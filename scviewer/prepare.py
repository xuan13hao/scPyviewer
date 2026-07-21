#!/usr/bin/env python
"""
prepare.py — Turn a raw or partially-processed AnnData (.h5ad) into a
viewer-ready object for the scviewer Streamlit app.

Design goals
------------
* Dataset-agnostic: works on any single-cell .h5ad regardless of species,
  annotation vocabulary, or which fields exist.
* Idempotent / robust: every step is guarded — if the input already carries a
  normalized layer, embeddings, or DE tables, that step is skipped rather than
  recomputed or overwritten.
* Transparent: writes a JSON manifest recording exactly what was computed vs.
  already present, so the preprocessing is auditable and reproducible.

What it guarantees on output
----------------------------
* X holds a log-normalized matrix (for coloring/violins); raw counts preserved
  in layers['counts'].
* A log-normalized matrix is available in layers['lognorm'] (== X).
* Highly-variable-gene flags in var['highly_variable'].
* PCA in obsm['X_pca']; UMAP in obsm['X_umap']; t-SNE in obsm['X_tsne'].
  (Pre-existing embeddings such as X_uce are left untouched.)
* A marker/DE table per grouping in uns['rank_genes_groups'] and a tidy,
  app-friendly copy in uns['scviewer_markers'].
* uns['scviewer'] metadata block: schema hints (categorical vs numeric obs
  fields), the embeddings available, and the grouping used for DE.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


# ----------------------------------------------------------------------------- helpers
def _looks_lognormed(X, n: int = 2000) -> bool:
    """Heuristic: log-normalized data is non-integer and bounded (< ~20)."""
    sub = X[:n]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    if sub.size == 0:
        return False
    mx = float(sub.max())
    frac_int = float(np.mean(sub == np.round(sub)))
    # counts are integer-valued and can be large; lognorm is fractional & small
    return (mx < 20.0) and (frac_int < 0.9)


def _pick_grouping(adata, user_group: str | None) -> str | None:
    """Choose the obs column to compute markers/DE against."""
    if user_group and user_group in adata.obs.columns:
        return user_group
    # prefer a cell-type-like column, then a cluster/leiden column
    preferred = ["cell_type", "celltype", "CellType", "cell.type",
                 "annotation", "leiden", "louvain", "cluster", "clusters"]
    for c in preferred:
        if c in adata.obs.columns:
            return c
    # fall back to any categorical column with a sensible number of levels
    for c in adata.obs.columns:
        s = adata.obs[c]
        if (s.dtype.name in ("category", "object")) and (2 <= s.nunique() <= 100):
            return c
    return None


def _schema(adata) -> dict[str, Any]:
    """Classify obs columns into categorical vs numeric for the app UI."""
    cats, nums = [], []
    for c in adata.obs.columns:
        s = adata.obs[c]
        if s.dtype.name in ("category", "object", "bool"):
            if s.nunique() <= 200:
                cats.append(c)
        elif np.issubdtype(s.dtype, np.number):
            nums.append(c)
    return {"categorical_obs": cats, "numeric_obs": nums}


def _tidy_markers(adata, group_key: str, n_top: int = 100) -> pd.DataFrame:
    """Flatten scanpy rank_genes_groups into a long tidy dataframe."""
    rg = adata.uns["rank_genes_groups"]
    groups = list(rg["names"].dtype.names)
    rows = []
    for g in groups:
        names = rg["names"][g][:n_top]
        lfc = rg["logfoldchanges"][g][:n_top] if "logfoldchanges" in rg else [np.nan] * len(names)
        pv = rg["pvals"][g][:n_top] if "pvals" in rg else [np.nan] * len(names)
        pva = rg["pvals_adj"][g][:n_top] if "pvals_adj" in rg else [np.nan] * len(names)
        sc_ = rg["scores"][g][:n_top] if "scores" in rg else [np.nan] * len(names)
        for rank, (nm, l, p, pa, s) in enumerate(zip(names, lfc, pv, pva, sc_), start=1):
            rows.append({
                "group": g, "rank": rank, "gene": str(nm),
                "logfoldchange": float(l), "score": float(s),
                "pval": float(p), "pval_adj": float(pa),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def prepare(in_path: str, out_path: str, group_key: str | None = None,
            n_top_genes: int = 2000, do_tsne: bool = True) -> dict[str, Any]:
    t0 = time.time()
    manifest: dict[str, Any] = {"input": in_path, "output": out_path,
                                "computed": [], "skipped": [], "warnings": []}

    adata = sc.read_h5ad(in_path)
    manifest["n_obs"], manifest["n_vars"] = int(adata.n_obs), int(adata.n_vars)

    # --- 1. preserve raw counts + build log-normalized matrix -----------------
    if _looks_lognormed(adata.X):
        # input X is already log-normalized
        manifest["skipped"].append("normalize (X already log-normalized)")
        adata.layers["lognorm"] = adata.X.copy()
        if "counts" not in adata.layers:
            manifest["warnings"].append("no raw counts found; counts layer == X")
            adata.layers["counts"] = adata.X.copy()
    else:
        # input X is raw counts
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers["lognorm"] = adata.X.copy()
        manifest["computed"].append("normalize_total(1e4) + log1p")

    # --- 2. highly variable genes ---------------------------------------------
    if "highly_variable" in adata.var.columns:
        manifest["skipped"].append("HVG (already present)")
    else:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars),
                                        flavor="seurat")
            manifest["computed"].append(f"highly_variable_genes(n_top={n_top_genes})")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"HVG failed: {e}")

    # --- 3. PCA ----------------------------------------------------------------
    if "X_pca" in adata.obsm:
        manifest["skipped"].append("PCA (already present)")
    else:
        try:
            sc.pp.pca(adata, n_comps=min(50, adata.n_obs - 1, adata.n_vars - 1))
            manifest["computed"].append("pca(50)")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"PCA failed: {e}")

    # --- 4. neighbors (needed for UMAP if absent) -----------------------------
    have_neighbors = "neighbors" in adata.uns
    if not have_neighbors and "X_pca" in adata.obsm:
        try:
            sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
            manifest["computed"].append("neighbors(15, use_rep=X_pca)")
            have_neighbors = True
        except Exception as e:  # noqa
            manifest["warnings"].append(f"neighbors failed: {e}")

    # --- 5. UMAP ---------------------------------------------------------------
    if "X_umap" in adata.obsm:
        manifest["skipped"].append("UMAP (already present)")
    elif have_neighbors:
        try:
            sc.tl.umap(adata)
            manifest["computed"].append("umap")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"UMAP failed: {e}")

    # --- 6. t-SNE --------------------------------------------------------------
    if "X_tsne" in adata.obsm:
        manifest["skipped"].append("t-SNE (already present)")
    elif do_tsne and "X_pca" in adata.obsm:
        try:
            sc.tl.tsne(adata, use_rep="X_pca", n_jobs=4)
            manifest["computed"].append("tsne")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"t-SNE failed: {e}")

    # --- 7. marker / DE table --------------------------------------------------
    gk = _pick_grouping(adata, group_key)
    manifest["group_key"] = gk
    if gk is None:
        manifest["warnings"].append("no suitable grouping column; skipped DE")
    elif "rank_genes_groups" in adata.uns and not group_key:
        manifest["skipped"].append("rank_genes_groups (already present)")
        try:
            adata.uns["scviewer_markers"] = _tidy_markers(adata, gk).to_dict("list")
            manifest["computed"].append("scviewer_markers (tidied from existing DE)")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"tidy markers failed: {e}")
    else:
        # ensure the grouping is categorical
        adata.obs[gk] = adata.obs[gk].astype("category")
        try:
            sc.tl.rank_genes_groups(adata, groupby=gk, method="wilcoxon",
                                    use_raw=False, layer="lognorm")
            manifest["computed"].append(f"rank_genes_groups(wilcoxon, groupby={gk})")
            adata.uns["scviewer_markers"] = _tidy_markers(adata, gk).to_dict("list")
            manifest["computed"].append("scviewer_markers")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"rank_genes_groups failed: {e}")

    # --- 8. viewer metadata block ---------------------------------------------
    embeddings = sorted([k for k in adata.obsm.keys() if k.startswith("X_")])
    adata.uns["scviewer"] = {
        "schema": _schema(adata),
        "embeddings": embeddings,
        "group_key": gk,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "prepared_by": "scviewer.prepare",
    }
    manifest["embeddings"] = embeddings
    manifest["schema"] = adata.uns["scviewer"]["schema"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    adata.write_h5ad(out_path)
    manifest["elapsed_sec"] = round(time.time() - t0, 2)
    manifest["output_size_mb"] = round(os.path.getsize(out_path) / 1e6, 1)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Prepare an AnnData .h5ad for the scviewer app.")
    ap.add_argument("input", help="path to input .h5ad")
    ap.add_argument("-o", "--output", help="path to output prepared .h5ad")
    ap.add_argument("-g", "--group-key", default=None,
                    help="obs column for marker/DE computation (auto-detected if omitted)")
    ap.add_argument("--n-top-genes", type=int, default=2000)
    ap.add_argument("--no-tsne", action="store_true", help="skip t-SNE (slow on big data)")
    args = ap.parse_args()

    out = args.output
    if out is None:
        base = os.path.basename(args.input)
        stem = base[:-5] if base.endswith(".h5ad") else base
        out = os.path.join(os.path.dirname(args.input) or ".", f"{stem}.prepared.h5ad")

    manifest = prepare(args.input, out, group_key=args.group_key,
                       n_top_genes=args.n_top_genes, do_tsne=not args.no_tsne)

    man_path = out.replace(".h5ad", ".manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out}\nWrote {man_path}")


if __name__ == "__main__":
    main()

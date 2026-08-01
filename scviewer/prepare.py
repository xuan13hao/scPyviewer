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

import psutil

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
def _available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _alt_embedding(adata) -> str | None:
    """Return the name of an existing non-PCA embedding suitable for neighbors."""
    preferred = ["X_scvi", "X_uce", "X_harmony", "X_diffmap"]
    for k in preferred:
        if k in adata.obsm:
            return k
    return None


def _looks_lognormed_backed(adata) -> bool:
    """Like _looks_lognormed but safe for backed datasets (reads one column)."""
    try:
        n_sample = min(2000, adata.n_obs)
        raw = adata.X[:n_sample, 0]
        arr = raw.toarray().ravel() if sp.issparse(raw) else np.asarray(raw).ravel()
        if arr.size == 0:
            return False
        return (float(arr.max()) < 20.0) and (float(np.mean(arr == np.round(arr))) < 0.9)
    except Exception:
        return False


def prepare_backed_only(in_path: str, out_path: str,
                        group_key: str | None = None) -> dict[str, Any]:
    """Memory-efficient prepare for large files on low-RAM machines.

    Opens in backed='r' mode so X stays on disk throughout. Computes UMAP from
    any existing alternative embedding (X_uce, X_scvi, ...) then discards it so
    the viewer's RAM footprint stays small. X is NOT normalised -- gene_vector in
    the viewer applies log1p on demand when x_is_raw is set in uns['scviewer'].
    Peak RAM: ~2-3 GB regardless of file size.
    """
    import gc as _gc
    t0 = time.time()
    manifest: dict[str, Any] = {
        "input": in_path, "output": out_path, "mode": "backed-only",
        "computed": [], "skipped": [], "warnings": [],
    }

    adata = sc.read_h5ad(in_path, backed="r")
    manifest["n_obs"], manifest["n_vars"] = int(adata.n_obs), int(adata.n_vars)

    # Drop all layers -- backed mode loads layers fully into RAM; viewer doesn't need them.
    dropped = list(adata.layers.keys())
    for _lyr in dropped:
        del adata.layers[_lyr]
    if dropped:
        manifest["computed"].append(f"dropped layers {dropped} to save RAM")

    # Detect raw counts so the viewer can apply log1p automatically.
    x_is_raw = not _looks_lognormed_backed(adata)
    if x_is_raw:
        needed_gb = os.path.getsize(in_path) / 1e9 * 1.6
        manifest["warnings"].append(
            f"X appears to be raw counts; viewer will apply log1p(x) on the fly. "
            f"For normalised expression + DE markers, re-run prepare on a machine "
            f"with >= {needed_gb:.0f} GB RAM."
        )

    # --- 1. UMAP from pre-existing embedding (e.g. X_uce) ----------------------
    alt_emb = _alt_embedding(adata)
    if "X_umap" in adata.obsm:
        manifest["skipped"].append("UMAP (already present)")
    elif "neighbors" in adata.uns:
        manifest["skipped"].append("neighbors (already present)")
        try:
            sc.tl.umap(adata)
            manifest["computed"].append("umap (from existing neighbors)")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"UMAP failed: {e}")
    elif alt_emb is not None:
        try:
            sc.pp.neighbors(adata, n_neighbors=15, use_rep=alt_emb)
            manifest["computed"].append(f"neighbors(15, use_rep={alt_emb})")
            sc.tl.umap(adata)
            manifest["computed"].append("umap")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"neighbors/UMAP failed: {e}")
    else:
        manifest["warnings"].append(
            "No alternative embedding (X_uce, X_scvi, ...) found -- skipping UMAP. "
            "Run prepare on a machine with more RAM for a full embedding."
        )

    # --- 2. Drop alt embedding from output to keep viewer RAM small -------------
    if alt_emb is not None and alt_emb in adata.obsm:
        del adata.obsm[alt_emb]
        _gc.collect()
        manifest["computed"].append(f"dropped {alt_emb} from output (saves viewer RAM)")

    # --- 3. Viewer metadata block -----------------------------------------------
    gk = _pick_grouping(adata, group_key)
    embeddings = sorted(k for k in adata.obsm.keys() if k.startswith("X_"))
    adata.uns["scviewer"] = {
        "schema": _schema(adata),
        "embeddings": embeddings,
        "group_key": gk,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "x_is_raw": x_is_raw,
        "prepared_by": "scviewer.prepare (backed-only)",
    }
    manifest["embeddings"] = embeddings
    manifest["schema"] = adata.uns["scviewer"]["schema"]
    manifest["group_key"] = gk

    # Remove null-encoded uns entries that backed='r' cannot deserialise.
    adata.uns.pop("log1p", None)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    adata.write_h5ad(out_path)
    adata.file.close()

    manifest["elapsed_sec"] = round(time.time() - t0, 2)
    manifest["output_size_mb"] = round(os.path.getsize(out_path) / 1e6, 1)
    return manifest


def prepare(in_path: str, out_path: str, group_key: str | None = None,
            n_top_genes: int = 2000, do_tsne: bool = True,
            convert_csc: bool = True, backed_only: bool = False) -> dict[str, Any]:
    """Prepare an AnnData for the scviewer app.

    convert_csc controls whether X is converted to CSC format (fastest backed
    column access) at the cost of 2x peak RAM for the conversion step.  Pass
    False when RAM is limited; gene-expression coloring will work but will read
    the entire matrix per gene (much slower).

    backed_only=True (or auto-triggered when file > 0.66x available RAM) keeps X
    on disk the entire time -- skips normalisation and DE, but peak RAM stays
    under 3 GB regardless of file size.
    """
    t0 = time.time()
    manifest: dict[str, Any] = {"input": in_path, "output": out_path,
                                "computed": [], "skipped": [], "warnings": []}

    # Auto-detect backed-only mode when the file is too large to fit in RAM.
    # Full load needs roughly 1.5x the file size (sparse X + obsm + metadata).
    file_size_gb = os.path.getsize(in_path) / 1e9
    avail_gb = _available_ram_gb()
    if backed_only or (file_size_gb * 1.5 > avail_gb):
        if not backed_only:
            manifest["warnings"].append(
                f"File is {file_size_gb:.1f} GB but only {avail_gb:.1f} GB available "
                f"-- switching to backed-only mode (no normalisation / DE). "
                f"Re-run on a machine with >= {file_size_gb * 1.6:.0f} GB RAM for a full prepare."
            )
        return prepare_backed_only(in_path, out_path, group_key=group_key)

    adata = sc.read_h5ad(in_path)
    manifest["n_obs"], manifest["n_vars"] = int(adata.n_obs), int(adata.n_vars)

    # Cast X to float32 immediately to halve peak memory on large datasets.
    if sp.issparse(adata.X):
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    # Drop all pre-existing layers immediately — the viewer only needs X.
    for _lyr in list(adata.layers.keys()):
        del adata.layers[_lyr]

    # --- 1. log-normalize X (in-place, no copies) ------------------------------
    if _looks_lognormed(adata.X):
        manifest["skipped"].append("normalize (X already log-normalized)")
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
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

    # --- 3. PCA — skip if a suitable embedding already exists -----------------
    alt_emb = _alt_embedding(adata)  # e.g. X_uce, X_scvi, X_harmony
    if "X_pca" in adata.obsm:
        manifest["skipped"].append("PCA (already present)")
    elif alt_emb is not None:
        manifest["skipped"].append(
            f"PCA (skipped — will use existing {alt_emb} for neighbors/UMAP)"
        )
    else:
        try:
            use_hvg = "highly_variable" in adata.var.columns
            sc.pp.pca(adata, n_comps=min(50, adata.n_obs - 1, adata.n_vars - 1),
                      use_highly_variable=use_hvg)
            manifest["computed"].append(f"pca(50, use_hvg={use_hvg})")
        except Exception as e:  # noqa
            manifest["warnings"].append(f"PCA failed: {e}")

    # --- 4. neighbors ---------------------------------------------------------
    have_neighbors = "neighbors" in adata.uns
    if not have_neighbors:
        # Prefer the pre-existing non-X embedding (saves the PCA dense matrix).
        use_rep = "X_pca" if "X_pca" in adata.obsm else alt_emb
        if use_rep:
            try:
                sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep)
                manifest["computed"].append(f"neighbors(15, use_rep={use_rep})")
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

    # Free large intermediate embeddings that are no longer needed.
    # Dropping alt_emb (e.g. X_uce at 1.6 GB) before DE frees critical RAM.
    import gc as _gc
    if alt_emb is not None and alt_emb in adata.obsm:
        del adata.obsm[alt_emb]
        _gc.collect()
        manifest["computed"].append(f"freed {alt_emb} from obsm to reclaim RAM")

    # --- 6. t-SNE — only if PCA available and t-SNE requested -----------------
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
        adata.obs[gk] = adata.obs[gk].astype("category")
        try:
            sc.tl.rank_genes_groups(adata, groupby=gk, method="wilcoxon",
                                    use_raw=False, layer=None)
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

    # Remove uns entries that carry null-encoded h5py values (e.g. log1p.base).
    # Older scanpy writes these; anndata 0.11.x backed='r' cannot read them.
    adata.uns.pop("log1p", None)

    # --- 9. CSC conversion for fast backed column access ----------------------
    # Converting CSR → CSC needs two copies of X in RAM simultaneously (~2× size).
    # Skip automatically if available RAM is less than 2.5× the X matrix size.
    if sp.issparse(adata.X) and not isinstance(adata.X, sp.csc_matrix):
        x_bytes = (adata.X.data.nbytes + adata.X.indices.nbytes
                   + adata.X.indptr.nbytes)
        ram_needed_gb = x_bytes * 2.5 / 1e9
        available_gb = _available_ram_gb()
        if convert_csc and available_gb >= ram_needed_gb:
            adata.X = adata.X.tocsc()
            manifest["computed"].append(
                f"convert X to CSC (RAM ok: {available_gb:.1f} GB available)")
        else:
            manifest["warnings"].append(
                f"CSC conversion skipped: need ~{ram_needed_gb:.1f} GB, "
                f"only {available_gb:.1f} GB available.  "
                "Gene-expression coloring will work but each gene query reads "
                "the full matrix from disk (slow for very large files).  "
                "Re-run on a machine with more RAM to enable fast access.")

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
    ap.add_argument("--no-csc", action="store_true",
                    help="skip CSR→CSC conversion (saves RAM; gene coloring will be slower)")
    ap.add_argument("--backed-only", action="store_true",
                    help="force backed-only mode: X stays on disk, no normalisation/DE, "
                         "peak RAM ~2-3 GB regardless of file size (auto-triggered when "
                         "file > 0.66x available RAM)")
    args = ap.parse_args()

    out = args.output
    if out is None:
        base = os.path.basename(args.input)
        stem = base[:-5] if base.endswith(".h5ad") else base
        out = os.path.join(os.path.dirname(args.input) or ".", f"{stem}.prepared.h5ad")

    manifest = prepare(args.input, out, group_key=args.group_key,
                       n_top_genes=args.n_top_genes, do_tsne=not args.no_tsne,
                       convert_csc=not args.no_csc,
                       backed_only=args.backed_only)

    man_path = out.replace(".h5ad", ".manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out}\nWrote {man_path}")


if __name__ == "__main__":
    main()

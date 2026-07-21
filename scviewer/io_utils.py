"""
io_utils.py — dataset discovery, cached AnnData loading, and value extraction
for the scviewer Streamlit app. All functions are UI-framework-agnostic
(no Streamlit imports) so they can be reused by the benchmark harness and the
figure-generation scripts.
"""
from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


# ------------------------------------------------------------------ discovery
def discover_datasets(data_dir: str) -> dict[str, str]:
    """Return {display_name: path} for every .h5ad in data_dir.

    Prefers *.prepared.h5ad; a raw file is only listed if no prepared
    counterpart exists.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.h5ad")))
    prepared = {p for p in paths if p.endswith(".prepared.h5ad")}
    prepared_stems = {os.path.basename(p)[: -len(".prepared.h5ad")] for p in prepared}
    out: dict[str, str] = {}
    for p in paths:
        base = os.path.basename(p)
        if p in prepared:
            out[base[: -len(".prepared.h5ad")]] = p
        else:
            stem = base[: -len(".h5ad")]
            if stem not in prepared_stems:  # raw with no prepared version
                out[stem + " (raw)"] = p
    return out


# ------------------------------------------------------------------ loading
def load_adata(path: str):
    """Load an AnnData from disk. (Caching is applied at the app layer.)"""
    return sc.read_h5ad(path)


def get_meta(adata) -> dict[str, Any]:
    """Return the scviewer metadata block, reconstructing it if absent so the
    app also works on AnnData objects that were not run through prepare.py."""
    sv = adata.uns.get("scviewer")
    if sv is not None:
        # h5ad round-trips numpy arrays; coerce to plain python
        return {
            "embeddings": list(sv["embeddings"]),
            "group_key": (None if sv.get("group_key") is None else str(sv["group_key"])),
            "schema": {
                "categorical_obs": list(sv["schema"]["categorical_obs"]),
                "numeric_obs": list(sv["schema"]["numeric_obs"]),
            },
        }
    # fallback: infer on the fly
    cats, nums = [], []
    for c in adata.obs.columns:
        s = adata.obs[c]
        if s.dtype.name in ("category", "object", "bool") and s.nunique() <= 200:
            cats.append(c)
        elif np.issubdtype(s.dtype, np.number):
            nums.append(c)
    gk = next((c for c in ["cell_type", "celltype", "leiden", "louvain", "cluster"]
               if c in adata.obs.columns), (cats[0] if cats else None))
    return {"embeddings": sorted(k for k in adata.obsm if k.startswith("X_")),
            "group_key": gk,
            "schema": {"categorical_obs": cats, "numeric_obs": nums}}


def embedding_2d(adata, key: str) -> np.ndarray:
    """Return the first two dimensions of an embedding as a dense (n,2) array."""
    X = adata.obsm[key]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return X[:, :2]


def gene_vector(adata, gene: str, layer: str = "lognorm") -> np.ndarray:
    """Return the expression vector for a single gene as a 1-D dense array."""
    if gene not in adata.var_names:
        raise KeyError(gene)
    idx = adata.var_names.get_loc(gene)
    mat = adata.layers[layer] if layer in adata.layers else adata.X
    col = mat[:, idx]
    col = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
    return col


def category_colors(adata, field: str) -> dict[str, str] | None:
    """Return {category: hex} using a stored *_colors palette if available."""
    key = f"{field}_colors"
    if key in adata.uns and adata.obs[field].dtype.name == "category":
        cats = list(adata.obs[field].cat.categories)
        cols = list(adata.uns[key])
        if len(cols) >= len(cats):
            return {str(c): str(col) for c, col in zip(cats, cols)}
    return None


def markers_df(adata) -> pd.DataFrame | None:
    """Return the tidy marker/DE table, or None if absent."""
    if "scviewer_markers" in adata.uns:
        return pd.DataFrame(adata.uns["scviewer_markers"])
    if "rank_genes_groups" in adata.uns:
        rg = adata.uns["rank_genes_groups"]
        groups = list(rg["names"].dtype.names)
        rows = []
        for g in groups:
            names = rg["names"][g]
            for rank, nm in enumerate(names, start=1):
                rows.append({"group": g, "rank": rank, "gene": str(nm)})
        return pd.DataFrame(rows)
    return None


def gene_search(adata, query: str, limit: int = 50) -> list[str]:
    """Case-insensitive prefix+substring search over var_names."""
    if not query:
        return list(adata.var_names[:limit])
    q = query.lower()
    names = adata.var_names.astype(str)
    pref = [n for n in names if n.lower().startswith(q)]
    sub = [n for n in names if q in n.lower() and n not in pref]
    return (pref + sub)[:limit]

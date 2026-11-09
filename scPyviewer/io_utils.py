"""
io_utils.py — dataset discovery, cached AnnData loading, and value extraction
for the scPyviewer Streamlit app. All functions are UI-framework-agnostic
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
_BACKED_THRESHOLD = int(os.environ.get("SCPYVIEWER_BACKED_BYTES", 500_000_000))  # 500 MB


def _fits_in_ram(size_bytes: int) -> bool:
    """True if a file of ``size_bytes`` can be loaded fully into RAM comfortably.

    A sparse ``.h5ad`` expands to roughly 2–3× its on-disk size in memory, so we
    require that headroom against currently-available RAM.  Falls back to a
    conservative ``False`` (prefer backed) when available RAM can't be queried.
    """
    try:
        import psutil
        return size_bytes * 3 < psutil.virtual_memory().available
    except Exception:
        return False


def load_adata(path: str):
    """Load an AnnData from disk, choosing in-memory vs. backed by RAM headroom.

    Small files (< SCPYVIEWER_BACKED_BYTES, default 500 MB) always load into
    memory.  Larger files load into memory **only if they comfortably fit in
    available RAM** (≈3× on-disk size); otherwise they open in ``backed='r'``
    mode so the matrix stays on disk.  This keeps a subsampled 1-GB object fast
    (in-memory) while a 12-GB atlas stays memory-safe (backed).  Falls back to a
    full load if the backed open fails (e.g. null-encoded ``uns`` from older
    scanpy — re-run prepare.py to fix).
    """
    size = os.path.getsize(path)
    if size > _BACKED_THRESHOLD and not _fits_in_ram(size):
        try:
            return sc.read_h5ad(path, backed="r")
        except Exception:
            import warnings
            warnings.warn(
                f"{os.path.basename(path)}: backed='r' failed; falling back to "
                "full in-memory load. Re-run prepare.py on the file to fix "
                "compatibility and enable low-memory backed access.",
                stacklevel=2,
            )
    return sc.read_h5ad(path)


def get_meta(adata) -> dict[str, Any]:
    """Return the scPyviewer metadata block, reconstructing it if absent so the
    app also works on AnnData objects that were not run through prepare.py."""
    sv = adata.uns.get("scPyviewer")
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
        elif s.dtype.name not in ("category", "object", "bool"):
            try:
                if np.issubdtype(s.dtype, np.number):
                    nums.append(c)
            except TypeError:
                pass
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


def _read_backed_column(X, gene_idx: int, n_obs: int) -> np.ndarray:
    """Read one gene column from a backed sparse dataset with bounded memory.

    A **CSC** backed dataset slices a column cheaply.  A **CSR** backed dataset
    would load the *entire* matrix if column-sliced (anndata falls back to
    ``to_memory`` — e.g. a 1M-cell file tries to allocate tens of GB), so we
    stream row-blocks sized to ~128 MB and pull the column from each block.
    Peak RAM is one block, independent of file size.
    """
    fmt = getattr(X, "format", None)
    if fmt == "csc":
        c = X[:, gene_idx]
        return c.toarray().ravel() if sp.issparse(c) else np.asarray(c).ravel()
    # CSR or unknown format: chunked row-block streaming.
    try:
        nnz = X.group["data"].shape[0]
        per_row = max(1.0, nnz / max(1, n_obs)) * 12.0  # bytes/row (data + indices)
        block = int(np.clip(128e6 / per_row, 1000, 100_000))
    except Exception:
        block = int(os.environ.get("SCPYVIEWER_STREAM_ROWS", 20_000))
    out = np.empty(n_obs, dtype=np.float32)
    for r0 in range(0, n_obs, block):
        r1 = min(r0 + block, n_obs)
        sub = X[r0:r1]
        c = sub[:, gene_idx]
        out[r0:r1] = c.toarray().ravel() if sp.issparse(c) else np.asarray(c).ravel()
    return out


def gene_vector(adata, gene: str, layer: str = "lognorm") -> np.ndarray:
    """Return the expression vector for a single gene as a 1-D dense float32 array.

    Works with in-memory, sparse, and h5py-backed AnnData objects (including
    backed views).  Backed access is memory-bounded: CSC files read the column
    directly; CSR files are streamed in row-blocks so peak RAM stays ~constant
    regardless of file size (see :func:`_read_backed_column`).  For fast,
    low-memory viewing of atlases larger than RAM, prefer
    ``scpyviewer-prepare --subsample N`` which writes a small CSC object.
    """
    if gene not in adata.var_names:
        raise KeyError(gene)
    gene_idx = adata.var_names.get_loc(gene)

    if getattr(adata, "isbacked", False):
        # Backed mode uses X (backed layers load fully into RAM in anndata
        # 0.11.x). For backed *views* we read the parent column (bounded) then
        # apply the obs index as a cheap in-memory fancy index.
        if adata.is_view:
            ref = adata._adata_ref
            col = _read_backed_column(ref.X, gene_idx, ref.n_obs)
            col = col[np.asarray(adata._oidx)]
        else:
            col = _read_backed_column(adata.X, gene_idx, adata.n_obs)
    else:
        mat = adata.layers[layer] if layer in adata.layers else adata.X
        raw = mat[:, gene_idx]
        col = raw.toarray().ravel() if sp.issparse(raw) else np.asarray(raw).ravel()

    col = col.astype(np.float32, copy=False)

    # Backed-only prepare mode stores raw counts in X.  Apply log1p here so that
    # gene-expression colouring and violin plots use a log scale automatically.
    if adata.uns.get("scPyviewer", {}).get("x_is_raw"):
        col = np.log1p(col)

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
    if "scPyviewer_markers" in adata.uns:
        return pd.DataFrame(adata.uns["scPyviewer_markers"])
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

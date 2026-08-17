"""Shared pytest fixtures for scPyviewer tests.

Builds a minimal in-memory AnnData (100 cells × 50 genes, 3 cell types,
UMAP + t-SNE embeddings, lognorm layer, and synthetic marker/DE table) so
tests run fast without any disk I/O.
"""
import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp


# ------------------------------------------------------------------ fixture
@pytest.fixture(scope="session")
def toy_adata():
    """100 cells × 50 genes with embeddings, lognorm layer, and markers."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 50
    gene_names = [f"Gene{i:03d}" for i in range(n_genes)]

    # Sparse count matrix (raw counts for variety)
    counts = sp.random(n_cells, n_genes, density=0.3, format="csr",
                       random_state=42).toarray()
    counts = (counts * 10).astype(np.float32)
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)

    # Obs metadata
    cell_types = np.array(["TypeA"] * 40 + ["TypeB"] * 35 + ["TypeC"] * 25)
    batch = np.array(["batch1"] * 50 + ["batch2"] * 50)
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(cell_types),
        "batch": pd.Categorical(batch),
        "n_counts": counts.sum(axis=1),
    })

    adata = ad.AnnData(
        X=sp.csr_matrix(lognorm),
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )
    adata.layers["lognorm"] = sp.csr_matrix(lognorm)

    # Fake 2-D embeddings
    adata.obsm["X_umap"] = rng.standard_normal((n_cells, 2)).astype(np.float32)
    adata.obsm["X_tsne"] = rng.standard_normal((n_cells, 2)).astype(np.float32)

    # Synthetic marker table (scPyviewer_markers format)
    rows = []
    for g_idx, group in enumerate(["TypeA", "TypeB", "TypeC"]):
        for rank in range(1, 11):
            gene = gene_names[(g_idx * 10 + rank - 1) % n_genes]
            rows.append({
                "group": group, "rank": rank, "gene": gene,
                "logfoldchange": float(rng.uniform(0.5, 3.0)),
                "score": float(rng.uniform(1.0, 10.0)),
                "pval": float(rng.uniform(0, 0.05)),
                "pval_adj": float(rng.uniform(0, 0.1)),
            })
    mdf = pd.DataFrame(rows)
    adata.uns["scPyviewer_markers"] = mdf.to_dict("list")

    # scPyviewer metadata block
    adata.uns["scPyviewer"] = {
        "schema": {
            "categorical_obs": ["cell_type", "batch"],
            "numeric_obs": ["n_counts"],
        },
        "embeddings": ["X_tsne", "X_umap"],
        "group_key": "cell_type",
        "n_obs": n_cells,
        "n_vars": n_genes,
    }
    return adata


@pytest.fixture(scope="session")
def toy_dataset(toy_adata, tmp_path_factory):
    """Saved toy_adata as a .h5ad and loaded as a scPyviewer Dataset."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import scPyviewer as sv
    from scPyviewer import io_utils as io

    path = str(tmp_path_factory.mktemp("data") / "toy.prepared.h5ad")
    toy_adata.write_h5ad(path)
    return sv.load_dataset(path)

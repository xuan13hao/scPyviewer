#!/usr/bin/env python3
"""Generate a small synthetic single-cell AnnData toy example for scviewer."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

rng = np.random.default_rng(42)

N_CELLS = 500
N_GENES = 200
CELL_TYPES = ["T cell", "B cell", "Macrophage", "NK cell", "Dendritic cell"]
SAMPLES = ["sample_A", "sample_B"]

# --- counts matrix (integer, Poisson-like) ---
# Each cell type has a set of marker genes with higher expression
counts = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)

cell_labels = rng.choice(CELL_TYPES, size=N_CELLS)
sample_labels = rng.choice(SAMPLES, size=N_CELLS)

# Add marker signal per cell type
marker_blocks = {ct: slice(i * 20, i * 20 + 20) for i, ct in enumerate(CELL_TYPES)}
for i, ct in enumerate(cell_labels):
    counts[i, marker_blocks[ct]] += rng.poisson(8.0, size=20).astype(np.float32)

# --- gene names ---
gene_names = [f"Gene{i:04d}" for i in range(N_GENES)]
# Sprinkle some realistic-looking names for the marker genes
bio_names = {
    "T cell":        ["CD3D", "CD3E", "CD8A", "CD8B", "CD4", "TRAC", "TRBC1", "IL7R", "TCF7", "LEF1",
                      "CCR7", "SELL", "CD69", "GZMB", "PRF1", "IFNG", "TNF", "IL2", "FOXP3", "CTLA4"],
    "B cell":        ["CD19", "MS4A1", "CD79A", "CD79B", "IGHM", "IGHD", "IGKC", "IGLC2", "PAX5", "EBF1",
                      "BANK1", "BLK", "FCRL1", "CD22", "FCER2", "CXCR5", "BCL6", "AID", "MZB1", "XBP1"],
    "Macrophage":    ["CD14", "LYZ", "CST3", "FCGR3A", "MS4A7", "CD68", "MRC1", "MARCO", "ITGAM", "CCL2",
                      "IL1B", "TNF", "IL6", "CXCL8", "MMP9", "FN1", "C1QA", "C1QB", "APOE", "TREM2"],
    "NK cell":       ["NCAM1", "NKG7", "GNLY", "KLRB1", "KLRD1", "KLRC1", "NCR1", "NCR3", "GZMB", "GZMK",
                      "PRF1", "IFNG", "CX3CR1", "FGFBP2", "FCGR3A", "SPON2", "XCL1", "XCL2", "KIR2DL1", "KIR3DL1"],
    "Dendritic cell":["FCER1A", "CST7", "CLEC10A", "CD1C", "ITGAX", "HLA-DRA", "HLA-DRB1", "CCR7", "LAMP3",
                      "IDO1", "CCL17", "CCL19", "CLEC9A", "SIGLEC6", "XCR1", "CADM1", "IRF8", "BATF3", "THBD", "FLT3"],
}
used = set()
for i, (ct, names) in enumerate(bio_names.items()):
    start = i * 20
    for j, nm in enumerate(names):
        # deduplicate: if same gene appears in two cell-type blocks, append suffix
        key = nm
        if key in used:
            key = f"{nm}_{i}"
        used.add(key)
        gene_names[start + j] = key

# --- obs / var ---
obs = pd.DataFrame({
    "cell_type": pd.Categorical(cell_labels, categories=CELL_TYPES),
    "sample":    pd.Categorical(sample_labels, categories=SAMPLES),
    "n_genes":   (counts > 0).sum(axis=1).astype(int),
    "total_counts": counts.sum(axis=1).astype(float),
}, index=[f"cell_{i:04d}" for i in range(N_CELLS)])

var = pd.DataFrame(index=gene_names)
var.index.name = "gene"

adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)

# --- log-normalize ---
import scanpy as sc
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()

# --- HVG + PCA ---
sc.pp.highly_variable_genes(adata, n_top_genes=100, flavor="seurat")
sc.tl.pca(adata, n_comps=20, use_highly_variable=True)

# --- UMAP ---
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=15)
sc.tl.umap(adata, min_dist=0.3)

# --- t-SNE ---
sc.tl.tsne(adata, n_pcs=15)

# --- Louvain clustering ---
try:
    sc.tl.louvain(adata, resolution=0.5, key_added="leiden")
except Exception:
    # fallback: assign clusters from cell_type index
    adata.obs["leiden"] = pd.Categorical(
        adata.obs["cell_type"].cat.codes.astype(str),
        categories=[str(c) for c in sorted(adata.obs["cell_type"].cat.codes.unique())],
    )

# --- DE markers ---
sc.tl.rank_genes_groups(adata, groupby="cell_type", method="wilcoxon", use_raw=False)

# tidy marker table expected by scviewer
rg = adata.uns["rank_genes_groups"]
groups = list(rg["names"].dtype.names)
rows = []
for g in groups:
    for rank, (nm, sc_, lfc, pv) in enumerate(
        zip(rg["names"][g], rg["scores"][g], rg["logfoldchanges"][g], rg["pvals_adj"][g]),
        start=1,
    ):
        rows.append({"group": g, "rank": rank, "gene": str(nm),
                     "score": float(sc_), "logfoldchange": float(lfc), "pvals_adj": float(pv)})
adata.uns["scviewer_markers"] = pd.DataFrame(rows).to_dict(orient="list")

# CSR -> CSC for fast column access
adata.X = adata.X.tocsc()
if sp.issparse(adata.layers.get("counts")):
    adata.layers["counts"] = adata.layers["counts"].tocsc()
if sp.issparse(adata.layers.get("lognorm")):
    adata.layers["lognorm"] = adata.layers["lognorm"].tocsc()

# --- scviewer metadata block ---
emb = [k for k in adata.obsm if k.startswith("X_")]
cats = ["cell_type", "sample", "leiden"]
nums = ["n_genes", "total_counts"]
adata.uns["scviewer"] = {
    "embeddings": emb,
    "group_key": "cell_type",
    "schema": {"categorical_obs": cats, "numeric_obs": nums},
}

out = "data/toy_example.prepared.h5ad"
adata.write_h5ad(out)
print(f"Saved {adata.n_obs} cells x {adata.n_vars} genes -> {out}")
print(f"Embeddings: {emb}")
print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

#!/usr/bin/env python3
"""Generate fig_de_snapshot for human lung dataset, top 10 genes per group."""
from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scviewer import io_utils as io

DATA_PATH  = "data/human_lung_disease-003.prepared.h5ad"
GROUP_KEY  = "Manuscript_Identity"
OUT_DIR    = "results/figures/multidataset/human_lung"
N_TOP      = 10
MAX_CELLS_PER_GROUP = 500   # subsample cap for DE computation

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load backed ────────────────────────────────────────────────────────────
print("Loading adata (backed)…")
adata = io.load_adata(DATA_PATH)
groups = list(adata.obs[GROUP_KEY].astype("category").cat.categories)
print(f"  {adata.n_obs} cells  ×  {adata.n_vars} genes  |  {len(groups)} groups")

# ── 2. Stratified subsample: up to MAX_CELLS_PER_GROUP per group ─────────────
print(f"Subsampling ≤{MAX_CELLS_PER_GROUP} cells per group…")
rng = np.random.default_rng(42)
keep_idx = []
for g in groups:
    idx = np.where(adata.obs[GROUP_KEY].values == g)[0]
    if len(idx) > MAX_CELLS_PER_GROUP:
        idx = rng.choice(idx, MAX_CELLS_PER_GROUP, replace=False)
    keep_idx.append(idx)
keep_idx = np.sort(np.concatenate(keep_idx))
print(f"  keeping {len(keep_idx)} cells")

# Load subsample into memory
X_sub = adata.X[keep_idx]
if not sp.issparse(X_sub):
    X_sub = sp.csr_matrix(X_sub)
elif hasattr(X_sub, 'toarray'):
    X_sub = sp.csr_matrix(X_sub)

obs_sub = adata.obs.iloc[keep_idx].copy()
sub = sc.AnnData(X=X_sub, obs=obs_sub, var=adata.var.copy())
sub.obs[GROUP_KEY] = sub.obs[GROUP_KEY].astype("category")

# ── 3. Compute rank_genes_groups ─────────────────────────────────────────────
print("Running rank_genes_groups (t-test)…")
sc.tl.rank_genes_groups(sub, groupby=GROUP_KEY, method="t-test", n_genes=N_TOP)

# ── 4. Build tidy markers DataFrame ──────────────────────────────────────────
rg = sub.uns["rank_genes_groups"]
group_names = list(rg["names"].dtype.names)
rows = []
for g in group_names:
    for rank, nm in enumerate(rg["names"][g], start=1):
        rows.append({"group": g, "rank": rank, "gene": str(nm)})
mdf = pd.DataFrame(rows)

# ── 5. Plot ───────────────────────────────────────────────────────────────────
print("Plotting…")
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "figure.facecolor": "white",
    "figure.dpi": 110,
})

# Show all groups (up to 39) in a grid with wider figure
n_groups_to_show = len(groups)
cols_per_row = 8
n_rows = int(np.ceil(n_groups_to_show / cols_per_row))

fig_w = min(cols_per_row, n_groups_to_show) * 1.6 + 0.4
fig_h = n_rows * (N_TOP * 0.18 + 0.55)

fig, axes = plt.subplots(n_rows, cols_per_row,
                         figsize=(fig_w, fig_h),
                         squeeze=False)
fig.patch.set_facecolor("white")

for j, g in enumerate(groups):
    r, c = divmod(j, cols_per_row)
    ax = axes[r][c]
    ax.axis("off")
    sub_genes = mdf[mdf["group"] == g].sort_values("rank").head(N_TOP)
    ax.text(0.5, 1.0, g, ha="center", va="top",
            fontsize=6.0, fontweight="bold", transform=ax.transAxes,
            wrap=True)
    for rank_i, (_, row) in enumerate(sub_genes.iterrows()):
        ax.text(0.5, 0.88 - rank_i * (0.88 / N_TOP),
                row["gene"], ha="center", va="top",
                fontsize=5.5, style="italic", transform=ax.transAxes)

# Hide unused axes
for j in range(n_groups_to_show, n_rows * cols_per_row):
    r, c = divmod(j, cols_per_row)
    axes[r][c].axis("off")

fig.suptitle(f"Top {N_TOP} marker genes per cell type  (Human Lung Disease)",
             fontsize=9, y=1.01)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "fig_de_snapshot.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")

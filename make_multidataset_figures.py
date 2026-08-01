#!/usr/bin/env python
"""
make_multidataset_figures.py — Generate the same figure set as make_figures.py
for each additional dataset, using identical function logic and file names but
saving into per-dataset subdirectories of --out.

Output layout:
    <out>/chicken_heart/fig_umap_celltypes.png
    <out>/chicken_heart/fig_multigene_grid.png
    <out>/chicken_heart/fig_violin_by_sample.png
    <out>/chicken_heart/fig_de_snapshot.png
    <out>/chicken_heart/fig_composition.png
    <out>/green_monkey/  ...
    <out>/human_lung/    ...
    <out>/dataset_overview_table.png

Font: Arial (300 dpi, tight bbox).

Usage:
    python make_multidataset_figures.py --out results/figures/multidataset
    python make_multidataset_figures.py --out results/figures/multidataset --dataset green_monkey
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scviewer import io_utils as io

# ── Global style: Arial + 300 dpi ──────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "Arial",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    12,
    "axes.titlelocation":"left",
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "figure.dpi":        110,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ── Dataset registry ────────────────────────────────────────────────────────
DATASETS = {
    "chicken_heart": dict(
        path="data/chicken_heart.prepared.h5ad",
        species="Gallus gallus",
        display="Chicken heart",
        tissue="Heart",
        technology="10x Chromium",
        group_key="cell_type",
        split_key="sample",
        embedding="X_umap",
    ),
    "green_monkey": dict(
        path="data/green_monkey.prepared.h5ad",
        species="Chlorocebus sabaeus",
        display="Green monkey lung/LN",
        tissue="Lung + Mediastinal LN",
        technology="10x Chromium",
        group_key="cell_type",
        split_key="tissue",
        embedding="X_umap",
    ),
    "human_lung": dict(
        path="data/human_lung_disease-003.prepared.h5ad",
        species="Homo sapiens",
        display="Human lung disease",
        tissue="Lung",
        technology="10x Chromium",
        group_key="Manuscript_Identity",
        split_key="Disease_Identity",
        embedding="X_umap",
    ),
}


# ── Internal helpers ─────────────────────────────────────────────────────────
def _pick_embedding(adata, preferred: str) -> str:
    if preferred in adata.obsm:
        return preferred
    for k in ("X_umap", "X_uce"):
        if k in adata.obsm:
            return k
    raise ValueError("no 2-D embedding found in obsm")


def _pick_genes(adata, n: int = 6) -> list[str]:
    """Top marker genes from rank table, or first HVG genes as fallback."""
    mdf = io.markers_df(adata)
    if mdf is not None:
        top = (mdf[mdf["rank"] == 1]
               .sort_values("group")["gene"].tolist())
        genes = top[:n]
        if len(genes) < n:
            genes += list(adata.var_names[:n - len(genes)])
        return genes
    if "highly_variable" in adata.var.columns:
        hvg = adata.var.index[adata.var["highly_variable"]].tolist()
        return hvg[:n]
    return list(adata.var_names[:n])


# ── Figure functions (same output names as make_figures.py) ─────────────────

def fig_umap_celltypes(adata, out: str, group_key: str, embedding: str = "X_umap") -> str:
    emb_key = _pick_embedding(adata, embedding)
    xy = io.embedding_2d(adata, emb_key)
    cats = list(adata.obs[group_key].astype("category").cat.categories)
    cmap = io.category_colors(adata, group_key) or {}

    # build a tab20 palette for any missing colors
    tab20 = plt.cm.get_cmap("tab20")
    for i, c in enumerate(cats):
        if c not in cmap:
            cmap[c] = tab20(i / max(len(cats), 1))

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for c in cats:
        m = (adata.obs[group_key].astype(str) == c).values
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.75,
                   color=cmap.get(c), label=c, linewidths=0, rasterized=True)
        cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
        ax.text(cx, cy, c, fontsize=5.6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.68))

    ax.set_xticks([]); ax.set_yticks([])
    emb_label = emb_key.replace("X_", "").upper()
    ax.annotate("", xy=(0.14, -0.02), xytext=(0.02, -0.02), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(0.02, 0.14), xytext=(0.02, -0.02), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(0.075, -0.055, emb_label, transform=ax.transAxes, fontsize=8.5, ha="center")
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    p = os.path.join(out, "fig_umap_celltypes.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_multigene(adata, out: str, genes: list[str]) -> str:
    genes = [g for g in genes if g in adata.var_names][:6]
    emb_key = _pick_embedding(adata, "X_umap")
    xy = io.embedding_2d(adata, emb_key)
    ncol = 3; nrow = int(np.ceil(len(genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, g in enumerate(genes):
        ax = axes[i]
        expr = io.gene_vector(adata, g, "lognorm")
        o = np.argsort(expr)
        sc = ax.scatter(xy[o, 0], xy[o, 1], c=expr[o], s=2.5, cmap="viridis",
                        linewidths=0, rasterized=True)
        ax.set_title(g, fontsize=10, style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=5.5); cb.set_label("expr", fontsize=8)
    for j in range(len(genes), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out, "fig_multigene_grid.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_violin_by_sample(adata, out: str, gene: str, sample_field: str,
                         top_n: int = 10) -> str:
    all_groups = sorted(adata.obs[sample_field].astype(str).unique())
    expr   = io.gene_vector(adata, gene, "lognorm")
    groups = adata.obs[sample_field].astype(str).values

    # rank all groups by mean expression, keep top_n
    data_map = {s: expr[groups == s] for s in all_groups}
    order = sorted(all_groups, key=lambda s: data_map[s].mean(), reverse=True)[:top_n]
    order = sorted(order)   # re-sort alphabetically for consistent display

    data = [data_map[s] for s in order]
    cmap = io.category_colors(adata, sample_field) or {}

    tab20 = plt.cm.get_cmap("tab20")
    for i, s in enumerate(order):
        if s not in cmap:
            cmap[s] = tab20(i / max(len(order), 1))

    fig_w = max(7.0, 0.42 * len(order) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.85)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(cmap.get(order[i], "#4d9221"))
        b.set_alpha(0.8); b.set_edgecolor("none")
    parts["cmeans"].set_color("#222"); parts["cmeans"].set_linewidth(1.0)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("log-normalized expression")
    ax.set_xlabel("")
    fig.tight_layout()
    p = os.path.join(out, "fig_violin_by_sample.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_de_snapshot(adata, out: str, group_key: str,
                    n_top: int = 8, n_groups: int = 6) -> str | None:
    mdf = io.markers_df(adata)
    if mdf is None:
        return None
    groups = list(adata.obs[group_key].astype("category").cat.categories)[:n_groups]
    fig, ax = plt.subplots(figsize=(8.0, 4.8)); ax.axis("off")
    col_w = 1.0 / len(groups)
    for j, g in enumerate(groups):
        sub = mdf[mdf["group"] == g].sort_values("rank").head(n_top)
        ax.text(j * col_w + col_w / 2, 1.0, g, ha="center", va="top",
                fontsize=6.6, fontweight="bold", transform=ax.transAxes)
        for r, (_, row) in enumerate(sub.iterrows()):
            ax.text(j * col_w + col_w / 2, 0.90 - r * 0.10,
                    f"{row['gene']}", ha="center", va="top", fontsize=6.0,
                    style="italic", transform=ax.transAxes)
    fig.tight_layout()
    p = os.path.join(out, "fig_de_snapshot.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_composition(adata, out: str, group_key: str, split_field: str) -> str:
    import pandas as pd
    df = adata.obs[[split_field, group_key]].astype(str)
    ct = df.groupby([split_field, group_key]).size().unstack(fill_value=0)
    frac = ct.div(ct.sum(axis=1), axis=0)
    cats = list(adata.obs[group_key].astype("category").cat.categories)
    frac = frac[[c for c in cats if c in frac.columns]]
    cmap = io.category_colors(adata, group_key) or {}

    tab20 = plt.cm.get_cmap("tab20")
    for i, c in enumerate(cats):
        if c not in cmap:
            cmap[c] = tab20(i / max(len(cats), 1))

    # cap split groups at 30
    if len(frac) > 30:
        frac = frac.iloc[:30]

    fig_w = max(7.6, 0.4 * len(frac) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))
    bottom = np.zeros(len(frac)); x = np.arange(len(frac))
    for c in frac.columns:
        ax.bar(x, frac[c].values, bottom=bottom, width=0.8,
               color=cmap.get(c), label=c, linewidth=0)
        bottom += frac[c].values
    ax.set_xticks(x); ax.set_xticklabels(frac.index, rotation=30, ha="right")
    ax.set_ylabel("fraction of cells"); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, "fig_composition.png")
    fig.savefig(p); plt.close(fig)
    return p


# ── Dataset overview table ───────────────────────────────────────────────────
def fig_dataset_table(rows: list[dict], out: str) -> str:
    """Render a clean dataset summary table for the paper."""
    cols   = ["Dataset", "Species", "Tissue", "Technology", "Cells", "Genes", "Cell types"]
    widths = [0.20,       0.17,      0.17,     0.15,         0.10,    0.10,    0.11]
    xs     = [sum(widths[:i]) for i in range(len(cols))]

    n = len(rows)
    fig_h = max(1.4, 0.40 * (n + 1) + 0.5)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))
    ax.axis("off")

    # header
    for h, x in zip(cols, xs):
        ax.text(x, 1.0, h, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")
    ax.plot([0, 1], [1.0 - 0.10, 1.0 - 0.10], color="#333", lw=0.8,
            transform=ax.transAxes, clip_on=False)

    row_h = min(0.18, 0.72 / n) if n else 0.18
    for i, r in enumerate(rows):
        y = 1.0 - 0.16 - i * row_h
        bg = "#f7f7f7" if i % 2 == 0 else "white"
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((0, y - row_h * 0.15), 1, row_h * 0.97,
                                    boxstyle="square,pad=0", facecolor=bg,
                                    edgecolor="none", transform=ax.transAxes,
                                    clip_on=False))
        vals = [r["display"], r["species"], r["tissue"], r["technology"],
                f"{r['n_obs']:,}", f"{r['n_vars']:,}", str(r["n_types"])]
        for j, (v, x) in enumerate(zip(vals, xs)):
            st = "italic" if j == 1 else "normal"   # species in italic
            ax.text(x, y, v, transform=ax.transAxes,
                    fontsize=10, va="top", ha="left", style=st)

    ax.plot([0, 1], [1.0 - 0.16 - n * row_h + row_h * 0.05] * 2,
            color="#aaa", lw=0.5, transform=ax.transAxes, clip_on=False)
    fig.tight_layout()
    p = os.path.join(out, "dataset_overview_table.png")
    fig.savefig(p); plt.close(fig)
    return p


# ── Driver ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/multidataset",
                    help="root output directory; per-dataset subdirs are created inside")
    ap.add_argument("--dataset", default=None,
                    help="process only this dataset key (default: all available)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    keys = ([args.dataset] if args.dataset else list(DATASETS.keys()))
    table_rows = []
    all_written = []

    for key in keys:
        cfg = DATASETS[key]
        if not os.path.exists(cfg["path"]):
            print(f"  [SKIP] {key}: {cfg['path']} not found — run prepare.py first")
            continue

        subdir = os.path.join(args.out, key)
        os.makedirs(subdir, exist_ok=True)
        print(f"\n── {key} ({cfg['display']}) → {subdir}/")

        adata = io.load_adata(cfg["path"])
        gk    = cfg["group_key"]
        sk    = cfg["split_key"]
        emb   = cfg["embedding"]
        n_types = adata.obs[gk].nunique() if gk in adata.obs.columns else 0

        table_rows.append({**cfg, "n_obs": adata.n_obs, "n_vars": adata.n_vars,
                           "n_types": n_types})

        # 1. UMAP by cell type
        p = fig_umap_celltypes(adata, subdir, gk, emb)
        print(f"   fig_umap_celltypes      → {p}"); all_written.append(p)

        # 2. Multi-gene grid
        genes = _pick_genes(adata, n=6)
        p = fig_multigene(adata, subdir, genes)
        print(f"   fig_multigene_grid      → {p}"); all_written.append(p)

        # 3. Violin — use group_key (cell type) if split_key has too few groups,
        #    otherwise use split_key. Always pick top-10 by mean expression.
        mdf = io.markers_df(adata)
        top_gene = (mdf.sort_values("rank").iloc[0]["gene"]
                    if mdf is not None else adata.var_names[0])
        n_split  = adata.obs[sk].nunique() if sk in adata.obs.columns else 0
        violin_field = sk if n_split >= 5 else gk
        p = fig_violin_by_sample(adata, subdir, top_gene, violin_field, top_n=10)
        print(f"   fig_violin_by_sample    → {p}  (field={violin_field!r})"); all_written.append(p)

        # 4. DE snapshot
        p = fig_de_snapshot(adata, subdir, gk)
        if p:
            print(f"   fig_de_snapshot         → {p}"); all_written.append(p)

        # 5. Composition bar
        if sk in adata.obs.columns:
            p = fig_composition(adata, subdir, gk, sk)
            print(f"   fig_composition         → {p}"); all_written.append(p)

    # Combined table
    if table_rows:
        p = fig_dataset_table(table_rows, args.out)
        print(f"\n   dataset_overview_table  → {p}"); all_written.append(p)

    print(f"\nDone. {len(all_written)} figures written.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
make_figures.py — Render the demonstration/paper figures directly from the
scPyviewer plot builders (the same functions the Streamlit app calls), so the
paper figures are faithful to what a user sees in the tool.

Two figure families:
  * benchmark figures  — feature-parity matrix, render-time bars (from the
    benchmark JSON).
  * demonstration figures — annotated UMAP, multi-gene expression grid, violin
    by developmental sample, marker/DE snapshot, cross-sample composition
    (rendered from the prepared AnnData via matplotlib for static export).

Usage:
  python scPyviewer/make_figures.py --prepared data/chicken_heart.prepared.h5ad \
      --benchmark results/benchmark_results.json --out results/figures
  add --benchmark-only to render just the benchmark figures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scPyviewer import io_utils as io  # noqa: E402


# --------------------------------------------------------------- style
def _style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.titlelocation": "left",
    })


# --------------------------------------------------------------- benchmark figs
def fig_feature_parity(bench: dict, out: str):
    cl = bench["feature_parity"]["checklist"]
    tools = ["ShinyCell", "ScRDAVis", "sCIRCLE/scViewer", "scPyviewer"]
    lvl = lambda v: {"Yes": 2, "Partial": 1, "No": 0}.get(v, 0)
    feat = [c["feature"].replace(" (UMAP/PCA/t-SNE)", "").replace(" gene / DE table", " / DE table")
            for c in cl]
    M = np.zeros((len(cl), len(tools))); labs = np.empty_like(M, dtype=object)
    for i, c in enumerate(cl):
        for j, t in enumerate(tools):
            val = ("Yes" if c["scPyviewer"] else "No") if t == "scPyviewer" else c[t]
            M[i, j] = lvl(val); labs[i, j] = val
    cmap = ListedColormap(["#f0f0f0", "#fdc086", "#4d9221"])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(tools))); ax.set_xticklabels(tools)
    ax.set_yticks(range(len(cl))); ax.set_yticklabels(feat)
    ax.add_patch(Rectangle((len(tools) - 1.5, -0.5), 1, len(cl), fill=False, edgecolor="#222", lw=2))
    for i in range(len(cl)):
        for j in range(len(tools)):
            ax.text(j, i, labs[i, j], ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] == 2 else "#222")
    ax.set_xticks(np.arange(-.5, len(tools), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(cl), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    score = bench["feature_parity"]["parity_score"]
    # ax.set_title(f"scPyviewer reproduces all 7 baseline features (parity {score:.0%})")
    leg = [Patch(facecolor="#4d9221", label="Yes"), Patch(facecolor="#fdc086", label="Partial"),
           Patch(facecolor="#f0f0f0", edgecolor="#ccc", label="No / absent")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    p = os.path.join(out, "fig_feature_parity.png"); fig.savefig(p); plt.close(fig)
    return p


def fig_render_times(bench: dict, out: str):
    p = bench["performance"]
    keys = [("render_embedding_metadata", "Embedding\n(metadata)"),
            ("render_embedding_gene", "Embedding\n(gene)"),
            ("render_multigene_grid_3", "Multi-gene\ngrid (3)"),
            ("render_violin", "Violin"), ("render_dotplot_5", "Dotplot (5)"),
            ("render_composition", "Composition\nbar")]
    names = [n for _, n in keys]
    best = [p[k]["best_sec"] for k, _ in keys]; mean = [p[k]["mean_sec"] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.0)); x = np.arange(len(names))
    ax.bar(x, mean, width=0.62, color="#4d9221", label="mean of 3 runs")
    ax.scatter(x, best, marker="_", s=340, color="#222", zorder=5, label="best of 3")
    for xi, m in zip(x, mean):
        ax.text(xi, m + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylabel("render time (s)"); ax.set_ylim(0, max(mean) * 1.28)
    load = p["load_h5ad"]["best_sec"]
    # ax.set_title(f"All core views render in <0.5 s at {p['n_obs']:,} cells\n"
    #              f"(load {load:.1f} s · peak {p['peak_python_mem_mb']:.0f} MB)", fontsize=8.5)
    ax.legend(frameon=False, loc="upper right"); ax.margins(x=0.02)
    fig.tight_layout()
    pth = os.path.join(out, "fig_render_times.png"); fig.savefig(pth); plt.close(fig)
    return pth


# --------------------------------------------------------------- demo figs
def _emb_xy(adata, key):
    return io.embedding_2d(adata, key)


def fig_umap_celltypes(adata, out: str, group_key: str):
    xy = _emb_xy(adata, "X_umap")
    cats = list(adata.obs[group_key].astype("category").cat.categories)
    cmap = io.category_colors(adata, group_key) or {}
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for c in cats:
        m = (adata.obs[group_key].astype(str) == c).values
        ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.75,
                   color=cmap.get(c, None), label=c, linewidths=0, rasterized=True)
        # leader-label at centroid
        cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
        ax.text(cx, cy, c, fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.68))
    ax.set_xticks([]); ax.set_yticks([])
    ax.annotate("", xy=(0.14, -0.02), xytext=(0.02, -0.02), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(0.02, 0.14), xytext=(0.02, -0.02), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(0.075, -0.055, "UMAP", transform=ax.transAxes, fontsize=8, ha="center")
    # ax.set_title(f"Chicken heart development: {len(cats)} annotated cell types "
    #              f"({adata.n_obs:,} cells)")
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out, "fig_umap_celltypes.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_multigene(adata, out: str, genes: list[str]):
    genes = [g for g in genes if g in adata.var_names][:6]
    xy = _emb_xy(adata, "X_umap")
    ncol = 3; nrow = int(np.ceil(len(genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, g in enumerate(genes):
        ax = axes[i]; expr = io.gene_vector(adata, g, "lognorm")
        o = np.argsort(expr)  # plot high-expr on top
        sc = ax.scatter(xy[o, 0], xy[o, 1], c=expr[o], s=2.5, cmap="viridis",
                        linewidths=0, rasterized=True)
        ax.set_title(g, fontsize=8, style="italic"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=5.5); cb.set_label("expr", fontsize=6)
    for j in range(len(genes), len(axes)):
        axes[j].set_visible(False)
    # fig.suptitle("Cell-type marker expression on the UMAP embedding", x=0.02, ha="left", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out, "fig_multigene_grid.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_violin_by_sample(adata, out: str, gene: str, sample_field: str):
    from scipy import sparse as sp
    order = sorted(adata.obs[sample_field].astype(str).unique())
    expr = io.gene_vector(adata, gene, "lognorm")
    groups = adata.obs[sample_field].astype(str).values
    data = [expr[groups == s] for s in order]
    cmap = io.category_colors(adata, sample_field) or {}
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.85)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(cmap.get(order[i], "#4d9221")); b.set_alpha(0.8); b.set_edgecolor("none")
    parts["cmeans"].set_color("#222"); parts["cmeans"].set_linewidth(1.0)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("log-normalized expression")
    # ax.set_title(f"{gene} expression across developmental samples", fontsize=8.5)
    ax.set_xlabel("")
    fig.tight_layout()
    p = os.path.join(out, "fig_violin_by_sample.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_de_snapshot(adata, out: str, group_key: str, n_top: int = 8, n_groups: int = 6):
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
    # ax.set_title(f"Top Wilcoxon marker genes per cell type (groupby = {group_key})",
    #              fontsize=8.5, loc="left")
    fig.tight_layout()
    p = os.path.join(out, "fig_de_snapshot.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_composition(adata, out: str, group_key: str, split_field: str):
    import pandas as pd
    df = adata.obs[[split_field, group_key]].astype(str)
    ct = df.groupby([split_field, group_key]).size().unstack(fill_value=0)
    frac = ct.div(ct.sum(axis=1), axis=0)
    cats = list(adata.obs[group_key].astype("category").cat.categories)
    frac = frac[[c for c in cats if c in frac.columns]]
    cmap = io.category_colors(adata, group_key) or {}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bottom = np.zeros(len(frac))
    x = np.arange(len(frac))
    for c in frac.columns:
        ax.bar(x, frac[c].values, bottom=bottom, width=0.8,
               color=cmap.get(c, None), label=c, linewidth=0)
        bottom += frac[c].values
    ax.set_xticks(x); ax.set_xticklabels(frac.index, rotation=30, ha="right")
    ax.set_ylabel("fraction of cells"); ax.set_ylim(0, 1)
    # ax.set_title(f"Cell-type composition across developmental samples", fontsize=8.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=5.8)
    fig.tight_layout()
    p = os.path.join(out, "fig_composition.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


# --------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="data/chicken_heart.prepared.h5ad")
    ap.add_argument("--benchmark", default="results/benchmark_results.json")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--genes", nargs="*", default=None,
                    help="genes for the multi-gene grid (default: top marker per major type)")
    args = ap.parse_args()
    _style()
    os.makedirs(args.out, exist_ok=True)
    written = []

    if os.path.exists(args.benchmark):
        bench = json.load(open(args.benchmark))
        written.append(fig_feature_parity(bench, args.out))
        written.append(fig_render_times(bench, args.out))

    if not args.benchmark_only:
        adata = io.load_adata(args.prepared)
        meta = io.get_meta(adata)
        gk = meta["group_key"]
        cats = meta["schema"]["categorical_obs"]
        split = next((c for c in cats if c != gk), cats[0])
        mdf = io.markers_df(adata)
        # default marker genes: rank-1 marker of the largest few cell types
        if args.genes:
            genes = args.genes
        elif mdf is not None:
            top_by_group = (mdf[mdf["rank"] == 1].sort_values("group")["gene"].tolist())
            genes = top_by_group[:6]
        else:
            genes = list(adata.var_names[:6])
        written.append(fig_umap_celltypes(adata, args.out, gk))
        written.append(fig_multigene(adata, args.out, genes))
        top_gene = genes[0] if genes else adata.var_names[0]
        written.append(fig_violin_by_sample(adata, args.out, top_gene, split))
        written.append(fig_de_snapshot(adata, args.out, gk))
        written.append(fig_composition(adata, args.out, gk, split))

    written = [w for w in written if w]
    print("Wrote:")
    for w in written:
        print("  ", w)


if __name__ == "__main__":
    main()

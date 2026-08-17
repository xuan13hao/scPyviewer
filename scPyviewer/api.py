"""scPyviewer.api — a public, programmatic interface to scPyviewer.

This module exposes the same views the interactive Streamlit app offers as
plain Python functions that **return objects** (matplotlib ``Figure`` and pandas
``DataFrame``) and helpers that **write files** (figures and tables). It lets a
scanpy user drive scPyviewer from a script or notebook without launching the UI::

    import scPyviewer as sv

    ds = sv.load_dataset("data/chicken_heart.prepared.h5ad")
    fig = sv.plot_embedding(ds, color=ds.group_key)      # matplotlib Figure
    fig.savefig("umap.png", dpi=200)

    markers = sv.markers_table(ds)                        # pandas DataFrame
    sv.export_figures(ds, "out/figs")                     # writes png + pdf
    sv.export_tables(ds, "out/tables")                    # writes csv

The plotting functions here render with matplotlib for reliable static export
(PNG/PDF/SVG) with no extra system dependencies. The interactive Plotly views
used by the Streamlit app live in :mod:`scPyviewer.plots`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / script use
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

from . import io_utils as io


# ------------------------------------------------------------------ handle
@dataclass
class Dataset:
    """A loaded, viewer-ready dataset.

    Attributes
    ----------
    adata : AnnData
        The underlying AnnData object.
    path : str
        Source ``.h5ad`` path.
    group_key : str
        Auto-selected primary grouping column (e.g. ``cell_type``).
    embeddings : list[str]
        Available 2-D embedding keys in ``obsm`` (e.g. ``X_umap``).
    categorical : list[str]
        Categorical ``obs`` columns available for grouping/splitting.
    """
    adata: Any
    path: str
    group_key: str
    embeddings: list
    categorical: list

    @property
    def n_obs(self) -> int:
        return int(self.adata.n_obs)

    @property
    def n_vars(self) -> int:
        return int(self.adata.n_vars)

    def genes(self, query: str = "", limit: int = 50) -> list:
        """Search gene names (substring, case-insensitive)."""
        return io.gene_search(self.adata, query, limit=limit)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Dataset({os.path.basename(self.path)}: "
                f"{self.n_obs:,} cells x {self.n_vars:,} genes, "
                f"group_key={self.group_key!r})")


def load_dataset(path: str) -> Dataset:
    """Load a prepared ``.h5ad`` and return a :class:`Dataset` handle.

    Parameters
    ----------
    path : str
        Path to a ``*.prepared.h5ad`` (or any AnnData; run
        :mod:`scPyviewer.prepare` first for full functionality).
    """
    adata = io.load_adata(path)
    meta = io.get_meta(adata)
    return Dataset(
        adata=adata, path=path,
        group_key=meta["group_key"],
        embeddings=list(meta["embeddings"]),
        categorical=list(meta["schema"]["categorical_obs"]),
    )


# ------------------------------------------------------------------ helpers
def _resolve_embedding(ds: Dataset, embedding: str | None) -> str:
    if embedding and embedding in ds.embeddings:
        return embedding
    if "X_umap" in ds.embeddings:
        return "X_umap"
    if not ds.embeddings:
        raise ValueError("no 2-D embedding available in this dataset")
    return ds.embeddings[0]


def _group(ds: Dataset, group: str | None) -> str:
    return group or ds.group_key


def _emb_label(key: str) -> str:
    return key.replace("X_", "").upper()


# ------------------------------------------------------------------ plots
def plot_embedding(
    ds: Dataset,
    color: str | None = None,
    gene: str | None = None,
    embedding: str | None = None,
    point_size: float = 3.0,
    figsize: tuple = (7.4, 5.6),
    label_groups: bool = True,
    cmap: str = "viridis",
    alpha: float = 0.75,
    title: str | None = None,
    dpi: int = 150,
    show_legend: bool = True,
):
    """Scatter of a 2-D embedding, colored by a metadata column or a gene.

    Exactly one of ``color`` (an ``obs`` column) or ``gene`` should be given.
    If neither is given, colors by ``ds.group_key``. Returns a matplotlib
    ``Figure``.

    Parameters
    ----------
    color : str, optional
        ``obs`` column name to color by (categorical or numeric).
    gene : str, optional
        Gene name whose log-normalized expression drives the color scale.
    embedding : str, optional
        Embedding key in ``obsm`` (e.g. ``"X_umap"``). Auto-selected if None.
    point_size : float
        Scatter point diameter in points.
    figsize : tuple
        Figure (width, height) in inches.
    label_groups : bool
        Overlay group-centroid text labels when coloring by category.
    cmap : str
        Matplotlib colormap name for continuous/gene coloring.
    alpha : float
        Point transparency (0–1) for categorical scatter.
    title : str, optional
        Override the auto-generated figure title.
    dpi : int
        Dots-per-inch resolution of the figure.
    show_legend : bool
        Show legend when coloring by a categorical field without centroid labels.
    """
    key = _resolve_embedding(ds, embedding)
    xy = io.embedding_2d(ds.adata, key)
    lab = _emb_label(key)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if gene is not None:
        expr = io.gene_vector(ds.adata, gene, "lognorm")
        o = np.argsort(expr)
        sc = ax.scatter(xy[o, 0], xy[o, 1], c=expr[o], s=point_size, cmap=cmap,
                        linewidths=0, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("log-normalized expression", fontsize=8)
        ax.set_title(title or f"{gene} expression ({lab})", fontsize=9, style="italic")
    else:
        field = color or ds.group_key
        cats = list(ds.adata.obs[field].astype("category").cat.categories)
        cmap_dict = io.category_colors(ds.adata, field) or {}
        for c in cats:
            m = (ds.adata.obs[field].astype(str) == c).values
            ax.scatter(xy[m, 0], xy[m, 1], s=point_size, alpha=alpha,
                       color=cmap_dict.get(c, None), label=c, linewidths=0,
                       rasterized=True)
            if label_groups:
                cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
                ax.text(cx, cy, c, fontsize=5.6, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                  alpha=0.68))
        ax.set_title(title or f"{field} ({lab}, {ds.n_obs:,} cells)", fontsize=9)
        if show_legend and not label_groups:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      frameon=False, fontsize=5.8)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def plot_multigene(
    ds: Dataset,
    genes: list,
    embedding: str | None = None,
    ncol: int = 3,
    point_size: float = 2.5,
    cmap: str = "viridis",
    alpha: float = 0.8,
    max_genes: int = 12,
):
    """Grid of embedding scatters, one panel per gene. Returns a Figure.

    Parameters
    ----------
    genes : list[str]
        Gene names to plot. Filtered to those present in the dataset.
    embedding : str, optional
        Embedding key (auto-selected if None).
    ncol : int
        Number of columns in the grid.
    point_size : float
        Scatter point diameter in points.
    cmap : str
        Matplotlib colormap for expression coloring.
    alpha : float
        Point transparency (0–1).
    max_genes : int
        Hard cap on the number of genes shown (default 12).
    """
    genes = [g for g in genes if g in ds.adata.var_names][:max_genes]
    if not genes:
        raise ValueError("none of the requested genes are present")
    key = _resolve_embedding(ds, embedding)
    xy = io.embedding_2d(ds.adata, key)
    nrow = int(np.ceil(len(genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, g in enumerate(genes):
        ax = axes[i]
        expr = io.gene_vector(ds.adata, g, "lognorm")
        o = np.argsort(expr)
        sc = ax.scatter(xy[o, 0], xy[o, 1], c=expr[o], s=point_size, cmap=cmap,
                        alpha=alpha, linewidths=0, rasterized=True)
        ax.set_title(g, fontsize=8, style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=5.5); cb.set_label("expr", fontsize=6)
    for j in range(len(genes), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Gene expression on the {_emb_label(key)} embedding",
                 x=0.02, ha="left", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_violin(
    ds: Dataset,
    gene: str,
    group: str | None = None,
    figsize: tuple = (7.0, 4.0),
    kind: str = "violin",
    palette: dict | None = None,
    rotation: int = 30,
    show_points: bool = False,
):
    """Violin or box plot of a gene's expression grouped by a metadata column.

    Returns a matplotlib ``Figure``.

    Parameters
    ----------
    gene : str
        Gene name whose expression to plot.
    group : str, optional
        ``obs`` column for grouping (defaults to ``ds.group_key``).
    figsize : tuple
        Figure (width, height) in inches.
    kind : {"violin", "box"}
        Plot type. ``"violin"`` shows the full distribution; ``"box"`` shows
        quartiles and whiskers.
    palette : dict, optional
        ``{group_label: color}`` override. Falls back to dataset palette then
        a default green.
    rotation : int
        X-axis tick-label rotation in degrees.
    show_points : bool
        Overlay individual data points as a jittered strip plot.
    """
    if kind not in ("violin", "box"):
        raise ValueError(f"kind must be 'violin' or 'box', got {kind!r}")
    field = _group(ds, group)
    order = sorted(ds.adata.obs[field].astype(str).unique())
    expr = io.gene_vector(ds.adata, gene, "lognorm")
    groups = ds.adata.obs[field].astype(str).values
    data = [expr[groups == s] for s in order]
    cmap_dict = palette or io.category_colors(ds.adata, field) or {}
    colors = [cmap_dict.get(s, "#4d9221") for s in order]

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "violin":
        parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.85)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(colors[i]); b.set_alpha(0.8); b.set_edgecolor("none")
        parts["cmeans"].set_color("#222"); parts["cmeans"].set_linewidth(1.0)
    else:
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="#222", lw=1.5),
                        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                        flierprops=dict(marker=".", ms=2, alpha=0.4))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.8)

    if show_points:
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                       s=1.5, alpha=0.3, color=colors[i], linewidths=0)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=rotation, ha="right")
    ax.set_ylabel("log-normalized expression")
    ax.set_title(f"{gene} by {field}", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_dotplot(
    ds: Dataset,
    genes: list,
    group: str | None = None,
    cmap: str = "viridis",
    size_scale: float = 180,
    standard_scale: str | None = None,
):
    """Dot plot (mean expression + fraction expressing) of genes across groups.

    Returns a matplotlib ``Figure``.

    Parameters
    ----------
    genes : list[str]
        Genes to show as columns. Filtered to those present in the dataset.
    group : str, optional
        ``obs`` column for row grouping (defaults to ``ds.group_key``).
    cmap : str
        Matplotlib colormap for mean expression coloring.
    size_scale : float
        Maximum dot area in points² (fraction-expressing scales 0→this value).
    standard_scale : {None, "var", "group"}
        Normalize mean expression within genes (``"var"``) or within groups
        (``"group"``) to [0, 1] before coloring. ``None`` uses raw log-norm values.
    """
    from scipy import sparse as sp
    field = _group(ds, group)
    genes = [g for g in genes if g in ds.adata.var_names]
    if not genes:
        raise ValueError("none of the requested genes are present")
    if standard_scale not in (None, "var", "group"):
        raise ValueError(
            f"standard_scale must be None, 'var', or 'group', got {standard_scale!r}")
    cats = list(ds.adata.obs[field].astype("category").cat.categories)
    gidx = [ds.adata.var_names.get_loc(g) for g in genes]
    layer = ds.adata.layers["lognorm"] if "lognorm" in ds.adata.layers else ds.adata.X
    means = np.zeros((len(cats), len(genes)))
    fracs = np.zeros((len(cats), len(genes)))
    grp = ds.adata.obs[field].astype(str).values
    for i, c in enumerate(cats):
        m = grp == c
        sub = layer[m][:, gidx]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        means[i] = sub.mean(axis=0)
        fracs[i] = (sub > 0).mean(axis=0)

    if standard_scale == "var":
        mn, mx = means.min(axis=0), means.max(axis=0)
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        means = (means - mn) / rng
    elif standard_scale == "group":
        mn = means.min(axis=1, keepdims=True)
        mx = means.max(axis=1, keepdims=True)
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        means = (means - mn) / rng

    fig, ax = plt.subplots(figsize=(0.7 * len(genes) + 2.5, 0.4 * len(cats) + 1.5))
    X, Y = np.meshgrid(np.arange(len(genes)), np.arange(len(cats)))
    sizes = (fracs.ravel() * size_scale) + 5
    sc = ax.scatter(X.ravel(), Y.ravel(), s=sizes, c=means.ravel(),
                    cmap=cmap, linewidths=0.3, edgecolors="#444")
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=45,
                                                         ha="right", style="italic")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=7)
    scale_label = f" (scaled by {standard_scale})" if standard_scale else ""
    ax.set_title(f"Marker dot plot by {field}{scale_label}", fontsize=8.5)
    ax.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean expr", fontsize=7)
    for f, lbl in [(0.25, "25%"), (0.5, "50%"), (1.0, "100%")]:
        ax.scatter([], [], s=f * size_scale + 5, c="grey", label=lbl)
    ax.legend(title="% expressing", loc="upper left", bbox_to_anchor=(1.12, 1.0),
              frameon=False, fontsize=6, title_fontsize=6)
    fig.tight_layout()
    return fig


def plot_composition(
    ds: Dataset,
    group: str | None = None,
    split: str | None = None,
    normalize: bool = True,
    figsize: tuple = (7.6, 4.4),
    palette: dict | None = None,
    bar_width: float = 0.8,
    sort_groups: bool = False,
):
    """Stacked bar of ``group`` composition across ``split`` categories.

    Returns a matplotlib ``Figure``.

    Parameters
    ----------
    group : str, optional
        The cell-type / cluster column (defaults to ``ds.group_key``).
    split : str, optional
        The sample / condition column to split bars by. Auto-selected if None.
    normalize : bool
        Show fractions (True) or raw cell counts (False).
    figsize : tuple
        Figure (width, height) in inches.
    palette : dict, optional
        ``{group_label: color}`` override. Falls back to dataset palette.
    bar_width : float
        Width of each bar (0–1).
    sort_groups : bool
        Sort the stacked groups alphabetically (default False preserves
        category order from the AnnData object).
    """
    gk = _group(ds, group)
    if split is None:
        split = next((c for c in ds.categorical if c != gk), gk)
    df = ds.adata.obs[[split, gk]].astype(str)
    ct = df.groupby([split, gk]).size().unstack(fill_value=0)
    frac = ct.div(ct.sum(axis=1), axis=0) if normalize else ct
    if sort_groups:
        cats = sorted(ds.adata.obs[gk].astype("category").cat.categories)
    else:
        cats = list(ds.adata.obs[gk].astype("category").cat.categories)
    frac = frac[[c for c in cats if c in frac.columns]]
    cmap_dict = palette or io.category_colors(ds.adata, gk) or {}
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(frac)); x = np.arange(len(frac))
    for c in frac.columns:
        ax.bar(x, frac[c].values, bottom=bottom, width=bar_width,
               color=cmap_dict.get(c, None), label=c, linewidth=0)
        bottom += frac[c].values
    ax.set_xticks(x); ax.set_xticklabels(frac.index, rotation=30, ha="right")
    ax.set_ylabel("fraction of cells" if normalize else "cells")
    if normalize:
        ax.set_ylim(0, 1)
    ax.set_title(f"{gk} composition across {split}", fontsize=8.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=5.8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ tables
def markers_table(
    ds: Dataset,
    group: str | None = None,
    top_n: int | None = None,
    sort_by: str = "rank",
    ascending: bool = True,
) -> pd.DataFrame:
    """Return the per-group differential-expression (marker) table as a DataFrame.

    Columns: ``group``, ``gene``, ``rank`` and any score columns
    (``logfoldchanges``, ``pvals_adj``, ...) present in the prepared object.

    Parameters
    ----------
    group : str, optional
        Filter to a single group label.
    top_n : int, optional
        Keep the top-N genes per group.
    sort_by : str
        Column to sort by. Default ``"rank"``; other useful values:
        ``"logfoldchange"``, ``"pval_adj"``, ``"score"``.
    ascending : bool
        Sort direction.
    """
    mdf = io.markers_df(ds.adata)
    if mdf is None:
        raise ValueError("no marker/DE table in this dataset; run scPyviewer.prepare first")
    if group is not None:
        mdf = mdf[mdf["group"].astype(str) == str(group)]
    if sort_by in mdf.columns:
        mdf = mdf.sort_values(["group", sort_by], ascending=[True, ascending])
    else:
        mdf = mdf.sort_values(["group", "rank"])
    if top_n is not None:
        mdf = mdf.groupby("group", group_keys=False).head(top_n)
    return mdf.reset_index(drop=True)


def composition_table(ds: Dataset, group: str | None = None, split: str | None = None,
                      normalize: bool = True) -> pd.DataFrame:
    """Return the ``group`` x ``split`` composition matrix as a DataFrame."""
    gk = _group(ds, group)
    if split is None:
        split = next((c for c in ds.categorical if c != gk), gk)
    df = ds.adata.obs[[split, gk]].astype(str)
    ct = df.groupby([split, gk]).size().unstack(fill_value=0)
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0)
    return ct.reset_index()


def metadata_table(ds: Dataset) -> pd.DataFrame:
    """Return the per-cell metadata (``obs``) as a DataFrame."""
    return ds.adata.obs.copy()


# ------------------------------------------------------------------ exporters
def _save_fig(fig, base: str, formats, dpi: int = 200) -> list:
    paths = []
    for fmt in formats:
        p = f"{base}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def export_figures(
    ds: Dataset,
    outdir: str,
    formats=("png",),
    genes: list | None = None,
    dpi: int = 200,
) -> list:
    """Render the standard view set and write them to ``outdir``.

    Produces: embedding (by group), embedding (by top marker gene),
    multi-gene grid, violin, dot plot, and composition bar. Returns the list of
    written file paths.

    Parameters
    ----------
    outdir : str
        Directory to write figures into (created if absent).
    formats : sequence[str]
        Output formats — any of ``"png"``, ``"pdf"``, ``"svg"``.
    genes : list[str], optional
        Genes to highlight. Defaults to the top-ranked marker genes.
    dpi : int
        Dots-per-inch for raster formats (png). Has no effect on pdf/svg.
    """
    os.makedirs(outdir, exist_ok=True)
    formats = list(formats)
    try:
        mdf = io.markers_df(ds.adata)
        ranked = mdf.sort_values("rank")["gene"].tolist() if mdf is not None else []
    except Exception:
        ranked = []
    if genes is None:
        genes = ranked[:6] if ranked else list(ds.adata.var_names[:6])
    g1 = genes[0]
    written = []
    written += _save_fig(plot_embedding(ds, color=ds.group_key),
                         os.path.join(outdir, "embedding_group"), formats, dpi=dpi)
    written += _save_fig(plot_embedding(ds, gene=g1),
                         os.path.join(outdir, "embedding_gene"), formats, dpi=dpi)
    written += _save_fig(plot_multigene(ds, genes[:6]),
                         os.path.join(outdir, "multigene_grid"), formats, dpi=dpi)
    written += _save_fig(plot_violin(ds, g1),
                         os.path.join(outdir, "violin"), formats, dpi=dpi)
    written += _save_fig(plot_dotplot(ds, genes[:5]),
                         os.path.join(outdir, "dotplot"), formats, dpi=dpi)
    written += _save_fig(plot_composition(ds),
                         os.path.join(outdir, "composition"), formats, dpi=dpi)
    return written


def export_tables(
    ds: Dataset,
    outdir: str,
    formats=("csv",),
    top_n: int | None = 25,
) -> list:
    """Write the marker, composition, and metadata tables to ``outdir``.

    Parameters
    ----------
    outdir : str
        Directory to write tables into (created if absent).
    formats : sequence[str]
        Output formats — any of ``"csv"``, ``"tsv"``, ``"xlsx"``
        (xlsx requires ``openpyxl``).
    top_n : int, optional
        Limit marker table to the top N genes per group.

    Returns the list of written file paths.
    """
    os.makedirs(outdir, exist_ok=True)
    formats = list(formats)
    tables = {
        "markers": markers_table(ds, top_n=top_n),
        "composition": composition_table(ds),
        "metadata": metadata_table(ds),
    }
    written = []
    for name, tbl in tables.items():
        for fmt in formats:
            p = os.path.join(outdir, f"{name}.{fmt}")
            if fmt == "csv":
                tbl.to_csv(p, index=False)
            elif fmt == "tsv":
                tbl.to_csv(p, sep="\t", index=False)
            elif fmt == "xlsx":
                tbl.to_excel(p, index=False)
            else:
                raise ValueError(f"unsupported table format: {fmt}")
            written.append(p)
    return written


__all__ = [
    "Dataset", "load_dataset",
    "plot_embedding", "plot_multigene", "plot_violin", "plot_dotplot",
    "plot_composition",
    "markers_table", "composition_table", "metadata_table",
    "export_figures", "export_tables",
]

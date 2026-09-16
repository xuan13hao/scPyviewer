"""scPyviewer.api — public programmatic interface to scPyviewer.

Every ``plot_*`` function returns a :class:`matplotlib.figure.Figure`; every
``*_table`` function returns a :class:`pandas.DataFrame`. All plotting
functions expose fine-grained typography and layout controls so that
publication-quality figures can be produced without post-processing::

    import scPyviewer as sv

    ds = sv.load_dataset("data/chicken_heart.prepared.h5ad")

    # quick look
    fig = sv.plot_embedding(ds, color=ds.group_key)

    # publication-quality
    fig = sv.plot_embedding(
        ds, color=ds.group_key,
        figsize=(6, 5), dpi=300,
        title="Cell types — chicken heart",
        title_fontsize=14,
        label_fontsize=9,
        legend_fontsize=9,
        font_family="Arial",
    )
    fig.savefig("fig1.pdf", bbox_inches="tight")

Global style (applies to all subsequent plots in the session)::

    sv.set_style(font_family="Helvetica", base_fontsize=11, dpi=300)
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

from . import io_utils as io


# ------------------------------------------------------------------ style
@contextlib.contextmanager
def _font_ctx(family: str | None):
    """Temporarily switch font family inside a with-block."""
    if family:
        prev = mpl.rcParams.get("font.family", "sans-serif")
        mpl.rcParams["font.family"] = family
        try:
            yield
        finally:
            mpl.rcParams["font.family"] = prev
    else:
        yield


def set_style(
    font_family: str | None = None,
    base_fontsize: float | None = None,
    dpi: int | None = None,
    style: str | None = None,
) -> None:
    """Set global matplotlib defaults for all subsequent ``plot_*`` calls.

    Parameters
    ----------
    font_family : str, optional
        Font family name, e.g. ``"Arial"``, ``"Helvetica"``, ``"Times New Roman"``.
    base_fontsize : float, optional
        Base font size in points. Sets ``font.size`` in rcParams.
    dpi : int, optional
        Default figure resolution (dots per inch).
    style : str, optional
        Matplotlib style sheet, e.g. ``"seaborn-v0_8-whitegrid"``, ``"ggplot"``.
        Passed to :func:`matplotlib.pyplot.style.use`.

    Examples
    --------
    >>> sv.set_style(font_family="Arial", base_fontsize=11, dpi=300)
    """
    if style is not None:
        plt.style.use(style)
    if font_family is not None:
        mpl.rcParams["font.family"] = font_family
    if base_fontsize is not None:
        mpl.rcParams["font.size"] = base_fontsize
    if dpi is not None:
        mpl.rcParams["figure.dpi"] = dpi


# ------------------------------------------------------------------ handle
@dataclass
class Dataset:
    """A loaded, viewer-ready dataset.

    Attributes
    ----------
    adata : AnnData
    path : str
    group_key : str
    embeddings : list[str]
    categorical : list[str]
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

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Dataset({os.path.basename(self.path)}: "
                f"{self.n_obs:,} cells x {self.n_vars:,} genes, "
                f"group_key={self.group_key!r})")


def load_dataset(path: str) -> Dataset:
    """Load a prepared ``.h5ad`` and return a :class:`Dataset` handle.

    Parameters
    ----------
    path : str
        Path to a ``*.prepared.h5ad``. Run :mod:`scPyviewer.prepare` first
        for full functionality.
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
    # layout
    figsize: tuple = (7.4, 5.6),
    dpi: int = 150,
    # scatter
    point_size: float = 3.0,
    alpha: float = 0.75,
    cmap: str = "viridis",
    # labels & legend
    title: str | None = None,
    title_fontsize: float = 11,
    label_groups: bool = True,
    label_fontsize: float = 6,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlabel_fontsize: float = 9,
    ylabel_fontsize: float = 9,
    tick_fontsize: float = 8,
    show_legend: bool = True,
    legend_fontsize: float = 7,
    legend_title: str | None = None,
    legend_title_fontsize: float = 8,
    # colorbar (gene mode)
    colorbar_label: str | None = None,
    colorbar_fontsize: float = 8,
    # style
    font_family: str | None = None,
):
    """Scatter of a 2-D embedding colored by a metadata column or a gene.

    Parameters
    ----------
    color : str, optional
        ``obs`` column name (categorical or numeric).
    gene : str, optional
        Gene name; expression drives the color scale.
    embedding : str, optional
        Embedding key in ``obsm`` (e.g. ``"X_umap"``). Auto-selected if None.
    figsize : tuple
        Figure ``(width, height)`` in inches.
    dpi : int
        Figure resolution (dots per inch).
    point_size : float
        Scatter point diameter in points.
    alpha : float
        Point transparency (0–1) for categorical coloring.
    cmap : str
        Matplotlib colormap for continuous / gene coloring.
    title : str, optional
        Override the auto-generated title.
    title_fontsize : float
        Font size of the figure title.
    label_groups : bool
        Overlay centroid text labels when coloring by category.
    label_fontsize : float
        Font size of centroid group labels.
    xlabel / ylabel : str, optional
        Override axis labels (default: embedding axis name).
    xlabel_fontsize / ylabel_fontsize : float
        Axis-label font sizes.
    tick_fontsize : float
        Tick-label font size.
    show_legend : bool
        Show legend for categorical coloring when ``label_groups=False``.
    legend_fontsize : float
        Legend entry font size.
    legend_title : str, optional
        Override legend title.
    legend_title_fontsize : float
        Legend title font size.
    colorbar_label : str, optional
        Override colorbar label in gene mode.
    colorbar_fontsize : float
        Colorbar label and tick font size.
    font_family : str, optional
        Font family for this figure only (e.g. ``"Arial"``).
    """
    key = _resolve_embedding(ds, embedding)
    xy = io.embedding_2d(ds.adata, key)
    lab = _emb_label(key)

    with _font_ctx(font_family):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if gene is not None:
            expr = io.gene_vector(ds.adata, gene, "lognorm")
            o = np.argsort(expr)
            sc = ax.scatter(xy[o, 0], xy[o, 1], c=expr[o], s=point_size,
                            cmap=cmap, linewidths=0, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(colorbar_label or "log-norm expression",
                         fontsize=colorbar_fontsize)
            cb.ax.tick_params(labelsize=colorbar_fontsize - 1)
            ax.set_title(title or f"{gene} expression ({lab})",
                         fontsize=title_fontsize, style="italic")
        else:
            field = color or ds.group_key
            cats = list(ds.adata.obs[field].astype("category").cat.categories)
            cmap_dict = io.category_colors(ds.adata, field) or {}
            for c in cats:
                m = (ds.adata.obs[field].astype(str) == c).values
                ax.scatter(xy[m, 0], xy[m, 1], s=point_size, alpha=alpha,
                           color=cmap_dict.get(c, None), label=c,
                           linewidths=0, rasterized=True)
                if label_groups:
                    cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
                    ax.text(cx, cy, c, fontsize=label_fontsize,
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.12",
                                      fc="white", ec="none", alpha=0.68))
            ax.set_title(title or f"{field} ({lab}, {ds.n_obs:,} cells)",
                         fontsize=title_fontsize)
            if show_legend and not label_groups:
                leg = ax.legend(
                    title=legend_title or field,
                    loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    frameon=False, fontsize=legend_fontsize,
                    title_fontsize=legend_title_fontsize,
                )

        ax.set_xlabel(xlabel or f"{lab}1", fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel or f"{lab}2", fontsize=ylabel_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
    return fig


def plot_multigene(
    ds: Dataset,
    genes: list,
    embedding: str | None = None,
    # layout
    ncol: int = 3,
    figsize: tuple | None = None,
    dpi: int = 150,
    # scatter
    point_size: float = 2.5,
    cmap: str = "viridis",
    alpha: float = 0.8,
    max_genes: int = 12,
    # labels
    suptitle: str | None = None,
    suptitle_fontsize: float = 10,
    title_fontsize: float | None = None,  # alias for suptitle_fontsize
    panel_title_fontsize: float = 8,
    tick_fontsize: float | None = None,   # sets colorbar tick size when given
    colorbar_fontsize: float = 6,
    # style
    font_family: str | None = None,
):
    """Grid of embedding scatters, one panel per gene.

    Parameters
    ----------
    genes : list[str]
        Gene names to plot.
    embedding : str, optional
        Embedding key (auto-selected if None).
    ncol : int
        Number of columns in the grid.
    figsize : tuple, optional
        Figure ``(width, height)`` in inches. Auto-computed from ncol/nrow if None.
    dpi : int
        Figure resolution.
    point_size : float
        Scatter point size.
    cmap : str
        Colormap for expression values.
    alpha : float
        Point transparency (0–1).
    max_genes : int
        Hard cap on number of panels shown.
    suptitle : str, optional
        Override the figure super-title.
    suptitle_fontsize : float
        Font size of the figure super-title.
    panel_title_fontsize : float
        Font size of each panel's gene-name title.
    colorbar_fontsize : float
        Colorbar tick and label font size.
    font_family : str, optional
        Font family for this figure only.
    """
    genes = [g for g in genes if g in ds.adata.var_names][:max_genes]
    if not genes:
        raise ValueError("none of the requested genes are present")
    key = _resolve_embedding(ds, embedding)
    xy = io.embedding_2d(ds.adata, key)
    nrow = int(np.ceil(len(genes) / ncol))
    sup_fs = title_fontsize if title_fontsize is not None else suptitle_fontsize
    auto_size = (3.0 * ncol, 2.8 * nrow)
    fs = figsize if figsize is not None else auto_size

    with _font_ctx(font_family):
        fig, axes = plt.subplots(nrow, ncol, figsize=fs, dpi=dpi)
        axes = np.array(axes).reshape(-1)
        for i, g in enumerate(genes):
            ax = axes[i]
            expr = io.gene_vector(ds.adata, g, "lognorm")
            o = np.argsort(expr)
            sc = ax.scatter(xy[o, 0], xy[o, 1], c=expr[o], s=point_size,
                            cmap=cmap, alpha=alpha, linewidths=0, rasterized=True)
            ax.set_title(g, fontsize=panel_title_fontsize, style="italic")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            cb_tick = tick_fontsize if tick_fontsize is not None else colorbar_fontsize - 0.5
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=cb_tick)
            cb.set_label("expr", fontsize=colorbar_fontsize)
        for j in range(len(genes), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(
            suptitle or f"Gene expression — {_emb_label(key)}",
            x=0.02, ha="left", fontsize=sup_fs,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_violin(
    ds: Dataset,
    gene: str,
    group: str | None = None,
    # layout
    figsize: tuple = (7.0, 4.0),
    dpi: int = 150,
    # style
    kind: str = "violin",
    palette: dict | None = None,
    rotation: int = 30,
    show_points: bool = False,
    # typography
    title: str | None = None,
    title_fontsize: float = 11,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlabel_fontsize: float = 9,
    ylabel_fontsize: float = 9,
    tick_fontsize: float = 8,
    font_family: str | None = None,
):
    """Violin or box plot of a gene's expression grouped by a metadata column.

    Parameters
    ----------
    gene : str
        Gene name.
    group : str, optional
        ``obs`` grouping column (defaults to ``ds.group_key``).
    figsize : tuple
        Figure ``(width, height)`` in inches.
    dpi : int
        Figure resolution.
    kind : {"violin", "box"}
        Plot type.
    palette : dict, optional
        ``{label: color}`` override.
    rotation : int
        X-tick label rotation in degrees.
    show_points : bool
        Overlay jittered individual data points.
    title : str, optional
        Override figure title.
    title_fontsize : float
        Title font size.
    xlabel / ylabel : str, optional
        Axis label overrides.
    xlabel_fontsize / ylabel_fontsize : float
        Axis label font sizes.
    tick_fontsize : float
        Tick-label font size.
    font_family : str, optional
        Font family for this figure only.
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

    with _font_ctx(font_family):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if kind == "violin":
            parts = ax.violinplot(data, showmeans=True, showextrema=False,
                                  widths=0.85)
            for i, b in enumerate(parts["bodies"]):
                b.set_facecolor(colors[i]); b.set_alpha(0.8); b.set_edgecolor("none")
            parts["cmeans"].set_color("#222"); parts["cmeans"].set_linewidth(1.0)
        else:
            bp = ax.boxplot(data, patch_artist=True,
                            medianprops=dict(color="#222", lw=1.5),
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
        ax.set_xticklabels(order, rotation=rotation, ha="right",
                           fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_xlabel(xlabel or field, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel or "log-normalized expression",
                      fontsize=ylabel_fontsize)
        ax.set_title(title or f"{gene} by {field}", fontsize=title_fontsize)
        fig.tight_layout()
    return fig


def plot_dotplot(
    ds: Dataset,
    genes: list,
    group: str | None = None,
    # layout
    figsize: tuple | None = None,
    dpi: int = 150,
    # rendering
    cmap: str = "viridis",
    size_scale: float = 180,
    standard_scale: str | None = None,
    # typography
    title: str | None = None,
    title_fontsize: float = 11,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlabel_fontsize: float = 8,
    ylabel_fontsize: float = 8,
    tick_fontsize: float = 7,
    legend_fontsize: float = 6,
    legend_title_fontsize: float = 6,
    colorbar_label: str | None = None,
    colorbar_fontsize: float = 7,
    gene_label_rotation: int = 45,
    font_family: str | None = None,
):
    """Dot plot: mean expression × fraction expressing across groups.

    Parameters
    ----------
    genes : list[str]
        Genes to show as columns.
    group : str, optional
        ``obs`` row-grouping column (defaults to ``ds.group_key``).
    figsize : tuple, optional
        Figure ``(width, height)`` in inches. Auto-computed if None.
    dpi : int
        Figure resolution.
    cmap : str
        Colormap for mean expression.
    size_scale : float
        Maximum dot area in points² (fraction-expressing scales 0→this).
    standard_scale : {None, "var", "group"}
        Normalize mean expression within genes (``"var"``) or groups
        (``"group"``). ``None`` uses raw log-norm values.
    title : str, optional
        Override figure title.
    title_fontsize : float
        Title font size.
    xlabel / ylabel : str, optional
        Axis label overrides.
    xlabel_fontsize / ylabel_fontsize : float
        Axis label font sizes.
    tick_fontsize : float
        Tick-label font size.
    legend_fontsize : float
        Dot-size legend font size.
    legend_title_fontsize : float
        Dot-size legend title font size.
    colorbar_label : str, optional
        Override colorbar label.
    colorbar_fontsize : float
        Colorbar label and tick font size.
    gene_label_rotation : int
        X-axis gene label rotation in degrees.
    font_family : str, optional
        Font family for this figure only.
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
        r = np.where(mx - mn > 0, mx - mn, 1.0)
        means = (means - mn) / r
    elif standard_scale == "group":
        mn = means.min(axis=1, keepdims=True)
        mx = means.max(axis=1, keepdims=True)
        r = np.where(mx - mn > 0, mx - mn, 1.0)
        means = (means - mn) / r

    auto_size = (0.7 * len(genes) + 2.5, 0.4 * len(cats) + 1.5)
    fs = figsize if figsize is not None else auto_size

    with _font_ctx(font_family):
        fig, ax = plt.subplots(figsize=fs, dpi=dpi)
        X, Y = np.meshgrid(np.arange(len(genes)), np.arange(len(cats)))
        sizes = (fracs.ravel() * size_scale) + 5
        sc = ax.scatter(X.ravel(), Y.ravel(), s=sizes, c=means.ravel(),
                        cmap=cmap, linewidths=0.3, edgecolors="#444")

        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=gene_label_rotation, ha="right",
                           style="italic", fontsize=tick_fontsize)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=tick_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

        scale_label = f" (scaled by {standard_scale})" if standard_scale else ""
        ax.set_title(title or f"Dot plot — {field}{scale_label}",
                     fontsize=title_fontsize)
        ax.set_xlabel(xlabel or "Gene", fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel or field, fontsize=ylabel_fontsize)
        ax.invert_yaxis()

        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(colorbar_label or "mean expr", fontsize=colorbar_fontsize)
        cb.ax.tick_params(labelsize=colorbar_fontsize - 1)

        for f, lbl in [(0.25, "25%"), (0.5, "50%"), (1.0, "100%")]:
            ax.scatter([], [], s=f * size_scale + 5, c="grey", label=lbl)
        ax.legend(
            title="% expressing",
            loc="upper left", bbox_to_anchor=(1.12, 1.0),
            frameon=False, fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
        )
        fig.tight_layout()
    return fig


def plot_composition(
    ds: Dataset,
    group: str | None = None,
    split: str | None = None,
    # layout
    figsize: tuple = (7.6, 4.4),
    dpi: int = 150,
    # rendering
    normalize: bool = True,
    palette: dict | None = None,
    bar_width: float = 0.8,
    sort_groups: bool = False,
    # typography
    title: str | None = None,
    title_fontsize: float = 11,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlabel_fontsize: float = 9,
    ylabel_fontsize: float = 9,
    tick_fontsize: float = 8,
    rotation: int = 30,
    legend_fontsize: float = 6,
    legend_title_fontsize: float = 7,
    font_family: str | None = None,
):
    """Stacked bar of ``group`` composition across ``split`` categories.

    Parameters
    ----------
    group : str, optional
        Cell-type / cluster column (defaults to ``ds.group_key``).
    split : str, optional
        Sample / condition column to split bars by. Auto-selected if None.
    figsize : tuple
        Figure ``(width, height)`` in inches.
    dpi : int
        Figure resolution.
    normalize : bool
        Show fractions (True) or raw counts (False).
    palette : dict, optional
        ``{label: color}`` override.
    bar_width : float
        Bar width (0–1).
    sort_groups : bool
        Sort stacked groups alphabetically.
    title : str, optional
        Override figure title.
    title_fontsize : float
        Title font size.
    xlabel / ylabel : str, optional
        Axis label overrides.
    xlabel_fontsize / ylabel_fontsize : float
        Axis label font sizes.
    tick_fontsize : float
        Tick-label font size.
    rotation : int
        X-tick label rotation in degrees.
    legend_fontsize : float
        Legend entry font size.
    legend_title_fontsize : float
        Legend title font size.
    font_family : str, optional
        Font family for this figure only.
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

    with _font_ctx(font_family):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        bottom = np.zeros(len(frac)); x = np.arange(len(frac))
        for c in frac.columns:
            ax.bar(x, frac[c].values, bottom=bottom, width=bar_width,
                   color=cmap_dict.get(c, None), label=c, linewidth=0)
            bottom += frac[c].values

        ax.set_xticks(x)
        ax.set_xticklabels(frac.index, rotation=rotation, ha="right",
                           fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_xlabel(xlabel or split, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel or ("fraction of cells" if normalize else "cells"),
                      fontsize=ylabel_fontsize)
        if normalize:
            ax.set_ylim(0, 1)
        ax.set_title(title or f"{gk} composition across {split}",
                     fontsize=title_fontsize)
        ax.legend(
            title=gk,
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
            frameon=False, fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
        )
        fig.tight_layout()
    return fig



# ------------------------------------------------------------------ new plots
def plot_heatmap(
    ds: Dataset,
    genes: list,
    group: str | None = None,
    # layout
    figsize: tuple | None = None,
    dpi: int = 150,
    # rendering
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    standard_scale: str | None = None,
    swap_axes: bool = False,
    show_group_bar: bool = True,
    # typography
    title: str | None = None,
    title_fontsize: float = 11,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlabel_fontsize: float = 9,
    ylabel_fontsize: float = 9,
    tick_fontsize: float = 7,
    group_label_fontsize: float = 7,
    colorbar_label: str | None = None,
    colorbar_fontsize: float = 8,
    font_family: str | None = None,
):
    """Expression heatmap: genes × cells, grouped by a metadata column.

    Better than ``sc.pl.heatmap``: direct ``figsize``/``dpi`` control, all
    font sizes exposed as parameters, per-figure ``font_family`` override,
    and explicit ``vmin``/``vmax`` for color scaling.

    Parameters
    ----------
    genes : list[str]
        Genes to show (rows when ``swap_axes=False``).
    group : str, optional
        ``obs`` column used to sort and annotate cells.
    figsize : tuple, optional
        Figure ``(width, height)`` in inches. Auto-computed if None.
    dpi : int
        Figure resolution.
    cmap : str
        Matplotlib colormap.
    vmin / vmax : float, optional
        Color-scale limits. Auto from data if None.
    standard_scale : {None, "var", "obs"}
        Normalize each gene (``"var"``) or each cell (``"obs"``) to [0, 1]
        before plotting.
    swap_axes : bool
        If True, genes on X and cells on Y.
    show_group_bar : bool
        Render a color-coded group annotation bar.
    title : str, optional
        Figure title.
    title_fontsize : float
        Title font size.
    xlabel / ylabel : str, optional
        Axis label overrides.
    xlabel_fontsize / ylabel_fontsize : float
        Axis label font sizes.
    tick_fontsize : float
        Tick-label font size.
    group_label_fontsize : float
        Font size of group labels on the annotation bar.
    colorbar_label : str, optional
        Override colorbar label.
    colorbar_fontsize : float
        Colorbar label and tick font size.
    font_family : str, optional
        Font family for this figure only.
    """
    from scipy import sparse as sp
    field = _group(ds, group)
    genes = [g for g in genes if g in ds.adata.var_names]
    if not genes:
        raise ValueError("none of the requested genes are present")

    cats = list(ds.adata.obs[field].astype("category").cat.categories)
    cmap_dict = io.category_colors(ds.adata, field) or {}
    order = np.concatenate([
        np.where(ds.adata.obs[field].astype(str) == c)[0] for c in cats
    ])
    gidx = [ds.adata.var_names.get_loc(g) for g in genes]
    layer = ds.adata.layers["lognorm"] if "lognorm" in ds.adata.layers else ds.adata.X
    mat = layer[order][:, gidx]
    mat = mat.toarray() if sp.issparse(mat) else np.asarray(mat)
    mat = mat.astype(float)

    if standard_scale == "var":
        mn, mx = mat.min(axis=0), mat.max(axis=0)
        mat = (mat - mn) / np.where(mx - mn > 0, mx - mn, 1.0)
    elif standard_scale == "obs":
        mn = mat.min(axis=1, keepdims=True)
        mx = mat.max(axis=1, keepdims=True)
        mat = (mat - mn) / np.where(mx - mn > 0, mx - mn, 1.0)

    if swap_axes:
        mat = mat.T

    n_genes, n_cells = len(genes), len(order)
    if swap_axes:
        n_genes, n_cells = n_cells, n_genes
    bar_h = 0.06 if show_group_bar else 0.0
    auto_w = max(5.0, n_cells * 0.02)
    auto_h = max(3.0, n_genes * 0.25) + bar_h * max(3.0, n_genes * 0.25)
    fs = figsize if figsize is not None else (auto_w, auto_h)

    height_ratios = [bar_h, 1 - bar_h] if show_group_bar and not swap_axes else [1]
    n_rows = 2 if show_group_bar and not swap_axes else 1

    with _font_ctx(font_family):
        fig, axes = plt.subplots(
            n_rows, 1, figsize=fs, dpi=dpi,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.01}
            if n_rows == 2 else {},
        )
        ax_bar = axes[0] if n_rows == 2 else None
        ax = axes[1] if n_rows == 2 else axes

        im = ax.imshow(mat if not swap_axes else mat,
                       aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        if swap_axes:
            ax.set_xticks(range(len(genes)))
            ax.set_xticklabels(genes, rotation=45, ha="right",
                               style="italic", fontsize=tick_fontsize)
            ax.set_yticks([])
            ax.set_xlabel(xlabel or "Gene", fontsize=xlabel_fontsize)
            ax.set_ylabel(ylabel or f"Cells ({n_cells:,})", fontsize=ylabel_fontsize)
        else:
            ax.set_yticks(range(len(genes)))
            ax.set_yticklabels(genes, style="italic", fontsize=tick_fontsize)
            ax.set_xticks([])
            ax.set_xlabel(xlabel or f"Cells ({len(order):,})", fontsize=xlabel_fontsize)
            ax.set_ylabel(ylabel or "Gene", fontsize=ylabel_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

        if ax_bar is not None:
            boundaries = [0] + list(np.cumsum(
                [np.sum(ds.adata.obs[field].astype(str) == c) for c in cats]))
            for i, c in enumerate(cats):
                color = cmap_dict.get(c, f"C{i}")
                ax_bar.barh(0, boundaries[i + 1] - boundaries[i],
                            left=boundaries[i], height=1,
                            color=color, linewidth=0)
                mid = (boundaries[i] + boundaries[i + 1]) / 2
                ax_bar.text(mid, 0.5, c, ha="center", va="center",
                            fontsize=group_label_fontsize, color="white",
                            fontweight="bold")
            ax_bar.set_xlim(0, len(order))
            ax_bar.axis("off")

        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label(colorbar_label or (
            f"scaled expr ({standard_scale})" if standard_scale
            else "log-norm expression"), fontsize=colorbar_fontsize)
        cb.ax.tick_params(labelsize=colorbar_fontsize - 1)

        ax.set_title(title or f"Expression heatmap — {field}",
                     fontsize=title_fontsize)
        fig.tight_layout()
    return fig


def plot_matrixplot(
    ds: Dataset,
    genes: list,
    group: str | None = None,
    # layout
    figsize: tuple | None = None,
    dpi: int = 150,
    # rendering
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    standard_scale: str | None = None,
    swap_axes: bool = False,
    annotate: bool = False,
    annotation_fmt: str = ".2f",
    annotation_fontsize: float = 7,
    # typography
    title: str | None = None,
    title_fontsize: float = 11,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlabel_fontsize: float = 9,
    ylabel_fontsize: float = 9,
    tick_fontsize: float = 8,
    gene_label_rotation: int = 45,
    colorbar_label: str | None = None,
    colorbar_fontsize: float = 8,
    font_family: str | None = None,
):
    """Mean-expression matrix heatmap: groups × genes.

    Better than ``sc.pl.matrixplot``: direct ``figsize``/``dpi`` and full
    font control. ``annotate=True`` overlays the numeric mean in each cell.

    Parameters
    ----------
    genes : list[str]
        Genes to include.
    group : str, optional
        ``obs`` column for row-grouping.
    figsize : tuple, optional
        Figure ``(width, height)`` in inches. Auto if None.
    dpi : int
        Resolution.
    cmap : str
        Colormap.
    vmin / vmax : float, optional
        Color-scale limits.
    standard_scale : {None, "var", "group"}
        Normalize per gene (``"var"``) or per group (``"group"``).
    swap_axes : bool
        Transpose: genes on Y, groups on X.
    annotate : bool
        Overlay numeric mean-expression values in each cell.
    annotation_fmt : str
        Python format string for annotations (e.g. ``".2f"``).
    annotation_fontsize : float
        Font size of annotation text.
    title : str, optional
        Figure title.
    title_fontsize : float
        Title font size.
    xlabel / ylabel : str, optional
        Axis label overrides.
    xlabel_fontsize / ylabel_fontsize : float
        Axis label font sizes.
    tick_fontsize : float
        Tick-label font size.
    gene_label_rotation : int
        Gene label rotation in degrees.
    colorbar_label : str, optional
        Override colorbar label.
    colorbar_fontsize : float
        Colorbar text size.
    font_family : str, optional
        Font family for this figure only.
    """
    from scipy import sparse as sp
    field = _group(ds, group)
    genes = [g for g in genes if g in ds.adata.var_names]
    if not genes:
        raise ValueError("none of the requested genes are present")
    cats = list(ds.adata.obs[field].astype("category").cat.categories)
    gidx = [ds.adata.var_names.get_loc(g) for g in genes]
    layer = ds.adata.layers["lognorm"] if "lognorm" in ds.adata.layers else ds.adata.X
    means = np.zeros((len(cats), len(genes)))
    grp = ds.adata.obs[field].astype(str).values
    for i, c in enumerate(cats):
        sub = layer[grp == c][:, gidx]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        means[i] = sub.mean(axis=0)

    if standard_scale == "var":
        mn, mx = means.min(axis=0), means.max(axis=0)
        means = (means - mn) / np.where(mx - mn > 0, mx - mn, 1.0)
    elif standard_scale == "group":
        mn = means.min(axis=1, keepdims=True)
        mx = means.max(axis=1, keepdims=True)
        means = (means - mn) / np.where(mx - mn > 0, mx - mn, 1.0)

    mat = means.T if swap_axes else means
    row_labels = genes if swap_axes else cats
    col_labels = cats if swap_axes else genes

    auto_w = max(3.5, len(col_labels) * 0.55 + 1.5)
    auto_h = max(3.0, len(row_labels) * 0.4 + 1.0)
    fs = figsize if figsize is not None else (auto_w, auto_h)

    with _font_ctx(font_family):
        fig, ax = plt.subplots(figsize=fs, dpi=dpi)
        im = ax.imshow(mat, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")

        if annotate:
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = mat[r, c]
                    text_color = "white" if val > (mat.max() * 0.6) else "black"
                    ax.text(c, r, format(val, annotation_fmt),
                            ha="center", va="center",
                            fontsize=annotation_fontsize, color=text_color)

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=gene_label_rotation,
                           ha="right", fontsize=tick_fontsize,
                           style="italic" if not swap_axes else "normal")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=tick_fontsize,
                           style="italic" if swap_axes else "normal")
        ax.tick_params(labelsize=tick_fontsize)
        ax.set_xlabel(xlabel or ("Cell type" if not swap_axes else "Gene"),
                      fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel or ("Gene" if not swap_axes else "Cell type"),
                      fontsize=ylabel_fontsize)

        scale_tag = f" (scaled by {standard_scale})" if standard_scale else ""
        ax.set_title(title or f"Mean expression — {field}{scale_tag}",
                     fontsize=title_fontsize)

        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(colorbar_label or "mean log-norm expr",
                     fontsize=colorbar_fontsize)
        cb.ax.tick_params(labelsize=colorbar_fontsize - 1)
        fig.tight_layout()
    return fig


def plot_stacked_violin(
    ds: Dataset,
    genes: list,
    group: str | None = None,
    # layout
    figsize: tuple | None = None,
    dpi: int = 150,
    swap_axes: bool = False,
    # rendering
    palette: dict | None = None,
    inner: str | None = "box",
    linewidth: float = 0.6,
    # typography
    title: str | None = None,
    title_fontsize: float = 11,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlabel_fontsize: float = 9,
    ylabel_fontsize: float = 9,
    tick_fontsize: float = 7,
    gene_label_fontsize: float = 8,
    rotation: int = 30,
    font_family: str | None = None,
):
    """Stacked violin plots: one violin per gene, grouped by a metadata column.

    Better than ``sc.pl.stacked_violin``: direct ``figsize``/``dpi``,
    full font control, and ``swap_axes`` to orient genes horizontally.

    Parameters
    ----------
    genes : list[str]
        Genes to show (one row per gene).
    group : str, optional
        ``obs`` grouping column.
    figsize : tuple, optional
        Figure ``(width, height)`` in inches. Auto if None.
    dpi : int
        Resolution.
    swap_axes : bool
        If True, genes on X-axis, groups on Y-axis.
    palette : dict, optional
        ``{label: color}`` override.
    inner : {None, "box", "point"}
        Inner marks inside each violin body.
    linewidth : float
        Violin body edge linewidth.
    title : str, optional
        Figure title.
    title_fontsize : float
        Title font size.
    xlabel / ylabel : str, optional
        Shared axis label overrides.
    xlabel_fontsize / ylabel_fontsize : float
        Shared axis label font sizes.
    tick_fontsize : float
        Tick-label font size.
    gene_label_fontsize : float
        Gene name label font size.
    rotation : int
        Group-label rotation in degrees (non-swapped mode).
    font_family : str, optional
        Font family for this figure only.
    """
    field = _group(ds, group)
    genes = [g for g in genes if g in ds.adata.var_names]
    if not genes:
        raise ValueError("none of the requested genes are present")
    order = sorted(ds.adata.obs[field].astype(str).unique())
    cmap_dict = palette or io.category_colors(ds.adata, field) or {}
    colors = [cmap_dict.get(s, f"C{i}") for i, s in enumerate(order)]
    grp = ds.adata.obs[field].astype(str).values

    n = len(genes)
    if swap_axes:
        auto_fs = (max(4, n * 0.9 + 1), max(3, len(order) * 0.5 + 1.5))
    else:
        auto_fs = (max(5, len(order) * 0.7 + 1.5), max(2, n * 1.2 + 0.8))
    fs = figsize if figsize is not None else auto_fs

    with _font_ctx(font_family):
        if swap_axes:
            fig, axes = plt.subplots(1, n, figsize=fs, dpi=dpi,
                                     sharey=True, squeeze=False)
            axes = axes[0]
        else:
            fig, axes = plt.subplots(n, 1, figsize=fs, dpi=dpi,
                                     sharex=True, squeeze=False)
            axes = [r[0] for r in axes]

        for i, (ax, g) in enumerate(zip(axes, genes)):
            expr = io.gene_vector(ds.adata, g, "lognorm")
            data = [expr[grp == s] for s in order]

            if swap_axes:
                parts = ax.violinplot(data, vert=True, showmeans=False,
                                      showextrema=False, widths=0.8)
            else:
                parts = ax.violinplot(data, vert=True, showmeans=False,
                                      showextrema=False, widths=0.8)

            for j, body in enumerate(parts["bodies"]):
                body.set_facecolor(colors[j])
                body.set_alpha(0.82)
                body.set_edgecolor("none")
                body.set_linewidth(linewidth)

            if inner == "box":
                for j, vals in enumerate(data):
                    q1, med, q3 = np.percentile(vals, [25, 50, 75]) if len(vals) else (0, 0, 0)
                    ax.plot([j + 1, j + 1], [q1, q3], color="#333", lw=linewidth * 2)
                    ax.scatter([j + 1], [med], color="white", s=8, zorder=3, linewidths=0)
            elif inner == "point":
                rng = np.random.default_rng(0)
                for j, vals in enumerate(data):
                    jitter = rng.uniform(-0.08, 0.08, len(vals))
                    ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                               s=1.5, alpha=0.35, color=colors[j], linewidths=0)

            ax.set_xticks(range(1, len(order) + 1))
            if swap_axes or i == n - 1:
                ax.set_xticklabels(order, rotation=rotation, ha="right",
                                   fontsize=tick_fontsize)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=tick_fontsize)
            ax.set_ylabel(g, fontsize=gene_label_fontsize, style="italic",
                          rotation=0 if swap_axes else 90,
                          labelpad=4, ha="right" if not swap_axes else "center")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        if not swap_axes:
            axes[-1].set_xlabel(xlabel or field, fontsize=xlabel_fontsize)
        else:
            fig.text(0.5, 0.02, xlabel or field, ha="center",
                     fontsize=xlabel_fontsize)
        fig.text(0.01, 0.5, ylabel or "log-norm expression", ha="center",
                 va="center", rotation=90, fontsize=ylabel_fontsize)

        fig.suptitle(title or f"Expression by {field}", fontsize=title_fontsize,
                     x=0.5, ha="center")
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    return fig


def plot_tracksplot(
    ds: Dataset,
    genes: list,
    group: str | None = None,
    # layout
    figsize: tuple | None = None,
    dpi: int = 150,
    # rendering
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_group_labels: bool = True,
    track_height: float = 0.4,
    # typography
    title: str | None = None,
    title_fontsize: float = 11,
    gene_label_fontsize: float = 8,
    group_label_fontsize: float = 8,
    colorbar_label: str | None = None,
    colorbar_fontsize: float = 8,
    font_family: str | None = None,
):
    """Track-style expression plot: one horizontal color strip per gene.

    Cells are sorted by group; vertical dividers mark group boundaries.
    Better than ``sc.pl.tracksplot``: full typography control, direct
    ``figsize``/``dpi``, and configurable track height.

    Parameters
    ----------
    genes : list[str]
        Genes to show (one track each).
    group : str, optional
        ``obs`` column used to sort and annotate cells.
    figsize : tuple, optional
        Figure ``(width, height)`` in inches. Auto if None.
    dpi : int
        Resolution.
    cmap : str
        Colormap for expression values.
    vmin / vmax : float, optional
        Color-scale limits.
    show_group_labels : bool
        Annotate group boundaries at the bottom.
    track_height : float
        Height of each gene track in inches.
    title : str, optional
        Figure title.
    title_fontsize : float
        Title font size.
    gene_label_fontsize : float
        Gene name label font size.
    group_label_fontsize : float
        Group boundary label font size.
    colorbar_label : str, optional
        Override colorbar label.
    colorbar_fontsize : float
        Colorbar text size.
    font_family : str, optional
        Font family for this figure only.
    """
    from scipy import sparse as sp
    field = _group(ds, group)
    genes = [g for g in genes if g in ds.adata.var_names]
    if not genes:
        raise ValueError("none of the requested genes are present")

    cats = list(ds.adata.obs[field].astype("category").cat.categories)
    order = np.concatenate([
        np.where(ds.adata.obs[field].astype(str) == c)[0] for c in cats
    ])
    group_sizes = [int(np.sum(ds.adata.obs[field].astype(str) == c)) for c in cats]
    boundaries = np.cumsum([0] + group_sizes)
    n_cells = len(order)

    gidx = [ds.adata.var_names.get_loc(g) for g in genes]
    layer = ds.adata.layers["lognorm"] if "lognorm" in ds.adata.layers else ds.adata.X
    mat = layer[order][:, gidx]
    mat = mat.toarray() if sp.issparse(mat) else np.asarray(mat, dtype=float)
    mat = mat.T  # genes × cells

    label_h = 0.5 if show_group_labels else 0.0
    auto_w = max(6.0, n_cells * 0.015)
    auto_h = len(genes) * track_height + label_h + 0.5
    fs = figsize if figsize is not None else (auto_w, auto_h)

    with _font_ctx(font_family):
        fig = plt.figure(figsize=fs, dpi=dpi)
        gs = fig.add_gridspec(
            len(genes), 1, hspace=0.04,
            left=0.12, right=0.88, top=0.92,
            bottom=label_h / fs[1] + 0.05,
        )
        axes = [fig.add_subplot(gs[i]) for i in range(len(genes))]

        im = None
        for i, (ax, g) in enumerate(zip(axes, genes)):
            expr = mat[i].reshape(1, -1)
            im = ax.imshow(expr, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            for b in boundaries[1:-1]:
                ax.axvline(b, color="white", lw=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(g, fontsize=gene_label_fontsize, style="italic",
                          rotation=0, ha="right", va="center", labelpad=4)

        if show_group_labels and len(axes) > 0:
            ax_bot = axes[-1]
            for j, c in enumerate(cats):
                mid = (boundaries[j] + boundaries[j + 1]) / 2
                fig.text(
                    0.12 + (mid / n_cells) * 0.76,
                    0.01 + label_h / fs[1] * 0.3,
                    c, ha="center", va="bottom",
                    fontsize=group_label_fontsize,
                )

        if im is not None:
            cbar_ax = fig.add_axes([0.89, 0.15, 0.015, 0.7])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label(colorbar_label or "log-norm expression",
                         fontsize=colorbar_fontsize)
            cb.ax.tick_params(labelsize=colorbar_fontsize - 1)

        fig.suptitle(title or f"Expression tracks — {field}",
                     fontsize=title_fontsize, y=0.97)
    return fig


def plot_correlation(
    ds: Dataset,
    group: str | None = None,
    genes: list | None = None,
    method: str = "pearson",
    # layout
    figsize: tuple | None = None,
    dpi: int = 150,
    # rendering
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = True,
    annotation_fmt: str = ".2f",
    annotation_fontsize: float = 8,
    # typography
    title: str | None = None,
    title_fontsize: float = 11,
    tick_fontsize: float = 9,
    colorbar_label: str | None = None,
    colorbar_fontsize: float = 8,
    rotation: int = 45,
    font_family: str | None = None,
):
    """Pairwise correlation matrix between groups (based on mean expression).

    Better than ``sc.pl.correlation_matrix``: ``"pearson"`` or ``"spearman"``,
    value annotations in every cell, ``vmin``/``vmax`` control, and full
    font/size control.

    Parameters
    ----------
    group : str, optional
        ``obs`` column whose categories become the matrix rows/cols.
    genes : list[str], optional
        Subset of genes used to compute correlations. Defaults to all genes.
    method : {"pearson", "spearman"}
        Correlation method.
    figsize : tuple, optional
        Figure ``(width, height)`` in inches. Auto if None.
    dpi : int
        Resolution.
    cmap : str
        Diverging colormap (e.g. ``"vlag"``, ``"RdBu_r"``).
    vmin / vmax : float
        Color-scale limits (default ±1).
    annotate : bool
        Overlay correlation values in each cell.
    annotation_fmt : str
        Format string for annotations.
    annotation_fontsize : float
        Annotation font size.
    title : str, optional
        Figure title.
    title_fontsize : float
        Title font size.
    tick_fontsize : float
        Tick-label font size.
    colorbar_label : str, optional
        Override colorbar label.
    colorbar_fontsize : float
        Colorbar text size.
    rotation : int
        X-tick label rotation in degrees.
    font_family : str, optional
        Font family for this figure only.
    """
    from scipy import sparse as sp
    from scipy.stats import spearmanr

    field = _group(ds, group)
    cats = list(ds.adata.obs[field].astype("category").cat.categories)
    layer = ds.adata.layers["lognorm"] if "lognorm" in ds.adata.layers else ds.adata.X
    grp = ds.adata.obs[field].astype(str).values

    if genes is not None:
        genes = [g for g in genes if g in ds.adata.var_names]
        gidx = [ds.adata.var_names.get_loc(g) for g in genes]
    else:
        gidx = list(range(ds.adata.n_vars))

    means = np.zeros((len(cats), len(gidx)))
    for i, c in enumerate(cats):
        sub = layer[grp == c][:, gidx]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub, dtype=float)
        means[i] = sub.mean(axis=0)

    n = len(cats)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if method == "spearman":
                r, _ = spearmanr(means[i], means[j])
            else:
                denom = np.std(means[i]) * np.std(means[j])
                r = float(np.corrcoef(means[i], means[j])[0, 1]) if denom > 0 else 0.0
            corr[i, j] = corr[j, i] = r

    auto_n = max(4.0, n * 0.7 + 1.5)
    fs = figsize if figsize is not None else (auto_n, auto_n)

    with _font_ctx(font_family):
        fig, ax = plt.subplots(figsize=fs, dpi=dpi)
        im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="auto")

        if annotate:
            for i in range(n):
                for j in range(n):
                    text_color = "white" if abs(corr[i, j]) > 0.6 else "black"
                    ax.text(j, i, format(corr[i, j], annotation_fmt),
                            ha="center", va="center",
                            fontsize=annotation_fontsize, color=text_color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=rotation, ha="right",
                           fontsize=tick_fontsize)
        ax.set_yticks(range(n))
        ax.set_yticklabels(cats, fontsize=tick_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

        method_label = method.capitalize()
        ax.set_title(title or f"{method_label} correlation — {field}",
                     fontsize=title_fontsize)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(colorbar_label or f"{method_label} r",
                     fontsize=colorbar_fontsize)
        cb.ax.tick_params(labelsize=colorbar_fontsize - 1)
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
    """Return the per-group DE / marker table as a DataFrame.

    Parameters
    ----------
    group : str, optional
        Filter to a single group label.
    top_n : int, optional
        Keep the top-N genes per group.
    sort_by : str
        Column to sort by. Options: ``"rank"``, ``"logfoldchange"``,
        ``"pval_adj"``, ``"score"``.
    ascending : bool
        Sort direction.
    """
    mdf = io.markers_df(ds.adata)
    if mdf is None:
        raise ValueError("no marker/DE table; run scPyviewer.prepare first")
    if group is not None:
        mdf = mdf[mdf["group"].astype(str) == str(group)]
    if sort_by in mdf.columns:
        mdf = mdf.sort_values(["group", sort_by], ascending=[True, ascending])
    else:
        mdf = mdf.sort_values(["group", "rank"])
    if top_n is not None:
        mdf = mdf.groupby("group", group_keys=False).head(top_n)
    return mdf.reset_index(drop=True)


def composition_table(
    ds: Dataset,
    group: str | None = None,
    split: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Return the ``group`` × ``split`` composition matrix as a DataFrame."""
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
    figsize: tuple | None = None,
    title_fontsize: float = 11,
    tick_fontsize: float = 8,
    label_fontsize: float = 9,
    font_family: str | None = None,
) -> list:
    """Render the standard view set and write them to ``outdir``.

    Produces: embedding (by group), embedding (by top gene), multi-gene grid,
    violin, dot plot, composition. Returns the list of written file paths.

    Parameters
    ----------
    outdir : str
        Output directory (created if absent).
    formats : sequence[str]
        Output formats: ``"png"``, ``"pdf"``, ``"svg"``.
    genes : list[str], optional
        Genes to highlight. Defaults to top-ranked marker genes.
    dpi : int
        Raster resolution.
    figsize : tuple, optional
        Override figure size for all plots (passed to each ``plot_*`` call).
    title_fontsize : float
        Title font size applied to all plots.
    tick_fontsize : float
        Tick-label font size applied to all plots.
    label_fontsize : float
        Axis-label font size applied to all plots.
    font_family : str, optional
        Font family applied to all plots.
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

    common = dict(dpi=dpi, title_fontsize=title_fontsize,
                  tick_fontsize=tick_fontsize, font_family=font_family)
    fs_kw = dict(figsize=figsize) if figsize else {}

    written = []
    written += _save_fig(
        plot_embedding(ds, color=ds.group_key, **common, **fs_kw),
        os.path.join(outdir, "embedding_group"), formats, dpi=dpi)
    written += _save_fig(
        plot_embedding(ds, gene=g1, **common, **fs_kw),
        os.path.join(outdir, "embedding_gene"), formats, dpi=dpi)
    written += _save_fig(
        plot_multigene(ds, genes[:6], **common, **fs_kw),
        os.path.join(outdir, "multigene_grid"), formats, dpi=dpi)
    written += _save_fig(
        plot_violin(ds, g1, xlabel_fontsize=label_fontsize,
                    ylabel_fontsize=label_fontsize, **common, **fs_kw),
        os.path.join(outdir, "violin"), formats, dpi=dpi)
    written += _save_fig(
        plot_dotplot(ds, genes[:5], xlabel_fontsize=label_fontsize,
                     ylabel_fontsize=label_fontsize, **common, **fs_kw),
        os.path.join(outdir, "dotplot"), formats, dpi=dpi)
    written += _save_fig(
        plot_composition(ds, xlabel_fontsize=label_fontsize,
                         ylabel_fontsize=label_fontsize, **common, **fs_kw),
        os.path.join(outdir, "composition"), formats, dpi=dpi)
    return written


def export_tables(
    ds: Dataset,
    outdir: str,
    formats=("csv",),
    top_n: int | None = 25,
) -> list:
    """Write marker, composition, and metadata tables to ``outdir``.

    Parameters
    ----------
    outdir : str
        Output directory (created if absent).
    formats : sequence[str]
        Output formats: ``"csv"``, ``"tsv"``, ``"xlsx"``
        (xlsx requires ``openpyxl``).
    top_n : int, optional
        Limit marker table to top-N genes per group.

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
    "set_style",
    "plot_embedding", "plot_multigene", "plot_violin", "plot_dotplot",
    "plot_composition",
    "markers_table", "composition_table", "metadata_table",
    "export_figures", "export_tables",
]

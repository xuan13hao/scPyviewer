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
    panel_title_fontsize: float = 8,
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
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=colorbar_fontsize - 0.5)
            cb.set_label("expr", fontsize=colorbar_fontsize)
        for j in range(len(genes), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(
            suptitle or f"Gene expression — {_emb_label(key)}",
            x=0.02, ha="left", fontsize=suptitle_fontsize,
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

"""scviewer.api — a public, programmatic interface to scviewer.

This module exposes the same views the interactive Streamlit app offers as
plain Python functions that **return objects** (matplotlib ``Figure`` and pandas
``DataFrame``) and helpers that **write files** (figures and tables). It lets a
scanpy user drive scviewer from a script or notebook without launching the UI::

    import scviewer as sv

    ds = sv.load_dataset("data/chicken_heart.prepared.h5ad")
    fig = sv.plot_embedding(ds, color=ds.group_key)      # matplotlib Figure
    fig.savefig("umap.png", dpi=200)

    markers = sv.markers_table(ds)                        # pandas DataFrame
    sv.export_figures(ds, "out/figs")                     # writes png + pdf
    sv.export_tables(ds, "out/tables")                    # writes csv

The plotting functions here render with matplotlib for reliable static export
(PNG/PDF/SVG) with no extra system dependencies. The interactive Plotly views
used by the Streamlit app live in :mod:`scviewer.plots`.
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
        :mod:`scviewer.prepare` first for full functionality).
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
def plot_embedding(ds: Dataset, color: str | None = None, gene: str | None = None,
                   embedding: str | None = None, point_size: float = 3.0,
                   figsize=(7.4, 5.6), label_groups: bool = True):
    """Scatter of a 2-D embedding, colored by a metadata column or a gene.

    Exactly one of ``color`` (an ``obs`` column) or ``gene`` should be given.
    If neither is given, colors by ``ds.group_key``. Returns a matplotlib
    ``Figure``.
    """
    key = _resolve_embedding(ds, embedding)
    xy = io.embedding_2d(ds.adata, key)
    lab = _emb_label(key)
    fig, ax = plt.subplots(figsize=figsize)

    if gene is not None:
        expr = io.gene_vector(ds.adata, gene, "lognorm")
        o = np.argsort(expr)
        sc = ax.scatter(xy[o, 0], xy[o, 1], c=expr[o], s=point_size, cmap="viridis",
                        linewidths=0, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("log-normalized expression", fontsize=8)
        ax.set_title(f"{gene} expression ({lab})", fontsize=9, style="italic")
    else:
        field = color or ds.group_key
        cats = list(ds.adata.obs[field].astype("category").cat.categories)
        cmap = io.category_colors(ds.adata, field) or {}
        for c in cats:
            m = (ds.adata.obs[field].astype(str) == c).values
            ax.scatter(xy[m, 0], xy[m, 1], s=point_size, alpha=0.75,
                       color=cmap.get(c, None), label=c, linewidths=0, rasterized=True)
            if label_groups:
                cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
                ax.text(cx, cy, c, fontsize=5.6, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                  alpha=0.68))
        ax.set_title(f"{field} ({lab}, {ds.n_obs:,} cells)", fontsize=9)
        if not label_groups:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      frameon=False, fontsize=5.8)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def plot_multigene(ds: Dataset, genes: list, embedding: str | None = None,
                   ncol: int = 3, point_size: float = 2.5):
    """Grid of embedding scatters, one panel per gene (up to 12). Returns a Figure."""
    genes = [g for g in genes if g in ds.adata.var_names][:12]
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
        sc = ax.scatter(xy[o, 0], xy[o, 1], c=expr[o], s=point_size, cmap="viridis",
                        linewidths=0, rasterized=True)
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


def plot_violin(ds: Dataset, gene: str, group: str | None = None,
                figsize=(7.0, 4.0)):
    """Violin of a gene's expression grouped by a metadata column. Returns a Figure."""
    field = _group(ds, group)
    order = sorted(ds.adata.obs[field].astype(str).unique())
    expr = io.gene_vector(ds.adata, gene, "lognorm")
    groups = ds.adata.obs[field].astype(str).values
    data = [expr[groups == s] for s in order]
    cmap = io.category_colors(ds.adata, field) or {}
    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.85)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(cmap.get(order[i], "#4d9221")); b.set_alpha(0.8)
        b.set_edgecolor("none")
    parts["cmeans"].set_color("#222"); parts["cmeans"].set_linewidth(1.0)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("log-normalized expression")
    ax.set_title(f"{gene} by {field}", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_dotplot(ds: Dataset, genes: list, group: str | None = None):
    """Dot plot (mean expression + fraction expressing) of genes across groups.

    Returns a matplotlib ``Figure``.
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
    fracs = np.zeros((len(cats), len(genes)))
    grp = ds.adata.obs[field].astype(str).values
    for i, c in enumerate(cats):
        m = grp == c
        sub = layer[m][:, gidx]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        means[i] = sub.mean(axis=0)
        fracs[i] = (sub > 0).mean(axis=0)
    fig, ax = plt.subplots(figsize=(0.7 * len(genes) + 2.5, 0.4 * len(cats) + 1.5))
    X, Y = np.meshgrid(np.arange(len(genes)), np.arange(len(cats)))
    sizes = (fracs.ravel() * 180) + 5
    sc = ax.scatter(X.ravel(), Y.ravel(), s=sizes, c=means.ravel(),
                    cmap="viridis", linewidths=0.3, edgecolors="#444")
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=45,
                                                         ha="right", style="italic")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=7)
    ax.set_title(f"Marker dot plot by {field}", fontsize=8.5)
    ax.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean expr", fontsize=7)
    # size legend
    for f, lbl in [(0.25, "25%"), (0.5, "50%"), (1.0, "100%")]:
        ax.scatter([], [], s=f * 180 + 5, c="grey", label=lbl)
    ax.legend(title="% expressing", loc="upper left", bbox_to_anchor=(1.12, 1.0),
              frameon=False, fontsize=6, title_fontsize=6)
    fig.tight_layout()
    return fig


def plot_composition(ds: Dataset, group: str | None = None, split: str | None = None,
                     normalize: bool = True, figsize=(7.6, 4.4)):
    """Stacked bar of ``group`` composition across ``split`` categories. Returns a Figure."""
    gk = _group(ds, group)
    if split is None:
        split = next((c for c in ds.categorical if c != gk), gk)
    df = ds.adata.obs[[split, gk]].astype(str)
    ct = df.groupby([split, gk]).size().unstack(fill_value=0)
    frac = ct.div(ct.sum(axis=1), axis=0) if normalize else ct
    cats = list(ds.adata.obs[gk].astype("category").cat.categories)
    frac = frac[[c for c in cats if c in frac.columns]]
    cmap = io.category_colors(ds.adata, gk) or {}
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(frac)); x = np.arange(len(frac))
    for c in frac.columns:
        ax.bar(x, frac[c].values, bottom=bottom, width=0.8,
               color=cmap.get(c, None), label=c, linewidth=0)
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
def markers_table(ds: Dataset, group: str | None = None, top_n: int | None = None
                  ) -> pd.DataFrame:
    """Return the per-group differential-expression (marker) table as a DataFrame.

    Columns: ``group``, ``gene``, ``rank`` and any score columns
    (``logfoldchanges``, ``pvals_adj``, ...) present in the prepared object.
    ``group`` filters to a single group; ``top_n`` keeps the top-ranked genes
    per group.
    """
    mdf = io.markers_df(ds.adata)
    if mdf is None:
        raise ValueError("no marker/DE table in this dataset; run scviewer.prepare first")
    if group is not None:
        mdf = mdf[mdf["group"].astype(str) == str(group)]
    if top_n is not None:
        mdf = (mdf.sort_values(["group", "rank"])
                  .groupby("group", group_keys=False).head(top_n))
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
def _save_fig(fig, base: str, formats) -> list:
    paths = []
    for fmt in formats:
        p = f"{base}.{fmt}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def export_figures(ds: Dataset, outdir: str, formats=("png",),
                   genes: list | None = None) -> list:
    """Render the standard view set and write them to ``outdir``.

    Produces: embedding (by group), embedding (by top marker gene),
    multi-gene grid, violin, dot plot, and composition bar. Returns the list of
    written file paths. ``formats`` may include ``png``, ``pdf``, ``svg``.
    """
    os.makedirs(outdir, exist_ok=True)
    formats = list(formats)
    # pick representative genes from the marker table when not supplied
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
                         os.path.join(outdir, "embedding_group"), formats)
    written += _save_fig(plot_embedding(ds, gene=g1),
                         os.path.join(outdir, "embedding_gene"), formats)
    written += _save_fig(plot_multigene(ds, genes[:6]),
                         os.path.join(outdir, "multigene_grid"), formats)
    written += _save_fig(plot_violin(ds, g1),
                         os.path.join(outdir, "violin"), formats)
    written += _save_fig(plot_dotplot(ds, genes[:5]),
                         os.path.join(outdir, "dotplot"), formats)
    written += _save_fig(plot_composition(ds),
                         os.path.join(outdir, "composition"), formats)
    return written


def export_tables(ds: Dataset, outdir: str, formats=("csv",),
                  top_n: int | None = 25) -> list:
    """Write the marker, composition, and metadata tables to ``outdir``.

    ``formats`` may include ``csv``, ``tsv``, ``xlsx`` (xlsx needs ``openpyxl``).
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

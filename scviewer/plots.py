"""
plots.py — Plotly figure builders for the scviewer app. Pure functions:
take an AnnData (+ selections) and return a plotly.graph_objects.Figure.
No Streamlit dependency, so the same builders drive the app and headless
figure export.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import io_utils as io


_CONTINUOUS = "Viridis"
_POINT = dict(size=3, opacity=0.75)


def embedding_scatter(adata, embedding: str, color_field: str | None = None,
                      gene: str | None = None, layer: str = "lognorm",
                      title: str | None = None, max_points: int | None = None):
    """Scatter of a 2-D embedding, colored by a metadata field OR a gene.

    If both color_field and gene are None, points are drawn in a single grey.
    max_points optionally subsamples for responsiveness (kept deterministic).
    """
    xy = io.embedding_2d(adata, embedding)
    n = xy.shape[0]
    idx = np.arange(n)
    if max_points and n > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, max_points, replace=False))
        xy = xy[idx]

    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
    lab = {"x": f"{embedding.replace('X_', '').upper()} 1",
           "y": f"{embedding.replace('X_', '').upper()} 2"}

    if gene:
        df["expr"] = io.gene_vector(adata, gene, layer)[idx]
        fig = px.scatter(df, x="x", y="y", color="expr",
                         color_continuous_scale=_CONTINUOUS, labels=lab,
                         title=title or f"{gene} ({embedding.replace('X_', '').upper()})")
        fig.update_coloraxes(colorbar_title="expr")
    elif color_field:
        vals = adata.obs[color_field].values[idx]
        is_cat = adata.obs[color_field].dtype.name in ("category", "object", "bool")
        if is_cat:
            df["c"] = np.asarray(vals).astype(str)
            cmap = io.category_colors(adata, color_field)
            fig = px.scatter(df, x="x", y="y", color="c", labels=lab,
                             color_discrete_map=cmap,
                             category_orders={"c": sorted(df["c"].unique())},
                             title=title or f"{color_field} ({embedding.replace('X_', '').upper()})")
            fig.update_layout(legend_title_text=color_field)
        else:
            df["c"] = np.asarray(vals, dtype=float)
            fig = px.scatter(df, x="x", y="y", color="c",
                             color_continuous_scale=_CONTINUOUS, labels=lab,
                             title=title or f"{color_field} ({embedding.replace('X_', '').upper()})")
            fig.update_coloraxes(colorbar_title=color_field)
    else:
        fig = px.scatter(df, x="x", y="y", labels=lab, title=title)
        fig.update_traces(marker_color="lightgrey")

    fig.update_traces(marker=_POINT)
    fig.update_layout(template="plotly_white", height=560,
                      legend=dict(itemsizing="constant"))
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)
    return fig


def multi_gene_grid(adata, embedding: str, genes: list[str], layer: str = "lognorm",
                    ncol: int = 3, max_points: int | None = None):
    """Grid of embedding scatters, one panel per gene, shared layout."""
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        return go.Figure()
    ncol = min(ncol, len(genes))
    nrow = int(np.ceil(len(genes) / ncol))

    xy = io.embedding_2d(adata, embedding)
    n = xy.shape[0]
    idx = np.arange(n)
    if max_points and n > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, max_points, replace=False))
        xy = xy[idx]

    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=genes,
                        horizontal_spacing=0.04, vertical_spacing=0.08)
    for i, g in enumerate(genes):
        r, c = divmod(i, ncol)
        expr = io.gene_vector(adata, g, layer)[idx]
        fig.add_trace(go.Scattergl(
            x=xy[:, 0], y=xy[:, 1], mode="markers",
            marker=dict(size=2.5, color=expr, colorscale=_CONTINUOUS,
                        showscale=(i == 0), colorbar=dict(title="expr", len=0.9)),
            showlegend=False, hovertext=[f"{g}: {v:.2f}" for v in expr],
            hoverinfo="text"), row=r + 1, col=c + 1)
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)
    fig.update_layout(template="plotly_white", height=300 * nrow,
                      title_text=f"Expression on {embedding.replace('X_', '').upper()}")
    return fig


def violin(adata, genes: list[str], group_field: str, layer: str = "lognorm",
           kind: str = "violin", max_cells_per_group: int = 10_000):
    """Violin (or box) of gene expression grouped by a metadata field.

    Caps each group at max_cells_per_group cells so large datasets don't OOM
    when building the long-format DataFrame.
    """
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        return go.Figure()
    groups = np.asarray(adata.obs[group_field].astype(str).values)
    cmap = io.category_colors(adata, group_field)
    order = sorted(pd.unique(groups))

    # Build a per-group subsample index to cap memory usage.
    rng = np.random.default_rng(0)
    keep_idx: list[np.ndarray] = []
    for grp in order:
        idx = np.where(groups == grp)[0]
        if len(idx) > max_cells_per_group:
            idx = rng.choice(idx, max_cells_per_group, replace=False)
        keep_idx.append(idx)
    sel = np.sort(np.concatenate(keep_idx))
    sel_groups = groups[sel]

    frames = []
    for g in genes:
        vec = io.gene_vector(adata, g, layer)[sel]
        frames.append(pd.DataFrame({"expr": vec, "group": sel_groups, "gene": g}))
    long = pd.concat(frames, ignore_index=True)

    if len(genes) == 1:
        maker = px.violin if kind == "violin" else px.box
        fig = maker(long, x="group", y="expr", color="group",
                    color_discrete_map=cmap, category_orders={"group": order},
                    points=False, title=f"{genes[0]} by {group_field}")
        if kind == "violin":
            fig.update_traces(meanline_visible=True, spanmode="hard")
        fig.update_layout(showlegend=False)
    else:
        maker = px.violin if kind == "violin" else px.box
        fig = maker(long, x="gene", y="expr", color="group",
                    color_discrete_map=cmap, category_orders={"group": order},
                    points=False, title=f"Expression by {group_field}")
    fig.update_layout(template="plotly_white", height=480, xaxis_title="",
                      yaxis_title="log-normalized expression")
    fig.update_xaxes(tickangle=-40)
    return fig


def dotplot(adata, genes: list[str], group_field: str, layer: str = "lognorm"):
    """Dotplot: mean expression (color) and fraction expressing (size) per group.

    Stats are computed gene-by-gene so only one gene vector is in memory at a time.
    """
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        return go.Figure()
    groups = np.asarray(adata.obs[group_field].astype(str).values)
    order = sorted(pd.unique(groups))
    # Pre-compute group masks once (boolean arrays are cheap).
    group_masks = {grp: groups == grp for grp in order}
    rows = []
    for g in genes:
        v = io.gene_vector(adata, g, layer)  # one gene at a time
        for grp in order:
            sub = v[group_masks[grp]]
            rows.append({"gene": g, "group": grp,
                         "mean_expr": float(sub.mean()) if sub.size else 0.0,
                         "frac": float((sub > 0).mean()) if sub.size else 0.0})
    d = pd.DataFrame(rows)
    fig = px.scatter(d, x="gene", y="group", color="mean_expr", size="frac",
                     color_continuous_scale=_CONTINUOUS, size_max=18,
                     title=f"Dotplot by {group_field}")
    fig.update_layout(template="plotly_white",
                      height=max(320, 26 * len(order)), xaxis_title="", yaxis_title="")
    fig.update_xaxes(tickangle=-40)
    return fig


def composition_bar(adata, group_field: str, split_field: str, normalize: bool = True):
    """Stacked bar of group_field composition across split_field categories."""
    df = adata.obs[[group_field, split_field]].astype(str)
    ct = df.groupby([split_field, group_field]).size().reset_index(name="n")
    if normalize:
        tot = ct.groupby(split_field)["n"].transform("sum")
        ct["frac"] = ct["n"] / tot
        y = "frac"
    else:
        y = "n"
    cmap = io.category_colors(adata, group_field)
    fig = px.bar(ct, x=split_field, y=y, color=group_field, color_discrete_map=cmap,
                 title=f"{group_field} composition by {split_field}")
    fig.update_layout(template="plotly_white", height=480,
                      yaxis_title=("fraction" if normalize else "cells"))
    fig.update_xaxes(tickangle=-40)
    return fig

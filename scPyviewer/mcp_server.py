"""scPyviewer MCP server — drive scPyviewer from an LLM agent.

Exposes dataset discovery, gene/marker queries, and publication-quality plots
(returned as PNG images) over the Model Context Protocol, so a client such as
Claude Code / Claude Desktop can explore single-cell ``.h5ad`` data directly.

Run as a stdio server::

    scpyviewer-mcp

Then register it with your MCP client (see SKILL.md for the config snippet).
"""
from __future__ import annotations

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcp.server.fastmcp import FastMCP, Image

from . import api as _api
from . import io_utils as _io

mcp = FastMCP(
    "scPyviewer",
    instructions=(
        "Explore single-cell AnnData (.h5ad) datasets. Start with list_datasets "
        "(or pass an explicit path), then dataset_info to learn the grouping / "
        "embeddings / metadata fields, search_genes to find genes, and the plot_* "
        "tools (which return PNG images) or top_markers / composition (tables). "
        "Every path is a prepared .h5ad; run `scpyviewer-prepare` first if needed."
    ),
)

# One consistent publication style for every figure the server returns.
_api.set_publication_style()

_CACHE: dict[str, _api.Dataset] = {}


def _load(path: str) -> _api.Dataset:
    path = os.path.expanduser(path)
    if path not in _CACHE:
        _CACHE[path] = _api.load_dataset(path)
    return _CACHE[path]


def _png(fig) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


# ------------------------------------------------------------------ discovery
@mcp.tool()
def list_datasets(data_dir: str = "") -> list[dict]:
    """List prepared .h5ad datasets in a directory.

    Defaults to $SCPYVIEWER_DATA_DIR, or ./data. Returns name + path for each.
    """
    data_dir = data_dir or os.environ.get("SCPYVIEWER_DATA_DIR", "data")
    return [{"name": name, "path": path}
            for name, path in _io.discover_datasets(data_dir).items()]


@mcp.tool()
def dataset_info(path: str) -> dict:
    """Summarize a dataset: cell/gene counts, default grouping, embeddings, and
    which obs columns are categorical vs numeric (use these as `group`/`color`)."""
    ds = _load(path)
    meta = _io.get_meta(ds.adata)
    return {
        "path": path,
        "n_cells": ds.n_obs,
        "n_genes": ds.n_vars,
        "group_key": ds.group_key,
        "embeddings": ds.embeddings,
        "categorical_fields": meta["schema"]["categorical_obs"],
        "numeric_fields": meta["schema"]["numeric_obs"],
        "has_markers": _io.markers_df(ds.adata) is not None,
    }


@mcp.tool()
def search_genes(path: str, query: str, limit: int = 30) -> list[str]:
    """Search gene names in a dataset (case-insensitive prefix + substring)."""
    return _load(path).genes(query, limit=limit)


# ------------------------------------------------------------------ tables
@mcp.tool()
def top_markers(path: str, group: str | None = None, top_n: int = 10) -> str:
    """Top marker / DE genes per group as a text table."""
    df = _api.markers_table(_load(path), group=group, top_n=top_n)
    return df.to_string(index=False)


@mcp.tool()
def composition(path: str, group: str | None = None, split: str | None = None) -> str:
    """Group × split composition (fractions) as a text table."""
    df = _api.composition_table(_load(path), group=group, split=split)
    return df.to_string(index=False)


# ------------------------------------------------------------------ plots (PNG)
@mcp.tool()
def plot_embedding(path: str, color: str | None = None, gene: str | None = None,
                   embedding: str | None = None) -> Image:
    """Embedding scatter (UMAP/t-SNE/PCA) colored by an obs column or a gene."""
    return _png(_api.plot_embedding(_load(path), color=color, gene=gene,
                                    embedding=embedding))


@mcp.tool()
def plot_dotplot(path: str, genes: list[str], group: str | None = None) -> Image:
    """Dot plot: mean expression (colour) × fraction expressing (size) per group."""
    return _png(_api.plot_dotplot(_load(path), genes, group=group))


@mcp.tool()
def plot_violin(path: str, gene: str, group: str | None = None) -> Image:
    """Violin plot of a gene's expression grouped by a metadata column."""
    return _png(_api.plot_violin(_load(path), gene, group=group))


@mcp.tool()
def plot_heatmap(path: str, genes: list[str], group: str | None = None) -> Image:
    """Expression heatmap (genes × cells) sorted and annotated by group."""
    return _png(_api.plot_heatmap(_load(path), genes, group=group))


@mcp.tool()
def plot_bar(path: str, gene: str, group: str | None = None,
             sort: str | None = "descending") -> Image:
    """Grouped bar chart of mean ± SEM expression of a gene across groups."""
    return _png(_api.plot_bar(_load(path), gene, group=group, sort=sort))


@mcp.tool()
def plot_composition(path: str, group: str | None = None,
                     split: str | None = None) -> Image:
    """Stacked bar of group composition across a sample/condition column."""
    return _png(_api.plot_composition(_load(path), group=group, split=split))


def main() -> None:
    """Console-script entry point: run the stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

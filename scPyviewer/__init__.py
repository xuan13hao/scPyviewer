"""scPyviewer — a Python-native interactive viewer for single-cell data.

scPyviewer ingests AnnData (``.h5ad``) directly and exposes embeddings, gene
expression, metadata, and marker/DE tables both through an interactive
Streamlit app and through a programmatic API.

Quick start (programmatic)::

    import scPyviewer as sv
    ds = sv.load_dataset("data/chicken_heart.prepared.h5ad")
    fig = sv.plot_embedding(ds, color=ds.group_key)
    markers = sv.markers_table(ds)
    sv.export_figures(ds, "out/figs")
    sv.export_tables(ds, "out/tables")

Launch the interactive viewer with the ``scpyviewer`` console script, or
``python -m streamlit run scPyviewer/app.py``.
"""
__version__ = "0.3.0"

from .api import (  # noqa: E402,F401
    Dataset,
    load_dataset,
    set_style,
    plot_embedding,
    plot_multigene,
    plot_violin,
    plot_dotplot,
    plot_composition,
    markers_table,
    composition_table,
    metadata_table,
    export_figures,
    export_tables,
)

__all__ = [
    "__version__",
    "Dataset", "load_dataset",
    "set_style",
    "plot_embedding", "plot_multigene", "plot_violin", "plot_dotplot",
    "plot_composition",
    "markers_table", "composition_table", "metadata_table",
    "export_figures", "export_tables",
]

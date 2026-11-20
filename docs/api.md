# API reference

All `plot_*` functions return a {class}`matplotlib.figure.Figure`; all `*_table`
functions return a {class}`pandas.DataFrame`. Call {func}`~scPyviewer.set_publication_style`
once to style every subsequent figure consistently.

```{eval-rst}
.. currentmodule:: scPyviewer
```

## Loading & style

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   load_dataset
   Dataset
   set_publication_style
   set_style
   get_palette
```

## Embedding plots

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_embedding
   plot_multigene
   plot_embedding_3d
```

## Expression plots

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_violin
   plot_stacked_violin
   plot_dotplot
   plot_matrixplot
   plot_heatmap
   plot_tracksplot
   plot_bar
   plot_trend
   plot_radar
```

## Composition & relationships

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_composition
   plot_correlation
```

## Tables & export

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   markers_table
   composition_table
   metadata_table
   export_figures
   export_tables
```

## Preparation (CLI + API)

Turn a raw `.h5ad` into a viewer-ready object. For atlases larger than RAM, see
[Large data](large_data).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare.prepare
   prepare.prepare_subsample
```

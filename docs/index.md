---
sd_hide_title: true
---

# scPyviewer

```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/embedding_celltype.png
:align: center
:width: 65%
```

<h1 align="center">scPyviewer</h1>

<p align="center"><b>A Python-native interactive viewer and publication-quality plotting API for single-cell AnnData.</b></p>

scPyviewer ingests an **AnnData** (`.h5ad`) object directly — no Seurat conversion — and exposes embeddings, gene expression, metadata, and marker/DE tables through both an interactive **Streamlit viewer** and a programmatic **plotting API** that produces publication-ready figures. It scales from a laptop toy dataset to **million-cell atlases larger than RAM**.

```{code-block} bash
pip install scPyviewer          # core
pip install 'scPyviewer[all]'   # + viewer, clustering, xlsx export
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 Get started
:link: installation
:link-type: doc
Install, launch the viewer, and load your first dataset.
:::

:::{grid-item-card} 📚 Tutorial
:link: tutorial
:link-type: doc
Every `plot_*` and table function with runnable examples.
:::

:::{grid-item-card} 🎨 Styles & palettes
:link: styling
:link-type: doc
figures4papers / nature presets and the journal colour palettes (NPG, AAAS, NEJM, Lancet, JAMA).
:::

:::{grid-item-card} 🧬 API reference
:link: api
:link-type: doc
Full signatures and parameters, generated from the docstrings.
:::

:::{grid-item-card} 🗻 Large data
:link: large_data
:link-type: doc
Explore million-cell atlases smoothly in low memory.
:::

:::{grid-item-card} 📄 Citation
:link: citation
:link-type: doc
How to cite scPyviewer (bioRxiv + BibTeX + `CITATION.cff`).
:::
::::

## Publication-quality figures

Every figure below is one `plot_*` call under {func}`~scPyviewer.set_publication_style` (bold figures4papers-style default, colour-blind-safe Okabe-Ito palette; call `set_publication_style("nature")` for the compact journal look). See the [tutorial](tutorial) for the code.

::::{grid} 3
:gutter: 2

:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/dotplot.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/heatmap.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/bar.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/trend.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/radar.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/violin.png
```
:::
::::

### Palettes

Match any target venue with one argument — `set_publication_style(palette="npg")`
(Nature), `"aaas"` (Science), `"nejm"`, `"lancet"`, `"jama"`, or the
colour-blind-safe `"okabe-ito"` default.

```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/palettes.png
:width: 80%
```

```{toctree}
:hidden:
:maxdepth: 2

installation
tutorial
styling
large_data
mcp
api
citation
```

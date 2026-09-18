# scPyviewer API Reference

Complete parameter reference for every public function in `scPyviewer`.

```python
import scPyviewer as sv
```

Every `plot_*` function returns a `matplotlib.figure.Figure`.  
Every `*_table` function returns a `pandas.DataFrame`.

---

## Table of Contents

1. [Loading data](#1-loading-data)  — `load_dataset`, `Dataset`
2. [Global style](#2-global-style)  — `set_style`
3. [plot_embedding](#3-plot_embedding)  — UMAP / t-SNE scatter
4. [plot_multigene](#4-plot_multigene)  — multi-gene expression grid
5. [plot_violin](#5-plot_violin)  — violin / box plot
6. [plot_dotplot](#6-plot_dotplot)  — dot plot
7. [plot_composition](#7-plot_composition)  — stacked bar chart
8. [plot_heatmap](#8-plot_heatmap)  — gene × group heatmap
9. [plot_matrixplot](#9-plot_matrixplot)  — mean expression matrix
10. [plot_stacked_violin](#10-plot_stacked_violin)  — multi-gene violin stack
11. [plot_tracksplot](#11-plot_tracksplot)  — single-cell expression tracks
12. [plot_correlation](#12-plot_correlation)  — inter-group correlation heatmap
13. [markers_table](#13-markers_table)  — DE / marker gene table
14. [composition_table](#14-composition_table)  — cell-type composition table
15. [metadata_table](#15-metadata_table)  — per-cell metadata table
16. [export_figures](#16-export_figures)  — batch figure export
17. [export_tables](#17-export_tables)  — batch table export

---

## 1 · Loading data

### `load_dataset(path)`

Load a prepared `.h5ad` file and return a `Dataset` handle used by all plotting functions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to a `*.prepared.h5ad` file. Run `scpyviewer-prepare` first to create one from a raw `.h5ad`. |

**Returns** `Dataset`

**Example**
```python
ds = sv.load_dataset("data/chicken_heart.prepared.h5ad")
print(ds)
# Dataset(chicken_heart.prepared.h5ad: 22315 cells x 10031 genes, group_key='cell_type')
```

---

### `Dataset` — attributes and methods

After loading you can inspect the dataset directly:

| Attribute / Method | Type | Description |
|--------------------|------|-------------|
| `ds.n_obs` | `int` | Number of cells |
| `ds.n_vars` | `int` | Number of genes |
| `ds.group_key` | `str` | Primary grouping column (e.g. `"cell_type"`) |
| `ds.embeddings` | `list[str]` | Available 2-D embedding keys (e.g. `["X_umap", "X_tsne"]`) |
| `ds.categorical` | `list[str]` | All categorical `obs` columns available for grouping |
| `ds.genes(query, limit)` | method | Search gene names by substring (case-insensitive). Returns `list[str]`. |

**Example**
```python
print(ds.n_obs, ds.n_vars)        # 22315  10031
print(ds.group_key)                # "cell_type"
print(ds.embeddings)               # ["X_umap", "X_tsne"]
print(ds.categorical)              # ["cell_type", "sample", "batch"]

# search genes
ds.genes("CD3")                    # ["CD3D", "CD3E", "CD3G"]
ds.genes("CD", limit=5)            # first 5 matches
```

---

## 2 · Global style

### `set_style(font_family, base_fontsize, dpi, style)`

Set matplotlib defaults that apply to **all subsequent** `plot_*` calls in the session.  
Individual plot functions can still override any setting via their own parameters.

| Parameter | Type | Default | Options / Description |
|-----------|------|---------|-----------------------|
| `font_family` | `str` or `None` | `None` | Any font available on your system: `"Arial"`, `"Helvetica"`, `"DejaVu Sans"`, `"Times New Roman"`, etc. |
| `base_fontsize` | `float` or `None` | `None` | Base font size in points. Sets `matplotlib.rcParams["font.size"]`. Common values: `9`, `10`, `11`, `12`. |
| `dpi` | `int` or `None` | `None` | Default resolution for all figures. `150` for screen, `300` for publication. |
| `style` | `str` or `None` | `None` | Matplotlib style sheet name. Examples: `"seaborn-v0_8-whitegrid"`, `"ggplot"`, `"bmh"`, `"classic"`. |

**Returns** `None`

**Examples**
```python
# publication defaults
sv.set_style(font_family="Arial", base_fontsize=11, dpi=300)

# screen defaults with a grid style
sv.set_style(font_family="DejaVu Sans", base_fontsize=10, dpi=120,
             style="seaborn-v0_8-whitegrid")

# reset to matplotlib defaults
sv.set_style(style="default")
```

---

## 3 · `plot_embedding`

UMAP / t-SNE / PCA scatter colored by a metadata column or a single gene.

```python
fig = sv.plot_embedding(ds, color="cell_type")
fig = sv.plot_embedding(ds, gene="CD3D")
```

### Parameters

#### Data selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle from `load_dataset`. |
| `color` | `str` or `None` | `None` | `obs` column name to color by. Can be **categorical** (cell type, batch, …) or **numeric** (n_counts, …). Defaults to `ds.group_key` if both `color` and `gene` are `None`. |
| `gene` | `str` or `None` | `None` | Gene name. When set, colors cells by log-normalized expression and ignores `color`. |
| `embedding` | `str` or `None` | `None` | `obsm` key to plot (e.g. `"X_umap"`, `"X_tsne"`, `"X_pca"`). Auto-selects `X_umap` if available, otherwise the first listed embedding. |

#### Layout

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | `(float, float)` | `(7.4, 5.6)` | Figure width × height in inches. |
| `dpi` | `int` | `150` | Resolution. Use `300` for publication, `100–150` for screen. |

#### Scatter style

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `point_size` | `float` | `3.0` | Dot diameter in points². Increase for sparse datasets, decrease for dense. |
| `alpha` | `float` | `0.75` | Dot transparency for categorical coloring. Range 0 (invisible) – 1 (solid). |
| `cmap` | `str` | `"viridis"` | Matplotlib colormap for **continuous** / gene coloring. Options: `"viridis"`, `"magma"`, `"plasma"`, `"RdBu_r"`, `"Blues"`, etc. |

#### Labels and title

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` or `None` | `None` | Figure title. Auto-generated from `color`/`gene` if `None`. |
| `title_fontsize` | `float` | `11` | Title font size in points. |
| `label_groups` | `bool` | `True` | Overlay centroid text labels for each category (categorical coloring only). |
| `label_fontsize` | `float` | `6` | Font size of centroid group labels. |
| `xlabel` | `str` or `None` | `None` | X-axis label. Defaults to embedding axis name (e.g. `"UMAP1"`). |
| `ylabel` | `str` or `None` | `None` | Y-axis label. Defaults to embedding axis name (e.g. `"UMAP2"`). |
| `xlabel_fontsize` | `float` | `9` | X-axis label font size. |
| `ylabel_fontsize` | `float` | `9` | Y-axis label font size. |
| `tick_fontsize` | `float` | `8` | Tick-label font size. |

#### Legend (categorical coloring)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_legend` | `bool` | `True` | Show legend when `label_groups=False`. |
| `legend_fontsize` | `float` | `7` | Legend entry font size. |
| `legend_title` | `str` or `None` | `None` | Legend title override. Defaults to the column name. |
| `legend_title_fontsize` | `float` | `8` | Legend title font size. |

#### Colorbar (gene coloring)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `colorbar_label` | `str` or `None` | `None` | Colorbar label override. Default: `"log-norm expression"`. |
| `colorbar_fontsize` | `float` | `8` | Colorbar label and tick font size. |

#### Style

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `font_family` | `str` or `None` | `None` | Font family for this figure only (e.g. `"Arial"`). Overrides `set_style`. |

### Examples

```python
# minimal — color by default group key
fig = sv.plot_embedding(ds, color="cell_type")

# gene expression overlay
fig = sv.plot_embedding(ds, gene="CD3D", cmap="magma")

# t-SNE instead of UMAP
fig = sv.plot_embedding(ds, color="batch", embedding="X_tsne")

# publication-quality
fig = sv.plot_embedding(
    ds,
    color="cell_type",
    figsize=(6, 5),
    dpi=300,
    point_size=5,
    alpha=0.85,
    title="Immune cell types",
    title_fontsize=14,
    label_groups=True,
    label_fontsize=9,
    xlabel="UMAP 1",
    ylabel="UMAP 2",
    xlabel_fontsize=11,
    ylabel_fontsize=11,
    tick_fontsize=9,
    font_family="Arial",
)
fig.savefig("fig1a.pdf", bbox_inches="tight")
```

---

## 4 · `plot_multigene`

Grid of UMAP / t-SNE panels, one per gene, all colored by expression.

```python
fig = sv.plot_multigene(ds, genes=["CD3D", "CD19", "CD14", "NKG7"])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `genes` | `list[str]` | required | Gene names to plot. Non-existent genes are silently skipped. |
| `embedding` | `str` or `None` | `None` | Embedding key (auto-selected if `None`). |
| `ncol` | `int` | `3` | Number of columns in the panel grid. Rows are computed automatically. |
| `figsize` | `(float, float)` or `None` | `None` | Figure size in inches. Auto-computed as `(3 × ncol, 2.8 × nrow)` if `None`. |
| `dpi` | `int` | `150` | Resolution. |
| `point_size` | `float` | `2.5` | Scatter dot diameter. |
| `cmap` | `str` | `"viridis"` | Colormap for expression. |
| `alpha` | `float` | `0.8` | Dot transparency. |
| `max_genes` | `int` | `12` | Hard cap on the number of panels (protects from very long gene lists). |
| `suptitle` | `str` or `None` | `None` | Figure super-title (top of whole figure). |
| `suptitle_fontsize` | `float` | `10` | Super-title font size. |
| `title_fontsize` | `float` or `None` | `None` | Alias for `suptitle_fontsize`. |
| `panel_title_fontsize` | `float` | `8` | Font size of each individual panel's gene title. |
| `colorbar_fontsize` | `float` | `6` | Colorbar tick and label font size per panel. |
| `tick_fontsize` | `float` or `None` | `None` | Colorbar tick font size override. |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
markers = ["CD3D", "CD19", "CD14", "NKG7", "FCGR3A", "MS4A1"]

# default 3-column grid
fig = sv.plot_multigene(ds, genes=markers)

# 2-column layout, custom style
fig = sv.plot_multigene(
    ds,
    genes=markers,
    ncol=2,
    figsize=(7, 12),
    dpi=300,
    point_size=3,
    cmap="RdBu_r",
    suptitle="Top marker genes",
    suptitle_fontsize=14,
    panel_title_fontsize=11,
    colorbar_fontsize=8,
    font_family="Arial",
)
```

---

## 5 · `plot_violin`

Violin or box plot of a single gene's expression, split by a categorical column.

```python
fig = sv.plot_violin(ds, gene="CD3D", group="cell_type")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `gene` | `str` | required | Gene name. Must exist in `ds.adata.var_names`. |
| `group` | `str` or `None` | `None` | `obs` column to split by. Defaults to `ds.group_key`. Any categorical column works (e.g. `"batch"`, `"sample"`). |
| `figsize` | `(float, float)` | `(7.0, 4.0)` | Figure size in inches. |
| `dpi` | `int` | `150` | Resolution. |
| `kind` | `str` | `"violin"` | Plot type. **Options:** `"violin"` (kernel density), `"box"` (box-and-whisker). |
| `palette` | `dict` or `None` | `None` | Custom color map `{label: color}`. Example: `{"T cell": "#e41a1c", "B cell": "#377eb8"}`. Overrides dataset defaults. |
| `rotation` | `int` | `30` | X-tick label rotation in degrees. Use `0` for horizontal, `90` for vertical. |
| `show_points` | `bool` | `False` | Overlay jittered individual data points on top of violins/boxes. |
| `title` | `str` or `None` | `None` | Figure title. Auto: `"{gene} by {group}"`. |
| `title_fontsize` | `float` | `11` | Title font size. |
| `xlabel` | `str` or `None` | `None` | X-axis label. Defaults to the column name. |
| `ylabel` | `str` or `None` | `None` | Y-axis label. Default: `"log-normalized expression"`. |
| `xlabel_fontsize` | `float` | `9` | X-axis label font size. |
| `ylabel_fontsize` | `float` | `9` | Y-axis label font size. |
| `tick_fontsize` | `float` | `8` | Tick-label font size. |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
# default violin
fig = sv.plot_violin(ds, gene="CD3D")

# box plot with jittered points
fig = sv.plot_violin(ds, gene="CD3D", group="cell_type",
                     kind="box", show_points=True)

# split by batch/condition
fig = sv.plot_violin(ds, gene="CD3D", group="batch")

# publication style
fig = sv.plot_violin(
    ds,
    gene="CD3D",
    group="cell_type",
    kind="violin",
    show_points=True,
    figsize=(9, 4.5),
    dpi=300,
    title="CD3D expression by cell type",
    title_fontsize=14,
    xlabel="Cell type",
    ylabel="Log-normalized expression",
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    tick_fontsize=10,
    rotation=25,
    font_family="Arial",
)
```

---

## 6 · `plot_dotplot`

Dot plot: dot **size** = fraction of cells expressing the gene; dot **color** = mean log-normalized expression.

```python
fig = sv.plot_dotplot(ds, genes=["CD3D", "CD19", "CD14"], group="cell_type")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `genes` | `list[str]` | required | Genes to show as columns. |
| `group` | `str` or `None` | `None` | `obs` row-grouping column. Defaults to `ds.group_key`. |
| `figsize` | `(float, float)` or `None` | `None` | Figure size. Auto-computed from gene/group counts if `None`. |
| `dpi` | `int` | `150` | Resolution. |
| `cmap` | `str` | `"viridis"` | Colormap for mean expression. Try `"Blues"`, `"YlOrRd"`, `"RdBu_r"`. |
| `size_scale` | `float` | `180` | Maximum dot area in points². Scales with fraction-expressing (0 → `size_scale`). Increase for visibility, decrease to reduce overlap. |
| `standard_scale` | `str` or `None` | `None` | Normalize mean expression before plotting. **Options:** `None` (raw log-norm values), `"var"` (0–1 per gene column — highlight relative differences across groups), `"group"` (0–1 per group row — highlight which genes are highest within each group). |
| `title` | `str` or `None` | `None` | Figure title. |
| `title_fontsize` | `float` | `11` | Title font size. |
| `xlabel` | `str` or `None` | `None` | X-axis label. Default: `"Gene"`. |
| `ylabel` | `str` or `None` | `None` | Y-axis label. Default: group column name. |
| `xlabel_fontsize` | `float` | `8` | X-axis label font size. |
| `ylabel_fontsize` | `float` | `8` | Y-axis label font size. |
| `tick_fontsize` | `float` | `7` | Tick-label font size. |
| `gene_label_rotation` | `int` | `45` | X-axis gene label rotation in degrees. |
| `legend_fontsize` | `float` | `6` | Dot-size legend entry font size. |
| `legend_title_fontsize` | `float` | `6` | Dot-size legend title font size. |
| `colorbar_label` | `str` or `None` | `None` | Colorbar label. Default: `"mean expr"`. |
| `colorbar_fontsize` | `float` | `7` | Colorbar label and tick font size. |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
markers = ["CD3D", "CD19", "CD14", "NKG7", "MS4A1"]

# default
fig = sv.plot_dotplot(ds, genes=markers, group="cell_type")

# normalize 0-1 per gene — easier to see which group expresses most
fig = sv.plot_dotplot(ds, genes=markers, group="cell_type",
                      standard_scale="var", cmap="Blues")

# publication style
fig = sv.plot_dotplot(
    ds,
    genes=markers,
    group="cell_type",
    standard_scale="var",
    cmap="viridis",
    size_scale=300,
    figsize=(8, 3.5),
    dpi=300,
    title="Marker gene expression",
    title_fontsize=13,
    xlabel="Gene",
    ylabel="Cell type",
    xlabel_fontsize=11,
    ylabel_fontsize=11,
    tick_fontsize=9,
    gene_label_rotation=35,
    colorbar_label="Scaled mean expr",
    colorbar_fontsize=9,
    legend_fontsize=8,
    font_family="Arial",
)
```

---

## 7 · `plot_composition`

Stacked bar chart showing cell-type proportions (or counts) across samples or conditions.

```python
fig = sv.plot_composition(ds, group="cell_type", split="sample")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `group` | `str` or `None` | `None` | Cell-type / cluster column (stacked colors). Defaults to `ds.group_key`. |
| `split` | `str` or `None` | `None` | Sample / condition column (X-axis bars). Auto-selects the first categorical column that isn't `group`. |
| `figsize` | `(float, float)` | `(7.6, 4.4)` | Figure size in inches. |
| `dpi` | `int` | `150` | Resolution. |
| `normalize` | `bool` | `True` | `True` → show fractions (0–1). `False` → show raw cell counts. |
| `palette` | `dict` or `None` | `None` | Custom `{label: color}` override. |
| `bar_width` | `float` | `0.8` | Bar width, 0–1. |
| `sort_groups` | `bool` | `False` | Sort stacked categories alphabetically. |
| `title` | `str` or `None` | `None` | Figure title. |
| `title_fontsize` | `float` | `11` | Title font size. |
| `xlabel` | `str` or `None` | `None` | X-axis label. Defaults to `split` column name. |
| `ylabel` | `str` or `None` | `None` | Y-axis label. Default: `"fraction of cells"` or `"cells"`. |
| `xlabel_fontsize` | `float` | `9` | X-axis label font size. |
| `ylabel_fontsize` | `float` | `9` | Y-axis label font size. |
| `tick_fontsize` | `float` | `8` | Tick-label font size. |
| `rotation` | `int` | `30` | X-tick label rotation in degrees. |
| `legend_fontsize` | `float` | `6` | Legend entry font size. |
| `legend_title_fontsize` | `float` | `7` | Legend title font size. |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
# fractions (default)
fig = sv.plot_composition(ds, group="cell_type", split="batch")

# raw counts
fig = sv.plot_composition(ds, group="cell_type", split="batch",
                          normalize=False)

# publication style
fig = sv.plot_composition(
    ds,
    group="cell_type",
    split="sample",
    normalize=True,
    sort_groups=True,
    figsize=(6, 4.5),
    dpi=300,
    bar_width=0.7,
    title="Cell-type composition per sample",
    title_fontsize=13,
    xlabel="Sample",
    ylabel="Fraction of cells",
    xlabel_fontsize=11,
    ylabel_fontsize=11,
    tick_fontsize=9,
    rotation=0,
    legend_fontsize=8,
    legend_title_fontsize=9,
    font_family="Arial",
)
```

---

## 8 · `plot_heatmap`

Expression heatmap with cells sorted by group. Each column is a cell; each row is a gene (or transposed with `swap_axes`).

```python
fig = sv.plot_heatmap(ds, genes=["CD3D", "CD19", "CD14"], group="cell_type")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `genes` | `list[str]` | required | Genes to display (rows when `swap_axes=False`). |
| `group` | `str` or `None` | `None` | `obs` column used to sort cells and draw the group annotation bar. |
| `figsize` | `(float, float)` or `None` | `None` | Figure size. Auto-computed from cell/gene counts if `None`. |
| `dpi` | `int` | `150` | Resolution. |
| `cmap` | `str` | `"viridis"` | Colormap. Try `"Blues"`, `"YlOrRd"`, `"RdBu_r"`, `"magma"`. |
| `vmin` | `float` or `None` | `None` | Minimum color-scale value. Values below are clipped. |
| `vmax` | `float` or `None` | `None` | Maximum color-scale value. Values above are clipped. |
| `standard_scale` | `str` or `None` | `None` | Normalize before plotting. **Options:** `None` (raw), `"var"` (0–1 per gene across cells), `"obs"` (0–1 per cell across genes). |
| `swap_axes` | `bool` | `False` | `False`: genes on Y, cells on X. `True`: genes on X, cells on Y. |
| `show_group_bar` | `bool` | `True` | Draw a color-coded group annotation bar above the heatmap. |
| `title` | `str` or `None` | `None` | Figure title. |
| `title_fontsize` | `float` | `11` | Title font size. |
| `xlabel` | `str` or `None` | `None` | X-axis label override. |
| `ylabel` | `str` or `None` | `None` | Y-axis label override. |
| `xlabel_fontsize` | `float` | `9` | X-axis label font size. |
| `ylabel_fontsize` | `float` | `9` | Y-axis label font size. |
| `tick_fontsize` | `float` | `7` | Gene-name and tick font size. |
| `group_label_fontsize` | `float` | `7` | Font size of group labels on the annotation bar. |
| `colorbar_label` | `str` or `None` | `None` | Colorbar label override. |
| `colorbar_fontsize` | `float` | `8` | Colorbar text font size. |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
genes = ["CD3D", "CD3E", "CD19", "MS4A1", "CD14", "NKG7"]

# default — genes × cells
fig = sv.plot_heatmap(ds, genes=genes, group="cell_type")

# transposed — cells × genes
fig = sv.plot_heatmap(ds, genes=genes, group="cell_type", swap_axes=True)

# scale per gene, custom colormap
fig = sv.plot_heatmap(ds, genes=genes, group="cell_type",
                      standard_scale="var", cmap="Blues")

# diverging colormap with clipped range
fig = sv.plot_heatmap(ds, genes=genes, group="cell_type",
                      cmap="RdBu_r", vmin=-1, vmax=1)

# publication
fig = sv.plot_heatmap(
    ds,
    genes=genes,
    group="cell_type",
    standard_scale="var",
    show_group_bar=True,
    cmap="Blues",
    figsize=(10, 5),
    dpi=300,
    title="Marker gene heatmap",
    title_fontsize=13,
    tick_fontsize=9,
    colorbar_fontsize=9,
    font_family="Arial",
)
```

---

## 9 · `plot_matrixplot`

Color-coded matrix of **mean expression** per gene per group. Optionally annotates each cell with the numeric value.

```python
fig = sv.plot_matrixplot(ds, genes=["CD3D", "CD19", "CD14"], group="cell_type")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `genes` | `list[str]` | required | Genes to include (columns when `swap_axes=False`). |
| `group` | `str` or `None` | `None` | `obs` grouping column (rows when `swap_axes=False`). |
| `figsize` | `(float, float)` or `None` | `None` | Figure size. Auto-computed if `None`. |
| `dpi` | `int` | `150` | Resolution. |
| `cmap` | `str` | `"Blues"` | Colormap for mean expression. |
| `vmin` | `float` or `None` | `None` | Minimum color-scale value. |
| `vmax` | `float` or `None` | `None` | Maximum color-scale value. |
| `standard_scale` | `str` or `None` | `None` | Normalize before plotting. **Options:** `None`, `"var"` (0–1 per gene column), `"group"` (0–1 per group row). |
| `swap_axes` | `bool` | `False` | `False`: groups on Y, genes on X. `True`: genes on Y, groups on X. |
| `annotate` | `bool` | `False` | Overlay the numeric mean expression value in each matrix cell. |
| `annotation_fmt` | `str` | `".2f"` | Python format string for annotations. `".2f"` → two decimals, `".1f"` → one decimal, `".0f"` → integer. |
| `annotation_fontsize` | `float` | `7` | Font size of annotation text inside cells. |
| `title` | `str` or `None` | `None` | Figure title. |
| `title_fontsize` | `float` | `11` | Title font size. |
| `xlabel` | `str` or `None` | `None` | X-axis label override. |
| `ylabel` | `str` or `None` | `None` | Y-axis label override. |
| `xlabel_fontsize` | `float` | `9` | X-axis label font size. |
| `ylabel_fontsize` | `float` | `9` | Y-axis label font size. |
| `tick_fontsize` | `float` | `8` | Tick-label font size. |
| `gene_label_rotation` | `int` | `45` | Gene label rotation in degrees. |
| `colorbar_label` | `str` or `None` | `None` | Colorbar label override. Default: `"mean log-norm expr"`. |
| `colorbar_fontsize` | `float` | `8` | Colorbar text font size. |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
genes = ["CD3D", "CD19", "CD14", "NKG7", "MS4A1"]

# default
fig = sv.plot_matrixplot(ds, genes=genes, group="cell_type")

# with value annotations
fig = sv.plot_matrixplot(ds, genes=genes, group="cell_type",
                         annotate=True, annotation_fmt=".2f")

# swapped axes + scaled
fig = sv.plot_matrixplot(ds, genes=genes, group="cell_type",
                         swap_axes=True, standard_scale="var",
                         cmap="YlOrRd", annotate=True)

# publication
fig = sv.plot_matrixplot(
    ds,
    genes=genes,
    group="cell_type",
    standard_scale="var",
    cmap="YlOrRd",
    annotate=True,
    annotation_fmt=".1f",
    figsize=(7, 4),
    dpi=300,
    title="Mean expression matrix",
    title_fontsize=13,
    tick_fontsize=9,
    colorbar_fontsize=9,
    font_family="Arial",
)
```

---

## 10 · `plot_stacked_violin`

Stack of violin plots — one row per gene, showing expression distribution per group.

```python
fig = sv.plot_stacked_violin(ds, genes=["CD3D", "CD19", "CD14"], group="cell_type")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `genes` | `list[str]` | required | Genes to show. One row (or column) per gene. |
| `group` | `str` or `None` | `None` | `obs` grouping column. |
| `figsize` | `(float, float)` or `None` | `None` | Figure size. Auto-computed if `None`. |
| `dpi` | `int` | `150` | Resolution. |
| `swap_axes` | `bool` | `False` | `False`: genes on Y (stacked rows), groups on X. `True`: genes on X (side-by-side columns), groups on Y. |
| `palette` | `dict` or `None` | `None` | Custom `{label: color}` override. |
| `inner` | `str` or `None` | `"box"` | Inner mark inside each violin. **Options:** `"box"` (IQR box + median dot), `"point"` (jittered data points), `None` (violin only). |
| `linewidth` | `float` | `0.6` | Violin body edge linewidth. Increase for bolder outlines. |
| `title` | `str` or `None` | `None` | Figure title. |
| `title_fontsize` | `float` | `11` | Title font size. |
| `xlabel` | `str` or `None` | `None` | Shared X-axis label override. |
| `ylabel` | `str` or `None` | `None` | Shared Y-axis label override. Default: `"log-norm expression"`. |
| `xlabel_fontsize` | `float` | `9` | Shared X-axis label font size. |
| `ylabel_fontsize` | `float` | `9` | Shared Y-axis label font size. |
| `tick_fontsize` | `float` | `7` | Tick-label font size. |
| `gene_label_fontsize` | `float` | `8` | Gene name label font size (left-side row labels). |
| `rotation` | `int` | `30` | Group-label rotation on X-axis (degrees). |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
genes = ["CD3D", "CD19", "CD14", "NKG7"]

# default stack
fig = sv.plot_stacked_violin(ds, genes=genes, group="cell_type")

# swap: genes as columns
fig = sv.plot_stacked_violin(ds, genes=genes, group="cell_type", swap_axes=True)

# box inner
fig = sv.plot_stacked_violin(ds, genes=genes, group="cell_type", inner="box")

# point inner (jittered dots)
fig = sv.plot_stacked_violin(ds, genes=genes, group="cell_type", inner="point")

# no inner (clean violin only)
fig = sv.plot_stacked_violin(ds, genes=genes, group="cell_type", inner=None)

# publication
fig = sv.plot_stacked_violin(
    ds,
    genes=genes,
    group="cell_type",
    inner="box",
    linewidth=0.5,
    figsize=(9, 6),
    dpi=300,
    title="Multi-gene expression",
    title_fontsize=13,
    xlabel_fontsize=11,
    ylabel_fontsize=11,
    tick_fontsize=9,
    gene_label_fontsize=10,
    font_family="Arial",
)
```

---

## 11 · `plot_tracksplot`

Expression tracks — one horizontal color strip per gene, cells ordered by group.  
Great for visualizing continuous expression changes across cell types or along a trajectory.

```python
fig = sv.plot_tracksplot(ds, genes=["CD3D", "CD19", "CD14"], group="cell_type")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `genes` | `list[str]` | required | Genes to display (one horizontal track each). |
| `group` | `str` or `None` | `None` | `obs` column used to sort cells and mark group boundaries. |
| `figsize` | `(float, float)` or `None` | `None` | Figure size. Auto-computed from cell count and number of genes if `None`. |
| `dpi` | `int` | `150` | Resolution. |
| `cmap` | `str` | `"viridis"` | Colormap for expression values. |
| `vmin` | `float` or `None` | `None` | Minimum color-scale value (clip below). |
| `vmax` | `float` or `None` | `None` | Maximum color-scale value (clip above). |
| `show_group_labels` | `bool` | `True` | Annotate group boundaries at the bottom of the figure. |
| `track_height` | `float` | `0.4` | Height of each gene track in inches. Increase for taller, more visible tracks. |
| `title` | `str` or `None` | `None` | Figure title. |
| `title_fontsize` | `float` | `11` | Title font size. |
| `gene_label_fontsize` | `float` | `8` | Font size of gene name labels on the left. |
| `group_label_fontsize` | `float` | `8` | Font size of group boundary labels at the bottom. |
| `colorbar_label` | `str` or `None` | `None` | Colorbar label override. Default: `"log-norm expression"`. |
| `colorbar_fontsize` | `float` | `8` | Colorbar text font size. |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
genes = ["CD3D", "CD19", "CD14", "NKG7", "FCGR3A"]

# default
fig = sv.plot_tracksplot(ds, genes=genes, group="cell_type")

# magma colormap, no group labels
fig = sv.plot_tracksplot(ds, genes=genes, group="cell_type",
                         cmap="magma", show_group_labels=False)

# clip expression range for better contrast
fig = sv.plot_tracksplot(ds, genes=genes, group="cell_type",
                         vmin=0, vmax=3)

# publication — taller tracks
fig = sv.plot_tracksplot(
    ds,
    genes=genes,
    group="cell_type",
    cmap="viridis",
    vmin=0, vmax=3,
    track_height=0.6,
    show_group_labels=True,
    figsize=(12, 5),
    dpi=300,
    title="Expression tracks by cell type",
    title_fontsize=13,
    gene_label_fontsize=9,
    group_label_fontsize=9,
    colorbar_fontsize=9,
    font_family="Arial",
)
```

---

## 12 · `plot_correlation`

Pairwise **Pearson or Spearman** correlation matrix between groups, based on mean gene expression profiles.

```python
fig = sv.plot_correlation(ds, group="cell_type", method="pearson")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `group` | `str` or `None` | `None` | `obs` column whose categories become matrix rows and columns. Defaults to `ds.group_key`. |
| `genes` | `list[str]` or `None` | `None` | Subset of genes to use for computing correlations. `None` = all genes. |
| `method` | `str` | `"pearson"` | Correlation method. **Options:** `"pearson"` (linear correlation), `"spearman"` (rank-based, robust to outliers). |
| `figsize` | `(float, float)` or `None` | `None` | Figure size. Auto-computed from number of groups. |
| `dpi` | `int` | `150` | Resolution. |
| `cmap` | `str` | `"RdBu_r"` | Diverging colormap. Recommended: `"RdBu_r"`, `"coolwarm"`, `"vlag"`. |
| `vmin` | `float` | `-1.0` | Minimum color value. Keep at `-1` for a correlation matrix. |
| `vmax` | `float` | `1.0` | Maximum color value. Keep at `1` for a correlation matrix. |
| `annotate` | `bool` | `True` | Overlay numeric correlation values in each matrix cell. |
| `annotation_fmt` | `str` | `".2f"` | Format string for annotations. `".2f"` → two decimals. |
| `annotation_fontsize` | `float` | `8` | Annotation text font size. |
| `title` | `str` or `None` | `None` | Figure title. |
| `title_fontsize` | `float` | `11` | Title font size. |
| `tick_fontsize` | `float` | `9` | Tick-label font size. |
| `colorbar_label` | `str` or `None` | `None` | Colorbar label override. Default: `"Pearson r"` or `"Spearman r"`. |
| `colorbar_fontsize` | `float` | `8` | Colorbar text font size. |
| `rotation` | `int` | `45` | X-tick label rotation in degrees. |
| `font_family` | `str` or `None` | `None` | Per-figure font family. |

### Examples

```python
# Pearson (all genes)
fig = sv.plot_correlation(ds, group="cell_type", method="pearson")

# Spearman (robust to outliers)
fig = sv.plot_correlation(ds, group="cell_type", method="spearman")

# restrict to marker genes only
markers = ["CD3D", "CD19", "CD14", "NKG7", "MS4A1", "FCGR3A"]
fig = sv.plot_correlation(ds, group="cell_type", genes=markers,
                          method="pearson")

# no annotations (cleaner for many groups)
fig = sv.plot_correlation(ds, group="cell_type", annotate=False)

# publication
fig = sv.plot_correlation(
    ds,
    group="cell_type",
    method="pearson",
    annotate=True,
    annotation_fmt=".2f",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    figsize=(6, 5),
    dpi=300,
    title="Cell-type transcriptome correlation",
    title_fontsize=13,
    tick_fontsize=10,
    colorbar_fontsize=9,
    font_family="Arial",
)
```

---

## 13 · `markers_table`

Return the per-group differential expression / marker gene table as a DataFrame.

```python
mk = sv.markers_table(ds)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. Requires that `scpyviewer-prepare` has been run (populates `adata.uns["scPyviewer_markers"]`). |
| `group` | `str` or `None` | `None` | Filter to a single group label (e.g. `"T cell"`). `None` = return all groups. |
| `top_n` | `int` or `None` | `None` | Keep top-N genes per group. `None` = all genes. |
| `sort_by` | `str` | `"rank"` | Column to sort by. **Options:** `"rank"`, `"logfoldchange"`, `"pval_adj"`, `"score"`. |
| `ascending` | `bool` | `True` | Sort direction. `False` = descending (highest first). |

**Returns** `pandas.DataFrame` with columns: `gene`, `group`, `logfoldchange`, `pval`, `pval_adj`, `rank`, `score`.

### Examples

```python
# top 10 marker genes per group
mk = sv.markers_table(ds, top_n=10)

# only T cell markers
t_markers = sv.markers_table(ds, group="T cell", top_n=20)

# sort by fold change (highest first)
mk = sv.markers_table(ds, sort_by="logfoldchange", ascending=False, top_n=10)

# sort by adjusted p-value (most significant first)
mk = sv.markers_table(ds, sort_by="pval_adj", ascending=True, top_n=10)

# save to CSV
sv.markers_table(ds, top_n=25).to_csv("markers.csv", index=False)
```

---

## 14 · `composition_table`

Return the group × split composition matrix as a DataFrame.

```python
comp = sv.composition_table(ds, group="cell_type", split="sample")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `group` | `str` or `None` | `None` | Cell-type / cluster column (columns of output table). Defaults to `ds.group_key`. |
| `split` | `str` or `None` | `None` | Sample / condition column (rows of output table). Auto-selects first non-`group` categorical column. |
| `normalize` | `bool` | `True` | `True` → fractions (0–1). `False` → raw cell counts. |

**Returns** `pandas.DataFrame` with one row per split value and one column per group value.

### Examples

```python
# fractions per sample
comp = sv.composition_table(ds, group="cell_type", split="sample")

# raw counts per batch
comp = sv.composition_table(ds, group="cell_type", split="batch",
                             normalize=False)

# save to CSV
comp.to_csv("composition.csv", index=False)
```

---

## 15 · `metadata_table`

Return the per-cell metadata (`obs`) as a DataFrame.

```python
meta = sv.metadata_table(ds)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |

**Returns** `pandas.DataFrame` — one row per cell, columns are all `obs` fields (e.g. `cell_type`, `batch`, `n_counts`, etc.).

### Examples

```python
meta = sv.metadata_table(ds)

# filter and inspect
t_cells = meta[meta["cell_type"] == "T cell"]
print(t_cells["n_counts"].describe())

# save
meta.to_csv("metadata.csv", index=False)
```

---

## 16 · `export_figures`

Render the standard figure set and write them to disk.

Produces: UMAP by group, UMAP by gene, multi-gene grid, violin, dot plot, composition.

```python
paths = sv.export_figures(ds, outdir="results/figs", formats=["png", "pdf"])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `outdir` | `str` | required | Output directory. Created automatically if it doesn't exist. |
| `formats` | `list[str]` | `("png",)` | Output file formats. **Options:** `"png"`, `"pdf"`, `"svg"`. Multiple formats are written per figure. |
| `genes` | `list[str]` or `None` | `None` | Genes to highlight in UMAP and multi-gene panels. Defaults to top-ranked marker genes. |
| `dpi` | `int` | `200` | Raster resolution for PNG output. |
| `figsize` | `(float, float)` or `None` | `None` | Override figure size for all plots. |
| `title_fontsize` | `float` | `11` | Title font size applied to all plots. |
| `tick_fontsize` | `float` | `8` | Tick-label font size applied to all plots. |
| `label_fontsize` | `float` | `9` | Axis-label font size applied to all plots. |
| `font_family` | `str` or `None` | `None` | Font family applied to all plots. |

**Returns** `list[str]` — paths of all written files.

### Examples

```python
# PNG only
paths = sv.export_figures(ds, outdir="results/figs")

# multiple formats
paths = sv.export_figures(ds, outdir="results/figs",
                          formats=["png", "pdf", "svg"])

# publication settings
paths = sv.export_figures(
    ds,
    outdir="results/paper_figs",
    formats=["pdf", "svg"],
    dpi=300,
    title_fontsize=12,
    tick_fontsize=9,
    label_fontsize=10,
    font_family="Arial",
)
for p in paths:
    print(p)
```

---

## 17 · `export_tables`

Write marker, composition, and metadata tables to disk.

```python
paths = sv.export_tables(ds, outdir="results/tables", formats=["csv", "xlsx"])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ds` | `Dataset` | required | Dataset handle. |
| `outdir` | `str` | required | Output directory. Created if absent. |
| `formats` | `list[str]` | `("csv",)` | Output formats. **Options:** `"csv"`, `"tsv"`, `"xlsx"` (xlsx requires `pip install openpyxl`). |
| `top_n` | `int` or `None` | `25` | Limit the marker table to top-N genes per group. `None` = all genes. |

**Returns** `list[str]` — paths of all written files.

### Examples

```python
# CSV
paths = sv.export_tables(ds, outdir="results/tables")

# CSV + TSV + Excel, all markers
paths = sv.export_tables(ds, outdir="results/tables",
                         formats=["csv", "tsv", "xlsx"],
                         top_n=None)
```

---

## Common patterns

### Publication-quality figures

```python
# 1. set global style once
sv.set_style(font_family="Arial", base_fontsize=11, dpi=300)

# 2. produce figures with fine-tuned typography
fig = sv.plot_embedding(
    ds, color="cell_type",
    figsize=(5.5, 4.5), title="A  Cell types",
    title_fontsize=12, label_fontsize=8,
    xlabel_fontsize=10, ylabel_fontsize=10,
)

# 3. save directly
fig.savefig("fig1a.pdf", bbox_inches="tight")
fig.savefig("fig1a.png", dpi=300, bbox_inches="tight")
```

### Consistent panel style across many plots

```python
style = dict(
    figsize=(5.5, 4.5), dpi=300,
    title_fontsize=12, tick_fontsize=9,
    font_family="Arial",
)

fig_umap  = sv.plot_embedding(ds, color="cell_type", **style)
fig_dot   = sv.plot_dotplot(ds, genes=markers, group="cell_type", **style)
fig_comp  = sv.plot_composition(ds, group="cell_type", split="batch", **style)
```

### Save a figure to multiple formats

```python
fig = sv.plot_heatmap(ds, genes=markers, group="cell_type")
for fmt in ("png", "pdf", "svg"):
    fig.savefig(f"heatmap.{fmt}", dpi=300, bbox_inches="tight")
```

### Filter and export marker genes

```python
# get top 5 markers per group, sorted by fold change
mk = sv.markers_table(ds, top_n=5,
                       sort_by="logfoldchange", ascending=False)

# save as CSV and Excel
mk.to_csv("top_markers.csv", index=False)
mk.to_excel("top_markers.xlsx", index=False)
```

---

## Parameter cheat-sheet

| Parameter | Applies to | Type | What it controls |
|-----------|-----------|------|-----------------|
| `figsize` | all plots | `(float, float)` | Figure width × height in inches |
| `dpi` | all plots | `int` | Resolution. `150` screen, `300` publication |
| `title` | all plots | `str` | Figure title text |
| `title_fontsize` | all plots | `float` | Title font size in points |
| `tick_fontsize` | all plots | `float` | Tick-label font size |
| `font_family` | all plots | `str` | Per-figure font, e.g. `"Arial"` |
| `cmap` | embedding, multigene, dotplot, heatmap, matrixplot, tracksplot, correlation | `str` | Matplotlib colormap name |
| `vmin` / `vmax` | heatmap, matrixplot, tracksplot, correlation | `float` | Clip color scale |
| `xlabel` / `ylabel` | most plots | `str` | Axis label text |
| `xlabel_fontsize` / `ylabel_fontsize` | most plots | `float` | Axis label font size |
| `colorbar_fontsize` | embedding, multigene, dotplot, heatmap, matrixplot, tracksplot, correlation | `float` | Colorbar text size |
| `standard_scale` | dotplot, heatmap, matrixplot | `None \| "var" \| "group"` | Normalize expression for visual comparison |
| `swap_axes` | heatmap, matrixplot, stacked_violin | `bool` | Transpose gene / group axes |
| `annotate` | matrixplot, correlation | `bool` | Show numeric values in cells |
| `annotation_fmt` | matrixplot, correlation | `str` | Python format string, e.g. `".2f"` |
| `point_size` | embedding, multigene | `float` | Scatter dot size |
| `alpha` | embedding, multigene | `float` | Dot transparency 0–1 |
| `label_groups` | embedding | `bool` | Centroid text labels per category |
| `show_legend` | embedding | `bool` | Show legend for categorical color |
| `kind` | violin | `"violin" \| "box"` | Plot type |
| `show_points` | violin | `bool` | Overlay jittered raw data points |
| `inner` | stacked_violin | `None \| "box" \| "point"` | Violin inner mark style |
| `linewidth` | stacked_violin | `float` | Violin edge linewidth |
| `gene_label_fontsize` | stacked_violin, tracksplot | `float` | Gene name font size |
| `gene_label_rotation` | dotplot, matrixplot | `int` | Gene label rotation in degrees |
| `track_height` | tracksplot | `float` | Height per gene track in inches |
| `show_group_labels` | tracksplot | `bool` | Group boundary labels |
| `show_group_bar` | heatmap | `bool` | Color-coded group annotation bar |
| `method` | correlation | `"pearson" \| "spearman"` | Correlation method |
| `normalize` | composition | `bool` | Fractions vs. raw counts |
| `sort_groups` | composition | `bool` | Alphabetically sort stacked groups |
| `size_scale` | dotplot | `float` | Maximum dot area (fraction-expressing) |
| `rotation` | violin, composition, correlation | `int` | X-tick label rotation in degrees |
| `palette` | violin, composition, stacked_violin | `dict` | Custom `{label: color}` color map |

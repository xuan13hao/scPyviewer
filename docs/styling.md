# Styles & palettes

Every `plot_*` shares one global style, set with
{func}`~scPyviewer.set_publication_style`. It applies the bold, figures4papers
look **automatically on import** — switch style or palette with a single call.

## Style presets

```{code-block} python
sv.set_publication_style()           # "bold" (figures4papers) — the default
sv.set_publication_style("nature")   # compact, journal-clean
```

The same bar chart in each preset:

::::{grid} 2
:gutter: 3

:::{grid-item-card} bold (figures4papers) — default
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/preset_bold.png
```
Thick 2 pt axes, bold titles, black-edged bars — high impact for talks, posters,
and punchy paper panels.
:::

:::{grid-item-card} nature
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/preset_nature.png
```
Thin 0.8 pt line art, compact type, un-bolded — matches Nature's artwork
guidance for journal submission.
:::
::::

## Journal palettes

Match a target venue's colour scheme with one argument. The hex schemes mirror
the [ggsci](https://nanx.me/ggsci/) palettes that are the de-facto standard in
bioinformatics papers:

| palette | venue |
|---|---|
| `npg` | Nature Publishing Group |
| `aaas` | Science / AAAS |
| `nejm` | New England Journal of Medicine |
| `lancet` | The Lancet |
| `jama` | JAMA |
| `okabe-ito` | colour-blind-safe default |

```{code-block} python
sv.set_publication_style(palette="npg")      # Nature
sv.set_publication_style(palette="lancet")   # The Lancet
```

The same UMAP, coloured with each palette:

::::{grid} 3
:gutter: 2

:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/pal_npg.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/pal_aaas.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/pal_nejm.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/pal_lancet.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/pal_jama.png
```
:::
:::{grid-item}
```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/pal_okabe-ito.png
```
:::
::::

### Swatches

```{image} https://raw.githubusercontent.com/xuan13hao/scPyviewer/main/docs/images/palettes.png
:width: 80%
```

Continuous data (gene expression, heatmaps) uses the perceptually-uniform,
colour-blind-safe `viridis` by default; pass `cmap="cividis"` for maximal
CVD-safety, or any Matplotlib colormap. See {func}`~scPyviewer.get_palette` and
{func}`~scPyviewer.set_publication_style`.

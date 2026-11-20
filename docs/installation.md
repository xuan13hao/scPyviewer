# Installation

```{code-block} bash
# PyPI (recommended)
pip install scPyviewer

# with optional extras (Streamlit viewer + leiden clustering + xlsx export)
pip install 'scPyviewer[all]'

# from source
pip install -e '.[all]'
```

This installs the `scPyviewer` package and two console scripts:

- **`scpyviewer`** — launch the interactive Streamlit viewer
- **`scpyviewer-prepare`** — turn a raw `.h5ad` into a viewer-ready object

Optional dependency groups: `app` (Streamlit), `prepare` (leiden clustering),
`excel` (`.xlsx` export), `dev` (pytest), `all` (everything).

## Quick start

### Programmatic API

```{code-block} python
import scPyviewer as sv

sv.set_publication_style()                      # Nature-style, colour-blind safe
ds = sv.load_dataset("data/chicken_heart.prepared.h5ad")

fig = sv.plot_embedding(ds, color=ds.group_key)
fig = sv.plot_dotplot(ds, ["MYL2", "TNNC1", "HBA1"])
markers = sv.markers_table(ds, top_n=10)

sv.export_figures(ds, "out/figs", formats=("pdf", "png"))
```

### Interactive viewer

```{code-block} bash
scpyviewer --data-dir data            # open http://localhost:8501
```

### Prepare your own data

```{code-block} bash
scpyviewer-prepare your_dataset.h5ad  # -> your_dataset.prepared.h5ad
```

`prepare` log-normalizes, finds highly-variable genes, computes PCA / UMAP /
t-SNE where missing, runs per-group differential expression, and writes an
`uns['scPyviewer']` metadata block the viewer uses. For files larger than RAM,
see [Large data](large_data).

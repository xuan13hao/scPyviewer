# scviewer — a Python-native interactive viewer for single-cell data

`scviewer` is a lightweight, browser-based explorer for analyzed single-cell
datasets. It ingests an **AnnData** (`.h5ad`) object directly — no Seurat
conversion, no notebook — and lets a non-programmer explore embeddings, gene
expression, metadata, and marker/DE tables, then share the result with a single
command.

Every actively maintained tool in this space (ShinyCell, ScRDAVis,
sCIRCLE/scViewer) is built on R Shiny and requires a Seurat object. `scviewer`
fills the Python/scanpy gap: it stays entirely inside the Python stack that most
single-cell analysis already runs in.

---

## Quick start (one command per step)

```bash
./run.sh setup       # install pinned dependencies (Python 3.11)
./run.sh prepare     # raw .h5ad -> viewer-ready .prepared.h5ad
./run.sh app         # launch the viewer at http://localhost:8501
```

Everything is driven through **`run.sh`**, the single reproduction interface:

| Command | What it does |
|---|---|
| `./run.sh setup` | `pip install -r requirements.txt` |
| `./run.sh prepare [RAW.h5ad]` | preprocess raw counts → `*.prepared.h5ad` (lognorm, HVG, PCA, t-SNE, per-group DE) |
| `./run.sh app` | launch the Streamlit viewer |
| `./run.sh benchmark` | run the feature-parity + performance harness, write `results/benchmark_results.json`, render benchmark figures |
| `./run.sh figures` | regenerate all demonstration + paper figures |
| `./run.sh all` | `prepare → benchmark → figures` |
| `./run.sh help` | usage |

### Environment overrides

`run.sh` reads four optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PY` | `python` | Python interpreter to use |
| `DATA_DIR` | `data` | directory scanned for `.h5ad` files |
| `RAW` | `$DATA_DIR/chicken_heart.h5ad` | raw input for `prepare`/`all` |
| `PORT` | `8501` | Streamlit port |

Example — prepare and serve a different dataset on another port:

```bash
RAW=data/my_data.h5ad ./run.sh prepare
PORT=9000 ./run.sh app
```

---

## Full reproduction from scratch

```bash
./run.sh setup                       # 1. dependencies
./run.sh all                         # 2. prepare + benchmark + figures
./run.sh app                         # 3. explore interactively
```

`run.sh all` regenerates every artifact behind the paper:
`results/benchmark_results.json` and all figures under `results/figures/`.

---

## What `prepare` does

`prepare.py` is **dataset-agnostic and idempotent** — it guards every step, so
running it on an already-analyzed object only fills in what is missing:

1. Store raw counts in `layers['counts']`; build a log-normalized matrix in
   `layers['lognorm']` and set it as `X` (skipped if `X` already looks
   log-normalized).
2. Highly-variable genes → PCA (skipped if `X_pca` present).
3. UMAP and t-SNE embeddings (any embedding already present, e.g. `X_umap` or a
   precomputed `X_uce`, is preserved; UMAP is skipped when already there).
4. Per-group differential expression (Wilcoxon) over the auto-selected grouping
   column, written to `uns['rank_genes_groups']` and a tidy
   `uns['scviewer_markers']` table.
5. A `uns['scviewer']` metadata block + a sidecar `*.manifest.json`.

Pass `--no-tsne` to skip the (slower) t-SNE step; pass `-g COLUMN` to force the
DE grouping column.

---

## The viewer

The Streamlit app (`scviewer/app.py`) has five tabs:

- **Embedding** — any 2-D embedding, colored by metadata or gene expression.
- **Expression** — single/multi-gene overlays, violin, and dot plots grouped by
  any metadata column.
- **Markers / DE** — browse the per-group differential-expression table.
- **Compare** — cross-dataset / cross-species side-by-side comparison.
- **Export** — download the current view and filtered cell tables.

A sidebar picks the dataset (any `*.prepared.h5ad` in `DATA_DIR`) and applies
metadata filters shared across all tabs.

---

## Layout

```
run.sh                       single reproduction interface
requirements.txt             pinned dependencies (Python 3.11)
scviewer/
  prepare.py                 raw .h5ad -> viewer-ready .prepared.h5ad
  app.py                     Streamlit viewer (5 tabs)
  io_utils.py                AnnData loading / metadata / gene access
  plots.py                   pure Plotly plotting functions
  benchmark.py               feature-parity + performance + time-to-share
  make_figures.py            demonstration + paper figures
data/                        .h5ad inputs and *.prepared.h5ad outputs
results/
  benchmark_results.json     benchmark output
  figures/                   paper + demonstration figures
paper/                       manuscript
```

---

## Notes on reproducibility

- Dependencies are pinned in `requirements.txt` to the versions used for the
  paper's benchmarks (scanpy 1.11.5, anndata 0.12.19, streamlit 1.59.2,
  plotly 6.9.0, Python 3.11).
- Benchmark timings are machine-dependent; the absolute numbers in the paper
  were measured on the development host (16 CPU, 8 GiB RAM, no GPU) and will
  vary. Re-run `./run.sh benchmark` to obtain numbers for your own hardware.
- The tool is built to be dataset- and species-agnostic. The distributed
  benchmark is run on one attached dataset (`chicken_heart.h5ad`); the study design names additional datasets that
  were not attached to this build.

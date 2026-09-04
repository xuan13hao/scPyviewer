# scPyviewer — Python-native interactive viewer and Python API for single-cell data

[![PyPI version](https://img.shields.io/pypi/v/scPyviewer.svg)](https://pypi.org/project/scPyviewer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)

**scPyviewer** is a lightweight, browser-based explorer for analyzed single-cell datasets. It ingests an **AnnData** (`.h5ad`) object directly — no Seurat conversion, no notebook — and lets a non-programmer explore embeddings, gene expression, metadata, and marker/DE tables, then share the result with a single command.

Every actively maintained tool in this space (ShinyCell, ScRDAVis, sCIRCLE/scViewer) is built on R Shiny and requires a Seurat object. **scPyviewer fills the Python/scanpy gap**: it stays entirely inside the Python stack that most single-cell analysis already runs in.

---

## Highlights

| | scPyviewer | ShinyCell | ScRDAVis | sCIRCLE/scViewer |
|---|:---:|:---:|:---:|:---:|
| Embedding plot (UMAP/PCA/t-SNE) | ✓ | ✓ | ✓ | ✓ |
| Single/multi-gene expression overlay | ✓ | ✓ | ✓ | Partial |
| Violin/box plot grouped by metadata | ✓ | ✓ | ✓ | Partial |
| Marker gene / DE table browsing | ✓ | Partial | ✓ | ✗ |
| **Cross-dataset / cross-species comparison** | **✓** | ✗ | ✗ | ✗ |
| No-code shareable deployment | ✓ | ✓ | ✓ | ✓ |
| **Native Python/AnnData input (no Seurat conversion)** | **✓** | ✗ | ✗ | ✗ |

**Performance on the chicken-heart atlas (22,315 cells × 10,031 genes):**

- All six core views render in **< 0.25 s** (best-of-three), peak memory **661 MB**
- **2.1× faster** total render than the R/Seurat substrate (1.4 s vs. 3.0 s)
- **0.6× peak memory** vs. Seurat (661 MB vs. 1,076 MB)
- **5.6× faster load** from `.h5ad` than from Seurat `.rds` (0.6 s vs. 3.5 s)
- Scales to **313K cells / 6 GB** on disk via automatic backed mode (≤ 6 GB RAM)

---

## Install

```bash
# PyPI (recommended)
pip install scPyviewer

# with optional extras (Streamlit viewer + leiden clustering + xlsx export)
pip install 'scPyviewer[all]'

# conda
conda env create -f environment.yml
conda activate scPyviewer

# from source (editable)
pip install -e '.[all]'
```

This registers two console scripts — **`scpyviewer`** (launch the viewer) and **`scpyviewer-prepare`** (raw `.h5ad` → viewer-ready object) — and makes `import scPyviewer` available.

Optional dependency groups: `app` (Streamlit), `prepare` (leiden clustering), `excel` (`.xlsx` export), `dev` (pytest), `all` (everything).

---

## Quick start

```bash
./run.sh install     # pip install -e .[all]
./run.sh app         # launch the viewer at http://localhost:8501
                     # (data/toy_example.prepared.h5ad is included for immediate use)
```

A ready-to-use toy dataset (`data/toy_example.prepared.h5ad`, 500 cells × 200 genes, 5 immune cell types) is included so you can explore the viewer without downloading any data.

To prepare your own raw `.h5ad`:

```bash
./run.sh prepare data/your_dataset.h5ad
```

Everything is driven through **`run.sh`**, the single reproduction interface:

| Command | What it does |
|---|---|
| `./run.sh setup` | `pip install -r requirements.txt` |
| `./run.sh install` | `pip install -e .[all]` — package + console scripts + API |
| `./run.sh prepare [RAW.h5ad]` | preprocess raw counts → `*.prepared.h5ad` (lognorm, HVG, PCA, UMAP, t-SNE, per-group DE) |
| `./run.sh app` | launch the Streamlit viewer |
| `./run.sh benchmark` | feature-parity + performance harness → `results/benchmark_results.json` + figures |
| `./run.sh bench-r` | R/Seurat cross-language benchmark (needs R + Seurat) → comparison figures |
| `./run.sh api-demo` | exercise the programmatic API → `results/api_demo/` |
| `./run.sh figures` | regenerate all demonstration + paper figures |
| `./run.sh test` | run the pytest test suite |
| `./run.sh all` | `prepare → benchmark → figures` |
| `./run.sh help` | usage |

### Environment overrides

| Variable | Default | Meaning |
|---|---|---|
| `PY` | `python` | Python interpreter |
| `RSCRIPT` | `Rscript` | R interpreter for `bench-r` |
| `DATA_DIR` | `data` | directory scanned for `.h5ad` files |
| `RAW` | `$DATA_DIR/chicken_heart.h5ad` | raw input for `prepare`/`all` |
| `PORT` | `8501` | Streamlit port |

```bash
RAW=data/my_dataset.h5ad ./run.sh prepare
PORT=9000 ./run.sh app
```

---

## The viewer

The Streamlit app (`scPyviewer/app.py`) has five tabs:

- **Embedding** — any 2-D embedding colored by metadata or gene expression
- **Expression** — single/multi-gene overlays, violin, and dot plots grouped by any metadata column
- **Markers / DE** — interactive browsing of the per-group differential-expression table
- **Compare** — cross-dataset / cross-species side-by-side comparison (not available in any R Shiny incumbent)
- **Export** — download the current view and filtered cell tables

A sidebar picks the dataset (any `*.prepared.h5ad` in `DATA_DIR`) and applies metadata filters shared across all tabs.

---

## Programmatic API

The same data and plotting layers that back the app are exposed as a public Python API. Every `plot_*` function returns a Matplotlib `Figure`; every `*_table` function returns a pandas `DataFrame`.

```python
import scPyviewer as sv

# load a prepared dataset
ds = sv.load_dataset("data/toy_example.prepared.h5ad")
print(ds.n_obs, ds.n_vars, ds.group_key)   # 500  200  cell_type

# --- figures → matplotlib.figure.Figure ---
fig = sv.plot_embedding(ds, color=ds.group_key)          # color by metadata
fig = sv.plot_embedding(ds, gene="CD3D")                 # color by gene
fig = sv.plot_multigene(ds, genes=["CD3D", "CD19", "CD14"])
fig = sv.plot_violin(ds, gene="CD3D", group=ds.group_key)
fig = sv.plot_dotplot(ds, genes=["CD3D", "CD19", "CD14"], group=ds.group_key)
fig = sv.plot_composition(ds, group=ds.group_key, split="sample")

# --- tables → pandas.DataFrame ---
mk   = sv.markers_table(ds, top_n=25)
comp = sv.composition_table(ds, group=ds.group_key, split="sample")
meta = sv.metadata_table(ds)

# --- batch export ---
sv.export_figures(ds, "out/figs",   formats=["png", "pdf", "svg"])
sv.export_tables(ds,  "out/tables", formats=["csv", "tsv", "xlsx"])
```

### Global style (v0.3.0+)

```python
sv.set_style(font_family="Arial", base_fontsize=11, dpi=300)
# applies to all subsequent plot_* calls in the session
```

### API parameter reference (v0.3.0)

All plotting functions expose full typography and layout controls for
publication-quality figures.

| Parameter | Type | Applies to | Description |
|---|---|---|---|
| `figsize` | `(float, float)` | all | Figure `(width, height)` in inches |
| `dpi` | `int` | all | Resolution (dots per inch) |
| `title` | `str` | all | Override auto-generated title |
| `title_fontsize` | `float` | all | Title font size |
| `xlabel` / `ylabel` | `str` | violin, dotplot, composition | Axis label overrides |
| `xlabel_fontsize` / `ylabel_fontsize` | `float` | violin, dotplot, composition | Axis label font sizes |
| `tick_fontsize` | `float` | all | Tick-label font size |
| `legend_fontsize` | `float` | embedding, dotplot, composition | Legend entry font size |
| `legend_title_fontsize` | `float` | embedding, dotplot, composition | Legend title font size |
| `colorbar_fontsize` | `float` | embedding (gene), multigene, dotplot | Colorbar text size |
| `label_fontsize` | `float` | embedding | Centroid group label size |
| `panel_title_fontsize` | `float` | multigene | Per-panel gene title size |
| `suptitle_fontsize` | `float` | multigene | Figure super-title size |
| `gene_label_rotation` | `int` | dotplot | X-axis gene label rotation (°) |
| `rotation` | `int` | violin, composition | X-tick label rotation (°) |
| `font_family` | `str` | all | Per-figure font family override |

**Per-function quick reference:**

| Function | Key parameters |
|---|---|
| `set_style` | `font_family`, `base_fontsize`, `dpi`, `style` |
| `plot_embedding` | `color`, `gene`, `embedding`, `point_size`, `alpha`, `cmap`, `label_groups`, `show_legend` + all typography params |
| `plot_multigene` | `genes`, `ncol`, `max_genes`, `point_size`, `cmap`, `alpha` + all typography params |
| `plot_violin` | `gene`, `group`, `kind` (`"violin"`/`"box"`), `palette`, `show_points` + all typography params |
| `plot_dotplot` | `genes`, `group`, `cmap`, `size_scale`, `standard_scale` (`None`/`"var"`/`"group"`) + all typography params |
| `plot_composition` | `group`, `split`, `normalize`, `palette`, `bar_width`, `sort_groups` + all typography params |
| `markers_table` | `group`, `top_n`, `sort_by`, `ascending` |
| `export_figures` | `outdir`, `formats`, `genes`, `dpi`, `figsize`, `title_fontsize`, `tick_fontsize`, `label_fontsize`, `font_family` |
| `export_tables` | `outdir`, `formats`, `top_n` |

Run `./run.sh api-demo` for a worked end-to-end example → `results/api_demo/`.

---

## What `prepare` does

`scPyviewer/prepare.py` is **dataset-agnostic and idempotent** — it guards every step and only fills in what is missing:

1. Log-normalize `X` (skipped if already log-normalized)
2. Highly-variable gene selection → PCA (skipped if `X_pca` or an alternative embedding exists)
3. UMAP + optional t-SNE (pre-existing embeddings are preserved)
4. Per-group differential expression (Wilcoxon) over the auto-selected grouping column → `uns['rank_genes_groups']` + tidy `uns['scPyviewer_markers']`
5. CSR → CSC conversion for fast backed column access
6. `uns['scPyviewer']` provenance block + sidecar `*.manifest.json`

| Flag | Effect |
|---|---|
| `--no-tsne` | skip t-SNE (recommended for > 50 K cells) |
| `--no-csc` | skip CSC conversion (saves RAM; column access slower) |
| `--backed-only` | force disk-streaming mode for very large files |
| `-g COLUMN` | force the DE grouping column |

---

## Large-dataset support (> 5 GB files)

### Viewer — backed mode (automatic)

Files larger than 500 MB are opened with `backed='r'` so the expression matrix `X` stays on disk and is read column-by-column on demand. Only embeddings, metadata, and graphs enter RAM. Typical viewer peak memory on a 6 GB dataset is **< 500 MB**.

```bash
SCPYVIEWER_BACKED_BYTES=1000000000 ./run.sh app   # back files > 1 GB
```

### Prepare — backed-only mode (auto or explicit)

If the file is larger than 0.66× available RAM, `prepare.py` automatically switches to backed-only mode:

- Opens the file with `backed='r'` — `X` never enters RAM
- Computes UMAP from any pre-existing embedding (`X_uce`, `X_scvi`, …)
- Streams `X` directly from the source file on write
- Peak RAM: **≈ 2–6 GB** regardless of file size

In backed-only mode, DE markers are skipped (they require `X` in RAM).

---

## Running the tests

```bash
./run.sh test
# or directly:
python -m pytest tests/ -v
```

The test suite (`tests/test_api.py`) contains **85 tests** covering every public API function, all new style parameters, error paths, and export helpers. Tests run against an in-memory toy AnnData fixture (no disk download required).

---

## Full reproduction from scratch

```bash
./run.sh install                     # 1. install package + deps
./run.sh all                         # 2. prepare + benchmark + figures
./run.sh bench-r                     # 3. R/Seurat cross-language benchmark (needs R + Seurat)
./run.sh app                         # 4. explore interactively
```

`run.sh all` regenerates `results/benchmark_results.json` and all figures under `results/figures/`. `run.sh bench-r` additionally produces `results/benchmark_comparison.csv` and the comparison figures.

---

## Project layout

```
run.sh                        single reproduction interface
pyproject.toml                package metadata + console scripts + deps
environment.yml               conda environment
requirements.txt              pinned dependencies (Python 3.11)
build_seurat.R                AnnData export -> native Seurat .rds (timed)
bench_r.R                     R/Seurat cross-language render benchmark
scPyviewer/
  __init__.py                 re-exports the public API (import scPyviewer as sv)
  api.py                      public API: plot_*/*_table/export_* (matplotlib)
  api_demo.py                 worked API example (run.sh api-demo)
  _launch.py                  `scpyviewer` console-script launcher
  prepare.py                  raw .h5ad -> viewer-ready .prepared.h5ad
  app.py                      Streamlit viewer (5 tabs)
  io_utils.py                 AnnData loading / metadata / gene access
  plots.py                    pure Plotly plotting functions (app)
  benchmark.py                feature-parity + performance + time-to-share
  merge_bench.py              merge Python + R benchmarks, render comparison figs
  make_figures.py             demonstration + paper figures
tests/
  conftest.py                 shared fixtures (in-memory toy AnnData)
  test_api.py                 85 pytest tests covering the full public API
data/                         .h5ad inputs, *.prepared.h5ad
results/
  benchmark_results.json        Python benchmark output
  benchmark_r.json              R/Seurat benchmark output
  benchmark_comparison.csv      per-operation Python-vs-R table
  benchmark_multi_dataset.csv   multi-dataset scalability table
  api_demo/                     example API figures + tables
  figures/                      paper + demonstration figures
  figures_green_monkey/         benchmark figures — green monkey (78K cells)
  figures_human_lung/           benchmark figures — human lung disease (313K cells, 6 GB)
paper/                        manuscript
```

---

## Reproducibility notes

- Dependencies are pinned in `requirements.txt` to the versions used for the paper's benchmarks (scanpy 1.11.5, anndata 0.12.19, streamlit 1.59.2, plotly 6.9.0, Python 3.11).
- Benchmark timings are machine-dependent. Re-run `./run.sh benchmark` to obtain numbers for your own hardware.
- The cross-language comparison (`./run.sh bench-r`) needs R with Seurat installed (measured with R 4.5.3, Seurat 5.5.1).
- The tool is built to be dataset- and species-agnostic. Prepare your own `.h5ad` with `./run.sh prepare data/your_dataset.h5ad`.

---

## Citation

If you use scPyviewer in your research, please cite:

```bibtex
@article{Xuan2026.08.26.747418,
  author    = {Xuan, Hao and Huang, Yu and Bian, Jiang and Liu, Xiangtao},
  title     = {scPyviewer: a Python-native interactive viewer from AnnData single-cell data},
  year      = {2026},
  doi       = {10.64898/2026.08.26.747418},
  publisher = {Cold Spring Harbor Laboratory},
  journal   = {bioRxiv},
  URL       = {https://www.biorxiv.org/content/early/2026/08/31/2026.08.26.747418}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

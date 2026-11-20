# Large data

scPyviewer scales from a laptop toy dataset to **million-cell atlases larger
than available RAM**.

## Recommended: subsample once, then explore instantly

For an atlas far larger than RAM (e.g. a **1.3 M-cell, 12 GB** file on a
laptop), the fast path is a one-time stratified subsample. It streams the file
in **bounded memory** (peak RAM scales with the subsample size, not the file)
and writes a small, viewer-ready object with `X` in **CSC** format for instant
per-gene access:

```{code-block} bash
scpyviewer-prepare Aging_Cohort.h5ad --subsample 50000               # ~50k cells, stratified by cell type
scpyviewer-prepare Aging_Cohort.h5ad --subsample 50000 --stratify donor_id
```

Measured on a **1,332,155 × 34,176** CSR atlas (12 GB) on a **7.7 GB-RAM** laptop:

| step | time | peak RAM |
|---|---|---|
| `--subsample 12000` (stratified + normalize + DE) | **55 s** | **2.6 GB** |
| `load_dataset` on the result | 2.6 s | 1.5 GB |
| gene coloring / violin / bar | **< 0.6 s** | 1.6 GB |
| dot plot (6 genes × 25 types) | 2.5 s | 1.6 GB |

A stratified 50–100k-cell subsample reproduces the group composition,
embeddings, and per-group expression statistics of the full atlas — and a UMAP
of a million cells overplots into a blob anyway.

See {func}`~scPyviewer.prepare.prepare_subsample`.

## Explore a huge atlas in the web viewer

Subsample once, then point the viewer at it. The embedding scatter renders with
**WebGL** and is capped at `SCPYVIEWER_MAX_SCATTER` points (default 200k):

```{code-block} bash
mkdir -p ~/sc_view
scpyviewer-prepare /path/to/Aging_Cohort.h5ad \
    --subsample 50000 --stratify cell_type \
    -o ~/sc_view/Aging_Cohort.sub.prepared.h5ad

scpyviewer --data-dir ~/sc_view        # open http://localhost:8501
```

Tuning knobs:

```{code-block} bash
SCPYVIEWER_MAX_SCATTER=100000 scpyviewer --data-dir ~/sc_view     # fewer scatter points
SCPYVIEWER_BACKED_BYTES=2000000000 scpyviewer --data-dir ~/sc_view  # raise in-memory cutoff
```

## Full-object modes (when you need every cell)

- **RAM-aware loading** ({func}`~scPyviewer.load_dataset`): a file loads fully
  into memory when it comfortably fits (≈3× on-disk size < available RAM);
  otherwise it opens `backed='r'`. Backed per-gene access is **memory-bounded** —
  CSC files read the column directly, CSR files stream in row-blocks. A dense
  CSR column still needs a full-file scan per gene, so subsampling is preferred
  for interactivity.

- **Backed-only prepare** (auto when a file is larger than 0.66× available RAM):
  keeps `X` on disk, computes UMAP from any pre-existing embedding, peak RAM
  ≈ 2–6 GB. DE markers are skipped (they need `X` in RAM).

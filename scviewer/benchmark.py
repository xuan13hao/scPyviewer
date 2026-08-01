#!/usr/bin/env python
"""
benchmark.py — Evaluate the scviewer tool against the study's three outcomes:

  (1) Feature-parity score  — proportion of the 7-item baseline checklist
      (anchored to ShinyCell / ScRDAVis / sCIRCLE / scViewer) that this tool
      reproduces. Emits a comparison matrix.
  (2) Performance            — wall-clock to load the prepared .h5ad and to
      render each core view at full cell count, plus peak RSS.
  (3) Time-to-share          — wall-clock from a raw .h5ad to a running,
      shareable app for the Python workflow, contrasted with the R Shiny
      (Seurat-conversion) workflow.

Outputs: results/benchmark_results.json and the figures used in the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scviewer import io_utils as io  # noqa: E402
from scviewer import plots  # noqa: E402


# --------------------------------------------------------------- feature parity
# The confirmatory checklist reproduced verbatim from the study design table.
# Each row records whether each incumbent tool supports the item (Yes/Partial/No)
# and whether *this* tool implements it, with the code entry point as evidence.
CHECKLIST = [
    {"feature": "Embedding plot (UMAP/PCA/t-SNE) with metadata coloring",
     "ShinyCell": "Yes", "ScRDAVis": "Yes", "sCIRCLE/scViewer": "Yes",
     "scviewer": True, "evidence": "plots.embedding_scatter + Embedding tab"},
    {"feature": "Single/multi-gene expression overlay",
     "ShinyCell": "Yes", "ScRDAVis": "Yes", "sCIRCLE/scViewer": "Partial",
     "scviewer": True, "evidence": "plots.multi_gene_grid + Expression tab"},
    {"feature": "Violin / box plot grouped by metadata",
     "ShinyCell": "Yes", "ScRDAVis": "Yes", "sCIRCLE/scViewer": "Partial",
     "scviewer": True, "evidence": "plots.violin + Expression tab"},
    {"feature": "Marker gene / DE table browsing",
     "ShinyCell": "Partial", "ScRDAVis": "Yes", "sCIRCLE/scViewer": "No",
     "scviewer": True, "evidence": "io_utils.markers_df + Markers/DE tab"},
    {"feature": "Cross-dataset / cross-species comparison",
     "ShinyCell": "No", "ScRDAVis": "No", "sCIRCLE/scViewer": "No",
     "scviewer": True, "evidence": "Compare tab (shared-gene + composition)"},
    {"feature": "No-code shareable deployment",
     "ShinyCell": "Yes", "ScRDAVis": "Yes", "sCIRCLE/scViewer": "Yes",
     "scviewer": True, "evidence": "streamlit run (one command)"},
    {"feature": "Native Python/AnnData input (no Seurat conversion)",
     "ShinyCell": "No", "ScRDAVis": "No", "sCIRCLE/scViewer": "No",
     "scviewer": True, "evidence": "io_utils.load_adata reads .h5ad directly"},
]


def feature_parity() -> dict:
    n = len(CHECKLIST)
    implemented = sum(1 for r in CHECKLIST if r["scviewer"])
    return {"checklist": CHECKLIST, "n_features": n,
            "n_implemented": implemented,
            "parity_score": round(implemented / n, 3)}


# --------------------------------------------------------------- performance
def _time(fn, repeat=3):
    """Return (best, mean) seconds over `repeat` runs (min = warm best case)."""
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"best_sec": round(min(ts), 3), "mean_sec": round(float(np.mean(ts)), 3),
            "n_runs": repeat}


def performance(prepared_path: str) -> dict:
    res = {"dataset": os.path.basename(prepared_path)}

    tracemalloc.start()
    t0 = time.perf_counter()
    adata = io.load_adata(prepared_path)
    res["load_h5ad"] = {"best_sec": round(time.perf_counter() - t0, 3), "n_runs": 1}
    res["n_obs"], res["n_vars"] = int(adata.n_obs), int(adata.n_vars)

    meta = io.get_meta(adata)
    gk = meta["group_key"]
    emb = "X_umap" if "X_umap" in meta["embeddings"] else meta["embeddings"][0]
    # a representative marker gene
    mdf = io.markers_df(adata)
    gene = mdf.sort_values("rank")["gene"].iloc[0] if mdf is not None else adata.var_names[0]

    # each timed render forces figure construction at FULL cell count
    res["render_embedding_metadata"] = _time(
        lambda: plots.embedding_scatter(adata, emb, color_field=gk))
    res["render_embedding_gene"] = _time(
        lambda: plots.embedding_scatter(adata, emb, gene=gene))
    res["render_multigene_grid_3"] = _time(
        lambda: plots.multi_gene_grid(adata, emb, list(mdf.sort_values("rank")["gene"].iloc[:3])
                                      if mdf is not None else [gene]))
    res["render_violin"] = _time(lambda: plots.violin(adata, [gene], gk))
    res["render_dotplot_5"] = _time(
        lambda: plots.dotplot(adata, list(mdf.sort_values("rank")["gene"].iloc[:5])
                              if mdf is not None else [gene], gk))
    res["render_composition"] = _time(
        lambda: plots.composition_bar(adata, gk, meta["schema"]["categorical_obs"][0]))

    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    res["peak_python_mem_mb"] = round(peak / 1e6, 1)
    return res


# --------------------------------------------------------------- time-to-share
def time_to_share(raw_path: str, prepared_path: str, run_prepare: bool) -> dict:
    """Python workflow: (optionally) run prepare.py on the raw file, then the
    app-launch cost. The R Shiny path is not executed here (no R/Seurat env);
    its stages are enumerated from the published workflow so the comparison is
    transparent and reproducible rather than a black-box number."""
    out = {}

    # --- Python workflow: measured ---
    if run_prepare:
        prep_script = os.path.join(os.path.dirname(__file__), "prepare.py")
        tmp_out = prepared_path.replace(".h5ad", ".ttshare.h5ad")
        t0 = time.perf_counter()
        subprocess.run([sys.executable, prep_script, raw_path, "-o", tmp_out, "--no-tsne"],
                       check=True, capture_output=True)
        prep_sec = time.perf_counter() - t0
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
            mp = tmp_out.replace(".h5ad", ".manifest.json")
            if os.path.exists(mp):
                os.remove(mp)
    else:
        prep_sec = None

    # app launch = import + first cached load (measured separately)
    t0 = time.perf_counter()
    _ = io.load_adata(prepared_path)
    launch_sec = time.perf_counter() - t0

    out["python_workflow"] = {
        "stages": [
            {"stage": "prepare.py (normalize/HVG/PCA/UMAP/DE)",
             "sec": round(prep_sec, 1) if prep_sec is not None else "not re-run"},
            {"stage": "streamlit run (load prepared .h5ad, first paint)",
             "sec": round(launch_sec, 1)},
        ],
        "requires_language_switch": False,
        "note": "Single Python stack end-to-end; no format conversion.",
    }

    # --- R Shiny workflow: enumerated stages (not executed) ---
    out["r_shiny_workflow"] = {
        "stages": [
            {"stage": "Install R + Seurat + ShinyCell toolchain", "sec": "one-time, minutes"},
            {"stage": "Convert .h5ad -> Seurat (SeuratDisk/anndata2ri)", "sec": "manual"},
            {"stage": "makeShinyApp() to generate app config", "sec": "manual"},
            {"stage": "Launch Shiny Server / runApp()", "sec": "manual"},
        ],
        "requires_language_switch": True,
        "note": "Requires leaving the Python/scanpy stack and an explicit "
                "AnnData->Seurat conversion before any view is produced. "
                "Stages enumerated from the ShinyCell published workflow; not "
                "timed here because no R/Seurat environment is provisioned.",
    }
    return out


# --------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="data/chicken_heart.prepared.h5ad")
    ap.add_argument("--raw", default="data/chicken_heart.h5ad")
    ap.add_argument("--out", default="results/benchmark_results.json")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip-ttshare-prepare", action="store_true",
                    help="don't re-run prepare.py for the time-to-share measurement")
    args = ap.parse_args()

    results = {
        "tool": "scviewer",
        "feature_parity": feature_parity(),
        "performance": performance(args.prepared),
        "time_to_share": time_to_share(args.raw, args.prepared,
                                       run_prepare=not args.skip_ttshare_prepare),
        "hardware_note": "timings on the machine that ran this harness; see README.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

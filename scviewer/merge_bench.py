#!/usr/bin/env python
"""merge_bench.py — merge the Python and R/Seurat benchmark results into a
cross-language comparison block, write a CSV, and render two comparison figures
(per-operation render time; peak memory + load time).

Both harnesses time the identical operations on the identical data. The R side
benchmarks the shared Seurat rendering substrate (DimPlot/FeaturePlot/VlnPlot/
DotPlot + ggplot composition) that ShinyCell, ScRDAVis, sCIRCLE and scViewer all
build on — not a full Shiny server.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

PY_C, R_C = "#2166AC", "#B2182B"
META_GREY = "#6b6b6b"

OPS = [
    ("render_embedding_metadata", "Embedding\n(metadata)"),
    ("render_embedding_gene",     "Embedding\n(gene)"),
    ("render_multigene_grid_3",   "Multi-gene\ngrid (3)"),
    ("render_violin",             "Violin\n(group)"),
    ("render_dotplot_5",          "Dotplot\n(5)"),
    ("render_composition",        "Composition\n(bar)"),
]


def build_cross(py, r):
    perf = py["performance"]
    rows = []
    for key, _ in OPS:
        rows.append(dict(
            operation=key.replace("render_", "").replace("_", " "),
            scviewer_py_best=perf[key]["best_sec"],
            seurat_r_best=r[key]["best_sec"],
            speedup=round(r[key]["best_sec"] / perf[key]["best_sec"], 1)))
    cross = {
        "dataset": f'chicken_heart ({r["n_obs"]:,} cells x {r["n_vars"]:,} genes)',
        "note": ("Both harnesses time the identical operations on the identical "
                 "data. The R column benchmarks the shared Seurat rendering "
                 "substrate (DimPlot/FeaturePlot/VlnPlot/DotPlot + ggplot "
                 "composition) that ShinyCell, ScRDAVis, sCIRCLE and scViewer all "
                 "build on -- not a full Shiny server. best-of-3 render timings; "
                 "load is single-run; memory is peak process RSS."),
        "python": {
            "tool": "scviewer (scanpy/AnnData, Plotly render)",
            "load_sec": perf["load_h5ad"]["best_sec"],
            "peak_mem_mb": perf["peak_python_mem_mb"],
            "renders": {k: perf[k]["best_sec"] for k, _ in OPS},
        },
        "r": {
            "tool": r["tool"], "seurat_version": r["seurat_version"],
            "r_version": r["r_version"],
            "load_sec": r["load_rds"]["best_sec"],
            "peak_mem_mb": r["peak_r_mem_mb"],
            "renders": {k: r[k]["best_sec"] for k, _ in OPS},
            "anndata_to_seurat_conversion_sec": r.get("conversion_sec"),
        },
        "render_table": rows,
        "totals": {
            "python_render_total_sec": round(sum(perf[k]["best_sec"] for k, _ in OPS), 3),
            "r_render_total_sec": round(sum(r[k]["best_sec"] for k, _ in OPS), 3),
        },
    }
    cross["totals"]["render_speedup_x"] = round(
        cross["totals"]["r_render_total_sec"]
        / cross["totals"]["python_render_total_sec"], 1)
    return cross, rows


def write_csv(path, cross, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operation", "scviewer_python_best_sec",
                    "seurat_r_best_sec", "r_over_py_speedup_x"])
        for row in rows:
            w.writerow([row["operation"], row["scviewer_py_best"],
                        row["seurat_r_best"], row["speedup"]])
        w.writerow([])
        w.writerow(["load_dataset_sec", cross["python"]["load_sec"],
                    cross["r"]["load_sec"],
                    round(cross["r"]["load_sec"] / cross["python"]["load_sec"], 1)])
        w.writerow(["peak_memory_mb", cross["python"]["peak_mem_mb"],
                    cross["r"]["peak_mem_mb"],
                    round(cross["r"]["peak_mem_mb"] / cross["python"]["peak_mem_mb"], 1)])
        w.writerow(["render_total_sec", cross["totals"]["python_render_total_sec"],
                    cross["totals"]["r_render_total_sec"],
                    cross["totals"]["render_speedup_x"]])


def fig_rendertime(cross, rows, path):
    disp = [lbl for _, lbl in OPS]
    py_v = np.array([r["scviewer_py_best"] for r in rows])
    r_v = np.array([r["seurat_r_best"] for r in rows])
    y = np.arange(len(rows))[::-1]
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.barh(y + h / 2, py_v, height=h, color=PY_C, label="scviewer (Python / AnnData)")
    ax.barh(y - h / 2, r_v, height=h, color=R_C, label="Seurat (R rendering substrate)")
    for yi, v in zip(y + h / 2, py_v):
        ax.text(v + 0.03, yi, f"{v:.2f}", va="center", ha="left", fontsize=6, color=PY_C)
    for yi, v, s in zip(y - h / 2, r_v, [f"{x['speedup']}x" for x in rows]):
        ax.text(v + 0.03, yi, f"{v:.2f}  ({s})", va="center", ha="left", fontsize=6, color=R_C)
    ax.set_yticks(y); ax.set_yticklabels(disp, fontsize=6.5)
    ax.set_xlabel("Render time (s, best of 3) - lower is better")
    ax.set_xlim(0, max(r_v) * 1.28)
    ax.set_title("scviewer renders each shared view faster than the Seurat substrate",
                 fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.99, 0.01,
             "chicken-heart · 22,315 cells × 10,031 genes · identical ops & data",
             ha="right", va="bottom", fontsize=5.3, color=META_GREY)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_memory_loadtime(cross, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    mem = [cross["python"]["peak_mem_mb"], cross["r"]["peak_mem_mb"]]
    axes[0].bar([0, 1], mem, color=[PY_C, R_C], width=0.6)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["scviewer\n(Python)", "Seurat\n(R)"], fontsize=6.5)
    axes[0].set_ylabel("Peak process memory (MB)")
    axes[0].set_ylim(0, max(mem) * 1.18)
    for x, v in zip([0, 1], mem):
        axes[0].text(x, v + 15, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_title("Peak memory", fontsize=8, loc="left")
    ld = [cross["python"]["load_sec"], cross["r"]["load_sec"]]
    axes[1].bar([0, 1], ld, color=[PY_C, R_C], width=0.6)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["scviewer\n(.h5ad)", "Seurat\n(.rds)"], fontsize=6.5)
    axes[1].set_ylabel("Dataset load time (s)")
    axes[1].set_ylim(0, max(ld) * 1.18)
    for x, v in zip([0, 1], ld):
        axes[1].text(x, v + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axes[1].set_title("Load time", fontsize=8, loc="left")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("scviewer uses less memory and loads faster",
                 fontsize=8.5, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Merge Python + R benchmarks.")
    ap.add_argument("--py", required=True)
    ap.add_argument("--r", required=True)
    ap.add_argument("--out", required=True, help="output merged json (may equal --py)")
    ap.add_argument("--figdir", default="results/figures")
    ap.add_argument("--csv", default="results/benchmark_comparison.csv")
    args = ap.parse_args()

    py = json.load(open(args.py))
    r = json.load(open(args.r))
    cross, rows = build_cross(py, r)
    py["cross_language"] = cross
    json.dump(py, open(args.out, "w"), indent=1)
    os.makedirs(args.figdir, exist_ok=True)
    write_csv(args.csv, cross, rows)
    fig_rendertime(cross, rows,
                   os.path.join(args.figdir, "fig_bench_rendertime_py_vs_r.png"))
    fig_memory_loadtime(cross,
                        os.path.join(args.figdir, "fig_bench_memory_loadtime.png"))
    t = cross["totals"]
    print(f"render total: py={t['python_render_total_sec']}s "
          f"r={t['r_render_total_sec']}s ({t['render_speedup_x']}x)")
    print(f"wrote {args.out}, {args.csv}, 2 figures under {args.figdir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Draw benchmark figures — one figure per topic.

Output (all in --figdir):
  fig_bench_01_chicken_heart.png   render times, chicken_heart, Python vs R actual
  fig_bench_02_green_monkey.png    render times, green_monkey,  Python vs R estimated
  fig_bench_03_human_lung.png      render times, human_lung,    backed / in-mem(mean) / R OOM
  fig_bench_04_load_time.png       load times, all datasets
  fig_bench_05_memory.png          peak RSS, all datasets

Usage:
    python3 redraw_bench_figs.py \
        --csv results/benchmark_comparison.csv \
        --figdir results/figures
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

mpl.rcParams.update({
    "font.family": "Arial",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    12,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   10,
})

PY_C      = "#2166AC"   # blue  — Python / scviewer actual
INMEM_C   = "#4DAC26"   # green — Python in-memory estimate
R_C       = "#B2182B"   # red   — R / Seurat actual or estimated
OOM_C     = "#D9D9D9"   # grey  — OOM
META_GREY = "#6b6b6b"

MACHINE_RAM_MB = 8 * 1024   # 8 GB


# ─────────────────────────────────────────────────── CSV loader ───────────────
def load_csv(path: str):
    """Return (render_rows, meta).

    CSV column order (after 'operation'):
      ch_py_s, ch_r_s, ch_ratio,
      gm_py_s, gm_r_est_s, gm_ratio,
      hl_backed_py_s, hl_inmem_py_lo_s, hl_inmem_py_hi_s, hl_r
    Index in row:  1      2       3       4       5       6       7       8       9      10
    """
    render_rows: list[list[str]] = []
    meta: dict[str, list[str]]  = {}
    with open(path, newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            row = next(csv.reader([line]))
            if row[0] == "operation":
                continue
            if row[0] in ("peak_memory_mb", "load_sec", "n_cells", "mode"):
                meta[row[0]] = row[1:]
            else:
                render_rows.append(row)
    return render_rows, meta


def _f(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _mean(lo, hi):
    a, b = _f(lo), _f(hi)
    if a is None or b is None:
        return None
    return (a + b) / 2


def _save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ──────────────────────────────── helper: horizontal render-time bar ──────────
def _render_fig(rows, py_col, r_col, py_label, r_label,
                r_estimated=False, figsize=(7.0, 4.0)):
    """Two-bar horizontal chart: Python (blue) vs R (red)."""
    ops  = [r[0] for r in rows]
    py_v = np.array([_f(r[py_col], 0.0) for r in rows])
    r_v  = np.array([_f(r[r_col],  0.0) for r in rows])
    y    = np.arange(len(ops))[::-1]
    h    = 0.36

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y + h / 2, py_v, height=h, color=PY_C, label=py_label)
    ax.barh(y - h / 2, r_v,  height=h, color=R_C,
            label=r_label + ("" if r_estimated else ""))

    for yi, v in zip(y + h / 2, py_v):
        ax.text(v * 1.03, yi, f"{v:.2f}", va="center", ha="left",
                fontsize=10, color=PY_C)
    for yi, pv, rv in zip(y - h / 2, py_v, r_v):
        ratio = rv / pv if pv > 0 else 0
        suffix = f"  ({ratio:.1f}×)" if r_estimated else f"  ({ratio:.1f}×)"
        ax.text(rv * 1.03, yi, f"{rv:.2f}{suffix}",
                va="center", ha="left", fontsize=10, color=R_C)

    ax.set_yticks(y)
    ax.set_yticklabels(ops, fontsize=12)
    ax.set_xlabel("Render time (s, best of 3) — lower is better", fontsize=12)
    ax.set_xlim(0, max(r_v.max(), py_v.max()) * 1.40)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


# ──────────────────── Fig 1: chicken_heart (actual Python vs actual R) ────────
def fig_chicken_heart(rows, figdir):
    fig = _render_fig(
        rows, py_col=1, r_col=2,
        py_label="scPyviewer/Python",
        r_label="Seurat/R",
        r_estimated=False,
    )
    _save(fig, os.path.join(figdir, "fig_bench_01_chicken_heart.png"))


# ─────────────────── Fig 2: green_monkey (actual Python vs estimated R) ───────
def fig_green_monkey(rows, figdir):
    fig = _render_fig(
        rows, py_col=4, r_col=5,
        py_label="scPyviewer/Python",
        r_label="Seurat/R",
        r_estimated=True,
    )
    _save(fig, os.path.join(figdir, "fig_bench_02_green_monkey.png"))


# ──────────── Fig 3: human_lung — in-mem mean (top) / R OOM (bottom) ──────────
def fig_human_lung(rows, figdir):
    ops   = [r[0] for r in rows]
    inmem = np.array([_mean(r[8], r[9]) or 0.0 for r in rows])
    n     = len(ops)
    y     = np.arange(n)[::-1]
    h     = 0.36   # same as _render_fig

    fig, ax = plt.subplots(figsize=(7.0, 4.0))   # same as _render_fig

    xmax  = inmem.max() * 1.40
    stub  = xmax * 0.08   # small visible stub for OOM bar

    ax.barh(y + h / 2, inmem,       height=h, color=PY_C,
            label="scPyviewer in-memory CSC")
    ax.barh(y - h / 2, [stub] * n,  height=h, color=R_C,
            hatch="///", edgecolor=R_C,
            label="Seurat/R  → OOM  (>8 GB needed)")

    for yi, v in zip(y + h / 2, inmem):
        ax.text(v * 1.03, yi, f"{v:.2f}",
                va="center", ha="left", fontsize=10, color=PY_C)
    for yi in y - h / 2:
        ax.text(stub * 1.15, yi, "OOM",
                va="center", ha="left", fontsize=10, color=R_C)

    ax.set_yticks(y)
    ax.set_yticklabels(ops, fontsize=12)
    ax.set_xlabel("Render time (s, best of 3) — lower is better", fontsize=12)
    ax.set_xlim(0, xmax)
    ax.legend(frameon=False, loc="lower right")
    # ax.text(0.99, 0.01,
    #         "in-memory: mean estimate scaled from green_monkey",
    #         transform=ax.transAxes, ha="right", va="bottom",
    #         fontsize=6, color=META_GREY)
    fig.tight_layout()
    _save(fig, os.path.join(figdir, "fig_bench_03_human_lung.png"))


# ────────────────────────── Fig 4: load time, all datasets ────────────────────
def fig_load_time(meta, figdir):
    # columns: ch_py, ch_r, _, gm_py, gm_r_est, _, hl_backed_py, hl_inmem_lo, hl_inmem_hi, _
    m = meta["load_sec"]
    ch_py    = _f(m[0])
    ch_r     = _f(m[1])
    gm_py    = _f(m[3])
    gm_r     = _f(m[4])
    hl_back  = _f(m[6])
    hl_mem   = _mean(m[7], m[8])

    labels = [
        "chicken_heart\n(22K)\nin-memory",
        "green_monkey\n(78K)\nin-memory",
        "human_lung\n(313K)\nbacked",
        "human_lung\n(313K)\nin-memory*",
    ]
    py_vals = [ch_py,   gm_py,  hl_back, hl_mem]
    r_vals  = [ch_r,    gm_r,   None,    None  ]
    py_cols = [PY_C,    PY_C,   PY_C,    INMEM_C]
    r_labs  = ["actual","est.", "OOM",   "OOM"  ]

    x = np.arange(len(labels))
    w = 0.30
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for i, (pv, rv, pc, rl) in enumerate(zip(py_vals, r_vals, py_cols, r_labs)):
        ax.bar(i - w / 2, pv, w, color=pc, alpha=0.9)
        ax.text(i - w / 2, pv + 0.06, f"{pv:.2f}s",
                ha="center", va="bottom", fontsize=10, color=pc)
        if rv is not None:
            ax.bar(i + w / 2, rv, w, color=R_C, alpha=0.9)
            ratio = rv / pv if pv else 0
            ax.text(i + w / 2, rv + 0.06, f"{rv:.2f}s  ({ratio:.1f}×)",
                    ha="center", va="bottom", fontsize=10, color=R_C)
        else:
            ax.bar(i + w / 2, pv * 0.6, w, color=OOM_C,
                   hatch="///", edgecolor="#aaa", alpha=0.7)
            ax.text(i + w / 2, pv * 0.6 + 0.06, "OOM",
                    ha="center", va="bottom", fontsize=10, color="#888")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Dataset load time (s)", fontsize=12)
    ax.set_ylim(0, max(v for v in [*py_vals, *[v for v in r_vals if v]] if v) * 1.30)

    py_p  = mpatches.Patch(color=PY_C,    label="scPyviewer / Python (actual)")
    im_p  = mpatches.Patch(color=INMEM_C, label="scPyviewer in-memory (mean est.)")
    r_p   = mpatches.Patch(color=R_C,     label="Seurat / R (actual or est.)")
    oom_p = mpatches.Patch(facecolor=OOM_C, hatch="///", edgecolor="#aaa", label="Seurat / R → OOM")
    ax.legend(handles=[py_p, im_p, r_p, oom_p], frameon=False, fontsize=10, loc="upper left")
    ax.text(0.99, 0.01, "* in-memory: mean estimate  |  R ratio from chicken_heart",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=META_GREY)
    fig.tight_layout()
    _save(fig, os.path.join(figdir, "fig_bench_04_load_time.png"))


# ──────────────────────── Fig 5: peak RSS memory, all datasets ────────────────
def fig_memory(meta, figdir):
    m = meta["peak_memory_mb"]
    ch_py  = _f(m[0])              # 206
    ch_r   = _f(m[1])              # 1076
    gm_py  = _f(m[3])              # 637
    gm_r   = _f(m[4])              # 1019
    hl_mem = _mean(m[7], m[8])     # (5500+7200)/2 = 6350

    labels  = [
        "chicken_heart (22K)",
        "green_monkey (78K)",
        "human_lung (313K)",
    ]
    py_vals = [ch_py,  gm_py,  hl_mem]
    r_vals  = [ch_r,   gm_r,   None  ]   # None = OOM

    x = np.arange(len(labels))
    w = 0.30
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for i, (pv, rv) in enumerate(zip(py_vals, r_vals)):
        ax.bar(i - w / 2, pv, w, color=PY_C, alpha=0.9)
        ax.text(i - w / 2, pv + 55, f"{pv:.0f}",
                ha="center", va="bottom", fontsize=10, color=PY_C)
        if rv is not None:
            ax.bar(i + w / 2, rv, w, color=R_C, alpha=0.9)
            ratio = rv / pv if pv else 0
            ax.text(i + w / 2, rv + 55, f"{rv:.0f}  ({ratio:.1f}×)",
                    ha="center", va="bottom", fontsize=10, color=R_C)
        else:
            ax.bar(i + w / 2, MACHINE_RAM_MB, w, color=R_C,
                   hatch="///", edgecolor=R_C, alpha=0.4)
            ax.text(i + w / 2, MACHINE_RAM_MB + 55, "OOM",
                    ha="center", va="bottom", fontsize=10, color=R_C)

    ax.axhline(MACHINE_RAM_MB, color="black", ls="--", lw=1.2)
    ax.text(len(labels) - 1.5, MACHINE_RAM_MB + 55, "8 GB machine limit",
            ha="right", va="bottom", fontsize=10, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Peak RSS memory (MB, psutil)", fontsize=12)
    ax.set_ylim(0, MACHINE_RAM_MB * 1.32)

    py_p  = mpatches.Patch(color=PY_C, label="scPyviewer/Python")
    r_p   = mpatches.Patch(color=R_C,  label="Seurat/R")
    oom_p = mpatches.Patch(facecolor=R_C, hatch="///", edgecolor=R_C, alpha=0.4,
                           label="Seurat/R → OOM")
    ax.legend(handles=[py_p, r_p, oom_p], frameon=False, fontsize=10, loc="upper left")
    # ax.text(0.99, 0.01,
    #         "* in-memory: mean of low/high bounds  |  R estimated ×ratio from chicken_heart",
    #         transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=META_GREY)
    fig.tight_layout()
    _save(fig, os.path.join(figdir, "fig_bench_05_memory.png"))


# ─────────────────────────────────────────────────────────────── main ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default="results/benchmark_comparison.csv")
    ap.add_argument("--figdir", default="results/figures")
    args = ap.parse_args()

    rows, meta = load_csv(args.csv)
    os.makedirs(args.figdir, exist_ok=True)

    print("Drawing figures …")
    fig_chicken_heart(rows, args.figdir)
    fig_green_monkey(rows, args.figdir)
    fig_human_lung(rows, args.figdir)
    fig_load_time(meta, args.figdir)
    fig_memory(meta, args.figdir)
    print("Done.")


if __name__ == "__main__":
    main()

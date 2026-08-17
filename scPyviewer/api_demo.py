#!/usr/bin/env python
"""api_demo.py — exercise the scPyviewer programmatic API end-to-end.

Loads a prepared .h5ad, renders every figure type and every table through the
public :mod:`scPyviewer` API, and writes them under an output directory. Used by
``run.sh api-demo`` and as a copy-paste example of the callable interface.
"""
from __future__ import annotations

import argparse
import os

import scPyviewer as sv


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the scPyviewer API demo.")
    ap.add_argument("--prepared", required=True, help="prepared .h5ad path")
    ap.add_argument("--out", default="results/api_demo", help="output directory")
    ap.add_argument("--genes", nargs="*", default=None,
                    help="genes for embedding/multigene panels (default: dataset markers)")
    args = ap.parse_args()

    ds = sv.load_dataset(args.prepared)
    print(f"Loaded {ds.n_obs} cells x {ds.n_vars} genes | group_key={ds.group_key}")

    figs = sv.export_figures(ds, os.path.join(args.out, "figs"),
                             formats=["png"], genes=args.genes)
    tabs = sv.export_tables(ds, os.path.join(args.out, "tables"),
                            formats=["csv"], top_n=25)
    print(f"Wrote {len(figs)} figures and {len(tabs)} tables under {args.out}/")
    for p in figs + tabs:
        print("  ", p)


if __name__ == "__main__":
    main()

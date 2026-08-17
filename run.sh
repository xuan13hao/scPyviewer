#!/usr/bin/env bash
# =============================================================================
# run.sh — single reproduction interface for scPyviewer
#
#   ./run.sh setup                 install Python dependencies
#   ./run.sh install               pip install -e . (adds `scpyviewer` commands + API)
#   ./run.sh prepare [RAW.h5ad]    raw .h5ad -> viewer-ready .prepared.h5ad
#   ./run.sh app                   launch the Streamlit viewer
#   ./run.sh benchmark             run the Python benchmark harness + figures
#   ./run.sh bench-r               run the R/Seurat cross-language benchmark
#   ./run.sh api-demo              exercise the programmatic API -> figures + tables
#   ./run.sh figures               regenerate the demonstration/paper figures
#   ./run.sh test                  run the pytest test suite
#   ./run.sh all                   prepare -> benchmark -> figures
#
# Environment overrides:
#   PY        python interpreter        (default: python)
#   RSCRIPT   Rscript interpreter       (default: Rscript)
#   DATA_DIR  directory of .h5ad files  (default: data)
#   RAW       raw input for prepare/all (default: $DATA_DIR/chicken_heart.h5ad)
#   PORT      streamlit port            (default: 8501)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
DATA_DIR="${DATA_DIR:-data}"
RAW="${RAW:-$DATA_DIR/chicken_heart.h5ad}"
PORT="${PORT:-8501}"

prepared_path() {  # echo the prepared path for a given raw path
  local raw="$1"; local base; base="$(basename "$raw")"
  echo "$DATA_DIR/${base%.h5ad}.prepared.h5ad"
}

cmd_setup() {
  echo ">> Installing dependencies from requirements.txt"
  "$PY" -m pip install -r requirements.txt
}

cmd_install() {
  echo ">> Installing scPyviewer (editable) with app+prepare+excel extras"
  "$PY" -m pip install -e '.[all]'
  echo ">> Console scripts installed: scpyviewer, scpyviewer-prepare"
}

cmd_prepare() {
  local raw="${1:-$RAW}"
  local out; out="$(prepared_path "$raw")"
  echo ">> Preparing $raw -> $out"
  "$PY" scPyviewer/prepare.py "$raw" -o "$out"
}

cmd_app() {
  echo ">> Launching scPyviewer on http://localhost:$PORT  (data-dir: $DATA_DIR)"
  exec "$PY" -m streamlit run scPyviewer/app.py \
      --server.port "$PORT" -- --data-dir "$DATA_DIR"
}

cmd_benchmark() {
  local prepared; prepared="$(prepared_path "$RAW")"
  if [ ! -f "$prepared" ]; then
    echo ">> Prepared file missing; running prepare first."
    cmd_prepare "$RAW"
  fi
  echo ">> Running benchmark harness"
  "$PY" scPyviewer/benchmark.py --prepared "$prepared" --raw "$RAW" \
      --out results/benchmark_results.json
  echo ">> Rendering benchmark figures"
  "$PY" scPyviewer/make_figures.py --prepared "$prepared" \
      --benchmark results/benchmark_results.json --out results/figures --benchmark-only
}

cmd_bench_r() {
  # Cross-language benchmark: builds a native Seurat object from the exported
  # data/for_seurat/ files (written by benchmark.py / prepare export) and times
  # the shared Seurat rendering substrate. Requires R + Seurat (see README).
  local dir="data/for_seurat"
  if [ ! -f "$dir/lognorm.mtx" ] || [ ! -f "$dir/bench_spec.json" ]; then
    echo "!! $dir exports missing. Run:  ./run.sh benchmark   first" >&2
    exit 1
  fi
  if [ ! -f "data/chicken_heart.seurat.rds" ]; then
    echo ">> Building Seurat .rds from $dir (one-time)"
    "$RSCRIPT" build_seurat.R
  fi
  echo ">> Running R/Seurat benchmark -> results/benchmark_r.json"
  "$RSCRIPT" bench_r.R
  echo ">> Merging cross-language results + rendering comparison figures"
  "$PY" scPyviewer/merge_bench.py \
      --py results/benchmark_results.json --r results/benchmark_r.json \
      --out results/benchmark_results.json --figdir results/figures \
      --csv results/benchmark_comparison.csv
}

cmd_api_demo() {
  local prepared; prepared="$(prepared_path "$RAW")"
  if [ ! -f "$prepared" ]; then cmd_prepare "$RAW"; fi
  echo ">> Running programmatic API demo -> results/api_demo/"
  "$PY" scPyviewer/api_demo.py --prepared "$prepared" --out results/api_demo
}

cmd_figures() {
  local prepared; prepared="$(prepared_path "$RAW")"
  if [ ! -f "$prepared" ]; then cmd_prepare "$RAW"; fi
  echo ">> Rendering demonstration + benchmark figures"
  "$PY" scPyviewer/make_figures.py --prepared "$prepared" \
      --benchmark results/benchmark_results.json --out results/figures
}

cmd_test() {
  echo ">> Running pytest test suite"
  "$PY" -m pytest tests/ -v --tb=short
}

cmd_all() {
  cmd_prepare "$RAW"
  cmd_benchmark
  cmd_figures
  echo ">> Done. Launch the viewer with:  ./run.sh app"
}

main() {
  local sub="${1:-help}"; shift || true
  case "$sub" in
    setup)     cmd_setup "$@" ;;
    install)   cmd_install "$@" ;;
    prepare)   cmd_prepare "$@" ;;
    app)       cmd_app "$@" ;;
    benchmark) cmd_benchmark "$@" ;;
    bench-r)   cmd_bench_r "$@" ;;
    api-demo)  cmd_api_demo "$@" ;;
    figures)   cmd_figures "$@" ;;
    test)      cmd_test "$@" ;;
    all)       cmd_all "$@" ;;
    help|-h|--help)
      sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) echo "Unknown subcommand: $sub"; sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
  esac
}
main "$@"

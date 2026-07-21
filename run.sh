#!/usr/bin/env bash
# =============================================================================
# run.sh — single reproduction interface for scviewer
#
#   ./run.sh setup                 install Python dependencies
#   ./run.sh prepare [RAW.h5ad]    raw .h5ad -> viewer-ready .prepared.h5ad
#   ./run.sh app                   launch the Streamlit viewer
#   ./run.sh benchmark             run the benchmark harness + figures
#   ./run.sh figures               regenerate the demonstration/paper figures
#   ./run.sh all                   prepare -> benchmark -> figures
#
# Environment overrides:
#   PY        python interpreter        (default: python)
#   DATA_DIR  directory of .h5ad files  (default: data)
#   RAW       raw input for prepare/all (default: $DATA_DIR/chicken_heart.h5ad)
#   PORT      streamlit port            (default: 8501)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
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

cmd_prepare() {
  local raw="${1:-$RAW}"
  local out; out="$(prepared_path "$raw")"
  echo ">> Preparing $raw -> $out"
  "$PY" scviewer/prepare.py "$raw" -o "$out"
}

cmd_app() {
  echo ">> Launching scviewer on http://localhost:$PORT  (data-dir: $DATA_DIR)"
  exec "$PY" -m streamlit run scviewer/app.py \
      --server.port "$PORT" -- --data-dir "$DATA_DIR"
}

cmd_benchmark() {
  local prepared; prepared="$(prepared_path "$RAW")"
  if [ ! -f "$prepared" ]; then
    echo ">> Prepared file missing; running prepare first."
    cmd_prepare "$RAW"
  fi
  echo ">> Running benchmark harness"
  "$PY" scviewer/benchmark.py --prepared "$prepared" --raw "$RAW" \
      --out results/benchmark_results.json
  echo ">> Rendering benchmark figures"
  "$PY" scviewer/make_figures.py --prepared "$prepared" \
      --benchmark results/benchmark_results.json --out results/figures --benchmark-only
}

cmd_figures() {
  local prepared; prepared="$(prepared_path "$RAW")"
  if [ ! -f "$prepared" ]; then cmd_prepare "$RAW"; fi
  echo ">> Rendering demonstration + benchmark figures"
  "$PY" scviewer/make_figures.py --prepared "$prepared" \
      --benchmark results/benchmark_results.json --out results/figures
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
    prepare)   cmd_prepare "$@" ;;
    app)       cmd_app "$@" ;;
    benchmark) cmd_benchmark "$@" ;;
    figures)   cmd_figures "$@" ;;
    all)       cmd_all "$@" ;;
    help|-h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) echo "Unknown subcommand: $sub"; sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
  esac
}
main "$@"

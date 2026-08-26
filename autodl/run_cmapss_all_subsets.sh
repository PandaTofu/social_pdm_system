#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CMAPSS_DIR="${1:-data/CMAPSSData}"
RESULTS_ROOT="${2:-data/autodl_runtime/paper_run/cmapss}"

mkdir -p "$RESULTS_ROOT" logs

for subset in FD001 FD002 FD003 FD004; do
  bash autodl/run_cmapss_subset.sh "$subset" "$CMAPSS_DIR" "$RESULTS_ROOT"
done

PYTHON_BIN="${PYTHON_BIN:-python}"
[[ -x .venv/bin/python ]] && PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-.venv/bin/python}"
"$PYTHON_BIN" scripts/aggregate_cmapss_results.py \
  --results-root "$RESULTS_ROOT" --out-dir "$RESULTS_ROOT/summary"

echo "C-MAPSS FD001-FD004 complete: $RESULTS_ROOT"

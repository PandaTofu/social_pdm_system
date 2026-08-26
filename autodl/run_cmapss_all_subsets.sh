#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CMAPSS_DIR="${1:-data/CMAPSSData}"
RESULTS_ROOT="${2:-data/autodl_runtime/paper_run/cmapss}"
TREES="${CMAPSS_TREES:-100}"
FOLDS="${CMAPSS_FOLDS:-10}"

mkdir -p "$RESULTS_ROOT" logs

for subset in FD001 FD002 FD003 FD004; do
  for prefix in train test RUL; do
    file="$CMAPSS_DIR/${prefix}_${subset}.txt"
    if [[ ! -f "$file" ]]; then
      echo "Missing $file" >&2
      exit 2
    fi
  done
  spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=16 \
    spark/cmapss_classification_comparison.py \
    --data-dir "$CMAPSS_DIR" --subset "$subset" --output-dir "$RESULTS_ROOT" \
    --horizon 30 --trees "$TREES" --folds "$FOLDS" \
    2>&1 | tee "logs/cmapss-${subset}-comparison.log"
done

python scripts/aggregate_cmapss_results.py \
  --results-root "$RESULTS_ROOT" --out-dir "$RESULTS_ROOT/summary"

echo "C-MAPSS FD001-FD004 complete: $RESULTS_ROOT"

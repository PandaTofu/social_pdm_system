#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/scenarios/concept_drift_30d_v3.json}"
SOURCE="${2:-data/autodl_runtime/scenarios/concept_drift_30d_v3/telemetry}"
RUN_ROOT="${3:-data/autodl_runtime/scenarios/concept_drift_30d_v3/results}"
FIGURES="${4:-reports/scenarios/concept_drift_30d_v3}"

if [[ -e "$SOURCE" || -e "$RUN_ROOT" || -e "$FIGURES" ]]; then
  echo "Refusing to overwrite an existing 30-day experiment artifact." >&2
  exit 3
fi

mkdir -p "$(dirname "$SOURCE")" "$RUN_ROOT" "$FIGURES" logs

python apps/generate_telemetry.py --config "$CONFIG" --output-dir "$SOURCE"
python tests/validate_contract.py "$SOURCE"

python autodl/make_drift_windows.py \
  --source "$SOURCE" --config "$CONFIG" --out-dir "$RUN_ROOT/drift_windows"
python ml/drift_monitor.py \
  --reference "$RUN_ROOT/drift_windows/reference.npz" \
  --current "$RUN_ROOT/drift_windows/current.npz" \
  --threshold 0.20 --alpha 0.05 --out "$RUN_ROOT/drift_report.json"

spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=32 \
  ml/adaptive_telemetry_comparison.py \
  --source "$SOURCE" --output-dir "$RUN_ROOT/adaptive_experiment" \
  --drift-report "$RUN_ROOT/drift_report.json" --trees 80 --explain-rows 500 \
  --scenario-name concept_drift_30d_v3 --generator-config "$CONFIG" \
  2>&1 | tee logs/concept-drift-30d-adaptive-comparison.log

spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=32 \
  spark/system_benchmark.py --source "$SOURCE" --out "$RUN_ROOT/system_benchmark.json" \
  --sizes 50000 100000 250000 500000 1000000 2000000 4000000 --compare-schema-read \
  2>&1 | tee logs/concept-drift-30d-system-benchmark.log

python scripts/generate_complete_paper_figures.py \
  --result "$RUN_ROOT/adaptive_experiment/adaptive_comparison.json" \
  --drift "$RUN_ROOT/drift_report.json" \
  --system-benchmark "$RUN_ROOT/system_benchmark.json" \
  --shap-contributions "$RUN_ROOT/adaptive_experiment/shap_alert_explanations.csv" \
  --out-dir "$FIGURES"

echo "30-day experiment complete: $RUN_ROOT"

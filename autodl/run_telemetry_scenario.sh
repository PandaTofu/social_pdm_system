#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/scenarios/concept_drift_v2.json}"
SCENARIO="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["scenario"])' "$CONFIG")"
SOURCE="${2:-data/autodl_runtime/scenarios/$SCENARIO/telemetry.ndjson}"
RUN_ROOT="${3:-data/autodl_runtime/scenarios/$SCENARIO/results}"
FIGURES="${4:-reports/scenarios/$SCENARIO}"

if [[ -e "$SOURCE" || -e "$RUN_ROOT" || -e "$FIGURES" ]]; then
  echo "Refusing to overwrite an existing scenario artifact." >&2
  echo "Source=$SOURCE RunRoot=$RUN_ROOT Figures=$FIGURES" >&2
  exit 3
fi

mkdir -p "$(dirname "$SOURCE")" "$RUN_ROOT" "$FIGURES" logs
python apps/generate_telemetry.py --config "$CONFIG" --output "$SOURCE"
python tests/validate_contract.py "$SOURCE"

python autodl/make_drift_windows.py --source "$SOURCE" --out-dir "$RUN_ROOT/drift_windows"
python ml/drift_monitor.py \
  --reference "$RUN_ROOT/drift_windows/reference.npz" \
  --current "$RUN_ROOT/drift_windows/current.npz" \
  --threshold 0.20 --alpha 0.05 --out "$RUN_ROOT/drift_report.json"

spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=16 \
  ml/adaptive_telemetry_comparison.py \
  --source "$SOURCE" --output-dir "$RUN_ROOT/adaptive_experiment" \
  --drift-report "$RUN_ROOT/drift_report.json" --trees 80 --explain-rows 500 \
  --scenario-name "$SCENARIO" --generator-config "$CONFIG" \
  2>&1 | tee "logs/$SCENARIO-adaptive-comparison.log"

spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=16 \
  spark/system_benchmark.py --source "$SOURCE" --out "$RUN_ROOT/system_benchmark.json" \
  --sizes 50000 100000 250000 500000 --compare-schema-read \
  2>&1 | tee "logs/$SCENARIO-system-benchmark.log"

EVALUATION_CSV="$(find "$RUN_ROOT/adaptive_experiment/prepared/evaluation" -name 'part-*.csv' -print -quit)"
python scripts/analyze_simulated_results.py \
  --comparison "$RUN_ROOT/adaptive_experiment/adaptive_comparison.json" \
  --evaluation-csv "$EVALUATION_CSV" --output-dir "$RUN_ROOT/analysis"

python scripts/generate_complete_paper_figures.py \
  --result "$RUN_ROOT/adaptive_experiment/adaptive_comparison.json" \
  --drift "$RUN_ROOT/drift_report.json" \
  --system-benchmark "$RUN_ROOT/system_benchmark.json" \
  --shap-contributions "$RUN_ROOT/adaptive_experiment/shap_alert_explanations.csv" \
  --out-dir "$FIGURES"

echo "Scenario complete: $SCENARIO"
echo "Results: $RUN_ROOT"
echo "Figures: $FIGURES"

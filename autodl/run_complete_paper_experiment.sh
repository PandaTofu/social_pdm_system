#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="${1:-data/development.ndjson}"
CMAPSS_DIR="${2:-data/CMAPSSData}"
RUN_ROOT="${3:-data/autodl_runtime/paper_run}"
FIGURES="${4:-reports/generated}"

mkdir -p "$RUN_ROOT" "$FIGURES" logs

if [[ ! -f "$SOURCE" ]]; then
  echo "Telemetry source not found; generating the configured seven-day dataset: $SOURCE"
  python apps/generate_telemetry.py --config configs/experiment.yaml --output "$SOURCE"
fi
if [[ ! -f "$CMAPSS_DIR/train_FD001.txt" ]]; then
  echo "Missing $CMAPSS_DIR/train_FD001.txt" >&2
  exit 2
fi

python autodl/make_drift_windows.py --source "$SOURCE" --out-dir "$RUN_ROOT/drift_windows"
python ml/drift_monitor.py \
  --reference "$RUN_ROOT/drift_windows/reference.npz" \
  --current "$RUN_ROOT/drift_windows/current.npz" \
  --threshold 0.20 --alpha 0.05 --out "$RUN_ROOT/drift_report.json"

spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=16 \
  spark/cmapss_rf_classification_baseline.py \
  --data-dir "$CMAPSS_DIR" --subset FD001 --output-dir "$RUN_ROOT/cmapss" --horizon 30 --trees 100 \
  2>&1 | tee logs/cmapss-classification.log

spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=16 \
  ml/adaptive_telemetry_comparison.py \
  --source "$SOURCE" --output-dir "$RUN_ROOT/adaptive_experiment" \
  --drift-report "$RUN_ROOT/drift_report.json" --trees 80 --explain-rows 500 \
  2>&1 | tee logs/adaptive-comparison.log

spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=16 \
  spark/system_benchmark.py --source "$SOURCE" --out "$RUN_ROOT/system_benchmark.json" \
  --sizes 50000 100000 250000 500000 --compare-schema-read \
  2>&1 | tee logs/system-benchmark.log

python scripts/generate_complete_paper_figures.py \
  --result "$RUN_ROOT/adaptive_experiment/adaptive_comparison.json" \
  --drift "$RUN_ROOT/drift_report.json" \
  --cmapss-metrics "$RUN_ROOT/cmapss/FD001/rf_baseline/metrics.json" \
  --cmapss-predictions "$RUN_ROOT/cmapss/FD001/rf_baseline/predictions.csv" \
  --system-benchmark "$RUN_ROOT/system_benchmark.json" \
  --shap-contributions "$RUN_ROOT/adaptive_experiment/shap_alert_explanations.csv" \
  --out-dir "$FIGURES"

echo "Complete. Results: $RUN_ROOT"
echo "Figures: $FIGURES"
echo "Dashboard: PDM_RUNTIME=$RUN_ROOT flask --app apps.dashboard run --host 0.0.0.0 --port 8090"

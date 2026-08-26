#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CMAPSS_DIR="${1:-data/CMAPSSData}"
RUN_ROOT="${2:-data/autodl_runtime/revised_paper_run}"
FIGURES="${3:-reports/revised_paper_run}"
CONFIG="${4:-configs/scenarios/concept_drift_30d_v3.json}"
TELEMETRY="$RUN_ROOT/telemetry"

if [[ -e "$RUN_ROOT" || -e "$FIGURES" ]]; then
  echo "Refusing to overwrite revised paper artifacts." >&2
  exit 3
fi

mkdir -p "$RUN_ROOT" logs

bash autodl/run_cmapss_all_subsets.sh "$CMAPSS_DIR" "$RUN_ROOT/cmapss"
bash autodl/run_30d_paper_experiment.sh \
  "$CONFIG" "$TELEMETRY" "$RUN_ROOT/social" "$FIGURES"

# Regenerate the unified set with both the four-subset NASA summary and the
# thirty-day social-backend evidence. All inputs are persisted artifacts.
python scripts/generate_complete_paper_figures.py \
  --result "$RUN_ROOT/social/adaptive_experiment/adaptive_comparison.json" \
  --drift "$RUN_ROOT/social/drift_report.json" \
  --cmapss-comparison "$RUN_ROOT/cmapss/FD001/classification_comparison/comparison.json" \
  --cmapss-summary "$RUN_ROOT/cmapss/summary/cmapss_all_subsets.json" \
  --system-benchmark "$RUN_ROOT/social/system_benchmark.json" \
  --shap-contributions "$RUN_ROOT/social/adaptive_experiment/shap_alert_explanations.csv" \
  --out-dir "$FIGURES"

echo "Revised paper experiment complete: $RUN_ROOT"

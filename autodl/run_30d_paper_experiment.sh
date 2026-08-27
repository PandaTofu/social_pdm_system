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

if [[ -x .venv/bin/spark-submit ]]; then
  SPARK_SUBMIT_BIN="${SPARK_SUBMIT_BIN:-.venv/bin/spark-submit}"
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  SPARK_SUBMIT_BIN="${SPARK_SUBMIT_BIN:-spark-submit}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
export PATH="$(dirname "$SPARK_SUBMIT_BIN"):$JAVA_HOME/bin:$PATH"
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
export SPARK_LOG_LEVEL="${SPARK_LOG_LEVEL:-WARN}"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-$PYTHON_BIN}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-$PYTHON_BIN}"

MASTER="${PDM_MASTER:-local[1]}"
DRIVER_MEMORY="${PDM_DRIVER_MEMORY:-512m}"
SHUFFLE_PARTITIONS="${PDM_SHUFFLE_PARTITIONS:-4}"
H2O_MEMORY="${PDM_H2O_MEMORY:-1G}"
H2O_THREADS="${PDM_H2O_THREADS:-1}"
KS_THRESHOLD="${PDM_KS_THRESHOLD:-0.10}"
KS_ALPHA="${PDM_KS_ALPHA:-0.05}"

"$PYTHON_BIN" apps/generate_telemetry.py --config "$CONFIG" --output-dir "$SOURCE"
"$PYTHON_BIN" tests/validate_contract.py "$SOURCE"

"$PYTHON_BIN" autodl/make_drift_windows.py \
  --source "$SOURCE" --config "$CONFIG" --out-dir "$RUN_ROOT/drift_windows"
"$PYTHON_BIN" ml/drift_monitor.py \
  --reference "$RUN_ROOT/drift_windows/reference.npz" \
  --current "$RUN_ROOT/drift_windows/current.npz" \
  --threshold "$KS_THRESHOLD" --alpha "$KS_ALPHA" --out "$RUN_ROOT/drift_report.json"

"$SPARK_SUBMIT_BIN" --master "$MASTER" --driver-memory "$DRIVER_MEMORY" \
  --conf "spark.sql.shuffle.partitions=$SHUFFLE_PARTITIONS" \
  ml/prepare_adaptive_telemetry.py \
  --source "$SOURCE" --output-dir "$RUN_ROOT/adaptive_experiment" \
  --drift-report "$RUN_ROOT/drift_report.json" \
  --scenario-name concept_drift_30d_v3 --generator-config "$CONFIG" \
  2>&1 | tee logs/concept-drift-30d-spark-prepare.log

# Run H2O as a new ordinary Python process.  The spark-submit JVM above has
# exited before H2O starts, which materially lowers peak RAM on small hosts.
"$PYTHON_BIN" ml/train_adaptive_h2o.py \
  --prepared-manifest "$RUN_ROOT/adaptive_experiment/prepared_manifest.json" \
  --trees 80 --explain-rows 500 \
  --h2o-memory "$H2O_MEMORY" --h2o-threads "$H2O_THREADS" \
  2>&1 | tee logs/concept-drift-30d-h2o-training.log

"$SPARK_SUBMIT_BIN" --master "$MASTER" --driver-memory "$DRIVER_MEMORY" \
  --conf "spark.sql.shuffle.partitions=$SHUFFLE_PARTITIONS" \
  spark/system_benchmark.py --source "$SOURCE" --out "$RUN_ROOT/system_benchmark.json" \
  --sizes 50000 100000 250000 500000 1000000 2000000 4000000 --compare-schema-read \
  2>&1 | tee logs/concept-drift-30d-system-benchmark.log

"$PYTHON_BIN" scripts/generate_complete_paper_figures.py \
  --result "$RUN_ROOT/adaptive_experiment/adaptive_comparison.json" \
  --drift "$RUN_ROOT/drift_report.json" \
  --system-benchmark "$RUN_ROOT/system_benchmark.json" \
  --shap-contributions "$RUN_ROOT/adaptive_experiment/shap_alert_explanations.csv" \
  --out-dir "$FIGURES"

echo "30-day experiment complete: $RUN_ROOT"

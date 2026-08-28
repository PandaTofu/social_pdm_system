#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT="${1:-data/autodl_runtime/e1_e2_smoke}"
SOURCE="$RUN_ROOT/telemetry.ndjson"
RESULTS="$RUN_ROOT/results"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing smoke output: $RUN_ROOT" >&2
  exit 3
fi
mkdir -p "$RUN_ROOT" logs

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

"$PYTHON_BIN" apps/generate_telemetry.py \
  --config configs/scenarios/smoke_e1_e2.json --output "$SOURCE"
"$PYTHON_BIN" tests/validate_contract.py "$SOURCE"
"$SPARK_SUBMIT_BIN" --master "${PDM_MASTER:-local[2]}" \
  --driver-memory "${PDM_DRIVER_MEMORY:-2g}" \
  --conf "spark.sql.shuffle.partitions=${PDM_SHUFFLE_PARTITIONS:-4}" \
  spark/e1_e2_experiment.py --source "$SOURCE" --output-dir "$RESULTS" \
  2>&1 | tee logs/e1-e2-smoke.log

echo "E1/E2 smoke experiment complete: $RESULTS"

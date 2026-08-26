#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SUBSET="${1:?usage: bash autodl/run_cmapss_subset.sh FD001 [data-dir] [results-root]}"
CMAPSS_DIR="${2:-data/CMAPSSData}"
RESULTS_ROOT="${3:-data/autodl_runtime/paper_run/cmapss}"

case "$SUBSET" in
  FD001|FD002|FD003|FD004) ;;
  *) echo "Invalid subset: $SUBSET" >&2; exit 2 ;;
esac

for prefix in train test RUL; do
  file="$CMAPSS_DIR/${prefix}_${SUBSET}.txt"
  [[ -f "$file" ]] || { echo "Missing $file" >&2; exit 2; }
done

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

TREES="${CMAPSS_TREES:-100}"
FOLDS="${CMAPSS_FOLDS:-10}"
MASTER="${CMAPSS_MASTER:-local[1]}"
DRIVER_MEMORY="${CMAPSS_DRIVER_MEMORY:-512m}"
SHUFFLE_PARTITIONS="${CMAPSS_SHUFFLE_PARTITIONS:-2}"
H2O_MEMORY="${CMAPSS_H2O_MEMORY:-768m}"
H2O_THREADS="${CMAPSS_H2O_THREADS:-1}"

mkdir -p "$RESULTS_ROOT" logs
echo "Running $SUBSET: master=$MASTER driver_memory=$DRIVER_MEMORY h2o_memory=$H2O_MEMORY trees=$TREES folds=$FOLDS"

"$SPARK_SUBMIT_BIN" --master "$MASTER" --driver-memory "$DRIVER_MEMORY" \
  --conf "spark.sql.shuffle.partitions=$SHUFFLE_PARTITIONS" \
  spark/cmapss_classification_comparison.py \
  --data-dir "$CMAPSS_DIR" --subset "$SUBSET" --output-dir "$RESULTS_ROOT" \
  --horizon 30 --trees "$TREES" --folds "$FOLDS" \
  --h2o-memory "$H2O_MEMORY" --h2o-threads "$H2O_THREADS" \
  2>&1 | tee "logs/cmapss-${SUBSET}-comparison.log"

"$PYTHON_BIN" -c "import json,pathlib; p=pathlib.Path('$RESULTS_ROOT/$SUBSET/classification_comparison/comparison.json'); r=json.loads(p.read_text()); print('$SUBSET complete:', {k: round(v['operating_metrics']['f1'], 4) for k, v in r['models'].items()})"

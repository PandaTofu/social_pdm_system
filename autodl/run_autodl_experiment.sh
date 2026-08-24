#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUNTIME=data/autodl_runtime
mkdir -p "$RUNTIME/inbox" "$RUNTIME/output" "$RUNTIME/checkpoints" logs
rm -f "$RUNTIME/inbox"/*.json

python apps/generate_telemetry.py --config configs/experiment.yaml --output "$RUNTIME/development.ndjson"

SPARK_LOCAL_IP=127.0.0.1 SPARK_UI_PORT=6006 \
  spark-submit --master 'local[*]' --conf spark.sql.shuffle.partitions=16 \
  autodl/file_streaming_job.py > logs/autodl-spark.log 2>&1 &
SPARK_PID=$!
echo "Spark started (pid=$SPARK_PID); UI: http://127.0.0.1:6006"
sleep 15

python autodl/replay_to_files.py --source "$RUNTIME/development.ndjson" \
  --inbox "$RUNTIME/inbox" --records-per-file 10000 --delay-seconds 1
echo "Replay complete. Wait for Spark processing, then stop it: kill $SPARK_PID"

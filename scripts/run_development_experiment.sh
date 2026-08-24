#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
bash ./scripts/create_topics.sh
bash ./scripts/submit_spark.sh > logs/spark-streaming.log 2>&1 &
echo "Spark streaming submitted; logs: logs/spark-streaming.log"
sleep 15
docker compose exec -T generator python /app/apps/generate_telemetry.py \
  --config /app/configs/experiment.yaml --kafka-bootstrap kafka:29092

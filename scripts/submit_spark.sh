#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Packages are downloaded only on the first run. Pin versions in this script so
# that the connector binary is compatible with the Spark 3.5 / Scala 2.12 image.
docker compose exec -T \
  -e KAFKA_BOOTSTRAP=kafka:29092 \
  -e DATA_PATH=/data/telemetry \
  -e CHECKPOINT=/data/checkpoints/telemetry \
  -e METRICS_PATH=/data/metrics \
  -e MONGO_URI="mongodb://${MONGO_ROOT_USERNAME}:${MONGO_ROOT_PASSWORD}@mongo:27017/?authSource=admin" \
  spark-master spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.executor.instances=2 \
  --conf spark.executor.cores=2 \
  --conf spark.executor.memory=6g \
  --conf spark.executor.memoryOverhead=1g \
  --conf spark.sql.shuffle.partitions=48 \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir=file:///data/spark-events \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 \
  /opt/pdm/spark/streaming_job.py

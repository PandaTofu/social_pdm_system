#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for topic in telemetry-raw telemetry-quarantine predictions; do
  docker compose exec -T kafka kafka-topics --bootstrap-server kafka:29092 \
    --create --if-not-exists --topic "$topic" --partitions 6 --replication-factor 1
done
docker compose exec -T kafka kafka-topics --bootstrap-server kafka:29092 --list

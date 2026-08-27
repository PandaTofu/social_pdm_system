#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Profile for the AutoDL CPU instance used by the paper experiment:
# 16 vCPU, 60 GiB cgroup memory and a fast data disk under /root/autodl-tmp.
mkdir -p /root/autodl-tmp/spark-local
export SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-/root/autodl-tmp/spark-local}"
export PDM_MASTER="${PDM_MASTER:-local[12]}"
export PDM_DRIVER_MEMORY="${PDM_DRIVER_MEMORY:-10g}"
export PDM_SHUFFLE_PARTITIONS="${PDM_SHUFFLE_PARTITIONS:-48}"
export PDM_H2O_MEMORY="${PDM_H2O_MEMORY:-20G}"
export PDM_H2O_THREADS="${PDM_H2O_THREADS:-12}"

exec bash autodl/run_30d_paper_experiment.sh "$@"

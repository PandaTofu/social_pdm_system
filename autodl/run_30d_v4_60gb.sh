#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PDM_SCENARIO_NAME="concept_drift_30d_v4"
export PDM_TREES="${PDM_TREES:-150}"

exec bash autodl/run_30d_60gb.sh \
  configs/scenarios/concept_drift_30d_v4.json \
  data/autodl_runtime/scenarios/concept_drift_30d_v4/telemetry \
  data/autodl_runtime/scenarios/concept_drift_30d_v4/results \
  reports/scenarios/concept_drift_30d_v4

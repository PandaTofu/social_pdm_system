"""Fast contract check for generated NDJSON before a Spark submission."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED = {
    "event_id", "event_time", "ingest_time", "schema_version", "server_id",
    "cpu_util_pct", "memory_util_pct", "response_p95_ms", "error_rate",
    "queue_depth", "failure_within_30min", "is_injected_defect",
    "expected_quality_action", "duplicate_ordinal", "task_type", "criticality",
    "latency_sla_ms", "expected_route", "workload_units", "source_sequence",
}
QUALITY_CATEGORIES = {"schema", "range", "temporal", "completeness", "cross_field"}


def files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    result = sorted(path.rglob("*.ndjson"))
    if not result:
        raise FileNotFoundError(f"No NDJSON files found under {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("path", type=Path); args = parser.parse_args()
    counts: Counter[str] = Counter()
    for file in files(args.path):
        with file.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line); counts["rows"] += 1
                counts[f"schema_v{row.get('schema_version')}"] += 1
                counts["missing_required"] += bool(REQUIRED - row.keys())
                counts["v2_missing_extension"] += row.get("schema_version") == 2 and not {"cache_hit_ratio", "gc_pause_ms"} <= row.keys()
                counts["invalid_cpu"] += not 0 <= row.get("cpu_util_pct", -1) <= 100
                counts["positive_labels"] += row.get("failure_within_30min", 0)
                counts[f"quality_truth_{row.get('injected_defect_category')}"] += bool(row.get("is_injected_defect"))
                counts[f"expected_route_{row.get('expected_route')}"] += 1
                counts["invalid_quality_action"] += row.get("expected_quality_action") not in {"accept", "quarantine"}
                counts["invalid_route"] += row.get("expected_route") not in {"stream", "batch"}
    print(json.dumps(counts, indent=2))
    if counts["rows"] == 0 or counts["schema_v1"] == 0 or counts["schema_v2"] == 0:
        raise SystemExit("Dataset must contain non-empty v1 and v2 records")
    missing_categories = [name for name in QUALITY_CATEGORIES if counts[f"quality_truth_{name}"] == 0]
    if missing_categories:
        raise SystemExit(f"Dataset is missing labelled quality categories: {missing_categories}")
    if counts["expected_route_stream"] == 0 or counts["expected_route_batch"] == 0:
        raise SystemExit("Dataset must contain both Stream and Batch routing truth")
    if counts["invalid_quality_action"] or counts["invalid_route"]:
        raise SystemExit("Dataset contains invalid quality-action or route labels")


if __name__ == "__main__": main()

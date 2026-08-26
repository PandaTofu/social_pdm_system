"""Fast contract check for generated NDJSON before a Spark submission."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED = {"event_id", "event_time", "ingest_time", "schema_version", "server_id", "cpu_util_pct", "memory_util_pct", "response_p95_ms", "error_rate", "queue_depth", "failure_within_30min"}


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
    print(json.dumps(counts, indent=2))
    if counts["rows"] == 0 or counts["schema_v1"] == 0 or counts["schema_v2"] == 0:
        raise SystemExit("Dataset must contain non-empty v1 and v2 records")


if __name__ == "__main__": main()

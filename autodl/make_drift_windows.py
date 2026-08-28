"""Create deterministic pre-/post-deployment NPZ windows from generated telemetry."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

import numpy as np

FEATURES = ("cpu_util_pct", "memory_util_pct", "response_p95_ms", "error_rate", "queue_depth")


def source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted({*path.rglob("*.ndjson"), *path.rglob("*.json")})
    if not files:
        raise FileNotFoundError(f"No NDJSON/JSON files found under {path}")
    return files


def rows(path: Path) -> Iterator[dict[str, object]]:
    for file in source_files(path):
        with file.open(encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def collect(path: Path, start_day: int, end_day: int,
            experiment_start: date | None = None) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = {feature: [] for feature in FEATURES}
    origin = experiment_start
    day_cache: dict[str, int] = {}
    for row in rows(path):
        event_date = str(row["event_time"])[:10]
        if origin is None:
            origin = date.fromisoformat(event_date)
        day = day_cache.setdefault(event_date, (date.fromisoformat(event_date) - origin).days + 1)
        if start_day <= day <= end_day and all(feature in row and row[feature] is not None for feature in FEATURES):
            if 0 <= row["cpu_util_pct"] <= 100 and 0 <= row["memory_util_pct"] <= 100:
                for feature in FEATURES:
                    result[feature].append(float(row[feature]))
    return {feature: np.asarray(values, dtype=np.float64) for feature, values in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-days", nargs=2, type=int)
    parser.add_argument("--current-days", nargs=2, type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    windows = config.get("experiment_windows", {})
    reference_days = args.reference_days or windows.get("drift_reference", [1, 3])
    current_days = args.current_days or windows.get("drift_current", [5, 7])
    start = (datetime.fromisoformat(config["start_time"].replace("Z", "+00:00")).date()
             if config.get("start_time") else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = collect(args.source, *reference_days, experiment_start=start)
    current = collect(args.source, *current_days, experiment_start=start)
    np.savez(args.out_dir / "reference.npz", **reference)
    np.savez(args.out_dir / "current.npz", **current)
    print({"reference_days": reference_days, "current_days": current_days,
           "reference_rows": len(reference[FEATURES[0]]), "current_rows": len(current[FEATURES[0]])})


if __name__ == "__main__": main()

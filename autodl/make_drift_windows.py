"""Create deterministic pre-/post-deployment NPZ windows from generated telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES = ("cpu_util_pct", "memory_util_pct", "response_p95_ms", "error_rate", "queue_depth")


def collect(path: Path, start_day: int, end_day: int) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = {feature: [] for feature in FEATURES}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            day = int(row["event_time"][8:10])
            if start_day <= day <= end_day and all(feature in row and row[feature] is not None for feature in FEATURES):
                if 0 <= row["cpu_util_pct"] <= 100 and 0 <= row["memory_util_pct"] <= 100:
                    for feature in FEATURES:
                        result[feature].append(float(row[feature]))
    return {feature: np.asarray(values, dtype=np.float64) for feature, values in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-days", nargs=2, type=int, default=[1, 3])
    parser.add_argument("--current-days", nargs=2, type=int, default=[5, 7])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = collect(args.source, *args.reference_days)
    current = collect(args.source, *args.current_days)
    np.savez(args.out_dir / "reference.npz", **reference)
    np.savez(args.out_dir / "current.npz", **current)
    print({"reference_rows": len(reference[FEATURES[0]]), "current_rows": len(current[FEATURES[0]])})


if __name__ == "__main__": main()

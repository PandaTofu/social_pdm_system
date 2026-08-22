"""Dependency-light two-sample KS monitoring for persisted telemetry windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


FEATURES = ("cpu_util_pct", "memory_util_pct", "response_p95_ms", "error_rate", "queue_depth")


def ks_d(reference: np.ndarray, current: np.ndarray) -> float:
    values = np.sort(np.concatenate((reference, current)))
    return float(np.max(np.abs(np.searchsorted(np.sort(reference), values, side="right") / len(reference)
                               - np.searchsorted(np.sort(current), values, side="right") / len(current))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two NPZ feature windows using KS D.")
    parser.add_argument("--reference", type=Path, required=True); parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.20); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference, current = np.load(args.reference), np.load(args.current)
    stats = {f: ks_d(reference[f], current[f]) for f in FEATURES}
    report = {"threshold": args.threshold, "feature_ks_d": stats, "max_ks_d": max(stats.values()), "drift_detected": max(stats.values()) > args.threshold}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__": main()

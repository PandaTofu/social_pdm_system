"""Dependency-light two-sample KS monitoring for persisted telemetry windows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np


FEATURES = ("cpu_util_pct", "memory_util_pct", "response_p95_ms", "error_rate", "queue_depth")


def ks_d(reference: np.ndarray, current: np.ndarray) -> float:
    values = np.sort(np.concatenate((reference, current)))
    return float(np.max(np.abs(np.searchsorted(np.sort(reference), values, side="right") / len(reference)
                               - np.searchsorted(np.sort(current), values, side="right") / len(current))))


def asymptotic_p_value(d_statistic: float, reference_size: int, current_size: int) -> float:
    """Two-sided, asymptotic two-sample KS p-value.

    The D statistic remains the practical effect-size gate.  The p-value is
    reported as complementary statistical evidence and is not used alone on
    very large telemetry windows, where negligible shifts can be significant.
    """
    effective_n = reference_size * current_size / (reference_size + current_size)
    root_n = math.sqrt(effective_n)
    scaled = (root_n + 0.12 + 0.11 / root_n) * d_statistic
    probability = 0.0
    for index in range(1, 101):
        term = 2.0 * ((-1) ** (index - 1)) * math.exp(-2.0 * index * index * scaled * scaled)
        probability += term
        if abs(term) < 1e-12:
            break
    return float(min(1.0, max(0.0, probability)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two NPZ feature windows using KS D.")
    parser.add_argument("--reference", type=Path, required=True); parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.20)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference, current = np.load(args.reference), np.load(args.current)
    details = {}
    for feature in FEATURES:
        statistic = ks_d(reference[feature], current[feature])
        p_value = asymptotic_p_value(statistic, len(reference[feature]), len(current[feature]))
        details[feature] = {
            "ks_d": statistic,
            "p_value": p_value,
            "reference_rows": int(len(reference[feature])),
            "current_rows": int(len(current[feature])),
            "effect_threshold_exceeded": statistic > args.threshold,
            "statistically_significant": p_value < args.alpha,
        }
    stats = {feature: values["ks_d"] for feature, values in details.items()}
    detected = any(values["effect_threshold_exceeded"] and values["statistically_significant"]
                   for values in details.values())
    report = {
        "method": "two-sample KS with an effect-size and significance gate",
        "threshold": args.threshold,
        "alpha": args.alpha,
        "feature_ks_d": stats,
        "feature_details": details,
        "max_ks_d": max(stats.values()),
        "drift_detected": detected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__": main()

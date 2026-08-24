"""Stable JSON extraction helpers for H2O binomial metrics.

H2O exposes threshold metrics as two-dimensional tables.  Keeping the table
parsing here prevents experiment scripts from depending on pandas and ensures
that every reported F1/precision/recall value uses an explicit threshold.
"""
from __future__ import annotations

import math
from typing import Any


def _finite(value: Any) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def metric_at(performance: Any, method: str, threshold: float) -> float:
    """Return one H2O threshold metric at the requested operating point."""
    values = getattr(performance, method)(thresholds=[threshold])
    if not values:
        raise ValueError(f"H2O returned no {method} value at threshold {threshold}")
    return float(values[0][1])


def confusion_at(performance: Any, threshold: float) -> dict[str, int]:
    """Extract the four confusion counts without converting through pandas."""
    matrix = performance.confusion_matrix(thresholds=[threshold])
    if matrix is None:
        raise ValueError(f"H2O returned no confusion matrix at threshold {threshold}")
    values = matrix.to_list()
    return {"tn": int(values[0][0]), "fp": int(values[0][1]),
            "fn": int(values[1][0]), "tp": int(values[1][1])}


def threshold_curve(performance: Any, max_points: int = 250) -> list[dict[str, float | int | None]]:
    """Serialize H2O's common threshold table, downsampling only for file size."""
    table = performance._metric_json["thresholds_and_metric_scores"]
    headers = list(table.col_header)
    rows = list(table.cell_values)
    if len(rows) > max_points:
        step = max(1, math.ceil(len(rows) / max_points))
        selected = rows[::step]
        if selected[-1] != rows[-1]:
            selected.append(rows[-1])
        rows = selected
    wanted = ("threshold", "f1", "precision", "recall", "tpr", "fpr", "accuracy")
    indices = {name: headers.index(name) for name in wanted if name in headers}
    serialized = [{name: _finite(row[index]) for name, index in indices.items()} for row in rows]
    for record in serialized:
        if "recall" not in record and "tpr" in record:
            record["recall"] = record["tpr"]
    return serialized


def binary_summary(performance: Any, threshold: float) -> dict[str, Any]:
    """Return the metrics used in all thesis tables at one operating point."""
    return {
        "threshold": float(threshold),
        "precision": metric_at(performance, "precision", threshold),
        "recall": metric_at(performance, "recall", threshold),
        "f1": metric_at(performance, "F1", threshold),
        "accuracy": metric_at(performance, "accuracy", threshold),
        "confusion_matrix": confusion_at(performance, threshold),
        "roc_auc": float(performance.auc()),
        "pr_auc": float(performance.aucpr()),
    }

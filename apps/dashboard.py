"""Minimal operations dashboard for the thesis prototype.

The dashboard is deliberately read-only: it renders persisted Spark, drift,
model and TreeSHAP artefacts and never invents live metrics when files are
missing.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.getenv("PDM_RUNTIME", ROOT / "data" / "autodl_runtime" / "output"))
ADAPTIVE_REPORT = Path(os.getenv("PDM_ADAPTIVE_REPORT", RUNTIME / "adaptive_experiment" / "adaptive_comparison.json"))
DRIFT_REPORT = Path(os.getenv("PDM_DRIFT_REPORT", RUNTIME / "drift_report.json"))
ALERTS = Path(os.getenv("PDM_ALERTS", RUNTIME / "adaptive_experiment" / "shap_alert_explanations.csv"))
METRICS_DIR = Path(os.getenv("PDM_STREAM_METRICS", RUNTIME / "metrics"))

app = Flask(__name__)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def latest_stream_metric() -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    if not METRICS_DIR.exists():
        return None
    for path in METRICS_DIR.rglob("part-*"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
    return max(rows, key=lambda row: int(row.get("batch_id", -1)), default=None)


def alert_rows(limit: int = 20) -> list[dict[str, Any]]:
    if not ALERTS.exists():
        return []
    excluded = {"event_id", "event_time", "server_id", "failure_within_30min", "predict", "p0", "p1", "BiasTerm"}
    alerts: list[dict[str, Any]] = []
    with ALERTS.open(encoding="utf-8-sig", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            contributions = []
            for name, value in row.items():
                if name in excluded or value in (None, ""):
                    continue
                try:
                    contributions.append((name, float(value)))
                except ValueError:
                    continue
            top = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:3]
            alerts.append({
                "event_id": row.get("event_id") or f"evaluation-row-{index + 1}",
                "event_time": row.get("event_time") or "n/a",
                "server_id": row.get("server_id") or "n/a",
                "probability": float(row.get("p1", 0.0)),
                "top_contributors": [{"feature": name, "value": value} for name, value in top],
            })
            if len(alerts) >= limit:
                break
    return alerts


@app.get("/")
def index():
    return render_template("dashboard.html")


@app.get("/health")
def health():
    return {"status": "ok", "runtime": str(RUNTIME)}


@app.get("/api/status")
def status():
    adaptive = read_json(ADAPTIVE_REPORT)
    drift = read_json(DRIFT_REPORT)
    stream = latest_stream_metric()
    return jsonify({
        "availability": {
            "adaptive_report": adaptive is not None,
            "drift_report": drift is not None,
            "stream_metrics": stream is not None,
            "shap_alerts": ALERTS.exists(),
        },
        "stream": stream,
        "drift": drift,
        "model": {
            "protocol": adaptive.get("protocol") if adaptive else None,
            "variants": adaptive.get("variants") if adaptive else None,
            "threshold": adaptive.get("adaptive_operating_threshold") if adaptive else None,
            "retrain_triggered": adaptive.get("retrain_triggered") if adaptive else None,
            "shap": adaptive.get("shap_explanations") if adaptive else None,
        },
        "alerts": alert_rows(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PDM_DASHBOARD_PORT", "8090")))

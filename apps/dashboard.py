"""Read-only operations dashboard for persisted Social PDM evidence.

The UI separates model-derived server health, threshold-based warnings and
TreeSHAP explanations. It never treats SHAP contribution columns as raw
telemetry or claims a live production control loop.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.getenv("PDM_RUNTIME", ROOT / "data" / "autodl_runtime" / "output"))
ADAPTIVE_DIR = RUNTIME / "adaptive_experiment"
ADAPTIVE_REPORT = Path(os.getenv("PDM_ADAPTIVE_REPORT", ADAPTIVE_DIR / "adaptive_comparison.json"))
DRIFT_REPORT = Path(os.getenv("PDM_DRIFT_REPORT", RUNTIME / "drift_report.json"))
SHAP_ALERTS = Path(os.getenv("PDM_ALERTS", ADAPTIVE_DIR / "shap_alert_explanations.csv"))
HEALTH_SNAPSHOT = Path(os.getenv("PDM_HEALTH_SNAPSHOT", ADAPTIVE_DIR / "server_health_snapshot.csv"))
THRESHOLD_ALERTS = Path(os.getenv("PDM_THRESHOLD_ALERTS", ADAPTIVE_DIR / "threshold_alerts.csv"))
METRICS_DIR = Path(os.getenv("PDM_STREAM_METRICS", RUNTIME / "metrics"))

RAW_METRICS = (
    "cpu_util_pct", "memory_util_pct", "disk_util_pct", "response_p95_ms",
    "error_rate", "timeout_rate", "queue_depth", "request_rate_rps",
)
SHAP_EXCLUDED = {
    "event_id", "event_time", "server_id", "failure_within_30min",
    "predict", "p0", "p1", "BiasTerm", "day", "sample_weight",
}

app = Flask(__name__)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))
    except OSError:
        return []


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calibrated_threshold() -> float:
    report = read_json(ADAPTIVE_REPORT) or {}
    return as_float(report.get("adaptive_operating_threshold"), 0.5)


def requested_threshold() -> float:
    value = as_float(request.args.get("threshold"), calibrated_threshold())
    return min(1.0, max(0.0, value))


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


def health_status(probability: float, threshold: float) -> str:
    return "failure_warning" if probability >= threshold else "normal"


def _telemetry(row: dict[str, str]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name in RAW_METRICS:
        raw = row.get(name)
        values[name] = as_float(raw) if raw not in (None, "") else None
    return values


def health_rows(threshold: float) -> tuple[list[dict[str, Any]], str]:
    """Return the latest model score for each logical server."""
    rows = read_csv_rows(HEALTH_SNAPSHOT)
    source = "server_health_snapshot"
    has_raw_telemetry = True
    if not rows:
        rows = read_csv_rows(SHAP_ALERTS)
        source = "explained_alerts_fallback"
        has_raw_telemetry = False

    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        server = row.get("server_id") or "unknown"
        if server not in latest or (row.get("event_time") or "") > (latest[server].get("event_time") or ""):
            latest[server] = row

    result = []
    for server, row in latest.items():
        probability = as_float(row.get("p1"))
        result.append({
            "server_id": server,
            "event_id": row.get("event_id") or "n/a",
            "event_time": row.get("event_time") or "n/a",
            "probability": probability,
            "status": health_status(probability, threshold),
            "telemetry": _telemetry(row) if has_raw_telemetry else {},
            "raw_telemetry_available": has_raw_telemetry,
        })
    result.sort(key=lambda row: (row["status"] != "failure_warning", -row["probability"], row["server_id"]))
    return result, source


def threshold_alert_rows(threshold: float, limit: int = 200) -> tuple[list[dict[str, Any]], str]:
    rows = read_csv_rows(THRESHOLD_ALERTS)
    source = "threshold_alerts"
    if not rows:
        rows = read_csv_rows(HEALTH_SNAPSHOT)
        source = "server_health_snapshot"
    if not rows:
        rows = read_csv_rows(SHAP_ALERTS)
        source = "explained_alerts_fallback"

    alerts = []
    for index, row in enumerate(rows):
        probability = as_float(row.get("p1"))
        if probability < threshold:
            continue
        alerts.append({
            "event_id": row.get("event_id") or f"evaluation-row-{index + 1}",
            "event_time": row.get("event_time") or "n/a",
            "server_id": row.get("server_id") or "n/a",
            "probability": probability,
            "status": "failure_warning",
            "observed_label": int(as_float(row.get("failure_within_30min"))),
        })
    alerts.sort(key=lambda row: (row["event_time"], row["probability"]), reverse=True)
    return alerts[:limit], source


def _contributions(row: dict[str, str]) -> list[dict[str, float | str]]:
    values = []
    for name, value in row.items():
        if name in SHAP_EXCLUDED or value in (None, ""):
            continue
        try:
            values.append({"feature": name, "value": float(value)})
        except ValueError:
            continue
    return sorted(values, key=lambda item: abs(float(item["value"])), reverse=True)


def explanation_rows(limit: int = 100) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(read_csv_rows(SHAP_ALERTS)[:limit]):
        result.append({
            "event_id": row.get("event_id") or f"evaluation-row-{index + 1}",
            "event_time": row.get("event_time") or "n/a",
            "server_id": row.get("server_id") or "n/a",
            "probability": as_float(row.get("p1")),
            "top_contributors": _contributions(row)[:5],
        })
    return result


def global_shap(rows: Iterable[dict[str, str]] | None = None, limit: int = 10) -> list[dict[str, float | str]]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows if rows is not None else read_csv_rows(SHAP_ALERTS):
        for item in _contributions(row):
            name, value = str(item["feature"]), abs(float(item["value"]))
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    ranked = sorted(
        ((name, totals[name] / counts[name]) for name in totals if counts[name]),
        key=lambda item: item[1], reverse=True,
    )[:limit]
    return [{"feature": name, "mean_abs_shap": value} for name, value in ranked]


@app.get("/")
def index():
    return render_template("dashboard.html", page="overview")


@app.get("/server-health")
def server_health_page():
    return render_template("server_health.html", page="health")


@app.get("/failure-alerts")
def failure_alerts_page():
    return render_template("failure_alerts.html", page="alerts")


@app.get("/explanations")
def explanations_page():
    return render_template("explanations.html", page="explanations")


@app.get("/health")
def healthcheck():
    return {"status": "ok", "runtime": str(RUNTIME)}


@app.get("/api/overview")
def overview_api():
    adaptive = read_json(ADAPTIVE_REPORT) or {}
    drift = read_json(DRIFT_REPORT) or {}
    threshold = calibrated_threshold()
    health, health_source = health_rows(threshold)
    alerts, alert_source = threshold_alert_rows(threshold)
    return jsonify({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "health_summary": {
            "servers": len(health),
            "normal": sum(row["status"] == "normal" for row in health),
            "failure_warning": sum(row["status"] == "failure_warning" for row in health),
            "source": health_source,
        },
        "alert_count": len(alerts),
        "alert_source": alert_source,
        "stream": latest_stream_metric(),
        "drift": drift,
        "variants": adaptive.get("variants") or {},
        "availability": {
            "adaptive_report": bool(adaptive),
            "drift_report": bool(drift),
            "health_snapshot": HEALTH_SNAPSHOT.exists(),
            "threshold_alerts": THRESHOLD_ALERTS.exists(),
            "shap_explanations": SHAP_ALERTS.exists(),
        },
    })


@app.get("/api/health-status")
def health_status_api():
    threshold = requested_threshold()
    rows, source = health_rows(threshold)
    return jsonify({
        "generated_at": datetime.now(timezone.utc).isoformat(), "threshold": threshold,
        "source": source, "rows": rows,
        "summary": {
            "servers": len(rows),
            "normal": sum(row["status"] == "normal" for row in rows),
            "failure_warning": sum(row["status"] == "failure_warning" for row in rows),
        },
    })


@app.get("/api/alerts")
def alerts_api():
    requested = requested_threshold()
    # The persisted alert history contains every score at or above the
    # calibrated threshold, not all recent negative rows. Lower display
    # thresholds would therefore be incomplete and are intentionally clamped.
    threshold = max(requested, calibrated_threshold()) if THRESHOLD_ALERTS.exists() else requested
    try:
        limit = min(1000, max(1, int(request.args.get("limit", "200"))))
    except ValueError:
        limit = 200
    rows, source = threshold_alert_rows(threshold, limit)
    export_metadata = (read_json(ADAPTIVE_REPORT) or {}).get("dashboard_exports") or {}
    return jsonify({
        "generated_at": datetime.now(timezone.utc).isoformat(), "threshold": threshold,
        "source": source, "count": len(rows), "rows": rows,
        "requested_threshold": requested,
        "minimum_complete_threshold": calibrated_threshold() if THRESHOLD_ALERTS.exists() else None,
        "export_metadata": export_metadata,
    })


@app.get("/api/explanations")
def explanations_api():
    try:
        limit = min(500, max(1, int(request.args.get("limit", "100"))))
    except ValueError:
        limit = 100
    return jsonify({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": explanation_rows(limit), "global_shap": global_shap(),
        "source": "shap_alert_explanations" if SHAP_ALERTS.exists() else "unavailable",
    })


# Backward-compatible endpoint used by earlier screenshots and scripts.
@app.get("/api/status")
def status_api():
    adaptive = read_json(ADAPTIVE_REPORT) or {}
    return jsonify({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stream": latest_stream_metric(), "drift": read_json(DRIFT_REPORT),
        "model": {
            "protocol": adaptive.get("protocol"), "variants": adaptive.get("variants"),
            "threshold": adaptive.get("adaptive_operating_threshold"),
            "retrain_triggered": adaptive.get("retrain_triggered"),
            "shap": adaptive.get("shap_explanations"),
            "stability": adaptive.get("post_drift_daily_stability"),
            "global_shap": global_shap(),
        },
        "alerts": explanation_rows(20),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PDM_DASHBOARD_PORT", "6008")))

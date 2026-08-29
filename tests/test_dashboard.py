from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps import dashboard


class DashboardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.adaptive = self.root / "adaptive_comparison.json"
        self.health = self.root / "server_health_snapshot.csv"
        self.alerts = self.root / "threshold_alerts.csv"
        self.shap = self.root / "shap_alert_explanations.csv"
        self.adaptive.write_text(json.dumps({
            "adaptive_operating_threshold": 0.7,
            "variants": {"full_adaptive": {"f1": 0.8}},
        }), encoding="utf-8")
        fields = ["event_id", "event_time", "server_id", "failure_within_30min", "p1",
                  "cpu_util_pct", "memory_util_pct", "response_p95_ms", "error_rate", "queue_depth"]
        rows = [
            {"event_id": "e1", "event_time": "2025-01-30T00:00:00Z", "server_id": "srv-1",
             "failure_within_30min": 0, "p1": .2, "cpu_util_pct": 30, "memory_util_pct": 40,
             "response_p95_ms": 80, "error_rate": .01, "queue_depth": 2},
            {"event_id": "e2", "event_time": "2025-01-30T00:01:00Z", "server_id": "srv-2",
             "failure_within_30min": 1, "p1": .8, "cpu_util_pct": 90, "memory_util_pct": 85,
             "response_p95_ms": 500, "error_rate": .2, "queue_depth": 40},
        ]
        for path in (self.health, self.alerts):
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        with self.shap.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["event_id", "event_time", "server_id", "p1", "cpu_util_pct", "BiasTerm"])
            writer.writeheader()
            writer.writerow({"event_id": "e2", "event_time": "2025-01-30T00:01:00Z",
                             "server_id": "srv-2", "p1": .8, "cpu_util_pct": .35, "BiasTerm": -.1})
        self.paths = patch.multiple(
            dashboard, ADAPTIVE_REPORT=self.adaptive, HEALTH_SNAPSHOT=self.health,
            THRESHOLD_ALERTS=self.alerts, SHAP_ALERTS=self.shap,
            DRIFT_REPORT=self.root / "missing.json", METRICS_DIR=self.root / "missing-metrics",
        )
        self.paths.start()
        self.client = dashboard.app.test_client()

    def tearDown(self) -> None:
        self.paths.stop()
        self.temporary.cleanup()

    def test_pages_and_health_summary(self) -> None:
        for route in ("/", "/server-health", "/failure-alerts", "/explanations"):
            self.assertEqual(self.client.get(route).status_code, 200)
        payload = self.client.get("/api/health-status").get_json()
        self.assertEqual(payload["summary"], {"servers": 2, "normal": 1, "failure_warning": 1})
        self.assertTrue(payload["rows"][0]["raw_telemetry_available"])

    def test_display_threshold_is_non_persistent(self) -> None:
        self.assertEqual(self.client.get("/api/alerts?threshold=0.9").get_json()["count"], 0)
        self.assertEqual(self.client.get("/api/alerts?threshold=0.5").get_json()["count"], 1)
        self.assertEqual(self.client.get("/api/overview").get_json()["threshold"], .7)

    def test_shap_values_are_explanations_not_telemetry(self) -> None:
        payload = self.client.get("/api/explanations").get_json()
        self.assertEqual(payload["rows"][0]["top_contributors"][0]["feature"], "cpu_util_pct")
        self.assertAlmostEqual(payload["rows"][0]["top_contributors"][0]["value"], .35)


if __name__ == "__main__":
    unittest.main()

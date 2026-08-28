from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

from spark.quality import SCHEMA, classify_quality, quality_report
from spark.routing import route_tasks, routing_report, simulate_fixed_capacity


def valid_row(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "event_time": "2025-01-01T00:00:00Z",
        "ingest_time": "2025-01-01T00:00:03Z",
        "schema_version": 1,
        "data_center_id": "dc-01",
        "rack_id": "rack-01",
        "server_id": "dc-01-srv-001",
        "service_name": "feed",
        "cpu_util_pct": 50.0,
        "memory_util_pct": 60.0,
        "disk_util_pct": 40.0,
        "disk_iops": 500.0,
        "disk_queue_depth": 2.0,
        "network_in_mbps": 10.0,
        "network_out_mbps": 12.0,
        "request_rate_rps": 100.0,
        "response_p50_ms": 50.0,
        "response_p95_ms": 100.0,
        "error_rate": 0.01,
        "timeout_rate": 0.002,
        "queue_depth": 3.0,
        "active_connections": 200,
        "maintenance_flag": False,
        "traffic_campaign_flag": False,
        "failure_within_30min": 0,
        "is_injected_defect": False,
        "expected_quality_action": "accept",
        "duplicate_ordinal": 0,
        "task_type": "health_monitor",
        "criticality": "medium",
        "latency_sla_ms": 1000,
        "expected_route": "stream",
        "workload_units": 100.0,
        "source_sequence": 1,
    }


def mark(row: dict, category: str, defect_type: str) -> dict:
    row.update({
        "is_injected_defect": True,
        "injected_defect_category": category,
        "injected_defect_type": defect_type,
        "expected_quality_action": "quarantine",
    })
    return row


class QualityRoutingSparkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("quality-routing-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def frame(self, rows: list[dict]):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rows.ndjson"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            frame = self.spark.read.schema(SCHEMA).json(str(source)).cache()
            frame.count()
            return frame

    def test_five_categories_and_duplicate_are_scored_against_truth(self):
        rows = [valid_row("clean")]
        schema = mark(valid_row("schema"), "schema", "unsupported_schema")
        schema["schema_version"] = 99
        rows.append(schema)
        ranged = mark(valid_row("range"), "range", "cpu_out_of_range")
        ranged["cpu_util_pct"] = 125.0
        rows.append(ranged)
        temporal = mark(valid_row("temporal"), "temporal", "event_too_late")
        temporal["ingest_time"] = "2025-01-01T00:20:00Z"
        rows.append(temporal)
        completeness = mark(valid_row("missing"), "completeness", "missing_required_field")
        completeness.pop("memory_util_pct")
        rows.append(completeness)
        cross = mark(valid_row("cross"), "cross_field", "p95_below_p50")
        cross["response_p95_ms"] = 25.0
        rows.append(cross)
        original = valid_row("duplicate")
        duplicate = mark(valid_row("duplicate"), "completeness", "duplicate_event")
        duplicate["duplicate_ordinal"] = 1
        duplicate["source_sequence"] = 2
        rows.extend([original, duplicate])

        frame = self.frame(rows)
        try:
            report = quality_report(classify_quality(frame))
        finally:
            frame.unpersist()
        self.assertEqual(report["received"], 8)
        self.assertEqual(report["accepted"], 2)
        self.assertEqual(report["quarantined"], 6)
        self.assertEqual((report["tp"], report["fp"], report["fn"]), (6, 0, 0))
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual({row["category"] for row in report["per_category"]}, {
            "schema", "range", "temporal", "completeness", "cross_field"
        })

    def test_router_is_lossless_and_matches_labelled_routes(self):
        stream = valid_row("stream")
        batch = valid_row("batch")
        batch.update({
            "task_type": "capacity_analytics", "criticality": "low",
            "latency_sla_ms": 300_000, "expected_route": "batch",
        })
        frame = self.frame([stream, batch])
        try:
            report = routing_report(route_tasks(frame))
        finally:
            frame.unpersist()
        self.assertEqual(report["routed_rows"], 2)
        self.assertEqual(report["stream_rows"], 1)
        self.assertEqual(report["batch_rows"], 1)
        self.assertEqual(report["routing_accuracy"], 1.0)


class WorkloadControllerTests(unittest.TestCase):
    def test_priority_redistribution_reduces_stream_delay_under_burst(self):
        minutes = [
            {"event_minute": index, "stream_units": 50.0, "batch_units": 30.0, "is_burst": False}
            for index in range(3)
        ]
        minutes.extend([
            {"event_minute": 3, "stream_units": 100.0, "batch_units": 80.0, "is_burst": True},
            {"event_minute": 4, "stream_units": 50.0, "batch_units": 20.0, "is_burst": False},
            {"event_minute": 5, "stream_units": 30.0, "batch_units": 10.0, "is_burst": False},
        ])
        result = simulate_fixed_capacity(minutes, 100.0, 0.60, 0.85)
        self.assertGreater(result["baseline"]["max_backlog_units"], 0)
        self.assertLess(
            result["enhanced"]["max_stream_queue_delay_minutes"],
            result["baseline"]["max_queue_delay_minutes"],
        )


if __name__ == "__main__":
    unittest.main()

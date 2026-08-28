from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from apps.generate_telemetry import (
    deterministic_event_id,
    incident_schedule,
    precursor_severity,
    records,
    settings_from,
)


class GeneratorScenarioTests(unittest.TestCase):
    def test_v4_protocol_aligns_feedback_and_evaluation_incident_rates(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "concept_drift_30d_v4.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        settings = settings_from(config_path)
        self.assertEqual(settings.scenario, "concept_drift_30d_v4")
        self.assertTrue(raw["history_features"])
        self.assertEqual(raw["feedback_split_key"], "server_id")
        feedback_days = raw["experiment_windows"]["feedback"]
        evaluation_days = raw["experiment_windows"]["evaluation"]
        feedback_rate = 1 / (feedback_days[1] - feedback_days[0] + 1)
        evaluation_rate = 2 / (evaluation_days[1] - evaluation_days[0] + 1)
        self.assertLess(abs(feedback_rate - evaluation_rate) / evaluation_rate, 0.15)

    def test_concept_scenario_has_phase_stratified_incidents(self):
        config = {
            "scenario": "concept_drift_v2", "seed": 7,
            "start_time": "2025-01-01T00:00:00Z", "logical_servers": 3, "days": 7,
            "sample_interval_minutes": 1, "data_centers": 1,
            "schema_v2_start_day": 4, "drift_start_day": 5,
            "failure_horizon_minutes": 30, "precursor_window_minutes": 30,
            "concept_drift_strength": 1.0,
            "incident_windows": [{"start_day": 1, "end_day": 3},
                                 {"start_day": 5, "end_day": 5},
                                 {"start_day": 6, "end_day": 7}],
            "quality_error_rate": 0, "late_event_rate": 0, "duplicate_rate": 0,
            "burst_windows": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            settings = settings_from(path)
        incidents = incident_schedule(settings, np.random.default_rng(settings.seed), 7 * 1440)
        self.assertEqual(set(incidents), {0, 1, 2})
        for values in incidents.values():
            self.assertEqual(len(values), 3)
            days = [minute // 1440 + 1 for minute in values]
            self.assertTrue(1 <= days[0] <= 3)
            self.assertEqual(days[1], 5)
            self.assertTrue(6 <= days[2] <= 7)

    def test_precursor_is_gradual_and_horizon_bounded(self):
        self.assertEqual(precursor_severity(None, 30), 0.0)
        self.assertEqual(precursor_severity(30, 30), 0.0)
        self.assertAlmostEqual(precursor_severity(15, 30), 0.5)
        self.assertGreater(precursor_severity(1, 30), 0.9)

    def test_event_ids_are_seeded_and_stable(self):
        config = {
            "scenario": "concept_drift_v2", "seed": 7,
            "start_time": "2025-01-01T00:00:00Z", "logical_servers": 2, "days": 2,
            "sample_interval_minutes": 1, "data_centers": 1,
            "schema_v2_start_day": 2, "drift_start_day": 2,
            "failure_horizon_minutes": 30, "precursor_window_minutes": 30,
            "concept_drift_strength": 1.0,
            "incident_windows": [{"start_day": 1, "end_day": 1}],
            "quality_error_rate": 0, "late_event_rate": 0, "duplicate_rate": 0,
            "burst_windows": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            settings = settings_from(path)
        event_id = deterministic_event_id(settings, minute=10, server=1)
        self.assertEqual(event_id, deterministic_event_id(settings, minute=10, server=1))
        self.assertNotEqual(event_id, deterministic_event_id(settings, minute=11, server=1))

    def test_generator_emits_five_category_truth_and_routing_contract(self):
        config = {
            "scenario": "concept_drift_v2", "seed": 11,
            "start_time": "2025-01-01T00:00:00Z", "logical_servers": 2, "days": 2,
            "sample_interval_minutes": 5, "data_centers": 1,
            "schema_v2_start_day": 2, "drift_start_day": 2,
            "failure_horizon_minutes": 30, "precursor_window_minutes": 30,
            "concept_drift_strength": 1.0,
            "incident_windows": [{"start_day": 1, "end_day": 1}],
            "quality_error_rate": 0.25, "late_event_rate": 0,
            "quality_defect_rates": {
                "schema": 0.05, "range": 0.05, "temporal": 0.05,
                "completeness": 0.05, "cross_field": 0.05,
            },
            "duplicate_rate": 0.05, "burst_windows": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            generated = list(records(settings_from(path)))
        categories = {
            row["injected_defect_category"] for row in generated
            if row["is_injected_defect"]
        }
        self.assertTrue({"schema", "range", "temporal", "completeness", "cross_field"} <= categories)
        self.assertTrue(all(row["expected_route"] in {"stream", "batch"} for row in generated))
        self.assertTrue(all(row["workload_units"] > 0 for row in generated))
        self.assertTrue(any(row["duplicate_ordinal"] == 1 for row in generated))


if __name__ == "__main__":
    unittest.main()

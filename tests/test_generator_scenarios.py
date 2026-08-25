from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from apps.generate_telemetry import (
    incident_schedule,
    precursor_severity,
    settings_from,
)


class GeneratorScenarioTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

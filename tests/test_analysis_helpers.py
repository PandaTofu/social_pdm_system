from __future__ import annotations

import unittest

import numpy as np

from ml.drift_monitor import asymptotic_p_value, ks_d
from ml.h2o_binary_metrics import binary_summary, threshold_curve


class FakeTable:
    col_header = ["threshold", "f1", "precision", "recall", "tpr", "fpr", "accuracy"]
    cell_values = [
        [0.2, 0.6, 0.5, 0.75, 0.75, 0.20, 0.80],
        [0.5, 0.7, 0.8, 0.62, 0.62, 0.05, 0.90],
    ]


class FakePerformance:
    _metric_json = {"thresholds_and_metric_scores": FakeTable()}

    @staticmethod
    def _value(name: str):
        values = {"precision": 0.8, "recall": 0.62, "F1": 0.7, "accuracy": 0.9,
                  "tns": 90, "fps": 5, "fns": 8, "tps": 13}
        return lambda thresholds: [[thresholds[0], values[name]]]

    def __getattr__(self, name: str):
        return self._value(name)

    @staticmethod
    def confusion_matrix(thresholds):
        class Matrix:
            @staticmethod
            def to_list():
                return [[90, 5], [8, 13]]
        return Matrix()

    def auc(self):
        return 0.91

    def aucpr(self):
        return 0.72


class AnalysisHelperTests(unittest.TestCase):
    def test_ks_detects_shift(self):
        reference = np.arange(100, dtype=float)
        self.assertEqual(ks_d(reference, reference.copy()), 0.0)
        statistic = ks_d(reference, reference + 100)
        self.assertGreater(statistic, 0.9)
        self.assertLess(asymptotic_p_value(statistic, 100, 100), 0.05)

    def test_h2o_summary_contract(self):
        summary = binary_summary(FakePerformance(), 0.5)
        self.assertEqual(summary["confusion_matrix"], {"tn": 90, "fp": 5, "fn": 8, "tp": 13})
        self.assertAlmostEqual(summary["f1"], 0.7)
        self.assertAlmostEqual(summary["pr_auc"], 0.72)
        self.assertEqual(len(threshold_curve(FakePerformance())), 2)


if __name__ == "__main__":
    unittest.main()

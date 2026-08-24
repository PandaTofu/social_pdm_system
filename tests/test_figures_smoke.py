from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.generate_complete_paper_figures import (
    architecture,
    cmapss_distribution,
    cmapss_validation,
    drift_figure,
    model_comparison,
    stability_figure,
    system_figure,
    threshold_figure,
    write_summary,
)


class FigureSmokeTests(unittest.TestCase):
    def test_all_core_figure_functions(self):
        result = {
            "adaptive_operating_threshold": 0.7,
            "variants": {
                "static_rf": {"threshold": 0.5, "precision": .4, "recall": .5, "f1": .44, "pr_auc": .31},
                "unweighted_retraining": {"threshold": 0.5, "precision": .42, "recall": .52, "f1": .46, "pr_auc": .33},
                "weighted_retraining": {"threshold": 0.5, "precision": .45, "recall": .54, "f1": .49, "pr_auc": .35},
                "full_adaptive": {"threshold": 0.7, "precision": .5, "recall": .5, "f1": .5, "pr_auc": .35},
            },
            "adaptive_threshold_curve": [
                {"threshold": .2, "precision": .2, "recall": .9, "f1": .33},
                {"threshold": .7, "precision": .5, "recall": .5, "f1": .5},
            ],
            "post_drift_daily_stability": [
                {"day": 6, "static": {"f1": .44}, "full_adaptive": {"f1": .50}},
                {"day": 7, "static": {"f1": .42}, "full_adaptive": {"f1": .49}},
            ],
        }
        drift = {"feature_ks_d": {"cpu": .1, "memory": .25}, "threshold": .2,
                 "max_ks_d": .25, "drift_detected": True}
        cmapss = {
            "train_distribution": {"rows": 100, "positive_rows": 20, "rul_bands": {"0-30": 20}},
            "test_distribution": {"rows": 50, "positive_rows": 10,
                                  "rul_bands": {"0-30": 10, "31-60": 10, "61-120": 20, "121+": 10}},
            "operating_metrics": {"confusion_matrix": {"tn": 35, "fp": 5, "fn": 3, "tp": 7},
                                  "pr_auc": .72},
        }
        benchmark = {"scaling": [
            {"processed_rows": 100, "throughput_rows_per_second": 50, "latency_ms": 2000},
            {"processed_rows": 200, "throughput_rows_per_second": 80, "latency_ms": 2500},
        ]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.csv"
            with predictions.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["p1", "failure_within_horizon"])
                writer.writeheader()
                for score, actual in ((.9, 1), (.7, 1), (.6, 0), (.2, 0), (.1, 0)):
                    writer.writerow({"p1": score, "failure_within_horizon": actual})
            architecture(root / "architecture")
            cmapss_distribution(cmapss, root / "distribution")
            cmapss_validation(cmapss, predictions, root / "validation")
            drift_figure(drift, root / "drift")
            model_comparison(result, root / "models")
            threshold_figure(result, root / "threshold")
            stability_figure(result, root / "stability")
            system_figure(benchmark, root / "system")
            write_summary(result, drift, root / "summary.md")
            self.assertEqual(len(list(root.glob("*.png"))), 8)
            self.assertEqual(len(list(root.glob("*.pdf"))), 8)
            self.assertTrue((root / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()

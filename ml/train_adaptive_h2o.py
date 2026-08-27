"""Train and evaluate adaptive H2O models from Spark-prepared CSVs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h2o

from adaptive_telemetry_comparison import (
    binary_summary,
    select_f1_threshold,
    stability_by_window,
    threshold_curve,
    train_h2o,
    write_explanations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adaptive H2O RF variants without a Spark JVM.")
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--explain-rows", type=int, default=500)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--h2o-memory", default="4G")
    parser.add_argument("--h2o-threads", type=int, default=4)
    parser.add_argument("--h2o-port", type=int, default=6008)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    output_dir = args.prepared_manifest.resolve().parent
    csv_paths = {name: output_dir / value for name, value in manifest["csv_paths"].items()}
    missing = [str(path) for path in csv_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Prepared CSV files are missing: {missing}")
    windows = manifest["experiment_windows"]
    feedback_days = tuple(windows["feedback"])
    evaluation_days = tuple(windows["evaluation"])
    train_days = tuple(windows["initial_train"])
    drift_detected = bool(manifest["drift_detected"])

    h2o.init(ip="127.0.0.1", port=args.h2o_port,
             max_mem_size=args.h2o_memory, nthreads=args.h2o_threads)
    try:
        static_model = train_h2o(csv_paths["pre_drift_train"], args.trees)
        evaluation_frame = h2o.import_file(str(csv_paths["evaluation"]))
        static_performance = static_model.model_performance(evaluation_frame)
        retrained_summary = None
        if drift_detected and not args.skip_ablation:
            retrained_model = train_h2o(csv_paths["adaptive_train"], args.trees, use_recency_weight=False)
            try:
                retrained_summary = binary_summary(retrained_model.model_performance(evaluation_frame), 0.5)
            finally:
                h2o.remove(retrained_model.model_id)
        adaptive_model = train_h2o(
            csv_paths["adaptive_train"], args.trees, use_recency_weight=drift_detected
        )
        adaptive_performance = adaptive_model.model_performance(evaluation_frame)
        calibration_frame = h2o.import_file(str(csv_paths["feedback_calibration"]))
        try:
            calibration_performance = adaptive_model.model_performance(calibration_frame)
            adaptive_threshold, calibration_f1 = select_f1_threshold(calibration_performance)
        finally:
            h2o.remove(calibration_frame.frame_id)

        variants = {
            "static_rf": binary_summary(static_performance, 0.5),
            "weighted_retraining": binary_summary(adaptive_performance, 0.5),
            "full_adaptive": binary_summary(adaptive_performance, adaptive_threshold),
        }
        if retrained_summary is not None:
            variants["unweighted_retraining"] = retrained_summary
        explanations = write_explanations(
            adaptive_model, evaluation_frame, adaptive_threshold, output_dir, args.explain_rows
        )
        stability = stability_by_window(
            static_model, adaptive_model, evaluation_frame, adaptive_threshold,
            evaluation_days, int(windows["stability_window_days"]),
        )
        counts = manifest["row_counts"]
        report = {
            "scenario": manifest["scenario"],
            "generator_config": manifest.get("generator_config"),
            "experiment_start": manifest["experiment_start"],
            "experiment_windows": windows,
            "protocol": (f"train days {train_days[0]}-{train_days[1]}; KS-triggered feedback days "
                         f"{feedback_days[0]}-{feedback_days[1]} split 70/30 for retraining/calibration; "
                         f"evaluate held-out days {evaluation_days[0]}-{evaluation_days[1]}"),
            "class_balance_policy": "keep all positives and deterministically sample 8% of training negatives; H2O balance_classes disabled for every variant",
            "drift_detected": drift_detected,
            "retrain_triggered": drift_detected,
            "drift_monitor": manifest["drift_monitor"],
            "trees": args.trees,
            "h2o_runtime": {"max_memory": args.h2o_memory, "threads": args.h2o_threads},
            "pre_drift_train": counts["pre_drift_train"],
            "adaptive_train": counts["adaptive_train"],
            "feedback_calibration": counts["feedback_calibration"],
            "evaluation": counts["evaluation"],
            "variants": variants,
            "static_f1_at_0_5": variants["static_rf"]["f1"],
            "adaptive_f1_at_0_5": variants["weighted_retraining"]["f1"],
            "adaptive_operating_threshold": adaptive_threshold,
            "adaptive_calibration_f1": calibration_f1,
            "adaptive_f1_at_calibrated_threshold": variants["full_adaptive"]["f1"],
            "static_aucpr": float(static_performance.aucpr()),
            "adaptive_aucpr": float(adaptive_performance.aucpr()),
            "adaptive_threshold_curve": threshold_curve(adaptive_performance),
            "post_drift_daily_stability": stability,
            "shap_explanations": explanations,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "prepared_manifest": args.prepared_manifest.name,
        }
        report["f1_change_at_0_5"] = report["adaptive_f1_at_0_5"] - report["static_f1_at_0_5"]
        report["f1_change_at_operating_threshold"] = (
            report["adaptive_f1_at_calibrated_threshold"] - report["static_f1_at_0_5"]
        )
        report_path = output_dir / "adaptive_comparison.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({
            "scenario": report["scenario"], "protocol": report["protocol"],
            "drift_detected": report["drift_detected"], "variants": variants,
            "duration_seconds": report["duration_seconds"], "full_report": str(report_path),
        }, indent=2))
    finally:
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()

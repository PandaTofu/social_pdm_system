"""Compare a static and a KS-triggered adaptive H2O Random Forest on telemetry.

Timeline: days 1-3 train both systems; day 5 is the detected drift window and
supplies delayed feedback labels.  A deterministic 70/30 feedback split keeps
the threshold-calibration rows out of retraining; days 6-7 are held out for the
final, identical evaluation.  This prevents label and threshold leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spark.streaming_job import SCHEMA, classify
from ml.h2o_binary_metrics import binary_summary, threshold_curve

FEATURES = ["cpu_util_pct", "memory_util_pct", "disk_util_pct", "disk_iops", "disk_queue_depth",
            "network_in_mbps", "network_out_mbps", "request_rate_rps", "response_p50_ms",
            "response_p95_ms", "error_rate", "timeout_rate", "queue_depth", "active_connections"]
TARGET = "failure_within_30min"


def spark_session() -> SparkSession:
    return (SparkSession.builder.appName("adaptive-telemetry-comparison")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "16")
            .getOrCreate())


def export_csv(frame: DataFrame, path: Path, balance_sample: bool,
               add_recency_weight: bool = False) -> tuple[Path, int, int]:
    # Context columns are excluded from model features but retained so sampled
    # alerts can be traced back to a logical server and event time.
    selected = frame.select("event_id", "event_time", "server_id", *FEATURES, TARGET, "day").dropna(
        subset=FEATURES + [TARGET]
    )
    if balance_sample:
        # Keep every scarce positive row; sample only negatives before H2O's
        # class balancing. This bounds local AutoDL memory without changing time order.
        selected = selected.sampleBy(TARGET, fractions={0: 0.08, 1: 1.0}, seed=20260824)
    if add_recency_weight:
        # Drift-era labels are deliberately four times as influential as the
        # legacy regime.  This is an explicit, reproducible recency policy.
        selected = selected.withColumn("sample_weight", F.when(F.col("day") == 5, F.lit(4.0)).otherwise(F.lit(1.0)))
    selected = selected.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        total = selected.count()
        positives = selected.filter(F.col(TARGET) == 1).count()
        selected.coalesce(1).write.mode("overwrite").option("header", True).csv(str(path))
    finally:
        selected.unpersist()
    return next(path.glob("part-*.csv")), total, positives


def train_h2o(csv_path: Path, trees: int, use_recency_weight: bool = False):
    frame = h2o.import_file(str(csv_path))
    frame[TARGET] = frame[TARGET].asfactor()
    model = H2ORandomForestEstimator(ntrees=trees, max_depth=20, min_rows=5,
                                     # Keep class handling identical across all
                                     # ablations. Spark already applies the same
                                     # deterministic negative downsampling.
                                     balance_classes=False, seed=20260824)
    model.train(x=FEATURES, y=TARGET, training_frame=frame,
                weights_column="sample_weight" if use_recency_weight else None)
    return model


def select_f1_threshold(performance) -> tuple[float, float]:
    threshold, score = max(performance.F1(), key=lambda item: item[1])
    return float(threshold), float(score)


def write_explanations(model, evaluation_frame, threshold: float, output_dir: Path,
                       max_rows: int) -> dict[str, object]:
    """Persist real H2O TreeSHAP contributions for predicted alerts.

    Explanation support differs across H2O model builds.  An unsupported build
    is recorded explicitly instead of failing the accuracy experiment or
    fabricating an explanation.
    """
    try:
        predictions = model.predict(evaluation_frame)
        alert_mask = predictions["p1"] >= threshold
        alert_rows = evaluation_frame[alert_mask, :]
        alert_predictions = predictions[alert_mask, :]
        selected_rows = min(int(alert_rows.nrows), max_rows)
        if selected_rows == 0:
            return {"status": "no_alerts", "rows": 0}
        sample = alert_rows[0:selected_rows, :]
        contributions = model.predict_contributions(sample)
        context_columns = [name for name in ("event_id", "event_time", "server_id", TARGET)
                           if name in sample.columns]
        combined = sample[context_columns].cbind(alert_predictions[0:selected_rows, :]).cbind(contributions)
        destination = output_dir / "shap_alert_explanations.csv"
        h2o.download_csv(combined, str(destination))
        return {"status": "generated", "rows": selected_rows, "path": str(destination)}
    except Exception as error:  # H2O DRF SHAP availability is runtime/version dependent.
        return {"status": "unsupported", "rows": 0, "error": f"{type(error).__name__}: {error}"}


def stability_by_day(static_model, adaptive_model, evaluation_frame, adaptive_threshold: float) -> list[dict[str, object]]:
    """Measure post-drift performance without claiming a six-month simulation."""
    rows: list[dict[str, object]] = []
    for day in (6, 7):
        daily = evaluation_frame[evaluation_frame["day"] == day, :]
        if int(daily.nrows) == 0:
            continue
        rows.append({
            "day": day,
            "rows": int(daily.nrows),
            "static": binary_summary(static_model.model_performance(daily), 0.5),
            "full_adaptive": binary_summary(adaptive_model.model_performance(daily), adaptive_threshold),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Static versus KS-triggered adaptive telemetry comparison.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drift-report", type=Path, required=True,
                        help="JSON output created by ml/drift_monitor.py")
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--explain-rows", type=int, default=500)
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip the unweighted retraining model for a shorter smoke run")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    drift_report = json.loads(args.drift_report.read_text(encoding="utf-8"))
    drift_detected = bool(drift_report.get("drift_detected", False))

    spark = spark_session()
    try:
        data = (classify(spark.read.schema(SCHEMA).json(str(args.source)))
                .filter("is_valid")
                .dropDuplicates(["event_id"])
                .withColumn("day", F.dayofmonth("event_time")))
        pre = data.filter(F.col("day").between(1, 3))
        drift_feedback = data.filter(F.col("day") == 5)
        # Feedback split happens before both retraining and threshold tuning.
        feedback_train = drift_feedback.filter(F.pmod(F.xxhash64("event_id"), F.lit(10)) < 7)
        feedback_calibration = drift_feedback.filter(F.pmod(F.xxhash64("event_id"), F.lit(10)) >= 7)
        adaptive_training = pre.unionByName(feedback_train) if drift_detected else pre
        evaluation = data.filter(F.col("day").between(6, 7))
        pre_csv, pre_rows, pre_positive = export_csv(pre, args.output_dir / "prepared" / "pre_drift_train", True)
        adaptive_csv, adaptive_rows, adaptive_positive = export_csv(adaptive_training, args.output_dir / "prepared" / "adaptive_train", True, add_recency_weight=drift_detected)
        calibration_csv, calibration_rows, calibration_positive = export_csv(feedback_calibration, args.output_dir / "prepared" / "feedback_calibration", False)
        eval_csv, eval_rows, eval_positive = export_csv(evaluation, args.output_dir / "prepared" / "evaluation", False)
    finally:
        spark.stop()

    try:
        h2o.init(ip="127.0.0.1", port=6008, max_mem_size="4G", nthreads=4)
        static_model = train_h2o(pre_csv, args.trees)
        evaluation_frame = h2o.import_file(str(eval_csv))
        static_performance = static_model.model_performance(evaluation_frame)
        retrained_model = None
        retrained_performance = None
        if drift_detected and not args.skip_ablation:
            retrained_model = train_h2o(adaptive_csv, args.trees, use_recency_weight=False)
            retrained_performance = retrained_model.model_performance(evaluation_frame)
        adaptive_model = train_h2o(adaptive_csv, args.trees, use_recency_weight=drift_detected)
        adaptive_performance = adaptive_model.model_performance(evaluation_frame)
        calibration_performance = adaptive_model.model_performance(h2o.import_file(str(calibration_csv)))
        adaptive_threshold, calibration_f1 = select_f1_threshold(calibration_performance)
        variants = {
            "static_rf": binary_summary(static_performance, 0.5),
            "weighted_retraining": binary_summary(adaptive_performance, 0.5),
            "full_adaptive": binary_summary(adaptive_performance, adaptive_threshold),
        }
        if retrained_performance is not None:
            variants["unweighted_retraining"] = binary_summary(retrained_performance, 0.5)
        explanations = write_explanations(
            adaptive_model, evaluation_frame, adaptive_threshold, args.output_dir, args.explain_rows
        )
        stability = stability_by_day(static_model, adaptive_model, evaluation_frame, adaptive_threshold)
        report = {
            "protocol": "train days 1-3; KS-detected day-5 feedback split 70/30 for recency-weighted retraining/threshold calibration; evaluate days 6-7",
            "class_balance_policy": "keep all positives and deterministically sample 8% of training negatives; H2O balance_classes disabled for every variant",
            "drift_detected": drift_detected, "retrain_triggered": drift_detected,
            "drift_monitor": drift_report, "trees": args.trees,
            "pre_drift_train": {"rows": pre_rows, "positive_rows": pre_positive},
            "adaptive_train": {"rows": adaptive_rows, "positive_rows": adaptive_positive},
            "feedback_calibration": {"rows": calibration_rows, "positive_rows": calibration_positive},
            "evaluation": {"rows": eval_rows, "positive_rows": eval_positive},
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
        }
        report["f1_change_at_0_5"] = report["adaptive_f1_at_0_5"] - report["static_f1_at_0_5"]
        report["f1_change_at_operating_threshold"] = report["adaptive_f1_at_calibrated_threshold"] - report["static_f1_at_0_5"]
        (args.output_dir / "adaptive_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()

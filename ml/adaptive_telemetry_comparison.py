"""Compare a static and a KS-triggered adaptive H2O Random Forest on telemetry.

Timeline: days 1-3 train both systems; day 5 is the detected drift window and
supplies delayed feedback labels for retraining only; days 6-7 are held out for
the final, identical evaluation.  This prevents retraining-label leakage.
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

FEATURES = ["cpu_util_pct", "memory_util_pct", "disk_util_pct", "disk_iops", "disk_queue_depth",
            "network_in_mbps", "network_out_mbps", "request_rate_rps", "response_p50_ms",
            "response_p95_ms", "error_rate", "timeout_rate", "queue_depth", "active_connections"]
TARGET = "failure_within_30min"


def spark_session() -> SparkSession:
    return (SparkSession.builder.appName("adaptive-telemetry-comparison")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "16")
            .getOrCreate())


def export_csv(frame: DataFrame, path: Path, balance_sample: bool) -> tuple[Path, int, int]:
    selected = frame.select(*FEATURES, TARGET).dropna()
    if balance_sample:
        # Keep every scarce positive row; sample only negatives before H2O's
        # class balancing. This bounds local AutoDL memory without changing time order.
        selected = selected.sampleBy(TARGET, fractions={0: 0.08, 1: 1.0}, seed=20260824)
    selected = selected.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        total = selected.count()
        positives = selected.filter(F.col(TARGET) == 1).count()
        selected.coalesce(1).write.mode("overwrite").option("header", True).csv(str(path))
    finally:
        selected.unpersist()
    return next(path.glob("part-*.csv")), total, positives


def f1_at_half(performance) -> float:
    return float(performance.F1(thresholds=[0.5])[0][1])


def train_h2o(csv_path: Path, trees: int):
    frame = h2o.import_file(str(csv_path))
    frame[TARGET] = frame[TARGET].asfactor()
    model = H2ORandomForestEstimator(ntrees=trees, max_depth=20, min_rows=5,
                                     balance_classes=True, seed=20260824)
    model.train(x=FEATURES, y=TARGET, training_frame=frame)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Static versus KS-triggered adaptive telemetry comparison.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drift-report", type=Path, required=True,
                        help="JSON output created by ml/drift_monitor.py")
    parser.add_argument("--trees", type=int, default=80)
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
        adaptive_training = pre.unionByName(drift_feedback) if drift_detected else pre
        evaluation = data.filter(F.col("day").between(6, 7))
        pre_csv, pre_rows, pre_positive = export_csv(pre, args.output_dir / "prepared" / "pre_drift_train", True)
        adaptive_csv, adaptive_rows, adaptive_positive = export_csv(adaptive_training, args.output_dir / "prepared" / "adaptive_train", True)
        eval_csv, eval_rows, eval_positive = export_csv(evaluation, args.output_dir / "prepared" / "evaluation", False)
    finally:
        spark.stop()

    try:
        h2o.init(ip="127.0.0.1", port=6008, max_mem_size="4G", nthreads=4)
        static_model = train_h2o(pre_csv, args.trees)
        static_performance = static_model.model_performance(h2o.import_file(str(eval_csv)))
        adaptive_model = train_h2o(adaptive_csv, args.trees)
        adaptive_performance = adaptive_model.model_performance(h2o.import_file(str(eval_csv)))
        report = {
            "protocol": "train days 1-3; KS-detected feedback/retrain day 5; evaluate days 6-7",
            "drift_detected": drift_detected, "retrain_triggered": drift_detected,
            "drift_monitor": drift_report, "trees": args.trees,
            "pre_drift_train": {"rows": pre_rows, "positive_rows": pre_positive},
            "adaptive_train": {"rows": adaptive_rows, "positive_rows": adaptive_positive},
            "evaluation": {"rows": eval_rows, "positive_rows": eval_positive},
            "static_f1_at_0_5": f1_at_half(static_performance),
            "adaptive_f1_at_0_5": f1_at_half(adaptive_performance),
            "static_aucpr": float(static_performance.aucpr()),
            "adaptive_aucpr": float(adaptive_performance.aucpr()),
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
        report["f1_change"] = report["adaptive_f1_at_0_5"] - report["static_f1_at_0_5"]
        (args.output_dir / "adaptive_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()

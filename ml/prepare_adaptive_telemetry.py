"""Prepare leakage-safe telemetry splits with Spark, then exit the JVM.

This is intentionally a separate process from H2O training.  On local and
small cloud machines, running both JVMs in one Python process can retain the
Spark gateway while H2O starts and needlessly doubles peak memory usage.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyspark.sql import functions as F

from adaptive_telemetry_comparison import export_csv, resolve_windows, spark_session
from spark.streaming_job import SCHEMA, classify


def relative_csv(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare adaptive telemetry experiment CSVs with Spark.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drift-report", type=Path, required=True)
    parser.add_argument("--scenario-name", default="unspecified")
    parser.add_argument("--generator-config", type=Path)
    parser.add_argument("--experiment-start", help="UTC experiment date, YYYY-MM-DD")
    parser.add_argument("--train-days", nargs=2, type=int)
    parser.add_argument("--feedback-days", nargs=2, type=int)
    parser.add_argument("--evaluation-days", nargs=2, type=int)
    parser.add_argument("--stability-window-days", type=int)
    parser.add_argument("--recency-weight", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.output_dir / "prepared_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite explicitly")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    drift_report = json.loads(args.drift_report.read_text(encoding="utf-8"))
    drift_detected = bool(drift_report.get("drift_detected", False))
    generator_config = (json.loads(args.generator_config.read_text(encoding="utf-8"))
                        if args.generator_config else {})
    windows = resolve_windows(generator_config, args)
    train_days = tuple(windows["initial_train"])
    feedback_days = tuple(windows["feedback"])
    evaluation_days = tuple(windows["evaluation"])
    configured_start = str(generator_config.get("start_time", ""))[:10]
    experiment_start = args.experiment_start or configured_start or None

    spark = spark_session()
    try:
        data = (classify(spark.read.schema(SCHEMA).option("recursiveFileLookup", "true").json(str(args.source)))
                .filter("is_valid").dropDuplicates(["event_id"]))
        source_partitions = data.rdd.getNumPartitions()
        if experiment_start is None:
            experiment_start = str(data.select(F.min(F.to_date("event_time"))).first()[0])
        data = data.withColumn(
            "day", F.datediff(F.to_date("event_time"), F.lit(experiment_start).cast("date")) + F.lit(1)
        )
        pre = data.filter(F.col("day").between(*train_days))
        drift_feedback = data.filter(F.col("day").between(*feedback_days))
        feedback_train = drift_feedback.filter(F.pmod(F.xxhash64("event_id"), F.lit(10)) < 7)
        feedback_calibration = drift_feedback.filter(F.pmod(F.xxhash64("event_id"), F.lit(10)) >= 7)
        adaptive_training = pre.unionByName(feedback_train) if drift_detected else pre
        evaluation = data.filter(F.col("day").between(*evaluation_days))

        pre_csv, pre_rows, pre_positive = export_csv(
            pre, args.output_dir / "prepared" / "pre_drift_train", True
        )
        adaptive_csv, adaptive_rows, adaptive_positive = export_csv(
            adaptive_training, args.output_dir / "prepared" / "adaptive_train", True,
            add_recency_weight=drift_detected, recent_days=feedback_days,
            recency_weight=float(windows["recency_weight"]),
        )
        calibration_csv, calibration_rows, calibration_positive = export_csv(
            feedback_calibration, args.output_dir / "prepared" / "feedback_calibration", False
        )
        eval_csv, eval_rows, eval_positive = export_csv(
            evaluation, args.output_dir / "prepared" / "evaluation", False
        )
    finally:
        spark.stop()

    manifest = {
        "format_version": 1,
        "scenario": args.scenario_name,
        "generator_config": generator_config or None,
        "experiment_start": experiment_start,
        "experiment_windows": windows,
        "drift_detected": drift_detected,
        "drift_monitor": drift_report,
        "source_partitions": source_partitions,
        "csv_paths": {
            "pre_drift_train": relative_csv(pre_csv, args.output_dir),
            "adaptive_train": relative_csv(adaptive_csv, args.output_dir),
            "feedback_calibration": relative_csv(calibration_csv, args.output_dir),
            "evaluation": relative_csv(eval_csv, args.output_dir),
        },
        "row_counts": {
            "pre_drift_train": {"rows": pre_rows, "positive_rows": pre_positive},
            "adaptive_train": {"rows": adaptive_rows, "positive_rows": adaptive_positive},
            "feedback_calibration": {"rows": calibration_rows, "positive_rows": calibration_positive},
            "evaluation": {"rows": eval_rows, "positive_rows": eval_positive},
        },
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"prepared_manifest": str(manifest_path), **manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()

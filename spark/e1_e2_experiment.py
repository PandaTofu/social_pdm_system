"""Run the formal E1 quality and E2 routing/workload experiments.

The job uses the same explicit schema and DataFrame transformations as the
streaming profiles.  It writes auditable JSON/Parquet artefacts and never
claims that controller decisions are physical multi-node scale-out.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spark.quality import SCHEMA, classify_quality, quality_report
from spark.routing import route_tasks, routing_report, simulate_fixed_capacity


def session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("social-pdm-e1-e2-experiment")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", os.getenv("SHUFFLE_PARTITIONS", "48"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    return spark


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def quality_rule_counts(classified: DataFrame) -> list[dict[str, Any]]:
    rows = (
        classified.select(F.explode_outer("quality_failure_reasons").alias("rule_id"))
        .filter(F.col("rule_id").isNotNull())
        .groupBy("rule_id").count().orderBy("rule_id").collect()
    )
    return [{"rule_id": row["rule_id"], "count": int(row["count"])} for row in rows]


def timed_workload(frame: DataFrame) -> dict[str, Any]:
    """Execute a common aggregation that represents feature preparation."""
    started = time.perf_counter()
    grouped = (
        frame.select(
            "event_time", "data_center_id", "service_name", "request_rate_rps",
            "response_p95_ms", "error_rate", "queue_depth",
        )
        .groupBy(F.window("event_time", "5 minutes"), "data_center_id", "service_name")
        .agg(
            F.count("*").alias("records"),
            F.avg("response_p95_ms").alias("response_p95_mean"),
            F.avg("error_rate").alias("error_rate_mean"),
            F.max("queue_depth").alias("queue_depth_max"),
            F.sum("request_rate_rps").alias("request_rate_sum"),
        )
    )
    summary = grouped.agg(
        F.sum("records").alias("input_rows"),
        F.count("*").alias("output_groups"),
    ).first()
    duration = time.perf_counter() - started
    rows = int(summary["input_rows"] or 0)
    return {
        "input_rows": rows,
        "output_groups": int(summary["output_groups"] or 0),
        "duration_seconds": duration,
        "throughput_rows_per_second": rows / duration if duration else 0.0,
    }


def benchmark_routing_paths(frame: DataFrame) -> dict[str, Any]:
    baseline = timed_workload(frame)
    stream = timed_workload(frame.filter(F.col("route") == "stream"))
    batch = timed_workload(frame.filter(F.col("route") == "batch"))
    enhanced_duration = stream["duration_seconds"] + batch["duration_seconds"]
    return {
        "baseline_single_path": baseline,
        "enhanced_stream_path": stream,
        "enhanced_batch_path": batch,
        "enhanced_total_duration_seconds": enhanced_duration,
        "enhanced_total_throughput_rows_per_second": (
            baseline["input_rows"] / enhanced_duration if enhanced_duration else 0.0
        ),
        "critical_path_completion_seconds": stream["duration_seconds"],
    }


def minute_workload(frame: DataFrame) -> DataFrame:
    units = F.coalesce(F.col("workload_units"), F.lit(1.0))
    return (
        frame.groupBy(F.date_trunc("minute", "event_time").alias("event_minute"))
        .agg(
            F.sum(F.when(F.col("route") == "stream", units).otherwise(0.0)).alias("stream_units"),
            F.sum(F.when(F.col("route") == "batch", units).otherwise(0.0)).alias("batch_units"),
            F.max(F.coalesce(F.col("traffic_campaign_flag"), F.lit(False)).cast("int")).alias("is_burst"),
        )
        .orderBy("event_minute")
    )


def queue_experiment(frame: DataFrame, capacity_multiplier: float) -> dict[str, Any]:
    by_minute = minute_workload(frame).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        normal = by_minute.filter(F.col("is_burst") == 0).withColumn(
            "total_units", F.col("stream_units") + F.col("batch_units")
        )
        p95_total = normal.approxQuantile("total_units", [0.95], 0.001)
        p95_stream = normal.approxQuantile("stream_units", [0.95], 0.001)
        if not p95_total or not p95_stream:
            raise ValueError("No normal workload minutes are available")
        capacity = float(p95_total[0]) * capacity_multiplier
        normal_share = min(0.90, max(0.55, float(p95_stream[0]) / capacity + 0.02))
        burst_share = min(0.97, max(0.88, normal_share + 0.08))
        # One row per minute (43,200 rows for the formal run) is intentionally
        # small enough to collect for a transparent deterministic queue model.
        rows = [row.asDict() for row in by_minute.collect()]
        result = simulate_fixed_capacity(rows, capacity, normal_share, burst_share)
        result.update({
            "capacity_derivation": "1.10 x p95 normal-minute workload unless overridden",
            "capacity_multiplier": capacity_multiplier,
            "normal_stream_share": normal_share,
            "burst_stream_share": burst_share,
            "minute_count": len(rows),
        })
        return result
    finally:
        by_minute.unpersist()


def write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 quality and E2 routing/workload evidence.")
    parser.add_argument("--source", type=Path, required=True, help="Partitioned NDJSON root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity-multiplier", type=float, default=1.10)
    parser.add_argument("--skip-evidence-writes", action="store_true", help="Do not write Parquet audit rows")
    args = parser.parse_args()
    if args.capacity_multiplier <= 0:
        parser.error("--capacity-multiplier must be positive")

    spark = session()
    started = time.perf_counter()
    classified: DataFrame | None = None
    routed: DataFrame | None = None
    try:
        source = (
            spark.read.option("recursiveFileLookup", "true")
            .schema(SCHEMA).json(str(args.source))
        )
        source_partitions = source.rdd.getNumPartitions()
        classified = classify_quality(source).persist(StorageLevel.MEMORY_AND_DISK)
        classified.count()
        e1 = quality_report(classified)
        e1["rule_counts"] = quality_rule_counts(classified)
        e1["source_partitions"] = source_partitions

        accepted = classified.filter("is_valid")
        routed = route_tasks(accepted).persist(StorageLevel.MEMORY_AND_DISK)
        routed_count = routed.count()
        e2 = routing_report(routed)
        e2.update({
            "accepted_input_rows": int(e1["accepted"]),
            "lossless": routed_count == int(e1["accepted"]),
            "output_partitions": routed.rdd.getNumPartitions(),
        })

        normal = routed.filter(~F.coalesce(F.col("traffic_campaign_flag"), F.lit(False)))
        burst = routed.filter(F.coalesce(F.col("traffic_campaign_flag"), F.lit(False)))
        workload = {
            "scope_warning": "Fixed single-node capacity; no multi-node horizontal scaling is claimed.",
            "all_rows": benchmark_routing_paths(routed),
            "normal_rows": benchmark_routing_paths(normal),
            "burst_rows": benchmark_routing_paths(burst),
        }
        queue = queue_experiment(routed, args.capacity_multiplier)
        trace = queue.pop("trace")
        workload["queue_simulation"] = queue

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "quality" / "quality_report.json", e1)
        write_json(args.output_dir / "routing" / "routing_report.json", e2)
        write_json(args.output_dir / "workload" / "workload_report.json", workload)
        write_trace(args.output_dir / "workload" / "controller_trace.csv", trace)

        if not args.skip_evidence_writes:
            (
                classified.filter("is_valid").select(*SCHEMA.fieldNames())
                .write.mode("overwrite").json(str(args.output_dir / "quality" / "accepted"))
            )
            (
                classified.filter("NOT is_valid")
                .select(
                    "event_id", "event_time", "server_id", "is_injected_defect",
                    "injected_defect_category", "injected_defect_type",
                    "quality_failure_reasons", "detected_defect_categories", "quality_action",
                )
                .write.mode("overwrite").parquet(str(args.output_dir / "quality" / "quarantine"))
            )
            (
                routed.select(
                    "event_id", "event_time", "server_id", "task_type", "criticality",
                    "latency_sla_ms", "expected_route", "route", "route_rule_id",
                    "decision_reason", "decision_time", "route_correct", "workload_units",
                )
                .write.mode("overwrite").partitionBy("route").parquet(
                    str(args.output_dir / "routing" / "audit")
                )
            )

        manifest = {
            "experiment": "E1_E2_formal",
            "spark_version": spark.version,
            "spark_master": spark.sparkContext.master,
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "source": str(args.source),
            "output_dir": str(args.output_dir),
            "duration_seconds": time.perf_counter() - started,
            "quality_report": "quality/quality_report.json",
            "routing_report": "routing/routing_report.json",
            "workload_report": "workload/workload_report.json",
            "controller_trace": "workload/controller_trace.csv",
        }
        write_json(args.output_dir / "experiment_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    finally:
        if routed is not None:
            routed.unpersist()
        if classified is not None:
            classified.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()

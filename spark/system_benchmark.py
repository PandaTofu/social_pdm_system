"""Single-node Spark evidence benchmark for the AutoDL deployment profile.

The benchmark reports observed wall-clock latency and throughput.  It does not
claim Kafka, Kubernetes, NiFi, MongoDB, or multi-node scaling performance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spark.streaming_job import SCHEMA, classify, risk_scored


def session() -> SparkSession:
    spark = (SparkSession.builder.appName("social-pdm-system-benchmark")
             .config("spark.sql.adaptive.enabled", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    return spark


def evaluate(frame: DataFrame, requested_rows: int) -> dict[str, float | int]:
    started = time.perf_counter()
    classified = classify(frame.limit(requested_rows)).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        scored = risk_scored(classified.filter("is_valid"))
        summary = scored.agg(
            F.count("*").alias("accepted"),
            F.sum(F.col("predicted_failure").cast("long")).alias("predicted_alerts"),
        ).first().asDict()
        received = classified.count()
        quarantined = classified.filter("NOT is_valid").count()
    finally:
        classified.unpersist()
    duration = time.perf_counter() - started
    return {
        "requested_rows": requested_rows,
        "processed_rows": int(received),
        "accepted_rows": int(summary["accepted"]),
        "quarantined_rows": int(quarantined),
        "predicted_alerts": int(summary["predicted_alerts"] or 0),
        "duration_seconds": duration,
        "latency_ms": duration * 1000.0,
        "throughput_rows_per_second": received / duration if duration else 0.0,
    }


def timed_schema_read(spark: SparkSession, source: Path, explicit: bool) -> dict[str, float | int | str]:
    started = time.perf_counter()
    reader = (spark.read.option("recursiveFileLookup", "true").schema(SCHEMA)
              if explicit else spark.read.option("recursiveFileLookup", "true"))
    rows = reader.json(str(source)).count()
    duration = time.perf_counter() - started
    return {
        "mode": "explicit_schema" if explicit else "runtime_inference",
        "rows": int(rows),
        "duration_seconds": duration,
        "throughput_rows_per_second": rows / duration if duration else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the AutoDL single-node Spark profile.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50_000, 100_000, 250_000, 500_000])
    parser.add_argument("--compare-schema-read", action="store_true")
    args = parser.parse_args()
    spark = session()
    try:
        # Keep the source uncached so each timed run includes file parsing,
        # validation and scoring rather than measuring an in-memory DataFrame.
        source = spark.read.option("recursiveFileLookup", "true").schema(SCHEMA).json(str(args.source))
        available_rows = source.count()
        scaling = [evaluate(source, min(size, available_rows)) for size in sorted(set(args.sizes))]
        schema_read = []
        if args.compare_schema_read:
            schema_read = [timed_schema_read(spark, args.source, False), timed_schema_read(spark, args.source, True)]
        report = {
            "profile": "single-node AutoDL PySpark local mode",
            "scope_warning": "Results do not represent Kafka, Kubernetes or multi-node scaling.",
            "timed_scope": "file read + explicit-schema parse + quality validation + deterministic risk scoring actions",
            "available_rows": int(available_rows),
            "spark_version": spark.version,
            "spark_master": spark.sparkContext.master,
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "scaling": scaling,
            "schema_read_comparison": schema_read,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

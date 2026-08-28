"""Shared explicit schema and auditable five-category quality gates.

The rules use Spark SQL expressions only.  ``classify_quality`` operates on a
static DataFrame, including the DataFrame supplied to ``foreachBatch`` by the
streaming profiles, so duplicate detection is distributed and auditable.
"""
from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


QUALITY_CATEGORIES = ("schema", "range", "temporal", "completeness", "cross_field")

TELEMETRY_SCHEMA = StructType([
    StructField("event_id", StringType()),
    StructField("event_time", TimestampType()),
    StructField("ingest_time", TimestampType()),
    StructField("schema_version", IntegerType()),
    StructField("data_center_id", StringType()),
    StructField("rack_id", StringType()),
    StructField("server_id", StringType()),
    StructField("service_name", StringType()),
    StructField("cpu_util_pct", DoubleType()),
    StructField("memory_util_pct", DoubleType()),
    StructField("disk_util_pct", DoubleType()),
    StructField("disk_iops", DoubleType()),
    StructField("disk_queue_depth", DoubleType()),
    StructField("network_in_mbps", DoubleType()),
    StructField("network_out_mbps", DoubleType()),
    StructField("request_rate_rps", DoubleType()),
    StructField("response_p50_ms", DoubleType()),
    StructField("response_p95_ms", DoubleType()),
    StructField("error_rate", DoubleType()),
    StructField("timeout_rate", DoubleType()),
    StructField("queue_depth", DoubleType()),
    StructField("active_connections", IntegerType()),
    StructField("deployment_id", StringType()),
    StructField("software_version", StringType()),
    StructField("config_version", StringType()),
    StructField("maintenance_flag", BooleanType()),
    StructField("traffic_campaign_flag", BooleanType()),
    StructField("failure_within_30min", IntegerType()),
    StructField("cache_hit_ratio", DoubleType()),
    StructField("gc_pause_ms", DoubleType()),
    # Generator truth used only to score the quality and routing experiments.
    StructField("is_injected_defect", BooleanType()),
    StructField("injected_defect_category", StringType()),
    StructField("injected_defect_type", StringType()),
    StructField("expected_quality_action", StringType()),
    StructField("duplicate_ordinal", IntegerType()),
    StructField("task_type", StringType()),
    StructField("criticality", StringType()),
    StructField("latency_sla_ms", IntegerType()),
    StructField("expected_route", StringType()),
    StructField("workload_units", DoubleType()),
    StructField("source_sequence", LongType()),
])

# Compatibility aliases used by existing jobs.
SCHEMA = TELEMETRY_SCHEMA
REQUIRED_FIELDS = (
    "event_id", "event_time", "ingest_time", "schema_version",
    "data_center_id", "server_id", "cpu_util_pct", "memory_util_pct",
    "response_p50_ms", "response_p95_ms", "error_rate", "queue_depth",
)


def _rule(category: str, rule_id: str, condition: F.Column) -> tuple[str, str, F.Column]:
    return category, rule_id, condition


def classify_quality(frame: DataFrame) -> DataFrame:
    """Attach rule evidence, quality action and valid/quarantine routing.

    Duplicate detection deliberately retains the first observation and marks
    later occurrences.  ``duplicate_ordinal`` makes the injected truth
    deterministic; the row-number fallback also catches unlabelled duplicates.
    """
    order = Window.partitionBy("event_id").orderBy(
        F.coalesce(F.col("duplicate_ordinal"), F.lit(0)),
        F.col("ingest_time").asc_nulls_last(),
        F.coalesce(F.col("source_sequence"), F.lit(0)),
    )
    ranked = frame.withColumn("_duplicate_rank", F.row_number().over(order))
    duplicate = F.col("event_id").isNotNull() & (
        (F.coalesce(F.col("duplicate_ordinal"), F.lit(0)) > 0)
        | (F.col("_duplicate_rank") > 1)
    )

    missing_required = F.lit(False)
    for name in REQUIRED_FIELDS:
        missing_required = missing_required | F.col(name).isNull()

    rules = [
        _rule("schema", "unsupported_schema", ~F.col("schema_version").isin(1, 2)),
        _rule(
            "schema",
            "v2_required_field_missing",
            (F.col("schema_version") == 2)
            & (F.col("cache_hit_ratio").isNull() | F.col("gc_pause_ms").isNull()),
        ),
        _rule("range", "cpu_out_of_range", ~F.col("cpu_util_pct").between(0.0, 100.0)),
        _rule("range", "memory_out_of_range", ~F.col("memory_util_pct").between(0.0, 100.0)),
        _rule("range", "disk_out_of_range", ~F.col("disk_util_pct").between(0.0, 100.0)),
        _rule("range", "error_rate_out_of_range", ~F.col("error_rate").between(0.0, 1.0)),
        _rule("range", "timeout_rate_out_of_range", ~F.col("timeout_rate").between(0.0, 1.0)),
        _rule("range", "negative_queue_depth", F.col("queue_depth") < 0),
        _rule(
            "range",
            "cache_hit_out_of_range",
            (F.col("schema_version") == 2) & ~F.col("cache_hit_ratio").between(0.0, 1.0),
        ),
        _rule("temporal", "ingest_before_event", F.col("ingest_time") < F.col("event_time")),
        _rule(
            "temporal",
            "event_too_late",
            F.col("ingest_time") > F.col("event_time") + F.expr("INTERVAL 10 MINUTES"),
        ),
        _rule("completeness", "missing_required_field", missing_required),
        _rule("completeness", "duplicate_event", duplicate),
        _rule(
            "cross_field",
            "p95_below_p50",
            F.col("response_p95_ms") < F.col("response_p50_ms"),
        ),
        _rule(
            "cross_field",
            "timeout_exceeds_error_rate",
            F.col("timeout_rate") > F.col("error_rate") + F.lit(1e-9),
        ),
        _rule("cross_field", "negative_active_connections", F.col("active_connections") < 0),
    ]

    reason_values = [F.when(condition, F.lit(rule_id)) for _, rule_id, condition in rules]
    category_values = [F.when(condition, F.lit(category)) for category, _, condition in rules]
    reasons = F.filter(F.array(*reason_values), lambda value: value.isNotNull())
    categories = F.array_distinct(F.filter(F.array(*category_values), lambda value: value.isNotNull()))
    return (
        ranked
        .withColumn("quality_failure_reasons", reasons)
        .withColumn("detected_defect_categories", categories)
        .withColumn("quality_failure_type", F.concat_ws("|", F.col("quality_failure_reasons")))
        .withColumn("quality_rule_count", F.size("quality_failure_reasons"))
        .withColumn("is_valid", F.col("quality_rule_count") == 0)
        .withColumn("quality_action", F.when(F.col("is_valid"), "accept").otherwise("quarantine"))
        .withColumn("late_event_flag", F.array_contains("quality_failure_reasons", "event_too_late"))
        .withColumn("event_date", F.to_date("event_time"))
        .drop("_duplicate_rank")
    )


def quality_report(frame: DataFrame) -> dict[str, Any]:
    """Return dataset-level and per-category quality detection metrics."""
    truth = F.coalesce(F.col("is_injected_defect"), F.lit(False))
    detected = ~F.col("is_valid")
    expressions = [
        F.count("*").alias("received"),
        F.sum(F.col("is_valid").cast("long")).alias("accepted"),
        F.sum(detected.cast("long")).alias("quarantined"),
        F.sum(truth.cast("long")).alias("injected_defects"),
        F.sum((truth & detected).cast("long")).alias("tp"),
        F.sum((~truth & detected).cast("long")).alias("fp"),
        F.sum((truth & ~detected).cast("long")).alias("fn"),
        F.sum((~truth & ~detected).cast("long")).alias("tn"),
    ]
    for category in QUALITY_CATEGORIES:
        category_truth = F.col("injected_defect_category") == category
        category_detected = F.array_contains(F.col("detected_defect_categories"), category)
        expressions.extend([
            F.sum((category_truth & category_detected).cast("long")).alias(f"{category}_tp"),
            F.sum((~category_truth & category_detected).cast("long")).alias(f"{category}_fp"),
            F.sum((category_truth & ~category_detected).cast("long")).alias(f"{category}_fn"),
            F.sum(category_truth.cast("long")).alias(f"{category}_truth"),
        ])
    values = frame.agg(*expressions).first().asDict()

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    per_category: list[dict[str, Any]] = []
    for category in QUALITY_CATEGORIES:
        tp = int(values[f"{category}_tp"] or 0)
        fp = int(values[f"{category}_fp"] or 0)
        fn = int(values[f"{category}_fn"] or 0)
        per_category.append({
            "category": category,
            "truth_count": int(values[f"{category}_truth"] or 0),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
        })
    tp, fp, fn = (int(values[name] or 0) for name in ("tp", "fp", "fn"))
    clean = int(values["tn"] or 0) + fp
    return {
        "received": int(values["received"]),
        "accepted": int(values["accepted"] or 0),
        "quarantined": int(values["quarantined"] or 0),
        "injected_defects": int(values["injected_defects"] or 0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(values["tn"] or 0),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "clean_record_preservation": ratio(int(values["tn"] or 0), clean),
        "per_category": per_category,
    }

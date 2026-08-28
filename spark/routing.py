"""Lossless task routing and fixed-capacity workload-control helpers for E2."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from pyspark.sql import DataFrame, functions as F


STREAM_TASKS = ("health_monitor", "incident_response", "incident_enrichment")
BATCH_TASKS = ("capacity_analytics", "model_feedback", "audit_compaction")


def route_tasks(frame: DataFrame) -> DataFrame:
    """Attach an auditable Stream/Batch decision without dropping any row."""
    high_criticality = F.lower(F.coalesce(F.col("criticality"), F.lit(""))) == "high"
    tight_sla = F.col("latency_sla_ms").isNotNull() & (F.col("latency_sla_ms") <= 1000)
    declared_stream = F.col("task_type").isin(*STREAM_TASKS)
    declared_batch = F.col("task_type").isin(*BATCH_TASKS)
    route = (
        F.when(high_criticality, F.lit("stream"))
        .when(tight_sla, F.lit("stream"))
        .when(declared_stream, F.lit("stream"))
        .when(declared_batch, F.lit("batch"))
        .otherwise(F.lit("batch"))
    )
    rule_id = (
        F.when(high_criticality, F.lit("R1_HIGH_CRITICALITY"))
        .when(tight_sla, F.lit("R2_TIGHT_SLA"))
        .when(declared_stream, F.lit("R3_STREAM_TASK"))
        .when(declared_batch, F.lit("R4_DELAY_TOLERANT"))
        .otherwise(F.lit("R5_DEFAULT_BATCH"))
    )
    reason = (
        F.when(high_criticality, F.lit("high criticality requires immediate analysis"))
        .when(tight_sla, F.lit("latency SLA is at most 1000 ms"))
        .when(declared_stream, F.lit("task type is latency sensitive"))
        .when(declared_batch, F.lit("task type permits deferred execution"))
        .otherwise(F.lit("no stream rule matched"))
    )
    return (
        frame.withColumn("route", route)
        .withColumn("route_rule_id", rule_id)
        .withColumn("decision_reason", reason)
        .withColumn("decision_time", F.current_timestamp())
        .withColumn(
            "route_correct",
            F.when(F.col("expected_route").isNull(), F.lit(None).cast("boolean"))
            .otherwise(F.col("route") == F.col("expected_route")),
        )
    )


def routing_report(frame: DataFrame) -> dict[str, Any]:
    """Aggregate losslessness, route counts and labelled routing accuracy."""
    row = frame.agg(
        F.count("*").alias("routed_rows"),
        F.sum((F.col("route") == "stream").cast("long")).alias("stream_rows"),
        F.sum((F.col("route") == "batch").cast("long")).alias("batch_rows"),
        F.sum(F.col("expected_route").isNotNull().cast("long")).alias("labelled_rows"),
        F.sum(F.coalesce(F.col("route_correct"), F.lit(False)).cast("long")).alias("correct_rows"),
        F.sum(F.coalesce(F.col("workload_units"), F.lit(1.0))).alias("workload_units"),
    ).first().asDict()
    labelled = int(row["labelled_rows"] or 0)
    correct = int(row["correct_rows"] or 0)
    return {
        "routed_rows": int(row["routed_rows"]),
        "stream_rows": int(row["stream_rows"] or 0),
        "batch_rows": int(row["batch_rows"] or 0),
        "labelled_rows": labelled,
        "correct_rows": correct,
        "routing_accuracy": float(correct / labelled) if labelled else None,
        "workload_units": float(row["workload_units"] or 0.0),
    }


def simulate_fixed_capacity(
    minute_rows: Iterable[dict[str, Any]],
    total_capacity_units_per_minute: float,
    normal_stream_share: float = 0.55,
    burst_stream_share: float = 0.80,
) -> dict[str, Any]:
    """Compare a unified queue with priority-aware resource redistribution.

    Both alternatives receive exactly the same fixed physical capacity.  The
    enhanced controller changes only the Stream/Batch share, so the result is
    evidence of scheduling behaviour rather than multi-node scale-out.
    """
    if total_capacity_units_per_minute <= 0:
        raise ValueError("total capacity must be positive")
    if not 0 < normal_stream_share < 1 or not 0 < burst_stream_share < 1:
        raise ValueError("capacity shares must fall between zero and one")

    rows = list(minute_rows)
    baseline_backlog = 0.0
    stream_backlog = 0.0
    batch_backlog = 0.0
    peak_baseline = peak_stream = peak_batch = 0.0
    max_baseline_delay = 0.0
    max_stream_delay = 0.0
    trace: list[dict[str, Any]] = []
    burst_indices = [index for index, row in enumerate(rows) if bool(row.get("is_burst"))]
    last_burst_index = max(burst_indices) if burst_indices else None
    baseline_recovery: int | None = None
    enhanced_recovery: int | None = None

    previous_minute: Any = None
    elapsed_since_last_burst = 0.0
    for index, raw in enumerate(rows):
        stream_input = float(raw.get("stream_units") or 0.0)
        batch_input = float(raw.get("batch_units") or 0.0)
        burst = bool(raw.get("is_burst"))
        current_minute = raw.get("event_minute")
        if isinstance(current_minute, datetime) and isinstance(previous_minute, datetime):
            elapsed_minutes = max(1.0, (current_minute - previous_minute).total_seconds() / 60.0)
        else:
            elapsed_minutes = 1.0
        previous_minute = current_minute
        share = burst_stream_share if burst or stream_backlog > 0 else normal_stream_share
        stream_capacity_per_minute = total_capacity_units_per_minute * share
        batch_capacity_per_minute = total_capacity_units_per_minute - stream_capacity_per_minute
        total_capacity = total_capacity_units_per_minute * elapsed_minutes
        stream_capacity = stream_capacity_per_minute * elapsed_minutes
        batch_capacity = batch_capacity_per_minute * elapsed_minutes

        total_input = stream_input + batch_input
        baseline_backlog = max(0.0, baseline_backlog + total_input - total_capacity)
        stream_backlog = max(0.0, stream_backlog + stream_input - stream_capacity)
        batch_backlog = max(0.0, batch_backlog + batch_input - batch_capacity)
        peak_baseline = max(peak_baseline, baseline_backlog)
        peak_stream = max(peak_stream, stream_backlog)
        peak_batch = max(peak_batch, batch_backlog)
        max_stream_delay = max(max_stream_delay, stream_backlog / stream_capacity_per_minute)
        max_baseline_delay = max(max_baseline_delay, baseline_backlog / total_capacity_units_per_minute)

        action = "prioritize_stream" if share == burst_stream_share else "normal_allocation"
        trace.append({
            "index": index,
            "event_minute": str(raw.get("event_minute")),
            "is_burst": burst,
            "controller_action": action,
            "stream_capacity_share": share,
            "elapsed_minutes": elapsed_minutes,
            "stream_input_units": stream_input,
            "batch_input_units": batch_input,
            "baseline_backlog_units": baseline_backlog,
            "stream_backlog_units": stream_backlog,
            "batch_backlog_units": batch_backlog,
        })

        if last_burst_index is not None and index >= last_burst_index:
            if index > last_burst_index:
                elapsed_since_last_burst += elapsed_minutes
            if baseline_recovery is None and baseline_backlog == 0:
                baseline_recovery = int(round(elapsed_since_last_burst))
            if enhanced_recovery is None and stream_backlog == 0:
                enhanced_recovery = int(round(elapsed_since_last_burst))

    return {
        "scope": "fixed single-node capacity; scheduling/resource redistribution only",
        "total_capacity_units_per_minute": total_capacity_units_per_minute,
        "baseline": {
            "max_backlog_units": peak_baseline,
            "max_queue_delay_minutes": max_baseline_delay,
            "recovery_minutes_after_last_burst": baseline_recovery,
        },
        "enhanced": {
            "max_stream_backlog_units": peak_stream,
            "max_batch_backlog_units": peak_batch,
            "max_stream_queue_delay_minutes": max_stream_delay,
            "stream_recovery_minutes_after_last_burst": enhanced_recovery,
        },
        "trace": trace,
    }

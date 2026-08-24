"""AutoDL-compatible file-source Structured Streaming profile.

No Docker, Kafka, NiFi, MongoDB, or Kubernetes is required.  JSON batch files
are atomically replayed into an inbox and Spark writes auditable Parquet/JSON
outputs.  It shares the exact schema and five-dimensional quality gates used by
the Docker/Kafka profile.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from spark.streaming_job import SCHEMA, classify, risk_scored

INBOX = os.getenv("AUTODL_INBOX", str(ROOT / "data" / "autodl_runtime" / "inbox"))
OUT = os.getenv("AUTODL_OUTPUT", str(ROOT / "data" / "autodl_runtime" / "output"))
CHECKPOINT = os.getenv("AUTODL_CHECKPOINT", str(ROOT / "data" / "autodl_runtime" / "checkpoints"))


def session() -> SparkSession:
    return (SparkSession.builder.appName("social-pdm-autodl-file-stream")
            .master(os.getenv("SPARK_MASTER", "local[*]"))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", os.getenv("SHUFFLE_PARTITIONS", "16"))
            .config("spark.ui.port", os.getenv("SPARK_UI_PORT", "6006"))
            .config("spark.sql.files.maxRecordsPerFile", "250000")
            .getOrCreate())


def process_batch(batch: DataFrame, batch_id: int) -> None:
    # Cache only within a micro-batch: it is reused for quality metrics and
    # three sinks, then immediately released.
    started = time.perf_counter()
    classified = classify(batch).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        valid = classified.filter("is_valid")
        invalid = classified.filter("NOT is_valid")
        predictions = risk_scored(valid)
        counts = classified.agg(
            F.count("*").alias("received"),
            F.sum(F.col("is_valid").cast("long")).alias("accepted"),
            F.sum((~F.col("is_valid")).cast("long")).alias("quarantined"),
        ).first().asDict()
        valid.write.mode("append").partitionBy("event_date", "data_center_id").parquet(f"{OUT}/validated")
        invalid.write.mode("append").json(f"{OUT}/quarantine")
        predictions.write.mode("append").partitionBy("event_date").parquet(f"{OUT}/predictions")
        duration = time.perf_counter() - started
        received = int(counts["received"])
        metric_row = {
            "batch_id": int(batch_id),
            "received": received,
            "accepted": int(counts["accepted"] or 0),
            "quarantined": int(counts["quarantined"] or 0),
            "duration_ms": duration * 1000.0,
            "processing_rows_per_second": received / duration if duration else 0.0,
        }
        batch.sparkSession.createDataFrame([metric_row]).withColumn(
            "processed_at", F.current_timestamp()
        ).write.mode("append").json(f"{OUT}/metrics")
    finally:
        classified.unpersist()


def main() -> None:
    Path(INBOX).mkdir(parents=True, exist_ok=True)
    spark = session()
    source = (spark.readStream.schema(SCHEMA).option("maxFilesPerTrigger", os.getenv("MAX_FILES_PER_TRIGGER", "4"))
              .json(INBOX).withWatermark("event_time", "10 minutes").dropDuplicates(["event_id"]))
    query = (source.writeStream.foreachBatch(process_batch).outputMode("append")
             .option("checkpointLocation", CHECKPOINT).trigger(processingTime="10 seconds").start())
    query.awaitTermination()


if __name__ == "__main__": main()

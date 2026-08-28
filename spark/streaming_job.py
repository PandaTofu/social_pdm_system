"""Kafka -> PySpark Structured Streaming -> Parquet/Mongo/Kafka pipeline.

Run this with the Spark Kafka and MongoDB connector packages supplied by the
deployment manifest.  It deliberately uses explicit schemas and built-in Spark
functions only; no UDF or schema inference is used on the hot path.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spark.quality import SCHEMA, classify_quality


KAFKA = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
RAW_TOPIC = os.getenv("RAW_TOPIC", "telemetry-raw")
QUARANTINE_TOPIC = os.getenv("QUARANTINE_TOPIC", "telemetry-quarantine")
PREDICTIONS_TOPIC = os.getenv("PREDICTIONS_TOPIC", "predictions")
DATA_PATH = os.getenv("DATA_PATH", "/data/telemetry")
CHECKPOINT = os.getenv("CHECKPOINT", "/data/checkpoints/telemetry")
METRICS_PATH = os.getenv("METRICS_PATH", "/data/metrics")
MONGO_URI = os.getenv("MONGO_URI", "")

def spark_session() -> SparkSession:
    return (SparkSession.builder.appName("social-pdm-streaming")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.shuffle.partitions", os.getenv("SHUFFLE_PARTITIONS", "48"))
            .config("spark.sql.streaming.stateStore.rocksdb.changelogCheckpointing.enabled", "true")
            .getOrCreate())


def parsed_stream(spark: SparkSession) -> DataFrame:
    raw = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", KAFKA)
           .option("subscribe", RAW_TOPIC).option("startingOffsets", "earliest")
           .option("maxOffsetsPerTrigger", os.getenv("MAX_OFFSETS_PER_TRIGGER", "50000")).load())
    return (raw.select(F.col("value").cast("string").alias("raw_json"), F.col("timestamp").alias("kafka_time"),
                       F.col("partition").alias("kafka_partition"), F.col("offset").alias("kafka_offset"))
            .select("raw_json", "kafka_time", "kafka_partition", "kafka_offset", F.from_json("raw_json", SCHEMA).alias("d"))
            .select("raw_json", "kafka_time", "kafka_partition", "kafka_offset", "d.*"))


def classify(df: DataFrame) -> DataFrame:
    """Compatibility wrapper used by existing model and benchmark jobs."""
    return classify_quality(df)


def risk_scored(valid: DataFrame) -> DataFrame:
    # A deterministic transparent baseline; train_h2o.py replaces it with a
    # versioned H2O model in the complete experiment.
    score = (F.lit(-6.0) + F.col("cpu_util_pct") * .035 + F.col("memory_util_pct") * .018 +
             F.col("response_p95_ms") * .003 + F.col("error_rate") * 7.0 + F.col("queue_depth") * .025)
    return valid.withColumn("risk_probability", 1 / (1 + F.exp(-score))).withColumn("predicted_failure", F.col("risk_probability") >= F.lit(.5))


def kafka_write(df: DataFrame, topic: str) -> None:
    (df.select(F.col("server_id").alias("key"), F.to_json(F.struct("*")).alias("value")).write.format("kafka")
       .option("kafka.bootstrap.servers", KAFKA).option("topic", topic).save())


def process_batch(batch: DataFrame, batch_id: int) -> None:
    classified = classify(batch)
    valid = classified.filter("is_valid")
    invalid = classified.filter("NOT is_valid")
    predictions = risk_scored(valid)
    # Actions are intentionally limited to this per-batch metric and writes.
    metrics = classified.agg(F.count("*").alias("received"), F.sum(F.col("is_valid").cast("long")).alias("accepted"),
                             F.sum((~F.col("is_valid")).cast("long")).alias("quarantined")).withColumn("batch_id", F.lit(batch_id))
    metrics.write.mode("append").json(METRICS_PATH)
    (valid.write.mode("append").partitionBy("event_date", "data_center_id").parquet(DATA_PATH))
    kafka_write(invalid, QUARANTINE_TOPIC)
    kafka_write(predictions, PREDICTIONS_TOPIC)
    if MONGO_URI:
        (predictions.select("event_id", "event_time", "server_id", "service_name", "risk_probability", "predicted_failure", "schema_version")
         .write.format("mongodb").option("connection.uri", MONGO_URI).option("database", "pdm").option("collection", "predictions")
         .mode("append").save())


def main() -> None:
    spark = spark_session()
    # Keep duplicate observations until foreachBatch so the E1 quality gate can
    # quarantine and count them instead of silently deleting the evidence.
    query = (parsed_stream(spark).writeStream.foreachBatch(process_batch).outputMode("append")
             .option("checkpointLocation", CHECKPOINT).trigger(processingTime="10 seconds").start())
    query.awaitTermination()


if __name__ == "__main__":
    main()

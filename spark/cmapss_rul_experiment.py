"""Reproducible PySpark baseline for NASA C-MAPSS remaining-useful-life (RUL).

The C-MAPSS train files are run-to-failure trajectories.  RUL is therefore
constructed as each engine's final cycle minus its current cycle.  The official
test RUL labels describe the final observed cycle of each test engine, which is
the only test point used for scoring.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

SETTING_COLUMNS = [f"setting_{index}" for index in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
FEATURE_COLUMNS = ["cycle"] + SETTING_COLUMNS + SENSOR_COLUMNS
ALL_COLUMNS = ["unit_id", "cycle"] + SETTING_COLUMNS + SENSOR_COLUMNS


def build_session() -> SparkSession:
    return (SparkSession.builder.appName("cmapss-rul-baseline")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.sql.files.maxRecordsPerFile", "250000")
            .getOrCreate())


def read_trajectory(spark: SparkSession, path: str) -> DataFrame:
    """Read variable-whitespace C-MAPSS text with an explicit 26-column schema."""
    tokens = F.split(F.trim(F.col("value")), r"\s+")
    frame = spark.read.text(path).select(tokens.alias("tokens"))
    malformed = frame.filter(F.size("tokens") != len(ALL_COLUMNS)).count()
    if malformed:
        raise ValueError(f"{path} has {malformed} rows without {len(ALL_COLUMNS)} columns")
    typed = [F.element_at("tokens", index + 1).cast("long" if name in {"unit_id", "cycle"} else "double").alias(name)
             for index, name in enumerate(ALL_COLUMNS)]
    return frame.select(*typed)


def read_test_rul(spark: SparkSession, path: str) -> DataFrame:
    """Preserve line order without a Python/RDD-side index."""
    values = (spark.read.text(path, wholetext=True)
              .select(F.posexplode(F.split(F.trim(F.col("value")), r"\s+")).alias("position", "raw_rul")))
    return values.select((F.col("position") + 1).cast("long").alias("unit_id"),
                         F.col("raw_rul").cast("double").alias("actual_rul"))


def with_train_rul(frame: DataFrame, cap: float) -> DataFrame:
    terminal_cycle = F.max("cycle").over(Window.partitionBy("unit_id"))
    return frame.withColumn("label", F.least((terminal_cycle - F.col("cycle")).cast("double"), F.lit(cap)))


def final_test_cycles(frame: DataFrame) -> DataFrame:
    final_cycle = F.max("cycle").over(Window.partitionBy("unit_id"))
    return (frame.withColumn("_final_cycle", final_cycle)
            .filter(F.col("cycle") == F.col("_final_cycle"))
            .drop("_final_cycle"))


def build_model(model_name: str) -> Pipeline:
    """Keep preprocessing identical where relevant and use Spark-native models."""
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw", handleInvalid="error")
    if model_name == "linear":
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
        regression = LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction",
                                      regParam=0.1, elasticNetParam=0.0, maxIter=100)
        return Pipeline(stages=[assembler, scaler, regression])
    gbt = GBTRegressor(featuresCol="features_raw", labelCol="label", predictionCol="prediction",
                       maxIter=80, maxDepth=5, stepSize=0.05, maxBins=32, seed=20260824)
    return Pipeline(stages=[assembler, gbt])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PySpark C-MAPSS RUL baseline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rul-cap", type=float, default=125.0)
    parser.add_argument("--model", choices=["linear", "gbt"], default="linear")
    args = parser.parse_args()

    started = time.perf_counter()
    spark = build_session()
    try:
        train = with_train_rul(read_trajectory(spark, str(args.data_dir / f"train_{args.subset}.txt")), args.rul_cap)
        test = read_trajectory(spark, str(args.data_dir / f"test_{args.subset}.txt"))
        labels = read_test_rul(spark, str(args.data_dir / f"RUL_{args.subset}.txt")).withColumn(
            "actual_rul", F.least(F.col("actual_rul"), F.lit(args.rul_cap))
        )
        if final_test_cycles(test).select("unit_id").distinct().count() != labels.count():
            raise ValueError("The test-engine count does not match the official RUL label count")

        model = build_model(args.model).fit(train)

        scored = (model.transform(final_test_cycles(test))
                  .select("unit_id", F.greatest(F.lit(0.0), F.col("prediction")).alias("predicted_rul")))
        result = (scored.join(F.broadcast(labels), "unit_id", "inner")
                  .withColumn("error", F.col("predicted_rul") - F.col("actual_rul"))
                  .withColumn("absolute_error", F.abs("error"))
                  # Standard C-MAPSS asymmetric score: late predictions
                  # (positive error) receive the stronger exponential penalty.
                  .withColumn("nasa_penalty", F.when(
                      F.col("error") < 0,
                      F.exp(-F.col("error") / F.lit(13.0)) - F.lit(1.0),
                  ).otherwise(F.exp(F.col("error") / F.lit(10.0)) - F.lit(1.0))))
        output = args.output_dir / args.subset / args.model
        result.write.mode("overwrite").parquet(str(output / "predictions"))

        actual_mean = result.agg(F.avg("actual_rul").alias("actual_mean")).first()["actual_mean"]
        stats = result.agg(
            F.count("*").alias("test_engines"),
            F.sqrt(F.avg(F.pow("error", 2))).alias("rmse"),
            F.avg("absolute_error").alias("mae"),
            F.avg("error").alias("mean_error"),
            F.sum("nasa_penalty").alias("nasa_score"),
            F.avg(F.when(F.col("absolute_error") <= 10.0, F.lit(1.0)).otherwise(F.lit(0.0))).alias("within_10_cycles"),
            (F.lit(1.0) - F.sum(F.pow("error", 2)) /
             F.sum(F.pow(F.col("actual_rul") - F.lit(actual_mean), 2))).alias("r2"),
        ).first().asDict()
        stats.update({"subset": args.subset, "model": args.model, "rul_cap": args.rul_cap,
                      "train_rows": train.count(), "duration_seconds": round(time.perf_counter() - started, 3)})
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

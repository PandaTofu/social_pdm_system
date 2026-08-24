"""CMAPSS failure-horizon classification baseline aligned with Su et al. (2024).

PySpark performs explicit-schema ingestion and labels every engine cycle as
``failure_within_horizon``.  H2O Random Forest then trains with 10-fold
cross-validation, matching the baseline platform's modelling choice.  This is
classification, deliberately distinct from the separate RUL regression study.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle"] + SETTINGS + SENSORS
FEATURES = ["cycle"] + SETTINGS + SENSORS
TARGET = "failure_within_horizon"


def spark_session() -> SparkSession:
    return (SparkSession.builder.appName("cmapss-rf-classification-baseline")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "16")
            .getOrCreate())


def read_trajectory(spark: SparkSession, path: Path) -> DataFrame:
    fields = F.split(F.trim(F.col("value")), r"\s+")
    raw = spark.read.text(str(path)).select(fields.alias("fields"))
    malformed = raw.filter(F.size("fields") != len(COLUMNS)).count()
    if malformed:
        raise ValueError(f"{path} has {malformed} malformed rows")
    return raw.select(*[
        F.element_at("fields", i + 1).cast("long" if name in {"unit_id", "cycle"} else "double").alias(name)
        for i, name in enumerate(COLUMNS)
    ])


def read_rul(spark: SparkSession, path: Path) -> DataFrame:
    values = (spark.read.text(str(path), wholetext=True)
              .select(F.posexplode(F.split(F.trim("value"), r"\s+")).alias("position", "rul")))
    return values.select((F.col("position") + 1).cast("long").alias("unit_id"),
                         F.col("rul").cast("long").alias("rul_at_final_cycle"))


def label_train(frame: DataFrame, horizon: int) -> DataFrame:
    terminal = F.max("cycle").over(Window.partitionBy("unit_id"))
    return (frame.withColumn("_terminal_cycle", terminal)
            .withColumn(TARGET, (F.col("_terminal_cycle") - F.col("cycle") <= horizon).cast("int"))
            .drop("_terminal_cycle"))


def label_test(frame: DataFrame, rul: DataFrame, horizon: int) -> DataFrame:
    terminal = F.max("cycle").over(Window.partitionBy("unit_id"))
    with_terminal = frame.withColumn("_terminal_cycle", terminal)
    return (with_terminal.join(F.broadcast(rul), "unit_id", "inner")
            .withColumn("actual_rul", F.col("rul_at_final_cycle") + F.col("_terminal_cycle") - F.col("cycle"))
            .withColumn(TARGET, (F.col("actual_rul") <= horizon).cast("int"))
            .drop("_terminal_cycle", "rul_at_final_cycle", "actual_rul"))


def write_csv(frame: DataFrame, destination: Path) -> Path:
    frame.select(*FEATURES, TARGET).coalesce(1).write.mode("overwrite").option("header", True).csv(str(destination))
    return next(destination.glob("part-*.csv"))


def metric(performance, method: str) -> float:
    value = getattr(performance, method)(thresholds=[0.5])
    return float(value[0][1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H2O Random Forest CMAPSS classification baseline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args()

    started = time.perf_counter()
    prepared = args.output_dir / args.subset / "rf_baseline" / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    spark = spark_session()
    try:
        train = label_train(read_trajectory(spark, args.data_dir / f"train_{args.subset}.txt"), args.horizon)
        test = label_test(read_trajectory(spark, args.data_dir / f"test_{args.subset}.txt"),
                          read_rul(spark, args.data_dir / f"RUL_{args.subset}.txt"), args.horizon)
        train = train.persist(StorageLevel.MEMORY_AND_DISK)
        test = test.persist(StorageLevel.MEMORY_AND_DISK)
        train_rows, test_rows = train.count(), test.count()
        train_csv = write_csv(train, prepared / "train")
        test_csv = write_csv(test, prepared / "test")
    finally:
        spark.stop()

    try:
        h2o.init(ip="127.0.0.1", port=54321, max_mem_size="4G", nthreads=4)
        train_h2o, test_h2o = h2o.import_file(str(train_csv)), h2o.import_file(str(test_csv))
        train_h2o[TARGET] = train_h2o[TARGET].asfactor()
        test_h2o[TARGET] = test_h2o[TARGET].asfactor()
        model = H2ORandomForestEstimator(ntrees=args.trees, max_depth=20, min_rows=5, nfolds=10,
                                         fold_assignment="Stratified", seed=20260824)
        model.train(x=FEATURES, y=TARGET, training_frame=train_h2o)
        performance = model.model_performance(test_h2o)
        output = args.output_dir / args.subset / "rf_baseline"
        output.mkdir(parents=True, exist_ok=True)
        predictions = model.predict(test_h2o)
        h2o.download_csv(predictions.cbind(test_h2o[TARGET]), str(output / "predictions.csv"))
        report = {
            "subset": args.subset, "task": f"failure within {args.horizon} cycles", "model": "H2O Random Forest",
            "cross_validation_folds": 10, "trees": args.trees, "train_rows": train_rows, "test_rows": test_rows,
            "f1_at_0_5": metric(performance, "F1"), "precision_at_0_5": metric(performance, "precision"),
            "recall_at_0_5": metric(performance, "recall"), "auc": float(performance.auc()),
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()

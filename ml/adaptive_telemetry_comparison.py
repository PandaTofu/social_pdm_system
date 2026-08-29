"""Compare static and KS-triggered adaptive H2O Random Forest variants.

All experiment phases are expressed as continuous days from the configured UTC
start timestamp.  A deterministic 70/30 feedback split keeps threshold
calibration rows out of retraining, and the configured evaluation interval is
fully held out.  This prevents label, time and threshold leakage for both the
seven-day development scenario and the thirty-day paper scenario.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spark.streaming_job import SCHEMA, classify
from ml.h2o_binary_metrics import binary_summary, threshold_curve

BASE_FEATURES = ["cpu_util_pct", "memory_util_pct", "disk_util_pct", "disk_iops", "disk_queue_depth",
                 "network_in_mbps", "network_out_mbps", "request_rate_rps", "response_p50_ms",
                 "response_p95_ms", "error_rate", "timeout_rate", "queue_depth", "active_connections"]
HISTORY_FEATURES = [
    "error_rate_mean_5m", "error_rate_mean_15m", "error_rate_max_15m", "error_rate_delta_15m",
    "response_p95_mean_5m", "response_p95_mean_15m", "response_p95_delta_15m",
    "timeout_rate_mean_15m", "queue_depth_mean_15m", "cpu_util_mean_15m",
]
FEATURES = BASE_FEATURES
TARGET = "failure_within_30min"
DEFAULT_WINDOWS = {
    "initial_train": [1, 3],
    "feedback": [5, 5],
    "evaluation": [6, 7],
    "stability_window_days": 1,
    "recency_weight": 4.0,
}


def spark_session() -> SparkSession:
    session = (SparkSession.builder.appName("adaptive-telemetry-comparison")
               .config("spark.sql.adaptive.enabled", "true")
               # Generator scenario days are defined from a UTC start timestamp.
               # Pinning the session prevents host timezone from moving late day-7
               # incidents into day 8 and silently changing the held-out split.
               .config("spark.sql.session.timeZone", "UTC")
               .getOrCreate())
    session.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    return session


def add_history_features(frame: DataFrame) -> DataFrame:
    """Add past-only rolling features without using future observations."""
    order_seconds = F.col("event_time").cast("long")
    window_5m = Window.partitionBy("server_id").orderBy(order_seconds).rangeBetween(-4 * 60, 0)
    window_15m = Window.partitionBy("server_id").orderBy(order_seconds).rangeBetween(-14 * 60, 0)
    return (frame
            .withColumn("error_rate_mean_5m", F.avg("error_rate").over(window_5m))
            .withColumn("error_rate_mean_15m", F.avg("error_rate").over(window_15m))
            .withColumn("error_rate_max_15m", F.max("error_rate").over(window_15m))
            .withColumn("error_rate_delta_15m", F.col("error_rate") - F.col("error_rate_mean_15m"))
            .withColumn("response_p95_mean_5m", F.avg("response_p95_ms").over(window_5m))
            .withColumn("response_p95_mean_15m", F.avg("response_p95_ms").over(window_15m))
            .withColumn("response_p95_delta_15m", F.col("response_p95_ms") - F.col("response_p95_mean_15m"))
            .withColumn("timeout_rate_mean_15m", F.avg("timeout_rate").over(window_15m))
            .withColumn("queue_depth_mean_15m", F.avg("queue_depth").over(window_15m))
            .withColumn("cpu_util_mean_15m", F.avg("cpu_util_pct").over(window_15m)))


def export_csv(frame: DataFrame, path: Path, balance_sample: bool,
               feature_columns: list[str] | None = None, add_recency_weight: bool = False,
               recent_days: tuple[int, int] = (5, 5),
               recency_weight: float = 4.0) -> tuple[Path, int, int]:
    # Context columns are excluded from model features but retained so sampled
    # alerts can be traced back to a logical server and event time.
    model_features = feature_columns or FEATURES
    selected = frame.select("event_id", "event_time", "server_id", *model_features, TARGET, "day").dropna(
        subset=model_features + [TARGET]
    )
    if balance_sample:
        # Keep every scarce positive row; sample only negatives before H2O's
        # class balancing. This bounds local AutoDL memory without changing time order.
        selected = selected.sampleBy(TARGET, fractions={0: 0.08, 1: 1.0}, seed=20260824)
    if add_recency_weight:
        selected = selected.withColumn(
            "sample_weight",
            F.when(F.col("day").between(*recent_days), F.lit(recency_weight)).otherwise(F.lit(1.0)),
        )
    selected = selected.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        total = selected.count()
        positives = selected.filter(F.col(TARGET) == 1).count()
        selected.coalesce(1).write.mode("overwrite").option("header", True).csv(str(path))
    finally:
        selected.unpersist()
    return next(path.glob("part-*.csv")), total, positives


def train_h2o(csv_path: Path, trees: int, use_recency_weight: bool = False,
              feature_columns: list[str] | None = None):
    frame = h2o.import_file(str(csv_path))
    frame[TARGET] = frame[TARGET].asfactor()
    model = H2ORandomForestEstimator(ntrees=trees, max_depth=20, min_rows=5,
                                     # Keep class handling identical across all
                                     # ablations. Spark already applies the same
                                     # deterministic negative downsampling.
                                     balance_classes=False, seed=20260824)
    try:
        model.train(x=feature_columns or FEATURES, y=TARGET, training_frame=frame,
                    weights_column="sample_weight" if use_recency_weight else None)
    finally:
        h2o.remove(frame.frame_id)
    return model


def select_f1_threshold(performance) -> tuple[float, float]:
    threshold, score = max(performance.F1(), key=lambda item: item[1])
    return float(threshold), float(score)


def write_explanations(model, evaluation_frame, threshold: float, output_dir: Path,
                       max_rows: int) -> dict[str, object]:
    """Persist real H2O TreeSHAP contributions for predicted alerts.

    Explanation support differs across H2O model builds.  An unsupported build
    is recorded explicitly instead of failing the accuracy experiment or
    fabricating an explanation.
    """
    try:
        predictions = model.predict(evaluation_frame)
        alert_mask = predictions["p1"] >= threshold
        alert_rows = evaluation_frame[alert_mask, :]
        alert_predictions = predictions[alert_mask, :]
        selected_rows = min(int(alert_rows.nrows), max_rows)
        if selected_rows == 0:
            return {"status": "no_alerts", "rows": 0}
        sample = alert_rows[0:selected_rows, :]
        contributions = model.predict_contributions(sample)
        context_columns = [name for name in ("event_id", "event_time", "server_id", TARGET)
                           if name in sample.columns]
        combined = sample[context_columns].cbind(alert_predictions[0:selected_rows, :]).cbind(contributions)
        destination = output_dir / "shap_alert_explanations.csv"
        h2o.download_csv(combined, str(destination))
        return {"status": "generated", "rows": selected_rows, "path": str(destination)}
    except Exception as error:  # H2O DRF SHAP availability is runtime/version dependent.
        return {"status": "unsupported", "rows": 0, "error": f"{type(error).__name__}: {error}"}


def write_dashboard_predictions(model, health_frame, recent_frame, threshold: float,
                                output_dir: Path, recent_days: tuple[int, int],
                                max_alert_rows: int) -> dict[str, object]:
    """Export raw telemetry plus model scores for the read-only dashboard.

    ``health_frame`` contains one latest observation per logical server.
    ``recent_frame`` is deliberately limited to one or two evaluation days so
    dashboard evidence can be regenerated without scoring the complete
    simulation. TreeSHAP columns are exported separately by
    :func:`write_explanations` and are never mixed with raw telemetry here.
    """
    context = ["event_id", "event_time", "server_id", TARGET, "day"]
    display_columns = context + [name for name in BASE_FEATURES if name in health_frame.columns]
    health_predictions = model.predict(health_frame)
    health_combined = health_frame[display_columns].cbind(health_predictions)
    health_destination = output_dir / "server_health_snapshot.csv"
    h2o.download_csv(health_combined, str(health_destination))

    recent_predictions = model.predict(recent_frame)
    alert_mask = recent_predictions["p1"] >= threshold
    alert_frame = recent_frame[alert_mask, :]
    alert_predictions = recent_predictions[alert_mask, :]
    total_alert_rows = int(alert_frame.nrows)
    selected_rows = min(total_alert_rows, max_alert_rows)
    alert_destination = output_dir / "threshold_alerts.csv"
    if selected_rows:
        alert_columns = context + [name for name in BASE_FEATURES if name in alert_frame.columns]
        combined = alert_frame[0:selected_rows, alert_columns].cbind(alert_predictions[0:selected_rows, :])
        h2o.download_csv(combined, str(alert_destination))
    else:
        # Keep a valid header-only CSV so the dashboard can distinguish a real
        # zero-alert result from a missing experiment artefact.
        alert_destination.write_text(
            ",".join(display_columns + ["predict", "p0", "p1"]) + "\n", encoding="utf-8"
        )
    return {
        "status": "generated",
        "recent_days": list(recent_days),
        "health_rows": int(health_frame.nrows),
        "alert_rows": selected_rows,
        "total_alert_rows": total_alert_rows,
        "alerts_truncated": total_alert_rows > selected_rows,
        "health_snapshot": str(health_destination),
        "threshold_alerts": str(alert_destination),
    }


def stability_by_window(static_model, adaptive_model, evaluation_frame,
                        adaptive_threshold: float, evaluation_days: tuple[int, int],
                        window_days: int) -> list[dict[str, object]]:
    """Measure held-out post-drift performance in configured day windows."""
    rows: list[dict[str, object]] = []
    for start in range(evaluation_days[0], evaluation_days[1] + 1, window_days):
        end = min(start + window_days - 1, evaluation_days[1])
        window = evaluation_frame[(evaluation_frame["day"] >= start) & (evaluation_frame["day"] <= end), :]
        if int(window.nrows) == 0:
            continue
        rows.append({
            "day": end,
            "window_start_day": start,
            "window_end_day": end,
            "rows": int(window.nrows),
            "static": binary_summary(static_model.model_performance(window), 0.5),
            "full_adaptive": binary_summary(adaptive_model.model_performance(window), adaptive_threshold),
        })
    return rows


def resolve_windows(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    configured = config.get("experiment_windows", {}) if config else {}
    resolved = {**DEFAULT_WINDOWS, **configured}
    for name, value in (("initial_train", args.train_days),
                        ("feedback", args.feedback_days),
                        ("evaluation", args.evaluation_days)):
        if value:
            resolved[name] = value
    if args.stability_window_days:
        resolved["stability_window_days"] = args.stability_window_days
    if args.recency_weight:
        resolved["recency_weight"] = args.recency_weight
    for name in ("initial_train", "feedback", "evaluation"):
        start, end = map(int, resolved[name])
        if start <= 0 or end < start:
            raise ValueError(f"Invalid {name} day range: {resolved[name]}")
        resolved[name] = [start, end]
    if resolved["initial_train"][1] >= resolved["feedback"][0]:
        raise ValueError("Initial training must finish before delayed feedback begins")
    if resolved["feedback"][1] >= resolved["evaluation"][0]:
        raise ValueError("Feedback/calibration must finish before held-out evaluation begins")
    resolved["stability_window_days"] = int(resolved["stability_window_days"])
    resolved["recency_weight"] = float(resolved["recency_weight"])
    if resolved["stability_window_days"] <= 0 or resolved["recency_weight"] <= 0:
        raise ValueError("Stability window and recency weight must be positive")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Static versus KS-triggered adaptive telemetry comparison.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drift-report", type=Path, required=True,
                        help="JSON output created by ml/drift_monitor.py")
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--explain-rows", type=int, default=500)
    parser.add_argument("--dashboard-days", type=int, default=2,
                        help="Number of final evaluation days scored for dashboard alert history")
    parser.add_argument("--dashboard-alert-rows", type=int, default=2000,
                        help="Maximum recent threshold alerts persisted for the dashboard")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip the unweighted retraining model for a shorter smoke run")
    parser.add_argument("--scenario-name", default="unspecified")
    parser.add_argument("--generator-config", type=Path)
    parser.add_argument("--experiment-start", help="UTC experiment date, YYYY-MM-DD")
    parser.add_argument("--train-days", nargs=2, type=int)
    parser.add_argument("--feedback-days", nargs=2, type=int)
    parser.add_argument("--evaluation-days", nargs=2, type=int)
    parser.add_argument("--stability-window-days", type=int)
    parser.add_argument("--recency-weight", type=float)
    parser.add_argument("--h2o-memory", default="4G")
    parser.add_argument("--h2o-threads", type=int, default=4)
    args = parser.parse_args()
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
    if args.dashboard_days not in (1, 2):
        raise ValueError("--dashboard-days must be 1 or 2")
    if args.dashboard_alert_rows <= 0:
        raise ValueError("--dashboard-alert-rows must be positive")
    dashboard_days = (max(evaluation_days[0], evaluation_days[1] - args.dashboard_days + 1),
                      evaluation_days[1])
    configured_start = str(generator_config.get("start_time", ""))[:10]
    experiment_start = args.experiment_start or configured_start or None

    spark = spark_session()
    try:
        data = (classify(spark.read.schema(SCHEMA).option("recursiveFileLookup", "true").json(str(args.source)))
                .filter("is_valid").dropDuplicates(["event_id"]))
        if experiment_start is None:
            experiment_start = str(data.select(F.min(F.to_date("event_time"))).first()[0])
        data = data.withColumn(
            "day", F.datediff(F.to_date("event_time"), F.lit(experiment_start).cast("date")) + F.lit(1)
        )
        pre = data.filter(F.col("day").between(*train_days))
        drift_feedback = data.filter(F.col("day").between(*feedback_days))
        # Feedback split happens before both retraining and threshold tuning.
        feedback_train = drift_feedback.filter(F.pmod(F.xxhash64("event_id"), F.lit(10)) < 7)
        feedback_calibration = drift_feedback.filter(F.pmod(F.xxhash64("event_id"), F.lit(10)) >= 7)
        adaptive_training = pre.unionByName(feedback_train) if drift_detected else pre
        evaluation = data.filter(F.col("day").between(*evaluation_days))
        dashboard_recent = (evaluation.filter(F.col("day").between(*dashboard_days))
                            .orderBy(F.col("event_time").desc(), F.col("event_id").desc()))
        latest_window = Window.partitionBy("server_id").orderBy(
            F.col("event_time").desc(), F.col("event_id").desc()
        )
        dashboard_latest = (dashboard_recent
                            .withColumn("dashboard_row", F.row_number().over(latest_window))
                            .filter(F.col("dashboard_row") == 1).drop("dashboard_row"))
        pre_csv, pre_rows, pre_positive = export_csv(pre, args.output_dir / "prepared" / "pre_drift_train", True)
        adaptive_csv, adaptive_rows, adaptive_positive = export_csv(
            adaptive_training, args.output_dir / "prepared" / "adaptive_train", True,
            add_recency_weight=drift_detected, recent_days=feedback_days,
            recency_weight=float(windows["recency_weight"]),
        )
        calibration_csv, calibration_rows, calibration_positive = export_csv(feedback_calibration, args.output_dir / "prepared" / "feedback_calibration", False)
        eval_csv, eval_rows, eval_positive = export_csv(evaluation, args.output_dir / "prepared" / "evaluation", False)
        dashboard_recent_csv, dashboard_recent_rows, _ = export_csv(
            dashboard_recent, args.output_dir / "prepared" / "dashboard_recent", False
        )
        dashboard_latest_csv, dashboard_latest_rows, _ = export_csv(
            dashboard_latest, args.output_dir / "prepared" / "dashboard_latest", False
        )
    finally:
        spark.stop()

    try:
        h2o.init(
            ip="127.0.0.1", port=6008,
            max_mem_size=args.h2o_memory, nthreads=args.h2o_threads,
        )
        static_model = train_h2o(pre_csv, args.trees)
        evaluation_frame = h2o.import_file(str(eval_csv))
        static_performance = static_model.model_performance(evaluation_frame)
        retrained_summary = None
        if drift_detected and not args.skip_ablation:
            retrained_model = train_h2o(adaptive_csv, args.trees, use_recency_weight=False)
            try:
                retrained_summary = binary_summary(
                    retrained_model.model_performance(evaluation_frame), 0.5
                )
            finally:
                h2o.remove(retrained_model.model_id)
        adaptive_model = train_h2o(adaptive_csv, args.trees, use_recency_weight=drift_detected)
        adaptive_performance = adaptive_model.model_performance(evaluation_frame)
        calibration_frame = h2o.import_file(str(calibration_csv))
        try:
            calibration_performance = adaptive_model.model_performance(calibration_frame)
            adaptive_threshold, calibration_f1 = select_f1_threshold(calibration_performance)
        finally:
            h2o.remove(calibration_frame.frame_id)
        variants = {
            "static_rf": binary_summary(static_performance, 0.5),
            "weighted_retraining": binary_summary(adaptive_performance, 0.5),
            "full_adaptive": binary_summary(adaptive_performance, adaptive_threshold),
        }
        if retrained_summary is not None:
            variants["unweighted_retraining"] = retrained_summary
        explanations = write_explanations(
            adaptive_model, evaluation_frame, adaptive_threshold, args.output_dir, args.explain_rows
        )
        dashboard_recent_frame = h2o.import_file(str(dashboard_recent_csv))
        dashboard_latest_frame = h2o.import_file(str(dashboard_latest_csv))
        try:
            dashboard_exports = write_dashboard_predictions(
                adaptive_model, dashboard_latest_frame, dashboard_recent_frame,
                adaptive_threshold, args.output_dir, dashboard_days, args.dashboard_alert_rows,
            )
        finally:
            h2o.remove(dashboard_recent_frame.frame_id)
            h2o.remove(dashboard_latest_frame.frame_id)
        model_dir = args.output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        saved_model_path = Path(h2o.save_model(adaptive_model, path=str(model_dir), force=True))
        saved_model_reference = saved_model_path.resolve().relative_to(args.output_dir.resolve()).as_posix()
        stability = stability_by_window(
            static_model, adaptive_model, evaluation_frame, adaptive_threshold,
            evaluation_days, int(windows["stability_window_days"]),
        )
        report = {
            "scenario": args.scenario_name,
            "generator_config": generator_config or None,
            "experiment_start": experiment_start,
            "experiment_windows": windows,
            "protocol": (f"train days {train_days[0]}-{train_days[1]}; KS-triggered feedback days "
                         f"{feedback_days[0]}-{feedback_days[1]} split 70/30 for retraining/calibration; "
                         f"evaluate held-out days {evaluation_days[0]}-{evaluation_days[1]}"),
            "class_balance_policy": "keep all positives and deterministically sample 8% of training negatives; H2O balance_classes disabled for every variant",
            "drift_detected": drift_detected, "retrain_triggered": drift_detected,
            "drift_monitor": drift_report, "trees": args.trees,
            "h2o_runtime": {"max_memory": args.h2o_memory, "threads": args.h2o_threads},
            "pre_drift_train": {"rows": pre_rows, "positive_rows": pre_positive},
            "adaptive_train": {"rows": adaptive_rows, "positive_rows": adaptive_positive},
            "feedback_calibration": {"rows": calibration_rows, "positive_rows": calibration_positive},
            "evaluation": {"rows": eval_rows, "positive_rows": eval_positive},
            "dashboard_prediction_window": {
                "days": list(dashboard_days), "rows": dashboard_recent_rows,
                "latest_server_rows": dashboard_latest_rows,
            },
            "variants": variants,
            "static_f1_at_0_5": variants["static_rf"]["f1"],
            "adaptive_f1_at_0_5": variants["weighted_retraining"]["f1"],
            "adaptive_operating_threshold": adaptive_threshold,
            "adaptive_calibration_f1": calibration_f1,
            "adaptive_f1_at_calibrated_threshold": variants["full_adaptive"]["f1"],
            "static_aucpr": float(static_performance.aucpr()),
            "adaptive_aucpr": float(adaptive_performance.aucpr()),
            "adaptive_threshold_curve": threshold_curve(adaptive_performance),
            "post_drift_daily_stability": stability,
            "shap_explanations": explanations,
            "dashboard_exports": dashboard_exports,
            "adaptive_model_path": saved_model_reference,
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
        report["f1_change_at_0_5"] = report["adaptive_f1_at_0_5"] - report["static_f1_at_0_5"]
        report["f1_change_at_operating_threshold"] = report["adaptive_f1_at_calibrated_threshold"] - report["static_f1_at_0_5"]
        (args.output_dir / "adaptive_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({
            "scenario": report["scenario"],
            "protocol": report["protocol"],
            "drift_detected": report["drift_detected"],
            "variants": report["variants"],
            "duration_seconds": report["duration_seconds"],
            "full_report": str(args.output_dir / "adaptive_comparison.json"),
        }, indent=2))
    finally:
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()

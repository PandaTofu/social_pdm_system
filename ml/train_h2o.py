"""Train and register the paper's H2O Random Forest baseline on Parquet telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

FEATURES = ["cpu_util_pct", "memory_util_pct", "disk_util_pct", "disk_iops", "disk_queue_depth", "network_in_mbps", "network_out_mbps", "request_rate_rps", "response_p50_ms", "response_p95_ms", "error_rate", "timeout_rate", "queue_depth", "active_connections"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV exported from validated Parquet telemetry")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--h2o-url", default="http://h2o:54321")
    args = parser.parse_args()
    h2o.connect(url=args.h2o_url)
    frame = h2o.import_file(args.data)
    frame["failure_within_30min"] = frame["failure_within_30min"].asfactor()
    train, valid, test = frame.split_frame(ratios=[.6, .2], seed=20260821)
    model = H2ORandomForestEstimator(ntrees=200, max_depth=20, balance_classes=True, seed=20260821)
    model.train(x=FEATURES, y="failure_within_30min", training_frame=train, validation_frame=valid)
    perf = model.model_performance(test)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = h2o.save_model(model=model, path=str(args.model_dir), force=True)
    report = {"model_path": model_path, "test_aucpr": perf.aucpr(), "test_f1": perf.F1()[0][1] if perf.F1() else None}
    (args.model_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__": main()

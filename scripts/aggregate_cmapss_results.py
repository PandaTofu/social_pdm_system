"""Aggregate FD001-FD004 comparison outputs without recomputing model metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUBSETS = ("FD001", "FD002", "FD003", "FD004")
MODELS = ("logistic_regression", "random_forest", "gradient_boosting")
METRICS = ("precision", "recall", "f1", "pr_auc", "roc_auc", "threshold")


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "subset": report["subset"],
        "task": report["task"],
        "protocol": report["protocol"],
        "train_distribution": report["train_distribution"],
        "test_distribution": report["test_distribution"],
        "models": {
            name: {
                "display_name": report["models"][name]["display_name"],
                "operating_metrics": {
                    metric: report["models"][name]["operating_metrics"][metric]
                    for metric in METRICS
                },
            }
            for name in MODELS
        },
        "non_regression_check": {
            "passed": report["non_regression_check"]["passed"],
            "tolerance": report["non_regression_check"]["tolerance"],
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# NASA C-MAPSS FD001-FD004 分类比较",
        "",
        "统一协议：官方训练/测试划分、RUL <= 30 cycles、相同特征和请求阈值0.50。",
        "",
        "| Subset | Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Effective threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in SUBSETS:
        report = summary["subsets"][subset]
        for model in MODELS:
            entry = report["models"][model]
            metric = entry["operating_metrics"]
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    subset, entry["display_name"], metric["precision"], metric["recall"],
                    metric["f1"], metric["pr_auc"], metric["roc_auc"], metric["threshold"],
                )
            )
    lines.extend(["", "各子集分别评价；不将不同子集的轨迹行直接合并计算总F1。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate four persisted C-MAPSS comparison reports.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    reports: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for subset in SUBSETS:
        path = args.results_root / subset / "classification_comparison" / "comparison.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing comparison report for {subset}: {path}")
        reports[subset] = compact_report(json.loads(path.read_text(encoding="utf-8")))
        sources[subset] = str(path)
    summary = {
        "dataset": "NASA C-MAPSS",
        "subsets": reports,
        "source_reports": sources,
        "aggregation_rule": "Each subset is evaluated independently; no pooled F1 is reported.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cmapss_all_subsets.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out_dir / "cmapss_all_subsets.md").write_text(markdown(summary), encoding="utf-8")
    print(f"C-MAPSS summary written to {args.out_dir}")


if __name__ == "__main__":
    main()

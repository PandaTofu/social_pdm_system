"""Create an auditable diagnostic report for simulated telemetry results."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_evaluation(path: Path) -> dict[str, Any]:
    by_day: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_server: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    rows = positives = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = int(row["failure_within_30min"])
            rows += 1
            positives += label
            by_day[row["day"]][0] += 1
            by_day[row["day"]][1] += label
            by_server[row["server_id"]][0] += 1
            by_server[row["server_id"]][1] += label
    return {
        "rows": rows,
        "positive_rows": positives,
        "positive_rate": positives / rows if rows else 0.0,
        "by_day": {day: {"rows": value[0], "positives": value[1],
                         "positive_rate": value[1] / value[0] if value[0] else 0.0}
                   for day, value in sorted(by_day.items(), key=lambda item: int(item[0]))},
        "servers": len(by_server),
        "servers_without_positive_rows": sum(value[1] == 0 for value in by_server.values()),
    }


def variant_diagnostics(metrics: dict[str, Any], prevalence: float) -> dict[str, float]:
    matrix = metrics["confusion_matrix"]
    tn, fp, fn, tp = (int(matrix[name]) for name in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    alerts = fp + tp
    return {
        "threshold": float(metrics["threshold"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "pr_auc": float(metrics["pr_auc"]),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "alert_rate": alerts / total,
        "precision_lift_over_prevalence": float(metrics["precision"]) / prevalence,
        "false_alerts_per_true_alert": fp / tp if tp else float("inf"),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def analyze(report: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    prevalence = evaluation["positive_rate"]
    training = {}
    for name in ("pre_drift_train", "adaptive_train", "feedback_calibration", "evaluation"):
        values = report[name]
        rate = values["positive_rows"] / values["rows"]
        training[name] = {**values, "positive_rate": rate,
                          "prior_multiple_vs_evaluation": rate / prevalence}
    curve = report.get("adaptive_threshold_curve", [])
    oracle = max(curve, key=lambda row: row["f1"]) if curve else None
    config = report.get("generator_config") or {}
    return {
        "scenario": report.get("scenario", config.get("scenario", "unspecified")),
        "generator_contract": {
            "failure_horizon_minutes": config.get("failure_horizon_minutes"),
            "precursor_window_minutes": config.get("precursor_window_minutes"),
            "windows_aligned": (config.get("failure_horizon_minutes") == config.get("precursor_window_minutes")
                                if config else None),
            "concept_drift_strength": config.get("concept_drift_strength"),
        },
        "evaluation": evaluation,
        "training_and_calibration_priors": training,
        "variants": {name: variant_diagnostics(values, prevalence)
                     for name, values in report["variants"].items()},
        "post_hoc_evaluation_best_f1": oracle,
        "warning": "post_hoc_evaluation_best_f1 is diagnostic only and must not be used for threshold tuning",
    }


def markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    contract = result["generator_contract"]
    lines = ["# 模拟遥测数据诊断", "",
             f"- 场景：{result['scenario']}",
             f"- 评估记录：{evaluation['rows']:,}",
             f"- 正例：{evaluation['positive_rows']:,}（{evaluation['positive_rate']:.4%}）",
             f"- 逻辑服务器：{evaluation['servers']}；评估期无正例的服务器：{evaluation['servers_without_positive_rows']}", "",
             "## 模型告警诊断", "",
             "| 方案 | 阈值 | Precision | Recall | F1 | PR-AUC | FPR | 告警率 | 每个真告警对应的假告警 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in result["variants"].items():
        lines.append(f"| {name} | {values['threshold']:.4f} | {values['precision']:.4f} | "
                     f"{values['recall']:.4f} | {values['f1']:.4f} | {values['pr_auc']:.4f} | "
                     f"{values['false_positive_rate']:.4%} | {values['alert_rate']:.4%} | "
                     f"{values['false_alerts_per_true_alert']:.2f} |")
    lines.extend(["", "## 结论", "",
                  "1. Recall高而Precision低的首要原因是正例极稀少；很小的FPR也会产生远多于TP的FP。",
                  "2. 训练集只保留8%的负例，使训练先验显著高于真实评估先验；0.5概率阈值不再适合直接部署。",
                  ("3. 故障征兆窗口与标签窗口一致，已消除旧场景的90/30分钟结构性错配。"
                   if contract["windows_aligned"] else
                   "3. 故障征兆窗口与标签窗口不一致，窗口之外的相似负例会制造结构性假阳性。"),
                  "4. PR-AUC应与正例率基线一起解释；同时报告固定阈值下的告警负担。"])
    if result["scenario"] == "concept_drift_v2":
        lines.append("5. 本场景在部署后切换主要风险特征；是否成功证明自适应能力，应以同阈值重训练增益和完整消融结果判断。")
    else:
        lines.append("5. 当前场景主要用于协变量漂移；不要将其单独作为概念漂移证据。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    result = analyze(report, read_evaluation(args.evaluation_csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "simulated_data_diagnostics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "simulated_data_diagnostics.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()

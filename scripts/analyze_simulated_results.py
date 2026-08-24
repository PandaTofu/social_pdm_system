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
    return {
        "evaluation": evaluation,
        "training_and_calibration_priors": training,
        "variants": {name: variant_diagnostics(values, prevalence)
                     for name, values in report["variants"].items()},
        "post_hoc_evaluation_best_f1": oracle,
        "warning": "post_hoc_evaluation_best_f1 is diagnostic only and must not be used for threshold tuning",
    }


def markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = ["# 模拟遥测数据诊断", "",
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
                  "3. 生成器在故障前90分钟增强特征，但标签窗口只有30分钟；31–90分钟的负标签样本天然类似正例，会制造结构性假阳性。",
                  "4. PR-AUC应与正例率基线一起解释；本实验PR-AUC远高于随机基线，但固定阈值下的告警负担仍偏高。",
                  "5. 当前漂移主要是协变量漂移。若论文主张概念漂移，应让故障前后的特征—风险关系在部署后发生可控变化。", ""])
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

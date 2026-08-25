"""Compare persisted telemetry scenarios without mixing their test rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_spec(spec: str) -> tuple[str, dict[str, Any]]:
    label, separator, path = spec.partition("=")
    if not separator:
        raise ValueError("Scenario must use LABEL=JSON_PATH")
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def false_positive_rate(metrics: dict[str, Any]) -> float:
    matrix = metrics["confusion_matrix"]
    return matrix["fp"] / (matrix["fp"] + matrix["tn"])


def summarize(label: str, report: dict[str, Any]) -> dict[str, Any]:
    static = report["variants"]["static_rf"]
    weighted = report["variants"]["weighted_retraining"]
    full = report["variants"]["full_adaptive"]
    return {
        "label": label,
        "drift_detected": report["drift_detected"],
        "max_ks_d": report["drift_monitor"]["max_ks_d"],
        "evaluation_positive_rate": report["evaluation"]["positive_rows"] / report["evaluation"]["rows"],
        "static": {**static, "fpr": false_positive_rate(static)},
        "weighted_retraining": {**weighted, "fpr": false_positive_rate(weighted)},
        "full_adaptive": {**full, "fpr": false_positive_rate(full)},
        "gains": {
            "ranking_pr_auc_weighted_minus_static": weighted["pr_auc"] - static["pr_auc"],
            "threshold_f1_full_minus_weighted": full["f1"] - weighted["f1"],
            "end_to_end_f1_full_minus_static": full["f1"] - static["f1"],
        },
    }


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# 遥测场景对比", "",
             "不同场景使用各自独立测试集；本表用于验证场景行为，不把跨场景绝对分数当作同一数据集上的算法排名。", "",
             "| 场景 | 正例率 | KS max D | 静态F1 | 加权重训练F1 | 完整自适应F1 | 静态PR-AUC | 重训练PR-AUC | 完整FPR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {label} | {evaluation_positive_rate:.4%} | {max_ks_d:.4f} | "
                     "{static[f1]:.4f} | {weighted_retraining[f1]:.4f} | "
                     "{full_adaptive[f1]:.4f} | {static[pr_auc]:.4f} | "
                     "{weighted_retraining[pr_auc]:.4f} | {full_adaptive[fpr]:.4%} |".format(**row))
    lines.extend(["", "## 增益分解", "",
                  "- `重训练PR-AUC - 静态PR-AUC`衡量模型对新关系的排序适应。",
                  "- `完整F1 - 加权重训练F1`衡量独立阈值校准的额外贡献。",
                  "- v1主要是协变量漂移；v2显式改变部署后的特征—故障关系。", ""])
    for row in rows:
        gains = row["gains"]
        lines.append(f"- {row['label']}：排序增益={gains['ranking_pr_auc_weighted_minus_static']:+.4f}；"
                     f"阈值F1增益={gains['threshold_f1_full_minus_weighted']:+.4f}；"
                     f"端到端F1增益={gains['end_to_end_f1_full_minus_static']:+.4f}。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = (("f1", "F1"), ("pr_auc", "PR-AUC"),
               ("precision", "Precision"), ("recall", "Recall"))
    variants = (("static", "Static RF"), ("weighted_retraining", "Weighted retraining"),
                ("full_adaptive", "Full adaptive"))
    colors = ("#7F7F7F", "#4C78A8", "#59A14F")
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    x = np.arange(len(rows)); width = .24
    for axis, (metric, title) in zip(axes.flat, metrics):
        for index, ((variant, display), color) in enumerate(zip(variants, colors)):
            values = [row[variant][metric] for row in rows]
            bars = axis.bar(x + (index - 1) * width, values, width, label=display, color=color)
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=8)
        axis.set_xticks(x, [row["label"] for row in rows])
        axis.set(title=title, ylim=(0, 1.05))
        axis.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Held-out score")
    axes[1, 0].set_ylabel("Held-out score")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Scenario behavior: static, retrained and calibrated models")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", action="append", required=True,
                        help="Repeat LABEL=adaptive_comparison.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize(*load_spec(spec)) for spec in args.scenario]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenario_comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(rows, args.output_dir / "scenario_comparison.md")
    plot(rows, args.output_dir / "scenario_comparison")


if __name__ == "__main__":
    main()

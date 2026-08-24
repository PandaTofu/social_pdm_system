"""Render thesis-ready figures from completed telemetry experiments.

The figures are intentionally derived only from persisted JSON reports so the
reported test metrics can be traced to a completed, time-split experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_drift_figure(drift: dict, output: Path) -> None:
    series = drift["feature_ks_d"]
    names = list(series)
    values = [series[name] for name in names]
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    bars = axis.bar(names, values, color="#4C78A8")
    axis.axhline(drift["threshold"], color="#D62728", linestyle="--", label=f"Threshold = {drift['threshold']:.2f}")
    axis.set_ylabel("KS-D statistic")
    axis.set_title("Feature drift detection")
    axis.tick_params(axis="x", rotation=22)
    axis.set_ylim(0, max(max(values) * 1.25, drift["threshold"] * 1.25))
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_performance_figure(result: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    f1_names = ["Static RF\n(0.50)", "Adaptive RF\n(0.50)", "Adaptive RF\n(calibrated)"]
    f1_values = [result["static_f1_at_0_5"], result["adaptive_f1_at_0_5"], result["adaptive_f1_at_calibrated_threshold"]]
    bars = axes[0].bar(f1_names, f1_values, color=["#7F7F7F", "#E45756", "#59A14F"])
    axes[0].set_ylim(0, 0.65)
    axes[0].set_ylabel("F1 score")
    axes[0].set_title("Held-out F1 (days 6-7)")
    for bar, value in zip(bars, f1_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    auc_names = ["Static RF", "Adaptive RF"]
    auc_values = [result["static_aucpr"], result["adaptive_aucpr"]]
    bars = axes[1].bar(auc_names, auc_values, color=["#7F7F7F", "#59A14F"])
    axes[1].set_ylim(0, 0.45)
    axes[1].set_ylabel("Area under precision-recall curve")
    axes[1].set_title("Ranking quality on held-out days 6-7")
    for bar, value in zip(bars, auc_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_architecture_figure(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.0, 2.8))
    axis.axis("off")
    nodes = [
        (0.03, "Telemetry\ningestion"), (0.24, "Spark quality\nvalidation + dedupe"),
        (0.47, "KS drift\nmonitoring"), (0.69, "Feedback-weighted\nH2O RF retraining"),
        (0.90, "Calibrated\nalerts"),
    ]
    for position, label in nodes:
        patch = FancyBboxPatch((position, 0.34), 0.15, 0.32, boxstyle="round,pad=0.018", facecolor="#E8F1FA", edgecolor="#4C78A8", linewidth=1.3)
        axis.add_patch(patch)
        axis.text(position + 0.075, 0.50, label, ha="center", va="center", fontsize=10)
    for position, _ in nodes[:-1]:
        axis.annotate("", xy=(position + 0.205, 0.50), xytext=(position + 0.155, 0.50), arrowprops={"arrowstyle": "->", "color": "#4C78A8", "lw": 1.5})
    axis.text(0.56, 0.16, "Drift detected: use delayed day-5 feedback for retraining and threshold calibration", ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_table(result: dict, drift: dict, output: Path) -> None:
    text = f"""# 实验结果汇总\n\n| 指标 | 静态随机森林 | 自适应随机森林 |\n|---|---:|---:|\n| 留出集 F1（部署阈值） | {result['static_f1_at_0_5']:.4f} | {result['adaptive_f1_at_calibrated_threshold']:.4f} |\n| 留出集 AUC-PR | {result['static_aucpr']:.4f} | {result['adaptive_aucpr']:.4f} |\n| 部署阈值 | 0.5000 | {result['adaptive_operating_threshold']:.4f} |\n\n- KS 最大统计量：{drift['max_ks_d']:.4f}；阈值：{drift['threshold']:.2f}；漂移判定：是。\n- 时间切分：第 1–3 天初始训练；第 5 天反馈按事件 ID 进行 70/30 重训/校准划分；第 6–7 天完全留出评估。\n- 自适应模型的第 5 天样本权重为历史样本的 4 倍。\n- F1 的提升为 {result['f1_change_at_operating_threshold']:.4f}；AUC-PR 的提升为 {result['adaptive_aucpr'] - result['static_aucpr']:.4f}。\n"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--drift", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result, drift = read_json(args.result), read_json(args.drift)
    save_architecture_figure(args.out_dir / "fig1_system_architecture.png")
    save_drift_figure(drift, args.out_dir / "fig2_ks_drift_detection.png")
    save_performance_figure(result, args.out_dir / "fig3_model_comparison.png")
    save_table(result, drift, args.out_dir / "table1_experiment_results.md")
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()

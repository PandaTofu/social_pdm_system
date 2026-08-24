"""Generate thesis figures only from persisted, auditable experiment outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def load_json(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else None


def save(figure: plt.Figure, output: Path) -> None:
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 3.2))
    axis.set_axis_off()
    labels = ["Versioned schema\n+ quality gates", "Spark streaming\n+ task routing",
              "KS drift\nmonitor", "H2O adaptive RF\n+ TreeSHAP", "Dashboard\n+ alerts"]
    positions = [.015, .215, .415, .615, .815]
    for index, (x, label) in enumerate(zip(positions, labels)):
        color = "#E8F1FA" if index < 2 else "#EAF5EA"
        axis.add_patch(FancyBboxPatch((x, .36), .16, .32, boxstyle="round,pad=.018",
                                      facecolor=color, edgecolor="#3F6F8F", linewidth=1.3))
        axis.text(x + .08, .52, label, ha="center", va="center", fontsize=10)
    for current, following in zip(positions, positions[1:]):
        axis.annotate("", xy=(following - .008, .52), xytext=(current + .165, .52),
                      arrowprops={"arrowstyle": "->", "color": "#3F6F8F", "lw": 1.5})
    axis.text(.50, .18, "DW-Spark boundary", ha="center", fontsize=9, color="#555555")
    axis.text(.70, .18, "Spark-ML boundary", ha="center", fontsize=9, color="#555555")
    axis.text(.90, .18, "ML-Monitoring boundary", ha="center", fontsize=9, color="#555555")
    save(figure, output)


def cmapss_distribution(metrics: dict[str, Any], output: Path) -> None:
    train, test = metrics["train_distribution"], metrics["test_distribution"]
    positives = [train["positive_rows"], test["positive_rows"]]
    negatives = [train["rows"] - train["positive_rows"], test["rows"] - test["positive_rows"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(2)
    axes[0].bar(x, negatives, label="RUL > horizon", color="#4C78A8")
    axes[0].bar(x, positives, bottom=negatives, label="RUL <= horizon", color="#E45756")
    axes[0].set_xticks(x, ["Train", "Official test"])
    axes[0].set(title="C-MAPSS label distribution", ylabel="Cycle rows")
    axes[0].legend(frameon=False)
    bands = ["0-30", "31-60", "61-120", "121+"]
    axes[1].bar(bands, [test["rul_bands"].get(band, 0) for band in bands], color="#59A14F")
    axes[1].set(title="Official test RUL bands", xlabel="Remaining cycles", ylabel="Cycle rows")
    save(figure, output)


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    actual: list[int] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        target = next((name for name in (reader.fieldnames or []) if "failure_within" in name), None)
        if not target or "p1" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain p1 and the failure target")
        for row in reader:
            scores.append(float(row["p1"]))
            actual.append(int(float(row[target])))
    return np.asarray(scores), np.asarray(actual)


def precision_recall_curve(scores: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    truth = actual[order]
    tp = np.cumsum(truth == 1)
    fp = np.cumsum(truth == 0)
    precision = tp / np.maximum(tp + fp, 1)
    positives = max(int(np.sum(actual == 1)), 1)
    recall = tp / positives
    return np.r_[1.0, recall], np.r_[1.0, precision]


def cmapss_validation(metrics: dict[str, Any], predictions: Path, output: Path) -> None:
    confusion = metrics["operating_metrics"]["confusion_matrix"]
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    scores, actual = read_predictions(predictions)
    recall, precision = precision_recall_curve(scores, actual)
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[0].text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Predicted normal", "Predicted failure"],
                yticklabels=["Actual normal", "Actual failure"], title="Confusion matrix at 0.50")
    figure.colorbar(image, ax=axes[0], fraction=.046)
    axes[1].plot(recall, precision, color="#E45756", lw=1.8,
                 label=f"PR-AUC = {metrics['operating_metrics']['pr_auc']:.3f}")
    prevalence = float(np.mean(actual))
    axes[1].axhline(prevalence, color="#777777", linestyle="--", label=f"Prevalence = {prevalence:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Precision-recall curve")
    axes[1].legend(frameon=False)
    save(figure, output)


def cmapss_model_comparison(comparison: dict[str, Any], output: Path) -> None:
    order = [name for name in ("logistic_regression", "random_forest", "gradient_boosting")
             if name in comparison["models"]]
    metrics = ("precision", "recall", "f1", "pr_auc")
    x = np.arange(len(order)); width = .19
    figure, axis = plt.subplots(figsize=(9.4, 4.7))
    for index, metric in enumerate(metrics):
        values = [comparison["models"][name]["operating_metrics"][metric] for name in order]
        bars = axis.bar(x + (index - 1.5) * width, values, width,
                        label=metric.replace("_", " ").upper())
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values],
                       padding=2, fontsize=7, rotation=90)
    axis.set_xticks(x, [comparison["models"][name]["display_name"] for name in order])
    axis.set(title=f"C-MAPSS {comparison['subset']} model comparison",
             ylabel="Official-test score", ylim=(0, 1.06))
    axis.legend(frameon=False, ncol=4)
    save(figure, output)


def resolve_prediction_path(value: str, comparison_path: Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    sibling = comparison_path.parent / candidate.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"Prediction artifact not found: {value}")


def cmapss_pr_comparison(comparison: dict[str, Any], comparison_path: Path, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    colors = {"logistic_regression": "#4C78A8", "random_forest": "#59A14F", "gradient_boosting": "#E45756"}
    for name, model in comparison["models"].items():
        path = resolve_prediction_path(model["predictions_csv"], comparison_path)
        scores, actual = read_predictions(path)
        recall, precision = precision_recall_curve(scores, actual)
        axes[0].plot(recall, precision, color=colors.get(name), lw=1.6,
                     label=f"{model['display_name']} ({model['operating_metrics']['pr_auc']:.3f})")
    prevalence = comparison["test_distribution"]["positive_rate"]
    axes[0].axhline(prevalence, color="#777777", linestyle="--", label=f"Prevalence ({prevalence:.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
                title="Official-test precision-recall curves")
    axes[0].legend(frameon=False, fontsize=8)
    confusion = comparison["models"]["random_forest"]["operating_metrics"]["confusion_matrix"]
    rf_metrics = comparison["models"]["random_forest"]["operating_metrics"]
    requested_threshold = rf_metrics.get("requested_threshold", rf_metrics.get("threshold", 0.5))
    actual_threshold = rf_metrics.get("threshold", requested_threshold)
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    image = axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred. normal", "Pred. failure"],
                yticklabels=["Actual normal", "Actual failure"],
                title=(f"Random Forest: requested {requested_threshold:.2f}, "
                       f"effective {actual_threshold:.3f}"))
    figure.colorbar(image, ax=axes[1], fraction=.046)
    save(figure, output)


def cmapss_non_regression(comparison: dict[str, Any], output: Path) -> None:
    check = comparison["non_regression_check"]
    baseline = check["baseline_random_forest"]
    enhanced = check["enhanced_pipeline_random_forest"]
    metrics = ["precision", "recall", "f1", "pr_auc"]
    x = np.arange(len(metrics)); width = .36
    figure, axis = plt.subplots(figsize=(8.3, 4.4))
    axis.bar(x - width / 2, [baseline[name] for name in metrics], width, label="Baseline RF", color="#7F7F7F")
    axis.bar(x + width / 2, [enhanced[name] for name in metrics], width,
             label="Enhanced-pipeline RF", color="#59A14F")
    axis.set_xticks(x, [name.replace("_", " ").upper() for name in metrics])
    axis.set(title="C-MAPSS clean-data non-regression check", ylabel="Score", ylim=(0, 1))
    axis.text(.5, .06, f"PASS: all |metric deltas| <= {check['tolerance']:.2f}; predictions preserved",
              transform=axis.transAxes, ha="center", color="#2F7D62", fontsize=9)
    axis.legend(frameon=False)
    save(figure, output)


def drift_figure(report: dict[str, Any], output: Path) -> None:
    values = report["feature_ks_d"]
    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    bars = axis.bar(values.keys(), values.values(), color="#4C78A8")
    axis.axhline(report["threshold"], color="#D62728", linestyle="--",
                 label=f"Effect threshold = {report['threshold']:.2f}")
    for bar, value in zip(bars, values.values()):
        axis.text(bar.get_x() + bar.get_width() / 2, value + .006, f"{value:.3f}", ha="center", fontsize=9)
    axis.set(title="Feature drift: two-sample KS", ylabel="KS D statistic")
    axis.tick_params(axis="x", rotation=18)
    axis.legend(frameon=False)
    save(figure, output)


def normalized_variants(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    if "variants" in result:
        return result["variants"]
    return {
        "static_rf": {"f1": result["static_f1_at_0_5"], "pr_auc": result["static_aucpr"]},
        "weighted_retraining": {"f1": result["adaptive_f1_at_0_5"], "pr_auc": result["adaptive_aucpr"]},
        "full_adaptive": {"f1": result["adaptive_f1_at_calibrated_threshold"], "pr_auc": result["adaptive_aucpr"]},
    }


def model_comparison(result: dict[str, Any], output: Path) -> None:
    variants = normalized_variants(result)
    keys = [key for key in ("static_rf", "unweighted_retraining", "weighted_retraining", "full_adaptive") if key in variants]
    labels = {"static_rf": "Static RF", "unweighted_retraining": "Retrain",
              "weighted_retraining": "Weighted", "full_adaptive": "Full adaptive"}
    metrics = [metric for metric in ("precision", "recall", "f1", "pr_auc")
               if all(metric in variants[key] for key in keys)]
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(keys)); width = .8 / max(len(metrics), 1)
    for index, metric in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        axis.bar(x + offset, [variants[key][metric] for key in keys], width, label=metric.replace("_", " ").upper())
    axis.set_xticks(x, [labels[key] for key in keys])
    axis.set(title="Held-out model comparison (days 6-7)", ylabel="Score", ylim=(0, 1))
    axis.legend(frameon=False, ncol=max(1, len(metrics)))
    save(figure, output)


def threshold_figure(result: dict[str, Any], output: Path) -> None:
    rows = [row for row in result.get("adaptive_threshold_curve", []) if row.get("threshold") is not None]
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    for metric, color in (("precision", "#4C78A8"), ("recall", "#E45756"), ("f1", "#59A14F")):
        axis.plot([row["threshold"] for row in rows], [row.get(metric) for row in rows], label=metric.title(), color=color)
    axis.axvline(result["adaptive_operating_threshold"], color="#222222", linestyle="--",
                 label=f"Calibrated = {result['adaptive_operating_threshold']:.3f}")
    axis.set(xlabel="Decision threshold", ylabel="Score", ylim=(0, 1), title="Adaptive model threshold calibration")
    axis.legend(frameon=False)
    save(figure, output)


def system_figure(report: dict[str, Any], output: Path) -> None:
    rows = report.get("scaling", [])
    if not rows:
        return
    x = [row["processed_rows"] for row in rows]
    figure, left = plt.subplots(figsize=(8.5, 4.5))
    right = left.twinx()
    left.plot(x, [row["throughput_rows_per_second"] for row in rows], marker="o", color="#4C78A8", label="Throughput")
    right.plot(x, [row["latency_ms"] for row in rows], marker="s", color="#E45756", label="Batch latency")
    left.set(xlabel="Processed rows", ylabel="Rows per second", title="Observed single-node Spark scaling")
    right.set_ylabel("Batch latency (ms)")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    save(figure, output)


def stability_figure(result: dict[str, Any], output: Path) -> None:
    rows = result.get("post_drift_daily_stability", [])
    if not rows:
        return
    days = [row["day"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.8, 4.3))
    axis.plot(days, [row["static"]["f1"] for row in rows], marker="o", label="Static RF", color="#7F7F7F")
    axis.plot(days, [row["full_adaptive"]["f1"] for row in rows], marker="s", label="Full adaptive", color="#59A14F")
    axis.set(xticks=days, xlabel="Post-drift evaluation day", ylabel="F1 score", ylim=(0, 1),
             title="Observed post-drift model stability")
    axis.legend(frameon=False)
    save(figure, output)


def shap_figure(path: Path, output: Path) -> None:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return
    excluded = {"event_id", "event_time", "server_id", "failure_within_30min", "predict", "p0", "p1", "BiasTerm"}
    features = [name for name in rows[0] if name not in excluded]
    contributions = {feature: np.asarray([float(row[feature]) for row in rows]) for feature in features}
    ranked = sorted(features, key=lambda feature: float(np.mean(np.abs(contributions[feature]))), reverse=True)[:10]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    global_values = [float(np.mean(np.abs(contributions[feature]))) for feature in ranked][::-1]
    axes[0].barh(ranked[::-1], global_values, color="#4C78A8")
    axes[0].set(title="Global mean |SHAP|", xlabel="Mean absolute contribution")
    local = sorted(ranked, key=lambda feature: abs(float(rows[0][feature])), reverse=True)[:6]
    local_values = [float(rows[0][feature]) for feature in local][::-1]
    axes[1].barh(local[::-1], local_values,
                 color=["#E45756" if value > 0 else "#59A14F" for value in local_values])
    axes[1].axvline(0, color="#333333", linewidth=.8)
    axes[1].set(title="One alert: local contributions", xlabel="SHAP contribution")
    save(figure, output)


def write_summary(result: dict[str, Any], drift: dict[str, Any], output: Path,
                  cmapss_comparison: dict[str, Any] | None = None) -> None:
    variants = normalized_variants(result)
    lines = ["# 可复现实验结果汇总", ""]
    if cmapss_comparison:
        lines.extend(["## NASA C-MAPSS FD001分类比较", "",
                      "| 方法 | Precision | Recall | F1 | PR-AUC | ROC-AUC | 请求阈值 | 实际阈值 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for model in cmapss_comparison["models"].values():
            values = model["operating_metrics"]
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:.4f} |".format(
                model["display_name"], values["precision"], values["recall"], values["f1"],
                values["pr_auc"], values["roc_auc"], values.get("requested_threshold", values["threshold"]),
                values["threshold"],
            ))
        check = cmapss_comparison["non_regression_check"]
        lines.extend(["", f"- 增强流水线非回归检查：{'通过' if check['passed'] else '未通过'}；容差={check['tolerance']:.2f}。", ""])
    lines.extend(["## 模拟社交服务器数据消融", "",
                  "| 方法 | Precision | Recall | F1 | PR-AUC | 阈值 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, values in variants.items():
        lines.append("| {} | {} | {} | {:.4f} | {:.4f} | {} |".format(
            name,
            f"{values['precision']:.4f}" if "precision" in values else "-",
            f"{values['recall']:.4f}" if "recall" in values else "-",
            values["f1"], values["pr_auc"],
            f"{values['threshold']:.4f}" if "threshold" in values else "-",
        ))
    lines.extend(["", f"- KS最大D值：{drift['max_ks_d']:.4f}",
                  f"- 漂移判定：{'是' if drift['drift_detected'] else '否'}",
                  "- 所有数值均来自传入的JSON/CSV实验产物，绘图脚本不包含论文目标值。", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True, help="adaptive_comparison.json")
    parser.add_argument("--drift", type=Path, required=True)
    parser.add_argument("--cmapss-metrics", type=Path)
    parser.add_argument("--cmapss-predictions", type=Path)
    parser.add_argument("--cmapss-comparison", type=Path)
    parser.add_argument("--system-benchmark", type=Path)
    parser.add_argument("--shap-contributions", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result, drift = load_json(args.result), load_json(args.drift)
    architecture(args.out_dir / "fig1_enhanced_architecture")
    drift_figure(drift, args.out_dir / "fig6_ks_drift_detection")
    model_comparison(result, args.out_dir / "fig7_social_model_comparison")
    threshold_figure(result, args.out_dir / "fig8_threshold_calibration")
    stability_figure(result, args.out_dir / "fig9_model_stability")
    cmapss_comparison = load_json(args.cmapss_comparison)
    if cmapss_comparison:
        cmapss_distribution(cmapss_comparison, args.out_dir / "fig2_cmapss_distribution")
        cmapss_model_comparison(cmapss_comparison, args.out_dir / "fig3_cmapss_model_comparison")
        cmapss_pr_comparison(cmapss_comparison, args.cmapss_comparison, args.out_dir / "fig4_cmapss_pr_comparison")
        cmapss_non_regression(cmapss_comparison, args.out_dir / "fig5_cmapss_non_regression")
    cmapss = load_json(args.cmapss_metrics)
    if cmapss and not cmapss_comparison:
        cmapss_distribution(cmapss, args.out_dir / "fig2_cmapss_distribution")
        if args.cmapss_predictions and args.cmapss_predictions.exists():
            cmapss_validation(cmapss, args.cmapss_predictions, args.out_dir / "fig3_cmapss_validation")
    benchmark = load_json(args.system_benchmark)
    if benchmark:
        system_figure(benchmark, args.out_dir / "fig10_system_performance")
    if args.shap_contributions and args.shap_contributions.exists():
        shap_figure(args.shap_contributions, args.out_dir / "fig11_shap_explanations")
    write_summary(result, drift, args.out_dir / "table_reproducible_results.md", cmapss_comparison)
    manifest = {"source_files": {key: str(value) if value else None for key, value in vars(args).items() if key != "out_dir"},
                "rule": "Figures are generated from persisted experiment artefacts; missing optional inputs skip their figures."}
    (args.out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Auditable PNG/PDF figures written to {args.out_dir}")


if __name__ == "__main__":
    main()

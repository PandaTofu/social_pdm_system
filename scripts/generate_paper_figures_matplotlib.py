"""Generate 300-dpi PNG and vector PDF figures from persisted experiment JSON.

Example:
python scripts/generate_paper_figures_matplotlib.py \
  --result data/autodl_runtime/output/adaptive_experiment_weighted_rerun/adaptive_comparison.json \
  --drift data/autodl_runtime/output/drift_report.json \
  --out-dir reports/paper_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_label(result: dict) -> str:
    window = result.get("experiment_windows", {}).get("evaluation")
    if isinstance(window, list) and len(window) == 2:
        return f"days {window[0]}-{window[1]}"
    return "held-out window"


def save_both(figure: plt.Figure, stem: Path) -> None:
    figure.tight_layout()
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def drift_chart(drift: dict, stem: Path) -> None:
    values = drift["feature_ks_d"]
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    bars = axis.bar(values.keys(), values.values(), color="#4C78A8")
    axis.axhline(drift["threshold"], color="#D62728", linestyle="--", linewidth=1.5,
                 label=f"threshold = {drift['threshold']:.2f}")
    axis.set_title("Feature drift detection")
    axis.set_ylabel("KS-D statistic")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(frameon=False)
    for bar, value in zip(bars, values.values()):
        axis.text(bar.get_x() + bar.get_width() / 2, value + .006, f"{value:.3f}",
                  ha="center", va="bottom", fontsize=9)
    save_both(figure, stem)


def performance_chart(result: dict, stem: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    f1_names = ["Static RF\n(0.50)", "Adaptive RF\n(0.50)", "Adaptive RF\n(calibrated)"]
    f1_values = [result["static_f1_at_0_5"], result["adaptive_f1_at_0_5"],
                 result["adaptive_f1_at_calibrated_threshold"]]
    bars = axes[0].bar(f1_names, f1_values, color=["#7F7F7F", "#E45756", "#59A14F"])
    label = evaluation_label(result)
    axes[0].set(title=f"Held-out F1 ({label})", ylabel="F1 score", ylim=(0, .65))
    for bar, value in zip(bars, f1_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + .015, f"{value:.3f}", ha="center", fontsize=9)
    auc_values = [result["static_aucpr"], result["adaptive_aucpr"]]
    bars = axes[1].bar(["Static RF", "Adaptive RF"], auc_values, color=["#7F7F7F", "#59A14F"])
    axes[1].set(title=f"Ranking quality ({label})", ylabel="AUC-PR", ylim=(0, .45))
    for bar, value in zip(bars, auc_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + .012, f"{value:.3f}", ha="center", fontsize=9)
    save_both(figure, stem)


def architecture_chart(stem: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 2.8))
    axis.set_axis_off()
    labels = ["Telemetry\ningestion", "Spark quality\nvalidation + dedupe", "KS drift\nmonitoring",
              "Feedback-weighted\nH2O RF retraining", "Calibrated\nalerts"]
    positions = [.02, .22, .42, .62, .82]
    for x, label in zip(positions, labels):
        axis.add_patch(FancyBboxPatch((x, .34), .15, .32, boxstyle="round,pad=.018",
                                      facecolor="#E8F1FA", edgecolor="#4C78A8", linewidth=1.3))
        axis.text(x + .075, .50, label, ha="center", va="center", fontsize=10)
    for current, following in zip(positions, positions[1:]):
        axis.annotate("", xy=(following - .01, .50), xytext=(current + .155, .50),
                      arrowprops={"arrowstyle": "->", "color": "#4C78A8", "lw": 1.5})
    axis.text(.56, .16, "Drift detected: delayed feedback drives retraining and threshold calibration",
              ha="center", fontsize=9)
    save_both(figure, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--drift", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result, drift = load(args.result), load(args.drift)
    architecture_chart(args.out_dir / "fig1_system_architecture")
    drift_chart(drift, args.out_dir / "fig2_ks_drift_detection")
    performance_chart(result, args.out_dir / "fig3_model_comparison")
    print(f"PNG/PDF figures written to {args.out_dir}")


if __name__ == "__main__":
    main()

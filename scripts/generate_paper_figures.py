"""Render thesis-ready figures from completed telemetry experiments.

The figures are intentionally derived only from persisted JSON reports so the
reported test metrics can be traced to a completed, time-split experiment.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:20px;font-weight:bold}.label{font-size:14px}.small{font-size:12px}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1}.box{fill:#e8f1fa;stroke:#4c78a8;stroke-width:1.5}</style>']
    svg.extend(body)
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def save_drift_figure(drift: dict, output: Path) -> None:
    series = drift["feature_ks_d"]
    names = list(series)
    values = [series[name] for name in names]
    maximum, left, top, chart_height = max(max(values) * 1.25, drift["threshold"] * 1.25), 75, 55, 270
    body = ['<text x="300" y="28" text-anchor="middle" class="title">Feature drift detection</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>',
            f'<line x1="{left}" y1="{top + chart_height}" x2="570" y2="{top + chart_height}" class="axis"/>']
    for tick in (0, maximum / 2, maximum):
        y = top + chart_height - tick / maximum * chart_height
        body += [f'<line x1="{left}" y1="{y:.1f}" x2="570" y2="{y:.1f}" class="grid"/>', f'<text x="66" y="{y + 4:.1f}" text-anchor="end" class="small">{tick:.2f}</text>']
    threshold_y = top + chart_height - drift["threshold"] / maximum * chart_height
    body += [f'<line x1="{left}" y1="{threshold_y:.1f}" x2="570" y2="{threshold_y:.1f}" stroke="#d62728" stroke-width="2" stroke-dasharray="6 4"/>',
             f'<text x="566" y="{threshold_y - 7:.1f}" text-anchor="end" class="small">threshold = {drift["threshold"]:.2f}</text>']
    for index, (name, value) in enumerate(zip(names, values)):
        x, width, height = 95 + index * 96, 52, value / maximum * chart_height
        y = top + chart_height - height
        body += [f'<rect x="{x}" y="{y:.1f}" width="{width}" height="{height:.1f}" fill="#4c78a8"/>',
                 f'<text x="{x + width / 2}" y="{y - 7:.1f}" text-anchor="middle" class="small">{value:.3f}</text>',
                 f'<text x="{x + width / 2}" y="{top + chart_height + 18}" text-anchor="middle" class="small">{html.escape(name)}</text>']
    body.append('<text x="22" y="200" text-anchor="middle" class="label" transform="rotate(-90 22 200)">KS-D statistic</text>')
    write_svg(output, 620, 385, body)


def save_performance_figure(result: dict, output: Path) -> None:
    f1_names = ["Static RF (0.50)", "Adaptive RF (0.50)", "Adaptive RF (calibrated)"]
    f1_values = [result["static_f1_at_0_5"], result["adaptive_f1_at_0_5"], result["adaptive_f1_at_calibrated_threshold"]]
    auc_values = [result["static_aucpr"], result["adaptive_aucpr"]]
    body = ['<text x="450" y="28" text-anchor="middle" class="title">Held-out performance comparison (days 6-7)</text>']
    panels = [(60, 390, "F1 score", f1_names, f1_values, ["#7f7f7f", "#e45756", "#59a14f"], 0.65),
              (500, 810, "AUC-PR", ["Static RF", "Adaptive RF"], auc_values, ["#7f7f7f", "#59a14f"], 0.45)]
    for start, end, title, names, values, colors, maximum in panels:
        base, top = 325, 80
        body += [f'<text x="{(start + end) / 2}" y="55" text-anchor="middle" class="label">{title}</text>',
                 f'<line x1="{start}" y1="{top}" x2="{start}" y2="{base}" class="axis"/>', f'<line x1="{start}" y1="{base}" x2="{end}" y2="{base}" class="axis"/>']
        for index, (name, value, color) in enumerate(zip(names, values, colors)):
            width = 54; step = (end - start - 60) / len(names); x = start + 32 + index * step
            height = value / maximum * (base - top); y = base - height
            body += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{width}" height="{height:.1f}" fill="{color}"/>', f'<text x="{x + width / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" class="small">{value:.3f}</text>', f'<text x="{x + width / 2:.1f}" y="{base + 18}" text-anchor="middle" class="small">{html.escape(name)}</text>']
    write_svg(output, 875, 385, body)


def save_architecture_figure(output: Path) -> None:
    body = ['<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#4c78a8"/></marker></defs>']
    nodes = [
        (0.03, "Telemetry\ningestion"), (0.24, "Spark quality\nvalidation + dedupe"),
        (0.47, "KS drift\nmonitoring"), (0.69, "Feedback-weighted\nH2O RF retraining"),
        (0.90, "Calibrated\nalerts"),
    ]
    for position, label in nodes:
        x = int(position * 1000)
        body += [f'<rect x="{x}" y="85" width="150" height="90" rx="8" class="box"/>']
        for line_index, line in enumerate(label.split("\n")):
            body.append(f'<text x="{x + 75}" y="{122 + line_index * 20}" text-anchor="middle" class="label">{html.escape(line)}</text>')
    for position, _ in nodes[:-1]:
        x = int(position * 1000)
        body.append(f'<line x1="{x + 150}" y1="130" x2="{x + 205}" y2="130" stroke="#4c78a8" stroke-width="2" marker-end="url(#arrow)"/>')
    body += ['<text x="560" y="225" text-anchor="middle" class="label">Drift detected: use delayed day-5 feedback for retraining and threshold calibration</text>']
    write_svg(output, 1080, 255, body)


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
    save_architecture_figure(args.out_dir / "fig1_system_architecture.svg")
    save_drift_figure(drift, args.out_dir / "fig2_ks_drift_detection.svg")
    save_performance_figure(result, args.out_dir / "fig3_model_comparison.svg")
    save_table(result, drift, args.out_dir / "table1_experiment_results.md")
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()

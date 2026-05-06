"""Plot helpers for metrics.json.

This module currently focuses on a bar chart that shows how many accepted
references use each metric.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_metrics(metrics_path: Path) -> list[dict]:
	with metrics_path.open("r", encoding="utf-8") as file_handle:
		return json.load(file_handle)


def count_used_references(metrics: Sequence[dict]) -> list[dict[str, object]]:
	summarized_metrics = []

	for metric in metrics:
		used_in_ref = metric.get("used_in_ref") or []
		summarized_metrics.append(
			{
				"id": metric.get("id"),
				"name": metric.get("name", "Unknown metric"),
				"type": metric.get("type") or "Unknown",
				"reference_count": len(used_in_ref),
			}
		)

	summarized_metrics.sort(
		key=lambda item: (-int(item["reference_count"]), str(item["name"]))
	)
	return summarized_metrics


def plot_metric_reference_counts(
	metrics: Sequence[dict],
	output_path: Path,
	title: str = "References used by metric",
	top_n: int | None = None,
) -> Path:
	color_map = {
		"Execution-based": "#2F6FDB",
		"String-based": "#F28E2B",
		"Triple-based": "#3EA055",
		"Time-based": "#9B4E8F",
	}
	summarized_metrics = count_used_references(metrics)

	if top_n is not None:
		summarized_metrics = summarized_metrics[:top_n]

	if not summarized_metrics:
		raise ValueError("No metrics found to plot.")

	names = [str(item["name"]) for item in summarized_metrics]
	counts = [int(item["reference_count"]) for item in summarized_metrics]
	colors = [color_map.get(str(item["type"]), "#9AA0A6") for item in summarized_metrics]

	figure_height = max(6, 0.4 * len(names))
	figure, axis = plt.subplots(figsize=(12, figure_height))
	bars = axis.barh(names, counts, color=colors)
	axis.invert_yaxis()
	axis.set_xlabel("Number of references")
	axis.set_ylabel("Metric")
	axis.set_title(title)
	axis.grid(axis="x", linestyle="--", alpha=0.3)
	axis.legend(
		handles=[
			Patch(facecolor=color_map["Execution-based"], label="Execution-based"),
			Patch(facecolor=color_map["String-based"], label="String-based"),
			Patch(facecolor=color_map["Triple-based"], label="Triple-based"),
			Patch(facecolor=color_map["Time-based"], label="Time-based"),
		],
		loc="lower right",
		title="Metric type",
	)

	for bar, count in zip(bars, counts, strict=False):
		axis.text(
			bar.get_width() + 0.15,
			bar.get_y() + bar.get_height() / 2,
			str(count),
			va="center",
			ha="left",
			fontsize=9,
		)

	figure.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(figure)
	return output_path


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Plot metrics from metrics.json")
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("metrics.json"),
		help="Path to the metrics JSON file.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("plots/metric_reference_counts.png"),
		help="Path where the plot will be written.",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=None,
		help="Plot only the top N metrics by reference count.",
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	metrics = load_metrics(args.input)
	output_path = plot_metric_reference_counts(
		metrics=metrics,
		output_path=args.output,
		top_n=args.top_n,
	)
	print(f"Saved plot to {output_path}")


if __name__ == "__main__":
	main()

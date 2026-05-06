"""Correlation plots and feature-closeness scoring for metrics result files.

This script provides a simple CLI that accepts one or more results JSON files
and prints correlation matrices and feature-closeness scores to the terminal.
It also renders and (optionally) saves clustered heatmap plots.
"""

import json
import os
import sys
from argparse import ArgumentParser
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _load_json_records(results_path: str) -> list[dict]:
    """Load a JSON results file and return the raw records.

    The function supports both a single JSON array and newline-delimited JSON (jsonl).
    """
    with open(results_path, encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        try:
            return json.loads(text)
        except Exception:
            # try jsonlines
            f.seek(0)
            records = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records


def _filter_valid_metrics(records: list[dict]) -> list[dict]:
    """Keep only records that contain a non-empty metrics payload."""
    return [item for item in records if item.get("metrics")]


def _build_system_correlation_from_records(records: list[dict]) -> pd.DataFrame:
    """Build a system-by-system correlation matrix from metrics records."""
    frame = pd.DataFrame(
        [
            {"system_name": item.get("system_name", f"sys_{i}"), **item["metrics"]}
            for i, item in enumerate(records)
        ]
    )
    if frame.empty:
        raise ValueError("No valid metrics found in records")
    frame.set_index("system_name", inplace=True)
    return frame.T.corr().fillna(0)


def _build_metric_correlation_from_records(records: list[dict]) -> pd.DataFrame:
    """Build a metric-by-metric correlation matrix from metrics records."""
    frame = pd.DataFrame([item["metrics"] for item in records]).fillna(0)
    if frame.empty:
        raise ValueError("No valid metrics found in records")
    print(f"File shape: {frame.shape} (systems × metrics)")
    print(f"\nFile metrics: {sorted(frame.columns.tolist())}")
    return frame.corr().fillna(0)


def _plot_clustermap(
    correlation: pd.DataFrame, *, figsize: tuple[int, int] = (12, 12)
) -> plt.Figure:
    """Render a clustered heatmap for a correlation matrix and return the figure."""
    grid = sns.clustermap(
        correlation,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        figsize=figsize,
    )
    grid.fig.tight_layout()
    return grid.fig


def _compute_feature_closeness_score(correlation: pd.DataFrame) -> pd.Series:
    """Score features by how close their correlation profiles are to each other."""
    correlation = correlation.fillna(0)
    corr_array = correlation.to_numpy()

    diff_matrix = np.abs(corr_array[:, None, :] - corr_array[None, :, :])
    scores = diff_matrix.mean(axis=(1, 2))

    return pd.Series(scores, index=correlation.index).sort_values()


def _save_figure_for_input(
    fig: plt.Figure, save_target: str | None, input_path: str, suffix: str
) -> None:
    if save_target is None:
        return
    # If save_target is a directory, place file inside it with input basename
    if os.path.isdir(save_target):
        base = os.path.splitext(os.path.basename(input_path))[0]
        out = os.path.join(save_target, f"{base}{suffix}")
    else:
        # save_target is a path. If multiple inputs, inject input basename before extension
        if (
            isinstance(save_target, str)
            and save_target
            and len(save_target) > 0
            and os.pathsep not in save_target
        ):
            root, ext = os.path.splitext(save_target)
            # If the root ends with a path sep, treat as directory
            if root.endswith(os.sep):
                os.makedirs(save_target, exist_ok=True)
                base = os.path.splitext(os.path.basename(input_path))[0]
                out = os.path.join(save_target, f"{base}{suffix}")
            else:
                # if user provided a single filename but we have multiple inputs, insert basename
                out = (
                    f"{root}_{os.path.splitext(os.path.basename(input_path))[0]}{suffix if suffix.startswith('.') else '.' + suffix.lstrip('.')}"
                    if len(sys.argv) > 2
                    else save_target
                )
        else:
            out = save_target

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Read one or more metrics results JSON files and produce:\n"
            " - system correlation clustered heatmap (-os)\n"
            " - metric correlation clustered heatmap (-om)\n"
            " - feature closeness score CSV (-oc)\n"
            "All results are printed to terminal and plots are shown interactively."
        )
    )

    parser.add_argument(
        "-r",
        "--results",
        nargs="+",
        required=True,
        help="One or more paths to results JSON/jsonl files.",
    )

    parser.add_argument(
        "-os",
        "--out-system-plot",
        type=str,
        default=None,
        help="Optional output file (or directory) for system correlation plot(s).",
    )

    parser.add_argument(
        "-om",
        "--out-metric-plot",
        type=str,
        default=None,
        help="Optional output file (or directory) for metric correlation plot(s).",
    )

    parser.add_argument(
        "-oc",
        "--out-closeness",
        type=str,
        default=None,
        help="Optional output file for closeness scores (CSV or .csp).",
    )

    args = parser.parse_args()

    inputs = args.results

    for results_path in inputs:
        print(f"\n=== Processing: {results_path} ===\n")
        records = _filter_valid_metrics(_load_json_records(results_path))
        if not records:
            print(f"No valid records with 'metrics' found in {results_path}")
            continue

        # Metric correlation (metric × metric)
        metric_corr = _build_metric_correlation_from_records(records)
        print("Metric correlation matrix:\n", metric_corr, "\n")
        fig_metric = _plot_clustermap(metric_corr, figsize=(18, 18))
        # Save metric plot if requested
        _save_figure_for_input(
            fig_metric, args.out_metric_plot, results_path, "_metric_corr.png"
        )
        plt.show()

        # System correlation (system × system)
        system_corr = _build_system_correlation_from_records(records)
        print("System correlation matrix:\n", system_corr, "\n")
        fig_system = _plot_clustermap(system_corr, figsize=(12, 12))
        _save_figure_for_input(
            fig_system, args.out_system_plot, results_path, "_system_corr.png"
        )
        plt.show()

        # Feature closeness (from metric correlation)
        closeness = _compute_feature_closeness_score(metric_corr)
        print("Feature closeness scores (lower = closer):\n", closeness, "\n")
        if args.out_closeness:
            out_target = args.out_closeness
            if os.path.isdir(out_target) or out_target.endswith(os.sep):
                out_dir = out_target
                os.makedirs(out_dir, exist_ok=True)
                out = os.path.join(
                    out_dir,
                    f"{os.path.splitext(os.path.basename(results_path))[0]}.csv",
                )
            else:
                if len(inputs) > 1:
                    root, ext = os.path.splitext(out_target)
                    ext = ext or ".csv"
                    out = f"{root}_{os.path.splitext(os.path.basename(results_path))[0]}{ext}"
                else:
                    out = out_target

            makedirs(os.path.dirname(out) or ".", exist_ok=True)
            closeness.to_csv(out, header=["feature_closeness_score"])


if __name__ == "__main__":
    main()

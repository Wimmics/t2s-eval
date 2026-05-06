from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASET_RUNS: list[tuple[str, str, int]] = [
    ("ck25", "http://localhost:8886/", 0),
    ("ck26", "http://localhost:8870/", 0),
    ("db25", "http://localhost:8887/", 25000),
    ("db26", "http://localhost:8871/", 25000),
]

PROXY_METRICS = ["answerset_f1", "exact_match_spinach"]


def core():
    return importlib.import_module("pipeline_core")


def latest_result_file(dataset_dir: Path) -> Path:
    result_files = sorted(
        dataset_dir.glob("results/*.json"), key=lambda path: path.stat().st_mtime
    )
    if not result_files:
        raise FileNotFoundError(
            f"No JSON result files found under {dataset_dir / 'results'}"
        )
    return result_files[-1]


def markdown_table(frame) -> str:
    if frame is None or frame.empty:
        return "_No rows_\n"
    rows = frame.to_dict(orient="records")
    columns = list(frame.columns)
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body]) + "\n"


def save_merged_outputs(
    merged_name: str, merged_map: dict[str, dict[str, float]], output_dir: Path
) -> tuple[Path, Path]:
    pc = core()
    merged_list = pc.merged_mapping_to_list(merged_map)
    json_path = pc.save_json(output_dir / f"{merged_name}.json", merged_list)
    csv_path = pc.save_dataframe(
        output_dir / f"{merged_name}.csv", pc.merged_results_to_frame(merged_map)
    )
    return json_path, csv_path


def main() -> None:
    pc = core()
    merged_output_dir = pc.ensure_directory(ROOT / "datasets" / "_streamlit" / "merged")
    markdown_output = ROOT / "results" / "experiment_results.md"
    markdown_output.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = ["# Experiment Results", ""]
    latest_result_files: list[Path] = []
    merged_files: list[tuple[str, Path, Path]] = []

    sections.append("## Step 1: calculate metrics")
    for dataset_name, endpoint, safe_limit in DATASET_RUNS:
        dataset_dir = ROOT / "datasets" / dataset_name
        eval_dir = dataset_dir / "eval"
        sections.append(f"### {dataset_name}")
        result = pc.run_metrics_job(
            dataset=dataset_name,
            jsonl_evals=sorted(eval_dir.glob("*.jsonl")),
            metric_names=[
                "answerset_f1",
                "bleu",
                "exact_match_spinach",
                "qcan-bleu-strict",
                "qcan-bleu-flex",
                "rouge_4",
                "qcan-rouge-4-strict",
                "qcan-rouge-4-flex",
            ],
            execution_backend_endpoint_url=endpoint,
            parallel=True,
            per_query=True,
            safe_limit=safe_limit,
            verbose=True,
        )
        latest_result = latest_result_file(dataset_dir)
        latest_result_files.append(latest_result)
        sections.append(
            f"Calculated metrics with {len(result['jsonl_evals'])} eval files."
        )
        sections.append(f"Latest result file: {latest_result}")
        sections.append("")

    sections.append("## Step 2: merge latest result files")
    for dataset_name, _, _ in DATASET_RUNS:
        dataset_dir = ROOT / "datasets" / dataset_name
        latest_result = latest_result_file(dataset_dir)
        merged_map = pc.merge_result_files([latest_result])
        json_path, csv_path = save_merged_outputs(
            f"merged_{dataset_name}", merged_map, merged_output_dir
        )
        merged_files.append((dataset_name, json_path, csv_path))
        sections.append(f"### {dataset_name}")
        sections.append(f"JSON: {json_path}")
        sections.append(f"CSV: {csv_path}")
        sections.append("")

    combined_map = pc.merge_result_files(latest_result_files)
    combined_json, combined_csv = save_merged_outputs(
        "merged_all", combined_map, merged_output_dir
    )
    merged_files.append(("all", combined_json, combined_csv))
    sections.append("### merged_all")
    sections.append(f"JSON: {combined_json}")
    sections.append(f"CSV: {combined_csv}")
    sections.append("")

    sections.append("## Step 3: krippendorff alpha for merged all")
    available_metrics = pc.calculate_available_metric_set(combined_map)
    kripp_metrics = [
        metric for metric in pc.SUPPORTED_METRICS if metric in available_metrics
    ]
    alpha_result = pc.calculate_krippendorff_alpha_from_mapping(
        combined_map, kripp_metrics
    )
    sections.append(json.dumps(alpha_result, indent=2))
    sections.append("")

    sections.append("## Step 4: proving qcan value")
    for merged_name, json_path, _ in merged_files:
        with open(json_path, encoding="utf-8") as handle:
            merged_map = pc.normalize_merged_payload(json.load(handle))
        available_metrics = pc.calculate_available_metric_set(merged_map)
        for proxy_metric in PROXY_METRICS:
            sections.append(f"### {merged_name} / {proxy_metric}")
            if proxy_metric not in available_metrics:
                sections.append("Proxy metric not available in this merged file.")
                sections.append("")
                continue
            compare_metrics = [
                metric
                for metric in pc.SUPPORTED_METRICS
                if metric in available_metrics and metric != proxy_metric
            ]
            metric_df, comparison_df, examples_df, _ = pc.compute_metric_matrix(
                merged_map,
                proxy_metric=proxy_metric,
                compare_metrics=compare_metrics,
                scale_10=False,
                bootstrap_iters=1000,
                seed=7,
                top_k=5,
            )
            sections.append("Metric ranking:")
            sections.append(markdown_table(metric_df))
            sections.append("Significance checks:")
            sections.append(markdown_table(comparison_df))
            sections.append("Qualitative examples:")
            sections.append(markdown_table(examples_df))
            sections.append("")

    with open(markdown_output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))

    print(f"Wrote markdown summary to {markdown_output}")
    print(f"Created merged files under {merged_output_dir}")


if __name__ == "__main__":
    main()

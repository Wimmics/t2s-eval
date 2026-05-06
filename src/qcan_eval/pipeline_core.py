from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import krippendorff
except ImportError as exc:  # pragma: no cover - surfaced in the UI.
    krippendorff = None
    KRIPPENDORFF_IMPORT_ERROR = exc
else:
    KRIPPENDORFF_IMPORT_ERROR = None


SUPPORTED_METRICS = [
    "answerset_f1",
    "bleu",
    "exact_match_spinach",
    "qcan-bleu-strict",
    "qcan-bleu-flex",
    "rouge_4",
    "query_exact_match",
    "qcan-rouge-4-strict",
    "qcan-rouge-4-flex",
]


@dataclass
class MetricSummary:
    name: str
    pearson: float
    spearman: float
    kendall_tau_b: float
    pairwise_accuracy: float
    pearson_ci_low: float
    pearson_ci_high: float


def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def datasets_root() -> Path:
    return workspace_root() / "datasets"


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_dataset_dirs(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or datasets_root()
    datasets: list[dict[str, Any]] = []
    if not root.exists():
        return datasets

    for dataset_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        eval_dir = dataset_dir / "eval"
        if not eval_dir.exists():
            continue
        eval_files = sorted(eval_dir.glob("*.jsonl"))
        result_root = dataset_dir / "results"
        result_files = (
            sorted(result_root.glob("*.json")) if result_root.exists() else []
        )
        datasets.append(
            {
                "name": dataset_dir.name,
                "path": dataset_dir,
                "eval_dir": eval_dir,
                "eval_files": eval_files,
                "result_files": result_files,
            }
        )
    return datasets


def discover_merged_files(root: Path | None = None) -> list[Path]:
    root = root or datasets_root()
    if not root.exists():
        return []
    return sorted({path for path in root.rglob("merged_*.json") if path.is_file()})


def load_result_file(path: Path | str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path | str, data: Any) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return path


def save_dataframe(path: Path | str, frame: pd.DataFrame) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    frame.to_csv(path, index=False)
    return path


def merge_result_files(result_files: list[Path]) -> dict[str, dict[str, float]]:
    query_results: dict[str, dict[str, float]] = {}
    for result_file in result_files:
        entries = load_result_file(result_file)
        for entry in entries:
            dataset = entry.get("dataset", "unknown")
            system_name = entry.get("system_name", result_file.stem)
            for query_result in entry.get("per_query_results", []):
                query_id = query_result.get("id")
                metric = query_result.get("metric")
                score = query_result.get("score")
                if query_id is None or metric is None:
                    continue
                key = f"{dataset}_{system_name}_{query_id}"
                query_results.setdefault(key, {})[metric] = score

    return {
        query_key: dict(sorted(metric_scores.items(), key=lambda item: item[0]))
        for query_key, metric_scores in sorted(
            query_results.items(), key=lambda item: item[0]
        )
    }


def merged_results_to_frame(
    merged_results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = sorted(
        {
            metric
            for metric_scores in merged_results.values()
            for metric in metric_scores
        }
    )
    for query_key, metric_scores in merged_results.items():
        row = {"query_key": query_key}
        for metric in metric_names:
            row[metric] = metric_scores.get(metric)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame[["query_key", *metric_names]]
    return frame


def merged_mapping_to_list(
    merged_results: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    return [
        {
            "query_key": query_key,
            "metrics": metric_scores,
        }
        for query_key, metric_scores in sorted(
            merged_results.items(), key=lambda item: item[0]
        )
    ]


def normalize_merged_payload(payload: Any) -> dict[str, dict[str, float]]:
    if isinstance(payload, dict):
        return {
            str(query_key): {
                str(metric): float(score)
                for metric, score in metric_scores.items()
                if isinstance(score, (int, float))
            }
            for query_key, metric_scores in payload.items()
            if isinstance(metric_scores, dict)
        }

    if not isinstance(payload, list):
        raise ValueError("Merged file must be either a merged map or a merged list.")

    result: dict[str, dict[str, float]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue

        # Preferred list shape: {"query_key": "...", "metrics": {...}}
        if "query_key" in row and isinstance(row.get("metrics"), dict):
            query_key = str(row["query_key"])
            result[query_key] = {
                str(metric): float(score)
                for metric, score in row["metrics"].items()
                if isinstance(score, (int, float))
            }
            continue

        # Alternate list shape: {"query_key": "...", "metric_a": 0.1, ...}
        if "query_key" in row:
            query_key = str(row["query_key"])
            result[query_key] = {
                str(metric): float(score)
                for metric, score in row.items()
                if metric != "query_key" and isinstance(score, (int, float))
            }
            continue

        # Fallback list shape: {"some_query_key": {...metrics...}}
        if len(row) == 1:
            query_key, metric_scores = next(iter(row.items()))
            if isinstance(metric_scores, dict):
                result[str(query_key)] = {
                    str(metric): float(score)
                    for metric, score in metric_scores.items()
                    if isinstance(score, (int, float))
                }

    if not result:
        raise ValueError(
            "Could not parse merged list format. Expected entries with query_key and metrics."
        )
    return result


def calculate_available_metric_set(
    query_results: dict[str, dict[str, float]],
) -> list[str]:
    return sorted(
        {metric for metric_scores in query_results.values() for metric in metric_scores}
    )


def load_problematic_queries_from_mapping(
    query_results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    query_ids = list(query_results.keys())
    metric_names = metrics or sorted(
        {metric for metric_scores in query_results.values() for metric in metric_scores}
    )
    matrix: dict[str, np.ndarray] = {}
    for metric in metric_names:
        matrix[metric] = np.array(
            [float(query_results[q].get(metric, 0.0)) for q in query_ids], dtype=float
        )
    return query_ids, matrix


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    n = len(x)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                ties_x += 1
                continue
            if dy == 0.0:
                ties_y += 1
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denom == 0.0:
        return float("nan")
    return (concordant - discordant) / denom


def pairwise_accuracy(metric: np.ndarray, proxy: np.ndarray) -> float:
    agree = 0.0
    total = 0
    n = len(metric)
    for i in range(n - 1):
        for j in range(i + 1, n):
            proxy_pref = proxy[i] - proxy[j]
            if proxy_pref == 0.0:
                continue
            metric_pref = metric[i] - metric[j]
            total += 1
            if metric_pref == 0.0:
                agree += 0.5
            elif metric_pref * proxy_pref > 0:
                agree += 1.0

    if total == 0:
        return float("nan")
    return agree / total


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    corr_fn,
    rng: random.Random,
    iterations: int,
    alpha: float,
) -> tuple[float, float]:
    n = len(x)
    values = []
    for _ in range(iterations):
        idx = np.array([rng.randrange(n) for _ in range(n)], dtype=int)
        values.append(corr_fn(x[idx], y[idx]))
    low = float(np.nanpercentile(values, 100.0 * alpha / 2.0))
    high = float(np.nanpercentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def bootstrap_pvalue_for_difference(
    a: np.ndarray,
    b: np.ndarray,
    proxy: np.ndarray,
    corr_fn,
    rng: random.Random,
    iterations: int,
) -> tuple[float, float]:
    observed = corr_fn(a, proxy) - corr_fn(b, proxy)
    n = len(proxy)
    diffs = []
    for _ in range(iterations):
        idx = np.array([rng.randrange(n) for _ in range(n)], dtype=int)
        diffs.append(corr_fn(a[idx], proxy[idx]) - corr_fn(b[idx], proxy[idx]))
    diffs_np = np.array(diffs)
    one_sided = float(np.mean(diffs_np <= 0.0))
    return observed, one_sided


def qualitative_examples(
    query_ids: list[str],
    proxy: np.ndarray,
    challenger: np.ndarray,
    baseline: np.ndarray,
    top_k: int,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    deltas = np.abs(challenger - proxy) - np.abs(baseline - proxy)
    order = np.argsort(deltas)
    wins = [(query_ids[i], float(deltas[i])) for i in order[:top_k]]
    fails = [(query_ids[i], float(deltas[i])) for i in order[-top_k:][::-1]]
    return wins, fails


def calculate_krippendorff_alpha_from_mapping(
    query_results: dict[str, dict[str, float]],
    subset_metrics: list[str],
    scale_to_ten: bool = True,
) -> dict[str, Any]:
    if krippendorff is None:  # pragma: no cover - surfaced in the app.
        raise ImportError(
            "krippendorff is not available"
        ) from KRIPPENDORFF_IMPORT_ERROR
    if len(subset_metrics) < 2:
        raise ValueError("Select at least two metrics for Krippendorff alpha.")

    rows: list[list[int]] = []
    for metric_scores in query_results.values():
        row: list[int] = []
        for metric in subset_metrics:
            score = metric_scores.get(metric, 0.0)
            if not isinstance(score, (int, float)):
                score = 0.0
            if scale_to_ten:
                score = int(float(score) * 10)
            else:
                score = int(float(score))
            row.append(score)
        rows.append(row)

    value_counts = np.array(rows, dtype=np.int32)
    return {
        "metrics": subset_metrics,
        "scale_to_ten": scale_to_ten,
        "sample_size": int(value_counts.shape[0]),
        "alpha_nominal": float(
            krippendorff.alpha(
                value_counts=value_counts, level_of_measurement="nominal"
            )
        ),
        "alpha_interval": float(
            krippendorff.alpha(
                value_counts=value_counts, level_of_measurement="interval"
            )
        ),
        "alpha_ordinal": float(
            krippendorff.alpha(
                value_counts=value_counts, level_of_measurement="ordinal"
            )
        ),
        "alpha_ratio": float(
            krippendorff.alpha(value_counts=value_counts, level_of_measurement="ratio")
        ),
    }


def compute_metric_matrix(
    query_results: dict[str, dict[str, float]],
    proxy_metric: str,
    compare_metrics: list[str] | None = None,
    scale_10: bool = False,
    bootstrap_iters: int = 3000,
    seed: int = 7,
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[MetricSummary]]:
    query_ids, matrix = load_problematic_queries_from_mapping(query_results)
    all_metrics = calculate_available_metric_set(query_results)
    if scale_10:
        for metric in all_metrics:
            matrix[metric] = 10.0 * matrix[metric]

    proxy = matrix[proxy_metric]
    metrics_to_compare = compare_metrics or [
        metric for metric in all_metrics if metric != proxy_metric
    ]
    rng = random.Random(seed)
    summaries: list[MetricSummary] = []
    for metric in metrics_to_compare:
        if metric not in matrix:
            continue
        values = matrix[metric]
        pearson = pearson_corr(values, proxy)
        spearman = spearman_corr(values, proxy)
        tau = kendall_tau_b(values, proxy)
        pw = pairwise_accuracy(values, proxy)
        ci_low, ci_high = bootstrap_ci(
            values, proxy, pearson_corr, rng, bootstrap_iters, alpha=0.05
        )
        summaries.append(
            MetricSummary(
                name=metric,
                pearson=pearson,
                spearman=spearman,
                kendall_tau_b=tau,
                pairwise_accuracy=pw,
                pearson_ci_low=ci_low,
                pearson_ci_high=ci_high,
            )
        )

    summaries.sort(key=lambda item: item.pearson, reverse=True)
    metric_df = pd.DataFrame(
        [
            {
                "metric": summary.name,
                "pearson": summary.pearson,
                "pearson_ci_low": summary.pearson_ci_low,
                "pearson_ci_high": summary.pearson_ci_high,
                "spearman": summary.spearman,
                "kendall_tau_b": summary.kendall_tau_b,
                "pairwise_accuracy": summary.pairwise_accuracy,
            }
            for summary in summaries
        ]
    )

    comparison_rows: list[dict[str, Any]] = []
    comparison_pairs = [
        ("qcan-bleu-strict", "bleu"),
        ("qcan-bleu-flex", "bleu"),
        ("qcan-bleu-strict", "answerset_f1"),
        ("qcan-bleu-flex", "answerset_f1"),
    ]
    for better, baseline in comparison_pairs:
        if better not in matrix or baseline not in matrix or proxy_metric not in matrix:
            continue
        observed, pvalue = bootstrap_pvalue_for_difference(
            matrix[better],
            matrix[baseline],
            proxy,
            pearson_corr,
            rng,
            bootstrap_iters,
        )
        comparison_rows.append(
            {
                "better": better,
                "baseline": baseline,
                "delta_r": observed,
                "p_value": pvalue,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    examples_rows: list[dict[str, Any]] = []
    if "qcan-bleu-strict" in matrix and "bleu" in matrix and proxy_metric in matrix:
        wins, fails = qualitative_examples(
            query_ids, proxy, matrix["qcan-bleu-strict"], matrix["bleu"], top_k=top_k
        )
        for label, rows in (("wins", wins), ("fails", fails)):
            for query_id, delta in rows:
                index = query_ids.index(query_id)
                examples_rows.append(
                    {
                        "kind": label,
                        "query_id": query_id,
                        "delta": delta,
                        "proxy": float(proxy[index]),
                        "qcan-bleu-strict": float(matrix["qcan-bleu-strict"][index]),
                        "bleu": float(matrix["bleu"][index]),
                    }
                )

    examples_df = pd.DataFrame(examples_rows)
    return metric_df, comparison_df, examples_df, summaries


def build_metric_objects(metric_names: list[str]) -> list[Any]:
    try:
        from t2smetrics.metrics import (
            AnswerSetF1,
            Bleu,
            ExactMatchSpinach,
            QCanBleu,
            QCanRougeN,
            QueryExactMatch,
            RougeN,
        )
    except ImportError as exc:  # pragma: no cover - surfaced in the UI.
        raise ImportError(
            "t2smetrics is not installed. The metrics runner can only be used when that package is available."
        ) from exc

    metric_objects: list[Any] = []
    for metric_name in metric_names:
        if metric_name == "answerset_f1":
            metric_objects.append(AnswerSetF1())
        elif metric_name == "bleu":
            metric_objects.append(Bleu())
        elif metric_name == "exact_match_spinach":
            metric_objects.append(ExactMatchSpinach())
        elif metric_name == "qcan-bleu-strict":
            metric_objects.append(QCanBleu(calculation_type="strict"))
        elif metric_name == "qcan-bleu-flex":
            metric_objects.append(QCanBleu(calculation_type="flex"))
        elif metric_name == "rouge_4":
            metric_objects.append(RougeN(4))
        elif metric_name == "query_exact_match":
            metric_objects.append(QueryExactMatch())
        elif metric_name == "qcan-rouge-4-strict":
            metric_objects.append(QCanRougeN(n=4, calculation_type="strict"))
        elif metric_name == "qcan-rouge-4-flex":
            metric_objects.append(QCanRougeN(n=4, calculation_type="flex"))
    return metric_objects


def run_metrics_job(
    dataset: str,
    jsonl_evals: list[Path],
    metric_names: list[str],
    execution_backend_endpoint_url: str,
    parallel: bool,
    per_query: bool,
    safe_limit: int,
    verbose: bool,
) -> dict[str, Any]:
    try:
        from t2smetrics import run_experiments
        from t2smetrics.core.logging import setup_third_party_logging
    except ImportError as exc:  # pragma: no cover - surfaced in the UI.
        raise ImportError(
            "t2smetrics is not installed. Install or expose the package before running the metrics stage."
        ) from exc

    setup_third_party_logging(logging_level=logging.WARNING)
    metric_objects = build_metric_objects(metric_names)
    run_experiments.run(
        dataset=dataset,
        jsonl_evals=[str(path) for path in jsonl_evals],
        metrics_list=metric_objects,
        execution_backend_endpoint_url=execution_backend_endpoint_url,
        verbose=verbose,
        parallel=parallel,
        per_query=per_query,
        safe_limit=safe_limit,
    )

    return {
        "dataset": dataset,
        "jsonl_evals": [str(path) for path in jsonl_evals],
        "metrics": metric_names,
        "endpoint": execution_backend_endpoint_url,
        "parallel": parallel,
        "per_query": per_query,
        "safe_limit": safe_limit,
        "verbose": verbose,
    }

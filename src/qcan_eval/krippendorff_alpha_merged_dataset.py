import json

import krippendorff
import numpy as np
from loguru import logger


def normilise_int(num, scale: int = 10) -> int:
    """Convert a number to integer (scaled by scale)."""
    if isinstance(num, (int, float)):
        return int(num * scale)
    return 0


def load_merged_dataset(
    file_path: str = "./datasets/_streamlit/merged/merged_all_datasets.json",
):
    """Load the merged dataset from JSON file.

    Args:
        file_path: Path to the merged dataset JSON file

    Returns:
        Dictionary mapping query_key to metrics
    """
    with open(file_path) as f:
        data: list[dict] = json.load(f)

    # Transform array of objects to dict indexed by query_key
    query_results = {}
    for entry in data:
        query_key = entry.pop("query_key")
        query_results[query_key] = entry

    logger.info(f"Total unique queries loaded: {len(query_results)}")
    return query_results


def calculate_krippendorff_alpha_merged(subset_metrics: list[str] | None = None):
    """Calculate Krippendorff's alpha for the merged dataset.

    Args:
        subset_metrics: List of specific metrics to use. If None, uses all available metrics.

    Returns:
        Tuple of (alpha_nominal, alpha_interval, alpha_ordinal, alpha_ratio)
    """
    query_results = load_merged_dataset()

    scale = 10  # Scale factor for converting to integers

    # Convert all metric scores to integers (scaled by 10)
    for query_key, metrics in query_results.items():
        for metric, score in metrics.items():
            if isinstance(score, (int, float)) and score is not None:
                query_results[query_key][metric] = normilise_int(score, scale)
            elif score is None:
                query_results[query_key][metric] = 0
            else:
                logger.warning(
                    f"Non-numeric score found for {query_key} and metric {metric}: {score}"
                )

    # Get all unique metrics if not specified
    if subset_metrics is None:
        unique_metrics = set()
        for metrics in query_results.values():
            unique_metrics.update(metrics.keys())
        subset_metrics = list(sorted(unique_metrics))
        logger.info(f"Unique metrics found: {subset_metrics}")
    else:
        logger.info(f"Using subset metrics: {subset_metrics}")

    # Build value counts matrix
    value_counts = np.array(
        [
            [metrics.get(metric, 0) for metric in subset_metrics]
            for query_key, metrics in query_results.items()
        ],
        dtype=np.int32,
    )

    logger.info(f"Value counts matrix shape: {value_counts.shape}")

    # Calculate Krippendorff's alpha for different measurement levels
    alpha_nominal = krippendorff.alpha(
        value_counts=value_counts, level_of_measurement="nominal"
    )

    alpha_interval = krippendorff.alpha(
        value_counts=value_counts, level_of_measurement="interval"
    )

    alpha_ordinal = krippendorff.alpha(
        value_counts=value_counts, level_of_measurement="ordinal"
    )

    alpha_ratio = krippendorff.alpha(
        value_counts=value_counts, level_of_measurement="ratio"
    )

    return (alpha_nominal, alpha_interval, alpha_ordinal, alpha_ratio)


def test_calculate_krippendorff_alpha_merged():
    """Test Krippendorff's alpha calculation with merged dataset."""
    subset_metrics = [
        "answerset_f1",
        "exact_match_spinach",
        "bleu",
        "rouge_4",
        "sp-bleu",
        "qcan-bleu-flex",
        "qcan-bleu-strict",
        "qcan-rouge-4-flex",
        "qcan-rouge-4-strict",
        "naive-can-rouge-4",
        "naive-can-bleu",
    ]

    alpha_nominal, alpha_interval, alpha_ordinal, alpha_ratio = (
        calculate_krippendorff_alpha_merged(subset_metrics)
    )

    logger.info(
        f"Metrics: {subset_metrics}\n"
        f"  - Alpha Nominal: {alpha_nominal:.4f}\n"
        f"  - Alpha Interval: {alpha_interval:.4f}\n"
        f"  - Alpha Ordinal: {alpha_ordinal:.4f}\n"
        f"  - Alpha Ratio: {alpha_ratio:.4f}"
    )


if __name__ == "__main__":
    test_calculate_krippendorff_alpha_merged()

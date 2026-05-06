import json
from itertools import combinations

import krippendorff
import numpy as np
from loguru import logger


def to_ten_int(num):
    return int(num * 10)


def calculate_krippendorff_alpha(subset_metrics: list[str] | None = None):

    with open("./results/problematic_queries_all.json") as f:
        query_results: dict[str, dict] = json.load(f)
        # logger.info(f"Total unique queries loaded: {len(query_results)}")

    for query_key, metrics in query_results.items():
        for metric, score in metrics.items():
            if isinstance(score, (int, float)):
                query_results[query_key][metric] = to_ten_int(score)
            else:
                logger.warning(
                    f"Non-numeric score found for {query_key} and metric {metric}: {score}"
                )

    unique_metrics = set()
    for metrics in query_results.values():
        unique_metrics.update(metrics.keys())

    unique_metrics = list(sorted(unique_metrics))
    # logger.info(f"Unique metrics found: {unique_metrics}")

    used_metrics = subset_metrics

    value_counts = np.array(
        [
            [metrics.get(metric, 0) for metric in used_metrics]
            for query_key, metrics in query_results.items()
        ],
        dtype=np.int32,
    )

    logger.info(f"Value counts matrix: {value_counts}")

    # print("Value counts matrix shape: ", value_counts.shape)

    alpha_nominal = krippendorff.alpha(
        value_counts=value_counts, level_of_measurement="nominal"
    )
    # print("Krippendorff's alpha for nominal metric: ", alpha_nominal)

    alpha_interval = krippendorff.alpha(
        value_counts=value_counts, level_of_measurement="interval"
    )

    # print("Krippendorff's alpha for interval metric: ", alpha_interval)

    alpha_ordinal = krippendorff.alpha(
        value_counts=value_counts, level_of_measurement="ordinal"
    )
    # print("Krippendorff's alpha for ordinal metric: ", alpha_ordinal)

    alpha_ratio = krippendorff.alpha(
        value_counts=value_counts, level_of_measurement="ratio"
    )
    # print("Krippendorff's alpha for ratio metric: ", alpha_ratio)

    return (alpha_nominal, alpha_interval, alpha_ordinal, alpha_ratio)


def calculate_combinations_of_metrics():
    subset_metrics = [
        # "answerset_precision",
        # "answerset_recall",
        # "codebleu",
        # "cosine_sim",
        # "euclidean",
        # "f1_qald",
        # "f1_spinach",
        # "hit@1",
        # "jaccard",
        # "levenshtein",
        # "meteor",
        # "mrr",
        # "ndcg",
        # "p@1",
        # "precision_qald",
        # "query_exact_match",
        # "query_execution",
        # "recall_qald",
        # "rouge_4",
        # "sp-bleu",
        # "sp-f1",
        # "token_f1",
        # "token_precision",
        # "token_recall",
        # "uri_hallucination",
        "answerset_f1",
        "bleu",
        "exact_match_spinach",
        "qcan-bleu-strict",
        "qcan-bleu-flex",
        "qcan-rouge-4-strict",
        "qcan-rouge-4-flex",
    ]

    map_metrics = {}

    for size in range(2, 3):  # 2 to 5 inclusive
        for combo in combinations(subset_metrics, size):
            alpha_nominal, alpha_interval, alpha_ordinal, alpha_ratio = (
                calculate_krippendorff_alpha(list(combo))
            )
            map_metrics[combo] = {
                "alpha_nominal": alpha_nominal,
                "alpha_interval": alpha_interval,
                "alpha_ordinal": alpha_ordinal,
                "alpha_ratio": alpha_ratio,
            }

    sorted_map = dict(
        sorted(
            map_metrics.items(),
            key=lambda item: item[1]["alpha_ordinal"],
            # reverse=True,
        )
    )

    for combo, alphas in sorted_map.items():
        logger.info(
            f"Metrics: {combo}\n  - Alpha Nominal: {alphas['alpha_nominal']:.4f}\n  - Alpha Interval: {alphas['alpha_interval']:.4f}\n  - Alpha Ordinal: {alphas['alpha_ordinal']:.4f}\n  - Alpha Ratio: {alphas['alpha_ratio']:.4f}"
        )


def test_calculate_krippendorff_alpha():
    subset_metrics = [
        # "answerset_precision",
        # "answerset_recall",
        # "codebleu",
        # "cosine_sim",
        # "euclidean",
        # "f1_qald",
        # "f1_spinach",
        # "hit@1",
        # "jaccard",
        # "levenshtein",
        # "meteor",
        # "mrr",
        # "ndcg",
        # "p@1",
        # "precision_qald",
        # "query_exact_match",
        # "query_execution",
        # "recall_qald",
        "rouge_4",
        # "sp-bleu",
        # "sp-f1",
        # "token_f1",
        # "token_precision",
        # "token_recall",
        # "uri_hallucination",
        "answerset_f1",
        "bleu",
        "exact_match_spinach",
        "qcan-bleu-strict",
        "qcan-bleu-flex",
        "qcan-rouge-4-strict",
        "qcan-rouge-4-flex",
    ]

    alpha_nominal, alpha_interval, alpha_ordinal, alpha_ratio = (
        calculate_krippendorff_alpha(subset_metrics)
    )

    logger.info(
        f"Metrics: {subset_metrics}\n  - Alpha Nominal: {alpha_nominal:.4f}\n  - Alpha Interval: {alpha_interval:.4f}\n  - Alpha Ordinal: {alpha_ordinal:.4f}\n  - Alpha Ratio: {alpha_ratio:.4f}"
    )


if __name__ == "__main__":
    test_calculate_krippendorff_alpha()
    # calculate_combinations_of_metrics()

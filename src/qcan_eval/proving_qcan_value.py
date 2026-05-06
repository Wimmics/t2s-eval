import argparse
import json
import math
import random
from dataclasses import dataclass

import numpy as np


METRICS = [
	"answerset_f1",
	"bleu",
	"exact_match_spinach",
	"qcan-bleu-strict",
	"qcan-bleu-flex",
	"rouge_4",
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

	denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
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
	return (low, high)


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


def load_problematic_queries(path: str) -> tuple[list[str], dict[str, np.ndarray]]:
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	query_ids = list(data.keys())
	matrix: dict[str, np.ndarray] = {}
	for metric in METRICS:
		matrix[metric] = np.array([float(data[q].get(metric, 0.0)) for q in query_ids], dtype=float)
	return query_ids, matrix


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


def print_table(summaries: list[MetricSummary]) -> None:
	print("| Metric | Pearson r (95% CI) | Spearman r | Kendall tau-b | Pairwise accuracy |")
	print("|---|---:|---:|---:|---:|")
	for s in summaries:
		print(
			f"| {s.name} | {s.pearson:.4f} [{s.pearson_ci_low:.4f}, {s.pearson_ci_high:.4f}]"
			f" | {s.spearman:.4f} | {s.kendall_tau_b:.4f} | {s.pairwise_accuracy:.4f} |"
		)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Validate whether qcan metrics are closer to an execution-style proxy than other string metrics."
	)
	parser.add_argument(
		"--input",
		default="./problematic_queries.json",
		help="Path to problematic_queries.json",
	)
	parser.add_argument(
		"--proxy-metric",
		default="answerset_f1",
		choices=METRICS,
		help="Metric used as execution-proxy ground truth.",
	)
	parser.add_argument(
		"--scale-10",
		action="store_true",
		help="Scale scores from [0,1] to [0,10] before analysis.",
	)
	parser.add_argument(
		"--bootstrap-iters",
		type=int,
		default=3000,
		help="Bootstrap iterations for confidence intervals and p-values.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=7,
		help="Random seed for reproducibility.",
	)
	parser.add_argument(
		"--top-k-examples",
		type=int,
		default=5,
		help="Number of qualitative win/failure examples to print.",
	)
	args = parser.parse_args()

	query_ids, matrix = load_problematic_queries(args.input)
	if args.scale_10:
		for metric in METRICS:
			matrix[metric] = 10.0 * matrix[metric]

	proxy = matrix[args.proxy_metric]
	compare_metrics = [m for m in METRICS if m != args.proxy_metric]
	rng = random.Random(args.seed)

	summaries = []
	for metric in compare_metrics:
		values = matrix[metric]
		pearson = pearson_corr(values, proxy)
		spearman = spearman_corr(values, proxy)
		tau = kendall_tau_b(values, proxy)
		pw = pairwise_accuracy(values, proxy)
		ci_low, ci_high = bootstrap_ci(
			values,
			proxy,
			pearson_corr,
			rng,
			args.bootstrap_iters,
			alpha=0.05,
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

	summaries.sort(key=lambda s: s.pearson, reverse=True)

	print(f"Loaded {len(query_ids)} examples from {args.input}")
	print(f"Proxy execution metric: {args.proxy_metric}")
	print(f"Bootstrap iterations: {args.bootstrap_iters}")
	print()
	print_table(summaries)
	print()

	comparisons = []
	if args.proxy_metric != "bleu":
		comparisons.append(("qcan-bleu-strict", "bleu"))
	if args.proxy_metric != "answerset_f1":
		comparisons.append(("qcan-bleu-flex", "answerset_f1"))
	elif args.proxy_metric != "bleu":
		comparisons.append(("qcan-bleu-flex", "bleu"))

	print("Significance checks (one-sided bootstrap p-value for corr(A, proxy) > corr(B, proxy)):")
	for better, baseline in comparisons:
		if better == args.proxy_metric or baseline == args.proxy_metric:
			continue
		observed, pval = bootstrap_pvalue_for_difference(
			matrix[better],
			matrix[baseline],
			proxy,
			pearson_corr,
			rng,
			args.bootstrap_iters,
		)
		print(f"- {better} vs {baseline}: delta_r={observed:.4f}, p={pval:.6f}")

	print()
	print("Qualitative examples (delta = |challenger-proxy| - |baseline-proxy|; lower is better):")
	q_wins, q_fails = qualitative_examples(
		query_ids,
		proxy,
		matrix["qcan-bleu-strict"],
		matrix["bleu"],
		args.top_k_examples,
	)
	print("- qcan-bleu-strict beats bleu (best deltas):")
	for qid, delta in q_wins:
		print(
			f"  - {qid}: delta={delta:.4f}, proxy={proxy[query_ids.index(qid)]:.4f},"
			f" qcan-strict={matrix['qcan-bleu-strict'][query_ids.index(qid)]:.4f},"
			f" bleu={matrix['bleu'][query_ids.index(qid)]:.4f}"
		)
	print("- qcan-bleu-strict fails vs bleu (worst deltas):")
	for qid, delta in q_fails:
		print(
			f"  - {qid}: delta={delta:.4f}, proxy={proxy[query_ids.index(qid)]:.4f},"
			f" qcan-strict={matrix['qcan-bleu-strict'][query_ids.index(qid)]:.4f},"
			f" bleu={matrix['bleu'][query_ids.index(qid)]:.4f}"
		)


if __name__ == "__main__":
	main()

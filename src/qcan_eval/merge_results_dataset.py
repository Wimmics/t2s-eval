import json

import loguru

loguru.logger.info("Starting problematic query identification...")

result_files = [
    "./datasets/ck25/results/ck25-20260429-153052.json",
    "./datasets/ck26/results/ck26-20260504-110702.json",
    "./datasets/db25/results/db25-20260429-164041.json",
    "./datasets/db26/results/db26-20260429-174352.json",
]

query_results = {}

for result_file in result_files:
    loguru.logger.info(f"Processing {result_file}...")
    with open(result_file) as f:
        qa_results: list = json.load(f)
        for qa_result in qa_results:
            dataset = qa_result["dataset"]
            system_name = qa_result["system_name"]
            per_query_results: list = qa_result["per_query_results"]
            loguru.logger.info(
                f"Processing dataset: {dataset}, system: {system_name} with {len(per_query_results)} per-query results..."
            )
            for query_result in per_query_results:
                query_id = query_result["id"]
                metric = query_result["metric"]
                score = query_result["score"]

                query_results.setdefault(
                    f"{dataset}_{system_name}_{query_id}", {}
                ).update({metric: score})
loguru.logger.info(
    f"Finished processing all result files. Total unique queries: {len(query_results)}"
)

sorted_query_results = {}

for query_key, metrics in query_results.items():
    sorted_items = sorted(metrics.items(), key=lambda item: item[1])
    sorted_query_results[query_key] = dict(sorted_items)

with open("./problematic_queries_all.json", "w") as f:
    json.dump(sorted_query_results, f, indent=4)

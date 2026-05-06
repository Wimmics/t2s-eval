import json
from os import makedirs

import yaml

# datasets = ["ck26", "db26"]
# qa_systems = [
#     "ADFR",
#     "INFAI-ETI-AND-FRIENDS-A",
#     "INFAI-ETI-AND-FRIENDS-B",
#     "INFAI-ETI-AND-FRIENDS-C",
#     "IRIS",
#     "LIBER-AI-CLAUDE",
#     "LIBER-AI-QWEN",
#     "SPARQL-LLM"
# ]
datasets = ["ck25", "db25"]
qa_systems = [
    "AIFB",
    "DBPEDIA-CG",
    "DBPEDIA-CL",
    "DBPEDIA-SC",
    "FRANZ",
    "IIS-L",
    "IIS-Q",
    "INFAI",
    "LABIC",
    "LACODAM",
    "MIPT",
    "WSE",
]

languages = ["en", "es"]

for dataset in datasets:
    for qa_system in qa_systems:
        with open(
            f"datasets/{dataset}/raw_eval/{qa_system}/{dataset}_answers.json"
        ) as f_gen_ck25:
            f_gen_data = json.load(f_gen_ck25)

        with open(
            f"datasets/{dataset}/raw_eval/questions_{dataset}.yml"
        ) as f_gold_ck25:
            content = f_gold_ck25.read()
            f_gold_data = yaml.safe_load(content)

        collected_data = []
        for gold_item in f_gold_data["questions"]:
            for language in languages:
                try:
                    item = next(
                        item
                        for item in f_gen_data
                        if item["qname"] == f"{dataset}:{gold_item['id']}-{language}"
                    )
                    collected_data.append(
                        {
                            "id": f"{dataset}:{gold_item['id']}-{language}",
                            "question": gold_item["question"][language],
                            "golden": gold_item["query"]["sparql"],
                            "generated": item["query"],
                            "order_matters": bool(
                                "features" in gold_item
                                and "RESULT_ORDER_MATTERS" in gold_item["features"]
                            ),
                        }
                    )
                except StopIteration:
                    pass
        export_path = f"datasets/{dataset}/eval"

        makedirs(export_path, exist_ok=True)

        with open(f"{export_path}/{qa_system}.jsonl", "w") as f:
            for item in collected_data:
                f.write(json.dumps(item) + "\n")

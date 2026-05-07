# T2S Eval — T2S-Metrics Evaluation


<p align="center">
    <em>
        This repository contains evaluation tooling, datasets, and analysis scripts used for the "QCan Family Metrics" evaluation. 
    </em>
</p>

> [!IMPORTANT]
> The Results are also available online at: [https://wimmics.github.io/t2s-eval/](https://wimmics.github.io/t2s-eval/)


## Features

- Provide the results of evaluating all the metrics of the [t2s-metrics](https://github.com/Wimmics/t2s-metrics/) library on all the benchmarks from the [Text2SPARQL Challenges 2025 and 2026](https://text2sparql.aksw.org/).
- Analyze the behaviors of four frequently used execution-based and string-based metrics together with four QCan metrics.
- Produce dashboards of the results (interactive Streamlit or static HTML snapshots).
- Plot metric correlations and per-experiment summaries.

## Prerequisites

- [Python](https://www.python.org/) 3.12 or later.
- [uv](https://docs.astral.sh/uv/) (recommended for local development) or pip.
- A SPARQL endpoint only if you use execution metrics with a remote KG (for example QLever/Corese).
- [Ollama](http://ollama.com/) only if you enable LLM-based metrics.
- QCan jar only if you use qcan-related metrics. The repository includes it under [third_party_lib](https://github.com/Wimmics/t2s-metrics/tree/main/third_party_lib).
- [NLTK data](https://www.nltk.org/data.html) only if you use BLEU and METEOR realated metrics. 



## For development (editable install):

1. Clone the repository:
```bash
git clone https://github.com/Wimmics/t2s-eval.git
```

2. Navigate to the project directory:
```bash
cd t2s-eval
```

3. Install dependencies using `uv`:
```bash
uv sync
```

## Repository layout

- `datasets/` — JSONL evaluation files and per-dataset result directories (ck25, ck26, db25, db26, ...).
- `docs/` — exported dashboards and static pages.
- `results_analysis/` — CSV summaries and experiment notes.
- `src/qcan_eval/` — evaluation pipeline, metric calculation, merge tools, Streamlit interfaces.
- `src/t2s_soa/` — scripts for generating plots, inventories and paper-oriented helpers.
- `third_party_lib/` — redistributed third-party binaries (QCan jar).

## Common tasks / Usage

The repository contains small scripts for common evaluation workflows under `src/`.

Run the interactive Streamlit dashboard for ad-hoc exploration:

```bash
streamlit run src/qcan_eval/web_interface.py
```

Generate a static, self-contained dashboard snapshot (useful for sharing):

```bash
uv run src/qcan_eval/web_interface_static.py

# The exporter writes to docs/qcan-eval-static/index.html
```


Produce metric correlation plots for a given merged result file:

```bash
uv run src/qcan_eval/metrics_corr_plots.py -r datasets/db26/results/db26-20260429-174352.json -om results/ -os results/ -oc results/
```

Explore helper scripts under `src/t2s_soa/` for inventory generation, paper figures, and bibliography transformations.

## Datasets and expected formats

Evaluation inputs are JSON Lines (`.jsonl`) files with one JSON object per line. Each object should include at least:

- `id`: unique example identifier
- `golden`: reference SPARQL query (string)
- `generated`: system-generated SPARQL query (string)
- `order_matters`: boolean (whether result ordering matters)

Example files are available under `datasets/*/eval/`.

Results of metric runs are exported to `datasets/{dataset}/results/` as timestamped JSON files.

## Reproducing experiments from this repo

1. Ensure the relevant dataset folder is present under `datasets/` (e.g. `datasets/ck25/`).
2. Run the following script and generate the results through the GUI:
```bash
streamlit run src/qcan_eval/web_interface.py
```

Or 

2. Run the following script to generate a markdown file with all the results:

```bash
uv run src/qcan_eval/generate_markdown_experiments.py
```

## License

### T2S-Eval 

#### Software

t2s-eval scripts under `src` are provided under the terms of the [GNU Affero General Public License 3.0](https://github.com/Wimmics/t2s-eval/blob/main/LICENSES/AGPL-3.0.txt) (AGPL-3.0).

#### Datasets

t2s-eval datasets under `datasets/{dataset}/results`, `datasets/_streamlit` and `results_analysis` are provided under the terms of the [Creative Commons Attribution-ShareAlike 4.0 International](https://github.com/Wimmics/t2s-eval/blob/main/LICENSES/CC-BY-SA-4.0.txt) (CC-BY-SA-4.0).


### Redistribution of third-party software and data

This repository provides several third-party contributions redistributed with their original licenses.

#### CK25 Dataset

t2s-eval reuses the [CK25 Corporate Knowledge Reference Dataset for Benchmarking Text-2-SPARQL QA Approaches](https://github.com/eccenca/ck25-dataset/) that we modified to account for file format requirements (jsonl format).

The modified version is redistributed in directory [datasets/ck25](https://github.com/Wimmics/t2s-eval/blob/main/datasets/ck25) under the terms of the [Creative Commons Attribution 4.0 International license](https://github.com/Wimmics/t2s-eval/blob/main/LICENSES/CC-BY-4.0.txt) (CC-BY-4.0).

#### CK26, DB25 and DB26 Datasets

t2s-eval reuses the [CK26, DB25 and DB26](https://github.com/AKSW/text2sparql.aksw.org) that we modified to account for file format requirements (jsonl format).

The modified version is redistributed in directories [datasets/ck26](https://github.com/Wimmics/t2s-eval/blob/main/datasets/ck26), [datasets/db25](https://github.com/Wimmics/t2s-eval/blob/main/datasets/db25) and [datasets/db26](https://github.com/Wimmics/t2s-eval/blob/main/datasets/db26) under the terms of the [Creative Commons Attribution-ShareAlike 4.0 International](https://github.com/Wimmics/t2s-eval/blob/main/LICENSES/CC-BY-SA-4.0.txt) (CC-BY-SA-4.0).

#### QCan library

t2s-eval reuses the [QCan software for canonicalising SPARQL queries](https://github.com/RittoShadow/QCan).

QCan is written in Java. In this repository, we distribute the compiled jar of QCan v1.1, [third_party_lib/qcan-1.1-jar-with-dependencies.jar](https://github.com/Wimmics/t2s-eval/blob/main/third_party_lib/qcan-1.1-jar-with-dependencies.jar), under the terms of the [Apache 2.0 license](https://github.com/Wimmics/t2s-eval/blob/main/LICENSES/Apache-2.0.txt).
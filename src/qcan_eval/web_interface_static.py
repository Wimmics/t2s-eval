"""Static dashboard generator for the t2s-eval pipeline.

This module scans the repository datasets and precomputed artefacts, then writes a
single-file HTML snapshot that can be deployed online without Streamlit.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from pipeline_core import (  # noqa: E402
    calculate_available_metric_set,
    datasets_root,
    discover_dataset_dirs,
    discover_merged_files,
    ensure_directory,
    merged_results_to_frame,
    normalize_merged_payload,
)

PREVIEW_ROWS = 25
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "qcan-eval-static"


def _relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _format_number(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(num_bytes, 0))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _safe_json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _frame_to_table(
    frame: pd.DataFrame, preview_rows: int = PREVIEW_ROWS
) -> dict[str, Any]:
    if frame.empty:
        return {"columns": [], "rows": []}
    preview = frame.head(preview_rows).copy()
    preview = preview.where(pd.notna(preview), None)
    return {
        "columns": [str(column) for column in preview.columns],
        "rows": preview.to_dict(orient="records"),
        "rowCount": int(len(frame)),
    }


def _kv_rows(items: list[tuple[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, value in items:
        if isinstance(value, bool):
            rendered = "Yes" if value else "No"
        elif isinstance(value, (int, float)):
            rendered = _format_number(value)
        else:
            rendered = str(value)
        rows.append({"key": key, "value": rendered})
    return rows


def _numeric_values(values: dict[str, Any]) -> list[float]:
    return [
        float(value) for value in values.values() if isinstance(value, (int, float))
    ]


def _mean_score(values: dict[str, Any]) -> float | None:
    numbers = _numeric_values(values)
    if not numbers:
        return None
    return fmean(numbers)


def _dataset_artifact(dataset_item: dict[str, Any], root: Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_item["path"])
    eval_files = [Path(path) for path in dataset_item.get("eval_files", [])]
    result_files = [Path(path) for path in dataset_item.get("result_files", [])]
    latest_result = None
    system_rows: list[dict[str, Any]] = []

    if result_files:
        latest_result = sorted(result_files, key=lambda path: path.stat().st_mtime)[-1]
        try:
            entries = _safe_json_load(latest_result)
        except Exception:
            entries = []

        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            metrics = entry.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            numeric_metrics = _numeric_values(metrics)
            top_metric = None
            top_score = None
            if numeric_metrics:
                top_metric, top_score = max(
                    (
                        (metric, score)
                        for metric, score in metrics.items()
                        if isinstance(score, (int, float))
                    ),
                    key=lambda item: float(item[1]),
                )
            system_rows.append(
                {
                    "system_name": entry.get("system_name", latest_result.stem),
                    "metric_count": len(metrics),
                    "mean_score": _mean_score(metrics),
                    "best_metric": top_metric,
                    "best_score": top_score,
                    "query_rows": len(entry.get("per_query_results", []) or []),
                }
            )

    artifact_id = f"dataset::{dataset_item['name']}"
    file_rows = [
        {"type": "eval", "name": path.name, "count": 1} for path in eval_files
    ] + [{"type": "result", "name": path.name, "count": 1} for path in result_files]

    sections: list[dict[str, Any]] = [
        {
            "title": "Files",
            "kind": "table",
            "columns": ["type", "name", "count"],
            "rows": file_rows,
            "rowCount": len(file_rows),
        }
    ]

    if system_rows:
        sections.append(
            {
                "title": "Latest result systems",
                "kind": "table",
                "columns": [
                    "system_name",
                    "metric_count",
                    "mean_score",
                    "best_metric",
                    "best_score",
                    "query_rows",
                ],
                "rows": system_rows,
                "rowCount": len(system_rows),
            }
        )

    stats = [
        ("Eval files", len(eval_files)),
        ("Result files", len(result_files)),
        ("Latest result", latest_result.name if latest_result else "None"),
    ]

    chart = None
    if system_rows:
        chart = {
            "kind": "bar",
            "title": f"Average metric score by system - {dataset_item['name']}",
            "x": [row["system_name"] for row in system_rows],
            "y": [row["mean_score"] or 0.0 for row in system_rows],
        }

    return {
        "id": artifact_id,
        "group": "datasets",
        "title": f"Dataset {dataset_item['name']}",
        "subtitle": _relative_display(dataset_dir, root),
        "sourcePath": _relative_display(dataset_dir, root),
        "stats": _kv_rows(stats),
        "sections": sections,
        "chart": chart,
    }


def _result_artifact(path: Path, root: Path) -> dict[str, Any]:
    artifact_id = f"result::{path.as_posix()}"
    try:
        payload = _safe_json_load(path)
    except Exception as exc:
        return {
            "id": artifact_id,
            "group": "results",
            "title": path.name,
            "subtitle": "Could not read result file",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows([("Error", exc)]),
            "sections": [],
        }

    if not isinstance(payload, list):
        return {
            "id": artifact_id,
            "group": "results",
            "title": path.name,
            "subtitle": "Unexpected result file format",
            "sourcePath": _relative_display(path, root),
            "stats": [],
            "sections": [{"title": "Raw payload", "kind": "json", "value": payload}],
        }

    systems: list[dict[str, Any]] = []
    total_query_rows = 0
    all_metrics: set[str] = set()

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        all_metrics.update(
            {
                metric
                for metric, score in metrics.items()
                if isinstance(score, (int, float))
            }
        )
        query_rows = entry.get("per_query_results", []) or []
        total_query_rows += len(query_rows)
        numeric_items = [
            (metric, float(score))
            for metric, score in metrics.items()
            if isinstance(score, (int, float))
        ]
        best_metric = None
        best_score = None
        if numeric_items:
            best_metric, best_score = max(numeric_items, key=lambda item: item[1])
        systems.append(
            {
                "system_name": entry.get("system_name", "unknown"),
                "metric_count": len(metrics),
                "mean_score": _mean_score(metrics),
                "best_metric": best_metric,
                "best_score": best_score,
                "query_rows": len(query_rows),
            }
        )

    systems.sort(
        key=lambda row: (
            row["mean_score"] is None,
            -(row["mean_score"] or 0.0),
            row["system_name"],
        )
    )
    system_frame = pd.DataFrame(systems)
    sections = [
        {
            "title": "System summary",
            "kind": "table",
            "columns": list(system_frame.columns) if not system_frame.empty else [],
            "rows": system_frame.to_dict(orient="records")
            if not system_frame.empty
            else [],
            "rowCount": len(system_frame),
        }
    ]

    stats = [
        ("Systems", len(systems)),
        ("Metrics", len(all_metrics)),
        ("Per-query rows", total_query_rows),
    ]
    chart = None
    if systems:
        chart = {
            "kind": "bar",
            "title": f"Mean score by system - {path.stem}",
            "x": [row["system_name"] for row in systems],
            "y": [row["mean_score"] or 0.0 for row in systems],
        }

    return {
        "id": artifact_id,
        "group": "results",
        "title": path.name,
        "subtitle": f"{len(systems)} systems, {len(all_metrics)} metrics",
        "sourcePath": _relative_display(path, root),
        "stats": _kv_rows(stats),
        "sections": sections,
        "chart": chart,
    }


def _merged_artifact(path: Path, root: Path) -> dict[str, Any]:
    artifact_id = f"merged::{path.as_posix()}"
    try:
        payload = _safe_json_load(path)
        merged_map = normalize_merged_payload(payload)
    except Exception as exc:
        return {
            "id": artifact_id,
            "group": "merged",
            "title": path.name,
            "subtitle": "Could not parse merged payload",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows([("Error", exc)]),
            "sections": [],
        }

    frame = merged_results_to_frame(merged_map)
    metric_names = calculate_available_metric_set(merged_map)
    preview = _frame_to_table(frame)
    stats = [
        ("Rows", len(frame)),
        ("Metrics", len(metric_names)),
        ("Preview rows", min(PREVIEW_ROWS, len(frame))),
    ]

    metric_means: list[tuple[str, float]] = []
    if not frame.empty:
        for metric in metric_names:
            series = pd.to_numeric(frame.get(metric), errors="coerce").dropna()
            if not series.empty:
                metric_means.append((metric, float(series.mean())))
        metric_means.sort(key=lambda item: item[1], reverse=True)

    sections = [
        {
            "title": "Merged preview",
            "kind": "table",
            "columns": preview["columns"],
            "rows": preview["rows"],
            "rowCount": preview["rowCount"],
        }
    ]

    chart = None
    if metric_means:
        chart = {
            "kind": "bar",
            "title": f"Average metric value - {path.stem}",
            "x": [metric for metric, _ in metric_means],
            "y": [value for _, value in metric_means],
        }

    return {
        "id": artifact_id,
        "group": "merged",
        "title": path.name,
        "subtitle": f"{len(frame)} rows, {len(metric_names)} metrics",
        "sourcePath": _relative_display(path, root),
        "stats": _kv_rows(stats),
        "sections": sections,
        "chart": chart,
    }


def _alpha_artifact(path: Path, root: Path) -> dict[str, Any]:
    artifact_id = f"alpha::{path.as_posix()}"
    try:
        payload = _safe_json_load(path)
    except Exception as exc:
        return {
            "id": artifact_id,
            "group": "alpha",
            "title": path.name,
            "subtitle": "Could not read alpha file",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows([("Error", exc)]),
            "sections": [],
        }

    if isinstance(payload, dict):
        alpha_rows = [
            {"measure": key.removeprefix("alpha_"), "alpha": value}
            for key, value in payload.items()
            if key.startswith("alpha_") and isinstance(value, (int, float))
        ]
        if alpha_rows:
            alpha_rows.sort(key=lambda row: row["measure"])
        sections = [
            {
                "title": "Alpha summary",
                "kind": "table",
                "columns": ["measure", "alpha"],
                "rows": alpha_rows,
                "rowCount": len(alpha_rows),
            },
            {
                "title": "Raw payload",
                "kind": "json",
                "value": payload,
            },
        ]
        stats = [
            (
                "Metrics",
                len(payload.get("metrics", []))
                if isinstance(payload.get("metrics"), list)
                else 0,
            ),
            ("Sample size", payload.get("sample_size", 0)),
            ("Scale to ten", payload.get("scale_to_ten", False)),
        ]
        chart = None
        if alpha_rows:
            chart = {
                "kind": "bar",
                "title": f"Krippendorff alpha - {path.stem}",
                "x": [row["measure"] for row in alpha_rows],
                "y": [float(row["alpha"]) for row in alpha_rows],
            }
        return {
            "id": artifact_id,
            "group": "alpha",
            "title": path.name,
            "subtitle": f"Sample size {payload.get('sample_size', 0)}",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows(stats),
            "sections": sections,
            "chart": chart,
        }

    if isinstance(payload, list):
        frame = pd.DataFrame(payload)
        preview = _frame_to_table(frame)
        return {
            "id": artifact_id,
            "group": "alpha",
            "title": path.name,
            "subtitle": f"{len(frame)} rows",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows([("Rows", len(frame))]),
            "sections": [
                {
                    "title": "Preview",
                    "kind": "table",
                    "columns": preview["columns"],
                    "rows": preview["rows"],
                    "rowCount": preview["rowCount"],
                }
            ],
            "chart": None,
        }

    return {
        "id": artifact_id,
        "group": "alpha",
        "title": path.name,
        "subtitle": "Unsupported alpha payload",
        "sourcePath": _relative_display(path, root),
        "stats": [],
        "sections": [{"title": "Raw payload", "kind": "json", "value": payload}],
        "chart": None,
    }


def _proving_artifact(path: Path, root: Path) -> dict[str, Any]:
    artifact_id = f"proving::{path.as_posix()}"
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            return {
                "id": artifact_id,
                "group": "proving",
                "title": path.name,
                "subtitle": "Could not read CSV",
                "sourcePath": _relative_display(path, root),
                "stats": _kv_rows([("Error", exc)]),
                "sections": [],
            }
        preview = _frame_to_table(frame)
        return {
            "id": artifact_id,
            "group": "proving",
            "title": path.name,
            "subtitle": f"{len(frame)} rows",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows([("Rows", len(frame)), ("Columns", len(frame.columns))]),
            "sections": [
                {
                    "title": "Preview",
                    "kind": "table",
                    "columns": preview["columns"],
                    "rows": preview["rows"],
                    "rowCount": preview["rowCount"],
                }
            ],
            "chart": None,
        }

    try:
        payload = _safe_json_load(path)
    except Exception as exc:
        return {
            "id": artifact_id,
            "group": "proving",
            "title": path.name,
            "subtitle": "Could not read proving file",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows([("Error", exc)]),
            "sections": [],
        }

    if isinstance(payload, dict):
        summary_metrics = payload.get("summary_metrics", [])
        frame = (
            pd.DataFrame(summary_metrics)
            if isinstance(summary_metrics, list)
            else pd.DataFrame()
        )
        sections: list[dict[str, Any]] = []
        if not frame.empty:
            preview = _frame_to_table(frame)
            sections.append(
                {
                    "title": "Summary metrics",
                    "kind": "table",
                    "columns": preview["columns"],
                    "rows": preview["rows"],
                    "rowCount": preview["rowCount"],
                }
            )
        sections.append({"title": "Raw payload", "kind": "json", "value": payload})

        stats = [
            ("Proxy metric", payload.get("proxy_metric", "n/a")),
            ("Scale to 10", payload.get("scale_10", False)),
            ("Bootstrap iters", payload.get("bootstrap_iters", 0)),
            ("Top-k", payload.get("top_k", 0)),
        ]
        chart = None
        if not frame.empty and "pearson" in frame.columns:
            chart = {
                "kind": "bar",
                "title": f"Pearson by metric - {path.stem}",
                "x": frame.get("name", frame.index).tolist(),
                "y": pd.to_numeric(frame["pearson"], errors="coerce")
                .fillna(0.0)
                .tolist(),
            }
        return {
            "id": artifact_id,
            "group": "proving",
            "title": path.name,
            "subtitle": f"Proxy {payload.get('proxy_metric', 'n/a')}",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows(stats),
            "sections": sections,
            "chart": chart,
        }

    if isinstance(payload, list):
        frame = pd.DataFrame(payload)
        preview = _frame_to_table(frame)
        return {
            "id": artifact_id,
            "group": "proving",
            "title": path.name,
            "subtitle": f"{len(frame)} rows",
            "sourcePath": _relative_display(path, root),
            "stats": _kv_rows([("Rows", len(frame)), ("Columns", len(frame.columns))]),
            "sections": [
                {
                    "title": "Preview",
                    "kind": "table",
                    "columns": preview["columns"],
                    "rows": preview["rows"],
                    "rowCount": preview["rowCount"],
                }
            ],
            "chart": None,
        }

    return {
        "id": artifact_id,
        "group": "proving",
        "title": path.name,
        "subtitle": "Unsupported proving payload",
        "sourcePath": _relative_display(path, root),
        "stats": [],
        "sections": [{"title": "Raw payload", "kind": "json", "value": payload}],
        "chart": None,
    }


def _discover_sidecar_files(root: Path, parent_name: str) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.parent.name == parent_name
        and path.suffix.lower() in {".json", ".csv"}
    ]
    return sorted(files)


def build_payload(root: Path | None = None) -> dict[str, Any]:
    """Collect datasets and artefacts into a browser-friendly payload."""
    root = root or datasets_root()
    dataset_items = discover_dataset_dirs(root)

    datasets = [_dataset_artifact(item, root) for item in dataset_items]
    result_paths = [
        Path(path) for item in dataset_items for path in item.get("result_files", [])
    ]
    results = [_result_artifact(path, root) for path in result_paths]
    merged_files = [
        _merged_artifact(path, root) for path in discover_merged_files(root)
    ]
    alpha_files = [
        _alpha_artifact(path, root) for path in _discover_sidecar_files(root, "alpha")
    ]
    proving_files = [
        _proving_artifact(path, root)
        for path in _discover_sidecar_files(root, "proving")
    ]

    artifacts = datasets + results + merged_files + alpha_files + proving_files
    groups = {
        "datasets": [artifact["id"] for artifact in datasets],
        "results": [artifact["id"] for artifact in results],
        "merged": [artifact["id"] for artifact in merged_files],
        "alpha": [artifact["id"] for artifact in alpha_files],
        "proving": [artifact["id"] for artifact in proving_files],
        "all": [artifact["id"] for artifact in artifacts],
    }

    summary = {
        "datasetCount": len(datasets),
        "resultFileCount": len(results),
        "mergedFileCount": len(merged_files),
        "alphaFileCount": len(alpha_files),
        "provingFileCount": len(proving_files),
        "artifactCount": len(artifacts),
    }

    dataset_cards = [
        {
            "name": artifact["title"],
            "subtitle": artifact["subtitle"],
            "evalFiles": next(
                (
                    row["value"]
                    for row in artifact["stats"]
                    if row["key"] == "Eval files"
                ),
                "0",
            ),
            "resultFiles": next(
                (
                    row["value"]
                    for row in artifact["stats"]
                    if row["key"] == "Result files"
                ),
                "0",
            ),
            "latestResult": next(
                (
                    row["value"]
                    for row in artifact["stats"]
                    if row["key"] == "Latest result"
                ),
                "None",
            ),
        }
        for artifact in datasets
    ]

    return {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": _relative_display(root, root),
        "summary": summary,
        "datasetCards": dataset_cards,
        "groups": groups,
        "artifacts": artifacts,
    }


def _html_template() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>t2s-eval static dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #f6f1e8;
      --panel: #fffaf3;
      --panel-strong: #fff5e7;
      --ink: #1c2430;
      --muted: #6a6f7a;
      --accent: #c96824;
      --accent-soft: #ffe4c4;
      --border: #e8d7c4;
      --shadow: 0 16px 40px rgba(80, 52, 22, 0.11);
      --radius: 20px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 211, 160, 0.42), transparent 30%),
        radial-gradient(circle at 85% 0%, rgba(207, 114, 28, 0.16), transparent 24%),
        linear-gradient(180deg, #fffdf8 0%, var(--bg) 100%);
      font-family: "Aptos", "Segoe UI", "Noto Sans", sans-serif;
      min-height: 100vh;
    }

    .shell {
      width: min(1400px, 96vw);
      margin: 24px auto 40px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 16px;
      align-items: stretch;
      margin-bottom: 16px;
    }

    .hero-main, .hero-side, .panel, .card {
      background: rgba(255, 255, 255, 0.82);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .hero-main {
      padding: 26px;
      overflow: hidden;
      position: relative;
    }

    .hero-main::after {
      content: "";
      position: absolute;
      inset: auto -40px -90px auto;
      width: 230px;
      height: 230px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(201, 104, 36, 0.18), transparent 70%);
      pointer-events: none;
    }

    .kicker {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 0.76rem;
    }

    h1 {
      margin: 14px 0 10px;
      font-size: clamp(2rem, 3.2vw, 3.6rem);
      line-height: 1.02;
      letter-spacing: -0.04em;
      max-width: 12ch;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
      max-width: 75ch;
    }

    .hero-side {
      padding: 18px;
      display: grid;
      gap: 12px;
      align-content: start;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    .card {
      padding: 16px;
    }

    .card .label {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.74rem;
      margin-bottom: 6px;
    }

    .card .value {
      font-size: 2rem;
      line-height: 1;
      font-weight: 800;
      letter-spacing: -0.04em;
    }

    .card .hint {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }

    .panel {
      padding: 16px;
    }

    .stack {
      display: grid;
      gap: 12px;
    }

    .panel h2, .panel h3 {
      margin: 0 0 10px;
      letter-spacing: -0.02em;
    }

    .controls {
      display: grid;
      gap: 12px;
      margin-bottom: 14px;
    }

    .control {
      display: grid;
      gap: 6px;
    }

    label {
      color: var(--muted);
      font-size: 0.84rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    select, input {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      min-height: 42px;
      padding: 9px 12px;
      background: #fff;
      color: var(--ink);
    }

    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }

    .tab-button {
      border: 1px solid var(--border);
      background: #fff;
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 14px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.18s ease;
    }

    .tab-button.active {
      background: var(--accent-soft);
      color: var(--accent);
      border-color: #e3b17f;
    }

    .tab-content {
      display: none;
      animation: fade-in 0.2s ease;
    }

    .tab-content.active { display: block; }

    @keyframes fade-in {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .dataset-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }

    .mini-card {
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.7);
    }

    .mini-card .title {
      font-weight: 800;
      margin-bottom: 6px;
    }

    .mini-card .meta {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.4;
    }

    .artifact-header {
      display: grid;
      gap: 10px;
      margin-bottom: 12px;
    }

    .artifact-title {
      margin: 0;
      font-size: clamp(1.4rem, 2vw, 2rem);
      letter-spacing: -0.03em;
    }

    .artifact-subtitle {
      color: var(--muted);
      line-height: 1.5;
    }

    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .stat {
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--panel-strong);
      border: 1px solid var(--border);
      color: var(--ink);
      font-size: 0.88rem;
      font-weight: 700;
    }

    .chart {
      min-height: 360px;
      margin-bottom: 14px;
    }

    .section {
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fff;
      overflow: hidden;
      margin-bottom: 12px;
    }

    .section h4 {
      margin: 0;
      padding: 12px 14px;
      background: #fff8ef;
      border-bottom: 1px solid var(--border);
      font-size: 0.98rem;
    }

    .section-body {
      padding: 12px 14px;
      overflow: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }

    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid #f0e5d9;
      vertical-align: top;
      text-align: left;
    }

    th {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: #fffdf9;
      position: sticky;
      top: 0;
      z-index: 1;
    }

    .kv {
      display: grid;
      gap: 8px;
    }

    .kv-row {
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 12px;
      padding: 10px 12px;
      border: 1px solid #f1e3d2;
      border-radius: 12px;
      background: #fffefc;
    }

    .kv-row .key {
      font-weight: 700;
      color: var(--muted);
    }

    pre {
      margin: 0;
      padding: 12px;
      border-radius: 12px;
      background: #101521;
      color: #f7f3ec;
      overflow: auto;
      font-size: 0.86rem;
      line-height: 1.5;
    }

    .empty {
      padding: 20px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed var(--border);
      border-radius: 14px;
      background: #fff;
    }

    .note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    @media (max-width: 1100px) {
      .hero, .layout {
        grid-template-columns: 1fr;
      }

      .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 640px) {
      .shell { width: min(96vw, 100%); margin: 12px auto 28px; }
      .summary-grid { grid-template-columns: 1fr; }
      .kv-row { grid-template-columns: 1fr; }
      .hero-main { padding: 18px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-main">
        <span class="kicker">Static snapshot</span>
        <h1>t2s-eval pipeline dashboard</h1>
        <p>
          This build captures the current datasets and generated artefacts as a static
          browser-ready snapshot. It is designed for online deployment without a Python
          backend, while still letting you browse result files, merged outputs, alpha
          summaries, and proving artefacts.
        </p>
      </div>
      <div class="hero-side">
        <div>
          <strong>Generated at</strong>
          <div id="generated-at"></div>
        </div>
        <div>
          <strong>Workspace</strong>
          <div id="workspace-root"></div>
        </div>
        <div class="note">
          The dashboard content is read-only and reflects the files discovered when this
          snapshot was built.
        </div>
      </div>
    </section>

    <section class="summary-grid" id="summary-grid"></section>

    <section class="layout">
      <aside class="panel stack">
        <div>
          <h2>Browse artefacts</h2>
          <div class="controls">
            <div class="control">
              <label for="group-select">Group</label>
              <select id="group-select"></select>
            </div>
            <div class="control">
              <label for="artifact-select">Artefact</label>
              <select id="artifact-select"></select>
            </div>
          </div>
        </div>

        <div>
          <h3>Datasets</h3>
          <div id="dataset-grid" class="dataset-grid"></div>
        </div>
      </aside>

      <main class="panel">
        <div class="tabs">
          <button type="button" class="tab-button active" data-tab="artifact">Selected artefact</button>
          <button type="button" class="tab-button" data-tab="overview">Overview JSON</button>
        </div>

        <section id="tab-artifact" class="tab-content active">
          <div class="artifact-header">
            <h2 class="artifact-title" id="artifact-title"></h2>
            <div class="artifact-subtitle" id="artifact-subtitle"></div>
            <div class="stats" id="artifact-stats"></div>
            <div class="note" id="artifact-source"></div>
          </div>
          <div id="artifact-chart" class="chart"></div>
          <div id="artifact-sections"></div>
        </section>

        <section id="tab-overview" class="tab-content">
          <div class="section">
            <h4>Snapshot overview</h4>
            <div class="section-body">
              <pre id="overview-json"></pre>
            </div>
          </div>
        </section>
      </main>
    </section>
  </div>

  <script>
    window.__T2S_STATIC__ = __PAYLOAD__;

    const app = window.__T2S_STATIC__;

    const refs = {
      generatedAt: document.getElementById('generated-at'),
      workspaceRoot: document.getElementById('workspace-root'),
      summaryGrid: document.getElementById('summary-grid'),
      groupSelect: document.getElementById('group-select'),
      artifactSelect: document.getElementById('artifact-select'),
      datasetGrid: document.getElementById('dataset-grid'),
      artifactTitle: document.getElementById('artifact-title'),
      artifactSubtitle: document.getElementById('artifact-subtitle'),
      artifactStats: document.getElementById('artifact-stats'),
      artifactSource: document.getElementById('artifact-source'),
      artifactChart: document.getElementById('artifact-chart'),
      artifactSections: document.getElementById('artifact-sections'),
      overviewJson: document.getElementById('overview-json'),
      tabButtons: Array.from(document.querySelectorAll('.tab-button')),
      tabPanels: Array.from(document.querySelectorAll('.tab-content')),
    };

    const state = {
      group: 'all',
      artifactId: null,
    };

    function escapeText(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    function artifactById(id) {
      return app.artifacts.find((artifact) => artifact.id === id) || null;
    }

    function renderSummaryCards() {
      const entries = [
        { label: 'Datasets', value: app.summary.datasetCount, hint: 'folders with eval files' },
        { label: 'Result files', value: app.summary.resultFileCount, hint: 'per-dataset JSON outputs' },
        { label: 'Merged files', value: app.summary.mergedFileCount, hint: 'combined result artefacts' },
        { label: 'Alpha files', value: app.summary.alphaFileCount, hint: 'Krippendorff summaries' },
        { label: 'Proving files', value: app.summary.provingFileCount, hint: 'QCan evidence artefacts' },
      ];

      refs.summaryGrid.innerHTML = entries.map((entry) => `
        <article class="card">
          <div class="label">${escapeText(entry.label)}</div>
          <div class="value">${escapeText(entry.value)}</div>
          <div class="hint">${escapeText(entry.hint)}</div>
        </article>
      `).join('');
    }

    function renderDatasetCards() {
      if (!app.datasetCards.length) {
        refs.datasetGrid.innerHTML = '<div class="empty">No datasets were discovered.</div>';
        return;
      }

      refs.datasetGrid.innerHTML = app.datasetCards.map((card) => `
        <article class="mini-card">
          <div class="title">${escapeText(card.name)}</div>
          <div class="meta">${escapeText(card.subtitle)}</div>
          <div class="meta">Eval files: ${escapeText(card.evalFiles)}<br />Result files: ${escapeText(card.resultFiles)}<br />Latest result: ${escapeText(card.latestResult)}</div>
        </article>
      `).join('');
    }

    function renderGroupOptions() {
      const groupLabels = {
        datasets: 'Datasets',
        results: 'Result files',
        merged: 'Merged files',
        alpha: 'Alpha files',
        proving: 'Proving files',
        all: 'All artefacts',
      };

      refs.groupSelect.innerHTML = Object.entries(groupLabels).map(([value, label]) => `
        <option value="${escapeText(value)}">${escapeText(label)}</option>
      `).join('');
      refs.groupSelect.value = state.group;
    }

    function currentArtifacts() {
      const ids = app.groups[state.group] || [];
      return ids.map((id) => artifactById(id)).filter(Boolean);
    }

    function renderArtifactOptions() {
      const artifacts = currentArtifacts();
      refs.artifactSelect.innerHTML = artifacts.map((artifact) => `
        <option value="${escapeText(artifact.id)}">${escapeText(artifact.title)}</option>
      `).join('');

      if (!artifacts.length) {
        refs.artifactSelect.innerHTML = '<option value="">No artefacts available</option>';
        state.artifactId = null;
        return;
      }

      if (!artifacts.some((artifact) => artifact.id === state.artifactId)) {
        state.artifactId = artifacts[0].id;
      }

      refs.artifactSelect.value = state.artifactId;
    }

    function renderChart(artifact) {
      if (!window.Plotly || !artifact.chart || !artifact.chart.x.length) {
        refs.artifactChart.innerHTML = artifact.chart ? '<div class="empty">Chart unavailable.</div>' : '<div class="empty">This artefact does not expose a chart.</div>';
        return;
      }

      const trace = {
        type: 'bar',
        x: artifact.chart.x,
        y: artifact.chart.y,
        marker: { color: '#c96824' },
      };

      const layout = {
        title: artifact.chart.title,
        margin: { t: 50, l: 50, r: 20, b: 120 },
        height: 360,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { tickangle: -35 },
        yaxis: { gridcolor: '#eadcc7' },
      };

      Plotly.react(refs.artifactChart, [trace], layout, { responsive: true, displayModeBar: false });
    }

    function renderTable(section) {
      if (!section.rows || !section.rows.length) {
        return '<div class="empty">No rows available.</div>';
      }

      const columns = section.columns && section.columns.length
        ? section.columns
        : Object.keys(section.rows[0] || {});

      const header = `<tr>${columns.map((column) => `<th>${escapeText(column)}</th>`).join('')}</tr>`;
      const body = section.rows.map((row) => `<tr>${columns.map((column) => `<td>${escapeText(row?.[column] ?? '')}</td>`).join('')}</tr>`).join('');

      const summary = section.rowCount && section.rowCount > section.rows.length
        ? `<div class="note">Showing ${section.rows.length} of ${section.rowCount} rows.</div>`
        : '';

      return `${summary}<table>${header}${body}</table>`;
    }

    function renderSection(section) {
      const wrapper = document.createElement('section');
      wrapper.className = 'section';

      const heading = document.createElement('h4');
      heading.textContent = section.title || 'Section';
      wrapper.appendChild(heading);

      const body = document.createElement('div');
      body.className = 'section-body';

      if (section.kind === 'table') {
        body.innerHTML = renderTable(section);
      } else if (section.kind === 'kv') {
        body.innerHTML = `<div class="kv">${(section.rows || []).map((row) => `
          <div class="kv-row"><div class="key">${escapeText(row.key)}</div><div class="value">${escapeText(row.value)}</div></div>
        `).join('')}</div>`;
      } else if (section.kind === 'json') {
        body.innerHTML = `<pre>${escapeText(JSON.stringify(section.value, null, 2))}</pre>`;
      } else if (section.kind === 'text') {
        body.innerHTML = `<div>${escapeText(section.value)}</div>`;
      } else {
        body.innerHTML = '<div class="empty">Unsupported section.</div>';
      }

      wrapper.appendChild(body);
      return wrapper;
    }

    function renderSelectedArtifact() {
      const artifact = artifactById(state.artifactId) || currentArtifacts()[0] || null;
      if (!artifact) {
        refs.artifactTitle.textContent = 'No artefact selected';
        refs.artifactSubtitle.textContent = 'Nothing to show.';
        refs.artifactStats.innerHTML = '';
        refs.artifactSource.textContent = '';
        refs.artifactChart.innerHTML = '<div class="empty">No artefacts found for the selected group.</div>';
        refs.artifactSections.innerHTML = '';
        return;
      }

      state.artifactId = artifact.id;
      refs.artifactTitle.textContent = artifact.title;
      refs.artifactSubtitle.textContent = artifact.subtitle || '';
      refs.artifactSource.textContent = `Source: ${artifact.sourcePath || 'n/a'}`;
      refs.artifactStats.innerHTML = (artifact.stats || []).map((stat) => `<span class="stat">${escapeText(stat.key)}: ${escapeText(stat.value)}</span>`).join('');
      renderChart(artifact);
      refs.artifactSections.innerHTML = '';
      (artifact.sections || []).forEach((section) => {
        refs.artifactSections.appendChild(renderSection(section));
      });
      if (!artifact.sections || !artifact.sections.length) {
        refs.artifactSections.innerHTML = '<div class="empty">This artefact does not expose a tabular preview.</div>';
      }
    }

    function renderOverviewJson() {
      refs.overviewJson.textContent = JSON.stringify(app, null, 2);
    }

    function setTab(tabName) {
      refs.tabButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.tab === tabName);
      });
      refs.tabPanels.forEach((panel) => {
        panel.classList.toggle('active', panel.id === `tab-${tabName}`);
      });
    }

    function syncAndRender() {
      renderGroupOptions();
      renderArtifactOptions();
      renderSelectedArtifact();
    }

    refs.generatedAt.textContent = app.generatedAt;
    refs.workspaceRoot.textContent = app.root;
    renderSummaryCards();
    renderDatasetCards();
    renderOverviewJson();
    syncAndRender();

    refs.groupSelect.addEventListener('change', () => {
      state.group = refs.groupSelect.value;
      state.artifactId = null;
      syncAndRender();
    });

    refs.artifactSelect.addEventListener('change', () => {
      state.artifactId = refs.artifactSelect.value;
      renderSelectedArtifact();
    });

    refs.tabButtons.forEach((button) => {
      button.addEventListener('click', () => setTab(button.dataset.tab));
    });
  </script>
</body>
</html>
"""


def build_static_site(output_dir: Path | None = None, root: Path | None = None) -> Path:
    """Write the static dashboard HTML snapshot to ``output_dir``."""
    root = root or datasets_root()
    output_dir = ensure_directory(output_dir or DEFAULT_OUTPUT_DIR)
    payload = build_payload(root)
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = _html_template().replace("__PAYLOAD__", payload_json)

    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    """Generate the default static dashboard snapshot."""
    html_path = build_static_site()
    print(f"Wrote static dashboard to {html_path}")


if __name__ == "__main__":
    main()

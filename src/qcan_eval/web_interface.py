from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pipeline_core as pipeline_core
import streamlit as st  # type: ignore[import-not-found]
from pipeline_core import (
    SUPPORTED_METRICS,
    calculate_available_metric_set,
    calculate_krippendorff_alpha_from_mapping,
    datasets_root,
    discover_dataset_dirs,
    discover_merged_files,
    ensure_directory,
    merge_result_files,
    merged_mapping_to_list,
    merged_results_to_frame,
    normalize_merged_payload,
    run_metrics_job,
    save_dataframe,
    save_json,
)

st.set_page_config(page_title="QCanBlueFamilyMetric Pipeline", layout="wide")


def _as_json_bytes(data: object) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def _table_downloads(
    title: str, frame: pd.DataFrame, base_path: Path, stem: str
) -> None:
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if frame.empty:
        return
    ensure_directory(base_path)
    csv_path = base_path / f"{stem}.csv"
    json_path = base_path / f"{stem}.json"
    save_dataframe(csv_path, frame)
    save_json(json_path, frame.to_dict(orient="records"))
    col_left, col_right = st.columns(2)
    with col_left:
        st.download_button(
            f"Download {title} CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=csv_path.name,
            mime="text/csv",
        )
    with col_right:
        st.download_button(
            f"Download {title} JSON",
            data=_as_json_bytes(frame.to_dict(orient="records")),
            file_name=json_path.name,
            mime="application/json",
        )


def _dataset_options() -> list[dict[str, object]]:
    return discover_dataset_dirs(datasets_root())


def _selected_dataset(
    dataset_items: list[dict[str, object]], dataset_name: str
) -> dict[str, object] | None:
    for item in dataset_items:
        if item["name"] == dataset_name:
            return item
    return None


def _sanitize_output_stem(value: str, fallback: str) -> str:
    base = value.strip() or fallback
    base = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in base
    )
    base = base.strip("._-")
    return base or fallback


def _resolve_output_json_path(
    output_dir: Path, file_name: str, default_stem: str
) -> Path:
    base_name = Path(file_name.strip() or f"{default_stem}.json").name
    provided_stem = Path(base_name).stem
    stem = _sanitize_output_stem(provided_stem, default_stem)
    return output_dir / f"{stem}.json"


def _dataset_slug_from_merged_file(path: Path) -> str:
    stem = path.stem
    if stem.startswith("merged_"):
        stem = stem[len("merged_") :]
    return _sanitize_output_stem(stem, "dataset")


def _sync_output_filename_input(
    input_key: str, source_value: str, default_file_name: str
) -> None:
    source_key = f"_{input_key}_source"
    if st.session_state.get(source_key) != source_value:
        st.session_state[input_key] = default_file_name
        st.session_state[source_key] = source_value
    elif input_key not in st.session_state:
        st.session_state[input_key] = default_file_name


def _render_scan_tab(dataset_items: list[dict[str, object]]) -> None:
    st.subheader("Local dataset scan")
    if not dataset_items:
        st.warning("No datasets with an eval directory were found under datasets/.")
        return

    rows = [
        {
            "dataset": item["name"],
            "eval_files": len(item["eval_files"]),
            "result_files": len(item["result_files"]),
        }
        for item in dataset_items
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect dataset", [item["name"] for item in dataset_items])
    dataset = _selected_dataset(dataset_items, selected)
    if dataset is None:
        return

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Eval files**")
        for path in dataset["eval_files"]:
            st.write(Path(path).name)
    with col_right:
        st.markdown("**Result files**")
        for path in dataset["result_files"]:
            st.write(Path(path).name)


def _render_metrics_tab(dataset_items: list[dict[str, object]]) -> None:
    st.subheader("Calculate metrics")
    if not dataset_items:
        st.info("No datasets were found.")
        return

    dataset_name = st.selectbox(
        "Dataset", [item["name"] for item in dataset_items], key="metrics_dataset"
    )
    dataset = _selected_dataset(dataset_items, dataset_name)
    if dataset is None:
        return

    eval_files = [Path(path) for path in dataset["eval_files"]]
    selected_eval_files = st.multiselect(
        "Eval JSONL files",
        eval_files,
        default=eval_files,
        format_func=lambda path: path.name,
        key="metrics_eval_files",
    )
    metric_names = st.multiselect(
        "Metrics",
        SUPPORTED_METRICS,
        default=[
            "answerset_f1",
            "bleu",
            "exact_match_spinach",
            "qcan-bleu-strict",
            "qcan-bleu-flex",
            "rouge_4",
            "qcan-rouge-4-strict",
            "qcan-rouge-4-flex",
        ],
        key="metrics_names",
    )
    endpoint = st.text_input(
        "Execution backend endpoint",
        value="http://localhost:8887/",
        key="metrics_endpoint",
    )
    col_left, col_right = st.columns(2)
    with col_left:
        parallel = st.checkbox("Parallel", value=True, key="metrics_parallel")
        per_query = st.checkbox("Per query", value=True, key="metrics_per_query")
        verbose = st.checkbox("Verbose", value=True, key="metrics_verbose")
    with col_right:
        safe_limit = st.number_input(
            "Safe limit",
            min_value=0,
            max_value=1000000,
            value=25000,
            step=1000,
            key="metrics_safe_limit",
        )

    output_dir = ensure_directory(
        Path(dataset["path"]) / "results" / "streamlit" / "metrics"
    )
    st.caption(f"Outputs will be written to {output_dir}")

    if st.button("Run metric calculation", type="primary"):
        if not selected_eval_files:
            st.error("Select at least one eval file.")
            return
        if not metric_names:
            st.error("Select at least one metric.")
            return
        with st.spinner("Running metric pipeline..."):
            try:
                result = run_metrics_job(
                    dataset=dataset_name,
                    jsonl_evals=selected_eval_files,
                    metric_names=metric_names,
                    execution_backend_endpoint_url=endpoint,
                    parallel=parallel,
                    per_query=per_query,
                    safe_limit=int(safe_limit),
                    verbose=verbose,
                )
            except ImportError as exc:
                st.error(str(exc))
                st.info(
                    "Install or expose t2smetrics in the active environment, then rerun this step."
                )
                return
            st.success("Metric calculation finished.")
            st.json(result)


def _render_merge_tab(dataset_items: list[dict[str, object]]) -> None:
    st.subheader("Merge result files")
    if not dataset_items:
        st.info("No datasets were found.")
        return

    selected_scope = st.selectbox(
        "Dataset scope",
        ["All datasets", *[item["name"] for item in dataset_items]],
        key="merge_scope",
    )
    if selected_scope == "All datasets":
        candidate_files = [
            Path(path) for item in dataset_items for path in item["result_files"]
        ]
    else:
        dataset = _selected_dataset(dataset_items, selected_scope)
        candidate_files = (
            [Path(path) for path in dataset["result_files"]] if dataset else []
        )

    selected_files = st.multiselect(
        "Result JSON files",
        candidate_files,
        default=candidate_files,
        format_func=lambda path: path.name,
        key="merge_files",
    )
    output_dir_text = st.text_input(
        "Merged output folder",
        value=str(datasets_root() / "_streamlit" / "merged"),
        key="merge_output_dir",
    )
    output_dir = ensure_directory(Path(output_dir_text))
    default_merge_stem = (
        "merged_all_datasets"
        if selected_scope == "All datasets"
        else f"merged_{selected_scope}"
    )
    merge_output_name = st.text_input(
        "Merged file name", value=f"{default_merge_stem}.json", key="merge_output_name"
    )

    if st.button("Merge selected results", type="primary"):
        if not selected_files:
            st.error("Select at least one results file.")
            return
        with st.spinner("Merging result files..."):
            merged = merge_result_files(selected_files)
            merged_list = merged_mapping_to_list(merged)
            merged_path = _resolve_output_json_path(
                output_dir, merge_output_name, default_merge_stem
            )
            save_json(merged_path, merged_list)
            frame = merged_results_to_frame(merged)
            st.success(f"Merged {len(selected_files)} files into {merged_path}")
            st.write(
                f"Rows: {len(frame)}, metrics: {len(frame.columns) - 1 if not frame.empty else 0}"
            )
            _table_downloads("merged results", frame, output_dir, merged_path.stem)


def _render_alpha_tab() -> None:
    st.subheader("Krippendorff alpha")
    merged_files = discover_merged_files(datasets_root())
    if not merged_files:
        st.info("No merged JSON files were found. Run the merge step first.")
        return

    selected_file = st.selectbox(
        "Merged results file",
        merged_files,
        format_func=lambda path: str(path),
        key="alpha_file",
    )
    with open(selected_file, encoding="utf-8") as handle:
        merged_map = normalize_merged_payload(json.load(handle))
    metric_names = calculate_available_metric_set(merged_map)
    selected_metrics = st.multiselect(
        "Metric subset",
        metric_names,
        default=metric_names[: min(5, len(metric_names))],
        key="alpha_metrics",
    )
    scale_to_ten = st.checkbox(
        "Scale scores to 0-10 before alpha", value=True, key="alpha_scale"
    )
    output_dir_text = st.text_input(
        "Alpha output folder",
        value=str(selected_file.parent / "alpha"),
        key="alpha_output_dir",
    )
    output_dir = ensure_directory(Path(output_dir_text))
    default_alpha_stem = f"alpha_{_dataset_slug_from_merged_file(selected_file)}"
    default_alpha_file_name = f"{default_alpha_stem}.json"
    _sync_output_filename_input(
        "alpha_output_name", str(selected_file), default_alpha_file_name
    )
    alpha_output_name = st.text_input(
        "Alpha file name", value=default_alpha_file_name, key="alpha_output_name"
    )

    if st.button("Compute Krippendorff alpha", type="primary"):
        if len(selected_metrics) < 2:
            st.error("Pick at least two metrics.")
            return
        try:
            alpha_result = calculate_krippendorff_alpha_from_mapping(
                merged_map, selected_metrics, scale_to_ten=scale_to_ten
            )
        except Exception as exc:
            st.error(str(exc))
            return
        alpha_df = pd.DataFrame(
            [
                {"measure": "nominal", "alpha": alpha_result["alpha_nominal"]},
                {"measure": "interval", "alpha": alpha_result["alpha_interval"]},
                {"measure": "ordinal", "alpha": alpha_result["alpha_ordinal"]},
                {"measure": "ratio", "alpha": alpha_result["alpha_ratio"]},
            ]
        )
        alpha_path = _resolve_output_json_path(
            output_dir, alpha_output_name, default_alpha_stem
        )
        save_json(alpha_path, alpha_result)
        st.success(f"Saved alpha summary to {alpha_path}")
        st.dataframe(alpha_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download alpha JSON",
            data=_as_json_bytes(alpha_result),
            file_name=alpha_path.name,
            mime="application/json",
        )


def _render_proving_tab() -> None:
    st.subheader("Prove QCan value")
    merged_files = discover_merged_files(datasets_root())
    if not merged_files:
        st.info("No merged JSON files were found. Run the merge step first.")
        return

    selected_file = st.selectbox(
        "Merged results file",
        merged_files,
        format_func=lambda path: str(path),
        key="proof_file",
    )
    with open(selected_file, encoding="utf-8") as handle:
        merged_map = normalize_merged_payload(json.load(handle))
    available_metrics = calculate_available_metric_set(merged_map)
    proxy_metric = st.selectbox(
        "Proxy metric",
        available_metrics,
        index=available_metrics.index("answerset_f1")
        if "answerset_f1" in available_metrics
        else 0,
        key="proof_proxy",
    )
    compare_metrics = st.multiselect(
        "Comparison metrics",
        [metric for metric in available_metrics if metric != proxy_metric],
        default=[
            metric
            for metric in [
                "qcan-bleu-strict",
                "qcan-bleu-flex",
                "bleu",
                "exact_match_spinach",
                "rouge_4",
                "qcan-rouge-4-strict",
                "qcan-rouge-4-flex",
            ]
            if metric in available_metrics and metric != proxy_metric
        ],
        key="proof_compare_metrics",
    )
    col_left, col_right = st.columns(2)
    with col_left:
        scale_10 = st.checkbox(
            "Scale scores to 0-10 before analysis", value=False, key="proof_scale"
        )
        bootstrap_iters = st.number_input(
            "Bootstrap iterations",
            min_value=100,
            max_value=50000,
            value=3000,
            step=100,
            key="proof_bootstrap",
        )
    with col_right:
        seed = st.number_input(
            "Seed", min_value=0, max_value=100000, value=7, step=1, key="proof_seed"
        )
        top_k = st.number_input(
            "Top-k qualitative examples",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="proof_top_k",
        )
    output_dir_text = st.text_input(
        "Proving output folder",
        value=str(selected_file.parent / "proving"),
        key="proof_output_dir",
    )
    output_dir = ensure_directory(Path(output_dir_text))
    default_proof_stem = f"qcan_value_{_sanitize_output_stem(proxy_metric, 'proxy')}_{_dataset_slug_from_merged_file(selected_file)}"
    default_proof_file_name = f"{default_proof_stem}.json"
    _sync_output_filename_input(
        "proof_output_name", f"{selected_file}|{proxy_metric}", default_proof_file_name
    )
    proof_output_name = st.text_input(
        "Proving file name", value=default_proof_file_name, key="proof_output_name"
    )

    if st.button("Run proving analysis", type="primary"):
        metric_df, comparison_df, examples_df, summaries = (
            pipeline_core.compute_metric_matrix(
                merged_map,
                proxy_metric=proxy_metric,
                compare_metrics=compare_metrics,
                scale_10=scale_10,
                bootstrap_iters=int(bootstrap_iters),
                seed=int(seed),
                top_k=int(top_k),
            )
        )
        analysis = {
            "proxy_metric": proxy_metric,
            "scale_10": scale_10,
            "bootstrap_iters": int(bootstrap_iters),
            "seed": int(seed),
            "top_k": int(top_k),
            "summary_metrics": [summary.__dict__ for summary in summaries],
        }
        analysis_path = _resolve_output_json_path(
            output_dir, proof_output_name, default_proof_stem
        )
        save_json(analysis_path, analysis)
        st.success(f"Saved proving output to {analysis_path}")
        st.markdown("**Metric ranking**")
        _table_downloads(
            "metric ranking", metric_df, output_dir, f"{analysis_path.stem}_metrics"
        )
        st.markdown("**Significance checks**")
        _table_downloads(
            "significance checks",
            comparison_df,
            output_dir,
            f"{analysis_path.stem}_comparisons",
        )
        st.markdown("**Qualitative examples**")
        _table_downloads(
            "qualitative examples",
            examples_df,
            output_dir,
            f"{analysis_path.stem}_examples",
        )


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(num_bytes, 0))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _extract_case_from_jsonl(jsonl_path: str, case_id: str) -> dict[str, object] | None:
    """Extract a specific case from a JSONL file by ID."""
    try:
        with open(jsonl_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("id") == case_id:
                    return item
    except Exception:
        pass
    return None


def _parse_query_id(query_id: str) -> tuple[str, str] | None:
    """Parse query_id to extract JSONL path and case_id.

    Query ID format: {dataset}_{system_name}_with_JSONL_{path}_{case_id}
    Example: db25_System with JSONL /path/WSE_db25:52-en

    Returns (jsonl_path, case_id) or None if parsing fails.
    """
    parts = query_id.split("_")
    if len(parts) < 3:
        return None

    case_id_part = query_id.split("_")[-1]
    jsonl_start = query_id.find(" /")
    if jsonl_start == -1:
        return None

    remaining = query_id[jsonl_start + 1 :]
    jsonl_path = remaining.rsplit("_", 1)[0]

    return (jsonl_path, case_id_part)


def _render_case_details(query_id: str) -> None:
    """Render case details when clicking on a failing case."""
    parse_result = _parse_query_id(query_id)
    if parse_result is None:
        st.error(f"Invalid query_id format: {query_id}")
        return

    jsonl_path, case_id_part = parse_result

    if not Path(jsonl_path).exists():
        st.error(f"JSONL file not found: {jsonl_path}")
        return

    case = _extract_case_from_jsonl(jsonl_path, case_id_part)
    if case is None:
        st.error(f"Case with ID {case_id_part} not found in {jsonl_path}")
        return

    st.markdown("---")
    st.markdown(f"### Case Details: {case_id_part}")

    question = case.get("question", "")
    golden = case.get("golden", "")
    generated = case.get("generated", "")

    col_left, col_right = st.columns(2)

    with col_left:
        if question:
            st.markdown("**Question:**")
            st.text_area(
                "Question",
                value=question,
                height=100,
                disabled=True,
                label_visibility="collapsed",
            )

    with col_right:
        st.markdown("**Metadata:**")
        metadata = f"ID: {case_id_part}\nJSONL: {Path(jsonl_path).name}"
        if "order_matters" in case:
            metadata += f"\nOrder Matters: {case['order_matters']}"
        st.text_area(
            "Metadata",
            value=metadata,
            height=100,
            disabled=True,
            label_visibility="collapsed",
        )

    st.markdown("**Golden Query:**")
    st.text_area(
        "Golden Query",
        value=golden,
        height=150,
        disabled=True,
        label_visibility="collapsed",
    )

    st.markdown("**Generated Query:**")
    st.text_area(
        "Generated Query",
        value=generated,
        height=150,
        disabled=True,
        label_visibility="collapsed",
    )


def _render_proving_examples_table(frame: pd.DataFrame) -> None:
    """Render proving examples with interactive buttons for failing cases."""
    st.dataframe(frame, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Inspect Failing Cases")

    failing_cases = frame[frame["kind"] == "fails"]
    if failing_cases.empty:
        st.info("No failing cases found.")
        return

    for idx, row in failing_cases.iterrows():
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            if st.button("View Details", key=f"case_btn_{idx}"):
                st.session_state[f"selected_case_{idx}"] = row["query_id"]

        with col_info:
            query_id = row["query_id"]
            case_id = query_id.split("_")[-1]
            st.write(f"**{case_id}** - Delta: {row['delta']:.4f}")

    for idx, row in failing_cases.iterrows():
        if st.session_state.get(f"selected_case_{idx}"):
            _render_case_details(row["query_id"])
            if st.button("Close Details", key=f"case_close_{idx}"):
                st.session_state[f"selected_case_{idx}"] = None


def _history_file_preview(files: list[Path], label: str, key_prefix: str) -> None:
    st.markdown(f"### {label}")
    if not files:
        st.info("No files found.")
        return

    selected = st.selectbox(
        f"Select {label.lower()} file",
        files,
        format_func=lambda path: str(path),
        key=f"history_{key_prefix}_file",
    )

    metadata_col, download_col = st.columns([2, 1])
    with metadata_col:
        stat = selected.stat()
        st.caption(
            f"Size: {_format_bytes(stat.st_size)} | Modified: {pd.Timestamp(stat.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')}"
        )
    with download_col:
        with open(selected, "rb") as handle:
            raw = handle.read()
        st.download_button(
            f"Download {selected.name}",
            data=raw,
            file_name=selected.name,
            mime="application/octet-stream",
            key=f"history_{key_prefix}_download",
        )

    if selected.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(selected)
        except Exception as exc:
            st.error(f"Could not read CSV file: {exc}")
            return
        st.dataframe(frame, use_container_width=True, hide_index=True)
        return

    if selected.suffix.lower() == ".json":
        try:
            with open(selected, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            st.error(f"Could not read JSON file: {exc}")
            return

        if (
            isinstance(payload, list)
            and payload
            and all(isinstance(item, dict) for item in payload)
        ):
            frame = pd.DataFrame(payload)
            if (
                "kind" in frame.columns
                and "query_id" in frame.columns
                and "_examples" in selected.name
            ):
                _render_proving_examples_table(frame)
            else:
                st.dataframe(frame, use_container_width=True, hide_index=True)
        elif isinstance(payload, dict):
            st.json(payload)
        else:
            st.json(payload)
        return

    st.info("Preview is available for JSON and CSV files.")


def _render_history_tab(dataset_items: list[dict[str, object]]) -> None:
    st.subheader("Past results explorer")
    root = datasets_root()

    past_calc_files = sorted(
        {
            Path(path)
            for item in dataset_items
            for path in item["result_files"]
            if Path(path).is_file()
        }
    )
    past_calc_files.extend(sorted(root.glob("*/results/streamlit/metrics/*.json")))
    past_calc_files = sorted({path for path in past_calc_files if path.is_file()})

    merged_files = sorted(
        {*discover_merged_files(root), *root.glob("_streamlit/merged/merged_*.csv")}
    )
    alpha_files = sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.parent.name == "alpha"
            and path.suffix.lower() in {".json", ".csv"}
        ]
    )
    proving_files = sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.parent.name == "proving"
            and path.suffix.lower() in {".json", ".csv"}
        ]
    )

    total = (
        len(past_calc_files) + len(merged_files) + len(alpha_files) + len(proving_files)
    )
    st.caption(
        f"Found {total} files | Calculations: {len(past_calc_files)} | Merged: {len(merged_files)} | Alpha: {len(alpha_files)} | Proving: {len(proving_files)}"
    )

    _history_file_preview(past_calc_files, "Past calculations", "calc")
    _history_file_preview(merged_files, "Merged results", "merged")
    _history_file_preview(alpha_files, "Krippendorff alpha", "alpha")
    _history_file_preview(proving_files, "Prove QCan value", "proving")


def main() -> None:
    dataset_items = _dataset_options()
    st.title("QCanBlueFamilyMetric pipeline")
    st.write(
        "Scan local datasets, run metric calculations, merge result files, compute Krippendorff alpha, inspect QCan value evidence, and browse past outputs."
    )
    tab_scan, tab_metrics, tab_merge, tab_alpha, tab_prove, tab_history = st.tabs(
        [
            "Scan datasets",
            "Calculate metrics",
            "Merge results",
            "Krippendorff alpha",
            "Prove QCan value",
            "Past results",
        ]
    )
    with tab_scan:
        _render_scan_tab(dataset_items)
    with tab_metrics:
        _render_metrics_tab(dataset_items)
    with tab_merge:
        _render_merge_tab(dataset_items)
    with tab_alpha:
        _render_alpha_tab()
    with tab_prove:
        _render_proving_tab()
    with tab_history:
        _render_history_tab(dataset_items)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze DAG v6.8 with a source-256 Profiler-entry boundary."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_dag_v68_target_assisted_profiler_entry import (
    REPO,
    atomic_csv,
    atomic_json,
    atomic_text,
    build_entry_node,
    checked,
    configured_path,
    metric,
    replay,
    sha256,
    verify_v67_seal,
)


DEFAULT_CONFIG = REPO / (
    "case_224gpu_pp14_cp2_a2a/config/dag_v68_source256_profiler_entry_2026w36.toml"
)


def source_entry_table(event_windows_path: Path, phase_events_path: Path) -> pd.DataFrame:
    windows = pd.read_csv(checked(event_windows_path))
    phases = pd.read_csv(checked(phase_events_path))
    required_windows = {"iteration", "rank", "start_ns", "end_ns"}
    required_phases = {"iteration", "rank", "observed_start_ns", "observed_end_ns"}
    if not required_windows.issubset(windows.columns):
        raise ValueError("source256 event-window schema changed")
    if not required_phases.issubset(phases.columns):
        raise ValueError("source256 F/B phase schema changed")
    rows: list[dict[str, Any]] = []
    for iteration in sorted(phases["iteration"].astype(int).unique()):
        phase = phases[phases["iteration"].eq(iteration)]
        window = windows[windows["iteration"].eq(iteration)]
        if phase["rank"].nunique() != 256 or window["rank"].nunique() != 256:
            raise ValueError(f"source256 rank grid incomplete: iteration {iteration}")
        profiler_start_ns = int(window["start_ns"].min())
        profiler_end_ns = int(window["end_ns"].max())
        phase_start_ns = int(phase["observed_start_ns"].min())
        phase_end_ns = int(phase["observed_end_ns"].max())
        if not profiler_start_ns <= phase_start_ns < phase_end_ns <= profiler_end_ns:
            raise ValueError(f"source256 boundary order changed: iteration {iteration}")
        rows.append({
            "iteration": iteration,
            "profiler_start_ns": profiler_start_ns,
            "first_fb_envelope_start_ns": phase_start_ns,
            "last_fb_envelope_end_ns": phase_end_ns,
            "profiler_end_ns": profiler_end_ns,
            "source_rank_count": 256,
            "profiler_entry_to_phase_ms": (phase_start_ns - profiler_start_ns) / 1e6,
            "profiler_step_ms": (profiler_end_ns - profiler_start_ns) / 1e6,
            "phase_envelope_ms": (phase_end_ns - phase_start_ns) / 1e6,
        })
    return pd.DataFrame(rows)


def contiguous_window_scan(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Post-seal diagnostic only; never use the selected window to update parameters."""
    ordered = evaluation.sort_values("iteration").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for count in range(3, len(ordered) + 1):
        for offset in range(0, len(ordered) - count + 1):
            window = ordered.iloc[offset:offset + count]
            rows.append({
                "start_iteration": int(window.iloc[0]["iteration"]),
                "end_iteration": int(window.iloc[-1]["iteration"]),
                "iteration_count": count,
                "iterations": ",".join(str(int(value)) for value in window["iteration"]),
                "profiler_mape_pct": float(window["profiler_abs_error_pct"].mean()),
                "profiler_mae_ms": float(window["profiler_error_ms"].abs().mean()),
                "profiler_bias_ms": float(window["profiler_error_ms"].mean()),
                "selection_role": "post_seal_target_window_diagnostic_only",
                "eligible_as_independent_test_set": False,
            })
    return pd.DataFrame(rows).sort_values(
        ["profiler_mape_pct", "iteration_count", "start_iteration"]
    ).reset_index(drop=True)


def window_summary(row: pd.Series) -> dict[str, int | float]:
    return {
        "start_iteration": int(row["start_iteration"]),
        "end_iteration": int(row["end_iteration"]),
        "iteration_count": int(row["iteration_count"]),
        "profiler_mape_pct": float(row["profiler_mape_pct"]),
        "profiler_bias_ms": float(row["profiler_bias_ms"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    source_iteration = int(config["source_parameter"]["iteration"])
    expected_entry_ms = float(config["source_parameter"]["expected_duration_ms"])
    reference_iterations = tuple(int(value) for value in config["source_parameter"]["reference_iterations"])
    target_iterations = tuple(int(value) for value in config["target_evaluation"]["iterations"])
    early_iterations = tuple(int(value) for value in config["target_evaluation"]["early_development"])
    late_iterations = tuple(int(value) for value in config["target_evaluation"]["late_development"])
    if reference_iterations != tuple(range(60, 101, 5)):
        raise ValueError("source256 reference iteration grid changed")
    if target_iterations != reference_iterations:
        raise ValueError("target evaluator iteration grid changed")
    if early_iterations != (60, 65, 70, 75, 80) or late_iterations != (85, 90, 95, 100):
        raise ValueError("target development reporting split changed")

    v67 = configured_path(config["inputs"]["v67_run_dir"])
    v67_seal_path = checked(v67 / "predictions/prediction_seal.json")
    verify_v67_seal(v67_seal_path)
    v67_nodes_path = checked(v67 / "predictions/dag_v67_nodes.csv.gz")
    v67_edges_path = checked(v67 / "predictions/dag_v67_edges.csv.gz")
    v67_contract_path = checked(v67 / "prediction_contract.json")
    source_event_windows_path = checked(configured_path(config["inputs"]["source_event_windows"]))
    source_phase_events_path = checked(configured_path(config["inputs"]["source_phase_events"]))
    source_entries = source_entry_table(source_event_windows_path, source_phase_events_path)
    source_entries = source_entries[source_entries["iteration"].isin(reference_iterations)].copy()
    if source_entries["iteration"].astype(int).tolist() != list(reference_iterations):
        raise ValueError("source256 entry reference rows changed")
    selected = source_entries[source_entries["iteration"].eq(source_iteration)]
    if len(selected) != 1:
        raise ValueError("source256 selected entry row missing")
    entry_ms = float(selected.iloc[0]["profiler_entry_to_phase_ms"])
    if abs(entry_ms - expected_entry_ms) > 1e-9:
        raise ValueError(f"source256 frozen entry changed: {entry_ms} != {expected_entry_ms}")
    entry_ns = int(round(entry_ms * 1e6))

    nodes = pd.read_csv(v67_nodes_path, low_memory=False)
    edges = pd.read_csv(v67_edges_path, low_memory=False)
    entry_id = str(config["model"]["entry_node_id"])
    if entry_id in set(nodes["node_id"].astype(str)):
        raise ValueError("v6.8 entry node already exists")
    roots = nodes[~nodes["node_id"].isin(set(edges["dst"].astype(str)))].copy()
    if len(roots) != 16 or not roots["node_id"].str.match(
        r"^r(?:[0-9]|1[0-5]):s0:l(?:[0-9]|1[0-5]):fwd0:q0:start$"
    ).all():
        raise ValueError(f"v6.7 PP0 root contract changed: {len(roots)}")
    entry = build_entry_node(list(nodes.columns), roots.iloc[0], entry_id, entry_ns)
    entry.update({
        "cost_status": "SOURCE256_FROZEN",
        "cost_source": "source256_trace_iteration_85",
        "timing_source": "source256_profiler_to_first_fb_boundary",
        "source_parameter_key": "source256:iteration85:profiler_entry_to_phase",
    })
    nodes = pd.concat([nodes, pd.DataFrame([entry], columns=nodes.columns)], ignore_index=True)
    entry_edges = pd.DataFrame([
        {
            "case_id": str(config["target"]["case_id"]),
            "src": entry_id,
            "dst": str(root.node_id),
            "edge_type": "profiler_entry_dependency",
            "tensor_key": "none",
            "dependency_source": "source256_trace_boundary_frozen",
        }
        for root in roots.itertuples(index=False)
    ], columns=edges.columns)
    edges = pd.concat([edges, entry_edges], ignore_index=True)
    nodes, critical = replay(nodes, edges, "iteration:completion_join")
    completion_ns = int(nodes.loc[
        nodes["node_id"].eq("iteration:completion_join"), "predicted_end_ns"
    ].item())
    if not bool(nodes.loc[nodes["node_id"].eq(entry_id), "on_critical_path"].item()):
        raise ValueError("v6.8 source entry is not on the critical path")
    components = nodes[[
        "compute_exposed_ns_model", "compute_overlap_ns_model", "network_service_ns_model",
        "software_sync_ns_model", "framework_residual_ns_model",
    ]].fillna(0).sum(axis=1).round().astype("int64")
    if not components.eq(nodes["duration_ns"].astype("int64")).all():
        raise ValueError("v6.8 component conservation failed")

    v67_contract = json.loads(v67_contract_path.read_text(encoding="utf-8"))
    v67_prediction = v67_contract["prediction"]
    v67_raw_ms = float(v67_prediction["raw_graph_ms"])
    raw_ms = completion_ns / 1e6
    if abs(raw_ms - v67_raw_ms - entry_ms) > 0.002:
        raise ValueError("v6.8 graph delta is not exactly the frozen source entry")
    post_graph_reconciliation_ms = float(v67_prediction["profiler_step_ms"]) - v67_raw_ms
    profiler_ms = raw_ms + post_graph_reconciliation_ms
    outer_framework_ms = float(v67_prediction["outer_framework_ms"])
    training_ms = profiler_ms + outer_framework_ms
    mfu_pct = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"])
        * 1e12
        * training_ms / 1000.0
    )

    output = configured_path(config["outputs"]["output_dir"])
    predictions_dir = output / "predictions"
    evaluator = output / "evaluator_only"
    calibration = output / "calibration"
    logs = output / "logs"
    for directory in (predictions_dir, evaluator, calibration, logs):
        directory.mkdir(parents=True, exist_ok=True)
    node_path = predictions_dir / "dag_v68_nodes.csv.gz"
    edge_path = predictions_dir / "dag_v68_edges.csv.gz"
    critical_path = predictions_dir / "dag_v68_critical_path.csv"
    prediction_path = predictions_dir / "method_predictions.csv"
    contract_path = output / "prediction_contract.json"
    access_path = output / "input_access_audit.json"
    parameter_path = calibration / "profiler_entry_parameter.json"
    source_entry_path = calibration / "source256_profiler_entry_60_100.csv"
    atomic_csv(node_path, nodes, compression="gzip")
    atomic_csv(edge_path, edges, compression="gzip")
    atomic_csv(critical_path, critical)
    prediction_rows = pd.DataFrame([{
        "iteration": iteration,
        "split": "target_development_early" if iteration in early_iterations else "target_development_late",
        "method": "DAG v6.8 frozen source256 profiler entry",
        "predicted_profiler_step_ms": profiler_ms,
        "predicted_outer_framework_ms": outer_framework_ms,
        "predicted_training_step_ms": training_ms,
        "predicted_mfu_pct": mfu_pct,
        "target_timing_read": False,
        "entry_parameter_source": "source256_trace_iteration_85",
    } for iteration in target_iterations])
    atomic_csv(prediction_path, prediction_rows)
    parameter = {
        "schema": "dag-v6.8-source256-profiler-entry-parameter-v1",
        "status": "FROZEN_SOURCE256_TRACE",
        "source_case": "256gpu_pp16_cp2_a2a",
        "source_iteration": source_iteration,
        "duration_ms": entry_ms,
        "display_duration_s": float(config["source_parameter"]["display_duration_s"]),
        "semantic_boundary": "global earliest ProfilerStep start to global earliest 16-rank F/B envelope start",
        "selection": str(config["source_parameter"]["selection"]),
        "target_case_access": "none_before_prediction_seal",
    }
    atomic_json(parameter_path, parameter)
    atomic_csv(source_entry_path, source_entries)
    atomic_json(contract_path, {
        "schema": "dag-v6.8-source256-profiler-entry-contract-v1",
        "status": "FROZEN_SOURCE256_ENTRY_PREDICTION",
        "formal_blind_claim_allowed": False,
        "blind_extrapolation_claim_allowed": False,
        "reason_formal_blind_claim_blocked": "target224 scenario was previously inspected during method development",
        "target_timing_read": False,
        "final_step_residual_fit": False,
        "entry_node": {
            "node_id": entry_id,
            "duration_ms": entry_ms,
            "display_duration_s": float(config["source_parameter"]["display_duration_s"]),
            "root_edge_count": int(len(entry_edges)),
            "timing_component": "profiler_entry_to_first_phase",
            "source_case": "256gpu_pp16_cp2_a2a",
            "source_iteration": source_iteration,
        },
        "prediction": {
            "raw_graph_ms": raw_ms,
            "post_graph_reconciliation_ms": post_graph_reconciliation_ms,
            "profiler_step_ms": profiler_ms,
            "outer_framework_ms": outer_framework_ms,
            "training_step_ms": training_ms,
            "mfu_pct": mfu_pct,
        },
    })
    atomic_json(access_path, {
        "schema": "dag-v6.8-source256-input-access-v1",
        "status": "PASS_SOURCE256_ONLY_BEFORE_SEAL",
        "source_model": "sealed DAG v6.7",
        "source_timing_files_read": [str(source_event_windows_path), str(source_phase_events_path)],
        "source_iteration_used_for_parameter": source_iteration,
        "target_timing_read_before_seal": False,
        "target_files_read_before_seal": [],
        "target_fields_read_before_seal": [],
        "final_step_residual_read": False,
    })

    seal_path = predictions_dir / "prediction_seal.json"
    sealed_artifacts = [
        node_path, edge_path, critical_path, prediction_path, contract_path,
        access_path, parameter_path, source_entry_path,
    ]
    atomic_json(seal_path, {
        "schema": "dag-v6.8-source256-prediction-seal-v1",
        "status": "SEALED_BEFORE_V68_TARGET_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sealed_artifacts
        ],
    })

    # Target evaluator access begins only after the source-only prediction seal.
    truth_path = checked(configured_path(config["inputs"]["target_ground_truth"]))
    truth = pd.read_csv(truth_path)
    truth = truth[truth["iteration"].isin(target_iterations)].copy()
    evaluation = prediction_rows.merge(
        truth[["iteration", "actual_profiler_step_ms", "actual_training_step_ms"]],
        on="iteration",
        validate="one_to_one",
    )
    evaluation["profiler_error_ms"] = evaluation["predicted_profiler_step_ms"] - evaluation["actual_profiler_step_ms"]
    evaluation["profiler_abs_error_pct"] = evaluation["profiler_error_ms"].abs() / evaluation["actual_profiler_step_ms"] * 100.0
    evaluation["training_error_ms"] = evaluation["predicted_training_step_ms"] - evaluation["actual_training_step_ms"]
    evaluation["training_abs_error_pct"] = evaluation["training_error_ms"].abs() / evaluation["actual_training_step_ms"] * 100.0
    evaluation_path = evaluator / "iteration_evaluation.csv"
    metrics_path = evaluator / "metrics.json"
    evaluation_access_path = evaluator / "evaluation_access_audit.json"
    atomic_csv(evaluation_path, evaluation)
    window_scan = contiguous_window_scan(evaluation)
    window_scan_path = evaluator / "post60_contiguous_window_diagnostic.csv"
    atomic_csv(window_scan_path, window_scan)
    early = evaluation[evaluation["iteration"].isin(early_iterations)]
    late = evaluation[evaluation["iteration"].isin(late_iterations)]
    source_validation = source_entries[source_entries["iteration"].ne(source_iteration)].copy()
    source_validation["predicted_entry_ms"] = entry_ms
    source_entry_metric = metric(source_validation, "predicted_entry_ms", "profiler_entry_to_phase_ms")
    metrics = {
        "schema": "dag-v6.8-source256-entry-target-development-evaluation-v1",
        "status": "PASS_SOURCE256_FROZEN_ENTRY_TARGET_DEVELOPMENT",
        "scope": "source256-frozen parameter; target224 development evaluator because target scenario was previously inspected",
        "formal_blind_claim_allowed": False,
        "target_parameter_updates": 0,
        "source_entry_validation_excluding_selected_iteration": source_entry_metric,
        "target_development_all": {
            "profiler": metric(evaluation, "predicted_profiler_step_ms", "actual_profiler_step_ms"),
            "training": metric(evaluation, "predicted_training_step_ms", "actual_training_step_ms"),
        },
        "target_development_early": {
            "profiler": metric(early, "predicted_profiler_step_ms", "actual_profiler_step_ms"),
            "training": metric(early, "predicted_training_step_ms", "actual_training_step_ms"),
        },
        "target_development_late": {
            "profiler": metric(late, "predicted_profiler_step_ms", "actual_profiler_step_ms"),
            "training": metric(late, "predicted_training_step_ms", "actual_training_step_ms"),
        },
        "comparison": {
            "v67_target_development_all_profiler_mape_pct": 11.14314621034797,
            "v68_target_development_all_profiler_mape_pct": metric(
                evaluation, "predicted_profiler_step_ms", "actual_profiler_step_ms"
            )["mape_pct"],
        },
        "post60_window_diagnostic": {
            "status": "POST_SEAL_DIAGNOSTIC_NOT_TEST_SET_SELECTION",
            "selection_warning": "choosing a window after reading target errors is retrospective cherry-picking and cannot redefine the official 60-100 evaluation set",
            "best_contiguous_3_point": window_summary(window_scan[
                window_scan["iteration_count"].eq(3)
            ].iloc[0]),
            "best_contiguous_5_point": window_summary(window_scan[
                window_scan["iteration_count"].eq(5)
            ].iloc[0]),
            "best_strictly_after_60_contiguous_3_point": window_summary(window_scan[
                window_scan["iteration_count"].eq(3) & window_scan["start_iteration"].gt(60)
            ].iloc[0]),
            "worst_contiguous_3_point": window_summary(window_scan[
                window_scan["iteration_count"].eq(3)
            ].sort_values("profiler_mape_pct", ascending=False).iloc[0]),
            "worst_contiguous_5_point": window_summary(window_scan[
                window_scan["iteration_count"].eq(5)
            ].sort_values("profiler_mape_pct", ascending=False).iloc[0]),
        },
    }
    atomic_json(metrics_path, metrics)
    atomic_json(evaluation_access_path, {
        "schema": "dag-v6.8-source256-target-evaluation-access-v1",
        "status": "TARGET224_OPENED_AFTER_SOURCE256_PREDICTION_SEAL",
        "seal_path": str(seal_path.resolve()),
        "target_iterations": list(target_iterations),
        "parameter_updates": 0,
    })

    report_path = output / "DAG_V68_SOURCE256_PROFILER_ENTRY.md"
    legacy_report_path = output / "DAG_V68_TARGET_ASSISTED_PROFILER_ENTRY.md"
    reproduction_path = output / "reproduction_command.txt"
    build_log_path = logs / "build.log"
    all_mape = metrics["target_development_all"]["profiler"]["mape_pct"]
    late_mape = metrics["target_development_late"]["profiler"]["mape_pct"]
    atomic_text(report_path, f"""# DAG v6.8：256卡冻结Profiler入口节点

- 状态：`{metrics['status']}`
- 冻结来源：256卡 Trace iteration `{source_iteration}`
- 显式入口节点：`{entry_ms:.6f} ms`（页面显示 `{entry_ms / 1000.0:.4f} s`）
- v6.7 Profiler预测：`{float(v67_prediction['profiler_step_ms']):.6f} ms`
- v6.8 Profiler预测：`{profiler_ms:.6f} ms`
- 224卡开发集60–100 Profiler MAPE：`{all_mape:.6f}%`
- 224卡后半段85–100 Profiler MAPE：`{late_mape:.6f}%`
- 事后连续3点最低误差窗口：iter60–70，MAPE `{metrics['post60_window_diagnostic']['best_contiguous_3_point']['profiler_mape_pct']:.6f}%`
- 事后连续5点最低误差窗口：iter60–80，MAPE `{metrics['post60_window_diagnostic']['best_contiguous_5_point']['profiler_mape_pct']:.6f}%`
- 事后连续3点最高误差窗口：iter90–100，MAPE `{metrics['post60_window_diagnostic']['worst_contiguous_3_point']['profiler_mape_pct']:.6f}%`

v6.8在16个PP0 lane根节点之前增加一个由256卡Trace冻结的入口节点。预测seal之前不读取224卡时序，也不拟合最终Step残差；224卡仅在seal之后用于开发集评估。

上述窗口是在读取224卡误差后选择的诊断结果，不得重新命名为测试集。由于224场景此前已被人工反复分析，本结果仍不标记为独立盲测。
""")
    atomic_text(legacy_report_path, """# 已替换：DAG v6.8 target-assisted入口

原1.605579s目标辅助入口已被固定的source256入口版本替换。当前权威说明见 `DAG_V68_SOURCE256_PROFILER_ENTRY.md`；旧目录名只为保持已分享URL兼容。
""")
    atomic_text(reproduction_path, (
        ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
        "build_dag_v68_source256_profiler_entry.py --config "
        "case_224gpu_pp14_cp2_a2a/config/dag_v68_source256_profiler_entry_2026w36.toml\n"
    ))
    atomic_text(build_log_path, "\n".join([
        f"status={metrics['status']}",
        f"source_iteration={source_iteration}",
        f"entry_ms={entry_ms:.6f}",
        f"v67_profiler_ms={float(v67_prediction['profiler_step_ms']):.6f}",
        f"v68_profiler_ms={profiler_ms:.6f}",
        f"target_development_all_profiler_mape_pct={all_mape:.6f}",
        f"target_development_late_profiler_mape_pct={late_mape:.6f}",
        "target_timing_read_before_seal=false",
        "target_parameter_updates=0",
    ]) + "\n")

    provenance_path = output / "provenance.json"
    inputs = [
        config_path, v67_seal_path, v67_nodes_path, v67_edges_path, v67_contract_path,
        source_event_windows_path, source_phase_events_path, truth_path,
    ]
    outputs = [
        *sealed_artifacts, seal_path, evaluation_path, window_scan_path, metrics_path, evaluation_access_path,
        report_path, legacy_report_path, reproduction_path, build_log_path,
    ]
    atomic_json(provenance_path, {
        "schema": "dag-v6.8-source256-profiler-entry-provenance-v1",
        "status": metrics["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
    })
    artifact_manifest_path = output / "artifact_manifest.json"
    manifest_artifacts = [
        config_path, Path(__file__).resolve(), parameter_path, source_entry_path,
        contract_path, access_path, seal_path, evaluation_path, window_scan_path, metrics_path,
        evaluation_access_path, report_path, reproduction_path, build_log_path,
        provenance_path,
    ]
    atomic_json(artifact_manifest_path, {
        "schema": "dag-v6.8-source256-profiler-entry-artifact-manifest-v1",
        "status": metrics["status"],
        "entry_duration_ms": entry_ms,
        "source_case": "256gpu_pp16_cp2_a2a",
        "source_iteration": source_iteration,
        "target_timing_read_before_seal": False,
        "artifacts": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in manifest_artifacts
        ],
    })
    print(json.dumps({
        "status": metrics["status"],
        "output_dir": str(output),
        "source_iteration": source_iteration,
        "entry_ms": entry_ms,
        "predicted_profiler_ms": profiler_ms,
        "target_development_all_profiler_mape_pct": all_mape,
        "target_development_late_profiler_mape_pct": late_mape,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

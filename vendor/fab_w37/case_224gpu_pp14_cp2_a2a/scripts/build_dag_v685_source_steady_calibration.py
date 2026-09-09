#!/usr/bin/env python3
"""Refit available DAG costs on a source-only steady window and evaluate after seal."""

from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_dag_v61_kernel_calibration import parameter_frame
from build_dag_v681_causal_backward_trigger import (
    REPO,
    atomic_csv,
    atomic_json,
    atomic_text,
    component_conserved,
    sha256,
    verify_seal,
)
from build_dag_v682_stage_aware_pp_gradient import (
    SOURCE_COMPONENTS,
    TARGET_COMPONENTS,
    apply_source_gradient_wall,
    apply_target_gradient_wall,
    fit_gradient_wall,
)
from build_dag_v683_causal_program_order import topology_fingerprint
from build_dag_v68_target_assisted_profiler_entry import checked, configured_path, metric, replay


DEFAULT_CONFIG = REPO / (
    "case_224gpu_pp14_cp2_a2a/config/"
    "dag_v685_source_steady_calibration_2026w36.toml"
)
PHYSICAL_KEYS = (
    "phase", "layer_type", "layer_context", "execution_scope", "semantic_slot"
)


def coefficient_of_variation(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.mean(values)


def select_steady_suffix(
    boundaries: pd.DataFrame,
    candidates: tuple[int, ...],
    minimum_points: int,
    end_iteration: int,
) -> tuple[tuple[int, ...], pd.DataFrame]:
    indexed = boundaries.set_index("iteration")
    if not set(candidates).issubset(set(indexed.index.astype(int))):
        raise ValueError("source profiler boundary grid is incomplete")
    rows = []
    for start_index in range(0, len(candidates) - minimum_points + 1):
        window = candidates[start_index:]
        if window[-1] != end_iteration or len(window) < minimum_points:
            continue
        values = [float(indexed.loc[iteration, "profiler_step_ms"]) for iteration in window]
        rows.append({
            "start_iteration": window[0],
            "end_iteration": window[-1],
            "point_count": len(window),
            "profiler_mean_ms": statistics.mean(values),
            "profiler_median_ms": statistics.median(values),
            "profiler_std_ms": statistics.stdev(values),
            "profiler_cv_pct": 100.0 * coefficient_of_variation(values),
            "profiler_range_ms": max(values) - min(values),
            "end_to_start_pct": 100.0 * (values[-1] - values[0]) / values[0],
            "iterations": "|".join(map(str, window)),
        })
    table = pd.DataFrame(rows).sort_values(
        ["profiler_cv_pct", "point_count"], ascending=[True, False], kind="stable"
    )
    if table.empty:
        raise ValueError("no eligible steady source window")
    selected = tuple(int(value) for value in str(table.iloc[0]["iterations"]).split("|"))
    return selected, table


def physical_parameter_lookup(parameters: pd.DataFrame) -> dict[tuple[str, ...], int]:
    selected = parameters[
        parameters["parameter_view"].eq("window_split")
        & parameters["cost_group"].eq("__physical_slot_total__")
        & parameters["autograd_phase"].eq("physical_mixed")
    ]
    if selected.duplicated(list(PHYSICAL_KEYS)).any():
        raise ValueError("steady physical parameter keys are not unique")
    return {
        tuple(str(getattr(row, column)) for column in PHYSICAL_KEYS):
        int(row.median_active_union_ns)
        for row in selected.itertuples(index=False)
    }


def apply_steady_compute(
    nodes: pd.DataFrame,
    parameters: pd.DataFrame,
    v54_nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = nodes.copy()
    lookup = physical_parameter_lookup(parameters)
    v54_duration = {
        str(row.node_id): int(row.duration_ns)
        for row in v54_nodes[["node_id", "duration_ns"]].itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for index, node in output.iterrows():
        raw_key = str(node.get("source_parameter_key", ""))
        if not raw_key or raw_key == "nan":
            continue
        key = tuple(raw_key.split("|"))
        if key not in lookup:
            continue
        steady_compute_ns = lookup[key]
        old_duration_ns = int(node["duration_ns"])
        old_exposed_ns = int(node["compute_exposed_ns_model"])
        old_overlap_ns = int(node["compute_overlap_ns_model"])
        source_v54_id = str(node.get("source_v54_node_id", ""))
        if old_exposed_ns > 0:
            noncompute_floor_ns = int(v54_duration.get(source_v54_id, 0))
            new_duration_ns = max(steady_compute_ns, noncompute_floor_ns)
            output.at[index, "compute_exposed_ns_model"] = steady_compute_ns
            output.at[index, "framework_residual_ns_model"] = (
                new_duration_ns - steady_compute_ns
            )
            output.at[index, "duration_ns"] = new_duration_ns
            component = "compute_exposed"
        elif old_overlap_ns > 0:
            noncompute_floor_ns = 0
            new_duration_ns = steady_compute_ns
            output.at[index, "compute_overlap_ns_model"] = steady_compute_ns
            output.at[index, "duration_ns"] = new_duration_ns
            component = "compute_overlap"
        else:
            continue
        output.at[index, "timing_source"] = "source256_trace_iterations_85_100_median"
        rows.append({
            "node_id": str(node["node_id"]),
            "component": component,
            "source_parameter_key": raw_key,
            "old_duration_ns": old_duration_ns,
            "old_compute_ns": old_exposed_ns + old_overlap_ns,
            "steady_compute_ns": steady_compute_ns,
            "noncompute_floor_ns": noncompute_floor_ns,
            "new_duration_ns": new_duration_ns,
            "duration_delta_ns": new_duration_ns - old_duration_ns,
        })
    frame = pd.DataFrame(rows).sort_values(["component", "node_id"], kind="stable")
    if frame.empty:
        raise ValueError("steady compute parameters did not bind to target nodes")
    return output, frame


def render_html(payload: dict[str, Any]) -> str:
    p = payload["prediction"]
    m = payload["metrics"]
    s = payload["steady_window"]
    c = payload["calibration"]
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DAG v6.8.5 · 256卡稳态校准</title><style>:root{{--bg:#07111d;--panel:#0e1b2c;--line:#2b4059;--text:#e9f2fb;--muted:#93a8bd;--ok:#55d6be;--warn:#f5a524;--accent:#24c8ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:28px}}h1{{margin:4px 0}}h2{{margin:0 0 8px}}.eyebrow{{color:var(--ok);font-size:12px;font-weight:800;letter-spacing:.12em}}.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}}.card,.panel{{padding:15px;background:var(--panel);border:1px solid var(--line)}}.k{{font-size:12px;color:var(--muted)}}.v{{font-size:23px;font-weight:800}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}.flow{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.node{{padding:10px 13px;background:#091827;border:1px solid var(--line)}}.arrow{{color:var(--accent);font-size:20px}}.note{{border-left:4px solid var(--warn);background:#1b1b25;padding:11px 13px;margin-top:12px}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}</style></head><body><main><div class="eyebrow">DAG MFU · v6.8.5 SOURCE256 STEADY CALIBRATION</div><h1>用256卡稳定区间校准，再验证224卡稳定区间</h1><p class="muted">稳定区间只由256卡实测总时间波动选择；依赖图完全继承v6.8.4。</p><div class="cards"><div class="card"><div class="k">256卡校准轮次</div><div class="v">85–100</div><div class="muted">{s['source_point_count']}个点，波动{s['source_cv_pct']:.3f}%</div></div><div class="card"><div class="k">224卡验证轮次</div><div class="v">85–100</div><div class="muted">{s['target_point_count']}个点，波动{s['target_cv_pct']:.3f}%</div></div><div class="card"><div class="k">预测训练采样时间</div><div class="v">{p['profiler_step_ms']/1000:.3f}s</div><div class="muted">224卡实测均值{m['actual_profiler_mean_ms']/1000:.3f}s</div></div><div class="card"><div class="k">稳态验证平均误差</div><div class="v">{m['profiler_mape_pct']:.3f}%</div><div class="muted">平均低估{-m['profiler_bias_ms']/1000:.3f}s</div></div></div><section class="panel"><h2>校准内容</h2><table><tr><th>参数</th><th>256卡稳态输入</th><th>更新规模</th></tr><tr><td>计算节点成本</td><td>85、90、95、100轮次的中位数</td><td>{c['compute_parameter_rows']}项参数，命中{c['compute_node_updates']}个节点</td></tr><tr><td>流水线梯度传输与软件完成</td><td>按接收阶段、卡内位置和微批次取中位数</td><td>{c['pp_gradient_parameter_rows']}项参数，更新{c['target_pp_gradient_updates']}条目标边</td></tr><tr><td>训练轮次入口</td><td>四个轮次入口中位数</td><td>{c['entry_ms']:.3f}ms</td></tr><tr><td>源图未分类残差</td><td>稳态总时间中位数减源图回放</td><td>目标按微批次数比例折算{p['target_reconciliation_ms']:.3f}ms</td></tr></table></section><section class="panel"><h2>执行边界</h2><div class="flow"><div class="node">读取256卡85–100</div><div class="arrow">→</div><div class="node">冻结参数与预测</div><div class="arrow">→</div><div class="node">封存预测文件</div><div class="arrow">→</div><div class="node">读取224卡85–100评分</div></div><div class="note"><b>结论：</b>相同v6.8.4拓扑在原稳态口径下误差为{m['parent_profiler_mape_pct']:.3f}%，重新校准后为{m['profiler_mape_pct']:.3f}%，改善{m['improvement_pp']:.3f}个百分点；但仍低估约{-m['profiler_bias_ms']/1000:.3f}秒。说明稳定轮次校准减少了运行漂移影响，却没有解决256卡到224卡的成本迁移缺口。</div></section><script>window.DAG_V685={data};</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    parent = configured_path(config["inputs"]["v684_run_dir"])
    verify_seal(
        parent / "predictions/prediction_seal.json",
        "SEALED_BEFORE_V684_TARGET_EVALUATOR_ACCESS",
    )
    parent_nodes_path = checked(parent / "predictions/dag_v684_nodes.csv.gz")
    parent_edges_path = checked(parent / "predictions/dag_v684_edges.csv.gz")
    parent_source_nodes_path = checked(parent / "source_replay/dag_v684_source_nodes.csv.gz")
    parent_contract_path = checked(parent / "prediction_contract.json")
    topology_lock_path = checked(parent / "predictions/dependency_topology_lock.json")
    observations_path = checked(configured_path(config["inputs"]["v61_observations"]))
    source_pp_path = checked(configured_path(config["inputs"]["source_pp_trace_events"]))
    source_boundaries_path = checked(configured_path(config["inputs"]["source_profiler_boundaries"]))
    source_edges_path = checked(configured_path(config["inputs"]["source_edges"]))
    v54_nodes_path = checked(configured_path(config["inputs"]["v54_noncompute_nodes"]))

    boundaries = pd.read_csv(source_boundaries_path)
    candidates = tuple(int(value) for value in config["selection"]["candidate_iterations"])
    selected, window_table = select_steady_suffix(
        boundaries,
        candidates,
        int(config["selection"]["minimum_window_points"]),
        int(config["selection"]["window_must_end_at"]),
    )
    configured_selected = tuple(
        int(value) for value in config["selection"]["selected_source_calibration"]
    )
    if selected != configured_selected:
        raise ValueError(f"source-only steady selector chose {selected}, not {configured_selected}")

    observations = pd.read_csv(observations_path)
    steady_observations = observations[observations["iteration"].isin(selected)].copy()
    if set(steady_observations["iteration"].astype(int)) != set(selected):
        raise ValueError("steady compute observation split is incomplete")
    if steady_observations["source_path"].astype(str).str.contains("224gpu").any():
        raise ValueError("target path leaked into source compute observations")
    compute_parameters = parameter_frame(steady_observations)
    compute_parameters["fit_source"] = (
        "256gpu_iterations_85_100_lane0_all_pp_stages"
    )

    parent_nodes = pd.read_csv(parent_nodes_path, low_memory=False)
    edges = pd.read_csv(parent_edges_path, low_memory=False)
    v54_nodes = pd.read_csv(v54_nodes_path, low_memory=False)
    nodes, compute_updates = apply_steady_compute(parent_nodes, compute_parameters, v54_nodes)

    source_pp_events = pd.read_csv(source_pp_path)
    pp_parameters, pp_samples = fit_gradient_wall(source_pp_events, selected)
    pp_parameters["parameter_source"] = "source256_trace_iterations_85_100"
    nodes, target_pp_updates = apply_target_gradient_wall(
        nodes,
        pp_parameters,
        [int(value) for value in config["model"]["source_receiver_stage_for_target"]],
    )
    steady_boundaries = boundaries[boundaries["iteration"].isin(selected)].copy()
    entry_ms = float(steady_boundaries["profiler_entry_to_phase_ms"].median())
    entry_mask = nodes["kind"].eq("framework_entry")
    if int(entry_mask.sum()) != 1:
        raise ValueError("target framework-entry node is not unique")
    entry_ns = int(round(entry_ms * 1e6))
    nodes.loc[entry_mask, "duration_ns"] = entry_ns
    nodes.loc[entry_mask, "framework_residual_ns_model"] = entry_ns
    nodes.loc[entry_mask, "timing_source"] = "source256_iterations_85_100_entry_median"
    nodes, critical = replay(nodes, edges, config["model"]["target_completion_node_id"])
    if not component_conserved(nodes, TARGET_COMPONENTS):
        raise ValueError("target component conservation failed")
    target_raw_ms = float(nodes.loc[
        nodes["node_id"].eq(config["model"]["target_completion_node_id"]),
        "predicted_end_ns",
    ].item()) / 1e6

    source_nodes = pd.read_csv(parent_source_nodes_path, low_memory=False)
    source_edges = pd.read_csv(source_edges_path, low_memory=False)
    source_nodes, source_pp_updates = apply_source_gradient_wall(source_nodes, pp_parameters)
    source_nodes, source_critical = replay(
        source_nodes, source_edges, config["model"]["source_completion_node_id"]
    )
    if not component_conserved(source_nodes, SOURCE_COMPONENTS):
        raise ValueError("source component conservation failed")
    source_raw_ms = float(source_nodes.loc[
        source_nodes["node_id"].eq(config["model"]["source_completion_node_id"]),
        "predicted_end_ns",
    ].item()) / 1e6
    source_profiler_reference_ms = float(steady_boundaries["profiler_step_ms"].median())
    source_reconciliation_ms = source_profiler_reference_ms - source_raw_ms
    if source_reconciliation_ms < 0:
        raise ValueError("steady source graph exceeds source profiler reference")
    target_reconciliation_scale = (
        int(config["target"]["microbatches"]) / int(config["source"]["microbatches"])
    )
    target_reconciliation_ms = source_reconciliation_ms * target_reconciliation_scale
    profiler_ms = target_raw_ms + target_reconciliation_ms

    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
    topology_lock = json.loads(topology_lock_path.read_text(encoding="utf-8"))
    topology_sha = topology_fingerprint(edges)
    expected_topology_sha = str(topology_lock["child_topology_sha256"])
    if topology_sha != expected_topology_sha:
        raise ValueError("v6.8.5 changed the frozen v6.8.4 dependency topology")
    outer_ms = float(parent_contract["prediction"]["outer_framework_ms"])
    training_ms = profiler_ms + outer_ms
    mfu_pct = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"])
        * 1e12
        * training_ms
        / 1000.0
    )

    output = configured_path(config["outputs"]["output_dir"])
    prediction_dir = output / "predictions"
    calibration_dir = output / "calibration"
    source_dir = output / "source_replay"
    evaluator_dir = output / "evaluator_only"
    log_dir = output / "logs"
    for directory in (prediction_dir, calibration_dir, source_dir, evaluator_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    nodes_out = prediction_dir / "dag_v685_nodes.csv.gz"
    edges_out = prediction_dir / "dag_v685_edges.csv.gz"
    critical_out = prediction_dir / "dag_v685_critical_path.csv"
    methods_out = prediction_dir / "method_predictions.csv"
    compute_parameters_out = calibration_dir / "source256_steady_compute_parameters.csv"
    compute_updates_out = calibration_dir / "target_compute_node_updates.csv"
    pp_parameters_out = calibration_dir / "source256_steady_pp_gradient_parameters.csv"
    pp_samples_out = calibration_dir / "source256_steady_pp_gradient_samples.csv"
    target_pp_out = calibration_dir / "target_pp_gradient_updates.csv"
    window_table_out = calibration_dir / "source256_steady_window_scan.csv"
    source_nodes_out = source_dir / "dag_v685_source_nodes.csv.gz"
    source_critical_out = source_dir / "dag_v685_source_critical_path.csv"
    source_pp_out = source_dir / "source_pp_gradient_updates.csv"
    contract_out = output / "prediction_contract.json"
    access_out = output / "input_access_audit.json"
    command_out = output / "reproduction_command.txt"
    atomic_csv(nodes_out, nodes, compression="gzip")
    atomic_csv(edges_out, edges, compression="gzip")
    atomic_csv(critical_out, critical)
    atomic_csv(methods_out, pd.DataFrame([{
        "iteration": iteration,
        "method": "DAG v6.8.5 source256 steady calibration",
        "predicted_profiler_step_ms": profiler_ms,
        "predicted_outer_framework_ms": outer_ms,
        "predicted_training_step_ms": training_ms,
        "predicted_mfu_pct": mfu_pct,
        "target_timing_read": False,
    } for iteration in config["selection"]["target_evaluation"]]))
    atomic_csv(compute_parameters_out, compute_parameters)
    atomic_csv(compute_updates_out, compute_updates)
    atomic_csv(pp_parameters_out, pp_parameters)
    atomic_csv(pp_samples_out, pp_samples)
    atomic_csv(target_pp_out, target_pp_updates)
    atomic_csv(window_table_out, window_table)
    atomic_csv(source_nodes_out, source_nodes, compression="gzip")
    atomic_csv(source_critical_out, source_critical)
    atomic_csv(source_pp_out, source_pp_updates)
    atomic_json(contract_out, {
        "schema": "dag-v6.8.5-source-steady-calibration-contract-v1",
        "status": "SOURCE256_STEADY_PREDICTION_SEALED_BEFORE_TARGET_ACCESS",
        "version": "v6.8.5",
        "parent_version": "v6.8.4",
        "selected_source_iterations": list(selected),
        "target_timing_read_before_seal": False,
        "target_parameter_updates": 0,
        "dependency_topology_sha256": topology_sha,
        "prediction": {
            "source_raw_graph_ms": source_raw_ms,
            "source_profiler_reference_median_ms": source_profiler_reference_ms,
            "source_reconciliation_ms": source_reconciliation_ms,
            "target_raw_graph_ms": target_raw_ms,
            "target_reconciliation_scale": target_reconciliation_scale,
            "target_reconciliation_ms": target_reconciliation_ms,
            "profiler_step_ms": profiler_ms,
            "outer_framework_ms": outer_ms,
            "training_step_ms": training_ms,
            "mfu_pct": mfu_pct,
        },
    })
    atomic_json(access_out, {
        "schema": "dag-v6.8.5-input-access-v1",
        "status": "PASS_SOURCE256_ONLY_BEFORE_SEAL",
        "source_iterations_read": list(selected),
        "source_files_read_before_seal": [
            str(path.resolve()) for path in (
                observations_path, source_pp_path, source_boundaries_path,
                parent_nodes_path, parent_edges_path, parent_source_nodes_path,
                source_edges_path, v54_nodes_path,
            )
        ],
        "target_timing_files_read_before_seal": [],
        "target_parameter_updates": 0,
    })
    atomic_text(command_out, (
        ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
        "build_dag_v685_source_steady_calibration.py --config "
        "case_224gpu_pp14_cp2_a2a/config/"
        "dag_v685_source_steady_calibration_2026w36.toml\n"
    ))
    seal_out = prediction_dir / "prediction_seal.json"
    sealed_paths = [
        nodes_out, edges_out, critical_out, methods_out, compute_parameters_out,
        compute_updates_out, pp_parameters_out, pp_samples_out, target_pp_out,
        window_table_out, source_nodes_out, source_critical_out, source_pp_out,
        contract_out, access_out, command_out,
    ]
    atomic_json(seal_out, {
        "schema": "dag-v6.8.5-prediction-seal-v1",
        "status": "SEALED_BEFORE_V685_TARGET_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sealed_paths
        ],
    })

    target_truth_path = checked(configured_path(config["inputs"]["target_ground_truth"]))
    truth = pd.read_csv(target_truth_path)
    target_iterations = [int(value) for value in config["selection"]["target_evaluation"]]
    truth = truth[truth["iteration"].isin(target_iterations)][
        ["iteration", "actual_profiler_step_ms", "actual_training_step_ms"]
    ]
    methods = pd.read_csv(methods_out)
    evaluation = methods.merge(truth, on="iteration", validate="one_to_one")
    evaluation["profiler_error_ms"] = (
        evaluation["predicted_profiler_step_ms"] - evaluation["actual_profiler_step_ms"]
    )
    evaluation["profiler_abs_error_pct"] = (
        evaluation["profiler_error_ms"].abs() / evaluation["actual_profiler_step_ms"] * 100.0
    )
    evaluation["training_error_ms"] = (
        evaluation["predicted_training_step_ms"] - evaluation["actual_training_step_ms"]
    )
    evaluation["training_abs_error_pct"] = (
        evaluation["training_error_ms"].abs() / evaluation["actual_training_step_ms"] * 100.0
    )
    profiler_metric = metric(
        evaluation, "predicted_profiler_step_ms", "actual_profiler_step_ms"
    )
    training_metric = metric(
        evaluation, "predicted_training_step_ms", "actual_training_step_ms"
    )
    parent_eval = pd.read_csv(parent / "evaluator_only/iteration_evaluation.csv")
    parent_eval = parent_eval[parent_eval["iteration"].isin(target_iterations)]
    parent_mape = float(parent_eval["profiler_abs_error_pct"].mean())
    target_values = evaluation["actual_profiler_step_ms"].astype(float).tolist()
    selected_row = window_table[window_table["iterations"].eq("|".join(map(str, selected)))].iloc[0]
    payload = {
        "schema": "dag-v6.8.5-source-steady-evaluation-v1",
        "status": "PASS_SOURCE_STEADY_TARGET_DEVELOPMENT_EVALUATION",
        "formal_blind_claim_allowed": False,
        "target_parameter_updates": 0,
        "dependency_topology_sha256": topology_sha,
        "steady_window": {
            "source_iterations": list(selected),
            "source_point_count": len(selected),
            "source_cv_pct": float(selected_row["profiler_cv_pct"]),
            "source_range_ms": float(selected_row["profiler_range_ms"]),
            "target_iterations": target_iterations,
            "target_point_count": len(target_iterations),
            "target_cv_pct": 100.0 * coefficient_of_variation(target_values),
            "target_range_ms": max(target_values) - min(target_values),
            "selection_used_target_timing": False,
        },
        "calibration": {
            "compute_parameter_rows": len(physical_parameter_lookup(compute_parameters)),
            "compute_node_updates": len(compute_updates),
            "compute_duration_delta_ms": float(compute_updates["duration_delta_ns"].sum() / 1e6),
            "pp_gradient_parameter_rows": len(pp_parameters),
            "target_pp_gradient_updates": len(target_pp_updates),
            "entry_ms": entry_ms,
        },
        "prediction": json.loads(contract_out.read_text(encoding="utf-8"))["prediction"],
        "metrics": {
            "profiler_mape_pct": profiler_metric["mape_pct"],
            "profiler_bias_ms": profiler_metric["bias_ms"],
            "training_mape_pct": training_metric["mape_pct"],
            "training_bias_ms": training_metric["bias_ms"],
            "actual_profiler_mean_ms": float(evaluation["actual_profiler_step_ms"].mean()),
            "parent_profiler_mape_pct": parent_mape,
            "improvement_pp": parent_mape - profiler_metric["mape_pct"],
        },
    }
    evaluation_out = evaluator_dir / "iteration_evaluation.csv"
    metrics_out = evaluator_dir / "metrics.json"
    html_out = evaluator_dir / "dag_v685_source_steady_calibration.html"
    evaluator_access_out = evaluator_dir / "evaluation_access_audit.json"
    report_out = output / "DAG_V685_SOURCE_STEADY_CALIBRATION.md"
    log_out = log_dir / "build.log"
    provenance_out = output / "provenance.json"
    atomic_csv(evaluation_out, evaluation)
    atomic_json(metrics_out, payload)
    atomic_text(html_out, render_html(payload))
    atomic_json(evaluator_access_out, {
        "schema": "dag-v6.8.5-evaluator-access-v1",
        "status": "PASS_TARGET_READ_AFTER_SEAL",
        "prediction_seal": str(seal_out.resolve()),
        "target_ground_truth_read_after_seal": str(target_truth_path.resolve()),
        "target_iterations": target_iterations,
        "target_parameter_updates": 0,
    })
    atomic_text(report_out, f"""# DAG v6.8.5：256卡稳态校准

- 256卡稳态校准：`{list(selected)}`，Profiler CV `{float(selected_row['profiler_cv_pct']):.6f}%`。
- 224卡稳态验证：`{target_iterations}`，Profiler CV `{payload['steady_window']['target_cv_pct']:.6f}%`。
- 预测Profiler：`{profiler_ms:.6f} ms`；224卡实测均值：`{payload['metrics']['actual_profiler_mean_ms']:.6f} ms`。
- 224卡稳态Profiler MAPE：`{profiler_metric['mape_pct']:.6f}%`，平均偏差`{profiler_metric['bias_ms']:.6f} ms`。
- 父版v6.8.4在相同稳态验证集的MAPE：`{parent_mape:.6f}%`；改善`{parent_mape-profiler_metric['mape_pct']:.6f}`个百分点。
- 依赖拓扑保持：`{topology_sha}`；目标参数更新`0`。

该版本只重新估计当前已有且能由256卡稳态数据支持的计算槽、PP梯度、入口和源图残差。结果仍低估约`{-profiler_metric['bias_ms']:.3f} ms`，说明主要跨场景成本缺口不能靠挑选稳定轮次消除。224卡结果属于封存后的开发验证，不宣称独立blind validation。
""")
    atomic_text(log_out, "\n".join([
        "status=PASS_SOURCE_STEADY_TARGET_DEVELOPMENT_EVALUATION",
        f"source_iterations={','.join(map(str, selected))}",
        f"target_iterations={','.join(map(str, target_iterations))}",
        f"dependency_topology_sha256={topology_sha}",
        f"predicted_profiler_ms={profiler_ms:.6f}",
        f"profiler_mape_pct={profiler_metric['mape_pct']:.6f}",
        f"profiler_bias_ms={profiler_metric['bias_ms']:.6f}",
        "target_parameter_updates=0",
    ]) + "\n")
    all_inputs = [
        config_path, Path(__file__).resolve(), parent_nodes_path, parent_edges_path,
        parent_source_nodes_path, parent_contract_path, topology_lock_path,
        observations_path, source_pp_path, source_boundaries_path, source_edges_path,
        v54_nodes_path, target_truth_path,
    ]
    all_outputs = sealed_paths + [
        seal_out, evaluation_out, metrics_out, html_out, evaluator_access_out,
        report_out, log_out,
    ]
    atomic_json(provenance_out, {
        "schema": "dag-v6.8.5-source-steady-provenance-v1",
        "status": payload["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_parameter_updates": 0,
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in all_inputs
        ],
        "outputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in all_outputs
        ],
    })
    print(json.dumps({
        "status": payload["status"],
        "html": str(html_out.resolve()),
        "source_iterations": list(selected),
        "target_iterations": target_iterations,
        "predicted_profiler_ms": profiler_ms,
        "target_profiler_mape_pct": profiler_metric["mape_pct"],
        "target_profiler_bias_ms": profiler_metric["bias_ms"],
        "parent_same_window_mape_pct": parent_mape,
        "improvement_pp": parent_mape - profiler_metric["mape_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

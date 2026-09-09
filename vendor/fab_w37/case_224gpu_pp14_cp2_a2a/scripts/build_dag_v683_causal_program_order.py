#!/usr/bin/env python3
"""Build v6.8.3 without fitting DAG dependencies to observed wall gaps.

The v6.8.2 edge set is frozen.  This patch removes the duplicated B-to-B
observed gap from the local program-order branch while retaining all existing
program-order, autograd and PP-gradient prerequisites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_dag_v681_causal_backward_trigger import (
    REPO,
    atomic_csv,
    atomic_json,
    atomic_text,
    component_conserved,
    sha256,
    verify_seal,
    zero_nonterminal_f2b,
)
from build_dag_v682_stage_aware_pp_gradient import (
    SOURCE_COMPONENTS,
    TARGET_COMPONENTS,
    apply_source_gradient_wall,
)
from build_dag_v68_target_assisted_profiler_entry import (
    checked,
    configured_path,
    metric,
    replay,
)


DEFAULT_CONFIG = REPO / (
    "case_224gpu_pp14_cp2_a2a/config/"
    "dag_v683_causal_program_order_2026w36.toml"
)
EDGE_COLUMNS = (
    "case_id", "src", "dst", "edge_type", "tensor_key", "dependency_source",
)


def topology_fingerprint(edges: pd.DataFrame) -> str:
    """Hash dependency semantics, independent of row order and output paths."""
    missing = set(EDGE_COLUMNS) - set(edges.columns)
    if missing:
        raise ValueError(f"edge schema missing: {sorted(missing)}")
    frame = edges[list(EDGE_COLUMNS)].fillna("").astype(str).sort_values(
        list(EDGE_COLUMNS), kind="stable"
    )
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def zero_observed_b2b(
    nodes: pd.DataFrame,
    transfer: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep B->B program order but remove its duplicated observed wall cost."""
    selected = transfer[transfer["transition_category"].eq("B2B")].copy()
    if selected.empty or selected["node_id"].duplicated().any():
        raise ValueError("B2B transfer grid is empty or duplicated")
    output = nodes.copy()
    node_ids = set(selected["node_id"].astype(str))
    mask = output["node_id"].astype(str).isin(node_ids)
    if int(mask.sum()) != len(node_ids):
        raise ValueError(f"B2B node mismatch: {int(mask.sum())} != {len(node_ids)}")
    old = output.loc[mask, ["node_id", "duration_ns"]].copy().rename(
        columns={"duration_ns": "old_observed_b2b_gap_ns"}
    )
    selected = selected.merge(old, on="node_id", validate="one_to_one")
    if not selected["old_observed_b2b_gap_ns"].gt(0).all():
        raise ValueError("B2B input is no longer the v6.7 observed-gap assignment")
    selected["new_program_order_cost_ns"] = 0
    selected["semantic_action"] = "keep_edge_remove_observed_outcome_cost"
    selected["required_trigger"] = "activation_and_downstream_pp_gradient"

    output.loc[mask, "duration_ns"] = 0
    for column in component_columns:
        if column in output.columns:
            output.loc[mask, column] = 0
    assignments = {
        "op_name": "microbatch_program_order_b2b",
        "cost_status": "CAUSAL_SEMANTICS_PATCH",
        "cost_source": "no_independent_observed_b2b_wait_cost",
        "timing_component": "program_order_zero_duration",
        "timing_source": "backward_requires_activation_and_downstream_gradient",
        "model_component": "program_order",
        "semantic_region": "program_order_b2b_zero_cost",
    }
    for column, value in assignments.items():
        if column in output.columns:
            output.loc[mask, column] = value
    return output, selected.sort_values(
        ["pp_stage", "pp_lane", "sequence_index"], kind="stable"
    )


def target_dependency_audit(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    transfer: pd.DataFrame,
    *,
    pp: int,
    microbatches: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fail closed on the static target 1F1B prerequisite contract."""
    starts = nodes[
        nodes["kind"].eq("phase_boundary")
        & nodes["op_name"].isin(["fwd_start", "bwd_start"])
    ].copy()
    expected_starts = pp * 16 * microbatches * 2
    if len(starts) != expected_starts:
        raise ValueError(f"phase start grid incomplete: {len(starts)} != {expected_starts}")
    incoming = {str(key): value for key, value in edges.groupby(edges["dst"].astype(str))}
    allowed = {
        "FWD": {
            "profiler_entry_dependency", "pp_activation_recv",
            "microbatch_release_complete", "phase_handoff_complete",
        },
        "BWD": {
            "autograd_activation_dependency", "pp_gradient_recv",
            "microbatch_release_complete", "phase_handoff_complete",
        },
    }
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for node in starts.itertuples(index=False):
        node_id = str(node.node_id)
        phase = str(node.phase)
        stage = int(node.pp_stage)
        predecessors = incoming.get(node_id, edges.iloc[:0])
        types = predecessors["edge_type"].astype(str).tolist()
        unexpected = sorted(set(types) - allowed[phase])
        activation_count = types.count("pp_activation_recv")
        gradient_count = types.count("pp_gradient_recv")
        autograd_count = types.count("autograd_activation_dependency")
        expected_activation = 1 if phase == "FWD" and stage > 0 else 0
        expected_gradient = 1 if phase == "BWD" and stage < pp - 1 else 0
        expected_autograd = 1 if phase == "BWD" else 0
        passed = (
            not unexpected
            and activation_count == expected_activation
            and gradient_count == expected_gradient
            and autograd_count == expected_autograd
        )
        if not passed:
            failures.append(node_id)
        rows.append({
            "node_id": node_id,
            "phase": phase,
            "pp_stage": stage,
            "pp_lane": int(node.pp_lane),
            "microbatch": int(node.microbatch),
            "incoming_edge_types": "|".join(sorted(types)),
            "pp_activation_count": activation_count,
            "pp_gradient_count": gradient_count,
            "autograd_activation_count": autograd_count,
            "unexpected_edge_types": "|".join(unexpected),
            "semantic_contract_pass": passed,
        })
    if failures:
        raise ValueError(f"static prerequisite contract failed for {len(failures)} starts")

    b2b_ids = set(transfer.loc[
        transfer["transition_category"].eq("B2B"), "node_id"
    ].astype(str))
    b2b = nodes[nodes["node_id"].astype(str).isin(b2b_ids)]
    if len(b2b) != len(b2b_ids) or not b2b["duration_ns"].eq(0).all():
        raise ValueError("B2B program-order nodes are not all zero duration")
    summary = {
        "schema": "dag-v6.8.3-static-dependency-audit-v1",
        "status": "PASS_FROZEN_EDGE_SET_AND_STATIC_PREREQUISITES",
        "phase_start_rows": len(rows),
        "forward_start_rows": int(starts["phase"].eq("FWD").sum()),
        "backward_start_rows": int(starts["phase"].eq("BWD").sum()),
        "nonterminal_backward_with_gradient_rows": int(
            ((starts["phase"].eq("BWD")) & starts["pp_stage"].lt(pp - 1)).sum()
        ),
        "backward_with_activation_rows": int(starts["phase"].eq("BWD").sum()),
        "zero_duration_b2b_program_order_nodes": len(b2b),
        "dependency_edges_added": 0,
        "dependency_edges_removed": 0,
        "unresolved_cost_ownership": ["F2F_observed_gap", "B2F_observed_gap"],
    }
    return pd.DataFrame(rows).sort_values(
        ["phase", "pp_stage", "pp_lane", "microbatch"], kind="stable"
    ), summary


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    p = payload["prediction"]
    m = payload["metrics"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DAG v6.8.3 · 因果程序顺序</title><style>
:root{{--bg:#07111d;--panel:#0e1b2c;--line:#2b4059;--text:#e9f2fb;--muted:#93a8bd;--f:#24c8ff;--b:#ff725c;--grad:#ef79ff;--act:#55d6be;--warn:#ffe45e}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:28px}}h1{{margin:4px 0}}h2{{font-size:19px}}.muted{{color:var(--muted)}}.eyebrow{{color:var(--act);font-weight:800;font-size:12px;letter-spacing:.12em}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}}.card,.panel{{background:var(--panel);border:1px solid var(--line);padding:15px}}.v{{font-size:24px;font-weight:800}}.k{{font-size:12px;color:var(--muted)}}.flow{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;align-items:center;gap:9px;margin:15px 0}}.node{{padding:14px;border:1px solid var(--line);background:#091827;min-height:76px}}.grad{{border-color:var(--grad)}}.arrow{{font-size:23px;color:var(--muted)}}.formula{{padding:11px;border-left:4px solid var(--grad);background:#171629;font:600 15px ui-monospace,monospace}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-size:12px}}.pass{{color:#55d6be}}.partial{{color:var(--warn)}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}.flow{{display:block}}.node{{margin:7px 0}}.arrow{{text-align:center}}}}</style></head><body><main>
<div class="eyebrow">DAG MFU · v6.8.3 CAUSAL BASELINE</div><h1>依赖来自训练语义，不来自误差拟合</h1><p class="muted">本版冻结v6.8.2全部依赖边，只修正重复计时：B→B的Trace完整间隔不再作为一段额外本地耗时。</p>
<div class="cards"><div class="card"><div class="k">v6.8.2 Profiler</div><div class="v">{p['v682_profiler_ms']/1000:.3f}s</div></div><div class="card"><div class="k">v6.8.3 Profiler</div><div class="v">{p['v683_profiler_ms']/1000:.3f}s</div></div><div class="card"><div class="k">60–100开发集 MAPE</div><div class="v">{m['all_mape_pct']:.3f}%</div></div><div class="card"><div class="k">边集合变化</div><div class="v">0</div></div></div>
<section class="panel"><h2>反向开始的唯一逻辑</h2><div class="flow"><div class="node">本GPU程序顺序<br><span class="muted">只约束先后，0ms</span></div><div class="arrow">＋</div><div class="node">同microbatch激活<br><span class="muted">F完成</span></div><div class="arrow">＋</div><div class="node grad">下一级PP梯度<br><span class="muted">接收完成</span></div><div class="arrow">→</div><div class="node">本级B开始</div></div><div class="formula">B(s,m).start = max(program_order, activation_ready, gradient_recv(s+1→s,m))</div><p>这里没有“本地等待优先”或“梯度优先”。三个条件都是前提，最晚到达者自然成为关键前驱。</p></section>
<section class="panel"><h2>依赖审计门</h2><table><tr><th>检查</th><th>结果</th><th>含义</th></tr><tr><td>父子边指纹</td><td class="pass">一致</td><td>未人工补边、删边</td></tr><tr><td>目标阶段入口</td><td class="pass">{payload['audit']['phase_start_rows']} / {payload['audit']['phase_start_rows']}通过</td><td>F检查PP激活；B检查激活和PP梯度</td></tr><tr><td>B→B观测长间隔</td><td class="pass">{payload['audit']['zero_duration_b2b_program_order_nodes']}个已去重</td><td>保留顺序边，耗时归零</td></tr><tr><td>F→F、B→F成本归属</td><td class="partial">PARTIAL</td><td>依赖保留，尚未证明完整Trace间隔可迁移</td></tr></table></section>
<section class="panel"><h2>为什么精度不是本版唯一验收</h2><p>若去掉错误成本后误差增大，说明旧成本曾在数值上补偿其他缺项；不能因此把错误依赖或重复耗时加回来。后续autoresearch只能在固定边集合上补充有来源的计算、通信service和软件启动成本。</p></section>
<script>window.DAG_V683={data};</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    v682 = configured_path(config["inputs"]["v682_run_dir"])
    v67 = configured_path(config["inputs"]["v67_run_dir"])
    verify_seal(
        v682 / "predictions/prediction_seal.json",
        "SEALED_BEFORE_V682_TARGET_EVALUATOR_ACCESS",
    )
    verify_seal(
        v67 / "predictions/prediction_seal.json",
        "SEALED_BEFORE_V67_EVALUATOR_ACCESS",
    )
    parent_nodes_path = checked(v682 / "predictions/dag_v682_nodes.csv.gz")
    parent_edges_path = checked(v682 / "predictions/dag_v682_edges.csv.gz")
    parent_contract_path = checked(v682 / "prediction_contract.json")
    parent_source_contract_path = checked(v682 / "source_replay/source_replay_contract.json")
    gradient_parameters_path = checked(
        v682 / "calibration/source256_pp_gradient_wall_parameters.csv"
    )
    target_release_path = checked(v67 / "calibration/target_release_gap_transfer.csv")
    source_nodes_path = checked(v67 / "source_replay/dag_v67_source_nodes.csv.gz")
    source_edges_path = checked(v67 / "source_replay/dag_v67_source_edges.csv.gz")
    source_release_path = checked(v67 / "source_replay/source_release_gap_transfer.csv")
    source_contract_path = checked(v67 / "source_replay/source_replay_contract.json")

    parent_nodes = pd.read_csv(parent_nodes_path, low_memory=False)
    parent_edges = pd.read_csv(parent_edges_path, low_memory=False)
    target_release = pd.read_csv(target_release_path)
    nodes, target_correction = zero_observed_b2b(
        parent_nodes, target_release, component_columns=TARGET_COMPONENTS
    )
    # The edge frame is intentionally never mutated.  Equality is checked
    # before any prediction result can be sealed.
    edges = parent_edges.copy()
    if not edges.equals(parent_edges):
        raise ValueError("dependency edge frame changed")
    parent_topology = topology_fingerprint(parent_edges)
    child_topology = topology_fingerprint(edges)
    if parent_topology != child_topology:
        raise ValueError("dependency topology fingerprint changed")
    dependency_rows, dependency_summary = target_dependency_audit(
        nodes,
        edges,
        target_release,
        pp=int(config["target"]["pp"]),
        microbatches=int(config["target"]["microbatches"]),
    )
    nodes, critical = replay(nodes, edges, config["model"]["target_completion_node_id"])
    if not component_conserved(nodes, TARGET_COMPONENTS):
        raise ValueError("target component conservation failed")
    raw_ms = float(nodes.loc[
        nodes["node_id"].eq(config["model"]["target_completion_node_id"]),
        "predicted_end_ns",
    ].item()) / 1e6

    # Rebuild the source replay from frozen v6.7/v6.8.2 inputs so the opaque
    # source reconciliation does not silently retain the removed B2B cost.
    source_nodes = pd.read_csv(source_nodes_path, low_memory=False)
    source_edges = pd.read_csv(source_edges_path, low_memory=False)
    source_release = pd.read_csv(source_release_path)
    source_nodes, _ = zero_nonterminal_f2b(
        source_nodes,
        source_release,
        pp=int(config["source"]["pp"]),
        component_columns=SOURCE_COMPONENTS,
    )
    source_nodes, source_correction = zero_observed_b2b(
        source_nodes, source_release, component_columns=SOURCE_COMPONENTS
    )
    gradient_parameters = pd.read_csv(gradient_parameters_path)
    source_nodes, _ = apply_source_gradient_wall(source_nodes, gradient_parameters)
    source_nodes, source_critical = replay(
        source_nodes, source_edges, config["model"]["source_completion_node_id"]
    )
    if not component_conserved(source_nodes, SOURCE_COMPONENTS):
        raise ValueError("source component conservation failed")
    source_raw_ms = float(source_nodes.loc[
        source_nodes["node_id"].eq(config["model"]["source_completion_node_id"]),
        "predicted_end_ns",
    ].item()) / 1e6
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_profiler_ms = float(source_contract["source_profiler_median_ms"])
    source_reconciliation_ms = source_profiler_ms - source_raw_ms
    if source_reconciliation_ms < 0:
        raise ValueError("causal source graph exceeds source Profiler median")
    parent_source_contract = json.loads(parent_source_contract_path.read_text(encoding="utf-8"))
    reconciliation_scale = float(parent_source_contract["target_reconciliation_scale"])
    target_reconciliation_ms = source_reconciliation_ms * reconciliation_scale

    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
    parent_prediction = parent_contract["prediction"]
    profiler_ms = raw_ms + target_reconciliation_ms
    outer_ms = float(parent_prediction["outer_framework_ms"])
    training_ms = profiler_ms + outer_ms
    mfu_pct = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"])
        * 1e12
        * training_ms
        / 1000.0
    )

    output = configured_path(config["outputs"]["output_dir"])
    predictions = output / "predictions"
    calibration = output / "calibration"
    source_replay = output / "source_replay"
    evaluator = output / "evaluator_only"
    logs = output / "logs"
    for directory in (predictions, calibration, source_replay, evaluator, logs):
        directory.mkdir(parents=True, exist_ok=True)
    nodes_out = predictions / "dag_v683_nodes.csv.gz"
    edges_out = predictions / "dag_v683_edges.csv.gz"
    critical_out = predictions / "dag_v683_critical_path.csv"
    methods_out = predictions / "method_predictions.csv"
    target_correction_out = calibration / "target_b2b_cost_ownership_correction.csv"
    source_correction_out = calibration / "source256_b2b_cost_ownership_correction.csv"
    dependency_rows_out = calibration / "target_static_dependency_audit.csv"
    dependency_lock_out = predictions / "dependency_topology_lock.json"
    source_nodes_out = source_replay / "dag_v683_source_nodes.csv.gz"
    source_critical_out = source_replay / "dag_v683_source_critical_path.csv"
    source_contract_out = source_replay / "source_replay_contract.json"
    contract_out = output / "prediction_contract.json"
    access_out = output / "input_access_audit.json"
    command_out = output / "reproduction_command.txt"
    evaluation_iterations = [int(value) for value in config["evaluation"]["iterations"]]
    methods = pd.DataFrame([{
        "iteration": iteration,
        "method": "DAG v6.8.3 causal program order",
        "predicted_raw_graph_ms": raw_ms,
        "predicted_profiler_step_ms": profiler_ms,
        "predicted_outer_framework_ms": outer_ms,
        "predicted_training_step_ms": training_ms,
        "predicted_mfu_pct": mfu_pct,
        "target_timing_read": False,
    } for iteration in evaluation_iterations])
    atomic_csv(nodes_out, nodes, compression="gzip")
    atomic_csv(edges_out, edges, compression="gzip")
    atomic_csv(critical_out, critical)
    atomic_csv(methods_out, methods)
    atomic_csv(target_correction_out, target_correction)
    atomic_csv(source_correction_out, source_correction)
    atomic_csv(dependency_rows_out, dependency_rows)
    atomic_csv(source_nodes_out, source_nodes, compression="gzip")
    atomic_csv(source_critical_out, source_critical)
    atomic_json(dependency_lock_out, {
        **dependency_summary,
        "parent_version": "v6.8.2",
        "parent_edge_rows": len(parent_edges),
        "child_edge_rows": len(edges),
        "parent_topology_sha256": parent_topology,
        "child_topology_sha256": child_topology,
        "topology_identical": True,
        "optimization_may_change_topology": False,
    })
    atomic_json(source_contract_out, {
        "schema": "dag-v6.8.3-source256-causal-program-order-replay-v1",
        "status": "PASS_SOURCE256_REPLAY_PARTIAL_COST_OWNERSHIP",
        "source_raw_graph_ms": source_raw_ms,
        "source_profiler_median_ms": source_profiler_ms,
        "source_reconciliation_ms": source_reconciliation_ms,
        "target_reconciliation_scale": reconciliation_scale,
        "target_reconciliation_ms": target_reconciliation_ms,
        "zero_duration_source_b2b_nodes": len(source_correction),
        "target_timing_files_read": 0,
    })
    atomic_json(contract_out, {
        "schema": "dag-v6.8.3-causal-program-order-contract-v1",
        "status": "PARTIAL_CAUSAL_COST_OWNERSHIP_SOURCE256_ONLY_PREDICTION",
        "version": "v6.8.3",
        "parent_version": "v6.8.2",
        "target_timing_read_before_seal": False,
        "target_parameter_updates": 0,
        "dependency_policy": config["model"]["dependency_policy"],
        "dependency_topology_sha256": child_topology,
        "dependency_edges_added": 0,
        "dependency_edges_removed": 0,
        "causal_rule": {
            "nonterminal_backward_start": config["model"]["nonterminal_backward_start"],
            "observed_f2b_gap_role": "diagnostic_only_not_a_duration",
            "observed_b2b_gap_role": "diagnostic_only_not_a_duration",
            "terminal_backward_start": "activation_ready_and_local_loss_backward_launch",
            "unresolved_cost_ownership": ["F2F_observed_gap", "B2F_observed_gap"],
        },
        "prediction": {
            "v682_raw_graph_ms": float(parent_prediction["raw_graph_ms"]),
            "raw_graph_ms": raw_ms,
            "source_reconciliation_origin_ms": source_reconciliation_ms,
            "target_reconciliation_ms": target_reconciliation_ms,
            "profiler_step_ms": profiler_ms,
            "outer_framework_ms": outer_ms,
            "training_step_ms": training_ms,
            "mfu_pct": mfu_pct,
        },
    })
    atomic_json(access_out, {
        "schema": "dag-v6.8.3-input-access-v1",
        "status": "PASS_SOURCE256_AND_FROZEN_PARENT_ONLY_BEFORE_SEAL",
        "source_and_parent_files_read": [str(path.resolve()) for path in (
            parent_nodes_path, parent_edges_path, parent_contract_path,
            parent_source_contract_path, gradient_parameters_path,
            target_release_path, source_nodes_path, source_edges_path,
            source_release_path, source_contract_path,
        )],
        "target_timing_files_read_before_seal": [],
        "target_parameter_updates": 0,
    })
    atomic_text(command_out, (
        ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
        "build_dag_v683_causal_program_order.py --config "
        "case_224gpu_pp14_cp2_a2a/config/"
        "dag_v683_causal_program_order_2026w36.toml\n"
    ))
    seal_out = predictions / "prediction_seal.json"
    sealed = [
        nodes_out, edges_out, critical_out, methods_out, target_correction_out,
        source_correction_out, dependency_rows_out, dependency_lock_out,
        source_nodes_out, source_critical_out, source_contract_out, contract_out,
        access_out, command_out,
    ]
    atomic_json(seal_out, {
        "schema": "dag-v6.8.3-prediction-seal-v1",
        "status": "SEALED_BEFORE_V683_TARGET_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [{
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in sealed],
    })

    # Target truth is opened only after the predictor artifacts are sealed.
    truth_path = checked(configured_path(config["inputs"]["target_ground_truth"]))
    truth = pd.read_csv(truth_path)
    truth = truth[truth["iteration"].isin(evaluation_iterations)][
        ["iteration", "actual_profiler_step_ms", "actual_training_step_ms"]
    ]
    evaluation = methods.merge(truth, on="iteration", validate="one_to_one")
    evaluation["profiler_error_ms"] = (
        evaluation["predicted_profiler_step_ms"] - evaluation["actual_profiler_step_ms"]
    )
    evaluation["profiler_abs_error_pct"] = (
        evaluation["profiler_error_ms"].abs() / evaluation["actual_profiler_step_ms"] * 100.0
    )
    parent_eval = pd.read_csv(v682 / "evaluator_only/iteration_evaluation.csv")[[
        "iteration", "predicted_profiler_step_ms", "profiler_abs_error_pct",
    ]].rename(columns={
        "predicted_profiler_step_ms": "v682_predicted_profiler_step_ms",
        "profiler_abs_error_pct": "v682_profiler_abs_error_pct",
    })
    evaluation = evaluation.merge(parent_eval, on="iteration", validate="one_to_one")
    late_iterations = [int(value) for value in config["evaluation"]["late_development"]]
    late = evaluation[evaluation["iteration"].isin(late_iterations)]
    all_metric = metric(evaluation, "predicted_profiler_step_ms", "actual_profiler_step_ms")
    late_metric = metric(late, "predicted_profiler_step_ms", "actual_profiler_step_ms")
    v682_mape = float(evaluation["v682_profiler_abs_error_pct"].mean())
    evaluation_out = evaluator / "iteration_evaluation.csv"
    metrics_out = evaluator / "metrics.json"
    atomic_csv(evaluation_out, evaluation)
    metrics_payload = {
        "schema": "dag-v6.8.3-target-development-evaluation-v1",
        "status": "PASS_POST_SEAL_DEVELOPMENT_EVALUATION",
        "formal_blind_claim_allowed": False,
        "target_parameter_updates": 0,
        "all_60_100": {"profiler": all_metric},
        "late_85_100": {"profiler": late_metric},
        "comparison": {
            "v682_all_mape_pct": v682_mape,
            "v683_all_mape_pct": all_metric["mape_pct"],
            "mape_change_pp": all_metric["mape_pct"] - v682_mape,
        },
    }
    atomic_json(metrics_out, metrics_payload)
    evaluator_access_out = evaluator / "evaluation_access_audit.json"
    atomic_json(evaluator_access_out, {
        "schema": "dag-v6.8.3-evaluator-access-v1",
        "status": "PASS_TARGET_OPENED_AFTER_PREDICTION_SEAL",
        "prediction_seal": str(seal_out.resolve()),
        "target_timing_files_read": [str(truth_path.resolve())],
        "target_fields_read": [
            "iteration", "actual_profiler_step_ms", "actual_training_step_ms",
        ],
        "target_parameter_updates": 0,
    })
    payload = {
        "schema": "dag-v6.8.3-causal-program-order-visualization-v1",
        "status": "PARTIAL_CAUSAL_COST_OWNERSHIP",
        "prediction": {
            "v682_profiler_ms": float(parent_prediction["profiler_step_ms"]),
            "v683_profiler_ms": profiler_ms,
            "delta_ms": profiler_ms - float(parent_prediction["profiler_step_ms"]),
        },
        "metrics": {
            "all_mape_pct": all_metric["mape_pct"],
            "late_mape_pct": late_metric["mape_pct"],
            "mape_change_pp": all_metric["mape_pct"] - v682_mape,
        },
        "audit": dependency_summary,
        "topology_sha256": child_topology,
    }
    payload_out = evaluator / "dag_v683_causal_program_order_payload.json"
    html_out = evaluator / "dag_v683_causal_program_order.html"
    report_out = output / "DAG_V683_CAUSAL_PROGRAM_ORDER.md"
    log_out = logs / "build.log"
    atomic_json(payload_out, payload)
    atomic_text(html_out, render_html(payload))
    atomic_text(report_out, f"""# DAG v6.8.3：因果程序顺序基线

## 结论

v6.8.3不增加、不删除任何依赖边。它保留v6.8.2的1F1B程序顺序、PP激活、PP梯度和autograd依赖，只把`B→B`完整Trace间隔从“本地耗时”改为“零成本顺序约束”。

- 目标224修正B→B节点：`{len(target_correction)}`。
- 源256修正B→B节点：`{len(source_correction)}`。
- 父/子依赖边：`{len(parent_edges)}` / `{len(edges)}`，拓扑SHA256均为`{child_topology}`。
- 目标阶段入口审计：`{dependency_summary['phase_start_rows']}`行全部通过。
- v6.8.2 Profiler预测：`{float(parent_prediction['profiler_step_ms']):.6f} ms`。
- v6.8.3 Profiler预测：`{profiler_ms:.6f} ms`。
- 224卡60–100开发集MAPE：`{all_metric['mape_pct']:.6f}%`；不宣称blind validation。

## 当前边界

状态仍为`PARTIAL`。`F→F`与`B→F`的依赖边本身保留，但它们当前携带的完整观测间隔是否属于可迁移本地成本尚未证明。后续只能通过代码/Trace前驱归属审计拆分成本，不允许根据224误差人工补边或改变依赖方向。

## Autoresearch门槛

候选模型必须复用本版`dependency_topology_sha256`。允许搜索的只有有来源的节点耗时、OISA service FCT、软件启动成本和静态扩展参数；边集合变化必须由静态训练schedule或代码语义变更显式批准，否则失败。
""")
    atomic_text(log_out, "\n".join([
        "status=PARTIAL_CAUSAL_COST_OWNERSHIP",
        f"target_b2b_nodes={len(target_correction)}",
        f"source_b2b_nodes={len(source_correction)}",
        f"dependency_topology_sha256={child_topology}",
        "dependency_edges_added=0",
        "dependency_edges_removed=0",
        f"target_raw_graph_ms={raw_ms:.6f}",
        f"target_profiler_ms={profiler_ms:.6f}",
        f"all_60_100_mape_pct={all_metric['mape_pct']:.6f}",
        "target_parameter_updates=0",
    ]) + "\n")

    provenance_out = output / "provenance.json"
    input_paths = [
        config_path, Path(__file__).resolve(), parent_nodes_path, parent_edges_path,
        parent_contract_path, parent_source_contract_path, gradient_parameters_path,
        target_release_path, source_nodes_path, source_edges_path, source_release_path,
        source_contract_path, truth_path,
    ]
    output_paths = [
        nodes_out, edges_out, critical_out, methods_out, target_correction_out,
        source_correction_out, dependency_rows_out, dependency_lock_out,
        source_nodes_out, source_critical_out, source_contract_out, contract_out,
        access_out, command_out, seal_out, evaluation_out, metrics_out,
        evaluator_access_out, payload_out, html_out, report_out, log_out,
    ]
    atomic_json(provenance_out, {
        "schema": "dag-v6.8.3-causal-program-order-provenance-v1",
        "status": "PARTIAL_CAUSAL_COST_OWNERSHIP",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_parameter_updates": 0,
        "inputs": [{
            "path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in input_paths],
        "outputs": [{
            "path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in output_paths],
    })
    print(json.dumps({
        "status": "PARTIAL_CAUSAL_COST_OWNERSHIP",
        "html": str(html_out.resolve()),
        "dependency_topology_sha256": child_topology,
        "v682_profiler_ms": float(parent_prediction["profiler_step_ms"]),
        "v683_profiler_ms": profiler_ms,
        "all_60_100_mape_pct": all_metric["mape_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

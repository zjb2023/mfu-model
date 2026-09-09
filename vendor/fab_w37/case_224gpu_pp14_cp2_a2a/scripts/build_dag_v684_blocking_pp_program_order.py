#!/usr/bin/env python3
"""Build v6.8.4 with code-derived blocking PP-send dependencies."""

from __future__ import annotations

import argparse
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
)
from build_dag_v682_stage_aware_pp_gradient import SOURCE_COMPONENTS, TARGET_COMPONENTS
from build_dag_v683_causal_program_order import target_dependency_audit, topology_fingerprint
from build_dag_v68_target_assisted_profiler_entry import checked, configured_path, metric, replay


DEFAULT_CONFIG = REPO / (
    "case_224gpu_pp14_cp2_a2a/config/"
    "dag_v684_blocking_pp_program_order_2026w36.toml"
)


def build_blocking_edges(
    transfer: pd.DataFrame,
    *,
    case_id: str,
    pp: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in transfer.itertuples(index=False):
        stage = int(item.pp_stage)
        lane = int(item.pp_lane)
        microbatch = int(item.from_microbatch)
        if str(item.from_phase) == "forward" and stage < pp - 1:
            source = f"pp:lane{lane}:fwd{microbatch}:s{stage}_to_s{stage + 1}"
            edge_type = "blocking_forward_send"
            tensor_key = "activation"
        elif str(item.from_phase) == "backward" and stage > 0:
            source = f"pp:lane{lane}:bwd{microbatch}:s{stage}_to_s{stage - 1}"
            edge_type = "blocking_backward_send"
            tensor_key = "gradient"
        else:
            continue
        rows.append({
            "case_id": case_id,
            "src": source,
            "dst": str(item.node_id),
            "edge_type": edge_type,
            "tensor_key": tensor_key,
            "dependency_source": "static_1f1b_blocking_pp_api",
        })
    frame = pd.DataFrame(rows).sort_values(
        ["edge_type", "src", "dst"], kind="stable"
    ).reset_index(drop=True)
    if frame.duplicated(["src", "dst", "edge_type"]).any():
        raise ValueError("generated blocking dependency is duplicated")
    return frame


def zero_displaced_forward_transition_costs(
    nodes: pd.DataFrame,
    transfer: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = transfer[
        transfer["transition_category"].isin(["F2F", "B2F"])
    ].copy()
    node_ids = set(selected["node_id"].astype(str))
    output = nodes.copy()
    mask = output["node_id"].astype(str).isin(node_ids)
    if int(mask.sum()) != len(node_ids):
        raise ValueError(f"forward-transition node mismatch: {int(mask.sum())} != {len(node_ids)}")
    old = output.loc[mask, ["node_id", "duration_ns"]].copy().rename(
        columns={"duration_ns": "old_observed_transition_gap_ns"}
    )
    selected = selected.merge(old, on="node_id", validate="one_to_one")
    if not selected["old_observed_transition_gap_ns"].gt(0).all():
        raise ValueError("forward-transition input is no longer an observed wall assignment")
    selected["new_program_order_cost_ns"] = 0
    selected["semantic_action"] = "blocking_pp_edge_replaces_observed_wall_cost"
    output.loc[mask, "duration_ns"] = 0
    for column in component_columns:
        if column in output.columns:
            output.loc[mask, column] = 0
    assignments = {
        "cost_status": "CODE_DERIVED_DEPENDENCY_PATCH",
        "cost_source": "blocking_pp_send_dependency_not_observed_transition_wall",
        "timing_component": "program_order_zero_duration",
        "timing_source": "static_1f1b_blocking_pp_api",
        "model_component": "program_order",
        "semantic_region": "blocking_pp_program_order_zero_cost",
    }
    for column, value in assignments.items():
        if column in output.columns:
            output.loc[mask, column] = value
    return output, selected.sort_values(
        ["transition_category", "pp_stage", "pp_lane", "sequence_index"],
        kind="stable",
    )


def audit_blocking_edges(
    nodes: pd.DataFrame,
    parent_edges: pd.DataFrame,
    child_edges: pd.DataFrame,
    added: pd.DataFrame,
    transfer: pd.DataFrame,
    *,
    pp: int,
) -> dict[str, Any]:
    node_ids = set(nodes["node_id"].astype(str))
    if not added["src"].isin(node_ids).all() or not added["dst"].isin(node_ids).all():
        raise ValueError("blocking dependency endpoint missing")
    overlap = added.merge(parent_edges, on=["src", "dst", "edge_type"], how="inner")
    if not overlap.empty:
        raise ValueError("blocking dependency already existed in target graph")
    expected_forward = (pp - 1) * 16 * 3
    expected_backward = (pp - 1) * 16 * 2
    counts = added["edge_type"].value_counts().to_dict()
    if counts.get("blocking_forward_send", 0) != expected_forward:
        raise ValueError("blocking forward-send edge grid incomplete")
    if counts.get("blocking_backward_send", 0) != expected_backward:
        raise ValueError("blocking backward-send edge grid incomplete")
    if len(child_edges) != len(parent_edges) + len(added):
        raise ValueError("unexpected target dependency edge delta")
    parent_keys = set(map(tuple, parent_edges.fillna("").astype(str).to_numpy()))
    child_keys = set(map(tuple, child_edges.fillna("").astype(str).to_numpy()))
    if not parent_keys.issubset(child_keys):
        raise ValueError("a parent dependency edge was deleted or changed")
    transition_ids = set(transfer.loc[
        transfer["transition_category"].isin(["F2F", "B2F", "F2B", "B2B"]),
        "node_id",
    ].astype(str))
    transition_nodes = nodes[nodes["node_id"].astype(str).isin(transition_ids)]
    terminal_f2b = transfer[
        transfer["transition_category"].eq("F2B") & transfer["pp_stage"].eq(pp - 1)
    ]
    zero_ids = transition_ids - set(terminal_f2b["node_id"].astype(str))
    zero_nodes = nodes[nodes["node_id"].astype(str).isin(zero_ids)]
    if not zero_nodes["duration_ns"].eq(0).all():
        raise ValueError("a non-terminal phase transition still carries observed wall cost")
    terminal_nodes = nodes[nodes["node_id"].isin(terminal_f2b["node_id"])]
    if terminal_nodes.empty or not terminal_nodes["duration_ns"].gt(0).all():
        raise ValueError("terminal loss/backward launch cost disappeared")
    return {
        "schema": "dag-v6.8.4-blocking-pp-dependency-audit-v1",
        "status": "PASS_CODE_DERIVED_DEPENDENCY_GRID",
        "parent_edge_rows": len(parent_edges),
        "child_edge_rows": len(child_edges),
        "edges_added": len(added),
        "edges_removed": 0,
        "blocking_forward_send_edges": counts["blocking_forward_send"],
        "blocking_backward_send_edges": counts["blocking_backward_send"],
        "zero_duration_nonterminal_phase_transition_nodes": len(zero_nodes),
        "terminal_loss_backward_launch_nodes": len(terminal_nodes),
        "transition_nodes_audited": len(transition_nodes),
        "edge_change_authority": "static_1f1b_schedule_and_pp_api_order_not_target_error",
    }


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    p = payload["prediction"]
    m = payload["metrics"]
    a = payload["audit"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DAG v6.8.4 · 阻塞PP程序顺序</title><style>:root{{--bg:#07111d;--panel:#0e1b2c;--line:#2b4059;--text:#e9f2fb;--muted:#93a8bd;--f:#24c8ff;--b:#ff725c;--pp:#ef79ff;--ok:#55d6be}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:28px}}h1{{margin:4px 0}}.eyebrow{{color:var(--ok);font-size:12px;font-weight:800;letter-spacing:.12em}}.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}}.card,.panel{{padding:15px;background:var(--panel);border:1px solid var(--line)}}.k{{font-size:12px;color:var(--muted)}}.v{{font-size:23px;font-weight:800}}.flow{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:13px 0}}.node{{padding:12px 15px;border:1px solid var(--line);background:#091827}}.pp{{border-color:var(--pp)}}.arrow{{font-size:22px;color:var(--muted)}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}code{{color:#bde7ff}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}</style></head><body><main><div class="eyebrow">DAG MFU · v6.8.4 CODE-DERIVED TOPOLOGY</div><h1>PP发送完成，才允许本rank进入下一阶段</h1><p class="muted">边来自静态1F1B和Trace里的PP API顺序，不来自224卡误差拟合。</p><div class="cards"><div class="card"><div class="k">v6.8.3 Profiler</div><div class="v">{p['v683_profiler_ms']/1000:.3f}s</div></div><div class="card"><div class="k">v6.8.4 Profiler</div><div class="v">{p['v684_profiler_ms']/1000:.3f}s</div></div><div class="card"><div class="k">60–100开发集 MAPE</div><div class="v">{m['all_mape_pct']:.3f}%</div></div><div class="card"><div class="k">代码证明的新增边</div><div class="v">{a['edges_added']}</div></div></div><section class="panel"><h2>正确的本地程序顺序</h2><div class="flow"><div class="node">本rank F/B完成</div><div class="arrow">→</div><div class="node pp">PP send/recv完成</div><div class="arrow">→</div><div class="node">零成本程序顺序门</div><div class="arrow">→</div><div class="node">本rank下一F/B</div></div><p>forward阻塞边{a['blocking_forward_send_edges']}条，backward阻塞边{a['blocking_backward_send_edges']}条。接收侧原有PP激活/梯度依赖、autograd依赖全部保留；旧F→F/B→F完整Trace间隔不再重复计时。</p></section><section class="panel"><h2>结构审计</h2><table><tr><th>项目</th><th>结果</th></tr><tr><td>父图边</td><td>{a['parent_edge_rows']}</td></tr><tr><td>新增边</td><td>{a['edges_added']}（仅blocking send）</td></tr><tr><td>删除边</td><td>{a['edges_removed']}</td></tr><tr><td>非末级阶段切换零成本</td><td>{a['zero_duration_nonterminal_phase_transition_nodes']} / {a['zero_duration_nonterminal_phase_transition_nodes']}</td></tr><tr><td>新拓扑SHA256</td><td><code>{payload['topology_sha256']}</code></td></tr></table></section><section class="panel"><h2>后续autoresearch边界</h2><p>v6.8.4的新指纹是后续参数优化基线。只有代码或静态schedule证据才能再次改边；误差下降本身不能成为增删依赖的理由。</p></section><script>window.DAG_V684={data};</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    v683 = configured_path(config["inputs"]["v683_run_dir"])
    v67 = configured_path(config["inputs"]["v67_run_dir"])
    verify_seal(v683 / "predictions/prediction_seal.json", "SEALED_BEFORE_V683_TARGET_EVALUATOR_ACCESS")
    parent_nodes_path = checked(v683 / "predictions/dag_v683_nodes.csv.gz")
    parent_edges_path = checked(v683 / "predictions/dag_v683_edges.csv.gz")
    parent_contract_path = checked(v683 / "prediction_contract.json")
    parent_source_path = checked(v683 / "source_replay/source_replay_contract.json")
    source_nodes_path = checked(v683 / "source_replay/dag_v683_source_nodes.csv.gz")
    source_edges_path = checked(v67 / "source_replay/dag_v67_source_edges.csv.gz")
    target_release_path = checked(v67 / "calibration/target_release_gap_transfer.csv")
    source_release_path = checked(v67 / "source_replay/source_release_gap_transfer.csv")
    pp_api_path = checked(configured_path(config["inputs"]["pp_api_evidence"]))

    pp_api = pd.read_csv(pp_api_path, usecols=["iteration", "api_name"])
    required_api = {"send_forward_recv_backward", "send_backward_recv_forward"}
    if not required_api.issubset(set(pp_api["api_name"])):
        raise ValueError("fused blocking PP API evidence is incomplete")

    parent_nodes = pd.read_csv(parent_nodes_path, low_memory=False)
    parent_edges = pd.read_csv(parent_edges_path, low_memory=False)
    target_release = pd.read_csv(target_release_path)
    added = build_blocking_edges(
        target_release,
        case_id=str(config["target"]["case_id"]),
        pp=int(config["target"]["pp"]),
    )
    edges = pd.concat([parent_edges, added], ignore_index=True)
    nodes, target_correction = zero_displaced_forward_transition_costs(
        parent_nodes, target_release, component_columns=TARGET_COMPONENTS
    )
    blocking_audit = audit_blocking_edges(
        nodes, parent_edges, edges, added, target_release, pp=int(config["target"]["pp"])
    )
    phase_rows, phase_audit = target_dependency_audit(
        nodes, edges, target_release,
        pp=int(config["target"]["pp"]),
        microbatches=int(config["target"]["microbatches"]),
    )
    nodes, critical = replay(nodes, edges, config["model"]["target_completion_node_id"])
    if not component_conserved(nodes, TARGET_COMPONENTS):
        raise ValueError("target component conservation failed")
    raw_ms = float(nodes.loc[
        nodes["node_id"].eq(config["model"]["target_completion_node_id"]), "predicted_end_ns"
    ].item()) / 1e6

    source_nodes = pd.read_csv(source_nodes_path, low_memory=False)
    source_edges = pd.read_csv(source_edges_path, low_memory=False)
    source_release = pd.read_csv(source_release_path)
    source_counts = source_edges["edge_type"].value_counts().to_dict()
    if source_counts.get("blocking_forward_send", 0) != 15 * 16 * 4:
        raise ValueError("source blocking forward-send grid incomplete")
    if source_counts.get("blocking_backward_send", 0) != 15 * 16 * 3:
        raise ValueError("source blocking backward-send grid incomplete")
    source_nodes, source_correction = zero_displaced_forward_transition_costs(
        source_nodes, source_release, component_columns=SOURCE_COMPONENTS
    )
    source_nodes, source_critical = replay(
        source_nodes, source_edges, config["model"]["source_completion_node_id"]
    )
    if not component_conserved(source_nodes, SOURCE_COMPONENTS):
        raise ValueError("source component conservation failed")
    source_raw_ms = float(source_nodes.loc[
        source_nodes["node_id"].eq(config["model"]["source_completion_node_id"]),
        "predicted_end_ns",
    ].item()) / 1e6
    parent_source = json.loads(parent_source_path.read_text(encoding="utf-8"))
    source_profiler_ms = float(parent_source["source_profiler_median_ms"])
    source_reconciliation_ms = source_profiler_ms - source_raw_ms
    scale = float(parent_source["target_reconciliation_scale"])
    target_reconciliation_ms = source_reconciliation_ms * scale
    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
    parent_prediction = parent_contract["prediction"]
    profiler_ms = raw_ms + target_reconciliation_ms
    outer_ms = float(parent_prediction["outer_framework_ms"])
    training_ms = profiler_ms + outer_ms
    mfu_pct = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"])
        * 1e12 * training_ms / 1000.0
    )
    parent_topology = topology_fingerprint(parent_edges)
    child_topology = topology_fingerprint(edges)
    if parent_topology == child_topology:
        raise ValueError("code-derived blocking edges did not change topology fingerprint")

    output = configured_path(config["outputs"]["output_dir"])
    predictions, calibration, source_replay, evaluator, logs = (
        output / "predictions", output / "calibration", output / "source_replay",
        output / "evaluator_only", output / "logs",
    )
    for directory in (predictions, calibration, source_replay, evaluator, logs):
        directory.mkdir(parents=True, exist_ok=True)
    nodes_out = predictions / "dag_v684_nodes.csv.gz"
    edges_out = predictions / "dag_v684_edges.csv.gz"
    critical_out = predictions / "dag_v684_critical_path.csv"
    methods_out = predictions / "method_predictions.csv"
    lock_out = predictions / "dependency_topology_lock.json"
    added_out = calibration / "code_derived_blocking_pp_edges.csv"
    target_correction_out = calibration / "target_forward_transition_cost_correction.csv"
    source_correction_out = calibration / "source256_forward_transition_cost_correction.csv"
    phase_audit_out = calibration / "target_static_dependency_audit.csv"
    source_nodes_out = source_replay / "dag_v684_source_nodes.csv.gz"
    source_critical_out = source_replay / "dag_v684_source_critical_path.csv"
    source_contract_out = source_replay / "source_replay_contract.json"
    contract_out = output / "prediction_contract.json"
    access_out = output / "input_access_audit.json"
    command_out = output / "reproduction_command.txt"
    iterations = [int(value) for value in config["evaluation"]["iterations"]]
    methods = pd.DataFrame([{
        "iteration": iteration,
        "method": "DAG v6.8.4 code-derived blocking PP program order",
        "predicted_raw_graph_ms": raw_ms,
        "predicted_profiler_step_ms": profiler_ms,
        "predicted_outer_framework_ms": outer_ms,
        "predicted_training_step_ms": training_ms,
        "predicted_mfu_pct": mfu_pct,
        "target_timing_read": False,
    } for iteration in iterations])
    atomic_csv(nodes_out, nodes, compression="gzip")
    atomic_csv(edges_out, edges, compression="gzip")
    atomic_csv(critical_out, critical)
    atomic_csv(methods_out, methods)
    atomic_csv(added_out, added)
    atomic_csv(target_correction_out, target_correction)
    atomic_csv(source_correction_out, source_correction)
    atomic_csv(phase_audit_out, phase_rows)
    atomic_csv(source_nodes_out, source_nodes, compression="gzip")
    atomic_csv(source_critical_out, source_critical)
    atomic_json(lock_out, {
        **blocking_audit,
        "phase_start_contract_status": phase_audit["status"],
        "parent_topology_sha256": parent_topology,
        "child_topology_sha256": child_topology,
        "future_parameter_search_must_match_sha256": child_topology,
    })
    atomic_json(source_contract_out, {
        "schema": "dag-v6.8.4-source256-blocking-pp-replay-v1",
        "status": "PASS_SOURCE256_CODE_DERIVED_DEPENDENCIES",
        "source_raw_graph_ms": source_raw_ms,
        "source_profiler_median_ms": source_profiler_ms,
        "source_reconciliation_ms": source_reconciliation_ms,
        "target_reconciliation_scale": scale,
        "target_reconciliation_ms": target_reconciliation_ms,
        "zeroed_source_f2f_b2f_nodes": len(source_correction),
        "source_blocking_forward_send_edges": source_counts["blocking_forward_send"],
        "source_blocking_backward_send_edges": source_counts["blocking_backward_send"],
    })
    atomic_json(contract_out, {
        "schema": "dag-v6.8.4-blocking-pp-program-order-contract-v1",
        "status": "SOURCE256_ONLY_CODE_DERIVED_DEPENDENCY_PREDICTION",
        "version": "v6.8.4",
        "parent_version": "v6.8.3",
        "target_timing_read_before_seal": False,
        "target_parameter_updates": 0,
        "dependency_change": blocking_audit,
        "dependency_topology_sha256": child_topology,
        "prediction": {
            "v683_raw_graph_ms": float(parent_prediction["raw_graph_ms"]),
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
        "schema": "dag-v6.8.4-input-access-v1",
        "status": "PASS_SOURCE256_STATIC_SEMANTICS_ONLY_BEFORE_SEAL",
        "source_and_parent_files_read": [str(path.resolve()) for path in (
            parent_nodes_path, parent_edges_path, parent_contract_path, parent_source_path,
            source_nodes_path, source_edges_path, target_release_path, source_release_path,
            pp_api_path,
        )],
        "pp_api_fields_read": ["iteration", "api_name"],
        "target_timing_files_read_before_seal": [],
        "target_parameter_updates": 0,
    })
    atomic_text(command_out, (
        ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
        "build_dag_v684_blocking_pp_program_order.py --config "
        "case_224gpu_pp14_cp2_a2a/config/"
        "dag_v684_blocking_pp_program_order_2026w36.toml\n"
    ))
    seal_out = predictions / "prediction_seal.json"
    sealed = [
        nodes_out, edges_out, critical_out, methods_out, lock_out, added_out,
        target_correction_out, source_correction_out, phase_audit_out,
        source_nodes_out, source_critical_out, source_contract_out, contract_out,
        access_out, command_out,
    ]
    atomic_json(seal_out, {
        "schema": "dag-v6.8.4-prediction-seal-v1",
        "status": "SEALED_BEFORE_V684_TARGET_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [{
            "path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in sealed],
    })

    truth_path = checked(configured_path(config["inputs"]["target_ground_truth"]))
    truth = pd.read_csv(truth_path)
    truth = truth[truth["iteration"].isin(iterations)][
        ["iteration", "actual_profiler_step_ms", "actual_training_step_ms"]
    ]
    evaluation = methods.merge(truth, on="iteration", validate="one_to_one")
    evaluation["profiler_error_ms"] = evaluation["predicted_profiler_step_ms"] - evaluation["actual_profiler_step_ms"]
    evaluation["profiler_abs_error_pct"] = evaluation["profiler_error_ms"].abs() / evaluation["actual_profiler_step_ms"] * 100.0
    parent_eval = pd.read_csv(v683 / "evaluator_only/iteration_evaluation.csv")[[
        "iteration", "predicted_profiler_step_ms", "profiler_abs_error_pct",
    ]].rename(columns={
        "predicted_profiler_step_ms": "v683_predicted_profiler_step_ms",
        "profiler_abs_error_pct": "v683_profiler_abs_error_pct",
    })
    evaluation = evaluation.merge(parent_eval, on="iteration", validate="one_to_one")
    late = evaluation[evaluation["iteration"].isin(config["evaluation"]["late_development"])]
    all_metric = metric(evaluation, "predicted_profiler_step_ms", "actual_profiler_step_ms")
    late_metric = metric(late, "predicted_profiler_step_ms", "actual_profiler_step_ms")
    v683_mape = float(evaluation["v683_profiler_abs_error_pct"].mean())
    evaluation_out = evaluator / "iteration_evaluation.csv"
    metrics_out = evaluator / "metrics.json"
    evaluator_access_out = evaluator / "evaluation_access_audit.json"
    atomic_csv(evaluation_out, evaluation)
    atomic_json(metrics_out, {
        "schema": "dag-v6.8.4-target-development-evaluation-v1",
        "status": "PASS_POST_SEAL_DEVELOPMENT_EVALUATION",
        "formal_blind_claim_allowed": False,
        "target_parameter_updates": 0,
        "all_60_100": {"profiler": all_metric},
        "late_85_100": {"profiler": late_metric},
        "comparison": {
            "v683_all_mape_pct": v683_mape,
            "v684_all_mape_pct": all_metric["mape_pct"],
            "mape_change_pp": all_metric["mape_pct"] - v683_mape,
        },
    })
    atomic_json(evaluator_access_out, {
        "schema": "dag-v6.8.4-evaluator-access-v1",
        "status": "PASS_TARGET_OPENED_AFTER_PREDICTION_SEAL",
        "target_timing_files_read": [str(truth_path.resolve())],
        "target_parameter_updates": 0,
    })
    payload = {
        "schema": "dag-v6.8.4-blocking-pp-program-order-visualization-v1",
        "prediction": {
            "v683_profiler_ms": float(parent_prediction["profiler_step_ms"]),
            "v684_profiler_ms": profiler_ms,
            "delta_ms": profiler_ms - float(parent_prediction["profiler_step_ms"]),
        },
        "metrics": {
            "all_mape_pct": all_metric["mape_pct"],
            "late_mape_pct": late_metric["mape_pct"],
            "mape_change_pp": all_metric["mape_pct"] - v683_mape,
        },
        "audit": blocking_audit,
        "topology_sha256": child_topology,
    }
    payload_out = evaluator / "dag_v684_blocking_pp_program_order_payload.json"
    html_out = evaluator / "dag_v684_blocking_pp_program_order.html"
    report_out = output / "DAG_V684_BLOCKING_PP_PROGRAM_ORDER.md"
    log_out = logs / "build.log"
    atomic_json(payload_out, payload)
    atomic_text(html_out, render_html(payload))
    atomic_text(report_out, f"""# DAG v6.8.4：阻塞PP程序顺序

v6.8.4只增加可由静态1F1B和PP API顺序证明的依赖：PP发送完成后，本rank才可进入下一F/B。256源图已有同类边，目标图此前缺失。

- 新增forward阻塞边：`{blocking_audit['blocking_forward_send_edges']}`。
- 新增backward阻塞边：`{blocking_audit['blocking_backward_send_edges']}`。
- 删除父图边：`0`。
- 去重F→F/B→F完整观测间隔：目标`{len(target_correction)}`个，源`{len(source_correction)}`个。
- 新依赖拓扑SHA256：`{child_topology}`。
- v6.8.3 / v6.8.4 Profiler预测：`{float(parent_prediction['profiler_step_ms']):.6f}` / `{profiler_ms:.6f} ms`。
- 224卡60–100开发集MAPE：`{all_metric['mape_pct']:.6f}%`；目标参数更新为0，不宣称blind validation。

后续autoresearch必须匹配本版拓扑指纹。只有新的训练代码或静态schedule证据才能改变依赖；目标误差不能作为增删边依据。
""")
    atomic_text(log_out, "\n".join([
        "status=PASS_CODE_DERIVED_DEPENDENCY_BASELINE",
        f"blocking_forward_send_edges={blocking_audit['blocking_forward_send_edges']}",
        f"blocking_backward_send_edges={blocking_audit['blocking_backward_send_edges']}",
        f"dependency_topology_sha256={child_topology}",
        f"target_profiler_ms={profiler_ms:.6f}",
        f"all_60_100_mape_pct={all_metric['mape_pct']:.6f}",
        "target_parameter_updates=0",
    ]) + "\n")
    provenance_out = output / "provenance.json"
    input_paths = [
        config_path, Path(__file__).resolve(), parent_nodes_path, parent_edges_path,
        parent_contract_path, parent_source_path, source_nodes_path, source_edges_path,
        target_release_path, source_release_path, pp_api_path, truth_path,
    ]
    output_paths = [
        nodes_out, edges_out, critical_out, methods_out, lock_out, added_out,
        target_correction_out, source_correction_out, phase_audit_out, source_nodes_out,
        source_critical_out, source_contract_out, contract_out, access_out, command_out,
        seal_out, evaluation_out, metrics_out, evaluator_access_out, payload_out,
        html_out, report_out, log_out,
    ]
    atomic_json(provenance_out, {
        "schema": "dag-v6.8.4-blocking-pp-program-order-provenance-v1",
        "status": "PASS_CODE_DERIVED_DEPENDENCY_BASELINE",
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
        "status": "PASS_CODE_DERIVED_DEPENDENCY_BASELINE",
        "html": str(html_out.resolve()),
        "dependency_topology_sha256": child_topology,
        "v683_profiler_ms": float(parent_prediction["profiler_step_ms"]),
        "v684_profiler_ms": profiler_ms,
        "all_60_100_mape_pct": all_metric["mape_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

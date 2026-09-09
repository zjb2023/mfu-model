#!/usr/bin/env python3
"""Add an explicit per-PP-stage barrier between the two AG rounds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from build_dag_v63_ordered_optimizer_tail import (
    atomic_json,
    atomic_text,
    checked,
    configured_path,
    git_output,
    max_plus,
    sha256,
    target_role,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v631_ag_round_barrier_2026w36.toml"


def verify_v63_seal(path: Path) -> dict[str, Any]:
    seal = json.loads(checked(path).read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_BEFORE_V63_EVALUATOR_ACCESS":
        raise ValueError("v6.3 input prediction is not sealed")
    for artifact in seal["artifacts"]:
        source = checked(Path(artifact["path"]))
        if sha256(source) != artifact["sha256"] or source.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"sealed v6.3 artifact changed: {source}")
    return seal


def calibrate_stage_barrier(
    events: pd.DataFrame, source_iterations: tuple[int, ...], first_stage: int, last_stage: int
) -> dict[str, Any]:
    if tuple(sorted(events["iteration"].astype(int).unique())) != source_iterations:
        raise ValueError("source iteration grid changed")
    rows: list[dict[str, Any]] = []
    for (iteration, stage), frame in events.groupby(["iteration", "pp_stage"]):
        first_round = frame[
            frame["behavior"].eq("expert_dp_param_allgather") & frame["call_round"].eq(0)
        ]
        second_round = frame[
            frame["behavior"].eq("dp_param_allgather") & frame["call_round"].eq(1)
        ]
        if first_round.empty or second_round.empty:
            continue
        rows.append({
            "iteration": int(iteration), "pp_stage": int(stage),
            "pp_role": "first" if int(stage) == first_stage else "last" if int(stage) == last_stage else "internal",
            "first_round_expert_dp_group_count": int(first_round["group_id"].nunique()),
            "second_round_dp_rank_count": int(len(second_round)),
            "gap_ns": int(second_round["start_ns"].min() - first_round["end_ns"].max()),
        })
    samples = pd.DataFrame(rows)
    parameters: dict[str, Any] = {}
    for role in ("first", "internal", "last"):
        values = samples.loc[samples["pp_role"].eq(role), "gap_ns"].astype("int64")
        valid = values[values.ge(0)]
        if valid.empty:
            raise ValueError(f"no nonnegative stage barrier gaps for {role}")
        parameters[role] = {
            "median_nonnegative_ns": int(valid.median()),
            "sample_count": int(len(values)),
            "nonnegative_count": int(len(valid)),
            "negative_excluded_count": int(values.lt(0).sum()),
            "minimum_nonnegative_ns": int(valid.min()),
            "maximum_nonnegative_ns": int(valid.max()),
        }
    return {
        "schema": "dag-v6.3.1-source-ag-round-barrier-calibration-v1",
        "status": "FROZEN_SOURCE_ONLY",
        "source_iterations": list(source_iterations),
        "barrier_definition": {
            "left": "max end_ns of Expert-DP AG round 0 within one PP stage",
            "right": "min start_ns of DP AG round 1 within the same PP stage",
            "scope": "per_pp_stage_dp_group",
        },
        "roles": parameters,
        "negative_samples": samples[samples["gap_ns"].lt(0)].to_dict("records"),
        "target_timing_read": False,
    }


def set_zero_duration(nodes: pd.DataFrame, index: dict[str, int], node_id: str) -> None:
    row = index[node_id]
    for column in (
        "duration_ns", "compute_exposed_ns_model", "compute_overlap_ns_model",
        "network_service_ns_model", "software_sync_ns_model", "framework_residual_ns_model",
    ):
        nodes.loc[row, column] = 0
    nodes.loc[row, "timing_component"] = "zero_duration_dependency"
    nodes.loc[row, "timing_source"] = "v631_explicit_barrier_structure"
    nodes.loc[row, "source_parameter_key"] = ""


def new_edge(columns: list[str], src: str, dst: str, edge_type: str, tensor_key: str) -> dict[str, Any]:
    row = {column: "" for column in columns}
    row.update({
        "case_id": "224gpu_pp14_cp2_a2a", "src": src, "dst": dst,
        "edge_type": edge_type, "tensor_key": tensor_key,
        "dependency_source": "distributed_optimizer_v631_ag_round_barrier",
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    v63_run = configured_path(config["inputs"]["v63_run_dir"])
    output = configured_path(config["outputs"]["output_dir"])
    prediction_dir = output / "predictions"
    calibration_dir = output / "calibration"
    logs_dir = output / "logs"
    for directory in (prediction_dir, calibration_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seal_path = checked(v63_run / "predictions/prediction_seal.json")
    verify_v63_seal(seal_path)
    node_input = checked(v63_run / "predictions/dag_v63_nodes.csv.gz")
    edge_input = checked(v63_run / "predictions/dag_v63_edges.csv.gz")
    contract_input = checked(v63_run / "prediction_contract.json")
    source_path = checked(configured_path(config["inputs"]["source_rank_events"]))
    nodes = pd.read_csv(node_input, low_memory=False)
    edges = pd.read_csv(edge_input, low_memory=False)
    events = pd.read_csv(source_path)
    source_iterations = tuple(int(value) for value in config["split"]["source_calibration"])
    calibration = calibrate_stage_barrier(
        events, source_iterations, int(config["barrier"]["source_first_stage"]),
        int(config["barrier"]["source_last_stage"]),
    )
    calibration_path = calibration_dir / "ag_round_barrier_parameters.json"
    atomic_json(calibration_path, calibration)

    pp = int(config["target"]["pp"])
    last_stage = int(config["barrier"]["target_last_stage"])
    node_index = {str(node_id): position for position, node_id in enumerate(nodes["node_id"])}
    remove = pd.Series(False, index=edges.index)
    additions: list[dict[str, Any]] = []
    barrier_ids: list[str] = []
    for stage in range(pp):
        role = target_role(stage, last_stage)
        duration = int(calibration["roles"][role]["median_nonnegative_ns"])
        dp_ag1_arrival = f"tail:dp_with_cp_stage{stage}:ag1_arrival_join"
        # Replace the v6.3 implicit eight-edge join with one named barrier node.
        remove |= (
            edges["dst"].eq(dp_ag1_arrival)
            & edges["tensor_key"].eq("expert_dp_ag0_to_dp_ag1")
        )
        set_zero_duration(nodes, node_index, dp_ag1_arrival)
        barrier_id = f"tail:stage{stage}:ag0_round_barrier"
        barrier_ids.append(barrier_id)
        template = nodes.loc[nodes["node_id"].eq("tail:world_rs_done")].iloc[0].copy()
        template["node_id"] = barrier_id
        template["pp_stage"] = stage
        template["parallelism"] = "dp_stage"
        template["op_name"] = "parameter_all_gather_round_barrier"
        template["shape_key"] = "all_expert_dp_groups_in_stage"
        template["timing_component"] = "tail_ag_round_barrier_software"
        template["timing_source"] = "source256_stage_barrier_gap_median_nonnegative"
        template["source_parameter_key"] = f"ag_round_barrier:{role}"
        for column in (
            "compute_exposed_ns_model", "compute_overlap_ns_model", "network_service_ns_model",
            "framework_residual_ns_model", "predicted_start_ns", "predicted_end_ns",
        ):
            template[column] = 0
        template["duration_ns"] = duration
        template["software_sync_ns_model"] = duration
        template["critical_predecessor"] = ""
        template["on_critical_path"] = False
        nodes = pd.concat([nodes, pd.DataFrame([template])], ignore_index=True)
        node_index[barrier_id] = len(nodes) - 1
        for lane in range(8):
            additions.append(new_edge(
                list(edges.columns), f"tail:expert_dp_stage{stage}_lane{lane}:ag0_completion_sync",
                barrier_id, "ag_round_barrier_arrival", "all_expert_dp_ag0_groups_in_stage",
            ))
        additions.append(new_edge(
            list(edges.columns), barrier_id, dp_ag1_arrival,
            "ag_round_barrier_release", "dp_ag1_release",
        ))

    removed_count = int(remove.sum())
    if removed_count != pp * 8:
        raise ValueError(f"v6.3 implicit AG join changed: removed={removed_count}")
    edges = pd.concat([edges.loc[~remove], pd.DataFrame(additions)], ignore_index=True)
    start, end, predecessor, critical = max_plus(nodes, edges, "iteration:completion_join")
    nodes["predicted_start_ns"] = start
    nodes["predicted_end_ns"] = end
    ids = nodes["node_id"].astype(str).tolist()
    nodes["critical_predecessor"] = [ids[value] if value >= 0 else "" for value in predecessor]
    nodes["on_critical_path"] = [position in critical for position in range(len(nodes))]
    lookup = nodes.set_index("node_id")
    checks = []
    for stage, barrier_id in enumerate(barrier_ids):
        expert_done = nodes[nodes["node_id"].str.match(
            fr"^tail:expert_dp_stage{stage}_lane[0-9]+:ag0_completion_sync$"
        )]["predicted_end_ns"]
        barrier_start = int(lookup.loc[barrier_id, "predicted_start_ns"])
        barrier_end = int(lookup.loc[barrier_id, "predicted_end_ns"])
        dp_ag1_start = int(lookup.loc[f"tail:dp_with_cp_stage{stage}:ag1_service", "predicted_start_ns"])
        checks.append({
            "pp_stage": stage, "all_expert_dp_ag0_done_ns": int(expert_done.max()),
            "barrier_start_ns": barrier_start, "barrier_end_ns": barrier_end,
            "dp_ag1_start_ns": dp_ag1_start,
            "join_pass": barrier_start >= int(expert_done.max()),
            "release_pass": dp_ag1_start >= barrier_end,
        })
    audit = {
        "schema": "dag-v6.3.1-ag-round-barrier-audit-v1",
        "status": "PASS" if all(item["join_pass"] and item["release_pass"] for item in checks) else "FAIL",
        "scope": config["barrier"]["scope"], "barrier_count": len(barrier_ids),
        "checks": checks,
        "global_barrier_deliberately_not_used": True,
        "reason": "target trace permits different PP stages to overlap across AG rounds",
    }
    if audit["status"] != "PASS":
        raise ValueError("AG round barrier audit failed")
    audit_path = output / "ag_round_barrier_audit.json"
    atomic_json(audit_path, audit)

    v63_contract = json.loads(contract_input.read_text(encoding="utf-8"))
    raw_ms = int(lookup.loc["iteration:completion_join", "predicted_end_ns"]) / 1e6
    reconciliation = float(v63_contract["prediction"]["source_reconciliation_ms"])
    outer = float(v63_contract["prediction"]["outer_framework_ms"])
    profiler = raw_ms + reconciliation
    training = profiler + outer
    mfu = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        training / 1000.0 * int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"]) * 1e12
    )
    node_path = prediction_dir / "dag_v631_nodes.csv.gz"
    edge_path = prediction_dir / "dag_v631_edges.csv.gz"
    critical_path = prediction_dir / "dag_v631_critical_path.csv"
    nodes.to_csv(node_path, index=False, compression="gzip")
    edges.to_csv(edge_path, index=False, compression="gzip")
    nodes[nodes["on_critical_path"]].sort_values("predicted_start_ns").to_csv(critical_path, index=False)

    target_grid = [*config["split"]["target_regression_only"], *config["split"]["target_validation"]]
    rows = [{
        "protocol_id": config["protocol"]["id"], "source_case": "256gpu_pp16_cp2_a2a",
        "target_case": config["target"]["case_id"], "method": "DAG v6.3.1 explicit AG round barrier",
        "method_version": "dag-v6.3.1-source60-100-v1", "status": "PREDICTIVE_PARTIAL_UNEVALUATED",
        "split": "regression_only" if int(iteration) == 55 else "validation", "iteration": int(iteration),
        "predicted_raw_graph_ms": raw_ms, "predicted_source_reconciliation_ms": reconciliation,
        "predicted_profiler_step_ms": profiler, "predicted_outer_framework_ms": outer,
        "predicted_training_step_ms": training, "predicted_mfu_pct": mfu, "target_timing_read": False,
    } for iteration in target_grid]
    method_path = prediction_dir / "method_predictions.csv"
    pd.DataFrame(rows).to_csv(method_path, index=False)
    contract = {
        "schema": "dag-v6.3.1-ag-round-barrier-prediction-v1",
        "status": "PREDICTIVE_PARTIAL_UNEVALUATED",
        "prediction": {"raw_graph_ms": raw_ms, "source_reconciliation_ms": reconciliation,
                       "profiler_step_ms": profiler, "outer_framework_ms": outer,
                       "training_step_ms": training, "mfu_pct": mfu},
        "v63_raw_graph_ms": float(v63_contract["prediction"]["raw_graph_ms"]),
        "v631_minus_v63_raw_graph_ms": raw_ms - float(v63_contract["prediction"]["raw_graph_ms"]),
        "barrier": {"scope": config["barrier"]["scope"], "count": len(barrier_ids),
                    "removed_implicit_edges": removed_count, "added_edges": len(additions)},
        "target_timing_read": False, "accuracy_claim_allowed": False,
    }
    contract_path = output / "prediction_contract.json"
    atomic_json(contract_path, contract)
    access_path = output / "input_access_audit.json"
    inputs = [node_input, edge_input, contract_input, seal_path, source_path]
    atomic_json(access_path, {
        "schema": "dag-v6.3.1-input-access-audit-v1", "status": "PASS_NO_TARGET_TIMING",
        "source_iterations": list(source_iterations), "target_timing_files_read": 0,
        "source_fields_read": ["iteration", "pp_stage", "behavior", "call_round", "group_id", "start_ns", "end_ns"],
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in inputs],
    })
    report_path = output / "DAG_V631_AG_ROUND_BARRIER_REPORT.md"
    atomic_text(report_path, f"""# DAG v6.3.1：AG 两轮同步 barrier

在 v6.3 的 `DP AG0 → Expert-DP AG0 → DP AG1 → Expert-DP AG1` 顺序上，增加 14 个显式 barrier：每个 PP stage 汇合该 stage 的 8 个 Expert-DP AG0 group，再释放同 stage 的 DP AG1。

这不是 224-rank 全局 barrier。目标 Trace 显示不同 PP stage 可以跨 AG 轮次重叠；使用全局 barrier 会制造不存在的串行等待。

- v6.3 raw graph：`{contract['v63_raw_graph_ms']:.6f} ms`
- v6.3.1 raw graph：`{raw_ms:.6f} ms`
- 变化：`{contract['v631_minus_v63_raw_graph_ms']:.6f} ms`
- barrier 审计：`{audit['status']}`（{len(barrier_ids)} 个 stage barrier）
- 构图读取目标时间：`0`
""")
    reproduction = output / "reproduction_command.txt"
    atomic_text(reproduction, f"cd {REPO}\npython {Path(__file__).resolve()} --config {config_path}\n")
    log_path = logs_dir / "build.log"
    atomic_text(log_path, "\n".join([
        "DAG v6.3.1 AG round barrier PASS", f"generated_at={datetime.now(timezone.utc).isoformat()}",
        f"raw_graph_ms={raw_ms:.9f}", f"v631_minus_v63_raw_graph_ms={contract['v631_minus_v63_raw_graph_ms']:.9f}",
        f"barrier_count={len(barrier_ids)}", f"removed_edges={removed_count}", f"added_edges={len(additions)}",
        "barrier_audit=PASS", "target_timing_files_read=0", "status=PREDICTIVE_PARTIAL_UNEVALUATED", "",
    ]))
    provenance_path = output / "provenance.json"
    atomic_json(provenance_path, {
        "schema": "dag-v6.3.1-provenance-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO), "git_branch": git_output("branch", "--show-current"),
        "git_head": git_output("rev-parse", "HEAD"), "git_status_porcelain": git_output("status", "--short").splitlines(),
        "version_state": "PARTIAL_UNCOMMITTED_WORKTREE_PINNED_BY_SHA256",
        "code_and_config": [{"path": str(path), "sha256": sha256(path)} for path in (config_path, Path(__file__).resolve())],
        "v63_prediction_seal": {"path": str(seal_path), "sha256": sha256(seal_path)},
    })
    artifacts = [node_path, edge_path, critical_path, method_path, calibration_path, audit_path,
                 contract_path, access_path, report_path, reproduction, log_path, provenance_path]
    atomic_json(output / "artifact_manifest.json", {
        "schema": "dag-v6.3.1-artifact-manifest-v1",
        "artifacts": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts],
    })
    print(json.dumps({
        "status": contract["status"], "output": str(output), "raw_graph_ms": raw_ms,
        "v631_minus_v63_raw_graph_ms": contract["v631_minus_v63_raw_graph_ms"],
        "profiler_step_ms": profiler, "training_step_ms": training,
        "barrier_count": len(barrier_ids), "barrier_audit": audit["status"], "target_timing_files_read": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

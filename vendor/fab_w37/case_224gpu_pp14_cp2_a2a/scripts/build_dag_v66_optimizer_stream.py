#!/usr/bin/env python3
"""Refactor v6.5 optimizer tail into trace-ordered stream segments."""

from __future__ import annotations

import argparse
import json
import tempfile
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
    configured_path,
    git_output,
    max_plus,
    sha256,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v66_optimizer_stream_2026w36.toml"
SOURCE_ITERATIONS = tuple(range(60, 101, 5))


def checked(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def atomic_csv(path: Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def verify_seal(path: Path) -> dict[str, Any]:
    payload = json.loads(checked(path).read_text())
    if payload.get("status") != "SEALED_BEFORE_V65_EVALUATOR_ACCESS":
        raise ValueError("v6.5 input is not sealed")
    for artifact in payload["artifacts"]:
        source = checked(Path(artifact["path"]))
        if source.stat().st_size != int(artifact["size_bytes"]) or sha256(source) != artifact["sha256"]:
            raise ValueError(f"sealed v6.5 artifact changed: {source}")
    return payload


def target_role(stage: int, pp: int) -> str:
    return "first" if stage == 0 else ("last" if stage == pp - 1 else "internal")


def role_optimizer_parameters(parameters: pd.DataFrame) -> dict[str, dict[str, float]]:
    columns = [
        "trace_optimizer_compute_ns_median", "ar0_to_ar1_compute_ns_median",
        "ar1_to_dp_ag0_compute_ns_median",
    ]
    output: dict[str, dict[str, float]] = {}
    for role, frame in parameters.groupby("pp_role"):
        medians = {column: float(frame[column].median()) for column in columns}
        total = medians["trace_optimizer_compute_ns_median"]
        if total <= 0:
            raise ValueError(f"invalid optimizer total for {role}")
        output[str(role)] = {
            **medians,
            "ar0_to_ar1_fraction": medians["ar0_to_ar1_compute_ns_median"] / total,
            "ar1_to_dp_ag0_fraction": medians["ar1_to_dp_ag0_compute_ns_median"] / total,
        }
    if set(output) != {"first", "internal", "last"}:
        raise ValueError("optimizer role grid incomplete")
    return output


def global_allreduce_tails(parameters: pd.DataFrame) -> dict[int, int]:
    selected = parameters[
        parameters["behavior"].eq("global_control_allreduce")
        & parameters["call_ordinal"].isin([0, 1])
    ]
    if set(selected["call_ordinal"].astype(int)) != {0, 1}:
        raise ValueError("two source global AllReduce calls are required")
    return {
        int(row.call_ordinal): int(round(float(row.tail_after_last_arrival_ms_median) * 1e6))
        for row in selected.itertuples(index=False)
    }


def reset_components(row: pd.Series) -> pd.Series:
    for column in (
        "compute_exposed_ns_model", "compute_overlap_ns_model", "network_service_ns_model",
        "software_sync_ns_model", "framework_residual_ns_model",
    ):
        row[column] = 0
    return row


def node_from_template(
    template: pd.Series,
    *,
    node_id: str,
    kind: str,
    duration_ns: int,
    rank: int = -1,
    pp_stage: int = -1,
    op_name: str,
    op_family: str,
    timing_component: str,
    timing_source: str,
    compute_ns: int = 0,
    software_ns: int = 0,
) -> pd.Series:
    row = reset_components(template.copy())
    row["node_id"] = node_id
    row["kind"] = kind
    row["rank"] = rank
    row["pp_stage"] = pp_stage
    row["phase"] = "OPT"
    row["microbatch"] = -1
    row["layer_id"] = -1
    row["op_name"] = op_name
    row["op_family"] = op_family
    row["duration_ns"] = int(duration_ns)
    row["compute_exposed_ns_model"] = int(compute_ns)
    row["software_sync_ns_model"] = int(software_ns)
    row["timing_component"] = timing_component
    row["timing_source"] = timing_source
    row["source_parameter_key"] = timing_component
    row["source_v54_node_id"] = ""
    row["predicted_start_ns"] = 0
    row["predicted_end_ns"] = 0
    row["critical_predecessor"] = ""
    row["on_critical_path"] = False
    if compute_ns + software_ns != duration_ns:
        raise ValueError(f"new node component conservation failed: {node_id}")
    return row


def edge_row(columns: list[str], src: str, dst: str, edge_type: str, tensor_key: str) -> dict[str, Any]:
    row = {column: "" for column in columns}
    row.update({
        "case_id": "224gpu_pp14_cp2_a2a", "src": src, "dst": dst,
        "edge_type": edge_type, "tensor_key": tensor_key,
        "dependency_source": "dag_v66_source_trace_optimizer_stream",
    })
    return row


def set_compute_duration(
    nodes: pd.DataFrame, index: dict[str, int], node_id: str, duration_ns: int, component: str
) -> None:
    position = index[node_id]
    for column in (
        "compute_exposed_ns_model", "compute_overlap_ns_model", "network_service_ns_model",
        "software_sync_ns_model", "framework_residual_ns_model",
    ):
        nodes.loc[position, column] = 0
    nodes.loc[position, "duration_ns"] = int(duration_ns)
    nodes.loc[position, "compute_exposed_ns_model"] = int(duration_ns)
    nodes.loc[position, "timing_component"] = component
    nodes.loc[position, "timing_source"] = "source256_optimizer_phase_segment_fraction"
    nodes.loc[position, "source_parameter_key"] = component


def set_zero_barrier(nodes: pd.DataFrame, index: dict[str, int], node_id: str) -> None:
    position = index[node_id]
    for column in (
        "duration_ns", "compute_exposed_ns_model", "compute_overlap_ns_model",
        "network_service_ns_model", "software_sync_ns_model", "framework_residual_ns_model",
    ):
        nodes.loc[position, column] = 0
    nodes.loc[position, "timing_component"] = "zero_duration_dependency"
    nodes.loc[position, "timing_source"] = "v66_optimizer_post_ag0_join"
    nodes.loc[position, "source_parameter_key"] = ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if tuple(config["split"]["source_calibration"]) != SOURCE_ITERATIONS:
        raise ValueError("v6.6 source split must be stable iterations 60--100")
    v65 = configured_path(config["inputs"]["v65_run_dir"])
    seal_path = checked(v65 / "predictions/prediction_seal.json")
    verify_seal(seal_path)
    node_input = checked(v65 / "predictions/dag_v65_nodes.csv.gz")
    edge_input = checked(v65 / "predictions/dag_v65_edges.csv.gz")
    v65_contract_path = checked(v65 / "prediction_contract.json")
    optimizer_path = checked(configured_path(config["inputs"]["optimizer_parameters"]))
    optimizer_audit_path = checked(configured_path(config["inputs"]["optimizer_access_audit"]))
    control_path = checked(configured_path(config["inputs"]["global_control_parameters"]))
    control_audit_path = checked(configured_path(config["inputs"]["global_control_access_audit"]))
    source_replay_path = checked(configured_path(config["inputs"]["source_replay_contract"]))
    for path in (optimizer_audit_path, control_audit_path):
        audit = json.loads(path.read_text())
        if audit.get("status") != "PASS_SOURCE_ONLY" or audit.get("target_timing_files_read") != 0:
            raise ValueError(f"calibration audit is not source-only: {path}")
    source_replay = json.loads(source_replay_path.read_text())
    if source_replay.get("status") != "PASS_SOURCE_REPLAY":
        raise ValueError("v6.6 source graph replay has not passed")
    if int(source_replay.get("target_timing_files_read", -1)) != 0:
        raise ValueError("v6.6 source replay opened target timing")

    nodes = pd.read_csv(node_input, low_memory=False)
    edges = pd.read_csv(edge_input, low_memory=False)
    node_index = {str(node_id): position for position, node_id in enumerate(nodes["node_id"])}
    optimizer_roles = role_optimizer_parameters(pd.read_csv(optimizer_path))
    control_tails = global_allreduce_tails(pd.read_csv(control_path))
    pp = int(config["target"]["pp"])
    world = int(config["target"]["world_size"])
    template = nodes.loc[nodes["node_id"].eq("tail:world_rs_done")].iloc[0]
    edge_columns = list(edges.columns)
    additions: list[dict[str, Any]] = []
    new_nodes: list[pd.Series] = []
    remove = pd.Series(False, index=edges.index)

    global0 = "tail:global_control_allreduce0_completion_tail"
    global1_join = "tail:global_control_allreduce1_arrival_join"
    global1 = "tail:global_control_allreduce1_completion_tail"
    new_nodes.extend([
        node_from_template(
            template, node_id=global0, kind="collective_service",
            duration_ns=control_tails[0], op_name="global_control_allreduce_0_completion_tail",
            op_family="global_control", timing_component="global_control_completion_tail",
            timing_source="source256_tail_after_last_arrival_median", software_ns=control_tails[0],
        ),
        node_from_template(
            template, node_id=global1_join, kind="collective_boundary", duration_ns=0,
            op_name="global_control_allreduce_1_arrival_join", op_family="global_control",
            timing_component="zero_duration_dependency", timing_source="v66_dependency_structure",
        ),
        node_from_template(
            template, node_id=global1, kind="collective_service",
            duration_ns=control_tails[1], op_name="global_control_allreduce_1_completion_tail",
            op_family="global_control", timing_component="global_control_completion_tail",
            timing_source="source256_tail_after_last_arrival_median", software_ns=control_tails[1],
        ),
    ])
    additions.append(edge_row(edge_columns, "tail:world_rs_done", global0, "global_control_release", "world_rs_done"))
    additions.append(edge_row(edge_columns, global1_join, global1, "global_control_release", "global_ar1"))

    split_rows: list[dict[str, Any]] = []
    for rank in range(world):
        stage = rank // 16
        role = target_role(stage, pp)
        values = optimizer_roles[role]
        update = f"tail:r{rank}:optimizer_update"
        start = f"tail:r{rank}:optimizer_start"
        done = f"tail:r{rank}:optimizer_done"
        old_total = int(nodes.loc[node_index[update], "duration_ns"])
        ar0_ar1 = int(round(old_total * values["ar0_to_ar1_fraction"]))
        ar1_ag0 = int(round(old_total * values["ar1_to_dp_ag0_fraction"]))
        post_ag0 = max(old_total - ar0_ar1 - ar1_ag0, 0)
        set_compute_duration(nodes, node_index, update, ar0_ar1, "optimizer_ar0_to_ar1_compute")
        pre_node = f"tail:r{rank}:optimizer_pre_ag0"
        post_node = f"tail:r{rank}:optimizer_post_ag0"
        new_nodes.extend([
            node_from_template(
                template, node_id=pre_node, kind="optimizer", duration_ns=ar1_ag0,
                rank=rank, pp_stage=stage, op_name="optimizer_pre_ag0", op_family="optimizer",
                timing_component="optimizer_pre_ag0_compute",
                timing_source="source256_optimizer_phase_segment_fraction", compute_ns=ar1_ag0,
            ),
            node_from_template(
                template, node_id=post_node, kind="optimizer", duration_ns=post_ag0,
                rank=rank, pp_stage=stage, op_name="optimizer_post_ag0", op_family="optimizer",
                timing_component="optimizer_post_ag0_compute",
                timing_source="source256_optimizer_phase_segment_remainder", compute_ns=post_ag0,
            ),
        ])
        remove |= edges["src"].eq("tail:world_rs_done") & edges["dst"].eq(start)
        additions.append(edge_row(edge_columns, global0, start, "global_control_complete", f"rank{rank}_optimizer_ar0_ar1"))
        additions.append(edge_row(edge_columns, done, global1_join, "global_control_rank_arrival", f"rank{rank}_global_ar1"))
        additions.append(edge_row(edge_columns, global1, pre_node, "global_control_complete", f"rank{rank}_optimizer_pre_ag0"))
        for group_node in (
            f"tail:dp_with_cp_stage{stage}:ag0_arrival_join",
            f"tail:expert_dp_stage{stage}_lane{rank % 8}:ag0_arrival_join",
        ):
            remove |= edges["src"].eq(done) & edges["dst"].eq(group_node)
            additions.append(edge_row(edge_columns, pre_node, group_node, "optimizer_pre_ag0_complete", f"rank{rank}_ag0"))
        edp_done = f"tail:expert_dp_stage{stage}_lane{rank % 8}:ag0_completion_sync"
        barrier = f"tail:stage{stage}:ag0_round_barrier"
        additions.append(edge_row(edge_columns, edp_done, post_node, "optimizer_post_ag0_release", f"rank{rank}_post_ag0"))
        additions.append(edge_row(edge_columns, post_node, barrier, "optimizer_post_ag0_complete", f"rank{rank}_ag1_barrier"))
        split_rows.append({
            "rank": rank, "pp_stage": stage, "pp_role": role,
            "v65_optimizer_total_ns": old_total, "v66_ar0_to_ar1_compute_ns": ar0_ar1,
            "v66_ar1_to_ag0_compute_ns": ar1_ag0, "v66_post_ag0_compute_ns": post_ag0,
        })

    for stage in range(pp):
        barrier = f"tail:stage{stage}:ag0_round_barrier"
        set_zero_barrier(nodes, node_index, barrier)
        for lane in range(8):
            src = f"tail:expert_dp_stage{stage}_lane{lane}:ag0_completion_sync"
            remove |= edges["src"].eq(src) & edges["dst"].eq(barrier)

    removed = int(remove.sum())
    expected_removed = world + world * 2 + pp * 8
    if removed != expected_removed:
        raise ValueError(f"unexpected edge replacement count: {removed} != {expected_removed}")
    nodes = pd.concat([nodes, pd.DataFrame(new_nodes)], ignore_index=True)
    edges = pd.concat([edges.loc[~remove], pd.DataFrame(additions)], ignore_index=True)
    start, end, predecessor, critical = max_plus(nodes, edges, "iteration:completion_join")
    ids = nodes["node_id"].astype(str).tolist()
    nodes["predicted_start_ns"] = start
    nodes["predicted_end_ns"] = end
    nodes["critical_predecessor"] = [ids[value] if value >= 0 else "" for value in predecessor]
    nodes["on_critical_path"] = [position in critical for position in range(len(nodes))]
    component_columns = [
        "compute_exposed_ns_model", "compute_overlap_ns_model", "network_service_ns_model",
        "software_sync_ns_model", "framework_residual_ns_model",
    ]
    conserved = nodes[component_columns].fillna(0).sum(axis=1).round().astype("int64")
    if not conserved.eq(nodes["duration_ns"].astype("int64")).all():
        bad = nodes.loc[~conserved.eq(nodes["duration_ns"].astype("int64")), "node_id"].head().tolist()
        raise ValueError(f"node component conservation failed: {bad}")

    output = configured_path(config["outputs"]["output_dir"])
    prediction = output / "predictions"
    calibration = output / "calibration"
    node_path = prediction / "dag_v66_nodes.csv.gz"
    edge_path = prediction / "dag_v66_edges.csv.gz"
    critical_path = prediction / "dag_v66_critical_path.csv"
    split_path = calibration / "optimizer_stream_transfer.csv"
    atomic_csv(node_path, nodes, compression="gzip")
    atomic_csv(edge_path, edges, compression="gzip")
    atomic_csv(critical_path, nodes[nodes["on_critical_path"]].sort_values("predicted_start_ns"))
    atomic_csv(split_path, pd.DataFrame(split_rows))

    v65_contract = json.loads(v65_contract_path.read_text())
    old = v65_contract["prediction"]
    raw_ms = int(nodes.loc[nodes["node_id"].eq("iteration:completion_join"), "predicted_end_ns"].iloc[0]) / 1e6
    reconciliation_ms = float(source_replay["target_reconciliation_ms"])
    outer_ms = float(old["outer_framework_ms"])
    profiler_ms = raw_ms + reconciliation_ms
    training_ms = profiler_ms + outer_ms
    mfu = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        training_ms / 1000 * world * float(config["target"]["peak_tflops_per_gpu"]) * 1e12
    )
    contract = {
        "schema": "dag-v6.6-optimizer-stream-prediction-v2",
        "status": "PREDICTIVE_UNEVALUATED",
        "prediction": {
            "raw_graph_ms": raw_ms,
            "source_reconciliation_origin_ms": float(source_replay["source_reconciliation_ms"]),
            "target_reconciliation_ms": reconciliation_ms,
            # Compatibility with v6.2--v6.5 readers; this value is already
            # scaled for the target and the explicit field above is canonical.
            "source_reconciliation_ms": reconciliation_ms,
            "profiler_step_ms": profiler_ms, "outer_framework_ms": outer_ms,
            "training_step_ms": training_ms, "mfu_pct": mfu,
        },
        "v65_raw_graph_ms": float(old["raw_graph_ms"]),
        "v66_minus_v65_raw_graph_ms": raw_ms - float(old["raw_graph_ms"]),
        "source_replay_status": "PASS_SOURCE_REPLAY",
        "source_replay_contract_sha256": sha256(source_replay_path),
        "reconciliation_status": "REFIT_FROM_V66_SOURCE_GRAPH_AND_SCALED_BY_MICROBATCH_RATIO",
        "optimizer_semantics": "AR0 tail -> optimizer small segment -> AR1 tail -> pre-AG0 segment -> AG0 -> per-rank post-AG0 segment -> stage join -> AG1",
        "target_timing_read": False,
        "accuracy_claim_allowed": False,
    }
    contract_path = output / "prediction_contract.json"
    atomic_json(contract_path, contract)
    grid = [*config["split"]["target_regression_only"], *config["split"]["target_validation"]]
    methods = pd.DataFrame([{
        "protocol_id": config["protocol"]["id"], "source_case": "256gpu_pp16_cp2_a2a",
        "target_case": config["target"]["case_id"], "method": "DAG v6.6 optimizer stream",
        "method_version": "dag-v6.6-source60-100-v2", "status": "PREDICTIVE_UNEVALUATED",
        "split": "regression_only" if int(iteration) == 55 else "validation", "iteration": int(iteration),
        "predicted_raw_graph_ms": raw_ms, "predicted_source_reconciliation_ms": reconciliation_ms,
        "predicted_source_reconciliation_origin_ms": float(source_replay["source_reconciliation_ms"]),
        "predicted_target_reconciliation_ms": reconciliation_ms,
        "predicted_profiler_step_ms": profiler_ms, "predicted_outer_framework_ms": outer_ms,
        "predicted_training_step_ms": training_ms, "predicted_mfu_pct": mfu,
        "target_timing_read": False,
    } for iteration in grid])
    method_path = prediction / "method_predictions.csv"
    atomic_csv(method_path, methods)
    audit = {
        "schema": "dag-v6.6-optimizer-stream-audit-v2", "status": "PASS_STRUCTURE_AND_SOURCE_REPLAY",
        "target_timing_files_read": 0, "parameter_updates_from_target": 0,
        "optimizer_total_conserved_per_rank": bool((pd.DataFrame(split_rows)[[
            "v66_ar0_to_ar1_compute_ns", "v66_ar1_to_ag0_compute_ns", "v66_post_ag0_compute_ns"
        ]].sum(axis=1) == pd.DataFrame(split_rows)["v65_optimizer_total_ns"]).all()),
        "global_control_tail_ns": control_tails, "removed_edge_count": removed,
        "added_node_count": len(new_nodes), "added_edge_count": len(additions),
        "graph_node_count": len(nodes), "graph_edge_count": len(edges),
        "acyclic": True, "component_conservation": True,
        "source_replay": {
            "status": source_replay["status"],
            "source_raw_graph_ms": source_replay["v66_raw_graph_ms"],
            "source_profiler_median_ms": source_replay["source_profiler_median_ms"],
            "source_reconciliation_ms": source_replay["source_reconciliation_ms"],
            "target_reconciliation_ms": source_replay["target_reconciliation_ms"],
        },
    }
    audit_path = output / "optimizer_stream_audit.json"
    atomic_json(audit_path, audit)
    access_path = output / "input_access_audit.json"
    input_paths = [
        config_path, seal_path, node_input, edge_input, v65_contract_path,
        optimizer_path, optimizer_audit_path, control_path, control_audit_path,
        source_replay_path,
    ]
    atomic_json(access_path, {
        "schema": "dag-v6.6-input-access-v1", "status": "PASS_NO_TARGET_TIMING",
        "source_iterations": list(SOURCE_ITERATIONS), "target_timing_files_read": 0,
        "target_fields_read": ["sealed v6.5 target-static graph"],
        "inputs": [{"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in input_paths],
    })
    report_path = output / "DAG_V66_OPTIMIZER_STREAM.md"
    atomic_text(report_path, f"""# DAG v6.6 optimizer stream

- Status: `PREDICTIVE_UNEVALUATED`
- 52-layer target placement: `[2, 4×12, 2]`
- Raw graph: `{raw_ms:.6f} ms` (v6.5 delta `{raw_ms - float(old['raw_graph_ms']):+.6f} ms`)
- Source reconciliation: `{float(source_replay['source_reconciliation_ms']):.6f} ms`; target-scaled reconciliation: `{reconciliation_ms:.6f} ms`
- Profiler prediction: `{profiler_ms:.6f} ms`
- Training prediction: `{training_ms:.6f} ms`

v6.6 does not change total per-rank optimizer compute.  It moves that compute to
the order observed in source trace and allows the post-AG0 portion to overlap
across Expert-DP groups.  The old fixed-duration AG0→AG1 barrier is now only a
zero-duration dependency join.  Two world AllReduce nodes use source-only
tail-after-last-arrival, never the full observed rank duration.

The 256-GPU source graph was rebuilt with the same ordered tail before this
target prediction.  Its reconciliation was solved against the source stable
profiler clock and transferred by the 3/4 microbatch ratio.  No target timing
was opened by either the source replay or prediction process.
""")
    reproduction_path = output / "reproduction_command.txt"
    atomic_text(reproduction_path, f"cd {REPO}\n{REPO / '.venv/bin/python'} {Path(__file__).resolve()} --config {config_path}\n")
    log_path = output / "logs/build.log"
    atomic_text(log_path, "\n".join([
        "PASS_STRUCTURE_AND_SOURCE_REPLAY", f"raw_graph_ms={raw_ms:.6f}",
        f"profiler_step_ms={profiler_ms:.6f}", f"training_step_ms={training_ms:.6f}",
        "target_timing_files_read=0", "parameter_updates_from_target=0",
    ]) + "\n")
    provenance_path = output / "provenance.json"
    artifacts = [node_path, edge_path, critical_path, split_path, method_path, contract_path, audit_path, access_path, report_path, reproduction_path, log_path]
    atomic_json(provenance_path, {
        "schema": "dag-v6.6-provenance-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_output("rev-parse", "HEAD"), "git_branch": git_output("branch", "--show-current"),
        "git_status_short": git_output("status", "--short"),
        "artifacts": [{"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in artifacts],
    })
    seal_artifacts = [
        method_path, node_path, edge_path, critical_path, split_path, contract_path,
        audit_path, access_path, report_path, reproduction_path, log_path,
        config_path, Path(__file__).resolve(),
    ]
    atomic_json(prediction / "prediction_seal.json", {
        "schema": "dag-v6.6-prediction-seal-v1",
        "status": "SEALED_BEFORE_V66_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_timing_opened_by_sealer": False,
        "prediction_status": contract["status"],
        "parameter_mutation_after_seal": "FORBIDDEN",
        "artifacts": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in seal_artifacts
        ],
    })
    print(json.dumps({"status": audit["status"], "raw_graph_ms": raw_ms, "profiler_step_ms": profiler_ms, "training_step_ms": training_ms, "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Insert source-calibrated local F/B scheduler handoff branches into v6.3.1."""

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
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v64a_phase_handoff_2026w36.toml"


def verify_v631_seal(path: Path) -> dict[str, Any]:
    seal = json.loads(checked(path).read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_BEFORE_V631_EVALUATOR_ACCESS":
        raise ValueError("v6.3.1 input prediction is not sealed")
    for artifact in seal["artifacts"]:
        source = checked(Path(artifact["path"]))
        if sha256(source) != artifact["sha256"] or source.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"sealed v6.3.1 artifact changed: {source}")
    return seal


def schedule(stage: int, pp: int, microbatches: int) -> list[tuple[str, int]]:
    warmup = min(pp - stage - 1, microbatches)
    remaining = microbatches - warmup
    operations: list[tuple[str, int]] = [("forward", mb) for mb in range(warmup)]
    for index in range(remaining):
        operations.append(("forward", warmup + index))
        operations.append(("backward", index))
    operations.extend(("backward", mb) for mb in range(remaining, microbatches))
    return operations


def calibrate_handoff(
    segments: pd.DataFrame, iterations: tuple[int, ...], pp: int, microbatches: int, quantile: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    segments = segments[segments["iteration"].isin(iterations)].copy()
    if tuple(sorted(segments["iteration"].astype(int).unique())) != iterations:
        raise ValueError("source phase-segment iteration grid changed")
    phase = segments.groupby(
        ["iteration", "rank", "pp_stage", "phase", "microbatch"], as_index=False
    ).agg(start_ns=("observed_start_ns", "min"), end_ns=("observed_end_ns", "max"))
    lookup = {
        (int(row.iteration), int(row.rank), str(row.phase), int(row.microbatch)):
        (int(row.start_ns), int(row.end_ns))
        for row in phase.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for (iteration, rank, stage), _ in phase.groupby(["iteration", "rank", "pp_stage"]):
        for left, right in zip(schedule(int(stage), pp, microbatches), schedule(int(stage), pp, microbatches)[1:]):
            if left[0] == right[0]:
                continue
            left_time = lookup[(int(iteration), int(rank), left[0], int(left[1]))]
            right_time = lookup[(int(iteration), int(rank), right[0], int(right[1]))]
            rows.append({
                "iteration": int(iteration), "rank": int(rank), "pp_stage": int(stage),
                "transition": f"{left[0][0].upper()}2{right[0][0].upper()}",
                "from_phase": left[0], "from_microbatch": int(left[1]),
                "to_phase": right[0], "to_microbatch": int(right[1]),
                "observed_gap_ns": int(right_time[0] - left_time[1]),
            })
    samples = pd.DataFrame(rows)
    parameters: dict[str, Any] = {}
    for transition in ("F2B", "B2F"):
        values = samples.loc[samples["transition"].eq(transition), "observed_gap_ns"].astype("int64")
        valid = values[values.ge(0)]
        if valid.empty:
            raise ValueError(f"no nonnegative source handoff samples for {transition}")
        parameters[transition] = {
            "p10_nonnegative_ns": int(round(float(valid.quantile(quantile)))),
            "sample_count": int(len(values)), "nonnegative_count": int(len(valid)),
            "negative_excluded_count": int(values.lt(0).sum()),
            "minimum_nonnegative_ns": int(valid.min()),
            "median_nonnegative_ns": int(round(float(valid.median()))),
            "p90_nonnegative_ns": int(round(float(valid.quantile(0.90)))),
            "maximum_nonnegative_ns": int(valid.max()),
        }
    payload = {
        "schema": "dag-v6.4a-source-phase-handoff-calibration-v1",
        "status": "FROZEN_SOURCE_ONLY_LOWER_ENVELOPE",
        "source_iterations": list(iterations), "quantile": quantile,
        "interpretation": (
            "PP dependency can only enlarge an observed phase gap; the nonnegative p10 "
            "is used as a conservative local scheduler-handoff floor."
        ),
        "parameters": parameters, "target_timing_read": False,
    }
    return payload, samples


def edge_row(columns: list[str], src: str, dst: str, edge_type: str, tensor_key: str) -> dict[str, Any]:
    row = {column: "" for column in columns}
    row.update({
        "case_id": "224gpu_pp14_cp2_a2a", "src": src, "dst": dst,
        "edge_type": edge_type, "tensor_key": tensor_key,
        "dependency_source": "v64a_source_phase_handoff",
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    v631_run = configured_path(config["inputs"]["v631_run_dir"])
    output = configured_path(config["outputs"]["output_dir"])
    predictions_dir = output / "predictions"
    calibration_dir = output / "calibration"
    logs_dir = output / "logs"
    for directory in (predictions_dir, calibration_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seal_path = checked(v631_run / "predictions/prediction_seal.json")
    verify_v631_seal(seal_path)
    node_input = checked(v631_run / "predictions/dag_v631_nodes.csv.gz")
    edge_input = checked(v631_run / "predictions/dag_v631_edges.csv.gz")
    contract_input = checked(v631_run / "prediction_contract.json")
    segment_path = checked(configured_path(config["source"]["phase_segments"]))
    nodes = pd.read_csv(node_input, low_memory=False)
    edges = pd.read_csv(edge_input, low_memory=False)
    segments = pd.read_csv(segment_path, usecols=[
        "iteration", "rank", "pp_stage", "phase", "microbatch",
        "observed_start_ns", "observed_end_ns",
    ])
    iterations = tuple(int(value) for value in config["split"]["source_calibration"])
    calibration, samples = calibrate_handoff(
        segments, iterations, int(config["source"]["pp"]), int(config["source"]["microbatches"]),
        float(config["handoff"]["quantile"]),
    )
    calibration_path = calibration_dir / "phase_handoff_parameters.json"
    sample_path = calibration_dir / "source_phase_transition_samples.csv"
    atomic_json(calibration_path, calibration)
    samples.to_csv(sample_path, index=False)

    node_lookup = nodes.set_index("node_id")
    phase_end_ids = set(nodes.loc[nodes["op_name"].isin(["fwd_end", "bwd_end"]), "node_id"].astype(str))
    phase_start_ids = set(nodes.loc[nodes["op_name"].isin(["fwd_start", "bwd_start"]), "node_id"].astype(str))
    candidates = edges[
        edges["edge_type"].eq("rank_program_order")
        & edges["src"].isin(phase_end_ids)
        & edges["dst"].isin(phase_start_ids)
    ].copy()
    candidates["source_phase"] = candidates["src"].map(lambda node: str(node_lookup.loc[node, "phase"]))
    candidates["target_phase"] = candidates["dst"].map(lambda node: str(node_lookup.loc[node, "phase"]))
    candidates = candidates[candidates["source_phase"].ne(candidates["target_phase"])].copy()
    additions_nodes: list[dict[str, Any]] = []
    additions_edges: list[dict[str, Any]] = []
    for edge in candidates.itertuples(index=True):
        transition = "F2B" if edge.source_phase == "FWD" else "B2F"
        duration = int(calibration["parameters"][transition]["p10_nonnegative_ns"])
        target = node_lookup.loc[str(edge.dst)].to_dict()
        node_id = f"handoff:{edge.src}__to__{edge.dst}"
        target.update({
            "node_id": node_id, "kind": "scheduler_handoff", "phase": "HANDOFF",
            "op_name": f"phase_handoff_{transition.lower()}", "op_family": "software_sync",
            "stream": "runtime", "resource": "host_runtime", "parallelism": "local",
            "shape_key": transition, "payload_key": "phase_token", "flops_formula": "0",
            "bytes_formula": "0", "code_anchor": "pipeline_schedules",
            "duration_ns": duration, "compute_exposed_ns_model": 0, "compute_overlap_ns_model": 0,
            "network_service_ns_model": 0, "software_sync_ns_model": duration,
            "framework_residual_ns_model": 0, "timing_component": "phase_handoff_software",
            "timing_source": "source256_nonnegative_phase_gap_p10",
            "semantic_slot": transition, "source_parameter_key": f"phase_handoff:{transition}:p10",
            "source_v54_node_id": "", "predicted_start_ns": 0, "predicted_end_ns": 0,
            "critical_predecessor": "", "on_critical_path": False,
        })
        additions_nodes.append(target)
        additions_edges.extend([
            edge_row(list(edges.columns), str(edge.src), node_id, "phase_handoff_start", transition),
            edge_row(list(edges.columns), node_id, str(edge.dst), "phase_handoff_complete", transition),
        ])
    remove_indices = candidates.index
    nodes = pd.concat([nodes, pd.DataFrame(additions_nodes)], ignore_index=True)
    edges = pd.concat([edges.drop(index=remove_indices), pd.DataFrame(additions_edges)], ignore_index=True)
    start, end, predecessor, critical = max_plus(nodes, edges, "iteration:completion_join")
    nodes["predicted_start_ns"] = start
    nodes["predicted_end_ns"] = end
    ids = nodes["node_id"].astype(str).tolist()
    nodes["critical_predecessor"] = [ids[value] if value >= 0 else "" for value in predecessor]
    nodes["on_critical_path"] = [position in critical for position in range(len(nodes))]
    handoffs = nodes[nodes["kind"].eq("scheduler_handoff")]
    counts = handoffs["semantic_slot"].value_counts().to_dict()
    expected = {"F2B": 272, "B2F": 48}
    if counts != expected:
        raise ValueError(f"target handoff edge count changed: {counts} != {expected}")
    audit = {
        "schema": "dag-v6.4a-phase-handoff-audit-v1", "status": "PASS",
        "handoff_node_count": int(len(handoffs)), "transition_counts": counts,
        "replaced_zero_duration_rank_program_edges": int(len(remove_indices)),
        "placement": config["handoff"]["placement"],
        "hidden_by_external_dependency_allowed": True,
        "all_handoffs_nonnegative": bool(handoffs["duration_ns"].ge(0).all()),
    }
    audit_path = output / "phase_handoff_audit.json"
    atomic_json(audit_path, audit)

    lookup = nodes.set_index("node_id")
    previous = json.loads(contract_input.read_text(encoding="utf-8"))
    raw_ms = int(lookup.loc["iteration:completion_join", "predicted_end_ns"]) / 1e6
    reconciliation = float(previous["prediction"]["source_reconciliation_ms"])
    outer = float(previous["prediction"]["outer_framework_ms"])
    profiler = raw_ms + reconciliation
    training = profiler + outer
    mfu = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        training / 1000.0 * int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"]) * 1e12
    )
    node_path = predictions_dir / "dag_v64a_nodes.csv.gz"
    edge_path = predictions_dir / "dag_v64a_edges.csv.gz"
    critical_path = predictions_dir / "dag_v64a_critical_path.csv"
    nodes.to_csv(node_path, index=False, compression="gzip")
    edges.to_csv(edge_path, index=False, compression="gzip")
    nodes[nodes["on_critical_path"]].sort_values("predicted_start_ns").to_csv(critical_path, index=False)
    grid = [*config["split"]["target_regression_only"], *config["split"]["target_validation"]]
    rows = [{
        "protocol_id": config["protocol"]["id"], "source_case": "256gpu_pp16_cp2_a2a",
        "target_case": config["target"]["case_id"], "method": "DAG v6.4a source phase handoff",
        "method_version": "dag-v6.4a-source60-100-v1", "status": "PREDICTIVE_PARTIAL_UNEVALUATED",
        "split": "regression_only" if int(iteration) == 55 else "validation", "iteration": int(iteration),
        "predicted_raw_graph_ms": raw_ms, "predicted_source_reconciliation_ms": reconciliation,
        "predicted_profiler_step_ms": profiler, "predicted_outer_framework_ms": outer,
        "predicted_training_step_ms": training, "predicted_mfu_pct": mfu, "target_timing_read": False,
    } for iteration in grid]
    method_path = predictions_dir / "method_predictions.csv"
    pd.DataFrame(rows).to_csv(method_path, index=False)
    contract = {
        "schema": "dag-v6.4a-phase-handoff-prediction-v1", "status": "PREDICTIVE_PARTIAL_UNEVALUATED",
        "target_workload_contract": config["handoff"]["target_layer_contract"],
        "prediction": {"raw_graph_ms": raw_ms, "source_reconciliation_ms": reconciliation,
                       "profiler_step_ms": profiler, "outer_framework_ms": outer,
                       "training_step_ms": training, "mfu_pct": mfu},
        "v631_raw_graph_ms": float(previous["prediction"]["raw_graph_ms"]),
        "v64a_minus_v631_raw_graph_ms": raw_ms - float(previous["prediction"]["raw_graph_ms"]),
        "handoff": {"nodes": int(len(handoffs)), "transition_counts": counts,
                    "parameters_ns": {key: value["p10_nonnegative_ns"] for key, value in calibration["parameters"].items()},
                    "placement": config["handoff"]["placement"]},
        "target_timing_read": False, "accuracy_claim_allowed": False,
    }
    contract_path = output / "prediction_contract.json"
    atomic_json(contract_path, contract)
    access_path = output / "input_access_audit.json"
    inputs = [node_input, edge_input, contract_input, seal_path, segment_path]
    atomic_json(access_path, {
        "schema": "dag-v6.4a-input-access-audit-v1", "status": "PASS_NO_TARGET_TIMING",
        "source_iterations": list(iterations), "target_timing_files_read": 0,
        "source_fields_read": ["iteration", "rank", "pp_stage", "phase", "microbatch", "observed_start_ns", "observed_end_ns"],
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in inputs],
    })
    report_path = output / "DAG_V64A_PHASE_HANDOFF_REPORT.md"
    atomic_text(report_path, f"""# DAG v6.4a：F/B phase handoff

保持当前真实的 52 层/PP14 workload 不变，在本地 1F1B 程序顺序中增加 320 个 handoff 节点。handoff 与 PP/autograd 外部依赖在 phase start 取 max，所以调度开销可以被更晚的通信依赖隐藏，不会无条件串行叠加。

- F→B source p10：`{calibration['parameters']['F2B']['p10_nonnegative_ns']/1e6:.6f} ms`
- B→F source p10：`{calibration['parameters']['B2F']['p10_nonnegative_ns']/1e6:.6f} ms`
- v6.3.1 raw graph：`{contract['v631_raw_graph_ms']:.6f} ms`
- v6.4a raw graph：`{raw_ms:.6f} ms`
- 暴露增量：`{contract['v64a_minus_v631_raw_graph_ms']:.6f} ms`

本版本没有应用 `256/224` 统一计算倍率，因为当前 224 卡实测合同是 52 层，而256卡源是60层；60层/PP14应作为独立预测场景构图。
""")
    reproduction = output / "reproduction_command.txt"
    atomic_text(reproduction, f"cd {REPO}\npython {Path(__file__).resolve()} --config {config_path}\n")
    log_path = logs_dir / "build.log"
    atomic_text(log_path, "\n".join([
        "DAG v6.4a phase handoff PASS", f"generated_at={datetime.now(timezone.utc).isoformat()}",
        f"raw_graph_ms={raw_ms:.9f}", f"v64a_minus_v631_raw_graph_ms={contract['v64a_minus_v631_raw_graph_ms']:.9f}",
        f"f2b_handoff_ns={calibration['parameters']['F2B']['p10_nonnegative_ns']}",
        f"b2f_handoff_ns={calibration['parameters']['B2F']['p10_nonnegative_ns']}",
        f"handoff_nodes={len(handoffs)}", "target_timing_files_read=0", "status=PREDICTIVE_PARTIAL_UNEVALUATED", "",
    ]))
    provenance_path = output / "provenance.json"
    atomic_json(provenance_path, {
        "schema": "dag-v6.4a-provenance-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO), "git_branch": git_output("branch", "--show-current"),
        "git_head": git_output("rev-parse", "HEAD"), "git_status_porcelain": git_output("status", "--short").splitlines(),
        "version_state": "PARTIAL_UNCOMMITTED_WORKTREE_PINNED_BY_SHA256",
        "code_and_config": [{"path": str(path), "sha256": sha256(path)} for path in (config_path, Path(__file__).resolve())],
        "v631_prediction_seal": {"path": str(seal_path), "sha256": sha256(seal_path)},
    })
    artifacts = [node_path, edge_path, critical_path, method_path, calibration_path, sample_path,
                 audit_path, contract_path, access_path, report_path, reproduction, log_path, provenance_path]
    atomic_json(output / "artifact_manifest.json", {
        "schema": "dag-v6.4a-artifact-manifest-v1",
        "artifacts": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts],
    })
    print(json.dumps({
        "status": contract["status"], "output": str(output), "raw_graph_ms": raw_ms,
        "v64a_minus_v631_raw_graph_ms": contract["v64a_minus_v631_raw_graph_ms"],
        "profiler_step_ms": profiler, "training_step_ms": training,
        "handoff_parameters_ms": {key: value["p10_nonnegative_ns"] / 1e6 for key, value in calibration["parameters"].items()},
        "handoff_nodes": int(len(handoffs)), "target_timing_files_read": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

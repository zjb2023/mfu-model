#!/usr/bin/env python3
"""Preserve source-256 global layer identity in the 224-GPU DAG clocks.

v5.4/v6.4a pooled all source internal PP stages by stage-local position.  That
made target PP1--PP12 identical before the max-plus schedule was evaluated.
v6.5 instead transfers the same global layer id for local compute, CP/EP arrival
gaps, and CP/EP rank completion.  PP and the DP/Expert-DP optimizer tail remain
the sealed v6.4a backends.

The model process accepts only source-256 calibration artifacts plus the sealed
target-static graph.  It never opens target profiler/training timing.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
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
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v65_global_layer_transfer_2026w36.toml"
SOURCE_ITERATIONS = tuple(range(60, 101, 5))
PHASE = {"FWD": "forward", "BWD": "backward"}
LOCAL_KEY = [
    "pp_stage", "pp_lane", "phase", "microbatch", "stage_local_layer", "semantic",
]


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


def verify_seal(path: Path, status: str) -> dict[str, Any]:
    payload = json.loads(checked(path).read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise ValueError(f"unexpected prediction seal: {payload.get('status')} != {status}")
    for artifact in payload["artifacts"]:
        source = checked(Path(artifact["path"]))
        if source.stat().st_size != int(artifact["size_bytes"]) or sha256(source) != artifact["sha256"]:
            raise ValueError(f"sealed artifact changed: {source}")
    return payload


def source_compute_parameters(path: Path) -> pd.DataFrame:
    columns = [
        "iteration", "rank", "phase", "microbatch", "layer_id", "autograd_phase",
        "parameter_view", "execution_scope", "semantic_slot", "cost_group", "active_union_ns",
    ]
    observations = pd.read_csv(path, compression="gzip", usecols=columns)
    physical = observations[
        observations["parameter_view"].eq("window_split")
        & observations["cost_group"].eq("__physical_slot_total__")
        & observations["autograd_phase"].eq("physical_mixed")
        & observations["layer_id"].ge(0)
    ].copy()
    observed = tuple(sorted(physical["iteration"].astype(int).unique()))
    if observed != SOURCE_ITERATIONS:
        raise ValueError(f"source compute iteration grid changed: {observed}")
    keys = ["phase", "microbatch", "layer_id", "execution_scope", "semantic_slot"]
    grouped = physical.groupby(keys, as_index=False).agg(
        calibration_samples=("active_union_ns", "size"),
        calibration_iterations=("iteration", "nunique"),
        source_rank_count=("rank", "nunique"),
        median_active_union_ns=("active_union_ns", "median"),
        p10_active_union_ns=("active_union_ns", lambda values: values.quantile(0.10)),
        p90_active_union_ns=("active_union_ns", lambda values: values.quantile(0.90)),
        mean_active_union_ns=("active_union_ns", "mean"),
        std_active_union_ns=("active_union_ns", lambda values: values.std(ddof=0)),
    )
    integer_columns = [
        "median_active_union_ns", "p10_active_union_ns", "p90_active_union_ns",
        "mean_active_union_ns", "std_active_union_ns",
    ]
    for column in integer_columns:
        grouped[column] = grouped[column].round().astype("int64")
    grouped["coefficient_of_variation"] = grouped.apply(
        lambda row: float(row.std_active_union_ns / row.mean_active_union_ns)
        if int(row.mean_active_union_ns) else 0.0,
        axis=1,
    )
    grouped["parameter_unit"] = "ns_per_rank_layer_microbatch_semantic_slot"
    grouped["fit_source"] = "256gpu_iterations_60_100_lane0_exact_global_layer"
    grouped["target_timing_read"] = False
    return grouped.sort_values(keys).reset_index(drop=True)


def semantic_for_source(slot: str) -> str:
    return f"{slot}_wall" if slot.startswith("ep_") else slot


def local_lookup(frame: pd.DataFrame, values: list[str]) -> dict[tuple[Any, ...], dict[str, int]]:
    if frame.duplicated(LOCAL_KEY).any():
        raise ValueError("source local parameter key is not unique")
    result: dict[tuple[Any, ...], dict[str, int]] = {}
    for row in frame.itertuples(index=False):
        key = tuple(getattr(row, column) for column in LOCAL_KEY)
        result[key] = {column: int(getattr(row, column)) for column in values}
    return result


def render_report(contract: dict[str, Any], audit: dict[str, Any], stage_delta: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| PP{int(row.pp_stage)} | {row.prior_clock_ms:.3f} | {row.v65_clock_ms:.3f} | {row.delta_ms:+.3f} |"
        for row in stage_delta.itertuples(index=False)
    )
    return f"""# DAG v6.5：按全局层迁移 256 卡参数

v6.4a 将 256 卡 PP1--PP14 的相同 stage-local 位置取中位数，导致 224 卡 PP1--PP12 的本地时钟完全相同。v6.5 保留全局 `layer_id`，把同名源层的计算、CP/EP 前间隔、网络 service 和软件完成时间迁移到目标 DAG。

- 实际目标仍为 52 层，放置为 `[2, 4×12, 2]`；不是 42 层或 60 层假设。
- PP、DP/Expert-DP 尾部、AG barrier、F/B handoff 沿用封存的 v6.4a。
- PP13 的 BWD layer51 入口和 FWD layer51 退出在源 60 层图中不是同类边界，保留 v6.4a 末 stage 边界模板。
- model process 读取目标时序文件数：`{audit['target_timing_files_read']}`。
- raw graph：`{contract['prediction']['raw_graph_ms']:.6f} ms`；相对 v6.4a：`{contract['v65_minus_v64a_raw_graph_ms']:+.6f} ms`。
- profiler / training 双时钟：`{contract['prediction']['profiler_step_ms']:.6f} / {contract['prediction']['training_step_ms']:.6f} ms`。

## 各 PP stage 本地 clock 总量（每 lane，3 microbatch，FWD+BWD）

| stage | v6.4a ms | v6.5 ms | delta ms |
|---:|---:|---:|---:|
{rows}

本版模型结构是在查看过 224 卡诊断后选择的，因此后续 60--100 只能称为 retrospective development-set check，不能称为未见验证集。参数数值仍只来自 256 卡 iteration 60--100。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if bool(config["split"]["target_timing_allowed"]):
        raise ValueError("target timing must remain disabled")
    if tuple(int(value) for value in config["split"]["source_calibration"]) != SOURCE_ITERATIONS:
        raise ValueError("source calibration must be exact stable 60--100")

    inputs = {name: checked(configured_path(value)) for name, value in config["inputs"].items() if name != "v64a_run_dir"}
    prior_run = configured_path(config["inputs"]["v64a_run_dir"])
    prior_seal_path = checked(prior_run / "predictions/prediction_seal.json")
    verify_seal(prior_seal_path, "SEALED_BEFORE_V64A_EVALUATOR_ACCESS")
    prior_access = json.loads(checked(prior_run / "input_access_audit.json").read_text())
    compute_access = json.loads(inputs["source_compute_access_audit"].read_text())
    local_access = json.loads(inputs["source_local_access_audit"].read_text())
    if int(prior_access.get("target_timing_files_read", -1)) != 0:
        raise ValueError("v6.4a model process was not target-safe")
    if int(compute_access.get("target_timing_files_read", -1)) != 0:
        raise ValueError("source compute calibration was not target-safe")
    if local_access.get("target_profiler_or_training_timing_read") is not False:
        raise ValueError("source local calibration was not target-safe")

    output = configured_path(config["outputs"]["output_dir"])
    prediction_dir, calibration_dir, model_dir, logs_dir = (
        output / "predictions", output / "calibration", output / "model_inputs", output / "logs"
    )
    for directory in (prediction_dir, calibration_dir, model_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_map = pd.read_csv(inputs["source_layer_stage_map"])
    target_map = pd.read_csv(inputs["target_layer_stage_map"])
    if source_map["layer_id"].astype(int).tolist() != list(range(60)):
        raise ValueError("source layer map must contain contiguous layers 0--59")
    if target_map["layer_id"].astype(int).tolist() != list(range(52)):
        raise ValueError("target layer map must contain contiguous layers 0--51")
    source_by_layer = source_map.set_index("layer_id")
    transfer = target_map.rename(columns={
        "pp_stage": "target_pp_stage", "stage_local_layer": "target_stage_local_layer",
        "stage_layer_count": "target_stage_layer_count", "virtual_chunk": "target_virtual_chunk",
    }).copy()
    transfer["source_layer_id"] = transfer["layer_id"]
    for source_column, target_column in (
        ("pp_stage", "source_pp_stage"), ("stage_local_layer", "source_stage_local_layer"),
        ("stage_layer_count", "source_stage_layer_count"), ("virtual_chunk", "source_virtual_chunk"),
    ):
        transfer[target_column] = transfer["source_layer_id"].map(source_by_layer[source_column]).astype(int)
    transfer["transfer_policy"] = "same_global_layer_id"
    transfer_path = model_dir / "global_layer_transfer_map.csv"
    atomic_csv(transfer_path, transfer)

    compute_parameters = source_compute_parameters(inputs["source_compute_observations"])
    compute_path = calibration_dir / "source_global_layer_compute_parameters.csv"
    atomic_csv(compute_path, compute_parameters)
    compute_keys = ["phase", "microbatch", "layer_id", "execution_scope", "semantic_slot"]
    compute_lookup = {
        tuple(getattr(row, column) for column in compute_keys): int(row.median_active_union_ns)
        for row in compute_parameters.itertuples(index=False)
    }

    gaps = pd.read_csv(inputs["source_local_gap_parameters"])
    completions = pd.read_csv(inputs["source_local_communication_parameters"])
    gap_lookup = local_lookup(gaps, ["gap_before_ns_median"])
    completion_lookup = local_lookup(
        completions, ["predicted_duration_ns", "network_service_ns", "software_completion_ns"]
    )

    node_input = checked(prior_run / "predictions/dag_v64a_nodes.csv.gz")
    edge_input = checked(prior_run / "predictions/dag_v64a_edges.csv.gz")
    contract_input = checked(prior_run / "prediction_contract.json")
    nodes = pd.read_csv(node_input, low_memory=False)
    edges = pd.read_csv(edge_input, low_memory=False)
    prior_clock = nodes[["node_id", "pp_stage", "timing_component", "duration_ns"]].copy()
    prior_clock = prior_clock[prior_clock["timing_component"].isin({
        "local_collective_completion", "physical_exposed_slot_clock",
        "physical_overlap_slot_clock", "physical_step_exit_clock",
    })]
    source_location = {
        int(row.layer_id): (int(row.source_pp_stage), int(row.source_stage_local_layer))
        for row in transfer.itertuples(index=False)
    }
    stats: Counter[str] = Counter()
    bindings: list[dict[str, Any]] = []

    def source_key(row: pd.Series, semantic: str) -> tuple[Any, ...]:
        layer = int(row["layer_id"])
        source_stage, source_local = source_location[layer]
        return (
            source_stage, int(row["rank"]) % 16, PHASE[str(row["phase"])],
            int(row["microbatch"]), source_local, semantic_for_source(semantic),
        )

    services = nodes["timing_component"].eq("local_collective_completion")
    for index, row in nodes.loc[services].iterrows():
        key = source_key(row, str(row["semantic_slot"]))
        value = completion_lookup.get(key)
        if value is None:
            raise KeyError(f"missing source completion: {key}")
        nodes.at[index, "duration_ns"] = value["predicted_duration_ns"]
        nodes.at[index, "network_service_ns_model"] = value["network_service_ns"]
        nodes.at[index, "software_sync_ns_model"] = value["software_completion_ns"]
        nodes.at[index, "timing_source"] = "source256_exact_global_layer_local_completion"
        nodes.at[index, "source_parameter_key"] = "local_completion|" + "|".join(map(str, key))
        bindings.append({
            "node_id": row["node_id"], "binding_kind": "local_collective_completion",
            "target_pp_stage": int(row["pp_stage"]), "target_layer_id": int(row["layer_id"]),
            "source_parameter_key": "|".join(map(str, key)), "boundary_fallback": False,
            "compute_parameter_observed": False, "prior_duration_ns": int(row["duration_ns"]),
            "v65_duration_ns": value["predicted_duration_ns"],
        })
        stats["local_completion_exact_layer"] += 1

    exposed = nodes["timing_component"].eq("physical_exposed_slot_clock")
    for index, row in nodes.loc[exposed].iterrows():
        layer, slot = int(row["layer_id"]), str(row["semantic_slot"])
        if str(row["phase"]) == "BWD" and layer == 51 and slot == "before_cp0":
            stats["boundary_fallback_bwd_layer51_entry"] += 1
            bindings.append({
                "node_id": row["node_id"], "binding_kind": "physical_exposed_slot_clock",
                "target_pp_stage": int(row["pp_stage"]), "target_layer_id": layer,
                "source_parameter_key": str(row["source_parameter_key"]), "boundary_fallback": True,
                "compute_parameter_observed": bool(int(row["compute_exposed_ns_model"])),
                "prior_duration_ns": int(row["duration_ns"]), "v65_duration_ns": int(row["duration_ns"]),
            })
            continue
        key = source_key(row, slot.removeprefix("before_"))
        gap_ns = gap_lookup.get(key, {}).get("gap_before_ns_median")
        if gap_ns is None:
            raise KeyError(f"missing source local gap: {key}")
        compute_key = (PHASE[str(row["phase"])], int(row["microbatch"]), layer, "exposed_gap", slot)
        observed = compute_key in compute_lookup
        compute_ns = compute_lookup.get(compute_key, 0)
        residual_ns = max(int(gap_ns) - int(compute_ns), 0)
        duration_ns = max(int(gap_ns), int(compute_ns))
        nodes.at[index, "duration_ns"] = duration_ns
        nodes.at[index, "compute_exposed_ns_model"] = compute_ns
        nodes.at[index, "framework_residual_ns_model"] = residual_ns
        nodes.at[index, "timing_source"] = "source256_exact_global_layer_compute_plus_local_gap"
        nodes.at[index, "source_parameter_key"] = "compute_gap|" + "|".join(map(str, compute_key))
        bindings.append({
            "node_id": row["node_id"], "binding_kind": "physical_exposed_slot_clock",
            "target_pp_stage": int(row["pp_stage"]), "target_layer_id": layer,
            "source_parameter_key": "|".join(map(str, compute_key)), "boundary_fallback": False,
            "compute_parameter_observed": observed, "prior_duration_ns": int(row["duration_ns"]),
            "v65_duration_ns": duration_ns,
        })
        stats["exposed_exact_layer"] += 1
        stats["exposed_compute_observed" if observed else "exposed_zero_compute_unobserved"] += 1

    overlap = nodes["timing_component"].eq("physical_overlap_slot_clock")
    for index, row in nodes.loc[overlap].iterrows():
        compute_key = (
            PHASE[str(row["phase"])], int(row["microbatch"]), int(row["layer_id"]),
            "overlap_with_collective", str(row["semantic_slot"]),
        )
        observed = compute_key in compute_lookup
        compute_ns = compute_lookup.get(compute_key, 0)
        nodes.at[index, "duration_ns"] = compute_ns
        nodes.at[index, "compute_overlap_ns_model"] = compute_ns
        nodes.at[index, "timing_source"] = "source256_exact_global_layer_compute_overlap"
        nodes.at[index, "source_parameter_key"] = "compute_overlap|" + "|".join(map(str, compute_key))
        bindings.append({
            "node_id": row["node_id"], "binding_kind": "physical_overlap_slot_clock",
            "target_pp_stage": int(row["pp_stage"]), "target_layer_id": int(row["layer_id"]),
            "source_parameter_key": "|".join(map(str, compute_key)), "boundary_fallback": False,
            "compute_parameter_observed": observed, "prior_duration_ns": int(row["duration_ns"]),
            "v65_duration_ns": compute_ns,
        })
        stats["overlap_exact_layer"] += 1
        stats["overlap_compute_observed" if observed else "overlap_zero_compute_unobserved"] += 1

    target_layers = {
        int(stage): [int(value) for value in group.sort_values("stage_local_layer")["layer_id"]]
        for stage, group in target_map.groupby("pp_stage")
    }
    step_exit = nodes["timing_component"].eq("physical_step_exit_clock")
    for index, row in nodes.loc[step_exit].iterrows():
        stage, phase = int(row["pp_stage"]), str(row["phase"])
        # v6 boundary nodes carry layer_id=-1; infer the terminal layer from target placement.
        layer = target_layers[stage][-1] if phase == "FWD" else target_layers[stage][0]
        compute_key = (PHASE[phase], int(row["microbatch"]), layer, "exposed_gap", "step_exit")
        if phase == "FWD" and stage == 13:
            stats["boundary_fallback_fwd_layer51_exit"] += 1
            bindings.append({
                "node_id": row["node_id"], "binding_kind": "physical_step_exit_clock",
                "target_pp_stage": stage, "target_layer_id": layer,
                "source_parameter_key": str(row["source_parameter_key"]), "boundary_fallback": True,
                "compute_parameter_observed": bool(int(row["compute_exposed_ns_model"])),
                "prior_duration_ns": int(row["duration_ns"]), "v65_duration_ns": int(row["duration_ns"]),
            })
            continue
        if phase != "FWD" or compute_key not in compute_lookup:
            stats["step_exit_retained_no_exact_source_semantics"] += 1
            continue
        compute_ns = compute_lookup[compute_key]
        nodes.at[index, "duration_ns"] = compute_ns
        nodes.at[index, "compute_exposed_ns_model"] = compute_ns
        nodes.at[index, "timing_source"] = "source256_exact_global_layer_step_exit"
        nodes.at[index, "source_parameter_key"] = "step_exit|" + "|".join(map(str, compute_key))
        bindings.append({
            "node_id": row["node_id"], "binding_kind": "physical_step_exit_clock",
            "target_pp_stage": stage, "target_layer_id": layer,
            "source_parameter_key": "|".join(map(str, compute_key)), "boundary_fallback": False,
            "compute_parameter_observed": True, "prior_duration_ns": int(row["duration_ns"]),
            "v65_duration_ns": compute_ns,
        })
        stats["step_exit_exact_layer"] += 1

    start, end, predecessor, critical = max_plus(nodes, edges, "iteration:completion_join")
    nodes["predicted_start_ns"] = start
    nodes["predicted_end_ns"] = end
    ids = nodes["node_id"].astype(str).tolist()
    nodes["critical_predecessor"] = [ids[value] if value >= 0 else "" for value in predecessor]
    nodes["on_critical_path"] = [position in critical for position in range(len(nodes))]

    node_path = prediction_dir / "dag_v65_nodes.csv.gz"
    edge_path = prediction_dir / "dag_v65_edges.csv.gz"
    critical_path = prediction_dir / "dag_v65_critical_path.csv"
    atomic_csv(node_path, nodes, compression="gzip")
    atomic_csv(edge_path, edges, compression="gzip")
    atomic_csv(critical_path, nodes[nodes["on_critical_path"]].sort_values("predicted_start_ns"))
    binding_path = model_dir / "global_layer_clock_bindings.csv.gz"
    atomic_csv(binding_path, pd.DataFrame(bindings), compression="gzip")

    clock_kinds = {
        "local_collective_completion", "physical_exposed_slot_clock",
        "physical_overlap_slot_clock", "physical_step_exit_clock",
    }
    prior_stage = prior_clock.groupby("pp_stage", as_index=False)["duration_ns"].sum().rename(
        columns={"duration_ns": "prior_duration_ns"}
    )
    current_stage = nodes[nodes["timing_component"].isin(clock_kinds)].groupby(
        "pp_stage", as_index=False
    )["duration_ns"].sum().rename(columns={"duration_ns": "v65_duration_ns"})
    stage_delta = prior_stage.merge(current_stage, on="pp_stage", validate="one_to_one")
    # Both graphs contain 16 lanes; report one-lane total across three microbatches.
    stage_delta["prior_clock_ms"] = stage_delta["prior_duration_ns"] / 16e6
    stage_delta["v65_clock_ms"] = stage_delta["v65_duration_ns"] / 16e6
    stage_delta["delta_ms"] = stage_delta["v65_clock_ms"] - stage_delta["prior_clock_ms"]
    stage_delta_path = calibration_dir / "pp_stage_local_clock_delta.csv"
    atomic_csv(stage_delta_path, stage_delta)

    prior_contract = json.loads(contract_input.read_text(encoding="utf-8"))
    final_index = int(nodes.index[nodes["node_id"].eq("iteration:completion_join")][0])
    raw_ms = int(end[final_index]) / 1e6
    reconciliation = float(prior_contract["prediction"]["source_reconciliation_ms"])
    outer = float(prior_contract["prediction"]["outer_framework_ms"])
    profiler = raw_ms + reconciliation
    training = profiler + outer
    mfu = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        training / 1000.0 * int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"]) * 1e12
    )
    grid = [*config["split"]["target_regression_only"], *config["split"]["target_validation"]]
    method = pd.DataFrame([{
        "protocol_id": config["protocol"]["id"], "source_case": config["source"]["case_id"],
        "target_case": config["target"]["case_id"], "method": "DAG v6.5 exact global layer transfer",
        "method_version": "dag-v6.5-source60-100-v1",
        "status": "PREDICTIVE_RETROSPECTIVE_DEVELOPMENT_UNEVALUATED",
        "split": "regression_only" if int(iteration) == 55 else "validation",
        "iteration": int(iteration), "predicted_raw_graph_ms": raw_ms,
        "predicted_source_reconciliation_ms": reconciliation,
        "predicted_profiler_step_ms": profiler, "predicted_outer_framework_ms": outer,
        "predicted_training_step_ms": training, "predicted_mfu_pct": mfu,
        "target_timing_read": False,
    } for iteration in grid])
    method_path = prediction_dir / "method_predictions.csv"
    atomic_csv(method_path, method)

    audit = {
        "schema": "dag-v6.5-global-layer-transfer-audit-v1", "status": "PASS_SOURCE_ONLY_MODEL_PROCESS",
        "source_iterations": list(SOURCE_ITERATIONS), "source_global_layers_available": [0, 59],
        "target_global_layers_transferred": [0, 51], "target_layer_count": 52,
        "target_layer_placement": [2, *([4] * 12), 2], "target_timing_files_read": 0,
        "parameter_counts": dict(sorted(stats.items())),
        "changed_boundary_fallbacks": {
            "target_pp13_bwd_layer51_before_cp0": int(stats["boundary_fallback_bwd_layer51_entry"]),
            "target_pp13_fwd_layer51_step_exit": int(stats["boundary_fallback_fwd_layer51_exit"]),
        },
        "unchanged_backend_checks": {
            "node_count_unchanged": int(len(nodes)), "edge_count_unchanged": int(len(edges)),
            "pp_node_count": int(nodes["timing_component"].eq("pp_trace_wall").sum()),
            "optimizer_tail_node_count": int(nodes["phase"].eq("OPT").sum()),
            "phase_handoff_node_count": int(nodes["kind"].eq("scheduler_handoff").sum()),
        },
        "model_selection_disclosure": "target224 diagnostics informed the decision to preserve global layer identity; no target timing value is read or fitted by this builder",
    }
    audit_path = output / "global_layer_transfer_audit.json"
    atomic_json(audit_path, audit)
    contract = {
        "schema": "dag-v6.5-global-layer-prediction-v1",
        "status": "PREDICTIVE_RETROSPECTIVE_DEVELOPMENT_UNEVALUATED",
        "target_workload_contract": "actual 52-layer PP14 placement [2,4x12,2]",
        "prediction": {
            "raw_graph_ms": raw_ms, "source_reconciliation_ms": reconciliation,
            "profiler_step_ms": profiler, "outer_framework_ms": outer,
            "training_step_ms": training, "mfu_pct": mfu,
        },
        "v64a_raw_graph_ms": float(prior_contract["prediction"]["raw_graph_ms"]),
        "v65_minus_v64a_raw_graph_ms": raw_ms - float(prior_contract["prediction"]["raw_graph_ms"]),
        "parameter_source": "256gpu iterations 60--100 exact global layer id",
        "target_timing_read": False, "accuracy_claim_allowed": False,
        "untouched_validation_claim_allowed": False,
    }
    contract_path = output / "prediction_contract.json"
    atomic_json(contract_path, contract)

    input_files = [
        node_input, edge_input, contract_input, prior_seal_path, config_path,
        inputs["source_layer_stage_map"], inputs["target_layer_stage_map"],
        inputs["source_compute_observations"], inputs["source_compute_access_audit"],
        inputs["source_local_gap_parameters"], inputs["source_local_communication_parameters"],
        inputs["source_local_access_audit"],
    ]
    access_path = output / "input_access_audit.json"
    atomic_json(access_path, {
        "schema": "dag-v6.5-input-access-audit-v1", "status": "PASS_NO_TARGET_TIMING",
        "source_iterations": list(SOURCE_ITERATIONS), "target_timing_files_read": 0,
        "target_fields_read": ["static layer_stage_map", "sealed v6.4a graph structure and source-only clocks"],
        "forbidden_target_fields_read": [],
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in input_files],
    })
    report_path = output / "DAG_V65_GLOBAL_LAYER_TRANSFER.md"
    atomic_text(report_path, render_report(contract, audit, stage_delta))
    reproduction_path = output / "reproduction_command.txt"
    atomic_text(reproduction_path, f"cd {REPO}\npython {Path(__file__).resolve()} --config {config_path}\n")
    log_path = logs_dir / "build.log"
    atomic_text(log_path, "\n".join([
        "DAG v6.5 exact global layer transfer PASS",
        f"generated_at={datetime.now(timezone.utc).isoformat()}",
        f"raw_graph_ms={raw_ms:.9f}", f"profiler_step_ms={profiler:.9f}",
        f"training_step_ms={training:.9f}", f"v65_minus_v64a_raw_graph_ms={contract['v65_minus_v64a_raw_graph_ms']:.9f}",
        f"exact_local_completion_nodes={stats['local_completion_exact_layer']}",
        f"exact_exposed_nodes={stats['exposed_exact_layer']}",
        f"exact_overlap_nodes={stats['overlap_exact_layer']}",
        f"boundary_fallback_nodes={stats['boundary_fallback_bwd_layer51_entry'] + stats['boundary_fallback_fwd_layer51_exit']}",
        "target_timing_files_read=0", "status=PREDICTIVE_RETROSPECTIVE_DEVELOPMENT_UNEVALUATED", "",
    ]))
    provenance_path = output / "provenance.json"
    atomic_json(provenance_path, {
        "schema": "dag-v6.5-provenance-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO), "git_branch": git_output("branch", "--show-current"),
        "git_head": git_output("rev-parse", "HEAD"), "git_status_porcelain": git_output("status", "--short").splitlines(),
        "version_state": "PARTIAL_UNCOMMITTED_WORKTREE_PINNED_BY_SHA256",
        "code_and_config": [{"path": str(path), "sha256": sha256(path)} for path in (config_path, Path(__file__).resolve())],
        "prior_prediction_seal": {"path": str(prior_seal_path), "sha256": sha256(prior_seal_path)},
        "target_timing_files_read": 0,
    })
    artifacts = [
        node_path, edge_path, critical_path, method_path, compute_path, transfer_path,
        binding_path, stage_delta_path, audit_path, contract_path, access_path, report_path,
        reproduction_path, log_path, provenance_path,
    ]
    manifest_path = output / "artifact_manifest.json"
    atomic_json(manifest_path, {
        "schema": "dag-v6.5-artifact-manifest-v1",
        "artifacts": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts],
    })
    print(json.dumps({
        "status": contract["status"], "output": str(output), "raw_graph_ms": raw_ms,
        "v65_minus_v64a_raw_graph_ms": contract["v65_minus_v64a_raw_graph_ms"],
        "profiler_step_ms": profiler, "training_step_ms": training,
        "parameter_counts": dict(sorted(stats.items())), "target_timing_files_read": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

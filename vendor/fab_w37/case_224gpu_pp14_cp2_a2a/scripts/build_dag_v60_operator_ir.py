#!/usr/bin/env python3
"""Build the structure-only DAG v6 operator IR for the 256 -> 224 study.

This milestone deliberately assigns no durations.  It translates the captured
236B launch semantics into explicit compute, communication-service and
software-synchronisation nodes, then instantiates the same rules for PP16/M4
and PP14/M3.  Target profiler/training timestamps are not inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v60_operator_ir_2026w36.toml"

NODE_COLUMNS = [
    "case_id", "node_id", "kind", "rank", "pp_stage", "pp_lane", "phase",
    "microbatch", "schedule_region", "operation_sequence", "layer_id",
    "stage_local_layer", "layer_type", "autograd_phase", "op_name", "op_family",
    "stream", "resource", "parallelism", "dtype", "shape_key", "payload_key",
    "flops_formula", "bytes_formula", "cost_status", "cost_source", "code_anchor",
    "template_index",
]

EDGE_COLUMNS = [
    "case_id", "src", "dst", "edge_type", "tensor_key", "dependency_source",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = (
        gzip.open(path, "wt", newline="", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("w", newline="", encoding="utf-8")
    )
    with handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class OpSpec:
    name: str
    family: str
    resource: str = "gpu_compute"
    stream: str = "compute"
    parallelism: str = "local"
    shape: str = "hidden_state"
    payload: str = ""
    flops: str = "symbolic_unbound"
    bytes_: str = "symbolic_unbound"
    anchor: str = "fine_grained_callables"


def comm_triplet(prefix: str, parallelism: str, payload: str, anchor: str) -> list[OpSpec]:
    return [
        OpSpec(
            f"{prefix}_arrival_join", "software_sync", "host_runtime", "runtime",
            parallelism, "rank_release_vector", payload, "0", "0", anchor,
        ),
        OpSpec(
            f"{prefix}_service", "collective_service", "network", f"{parallelism}_network",
            parallelism, "collective_group", payload, "0", "payload_bytes", anchor,
        ),
        OpSpec(
            f"{prefix}_completion_sync", "software_sync", "host_runtime", "runtime",
            parallelism, "rank_completion_vector", payload, "0", "0", anchor,
        ),
    ]


def attention_forward() -> list[OpSpec]:
    ops = [
        OpSpec("input_rmsnorm", "normalization"),
        OpSpec("mla_q_down_projection", "attention_projection", shape="tokens_x_q_lora"),
        OpSpec("mla_kv_down_projection_rope", "attention_projection", shape="tokens_x_kv_lora"),
        OpSpec("mla_qkv_up_projection_and_rope", "attention_projection", shape="tokens_x_heads_x_head_dim"),
    ]
    # Raw AttnFuncWithCPAndQKVOA2A trace: Q/K/V A2As precede flash
    # attention and one output A2A follows it.  This is 3+1, not 2+2.
    ops += comm_triplet("cp_a2a_query", "cp", "cp_query_384mib_logical", "transformer_engine_wrapper")
    ops += comm_triplet("cp_a2a_key", "cp", "cp_key_384mib_logical", "transformer_engine_wrapper")
    ops += comm_triplet("cp_a2a_value", "cp", "cp_value_256mib_logical", "transformer_engine_wrapper")
    ops += [OpSpec("flash_attention", "attention_kernel", shape="heads_x_local_sequence_squared")]
    ops += comm_triplet("cp_a2a_output", "cp", "cp_output_256mib_logical", "transformer_engine_wrapper")
    ops += [
        OpSpec("attention_output_projection", "attention_projection"),
        OpSpec("attention_residual_add", "elementwise"),
    ]
    return ops


def dense_mlp_forward() -> list[OpSpec]:
    return [
        OpSpec("pre_mlp_rmsnorm", "normalization"),
        OpSpec("dense_fc1", "dense_gemm", shape="tokens_x_dense_ffn"),
        OpSpec("dense_swiglu", "activation", shape="tokens_x_dense_ffn"),
        OpSpec("dense_fc2", "dense_gemm", shape="tokens_x_hidden"),
        OpSpec("dense_mlp_residual_add", "elementwise"),
    ]


def moe_forward() -> list[OpSpec]:
    ops = [
        OpSpec("pre_moe_rmsnorm", "normalization"),
        OpSpec("router_logits", "moe_router", shape="tokens_x_experts"),
        OpSpec("router_topk", "moe_router", shape="tokens_x_topk"),
        OpSpec("token_permute", "moe_permutation", shape="routed_tokens"),
    ]
    ops += comm_triplet("ep_dispatch", "ep", "routed_token_payload", "fine_grained_callables")
    ops += [
        # MoELayer.experts_compute executes the non-overlapped shared expert
        # before dispatch_postprocess and routed experts.
        OpSpec("shared_expert_fc1", "shared_expert_gemm", shape="tokens_x_shared_expert"),
        OpSpec("shared_expert_swiglu", "activation", shape="tokens_x_shared_expert"),
        OpSpec("shared_expert_fc2", "shared_expert_gemm", shape="tokens_x_hidden"),
        OpSpec("dispatch_postprocess", "moe_permutation", shape="routed_tokens"),
        OpSpec("grouped_expert_fc1", "expert_gemm", shape="expert_tokens_x_moe_ffn"),
        OpSpec("expert_swiglu", "activation", shape="expert_tokens_x_moe_ffn"),
        OpSpec("grouped_expert_fc2", "expert_gemm", shape="expert_tokens_x_hidden"),
        OpSpec("combine_preprocess", "moe_permutation", shape="routed_tokens"),
    ]
    ops += comm_triplet("ep_combine", "ep", "routed_token_payload", "fine_grained_callables")
    ops += [
        OpSpec("token_unpermute", "moe_permutation", shape="routed_tokens"),
        OpSpec("moe_mlp_residual_add", "elementwise"),
    ]
    return ops


def attention_backward() -> list[OpSpec]:
    ops = [
        OpSpec("attention_output_projection_backward", "attention_projection_backward"),
    ]
    # Raw AttnFuncWithCPAndQKVOA2ABackward trace: two 256-MiB logical
    # A2As precede flash backward; Q/K/V gradient A2As follow it.
    ops += comm_triplet("cp_a2a_grad_output_0", "cp", "cp_grad_output_256mib_logical", "transformer_engine_wrapper")
    ops += comm_triplet("cp_a2a_grad_output_1", "cp", "cp_grad_output_256mib_logical", "transformer_engine_wrapper")
    ops.append(OpSpec("flash_attention_backward", "attention_kernel_backward"))
    ops += comm_triplet("cp_a2a_grad_query", "cp", "cp_grad_query_384mib_logical", "transformer_engine_wrapper")
    ops += comm_triplet("cp_a2a_grad_key", "cp", "cp_grad_key_384mib_logical", "transformer_engine_wrapper")
    ops += comm_triplet("cp_a2a_grad_value", "cp", "cp_grad_value_256mib_logical", "transformer_engine_wrapper")
    ops += [
        OpSpec("mla_up_projection_backward", "attention_projection_backward"),
        OpSpec("mla_kv_down_projection_backward", "attention_projection_backward"),
        OpSpec("mla_q_down_projection_backward", "attention_projection_backward"),
        OpSpec("input_rmsnorm_backward", "normalization_backward"),
    ]
    return ops


def dense_mlp_backward() -> list[OpSpec]:
    return [
        OpSpec("dense_fc2_backward", "dense_gemm_backward"),
        OpSpec("dense_swiglu_backward", "activation_backward"),
        OpSpec("dense_fc1_backward", "dense_gemm_backward"),
        OpSpec("pre_mlp_rmsnorm_backward", "normalization_backward"),
    ]


def moe_backward() -> list[OpSpec]:
    ops = [
        OpSpec("token_unpermute_backward", "moe_permutation_backward"),
    ]
    ops += comm_triplet("ep_combine_backward", "ep", "routed_gradient_payload", "fine_grained_callables")
    ops += [
        # The grouped/shared local backward order is a code-guided modelling
        # assumption; the source trace does not label these kernels separately.
        OpSpec("grouped_expert_fc2_backward", "expert_gemm_backward"),
        OpSpec("expert_swiglu_backward", "activation_backward"),
        OpSpec("grouped_expert_fc1_backward", "expert_gemm_backward"),
        OpSpec("shared_expert_fc2_backward", "shared_expert_gemm_backward"),
        OpSpec("shared_expert_swiglu_backward", "activation_backward"),
        OpSpec("shared_expert_fc1_backward", "shared_expert_gemm_backward"),
    ]
    ops += comm_triplet("ep_dispatch_backward", "ep", "routed_gradient_payload", "fine_grained_callables")
    ops += [
        OpSpec("token_permute_backward", "moe_permutation_backward"),
        OpSpec("router_topk_backward", "moe_router_backward"),
        OpSpec("router_logits_backward", "moe_router_backward"),
        OpSpec("pre_moe_rmsnorm_backward", "normalization_backward"),
    ]
    return ops


def template_catalog() -> dict[str, list[OpSpec]]:
    dense_fwd = attention_forward() + dense_mlp_forward()
    moe_fwd = attention_forward() + moe_forward()
    # Full activation recomputation is explicit.  Recompute nodes use the same
    # code operations but are tagged with a separate autograd phase.
    return {
        "dense_forward": dense_fwd,
        "moe_forward": moe_fwd,
        "dense_backward_recompute": dense_fwd + dense_mlp_backward() + attention_backward(),
        "moe_backward_recompute": moe_fwd + moe_backward() + attention_backward(),
        "mtp_forward_template_inactive": [
            OpSpec("mtp_input_projection", "mtp_compute"),
            OpSpec("mtp_transformer_layer", "mtp_compute"),
            OpSpec("mtp_loss", "mtp_compute"),
        ],
    }


def layer_map(path: Path, expected_layers: int, expected_pp: int) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    assert len(rows) == expected_layers, (path, len(rows), expected_layers)
    assert {row["pp_stage"] for row in rows} == set(range(expected_pp))
    assert [row["layer_id"] for row in rows] == list(range(expected_layers))
    return rows


def schedule(pp: int, stage: int, microbatches: int) -> list[tuple[str, int, str]]:
    warmup = min(pp - stage - 1, microbatches)
    remaining = microbatches - warmup
    result: list[tuple[str, int, str]] = []
    for microbatch in range(warmup):
        result.append(("FWD", microbatch, "warmup"))
    for index in range(remaining):
        fwd_mb = warmup + index
        result.append(("FWD", fwd_mb, "steady_or_drain"))
        result.append(("BWD", index, "steady_or_drain"))
    for microbatch in range(remaining, microbatches):
        result.append(("BWD", microbatch, "cooldown"))
    return result


class GraphBuilder:
    def __init__(self, case_id: str, dtype: str) -> None:
        self.case_id = case_id
        self.dtype = dtype
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.ids: set[str] = set()

    def node(self, node_id: str, **values: Any) -> str:
        if node_id in self.ids:
            raise ValueError(f"duplicate node: {node_id}")
        self.ids.add(node_id)
        row = {column: "" for column in NODE_COLUMNS}
        row.update(
            {
                "case_id": self.case_id,
                "node_id": node_id,
                "cost_status": "UNASSIGNED_STRUCTURE_ONLY",
                "cost_source": "none",
                "dtype": self.dtype,
            }
        )
        row.update(values)
        self.nodes.append(row)
        return node_id

    def edge(
        self,
        src: str,
        dst: str,
        edge_type: str = "data_dependency",
        tensor_key: str = "activation_or_gradient",
        dependency_source: str = "code_semantics",
    ) -> None:
        self.edges.append(
            {
                "case_id": self.case_id,
                "src": src,
                "dst": dst,
                "edge_type": edge_type,
                "tensor_key": tensor_key,
                "dependency_source": dependency_source,
            }
        )


def autograd_phase_for(template_name: str, index: int, specs: list[OpSpec]) -> str:
    if not template_name.endswith("_backward_recompute"):
        return "forward"
    fwd_size = len(attention_forward()) + (
        len(dense_mlp_forward()) if template_name.startswith("dense") else len(moe_forward())
    )
    return "recompute" if index < fwd_size else "backward"


def add_operator(
    graph: GraphBuilder,
    prefix: str,
    previous: str | None,
    spec: OpSpec,
    *,
    rank: int,
    stage: int,
    lane: int,
    phase: str,
    microbatch: int,
    region: str,
    operation_sequence: int,
    layer: dict[str, int],
    layer_type: str,
    autograd_phase: str,
    template_index: int,
) -> str:
    node_id = f"{prefix}:n{template_index:03d}:{spec.name}"
    graph.node(
        node_id,
        kind="operator",
        rank=rank,
        pp_stage=stage,
        pp_lane=lane,
        phase=phase,
        microbatch=microbatch,
        schedule_region=region,
        operation_sequence=operation_sequence,
        layer_id=layer["layer_id"],
        stage_local_layer=layer["stage_local_layer"],
        layer_type=layer_type,
        autograd_phase=autograd_phase,
        op_name=spec.name,
        op_family=spec.family,
        stream=spec.stream,
        resource=spec.resource,
        parallelism=spec.parallelism,
        shape_key=spec.shape,
        payload_key=spec.payload,
        flops_formula=spec.flops,
        bytes_formula=spec.bytes_,
        code_anchor=spec.anchor,
        template_index=template_index,
    )
    if previous:
        graph.edge(previous, node_id)
    return node_id


def add_phase_graph(
    graph: GraphBuilder,
    templates: dict[str, list[OpSpec]],
    stage_layers: list[dict[str, int]],
    *,
    rank: int,
    stage: int,
    lane: int,
    phase: str,
    microbatch: int,
    region: str,
    operation_sequence: int,
    previous: str | None,
    first_stage: bool,
    last_stage: bool,
) -> tuple[str, str]:
    phase_prefix = f"r{rank}:s{stage}:l{lane}:{phase.lower()}{microbatch}:q{operation_sequence}"
    start = graph.node(
        f"{phase_prefix}:start",
        kind="phase_boundary", rank=rank, pp_stage=stage, pp_lane=lane, phase=phase,
        microbatch=microbatch, schedule_region=region, operation_sequence=operation_sequence,
        layer_id=-1, stage_local_layer=-1, layer_type="boundary", autograd_phase=phase.lower(),
        op_name=f"{phase.lower()}_start", op_family="phase_boundary", stream="compute",
        resource="scheduler", parallelism="pp", shape_key="", payload_key="",
        flops_formula="0", bytes_formula="0", code_anchor="pipeline_schedules", template_index=-1,
    )
    if previous:
        graph.edge(previous, start, "rank_program_order", "rank_stream", "1f1b_schedule")
    current = start
    if phase == "FWD" and first_stage:
        preprocess = OpSpec("embedding_and_input_preprocess", "embedding")
        current = add_operator(
            graph, phase_prefix + ":pre", current, preprocess, rank=rank, stage=stage, lane=lane,
            phase=phase, microbatch=microbatch, region=region, operation_sequence=operation_sequence,
            layer={"layer_id": -1, "stage_local_layer": -1}, layer_type="preprocess",
            autograd_phase="forward", template_index=0,
        )
    ordered_layers = stage_layers if phase == "FWD" else list(reversed(stage_layers))
    for layer in ordered_layers:
        layer_type = "dense" if layer["layer_id"] == 0 else "moe"
        template_name = f"{layer_type}_{'forward' if phase == 'FWD' else 'backward_recompute'}"
        specs = templates[template_name]
        for index, spec in enumerate(specs):
            prefix = f"{phase_prefix}:layer{layer['layer_id']}:{template_name}"
            current = add_operator(
                graph, prefix, current, spec, rank=rank, stage=stage, lane=lane, phase=phase,
                microbatch=microbatch, region=region, operation_sequence=operation_sequence,
                layer=layer, layer_type=layer_type,
                autograd_phase=autograd_phase_for(template_name, index, specs), template_index=index,
            )
    if phase == "FWD" and last_stage:
        loss = OpSpec("output_projection_and_loss", "loss")
        current = add_operator(
            graph, phase_prefix + ":post", current, loss, rank=rank, stage=stage, lane=lane,
            phase=phase, microbatch=microbatch, region=region, operation_sequence=operation_sequence,
            layer={"layer_id": -1, "stage_local_layer": -1}, layer_type="postprocess",
            autograd_phase="forward", template_index=0,
        )
    if phase == "BWD" and last_stage:
        loss_bwd = OpSpec("loss_backward_seed", "loss_backward")
        seeded = add_operator(
            graph, phase_prefix + ":seed", start, loss_bwd, rank=rank, stage=stage, lane=lane,
            phase=phase, microbatch=microbatch, region=region, operation_sequence=operation_sequence,
            layer={"layer_id": -1, "stage_local_layer": -1}, layer_type="postprocess",
            autograd_phase="backward", template_index=0,
        )
        # The actual layer chain was already attached to start; the seed must
        # also gate its first layer node.
        first_layer_node = next(
            edge["dst"] for edge in reversed(graph.edges)
            if edge["src"] == start and edge["dst"].startswith(phase_prefix + ":layer")
        )
        graph.edge(seeded, first_layer_node, "gradient_seed", "loss_gradient", "autograd")
    end = graph.node(
        f"{phase_prefix}:end",
        kind="phase_boundary", rank=rank, pp_stage=stage, pp_lane=lane, phase=phase,
        microbatch=microbatch, schedule_region=region, operation_sequence=operation_sequence,
        layer_id=-1, stage_local_layer=-1, layer_type="boundary", autograd_phase=phase.lower(),
        op_name=f"{phase.lower()}_end", op_family="phase_boundary", stream="compute",
        resource="scheduler", parallelism="pp", shape_key="", payload_key="",
        flops_formula="0", bytes_formula="0", code_anchor="pipeline_schedules", template_index=-1,
    )
    graph.edge(current, end)
    return start, end


def group_definitions(pp: int, lanes: int) -> dict[str, list[dict[str, Any]]]:
    dp = [
        {"group_id": f"dp_with_cp_stage{stage}", "pp_stage": stage,
         "ranks": list(range(stage * lanes, (stage + 1) * lanes))}
        for stage in range(pp)
    ]
    expert_dp = [
        {"group_id": f"expert_dp_stage{stage}_lane{lane}", "pp_stage": stage,
         "ranks": [stage * lanes + lane, stage * lanes + lane + 8]}
        for stage in range(pp) for lane in range(8)
    ]
    return {"dp_with_cp": dp, "expert_dp": expert_dp}


def add_tail(
    graph: GraphBuilder,
    backward_ends: dict[tuple[int, int], str],
    *,
    pp: int,
    lanes: int,
    microbatches: int,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    groups = group_definitions(pp, lanes)
    rs_sync: dict[str, str] = {}
    ag1_sync: list[str] = []
    rank_groups: dict[int, list[str]] = defaultdict(list)
    for group_type, definitions in groups.items():
        parallelism = "dp" if group_type == "dp_with_cp" else "expert_dp"
        for definition in definitions:
            group_id = definition["group_id"]
            for rank in definition["ranks"]:
                rank_groups[rank].append(group_id)
            arrival = graph.node(
                f"tail:{group_id}:rs_arrival_join", kind="collective_boundary", rank=-1,
                pp_stage=definition["pp_stage"], pp_lane=-1, phase="OPT", microbatch=-1,
                schedule_region="optimizer_tail", operation_sequence=-1, layer_id=-1,
                stage_local_layer=-1, layer_type="tail", autograd_phase="optimizer",
                op_name="gradient_reduce_scatter_arrival_join", op_family="software_sync",
                stream="runtime", resource="host_runtime", parallelism=parallelism,
                shape_key="rank_release_vector", payload_key="gradient_shard",
                flops_formula="0", bytes_formula="0", code_anchor="distributed_optimizer",
                template_index=0,
            )
            for rank in definition["ranks"]:
                graph.edge(
                    backward_ends[(rank, microbatches - 1)], arrival, "collective_rank_arrival",
                    "gradient_ready", "distributed_optimizer",
                )
            service = graph.node(
                f"tail:{group_id}:rs_service", kind="collective_service", rank=-1,
                pp_stage=definition["pp_stage"], pp_lane=-1, phase="OPT", microbatch=-1,
                schedule_region="optimizer_tail", operation_sequence=-1, layer_id=-1,
                stage_local_layer=-1, layer_type="tail", autograd_phase="optimizer",
                op_name="gradient_reduce_scatter_service", op_family="collective_service",
                stream=f"{parallelism}_network", resource="network", parallelism=parallelism,
                shape_key="collective_group", payload_key="gradient_shard",
                flops_formula="0", bytes_formula="payload_bytes", code_anchor="distributed_optimizer",
                template_index=1,
            )
            sync = graph.node(
                f"tail:{group_id}:rs_completion_sync", kind="collective_boundary", rank=-1,
                pp_stage=definition["pp_stage"], pp_lane=-1, phase="OPT", microbatch=-1,
                schedule_region="optimizer_tail", operation_sequence=-1, layer_id=-1,
                stage_local_layer=-1, layer_type="tail", autograd_phase="optimizer",
                op_name="gradient_reduce_scatter_completion_sync", op_family="software_sync",
                stream="runtime", resource="host_runtime", parallelism=parallelism,
                shape_key="rank_completion_vector", payload_key="gradient_shard",
                flops_formula="0", bytes_formula="0", code_anchor="distributed_optimizer",
                template_index=2,
            )
            graph.edge(arrival, service, "collective_release", "release_offsets", "rank_release_protocol")
            graph.edge(service, sync, "collective_completion", "rank_done_offsets", "fct_backend")
            rs_sync[group_id] = sync
    optimizer_done: dict[int, str] = {}
    for rank in range(pp * lanes):
        stage, lane = divmod(rank, lanes)
        start = graph.node(
            f"tail:r{rank}:optimizer_start", kind="optimizer", rank=rank, pp_stage=stage,
            pp_lane=lane, phase="OPT", microbatch=-1, schedule_region="optimizer_tail",
            operation_sequence=-1, layer_id=-1, stage_local_layer=-1, layer_type="tail",
            autograd_phase="optimizer", op_name="optimizer_start", op_family="software_sync",
            stream="compute", resource="gpu_compute", parallelism="local", shape_key="parameters",
            payload_key="", flops_formula="0", bytes_formula="0", code_anchor="distributed_optimizer",
            template_index=0,
        )
        for group_id in rank_groups[rank]:
            graph.edge(rs_sync[group_id], start, "optimizer_gradient_dependency", "gradient_shard", "distributed_optimizer")
        update = graph.node(
            f"tail:r{rank}:optimizer_update", kind="optimizer", rank=rank, pp_stage=stage,
            pp_lane=lane, phase="OPT", microbatch=-1, schedule_region="optimizer_tail",
            operation_sequence=-1, layer_id=-1, stage_local_layer=-1, layer_type="tail",
            autograd_phase="optimizer", op_name="optimizer_update", op_family="optimizer_compute",
            stream="compute", resource="gpu_compute", parallelism="local", shape_key="parameter_shard",
            payload_key="", flops_formula="optimizer_symbolic_unbound", bytes_formula="parameter_bytes",
            code_anchor="distributed_optimizer", template_index=1,
        )
        done = graph.node(
            f"tail:r{rank}:optimizer_done", kind="optimizer", rank=rank, pp_stage=stage,
            pp_lane=lane, phase="OPT", microbatch=-1, schedule_region="optimizer_tail",
            operation_sequence=-1, layer_id=-1, stage_local_layer=-1, layer_type="tail",
            autograd_phase="optimizer", op_name="optimizer_done", op_family="phase_boundary",
            stream="compute", resource="scheduler", parallelism="local", shape_key="parameters",
            payload_key="", flops_formula="0", bytes_formula="0", code_anchor="distributed_optimizer",
            template_index=2,
        )
        graph.edge(start, update, "optimizer_program_order", "parameter_shard", "distributed_optimizer")
        graph.edge(update, done, "optimizer_program_order", "updated_parameter_shard", "distributed_optimizer")
        optimizer_done[rank] = done
    for group_type, definitions in groups.items():
        parallelism = "dp" if group_type == "dp_with_cp" else "expert_dp"
        for definition in definitions:
            group_id = definition["group_id"]
            previous_sync: str | None = None
            for round_index in (0, 1):
                arrival = graph.node(
                    f"tail:{group_id}:ag{round_index}_arrival_join", kind="collective_boundary", rank=-1,
                    pp_stage=definition["pp_stage"], pp_lane=-1, phase="OPT", microbatch=-1,
                    schedule_region="optimizer_tail", operation_sequence=-1, layer_id=-1,
                    stage_local_layer=-1, layer_type="tail", autograd_phase="optimizer",
                    op_name=f"parameter_all_gather_{round_index}_arrival_join", op_family="software_sync",
                    stream="runtime", resource="host_runtime", parallelism=parallelism,
                    shape_key="rank_release_vector", payload_key="parameter_shard",
                    flops_formula="0", bytes_formula="0", code_anchor="distributed_optimizer",
                    template_index=round_index * 3,
                )
                if round_index == 0:
                    for rank in definition["ranks"]:
                        graph.edge(optimizer_done[rank], arrival, "collective_rank_arrival", "updated_parameter_shard", "distributed_optimizer")
                else:
                    assert previous_sync
                    graph.edge(previous_sync, arrival, "allgather_round_order", "parameter_shard", "distributed_optimizer")
                service = graph.node(
                    f"tail:{group_id}:ag{round_index}_service", kind="collective_service", rank=-1,
                    pp_stage=definition["pp_stage"], pp_lane=-1, phase="OPT", microbatch=-1,
                    schedule_region="optimizer_tail", operation_sequence=-1, layer_id=-1,
                    stage_local_layer=-1, layer_type="tail", autograd_phase="optimizer",
                    op_name=f"parameter_all_gather_{round_index}_service", op_family="collective_service",
                    stream=f"{parallelism}_network", resource="network", parallelism=parallelism,
                    shape_key="collective_group", payload_key="parameter_shard",
                    flops_formula="0", bytes_formula="payload_bytes", code_anchor="distributed_optimizer",
                    template_index=round_index * 3 + 1,
                )
                sync = graph.node(
                    f"tail:{group_id}:ag{round_index}_completion_sync", kind="collective_boundary", rank=-1,
                    pp_stage=definition["pp_stage"], pp_lane=-1, phase="OPT", microbatch=-1,
                    schedule_region="optimizer_tail", operation_sequence=-1, layer_id=-1,
                    stage_local_layer=-1, layer_type="tail", autograd_phase="optimizer",
                    op_name=f"parameter_all_gather_{round_index}_completion_sync", op_family="software_sync",
                    stream="runtime", resource="host_runtime", parallelism=parallelism,
                    shape_key="rank_completion_vector", payload_key="parameter_shard",
                    flops_formula="0", bytes_formula="0", code_anchor="distributed_optimizer",
                    template_index=round_index * 3 + 2,
                )
                graph.edge(arrival, service, "collective_release", "release_offsets", "rank_release_protocol")
                graph.edge(service, sync, "collective_completion", "rank_done_offsets", "fct_backend")
                previous_sync = sync
            assert previous_sync
            ag1_sync.append(previous_sync)
    iteration_join = graph.node(
        "iteration:completion_join", kind="iteration_boundary", rank=-1, pp_stage=-1, pp_lane=-1,
        phase="ITER", microbatch=-1, schedule_region="iteration_tail", operation_sequence=-1,
        layer_id=-1, stage_local_layer=-1, layer_type="boundary", autograd_phase="iteration",
        op_name="iteration_completion_join", op_family="software_sync", stream="runtime",
        resource="host_runtime", parallelism="global", shape_key="all_parameter_groups",
        payload_key="", flops_formula="0", bytes_formula="0", code_anchor="training_loop",
        template_index=0,
    )
    for sync in ag1_sync:
        graph.edge(sync, iteration_join, "iteration_completion", "updated_parameters", "training_loop")
    return groups, iteration_join


def validate_graph(graph: GraphBuilder) -> dict[str, Any]:
    missing = [edge for edge in graph.edges if edge["src"] not in graph.ids or edge["dst"] not in graph.ids]
    if missing:
        raise ValueError(f"edges reference missing nodes: {missing[:3]}")
    indegree = {node_id: 0 for node_id in graph.ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        indegree[edge["dst"]] += 1
        outgoing[edge["src"]].append(edge["dst"])
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for target in outgoing.get(node_id, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(graph.ids):
        cyclic = [node_id for node_id, degree in indegree.items() if degree > 0][:20]
        raise ValueError(f"graph is cyclic; sample={cyclic}")
    return {
        "unique_node_ids": True,
        "edge_references_valid": True,
        "acyclic_verified_by_kahn": True,
        "topologically_visited_nodes": visited,
    }


def instantiate_case(
    section: dict[str, Any], architecture: dict[str, Any], templates: dict[str, list[OpSpec]]
) -> tuple[GraphBuilder, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    case_id = str(section["case_id"])
    pp = int(section["pp"])
    lanes = int(section["pp_lanes"])
    microbatches = int(section["microbatches"])
    world_size = int(section["world_size"])
    if pp * lanes != world_size:
        raise ValueError(f"{case_id}: pp*lanes != world_size")
    rows = layer_map(REPO / section["layer_stage_map"], int(section["layers"]), pp)
    layers_by_stage: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        layers_by_stage[row["pp_stage"]].append(row)
    graph = GraphBuilder(case_id, str(architecture["dtype"]))
    starts: dict[tuple[int, str, int], str] = {}
    ends: dict[tuple[int, str, int], str] = {}
    backward_ends: dict[tuple[int, int], str] = {}
    for rank in range(world_size):
        stage, lane = divmod(rank, lanes)
        previous: str | None = None
        for sequence_index, (phase, microbatch, region) in enumerate(schedule(pp, stage, microbatches)):
            start, end = add_phase_graph(
                graph, templates, layers_by_stage[stage], rank=rank, stage=stage, lane=lane,
                phase=phase, microbatch=microbatch, region=region,
                operation_sequence=sequence_index, previous=previous,
                first_stage=stage == 0, last_stage=stage == pp - 1,
            )
            starts[(rank, phase, microbatch)] = start
            ends[(rank, phase, microbatch)] = end
            if phase == "BWD":
                backward_ends[(rank, microbatch)] = end
            previous = end
    for lane in range(lanes):
        for microbatch in range(microbatches):
            for stage in range(pp - 1):
                src_rank = stage * lanes + lane
                dst_rank = (stage + 1) * lanes + lane
                pp_sendrecv = graph.node(
                    f"pp:lane{lane}:fwd{microbatch}:s{stage}_to_s{stage + 1}",
                    kind="pp_p2p", rank=-1, pp_stage=stage, pp_lane=lane, phase="FWD",
                    microbatch=microbatch, schedule_region="pipeline_transfer", operation_sequence=-1,
                    layer_id=-1, stage_local_layer=-1, layer_type="boundary", autograd_phase="forward",
                    op_name="pp_sendrecv_trace_wall", op_family="pp_p2p_trace_wall", stream="pp_network",
                    resource="network_and_runtime", parallelism="pp", shape_key="hidden_state",
                    payload_key="pp_activation", flops_formula="0", bytes_formula="captured_payload_or_unknown",
                    cost_status="UNASSIGNED_STRUCTURE_ONLY", cost_source="future_source_trace_end_to_end",
                    code_anchor="pipeline_schedules", template_index=stage,
                )
                graph.edge(ends[(src_rank, "FWD", microbatch)], pp_sendrecv, "pp_activation_send", "activation", "pipeline_schedule")
                graph.edge(pp_sendrecv, starts[(dst_rank, "FWD", microbatch)], "pp_activation_recv", "activation", "pipeline_schedule")
            for stage in range(pp - 1, 0, -1):
                src_rank = stage * lanes + lane
                dst_rank = (stage - 1) * lanes + lane
                pp_sendrecv = graph.node(
                    f"pp:lane{lane}:bwd{microbatch}:s{stage}_to_s{stage - 1}",
                    kind="pp_p2p", rank=-1, pp_stage=stage, pp_lane=lane, phase="BWD",
                    microbatch=microbatch, schedule_region="pipeline_transfer", operation_sequence=-1,
                    layer_id=-1, stage_local_layer=-1, layer_type="boundary", autograd_phase="backward",
                    op_name="pp_sendrecv_trace_wall", op_family="pp_p2p_trace_wall", stream="pp_network",
                    resource="network_and_runtime", parallelism="pp", shape_key="hidden_gradient",
                    payload_key="pp_gradient", flops_formula="0", bytes_formula="captured_payload_or_unknown",
                    cost_status="UNASSIGNED_STRUCTURE_ONLY", cost_source="future_source_trace_end_to_end",
                    code_anchor="pipeline_schedules", template_index=pp - stage,
                )
                graph.edge(ends[(src_rank, "BWD", microbatch)], pp_sendrecv, "pp_gradient_send", "gradient", "pipeline_schedule")
                graph.edge(pp_sendrecv, starts[(dst_rank, "BWD", microbatch)], "pp_gradient_recv", "gradient", "pipeline_schedule")
            for stage in range(pp):
                rank = stage * lanes + lane
                graph.edge(
                    ends[(rank, "FWD", microbatch)], starts[(rank, "BWD", microbatch)],
                    "autograd_activation_dependency", "saved_or_recomputed_activation", "autograd",
                )
    groups, iteration_join = add_tail(
        graph, backward_ends, pp=pp, lanes=lanes, microbatches=microbatches
    )
    validation = validate_graph(graph)
    kinds = Counter(str(row["kind"]) for row in graph.nodes)
    families = Counter(str(row["op_family"]) for row in graph.nodes)
    phases = Counter(str(row["phase"]) for row in graph.nodes)
    summary = {
        "case_id": case_id,
        "world_size": world_size,
        "pp": pp,
        "pp_lanes": lanes,
        "microbatches": microbatches,
        "layers": int(section["layers"]),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "node_kinds": dict(sorted(kinds.items())),
        "op_families": dict(sorted(families.items())),
        "phases": dict(sorted(phases.items())),
        "mtp_instantiated_nodes": 0,
        "iteration_join": iteration_join,
        "validation": validation,
    }
    return graph, summary, groups


def template_payload(templates: dict[str, list[OpSpec]], semantics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, specs in templates.items():
        result[name] = {
            "active": not name.startswith("mtp_"),
            "nodes": [spec.__dict__ for spec in specs],
            "cp_service_calls": sum(spec.family == "collective_service" and spec.parallelism == "cp" for spec in specs),
            "ep_service_calls": sum(spec.family == "collective_service" and spec.parallelism == "ep" for spec in specs),
        }
    return {
        "status": "STRUCTURE_ONLY_COSTS_UNASSIGNED",
        "templates": result,
        "expected_call_counts": {
            "forward_cp_per_layer": semantics["cp_forward_calls_per_layer"],
            "backward_including_recompute_cp_per_layer": semantics["cp_backward_calls_per_layer_including_recompute"],
            "forward_ep_per_moe_layer": semantics["ep_forward_calls_per_moe_layer"],
            "backward_including_recompute_ep_per_moe_layer": semantics["ep_backward_calls_per_moe_layer_including_recompute"],
        },
        "mtp": {"enabled": semantics["mtp_enabled"], "reason": semantics["mtp_reason"]},
        "combined_ep_overlap": {
            "enabled": semantics["combined_ep_overlap_enabled"],
            "reason": semantics["combined_ep_overlap_reason"],
        },
    }


def semantic_evidence(config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        name: (REPO / value if not Path(value).is_absolute() else Path(value))
        for name, value in config["semantic_evidence"].items()
    }
    signatures: dict[str, dict[int, set[int]]] = {
        "forward": defaultdict(set), "backward_including_recompute": defaultdict(set)
    }
    with paths["cp_slot_parameters"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phase = "forward" if row["phase"] == "forward" else "backward_including_recompute"
            signatures[phase][int(row["cp_slot_in_layer"])].add(int(float(row["logical_input_bytes"])))
    slot_signatures = {
        phase: [
            {"slot": slot, "logical_input_bytes": sorted(values)}
            for slot, values in sorted(slots.items())
        ]
        for phase, slots in signatures.items()
    }

    schedule_samples: dict[str, list[str]] = defaultdict(list)
    pp_rows: dict[int, list[tuple[int, str]]] = defaultdict(list)
    with paths["pp_trace_events"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rank = int(row["rank"])
            if int(row["iteration"]) == 60 and rank in (0, 16, 224, 240):
                pp_rows[rank].append(
                    (int(row["observed_start_ns"]), f"{row['phase'][0].upper()}{row['microbatch']}")
                )
    for rank, rows in pp_rows.items():
        schedule_samples[str(rank)] = [label for _, label in sorted(rows)]

    ep_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with paths["deepep_group_operations"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["iteration"]) == 60
                and int(row["ep_group"]) == 0
                and int(row["microbatch"]) == 0
                and int(row["layer_id"]) == 1
            ):
                ep_rows[row["phase"]].append(
                    (int(row["first_arrival_ns"]), row["semantic_region"])
                )
    ep_order = {
        phase: [name for _, name in sorted(rows)] for phase, rows in ep_rows.items()
    }

    cp_final: dict[str, int] | None = None
    with paths["cp_rank_events"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["iteration"]) == 60 and int(row["rank"]) == 16
                and row["phase"] == "backward" and int(row["microbatch"]) == 0
                and int(row["layer_id"]) == 2 and int(row["cp_slot_in_layer"]) == 8
            ):
                cp_final = {"start_ns": int(row["start_ns"]), "end_ns": int(row["end_ns"])}
                break
    pp_api: dict[str, int] | None = None
    with paths["pp_api_events"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["iteration"]) == 60 and int(row["rank"]) == 16
                and row["api_name"] == "send_backward" and int(row["occurrence"]) == 0
            ):
                pp_api = {
                    "start_ns": int(row["observed_start_ns"]),
                    "end_ns": int(row["observed_end_ns"]),
                }
                break
    if cp_final is None or pp_api is None:
        raise ValueError("semantic evidence sample for PP/CP overlap is missing")

    raw = json.loads(paths["raw_trace_example"].read_text(encoding="utf-8"))["traceEvents"]

    def raw_attention_order(wrapper_name: str, flash_token: str) -> dict[str, Any]:
        wrapper = next(
            event for event in raw
            if event.get("name") == wrapper_name and event.get("cat") == "cpu_op"
        )
        start = float(wrapper["ts"])
        end = start + float(wrapper["dur"])
        collectives = sorted(
            (
                event for event in raw
                if event.get("name") == "c10d::alltoall_base_"
                and start <= float(event.get("ts", -1)) <= end
            ),
            key=lambda event: float(event["ts"]),
        )
        flash = next(
            event for event in raw
            if flash_token in str(event.get("name", ""))
            and start <= float(event.get("ts", -1)) <= end
        )
        flash_ts = float(flash["ts"])
        return {
            "wrapper": wrapper_name,
            "collective_count": len(collectives),
            "before_flash": sum(float(event["ts"]) < flash_ts for event in collectives),
            "after_flash": sum(float(event["ts"]) > flash_ts for event in collectives),
            "logical_input_shapes": [
                event.get("args", {}).get("Input Dims", [[]])[0] for event in collectives
            ],
        }

    forward_raw = raw_attention_order(
        "AttnFuncWithCPAndQKVOA2A", "_scaled_dot_product_attention_flash_musa"
    )
    backward_raw = raw_attention_order(
        "AttnFuncWithCPAndQKVOA2ABackward", "_scaled_dot_product_attention_flash_musa_backward"
    )
    return {
        "status": "PASS_SOURCE_SEMANTIC_AUDIT_WITH_DOCUMENTED_ASSUMPTIONS",
        "evidence_paths": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "confirmed_facts": {
            "mtp_enabled": False,
            "mtp_basis": "launch enable flags are commented; MTP=1 alone does not activate MTP",
            "combined_ep_overlap_enabled": False,
            "combined_ep_overlap_basis": "--overlap-moe-expert-parallel-comm is commented and the code branches on this flag",
            "shared_expert_overlap_enabled": False,
            "shared_expert_overlap_basis": "SE_OVERLAP defaults false; the flag is only appended when true",
            "pipeline_schedule": "conventional_non_interleaved_1f1b",
            "full_recompute": True,
        },
        "trace_observations": {
            "attention_forward": forward_raw,
            "attention_backward_true_only": backward_raw,
            "cp_slot_payload_signatures": slot_signatures,
            "pp_operation_order_iteration60": dict(schedule_samples),
            "deepep_group0_layer1_microbatch0_order": ep_order,
            "pp_api_wrapper_overlap_sample": {
                "iteration": 60, "rank": 16, "phase": "backward", "microbatch": 0,
                "final_cp_slot": cp_final, "pp_send_backward_api": pp_api,
                "api_starts_before_final_cp_finishes": pp_api["start_ns"] < cp_final["end_ns"],
                "final_cp_finishes_before_api_returns": cp_final["end_ns"] < pp_api["end_ns"],
            },
        },
        "modeling_assumptions": [
            {
                "topic": "grouped_vs_shared_expert_backward_internal_order",
                "current_policy": "grouped expert backward then shared expert backward",
                "basis": "MoELayer.backward_dw code order; raw trace does not label the two kernel families separately",
                "impact": "structure-only today; must be rechecked before operator-level cost binding",
            },
            {
                "topic": "pp_trace_wall_cost_boundary",
                "current_policy": "logical PP send depends on final layer work; cost remains unassigned",
                "basis": "PP API wrapper can start before final CP finishes and return afterward",
                "impact": "do not append the full PP API wall after CP or CP wait will be double counted",
            },
            {
                "topic": "operator_kernel_fusion",
                "current_policy": "retain semantic operator families even when runtime fuses kernels",
                "basis": "code-level graph is finer than available labelled trace kernels",
                "impact": "fused groups need a cost-sharing rule before timing prediction",
            },
        ],
    }


def build_semantics_report(output: Path, evidence: dict[str, Any]) -> None:
    observations = evidence["trace_observations"]
    pp = observations["pp_operation_order_iteration60"]
    ep = observations["deepep_group0_layer1_microbatch0_order"]
    overlap = observations["pp_api_wrapper_overlap_sample"]
    report = f"""# DAG v6.0 执行语义审计

状态：**{evidence['status']}**。

本文只回答“本次 236B 训练实际启用了什么，以及 DAG 节点按什么顺序连接”。它不绑定耗时，也不产生 MFU 精度结论。证据优先级是：raw profiler trace / DeepEP 日志衍生表 > 实际启动参数与执行分支 > 代码模板 > 解释性 HTML。

## 已确认事实

| 项目 | 结论 | 证据 |
|---|---|---|
| Pipeline | 普通 non-interleaved 1F1B | combined EP overlap 开关未启用；iteration 60 的 rank 顺序与 1F1B 一致 |
| MTP | **未启用** | `MTP=1` 只是变量；`--use-multi-token-prediction` 和 `--mtp-num-layers` 均被注释 |
| combined EP overlap | **未启用** | `--overlap-moe-expert-parallel-comm` 被注释，combined schedule 分支不会进入 |
| shared-expert overlap | **未启用** | `SE_OVERLAP` 默认 false，只有 true 才追加启动参数 |
| activation recompute | full/block，最多 4 层 | 启动参数直接确认，trace 中出现完整 attention/MoE recompute 通信 |

`model_chunk_schedule_plan.py` 因此只用来说明代码具备的可选能力，**不是本次运行的真实 scheduler**。

## Trace 确认的真实层内顺序

### Attention forward

raw trace 的 `AttnFuncWithCPAndQKVOA2A` 给出：

```text
MLA Q/K/V projection
  -> CP A2A(Q, 384 MiB logical)
  -> CP A2A(K, 384 MiB logical)
  -> CP A2A(V, 256 MiB logical)
  -> FlashAttention
  -> CP A2A(Output, 256 MiB logical)
```

即三次 A2A 在 FlashAttention 前，一次在其后；不是旧模板中的 2+2。raw trace 自动审计结果是 before={observations['attention_forward']['before_flash']}、after={observations['attention_forward']['after_flash']}。

### MoE forward

```text
Attention/CP -> RMSNorm/router -> DeepEP dispatch
  -> shared expert（overlap 关闭，代码先执行）
  -> dispatch postprocess / routed experts / combine preprocess
  -> DeepEP combine -> residual
```

DeepEP 的 `bw` 日志负责确认 dispatch/combine 网络事务顺序；CPU `FusedDispatch/FusedCombine` wrapper 只作定位，不能把 wrapper 中的同步等待全部算成 EP service。

### MoE backward（含 full recompute）

```text
recompute attention: CP Q/K/V -> Flash -> CP Output
  -> recompute EP dispatch -> recompute EP combine
  -> EP combine backward -> local expert backward -> EP dispatch backward
  -> attention true backward:
       CP 256 MiB x2 -> FlashAttention backward -> CP 384/384/256 MiB
```

iteration 60、EP group 0、layer 1 的记录顺序：

- forward：{', '.join(ep.get('forward', []))}
- backward：{', '.join(ep.get('backward', []))}

### PP 1F1B 顺序

- rank 0：{' '.join(pp.get('0', []))}
- rank 16：{' '.join(pp.get('16', []))}
- rank 224：{' '.join(pp.get('224', []))}
- rank 240：{' '.join(pp.get('240', []))}

这同时确认末 stage 的 BWD 先发生，再沿 PP15→PP0 返回。

## 仍保留的建模说明

1. **grouped/shared expert backward 内部次序**：当前按 `MoELayer.backward_dw()` 的 grouped→shared 代码顺序表达；trace 没有给两类 kernel 独立标签，绑定细粒度时钟前仍需 kernel attribution。
2. **PP API wall 不能整体串接在 CP 后面**：样本 rank 16 的 `send_backward` API 在最终 CP 结束前已经进入，并在 CP 结束后返回（关系检查：{overlap['api_starts_before_final_cp_finishes']} / {overlap['final_cp_finishes_before_api_returns']}）。这说明 wrapper 内含等待；后续若使用完整 API wall，会重复计算 CP 等待。
3. **kernel fusion**：DAG 保留代码级算子名，但 runtime 可能把多个算子融合为一个 kernel。成本绑定必须按 fused group 分摊，不能给每个语义节点重复赋完整 kernel 时间。

## HTML 的使用边界

`HWN_ANA_20260730_Kimi236B_256卡集合通信_Profile时间线.html` 适合解释宏观依赖：PP recv→层内 CP/EP→PP send，以及 forward→recompute/backward→optimizer。它会合并部分 CP 逻辑操作，所以调用数量和精确先后以 `semantic_evidence.json`、raw trace 与 v5.3 事件表为准。
"""
    (output / "DAG_V60_SEMANTICS_AUDIT.md").write_text(report, encoding="utf-8")


def build_html(
    output: Path, summaries: list[dict[str, Any]], templates: dict[str, Any], evidence: dict[str, Any]
) -> None:
    source, target = summaries
    cards = "".join(
        f"<div class='card'><h3>{escape(item['case_id'])}</h3>"
        f"<b>{item['nodes']:,}</b><span> nodes</span><br>"
        f"<b>{item['edges']:,}</b><span> edges</span><br>"
        f"PP{item['pp']} × M{item['microbatches']} · {item['layers']} layers</div>"
        for item in summaries
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAG v6 operator IR</title><style>
:root{{--bg:#08111f;--panel:#101d31;--ink:#e8f0ff;--muted:#9eb0ca;--compute:#35c89f;--cp:#4da3ff;--ep:#c77dff;--sync:#ffba5c;--pp:#ff6b7d;}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07101d,#0d1830);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 22px 70px}} h1{{font-size:34px;margin:.2em 0}} h2{{margin-top:36px}} .sub,.note{{color:var(--muted)}}
.warning{{border:1px solid #ffba5c;background:#2a2011;padding:14px 18px;border-radius:12px}} .cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:22px 0}}
.card,.panel{{background:rgba(16,29,49,.92);border:1px solid #29405f;border-radius:14px;padding:18px}} .card b{{font-size:24px}} .flow{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.node{{padding:9px 12px;border-radius:9px;border:1px solid #44617e;background:#162842}} .compute{{border-color:var(--compute)}} .cp{{border-color:var(--cp)}} .ep{{border-color:var(--ep)}} .sync{{border-color:var(--sync)}} .pp{{border-color:var(--pp)}} .arrow{{color:#7188a5}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;border-bottom:1px solid #29405f;padding:9px}} code{{color:#b9d8ff}} @media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="sub">2026-W36 · structure milestone</div><h1>DAG v6：先把训练程序画成图，再绑定时钟</h1>
<div class="warning"><b>当前不是精度结果。</b> 本页只证明 PP16×M4 与 PP14×M3 都能由同一套代码语义生成无环算子图；所有节点时长仍为 UNASSIGNED，且未读取 224 卡 profiler/training timing。</div>
<div class="cards">{cards}</div>
<h2>1 · 单层前向语义</h2><div class="panel"><div class="flow">
<span class="node compute">RMSNorm / MLA projection</span><span class="arrow">→</span><span class="node cp">CP Q/K/V ×3</span><span class="arrow">→</span><span class="node compute">FlashAttention</span><span class="arrow">→</span><span class="node cp">CP Output ×1</span><span class="arrow">→</span><span class="node compute">Router</span><span class="arrow">→</span><span class="node ep">EP dispatch</span><span class="arrow">→</span><span class="node compute">Shared → Routed experts</span><span class="arrow">→</span><span class="node ep">EP combine</span>
</div><p class="note">通信不是一个不可拆分的大框：rank 到达、网络 service、完成同步分别成节点，后续才能判断 service 被计算隐藏了多少、同步又暴露了多少。</p></div>
<h2>2 · 反向与重计算</h2><div class="panel"><div class="flow"><span class="node compute">FWD 子图重计算</span><span class="arrow">→</span><span class="node ep">MoE backward EP ×2</span><span class="arrow">→</span><span class="node compute">MLP / router backward</span><span class="arrow">→</span><span class="node cp">Attention backward CP ×5</span><span class="arrow">→</span><span class="node compute">MLA backward</span></div>
<p class="note">因此每层 BWD（含重计算）是 CP 4+5=9；MoE 层 EP 2+2=4。它们是 trace 调用次数约束，不是目标 case 的耗时。</p></div>
<h2>3 · PP 与 iteration tail</h2><div class="panel"><div class="flow"><span class="node compute">stage FWD</span><span class="arrow">→</span><span class="node pp">PP activation trace-wall</span><span class="arrow">→</span><span class="node compute">next stage FWD</span><span class="arrow">…</span><span class="node compute">stage BWD</span><span class="arrow">→</span><span class="node pp">PP gradient trace-wall</span></div><br><div class="flow"><span class="node sync">DP / EDP RS arrival</span><span class="arrow">→</span><span class="node ep">RS service</span><span class="arrow">→</span><span class="node compute">optimizer</span><span class="arrow">→</span><span class="node ep">AG0 / AG1 service</span><span class="arrow">→</span><span class="node sync">iteration join</span></div></div>
<h2>4 · v6 与 v5.5 的关键差异</h2><table><tr><th>维度</th><th>v5.5</th><th>v6 当前里程碑</th></tr><tr><td>计算</td><td>trace gap 拆分为 compute family</td><td>由层代码语义生成 RMSNorm、MLA、FlashAttention、router、expert 等节点</td></tr><tr><td>通信</td><td>按观测事件夹在 gap 之间</td><td>固定插在代码调用点，并拆 arrival/service/completion</td></tr><tr><td>并行策略</td><td>主要转移 stage 模板</td><td>用 layer map + PP/M 重新实例化完整图</td></tr><tr><td>时钟</td><td>已有校准/预测</td><td>尚未绑定；不能宣称精度</td></tr></table>
<h2>5 · 代码与 trace 的确认边界</h2><div class="panel"><table><tr><th>项目</th><th>本次结论</th><th>状态</th></tr><tr><td>MTP</td><td>启用参数被注释</td><td>确认未启用</td></tr><tr><td>combined EP overlap</td><td>启用参数被注释，普通 1F1B</td><td>确认未启用</td></tr><tr><td>shared-expert overlap</td><td>默认 false</td><td>确认未启用</td></tr><tr><td>CP forward</td><td>Q/K/V 三次在 Flash 前，Output 一次在后</td><td>raw trace 确认</td></tr><tr><td>local expert backward</td><td>grouped→shared</td><td>代码引导假设，待 kernel 标签确认</td></tr><tr><td>PP API wall</td><td>wrapper 可与最终 CP 重叠</td><td>绑定成本前必须拆分</td></tr></table><p class="note">详细证据见 DAG_V60_SEMANTICS_AUDIT.md 与 model_inputs/semantic_evidence.json。</p></div>
<h2>6 · 审计结论</h2><div class="panel"><ul><li>source/target 均通过 Kahn 拓扑检查，节点和边引用完整。</li><li>MTP 模板保留但未实例化；combined EP overlap 与 shared-expert overlap 均按本次未启用处理。</li><li>解释性 trace HTML 用来说明宏观顺序；精确调用数与先后由 raw trace/事件表确认。</li><li>下一里程碑才会给 shape/flops、source calibration cost 与 FCT lookup 绑定成本。</li></ul></div>
<p class="note">模板统计：dense FWD CP={templates['templates']['dense_forward']['cp_service_calls']}；MoE FWD EP={templates['templates']['moe_forward']['ep_service_calls']}；MoE BWD(含重计算) CP={templates['templates']['moe_backward_recompute']['cp_service_calls']}、EP={templates['templates']['moe_backward_recompute']['ep_service_calls']}。</p>
</main></body></html>"""
    (output / "dag_v60_operator_ir.html").write_text(html, encoding="utf-8")


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()
    return {
        "branch": run("git", "branch", "--show-current"),
        "head": run("git", "rev-parse", "HEAD"),
        "status_porcelain": run("git", "status", "--porcelain=v1"),
        "diff_sha256": hashlib.sha256(run("git", "diff", "--binary").encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    output = REPO / config["outputs"]["output_dir"]
    model_inputs = output / "model_inputs"
    output.mkdir(parents=True, exist_ok=True)
    model_inputs.mkdir(parents=True, exist_ok=True)
    templates = template_catalog()
    template_doc = template_payload(templates, config["semantics"])
    json_dump(model_inputs / "operator_templates.json", template_doc)
    evidence = semantic_evidence(config)
    json_dump(model_inputs / "semantic_evidence.json", evidence)
    summaries: list[dict[str, Any]] = []
    all_groups: dict[str, Any] = {}
    for name in ("source", "target"):
        graph, summary, groups = instantiate_case(config[name], config["architecture"], templates)
        write_csv(model_inputs / f"{name}_operator_nodes.csv.gz", graph.nodes, NODE_COLUMNS)
        write_csv(model_inputs / f"{name}_operator_edges.csv.gz", graph.edges, EDGE_COLUMNS)
        summaries.append(summary)
        all_groups[name] = groups
        del graph
    json_dump(model_inputs / "collective_group_definitions.json", all_groups)
    comparison = [
        {
            "case_role": name,
            "case_id": summary["case_id"],
            "world_size": summary["world_size"],
            "pp": summary["pp"],
            "microbatches": summary["microbatches"],
            "layers": summary["layers"],
            "nodes": summary["nodes"],
            "edges": summary["edges"],
            "mtp_instantiated_nodes": summary["mtp_instantiated_nodes"],
            "acyclic": summary["validation"]["acyclic_verified_by_kahn"],
        }
        for name, summary in zip(("source", "target"), summaries)
    ]
    write_csv(output / "structure_comparison.csv", comparison, list(comparison[0]))
    contract = {
        "protocol": config["protocol"],
        "status": "PARTIAL_STRUCTURE_ONLY_COSTS_UNASSIGNED",
        "summaries": summaries,
        "invariants": {
            "same_template_catalog_for_source_and_target": True,
            "target_timing_read": False,
            "costs_assigned": False,
            "accuracy_claim_allowed": False,
            "pp_direction": "FWD 0->last; BWD last->0",
            "communication_interface": "arrival_join -> service -> completion_sync",
        },
        "future_cost_policies": {
            "pp": config["semantics"]["pp_cost_policy"],
            "cp": config["semantics"]["cp_service_policy"],
            "ep": config["semantics"]["ep_service_policy"],
            "dp_edp": config["semantics"]["dp_edp_service_policy"],
        },
        "semantic_audit": {
            "status": evidence["status"],
            "confirmed_facts": evidence["confirmed_facts"],
            "documented_assumption_count": len(evidence["modeling_assumptions"]),
        },
    }
    json_dump(output / "graph_contract.json", contract)
    inputs = [config_path]
    inputs += [REPO / config[name]["layer_stage_map"] for name in ("source", "target")]
    inputs += [REPO / config["source"]["training_script"], REPO / config["target"]["run_contract"]]
    inputs += [
        (REPO / path if not Path(path).is_absolute() else Path(path))
        for path in config["code_sources"].values()
    ]
    inputs += [
        (REPO / path if not Path(path).is_absolute() else Path(path))
        for path in config["semantic_evidence"].values()
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing declared inputs: " + ", ".join(missing))
    access = {
        "status": "PASS_STRUCTURE_INPUT_ISOLATION",
        "files_read": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs],
        "fields_read": {
            "layer_stage_maps": ["layer_id", "pp_stage", "stage_local_layer", "stage_layer_count", "virtual_chunk"],
            "config": "architecture and launch semantics only",
            "source_semantic_trace": "operator order, payload shape and PP schedule only; no target timing",
        },
        "target_profiler_or_training_timing_read": False,
        "target_trace_timestamp_or_duration_read": False,
        "target_accuracy_or_residual_read": False,
        "ns3_or_oisa_executed": False,
    }
    json_dump(output / "input_access_audit.json", access)
    report = f"""# DAG v6.0 code-native operator IR

状态：**PARTIAL — structure only, costs unassigned**。

本里程碑把 236B 训练代码的层级语义实例化为 source PP{summaries[0]['pp']}×M{summaries[0]['microbatches']}×{summaries[0]['layers']}-layer 和 target PP{summaries[1]['pp']}×M{summaries[1]['microbatches']}×{summaries[1]['layers']}-layer 两张 DAG。它不读取 224 卡的 profiler/training timing，不产生 step prediction，也不声明精度。

## 已建立的语义

- Attention：RMSNorm、MLA 投影、4 次 CP service、FlashAttention、输出投影。
- Dense MLP：FC1、SwiGLU、FC2 与 residual。
- MoE：router/top-k、permute、EP dispatch、grouped experts、shared expert、EP combine、unpermute。
- BWD：完整 forward recompute 子图 + 真正 backward；每层 CP 4+5=9，MoE 层 EP 2+2=4。
- PP：FWD 从 stage 0 到末 stage；BWD 从末 stage 返回 stage 0；PP 节点未来只绑定 trace wall time。
- Tail：DP/EDP RS arrival/service/sync、rank optimizer、AG0/AG1 arrival/service/sync、iteration join。

执行语义证据与仍保留的假设单列在 `DAG_V60_SEMANTICS_AUDIT.md`：MTP、combined EP overlap、shared-expert overlap 均确认未启用；attention forward 已由 raw trace 修正为 CP Q/K/V 三次在 FlashAttention 前、Output 一次在后。

## 结构结果

| role | case | nodes | edges | topology |
|---|---|---:|---:|---|
| source | {summaries[0]['case_id']} | {summaries[0]['nodes']:,} | {summaries[0]['edges']:,} | PP{summaries[0]['pp']} × M{summaries[0]['microbatches']} × L{summaries[0]['layers']} |
| target | {summaries[1]['case_id']} | {summaries[1]['nodes']:,} | {summaries[1]['edges']:,} | PP{summaries[1]['pp']} × M{summaries[1]['microbatches']} × L{summaries[1]['layers']} |

两张图均通过节点唯一性、边引用完整性和 Kahn 无环检查。

## 未做事项

- 未给任何 operator、PP、CP、EP、DP/EDP 节点赋时钟。
- 未读取 224 卡目标 trace 的 timestamp/duration 或 step ground truth。
- 未执行 OISA/ns-3。
- 未实例化 MTP 与 combined EP overlap，因为实测启动脚本中的启用选项被注释。
- 未进行精度对比；当前 HTML 是计算图说明页，不是预测结果页。

## 下一里程碑

为算子绑定 shape/flops 与 source-only cost parameters，并通过统一 FCT lookup 给 CP/EP/DP/EDP service 节点赋值；之后才可封存 224 卡预测并由 evaluator 解封验证。
"""
    (output / "DAG_V60_OPERATOR_IR_REPORT.md").write_text(report, encoding="utf-8")
    build_semantics_report(output, evidence)
    build_html(output, summaries, template_doc, evidence)
    builder = Path(__file__).resolve()
    test_file = REPO / "case_224gpu_pp14_cp2_a2a/tests/test_dag_v60_operator_ir.py"
    finalizer = REPO / "case_224gpu_pp14_cp2_a2a/scripts/finalize_dag_v60_operator_ir.py"
    commands = [
        f"cd {REPO}",
        f"{sys.executable} {builder} --config {config_path}",
        f"{sys.executable} -m pytest -q {test_file} --junitxml={output / 'logs/pytest.xml'}",
        f"{sys.executable} {finalizer} --run {output}",
    ]
    command_separator = " && " + "\\" + "\n  "
    (output / "reproduction_command.txt").write_text(command_separator.join(commands) + "\n", encoding="utf-8")
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_STRUCTURE_ONLY_COSTS_UNASSIGNED",
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "git": git_snapshot(),
        "limitations": [
            "dirty worktree is frozen by file hashes rather than a clean commit",
            "operator costs are not assigned",
            "no accuracy claim is permitted",
        ],
    }
    json_dump(output / "provenance.json", provenance)
    generated = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    json_dump(
        output / "artifact_manifest.json",
        {"status": provenance["status"], "artifacts": [{"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size} for path in generated]},
    )
    print(json.dumps({"output": str(output.resolve()), "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

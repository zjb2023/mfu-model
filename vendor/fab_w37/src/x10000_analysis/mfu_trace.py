"""Trace-side primitives for the stage-aware MFU model.

The profiler JSON is regular enough to scan with bytes regexes.  This avoids
materialising roughly 100,000 Chrome events per trace while still failing
closed if a trace layout changes.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


PHASES = ("forward", "backward", "optimizer")
COMPUTE_FAMILIES = (
    "gemm",
    "attention_forward",
    "attention_backward",
    "optimizer",
    "normalization",
    "routing",
    "memory",
    "elementwise",
    "other_compute",
)
COMMUNICATION_FAMILIES = ("mccl_communication", "deepep_communication")

_CPU_EVENT = re.compile(
    rb'"ph": "X", "cat": "(?:cpu_op|user_annotation|privateuse1_runtime)", '
    rb'"name": "([^"]*)"[^\n]*\n\s*"ts": ([0-9.]+), "dur": ([0-9.]+),'
    rb'\n\s*"args": \{\n\s*"External id": ([0-9]+)'
)
_KERNEL_EVENT = re.compile(
    rb'"ph": "X", "cat": "kernel", "name": "([^"]*)"[^\n]*'
    rb'"tid": ([0-9]+),\n\s*"ts": ([0-9.]+), "dur": ([0-9.]+),'
    rb'\n\s*"args": \{\n\s*([^\n]*)\n\s*\}'
)
_EXTERNAL_ID = re.compile(rb'"External id": ([0-9]+)')
_OCCUPANCY = re.compile(rb'"est\. achieved occupancy %": ([0-9.]+)')

_PHASE_ANNOTATIONS = {
    b"forward_step": "forward",
    b"backward_step": "backward",
    b"finalize_model_grads": "backward",
    b"step": "optimizer",
}


@dataclass(frozen=True)
class Architecture:
    hidden_size: int
    sequence_length: int
    global_batch_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_pos_emb_head_dim: int
    v_head_dim: int
    dense_ffn_hidden_size: int
    moe_ffn_hidden_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    moe_router_topk: int
    vocab_size: int
    first_dense_layers: int

    def validate(self) -> None:
        for name, value in vars(self).items():
            if value <= 0 and name != "first_dense_layers":
                raise ValueError(f"architecture {name} must be positive")
        if not 0 <= self.first_dense_layers:
            raise ValueError("first_dense_layers must be non-negative")


def interval_union_us(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the union length of half-open microsecond intervals."""
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def classify_kernel(name: bytes | str) -> str:
    raw = name.lower() if isinstance(name, bytes) else name.lower().encode()
    if b"mccl" in raw:
        return "mccl_communication"
    if b"deep_ep" in raw or b"deepep" in raw:
        return "deepep_communication"
    if b"flash_atten" in raw or b"flashatt" in raw:
        return "attention_backward" if b"bwd" in raw or b"backward" in raw else "attention_forward"
    if b"gemm" in raw or b"gemv" in raw or b"matmul" in raw:
        return "gemm"
    if b"adam" in raw or b"optimizer" in raw:
        return "optimizer"
    if b"norm" in raw:
        return "normalization"
    if any(token in raw for token in (b"topk", b"index_select", b"scatter", b"permute", b"sort")):
        return "routing"
    if any(token in raw for token in (b"memcpy", b"memset", b"kernelfill", b"copy")):
        return "memory"
    if any(token in raw for token in (b"triton_poi", b"unary", b"binary", b"elementwise", b"pointwise")):
        return "elementwise"
    return "other_compute"


def _phase_at(timestamp: float, intervals: list[tuple[float, float, str]]) -> str | None:
    matches = [item for item in intervals if item[0] <= timestamp <= item[1]]
    return min(matches, key=lambda item: item[1] - item[0])[2] if matches else None


def _decode_kernel_name(name: bytes) -> str:
    return name.decode("utf-8", errors="replace").replace(r"\u003c", "<").replace(r"\u003e", ">")


def parse_trace(
    path: Path,
    iteration: int,
    rank: int,
    pp_stage: int,
) -> dict[str, object]:
    """Extract phase/family active-time facts from one profiler trace."""
    data = path.read_bytes()
    raw_kernel_count = data.count(b'"cat": "kernel"')
    cpu_events: list[tuple[bytes, float, float, int]] = []
    phase_intervals: list[tuple[float, float, str]] = []
    phase_annotation_us = Counter()
    annotation_counts = Counter()
    profiler_step_us = 0.0

    for match in _CPU_EVENT.finditer(data):
        name, raw_ts, raw_dur, raw_external_id = match.groups()
        timestamp = float(raw_ts)
        duration = float(raw_dur)
        cpu_events.append((name, timestamp, duration, int(raw_external_id)))
        phase = _PHASE_ANNOTATIONS.get(name)
        if phase:
            phase_intervals.append((timestamp, timestamp + duration, phase))
            phase_annotation_us[phase] += duration
            annotation_counts[name.decode()] += 1
        elif name.startswith(b"ProfilerStep#"):
            profiler_step_us = max(profiler_step_us, duration)

    if annotation_counts["forward_step"] != 4 or annotation_counts["backward_step"] != 4:
        raise ValueError(
            f"unexpected fwd/bwd annotation count in {path}: {dict(annotation_counts)}"
        )
    if not profiler_step_us or not phase_annotation_us["optimizer"]:
        raise ValueError(f"missing ProfilerStep/optimizer annotation in {path}")

    external_phase_weights: dict[int, Counter[str]] = defaultdict(Counter)
    for _, timestamp, duration, external_id in cpu_events:
        phase = _phase_at(timestamp, phase_intervals)
        if phase:
            external_phase_weights[external_id][phase] += max(duration, 1.0)
    external_phases = {
        external_id: weights.most_common(1)[0][0]
        for external_id, weights in external_phase_weights.items()
    }

    family_stats: dict[tuple[str, str], list[object]] = {}
    kernel_stats: dict[tuple[str, str, bytes], list[float]] = {}
    compute_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    communication_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    mapped_count = 0
    mapped_duration_us = 0.0
    noncommunication_count = 0
    noncommunication_mapped_count = 0
    noncommunication_duration_us = 0.0
    noncommunication_mapped_duration_us = 0.0
    parsed_kernel_count = 0
    pp_p2p_kernel_count = 0
    pp_p2p_duration_us = 0.0

    for match in _KERNEL_EVENT.finditer(data):
        name, _, raw_ts, raw_dur, args = match.groups()
        parsed_kernel_count += 1
        timestamp = float(raw_ts)
        duration = float(raw_dur)
        end = timestamp + duration
        family = classify_kernel(name)
        is_communication = family in COMMUNICATION_FAMILIES
        is_pp_p2p = (
            b"mcclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" in name
            and b'"Collective name"' not in args
        )
        if is_pp_p2p:
            pp_p2p_kernel_count += 1
            pp_p2p_duration_us += duration
        external_match = _EXTERNAL_ID.search(args)
        external_id = int(external_match.group(1)) if external_match else None
        phase = external_phases.get(external_id)
        # Optimizer kernels commonly omit External id.  Timestamp fallback is
        # safe only for non-communication kernels; resident MCCL kernels span
        # pipeline waits and must not be assigned from their midpoint.
        if phase is None and not is_communication:
            phase = _phase_at(timestamp + duration / 2.0, phase_intervals)
        output_phase = phase or "unmapped"
        occupancy_match = _OCCUPANCY.search(args)
        occupancy = float(occupancy_match.group(1)) if occupancy_match else 0.0

        if phase:
            mapped_count += 1
            mapped_duration_us += duration
        if not is_communication:
            noncommunication_count += 1
            noncommunication_duration_us += duration
            if phase:
                noncommunication_mapped_count += 1
                noncommunication_mapped_duration_us += duration

        key = (output_phase, family)
        values = family_stats.setdefault(key, [0, 0.0, 0.0, []])
        values[0] += 1
        values[1] += duration
        values[2] += occupancy * duration
        values[3].append((timestamp, end))
        kernel_key = (output_phase, family, name)
        kernel_values = kernel_stats.setdefault(kernel_key, [0.0, 0.0, 0.0])
        kernel_values[0] += 1
        kernel_values[1] += duration
        kernel_values[2] += occupancy * duration
        if phase:
            target = communication_intervals if is_communication else compute_intervals
            target[phase].append((timestamp, end))

    if parsed_kernel_count != raw_kernel_count:
        raise ValueError(
            f"kernel scanner mismatch in {path}: parsed={parsed_kernel_count}, raw={raw_kernel_count}"
        )

    family_rows = []
    for (phase, family), values in family_stats.items():
        count, duration_us, occupancy_duration, intervals = values
        family_rows.append(
            {
                "iteration": iteration,
                "rank": rank,
                "pp_stage": pp_stage,
                "phase": phase,
                "family": family,
                "kernel_count": int(count),
                "kernel_duration_sum_ms": duration_us / 1000.0,
                "active_union_ms": interval_union_us(intervals) / 1000.0,
                "duration_weighted_occupancy_pct": occupancy_duration / duration_us if duration_us else 0.0,
            }
        )

    phase_rows = []
    for phase in PHASES:
        phase_rows.append(
            {
                "iteration": iteration,
                "rank": rank,
                "pp_stage": pp_stage,
                "phase": phase,
                "cpu_annotation_ms": phase_annotation_us[phase] / 1000.0,
                "compute_active_union_ms": interval_union_us(compute_intervals[phase]) / 1000.0,
                "communication_kernel_union_ms": interval_union_us(communication_intervals[phase]) / 1000.0,
            }
        )

    semantic_cpu_ms = sum(phase_annotation_us.values()) / 1000.0
    summary = {
        "iteration": iteration,
        "rank": rank,
        "pp_stage": pp_stage,
        "trace_path": str(path),
        "profiler_step_ms": profiler_step_us / 1000.0,
        "semantic_cpu_annotation_ms": semantic_cpu_ms,
        "scheduler_wait_residual_ms": profiler_step_us / 1000.0 - semantic_cpu_ms,
        "kernel_count": parsed_kernel_count,
        "mapped_kernel_count": mapped_count,
        "mapped_kernel_duration_ms": mapped_duration_us / 1000.0,
        "noncommunication_kernel_count": noncommunication_count,
        "mapped_noncommunication_kernel_count": noncommunication_mapped_count,
        "noncommunication_kernel_duration_ms": noncommunication_duration_us / 1000.0,
        "mapped_noncommunication_kernel_duration_ms": noncommunication_mapped_duration_us / 1000.0,
        "pp_p2p_kernel_count": pp_p2p_kernel_count,
        "pp_p2p_duration_ns_sum": round(pp_p2p_duration_us * 1000.0),
    }
    compact_kernels = [
        (
            phase,
            family,
            _decode_kernel_name(name),
            int(values[0]),
            values[1] / 1000.0,
            values[2] / values[1] if values[1] else 0.0,
        )
        for (phase, family, name), values in kernel_stats.items()
    ]
    return {
        "summary": summary,
        "phases": phase_rows,
        "families": family_rows,
        "kernels": compact_kernels,
    }


def analytical_forward_flop_weights(
    layers_by_stage: Mapping[int, Iterable[int]],
    architecture: Architecture,
) -> list[dict[str, float | int]]:
    """Build relative forward FLOP weights from the captured model config.

    These are rescaled to the training log's model-FLOP total by the caller;
    only the stage split comes from the analytical formula.
    """
    architecture.validate()
    a = architecture
    tokens = a.global_batch_size * a.sequence_length
    query_width = a.num_attention_heads * (a.qk_head_dim + a.qk_pos_emb_head_dim)
    kv_down_width = a.kv_lora_rank + a.qk_pos_emb_head_dim
    kv_up_width = a.num_attention_heads * (a.qk_head_dim + a.v_head_dim)
    value_width = a.num_attention_heads * a.v_head_dim
    mla_linear_per_token = 2 * (
        a.hidden_size * a.q_lora_rank
        + a.q_lora_rank * query_width
        + a.hidden_size * kv_down_width
        + a.kv_lora_rank * kv_up_width
        + value_width * a.hidden_size
    )
    attention_per_token = (
        2
        * a.num_attention_heads
        * a.sequence_length
        * (a.qk_head_dim + a.qk_pos_emb_head_dim + a.v_head_dim)
    )
    dense_mlp_per_token = 6 * a.hidden_size * a.dense_ffn_hidden_size
    moe_mlp_per_token = (
        2 * a.hidden_size * a.num_experts
        + 6 * a.moe_router_topk * a.hidden_size * a.moe_ffn_hidden_size
        + 6 * a.hidden_size * a.shared_expert_intermediate_size
    )
    dense_layer = tokens * (mla_linear_per_token + attention_per_token + dense_mlp_per_token)
    moe_layer = tokens * (mla_linear_per_token + attention_per_token + moe_mlp_per_token)
    output_head = tokens * 2 * a.hidden_size * a.vocab_size
    last_stage = max(layers_by_stage)
    rows = []
    for stage, layer_ids_iter in sorted(layers_by_stage.items()):
        layer_ids = tuple(sorted(int(value) for value in layer_ids_iter))
        dense_count = sum(layer < a.first_dense_layers for layer in layer_ids)
        moe_count = len(layer_ids) - dense_count
        transformer_weight = dense_count * dense_layer + moe_count * moe_layer
        rows.append(
            {
                "pp_stage": int(stage),
                "layer_count": len(layer_ids),
                "dense_layer_count": dense_count,
                "moe_layer_count": moe_count,
                "transformer_forward_flops_raw": float(transformer_weight),
                "output_head_forward_flops_raw": float(output_head if stage == last_stage else 0.0),
                "forward_flops_raw": float(transformer_weight + (output_head if stage == last_stage else 0.0)),
            }
        )
    return rows

#!/usr/bin/env python3
"""Map stable 256-GPU trace kernels to DAG v6 operators and fit compute costs.

The mapping is deliberately honest about profiler fusion.  Exact kernels such
as FlashAttention bind to one v6 node.  Ambiguous GEMM/layout kernels bind to a
fused cost group plus a set of candidate v6 nodes.  Communication kernels are
retained for audit but never enter the compute parameter table.

Only source iterations 60--100 and PP lane 0 are accepted.  No 224-GPU timing
or validation artifact is opened by this program.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 environment
    import tomli as tomllib


REPO = Path(__file__).resolve().parents[2]
SOURCE_CASE = REPO / "case_256gpu_pp16_cp2_a2a"
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v61_kernel_calibration_2026w36.toml"
SOURCE_ITERATIONS = tuple(range(60, 101, 5))

for directory in (REPO / "src",):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from x10000_analysis.mfu_trace import (  # noqa: E402
    COMMUNICATION_FAMILIES,
    _CPU_EVENT,
    _EXTERNAL_ID,
    _KERNEL_EVENT,
    classify_kernel,
)


_BASE_TIME = re.compile(rb'"baseTimeNanoseconds":\s*([0-9]+)')
_LOCAL_LAYER = re.compile(r"(?:before:)?L([0-9]+):")
_CP_SLOT = re.compile(r":cp([0-9]+)$")


KERNEL_COLUMNS = [
    "iteration", "rank", "pp_stage", "pp_lane", "kernel_index", "kernel_name",
    "kernel_family", "external_id", "cpu_op_candidates", "kernel_start_ns",
    "kernel_end_ns", "kernel_duration_ns", "modeled_window_overlap_ns",
    "dominant_fragment_ns", "cross_segment_count", "phase", "v6_phase",
    "microbatch", "layer_id", "previous_layer_id", "layer_type",
    "layer_context", "autograd_phase", "segment_kind", "semantic_region",
    "execution_scope", "mapping_status", "mapping_method", "confidence",
    "binding_scope", "cost_group", "v6_op_families", "candidate_v6_op_names",
    "candidate_v6_node_ids", "representative_v6_node_id", "source_path",
]

OBSERVATION_COLUMNS = [
    "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch",
    "layer_id", "previous_layer_id", "layer_type", "layer_context",
    "autograd_phase", "parameter_view", "execution_scope", "semantic_slot", "cost_group",
    "binding_scope", "active_union_ns", "kernel_duration_sum_ns",
    "kernel_fragment_count", "exact_active_ns", "family_active_ns",
    "fused_group_active_ns", "support_active_ns", "source_path",
]


@dataclass(frozen=True)
class Mapping:
    status: str
    method: str
    confidence: str
    binding_scope: str
    cost_group: str
    families: tuple[str, ...] = ()
    op_names: tuple[str, ...] = ()
    layer_selector: str = "current"
    autograd_override: str = ""


def configured_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def checked(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def execution_autograd(phase: str, semantic_region: str, kernel_family: str = "", kernel_name: str = "") -> str:
    if phase == "forward":
        return "forward"
    if phase != "backward":
        return ""
    clean = semantic_region.removeprefix("before:")
    match = _CP_SLOT.search(clean)
    if kernel_family == "attention_forward":
        return "recompute"
    if kernel_family == "attention_backward" or "backward" in kernel_name.lower() or "bwd" in kernel_name.lower():
        return "backward"
    if match:
        return "recompute" if int(match.group(1)) <= 3 else "backward"
    if "ep_recompute_" in clean:
        return "recompute"
    if "_backward_" in clean or clean.endswith("_backward_wall") or semantic_region == "FINAL":
        return "backward"
    return "boundary_mixed"


def semantic_slot(semantic_region: str) -> str:
    """Normalize a stage-local trace region into a PP-transfer-safe slot key."""
    if semantic_region == "FINAL":
        return "step_exit"
    before = semantic_region.startswith("before:")
    clean = semantic_region.removeprefix("before:")
    clean = re.sub(r"^L[0-9]+:", "", clean)
    clean = clean.removesuffix("_wall")
    return f"before_{clean}" if before else clean


def communication_service_name(phase: str, semantic_region: str) -> tuple[str, str]:
    clean = semantic_region.removeprefix("before:")
    match = _CP_SLOT.search(clean)
    if match:
        slot = int(match.group(1))
        if phase == "forward" or slot <= 3:
            names = ("cp_a2a_query", "cp_a2a_key", "cp_a2a_value", "cp_a2a_output")
            return names[slot] + "_service", "recompute" if phase == "backward" else "forward"
        backward = (
            "cp_a2a_grad_output_0", "cp_a2a_grad_output_1", "cp_a2a_grad_query",
            "cp_a2a_grad_key", "cp_a2a_grad_value",
        )
        return backward[slot - 4] + "_service", "backward"
    ep_names = (
        "ep_recompute_dispatch", "ep_recompute_combine", "ep_combine_backward",
        "ep_dispatch_backward", "ep_dispatch", "ep_combine",
    )
    for name in ep_names:
        if name + "_wall" in clean:
            mapped = name.removeprefix("ep_recompute_")
            autograd = "recompute" if "recompute" in name else ("backward" if "backward" in name else "forward")
            return mapped + "_service", autograd
    return "", ""


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def map_kernel(
    kernel_name: str,
    family: str,
    phase: str,
    autograd: str,
    semantic_region: str,
    segment_kind: str,
    layer_type: str,
    boundary: bool,
    step_exit: bool,
    cpu_ops: str,
) -> Mapping:
    """Return a conservative kernel -> v6 binding without inventing fusion splits."""
    clean = semantic_region.removeprefix("before:")
    combined = f"{kernel_name} {cpu_ops}".lower()
    if family in COMMUNICATION_FAMILIES:
        service, service_phase = communication_service_name(phase, semantic_region)
        return Mapping(
            "EXCLUDED_FROM_COMPUTE_PARAMETERS", "semantic_collective_window", "excluded",
            "collective_service_audit_only", "", ("collective_service",),
            (service,) if service else (), "current", service_phase,
        )

    if family == "optimizer":
        return Mapping("OUTSIDE_V6_LAYER_COMPUTE", "kernel_family", "excluded", "optimizer", "")

    if family == "attention_forward":
        return Mapping(
            "MAPPED", "exact_kernel_signature", "exact", "single_v6_operator",
            "attention_kernel", ("attention_kernel",), ("flash_attention",),
            "current", "recompute" if phase == "backward" else "forward",
        )
    if family == "attention_backward":
        return Mapping(
            "MAPPED", "exact_kernel_signature", "exact", "single_v6_operator",
            "attention_kernel_backward", ("attention_kernel_backward",),
            ("flash_attention_backward",), "previous" if boundary else "current", "backward",
        )

    if _contains_any(combined, ("routergatinglinear",)):
        backward = autograd == "backward"
        return Mapping(
            "MAPPED", "cpu_op_plus_kernel_signature", "exact", "single_v6_operator",
            "moe_router_backward" if backward else "moe_router",
            ("moe_router_backward" if backward else "moe_router",),
            ("router_logits_backward" if backward else "router_logits",),
            "previous" if boundary and backward else "current", "backward" if backward else autograd,
        )
    if family == "routing" and _contains_any(combined, ("topk",)):
        backward = autograd == "backward"
        return Mapping(
            "MAPPED", "cpu_op_plus_kernel_signature", "exact", "single_v6_operator",
            "moe_router_backward" if backward else "moe_router",
            ("moe_router_backward" if backward else "moe_router",),
            ("router_topk_backward" if backward else "router_topk",),
            "previous" if boundary and backward else "current", "backward" if backward else autograd,
        )

    if boundary:
        if phase == "forward":
            return Mapping(
                "MAPPED_PARTIAL", "cross_layer_trace_window", "fused_group",
                "previous_layer_tail_plus_current_layer_prefix", "forward_layer_boundary_fused",
                ("dense_gemm", "expert_gemm", "shared_expert_gemm", "moe_permutation", "elementwise",
                 "normalization", "attention_projection"), (), "both", "forward",
            )
        return Mapping(
            "MAPPED_PARTIAL", "cross_layer_trace_window", "fused_group",
            "previous_layer_backward_tail_plus_current_recompute_prefix", "backward_layer_boundary_fused",
            ("attention_projection_backward", "normalization_backward", "moe_router_backward",
             "moe_permutation_backward", "normalization", "attention_projection"), (), "both", "boundary_mixed",
        )

    if step_exit:
        backward = phase == "backward"
        return Mapping(
            "MAPPED_PARTIAL", "step_exit_trace_window", "fused_group", "last_layer_tail_plus_step_exit",
            "backward_step_exit_fused" if backward else "forward_step_exit_fused",
            ("attention_projection_backward", "normalization_backward", "moe_permutation_backward", "elementwise")
            if backward else ("dense_gemm", "expert_gemm", "moe_permutation", "elementwise"),
            (), "current", "backward" if backward else "forward",
        )

    if family == "normalization":
        backward = autograd == "backward" or _contains_any(combined, ("normbwd", "norm_backward"))
        return Mapping(
            "MAPPED", "kernel_family_plus_autograd_window", "family", "v6_operator_family",
            "normalization_backward" if backward else "normalization",
            ("normalization_backward" if backward else "normalization",), (), "current",
            "backward" if backward else autograd,
        )

    if family == "routing":
        backward = autograd == "backward"
        if _contains_any(clean, (":cp",)) and _contains_any(combined, ("index_select", "clone_index_select")):
            return Mapping(
                "MAPPED_PARTIAL", "cp_window_layout_kernel", "support", "attention_fused_support",
                "attention_data_transform_support", ("attention_projection_backward",)
                if backward else ("attention_projection",), (), "current", autograd,
            )
        return Mapping(
            "MAPPED", "routing_signature_plus_semantic_window", "family", "v6_operator_family",
            "moe_permutation_backward" if backward else "moe_permutation",
            ("moe_permutation_backward" if backward else "moe_permutation",), (), "current", autograd,
        )

    if family == "gemm":
        backward = autograd == "backward"
        if _contains_any(combined, ("_groupedlinear", "groupedlinear")):
            return Mapping(
                "MAPPED", "cpu_op_plus_kernel_signature", "family", "fused_v6_operator_family",
                "moe_expert_gemm_backward" if backward else "moe_expert_gemm",
                ("expert_gemm_backward", "shared_expert_gemm_backward") if backward
                else ("expert_gemm", "shared_expert_gemm"), (), "current", autograd,
            )
        if "ep_dispatch_backward" in clean:
            return Mapping(
                "MAPPED", "moe_backward_semantic_window", "fused_group", "fused_v6_operator_family",
                "moe_backward_expert_fused", ("expert_gemm_backward", "shared_expert_gemm_backward"),
                (), "current", "backward",
            )
        if "ep_combine" in clean and "backward" not in clean:
            return Mapping(
                "MAPPED", "moe_expert_semantic_window", "fused_group", "fused_v6_operator_family",
                "moe_grouped_expert_fused", ("expert_gemm",), (), "current", autograd,
            )
        if "ep_dispatch" in clean and "backward" not in clean:
            return Mapping(
                "MAPPED_PARTIAL", "moe_predispatch_semantic_window", "fused_group",
                "router_and_shared_expert_fused", "moe_router_shared_expert_fused",
                ("moe_router", "shared_expert_gemm", "attention_projection"), (), "current", autograd,
            )
        if ":cp" in clean:
            return Mapping(
                "MAPPED", "attention_semantic_window", "family", "fused_v6_operator_family",
                "attention_projection_backward" if backward else "attention_projection",
                ("attention_projection_backward" if backward else "attention_projection",),
                (), "current", autograd,
            )
        if layer_type == "dense":
            return Mapping(
                "MAPPED", "dense_layer_semantic_window", "fused_group", "fused_v6_operator_family",
                "dense_mlp_gemm_backward" if backward else "dense_mlp_gemm",
                ("dense_gemm_backward" if backward else "dense_gemm",), (), "current", autograd,
            )
        return Mapping(
            "MAPPED_PARTIAL", "gemm_kernel_in_layer_window", "fused_group", "fused_v6_operator_family",
            "moe_gemm_backward_fused" if backward else "moe_gemm_fused",
            ("expert_gemm_backward", "shared_expert_gemm_backward", "attention_projection_backward")
            if backward else ("expert_gemm", "shared_expert_gemm", "attention_projection"),
            (), "current", autograd,
        )

    if _contains_any(combined, ("rotary_", "rotaryemb")):
        backward = autograd == "backward"
        return Mapping(
            "MAPPED", "rotary_kernel_signature", "family", "attention_fused_support",
            "attention_rotary_backward" if backward else "attention_rotary",
            ("attention_projection_backward" if backward else "attention_projection",), (),
            "current", autograd,
        )

    backward = autograd == "backward"
    if family == "elementwise" and _contains_any(combined, ("silu", "swiglu")):
        return Mapping(
            "MAPPED", "activation_kernel_signature", "family", "fused_v6_operator_family",
            "activation_backward" if backward else "activation",
            ("activation_backward" if backward else "activation",), (), "current", autograd,
        )
    support_family = "elementwise"
    if "ep_" in clean:
        support_family = "moe_permutation_backward" if backward else "moe_permutation"
    elif ":cp" in clean:
        support_family = "attention_projection_backward" if backward else "attention_projection"
    return Mapping(
        "MAPPED_PARTIAL", "coarse_kernel_family_in_semantic_window", "support", "fused_support_group",
        f"{family}_{'backward_' if backward else ''}support", (support_family,), (), "current", autograd,
    )


def useful_cpu_names(data: bytes) -> dict[int, str]:
    by_external: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for match in _CPU_EVENT.finditer(data):
        raw_name, _, raw_duration, raw_external = match.groups()
        name = raw_name.decode("utf-8", errors="replace")
        lowered = name.lower()
        if lowered.startswith("musa") or name in {"aten::empty", "aten::empty_like", "aten::as_strided"}:
            continue
        by_external[int(raw_external)].append((float(raw_duration), name))
    result: dict[int, str] = {}
    for external, values in by_external.items():
        unique: dict[str, float] = {}
        for duration, name in values:
            unique[name] = max(unique.get(name, 0.0), duration)
        ordered = sorted(unique.items(), key=lambda item: (-item[1], item[0]))[:5]
        result[external] = "|".join(name for name, _ in ordered)
    return result


def parse_kernels(path: Path) -> tuple[list[dict[str, Any]], str, int]:
    data = path.read_bytes()
    base_match = _BASE_TIME.search(data)
    if not base_match:
        raise ValueError(f"missing baseTimeNanoseconds in {path}")
    base = int(base_match.group(1))
    cpu_names = useful_cpu_names(data)
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(_KERNEL_EVENT.finditer(data)):
        raw_name, _, raw_timestamp, raw_duration, args = match.groups()
        duration = int(round(float(raw_duration) * 1000.0))
        start = base + int(round(float(raw_timestamp) * 1000.0))
        external_match = _EXTERNAL_ID.search(args)
        external = int(external_match.group(1)) if external_match else -1
        rows.append(
            {
                "kernel_index": index,
                "kernel_name": raw_name.decode("utf-8", errors="replace").replace(r"\u003c", "<").replace(r"\u003e", ">"),
                "kernel_family": classify_kernel(raw_name),
                "external_id": external,
                "cpu_op_candidates": cpu_names.get(external, ""),
                "start_ns": start,
                "end_ns": start + duration,
                "duration_ns": duration,
            }
        )
    raw_count = data.count(b'"cat": "kernel"')
    if len(rows) != raw_count:
        raise ValueError(f"kernel scan mismatch for {path}: parsed={len(rows)} raw={raw_count}")
    return rows, hashlib.sha256(data).hexdigest(), len(data)


def annotate_segments(frame: pd.DataFrame, stage_layers: dict[int, list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in frame.sort_values(["observed_start_ns", "observed_end_ns", "segment_id"]).itertuples(index=False):
        stage = int(item.pp_stage)
        phase = str(item.phase)
        layers = stage_layers[stage]
        execution = layers if phase == "forward" else list(reversed(layers))
        region = str(item.semantic_region)
        local_match = _LOCAL_LAYER.search(region)
        local = int(local_match.group(1)) if local_match else -1
        layer_id = int(item.layer_id)
        step_exit = region == "FINAL"
        if step_exit:
            layer_id = execution[-1]
            local = layers.index(layer_id)
        boundary = bool(
            str(item.segment_kind) == "gap"
            and region.startswith("before:")
            and region.endswith(":cp0")
            and layer_id != execution[0]
        )
        previous_layer = -1
        if boundary:
            position = execution.index(layer_id)
            previous_layer = execution[position - 1]
        entry = bool(region.startswith("before:") and region.endswith(":cp0") and layer_id == execution[0])
        layer_type = "dense" if layer_id == 0 else "moe"
        previous_type = "dense" if previous_layer == 0 else ("moe" if previous_layer >= 0 else "")
        if boundary:
            context = f"{previous_type}_to_{layer_type}"
        elif step_exit:
            context = f"{layer_type}_to_step_exit"
        elif entry:
            context = f"step_entry_to_{layer_type}"
        else:
            context = layer_type
        rows.append(
            {
                **item._asdict(),
                "observed_start_ns": int(item.observed_start_ns),
                "observed_end_ns": int(item.observed_end_ns),
                "layer_id": layer_id,
                "previous_layer_id": previous_layer,
                "layer_type": layer_type,
                "layer_context": context,
                "boundary": boundary,
                "step_exit": step_exit,
                "entry": entry,
                "local_layer": local,
            }
        )
    for left, right in zip(rows, rows[1:]):
        if int(right["observed_start_ns"]) < int(left["observed_end_ns"]):
            raise ValueError(
                f"overlapping semantic segments rank={left['rank']} iter={left['iteration']}: "
                f"{left['segment_id']} -> {right['segment_id']}"
            )
    return rows


def load_v6_node_index(path: Path, lane: int) -> tuple[dict[tuple[Any, ...], str], dict[tuple[Any, ...], list[tuple[int, str, str]]]]:
    by_name: dict[tuple[Any, ...], str] = {}
    by_family: dict[tuple[Any, ...], list[tuple[int, str, str]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["pp_lane"]) != lane or row["kind"] != "operator" or row["resource"] != "gpu_compute":
                continue
            prefix = (
                int(row["rank"]), row["phase"], int(row["microbatch"]), int(row["layer_id"]),
                row["autograd_phase"],
            )
            by_name[(*prefix, row["op_name"])] = row["node_id"]
            by_family[(*prefix, row["op_family"])].append(
                (int(row["template_index"]), row["op_name"], row["node_id"])
            )
    for values in by_family.values():
        values.sort()
    return by_name, by_family


def resolve_v6_nodes(
    mapping: Mapping,
    segment: dict[str, Any],
    rank: int,
    phase: str,
    microbatch: int,
    by_name: dict[tuple[Any, ...], str],
    by_family: dict[tuple[Any, ...], list[tuple[int, str, str]]],
) -> tuple[list[str], list[str], str]:
    v6_phase = "FWD" if phase == "forward" else "BWD"
    current = int(segment["layer_id"])
    previous = int(segment["previous_layer_id"])
    autograd = mapping.autograd_override or execution_autograd(phase, str(segment["semantic_region"]))
    scopes: list[tuple[int, str]] = []
    if mapping.layer_selector in {"current", "both"}:
        current_phase = "recompute" if autograd == "boundary_mixed" and phase == "backward" else autograd
        scopes.append((current, current_phase))
    if mapping.layer_selector in {"previous", "both"} and previous >= 0:
        previous_phase = "backward" if phase == "backward" else "forward"
        scopes.append((previous, previous_phase))
    names: list[str] = []
    node_ids: list[str] = []
    for layer_id, auto in scopes:
        for op_name in mapping.op_names:
            node = by_name.get((rank, v6_phase, microbatch, layer_id, auto, op_name))
            if node:
                names.append(op_name)
                node_ids.append(node)
        # An explicit op name is a stronger binding than its broader family.
        # Do not add sibling nodes (for example router_logits beside an exact
        # router_topk match), otherwise an exact mapping becomes ambiguous.
        if not mapping.op_names:
            for family in mapping.families:
                for _, op_name, node in by_family.get((rank, v6_phase, microbatch, layer_id, auto, family), []):
                    names.append(op_name)
                    node_ids.append(node)
    dedup_names = list(dict.fromkeys(names))[:12]
    dedup_nodes = list(dict.fromkeys(node_ids))[:12]
    representative = dedup_nodes[0] if mapping.confidence == "exact" and len(dedup_nodes) == 1 else ""
    return dedup_names, dedup_nodes, representative


def parameter_frame(observations: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "phase", "autograd_phase", "layer_type", "layer_context", "parameter_view",
        "execution_scope", "semantic_slot", "cost_group", "binding_scope",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in observations.groupby(keys, dropna=False, sort=True):
        values = group["active_union_ns"].astype(float)
        median = float(values.median())
        mad = float((values - median).abs().median())
        mean = float(values.mean())
        confidence_totals = {
            name: int(group[f"{name}_active_ns"].sum())
            for name in ("exact", "family", "fused_group", "support")
        }
        confidence_sum = sum(confidence_totals.values())
        cost_group = str(key[7])
        if cost_group.startswith("__"):
            status = "READY_SOURCE_ACTIVE_UNION" if len(group) >= 9 else "PARTIAL_LOW_SAMPLE_COUNT"
        elif confidence_totals["fused_group"] or "fused" in cost_group:
            status = "PARTIAL_FUSED_GROUP_REQUIRES_SINGLE_COST_BINDING"
        elif len(group) < 9:
            status = "PARTIAL_LOW_SAMPLE_COUNT"
        else:
            status = "READY_SOURCE_CALIBRATED"
        rows.append(
            {
                **dict(zip(keys, key)),
                "calibration_samples": int(len(group)),
                "source_iteration_count": int(group["iteration"].nunique()),
                "source_rank_count": int(group["rank"].nunique()),
                "median_active_union_ns": int(round(median)),
                "p10_active_union_ns": int(round(float(values.quantile(0.10)))),
                "p90_active_union_ns": int(round(float(values.quantile(0.90)))),
                "mad_active_union_ns": int(round(mad)),
                "mean_active_union_ns": int(round(mean)),
                "coefficient_of_variation": float(values.std(ddof=0) / mean) if mean else 0.0,
                "exact_duration_share": confidence_totals["exact"] / confidence_sum if confidence_sum else 0.0,
                "family_duration_share": confidence_totals["family"] / confidence_sum if confidence_sum else 0.0,
                "fused_group_duration_share": confidence_totals["fused_group"] / confidence_sum if confidence_sum else 0.0,
                "support_duration_share": confidence_totals["support"] / confidence_sum if confidence_sum else 0.0,
                "parameter_unit": "ns_per_rank_layer_occurrence",
                "fit_source": "256gpu_iterations_60_100_lane0_all_pp_stages",
                "transfer_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def render_html(coverage: dict[str, Any], parameters: pd.DataFrame, bindings: pd.DataFrame) -> str:
    by_conf = coverage["compute_fragment_duration_by_confidence_ns"]
    total = max(sum(by_conf.values()), 1)
    top = parameters[parameters["parameter_view"].eq("all_windows")].sort_values(
        "median_active_union_ns", ascending=False
    ).head(18)
    parameter_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.phase))}</td><td>{escape(str(row.autograd_phase))}</td>"
        f"<td>{escape(str(row.layer_context))}</td><td><code>{escape(str(row.cost_group))}</code></td>"
        f"<td>{row.median_active_union_ns / 1e6:.3f}</td><td>{row.p10_active_union_ns / 1e6:.3f}</td>"
        f"<td>{row.p90_active_union_ns / 1e6:.3f}</td><td>{int(row.calibration_samples)}</td>"
        f"<td>{escape(str(row.transfer_status))}</td></tr>"
        for row in top.itertuples(index=False)
    )
    binding_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row.cost_group))}</code></td>"
        f"<td>{escape(str(row.v6_op_families))}</td><td>{escape(str(row.candidate_v6_op_names))}</td>"
        f"<td>{escape(str(row.binding_scope))}</td><td>{int(row.kernel_fragment_count)}</td></tr>"
        for row in bindings.sort_values("kernel_fragment_duration_ns", ascending=False).head(20).itertuples(index=False)
    )
    bars = "".join(
        f'<div class="bar"><span>{escape(name)}</span><i style="width:{100 * value / total:.2f}%"></i>'
        f'<b>{100 * value / total:.1f}%</b></div>'
        for name, value in by_conf.items()
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DAG v6.1 kernel calibration</title>
<style>
body{{margin:0;background:#09111f;color:#dbe7f5;font:14px/1.55 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:34px}}
h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:34px}}.muted{{color:#90a4bc}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card,.panel{{background:#111d2e;border:1px solid #263850;border-radius:12px;padding:16px}}.value{{font-size:25px;font-weight:700;color:#68d5ff}}
.flow{{display:flex;gap:8px;align-items:stretch;overflow:auto}}.flow div{{min-width:170px;background:#14243a;border:1px solid #315071;border-radius:10px;padding:14px}}.arrow{{align-self:center;color:#68d5ff;font-size:22px}}
.bar{{display:grid;grid-template-columns:110px 1fr 56px;gap:10px;align-items:center;margin:8px 0}}.bar i{{height:12px;background:#36c5a4;border-radius:8px;min-width:2px}}.bar b{{text-align:right}}
table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #263850;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#111d2e}}code{{color:#9ae6ff}}.warn{{color:#ffd27a}}
@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>DAG v6.1 · 256 卡 trace kernel → 算子节点</h1>
<p class="muted">源端稳定区间 iteration 60–100；lane0 覆盖全部 16 个 PP stage。该页只说明映射与计算参数，不给出 224 卡精度结论。</p>
<section class="cards">
<div class="card"><div class="muted">原始 trace</div><div class="value">{coverage['raw_trace_count']}</div></div>
<div class="card"><div class="muted">GPU kernel</div><div class="value">{coverage['raw_kernel_count']:,}</div></div>
<div class="card"><div class="muted">计算参数行</div><div class="value">{len(parameters):,}</div></div>
<div class="card"><div class="muted">实际读取目标 timing</div><div class="value">0</div></div>
</section>
<h2>映射路径</h2><section class="flow">
<div><b>1 · 原始 kernel</b><br>保留名称、时间、External id 与可用 CPU op。</div><span class="arrow">→</span>
<div><b>2 · 语义窗口</b><br>按 trace 时间切入 FWD/BWD、microbatch、layer、CP/EP 窗口。</div><span class="arrow">→</span>
<div><b>3 · v6 绑定</b><br>Flash 等精确绑定；融合 GEMM 绑定候选节点组。</div><span class="arrow">→</span>
<div><b>4 · 计算参数</b><br>同组 kernel 先做时间并集，再对 source occurrence 取中位数。</div>
</section>
<h2>为什么不是所有 kernel 都一对一</h2>
<div class="panel"><p>Profiler 中多个 v6 语义算子可能融合成一个 GEMM/布局 kernel；反过来，一个跨层窗口也可能同时包含“上一层尾部 + 下一层前缀”。</p>
<p class="warn">因此 fused_group 只能作为一个整体成本应用一次，不能给候选节点逐个重复赋相同耗时。</p>{bars}</div>
<h2>主要计算参数（all_windows，中位数按耗时排序）</h2><div class="panel" style="overflow:auto"><table><thead><tr><th>phase</th><th>autograd</th><th>context</th><th>cost group</th><th>median ms</th><th>p10</th><th>p90</th><th>N</th><th>状态</th></tr></thead><tbody>{parameter_rows}</tbody></table></div>
<h2>kernel 到 v6 候选节点组</h2><div class="panel" style="overflow:auto"><table><thead><tr><th>cost group</th><th>v6 family</th><th>候选节点名</th><th>绑定范围</th><th>fragment</th></tr></thead><tbody>{binding_rows}</tbody></table></div>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    requested = tuple(int(value) for value in config["split"]["source_iterations"])
    if requested != SOURCE_ITERATIONS:
        raise ValueError(f"source split must be exact stable 60--100: {requested}")
    if bool(config["split"]["target_timing_allowed"]):
        raise ValueError("target timing access must remain disabled")
    lane = int(config["split"]["source_pp_lane"])
    output = configured_path(config["outputs"]["output_dir"])
    model_dir = output / "model_inputs"
    calibration_dir = output / "calibration"
    logs_dir = output / "logs"
    for directory in (model_dir, calibration_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_paths = {name: checked(configured_path(value)) for name, value in config["inputs"].items()}
    inventory = pd.read_csv(input_paths["trace_inventory"])
    inventory = inventory[inventory["iteration"].isin(SOURCE_ITERATIONS) & inventory["pp_lane"].eq(lane)].copy()
    inventory = inventory.sort_values(["iteration", "pp_stage", "rank"])
    expected = len(SOURCE_ITERATIONS) * int(config["split"]["source_pp_stages"])
    if len(inventory) != expected:
        raise ValueError(f"expected {expected} lane{lane} traces, got {len(inventory)}")
    if tuple(sorted(int(value) for value in inventory["iteration"].unique())) != SOURCE_ITERATIONS:
        raise ValueError("inventory iteration split mismatch")

    layer_map = pd.read_csv(input_paths["source_layer_stage_map"])
    stage_layers = {
        int(stage): [int(value) for value in group.sort_values("stage_local_layer")["layer_id"]]
        for stage, group in layer_map.groupby("pp_stage")
    }
    segments_frame = pd.read_csv(input_paths["trace_semantic_segments"])
    segments_frame = segments_frame[
        segments_frame["iteration"].isin(SOURCE_ITERATIONS) & segments_frame["pp_lane"].eq(lane)
    ].copy()
    if tuple(sorted(int(value) for value in segments_frame["iteration"].unique())) != SOURCE_ITERATIONS:
        raise ValueError("semantic segment split mismatch")
    segments_by_trace = {
        (int(iteration), int(rank), str(path)): annotate_segments(group, stage_layers)
        for (iteration, rank, path), group in segments_frame.groupby(["iteration", "rank", "source_path"], sort=False)
    }
    by_name, by_family = load_v6_node_index(input_paths["v6_source_operator_nodes"], lane)

    kernel_output = model_dir / "kernel_to_v6_operator_mapping.csv.gz"
    temporary_kernel = kernel_output.with_name(kernel_output.name + ".tmp")
    coverage = Counter()
    confidence_duration = Counter()
    method_duration = Counter()
    binding_accumulator: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    observation_rows: list[dict[str, Any]] = []
    raw_inputs: list[dict[str, Any]] = []

    with gzip.open(temporary_kernel, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KERNEL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for trace_number, item in enumerate(inventory.itertuples(index=False), start=1):
            trace_path = checked(Path(str(item.path)))
            trace_key = (int(item.iteration), int(item.rank), str(trace_path))
            segments = segments_by_trace.get(trace_key)
            if not segments:
                raise ValueError(f"no semantic segments for {trace_key}")
            kernels, trace_sha, trace_size = parse_kernels(trace_path)
            raw_inputs.append(
                {"path": str(trace_path), "role": "source_trace_iteration_60_100_lane0", "sha256": trace_sha, "size_bytes": trace_size}
            )
            starts = [int(segment["observed_start_ns"]) for segment in segments]
            ends = [int(segment["observed_end_ns"]) for segment in segments]
            local_intervals: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
            local_sums = Counter()
            local_counts = Counter()
            local_confidence: dict[tuple[Any, ...], dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))

            for kernel in kernels:
                coverage["raw_kernel_count"] += 1
                coverage["raw_kernel_duration_ns"] += int(kernel["duration_ns"])
                left = bisect.bisect_right(ends, int(kernel["start_ns"]))
                right = bisect.bisect_left(starts, int(kernel["end_ns"]))
                fragments: list[tuple[int, dict[str, Any], int, int, Mapping, str, list[str], list[str], str]] = []
                for segment in segments[left:right]:
                    fragment_start = max(int(kernel["start_ns"]), int(segment["observed_start_ns"]))
                    fragment_end = min(int(kernel["end_ns"]), int(segment["observed_end_ns"]))
                    if fragment_end <= fragment_start:
                        continue
                    autograd = execution_autograd(
                        str(segment["phase"]), str(segment["semantic_region"]),
                        str(kernel["kernel_family"]), str(kernel["kernel_name"]),
                    )
                    mapping = map_kernel(
                        str(kernel["kernel_name"]), str(kernel["kernel_family"]), str(segment["phase"]),
                        autograd, str(segment["semantic_region"]), str(segment["segment_kind"]),
                        str(segment["layer_type"]), bool(segment["boundary"]), bool(segment["step_exit"]),
                        str(kernel["cpu_op_candidates"]),
                    )
                    if mapping.autograd_override:
                        autograd = mapping.autograd_override
                    candidate_names, candidate_nodes, representative = resolve_v6_nodes(
                        mapping, segment, int(item.rank), str(segment["phase"]), int(segment["microbatch"]),
                        by_name, by_family,
                    )
                    fragments.append(
                        (fragment_end - fragment_start, segment, fragment_start, fragment_end, mapping,
                         autograd, candidate_names, candidate_nodes, representative)
                    )
                    if str(kernel["kernel_family"]) not in COMMUNICATION_FAMILIES and mapping.cost_group:
                        execution_scope = "overlap_with_collective" if segment["segment_kind"] == "communication" else "exposed_gap"
                        common = (
                            int(item.iteration), int(item.rank), int(item.pp_stage), int(item.pp_lane),
                            str(segment["phase"]), int(segment["microbatch"]), int(segment["layer_id"]),
                            int(segment["previous_layer_id"]), str(segment["layer_type"]), str(segment["layer_context"]),
                            autograd,
                        )
                        for parameter_view, scope in (("window_split", execution_scope), ("all_windows", "all_windows")):
                            slot = semantic_slot(str(segment["semantic_region"])) if parameter_view == "window_split" else "ALL"
                            key = (*common, parameter_view, scope, slot, mapping.cost_group, mapping.binding_scope, str(trace_path))
                            local_intervals[key].append((fragment_start, fragment_end))
                            local_sums[key] += fragment_end - fragment_start
                            local_counts[key] += 1
                            local_confidence[key][mapping.confidence].append((fragment_start, fragment_end))
                            aggregate_group = "__slot_total__" if parameter_view == "window_split" else "__occurrence_total__"
                            aggregate_key = (*common, parameter_view, scope, slot, aggregate_group, "active_union_no_double_count", str(trace_path))
                            local_intervals[aggregate_key].append((fragment_start, fragment_end))
                            local_sums[aggregate_key] += fragment_end - fragment_start
                            local_counts[aggregate_key] += 1
                            local_confidence[aggregate_key][mapping.confidence].append((fragment_start, fragment_end))
                            physical_common = (*common[:-1], "physical_mixed")
                            physical_group = "__physical_slot_total__" if parameter_view == "window_split" else "__physical_occurrence_total__"
                            physical_key = (*physical_common, parameter_view, scope, slot, physical_group, "physical_active_union_no_double_count", str(trace_path))
                            local_intervals[physical_key].append((fragment_start, fragment_end))
                            local_sums[physical_key] += fragment_end - fragment_start
                            local_counts[physical_key] += 1
                            local_confidence[physical_key][mapping.confidence].append((fragment_start, fragment_end))
                        confidence_duration[mapping.confidence] += fragment_end - fragment_start
                        method_duration[mapping.method] += fragment_end - fragment_start
                        binding_key = (
                            mapping.cost_group, "|".join(mapping.families), "|".join(candidate_names),
                            mapping.binding_scope, mapping.confidence,
                        )
                        binding = binding_accumulator.setdefault(
                            binding_key,
                            {"kernel_fragment_count": 0, "kernel_fragment_duration_ns": 0,
                             "candidate_v6_node_examples": []},
                        )
                        binding["kernel_fragment_count"] += 1
                        binding["kernel_fragment_duration_ns"] += fragment_end - fragment_start
                        for node in candidate_nodes:
                            if node not in binding["candidate_v6_node_examples"] and len(binding["candidate_v6_node_examples"]) < 8:
                                binding["candidate_v6_node_examples"].append(node)

                if fragments:
                    coverage["kernels_overlapping_v6_windows"] += 1
                    dominant = max(fragments, key=lambda value: value[0])
                    fragment_ns, segment, _, _, mapping, autograd, candidate_names, candidate_nodes, representative = dominant
                    overlap_ns = sum(value[0] for value in fragments)
                    if str(kernel["kernel_family"]) in COMMUNICATION_FAMILIES:
                        coverage["communication_kernels_in_v6_windows"] += 1
                    else:
                        coverage["compute_kernels_in_v6_windows"] += 1
                        coverage["compute_kernel_modeled_overlap_ns"] += overlap_ns
                    row = {
                        "iteration": int(item.iteration), "rank": int(item.rank), "pp_stage": int(item.pp_stage),
                        "pp_lane": int(item.pp_lane), **kernel,
                        "kernel_start_ns": kernel["start_ns"], "kernel_end_ns": kernel["end_ns"],
                        "kernel_duration_ns": kernel["duration_ns"], "modeled_window_overlap_ns": overlap_ns,
                        "dominant_fragment_ns": fragment_ns, "cross_segment_count": len(fragments),
                        "phase": segment["phase"], "v6_phase": "FWD" if segment["phase"] == "forward" else "BWD",
                        "microbatch": segment["microbatch"], "layer_id": segment["layer_id"],
                        "previous_layer_id": segment["previous_layer_id"], "layer_type": segment["layer_type"],
                        "layer_context": segment["layer_context"], "autograd_phase": autograd,
                        "segment_kind": segment["segment_kind"], "semantic_region": segment["semantic_region"],
                        "execution_scope": "overlap_with_collective" if segment["segment_kind"] == "communication" else "exposed_gap",
                        "mapping_status": mapping.status, "mapping_method": mapping.method,
                        "confidence": mapping.confidence, "binding_scope": mapping.binding_scope,
                        "cost_group": mapping.cost_group, "v6_op_families": "|".join(mapping.families),
                        "candidate_v6_op_names": "|".join(candidate_names),
                        "candidate_v6_node_ids": "|".join(candidate_nodes),
                        "representative_v6_node_id": representative, "source_path": str(trace_path),
                    }
                else:
                    coverage["kernels_outside_v6_windows"] += 1
                    row = {
                        "iteration": int(item.iteration), "rank": int(item.rank), "pp_stage": int(item.pp_stage),
                        "pp_lane": int(item.pp_lane), **kernel,
                        "kernel_start_ns": kernel["start_ns"], "kernel_end_ns": kernel["end_ns"],
                        "kernel_duration_ns": kernel["duration_ns"], "modeled_window_overlap_ns": 0,
                        "dominant_fragment_ns": 0, "cross_segment_count": 0, "phase": "", "v6_phase": "",
                        "microbatch": -1, "layer_id": -1, "previous_layer_id": -1, "layer_type": "",
                        "layer_context": "", "autograd_phase": "", "segment_kind": "",
                        "semantic_region": "", "execution_scope": "", "mapping_status": "OUTSIDE_V6_LAYER_WINDOWS",
                        "mapping_method": "none", "confidence": "excluded", "binding_scope": "",
                        "cost_group": "", "v6_op_families": "", "candidate_v6_op_names": "",
                        "candidate_v6_node_ids": "", "representative_v6_node_id": "", "source_path": str(trace_path),
                    }
                writer.writerow({column: row.get(column, "") for column in KERNEL_COLUMNS})

            for key, intervals in local_intervals.items():
                (
                    iteration, rank, stage, pp_lane, phase, microbatch, layer_id, previous_layer_id,
                    layer_type, layer_context, autograd, parameter_view, execution_scope, semantic_slot_key,
                    cost_group, binding_scope, source_path,
                ) = key
                confidence = local_confidence[key]
                observation_rows.append(
                    {
                        "iteration": iteration, "rank": rank, "pp_stage": stage, "pp_lane": pp_lane,
                        "phase": phase, "microbatch": microbatch, "layer_id": layer_id,
                        "previous_layer_id": previous_layer_id, "layer_type": layer_type,
                        "layer_context": layer_context, "autograd_phase": autograd,
                        "parameter_view": parameter_view, "execution_scope": execution_scope,
                        "semantic_slot": semantic_slot_key,
                        "cost_group": cost_group, "binding_scope": binding_scope,
                        "active_union_ns": interval_union_ns(intervals), "kernel_duration_sum_ns": local_sums[key],
                        "kernel_fragment_count": local_counts[key],
                        "exact_active_ns": interval_union_ns(confidence.get("exact", [])),
                        "family_active_ns": interval_union_ns(confidence.get("family", [])),
                        "fused_group_active_ns": interval_union_ns(confidence.get("fused_group", [])),
                        "support_active_ns": interval_union_ns(confidence.get("support", [])),
                        "source_path": source_path,
                    }
                )
            if trace_number == 1 or trace_number % 8 == 0 or trace_number == len(inventory):
                print(
                    f"kernel_mapping {trace_number}/{len(inventory)} iteration={item.iteration} "
                    f"rank={item.rank} kernels={len(kernels)}",
                    flush=True,
                )
    temporary_kernel.replace(kernel_output)

    observations = pd.DataFrame(observation_rows, columns=OBSERVATION_COLUMNS)
    observation_path = calibration_dir / "operator_cost_group_observations.csv.gz"
    observations.to_csv(observation_path, index=False, compression="gzip")
    parameters = parameter_frame(observations)
    parameter_path = calibration_dir / "compute_parameters.csv"
    parameters.to_csv(parameter_path, index=False)

    binding_rows = []
    for key, values in binding_accumulator.items():
        cost_group, families, names, scope, confidence = key
        binding_rows.append(
            {
                "cost_group": cost_group, "v6_op_families": families,
                "candidate_v6_op_names": names, "binding_scope": scope,
                "confidence": confidence, "kernel_fragment_count": values["kernel_fragment_count"],
                "kernel_fragment_duration_ns": values["kernel_fragment_duration_ns"],
                "candidate_v6_node_examples": "|".join(values["candidate_v6_node_examples"]),
                "cost_application_policy": "apply_once_to_group_representative_not_each_candidate",
            }
        )
    bindings = pd.DataFrame(binding_rows).sort_values(
        ["kernel_fragment_duration_ns", "cost_group"], ascending=[False, True]
    )
    binding_path = model_dir / "v6_operator_cost_bindings.csv"
    bindings.to_csv(binding_path, index=False)

    coverage_payload = {
        "schema": "dag-v6.1-kernel-mapping-coverage-v1",
        "status": "PARTIAL_LANE0_ALL_PP_STAGES_SOURCE_ONLY",
        "source_iterations": list(SOURCE_ITERATIONS),
        "source_pp_lane": lane,
        "source_pp_stages": sorted(int(value) for value in inventory["pp_stage"].unique()),
        "raw_trace_count": len(raw_inputs),
        **{name: int(value) for name, value in coverage.items()},
        "compute_fragment_duration_by_confidence_ns": {
            name: int(confidence_duration.get(name, 0))
            for name in ("exact", "family", "fused_group", "support")
        },
        "compute_fragment_duration_by_mapping_method_ns": dict(sorted(method_duration.items())),
        "communication_compute_parameter_rows": 0,
        "target_timing_files_read": 0,
        "interpretation": {
            "exact": "one kernel signature resolves to one semantic v6 operator",
            "family": "kernel resolves to a v6 operator family but not a unique node",
            "fused_group": "one calibrated cost spans several candidate v6 nodes and must be applied once",
            "support": "layout/memory/elementwise support is retained as a fused support cost",
        },
        "known_gap": "other PP lanes are not yet used for variance validation",
    }
    coverage_path = output / "kernel_mapping_coverage.json"
    atomic_json(coverage_path, coverage_payload)

    calibration_payload = {
        "schema": "dag-v6.1-source-compute-parameters-v1",
        "status": "PARTIAL_SOURCE_PARAMETERS_FROZEN_TARGET_BINDING_NOT_RUN",
        "fit_iterations": list(SOURCE_ITERATIONS),
        "fit_ranks": sorted(int(value) for value in inventory["rank"].unique()),
        "fit_pp_lane": lane,
        "parameter_statistic": "median active GPU interval union per rank/layer occurrence",
        "parameter_views": {
            "all_windows": "full cost group across gap and communication-overlap windows",
            "window_split": "same cost split into exposed_gap and overlap_with_collective parts",
        },
        "parameter_table": str(parameter_path),
        "observation_table": str(observation_path),
        "parameter_rows": len(parameters),
        "stage_id_is_parameter": False,
        "target_trace_used": False,
        "captured_time_identity_replay_claim": False,
        "cost_binding_rule": "a fused cost group is charged once; candidate v6 nodes are not each charged the group median",
    }
    calibration_json = calibration_dir / "calibration_parameters.json"
    atomic_json(calibration_json, calibration_payload)

    base_inputs = []
    for role, path in input_paths.items():
        base_inputs.append(
            {"path": str(path), "role": role, "sha256": sha256(path), "size_bytes": path.stat().st_size}
        )
    access_payload = {
        "schema": "dag-v6.1-input-access-audit-v1",
        "allowed_split": {"iterations": list(SOURCE_ITERATIONS), "pp_lane": lane},
        "static_and_source_inputs": base_inputs,
        "raw_trace_inputs": raw_inputs,
        "model_process_read_fields": {
            "trace_inventory": ["iteration", "rank", "pp_stage", "pp_lane", "path"],
            "semantic_segments": [
                "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch", "layer_id",
                "segment_kind", "semantic_region", "observed_start_ns", "observed_end_ns", "source_path",
            ],
            "raw_trace": [
                "baseTimeNanoseconds", "kernel.name", "kernel.ts", "kernel.dur", "kernel.External id",
                "cpu_op/user_annotation/privateuse1_runtime.name", "External id",
            ],
            "v6_source_operator_nodes": [
                "rank", "pp_lane", "phase", "microbatch", "layer_id", "autograd_phase", "op_name",
                "op_family", "resource", "node_id", "template_index",
            ],
        },
        "forbidden_target_timing_or_ground_truth_inputs": [],
        "target_timing_files_read": 0,
        "note": "224 topology/timing/step/TFLOP/error fields are not inputs to this source calibration program",
    }
    access_path = output / "input_access_audit.json"
    atomic_json(access_path, access_payload)

    reproduction = (
        f"cd {REPO}\n"
        f"python {Path(__file__).resolve()} --config {config_path}\n"
        f"pytest -q case_224gpu_pp14_cp2_a2a/tests/test_dag_v60_operator_ir.py "
        f"case_224gpu_pp14_cp2_a2a/tests/test_dag_v61_kernel_calibration.py "
        f"--junitxml={logs_dir / 'pytest.xml'}\n"
        f"python {REPO / 'case_224gpu_pp14_cp2_a2a/scripts/finalize_dag_v61_kernel_calibration.py'} "
        f"--config {config_path}\n"
    )
    reproduction_path = output / "reproduction_command.txt"
    atomic_text(reproduction_path, reproduction)

    report = f"""# DAG v6.1：256 卡 trace kernel → v6 算子节点

状态：`PARTIAL_SOURCE_PARAMETERS_FROZEN_TARGET_BINDING_NOT_RUN`。

本阶段完成了两件事：

1. 将 source 256 卡稳定区间 iteration 60–100、lane0 的全部 16 个 PP stage 原始 GPU kernel 映射到 v6 算子或融合算子组；
2. 对映射后的计算 fragment 做区间并集，并生成 source-only 计算参数表。

## 映射口径

- `exact`：FlashAttention、router top-k 等可以落到单个 v6 节点；
- `family`：能确认 v6 算子族，但 profiler fusion 不允许确认唯一节点；
- `fused_group`：一个 trace 窗口同时覆盖多个 v6 节点，尤其是“上一层尾部 + 下一层前缀”；
- `support`：layout/memory/elementwise 等支持 kernel，保留为融合支持成本；
- 通信 kernel 只用于审计候选 collective service，**不进入计算参数**。

融合组的耗时只能应用一次，不能给候选节点逐个重复赋值。

## 参数口径

- `all_windows`：一个 cost group 在 gap 与通信重叠窗口中的完整 active union；
- `window_split`：拆为 `exposed_gap` 与 `overlap_with_collective`，供后续 DAG overlap 绑定；
- 冻结值为 rank/layer occurrence 的中位 active union，并同时保存 p10、p90、MAD、CV；
- 参数键不包含 source `pp_stage`，未读取任何 224 卡 timing/Step/TFLOP/error 字段。

## 产物

- kernel 映射：`{kernel_output}`
- v6 成本绑定：`{binding_path}`
- occurrence 观测：`{observation_path}`
- 冻结参数：`{parameter_path}`
- 覆盖审计：`{coverage_path}`
- 输入审计：`{access_path}`

## 尚未完成

- lane1–15 尚未用于跨 lane 方差验证；
- fused group 尚未选择 target graph 中“只计一次”的代表节点；
- 本阶段没有生成 224 卡预测或准确率结论。
"""
    report_path = output / "DAG_V61_KERNEL_CALIBRATION_REPORT.md"
    atomic_text(report_path, report)
    html_path = output / "dag_v61_kernel_calibration.html"
    atomic_text(html_path, render_html(coverage_payload, parameters, bindings))

    build_log_path = logs_dir / "build.log"
    atomic_text(
        build_log_path,
        "\n".join(
            [
                "DAG v6.1 kernel calibration build PASS",
                f"generated_at={datetime.now(timezone.utc).isoformat()}",
                f"source_iterations={','.join(map(str, SOURCE_ITERATIONS))}",
                f"source_lane={lane}",
                f"raw_trace_count={len(raw_inputs)}",
                f"raw_kernel_count={coverage['raw_kernel_count']}",
                f"observation_rows={len(observations)}",
                f"parameter_rows={len(parameters)}",
                "target_timing_files_read=0",
                "status=PARTIAL_SOURCE_PARAMETERS_FROZEN_TARGET_BINDING_NOT_RUN",
            ]
        ) + "\n",
    )

    code_hash_entries = [config_path, Path(__file__).resolve(), input_paths["v6_source_operator_nodes"], input_paths["v6_operator_templates"]]
    tree_digest = hashlib.sha256()
    for path in code_hash_entries:
        tree_digest.update(str(path).encode())
        tree_digest.update(sha256(path).encode())
    provenance_path = output / "provenance.json"
    atomic_json(
        provenance_path,
        {
            "schema": "dag-v6.1-provenance-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": str(REPO),
            "git_branch": git_output("branch", "--show-current"),
            "git_head": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--short").splitlines(),
            "version_state": "PARTIAL_UNCOMMITTED_WORKTREE_PINNED_BY_SHA256",
            "source_code_tree_hash": tree_digest.hexdigest(),
            "code_and_config": [
                {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in code_hash_entries
            ],
            "input_access_audit": str(access_path),
        },
    )

    artifacts = [
        kernel_output, observation_path, parameter_path, binding_path, coverage_path,
        calibration_json, access_path, reproduction_path, report_path, html_path,
        build_log_path, provenance_path,
    ]
    manifest_path = output / "artifact_manifest.json"
    atomic_json(
        manifest_path,
        {
            "schema": "dag-v6.1-artifact-manifest-v1",
            "artifacts": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in artifacts
            ],
        },
    )
    print(json.dumps({
        "status": "PARTIAL_SOURCE_PARAMETERS_FROZEN_TARGET_BINDING_NOT_RUN",
        "output": str(output), "raw_traces": len(raw_inputs),
        "raw_kernels": int(coverage["raw_kernel_count"]),
        "observations": len(observations), "parameters": len(parameters),
        "target_timing_files_read": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

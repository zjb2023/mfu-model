#!/usr/bin/env python3
"""Bind source-only v6.1 compute clocks to v6 and predict the 224-GPU DAG.

Compute timing advances only through the physical slot active-union parameter.
Per-operator cost groups remain attribution metadata.  Compute observed inside a
collective window is emitted as a parallel branch and joins the sealed v5.4
communication completion by max, rather than being appended serially.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v62_cost_bound_prediction_2026w36.toml"
_LOCAL_LAYER_PREFIX = re.compile(r"^L[0-9]+:")
_PP_NODE = re.compile(r"^pp:lane([0-9]+):(fwd|bwd)([0-9]+):s([0-9]+)_to_s([0-9]+)$")
_TAIL_SERVICE = re.compile(
    r"^tail:(dp_with_cp_stage([0-9]+)|expert_dp_stage([0-9]+)_lane([0-9]+)):(rs|ag0|ag1)_service$"
)


EXTRA_COLUMNS = [
    "duration_ns", "compute_exposed_ns_model", "compute_overlap_ns_model",
    "network_service_ns_model", "software_sync_ns_model", "framework_residual_ns_model",
    "timing_component", "timing_source", "semantic_slot", "source_parameter_key",
    "source_v54_node_id", "predicted_start_ns", "predicted_end_ns", "critical_predecessor",
    "on_critical_path",
]


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


def semantic_slot(region: str) -> str:
    if region == "FINAL":
        return "step_exit"
    before = region.startswith("before:")
    clean = region.removeprefix("before:")
    clean = _LOCAL_LAYER_PREFIX.sub("", clean).removesuffix("_wall")
    return f"before_{clean}" if before else clean


def dag_phase(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {"forward": "FWD", "fwd": "FWD", "backward": "BWD", "bwd": "BWD"}
    if normalized not in aliases:
        raise KeyError(value)
    return aliases[normalized]


def parameter_phase(value: str) -> str:
    return {"FWD": "forward", "BWD": "backward"}[dag_phase(value)]


def service_slot(op_name: str, autograd_phase: str) -> str:
    forward = {
        "cp_a2a_query_service": "cp0",
        "cp_a2a_key_service": "cp1",
        "cp_a2a_value_service": "cp2",
        "cp_a2a_output_service": "cp3",
    }
    backward = {
        "cp_a2a_grad_output_0_service": "cp4",
        "cp_a2a_grad_output_1_service": "cp5",
        "cp_a2a_grad_query_service": "cp6",
        "cp_a2a_grad_key_service": "cp7",
        "cp_a2a_grad_value_service": "cp8",
    }
    if op_name in forward:
        return forward[op_name]
    if op_name in backward:
        return backward[op_name]
    if op_name == "ep_dispatch_service":
        return "ep_recompute_dispatch" if autograd_phase == "recompute" else "ep_dispatch"
    if op_name == "ep_combine_service":
        return "ep_recompute_combine" if autograd_phase == "recompute" else "ep_combine"
    if op_name == "ep_combine_backward_service":
        return "ep_combine_backward"
    if op_name == "ep_dispatch_backward_service":
        return "ep_dispatch_backward"
    raise KeyError((op_name, autograd_phase))


def tail_group(group_name: str) -> tuple[str, int]:
    dp = re.fullmatch(r"dp_with_cp_stage([0-9]+)", group_name)
    if dp:
        return "dp_with_cp", int(dp.group(1))
    expert = re.fullmatch(r"expert_dp_stage([0-9]+)_lane([0-9]+)", group_name)
    if expert:
        return "expert_dp", int(expert.group(1)) * 8 + int(expert.group(2))
    raise KeyError(group_name)


def layer_context(
    stage_layers: dict[int, list[int]], stage: int, phase: str, layer_id: int, slot: str
) -> tuple[str, str]:
    layers = stage_layers[stage]
    execution = layers if phase == "FWD" else list(reversed(layers))
    layer_type = "dense" if layer_id == 0 else "moe"
    if slot == "before_cp0":
        position = execution.index(layer_id)
        if position == 0:
            return layer_type, f"step_entry_to_{layer_type}"
        previous = execution[position - 1]
        previous_type = "dense" if previous == 0 else "moe"
        return layer_type, f"{previous_type}_to_{layer_type}"
    if slot == "step_exit":
        return layer_type, f"{layer_type}_to_step_exit"
    return layer_type, layer_type


def max_plus(
    nodes: pd.DataFrame, edges: pd.DataFrame, final_node: str
) -> tuple[list[int], list[int], list[int], set[int]]:
    ids = nodes["node_id"].astype(str).tolist()
    index = {node_id: position for position, node_id in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("duplicate node ids")
    duration = nodes["duration_ns"].astype("int64").tolist()
    indegree = [0] * len(ids)
    outgoing: list[list[int]] = [[] for _ in ids]
    for row in edges.itertuples(index=False):
        try:
            source, target = index[str(row.src)], index[str(row.dst)]
        except KeyError as error:
            raise ValueError(f"edge references missing node: {error}") from error
        outgoing[source].append(target)
        indegree[target] += 1
    queue = deque(position for position, degree in enumerate(indegree) if degree == 0)
    start = [0] * len(ids)
    end = [0] * len(ids)
    predecessor = [-1] * len(ids)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        end[source] = start[source] + duration[source]
        for target in outgoing[source]:
            if end[source] > start[target]:
                start[target] = end[source]
                predecessor[target] = source
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(ids):
        raise ValueError(f"graph cycle detected: visited={visited} nodes={len(ids)}")
    final_index = index[final_node]
    critical: set[int] = set()
    current = final_index
    while current >= 0 and current not in critical:
        critical.add(current)
        current = predecessor[current]
    return start, end, predecessor, critical


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def render_html(
    raw_ms: float,
    reconciled_ms: float,
    training_ms: float,
    mfu: float,
    binding_stats: dict[str, Any],
    critical_components: dict[str, float],
    v54_ms: float,
) -> str:
    component_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{value:.3f}</td><td>{100 * value / max(raw_ms, 1e-9):.1f}%</td></tr>"
        for name, value in sorted(critical_components.items(), key=lambda item: -item[1])
    )
    stats = "".join(
        f"<tr><td>{escape(str(name))}</td><td>{escape(str(value))}</td></tr>"
        for name, value in binding_stats.items()
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DAG v6.2 224-GPU prediction</title>
<style>body{{margin:0;background:#08111e;color:#dce8f5;font:14px/1.55 system-ui}}main{{max-width:1280px;margin:auto;padding:34px}}h1{{margin:0}}h2{{margin-top:34px}}.muted{{color:#91a7bd}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card,.panel{{background:#111e30;border:1px solid #293d56;border-radius:12px;padding:16px}}.value{{font-size:25px;font-weight:700;color:#61d6ff}}.flow{{display:flex;gap:8px;align-items:center;overflow:auto}}.node{{min-width:180px;border-radius:9px;padding:12px;background:#172a42;border:1px solid #345575}}.arrow{{font-size:22px;color:#61d6ff}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #293d56;text-align:left}}code{{color:#9ee8ff}}.warn{{color:#ffd17a}}@media(max-width:850px){{.cards{{grid-template-columns:1fr 1fr}}}}</style></head><body><main>
<h1>DAG v6.2 · 256 卡校准 → 224 卡首次预测</h1>
<p class="muted">未读取 224 卡 profiler/training timing；当前是未评估预测，不是精度或胜负结论。</p>
<section class="cards"><div class="card"><div class="muted">raw graph</div><div class="value">{raw_ms:.3f} ms</div></div><div class="card"><div class="muted">source-reconciled profiler</div><div class="value">{reconciled_ms:.3f} ms</div></div><div class="card"><div class="muted">training clock</div><div class="value">{training_ms:.3f} ms</div></div><div class="card"><div class="muted">predicted MFU</div><div class="value">{mfu:.3f}%</div></div></section>
<h2>时钟如何进入 DAG</h2><section class="flow"><div class="node"><b>exposed gap</b><br><code>max(v61 compute, v54 gap)</code><br>只在 arrival token 计一次</div><span class="arrow">→</span><div class="node"><b>collective fork</b><br>v54 communication completion<br>∥ v61 overlap compute</div><span class="arrow">→</span><div class="node"><b>completion sync</b><br>两条分支取 max</div><span class="arrow">→</span><div class="node"><b>PP / optimizer tail</b><br>继承 sealed v5.4 非计算后端</div></section>
<h2>关键路径组成</h2><div class="panel"><table><thead><tr><th>component</th><th>ms</th><th>raw share</th></tr></thead><tbody>{component_rows}</tbody></table></div>
<h2>绑定审计</h2><div class="panel"><table><tbody>{stats}</tbody></table><p class="warn">v5.4 profiler prediction 为 {v54_ms:.3f} ms，仅作为非计算后端来源与结构参照；v6.2 尚未打开目标实测进行评价。</p></div>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if bool(config["split"]["target_timing_allowed"]):
        raise ValueError("target timing access must remain disabled")
    source_iterations = tuple(int(value) for value in config["split"]["source_calibration"])
    if source_iterations != tuple(range(60, 101, 5)):
        raise ValueError("source calibration split must be exact stable 60--100")
    input_paths = {name: checked(configured_path(value)) for name, value in config["inputs"].items()}
    output = configured_path(config["outputs"]["output_dir"])
    prediction_dir = output / "predictions"
    model_dir = output / "model_inputs"
    logs_dir = output / "logs"
    for directory in (prediction_dir, model_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    v61_access = json.loads(input_paths["v61_input_access_audit"].read_text(encoding="utf-8"))
    v54_access = json.loads(input_paths["v54_input_access_audit"].read_text(encoding="utf-8"))
    v54_seal = json.loads(input_paths["v54_prediction_seal"].read_text(encoding="utf-8"))
    if v61_access.get("target_timing_files_read") != 0:
        raise ValueError("v6.1 input audit is not target-safe")
    if v54_access.get("target_profiler_or_training_timing_read") is not False:
        raise ValueError("v5.4 input audit is not target-safe")
    if v54_seal.get("target_timing_opened") is not False:
        raise ValueError("v5.4 prediction was not sealed before target timing")
    for artifact in v54_seal["artifacts"]:
        path = checked(Path(artifact["path"]))
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"v5.4 seal hash mismatch: {path}")

    nodes = pd.read_csv(input_paths["v6_target_nodes"], compression="gzip")
    edges = pd.read_csv(input_paths["v6_target_edges"], compression="gzip")
    original_columns = nodes.columns.tolist()
    for column in EXTRA_COLUMNS:
        if column in {"timing_component", "timing_source", "semantic_slot", "source_parameter_key", "source_v54_node_id", "critical_predecessor"}:
            nodes[column] = ""
        elif column == "on_critical_path":
            nodes[column] = False
        else:
            nodes[column] = 0
    node_position = {str(node_id): index for index, node_id in enumerate(nodes["node_id"])}
    if len(node_position) != len(nodes):
        raise ValueError("v6 target nodes are not unique")

    layer_map = pd.read_csv(input_paths["target_layer_stage_map"])
    stage_layers = {
        int(stage): [int(value) for value in group.sort_values("stage_local_layer")["layer_id"]]
        for stage, group in layer_map.groupby("pp_stage")
    }
    local_transfer_policy = str(config["binding"].get("local_noncompute_policy", "exact_layer_id"))
    template_layer_lookup = {
        (stage, layer_id): layer_id
        for stage, layers in stage_layers.items()
        for layer_id in layers
    }
    layer_transfer_rows: list[dict[str, Any]] = []
    if local_transfer_policy == "stage_local_normalized_template":
        template_path = input_paths.get("local_template_layer_stage_map")
        if template_path is None:
            raise ValueError("stage-local transfer requires local_template_layer_stage_map")
        template_map = pd.read_csv(template_path)
        template_stage_layers = {
            int(stage): [int(value) for value in group.sort_values("stage_local_layer")["layer_id"]]
            for stage, group in template_map.groupby("pp_stage")
        }
        if set(template_stage_layers) != set(stage_layers):
            raise ValueError("candidate and template PP stage grids differ")
        template_layer_lookup = {}
        for stage, candidate_layers in stage_layers.items():
            templates = template_stage_layers[stage]
            for position, layer_id in enumerate(candidate_layers):
                if len(candidate_layers) == 1:
                    template_position = 0
                else:
                    template_position = round(position * (len(templates) - 1) / (len(candidate_layers) - 1))
                template_layer_id = templates[template_position]
                template_layer_lookup[(stage, layer_id)] = template_layer_id
                layer_transfer_rows.append({
                    "pp_stage": stage, "candidate_layer_id": layer_id,
                    "candidate_stage_local_layer": position,
                    "candidate_stage_layer_count": len(candidate_layers),
                    "template_layer_id": template_layer_id,
                    "template_stage_local_layer": template_position,
                    "template_stage_layer_count": len(templates),
                    "transfer_policy": local_transfer_policy,
                })
    elif local_transfer_policy != "exact_layer_id":
        raise ValueError(f"unsupported local_noncompute_policy: {local_transfer_policy}")
    parameters = pd.read_csv(input_paths["v61_compute_parameters"])
    physical = parameters[
        parameters["parameter_view"].eq("window_split")
        & parameters["cost_group"].eq("__physical_slot_total__")
        & parameters["autograd_phase"].eq("physical_mixed")
    ].copy()
    parameter_keys = ["phase", "layer_type", "layer_context", "execution_scope", "semantic_slot"]
    if physical.duplicated(parameter_keys).any():
        raise ValueError("physical slot parameters are not unique")
    physical_lookup = {
        tuple(getattr(row, column) for column in parameter_keys): int(row.median_active_union_ns)
        for row in physical.itertuples(index=False)
    }
    used_parameter_keys: set[tuple[Any, ...]] = set()

    v54 = pd.read_csv(input_paths["v54_noncompute_nodes"])
    local_gap_lookup: dict[tuple[Any, ...], Any] = {}
    local_completion_lookup: dict[tuple[Any, ...], Any] = {}
    for row in v54.itertuples(index=False):
        if str(row.kind) not in {"local_gap", "cp_rank_completion", "ep_rank_completion"}:
            continue
        slot = semantic_slot(str(row.semantic_region))
        key = (int(row.rank), dag_phase(str(row.phase)), int(row.microbatch), int(row.layer_id), slot)
        target = local_gap_lookup if row.kind == "local_gap" else local_completion_lookup
        if key in target:
            raise ValueError(f"duplicate v5.4 local timing key: {key}")
        target[key] = row

    binding_rows: list[dict[str, Any]] = []
    binding_stats = Counter()

    def assign(node_id: str, **values: Any) -> None:
        position = node_position[node_id]
        for name, value in values.items():
            nodes.at[position, name] = value

    local_services: dict[tuple[Any, ...], dict[str, Any]] = {}
    local_node_lookup = {
        (str(row.node_id).rsplit(":", 2)[0], str(row.op_name)): str(row.node_id)
        for row in nodes.itertuples(index=False)
        if str(row.phase) in {"FWD", "BWD"}
    }
    local_service_rows = nodes[
        nodes["phase"].isin(["FWD", "BWD"])
        & nodes["op_family"].eq("collective_service")
        & nodes["parallelism"].isin(["cp", "ep"])
    ]
    for row in local_service_rows.itertuples(index=False):
        slot = service_slot(str(row.op_name), str(row.autograd_phase))
        subgraph_id = str(row.node_id).rsplit(":", 2)[0]
        arrival_name = str(row.op_name).removesuffix("_service") + "_arrival_join"
        sync_name = str(row.op_name).removesuffix("_service") + "_completion_sync"
        arrival_id = local_node_lookup.get((subgraph_id, arrival_name))
        sync_id = local_node_lookup.get((subgraph_id, sync_name))
        if arrival_id is None or sync_id is None:
            raise ValueError(f"missing local collective boundary for {row.node_id}")
        key = (int(row.rank), str(row.phase), int(row.microbatch), int(row.layer_id), slot)
        if key in local_services:
            raise ValueError(f"duplicate v6 local service key: {key}")
        local_services[key] = {
            "service_id": str(row.node_id), "arrival_id": arrival_id, "sync_id": sync_id,
            "stage": int(row.pp_stage), "layer_type": str(row.layer_type),
            "autograd_phase": str(row.autograd_phase), "slot": slot,
        }

    # Bind sealed CP/EP completion and the exposed local gap before each call.
    missing_completion: list[tuple[Any, ...]] = []
    missing_gap_node: list[tuple[Any, ...]] = []
    for key, service in local_services.items():
        rank, phase, microbatch, layer_id, slot = key
        template_layer_id = template_layer_lookup[(int(service["stage"]), layer_id)]
        template_key = (rank, phase, microbatch, template_layer_id, slot)
        completion = local_completion_lookup.get(template_key)
        if completion is None:
            missing_completion.append(template_key)
            continue
        assign(
            service["service_id"], duration_ns=int(completion.duration_ns),
            network_service_ns_model=int(completion.network_service_ns_model),
            software_sync_ns_model=int(completion.software_sync_ns_model),
            timing_component="local_collective_completion", timing_source="sealed_v54_noncompute",
            semantic_slot=slot, source_v54_node_id=str(completion.node_id),
        )
        assign(service["sync_id"], timing_component="collective_max_join", timing_source="v62_overlap_protocol", semantic_slot=slot)
        binding_stats["local_collective_services"] += 1

        gap_slot = f"before_{slot}"
        gap_key = (rank, phase, microbatch, template_layer_id, gap_slot)
        gap = local_gap_lookup.get(gap_key)
        if gap is None:
            missing_gap_node.append(gap_key)
            continue
        layer_type, context = layer_context(stage_layers, int(service["stage"]), phase, layer_id, gap_slot)
        parameter_key = (parameter_phase(phase), layer_type, context, "exposed_gap", gap_slot)
        compute_ns = int(physical_lookup.get(parameter_key, 0))
        if parameter_key in physical_lookup:
            used_parameter_keys.add(parameter_key)
        gap_ns = int(gap.duration_ns)
        residual_ns = max(gap_ns - compute_ns, 0)
        duration_ns = compute_ns + residual_ns
        assign(
            service["arrival_id"], duration_ns=duration_ns,
            compute_exposed_ns_model=compute_ns, framework_residual_ns_model=residual_ns,
            timing_component="physical_exposed_slot_clock", timing_source="v61_compute_plus_sealed_v54_gap",
            semantic_slot=gap_slot, source_parameter_key="|".join(map(str, parameter_key)),
            source_v54_node_id=str(gap.node_id),
        )
        binding_rows.append(
            {
                "clock_node_id": service["arrival_id"], "rank": rank, "phase": phase,
                "microbatch": microbatch, "layer_id": layer_id, "semantic_slot": gap_slot,
                "template_layer_id": template_layer_id,
                "local_transfer_policy": local_transfer_policy,
                "binding_kind": "exposed_gap_single_clock", "compute_ns": compute_ns,
                "communication_ns": 0, "framework_residual_ns": residual_ns,
                "duration_ns": duration_ns, "source_parameter_key": "|".join(map(str, parameter_key)),
                "source_v54_node_id": str(gap.node_id),
            }
        )
        binding_stats["exposed_gap_clocks"] += 1
        if template_layer_id != layer_id:
            binding_stats["local_template_transfers"] += 1

    if missing_completion or missing_gap_node:
        raise ValueError(
            f"local noncompute mapping incomplete: completion={missing_completion[:3]} gap={missing_gap_node[:3]}"
        )

    # Add one physical compute branch per collective slot.  Cost groups are
    # attribution only; this branch advances the union clock exactly once.
    new_nodes: list[dict[str, Any]] = []
    new_edges: list[dict[str, Any]] = []
    for key, service in local_services.items():
        rank, phase, microbatch, layer_id, slot = key
        layer_type, context = layer_context(stage_layers, int(service["stage"]), phase, layer_id, slot)
        parameter_key = (parameter_phase(phase), layer_type, context, "overlap_with_collective", slot)
        overlap_ns = int(physical_lookup.get(parameter_key, 0))
        if not overlap_ns:
            continue
        used_parameter_keys.add(parameter_key)
        service_position = node_position[service["service_id"]]
        template = nodes.iloc[service_position].to_dict()
        overlap_id = service["service_id"] + "::physical_compute_overlap"
        template.update(
            {
                "node_id": overlap_id, "kind": "compute_overlap_token", "op_name": "physical_slot_compute_overlap",
                "op_family": "compute_overlap_active_union", "stream": "compute", "resource": "gpu_compute",
                "parallelism": "local", "duration_ns": overlap_ns, "compute_exposed_ns_model": 0,
                "compute_overlap_ns_model": overlap_ns, "network_service_ns_model": 0,
                "software_sync_ns_model": 0, "framework_residual_ns_model": 0,
                "timing_component": "physical_overlap_slot_clock", "timing_source": "v61_physical_slot_active_union",
                "semantic_slot": slot, "source_parameter_key": "|".join(map(str, parameter_key)),
                "source_v54_node_id": "", "predicted_start_ns": 0, "predicted_end_ns": 0,
                "critical_predecessor": "", "on_critical_path": False,
            }
        )
        new_nodes.append(template)
        new_edges.extend(
            [
                {
                    "case_id": template["case_id"], "src": service["arrival_id"], "dst": overlap_id,
                    "edge_type": "compute_overlap_branch", "tensor_key": "physical_slot_active_union",
                    "dependency_source": "v61_trace_overlap",
                },
                {
                    "case_id": template["case_id"], "src": overlap_id, "dst": service["sync_id"],
                    "edge_type": "compute_overlap_branch_join", "tensor_key": "physical_slot_active_union",
                    "dependency_source": "v62_max_join",
                },
            ]
        )
        binding_rows.append(
            {
                "clock_node_id": overlap_id, "rank": rank, "phase": phase,
                "microbatch": microbatch, "layer_id": layer_id, "semantic_slot": slot,
                "binding_kind": "collective_parallel_compute_branch", "compute_ns": overlap_ns,
                "communication_ns": int(nodes.at[service_position, "duration_ns"]),
                "framework_residual_ns": 0, "duration_ns": overlap_ns,
                "source_parameter_key": "|".join(map(str, parameter_key)),
                "source_v54_node_id": str(nodes.at[service_position, "source_v54_node_id"]),
            }
        )
        binding_stats["collective_overlap_branches"] += 1

    # The final post-collective compute has no following arrival token.  Charge
    # it exactly once on the phase end boundary.
    phase_ends = {
        (int(row.rank), str(row.phase), int(row.microbatch)): str(row.node_id)
        for row in nodes[nodes["op_name"].isin(["fwd_end", "bwd_end"])].itertuples(index=False)
    }
    pp = int(config["target"]["pp"])
    lanes = int(config["target"]["pp_lanes"])
    microbatches = int(config["target"]["microbatches"])
    for rank in range(pp * lanes):
        stage = rank // lanes
        for phase in ("FWD", "BWD"):
            execution = stage_layers[stage] if phase == "FWD" else list(reversed(stage_layers[stage]))
            layer_id = execution[-1]
            layer_type, context = layer_context(stage_layers, stage, phase, layer_id, "step_exit")
            parameter_key = (parameter_phase(phase), layer_type, context, "exposed_gap", "step_exit")
            compute_ns = int(physical_lookup.get(parameter_key, 0))
            if parameter_key in physical_lookup:
                used_parameter_keys.add(parameter_key)
            for microbatch in range(microbatches):
                end_id = phase_ends[(rank, phase, microbatch)]
                assign(
                    end_id, duration_ns=compute_ns, compute_exposed_ns_model=compute_ns,
                    timing_component="physical_step_exit_clock", timing_source="v61_physical_slot_active_union",
                    semantic_slot="step_exit", source_parameter_key="|".join(map(str, parameter_key)),
                )
                binding_rows.append(
                    {
                        "clock_node_id": end_id, "rank": rank, "phase": phase,
                        "microbatch": microbatch, "layer_id": layer_id, "semantic_slot": "step_exit",
                        "binding_kind": "step_exit_single_clock", "compute_ns": compute_ns,
                        "communication_ns": 0, "framework_residual_ns": 0, "duration_ns": compute_ns,
                        "source_parameter_key": "|".join(map(str, parameter_key)), "source_v54_node_id": "",
                    }
                )
                binding_stats["step_exit_clocks"] += 1

    # Point-to-point PP wall from the sealed trace-calibrated backend.
    v54_by_id = {str(row.node_id): row for row in v54.itertuples(index=False)}
    for row in nodes[nodes["kind"].eq("pp_p2p")].itertuples(index=False):
        match = _PP_NODE.fullmatch(str(row.node_id))
        if not match:
            raise ValueError(f"unexpected PP node id: {row.node_id}")
        lane, direction, microbatch, source_stage, target_stage = match.groups()
        v54_id = f"lane{lane}:{direction[0].upper()}{microbatch}:s{source_stage}->s{target_stage}"
        timing = v54_by_id[v54_id]
        assign(
            str(row.node_id), duration_ns=int(timing.duration_ns),
            network_service_ns_model=int(timing.network_service_ns_model),
            software_sync_ns_model=int(timing.software_sync_ns_model),
            timing_component="pp_trace_wall", timing_source="sealed_v54_pp",
            source_v54_node_id=v54_id,
        )
        binding_stats["pp_nodes"] += 1

    # Tail collective services and AG0->AG1 software gaps.
    tail_group_meta: dict[str, tuple[str, int]] = {}
    for row in nodes[
        nodes["phase"].eq("OPT") & nodes["kind"].eq("collective_service")
    ].itertuples(index=False):
        match = _TAIL_SERVICE.fullmatch(str(row.node_id))
        if not match:
            raise ValueError(f"unexpected tail service id: {row.node_id}")
        group_name, _, _, _, round_name = match.groups()
        group_type, group_index = tail_group(group_name)
        v54_id = f"{group_type}:{round_name}:g{group_index}"
        timing = v54_by_id[v54_id]
        assign(
            str(row.node_id), duration_ns=int(timing.duration_ns),
            network_service_ns_model=int(timing.network_service_ns_model),
            software_sync_ns_model=int(timing.software_sync_ns_model),
            timing_component=f"tail_{round_name}_completion", timing_source="sealed_v54_tail_collective",
            source_v54_node_id=v54_id,
        )
        tail_group_meta[group_name] = (group_type, group_index)
        if round_name == "ag1":
            arrival_id = str(row.node_id).removesuffix("_service") + "_arrival_join"
            gap_id = f"{group_type}:ag_gap:g{group_index}"
            gap = v54_by_id[gap_id]
            assign(
                arrival_id, duration_ns=int(gap.duration_ns),
                software_sync_ns_model=int(gap.software_sync_ns_model),
                timing_component="tail_ag0_to_ag1_software", timing_source="sealed_v54_tail_gap",
                source_v54_node_id=gap_id,
            )
        binding_stats["tail_collective_services"] += 1

    # Preserve rank-specific RS release skew by inserting a delay node on each
    # rank->group arrival edge, instead of replacing it with one group average.
    release_lookup = {
        (int(row.rank), "dp_with_cp" if ":dp_with_cp:" in str(row.node_id) else "expert_dp"): row
        for row in v54[v54["kind"].eq("collective_release")].itertuples(index=False)
    }
    nodes_by_id = {str(row.node_id): row for row in nodes.itertuples(index=False)}
    retained_edges: list[dict[str, Any]] = []
    release_nodes: list[dict[str, Any]] = []
    release_edges: list[dict[str, Any]] = []
    for edge in edges.to_dict("records"):
        dst = str(edge["dst"])
        if edge["edge_type"] != "collective_rank_arrival" or not dst.endswith(":rs_arrival_join"):
            retained_edges.append(edge)
            continue
        group_name = dst.split(":")[1]
        group_type, _ = tail_group_meta[group_name]
        source = nodes_by_id[str(edge["src"])]
        rank = int(source.rank)
        timing = release_lookup.get((rank, group_type))
        delay = int(timing.duration_ns) if timing is not None else 0
        arrival_position = node_position[dst]
        template = nodes.iloc[arrival_position].to_dict()
        release_id = f"{dst}::release:r{rank}"
        template.update(
            {
                "node_id": release_id, "kind": "collective_release_token", "rank": rank,
                "pp_stage": int(source.pp_stage), "pp_lane": int(source.pp_lane),
                "op_name": "rank_collective_release_delay", "op_family": "software_sync",
                "duration_ns": delay, "compute_exposed_ns_model": 0, "compute_overlap_ns_model": 0,
                "network_service_ns_model": 0, "software_sync_ns_model": delay,
                "framework_residual_ns_model": 0, "timing_component": "tail_rank_release_delay",
                "timing_source": "sealed_v54_rank_release", "semantic_slot": "rs_release",
                "source_parameter_key": "", "source_v54_node_id": str(timing.node_id) if timing else "zero_delay",
                "predicted_start_ns": 0, "predicted_end_ns": 0, "critical_predecessor": "",
                "on_critical_path": False,
            }
        )
        release_nodes.append(template)
        release_edges.extend(
            [
                {**edge, "dst": release_id, "edge_type": "collective_rank_release_start"},
                {**edge, "src": release_id, "edge_type": "collective_rank_release_complete"},
            ]
        )
        binding_stats["tail_rank_release_nodes"] += 1

    # Per-rank optimizer compute.
    optimizer_lookup = {
        int(row.rank): row for row in v54[v54["kind"].eq("optimizer_compute")].itertuples(index=False)
    }
    for row in nodes[nodes["op_family"].eq("optimizer_compute")].itertuples(index=False):
        timing = optimizer_lookup[int(row.rank)]
        assign(
            str(row.node_id), duration_ns=int(timing.duration_ns),
            compute_exposed_ns_model=int(timing.compute_work_ns),
            timing_component="optimizer_compute", timing_source="sealed_v54_optimizer",
            source_v54_node_id=str(timing.node_id),
        )
        binding_stats["optimizer_nodes"] += 1

    # All remaining GPU operators are semantic structure.  Their time has been
    # charged once on a physical slot token.
    structural = nodes["resource"].eq("gpu_compute") & nodes["timing_component"].eq("")
    nodes.loc[structural, "timing_component"] = "structural_operator_zero_cost"
    nodes.loc[structural, "timing_source"] = "cost_carried_by_physical_slot_token"
    zero_other = nodes["timing_component"].eq("")
    nodes.loc[zero_other, "timing_component"] = "zero_duration_dependency"
    nodes.loc[zero_other, "timing_source"] = "v6_structure"

    if new_nodes or release_nodes:
        nodes = pd.concat([nodes, pd.DataFrame([*new_nodes, *release_nodes])], ignore_index=True)
    edges = pd.DataFrame([*retained_edges, *new_edges, *release_edges], columns=edges.columns)
    final_node = "iteration:completion_join"
    start, end, predecessor, critical = max_plus(nodes, edges, final_node)
    ids = nodes["node_id"].astype(str).tolist()
    nodes["predicted_start_ns"] = start
    nodes["predicted_end_ns"] = end
    nodes["critical_predecessor"] = [ids[value] if value >= 0 else "" for value in predecessor]
    nodes["on_critical_path"] = [index in critical for index in range(len(nodes))]
    raw_ns = int(end[ids.index(final_node)])
    raw_ms = raw_ns / 1e6

    v54_parameters = json.loads(input_paths["v54_calibration_parameters"].read_text(encoding="utf-8"))
    reconciliation_ms = float(v54_parameters["target"]["reconciliation_ms"])
    outer_ms = float(v54_parameters["target"]["outer_framework_ms"])
    v54_profiler_ms = float(v54_parameters["target"]["dag_profiler_ms"])
    reconciled_ms = raw_ms + reconciliation_ms
    training_ms = reconciled_ms + outer_ms
    world = int(config["target"]["world_size"])
    peak = float(config["target"]["peak_tflops_per_gpu"]) * 1e12
    model_flops = float(config["target"]["model_flops_per_iteration"])
    predicted_mfu = 100.0 * model_flops / ((training_ms / 1000.0) * world * peak)

    critical_frame = nodes[nodes["on_critical_path"]].copy().sort_values("predicted_start_ns")
    component_columns = {
        "compute_exposed": "compute_exposed_ns_model",
        "compute_overlap": "compute_overlap_ns_model",
        "network_service": "network_service_ns_model",
        "software_sync": "software_sync_ns_model",
        "framework_residual": "framework_residual_ns_model",
    }
    critical_components = {
        name: float(critical_frame[column].sum()) / 1e6 for name, column in component_columns.items()
    }
    unexplained_ms = raw_ms - sum(critical_components.values())
    critical_components["unexplained_or_zero_boundary"] = max(unexplained_ms, 0.0)

    node_path = prediction_dir / "dag_v62_nodes.csv.gz"
    edge_path = prediction_dir / "dag_v62_edges.csv.gz"
    critical_path = prediction_dir / "dag_v62_critical_path.csv"
    binding_path = model_dir / "cost_binding_table.csv.gz"
    layer_transfer_path = model_dir / "local_noncompute_layer_transfer.csv"
    nodes.to_csv(node_path, index=False, compression="gzip")
    edges.to_csv(edge_path, index=False, compression="gzip")
    critical_frame.to_csv(critical_path, index=False)
    pd.DataFrame(binding_rows).to_csv(binding_path, index=False, compression="gzip")
    pd.DataFrame(layer_transfer_rows).to_csv(layer_transfer_path, index=False)

    iterations = [*config["split"]["target_regression_only"], *config["split"]["target_validation"]]
    prediction_rows = []
    for iteration in iterations:
        split = "regression_only" if int(iteration) == 55 else "validation"
        prediction_rows.append(
            {
                "protocol_id": config["protocol"]["id"], "source_case": "256gpu_pp16_cp2_a2a",
                "target_case": config["target"]["case_id"], "method": "DAG v6.2 physical-slot operator graph",
                "method_version": "dag-v6.2-source60-100-v1", "status": "PREDICTIVE_PARTIAL_UNEVALUATED",
                "split": split, "iteration": int(iteration), "predicted_raw_graph_ms": raw_ms,
                "predicted_source_reconciliation_ms": reconciliation_ms,
                "predicted_profiler_step_ms": reconciled_ms,
                "predicted_outer_framework_ms": outer_ms,
                "predicted_training_step_ms": training_ms, "predicted_mfu_pct": predicted_mfu,
                "target_timing_read": False,
            }
        )
    method_path = prediction_dir / "method_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(method_path, index=False)

    unused_parameters = sorted(set(physical_lookup) - used_parameter_keys)
    binding_stats_payload = {
        **{name: int(value) for name, value in binding_stats.items()},
        "v6_base_nodes": int(len(nodes) - len(new_nodes) - len(release_nodes)),
        "v62_nodes": int(len(nodes)), "v62_edges": int(len(edges)),
        "critical_path_nodes": int(len(critical)),
        "physical_parameter_count": int(len(physical_lookup)),
        "used_physical_parameter_count": int(len(used_parameter_keys)),
        "unused_physical_parameter_keys": [list(key) for key in unused_parameters],
        "target_timing_files_read": 0,
    }
    contract_path = output / "prediction_contract.json"
    atomic_json(
        contract_path,
        {
            "schema": "dag-v6.2-cost-bound-prediction-v1",
            "status": "PREDICTIVE_PARTIAL_UNEVALUATED",
            "clock_policy": config["binding"],
            "prediction": {
                "raw_graph_ms": raw_ms, "source_reconciliation_ms": reconciliation_ms,
                "profiler_step_ms": reconciled_ms, "outer_framework_ms": outer_ms,
                "training_step_ms": training_ms, "mfu_pct": predicted_mfu,
            },
            "critical_path_components_ms": critical_components,
            "binding_stats": binding_stats_payload,
            "local_noncompute_transfer": {
                "policy": local_transfer_policy,
                "mapping_rows": len(layer_transfer_rows),
                "mapping_path": str(layer_transfer_path),
                "uses_target_timing": False,
            },
            "accuracy_claim_allowed": False,
            "target_timing_read": False,
        },
    )

    access_path = output / "input_access_audit.json"
    atomic_json(
        access_path,
        {
            "schema": "dag-v6.2-input-access-audit-v1",
            "status": "PASS_NO_TARGET_TIMING",
            "source_iterations": list(source_iterations),
            "inputs": [
                {"path": str(path), "role": role, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for role, path in input_paths.items()
            ],
            "model_process_read_fields": {
                "v61_compute_parameters": parameter_keys + ["cost_group", "median_active_union_ns"],
                "v54_noncompute_nodes": [
                    "node_id", "kind", "rank", "pp_stage", "phase", "microbatch", "duration_ns",
                    "network_service_ns_model", "software_sync_ns_model", "compute_work_ns",
                    "semantic_region", "layer_id",
                ],
                "v6_target_graph": ["static dependency and operator metadata only"],
                "local_template_layer_stage_map": [
                    "layer_id", "pp_stage", "stage_local_layer", "stage_layer_count"
                ] if "local_template_layer_stage_map" in input_paths else [],
            },
            "forbidden_target_profiler_training_tflops_error_files": [],
            "target_timing_files_read": 0,
            "ns3_invocations": 0,
        },
    )

    report_path = output / "DAG_V62_COST_BOUND_PREDICTION_REPORT.md"
    atomic_text(
        report_path,
        f"""# DAG v6.2：256 卡校准到 224 卡首次预测

状态：`PREDICTIVE_PARTIAL_UNEVALUATED`。

本阶段把 v6.1 的物理 slot 计算参数放回 v6 operator DAG，并生成未读取 224 卡实测时间的首次预测。

## 预测

- raw graph：{raw_ms:.6f} ms
- source reconciliation：{reconciliation_ms:.6f} ms
- profiler clock：{reconciled_ms:.6f} ms
- outer framework：{outer_ms:.6f} ms
- training clock：{training_ms:.6f} ms
- predicted MFU：{predicted_mfu:.6f}%

## 计时规则

1. 每个 exposed gap 只使用一次 `__physical_slot_total__`；算子 cost group 只解释归属，不再求和。
2. gap 时钟为 `max(v6.1 exposed compute, sealed v5.4 local gap)`，差额记为 framework residual。
3. 通信窗口内计算形成独立分支，与 CP/EP completion 在 sync 节点取 max。
4. PP、DP/EDP RS、rank release、optimizer、AG0/gap/AG1 使用 target-safe、预先 sealed 的 v5.4 非计算后端；本阶段没有运行 ns-3。
5. 同时输出 raw graph 与 source-reconciled 时钟，未隐藏 reconciliation。

## 关键路径组成

{json.dumps(critical_components, ensure_ascii=False, indent=2)}

## 当前限制

- v6.1 计算校准仍是 lane0 覆盖全部 PP stage，尚未做 lane1–15 方差校验；
- 非计算后端继承 v5.4，因此这是 v6 计算图替换后的首次预测，不是全新独立通信校准；
- 未读取目标实测，不给出误差或方法胜负。
""",
    )
    html_path = output / "dag_v62_cost_bound_prediction.html"
    atomic_text(
        html_path,
        render_html(raw_ms, reconciled_ms, training_ms, predicted_mfu, binding_stats_payload, critical_components, v54_profiler_ms),
    )
    reproduction_path = output / "reproduction_command.txt"
    atomic_text(
        reproduction_path,
        f"cd {REPO}\npython {Path(__file__).resolve()} --config {config_path}\n"
        f"pytest -q case_224gpu_pp14_cp2_a2a/tests/test_dag_v62_cost_bound_prediction.py "
        f"--junitxml={logs_dir / 'pytest.xml'}\n",
    )
    atomic_text(
        logs_dir / "build.log",
        "\n".join(
            [
                "DAG v6.2 cost-bound prediction PASS", f"generated_at={datetime.now(timezone.utc).isoformat()}",
                f"raw_graph_ms={raw_ms:.9f}", f"profiler_step_ms={reconciled_ms:.9f}",
                f"training_step_ms={training_ms:.9f}", f"predicted_mfu_pct={predicted_mfu:.9f}",
                "target_timing_files_read=0", "status=PREDICTIVE_PARTIAL_UNEVALUATED",
            ]
        ) + "\n",
    )

    provenance_path = output / "provenance.json"
    code_paths = [config_path, Path(__file__).resolve()]
    atomic_json(
        provenance_path,
        {
            "schema": "dag-v6.2-provenance-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": str(REPO), "git_branch": git_output("branch", "--show-current"),
            "git_head": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--short").splitlines(),
            "version_state": "PARTIAL_UNCOMMITTED_WORKTREE_PINNED_BY_SHA256",
            "code_and_config": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in code_paths
            ],
            "input_access_audit": str(access_path), "v54_prediction_seal": str(input_paths["v54_prediction_seal"]),
        },
    )
    artifacts = [
        node_path, edge_path, critical_path, binding_path, method_path, contract_path,
        access_path, report_path, html_path, reproduction_path, logs_dir / "build.log", provenance_path,
    ]
    atomic_json(
        output / "artifact_manifest.json",
        {
            "schema": "dag-v6.2-artifact-manifest-v1",
            "artifacts": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in artifacts
            ],
        },
    )
    print(json.dumps({
        "status": "PREDICTIVE_PARTIAL_UNEVALUATED", "output": str(output),
        "raw_graph_ms": raw_ms, "profiler_step_ms": reconciled_ms,
        "training_step_ms": training_ms, "predicted_mfu_pct": predicted_mfu,
        "target_timing_files_read": 0, "binding_stats": binding_stats_payload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

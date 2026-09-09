#!/usr/bin/env python3
"""Correct DP/Expert-DP optimizer-tail order without opening target timing.

The frozen v6.2 graph already contains the target static graph and source-only
cost transfer.  v6.3 changes only the optimizer-tail dependency semantics:

DP RS -> Expert-DP RS -> world RS done -> optimizer ->
DP AG0 -> Expert-DP AG0 -> DP AG1 -> Expert-DP AG1.

Source 256-GPU stable rank events provide only the nonnegative program gaps.
The target 224-GPU trace/timing is deliberately unavailable to this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v63_ordered_optimizer_tail_2026w36.toml"
PAIR_SPECS = (
    ("dp_rs_to_expert_dp_rs", "dp_grad_reduce_scatter", 0, "expert_dp_grad_reduce_scatter", 0),
    ("dp_ag0_to_expert_dp_ag0", "dp_param_allgather", 0, "expert_dp_param_allgather", 0),
    ("expert_dp_ag0_to_dp_ag1", "expert_dp_param_allgather", 0, "dp_param_allgather", 1),
    ("dp_ag1_to_expert_dp_ag1", "dp_param_allgather", 1, "expert_dp_param_allgather", 1),
)


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


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def verify_v62_seal(seal_path: Path) -> dict[str, Any]:
    seal = json.loads(checked(seal_path).read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_BEFORE_V62_EVALUATOR_ACCESS":
        raise ValueError("v6.2 source prediction is not sealed")
    for artifact in seal["artifacts"]:
        path = checked(Path(artifact["path"]))
        if sha256(path) != artifact["sha256"] or path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"v6.2 sealed artifact changed: {path}")
    return seal


def source_role(stage: int, first: int, last: int) -> str:
    return "first" if stage == first else "last" if stage == last else "internal"


def calibrate_program_gaps(
    events: pd.DataFrame, source_iterations: tuple[int, ...], first_stage: int, last_stage: int
) -> dict[str, Any]:
    observed_iterations = tuple(sorted(events["iteration"].astype(int).unique()))
    if observed_iterations != source_iterations:
        raise ValueError(f"source iteration grid changed: {observed_iterations}")
    events = events.copy()
    events["pp_role"] = events["pp_stage"].astype(int).map(
        lambda stage: source_role(stage, first_stage, last_stage)
    )
    keys = ["iteration", "rank"]
    output: dict[str, Any] = {}
    for name, left_behavior, left_round, right_behavior, right_round in PAIR_SPECS:
        left = events[
            events["behavior"].eq(left_behavior) & events["call_round"].eq(left_round)
        ][keys + ["pp_role", "start_ns", "end_ns"]]
        right = events[
            events["behavior"].eq(right_behavior) & events["call_round"].eq(right_round)
        ][keys + ["pp_role", "start_ns", "end_ns"]]
        joined = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
        if not joined["pp_role_left"].eq(joined["pp_role_right"]).all():
            raise ValueError(f"PP role mismatch for {name}")
        joined["gap_ns"] = joined["start_ns_right"].astype("int64") - joined["end_ns_left"].astype("int64")
        roles: dict[str, Any] = {}
        for role in ("first", "internal", "last"):
            values = joined.loc[joined["pp_role_left"].eq(role), "gap_ns"].astype("int64")
            valid = values[values.ge(0)]
            if valid.empty:
                raise ValueError(f"no nonnegative source gaps for {name}/{role}")
            roles[role] = {
                "median_nonnegative_ns": int(valid.median()),
                "pair_count": int(len(values)),
                "nonnegative_count": int(len(valid)),
                "negative_excluded_count": int(values.lt(0).sum()),
                "minimum_nonnegative_ns": int(valid.min()),
                "maximum_nonnegative_ns": int(valid.max()),
            }
        output[name] = {
            "left": {"behavior": left_behavior, "round": left_round, "timestamp": "end_ns"},
            "right": {"behavior": right_behavior, "round": right_round, "timestamp": "start_ns"},
            "roles": roles,
        }
    return output


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
        if str(row.src) not in index or str(row.dst) not in index:
            raise ValueError(f"edge references missing node: {row.src} -> {row.dst}")
        source, target = index[str(row.src)], index[str(row.dst)]
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
        raise ValueError(f"graph cycle detected: visited={visited}, nodes={len(ids)}")
    final = index[final_node]
    critical: set[int] = set()
    current = final
    while current >= 0 and current not in critical:
        critical.add(current)
        current = predecessor[current]
    return start, end, predecessor, critical


def role_gap(parameters: dict[str, Any], name: str, role: str) -> int:
    return int(parameters[name]["roles"][role]["median_nonnegative_ns"])


def target_role(stage: int, last_stage: int) -> str:
    return "first" if stage == 0 else "last" if stage == last_stage else "internal"


def set_software_duration(
    nodes: pd.DataFrame, node_index: dict[str, int], node_id: str, duration_ns: int,
    component: str,
) -> None:
    row = node_index[node_id]
    nodes.loc[row, "duration_ns"] = int(duration_ns)
    for column in (
        "compute_exposed_ns_model", "compute_overlap_ns_model", "network_service_ns_model",
        "framework_residual_ns_model",
    ):
        nodes.loc[row, column] = 0
    nodes.loc[row, "software_sync_ns_model"] = int(duration_ns)
    nodes.loc[row, "timing_component"] = component
    nodes.loc[row, "timing_source"] = "source256_rank_program_gap_median_nonnegative"
    nodes.loc[row, "source_parameter_key"] = component


def edge_row(columns: list[str], src: str, dst: str, edge_type: str, tensor_key: str) -> dict[str, Any]:
    row = {column: "" for column in columns}
    row.update({
        "case_id": "224gpu_pp14_cp2_a2a", "src": src, "dst": dst,
        "edge_type": edge_type, "tensor_key": tensor_key,
        "dependency_source": "distributed_optimizer_v63_ordered_tail",
    })
    return row


def order_audit(nodes: pd.DataFrame, pp: int) -> dict[str, Any]:
    lookup = nodes.set_index("node_id")
    checks: list[dict[str, Any]] = []
    world_end = int(lookup.loc["tail:world_rs_done", "predicted_end_ns"])
    optimizer_min = int(nodes[nodes["node_id"].str.match(r"^tail:r[0-9]+:optimizer_start$")]["predicted_start_ns"].min())
    checks.append({
        "name": "world_rs_done_before_every_optimizer", "minimum_gap_ns": optimizer_min - world_end,
        "pass": optimizer_min >= world_end,
    })
    for stage in range(pp):
        dp_rs_end = int(lookup.loc[f"tail:dp_with_cp_stage{stage}:rs_completion_sync", "predicted_end_ns"])
        edp_rs_start = int(nodes[
            nodes["node_id"].str.match(fr"^tail:expert_dp_stage{stage}_lane[0-9]+:rs_service$")
        ]["predicted_start_ns"].min())
        dp_ag0_end = int(lookup.loc[f"tail:dp_with_cp_stage{stage}:ag0_completion_sync", "predicted_end_ns"])
        edp_ag0 = nodes[nodes["node_id"].str.match(
            fr"^tail:expert_dp_stage{stage}_lane[0-9]+:ag0_(service|completion_sync)$"
        )]
        edp_ag0_start = int(edp_ag0[edp_ag0["node_id"].str.endswith("_service")]["predicted_start_ns"].min())
        edp_ag0_end = int(edp_ag0[edp_ag0["node_id"].str.endswith("_completion_sync")]["predicted_end_ns"].max())
        dp_ag1_start = int(lookup.loc[f"tail:dp_with_cp_stage{stage}:ag1_service", "predicted_start_ns"])
        dp_ag1_end = int(lookup.loc[f"tail:dp_with_cp_stage{stage}:ag1_completion_sync", "predicted_end_ns"])
        edp_ag1_start = int(nodes[
            nodes["node_id"].str.match(fr"^tail:expert_dp_stage{stage}_lane[0-9]+:ag1_service$")
        ]["predicted_start_ns"].min())
        for name, gap in (
            ("dp_rs_before_expert_dp_rs", edp_rs_start - dp_rs_end),
            ("dp_ag0_before_expert_dp_ag0", edp_ag0_start - dp_ag0_end),
            ("all_expert_dp_ag0_before_dp_ag1", dp_ag1_start - edp_ag0_end),
            ("dp_ag1_before_expert_dp_ag1", edp_ag1_start - dp_ag1_end),
        ):
            checks.append({"name": name, "pp_stage": stage, "minimum_gap_ns": gap, "pass": gap >= 0})
    failures = [item for item in checks if not item["pass"]]
    return {
        "schema": "dag-v6.3-optimizer-tail-order-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "expected_rank_program": [
            "DP RS", "Expert-DP RS", "world RS done", "optimizer",
            "DP AG0", "Expert-DP AG0", "DP AG1", "Expert-DP AG1",
        ],
        "checks": checks,
        "failure_count": len(failures),
        "limitation": (
            "v6.2 carries group completion but not per-rank network completion offsets; "
            "v6.3 conservatively lowers rank dependencies to group completion."
        ),
    }


def render_html(contract: dict[str, Any], audit: dict[str, Any], params: dict[str, Any]) -> str:
    prediction = contract["prediction"]
    rows = "".join(
        f"<tr><td>{name}</td><td>{role}</td><td>{value['median_nonnegative_ns']/1e6:.3f}</td>"
        f"<td>{value['negative_excluded_count']}/{value['pair_count']}</td></tr>"
        for name, payload in params["program_gaps"].items()
        for role, value in payload["roles"].items()
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAG v6.3 · optimizer tail order</title><style>body{{margin:0;background:#07111d;color:#e9f2fb;font:14px/1.55 system-ui}}main{{max-width:1180px;margin:auto;padding:32px}}.muted{{color:#93a8bd}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.card,.panel{{background:#0e1b2c;border:1px solid #2b4059;padding:15px;margin:12px 0}}.v{{font-size:25px;font-weight:800;color:#58d6c4}}.chain{{display:flex;gap:8px;align-items:center;overflow:auto;padding:18px 0}}.n{{white-space:nowrap;background:#142840;border:1px solid #3c5c78;padding:10px}}.a{{color:#ffd166;font-size:20px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #2b4059}}.warn{{border-left:4px solid #f59e0b;padding:10px;background:#1b1d28}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}}}</style></head><body><main>
<p class="muted">DAG v6.3 · source-only dependency correction</p><h1>DP / Expert-DP optimizer tail 顺序修正</h1>
<div class="cards"><div class="card"><div class="muted">raw graph</div><div class="v">{prediction['raw_graph_ms']:.3f} ms</div></div><div class="card"><div class="muted">training clock</div><div class="v">{prediction['training_step_ms']:.3f} ms</div></div><div class="card"><div class="muted">顺序审计</div><div class="v">{audit['status']}</div></div></div>
<section class="panel"><h2>模型中的真实程序顺序</h2><div class="chain">{''.join(f'<span class="n">{name}</span>' + ('<span class="a">→</span>' if i < 7 else '') for i,name in enumerate(audit['expected_rank_program']))}</div><p>RS 阶段先 DP、后 Expert-DP，全部 RS 完成后进入 optimizer；AG 阶段同样不是并发，而是 DP AG0、Expert-DP AG0、DP AG1、Expert-DP AG1。</p></section>
<section class="panel"><h2>256 卡稳定段提取的软件间隔</h2><table><thead><tr><th>相邻步骤</th><th>PP 位置</th><th>median ms</th><th>排除负值/样本</th></tr></thead><tbody>{rows}</tbody></table></section>
<p class="warn">当前 FCT 仍沿用已冻结 v6.2 后端；由于后端没有逐 rank completion offset，rank 顺序暂按 group completion 保守下沉。该版本已修正错误的并发关系，但仍标记 PARTIAL，不能把这一近似当作最终逐-rank 网络模型。</p>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    v62_run = configured_path(config["inputs"]["v62_run_dir"])
    output = configured_path(config["outputs"]["output_dir"])
    prediction_dir = output / "predictions"
    calibration_dir = output / "calibration"
    logs_dir = output / "logs"
    for directory in (prediction_dir, calibration_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seal_path = checked(v62_run / "predictions/prediction_seal.json")
    verify_v62_seal(seal_path)
    node_input = checked(v62_run / "predictions/dag_v62_nodes.csv.gz")
    edge_input = checked(v62_run / "predictions/dag_v62_edges.csv.gz")
    contract_input = checked(v62_run / "prediction_contract.json")
    source_events_path = checked(configured_path(config["inputs"]["source_rank_events"]))
    nodes = pd.read_csv(node_input, low_memory=False)
    edges = pd.read_csv(edge_input, low_memory=False)
    events = pd.read_csv(source_events_path)
    source_iterations = tuple(int(value) for value in config["split"]["source_calibration"])
    gaps = calibrate_program_gaps(
        events, source_iterations,
        int(config["ordering"]["source_first_stage"]),
        int(config["ordering"]["source_last_stage"]),
    )
    calibration = {
        "schema": "dag-v6.3-source-program-gap-calibration-v1",
        "status": "FROZEN_SOURCE_ONLY",
        "source_case": "256gpu_pp16_cp2_a2a",
        "source_iterations": list(source_iterations),
        "statistic": config["ordering"]["gap_statistic"],
        "program_gaps": gaps,
        "target_timing_read": False,
    }
    calibration_path = calibration_dir / "optimizer_tail_program_gaps.json"
    atomic_json(calibration_path, calibration)

    pp = int(config["target"]["pp"])
    lanes = int(config["target"]["pp_lanes"])
    last_stage = int(config["ordering"]["target_last_stage"])
    node_index = {str(node_id): position for position, node_id in enumerate(nodes["node_id"])}
    additions: list[dict[str, Any]] = []
    remove = pd.Series(False, index=edges.index)

    # DP RS -> Expert-DP RS.  Replace the old BWD-direct Expert-DP releases.
    for stage in range(pp):
        role = target_role(stage, last_stage)
        duration = role_gap(gaps, "dp_rs_to_expert_dp_rs", role)
        dp_done = f"tail:dp_with_cp_stage{stage}:rs_completion_sync"
        for lane in range(8):
            prefix = f"tail:expert_dp_stage{stage}_lane{lane}:rs_arrival_join::release:"
            tokens = nodes.loc[nodes["node_id"].str.startswith(prefix), "node_id"].astype(str).tolist()
            if len(tokens) != 2:
                raise ValueError(f"Expert-DP RS release vector changed: {prefix} -> {tokens}")
            for token in tokens:
                remove |= edges["dst"].eq(token) & edges["edge_type"].eq("collective_rank_release_start")
                set_software_duration(nodes, node_index, token, duration, "tail_dp_rs_to_expert_dp_rs_software")
                additions.append(edge_row(list(edges.columns), dp_done, token, "rank_program_order", "dp_rs_to_expert_dp_rs"))

    # A global join is required by the observed optimizer contract.
    world_node = "tail:world_rs_done"
    template = nodes.loc[nodes["node_id"].eq("iteration:completion_join")].iloc[0].copy()
    template["node_id"] = world_node
    template["kind"] = "collective_boundary"
    template["phase"] = "OPT"
    template["schedule_region"] = "optimizer_tail"
    template["op_name"] = "world_rs_done"
    template["parallelism"] = "global"
    template["timing_source"] = "v63_dependency_structure"
    template["timing_component"] = "zero_duration_dependency"
    template["source_parameter_key"] = ""
    for column in (
        "duration_ns", "compute_exposed_ns_model", "compute_overlap_ns_model",
        "network_service_ns_model", "software_sync_ns_model", "framework_residual_ns_model",
        "predicted_start_ns", "predicted_end_ns",
    ):
        template[column] = 0
    template["critical_predecessor"] = ""
    template["on_critical_path"] = False
    nodes = pd.concat([nodes, pd.DataFrame([template])], ignore_index=True)
    node_index[world_node] = len(nodes) - 1
    for stage in range(pp):
        group_ids = [f"dp_with_cp_stage{stage}"] + [f"expert_dp_stage{stage}_lane{lane}" for lane in range(8)]
        for group_id in group_ids:
            additions.append(edge_row(
                list(edges.columns), f"tail:{group_id}:rs_completion_sync", world_node,
                "world_rs_completion", "all_gradient_shards",
            ))
    for rank in range(pp * lanes):
        additions.append(edge_row(
            list(edges.columns), world_node, f"tail:r{rank}:optimizer_start",
            "optimizer_world_rs_dependency", "all_gradient_shards",
        ))

    # AG is a four-call per-rank chain.  Group joins lower those dependencies.
    for stage in range(pp):
        role = target_role(stage, last_stage)
        dp_ag0_done = f"tail:dp_with_cp_stage{stage}:ag0_completion_sync"
        for lane in range(8):
            arrival = f"tail:expert_dp_stage{stage}_lane{lane}:ag0_arrival_join"
            set_software_duration(
                nodes, node_index, arrival,
                role_gap(gaps, "dp_ag0_to_expert_dp_ag0", role),
                "tail_dp_ag0_to_expert_dp_ag0_software",
            )
            additions.append(edge_row(
                list(edges.columns), dp_ag0_done, arrival,
                "rank_program_order_lowered_to_group", "dp_ag0_to_expert_dp_ag0",
            ))

        dp_ag1_arrival = f"tail:dp_with_cp_stage{stage}:ag1_arrival_join"
        remove |= edges["dst"].eq(dp_ag1_arrival) & edges["edge_type"].eq("allgather_round_order")
        set_software_duration(
            nodes, node_index, dp_ag1_arrival,
            role_gap(gaps, "expert_dp_ag0_to_dp_ag1", role),
            "tail_expert_dp_ag0_to_dp_ag1_software",
        )
        for lane in range(8):
            additions.append(edge_row(
                list(edges.columns), f"tail:expert_dp_stage{stage}_lane{lane}:ag0_completion_sync",
                dp_ag1_arrival, "rank_program_order_lowered_to_group", "expert_dp_ag0_to_dp_ag1",
            ))

        dp_ag1_done = f"tail:dp_with_cp_stage{stage}:ag1_completion_sync"
        for lane in range(8):
            arrival = f"tail:expert_dp_stage{stage}_lane{lane}:ag1_arrival_join"
            remove |= edges["dst"].eq(arrival) & edges["edge_type"].eq("allgather_round_order")
            set_software_duration(
                nodes, node_index, arrival,
                role_gap(gaps, "dp_ag1_to_expert_dp_ag1", role),
                "tail_dp_ag1_to_expert_dp_ag1_software",
            )
            additions.append(edge_row(
                list(edges.columns), dp_ag1_done, arrival,
                "rank_program_order_lowered_to_group", "dp_ag1_to_expert_dp_ag1",
            ))

    removed_count = int(remove.sum())
    edges = pd.concat([edges.loc[~remove], pd.DataFrame(additions)], ignore_index=True)
    start, end, predecessor, critical = max_plus(nodes, edges, "iteration:completion_join")
    nodes["predicted_start_ns"] = start
    nodes["predicted_end_ns"] = end
    node_ids = nodes["node_id"].astype(str).tolist()
    nodes["critical_predecessor"] = [node_ids[value] if value >= 0 else "" for value in predecessor]
    nodes["on_critical_path"] = [position in critical for position in range(len(nodes))]

    audit = order_audit(nodes, pp)
    if audit["status"] != "PASS":
        raise ValueError(f"v6.3 order audit failed: {audit['failure_count']}")
    audit_path = output / "optimizer_tail_order_audit.json"
    atomic_json(audit_path, audit)

    raw_ns = int(nodes.loc[nodes["node_id"].eq("iteration:completion_join"), "predicted_end_ns"].iloc[0])
    raw_ms = raw_ns / 1e6
    v62_contract = json.loads(contract_input.read_text(encoding="utf-8"))
    reconciliation_ms = float(v62_contract["prediction"]["source_reconciliation_ms"])
    outer_ms = float(v62_contract["prediction"]["outer_framework_ms"])
    profiler_ms = raw_ms + reconciliation_ms
    training_ms = profiler_ms + outer_ms
    world = int(config["target"]["world_size"])
    peak = float(config["target"]["peak_tflops_per_gpu"]) * 1e12
    flops = float(config["target"]["model_flops_per_iteration"])
    mfu = 100.0 * flops / ((training_ms / 1000.0) * world * peak)

    node_path = prediction_dir / "dag_v63_nodes.csv.gz"
    edge_path = prediction_dir / "dag_v63_edges.csv.gz"
    critical_path = prediction_dir / "dag_v63_critical_path.csv"
    nodes.to_csv(node_path, index=False, compression="gzip")
    edges.to_csv(edge_path, index=False, compression="gzip")
    nodes[nodes["on_critical_path"]].sort_values("predicted_start_ns").to_csv(critical_path, index=False)

    target_grid = [*config["split"]["target_regression_only"], *config["split"]["target_validation"]]
    method_rows = []
    for iteration in target_grid:
        method_rows.append({
            "protocol_id": config["protocol"]["id"], "source_case": "256gpu_pp16_cp2_a2a",
            "target_case": config["target"]["case_id"],
            "method": "DAG v6.3 ordered DP/Expert-DP optimizer tail",
            "method_version": "dag-v6.3-source60-100-v1",
            "status": "PREDICTIVE_PARTIAL_UNEVALUATED",
            "split": "regression_only" if int(iteration) == 55 else "validation",
            "iteration": int(iteration), "predicted_raw_graph_ms": raw_ms,
            "predicted_source_reconciliation_ms": reconciliation_ms,
            "predicted_profiler_step_ms": profiler_ms,
            "predicted_outer_framework_ms": outer_ms,
            "predicted_training_step_ms": training_ms, "predicted_mfu_pct": mfu,
            "target_timing_read": False,
        })
    method_path = prediction_dir / "method_predictions.csv"
    pd.DataFrame(method_rows).to_csv(method_path, index=False)

    critical_frame = nodes[nodes["on_critical_path"]]
    components = {
        label: float(critical_frame[column].sum()) / 1e6
        for label, column in {
            "compute_exposed": "compute_exposed_ns_model",
            "compute_overlap": "compute_overlap_ns_model",
            "network_service": "network_service_ns_model",
            "software_sync": "software_sync_ns_model",
            "framework_residual": "framework_residual_ns_model",
        }.items()
    }
    contract = {
        "schema": "dag-v6.3-ordered-optimizer-tail-prediction-v1",
        "status": "PREDICTIVE_PARTIAL_UNEVALUATED",
        "prediction": {
            "raw_graph_ms": raw_ms, "source_reconciliation_ms": reconciliation_ms,
            "profiler_step_ms": profiler_ms, "outer_framework_ms": outer_ms,
            "training_step_ms": training_ms, "mfu_pct": mfu,
        },
        "v62_raw_graph_ms": float(v62_contract["prediction"]["raw_graph_ms"]),
        "v63_minus_v62_raw_graph_ms": raw_ms - float(v62_contract["prediction"]["raw_graph_ms"]),
        "critical_path_components_ms": components,
        "dependency_patch": {
            "removed_edges": removed_count, "added_edges": len(additions),
            "world_rs_join_nodes_added": 1,
            "rank_program": config["ordering"]["rank_program"],
            "rank_completion_policy": config["ordering"]["rank_completion_policy"],
        },
        "target_timing_read": False,
        "accuracy_claim_allowed": False,
    }
    contract_path = output / "prediction_contract.json"
    atomic_json(contract_path, contract)

    access_path = output / "input_access_audit.json"
    input_paths = [node_input, edge_input, contract_input, seal_path, source_events_path]
    atomic_json(access_path, {
        "schema": "dag-v6.3-input-access-audit-v1", "status": "PASS_NO_TARGET_TIMING",
        "source_iterations": list(source_iterations), "target_timing_files_read": 0,
        "model_process_read_fields": {
            "v62_nodes": ["static graph", "frozen source-transferred durations", "predicted clocks"],
            "v62_edges": ["src", "dst", "edge_type", "tensor_key", "dependency_source"],
            "source_rank_events": [
                "iteration", "behavior", "call_round", "rank", "pp_stage", "start_ns", "end_ns",
            ],
        },
        "forbidden_target_fields_read": [], "ns3_invocations": 0,
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in input_paths
        ],
    })

    report_path = output / "DAG_V63_ORDERED_OPTIMIZER_TAIL_REPORT.md"
    atomic_text(report_path, f"""# DAG v6.3：DP / Expert-DP optimizer tail 顺序修正

状态：`PREDICTIVE_PARTIAL_UNEVALUATED`。

## 修正内容

v6.2 把 DP RS 与 Expert-DP RS 从 BWD 直接并发释放，optimizer 也只等待本 rank 关联的局部 RS；AG 的 DP 与 Expert-DP 也没有形成真实的跨类型程序顺序。v6.3 固化为：

`DP RS → Expert-DP RS → world RS done → optimizer → DP AG0 → Expert-DP AG0 → DP AG1 → Expert-DP AG1`

相邻调用之间的软件间隔只从 256 卡 iteration 60–100 的稳定 trace 提取，按 first/internal/last PP stage 的非负 median 冻结；构图阶段没有读取 224 卡时间。

## 预测变化

- v6.2 raw graph：`{contract['v62_raw_graph_ms']:.6f} ms`
- v6.3 raw graph：`{raw_ms:.6f} ms`
- 顺序修正增量：`{contract['v63_minus_v62_raw_graph_ms']:.6f} ms`
- v6.3 profiler：`{profiler_ms:.6f} ms`
- v6.3 training：`{training_ms:.6f} ms`
- v6.3 MFU：`{mfu:.6f}%`

## 限制

v6.2 的通信后端只有 group completion，没有每个 rank 的 network completion offset。因此 v6.3 将 rank 程序依赖保守地下沉到 group completion；顺序已经正确，但 EDP 的释放可能比真实 rank completion 略晚。正式逐-rank 接口应由 OISA 返回并绑定 `rank_network_done_offsets_ns` 后替换该近似。
""")
    html_path = output / "dag_v63_optimizer_tail_order.html"
    atomic_text(html_path, render_html(contract, audit, calibration))
    reproduction_path = output / "reproduction_command.txt"
    atomic_text(reproduction_path, f"cd {REPO}\npython {Path(__file__).resolve()} --config {config_path}\n")
    log_path = logs_dir / "build.log"
    atomic_text(log_path, "\n".join([
        "DAG v6.3 ordered optimizer tail PASS",
        f"generated_at={datetime.now(timezone.utc).isoformat()}",
        f"raw_graph_ms={raw_ms:.9f}", f"v63_minus_v62_raw_graph_ms={contract['v63_minus_v62_raw_graph_ms']:.9f}",
        f"profiler_step_ms={profiler_ms:.9f}", f"training_step_ms={training_ms:.9f}",
        f"removed_edges={removed_count}", f"added_edges={len(additions)}",
        "optimizer_tail_order_audit=PASS", "target_timing_files_read=0", "status=PREDICTIVE_PARTIAL_UNEVALUATED", "",
    ]))
    provenance_path = output / "provenance.json"
    code_paths = [config_path, Path(__file__).resolve()]
    atomic_json(provenance_path, {
        "schema": "dag-v6.3-provenance-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO), "git_branch": git_output("branch", "--show-current"),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--short").splitlines(),
        "version_state": "PARTIAL_UNCOMMITTED_WORKTREE_PINNED_BY_SHA256",
        "code_and_config": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in code_paths
        ],
        "v62_prediction_seal": {"path": str(seal_path), "sha256": sha256(seal_path)},
        "source_rank_events": {"path": str(source_events_path), "sha256": sha256(source_events_path)},
    })
    artifacts = [
        node_path, edge_path, critical_path, method_path, calibration_path, audit_path,
        contract_path, access_path, report_path, html_path, reproduction_path, log_path, provenance_path,
    ]
    atomic_json(output / "artifact_manifest.json", {
        "schema": "dag-v6.3-artifact-manifest-v1",
        "artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts
        ],
    })
    print(json.dumps({
        "status": contract["status"], "output": str(output), "raw_graph_ms": raw_ms,
        "v63_minus_v62_raw_graph_ms": contract["v63_minus_v62_raw_graph_ms"],
        "profiler_step_ms": profiler_ms, "training_step_ms": training_ms,
        "optimizer_tail_order_audit": audit["status"], "target_timing_files_read": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

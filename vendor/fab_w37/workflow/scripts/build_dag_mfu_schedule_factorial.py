#!/usr/bin/env python3
"""Generate unit-cost 1F1B schedule DAGs for a PP x microbatch factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def schedule(stage: int, pp: int, microbatches: int) -> list[tuple[str, int]]:
    warmup = min(pp - stage - 1, microbatches)
    remaining = microbatches - warmup
    operations: list[tuple[str, int]] = [("F", microbatch) for microbatch in range(warmup)]
    for index in range(remaining):
        operations.append(("F", warmup + index))
        operations.append(("B", index))
    operations.extend(("B", microbatch) for microbatch in range(remaining, microbatches))
    return operations


def build_graph(scenario: str, pp: int, microbatches: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = []
    edge_rows = []
    for stage in range(pp):
        operations = schedule(stage, pp, microbatches)
        for local_order, (phase, microbatch) in enumerate(operations):
            node_rows.append({
                "scenario": scenario,
                "node_id": f"s{stage}:{phase}{microbatch}",
                "pp_stage": stage,
                "phase": phase,
                "microbatch": microbatch,
                "local_order": local_order,
                "duration_units": 1,
            })
        for left, right in zip(operations, operations[1:]):
            edge_rows.append({
                "scenario": scenario,
                "src": f"s{stage}:{left[0]}{left[1]}",
                "dst": f"s{stage}:{right[0]}{right[1]}",
                "edge_type": "local_program_order",
            })
    for stage in range(pp - 1):
        for microbatch in range(microbatches):
            edge_rows.append({
                "scenario": scenario,
                "src": f"s{stage}:F{microbatch}",
                "dst": f"s{stage + 1}:F{microbatch}",
                "edge_type": "pp_forward_dependency",
            })
            edge_rows.append({
                "scenario": scenario,
                "src": f"s{stage + 1}:B{microbatch}",
                "dst": f"s{stage}:B{microbatch}",
                "edge_type": "pp_backward_dependency",
            })
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def replay(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    node_ids = nodes["node_id"].tolist()
    duration = nodes.set_index("node_id")["duration_units"].astype(int).to_dict()
    successors: dict[str, list[str]] = defaultdict(list)
    predecessors: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for row in edges.itertuples(index=False):
        successors[row.src].append(row.dst)
        predecessors[row.dst].append(row.src)
        indegree[row.dst] += 1
    ready = deque(sorted(node_id for node_id, value in indegree.items() if value == 0))
    start: dict[str, int] = {}
    end: dict[str, int] = {}
    critical_predecessor: dict[str, str | None] = {}
    order: list[str] = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        candidates = predecessors[node_id]
        parent = max(candidates, key=lambda value: (end[value], value)) if candidates else None
        critical_predecessor[node_id] = parent
        start[node_id] = end[parent] if parent else 0
        end[node_id] = start[node_id] + duration[node_id]
        for child in successors[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise ValueError("schedule graph is cyclic")
    completion = max(end, key=lambda value: (end[value], value))
    critical = []
    cursor: str | None = completion
    while cursor is not None:
        critical.append(cursor)
        cursor = critical_predecessor[cursor]
    critical.reverse()
    output = nodes.copy()
    output["start_unit"] = output["node_id"].map(start)
    output["end_unit"] = output["node_id"].map(end)
    output["critical_predecessor"] = output["node_id"].map(critical_predecessor)
    output["on_critical_path"] = output["node_id"].isin(critical)
    return output, critical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.source_contract.resolve(), args.target_contract.resolve()]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = json.loads(paths[0].read_text(encoding="utf-8"))
    target = json.loads(paths[1].read_text(encoding="utf-8"))
    source_pp = int(source["parallelism"]["pp"])
    target_pp = int(target["parallelism"]["pp"])
    source_mb = int(source["workload"]["microbatches"])
    target_mb = int(target["workload"]["microbatches"])
    variants = [
        ("pp16_m4_source_structure", source_pp, source_mb, "observed_configuration_shape"),
        ("pp14_m4_pp_only", target_pp, source_mb, "structural_counterfactual"),
        ("pp16_m3_mb_only", source_pp, target_mb, "structural_counterfactual"),
        ("pp14_m3_target_structure", target_pp, target_mb, "observed_configuration_shape"),
    ]
    all_nodes = []
    all_edges = []
    summary_rows = []
    for scenario, pp, microbatches, role in variants:
        nodes, edges = build_graph(scenario, pp, microbatches)
        replayed, critical = replay(nodes, edges)
        makespan = int(replayed["end_unit"].max())
        total_work = int(replayed["duration_units"].sum())
        summary_rows.append({
            "scenario": scenario,
            "scenario_role": role,
            "pp": pp,
            "microbatches": microbatches,
            "node_count": len(replayed),
            "edge_count": len(edges),
            "local_edge_count": int(edges["edge_type"].eq("local_program_order").sum()),
            "pp_forward_edge_count": int(edges["edge_type"].eq("pp_forward_dependency").sum()),
            "pp_backward_edge_count": int(edges["edge_type"].eq("pp_backward_dependency").sum()),
            "unit_cost_makespan": makespan,
            "critical_path_node_count": len(critical),
            "total_work_units": total_work,
            "average_concurrency": total_work / makespan,
            "unit_cost_semantics": "one unit per F/B phase; not time or MFU",
        })
        all_nodes.append(replayed)
        all_edges.append(edges)
    nodes = pd.concat(all_nodes, ignore_index=True)
    edges = pd.concat(all_edges, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    baseline = summary.set_index("scenario").loc["pp16_m4_source_structure"]
    summary["makespan_change_vs_source_pct"] = (
        summary["unit_cost_makespan"] / baseline["unit_cost_makespan"] - 1
    ) * 100
    summary["average_concurrency_change_vs_source_pct"] = (
        summary["average_concurrency"] / baseline["average_concurrency"] - 1
    ) * 100
    audit = {
        "schema": "dag-mfu-unit-cost-1f1b-schedule-factorial-v1",
        "status": "PASS_STRUCTURE_ONLY_GENERIC_PP_MB_GRAPH_FACTORY",
        "variants": summary.to_dict("records"),
        "checks": {
            "all_graphs_acyclic": True,
            "node_formula": "2 * pp * microbatches",
            "all_node_counts_match": bool(
                (summary["node_count"] == 2 * summary["pp"] * summary["microbatches"]).all()
            ),
            "critical_path_equals_unit_makespan": bool(
                summary["critical_path_node_count"].eq(summary["unit_cost_makespan"]).all()
            ),
        },
        "limitations": [
            "each F/B phase has unit cost",
            "no layers, ranks, token load, communication, host placement or optimizer nodes",
            "counterfactual structures were not compared with target timing",
            "unit-cost makespan is not milliseconds, MFU or a model accuracy result",
        ],
        "target_trace_files_read": 0,
        "model_parameter_updates": 0,
        "released_model_version": None,
        "network_downloads": [],
    }
    if not all(audit["checks"].values()):
        raise ValueError(f"schedule graph checks failed: {audit['checks']}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nodes_path = output / "schedule_factorial_nodes.csv.gz"
    edges_path = output / "schedule_factorial_edges.csv.gz"
    summary_path = output / "schedule_factorial_summary.csv"
    audit_path = output / "schedule_factorial_audit.json"
    report_path = output / "SCHEDULE_FACTORIAL.md"
    nodes.to_csv(nodes_path, index=False, compression="gzip")
    edges.to_csv(edges_path, index=False, compression="gzip")
    summary.to_csv(summary_path, index=False)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    s = summary.set_index("scenario")
    report_path.write_text(f"""# PP×microbatch 通用1F1B结构DAG

状态：**{audit['status']}**。这是下一场景生成器的最小结构原型，不是v6.9，也不带毫秒成本。

每个 `PP stage × F/B × microbatch` 是一个单位节点，边只包含stage内程序顺序、FWD向后stage传播和BWD向前stage传播。四个场景均通过无环、节点数和关键路径重放检查：

| 场景 | 节点 | 边 | 单位关键路径 | 平均并发 | 相对source单位路径 |
|---|---:|---:|---:|---:|---:|
| PP16 / MB4 | {int(s.at['pp16_m4_source_structure', 'node_count'])} | {int(s.at['pp16_m4_source_structure', 'edge_count'])} | {int(s.at['pp16_m4_source_structure', 'unit_cost_makespan'])} | {s.at['pp16_m4_source_structure', 'average_concurrency']:.3f} | {s.at['pp16_m4_source_structure', 'makespan_change_vs_source_pct']:+.2f}% |
| PP14 / MB4（只改PP） | {int(s.at['pp14_m4_pp_only', 'node_count'])} | {int(s.at['pp14_m4_pp_only', 'edge_count'])} | {int(s.at['pp14_m4_pp_only', 'unit_cost_makespan'])} | {s.at['pp14_m4_pp_only', 'average_concurrency']:.3f} | {s.at['pp14_m4_pp_only', 'makespan_change_vs_source_pct']:+.2f}% |
| PP16 / MB3（只改MB） | {int(s.at['pp16_m3_mb_only', 'node_count'])} | {int(s.at['pp16_m3_mb_only', 'edge_count'])} | {int(s.at['pp16_m3_mb_only', 'unit_cost_makespan'])} | {s.at['pp16_m3_mb_only', 'average_concurrency']:.3f} | {s.at['pp16_m3_mb_only', 'makespan_change_vs_source_pct']:+.2f}% |
| PP14 / MB3 | {int(s.at['pp14_m3_target_structure', 'node_count'])} | {int(s.at['pp14_m3_target_structure', 'edge_count'])} | {int(s.at['pp14_m3_target_structure', 'unit_cost_makespan'])} | {s.at['pp14_m3_target_structure', 'average_concurrency']:.3f} | {s.at['pp14_m3_target_structure', 'makespan_change_vs_source_pct']:+.2f}% |

这个原型证明PP/MB可以从scenario参数重新生成依赖，而不需要在v6.7里写死PP14/MB3。它也说明单位路径变短并不等于真实耗时变短：要成为完整预测图，仍需加入层级算子、16 rank/stage、token条件成本、PP/CP/DP/EDP/EP通信、optimizer与双时钟节点，并通过source replay。
""", encoding="utf-8")
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps({
        "schema": "dag-mfu-schedule-factorial-provenance-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": artifact(Path(__file__)),
        "inputs": [artifact(path) for path in paths],
        "outputs": [artifact(path) for path in (
            nodes_path, edges_path, summary_path, audit_path, report_path
        )],
        "target_trace_files_read": 0,
        "network_downloads": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "dag-mfu-schedule-factorial-manifest-v1",
        "status": audit["status"],
        "artifacts": [artifact(path) for path in (
            nodes_path, edges_path, summary_path, audit_path, report_path, provenance_path
        )],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "variants": summary_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

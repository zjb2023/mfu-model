#!/usr/bin/env python3
"""Visualize the source256-frozen v6.8 profiler-entry node.

The target Trace is evaluator-only.  The page never updates model parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_V67 = REPO / (
    "case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/"
    "dag_v67_microbatch_runtime_shape_256_to_224"
)
DEFAULT_V68 = REPO / (
    "case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/"
    "dag_v68_target_assisted_profiler_entry_256_to_224"
)
DEFAULT_SOURCE_EVENT_WINDOWS = REPO / "case_256gpu_pp16_cp2_a2a/results/readiness/event_windows.csv"
DEFAULT_SOURCE_PHASE_EVENTS = REPO / (
    "case_256gpu_pp16_cp2_a2a/results/mfu_accuracy_comparison_2026w36/"
    "dag_v53_ep_group_source60_100/source_pp_trace_events_60_100.csv"
)


def checked(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(checked(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with checked(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mape(rows: list[dict[str, Any]], predicted: str, actual: str) -> float:
    return 100.0 * sum(abs(float(row[predicted]) - float(row[actual])) / float(row[actual]) for row in rows) / len(rows)


def critical_label(row: pd.Series) -> str:
    stage = int(row["pp_stage"])
    phase = str(row["phase"])
    microbatch = int(row["microbatch"])
    kind = str(row["kind"])
    node_id = str(row["node_id"])
    if kind == "scheduler_handoff":
        return f"PP{stage} 本地F→B就绪等待"
    if kind == "microbatch_release_gap":
        return f"PP{stage} 本地microbatch就绪等待"
    if kind == "pp_p2p":
        if ":bwd" in node_id:
            return f"PP梯度传输（{node_id.split(':')[-1]}）"
        return f"PP激活传输（{node_id.split(':')[-1]}）"
    if phase in {"FWD", "BWD"} and microbatch >= 0:
        return f"PP{stage} {phase[0]}{microbatch}"
    if phase == "OPT":
        return f"PP{stage} optimizer" if stage >= 0 else "全局optimizer尾部"
    return node_id


def critical_track(row: pd.Series) -> str:
    phase = str(row["phase"])
    kind = str(row["kind"])
    if kind == "framework_entry":
        return "entry"
    if kind in {"scheduler_handoff", "microbatch_release_gap"}:
        return "wait"
    if kind == "pp_p2p":
        return "p2p"
    if phase == "FWD":
        return "forward"
    if phase == "BWD":
        return "backward"
    if phase == "OPT":
        return "optimizer"
    return "global"


def build_critical_path_payload(
    nodes_path: Path,
    edges_path: Path,
    profiler_ms: float,
    completion_id: str = "iteration:completion_join",
) -> dict[str, Any]:
    nodes = pd.read_csv(checked(nodes_path), low_memory=False)
    edges = pd.read_csv(checked(edges_path), low_memory=False)
    if nodes["node_id"].duplicated().any():
        raise ValueError("v6.8 node ids are not unique")
    lookup = nodes.set_index("node_id", drop=False)
    chain: list[pd.Series] = []
    current = completion_id
    seen: set[str] = set()
    while current:
        if current in seen or current not in lookup.index:
            raise ValueError(f"broken v6.8 critical predecessor chain at {current}")
        seen.add(current)
        row = lookup.loc[current]
        chain.append(row)
        predecessor = row.get("critical_predecessor", "")
        current = "" if pd.isna(predecessor) else str(predecessor)
    chain.reverse()
    declared = set(nodes.loc[nodes["on_critical_path"].astype(bool), "node_id"].astype(str))
    if set(seen) != declared:
        raise ValueError("v6.8 reconstructed critical chain differs from on_critical_path flags")

    positive = [row for row in chain if int(row["duration_ns"]) > 0]
    completion_ms = float(lookup.loc[completion_id, "predicted_end_ns"]) / 1e6
    duration_sum_ms = sum(int(row["duration_ns"]) for row in positive) / 1e6
    if abs(completion_ms - duration_sum_ms) > 0.002:
        raise ValueError("v6.8 critical path does not conserve raw graph duration")

    spans: list[dict[str, Any]] = []
    for row in positive:
        item = {
            "start_ms": float(row["predicted_start_ns"]) / 1e6,
            "end_ms": float(row["predicted_end_ns"]) / 1e6,
            "duration_ms": float(row["duration_ns"]) / 1e6,
            "stage": int(row["pp_stage"]),
            "phase": str(row["phase"]),
            "microbatch": int(row["microbatch"]),
            "kind": str(row["kind"]),
            "track": critical_track(row),
            "label": critical_label(row),
            "first_node_id": str(row["node_id"]),
            "last_node_id": str(row["node_id"]),
            "node_count": 1,
        }
        if spans:
            previous = spans[-1]
            same_track = all(previous[key] == item[key] for key in ("stage", "phase", "microbatch", "track"))
            contiguous = abs(previous["end_ms"] - item["start_ms"]) <= 0.002
            if same_track and contiguous:
                previous["end_ms"] = item["end_ms"]
                previous["duration_ms"] += item["duration_ms"]
                previous["last_node_id"] = item["last_node_id"]
                previous["node_count"] += 1
                continue
        spans.append(item)

    incoming: dict[str, list[str]] = {}
    for edge in edges.itertuples(index=False):
        incoming.setdefault(str(edge.dst), []).append(str(edge.src))
    decisions: list[dict[str, Any]] = []
    for row in chain:
        if str(row["kind"]) != "phase_boundary" or str(row["op_name"]) not in {"fwd_start", "bwd_start"}:
            continue
        candidates = [source for source in incoming.get(str(row["node_id"]), []) if source in lookup.index]
        if len(candidates) < 2:
            continue
        ranked = sorted(
            candidates,
            key=lambda source: float(lookup.loc[source, "predicted_end_ns"]),
            reverse=True,
        )
        winner = ranked[0]
        runner_up = ranked[1]
        declared_winner = str(row["critical_predecessor"])
        if winner != declared_winner:
            raise ValueError(f"critical winner mismatch at {row['node_id']}")
        winner_row = lookup.loc[winner]
        runner_row = lookup.loc[runner_up]
        decisions.append({
            "target": critical_label(row),
            "target_node_id": str(row["node_id"]),
            "target_stage": int(row["pp_stage"]),
            "target_phase": str(row["phase"]),
            "target_microbatch": int(row["microbatch"]),
            "target_start_ms": float(row["predicted_start_ns"]) / 1e6,
            "winner": critical_label(winner_row),
            "winner_node_id": winner,
            "winner_arrival_ms": float(winner_row["predicted_end_ns"]) / 1e6,
            "runner_up": critical_label(runner_row),
            "runner_up_node_id": runner_up,
            "runner_up_arrival_ms": float(runner_row["predicted_end_ns"]) / 1e6,
            "winning_margin_ms": (
                float(winner_row["predicted_end_ns"]) - float(runner_row["predicted_end_ns"])
            ) / 1e6,
            "candidate_count": len(ranked),
        })
    decisions.sort(key=lambda item: item["winning_margin_ms"], reverse=True)
    return {
        "definition": "single max-plus predecessor chain from iteration completion back to the graph root",
        "raw_graph_ms": completion_ms,
        "profiler_ms": profiler_ms,
        "post_graph_reconciliation_ms": profiler_ms - completion_ms,
        "positive_duration_node_count": len(positive),
        "merged_span_count": len(spans),
        "duration_coverage_pct": 100.0 * duration_sum_ms / completion_ms,
        "spans": spans,
        "join_decisions": decisions,
        "dominant_join_decision": decisions[0],
    }


def shifted_critical_path(path: dict[str, Any], shift_ms: float, profiler_ms: float) -> dict[str, Any]:
    spans = []
    for item in path["spans"]:
        if item["track"] == "entry":
            continue
        shifted = dict(item)
        shifted["start_ms"] -= shift_ms
        shifted["end_ms"] -= shift_ms
        spans.append(shifted)
    decisions = []
    for item in path["join_decisions"]:
        shifted = dict(item)
        for key in ("target_start_ms", "winner_arrival_ms", "runner_up_arrival_ms"):
            shifted[key] -= shift_ms
        decisions.append(shifted)
    raw_graph_ms = path["raw_graph_ms"] - shift_ms
    return {
        **path,
        "raw_graph_ms": raw_graph_ms,
        "profiler_ms": profiler_ms,
        "post_graph_reconciliation_ms": profiler_ms - raw_graph_ms,
        "spans": spans,
        "join_decisions": decisions,
        "dominant_join_decision": decisions[0],
    }


def source_profiler_boundaries(
    event_windows_path: Path,
    phase_events_path: Path,
    iterations: list[int],
) -> dict[int, dict[str, float]]:
    windows: dict[int, dict[str, Any]] = {}
    for row in read_csv(event_windows_path):
        iteration = int(row["iteration"])
        if iteration not in iterations:
            continue
        item = windows.setdefault(iteration, {"starts": [], "ends": [], "ranks": set()})
        item["starts"].append(int(row["start_ns"]))
        item["ends"].append(int(row["end_ns"]))
        item["ranks"].add(int(row["rank"]))
    phases: dict[int, dict[str, Any]] = {}
    for row in read_csv(phase_events_path):
        iteration = int(row["iteration"])
        if iteration not in iterations:
            continue
        item = phases.setdefault(iteration, {"starts": [], "ends": [], "ranks": set()})
        item["starts"].append(int(row["observed_start_ns"]))
        item["ends"].append(int(row["observed_end_ns"]))
        item["ranks"].add(int(row["rank"]))
    result: dict[int, dict[str, float]] = {}
    for iteration in iterations:
        if len(windows.get(iteration, {}).get("ranks", set())) != 256:
            raise ValueError(f"source256 ProfilerStep rank grid incomplete: {iteration}")
        if len(phases.get(iteration, {}).get("ranks", set())) != 256:
            raise ValueError(f"source256 F/B rank grid incomplete: {iteration}")
        profiler_start_ns = min(windows[iteration]["starts"])
        profiler_end_ns = max(windows[iteration]["ends"])
        phase_start_ns = min(phases[iteration]["starts"])
        phase_end_ns = max(phases[iteration]["ends"])
        if not profiler_start_ns <= phase_start_ns < phase_end_ns <= profiler_end_ns:
            raise ValueError(f"source256 Profiler/F/B boundary order changed: {iteration}")
        profiler_ms = (profiler_end_ns - profiler_start_ns) / 1e6
        phase_ms = (phase_end_ns - phase_start_ns) / 1e6
        result[iteration] = {
            "profiler_start_ns": profiler_start_ns,
            "phase_start_ns": phase_start_ns,
            "phase_end_ns": phase_end_ns,
            "profiler_end_ns": profiler_end_ns,
            "actual_entry_ms": (phase_start_ns - profiler_start_ns) / 1e6,
            "actual_profiler_ms": profiler_ms,
            "actual_phase_envelope_ms": phase_ms,
            "actual_outside_phase_envelope_ms": profiler_ms - phase_ms,
        }
    return result


def build_payload(
    v67: Path,
    v68: Path,
    source_event_windows_path: Path = DEFAULT_SOURCE_EVENT_WINDOWS,
    source_phase_events_path: Path = DEFAULT_SOURCE_PHASE_EVENTS,
) -> tuple[dict[str, Any], list[Path]]:
    v67_timeline_path = checked(v67 / "evaluator_only/dag_v67_trace_timeline_payload.json")
    v67_contract_path = checked(v67 / "prediction_contract.json")
    v68_contract_path = checked(v68 / "prediction_contract.json")
    v68_metrics_path = checked(v68 / "evaluator_only/metrics.json")
    v68_evaluation_path = checked(v68 / "evaluator_only/iteration_evaluation.csv")
    entry_parameter_path = checked(v68 / "calibration/profiler_entry_parameter.json")
    decomposition_path = checked(
        v67 / "evaluator_only/error_module_diagnosis/iteration_error_decomposition.csv"
    )
    v68_access_path = checked(v68 / "input_access_audit.json")
    v68_nodes_path = checked(v68 / "predictions/dag_v68_nodes.csv.gz")
    v68_edges_path = checked(v68 / "predictions/dag_v68_edges.csv.gz")

    v67_timeline = read_json(v67_timeline_path)
    v67_contract = read_json(v67_contract_path)
    v68_contract = read_json(v68_contract_path)
    metrics = read_json(v68_metrics_path)
    parameter = read_json(entry_parameter_path)
    access = read_json(v68_access_path)
    evaluation = read_csv(v68_evaluation_path)
    decomposition = read_csv(decomposition_path)

    if v67_timeline.get("model_version") != "v67":
        raise ValueError("v6.7 timeline payload changed")
    if access.get("status") != "PASS_SOURCE256_ONLY_BEFORE_SEAL":
        raise ValueError("v6.8 source256-only access audit failed")
    if access.get("target_timing_read_before_seal") is not False:
        raise ValueError("v6.8 target timing was read before prediction seal")
    if metrics.get("status") != "PASS_SOURCE256_FROZEN_ENTRY_TARGET_DEVELOPMENT":
        raise ValueError("v6.8 source256-frozen evaluation status changed")
    if metrics.get("formal_blind_claim_allowed") is not False:
        raise ValueError("v6.8 must not be presented as blind extrapolation")

    entry_ms = float(v68_contract["entry_node"]["duration_ms"])
    if abs(entry_ms - float(parameter["duration_ms"])) > 1e-9:
        raise ValueError("v6.8 entry parameter and graph node differ")
    v67_profiler_ms = float(v67_contract["prediction"]["profiler_step_ms"])
    v68_profiler_ms = float(v68_contract["prediction"]["profiler_step_ms"])
    if abs((v68_profiler_ms - v67_profiler_ms) - entry_ms) > 0.002:
        raise ValueError("v6.8 delta is not exactly the explicit entry node")

    actual_entry = {
        int(row["iteration"]): float(row["actual_profiler_entry_to_phase_ms"])
        for row in decomposition
        if int(row["iteration"]) in range(60, 101, 5)
    }
    iterations = [int(row["iteration"]) for row in evaluation]
    if iterations != list(range(60, 101, 5)) or set(actual_entry) != set(iterations):
        raise ValueError("v6.8 visualization iteration grid changed")
    if any(str(iteration) not in v67_timeline["trace"] for iteration in iterations):
        raise ValueError("v6.7 evaluator timeline is missing a target iteration")
    source_boundaries = source_profiler_boundaries(
        checked(source_event_windows_path), checked(source_phase_events_path), iterations
    )
    if any(str(iteration) not in v67_timeline["source_256_trace"] for iteration in iterations):
        raise ValueError("v6.7 evaluator timeline is missing a source256 iteration")
    for iteration in iterations:
        payload_phase_ms = float(
            v67_timeline["source_256_trace"][str(iteration)]["phase_envelope_ms"]
        )
        if abs(payload_phase_ms - source_boundaries[iteration]["actual_phase_envelope_ms"]) > 0.002:
            raise ValueError(f"source256 phase envelope clocks differ: {iteration}")

    evaluation_rows: list[dict[str, Any]] = []
    for row in evaluation:
        iteration = int(row["iteration"])
        actual_profiler_ms = float(row["actual_profiler_step_ms"])
        predicted_v68_ms = float(row["predicted_profiler_step_ms"])
        evaluation_rows.append({
            "iteration": iteration,
            "split": row["split"],
            "actual_entry_ms": actual_entry[iteration],
            "actual_profiler_ms": actual_profiler_ms,
            "v67_profiler_ms": v67_profiler_ms,
            "v68_profiler_ms": predicted_v68_ms,
            "v67_ape_pct": 100.0 * abs(v67_profiler_ms - actual_profiler_ms) / actual_profiler_ms,
            "v68_ape_pct": 100.0 * abs(predicted_v68_ms - actual_profiler_ms) / actual_profiler_ms,
        })

    early = [row for row in evaluation_rows if row["split"] == "target_development_early"]
    late = [row for row in evaluation_rows if row["split"] == "target_development_late"]
    if [row["iteration"] for row in early] != [60, 65, 70, 75, 80]:
        raise ValueError("v6.8 early development split changed")
    if [row["iteration"] for row in late] != [85, 90, 95, 100]:
        raise ValueError("v6.8 late development split changed")
    v67_late_mape = mape(late, "v67_profiler_ms", "actual_profiler_ms")
    v68_late_mape = mape(late, "v68_profiler_ms", "actual_profiler_ms")
    v68_critical_path = build_critical_path_payload(
        v68_nodes_path, v68_edges_path, v68_profiler_ms
    )
    v67_critical_path = shifted_critical_path(
        v68_critical_path, entry_ms, v67_profiler_ms
    )

    payload = {
        "schema": "dag-v6.8-profiler-entry-visualization-v1",
        "status": "PASS_SOURCE256_FROZEN_V68_VISUALIZATION",
        "claim": "source256-frozen entry with target224 development evaluation; not a formal blind result",
        "iterations": iterations,
        "default_iteration": 85,
        "representative_lane": v67_timeline["representative_lane"],
        "entry_node": {
            "id": v68_contract["entry_node"]["node_id"],
            "duration_ms": entry_ms,
            "root_edge_count": int(v68_contract["entry_node"]["root_edge_count"]),
            "semantic_boundary": parameter["semantic_boundary"],
            "source_case": parameter["source_case"],
            "source_iteration": int(parameter["source_iteration"]),
            "parameter_source": "source256_trace",
        },
        "prediction": {
            "v67_profiler_ms": v67_profiler_ms,
            "v68_profiler_ms": v68_profiler_ms,
            "delta_ms": v68_profiler_ms - v67_profiler_ms,
            "base_timeline": v67_timeline["predicted"],
            "v67_critical_path": v67_critical_path,
            "v68_critical_path": v68_critical_path,
        },
        "trace": v67_timeline["trace"],
        "source_256_trace": v67_timeline["source_256_trace"],
        "source_256_profiler_boundaries": {
            str(iteration): source_boundaries[iteration] for iteration in iterations
        },
        "evaluation": evaluation_rows,
        "metrics": {
            "v67_all_development_mape_pct": float(
                metrics["comparison"]["v67_target_development_all_profiler_mape_pct"]
            ),
            "v68_all_development_mape_pct": float(
                metrics["comparison"]["v68_target_development_all_profiler_mape_pct"]
            ),
            "v67_same_late_development_mape_pct": v67_late_mape,
            "v68_same_late_development_mape_pct": v68_late_mape,
            "same_late_development_improvement_pp": v67_late_mape - v68_late_mape,
            "same_late_development_relative_reduction_pct": (
                100.0 * (v67_late_mape - v68_late_mape) / v67_late_mape
            ),
            "v68_early_development_mape_pct": mape(
                early, "v68_profiler_ms", "actual_profiler_ms"
            ),
            "post60_window_diagnostic": metrics["post60_window_diagnostic"],
        },
        "data_scope": {
            "source_parameter_iteration": int(parameter["source_iteration"]),
            "target_development_early": [60, 65, 70, 75, 80],
            "target_development_late": [85, 90, 95, 100],
            "target_parameter_updates": int(metrics["target_parameter_updates"]),
            "target_trace_role": "post-seal development evaluator and timeline only",
            "source_256_trace_role": "entry parameter source and source Trace display",
            "blind_extrapolation_claim_allowed": False,
        },
    }
    inputs = [
        v67_timeline_path,
        v67_contract_path,
        v68_contract_path,
        v68_metrics_path,
        v68_evaluation_path,
        entry_parameter_path,
        decomposition_path,
        v68_access_path,
        v68_nodes_path,
        v68_edges_path,
        checked(source_event_windows_path),
        checked(source_phase_events_path),
    ]
    return payload, inputs


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    metrics = payload["metrics"]
    entry_ms = payload["entry_node"]["duration_ms"]
    window_diagnostic = metrics["post60_window_diagnostic"]
    best3 = window_diagnostic["best_contiguous_3_point"]
    best5 = window_diagnostic["best_contiguous_5_point"]
    strict3 = window_diagnostic["best_strictly_after_60_contiguous_3_point"]
    worst3 = window_diagnostic["worst_contiguous_3_point"]
    dominant = payload["prediction"]["v68_critical_path"]["dominant_join_decision"]
    decision_rows = "".join(
        "<tr>"
        f"<td>{item['target']}</td>"
        f"<td>{item['winner']}</td>"
        f"<td>{item['winner_arrival_ms']/1000.0:.3f}s</td>"
        f"<td>{item['runner_up']}</td>"
        f"<td>{item['runner_up_arrival_ms']/1000.0:.3f}s</td>"
        f"<td>{item['winning_margin_ms']:.1f}ms</td>"
        "</tr>"
        for item in payload["prediction"]["v68_critical_path"]["join_decisions"][:8]
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAG v6.8 · 256卡冻结ENTRY可视化</title>
<style>
:root{{--bg:#07111d;--panel:#0e1b2c;--line:#2b4059;--text:#e9f2fb;--muted:#93a8bd;--fwd:#24c8ff;--bwd:#ff725c;--entry:#f5a524;--entry2:#ffe08a;--critical:#ffe45e;--critical-glow:#fff3a0;--dp-rs:#27d17f;--edp-rs:#00b8a9;--dp-ag:#ffb54a;--edp-ag:#b78cff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}}main{{max-width:1580px;margin:auto;padding:28px}}h1{{font-size:34px;margin:4px 0 8px}}h2{{font-size:20px;margin:0 0 5px}}p{{margin:5px 0}}code{{color:#9cecff}}.muted{{color:var(--muted)}}.eyebrow{{color:#55d6be;font-weight:800;letter-spacing:.12em;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:16px 0}}.card,.panel{{background:var(--panel);border:1px solid var(--line)}}.card{{padding:14px}}.card .v{{font-size:23px;font-weight:800;margin-top:5px;font-variant-numeric:tabular-nums}}.card .k{{color:var(--muted);font-size:12px}}.notice{{border-left:4px solid var(--entry);padding:12px 14px;background:#1b1b25;margin:12px 0}}.flow{{display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:12px 0}}.flow .node{{min-width:170px;padding:12px;border:1px solid var(--line);background:#0a1727}}.flow .hot{{border-color:var(--entry);background:#332515}}.flow .arrow{{display:grid;place-items:center;color:var(--muted);font-size:22px}}.controls{{display:flex;gap:7px;flex-wrap:wrap;margin:14px 0}}button{{border:1px solid var(--line);background:#0b1828;color:var(--muted);padding:7px 10px;border-radius:6px;cursor:pointer}}button.active{{border-color:#55d6be;color:#fff;background:#14383e}}button.cal{{border-bottom:3px solid var(--entry)}}button.hold{{border-bottom:3px solid #55d6be}}.panel{{padding:17px;margin:12px 0}}.head{{display:flex;justify-content:space-between;gap:15px;align-items:flex-start}}.badge{{border:1px solid var(--line);padding:4px 8px;color:var(--muted);font-size:12px;white-space:nowrap}}.legend{{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:10px 0}}.sw{{display:inline-block;width:17px;height:9px;margin-right:5px;vertical-align:middle}}.timeline{{overflow-x:auto;background:#081523;border:1px solid #22384f;padding:5px}}.timeline svg{{display:block;min-width:1350px;width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{font-size:11px;color:var(--muted)}}tr.selected{{background:#14283b}}.scope{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}.scope{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">DAG MFU · v6.8 SOURCE256-FROZEN DEVELOPMENT</div>
<h1>Profiler入口节点：v6.7 → v6.8 的唯一图变化</h1>
<p class="muted">四张图使用同一 ProfilerStep 起点和横轴。FWD/BWD、DP/EDP 配色与 v6.7 页面一致。</p>
<div class="notice"><b>固定口径：</b>v6.8 的 ENTRY 取自 256 卡 Trace iter 85，为 <b>{entry_ms/1000:.4f}s</b>；预测封存前不读取224卡时序。224卡只用于封存后的开发评估。由于该目标场景此前已被分析，本页仍不宣称独立盲测。</div>
<div class="cards">
  <div class="card"><div class="k">v6.7 Profiler预测</div><div class="v">{payload['prediction']['v67_profiler_ms']/1000:.3f}s</div></div>
  <div class="card"><div class="k">v6.8 ENTRY（256卡 iter85）</div><div class="v">{entry_ms/1000:.4f}s</div></div>
  <div class="card"><div class="k">v6.8 Profiler预测</div><div class="v">{payload['prediction']['v68_profiler_ms']/1000:.3f}s</div></div>
  <div class="card"><div class="k">85–100 同集 v6.7 MAPE</div><div class="v">{metrics['v67_same_late_development_mape_pct']:.3f}%</div></div>
  <div class="card"><div class="k">85–100 同集 v6.8 MAPE</div><div class="v">{metrics['v68_same_late_development_mape_pct']:.3f}%</div></div>
</div>
<section class="panel"><h2>图语义：入口节点放在哪里</h2><div class="flow">
  <div class="node"><b>ProfilerStep 开始</b><br><span class="muted">全 rank 最早开始</span></div><div class="arrow">→</div>
  <div class="node hot"><b>ENTRY</b><br><span>{entry_ms/1000:.4f} s</span><br><span class="muted">256卡 Trace iter85 固定值</span></div><div class="arrow">→</div>
  <div class="node"><b>16 个 PP0 lane 根</b><br><span class="muted">首个 F0 计算块</span></div><div class="arrow">→</div>
  <div class="node"><b>PP14 × 3 microbatch</b><br><span class="muted">F/B、DP/EDP 与依赖传播</span></div><div class="arrow">→</div>
  <div class="node"><b>ProfilerStep 完成</b><br><span class="muted">不是在末尾补残差</span></div>
</div></section>
<div class="controls" id="iterationControls"></div><div class="controls" id="dependencyControls"><button data-dep="none">隐藏依赖</button><button class="active" data-dep="pp">只看PP依赖</button><button data-dep="all">全部依赖</button></div>
<div class="legend"><span><i class="sw" style="background:var(--entry)"></i>Profiler入口</span><span><i class="sw" style="background:var(--fwd)"></i>FWD</span><span><i class="sw" style="background:var(--bwd)"></i>BWD</span><span><i class="sw" style="background:var(--dp-rs)"></i>DP RS</span><span><i class="sw" style="background:var(--edp-rs)"></i>EDP RS</span><span><i class="sw" style="background:var(--dp-ag)"></i>DP AG</span><span><i class="sw" style="background:var(--edp-ag)"></i>EDP AG</span><span><i class="sw" style="background:var(--critical);box-shadow:0 0 5px var(--critical-glow)"></i>最终最长路径（全段黄色）</span></div>
<section class="panel"><div class="head"><div><h2>A · v6.7 预测：入口未建模</h2><p class="muted">F0 从 Profiler t=0 开始，表示图里缺少入口边界，不表示真实系统的准备时间为0。</p></div><span class="badge">冻结 v6.7</span></div><div class="timeline" id="v67Timeline"></div></section>
<section class="panel"><div class="head"><div><h2>B · v6.8 预测：ENTRY 固定为 1.2522s</h2><p class="muted">橙色 ENTRY 连接全部16个PP0 lane根；后面的v6.7图整体平移 {entry_ms:.1f} ms，其他节点与依赖不变。</p></div><span class="badge">source256 固定 v6.8</span></div><div class="timeline" id="v68Timeline"></div></section>
<section class="panel"><div class="head"><div><h2>v6.8 最长路径的最终决胜</h2><p class="muted">黄色不是按PP编号猜测，而是从iteration完成节点沿每个汇合点“最后到达的前驱”反向追溯得到；正时长节点覆盖率为 {payload['prediction']['v68_critical_path']['duration_coverage_pct']:.1f}%。</p></div><span class="badge">MAX-PLUS WINNER CHAIN</span></div>
<div class="notice"><b>主决胜点：</b>{dominant['target']}同时等待“{dominant['winner']}”和“{dominant['runner_up']}”。前者在{dominant['winner_arrival_ms']/1000.0:.3f}s到达，后者在{dominant['runner_up_arrival_ms']/1000.0:.3f}s到达，前者晚 <b>{dominant['winning_margin_ms']:.1f}ms</b>，因此赢得关键前驱。PP4–PP13不是消失，而是在该汇合点之前完成，属于非决胜分支。</div>
<div style="overflow-x:auto"><table><thead><tr><th>汇合目标</th><th>最终获胜前驱</th><th>到达</th><th>第二前驱</th><th>到达</th><th>决胜差</th></tr></thead><tbody>{decision_rows}</tbody></table></div></section>
<section class="panel"><div class="head"><div><h2>C · 224 卡 Trace：实际入口与完整时序</h2><p class="muted">当前 iteration 的橙色长度来自封存后 evaluator-only Trace；它只用于画图和误差评估，不回填 v6.8 ENTRY。</p></div><span class="badge" id="traceBadge"></span></div><div class="timeline" id="traceTimeline"></div></section>
<section class="panel"><div class="head"><div><h2>D · 256 卡源 Trace：源场景自身的实际入口</h2><p class="muted">PP16 × 4 microbatch；橙色入口由256卡全rank ProfilerStep最早开始到首个F/B包络开始实测得到，其中 iter85 的1.2522s被固定为v6.8 ENTRY。</p></div><span class="badge" id="sourceBadge"></span></div><div class="timeline" id="sourceTimeline"></div></section>
<section class="panel"><h2>同一 85–100 后半段开发集比较</h2><p class="muted">v6.7 和 v6.8 在下面使用完全相同的四个 iteration。v6.8 相对误差下降 <b>{metrics['same_late_development_improvement_pp']:.3f} 个百分点</b>（相对减少 {metrics['same_late_development_relative_reduction_pct']:.1f}%）；224卡数据只做封存后的开发评估。</p><div style="overflow-x:auto"><table id="accuracyTable"></table></div></section>
<section class="panel"><h2>iter60以后窗口敏感性诊断</h2><p class="muted">若“降到最低”指把<b>误差</b>降到最低：连续3点最低为iter{best3['start_iteration']}–{best3['end_iteration']}，MAPE <b>{best3['profiler_mape_pct']:.3f}%</b>；连续5点最低为iter{best5['start_iteration']}–{best5['end_iteration']}，MAPE <b>{best5['profiler_mape_pct']:.3f}%</b>。若严格排除iter60，连续3点最低为iter{strict3['start_iteration']}–{strict3['end_iteration']}，MAPE <b>{strict3['profiler_mape_pct']:.3f}%</b>。若原意是找<b>精度最低/误差最高</b>的连续3点，则为iter{worst3['start_iteration']}–{worst3['end_iteration']}，MAPE <b>{worst3['profiler_mape_pct']:.3f}%</b>。</p><div class="notice"><b>统计边界：</b>这些窗口是在看过224卡误差后选出的，只能说明误差随iteration漂移，不能替代预先固定的60–100评估集，也不能作为新的独立测试精度。</div></section>
<section class="scope"><div class="panel"><h2>v6.8 改了什么</h2><ul><li>新增一个耗时 {entry_ms:.3f} ms（{entry_ms/1000:.4f}s）的显式节点。</li><li>参数只来自256卡Trace iter85，不使用224卡时序拟合。</li><li>新增16条边，分别连接16个PP0 lane根节点。</li><li>没有调整F/B、DP/EDP、PP依赖或OISA service。</li></ul></div><div class="panel"><h2>结论边界</h2><ul><li>不能声称v6.8是独立的256→224盲测，因为224场景此前已被分析。</li><li>ENTRY只是实测时间边界，不直接等同于某个具体kernel或PP bubble。</li><li>当前改动仅补齐入口，不解决F/B包络和逐rank就绪传播问题。</li></ul></div></section>
<script>const DATA={data};const ns='http://www.w3.org/2000/svg';const $=id=>document.getElementById(id);const fmt=(v,n=2)=>Number(v).toFixed(n);let selected=String(DATA.default_iteration),depMode='pp';
function add(parent,name,attrs,text){{const e=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;parent.appendChild(e);return e}}
function visibleDeps(){{const deps=DATA.prediction.base_timeline.dependencies;if(depMode==='none')return[];if(depMode==='pp')return deps.filter(d=>d.kind==='pp_activation'||d.kind==='pp_gradient');return deps}}
function collectiveColor(d){{if(d.group_type==='dp')return d.collective==='rs'?'#27d17f':'#ffb54a';if(d.group_type==='expert_dp')return d.collective==='rs'?'#00b8a9':'#b78cff';return '#ff66c4'}}
function drawTimeline(target,base,entryMs,profilerMs,traceMode,label){{const W=1480,rowH=42,m={{l:62,r:24,t:58,b:38}},H=m.t+14*rowH+m.b,total=Math.ceil(Math.max(...DATA.evaluation.map(r=>r.actual_profiler_ms),DATA.prediction.v68_profiler_ms)/5000)*5000,x=v=>m.l+(W-m.l-m.r)*v/total,shift=entryMs;const svg=add(document.createElementNS(ns,'svg'),'g',{{}}).ownerSVGElement||document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);add(svg,'rect',{{x:m.l,y:19,width:W-m.l-m.r,height:22,fill:'#0a1727',stroke:'#2b4059'}});add(svg,'text',{{x:m.l-8,y:34,'text-anchor':'end',fill:'#93a8bd','font-size':11}},'ENTRY');if(entryMs>0){{const r=add(svg,'rect',{{x:x(0),y:22,width:Math.max(2,x(entryMs)-x(0)),height:16,rx:3,fill:'#f5a524',stroke:traceMode?'#ffe08a':'#ffe45e','stroke-width':traceMode?1:2}});const title=add(r,'title',{{}},`${{label}} Profiler入口 · 0–${{fmt(entryMs,3)}}ms · ProfilerStep开始到首个16-rank F/B包络`);r.appendChild(title);if(x(entryMs)-x(0)>52)add(svg,'text',{{x:x(0)+5,y:34,fill:'#241803','font-size':10,'font-weight':800}},`${{fmt(entryMs,1)}}ms`)}}else{{add(svg,'line',{{x1:x(0),y1:20,x2:x(0),y2:40,stroke:'#f5a524','stroke-width':3}});add(svg,'text',{{x:x(0)+5,y:34,fill:'#f5a524','font-size':10}},'未建模')}}for(let s=0;s<14;s++){{const y=m.t+s*rowH;add(svg,'rect',{{x:m.l,y,width:W-m.l-m.r,height:rowH,fill:s%2?'#0a1827':'#0c1b2c'}});add(svg,'text',{{x:m.l-8,y:y+24,'text-anchor':'end',fill:'#93a8bd','font-size':11}},`PP${{s}}`)}}for(let t=0;t<=total;t+=5000){{const xx=x(t);add(svg,'line',{{x1:xx,y1:15,x2:xx,y2:H-m.b,stroke:'#294057','stroke-dasharray':'3 4'}});add(svg,'text',{{x:xx,y:H-13,'text-anchor':'middle',fill:'#93a8bd','font-size':10}},`${{fmt(t/1000,0)}}s`)}}const pos=new Map();base.bars.forEach(d=>{{const start=d.start_ms+shift,end=d.end_ms+shift,yy=m.t+d.stage*rowH+(d.phase==='forward'?3:17),h=9,w=Math.max(1.5,x(end)-x(start));pos.set(d.id,{{start:x(start),end:x(end),y:yy+h/2}});const r=add(svg,'rect',{{x:x(start),y:yy,width:w,height:h,rx:2,fill:d.phase==='forward'?'#24c8ff':'#ff725c',stroke:!traceMode&&d.critical?'#ffe45e':'none','stroke-width':!traceMode&&d.critical?2:0}});add(r,'title',{{}},`${{label}} · PP${{d.stage}} ${{d.phase==='forward'?'F':'B'}}${{d.microbatch}} · ${{fmt(start/1000,3)}}–${{fmt(end/1000,3)}}s · 16-rank envelope ${{fmt(d.duration_ms,2)}}ms`);if(w>26)add(svg,'text',{{x:x(start)+3,y:yy+8,fill:'#07111d','font-size':7,'font-weight':800}},`${{d.phase==='forward'?'F':'B'}}${{d.microbatch}}`)}});base.collectives.filter(d=>d.group_type!=='handoff').forEach(d=>{{const start=d.start_ms+shift,end=d.end_ms+shift,yy=m.t+d.stage*rowH+(d.group_type==='dp'?30:36),h=5,r=add(svg,'rect',{{x:x(start),y:yy,width:Math.max(1.4,x(end)-x(start)),height:h,rx:1,fill:collectiveColor(d),stroke:!traceMode&&d.critical?'#ffe45e':'none','stroke-width':!traceMode&&d.critical?1.3:0}});add(r,'title',{{}},`${{label}} · PP${{d.stage}} ${{d.group_type==='dp'?'DP':'EDP'}} ${{String(d.collective).toUpperCase()}} · ${{fmt(start/1000,3)}}–${{fmt(end/1000,3)}}s · ${{fmt(d.duration_ms,2)}}ms`)}});if(!traceMode)visibleDeps().forEach(d=>{{const a=pos.get(d.source),b=pos.get(d.target);if(!a||!b)return;const color=d.kind==='pp_activation'?'#55d6be':d.kind==='pp_gradient'?'#ef79ff':'#ffb454',mid=(a.end+b.start)/2;const p=add(svg,'path',{{d:`M ${{a.end}} ${{a.y}} C ${{mid}} ${{a.y}}, ${{mid}} ${{b.y}}, ${{b.start}} ${{b.y}}`,fill:'none',stroke:color,'stroke-width':1.2,'stroke-opacity':.55}});add(p,'title',{{}},d.label)}});if(entryMs>0&&pos.has('s0:F0')){{const b=pos.get('s0:F0');add(svg,'path',{{d:`M ${{x(entryMs)}} 38 C ${{x(entryMs)}} 47, ${{b.start}} 47, ${{b.start}} ${{b.y}}`,fill:'none',stroke:'#f5a524','stroke-width':2,'stroke-dasharray':'4 3'}})}}add(svg,'line',{{x1:x(profilerMs),y1:15,x2:x(profilerMs),y2:H-m.b,stroke:'#fff','stroke-width':1.4}});add(svg,'text',{{x:x(profilerMs)-4,y:14,'text-anchor':'end',fill:'#fff','font-size':10}},`Profiler完成 ${{fmt(profilerMs/1000,3)}}s`);target.replaceChildren(svg)}}
function overlayCriticalPath(target,path){{const svg=target.querySelector('svg');if(!svg||svg.dataset.criticalPainted==='1')return;svg.dataset.criticalPainted='1';const W=1480,rowH=42,m={{l:62,r:24,t:58}},total=Math.ceil(Math.max(...DATA.evaluation.map(r=>r.actual_profiler_ms),DATA.prediction.v68_profiler_ms)/5000)*5000,x=v=>m.l+(W-m.l-m.r)*v/total,y=d=>d.track==='entry'?30:d.stage<0?48:m.t+d.stage*rowH+(d.track==='forward'?2:d.track==='backward'?16:d.track==='wait'?28:d.track==='p2p'?26:d.track==='optimizer'?36:39);const spans=path.spans;for(let i=1;i<spans.length;i++){{const a=spans[i-1],b=spans[i],ay=y(a),by=y(b),mid=(x(a.end_ms)+x(b.start_ms))/2,p=add(svg,'path',{{d:`M ${{x(a.end_ms)}} ${{ay}} C ${{mid}} ${{ay}}, ${{mid}} ${{by}}, ${{x(b.start_ms)}} ${{by}}`,fill:'none',stroke:'#ffe45e','stroke-width':2.2,'stroke-opacity':.8,'pointer-events':'stroke'}});add(p,'title',{{}},`最终最长路径依赖 · ${{a.label}} → ${{b.label}}`)}}for(const d of spans){{const yy=y(d),glow=add(svg,'line',{{x1:x(d.start_ms),y1:yy,x2:x(d.end_ms),y2:yy,stroke:'#fff3a0','stroke-width':7,'stroke-opacity':.22,'stroke-linecap':'round'}}),line=add(svg,'line',{{x1:x(d.start_ms),y1:yy,x2:x(d.end_ms),y2:yy,stroke:'#ffe45e','stroke-width':3.2,'stroke-linecap':'round','pointer-events':'stroke'}});add(glow,'title',{{}},`最终最长路径 · ${{d.label}}`);add(line,'title',{{}},`最终最长路径 · ${{d.label}} · ${{fmt(d.start_ms/1000,3)}}–${{fmt(d.end_ms/1000,3)}}s · ${{fmt(d.duration_ms,3)}}ms · ${{d.node_count}}节点`);if(d.track==='wait'&&d.duration_ms>500)add(svg,'text',{{x:x(d.start_ms)+5,y:yy-4,fill:'#ffe45e','font-size':9,'font-weight':800}},`${{fmt(d.duration_ms/1000,3)}}s 本地等待胜出`)}}if(path.post_graph_reconciliation_ms>0){{const yTail=48,line=add(svg,'line',{{x1:x(path.raw_graph_ms),y1:yTail,x2:x(path.profiler_ms),y2:yTail,stroke:'#ffe45e','stroke-width':2,'stroke-dasharray':'4 3'}});add(line,'title',{{}},`图外Profiler对齐 ${{fmt(path.post_graph_reconciliation_ms,3)}}ms；不属于DAG最长路径节点`)}}}}
function markDominantWinner(target,path){{const svg=target.querySelector('svg'),d=path.dominant_join_decision;if(!svg||!d)return;const W=1480,rowH=42,m={{l:62,r:24,t:58}},total=Math.ceil(Math.max(...DATA.evaluation.map(r=>r.actual_profiler_ms),DATA.prediction.v68_profiler_ms)/5000)*5000,x=v=>m.l+(W-m.l-m.r)*v/total,yy=m.t+d.target_stage*rowH+(d.target_phase==='FWD'?6:20);add(svg,'circle',{{cx:x(d.target_start_ms),cy:yy,r:5,fill:'#ffe45e',stroke:'#07111d','stroke-width':1.5}});const t=add(svg,'text',{{x:x(d.target_start_ms)+7,y:yy-7,fill:'#ffe45e','font-size':10,'font-weight':800}},`WIN +${{fmt(d.winning_margin_ms,1)}}ms`);add(t,'title',{{}},`${{d.winner}}比${{d.runner_up}}晚到${{fmt(d.winning_margin_ms,3)}}ms，因此成为最终关键前驱`)}}
function drawSourceTimeline(target,base,entryMs,profilerMs,label){{const W=1480,rowH=42,stageCount=16,m={{l:62,r:24,t:58,b:38}},H=m.t+stageCount*rowH+m.b,total=Math.ceil(Math.max(...Object.values(DATA.source_256_profiler_boundaries).map(r=>r.actual_profiler_ms),...DATA.evaluation.map(r=>r.actual_profiler_ms))/5000)*5000,x=v=>m.l+(W-m.l-m.r)*v/total,svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);add(svg,'rect',{{x:m.l,y:19,width:W-m.l-m.r,height:22,fill:'#0a1727',stroke:'#2b4059'}});add(svg,'text',{{x:m.l-8,y:34,'text-anchor':'end',fill:'#93a8bd','font-size':11}},'ENTRY');const entry=add(svg,'rect',{{x:x(0),y:22,width:Math.max(2,x(entryMs)-x(0)),height:16,rx:3,fill:'#f5a524',stroke:'#ffe08a'}});add(entry,'title',{{}},`${{label}} Profiler入口 · 0–${{fmt(entryMs,3)}}ms · 256卡自身实测边界`);if(x(entryMs)-x(0)>52)add(svg,'text',{{x:x(0)+5,y:34,fill:'#241803','font-size':10,'font-weight':800}},`${{fmt(entryMs,1)}}ms`);for(let s=0;s<stageCount;s++){{const y=m.t+s*rowH;add(svg,'rect',{{x:m.l,y,width:W-m.l-m.r,height:rowH,fill:s%2?'#0a1827':'#0c1b2c'}});add(svg,'text',{{x:m.l-8,y:y+24,'text-anchor':'end',fill:'#93a8bd','font-size':11}},`PP${{s}}`)}}for(let t=0;t<=total;t+=5000){{const xx=x(t);add(svg,'line',{{x1:xx,y1:15,x2:xx,y2:H-m.b,stroke:'#294057','stroke-dasharray':'3 4'}});add(svg,'text',{{x:xx,y:H-13,'text-anchor':'middle',fill:'#93a8bd','font-size':10}},`${{fmt(t/1000,0)}}s`)}}base.bars.forEach(d=>{{const start=d.start_ms+entryMs,end=d.end_ms+entryMs,yy=m.t+d.stage*rowH+(d.phase==='forward'?3:17),w=Math.max(1.5,x(end)-x(start)),r=add(svg,'rect',{{x:x(start),y:yy,width:w,height:9,rx:2,fill:d.phase==='forward'?'#24c8ff':'#ff725c'}});add(r,'title',{{}},`${{label}} · PP${{d.stage}} ${{d.phase==='forward'?'F':'B'}}${{d.microbatch}} · ${{fmt(start/1000,3)}}–${{fmt(end/1000,3)}}s · 16-rank envelope ${{fmt(d.duration_ms,2)}}ms`);if(w>26)add(svg,'text',{{x:x(start)+3,y:yy+8,fill:'#07111d','font-size':7,'font-weight':800}},`${{d.phase==='forward'?'F':'B'}}${{d.microbatch}}`)}});base.collectives.forEach(d=>{{const start=d.start_ms+entryMs,end=d.end_ms+entryMs,yy=m.t+d.stage*rowH+(d.group_type==='dp'?30:36),r=add(svg,'rect',{{x:x(start),y:yy,width:Math.max(1.4,x(end)-x(start)),height:5,rx:1,fill:collectiveColor(d)}});add(r,'title',{{}},`${{label}} · PP${{d.stage}} ${{d.group_type==='dp'?'DP':'EDP'}} ${{String(d.collective).toUpperCase()}} · ${{fmt(start/1000,3)}}–${{fmt(end/1000,3)}}s · ${{fmt(d.duration_ms,2)}}ms`)}});add(svg,'line',{{x1:x(profilerMs),y1:15,x2:x(profilerMs),y2:H-m.b,stroke:'#fff','stroke-width':1.4}});add(svg,'text',{{x:x(profilerMs)-4,y:14,'text-anchor':'end',fill:'#fff','font-size':10}},`Profiler完成 ${{fmt(profilerMs/1000,3)}}s`);target.replaceChildren(svg)}}
function table(){{const rows=DATA.evaluation.map(r=>`<tr class="${{String(r.iteration)===selected?'selected':''}}"><td>iter ${{r.iteration}} · ${{r.split==='target_development_early'?'前半段':'后半段'}}</td><td>${{fmt(r.actual_entry_ms,1)}} ms</td><td>${{fmt(r.actual_profiler_ms/1000,3)}} s</td><td>${{fmt(r.v67_ape_pct,3)}}%</td><td>${{fmt(r.v68_ape_pct,3)}}%</td><td>${{fmt(r.v67_ape_pct-r.v68_ape_pct,3)}} pp</td></tr>`).join('');$('accuracyTable').innerHTML='<thead><tr><th>Iteration</th><th>Trace入口</th><th>Trace Profiler</th><th>v6.7 APE</th><th>v6.8 APE</th><th>下降</th></tr></thead><tbody>'+rows+'</tbody>'}}
function render(){{const row=DATA.evaluation.find(r=>String(r.iteration)===selected),trace=DATA.trace[selected],source=DATA.source_256_trace[selected],sourceBoundary=DATA.source_256_profiler_boundaries[selected];drawTimeline($('v67Timeline'),DATA.prediction.base_timeline,0,DATA.prediction.v67_profiler_ms,false,'v6.7');drawTimeline($('v68Timeline'),DATA.prediction.base_timeline,DATA.entry_node.duration_ms,DATA.prediction.v68_profiler_ms,false,'v6.8');drawTimeline($('traceTimeline'),trace,row.actual_entry_ms,row.actual_profiler_ms,true,`224 Trace iter${{selected}}`);drawSourceTimeline($('sourceTimeline'),source,sourceBoundary.actual_entry_ms,sourceBoundary.actual_profiler_ms,`256 Trace iter${{selected}}`);$('traceBadge').textContent=`iter ${{selected}} · ${{row.split==='target_development_early'?'前半段开发评估':'后半段开发评估'}} · 实际入口 ${{fmt(row.actual_entry_ms,1)}}ms`;$('sourceBadge').textContent=`iter ${{selected}} · PP16×4 · 实际入口 ${{fmt(sourceBoundary.actual_entry_ms,1)}}ms`;for(const b of $('iterationControls').querySelectorAll('button'))b.classList.toggle('active',b.dataset.it===selected);for(const b of $('dependencyControls').querySelectorAll('button'))b.classList.toggle('active',b.dataset.dep===depMode);table()}}
function paintCriticalPaths(){{overlayCriticalPath($('v67Timeline'),DATA.prediction.v67_critical_path);markDominantWinner($('v67Timeline'),DATA.prediction.v67_critical_path);overlayCriticalPath($('v68Timeline'),DATA.prediction.v68_critical_path);markDominantWinner($('v68Timeline'),DATA.prediction.v68_critical_path)}}
$('iterationControls').innerHTML=DATA.iterations.map(it=>`<button class="${{it<=80?'cal':'hold'}}" data-it="${{it}}">iter ${{it}}</button>`).join('');$('iterationControls').addEventListener('click',e=>{{if(e.target.dataset.it){{selected=e.target.dataset.it;render();paintCriticalPaths()}}}});$('dependencyControls').addEventListener('click',e=>{{if(e.target.dataset.dep){{depMode=e.target.dataset.dep;render();paintCriticalPaths()}}}});render();paintCriticalPaths();</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v67-run-dir", type=Path, default=DEFAULT_V67)
    parser.add_argument("--v68-run-dir", type=Path, default=DEFAULT_V68)
    parser.add_argument("--source-event-windows", type=Path, default=DEFAULT_SOURCE_EVENT_WINDOWS)
    parser.add_argument("--source-phase-events", type=Path, default=DEFAULT_SOURCE_PHASE_EVENTS)
    args = parser.parse_args()
    v67 = args.v67_run_dir.resolve()
    v68 = args.v68_run_dir.resolve()
    payload, inputs = build_payload(
        v67, v68, args.source_event_windows.resolve(), args.source_phase_events.resolve()
    )
    output = v68 / "evaluator_only"
    payload_path = output / "dag_v68_profiler_entry_visualization_payload.json"
    html_path = output / "dag_v68_predicted_vs_trace_global_timeline.html"
    report_path = output / "DAG_V68_PROFILER_ENTRY_VISUALIZATION.md"
    evolution_path = output / "dag_v68_html_evolution_log.md"
    command_path = output / "visualization_reproduction_command.txt"
    provenance_path = output / "dag_v68_profiler_entry_visualization_provenance.json"
    manifest_path = output / "dag_v68_profiler_entry_visualization_manifest.json"
    report_window_diagnostic = payload["metrics"]["post60_window_diagnostic"]
    best3 = report_window_diagnostic["best_contiguous_3_point"]
    best5 = report_window_diagnostic["best_contiguous_5_point"]

    atomic_json(payload_path, payload)
    atomic_text(html_path, render_html(payload))
    atomic_text(report_path, f"""# DAG v6.8 256卡冻结ENTRY可视化

- 状态：`{payload['status']}`
- v6.7预测：`{payload['prediction']['v67_profiler_ms']:.6f} ms`
- v6.8入口节点：`{payload['entry_node']['duration_ms']:.6f} ms`
- v6.8预测：`{payload['prediction']['v68_profiler_ms']:.6f} ms`
- ENTRY来源：256卡Trace iteration `{payload['entry_node']['source_iteration']}`，页面固定显示 `{payload['entry_node']['duration_ms'] / 1000.0:.4f} s`
- 同一85–100后半段开发集：v6.7 MAPE `{payload['metrics']['v67_same_late_development_mape_pct']:.6f}%`，v6.8 MAPE `{payload['metrics']['v68_same_late_development_mape_pct']:.6f}%`
- 事后连续3点最低误差窗口：iter{best3['start_iteration']}–{best3['end_iteration']}，MAPE `{best3['profiler_mape_pct']:.6f}%`
- 事后连续5点最低误差窗口：iter{best5['start_iteration']}–{best5['end_iteration']}，MAPE `{best5['profiler_mape_pct']:.6f}%`
- DAG最长路径正时长覆盖率：`{payload['prediction']['v68_critical_path']['duration_coverage_pct']:.6f}%`

页面以ProfilerStep开始为共同零点，依次展示v6.7、v6.8、224卡Trace和256卡源Trace。v6.8 ENTRY直接冻结自256卡iter85；224卡Trace只在预测封存后用于开发评估。v6.8只新增入口节点与16条根依赖；F/B、PP、DP/EDP及service均未改变。

黄色最长路径从iteration完成节点逐个反查最终获胜前驱，覆盖计算、PP传输、本地等待、通信和optimizer正时长节点，并用黄色连接线补齐零时长依赖。PP4–PP13在PP3 B0汇合点之前完成，因此不是最终获胜分支。

因为224卡场景此前已被分析，本结果不声明为独立盲测；事后最低误差窗口也不能重新命名为测试集。
""")
    atomic_text(evolution_path, """# v6.8 HTML演进日志

- `v6.8-ui1`：沿用v6.7的PP14×3、F/B颜色、上下轨和DP/EDP细条；共同零点从首个FWD改为ProfilerStep开始。
- `v6.8-ui1`：并列显示v6.7无入口、v6.8显式入口与224 Trace实际入口。
- `v6.8-ui1`：增加85–100相同伪留出集的逐iteration误差表，避免跨不同子集直接比较。
- `v6.8-ui2`：增加D图256卡源Trace，按每个iteration自身的ProfilerStep→首个F/B实测入口平移PP16×4时序，不复用224入口参数。
- `v6.8-ui3`：B图ENTRY从224卡目标辅助中位数改为256卡Trace iter85固定值`1.2522s`；同步更新数据边界、误差表和版本说明。
- `v6.8-ui4`：从completion反向重建单条max-plus最长前驱链，将全部正时长段和连接依赖叠加为黄色；新增汇合点获胜前驱表，明确PP3本地等待为何压过PP4梯度分支。
- `v6.8-ui4`：增加iter60后连续窗口误差扫描；最低窗口只标记为事后诊断，不改变固定的60–100评估集。
""")
    atomic_text(command_path, (
        ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
        "build_dag_v68_profiler_entry_visualization.py\n"
    ))
    output_files = [payload_path, html_path, report_path, evolution_path, command_path]
    atomic_json(provenance_path, {
        "schema": "dag-v6.8-profiler-entry-visualization-provenance-v1",
        "status": payload["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_parameter_updates": 0,
        "blind_extrapolation_claim_allowed": False,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        ],
    })
    output_files.append(provenance_path)
    atomic_json(manifest_path, {
        "schema": "dag-v6.8-profiler-entry-visualization-manifest-v1",
        "status": payload["status"],
        "artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        ],
    })
    print(json.dumps({
        "status": payload["status"],
        "html": str(html_path),
        "entry_ms": payload["entry_node"]["duration_ms"],
        "v67_same_late_development_mape_pct": payload["metrics"]["v67_same_late_development_mape_pct"],
        "v68_same_late_development_mape_pct": payload["metrics"]["v68_same_late_development_mape_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal trace-driven DAG for a non-interleaved 1F1B pipeline.

The graph intentionally stops at the optimizer frontier.  Forward/backward
annotations provide compute-node durations; point-to-point service is a
replaceable FCT.  Pipeline waiting is an output of the dependency graph, not a
duration copied from a representative rank.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


TRACE_EVENT_COLUMNS = {
    "iteration",
    "rank",
    "pp_stage",
    "pp_lane",
    "phase",
    "microbatch",
    "duration_ns",
}


@dataclass(frozen=True)
class PipelineDagResult:
    """Materialized graph and its max-plus replay."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    rank_summary: pd.DataFrame
    lane_summary: pd.DataFrame
    optimizer_frontier: pd.DataFrame


def non_interleaved_1f1b_schedule(
    pp_size: int, microbatches: int, pp_stage: int
) -> pd.DataFrame:
    """Return Megatron's local operation order for one pipeline stage.

    With PP=16 and M=4 this produces no pipeline-wide steady plateau.  The
    later stages still execute a short local steady loop exactly as Megatron's
    schedule does.
    """
    if pp_size <= 0 or microbatches <= 0:
        raise ValueError("pp_size and microbatches must be positive")
    if not 0 <= pp_stage < pp_size:
        raise ValueError("pp_stage must be inside the pipeline")

    warmup = min(pp_size - pp_stage - 1, microbatches)
    remaining = microbatches - warmup
    operations: list[tuple[str, int, str]] = []
    operations.extend(("forward", mb, "warmup") for mb in range(warmup))
    for index in range(remaining):
        operations.append(("forward", warmup + index, "steady"))
        operations.append(("backward", index, "steady"))
    operations.extend(
        ("backward", mb, "cooldown") for mb in range(remaining, microbatches)
    )
    return pd.DataFrame(
        [
            {
                "pp_stage": pp_stage,
                "sequence": sequence,
                "phase": phase,
                "microbatch": microbatch,
                "region": region,
            }
            for sequence, (phase, microbatch, region) in enumerate(operations)
        ]
    )


def build_schedule_template(pp_size: int, microbatches: int) -> pd.DataFrame:
    """Return the local operation order for every stage."""
    return pd.concat(
        [
            non_interleaved_1f1b_schedule(pp_size, microbatches, stage)
            for stage in range(pp_size)
        ],
        ignore_index=True,
    )


def validate_trace_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate one iteration of per-rank FWD/BWD trace annotations."""
    missing = sorted(TRACE_EVENT_COLUMNS - set(events.columns))
    if missing:
        raise ValueError(f"pipeline trace events missing columns: {missing}")
    optional = [
        column
        for column in ("observed_start_ns", "observed_end_ns", "source_path")
        if column in events
    ]
    work = events[[*sorted(TRACE_EVENT_COLUMNS), *optional]].copy()
    numeric = [
        "iteration",
        "rank",
        "pp_stage",
        "pp_lane",
        "microbatch",
        "duration_ns",
    ]
    numeric.extend(
        column
        for column in ("observed_start_ns", "observed_end_ns")
        if column in work
    )
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="raise")
    if work[numeric].isna().any().any() or (~np.isfinite(work[numeric])).any().any():
        raise ValueError("pipeline trace numeric values must be finite")
    if work["iteration"].nunique() != 1:
        raise ValueError("build one pipeline DAG iteration at a time")
    if not set(work["phase"]).issubset({"forward", "backward"}):
        raise ValueError("pipeline phase must be forward or backward")
    if (work["duration_ns"] <= 0).any():
        raise ValueError("pipeline compute durations must be positive")
    key = ["rank", "phase", "microbatch"]
    if work.duplicated(key).any():
        raise ValueError("duplicate rank/phase/microbatch trace event")

    pp_size = int(work["pp_stage"].max()) + 1
    microbatches = int(work["microbatch"].max()) + 1
    expected_stages = set(range(pp_size))
    if set(work["pp_stage"].astype(int)) != expected_stages:
        raise ValueError("pipeline stages must be dense from zero")
    expected_events = 2 * microbatches
    counts = work.groupby("rank").size()
    if not counts.eq(expected_events).all():
        raise ValueError("every rank must have one FWD and BWD event per microbatch")
    phase_grid = work.groupby(["rank", "phase"])["microbatch"].agg(set)
    expected_microbatches = set(range(microbatches))
    if not phase_grid.map(lambda values: values == expected_microbatches).all():
        raise ValueError("rank phase microbatch grid is incomplete")
    rank_topology = work.groupby("rank")[["pp_stage", "pp_lane"]].nunique()
    if not rank_topology.eq(1).all().all():
        raise ValueError("rank topology changes inside the trace event table")
    lane_stages = (
        work[["rank", "pp_stage", "pp_lane"]]
        .drop_duplicates()
        .groupby("pp_lane")["pp_stage"]
        .agg(set)
    )
    if not lane_stages.map(lambda values: values == expected_stages).all():
        raise ValueError("every PP lane must cover every pipeline stage")
    if "observed_end_ns" in work and not np.allclose(
        work["observed_end_ns"] - work["observed_start_ns"],
        work["duration_ns"],
        rtol=0,
        atol=1,
    ):
        raise ValueError("observed event interval does not conserve duration")
    return work.sort_values(["pp_lane", "pp_stage", "phase", "microbatch"]).reset_index(
        drop=True
    )


def _compute_id(rank: int, phase: str, microbatch: int) -> str:
    letter = "F" if phase == "forward" else "B"
    return f"rank{rank}:{letter}{microbatch}"


def _message_id(
    lane: int, phase: str, microbatch: int, src_stage: int, dst_stage: int
) -> str:
    letter = "F" if phase == "forward" else "B"
    return f"lane{lane}:{letter}{microbatch}:s{src_stage}->s{dst_stage}"


def _max_plus_replay(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    duration = nodes.set_index("node_id")["duration_ns"].astype(int).to_dict()
    node_ids = set(duration)
    if not set(edges["src"]).issubset(node_ids) or not set(edges["dst"]).issubset(node_ids):
        raise ValueError("DAG edge references an unknown node")

    successors: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    predecessors: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges.itertuples(index=False):
        successors[edge.src].append(edge.dst)
        predecessors[edge.dst].append(edge.src)
        indegree[edge.dst] += 1

    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    critical_predecessor: dict[str, str] = {}
    visited = 0
    while ready:
        node_id = heapq.heappop(ready)
        visited += 1
        if predecessors[node_id]:
            predecessor = max(predecessors[node_id], key=lambda item: (ends[item], item))
            start = ends[predecessor]
            critical_predecessor[node_id] = predecessor
        else:
            start = 0
            critical_predecessor[node_id] = ""
        starts[node_id] = start
        ends[node_id] = start + duration[node_id]
        for successor in successors[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if visited != len(node_ids):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"pipeline dependency graph contains a cycle: {cyclic[:5]}")

    result = nodes.copy()
    result["predicted_start_ns"] = result["node_id"].map(starts).astype("int64")
    result["predicted_end_ns"] = result["node_id"].map(ends).astype("int64")
    result["critical_predecessor"] = result["node_id"].map(critical_predecessor)
    return result


def build_pipeline_dag(
    events: pd.DataFrame,
    pp_service_ns: int | float,
    pp_software_completion_ns: Mapping[str, int | float] | None = None,
) -> PipelineDagResult:
    """Build and replay the minimal PP DAG for one captured iteration.

    Point-to-point sends are blocking: the sender's next compute node waits for
    the outgoing message, and the receiver waits for its payload/gradient.
    ``pp_service_ns`` remains network service.  Optional direction-specific
    software completion is added as a separate component, preserving the
    distinction needed for network counterfactuals.
    """
    facts = validate_trace_events(events)
    if not np.isfinite(pp_service_ns) or pp_service_ns < 0:
        raise ValueError("pp_service_ns must be finite and non-negative")
    pp_service = int(round(pp_service_ns))
    software_completion = {"forward": 0, "backward": 0}
    if pp_software_completion_ns:
        unknown = sorted(set(pp_software_completion_ns) - set(software_completion))
        if unknown:
            raise ValueError(f"unknown PP software-completion directions: {unknown}")
        for direction, value in pp_software_completion_ns.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "PP software-completion durations must be finite and non-negative"
                )
            software_completion[str(direction)] = int(round(value))
    pp_size = int(facts["pp_stage"].max()) + 1
    microbatches = int(facts["microbatch"].max()) + 1
    iteration = int(facts["iteration"].iloc[0])
    schedule = build_schedule_template(pp_size, microbatches)
    schedule_lookup = {
        (int(row.pp_stage), str(row.phase), int(row.microbatch)): (
            int(row.sequence),
            str(row.region),
        )
        for row in schedule.itertuples(index=False)
    }

    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, str]] = []
    compute_order: dict[int, list[str]] = {}
    compute_position: dict[str, int] = {}
    topology = facts[["rank", "pp_stage", "pp_lane"]].drop_duplicates()
    rank_for = {
        (int(row.pp_stage), int(row.pp_lane)): int(row.rank)
        for row in topology.itertuples(index=False)
    }

    for row in facts.itertuples(index=False):
        rank = int(row.rank)
        stage = int(row.pp_stage)
        lane = int(row.pp_lane)
        phase = str(row.phase)
        microbatch = int(row.microbatch)
        sequence, region = schedule_lookup[(stage, phase, microbatch)]
        node = {
            "iteration": iteration,
            "node_id": _compute_id(rank, phase, microbatch),
            "kind": "compute",
            "phase": phase,
            "microbatch": microbatch,
            "rank": rank,
            "pp_stage": stage,
            "pp_lane": lane,
            "src_rank": rank,
            "dst_rank": rank,
            "src_stage": stage,
            "dst_stage": stage,
            "sequence": sequence,
            "region": region,
            "duration_ns": int(row.duration_ns),
            "network_service_ns": 0,
            "software_completion_ns": 0,
        }
        for optional in ("observed_start_ns", "observed_end_ns", "source_path"):
            if hasattr(row, optional):
                node[optional] = getattr(row, optional)
        node_rows.append(node)

    for rank, frame in facts.groupby("rank"):
        ordered = sorted(
            (
                schedule_lookup[(int(row.pp_stage), str(row.phase), int(row.microbatch))][0],
                _compute_id(int(rank), str(row.phase), int(row.microbatch)),
            )
            for row in frame.itertuples(index=False)
        )
        compute_order[int(rank)] = [node_id for _, node_id in ordered]
        compute_position.update({node_id: index for index, node_id in enumerate(compute_order[int(rank)])})
        for (_, previous), (_, current) in zip(ordered, ordered[1:]):
            edge_rows.append(
                {"src": previous, "dst": current, "edge_type": "local_compute_order"}
            )

    lanes = sorted(topology["pp_lane"].astype(int).unique())
    for lane in lanes:
        for microbatch in range(microbatches):
            for src_stage in range(pp_size - 1):
                dst_stage = src_stage + 1
                src_rank = rank_for[(src_stage, lane)]
                dst_rank = rank_for[(dst_stage, lane)]
                compute_src = _compute_id(src_rank, "forward", microbatch)
                compute_dst = _compute_id(dst_rank, "forward", microbatch)
                message = _message_id(
                    lane, "forward", microbatch, src_stage, dst_stage
                )
                node_rows.append(
                    {
                        "iteration": iteration,
                        "node_id": message,
                        "kind": "pp_message",
                        "phase": "forward",
                        "microbatch": microbatch,
                        "rank": -1,
                        "pp_stage": src_stage,
                        "pp_lane": lane,
                        "src_rank": src_rank,
                        "dst_rank": dst_rank,
                        "src_stage": src_stage,
                        "dst_stage": dst_stage,
                        "sequence": -1,
                        "region": "p2p",
                        "duration_ns": pp_service + software_completion["forward"],
                        "network_service_ns": pp_service,
                        "software_completion_ns": software_completion["forward"],
                    }
                )
                edge_rows.extend(
                    [
                        {
                            "src": compute_src,
                            "dst": message,
                            "edge_type": "forward_payload_ready",
                        },
                        {
                            "src": message,
                            "dst": compute_dst,
                            "edge_type": "forward_receive",
                        },
                    ]
                )
                position = compute_position[compute_src]
                order = compute_order[src_rank]
                if position + 1 < len(order):
                    edge_rows.append(
                        {
                            "src": message,
                            "dst": order[position + 1],
                            "edge_type": "blocking_forward_send",
                        }
                    )

            for src_stage in range(pp_size - 1, 0, -1):
                dst_stage = src_stage - 1
                src_rank = rank_for[(src_stage, lane)]
                dst_rank = rank_for[(dst_stage, lane)]
                compute_src = _compute_id(src_rank, "backward", microbatch)
                compute_dst = _compute_id(dst_rank, "backward", microbatch)
                message = _message_id(
                    lane, "backward", microbatch, src_stage, dst_stage
                )
                node_rows.append(
                    {
                        "iteration": iteration,
                        "node_id": message,
                        "kind": "pp_message",
                        "phase": "backward",
                        "microbatch": microbatch,
                        "rank": -1,
                        "pp_stage": src_stage,
                        "pp_lane": lane,
                        "src_rank": src_rank,
                        "dst_rank": dst_rank,
                        "src_stage": src_stage,
                        "dst_stage": dst_stage,
                        "sequence": -1,
                        "region": "p2p",
                        "duration_ns": pp_service + software_completion["backward"],
                        "network_service_ns": pp_service,
                        "software_completion_ns": software_completion["backward"],
                    }
                )
                edge_rows.extend(
                    [
                        {
                            "src": compute_src,
                            "dst": message,
                            "edge_type": "backward_gradient_ready",
                        },
                        {
                            "src": message,
                            "dst": compute_dst,
                            "edge_type": "backward_receive",
                        },
                    ]
                )
                position = compute_position[compute_src]
                order = compute_order[src_rank]
                if position + 1 < len(order):
                    edge_rows.append(
                        {
                            "src": message,
                            "dst": order[position + 1],
                            "edge_type": "blocking_backward_send",
                        }
                    )

    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(edge_rows).drop_duplicates().reset_index(drop=True)
    nodes = _max_plus_replay(nodes, edges)

    local_previous_end: dict[str, int] = {}
    for rank, order in compute_order.items():
        previous_end = 0
        for node_id in order:
            local_previous_end[node_id] = previous_end
            previous_end = int(
                nodes.loc[nodes["node_id"].eq(node_id), "predicted_end_ns"].iloc[0]
            )
    nodes["local_wait_ns"] = 0
    compute_mask = nodes["kind"].eq("compute")
    nodes.loc[compute_mask, "local_wait_ns"] = (
        nodes.loc[compute_mask, "predicted_start_ns"]
        - nodes.loc[compute_mask, "node_id"].map(local_previous_end)
    )

    compute_nodes = nodes[compute_mask].copy()
    rank_summary = (
        compute_nodes.groupby(["iteration", "rank", "pp_stage", "pp_lane"], as_index=False)
        .agg(
            compute_ns=("duration_ns", "sum"),
            pipeline_wait_ns=("local_wait_ns", "sum"),
            predicted_first_compute_ns=("predicted_start_ns", "min"),
            predicted_backward_done_ns=("predicted_end_ns", "max"),
        )
        .sort_values(["pp_lane", "pp_stage"])
        .reset_index(drop=True)
    )
    rank_summary["front_envelope_ns"] = (
        rank_summary["predicted_backward_done_ns"]
        - rank_summary["predicted_first_compute_ns"]
    )
    rank_summary["wait_share_pct"] = np.where(
        rank_summary["front_envelope_ns"] > 0,
        100.0 * rank_summary["pipeline_wait_ns"] / rank_summary["front_envelope_ns"],
        0.0,
    )
    if {"observed_start_ns", "observed_end_ns"}.issubset(facts.columns):
        observed_rank = (
            facts.groupby(["iteration", "rank", "pp_stage", "pp_lane"], as_index=False)
            .agg(
                observed_first_compute_ns=("observed_start_ns", "min"),
                observed_backward_done_ns=("observed_end_ns", "max"),
            )
        )
        observed_rank["observed_front_envelope_ns"] = (
            observed_rank["observed_backward_done_ns"]
            - observed_rank["observed_first_compute_ns"]
        )
        rank_summary = rank_summary.merge(
            observed_rank,
            on=["iteration", "rank", "pp_stage", "pp_lane"],
            validate="one_to_one",
        )
        rank_summary["observed_local_gap_ns"] = (
            rank_summary["observed_front_envelope_ns"] - rank_summary["compute_ns"]
        )
        rank_summary["unmodeled_local_gap_ns"] = (
            rank_summary["observed_local_gap_ns"] - rank_summary["pipeline_wait_ns"]
        )

    lane_summary = (
        rank_summary.groupby(["iteration", "pp_lane"], as_index=False)
        .agg(
            predicted_front_start_ns=("predicted_first_compute_ns", "min"),
            predicted_optimizer_ready_ns=("predicted_backward_done_ns", "max"),
            rank_compute_ns_sum=("compute_ns", "sum"),
            rank_pipeline_wait_ns_sum=("pipeline_wait_ns", "sum"),
        )
        .sort_values("pp_lane")
        .reset_index(drop=True)
    )
    lane_summary["predicted_front_makespan_ns"] = (
        lane_summary["predicted_optimizer_ready_ns"]
        - lane_summary["predicted_front_start_ns"]
    )
    if {"observed_start_ns", "observed_end_ns"}.issubset(facts.columns):
        observed_lane = (
            facts.groupby(["iteration", "pp_lane"], as_index=False)
            .agg(
                observed_front_start_ns=("observed_start_ns", "min"),
                observed_optimizer_ready_ns=("observed_end_ns", "max"),
            )
        )
        observed_lane["observed_front_makespan_ns"] = (
            observed_lane["observed_optimizer_ready_ns"]
            - observed_lane["observed_front_start_ns"]
        )
        lane_summary = lane_summary.merge(
            observed_lane, on=["iteration", "pp_lane"], validate="one_to_one"
        )
        lane_summary["front_replay_error_ns"] = (
            lane_summary["predicted_front_makespan_ns"]
            - lane_summary["observed_front_makespan_ns"]
        )
        lane_summary["front_replay_error_pct"] = (
            100.0
            * lane_summary["front_replay_error_ns"]
            / lane_summary["observed_front_makespan_ns"]
        )

    backward_last = compute_nodes[compute_nodes["phase"].eq("backward")].sort_values(
        ["rank", "sequence"]
    ).groupby("rank", as_index=False).tail(1)
    optimizer_frontier = backward_last[
        [
            "iteration",
            "rank",
            "pp_stage",
            "pp_lane",
            "node_id",
            "predicted_end_ns",
        ]
    ].rename(
        columns={
            "node_id": "predecessor_node_id",
            "predicted_end_ns": "predicted_optimizer_ready_ns",
        }
    )
    optimizer_frontier["next_node_contract"] = "first_optimizer_collective_or_local_tail"

    return PipelineDagResult(
        nodes=nodes.sort_values(["pp_lane", "predicted_start_ns", "node_id"]).reset_index(
            drop=True
        ),
        edges=edges.sort_values(["src", "dst", "edge_type"]).reset_index(drop=True),
        rank_summary=rank_summary,
        lane_summary=lane_summary,
        optimizer_frontier=optimizer_frontier.sort_values("rank").reset_index(drop=True),
    )

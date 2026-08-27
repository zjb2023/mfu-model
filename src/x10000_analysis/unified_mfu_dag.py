"""Join the trace-driven PP front graph to the optimizer RS/AG timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .mfu_timeline import TimelineResult, build_optimizer_timeline_model
from .pp_dag import PipelineDagResult, build_pipeline_dag


@dataclass(frozen=True)
class UnifiedDagResult:
    pipeline: PipelineDagResult
    optimizer: TimelineResult
    front_anchors: pd.DataFrame
    pp_nodes_absolute: pd.DataFrame
    combined_timeline: pd.DataFrame
    dependency_edges: pd.DataFrame
    iteration: pd.DataFrame


def _optimizer_node_id(row: object) -> str:
    safe_group = str(row.group_key).replace(":", "_").replace("=", "_")
    return (
        f"opt:{row.kind}:round{int(row.round)}:{safe_group}:rank{int(row.rank)}"
    )


def build_unified_mfu_dag(
    trace_events: pd.DataFrame,
    pp_service_ns: int | float,
    optimizer_calls: pd.DataFrame,
    clocks: pd.DataFrame,
    model_flops_per_iteration: float,
    world_size: int,
    peak_tflops_per_gpu: float,
    optimizer_service_scales: Mapping[str, float] | None = None,
    pp_software_completion_ns: Mapping[str, int | float] | None = None,
) -> UnifiedDagResult:
    """Build one iteration from PP compute through optimizer RS/AG completion."""
    if model_flops_per_iteration <= 0 or world_size <= 0 or peak_tflops_per_gpu <= 0:
        raise ValueError("FLOPs, world size and peak throughput must be positive")
    pipeline = build_pipeline_dag(
        trace_events,
        pp_service_ns,
        pp_software_completion_ns=pp_software_completion_ns,
    )
    iteration = int(trace_events["iteration"].iloc[0])
    selected_calls = optimizer_calls[optimizer_calls["iteration"].eq(iteration)].copy()
    selected_clocks = clocks[clocks["iteration"].eq(iteration)].copy()
    if selected_calls.empty or len(selected_clocks) != 1:
        raise ValueError(f"missing optimizer calls or unique clock for iteration {iteration}")

    topology = trace_events[["rank", "pp_stage", "pp_lane"]].drop_duplicates()
    if topology["rank"].nunique() != world_size:
        raise ValueError("trace topology does not match configured world size")
    lane_origins = (
        trace_events.groupby("pp_lane", as_index=False)["observed_start_ns"]
        .min()
        .rename(columns={"observed_start_ns": "predicted_lane_origin_ns"})
    )

    pp_nodes = pipeline.nodes.copy()
    pp_nodes = pp_nodes.rename(
        columns={
            "predicted_start_ns": "predicted_relative_start_ns",
            "predicted_end_ns": "predicted_relative_end_ns",
        }
    ).merge(lane_origins, on="pp_lane", validate="many_to_one")
    pp_nodes["predicted_start_ns"] = (
        pp_nodes["predicted_lane_origin_ns"]
        + pp_nodes["predicted_relative_start_ns"]
    )
    pp_nodes["predicted_end_ns"] = (
        pp_nodes["predicted_lane_origin_ns"]
        + pp_nodes["predicted_relative_end_ns"]
    )

    last_microbatch = int(trace_events["microbatch"].max())
    observed_last_bwd = trace_events[
        trace_events["phase"].eq("backward")
        & trace_events["microbatch"].eq(last_microbatch)
    ][["rank", "observed_end_ns"]].rename(
        columns={"observed_end_ns": "observed_backward_done_ns"}
    )
    first_dense_rs = (
        selected_calls[selected_calls["kind"].eq("dp_rs")]
        .groupby("rank", as_index=False)["start_ns"]
        .min()
        .rename(columns={"start_ns": "observed_first_rs_start_ns"})
    )
    frontier = (
        pipeline.optimizer_frontier.rename(
            columns={
                "predicted_optimizer_ready_ns": "predicted_backward_done_relative_ns"
            }
        )
        .merge(lane_origins, on="pp_lane", validate="many_to_one")
        .merge(observed_last_bwd, on="rank", validate="one_to_one")
        .merge(first_dense_rs, on="rank", validate="one_to_one")
    )
    frontier["predicted_backward_done_ns"] = (
        frontier["predicted_lane_origin_ns"]
        + frontier["predicted_backward_done_relative_ns"]
    )
    frontier["measured_tail_launch_lag_ns"] = (
        frontier["observed_first_rs_start_ns"]
        - frontier["observed_backward_done_ns"]
    ).clip(lower=0)
    frontier["predicted_start_ns"] = (
        frontier["predicted_backward_done_ns"]
        + frontier["measured_tail_launch_lag_ns"]
    )
    front_anchors = frontier[
        [
            "iteration",
            "rank",
            "pp_stage",
            "pp_lane",
            "predecessor_node_id",
            "predicted_lane_origin_ns",
            "predicted_backward_done_relative_ns",
            "predicted_backward_done_ns",
            "observed_backward_done_ns",
            "observed_first_rs_start_ns",
            "measured_tail_launch_lag_ns",
            "predicted_start_ns",
        ]
    ].sort_values("rank").reset_index(drop=True)

    optimizer = build_optimizer_timeline_model(
        selected_calls,
        selected_clocks,
        service_scales=optimizer_service_scales,
        front_anchors=front_anchors[["iteration", "rank", "predicted_start_ns"]],
    )
    optimizer_calls_predicted = optimizer.calls.merge(
        topology, on=["rank", "pp_stage"], validate="many_to_one"
    )
    optimizer_calls_predicted["node_id"] = [
        _optimizer_node_id(row)
        for row in optimizer_calls_predicted.itertuples(index=False)
    ]

    combined_rows: list[pd.DataFrame] = []
    pp_timeline = pp_nodes.copy()
    pp_timeline["source_model"] = "pipeline"
    pp_timeline["category"] = pp_timeline["phase"]
    pp_timeline["group_key"] = ""
    pp_timeline["dependency"] = pp_timeline["critical_predecessor"]
    combined_rows.append(
        pp_timeline[
            [
                "iteration",
                "node_id",
                "source_model",
                "kind",
                "category",
                "rank",
                "pp_stage",
                "pp_lane",
                "microbatch",
                "predicted_start_ns",
                "predicted_end_ns",
                "duration_ns",
                "network_service_ns",
                "software_completion_ns",
                "group_key",
                "dependency",
            ]
        ]
    )

    tail = front_anchors.copy()
    tail["node_id"] = tail["rank"].map(lambda rank: f"tail:rank{int(rank)}")
    tail["source_model"] = "pipeline_optimizer_bridge"
    tail["kind"] = "software_tail"
    tail["category"] = "optimizer_tail"
    tail["microbatch"] = last_microbatch
    tail["predicted_end_ns"] = tail["predicted_start_ns"]
    tail["predicted_start_ns"] = tail["predicted_backward_done_ns"]
    tail["duration_ns"] = tail["predicted_end_ns"] - tail["predicted_start_ns"]
    tail["network_service_ns"] = 0
    tail["software_completion_ns"] = tail["duration_ns"]
    tail["group_key"] = ""
    tail["dependency"] = tail["predecessor_node_id"]
    combined_rows.append(
        tail[
            [
                "iteration",
                "node_id",
                "source_model",
                "kind",
                "category",
                "rank",
                "pp_stage",
                "pp_lane",
                "microbatch",
                "predicted_start_ns",
                "predicted_end_ns",
                "duration_ns",
                "network_service_ns",
                "software_completion_ns",
                "group_key",
                "dependency",
            ]
        ]
    )

    opt_timeline = optimizer_calls_predicted.copy()
    opt_timeline["source_model"] = "optimizer"
    opt_timeline["category"] = opt_timeline["kind"]
    opt_timeline["microbatch"] = -1
    opt_timeline["duration_ns"] = (
        opt_timeline["predicted_end_ns"] - opt_timeline["predicted_start_ns"]
    )
    opt_timeline["network_service_ns"] = 0
    opt_timeline["software_completion_ns"] = 0
    opt_timeline["kind"] = "optimizer_collective"
    combined_rows.append(
        opt_timeline[
            [
                "iteration",
                "node_id",
                "source_model",
                "kind",
                "category",
                "rank",
                "pp_stage",
                "pp_lane",
                "microbatch",
                "predicted_start_ns",
                "predicted_end_ns",
                "duration_ns",
                "network_service_ns",
                "software_completion_ns",
                "group_key",
                "dependency",
            ]
        ]
    )
    world_rs_done_ns = int(
        optimizer_calls_predicted.loc[
            optimizer_calls_predicted["kind"].isin({"dp_rs", "edp_rs"}),
            "predicted_end_ns",
        ].max()
    )
    combined_rows.append(
        pd.DataFrame(
            [
                {
                    "iteration": iteration,
                    "node_id": "virtual:world_rs_done",
                    "source_model": "optimizer",
                    "kind": "virtual_join",
                    "category": "world_rs_done",
                    "rank": -1,
                    "pp_stage": -1,
                    "pp_lane": -1,
                    "microbatch": -1,
                    "predicted_start_ns": world_rs_done_ns,
                    "predicted_end_ns": world_rs_done_ns,
                    "duration_ns": 0,
                    "network_service_ns": 0,
                    "software_completion_ns": 0,
                    "group_key": "world",
                    "dependency": "all_rs_calls",
                }
            ]
        )
    )
    combined = pd.concat(combined_rows, ignore_index=True).sort_values(
        ["pp_lane", "predicted_start_ns", "node_id"]
    ).reset_index(drop=True)

    optimizer_node_for = {
        (str(row.kind), int(row.rank)): str(row.node_id)
        for row in optimizer_calls_predicted.itertuples(index=False)
    }
    dependency_rows = pipeline.edges.to_dict(orient="records")
    for row in front_anchors.itertuples(index=False):
        tail_id = f"tail:rank{int(row.rank)}"
        dependency_rows.append(
            {
                "src": str(row.predecessor_node_id),
                "dst": tail_id,
                "edge_type": "backward_to_optimizer_tail",
            }
        )
        dependency_rows.append(
            {
                "src": tail_id,
                "dst": optimizer_node_for[("dp_rs", int(row.rank))],
                "edge_type": "pp_frontier_to_dense_rs",
            }
        )
    for row in optimizer_calls_predicted.itertuples(index=False):
        dependency = str(row.dependency)
        if dependency.startswith("same_rank_"):
            previous_kind = dependency.removeprefix("same_rank_")
            dependency_rows.append(
                {
                    "src": optimizer_node_for[(previous_kind, int(row.rank))],
                    "dst": str(row.node_id),
                    "edge_type": dependency,
                }
            )
        elif dependency.startswith("stage_fallback_"):
            previous_kind = dependency.removeprefix("stage_fallback_")
            candidates = optimizer_calls_predicted[
                optimizer_calls_predicted["kind"].eq(previous_kind)
                & optimizer_calls_predicted["pp_stage"].eq(int(row.pp_stage))
            ]
            previous = candidates.loc[candidates["predicted_end_ns"].idxmax()]
            dependency_rows.append(
                {
                    "src": str(previous["node_id"]),
                    "dst": str(row.node_id),
                    "edge_type": dependency,
                }
            )
        elif dependency == "world_rs_done":
            dependency_rows.append(
                {
                    "src": "virtual:world_rs_done",
                    "dst": str(row.node_id),
                    "edge_type": dependency,
                }
            )
    dependency_rows.extend(
        {
            "src": str(row.node_id),
            "dst": "virtual:world_rs_done",
            "edge_type": "world_rs_join",
        }
        for row in optimizer_calls_predicted[
            optimizer_calls_predicted["kind"].isin({"dp_rs", "edp_rs"})
        ].itertuples(index=False)
    )
    dependency_edges = pd.DataFrame(dependency_rows).drop_duplicates().reset_index(drop=True)

    clock = selected_clocks.iloc[0]
    step_start_ns = int(selected_clocks["step_start_ns"].iloc[0])
    actual_profiler_ns = int(selected_clocks["profiler_step_ns"].iloc[0])
    actual_training_ns = int(selected_clocks["training_log_ns"].iloc[0])
    metrics = optimizer.iterations.iloc[0]
    predicted_training_ns = int(round(float(metrics["predicted_training_log_ms"]) * 1e6))
    predicted_mfu = 100.0 * model_flops_per_iteration / (
        predicted_training_ns / 1e9
        * world_size
        * peak_tflops_per_gpu
        * 1e12
    )
    actual_mfu = 100.0 * model_flops_per_iteration / (
        actual_training_ns / 1e9
        * world_size
        * peak_tflops_per_gpu
        * 1e12
    )
    pp_observed_p50 = float(
        pipeline.lane_summary["observed_front_makespan_ns"].median() / 1e6
    )
    pp_predicted_p50 = float(
        pipeline.lane_summary["predicted_front_makespan_ns"].median() / 1e6
    )
    iteration_frame = pd.DataFrame(
        [
            {
                "iteration": iteration,
                "step_start_ns": step_start_ns,
                "observed_profiler_step_ms": actual_profiler_ns / 1e6,
                "predicted_profiler_step_ms": float(metrics["predicted_profiler_step_ms"]),
                "profiler_replay_error_ms": float(metrics["profiler_replay_error_ms"]),
                "profiler_replay_error_pct": 100.0
                * float(metrics["profiler_replay_error_ms"])
                / (actual_profiler_ns / 1e6),
                "observed_training_log_ms": actual_training_ns / 1e6,
                "predicted_training_log_ms": predicted_training_ns / 1e6,
                "training_log_replay_error_ms": (predicted_training_ns - actual_training_ns)
                / 1e6,
                "actual_mfu_pct_fixed_flops": actual_mfu,
                "predicted_mfu_pct": predicted_mfu,
                "pp_front_observed_ms_p50": pp_observed_p50,
                "pp_front_predicted_ms_p50": pp_predicted_p50,
                "pp_front_replay_error_pct_p50": 100.0
                * (pp_predicted_p50 - pp_observed_p50)
                / pp_observed_p50,
                "first_rs_offset_ms": float(metrics["first_rs_offset_ms"]),
                "gradient_rs_done_offset_ms": float(metrics["gradient_rs_done_offset_ms"]),
                "first_ag0_offset_ms": float(metrics["first_ag0_offset_ms"]),
                "last_ag_offset_ms": float(metrics["last_ag_offset_ms"]),
                "all_dp_service_exposed_ms": float(metrics["all_dp_service_exposed_ms"]),
                "model_flops_per_iteration": float(model_flops_per_iteration),
                "world_size": world_size,
                "peak_tflops_per_gpu": peak_tflops_per_gpu,
            }
        ]
    )
    if not np.isfinite(iteration_frame.select_dtypes(include=[np.number])).all().all():
        raise ValueError("unified iteration metrics must be finite")
    return UnifiedDagResult(
        pipeline=pipeline,
        optimizer=optimizer,
        front_anchors=front_anchors,
        pp_nodes_absolute=pp_nodes,
        combined_timeline=combined,
        dependency_edges=dependency_edges,
        iteration=iteration_frame,
    )

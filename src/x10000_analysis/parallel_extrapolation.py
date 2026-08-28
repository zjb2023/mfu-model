"""Trace-preserving helpers for pipeline-parallel strategy extrapolation.

The captured 256-GPU case has 16 pipeline stages and 16 independent pipeline
lanes.  A 224-GPU ``PP14 / CP2 / DP8 / EP8`` strategy preserves the lane and
optimizer communicator shapes, but it still needs a model for repartitioning
the work of 16 captured stages over 14 target stages.  This module makes that
assumption explicit and auditable:

* choose 14 of the 16 captured stage templates and keep their order;
* conserve FWD/BWD compute work for every lane/phase/microbatch route;
* conserve optimizer payload by collective kind;
* preserve the captured optimizer dependency lags and completion skew; and
* remap ranks densely to the target topology.

Enumerating every pair of omitted source templates exposes the structural
uncertainty instead of presenting one arbitrary remapping as measured truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .mfu_timeline import AG_KINDS, TIMELINE_KINDS


@dataclass(frozen=True)
class PipelineRepartition:
    trace_events: pd.DataFrame
    stage_mapping: pd.DataFrame
    compute_conservation: pd.DataFrame


@dataclass(frozen=True)
class OptimizerRepartition:
    calls: pd.DataFrame
    payload_conservation: pd.DataFrame
    service_scales: dict[str, float]


@dataclass(frozen=True)
class RepartitionedCase:
    pipeline: PipelineRepartition
    optimizer: OptimizerRepartition
    clocks: pd.DataFrame
    target_world_size: int


def _one_iteration(frame: pd.DataFrame, iteration: int, label: str) -> pd.DataFrame:
    selected = frame[frame["iteration"].eq(iteration)].copy()
    if selected.empty:
        raise ValueError(f"{label} does not contain iteration {iteration}")
    return selected


def _stage_plan(
    trace_events: pd.DataFrame, retained_stages: Iterable[int]
) -> tuple[tuple[int, ...], dict[int, int], pd.DataFrame, pd.DataFrame]:
    topology = (
        trace_events[["rank", "pp_stage", "pp_lane"]]
        .drop_duplicates()
        .sort_values(["pp_stage", "pp_lane"])
        .reset_index(drop=True)
    )
    source_stages = tuple(sorted(topology["pp_stage"].astype(int).unique()))
    expected_stages = tuple(range(len(source_stages)))
    if source_stages != expected_stages:
        raise ValueError("source pipeline stages must be dense from zero")
    source_lanes = tuple(sorted(topology["pp_lane"].astype(int).unique()))
    if source_lanes != tuple(range(len(source_lanes))):
        raise ValueError("source pipeline lanes must be dense from zero")
    expected_grid = pd.MultiIndex.from_product(
        [source_stages, source_lanes], names=["pp_stage", "pp_lane"]
    )
    observed_grid = pd.MultiIndex.from_frame(topology[["pp_stage", "pp_lane"]])
    if len(topology) != len(expected_grid) or set(observed_grid) != set(expected_grid):
        raise ValueError("source rank topology is not a complete stage/lane grid")

    retained = tuple(int(stage) for stage in retained_stages)
    if retained != tuple(sorted(set(retained))):
        raise ValueError("retained stages must be unique and sorted")
    if not retained or not set(retained).issubset(source_stages):
        raise ValueError("retained stages must be a non-empty source-stage subset")
    stage_map = {source: target for target, source in enumerate(retained)}
    mapping = topology[topology["pp_stage"].isin(retained)].copy()
    mapping = mapping.rename(
        columns={
            "rank": "source_rank",
            "pp_stage": "source_pp_stage",
            "pp_lane": "target_pp_lane",
        }
    )
    mapping["target_pp_stage"] = mapping["source_pp_stage"].map(stage_map)
    mapping["target_rank"] = (
        mapping["target_pp_stage"] * len(source_lanes) + mapping["target_pp_lane"]
    )
    mapping = mapping[
        [
            "source_rank",
            "source_pp_stage",
            "target_rank",
            "target_pp_stage",
            "target_pp_lane",
        ]
    ].sort_values(["target_pp_stage", "target_pp_lane"])
    return retained, stage_map, topology, mapping.reset_index(drop=True)


def _conserved_durations(
    source: pd.DataFrame, selected: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    route = ["pp_lane", "phase", "microbatch"]
    source_totals = source.groupby(route)["duration_ns"].sum().rename("source_total_ns")
    selected_totals = (
        selected.groupby(route)["duration_ns"].sum().rename("selected_total_ns")
    )
    totals = pd.concat([source_totals, selected_totals], axis=1)
    if totals.isna().any().any() or (totals <= 0).any().any():
        raise ValueError("every pipeline route must retain positive compute work")

    result = pd.Series(index=selected.index, dtype="int64")
    audit_rows: list[dict[str, object]] = []
    for key, group in selected.groupby(route, sort=True):
        source_total = int(totals.loc[key, "source_total_ns"])
        selected_total = int(totals.loc[key, "selected_total_ns"])
        exact = group["duration_ns"].to_numpy(dtype="float64") * (
            source_total / selected_total
        )
        rounded = np.floor(exact).astype("int64")
        residual = source_total - int(rounded.sum())
        if residual:
            order = np.argsort(-(exact - rounded), kind="stable")
            rounded[order[:residual]] += 1
        if (rounded <= 0).any() or int(rounded.sum()) != source_total:
            raise ValueError("integer compute-work conservation failed")
        result.loc[group.index] = rounded
        lane, phase, microbatch = key
        audit_rows.append(
            {
                "pp_lane": int(lane),
                "phase": str(phase),
                "microbatch": int(microbatch),
                "source_total_ns": source_total,
                "selected_total_ns_before_scaling": selected_total,
                "target_total_ns": int(rounded.sum()),
                "scale": source_total / selected_total,
                "conservation_error_ns": int(rounded.sum()) - source_total,
            }
        )
    return result.astype("int64"), pd.DataFrame(audit_rows)


def repartition_pipeline_trace(
    trace_events: pd.DataFrame, retained_stages: Iterable[int]
) -> PipelineRepartition:
    """Build a dense target PP trace while conserving per-route compute work."""
    if trace_events["iteration"].nunique() != 1:
        raise ValueError("pipeline repartition expects exactly one iteration")
    retained, stage_map, _topology, mapping = _stage_plan(
        trace_events, retained_stages
    )
    selected = trace_events[trace_events["pp_stage"].isin(retained)].copy()
    durations, conservation = _conserved_durations(trace_events, selected)
    selected["duration_ns"] = durations

    rank_map = mapping.set_index("source_rank")["target_rank"].to_dict()
    selected["source_rank"] = selected["rank"].astype(int)
    selected["source_pp_stage"] = selected["pp_stage"].astype(int)
    selected["rank"] = selected["source_rank"].map(rank_map).astype("int64")
    selected["pp_stage"] = selected["source_pp_stage"].map(stage_map).astype("int64")

    # Observed timestamps have two calibration roles in the unified model:
    # lane origin and last-BWD-to-optimizer launch lag.  Reconstruct intervals
    # after duration scaling, then explicitly preserve both anchors.
    if {"observed_start_ns", "observed_end_ns"}.issubset(selected.columns):
        original_end = selected["observed_end_ns"].copy()
        selected["observed_end_ns"] = (
            selected["observed_start_ns"].astype("int64") + selected["duration_ns"]
        )
        last_microbatch = int(trace_events["microbatch"].max())
        last_backward = selected["phase"].eq("backward") & selected[
            "microbatch"
        ].eq(last_microbatch)
        selected.loc[last_backward, "observed_end_ns"] = original_end[last_backward]
        selected.loc[last_backward, "observed_start_ns"] = (
            selected.loc[last_backward, "observed_end_ns"]
            - selected.loc[last_backward, "duration_ns"]
        )

        source_origins = trace_events.groupby("pp_lane")["observed_start_ns"].min()
        for lane, group in selected.groupby("pp_lane"):
            earliest_index = group["observed_start_ns"].idxmin()
            selected.loc[earliest_index, "observed_start_ns"] = int(
                source_origins.loc[lane]
            )
            selected.loc[earliest_index, "observed_end_ns"] = (
                selected.loc[earliest_index, "observed_start_ns"]
                + selected.loc[earliest_index, "duration_ns"]
            )

    selected = selected.sort_values(
        ["pp_lane", "pp_stage", "phase", "microbatch"]
    ).reset_index(drop=True)
    return PipelineRepartition(
        trace_events=selected,
        stage_mapping=mapping,
        compute_conservation=conservation,
    )


def repartition_optimizer_calls(
    optimizer_calls: pd.DataFrame,
    source_trace_events: pd.DataFrame,
    retained_stages: Iterable[int],
) -> OptimizerRepartition:
    """Remap optimizer groups and conserve aggregate payload by kind."""
    iteration = int(source_trace_events["iteration"].iloc[0])
    calls = _one_iteration(optimizer_calls, iteration, "optimizer calls")
    retained, stage_map, _topology, mapping = _stage_plan(
        source_trace_events, retained_stages
    )
    selected = calls[calls["pp_stage"].isin(retained)].copy()
    rank_map = mapping.set_index("source_rank")["target_rank"].to_dict()
    selected["source_rank"] = selected["rank"].astype(int)
    selected["source_pp_stage"] = selected["pp_stage"].astype(int)
    if selected["source_rank"].map(rank_map).isna().any():
        raise ValueError("optimizer rank is missing from the retained PP topology")
    selected["rank"] = selected["source_rank"].map(rank_map).astype("int64")
    selected["pp_stage"] = selected["source_pp_stage"].map(stage_map).astype("int64")
    selected["group_key"] = [
        f"target_pp{int(target)}:{key}"
        for target, key in zip(selected["pp_stage"], selected["group_key"])
    ]

    group_key = ["kind", "round", "group_key"]
    source_groups = calls.drop_duplicates(["kind", "round", "group_key"])
    selected_groups = selected.drop_duplicates(group_key)
    audit_rows: list[dict[str, object]] = []
    scales: dict[str, float] = {}
    for kind in TIMELINE_KINDS:
        source_kind = source_groups[source_groups["kind"].eq(kind)]
        selected_kind = selected_groups[selected_groups["kind"].eq(kind)]
        if source_kind.empty and selected_kind.empty:
            scales[kind] = 1.0
            continue
        if source_kind.empty or selected_kind.empty:
            raise ValueError(f"retained topology loses all {kind} groups")
        source_payload = int(source_kind["payload_bytes"].sum())
        selected_payload = int(selected_kind["payload_bytes"].sum())
        scale = source_payload / selected_payload
        scales[kind] = scale
        mask = selected["kind"].eq(kind)
        selected.loc[mask, "payload_bytes"] = np.rint(
            selected.loc[mask, "payload_bytes"].astype("float64") * scale
        ).astype("int64")
        target_groups = selected[mask].drop_duplicates(group_key)
        target_payload = int(target_groups["payload_bytes"].sum())
        audit_rows.append(
            {
                "kind": kind,
                "source_groups": len(source_kind),
                "target_groups": len(target_groups),
                "source_payload_bytes": source_payload,
                "selected_payload_bytes_before_scaling": selected_payload,
                "target_payload_bytes": target_payload,
                "payload_scale_and_service_scale": scale,
                "payload_conservation_error_bytes": target_payload - source_payload,
            }
        )

    selected = selected.sort_values(
        ["pp_stage", "kind", "round", "group_key", "rank"]
    ).reset_index(drop=True)
    return OptimizerRepartition(
        calls=selected,
        payload_conservation=pd.DataFrame(audit_rows),
        service_scales=scales,
    )


def extrapolated_clocks(
    iteration_clocks: pd.DataFrame,
    source_optimizer_calls: pd.DataFrame,
    target_optimizer_calls: pd.DataFrame,
    iteration: int,
) -> pd.DataFrame:
    """Preserve the source post-AG and outer-clock residual calibrations."""
    clock = _one_iteration(iteration_clocks, iteration, "iteration clocks")
    if len(clock) != 1:
        raise ValueError("iteration clocks must contain one selected row")
    source = _one_iteration(source_optimizer_calls, iteration, "optimizer calls")
    source_last_ag = int(source.loc[source["kind"].isin(AG_KINDS), "end_ns"].max())
    target_last_ag = int(
        target_optimizer_calls.loc[
            target_optimizer_calls["kind"].isin(AG_KINDS), "end_ns"
        ].max()
    )
    post_ag_ns = int(clock.iloc[0]["step_end_ns"]) - source_last_ag
    if post_ag_ns < 0:
        raise ValueError("source post-AG residual is negative")
    result = clock.copy()
    result.loc[:, "step_end_ns"] = target_last_ag + post_ag_ns
    result.loc[:, "profiler_step_ns"] = (
        result["step_end_ns"] - result["step_start_ns"]
    )
    result.loc[:, "training_log_ns"] = (
        result["profiler_step_ns"] + result["outer_residual_ns"]
    )
    return result.reset_index(drop=True)


def build_repartitioned_case(
    trace_events: pd.DataFrame,
    optimizer_calls: pd.DataFrame,
    iteration_clocks: pd.DataFrame,
    retained_stages: Iterable[int],
) -> RepartitionedCase:
    pipeline = repartition_pipeline_trace(trace_events, retained_stages)
    optimizer = repartition_optimizer_calls(
        optimizer_calls, trace_events, retained_stages
    )
    iteration = int(trace_events["iteration"].iloc[0])
    clocks = extrapolated_clocks(
        iteration_clocks, optimizer_calls, optimizer.calls, iteration
    )
    return RepartitionedCase(
        pipeline=pipeline,
        optimizer=optimizer,
        clocks=clocks,
        target_world_size=int(pipeline.trace_events["rank"].nunique()),
    )


def stage_omission_scenarios(
    source_pp_size: int, target_pp_size: int
) -> list[tuple[int, ...]]:
    """Return all ordered stage-template subsets for a smaller PP target."""
    if source_pp_size <= 0 or target_pp_size <= 0 or target_pp_size > source_pp_size:
        raise ValueError("PP sizes must be positive and target cannot exceed source")
    omitted_count = source_pp_size - target_pp_size
    stages = tuple(range(source_pp_size))
    return [
        tuple(stage for stage in stages if stage not in omitted)
        for omitted in combinations(stages, omitted_count)
    ]


def enumerate_legal_strategies(
    *,
    world_size: int,
    expert_parallel_size: int,
    source_cp_size: int,
    source_dp_size: int,
    source_microbatches: int,
    pp_sizes: Iterable[int],
    cp_sizes: Iterable[int],
    captured_microbatches: int | None = None,
) -> pd.DataFrame:
    """Enumerate factor-compatible TP1 strategies and their trace distance.

    ``source_*`` describes the fixed-global-batch reference strategy.  A
    separate captured microbatch count exposes when that legal reference no
    longer shares the captured Trace schedule.
    """
    trace_microbatches = (
        source_microbatches
        if captured_microbatches is None
        else int(captured_microbatches)
    )
    rows: list[dict[str, object]] = []
    for pp_size in sorted(set(int(value) for value in pp_sizes)):
        for cp_size in sorted(set(int(value) for value in cp_sizes)):
            denominator = pp_size * cp_size
            if pp_size <= 0 or cp_size <= 0 or world_size % denominator:
                continue
            dp_size = world_size // denominator
            dense_size = cp_size * dp_size
            if dense_size % expert_parallel_size:
                continue
            microbatches = source_microbatches * source_dp_size / dp_size
            if not float(microbatches).is_integer() or microbatches < 1:
                continue
            preserves_cp_dp = (
                cp_size == source_cp_size and dp_size == source_dp_size
            )
            preserves_communicators = dense_size == source_cp_size * source_dp_size
            preserves_captured_microbatches = microbatches == trace_microbatches
            if preserves_cp_dp and preserves_captured_microbatches:
                identifiability = "high_pp_only"
            elif preserves_cp_dp:
                identifiability = "medium_same_topology_new_microbatch"
            elif preserves_communicators:
                identifiability = "medium_same_communicators"
            else:
                identifiability = "low_new_communicators"
            rows.append(
                {
                    "world_size": world_size,
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": pp_size,
                    "context_parallel_size": cp_size,
                    "data_parallel_size": dp_size,
                    "expert_parallel_size": expert_parallel_size,
                    "dense_communicator_size": dense_size,
                    "expert_communicator_size": dense_size
                    // expert_parallel_size,
                    "pipeline_lanes": dense_size,
                    "microbatches_at_fixed_global_batch": int(microbatches),
                    "captured_microbatches": trace_microbatches,
                    "preserves_captured_microbatch_count": bool(
                        preserves_captured_microbatches
                    ),
                    "trace_identifiability": identifiability,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["pipeline_parallel_size", "context_parallel_size"]
    ).reset_index(drop=True)

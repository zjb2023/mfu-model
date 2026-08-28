from __future__ import annotations

import pandas as pd

from x10000_analysis.parallel_extrapolation import (
    build_repartitioned_case,
    enumerate_legal_strategies,
    repartition_pipeline_trace,
    stage_omission_scenarios,
)


def _trace(pp_size: int = 4, lanes: int = 2, microbatches: int = 2) -> pd.DataFrame:
    rows = []
    for stage in range(pp_size):
        for lane in range(lanes):
            rank = stage * lanes + lane
            for phase_index, phase in enumerate(("forward", "backward")):
                for microbatch in range(microbatches):
                    duration = 10 + stage * 2 + lane + phase_index + microbatch
                    start = (
                        1_000
                        + lane * 10
                        + stage * 100
                        + phase_index * 1_000
                        + microbatch * 20
                    )
                    rows.append(
                        {
                            "iteration": 1,
                            "rank": rank,
                            "pp_stage": stage,
                            "pp_lane": lane,
                            "phase": phase,
                            "microbatch": microbatch,
                            "observed_start_ns": start,
                            "observed_end_ns": start + duration,
                            "duration_ns": duration,
                        }
                    )
    return pd.DataFrame(rows)


def _optimizer_calls(pp_size: int = 4, lanes: int = 2) -> pd.DataFrame:
    rows = []
    for kind_index, (kind, round_index) in enumerate(
        (("dp_rs", 0), ("dp_ag0", 0), ("dp_ag1", 1))
    ):
        for stage in range(pp_size):
            group_key = f"{kind}:stage{stage}"
            starts = []
            ends = []
            for lane in range(lanes):
                start = 4_000 + kind_index * 1_000 + stage * 50 + lane
                end = start + 20 + stage
                starts.append(start)
                ends.append(end)
                rows.append(
                    {
                        "iteration": 1,
                        "pp_stage": stage,
                        "rank": stage * lanes + lane,
                        "behavior": kind,
                        "kind": kind,
                        "round": round_index,
                        "group_key": group_key,
                        "group_size": lanes,
                        "start_ns": start,
                        "end_ns": end,
                        "payload_bytes": 1_000 + stage * 100,
                    }
                )
            first = min(starts)
            ready = max(starts)
            group_end = max(ends)
            for row in rows[-lanes:]:
                row.update(
                    {
                        "group_first_start_ns": first,
                        "group_ready_ns": ready,
                        "group_end_ns": group_end,
                        "arrival_wait_ns": ready - row["start_ns"],
                        "service_ns": group_end - ready,
                        "completion_advance_ns": group_end - row["end_ns"],
                        "observed_ranks": lanes,
                    }
                )
    return pd.DataFrame(rows)


def _clocks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iteration": 1,
                "step_start_ns": 900,
                "step_end_ns": 7_000,
                "profiler_step_ns": 6_100,
                "training_log_ns": 6_200,
                "outer_residual_ns": 100,
                "reported_tflops_per_gpu": 10.0,
            }
        ]
    )


def test_pipeline_repartition_conserves_every_compute_route() -> None:
    source = _trace()
    result = repartition_pipeline_trace(source, retained_stages=(0, 2, 3))
    target = result.trace_events
    route = ["pp_lane", "phase", "microbatch"]
    assert target.groupby(route)["duration_ns"].sum().equals(
        source.groupby(route)["duration_ns"].sum()
    )
    assert result.compute_conservation["conservation_error_ns"].eq(0).all()
    assert target["rank"].nunique() == 6
    assert set(target["rank"]) == set(range(6))
    assert set(target["pp_stage"]) == {0, 1, 2}
    assert (
        target.groupby("pp_lane")["observed_start_ns"].min()
        == source.groupby("pp_lane")["observed_start_ns"].min()
    ).all()


def test_complete_case_conserves_optimizer_payload_and_builds_target_clock() -> None:
    source_trace = _trace()
    result = build_repartitioned_case(
        source_trace,
        _optimizer_calls(),
        _clocks(),
        retained_stages=(0, 2, 3),
    )
    assert result.target_world_size == 6
    assert result.optimizer.calls["rank"].nunique() == 6
    assert set(result.optimizer.calls["pp_stage"]) == {0, 1, 2}
    assert result.optimizer.payload_conservation[
        "payload_conservation_error_bytes"
    ].abs().max() <= result.optimizer.payload_conservation["target_groups"].max()
    assert result.clocks.iloc[0]["training_log_ns"] == (
        result.clocks.iloc[0]["profiler_step_ns"] + 100
    )


def test_pp16_to_pp14_has_all_120_template_omission_scenarios() -> None:
    scenarios = stage_omission_scenarios(16, 14)
    assert len(scenarios) == 120
    assert len(set(scenarios)) == 120
    assert all(len(scenario) == 14 for scenario in scenarios)


def test_224_strategy_enumeration_uses_target_gbs48_microbatch_count() -> None:
    strategies = enumerate_legal_strategies(
        world_size=224,
        expert_parallel_size=8,
        source_cp_size=2,
        source_dp_size=8,
        source_microbatches=3,
        pp_sizes=(7, 14, 28),
        cp_sizes=(1, 2, 4),
        captured_microbatches=4,
    )
    assert len(strategies) == 6
    target = strategies[
        strategies["trace_identifiability"].eq(
            "medium_same_topology_new_microbatch"
        )
    ]
    assert len(target) == 1
    assert target.iloc[0][
        [
            "pipeline_parallel_size",
            "context_parallel_size",
            "data_parallel_size",
            "dense_communicator_size",
            "expert_communicator_size",
            "microbatches_at_fixed_global_batch",
        ]
    ].tolist() == [14, 2, 8, 16, 2, 3]
    assert not bool(target.iloc[0]["preserves_captured_microbatch_count"])

from __future__ import annotations

import pandas as pd

from x10000_analysis.unified_mfu_dag import build_unified_mfu_dag


def _trace_events() -> pd.DataFrame:
    rows = []
    topology = [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]
    for rank, stage, lane in topology:
        for phase_index, phase in enumerate(("forward", "backward")):
            for microbatch in range(2):
                start = 1_000 + lane + stage * 20 + phase_index * 80 + microbatch * 15
                rows.append(
                    {
                        "iteration": 1,
                        "rank": rank,
                        "pp_stage": stage,
                        "pp_lane": lane,
                        "phase": phase,
                        "microbatch": microbatch,
                        "duration_ns": 10,
                        "observed_start_ns": start,
                        "observed_end_ns": start + 10,
                    }
                )
    return pd.DataFrame(rows)


def _optimizer_calls() -> pd.DataFrame:
    rows = []
    topology = [(0, 0), (1, 0), (2, 1), (3, 1)]
    specifications = [("dp_rs", 1_200), ("dp_ag0", 1_300), ("dp_ag1", 1_400)]
    for kind, base in specifications:
        for rank, stage in topology:
            rows.append(
                {
                    "iteration": 1,
                    "pp_stage": stage,
                    "rank": rank,
                    "behavior": kind,
                    "kind": kind,
                    "round": 1 if kind == "dp_ag1" else 0,
                    "group_key": f"{kind}:stage{stage}",
                    "group_size": 2,
                    "start_ns": base + stage * 5 + rank % 2,
                    "end_ns": base + stage * 5 + rank % 2 + 10,
                }
            )
    source = pd.DataFrame(rows)
    keys = ["iteration", "pp_stage", "behavior", "kind", "round", "group_key"]
    grouped = source.groupby(keys, as_index=False).agg(
        group_first_start_ns=("start_ns", "min"),
        group_ready_ns=("start_ns", "max"),
        group_end_ns=("end_ns", "max"),
        observed_ranks=("rank", "nunique"),
    )
    grouped["service_ns"] = grouped["group_end_ns"] - grouped["group_ready_ns"]
    result = source.merge(grouped, on=keys, validate="many_to_one")
    result["arrival_wait_ns"] = result["group_ready_ns"] - result["start_ns"]
    result["completion_advance_ns"] = result["group_end_ns"] - result["end_ns"]
    return result


def _clocks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iteration": 1,
                "step_start_ns": 900,
                "step_end_ns": 1_500,
                "profiler_step_ns": 600,
                "training_log_ns": 700,
                "outer_residual_ns": 100,
                "reported_tflops_per_gpu": 10.0,
            }
        ]
    )


def test_pp_frontier_drives_rs_and_ag_timeline() -> None:
    result = build_unified_mfu_dag(
        _trace_events(),
        pp_service_ns=3,
        optimizer_calls=_optimizer_calls(),
        clocks=_clocks(),
        model_flops_per_iteration=1e6,
        world_size=4,
        peak_tflops_per_gpu=100.0,
    )
    assert len(result.front_anchors) == 4
    assert set(result.optimizer.calls.loc[
        result.optimizer.calls["kind"].eq("dp_rs"), "dependency"
    ]) == {"pp_frontier"}
    assert {"forward", "backward", "dp_rs", "dp_ag0", "dp_ag1"}.issubset(
        set(result.combined_timeline["category"])
    )
    assert "pp_frontier_to_dense_rs" in set(result.dependency_edges["edge_type"])


def test_pp_fct_change_propagates_to_final_step() -> None:
    arguments = {
        "trace_events": _trace_events(),
        "optimizer_calls": _optimizer_calls(),
        "clocks": _clocks(),
        "model_flops_per_iteration": 1e6,
        "world_size": 4,
        "peak_tflops_per_gpu": 100.0,
    }
    fast = build_unified_mfu_dag(pp_service_ns=0, **arguments)
    slow = build_unified_mfu_dag(pp_service_ns=8, **arguments)
    assert slow.iteration.iloc[0]["predicted_profiler_step_ms"] > fast.iteration.iloc[0][
        "predicted_profiler_step_ms"
    ]

from __future__ import annotations

import pandas as pd
import pytest

from x10000_analysis.pp_dag import (
    build_pipeline_dag,
    non_interleaved_1f1b_schedule,
)


def _symbols(schedule: pd.DataFrame) -> list[str]:
    return [
        f"{'F' if row.phase == 'forward' else 'B'}{int(row.microbatch)}"
        for row in schedule.itertuples(index=False)
    ]


def _events(pp_size: int, microbatches: int, duration_ns: int = 10) -> pd.DataFrame:
    rows = []
    for stage in range(pp_size):
        for phase in ("forward", "backward"):
            for microbatch in range(microbatches):
                rows.append(
                    {
                        "iteration": 1,
                        "rank": stage,
                        "pp_stage": stage,
                        "pp_lane": 0,
                        "phase": phase,
                        "microbatch": microbatch,
                        "duration_ns": duration_ns,
                    }
                )
    return pd.DataFrame(rows)


def test_pp16_m4_schedule_matches_non_interleaved_1f1b() -> None:
    assert _symbols(non_interleaved_1f1b_schedule(16, 4, 0)) == [
        "F0", "F1", "F2", "F3", "B0", "B1", "B2", "B3"
    ]
    assert _symbols(non_interleaved_1f1b_schedule(16, 4, 13)) == [
        "F0", "F1", "F2", "B0", "F3", "B1", "B2", "B3"
    ]
    assert _symbols(non_interleaved_1f1b_schedule(16, 4, 14)) == [
        "F0", "F1", "B0", "F2", "B1", "F3", "B2", "B3"
    ]
    assert _symbols(non_interleaved_1f1b_schedule(16, 4, 15)) == [
        "F0", "B0", "F1", "B1", "F2", "B2", "F3", "B3"
    ]


def test_backward_messages_run_from_last_stage_to_first() -> None:
    result = build_pipeline_dag(_events(pp_size=3, microbatches=2), pp_service_ns=3)
    backward = result.nodes[
        result.nodes["kind"].eq("pp_message")
        & result.nodes["phase"].eq("backward")
    ]
    assert set(zip(backward["src_stage"], backward["dst_stage"])) == {(2, 1), (1, 0)}
    assert (backward["src_stage"] > backward["dst_stage"]).all()


def test_message_service_propagates_to_pipeline_makespan() -> None:
    events = _events(pp_size=2, microbatches=2)
    zero = build_pipeline_dag(events, pp_service_ns=0)
    slower = build_pipeline_dag(events, pp_service_ns=5)
    assert slower.lane_summary.iloc[0]["predicted_front_makespan_ns"] > zero.lane_summary.iloc[0][
        "predicted_front_makespan_ns"
    ]
    assert slower.nodes["local_wait_ns"].ge(0).all()


def test_software_completion_is_separate_from_network_service() -> None:
    result = build_pipeline_dag(
        _events(pp_size=2, microbatches=2),
        pp_service_ns=3,
        pp_software_completion_ns={"backward": 7},
    )
    messages = result.nodes[result.nodes["kind"].eq("pp_message")]
    forward = messages[messages["phase"].eq("forward")]
    backward = messages[messages["phase"].eq("backward")]
    assert set(forward["duration_ns"]) == {3}
    assert set(backward["duration_ns"]) == {10}
    assert set(backward["network_service_ns"]) == {3}
    assert set(backward["software_completion_ns"]) == {7}


def test_optimizer_frontier_is_every_ranks_last_backward() -> None:
    result = build_pipeline_dag(_events(pp_size=4, microbatches=2), pp_service_ns=2)
    assert len(result.optimizer_frontier) == 4
    assert set(result.optimizer_frontier["predecessor_node_id"]) == {
        "rank0:B1", "rank1:B1", "rank2:B1", "rank3:B1"
    }


def test_incomplete_rank_event_grid_fails_closed() -> None:
    events = _events(pp_size=2, microbatches=2).iloc[:-1]
    with pytest.raises(ValueError, match="every rank"):
        build_pipeline_dag(events, pp_service_ns=1)

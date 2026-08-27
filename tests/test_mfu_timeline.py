from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from x10000_analysis.mfu_timeline import (
    build_collective_slack_audit,
    build_optimizer_timeline_model,
    validate_iteration_clocks,
)
from x10000_analysis.oisa_fct import (
    RecordedOisaFctProvider,
    ScaledMockOisaFctProvider,
    TraceReferenceFctProvider,
)


def _facts(rows: list[dict[str, object]]) -> pd.DataFrame:
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


def _row(
    rank: int,
    behavior: str,
    kind: str,
    round_index: int,
    group: str,
    group_size: int,
    start: int,
    end: int,
) -> dict[str, object]:
    return {
        "iteration": 1,
        "pp_stage": 0,
        "rank": rank,
        "behavior": behavior,
        "kind": kind,
        "round": round_index,
        "group_key": group,
        "group_size": group_size,
        "start_ns": start,
        "end_ns": end,
    }


def _full_calls() -> pd.DataFrame:
    return _facts(
        [
            _row(0, "dense_rs", "dp_rs", 0, "dense", 2, 100, 150),
            _row(1, "dense_rs", "dp_rs", 0, "dense", 2, 110, 160),
            _row(0, "expert_rs", "edp_rs", 0, "expert0", 1, 170, 180),
            _row(1, "expert_rs", "edp_rs", 0, "expert1", 1, 180, 190),
            _row(0, "dense_ag", "dp_ag0", 0, "dense", 2, 200, 220),
            _row(1, "dense_ag", "dp_ag0", 0, "dense", 2, 205, 225),
            _row(0, "expert_ag", "edp_ag0", 0, "expert0", 1, 230, 235),
            _row(1, "expert_ag", "edp_ag0", 0, "expert1", 1, 235, 240),
            _row(0, "dense_ag", "dp_ag1", 1, "dense", 2, 250, 265),
            _row(1, "dense_ag", "dp_ag1", 1, "dense", 2, 255, 270),
            _row(0, "expert_ag", "edp_ag1", 1, "expert0", 1, 275, 280),
            _row(1, "expert_ag", "edp_ag1", 1, "expert1", 1, 280, 285),
        ]
    )


def _clocks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iteration": 1,
                "step_start_ns": 1,
                "step_end_ns": 300,
                "profiler_step_ns": 299,
                "training_log_ns": 329,
                "outer_residual_ns": 30,
                "reported_tflops_per_gpu": 10.0,
            }
        ]
    )


def test_identity_replay_preserves_observed_timestamps() -> None:
    calls = _full_calls()
    calls["payload_bytes"] = 1024
    result = build_optimizer_timeline_model(calls, _clocks())
    assert (result.calls["predicted_start_ns"] == result.calls["start_ns"]).all()
    assert (result.calls["predicted_end_ns"] == result.calls["end_ns"]).all()
    assert result.groups["payload_bytes"].eq(1024).all()
    iteration = result.iterations.iloc[0]
    assert iteration["profiler_replay_error_ms"] == 0
    assert iteration["predicted_training_log_ms"] == pytest.approx(329 / 1e6)
    assert iteration["fallback_dependencies"] == 0


def test_dense_rs_service_change_propagates_through_dependency_graph() -> None:
    result = build_optimizer_timeline_model(
        _full_calls(), _clocks(), service_scales={"dp_rs": 0.0}
    )
    calls = result.calls
    dense_rs = calls[calls["kind"].eq("dp_rs")]
    assert dense_rs["predicted_group_end_ns"].nunique() == 1
    assert dense_rs["predicted_group_end_ns"].iloc[0] == 110
    # The downstream expert RS and AG chain move; no CPU wrapper duration is
    # separately added to the collective service.
    assert calls.loc[calls["kind"].eq("edp_rs"), "predicted_start_ns"].max() < 180
    assert result.iterations.iloc[0]["predicted_profiler_step_ms"] < 299 / 1e6


def test_oisa_elapsed_starts_at_first_release_and_completion_is_not_scaled() -> None:
    result = build_optimizer_timeline_model(
        _full_calls(),
        _clocks(),
        fct_provider=ScaledMockOisaFctProvider(default_scale=2.0),
    )
    dense_rs = result.calls[result.calls["kind"].eq("dp_rs")].sort_values("rank")
    # Starts are 100 and 110: OISA elapsed is 10 arrival span + 100 tail.
    # group_end must be first_release 100 + elapsed 110 = 210, not
    # last_release 110 + elapsed 110 = 220.
    assert dense_rs["oisa_collective_elapsed_ns"].eq(110).all()
    assert dense_rs["predicted_group_end_ns"].eq(210).all()
    # The measured rank completion skew is 10 ns and remains 10 ns.  It is a
    # separate Trace property, not another network-service multiplier.
    assert dense_rs["predicted_completion_advance_ns"].tolist() == [10, 0]
    assert dense_rs["predicted_end_ns"].tolist() == [200, 210]


def test_collective_slack_audit_separates_hidden_and_exposed_tail() -> None:
    audit = build_collective_slack_audit(
        _full_calls(),
        _clocks(),
        reference_provider=ScaledMockOisaFctProvider(default_scale=1.0),
        candidate_provider=ScaledMockOisaFctProvider(default_scale=2.0),
    )
    final = audit[
        audit["kind"].eq("edp_ag1") & audit["group_key"].eq("expert1")
    ].iloc[0]
    earlier = audit[
        audit["kind"].eq("edp_ag1") & audit["group_key"].eq("expert0")
    ].iloc[0]
    assert final["group_finish_shift_ns"] == 5
    assert final["iteration_drag_ns"] == 5
    assert final["exposure_ratio"] == 1
    assert earlier["group_finish_shift_ns"] == 5
    assert earlier["iteration_drag_ns"] == 0
    assert bool(earlier["fully_hidden"])


def test_finite_recorded_oisa_table_replays_baseline_without_extra_queries() -> None:
    online = build_optimizer_timeline_model(
        _full_calls(),
        _clocks(),
        fct_provider=ScaledMockOisaFctProvider(default_scale=1.1),
    )
    records = []
    for group in online.groups.itertuples(index=False):
        records.append(
            {
                "request_id": group.fct_request_id,
                "first_release_ns": 0,
                "last_release_ns": group.predicted_arrival_skew_ns,
                "last_flow_end_ns": group.oisa_collective_elapsed_ns,
                "collective_elapsed_ns": group.oisa_collective_elapsed_ns,
                "arrival_span_ns": group.predicted_arrival_skew_ns,
                "tail_after_last_release_ns": group.oisa_tail_after_last_release_ns,
            }
        )
    recorded = build_optimizer_timeline_model(
        _full_calls(),
        _clocks(),
        fct_provider=RecordedOisaFctProvider(records),
    )
    assert recorded.calls["predicted_end_ns"].equals(online.calls["predicted_end_ns"])
    assert not bool(recorded.iterations.iloc[0]["service_marginals_available"])
    assert recorded.iterations.iloc[0]["all_dp_service_exposed_ms"] == 0


def test_oisa_rank_done_offsets_replace_trace_completion_skew() -> None:
    class RankDoneProvider:
        supports_dynamic_arrivals = False

        def __call__(self, request):
            result = TraceReferenceFctProvider()(request)
            if request.kind != "dp_rs":
                return result
            return replace(
                result,
                source="oisa_rank_done_test",
                rank_network_done_offsets_ns={
                    rank: result.last_flow_end_ns for rank in request.ranks
                },
            )

    result = build_optimizer_timeline_model(
        _full_calls(), _clocks(), fct_provider=RankDoneProvider()
    )
    dense = result.calls[result.calls["kind"].eq("dp_rs")].sort_values("rank")
    assert dense["predicted_group_end_ns"].tolist() == [160, 160]
    assert dense["predicted_end_ns"].tolist() == [160, 160]
    assert dense["predicted_completion_advance_ns"].tolist() == [0, 0]
    assert set(dense["completion_model"]) == {"oisa_rank_network_done"}


def test_pp_frontier_replaces_measured_dense_rs_arrivals() -> None:
    anchors = pd.DataFrame(
        {
            "iteration": [1, 1],
            "rank": [0, 1],
            "predicted_start_ns": [50, 70],
        }
    )
    result = build_optimizer_timeline_model(
        _full_calls(), _clocks(), front_anchors=anchors
    )
    dense_rs = result.calls[result.calls["kind"].eq("dp_rs")].sort_values("rank")
    assert dense_rs["predicted_start_ns"].tolist() == [50, 70]
    assert set(dense_rs["dependency"]) == {"pp_frontier"}
    assert dense_rs["predicted_group_ready_ns"].tolist() == [70, 70]
    assert result.calls.loc[
        result.calls["kind"].eq("edp_rs"), "predicted_start_ns"
    ].max() < 180


def test_pp_frontier_must_cover_every_dense_rs_rank() -> None:
    anchors = pd.DataFrame(
        {"iteration": [1], "rank": [0], "predicted_start_ns": [50]}
    )
    with pytest.raises(ValueError, match="front-anchor grid mismatch"):
        build_optimizer_timeline_model(_full_calls(), _clocks(), front_anchors=anchors)


def test_counterfactuals_are_relative_to_requested_baseline_scale() -> None:
    result = build_optimizer_timeline_model(
        _full_calls(), _clocks(), service_scales={"dp_rs": 0.5}
    )
    iteration = result.iterations.iloc[0]
    assert iteration["dense_dp_service_marginal_ms"] > 0
    assert iteration["all_dp_service_exposed_ms"] >= iteration[
        "dense_dp_service_marginal_ms"
    ]


def test_dense_only_case_uses_ag0_as_ag1_predecessor() -> None:
    calls = _full_calls()
    calls = calls[~calls["kind"].str.startswith("edp_")].copy()
    result = build_optimizer_timeline_model(calls, _clocks())
    ag1 = result.calls[result.calls["kind"].eq("dp_ag1")]
    assert set(ag1["dependency"]) == {"same_rank_dp_ag0"}
    assert result.iterations.iloc[0]["fallback_dependencies"] == 0


def test_clock_domains_must_conserve_outer_residual() -> None:
    clocks = _clocks()
    clocks.loc[0, "outer_residual_ns"] = 32
    with pytest.raises(ValueError, match="outer clock residual"):
        validate_iteration_clocks(clocks)

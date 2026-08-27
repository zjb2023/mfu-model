"""Timestamp-driven MFU timeline model for the distributed-optimizer tail.

The legacy MFU model aggregates communication duration into semantic phase
buckets.  This module instead preserves rank/group dependencies and recomputes
the wall-clock makespan when collective service changes.  CPU annotations such
as ``finalize_model_grads`` and optimizer ``step`` are validation wrappers; they
are deliberately not added to the collective durations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .oisa_fct import (
    CollectiveFctProvider,
    TraceReferenceFctProvider,
    make_collective_request,
    validate_fct_result,
)


TIMELINE_KINDS = (
    "dp_rs",
    "edp_rs",
    "dp_ag0",
    "edp_ag0",
    "dp_ag1",
    "edp_ag1",
)
RS_KINDS = frozenset({"dp_rs", "edp_rs"})
AG_KINDS = frozenset({"dp_ag0", "edp_ag0", "dp_ag1", "edp_ag1"})
DENSE_KINDS = frozenset({"dp_rs", "dp_ag0", "dp_ag1"})
EXPERT_KINDS = frozenset({"edp_rs", "edp_ag0", "edp_ag1"})

CALL_REQUIRED_COLUMNS = {
    "iteration",
    "pp_stage",
    "rank",
    "behavior",
    "kind",
    "round",
    "group_key",
    "group_size",
    "start_ns",
    "end_ns",
    "group_first_start_ns",
    "group_ready_ns",
    "group_end_ns",
    "arrival_wait_ns",
    "service_ns",
    "completion_advance_ns",
    "observed_ranks",
}
CALL_OPTIONAL_COLUMNS = {"payload_bytes"}
CLOCK_REQUIRED_COLUMNS = {
    "iteration",
    "step_start_ns",
    "step_end_ns",
    "profiler_step_ns",
    "training_log_ns",
    "outer_residual_ns",
    "reported_tflops_per_gpu",
}
FRONT_ANCHOR_REQUIRED_COLUMNS = {"iteration", "rank", "predicted_start_ns"}


@dataclass(frozen=True)
class TimelineResult:
    calls: pd.DataFrame
    groups: pd.DataFrame
    iterations: pd.DataFrame


def validate_optimizer_calls(calls: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the per-rank optimizer collective fact table."""
    missing = sorted(CALL_REQUIRED_COLUMNS - set(calls.columns))
    if missing:
        raise ValueError(f"optimizer timeline calls missing columns: {missing}")
    optional = sorted(CALL_OPTIONAL_COLUMNS & set(calls.columns))
    work = calls[[*CALL_REQUIRED_COLUMNS, *optional]].copy()
    numeric = sorted(
        (CALL_REQUIRED_COLUMNS | set(optional)) - {"behavior", "kind", "group_key"}
    )
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="raise")
    if work[numeric].isna().any().any() or (~np.isfinite(work[numeric])).any().any():
        raise ValueError("optimizer timeline numeric values must be finite")
    if not set(work["kind"]).issubset(TIMELINE_KINDS):
        extra = sorted(set(work["kind"]) - set(TIMELINE_KINDS))
        raise ValueError(f"unknown optimizer timeline kinds: {extra}")
    if (
        (work["end_ns"] <= work["start_ns"])
        | (work["group_end_ns"] < work["group_ready_ns"])
        | (work["group_ready_ns"] < work["group_first_start_ns"])
        | (work["arrival_wait_ns"] < 0)
        | (work["service_ns"] < 0)
        | (work["completion_advance_ns"] < 0)
        | (work["group_size"] <= 0)
        | (work["observed_ranks"] <= 0)
    ).any():
        raise ValueError("optimizer timeline contains invalid interval/count values")
    if "payload_bytes" in work and (work["payload_bytes"] <= 0).any():
        raise ValueError("optimizer timeline payload must be positive")
    if not np.array_equal(
        work["group_ready_ns"] - work["start_ns"], work["arrival_wait_ns"]
    ):
        raise ValueError("arrival-wait decomposition does not conserve start time")
    if not np.array_equal(work["group_end_ns"] - work["group_ready_ns"], work["service_ns"]):
        raise ValueError("service decomposition does not conserve group duration")
    if not np.array_equal(
        work["group_end_ns"] - work["end_ns"], work["completion_advance_ns"]
    ):
        raise ValueError("completion-advance decomposition does not conserve rank end")
    key = ["iteration", "kind", "round", "group_key", "rank"]
    if work.duplicated(key).any():
        raise ValueError("optimizer timeline contains duplicate rank calls")
    return work.sort_values(["iteration", "pp_stage", "kind", "group_key", "rank"]).reset_index(drop=True)


def validate_iteration_clocks(clocks: pd.DataFrame) -> pd.DataFrame:
    """Validate ProfilerStep and outer training-log clock domains."""
    missing = sorted(CLOCK_REQUIRED_COLUMNS - set(clocks.columns))
    if missing:
        raise ValueError(f"iteration clocks missing columns: {missing}")
    work = clocks[list(CLOCK_REQUIRED_COLUMNS)].copy()
    for column in CLOCK_REQUIRED_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="raise")
    if work[list(CLOCK_REQUIRED_COLUMNS)].isna().any().any() or (
        ~np.isfinite(work[list(CLOCK_REQUIRED_COLUMNS)])
    ).any().any():
        raise ValueError("iteration clock values must be finite")
    if work["iteration"].duplicated().any():
        raise ValueError("iteration clocks contain duplicates")
    if (
        (work["step_end_ns"] <= work["step_start_ns"])
        | (work["profiler_step_ns"] <= 0)
        | (work["training_log_ns"] <= 0)
        | (work["outer_residual_ns"] < 0)
    ).any():
        raise ValueError("iteration clocks contain invalid durations")
    if not np.allclose(
        work["step_end_ns"] - work["step_start_ns"], work["profiler_step_ns"], rtol=0, atol=1
    ):
        raise ValueError("ProfilerStep clock does not conserve its interval")
    if not np.allclose(
        work["training_log_ns"] - work["profiler_step_ns"],
        work["outer_residual_ns"],
        rtol=0,
        atol=1,
    ):
        raise ValueError("outer clock residual does not conserve training-log time")
    return work.sort_values("iteration").reset_index(drop=True)


def validate_front_anchors(front_anchors: pd.DataFrame) -> pd.DataFrame:
    """Validate externally predicted dense-RS arrivals, normally from a PP DAG."""
    missing = sorted(FRONT_ANCHOR_REQUIRED_COLUMNS - set(front_anchors.columns))
    if missing:
        raise ValueError(f"optimizer front anchors missing columns: {missing}")
    work = front_anchors[list(FRONT_ANCHOR_REQUIRED_COLUMNS)].copy()
    for column in FRONT_ANCHOR_REQUIRED_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="raise")
    if work[list(FRONT_ANCHOR_REQUIRED_COLUMNS)].isna().any().any() or (
        ~np.isfinite(work[list(FRONT_ANCHOR_REQUIRED_COLUMNS)])
    ).any().any():
        raise ValueError("optimizer front anchors must be finite")
    if work.duplicated(["iteration", "rank"]).any():
        raise ValueError("optimizer front anchors contain duplicate iteration/rank rows")
    return work.sort_values(["iteration", "rank"]).reset_index(drop=True)


def _normalized_scales(scales: Mapping[str, float] | None) -> dict[str, float]:
    result = {kind: 1.0 for kind in TIMELINE_KINDS}
    if scales:
        unknown = sorted(set(scales) - set(TIMELINE_KINDS))
        if unknown:
            raise ValueError(f"unknown timeline service scales: {unknown}")
        result.update({str(key): float(value) for key, value in scales.items()})
    invalid = {
        key: value
        for key, value in result.items()
        if not np.isfinite(value) or value < 0
    }
    if invalid:
        raise ValueError(f"timeline service scales must be finite and non-negative: {invalid}")
    return result


def _simulate_iteration(
    actual: pd.DataFrame,
    clock: pd.Series,
    scales: Mapping[str, float] | None = None,
    front_anchors: Mapping[int, int] | None = None,
    fct_provider: CollectiveFctProvider | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    service_scales = _normalized_scales(scales)
    provider = fct_provider or TraceReferenceFctProvider()
    actual = actual.copy()
    actual_end = {
        (str(row.kind), int(row.pp_stage), int(row.rank)): int(row.end_ns)
        for row in actual.itertuples(index=False)
    }
    actual_stage_end = (
        actual.groupby(["kind", "pp_stage"])["end_ns"].max().to_dict()
    )
    actual_world_rs_end = int(actual.loc[actual["kind"].isin(RS_KINDS), "end_ns"].max())

    simulated_end: dict[tuple[str, int, int], int] = {}
    simulated_stage_end: dict[tuple[str, int], int] = {}
    output: list[pd.DataFrame] = []
    clamped_negative_lags = 0
    fallback_dependencies = 0

    has_edp_ag0 = "edp_ag0" in set(actual["kind"])

    def predecessor(row: object, kind: str) -> tuple[int, int, str]:
        nonlocal clamped_negative_lags, fallback_dependencies
        stage, rank = int(row.pp_stage), int(row.rank)
        if kind == "dp_rs":
            if front_anchors is None:
                return int(row.start_ns), 0, "measured_front_anchor"
            if rank not in front_anchors:
                raise ValueError(
                    f"missing PP frontier for iteration={int(row.iteration)} rank={rank}"
                )
            return int(front_anchors[rank]), 0, "pp_frontier"
        if kind == "dp_ag0":
            actual_predecessor = actual_world_rs_end
            predicted_predecessor = max(
                value
                for (node_kind, _stage, _rank), value in simulated_end.items()
                if node_kind in RS_KINDS
            )
            dependency = "world_rs_done"
        else:
            previous_kind = {
                "edp_rs": "dp_rs",
                "edp_ag0": "dp_ag0",
                "dp_ag1": "edp_ag0" if has_edp_ag0 else "dp_ag0",
                "edp_ag1": "dp_ag1",
            }[kind]
            key = (previous_kind, stage, rank)
            if key in actual_end and key in simulated_end:
                actual_predecessor = actual_end[key]
                predicted_predecessor = simulated_end[key]
                dependency = f"same_rank_{previous_kind}"
            else:
                fallback_dependencies += 1
                stage_key = (previous_kind, stage)
                if stage_key not in actual_stage_end or stage_key not in simulated_stage_end:
                    raise ValueError(
                        f"missing predecessor for iteration={int(row.iteration)} "
                        f"stage={stage} rank={rank} kind={kind}"
                    )
                actual_predecessor = int(actual_stage_end[stage_key])
                predicted_predecessor = int(simulated_stage_end[stage_key])
                dependency = f"stage_fallback_{previous_kind}"
        raw_lag = int(row.start_ns) - actual_predecessor
        # Cross-host timestamps have rare single-digit-us inversions.  Preserve
        # the causal edge and expose the clamp count rather than introducing a
        # negative dependency into extrapolation.
        lag = max(raw_lag, 0)
        if raw_lag < 0:
            clamped_negative_lags += 1
        return predicted_predecessor + lag, lag, dependency

    for kind in TIMELINE_KINDS:
        selected = actual[actual["kind"].eq(kind)].copy()
        if selected.empty:
            continue
        group_columns = ["pp_stage", "kind", "round", "group_key"]
        predicted_starts, lags, dependencies = [], [], []
        for row in selected.itertuples(index=False):
            start, lag, dependency = predecessor(row, kind)
            predicted_starts.append(start)
            lags.append(lag)
            dependencies.append(dependency)
        selected["dependency"] = dependencies
        selected["dependency_lag_ns"] = lags
        selected["predicted_start_ns"] = np.asarray(predicted_starts, dtype="int64")
        selected["predicted_group_ready_ns"] = selected.groupby(group_columns)[
            "predicted_start_ns"
        ].transform("max")
        selected["predicted_group_first_start_ns"] = selected.groupby(group_columns)[
            "predicted_start_ns"
        ].transform("min")

        # OISA receives rank release offsets and returns elapsed time from the
        # first release.  Only its tail after the last release is eligible for
        # service scaling; multiplying the arrival span would double-count DAG
        # synchronization.
        group_results: list[pd.DataFrame] = []
        scale = service_scales[kind]
        for _group_key, group in selected.groupby(group_columns, sort=False):
            request = make_collective_request(group)
            response = validate_fct_result(request, provider(request))
            predicted_tail = int(round(response.tail_after_last_release_ns * scale))
            predicted_elapsed = request.arrival_span_ns + predicted_tail
            group = group.copy()
            group["service_scale"] = scale
            group["fct_group_id"] = request.group_id
            group["fct_request_id"] = request.request_id
            group["fct_source"] = response.source
            group["oisa_collective_elapsed_ns"] = response.collective_elapsed_ns
            group["oisa_arrival_span_ns"] = response.arrival_span_ns
            group["oisa_tail_after_last_release_ns"] = (
                response.tail_after_last_release_ns
            )
            group["predicted_collective_elapsed_ns"] = predicted_elapsed
            group["predicted_service_ns"] = predicted_tail
            group["predicted_group_end_ns"] = (
                group["predicted_group_first_start_ns"] + predicted_elapsed
            )
            group_results.append(group)
        selected = pd.concat(group_results).sort_index()

        # Rank completion skew is a separate Trace-calibrated property.  It is
        # not multiplied by the network FCT.  Clamp only when a synthetic fast
        # network would otherwise place a rank completion before its release.
        max_completion_advance = (
            selected["predicted_group_end_ns"] - selected["predicted_start_ns"]
        ).clip(lower=0)
        selected["predicted_completion_advance_ns"] = np.minimum(
            selected["completion_advance_ns"].astype("int64"),
            max_completion_advance.astype("int64"),
        )
        selected["completion_model"] = "trace_group_end_lag_clamped"
        selected["predicted_end_ns"] = (
            selected["predicted_group_end_ns"]
            - selected["predicted_completion_advance_ns"]
        )
        simulated_kind = selected
        for row in simulated_kind.itertuples(index=False):
            simulated_end[(kind, int(row.pp_stage), int(row.rank))] = int(
                row.predicted_end_ns
            )
        simulated_stage_end.update(
            {
                (kind, int(stage)): int(value)
                for stage, value in simulated_kind.groupby("pp_stage")[
                    "predicted_end_ns"
                ].max().items()
            }
        )
        output.append(simulated_kind)

    result = pd.concat(output, ignore_index=True)
    actual_last_ag = int(result.loc[result["kind"].isin(AG_KINDS), "end_ns"].max())
    predicted_last_ag = int(
        result.loc[result["kind"].isin(AG_KINDS), "predicted_end_ns"].max()
    )
    post_ag_ns = int(clock["step_end_ns"]) - actual_last_ag
    if post_ag_ns < 0:
        raise ValueError("last optimizer collective falls after ProfilerStep end")
    predicted_step_end = predicted_last_ag + post_ag_ns
    predicted_profiler = predicted_step_end - int(clock["step_start_ns"])
    metrics = {
        "predicted_step_end_ns": predicted_step_end,
        "predicted_profiler_step_ns": predicted_profiler,
        "predicted_training_log_ns": predicted_profiler
        + int(clock["outer_residual_ns"]),
        "first_rs_ns": int(
            result.loc[result["kind"].isin(RS_KINDS), "predicted_start_ns"].min()
        ),
        "actual_gradient_rs_done_ns": int(
            result.loc[result["kind"].isin(RS_KINDS), "end_ns"].max()
        ),
        "predicted_gradient_rs_done_ns": int(
            result.loc[result["kind"].isin(RS_KINDS), "predicted_end_ns"].max()
        ),
        "actual_first_ag0_ns": int(
            result.loc[result["kind"].isin({"dp_ag0", "edp_ag0"}), "start_ns"].min()
        ),
        "predicted_first_ag0_ns": int(
            result.loc[result["kind"].isin({"dp_ag0", "edp_ag0"}), "predicted_start_ns"].min()
        ),
        "actual_last_ag_ns": actual_last_ag,
        "predicted_last_ag_ns": predicted_last_ag,
        "post_ag_ns": post_ag_ns,
        "clamped_negative_dependency_lags": clamped_negative_lags,
        "fallback_dependencies": fallback_dependencies,
    }
    return result, metrics


def build_optimizer_timeline_model(
    calls: pd.DataFrame,
    clocks: pd.DataFrame,
    service_scales: Mapping[str, float] | None = None,
    front_anchors: pd.DataFrame | None = None,
    fct_provider: CollectiveFctProvider | None = None,
    calculate_service_marginals: bool | None = None,
) -> TimelineResult:
    """Replay the measured optimizer tail and evaluate communication counterfactuals.

    Without ``front_anchors``, dense DP-RS arrivals replay measured timestamps.
    A PP DAG can instead provide ``iteration/rank/predicted_start_ns`` rows;
    every RS/AG rendezvous and the final makespan is then recomputed from those
    dynamic arrivals.
    """
    facts = validate_optimizer_calls(calls)
    clock_frame = validate_iteration_clocks(clocks)
    if set(facts["iteration"]) != set(clock_frame["iteration"]):
        raise ValueError("optimizer calls and iteration clocks cover different iterations")
    anchor_frame = validate_front_anchors(front_anchors) if front_anchors is not None else None
    if anchor_frame is not None:
        expected = set(
            map(
                tuple,
                facts.loc[facts["kind"].eq("dp_rs"), ["iteration", "rank"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        observed = set(map(tuple, anchor_frame[["iteration", "rank"]].to_numpy()))
        if observed != expected:
            missing = len(expected - observed)
            extra = len(observed - expected)
            raise ValueError(
                f"optimizer front-anchor grid mismatch: missing={missing}, extra={extra}"
            )

    baseline_scales = _normalized_scales(service_scales)
    if calculate_service_marginals is None:
        calculate_service_marginals = bool(
            getattr(fct_provider, "supports_dynamic_arrivals", True)
        )

    def without(kinds: frozenset[str]) -> dict[str, float]:
        return {
            kind: (0.0 if kind in kinds else baseline_scales[kind])
            for kind in TIMELINE_KINDS
        }

    scenario_scales = {
        "baseline": baseline_scales,
    }
    if calculate_service_marginals:
        scenario_scales.update(
            {
                "no_all_dp": without(frozenset(TIMELINE_KINDS)),
                "no_dense_dp": without(DENSE_KINDS),
                "no_expert_dp": without(EXPERT_KINDS),
                "no_rs": without(RS_KINDS),
                "no_ag": without(AG_KINDS),
            }
        )
    base_calls, iteration_rows = [], []
    for iteration, actual in facts.groupby("iteration", sort=True):
        clock = clock_frame[clock_frame["iteration"].eq(iteration)].iloc[0]
        iteration_anchors = None
        if anchor_frame is not None:
            iteration_anchors = {
                int(row.rank): int(row.predicted_start_ns)
                for row in anchor_frame[anchor_frame["iteration"].eq(iteration)].itertuples(
                    index=False
                )
            }
        scenario_metrics = {}
        for scenario, scales in scenario_scales.items():
            simulated, metrics = _simulate_iteration(
                actual,
                clock,
                scales,
                front_anchors=iteration_anchors,
                fct_provider=fct_provider,
            )
            scenario_metrics[scenario] = metrics
            if scenario == "baseline":
                base_calls.append(simulated)
        base = scenario_metrics["baseline"]
        # A finite recorded OISA table covers one exact arrival pattern.  Its
        # baseline prediction remains valid, but extra no-service scenarios
        # require new OISA queries because upstream changes alter later rank
        # releases.  Report zero marginal placeholders and an explicit flag
        # instead of silently reusing mismatched FCT records.
        no_all = scenario_metrics.get("no_all_dp", base)
        no_dense = scenario_metrics.get("no_dense_dp", base)
        no_expert = scenario_metrics.get("no_expert_dp", base)
        no_rs = scenario_metrics.get("no_rs", base)
        no_ag = scenario_metrics.get("no_ag", base)
        step_start = int(clock["step_start_ns"])
        iteration_rows.append(
            {
                "iteration": int(iteration),
                "profiler_step_ms": float(clock["profiler_step_ns"]) / 1e6,
                "predicted_profiler_step_ms": base["predicted_profiler_step_ns"] / 1e6,
                "profiler_replay_error_ms": (
                    base["predicted_profiler_step_ns"] - float(clock["profiler_step_ns"])
                )
                / 1e6,
                "training_log_ms": float(clock["training_log_ns"]) / 1e6,
                "outer_residual_ms": float(clock["outer_residual_ns"]) / 1e6,
                "predicted_training_log_ms": base["predicted_training_log_ns"] / 1e6,
                "first_rs_offset_ms": (base["first_rs_ns"] - step_start) / 1e6,
                "gradient_rs_done_offset_ms": (
                    base["predicted_gradient_rs_done_ns"] - step_start
                )
                / 1e6,
                "first_ag0_offset_ms": (base["predicted_first_ag0_ns"] - step_start) / 1e6,
                "last_ag_offset_ms": (base["predicted_last_ag_ns"] - step_start) / 1e6,
                "dp_envelope_ms": (
                    base["predicted_last_ag_ns"] - base["first_rs_ns"]
                )
                / 1e6,
                "post_ag_ms": base["post_ag_ns"] / 1e6,
                "all_dp_service_exposed_ms": (
                    base["predicted_profiler_step_ns"]
                    - no_all["predicted_profiler_step_ns"]
                )
                / 1e6,
                "dense_dp_service_marginal_ms": (
                    base["predicted_profiler_step_ns"]
                    - no_dense["predicted_profiler_step_ns"]
                )
                / 1e6,
                "expert_dp_service_marginal_ms": (
                    base["predicted_profiler_step_ns"]
                    - no_expert["predicted_profiler_step_ns"]
                )
                / 1e6,
                "rs_service_marginal_ms": (
                    base["predicted_profiler_step_ns"]
                    - no_rs["predicted_profiler_step_ns"]
                )
                / 1e6,
                "ag_service_marginal_ms": (
                    base["predicted_profiler_step_ns"]
                    - no_ag["predicted_profiler_step_ns"]
                )
                / 1e6,
                "counterfactual_no_dp_profiler_ms": no_all[
                    "predicted_profiler_step_ns"
                ]
                / 1e6,
                "counterfactual_no_dp_training_log_ms": no_all[
                    "predicted_training_log_ns"
                ]
                / 1e6,
                "clamped_negative_dependency_lags": int(
                    base["clamped_negative_dependency_lags"]
                ),
                "fallback_dependencies": int(base["fallback_dependencies"]),
                "service_marginals_available": bool(calculate_service_marginals),
            }
        )

    call_result = pd.concat(base_calls, ignore_index=True)
    group_columns = [
        "iteration",
        "pp_stage",
        "behavior",
        "kind",
        "round",
        "group_key",
        "group_size",
        "observed_ranks",
    ]
    group_aggregation = {
        "actual_first_start_ns": ("start_ns", "min"),
        "actual_ready_ns": ("start_ns", "max"),
        "actual_end_ns": ("end_ns", "max"),
        "predicted_first_start_ns": ("predicted_start_ns", "min"),
        "predicted_ready_ns": ("predicted_start_ns", "max"),
        "predicted_end_ns": ("predicted_end_ns", "max"),
        "baseline_service_ns": ("service_ns", "first"),
        "predicted_service_ns": ("predicted_service_ns", "first"),
        "fct_group_id": ("fct_group_id", "first"),
        "fct_request_id": ("fct_request_id", "first"),
        "fct_source": ("fct_source", "first"),
        "oisa_collective_elapsed_ns": ("oisa_collective_elapsed_ns", "first"),
        "oisa_tail_after_last_release_ns": (
            "oisa_tail_after_last_release_ns",
            "first",
        ),
        "predicted_collective_elapsed_ns": (
            "predicted_collective_elapsed_ns",
            "first",
        ),
        "dependency_lag_ns_min": ("dependency_lag_ns", "min"),
        "dependency_lag_ns_max": ("dependency_lag_ns", "max"),
    }
    if "payload_bytes" in call_result:
        group_aggregation["payload_bytes"] = ("payload_bytes", "median")
    groups = (
        call_result.groupby(group_columns, as_index=False)
        .agg(**group_aggregation)
        .sort_values(["iteration", "pp_stage", "kind", "group_key"])
        .reset_index(drop=True)
    )
    groups["actual_arrival_skew_ns"] = (
        groups["actual_ready_ns"] - groups["actual_first_start_ns"]
    )
    groups["predicted_arrival_skew_ns"] = (
        groups["predicted_ready_ns"] - groups["predicted_first_start_ns"]
    )
    groups["actual_group_fct_ns"] = groups["actual_end_ns"] - groups[
        "actual_first_start_ns"
    ]
    groups["predicted_group_fct_ns"] = groups["predicted_end_ns"] - groups[
        "predicted_first_start_ns"
    ]
    return TimelineResult(
        calls=call_result.sort_values(
            ["iteration", "pp_stage", "kind", "group_key", "rank"]
        ).reset_index(drop=True),
        groups=groups,
        iterations=pd.DataFrame(iteration_rows).sort_values("iteration").reset_index(drop=True),
    )


def build_collective_slack_audit(
    calls: pd.DataFrame,
    clocks: pd.DataFrame,
    *,
    reference_provider: CollectiveFctProvider,
    candidate_provider: CollectiveFctProvider,
    front_anchors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Measure how a one-group OISA slowdown propagates to iteration time.

    The reference graph is replayed once.  A reverse max-plus pass then derives
    the total downstream float of every group.  For a one-group slowdown
    ``delta``, the exact iteration drag is ``max(0, delta - downstream_slack)``.
    This is equivalent to isolated replays for non-negative FCT changes, but it
    does not issue hundreds of redundant provider queries.
    """
    facts = validate_optimizer_calls(calls)
    clock_frame = validate_iteration_clocks(clocks)
    anchor_frame = validate_front_anchors(front_anchors) if front_anchors is not None else None
    rows: list[dict[str, object]] = []

    for iteration, actual in facts.groupby("iteration", sort=True):
        selected_clock = clock_frame[clock_frame["iteration"].eq(iteration)]
        if len(selected_clock) != 1:
            raise ValueError(f"missing unique clock for slack audit iteration {iteration}")
        clock = selected_clock.iloc[0]
        iteration_anchors = None
        if anchor_frame is not None:
            selected_anchors = anchor_frame[anchor_frame["iteration"].eq(iteration)]
            iteration_anchors = {
                int(row.rank): int(row.predicted_start_ns)
                for row in selected_anchors.itertuples(index=False)
            }

        reference_calls, reference_metrics = _simulate_iteration(
            actual,
            clock,
            front_anchors=iteration_anchors,
            fct_provider=reference_provider,
        )
        reference_last_ag = int(reference_metrics["predicted_last_ag_ns"])
        reference_groups = (
            reference_calls.groupby("fct_group_id", sort=False)
            .agg(
                iteration=("iteration", "first"),
                pp_stage=("pp_stage", "first"),
                behavior=("behavior", "first"),
                kind=("kind", "first"),
                round=("round", "first"),
                group_key=("group_key", "first"),
                group_size=("group_size", "first"),
                observed_ranks=("observed_ranks", "first"),
                payload_bytes=("payload_bytes", "median")
                if "payload_bytes" in reference_calls
                else ("group_size", "size"),
                reference_first_start_ns=("predicted_start_ns", "min"),
                reference_ready_ns=("predicted_start_ns", "max"),
                reference_group_end_ns=("predicted_group_end_ns", "first"),
                reference_tail_ns=("oisa_tail_after_last_release_ns", "first"),
            )
            .reset_index()
        )

        # Convert rank-level dependency facts into a compact group max-plus
        # graph.  An edge src->dst with weight w encodes end(dst)>=end(src)+w.
        rank_call = {
            (str(row.kind), int(row.pp_stage), int(row.rank)): row
            for row in reference_calls.itertuples(index=False)
        }
        group_for_rank_call = {
            key: str(row.fct_group_id) for key, row in rank_call.items()
        }
        group_service = {
            str(group_id): int(frame["predicted_service_ns"].iloc[0])
            for group_id, frame in reference_calls.groupby("fct_group_id", sort=False)
        }
        group_end = {
            str(group_id): int(frame["predicted_group_end_ns"].iloc[0])
            for group_id, frame in reference_calls.groupby("fct_group_id", sort=False)
        }
        group_kind = {
            str(group_id): str(frame["kind"].iloc[0])
            for group_id, frame in reference_calls.groupby("fct_group_id", sort=False)
        }
        edge_weights: dict[tuple[str, str], int] = {}

        def add_edge(source: str, destination: str, weight: int) -> None:
            key = (source, destination)
            edge_weights[key] = max(edge_weights.get(key, weight), weight)

        rs_group_ids = sorted(
            group_id for group_id, kind in group_kind.items() if kind in RS_KINDS
        )
        for destination, target in reference_calls.groupby(
            "fct_group_id", sort=False
        ):
            destination = str(destination)
            kind = str(target["kind"].iloc[0])
            service = group_service[destination]
            if kind == "dp_rs":
                continue
            dependencies = set(target["dependency"])
            if dependencies == {"world_rs_done"}:
                max_lag = int(target["dependency_lag_ns"].max())
                for source in rs_group_ids:
                    # Every group_end is the max completion of one observed
                    # rank in that group, hence its best completion advance is
                    # zero and world_rs_done=max(group_end).
                    add_edge(source, destination, max_lag + service)
                continue
            for row in target.itertuples(index=False):
                dependency = str(row.dependency)
                if dependency.startswith("same_rank_"):
                    previous_kind = dependency.removeprefix("same_rank_")
                    previous_key = (
                        previous_kind,
                        int(row.pp_stage),
                        int(row.rank),
                    )
                    previous = rank_call[previous_key]
                elif dependency.startswith("stage_fallback_"):
                    previous_kind = dependency.removeprefix("stage_fallback_")
                    candidates = reference_calls[
                        reference_calls["kind"].eq(previous_kind)
                        & reference_calls["pp_stage"].eq(int(row.pp_stage))
                    ]
                    previous = next(
                        candidate
                        for candidate in candidates.itertuples(index=False)
                        if int(candidate.predicted_end_ns)
                        == int(candidates["predicted_end_ns"].max())
                    )
                    previous_key = (
                        previous_kind,
                        int(previous.pp_stage),
                        int(previous.rank),
                    )
                else:
                    raise ValueError(
                        f"unsupported slack-audit dependency {dependency!r}"
                    )
                source = group_for_rank_call[previous_key]
                weight = (
                    -int(previous.predicted_completion_advance_ns)
                    + int(row.dependency_lag_ns)
                    + service
                )
                add_edge(source, destination, weight)

        sink = "virtual:optimizer_end"
        for source, kind in group_kind.items():
            if kind in AG_KINDS:
                add_edge(source, sink, 0)
        successors: dict[str, list[tuple[str, int]]] = {
            group_id: [] for group_id in group_kind
        }
        successors[sink] = []
        for (source, destination), weight in edge_weights.items():
            successors[source].append((destination, weight))
        remaining = {sink: 0}
        kind_order = {kind: index for index, kind in enumerate(TIMELINE_KINDS)}
        reverse_nodes = sorted(
            group_kind,
            key=lambda group_id: kind_order[group_kind[group_id]],
            reverse=True,
        )
        for group_id in reverse_nodes:
            options = [
                weight + remaining[destination]
                for destination, weight in successors[group_id]
                if destination in remaining
            ]
            if not options:
                raise ValueError(f"collective group has no path to optimizer end: {group_id}")
            remaining[group_id] = max(options)
        downstream_slack = {
            group_id: reference_last_ag - (group_end[group_id] + remaining[group_id])
            for group_id in group_kind
        }
        if min(downstream_slack.values()) < 0:
            raise ValueError("derived collective downstream slack is negative")

        for baseline in reference_groups.itertuples(index=False):
            group_id = str(baseline.fct_group_id)
            target = reference_calls[reference_calls["fct_group_id"].eq(group_id)]
            request = make_collective_request(target)
            response = validate_fct_result(request, candidate_provider(request))
            simulated_tail = int(response.tail_after_last_release_ns)
            group_shift = simulated_tail - int(baseline.reference_tail_ns)
            if group_shift < 0:
                raise ValueError(
                    "analytical slack audit supports slowdown/equal FCT only; "
                    f"group {group_id} changed by {group_shift} ns"
                )
            slack = downstream_slack[group_id]
            iteration_drag = max(group_shift - slack, 0)
            hidden = group_shift - iteration_drag
            simulated_first = int(baseline.reference_first_start_ns)
            simulated_ready = int(baseline.reference_ready_ns)
            simulated_end = int(baseline.reference_group_end_ns) + group_shift
            if group_shift > 0:
                exposure_ratio = iteration_drag / group_shift
                critical_before = slack == 0
                critical_after = group_shift >= slack
            else:
                exposure_ratio = np.nan
                critical_before = False
                critical_after = False
            rows.append(
                {
                    "iteration": int(baseline.iteration),
                    "pp_stage": int(baseline.pp_stage),
                    "behavior": str(baseline.behavior),
                    "kind": str(baseline.kind),
                    "round": int(baseline.round),
                    "group_key": str(baseline.group_key),
                    "fct_group_id": str(baseline.fct_group_id),
                    "group_size": int(baseline.group_size),
                    "observed_ranks": int(baseline.observed_ranks),
                    "payload_bytes": int(baseline.payload_bytes),
                    "reference_first_start_ns": int(baseline.reference_first_start_ns),
                    "reference_ready_ns": int(baseline.reference_ready_ns),
                    "reference_group_end_ns": int(baseline.reference_group_end_ns),
                    "simulated_first_start_ns": simulated_first,
                    "simulated_ready_ns": simulated_ready,
                    "simulated_group_end_ns": simulated_end,
                    "reference_tail_after_last_release_ns": int(
                        baseline.reference_tail_ns
                    ),
                    "simulated_tail_after_last_release_ns": simulated_tail,
                    "network_tail_shift_ns": (
                        simulated_tail - int(baseline.reference_tail_ns)
                    ),
                    "group_finish_shift_ns": group_shift,
                    "iteration_drag_ns": iteration_drag,
                    "hidden_by_slack_ns": hidden,
                    "downstream_slack_ns": slack,
                    "exposure_ratio": exposure_ratio,
                    "fully_hidden": bool(group_shift > 0 and iteration_drag == 0),
                    "critical_before": bool(critical_before),
                    "critical_after": bool(critical_after),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["iteration", "pp_stage", "kind", "round", "group_key"]
    ).reset_index(drop=True)

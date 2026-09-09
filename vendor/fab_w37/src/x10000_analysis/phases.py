from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .intervals import Interval, LabeledInterval, merge_intervals


@dataclass(frozen=True)
class DirectPhaseResult:
    iter_windows: pd.DataFrame
    phase_envelopes: pd.DataFrame
    atomic_intervals: pd.DataFrame
    stage_share: pd.DataFrame


def is_deepep_kernel_name(name: object) -> bool:
    text = str(name)
    return "deep_ep::" in text or "musa::dnn::reduce_column::ReduceColumnShflKernel" in text


def classify_collective_stage(pg_description: object, collective: object, size_bytes: int) -> str:
    pg = str(pg_description or "")
    op = str(collective or "").lower()
    if pg == "EXPERT_DATA_PARALLEL_GROUP":
        if "reduce_scatter" in op:
            return "edp_rs_scaleout"
        if "allgather" in op or "all_gather" in op:
            return "edp_ag_scaleout"
    if pg == "DATA_PARALLEL_GROUP_WITH_CP":
        if "reduce_scatter" in op:
            return "dp_rs_hybrid"
        if "allgather" in op or "all_gather" in op:
            return "dp_ag_hybrid"
        if "allreduce" in op or "all_reduce" in op:
            return "gradient_finalize_sync"
    if pg == "DATA_PARALLEL_GROUP":
        return "tiny_sync_collective"
    if pg == "default_pg":
        if "allreduce" in op or "all_reduce" in op or "barrier" in op:
            return "tiny_sync_collective"
        return "other_communication"
    return "other_communication"


def _merge_pairs(
    starts: Iterable[int],
    ends: Iterable[int],
    window: Interval,
) -> list[Interval]:
    intervals: list[Interval] = []
    for start_value, end_value in zip(starts, ends):
        start = max(window.start_ns, int(start_value))
        end = min(window.end_ns, int(end_value))
        if end > start:
            intervals.append(Interval(start, end))
    return merge_intervals(intervals)


def _duration(intervals: Iterable[Interval]) -> int:
    return sum(interval.duration_ns for interval in merge_intervals(intervals))


def _duration_union(intervals: Iterable[Interval]) -> int:
    return _duration(intervals)


def _extract_stage_names(labels: frozenset[str]) -> set[str]:
    return _stages_from_labels(labels)


def _merged_for_label(
    frame: pd.DataFrame,
    window: Interval,
    label: str,
) -> list[LabeledInterval]:
    intervals: list[Interval] = []
    for row in frame.itertuples(index=False):
        start_ns = max(window.start_ns, int(row.start_ns))
        end_ns = min(window.end_ns, int(row.end_ns))
        if end_ns > start_ns:
            intervals.append(Interval(start_ns, end_ns))
    return [
        LabeledInterval(interval.start_ns, interval.end_ns, label)
        for interval in merge_intervals(intervals)
    ]


def _sweep_atomic(
    window: Interval,
    labeled: list[LabeledInterval],
) -> list[tuple[int, int, frozenset[str]]]:
    changes: dict[int, Counter[str]] = defaultdict(Counter)
    changes[window.start_ns]
    changes[window.end_ns]
    for interval in labeled:
        start = max(window.start_ns, interval.start_ns)
        end = min(window.end_ns, interval.end_ns)
        if end <= start:
            continue
        changes[start][interval.label] += 1
        changes[end][interval.label] -= 1

    active: Counter[str] = Counter()
    result: list[tuple[int, int, frozenset[str]]] = []
    previous: int | None = None
    for point in sorted(changes):
        if previous is not None and point > previous:
            labels = frozenset(label for label, count in active.items() if count > 0)
            result.append((previous, point, labels))
        for label, delta in changes[point].items():
            active[label] += delta
            if active[label] == 0:
                del active[label]
        previous = point
    return result


def _stages_from_labels(labels: frozenset[str]) -> set[str]:
    stages: set[str] = set()
    for label in labels:
        for prefix in ("comm_exposure::", "comm_service::", "deepep_active::"):
            if label.startswith(prefix):
                stages.add(label.removeprefix(prefix))
                break
    return stages


def _exclusive_category(labels: frozenset[str]) -> str:
    compute = "compute_active" in labels
    stages = sorted(_stages_from_labels(labels))
    if not stages:
        return "compute_only" if compute else "idle_or_unattributed"
    if len(stages) == 1:
        return f"compute_comm_overlap::{stages[0]}" if compute else stages[0]
    return "multi_stage_compute_overlap" if compute else "multi_stage_overlap"


def build_direct_phase_intervals(
    profiler_events: pd.DataFrame,
    deepep_events: pd.DataFrame,
    calls: pd.DataFrame,
) -> DirectPhaseResult:
    """Build direct, epoch-aligned phase facts for profiled iterations."""
    window_rows: list[dict[str, object]] = []
    envelope_rows: list[dict[str, object]] = []
    atomic_rows: list[dict[str, object]] = []
    share_rows: list[dict[str, object]] = []

    profiler_iters = sorted(int(value) for value in profiler_events["iter"].unique())
    for iter_id in profiler_iters:
        p_iter = profiler_events[profiler_events["iter"] == iter_id]
        step_mask = np.logical_and(
            p_iter["event_type"] == "user_annotation",
            p_iter["name"].astype(str).str.startswith("ProfilerStep#"),
        )
        steps = p_iter[step_mask]
        if steps.empty:
            raise ValueError(f"Missing ProfilerStep user annotation for iter {iter_id}")
        window = Interval(int(steps["start_ns"].min()), int(steps["end_ns"].max()))
        window_rows.append(
            {
                "iter": iter_id,
                "start_ns": window.start_ns,
                "end_ns": window.end_ns,
                "wall_ns": window.duration_ns,
                "rank_count": int(steps["rank"].nunique()),
                "provenance": "direct_profiled",
            }
        )

        phase_name_map = {
            "forward_step": "forward_phase",
            "backward_step": "backward_phase",
            "finalize_model_grads": "gradient_finalize_phase",
            "step": "optimizer_phase",
        }
        user = p_iter[p_iter["event_type"] == "user_annotation"]
        for annotation, phase in phase_name_map.items():
            selected = user[user["name"] == annotation]
            merged = _merge_pairs(selected["start_ns"], selected["end_ns"], window)
            for interval in merged:
                envelope_rows.append(
                    {
                        "iter": iter_id,
                        "phase": phase,
                        "start_ns": interval.start_ns,
                        "end_ns": interval.end_ns,
                        "duration_ns": interval.duration_ns,
                        "provenance": "direct_profiled_annotation_envelope",
                    }
                )

        labeled: list[LabeledInterval] = []
        exposure_by_stage: dict[str, list[Interval]] = defaultdict(list)
        service_by_stage: dict[str, list[Interval]] = defaultdict(list)

        compute = p_iter[p_iter["event_type"] == "compute_kernel"]
        if not compute.empty:
            compute = compute[~compute["name"].map(is_deepep_kernel_name)]
        compute_merged = _merge_pairs(compute["start_ns"], compute["end_ns"], window)
        for interval in compute_merged:
            labeled.append(LabeledInterval(interval.start_ns, interval.end_ns, "compute_active"))

        d_iter = deepep_events[deepep_events["iter"] == iter_id] if not deepep_events.empty else deepep_events
        if not d_iter.empty:
            for stage, group in d_iter.groupby("stage_hint"):
                merged = _merge_pairs(group["start_ns"], group["end_ns"], window)
                exposure_by_stage[str(stage)].extend(merged)
                for interval in merged:
                    labeled.append(
                        LabeledInterval(
                            interval.start_ns,
                            interval.end_ns,
                            f"deepep_active::{stage}",
                        )
                    )

        c_iter = calls[calls["iter"] == iter_id] if not calls.empty else calls
        for row in c_iter.itertuples(index=False):
            stage = classify_collective_stage(
                getattr(row, "pg_description", None),
                row.collective,
                int(row.size_bytes),
            )
            exposure = Interval(
                max(window.start_ns, int(row.earliest_start_ns)),
                min(window.end_ns, int(row.latest_end_ns)),
            )
            service = Interval(
                max(window.start_ns, int(row.latest_start_ns)),
                min(window.end_ns, int(row.latest_end_ns)),
            )
            exposure_by_stage[stage].append(exposure)
            service_by_stage[stage].append(service)

        for stage in list(exposure_by_stage):
            exposure_by_stage[stage] = merge_intervals(exposure_by_stage[stage])
            for interval in exposure_by_stage[stage]:
                labeled.append(
                    LabeledInterval(
                        interval.start_ns,
                        interval.end_ns,
                        f"comm_exposure::{stage}",
                    )
                )
        for stage in list(service_by_stage):
            service_by_stage[stage] = merge_intervals(service_by_stage[stage])
            for interval in service_by_stage[stage]:
                labeled.append(
                    LabeledInterval(
                        interval.start_ns,
                        interval.end_ns,
                        f"comm_service::{stage}",
                    )
                )

        atomic = _sweep_atomic(window, labeled)
        for start_ns, end_ns, labels in atomic:
            atomic_rows.append(
                {
                    "iter": iter_id,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": end_ns - start_ns,
                    "active_labels": "|".join(sorted(labels)),
                    "active_stage_count": len(_stages_from_labels(labels)),
                    "compute_active": "compute_active" in labels,
                    "exclusive_category": _exclusive_category(labels),
                    "provenance": "direct_profiled_atomic_partition",
                }
            )

        stage_names = sorted(exposure_by_stage)
        for stage in stage_names:
            exposure_ns = _duration(exposure_by_stage[stage])
            service_ns_value: float | int
            service_provenance: str
            if stage in service_by_stage:
                service_ns_value = _duration(service_by_stage[stage])
                service_provenance = "direct_profiled_post_last_arrival"
            else:
                service_ns_value = float("nan")
                service_provenance = "not_identifiable_from_deepep_logger"
            exclusive_ns = 0
            overlap_compute_ns = 0
            for start_ns, end_ns, labels in atomic:
                stages = _stages_from_labels(labels)
                duration_ns = end_ns - start_ns
                if stage not in stages:
                    continue
                if "compute_active" in labels:
                    overlap_compute_ns += duration_ns
                if stages == {stage} and "compute_active" not in labels:
                    exclusive_ns += duration_ns
            share_rows.append(
                {
                    "iter": iter_id,
                    "stage": stage,
                    "wall_ns": window.duration_ns,
                    "critical_exposure_ns": exposure_ns,
                    "service_ns": service_ns_value,
                    "exclusive_wall_ns": exclusive_ns,
                    "overlap_with_compute_ns": overlap_compute_ns,
                    "critical_exposure_pct": exposure_ns / window.duration_ns * 100,
                    "service_pct": service_ns_value / window.duration_ns * 100,
                    "exclusive_wall_pct": exclusive_ns / window.duration_ns * 100,
                    "overlap_with_compute_pct": overlap_compute_ns / window.duration_ns * 100,
                    "service_provenance": service_provenance,
                    "provenance": "direct_profiled",
                }
            )

        all_comm = merge_intervals(
            [interval for intervals in exposure_by_stage.values() for interval in intervals]
        )
        all_service = merge_intervals(
            [interval for intervals in service_by_stage.values() for interval in intervals]
        )
        comm_ns = _duration(all_comm)
        known_service_ns = _duration(all_service)
        compute_comm_overlap_ns = sum(
            end_ns - start_ns
            for start_ns, end_ns, labels in atomic
            if "compute_active" in labels and _stages_from_labels(labels)
        )
        share_rows.extend(
            [
                {
                    "iter": iter_id,
                    "stage": "compute_active",
                    "wall_ns": window.duration_ns,
                    "critical_exposure_ns": _duration(compute_merged),
                    "service_ns": float("nan"),
                    "exclusive_wall_ns": sum(
                        end_ns - start_ns
                        for start_ns, end_ns, labels in atomic
                        if _exclusive_category(labels) == "compute_only"
                    ),
                    "overlap_with_compute_ns": 0,
                    "critical_exposure_pct": _duration(compute_merged) / window.duration_ns * 100,
                    "service_pct": float("nan"),
                    "exclusive_wall_pct": sum(
                        end_ns - start_ns
                        for start_ns, end_ns, labels in atomic
                        if _exclusive_category(labels) == "compute_only"
                    )
                    / window.duration_ns
                    * 100,
                    "overlap_with_compute_pct": 0.0,
                    "service_provenance": "not_applicable",
                    "provenance": "direct_profiled",
                },
                {
                    "iter": iter_id,
                    "stage": "all_communication",
                    "wall_ns": window.duration_ns,
                    "critical_exposure_ns": comm_ns,
                    "service_ns": known_service_ns,
                    "exclusive_wall_ns": sum(
                        end_ns - start_ns
                        for start_ns, end_ns, labels in atomic
                        if _stages_from_labels(labels) and "compute_active" not in labels
                    ),
                    "overlap_with_compute_ns": compute_comm_overlap_ns,
                    "critical_exposure_pct": comm_ns / window.duration_ns * 100,
                    "service_pct": known_service_ns / window.duration_ns * 100,
                    "exclusive_wall_pct": sum(
                        end_ns - start_ns
                        for start_ns, end_ns, labels in atomic
                        if _stages_from_labels(labels) and "compute_active" not in labels
                    )
                    / window.duration_ns
                    * 100,
                    "overlap_with_compute_pct": compute_comm_overlap_ns / window.duration_ns * 100,
                    "service_provenance": "collective_only_lower_bound_excludes_deepep",
                    "provenance": "direct_profiled",
                },
            ]
        )

    return DirectPhaseResult(
        iter_windows=pd.DataFrame(window_rows),
        phase_envelopes=pd.DataFrame(envelope_rows),
        atomic_intervals=pd.DataFrame(atomic_rows),
        stage_share=pd.DataFrame(share_rows),
    )


def build_full_cycle_phase_share(
    windows: pd.DataFrame,
    deepep_events: pd.DataFrame,
    direct_calls: pd.DataFrame,
    inferred_collective_intervals: pd.DataFrame,
    direct_compute_intervals: pd.DataFrame,
) -> DirectPhaseResult:
    """Build a consistent all-iter cycle view with explicit provenance and uncertainty."""
    window_rows: list[dict[str, object]] = []
    atomic_rows: list[dict[str, object]] = []
    share_rows: list[dict[str, object]] = []

    for window_row in windows.sort_values("iter").itertuples(index=False):
        iter_id = int(window_row.iter)
        window = Interval(int(window_row.start_ns), int(window_row.end_ns))
        labeled: list[LabeledInterval] = []
        exposure: dict[str, list[Interval]] = defaultdict(list)
        service: dict[str, list[Interval]] = defaultdict(list)
        provenance: dict[str, str] = {}

        compute = direct_compute_intervals[
            direct_compute_intervals["iter"] == iter_id
        ] if not direct_compute_intervals.empty else pd.DataFrame()
        if not compute.empty:
            merged_compute = _merged_for_label(
                compute,
                window,
                "compute_active",
            )
            labeled.extend(merged_compute)

        deep = deepep_events[deepep_events["iter"] == iter_id]
        for stage, group in deep.groupby("stage_hint"):
            intervals = _merged_for_label(group, window, f"deepep_active::{stage}")
            labeled.extend(intervals)
            exposure[str(stage)].extend(
                Interval(item.start_ns, item.end_ns) for item in intervals
            )
            provenance[str(stage)] = "direct_deepep_instrumented"

        calls = direct_calls[direct_calls["iter"] == iter_id] if not direct_calls.empty else pd.DataFrame()
        if not calls.empty:
            for call in calls.itertuples(index=False):
                stage = classify_collective_stage(
                    call.pg_description,
                    call.collective,
                    int(call.size_bytes),
                )
                exp_start = max(window.start_ns, int(call.earliest_start_ns))
                exp_end = min(window.end_ns, int(call.latest_end_ns))
                if exp_end > exp_start:
                    exposure[stage].append(Interval(exp_start, exp_end))
                svc_start = max(window.start_ns, int(call.latest_start_ns))
                svc_end = min(window.end_ns, int(call.latest_end_ns))
                if svc_end > svc_start:
                    service[stage].append(Interval(svc_start, svc_end))
                provenance[stage] = "direct_profiled_collective"

        inferred = inferred_collective_intervals[
            inferred_collective_intervals["iter"] == iter_id
        ] if not inferred_collective_intervals.empty else pd.DataFrame()
        if not inferred.empty:
            for interval_row in inferred.itertuples(index=False):
                stage = str(interval_row.stage)
                start = max(window.start_ns, int(interval_row.start_ns))
                end = min(window.end_ns, int(interval_row.end_ns))
                if end > start:
                    exposure[stage].append(Interval(start, end))
                    provenance[stage] = str(interval_row.provenance)

        for stage, intervals in exposure.items():
            for interval in merge_intervals(intervals):
                labeled.append(
                    LabeledInterval(
                        interval.start_ns,
                        interval.end_ns,
                        f"comm_exposure::{stage}",
                    )
                )
        for stage, intervals in service.items():
            for interval in merge_intervals(intervals):
                labeled.append(
                    LabeledInterval(
                        interval.start_ns,
                        interval.end_ns,
                        f"comm_service::{stage}",
                    )
                )

        atomic = _sweep_atomic(window, labeled)
        for start_ns, end_ns, labels in atomic:
            atomic_rows.append(
                {
                    "iter": iter_id,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": end_ns - start_ns,
                    "active_labels": "|".join(sorted(labels)),
                    "exclusive_category": _exclusive_category(labels),
                    "provenance": "cycle_v5",
                }
            )

        wall_ns = window.duration_ns
        stages = sorted(exposure)
        for stage in stages:
            exposure_ns = _duration_union(exposure[stage])
            service_ns = _duration_union(service.get(stage, []))
            exclusive_ns = 0
            overlap_ns = 0
            for start_ns, end_ns, labels in atomic:
                stage_names = _extract_stage_names(labels)
                if stage not in stage_names:
                    continue
                duration_ns = end_ns - start_ns
                if "compute_active" in labels:
                    overlap_ns += duration_ns
                elif len(stage_names) == 1:
                    exclusive_ns += duration_ns
            has_service = bool(service.get(stage))
            compute_observed = not compute.empty
            reported_exclusive_ns = exclusive_ns if compute_observed else np.nan
            reported_overlap_ns = overlap_ns if compute_observed else np.nan
            share_rows.append(
                {
                    "iter": iter_id,
                    "stage": stage,
                    "wall_ns": wall_ns,
                    "critical_exposure_ns": exposure_ns,
                    "service_ns": service_ns if has_service else np.nan,
                    "exclusive_ns": reported_exclusive_ns,
                    "overlap_with_compute_ns": reported_overlap_ns,
                    "critical_exposure_pct": exposure_ns / wall_ns * 100,
                    "service_pct": service_ns / wall_ns * 100 if has_service else np.nan,
                    "exclusive_wall_pct": (
                        exclusive_ns / wall_ns * 100 if compute_observed else np.nan
                    ),
                    "overlap_with_compute_pct": (
                        overlap_ns / wall_ns * 100 if compute_observed else np.nan
                    ),
                    "service_provenance": (
                        "direct_profiled_post_last_arrival" if has_service else "not_observed"
                    ),
                    "provenance": provenance[stage],
                }
            )

        all_comm = [interval for intervals in exposure.values() for interval in intervals]
        all_comm_ns = _duration_union(all_comm)
        comm_overlap_ns = sum(
            end_ns - start_ns
            for start_ns, end_ns, labels in atomic
            if _extract_stage_names(labels) and "compute_active" in labels
        )
        share_rows.append(
            {
                "iter": iter_id,
                "stage": "all_communication",
                "wall_ns": wall_ns,
                "critical_exposure_ns": all_comm_ns,
                "service_ns": np.nan,
                "exclusive_ns": np.nan,
                "overlap_with_compute_ns": comm_overlap_ns if not compute.empty else np.nan,
                "critical_exposure_pct": all_comm_ns / wall_ns * 100,
                "service_pct": np.nan,
                "exclusive_wall_pct": np.nan,
                "overlap_with_compute_pct": (
                    comm_overlap_ns / wall_ns * 100 if not compute.empty else np.nan
                ),
                "service_provenance": "mixed_direct_and_unobserved",
                "provenance": "derived_union",
            }
        )
        compute_ns = sum(
            end_ns - start_ns
            for start_ns, end_ns, labels in atomic
            if "compute_active" in labels
        )
        share_rows.append(
            {
                "iter": iter_id,
                "stage": "compute_active",
                "wall_ns": wall_ns,
                "critical_exposure_ns": compute_ns,
                "service_ns": np.nan,
                "exclusive_ns": np.nan,
                "overlap_with_compute_ns": np.nan,
                "critical_exposure_pct": compute_ns / wall_ns * 100,
                "service_pct": np.nan,
                "exclusive_wall_pct": np.nan,
                "overlap_with_compute_pct": np.nan,
                "service_provenance": "not_applicable",
                "provenance": "direct_profiled_kernel_union" if not compute.empty else "not_observed",
            }
        )
        idle_ns = sum(
            end_ns - start_ns
            for start_ns, end_ns, labels in atomic
            if not labels
        )
        share_rows.append(
            {
                "iter": iter_id,
                "stage": "idle_or_unattributed",
                "wall_ns": wall_ns,
                "critical_exposure_ns": idle_ns,
                "service_ns": np.nan,
                "exclusive_ns": idle_ns,
                "overlap_with_compute_ns": 0,
                "critical_exposure_pct": idle_ns / wall_ns * 100,
                "service_pct": np.nan,
                "exclusive_wall_pct": idle_ns / wall_ns * 100,
                "overlap_with_compute_pct": 0.0,
                "service_provenance": "not_applicable",
                "provenance": "direct_gap" if not compute.empty else "compute_not_observed",
            }
        )
        window_rows.append(
            {
                "iter": iter_id,
                "start_ns": window.start_ns,
                "end_ns": window.end_ns,
                "wall_ns": wall_ns,
                "window_provenance": getattr(window_row, "window_provenance", "provided"),
            }
        )

    return DirectPhaseResult(
        iter_windows=pd.DataFrame(window_rows),
        phase_envelopes=pd.DataFrame(),
        atomic_intervals=pd.DataFrame(atomic_rows),
        stage_share=pd.DataFrame(share_rows),
    )


def build_rank_phase_intervals(
    profiler_events: pd.DataFrame,
    deepep_events: pd.DataFrame,
    call_metadata: pd.DataFrame,
    call_per_rank: pd.DataFrame,
) -> DirectPhaseResult:
    """Build the same phase facts independently for each rank's own step window."""
    metadata_fields = ["pg_description", "collective", "size_bytes"]
    missing_fields = [field for field in metadata_fields if field not in call_per_rank.columns]
    if missing_fields:
        metadata = call_metadata[["call_id", *missing_fields]].drop_duplicates("call_id")
        rank_calls = call_per_rank.merge(
            metadata,
            on="call_id",
            how="left",
            validate="many_to_one",
        )
    else:
        rank_calls = call_per_rank.copy()
    if rank_calls[metadata_fields].isna().any().any():
        raise ValueError("call_per_rank contains call_id missing from call_metadata")

    all_windows: list[pd.DataFrame] = []
    all_envelopes: list[pd.DataFrame] = []
    all_atomics: list[pd.DataFrame] = []
    all_shares: list[pd.DataFrame] = []
    identities = (
        profiler_events[["iter", "rank", "host", "local_rank"]]
        .drop_duplicates()
        .sort_values(["iter", "rank"])
    )
    for identity in identities.itertuples(index=False):
        iter_id = int(identity.iter)
        rank = int(identity.rank)
        p_rank = profiler_events[
            np.logical_and(profiler_events["iter"] == iter_id, profiler_events["rank"] == rank)
        ]
        if deepep_events.empty:
            d_rank = deepep_events
        else:
            d_rank = deepep_events[
                np.logical_and(deepep_events["iter"] == iter_id, deepep_events["rank"] == rank)
            ]
        c_rank = rank_calls[
            np.logical_and(rank_calls["iter"] == iter_id, rank_calls["rank"] == rank)
        ].copy()
        c_rank["earliest_start_ns"] = c_rank["start_ns"]
        c_rank["latest_start_ns"] = c_rank["start_ns"]
        c_rank["latest_end_ns"] = c_rank["end_ns"]

        result = build_direct_phase_intervals(p_rank, d_rank, c_rank)
        frames = (
            (result.iter_windows, all_windows),
            (result.phase_envelopes, all_envelopes),
            (result.atomic_intervals, all_atomics),
            (result.stage_share, all_shares),
        )
        for frame, destination in frames:
            enriched = frame.copy()
            enriched["rank"] = rank
            enriched["host"] = identity.host
            enriched["local_rank"] = int(identity.local_rank)
            destination.append(enriched)
        call_stages = {
            classify_collective_stage(row.pg_description, row.collective, int(row.size_bytes))
            for row in c_rank.itertuples(index=False)
        }
        share_frame = all_shares[-1]
        share_frame.loc[
            share_frame["stage"].isin(call_stages), "service_provenance"
        ] = "per_rank_kernel_duration_includes_wait"
        share_frame.loc[
            share_frame["stage"] == "all_communication", "service_provenance"
        ] = "per_rank_collective_kernel_union_excludes_deepep_service"

    return DirectPhaseResult(
        iter_windows=pd.concat(all_windows, ignore_index=True),
        phase_envelopes=pd.concat(all_envelopes, ignore_index=True)
        if all_envelopes
        else pd.DataFrame(),
        atomic_intervals=pd.concat(all_atomics, ignore_index=True),
        stage_share=pd.concat(all_shares, ignore_index=True),
    )

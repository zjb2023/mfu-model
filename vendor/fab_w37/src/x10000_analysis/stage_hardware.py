from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from .intervals import Interval, merge_intervals


def counter_stream_overlap_pct(
    counters: pd.DataFrame,
    stream_columns: list[str] | tuple[str, ...] = (),
) -> float:
    """Measure within-stream interval overlap while preserving distinct links."""
    required = {"start_ns", "end_ns", "duration_ns", *stream_columns}
    missing = sorted(required - set(counters.columns))
    if missing:
        raise ValueError(f"counters missing columns: {missing}")
    actual_duration = counters["end_ns"].astype("int64") - counters["start_ns"].astype("int64")
    if not actual_duration.equals(counters["duration_ns"].astype("int64")):
        raise ValueError("counter duration_ns differs from end_ns - start_ns")
    if counters.empty:
        return 0.0
    groups: Iterable[tuple[object, pd.DataFrame]]
    if stream_columns:
        groups = counters.groupby(list(stream_columns), sort=False, dropna=False)
    else:
        groups = [("single_stream", counters)]
    total_duration = 0
    union_duration = 0
    for _, group in groups:
        intervals = [
            Interval(int(start), int(end))
            for start, end in zip(group["start_ns"], group["end_ns"])
            if int(end) > int(start)
        ]
        total_duration += sum(interval.duration_ns for interval in intervals)
        union_duration += sum(interval.duration_ns for interval in merge_intervals(intervals))
    if total_duration <= 0:
        return 0.0
    return 100.0 * (total_duration - union_duration) / total_duration


def classify_deepep_kernel_stage(name: object) -> str | None:
    """Classify exact DeepEP profiler kernels using the colleague-dashboard semantics."""
    text = str(name)
    if "deep_ep::intranode::" not in text:
        return None
    if "notify_dispatch" in text or "notify_combine" in text:
        return "deepep_notify"
    if "::dispatch<" in text or "::combine<" in text:
        return "deepep_comm"
    return None


def nic_device_for_local_rank(local_rank: int) -> str:
    """Map the two local GPUs shown in each dashboard panel to one bond."""
    value = int(local_rank)
    if value < 0 or value > 7:
        raise ValueError(f"local_rank must be in [0, 7], got {local_rank}")
    return f"mlx5_bond_{2 + value // 2}_port1"


def nominal_nic_tx_capacity_gbps(active_rank_count: int) -> float:
    count = int(active_rank_count)
    if count not in (1, 2):
        raise ValueError(f"active_rank_count must be 1 or 2, got {active_rank_count}")
    return 200.0 * count


def parallel_fabric_critical_reduction_ns(
    current_duration_ns: dict[str, float | int],
    target_duration_ns: dict[str, float | int],
) -> float:
    if not current_duration_ns or set(current_duration_ns) != set(target_duration_ns):
        raise ValueError("current and target fabric maps must be non-empty with identical keys")
    current_critical = max(float(value) for value in current_duration_ns.values())
    target_critical = max(float(value) for value in target_duration_ns.values())
    return max(0.0, current_critical - target_critical)


def resolve_bulk_stage_over_control(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve an overlap only when one bulk-data stage overlaps control traffic.

    This preserves the colleague dashboard's visual semantics for cases such as
    EDP ReduceScatter overlapping a 4-byte synchronization AllReduce, while
    retaining ambiguity when two bulk stages overlap.
    """
    if "stage" not in frame or "active_stages" not in frame:
        raise ValueError("frame must contain stage and active_stages")
    bulk = {
        "deepep_comm",
        "dp_rs_hybrid",
        "dp_ag_hybrid",
        "edp_rs_scaleout",
        "edp_ag_scaleout",
    }
    control = {
        "deepep_notify",
        "tiny_sync_collective",
        "gradient_finalize_sync",
        "other_communication",
    }
    result = frame.copy()
    resolutions: list[str] = []
    stages_out: list[str] = []
    for row in result.itertuples(index=False):
        stage = str(row.stage)
        stages = set(str(row.active_stages).split("|"))
        if stage != "multi_stage_overlap":
            stages_out.append(stage)
            resolutions.append("not_overlapping")
            continue
        bulk_stages = sorted(stages & bulk)
        other_stages = stages - set(bulk_stages)
        if len(bulk_stages) == 1 and other_stages <= control:
            stages_out.append(bulk_stages[0])
            resolutions.append("single_bulk_stage_dominates_control")
        else:
            stages_out.append(stage)
            resolutions.append("unresolved_multiple_bulk_stages")
    result["stage"] = stages_out
    result["overlap_resolution"] = resolutions
    return result


def _active_rank_count(active: dict[str, Counter[int]]) -> int:
    if not active:
        return 0
    return max(len([rank for rank, count in ranks.items() if count > 0]) for ranks in active.values())


def build_atomic_entity_segments(intervals: pd.DataFrame) -> pd.DataFrame:
    """Partition labeled rank intervals without convex-envelope or double attribution.

    A segment with multiple active semantic stages is labeled ``multi_stage_overlap``
    rather than assigning the same hardware bytes to every stage. ``active_rank_count``
    is the maximum same-stage rank concurrency, which exposes one-vs-two GPU demand on
    a shared NIC bond.
    """
    required = {"iter", "host", "device", "rank", "stage", "start_ns", "end_ns"}
    missing = sorted(required - set(intervals.columns))
    if missing:
        raise ValueError(f"intervals missing columns: {missing}")

    rows: list[dict[str, object]] = []
    group_cols = ["iter", "host", "device"]
    for (iter_id, host, device), group in intervals.groupby(group_cols, sort=True):
        changes: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
        for row in group.itertuples(index=False):
            start_ns = int(row.start_ns)
            end_ns = int(row.end_ns)
            if end_ns <= start_ns:
                raise ValueError(f"non-positive interval for iter={iter_id}, device={device}")
            stage = str(row.stage)
            rank = int(row.rank)
            changes[start_ns].append((stage, rank, 1))
            changes[end_ns].append((stage, rank, -1))

        active: dict[str, Counter[int]] = defaultdict(Counter)
        previous: int | None = None
        for point in sorted(changes):
            if previous is not None and point > previous:
                stages = sorted(
                    stage for stage, ranks in active.items() if any(count > 0 for count in ranks.values())
                )
                if stages:
                    rows.append(
                        {
                            "iter": int(iter_id),
                            "host": str(host),
                            "device": str(device),
                            "stage": stages[0] if len(stages) == 1 else "multi_stage_overlap",
                            "active_stages": "|".join(stages),
                            "active_rank_count": _active_rank_count({stage: active[stage] for stage in stages}),
                            "start_ns": previous,
                            "end_ns": point,
                            "duration_ns": point - previous,
                            "provenance": "direct_profiled_atomic_hardware_stage",
                        }
                    )
            for stage, rank, delta in changes[point]:
                active[stage][rank] += delta
                if active[stage][rank] == 0:
                    del active[stage][rank]
                if not active[stage]:
                    del active[stage]
            previous = point

    columns = [
        "iter", "host", "device", "stage", "active_stages", "active_rank_count",
        "start_ns", "end_ns", "duration_ns", "provenance",
    ]
    return pd.DataFrame(rows, columns=columns)


def _union_duration_ns(starts: Iterable[int], ends: Iterable[int], window: Interval) -> int:
    clipped: list[Interval] = []
    for start_ns, end_ns in zip(starts, ends):
        start = max(window.start_ns, int(start_ns))
        end = min(window.end_ns, int(end_ns))
        if end > start:
            clipped.append(Interval(start, end))
    return sum(interval.duration_ns for interval in merge_intervals(clipped))


def allocate_counters_to_segments(
    counters: pd.DataFrame,
    segments: pd.DataFrame,
    rate_scale: float,
) -> pd.DataFrame:
    """Allocate interval-delta bytes by exact overlap fraction.

    ``rate_scale=8`` yields Gbps from bytes/ns; ``rate_scale=1`` yields decimal
    GB/s. Invalid or long-gap rows are retained as uncertainty but excluded from
    the allocatable rate numerator.
    """
    required_counter = {"start_ns", "end_ns", "duration_ns", "tx_bytes", "rx_bytes", "allocation_status"}
    required_segment = {
        "iter", "host", "device", "stage", "active_stages", "active_rank_count",
        "start_ns", "end_ns", "duration_ns",
    }
    missing_counter = sorted(required_counter - set(counters.columns))
    missing_segment = sorted(required_segment - set(segments.columns))
    if missing_counter:
        raise ValueError(f"counters missing columns: {missing_counter}")
    if missing_segment:
        raise ValueError(f"segments missing columns: {missing_segment}")

    if counters.empty or segments.empty:
        return segments.iloc[0:0].copy()

    counter = counters.sort_values("start_ns").reset_index(drop=True)
    starts = counter["start_ns"].to_numpy(dtype=np.int64)
    ends = counter["end_ns"].to_numpy(dtype=np.int64)
    prefix_max_ends = np.maximum.accumulate(ends)
    durations = counter["duration_ns"].to_numpy(dtype=np.int64)
    tx = counter["tx_bytes"].to_numpy(dtype=np.float64)
    rx = counter["rx_bytes"].to_numpy(dtype=np.float64)
    valid = counter["allocation_status"].eq("allocatable").to_numpy()

    rows: list[dict[str, object]] = []
    for segment in segments.itertuples(index=False):
        seg_start = int(segment.start_ns)
        seg_end = int(segment.end_ns)
        left = int(np.searchsorted(prefix_max_ends, seg_start, side="right"))
        right = int(np.searchsorted(starts, seg_end, side="left"))
        overlap = np.maximum(0, np.minimum(ends[left:right], seg_end) - np.maximum(starts[left:right], seg_start))
        fractions = np.divide(
            overlap,
            durations[left:right],
            out=np.zeros_like(overlap, dtype=np.float64),
            where=durations[left:right] > 0,
        )
        valid_slice = valid[left:right]
        allocated_tx = float(np.sum(tx[left:right] * fractions * valid_slice))
        allocated_rx = float(np.sum(rx[left:right] * fractions * valid_slice))
        uncertain_tx = float(np.sum(tx[left:right] * fractions * ~valid_slice))
        uncertain_rx = float(np.sum(rx[left:right] * fractions * ~valid_slice))
        window = Interval(seg_start, seg_end)
        coverage = _union_duration_ns(
            starts[left:right][valid_slice],
            ends[left:right][valid_slice],
            window,
        )
        duration_ns = int(segment.duration_ns)
        base = segment._asdict()
        base.update(
            {
                "allocated_tx_bytes": allocated_tx,
                "allocated_rx_bytes": allocated_rx,
                "uncertain_tx_bytes": uncertain_tx,
                "uncertain_rx_bytes": uncertain_rx,
                "tx_rate": allocated_tx * float(rate_scale) / duration_ns,
                "rx_rate": allocated_rx * float(rate_scale) / duration_ns,
                "bidir_rate": (allocated_tx + allocated_rx) * float(rate_scale) / duration_ns,
                "coverage_pct": min(100.0, 100.0 * coverage / duration_ns),
                "allocation_provenance": "counter_interval_overlap_fraction",
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)

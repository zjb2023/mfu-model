#!/usr/bin/env python3
"""Build per-GPU EP payload / MTLink-active-time audit tables.

The DeepEP payload and elapsed time remain the canonical phase-level logical
measurement.  MTLink samples are used as interval-censored activity evidence:
the duration of the sample intervals is unioned across all 14 links, never
summed per link.  Samples shared by multiple EP phases or other no-PP
collectives are retained as quality metadata instead of being silently
declared exclusive phase traffic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PHASES = (
    "fwd_dispatch",
    "fwd_combine",
    "bwd_dispatch",
    "bwd_combine",
)
BEHAVIOR_TO_PHASE = {
    "ep_fwd_dispatch": "fwd_dispatch",
    "ep_fwd_combine": "fwd_combine",
    "ep_bwd_combine_backward_dispatch": "bwd_dispatch",
    "ep_bwd_dispatch_backward_combine": "bwd_combine",
}
PHASE_TO_BEHAVIOR = {value: key for key, value in BEHAVIOR_TO_PHASE.items()}
MAX_I64 = np.uint64(2**63 - 1)
PCIE_LINK_ID = 2**32 - 1
COMPOUND_BEHAVIOR = "recompute_fwd_combine_plus_bwd_dispatch"


@dataclass(frozen=True)
class RankTask:
    rank: int
    event_frame: pd.DataFrame
    other_frame: pd.DataFrame
    mtlink_paths: tuple[Path, ...]
    iterations: tuple[int, ...]
    min_link_tx_rate_gbps: float
    long_gap_ns: int


def parse_args() -> argparse.Namespace:
    default_case = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, default=default_case)
    parser.add_argument("--event-parquet", type=Path)
    parser.add_argument("--source-cell-csv", type=Path)
    parser.add_argument("--rank-topology", type=Path)
    parser.add_argument("--mtlink-manifest", type=Path)
    parser.add_argument("--hardware-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--min-link-tx-rate-gbps",
        type=float,
        default=0.05,
        help="Per-link TX activity floor in decimal GB/s (bytes/ns).",
    )
    parser.add_argument("--mtlink-long-gap-ms", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Regenerate validation summary/audit Markdown from existing output CSVs.",
    )
    parser.add_argument(
        "--refresh-quality-only",
        action="store_true",
        help="Recompute derived quality fractions/confidence in existing CSVs, then audit.",
    )
    parser.add_argument(
        "--ranks",
        help="Optional comma-separated rank subset for development/audit.",
    )
    args = parser.parse_args()
    case_root = args.case_root.resolve()
    collective = case_root / "results" / "collective_bw_no_pp"
    args.event_parquet = args.event_parquet or (
        collective / "collective_events_no_pp_pre_mtlink.parquet"
    )
    args.source_cell_csv = args.source_cell_csv or (
        collective / "collective_iter_rank_bandwidth_core_no_pp.csv"
    )
    args.rank_topology = args.rank_topology or (
        case_root / "results" / "topology" / "rank_topology.csv"
    )
    args.mtlink_manifest = args.mtlink_manifest or (
        case_root / "results" / "readiness" / "mtlink_files.csv"
    )
    args.hardware_root = args.hardware_root or (case_root / ".raw_view")
    args.output_dir = args.output_dir or (
        case_root / "results" / "ep_active_time_bw"
    )
    if args.min_link_tx_rate_gbps < 0:
        parser.error("--min-link-tx-rate-gbps must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_rank_subset(text: str | None, available: Iterable[int]) -> list[int]:
    all_ranks = sorted(int(value) for value in available)
    if not text:
        return all_ranks
    selected = sorted({int(piece.strip()) for piece in text.split(",") if piece.strip()})
    unknown = sorted(set(selected) - set(all_ranks))
    if unknown:
        raise ValueError(f"requested ranks are absent from events: {unknown}")
    return selected


def resolve_manifest_paths(manifest_path: Path, hardware_root: Path) -> dict[tuple[str, int], tuple[Path, ...]]:
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("error"):
                continue
            key = (str(row["host"]), int(row["gpu_id"]))
            path = hardware_root / str(row["path"])
            if not path.exists():
                raise FileNotFoundError(path)
            grouped[key].append(path)
    return {key: tuple(sorted(paths)) for key, paths in grouped.items()}


def merge_interval_arrays(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.lexsort((ends, starts))
    starts = starts[order].astype(np.int64, copy=False)
    ends = ends[order].astype(np.int64, copy=False)
    merged_starts: list[int] = []
    merged_ends: list[int] = []
    current_start = int(starts[0])
    current_end = int(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        start_i = int(start)
        end_i = int(end)
        if start_i <= current_end:
            current_end = max(current_end, end_i)
        else:
            merged_starts.append(current_start)
            merged_ends.append(current_end)
            current_start = start_i
            current_end = end_i
    merged_starts.append(current_start)
    merged_ends.append(current_end)
    return np.asarray(merged_starts, dtype=np.int64), np.asarray(merged_ends, dtype=np.int64)


def overlaps_merged_intervals(
    sample_starts: np.ndarray,
    sample_ends: np.ndarray,
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
) -> np.ndarray:
    if len(interval_starts) == 0:
        return np.zeros(len(sample_starts), dtype=bool)
    index = np.searchsorted(interval_ends, sample_starts, side="right")
    safe = np.minimum(index, len(interval_starts) - 1)
    return (index < len(interval_starts)) & (interval_starts[safe] < sample_ends)


def grouped_interval_union(
    groups: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    group_count: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = np.zeros(group_count, dtype=np.int64)
    first = np.full(group_count, -1, dtype=np.int64)
    last = np.full(group_count, -1, dtype=np.int64)
    if mask is not None:
        groups = groups[mask]
        starts = starts[mask]
        ends = ends[mask]
    valid = (groups >= 0) & (starts < ends)
    groups = groups[valid].astype(np.int64, copy=False)
    starts = starts[valid].astype(np.int64, copy=False)
    ends = ends[valid].astype(np.int64, copy=False)
    if len(groups) == 0:
        return duration, first, last
    order = np.lexsort((ends, starts, groups))
    groups = groups[order]
    starts = starts[order]
    ends = ends[order]
    current_group = int(groups[0])
    current_start = int(starts[0])
    current_end = int(ends[0])
    first[current_group] = current_start
    for group, start, end in zip(groups[1:], starts[1:], ends[1:]):
        group_i = int(group)
        start_i = int(start)
        end_i = int(end)
        if group_i == current_group and start_i <= current_end:
            current_end = max(current_end, end_i)
            continue
        duration[current_group] += current_end - current_start
        last[current_group] = current_end
        current_group = group_i
        current_start = start_i
        current_end = end_i
        if first[current_group] < 0:
            first[current_group] = current_start
    duration[current_group] += current_end - current_start
    last[current_group] = current_end
    return duration, first, last


def grouped_median(groups: np.ndarray, values: np.ndarray, group_count: int) -> np.ndarray:
    result = np.full(group_count, np.nan, dtype=np.float64)
    if len(groups) == 0:
        return result
    frame = pd.DataFrame({"group": groups, "value": values})
    medians = frame.groupby("group", sort=False)["value"].median()
    result[medians.index.to_numpy(dtype=np.int64)] = medians.to_numpy(dtype=np.float64)
    return result


def confidence_label(
    resolution_ratio: float,
    shared_fraction: float,
    other_fraction: float,
    long_gap_fraction: float,
    active_count: int,
) -> str:
    if active_count == 0 or not math.isfinite(resolution_ratio):
        return "NO_ACTIVITY"
    if (
        resolution_ratio >= 2.0
        and shared_fraction < 0.10
        and other_fraction < 0.10
        and long_gap_fraction < 0.10
    ):
        return "HIGH"
    if (
        resolution_ratio >= 1.0
        and shared_fraction < 0.30
        and other_fraction < 0.30
        and long_gap_fraction < 0.30
    ):
        return "MEDIUM"
    return "LOW"


def build_event_facts(events: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    result = events.copy().sort_values(["rank", "start_ns", "end_ns", "behavior"]).reset_index(drop=True)
    result["phase"] = result["behavior"].map(BEHAVIOR_TO_PHASE)
    if result["phase"].isna().any():
        raise ValueError("unmapped DeepEP behavior")
    topo = topology[["rank", "pp_stage", "local_rank", "host", "gpu_id"]].copy()
    topo = topo.rename(columns={"host": "topology_host", "gpu_id": "topology_gpu_id"})
    result = result.merge(topo, on="rank", how="left", validate="many_to_one")
    if (
        result["pp_stage"].isna().any()
        or result["host"].ne(result["topology_host"]).any()
        or result["gpu_id"].ne(result["topology_gpu_id"]).any()
    ):
        raise ValueError("DeepEP event identity differs from rank topology")
    result = result.drop(columns=["topology_host", "topology_gpu_id"])
    result["previous_behavior"] = result.groupby("rank", sort=False)["behavior"].shift()
    result["previous_end_ns"] = result.groupby("rank", sort=False)["end_ns"].shift()
    result["gap_from_previous_ns"] = result["start_ns"] - result["previous_end_ns"]
    result["iteration_phase_call_index"] = result.groupby(
        ["rank", "iteration", "phase"], sort=False
    ).cumcount()
    result["rank_event_index"] = result.groupby("rank", sort=False).cumcount()
    result["compound_member"] = False
    result["compound_pair_id"] = pd.NA
    bwd_mask = result["behavior"].eq("ep_bwd_combine_backward_dispatch").to_numpy()
    if result.loc[bwd_mask, "rank_event_index"].eq(0).any():
        ranks = result.loc[bwd_mask & result["rank_event_index"].eq(0), "rank"].tolist()
        raise ValueError(f"bwd-dispatch has no predecessor on ranks {ranks}")
    bad_predecessor = bwd_mask & result["previous_behavior"].ne("ep_fwd_combine").to_numpy()
    if np.any(bad_predecessor):
        example = result.loc[
            bad_predecessor, ["rank", "event_id", "previous_behavior"]
        ].head(10)
        raise ValueError(f"bwd-dispatch predecessor mismatch:\n{example}")
    bwd_index = np.flatnonzero(bwd_mask)
    previous_index = bwd_index - 1
    pair_ids = (
        "r"
        + result.loc[bwd_index, "rank"].astype(str).to_numpy()
        + ":e"
        + result.loc[bwd_index, "event_id"].astype(str).to_numpy()
    )
    member_column = result.columns.get_loc("compound_member")
    pair_column = result.columns.get_loc("compound_pair_id")
    result.iloc[np.concatenate([previous_index, bwd_index]), member_column] = True
    result.iloc[previous_index, pair_column] = pair_ids
    result.iloc[bwd_index, pair_column] = pair_ids
    return result


def read_samples(paths: tuple[Path, ...], min_rate: float) -> pd.DataFrame:
    usecols = [
        "host_realtime_ns",
        "link_id",
        "tx_delta_bytes",
        "rx_delta_bytes",
        "mt_timestamp_begin_ns",
        "mt_timestamp_end_ns",
    ]
    frames = [
        pd.read_csv(
            path,
            usecols=usecols,
            dtype={
                "host_realtime_ns": "int64",
                "link_id": "int64",
                "tx_delta_bytes": "uint64",
                "rx_delta_bytes": "uint64",
                "mt_timestamp_begin_ns": "int64",
                "mt_timestamp_end_ns": "int64",
            },
        )
        for path in paths
    ]
    samples = pd.concat(frames, ignore_index=True)
    samples = samples[
        samples["link_id"].between(0, 13)
        & samples["link_id"].ne(PCIE_LINK_ID)
    ].drop_duplicates()
    duration = (
        samples["mt_timestamp_end_ns"].to_numpy(dtype=np.int64)
        - samples["mt_timestamp_begin_ns"].to_numpy(dtype=np.int64)
    )
    tx = samples["tx_delta_bytes"].to_numpy(dtype=np.uint64)
    rx = samples["rx_delta_bytes"].to_numpy(dtype=np.uint64)
    valid = (duration > 0) & (tx <= MAX_I64) & (rx <= MAX_I64)
    rate = np.zeros(len(samples), dtype=np.float64)
    rate[valid] = tx[valid].astype(np.float64) / duration[valid]
    active = valid & (tx > 0) & (rate >= min_rate)
    samples = samples.loc[active].copy()
    duration = duration[active]
    samples["sample_duration_ns"] = duration
    samples["sample_end_ns"] = samples["host_realtime_ns"].to_numpy(dtype=np.int64)
    samples["sample_start_ns"] = samples["sample_end_ns"].to_numpy(dtype=np.int64) - duration
    samples["link_tx_rate_GBps"] = rate[active]
    return samples.sort_values("sample_end_ns", kind="stable").reset_index(drop=True)


def build_pairs(events: pd.DataFrame, samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_start = samples["sample_start_ns"].to_numpy(dtype=np.int64)
    sample_end = samples["sample_end_ns"].to_numpy(dtype=np.int64)
    sample_duration = samples["sample_duration_ns"].to_numpy(dtype=np.int64)
    maximum_duration = int(sample_duration.max())
    duration_bins: list[tuple[np.ndarray, int]] = []
    lower = 0
    for upper in (10_000_000, 25_000_000, 50_000_000, 100_000_000, maximum_duration):
        upper = int(upper)
        if upper < lower:
            continue
        indices = np.flatnonzero((sample_duration > lower) & (sample_duration <= upper))
        if len(indices):
            duration_bins.append((indices, upper))
        lower = upper
    pair_samples: list[np.ndarray] = []
    pair_events: list[np.ndarray] = []
    pair_overlaps: list[np.ndarray] = []
    starts = events["start_ns"].to_numpy(dtype=np.int64)
    ends = events["end_ns"].to_numpy(dtype=np.int64)
    for event_index, (event_start, event_end) in enumerate(zip(starts, ends)):
        for bin_indices, upper in duration_bins:
            bin_ends = sample_end[bin_indices]
            left = int(np.searchsorted(bin_ends, event_start, side="right"))
            right = int(np.searchsorted(bin_ends, event_end + upper, side="right"))
            if right <= left:
                continue
            indices = bin_indices[left:right]
            overlap = np.minimum(sample_end[indices], event_end) - np.maximum(
                sample_start[indices], event_start
            )
            keep = overlap > 0
            if not np.any(keep):
                continue
            indices = indices[keep]
            pair_samples.append(indices)
            pair_events.append(np.full(len(indices), event_index, dtype=np.int64))
            pair_overlaps.append(overlap[keep].astype(np.int64))
    if not pair_samples:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    return (
        np.concatenate(pair_samples),
        np.concatenate(pair_events),
        np.concatenate(pair_overlaps),
    )


def metric_arrays(
    group_index: np.ndarray,
    group_count: int,
    sample_index: np.ndarray,
    overlap_ns: np.ndarray,
    samples: pd.DataFrame,
    ep_pair_count: np.ndarray,
    other_sample: np.ndarray,
    external_shared: np.ndarray | None = None,
    normalization_overlap_sum: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    sample_duration = samples["sample_duration_ns"].to_numpy(dtype=np.int64)
    tx = samples["tx_delta_bytes"].to_numpy(dtype=np.uint64).astype(np.float64)
    if external_shared is None:
        shared_pair = ep_pair_count[sample_index] > 1
    else:
        shared_pair = external_shared
    other_pair = other_sample[sample_index]
    exclusive_pair = (~shared_pair) & (~other_pair)

    # Quality fractions count each physical sample once per output group.  The
    # pair count is retained separately because one 5-ms sample can touch more
    # than one sub-ms DeepEP call in the same iteration/phase cell.
    sample_count = len(samples)
    unique_key = group_index.astype(np.int64) * sample_count + sample_index
    _, unique_position = np.unique(unique_key, return_index=True)
    unique_group = group_index[unique_position]
    unique_sample = sample_index[unique_position]
    unique_shared = shared_pair[unique_position]
    unique_other = other_pair[unique_position]
    unique_exclusive = exclusive_pair[unique_position]
    output: dict[str, np.ndarray] = {}
    output["sample_pair_count"] = np.bincount(group_index, minlength=group_count)
    output["touched_sample_count"] = np.bincount(
        unique_group, minlength=group_count
    )
    output["shared_sample_count"] = np.bincount(
        unique_group, weights=unique_shared, minlength=group_count
    )
    output["other_collective_sample_count"] = np.bincount(
        unique_group, weights=unique_other, minlength=group_count
    )
    output["exclusive_sample_count"] = np.bincount(
        unique_group, weights=unique_exclusive, minlength=group_count
    )
    output["touched_sample_duration_p50_ns"] = grouped_median(
        unique_group, sample_duration[unique_sample], group_count
    )
    output["mtlink_proportional_tx_bytes"] = np.bincount(
        group_index,
        weights=tx[sample_index] * overlap_ns / sample_duration[sample_index],
        minlength=group_count,
    )
    if normalization_overlap_sum is None:
        normalization_overlap_sum = np.bincount(
            sample_index, weights=overlap_ns, minlength=len(samples)
        )
    normalized_weight = overlap_ns / normalization_overlap_sum[sample_index]
    output["mtlink_ep_normalized_tx_bytes"] = np.bincount(
        group_index,
        weights=tx[sample_index] * normalized_weight,
        minlength=group_count,
    )
    output["mtlink_exclusive_full_bucket_tx_bytes"] = np.bincount(
        group_index[exclusive_pair],
        weights=tx[sample_index[exclusive_pair]],
        minlength=group_count,
    )
    output["long_gap_sample_count"] = np.zeros(group_count, dtype=np.float64)
    return output


def add_interval_metrics(
    output: dict[str, np.ndarray],
    group_index: np.ndarray,
    group_count: int,
    pair_event_index: np.ndarray,
    sample_index: np.ndarray,
    events: pd.DataFrame,
    samples: pd.DataFrame,
    ep_pair_count: np.ndarray,
    other_sample: np.ndarray,
    external_shared: np.ndarray | None = None,
    long_gap_ns: int = 10_000_000,
) -> None:
    event_start = events["start_ns"].to_numpy(dtype=np.int64)
    event_end = events["end_ns"].to_numpy(dtype=np.int64)
    sample_start = samples["sample_start_ns"].to_numpy(dtype=np.int64)
    sample_end = samples["sample_end_ns"].to_numpy(dtype=np.int64)
    sample_duration = samples["sample_duration_ns"].to_numpy(dtype=np.int64)
    clipped_start = np.maximum(sample_start[sample_index], event_start[pair_event_index])
    clipped_end = np.minimum(sample_end[sample_index], event_end[pair_event_index])
    if external_shared is None:
        shared_pair = ep_pair_count[sample_index] > 1
    else:
        shared_pair = external_shared
    other_pair = other_sample[sample_index]
    exclusive_pair = (~shared_pair) & (~other_pair)
    all_clipped, first_clipped, last_clipped = grouped_interval_union(
        group_index, clipped_start, clipped_end, group_count
    )
    all_full, first_full, last_full = grouped_interval_union(
        group_index,
        sample_start[sample_index],
        sample_end[sample_index],
        group_count,
    )
    exclusive_clipped, _, _ = grouped_interval_union(
        group_index, clipped_start, clipped_end, group_count, exclusive_pair
    )
    exclusive_full, _, _ = grouped_interval_union(
        group_index,
        sample_start[sample_index],
        sample_end[sample_index],
        group_count,
        exclusive_pair,
    )
    output["mtlink_active_time_clipped_ns"] = all_clipped
    output["mtlink_active_time_full_bucket_ns"] = all_full
    output["mtlink_exclusive_active_time_clipped_ns"] = exclusive_clipped
    output["mtlink_exclusive_active_time_full_bucket_ns"] = exclusive_full
    output["mtlink_active_start_ns"] = first_full
    output["mtlink_active_end_ns"] = last_full
    unique_key = group_index.astype(np.int64) * len(samples) + sample_index
    _, unique_position = np.unique(unique_key, return_index=True)
    output["long_gap_sample_count"] = np.bincount(
        group_index[unique_position],
        weights=sample_duration[sample_index[unique_position]] > long_gap_ns,
        minlength=group_count,
    )


def build_rows(
    base: pd.DataFrame,
    metrics: dict[str, np.ndarray],
    duration_for_resolution: np.ndarray,
) -> pd.DataFrame:
    result = base.reset_index(drop=True).copy()
    for name, values in metrics.items():
        result[name] = values
    active = result["touched_sample_count"].to_numpy(dtype=np.float64)
    result["shared_sample_fraction"] = np.divide(
        result["shared_sample_count"], active, out=np.zeros(len(result)), where=active > 0
    )
    result["other_collective_sample_fraction"] = np.divide(
        result["other_collective_sample_count"],
        active,
        out=np.zeros(len(result)),
        where=active > 0,
    )
    result["long_gap_sample_fraction"] = np.divide(
        result["long_gap_sample_count"],
        active,
        out=np.zeros(len(result)),
        where=active > 0,
    )
    sample_p50 = result["touched_sample_duration_p50_ns"].to_numpy(dtype=np.float64)
    result["event_to_sample_resolution_ratio"] = np.divide(
        duration_for_resolution,
        sample_p50,
        out=np.full(len(result), np.nan),
        where=sample_p50 > 0,
    )
    send = result["deepep_send_bytes"].to_numpy(dtype=np.float64)
    clipped = result["mtlink_active_time_clipped_ns"].to_numpy(dtype=np.float64)
    full = result["mtlink_active_time_full_bucket_ns"].to_numpy(dtype=np.float64)
    result["deepep_payload_bw_mtlink_clipped_GBps"] = np.divide(
        send, clipped, out=np.full(len(result), np.nan), where=clipped > 0
    )
    result["deepep_payload_bw_mtlink_full_bucket_GBps"] = np.divide(
        send, full, out=np.full(len(result), np.nan), where=full > 0
    )
    result["mtlink_proportional_tx_rate_GBps"] = np.divide(
        result["mtlink_proportional_tx_bytes"],
        clipped,
        out=np.full(len(result), np.nan),
        where=clipped > 0,
    )
    result["confidence"] = [
        confidence_label(
            float(ratio),
            float(shared),
            float(other),
            float(long_gap),
            int(count),
        )
        for ratio, shared, other, long_gap, count in zip(
            result["event_to_sample_resolution_ratio"],
            result["shared_sample_fraction"],
            result["other_collective_sample_fraction"],
            result["long_gap_sample_fraction"],
            result["touched_sample_count"],
        )
    ]
    return result


def refresh_quality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Refresh quality fields without rescanning raw MTLink samples."""

    result = frame.copy()
    active = result["touched_sample_count"].to_numpy(dtype=np.float64)
    result["shared_sample_fraction"] = np.divide(
        result["shared_sample_count"],
        active,
        out=np.zeros(len(result)),
        where=active > 0,
    )
    result["other_collective_sample_fraction"] = np.divide(
        result["other_collective_sample_count"],
        active,
        out=np.zeros(len(result)),
        where=active > 0,
    )
    result["long_gap_sample_fraction"] = np.divide(
        result["long_gap_sample_count"],
        active,
        out=np.zeros(len(result)),
        where=active > 0,
    )
    result["confidence"] = [
        confidence_label(
            float(ratio),
            float(shared),
            float(other),
            float(long_gap),
            int(count),
        )
        for ratio, shared, other, long_gap, count in zip(
            result["event_to_sample_resolution_ratio"],
            result["shared_sample_fraction"],
            result["other_collective_sample_fraction"],
            result["long_gap_sample_fraction"],
            result["touched_sample_count"],
        )
    ]
    return result


def process_rank(task: RankTask) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    events = task.event_frame.sort_values(["start_ns", "end_ns", "behavior"]).reset_index(drop=True)
    samples = read_samples(task.mtlink_paths, task.min_link_tx_rate_gbps)
    sample_index, event_index, overlap_ns = build_pairs(events, samples)
    if len(sample_index) == 0:
        raise RuntimeError(f"rank {task.rank} has no active MTLink/EP intersections")
    sample_count = len(samples)
    event_count = len(events)
    ep_pair_count = np.bincount(sample_index, minlength=sample_count)
    ep_overlap_sum = np.bincount(
        sample_index, weights=overlap_ns, minlength=sample_count
    )
    other_starts, other_ends = merge_interval_arrays(
        task.other_frame["start_ns"].to_numpy(dtype=np.int64),
        task.other_frame["end_ns"].to_numpy(dtype=np.int64),
    )
    other_sample = overlaps_merged_intervals(
        samples["sample_start_ns"].to_numpy(dtype=np.int64),
        samples["sample_end_ns"].to_numpy(dtype=np.int64),
        other_starts,
        other_ends,
    )

    event_metrics = metric_arrays(
        event_index,
        event_count,
        sample_index,
        overlap_ns,
        samples,
        ep_pair_count,
        other_sample,
        normalization_overlap_sum=ep_overlap_sum,
    )
    add_interval_metrics(
        event_metrics,
        event_index,
        event_count,
        event_index,
        sample_index,
        events,
        samples,
        ep_pair_count,
        other_sample,
        long_gap_ns=task.long_gap_ns,
    )
    call_base = events.rename(
        columns={
            "expected_tx_bytes": "deepep_send_bytes",
            "expected_rx_bytes": "deepep_recv_bytes",
            "logical_input_bytes": "deepep_comm_bytes",
            "duration_ns": "deepep_elapsed_ns",
        }
    ).copy()
    call_base["deepep_tx_bw_GBps"] = (
        call_base["deepep_send_bytes"] / call_base["deepep_elapsed_ns"]
    )
    call_rows = build_rows(
        call_base,
        event_metrics,
        call_base["deepep_elapsed_ns"].to_numpy(dtype=np.float64),
    )

    iteration_to_index = {iteration: index for index, iteration in enumerate(task.iterations)}
    phase_to_index = {phase: index for index, phase in enumerate(PHASES)}
    event_cell = np.asarray(
        [
            phase_to_index[phase] * len(task.iterations) + iteration_to_index[int(iteration)]
            for phase, iteration in zip(events["phase"], events["iteration"])
        ],
        dtype=np.int64,
    )
    cell_count = len(PHASES) * len(task.iterations)
    cell_group = event_cell[event_index]
    cell_metrics = metric_arrays(
        cell_group,
        cell_count,
        sample_index,
        overlap_ns,
        samples,
        ep_pair_count,
        other_sample,
        normalization_overlap_sum=ep_overlap_sum,
    )
    add_interval_metrics(
        cell_metrics,
        cell_group,
        cell_count,
        event_index,
        sample_index,
        events,
        samples,
        ep_pair_count,
        other_sample,
        long_gap_ns=task.long_gap_ns,
    )
    cell_event_count = np.bincount(event_cell, minlength=cell_count)
    cell_send = np.bincount(
        event_cell,
        weights=events["expected_tx_bytes"].to_numpy(dtype=np.float64),
        minlength=cell_count,
    )
    cell_recv = np.bincount(
        event_cell,
        weights=events["expected_rx_bytes"].to_numpy(dtype=np.float64),
        minlength=cell_count,
    )
    cell_elapsed = np.bincount(
        event_cell,
        weights=events["duration_ns"].to_numpy(dtype=np.float64),
        minlength=cell_count,
    )
    cell_duration_median = grouped_median(
        event_cell,
        events["duration_ns"].to_numpy(dtype=np.float64),
        cell_count,
    )
    cell_base_rows: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(PHASES):
        for iteration_index, iteration in enumerate(task.iterations):
            cell = phase_index * len(task.iterations) + iteration_index
            cell_base_rows.append(
                {
                    "iteration": iteration,
                    "rank": task.rank,
                    "host": str(events["host"].iloc[0]),
                    "gpu_id": int(events["gpu_id"].iloc[0]),
                    "local_rank": int(events["local_rank"].iloc[0]),
                    "pp_stage": int(events["pp_stage"].iloc[0]),
                    "phase": phase,
                    "behavior": PHASE_TO_BEHAVIOR[phase],
                    "event_count": int(cell_event_count[cell]),
                    "deepep_send_bytes": cell_send[cell],
                    "deepep_recv_bytes": cell_recv[cell],
                    "deepep_elapsed_ns": cell_elapsed[cell],
                    "deepep_tx_bw_GBps": (
                        cell_send[cell] / cell_elapsed[cell]
                        if cell_elapsed[cell] > 0
                        else np.nan
                    ),
                }
            )
    cell_rows = build_rows(
        pd.DataFrame(cell_base_rows),
        cell_metrics,
        cell_duration_median,
    )

    compound_member = events["compound_member"].to_numpy(dtype=bool)
    compound_event_cell = np.asarray(
        [iteration_to_index[int(value)] if member else -1 for value, member in zip(events["iteration"], compound_member)],
        dtype=np.int64,
    )
    compound_pair_mask = compound_member[event_index]
    compound_sample_index = sample_index[compound_pair_mask]
    compound_event_index = event_index[compound_pair_mask]
    compound_overlap = overlap_ns[compound_pair_mask]
    compound_group = compound_event_cell[compound_event_index]
    selected_pair_count = np.bincount(compound_sample_index, minlength=sample_count)
    external_shared = selected_pair_count[compound_sample_index] < ep_pair_count[compound_sample_index]
    compound_metrics = metric_arrays(
        compound_group,
        len(task.iterations),
        compound_sample_index,
        compound_overlap,
        samples,
        ep_pair_count,
        other_sample,
        external_shared,
        ep_overlap_sum,
    )
    add_interval_metrics(
        compound_metrics,
        compound_group,
        len(task.iterations),
        compound_event_index,
        compound_sample_index,
        events,
        samples,
        ep_pair_count,
        other_sample,
        external_shared,
        task.long_gap_ns,
    )
    compound_event_count = np.bincount(
        compound_event_cell[compound_member], minlength=len(task.iterations)
    )
    compound_send = np.bincount(
        compound_event_cell[compound_member],
        weights=events.loc[compound_member, "expected_tx_bytes"].to_numpy(dtype=np.float64),
        minlength=len(task.iterations),
    )
    compound_recv = np.bincount(
        compound_event_cell[compound_member],
        weights=events.loc[compound_member, "expected_rx_bytes"].to_numpy(dtype=np.float64),
        minlength=len(task.iterations),
    )
    compound_elapsed = np.bincount(
        compound_event_cell[compound_member],
        weights=events.loc[compound_member, "duration_ns"].to_numpy(dtype=np.float64),
        minlength=len(task.iterations),
    )
    # Sampling resolution for the compound is based on each causal pair's
    # envelope (fwd-combine start through bwd-dispatch end), including the
    # small dependency gap between them.  Using individual member durations
    # would understate how well a 5-ms bucket resolves the compound burst.
    pair_envelopes = (
        events.loc[
            compound_member,
            ["compound_pair_id", "iteration", "start_ns", "end_ns"],
        ]
        .groupby(["compound_pair_id", "iteration"], as_index=False, sort=False)
        .agg(pair_start_ns=("start_ns", "min"), pair_end_ns=("end_ns", "max"))
    )
    pair_envelopes["pair_envelope_ns"] = (
        pair_envelopes["pair_end_ns"] - pair_envelopes["pair_start_ns"]
    )
    envelope_by_iteration = pair_envelopes.groupby("iteration", sort=False)[
        "pair_envelope_ns"
    ].median()
    compound_duration_median = np.asarray(
        [float(envelope_by_iteration.get(iteration, np.nan)) for iteration in task.iterations],
        dtype=np.float64,
    )
    compound_base = pd.DataFrame(
        {
            "iteration": task.iterations,
            "rank": task.rank,
            "host": str(events["host"].iloc[0]),
            "gpu_id": int(events["gpu_id"].iloc[0]),
            "local_rank": int(events["local_rank"].iloc[0]),
            "pp_stage": int(events["pp_stage"].iloc[0]),
            "compound_behavior": COMPOUND_BEHAVIOR,
            "event_count": compound_event_count,
            "deepep_send_bytes": compound_send,
            "deepep_recv_bytes": compound_recv,
            "deepep_elapsed_ns": compound_elapsed,
            "pair_envelope_duration_p50_ns": compound_duration_median,
            "deepep_tx_bw_GBps": np.divide(
                compound_send,
                compound_elapsed,
                out=np.full(len(task.iterations), np.nan),
                where=compound_elapsed > 0,
            ),
        }
    )
    compound_rows = build_rows(
        compound_base,
        compound_metrics,
        compound_duration_median,
    )
    check = {
        "rank": task.rank,
        "event_count": event_count,
        "active_sample_count": sample_count,
        "event_sample_pair_count": int(len(sample_index)),
        "maximum_ep_proportional_weight_sum": float(
            np.bincount(
                sample_index,
                weights=overlap_ns
                / samples["sample_duration_ns"].to_numpy(dtype=np.int64)[sample_index],
                minlength=sample_count,
            ).max()
        ),
        "shared_active_sample_fraction": float(np.mean(ep_pair_count > 1)),
        "other_collective_active_sample_fraction": float(np.mean(other_sample)),
    }
    return call_rows, cell_rows, compound_rows, check


def portable(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def markdown_table(frame: pd.DataFrame, float_digits: int = 6) -> str:
    """Render a small audit table without an optional tabulate dependency."""

    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column).replace("|", "\\|") for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def audit_markdown(
    case_id: str,
    validation: dict[str, Any],
    phase_rows: pd.DataFrame,
    compound_rows: pd.DataFrame,
) -> str:
    phase_summary = (
        phase_rows.groupby("phase", as_index=False)
        .agg(
            event_count=("event_count", "sum"),
            deepep_send_bytes=("deepep_send_bytes", "sum"),
            deepep_elapsed_ns=("deepep_elapsed_ns", "sum"),
            sum_of_cell_active_clipped_ns=("mtlink_active_time_clipped_ns", "sum"),
            sum_of_cell_active_full_bucket_ns=("mtlink_active_time_full_bucket_ns", "sum"),
            shared_fraction_median=("shared_sample_fraction", "median"),
            resolution_ratio_median=("event_to_sample_resolution_ratio", "median"),
        )
        .sort_values("phase")
    )
    phase_summary["deepep_tx_bw_GBps"] = (
        phase_summary["deepep_send_bytes"] / phase_summary["deepep_elapsed_ns"]
    )
    phase_summary["payload_bw_full_bucket_GBps"] = (
        phase_summary["deepep_send_bytes"]
        / phase_summary["sum_of_cell_active_full_bucket_ns"]
    )
    confidence = (
        phase_rows.groupby(["phase", "confidence"]).size().unstack(fill_value=0)
    )
    trend_rows: list[dict[str, Any]] = []
    for phase, group in phase_rows.groupby("phase", sort=True):
        by_iteration = group.groupby("iteration", as_index=True).agg(
            deepep_tx_bw_GBps=("deepep_tx_bw_GBps", "median"),
            payload_full_bucket_GBps=(
                "deepep_payload_bw_mtlink_full_bucket_GBps",
                "median",
            ),
            mtlink_physical_rate_GBps=(
                "mtlink_proportional_tx_rate_GBps",
                "median",
            ),
        )
        trend_rows.append(
            {
                "phase": phase,
                "deepep_vs_physical_corr": by_iteration[
                    "deepep_tx_bw_GBps"
                ].corr(by_iteration["mtlink_physical_rate_GBps"]),
                "payload_full_vs_physical_corr": by_iteration[
                    "payload_full_bucket_GBps"
                ].corr(by_iteration["mtlink_physical_rate_GBps"]),
                "deepep_iter_first_GBps": by_iteration["deepep_tx_bw_GBps"].iloc[0],
                "deepep_iter_last_GBps": by_iteration["deepep_tx_bw_GBps"].iloc[-1],
                "physical_iter_first_GBps": by_iteration[
                    "mtlink_physical_rate_GBps"
                ].iloc[0],
                "physical_iter_last_GBps": by_iteration[
                    "mtlink_physical_rate_GBps"
                ].iloc[-1],
            }
        )
    compound_by_iteration = compound_rows.groupby("iteration", as_index=True).agg(
        deepep_tx_bw_GBps=("deepep_tx_bw_GBps", "median"),
        payload_full_bucket_GBps=(
            "deepep_payload_bw_mtlink_full_bucket_GBps",
            "median",
        ),
        mtlink_physical_rate_GBps=("mtlink_proportional_tx_rate_GBps", "median"),
    )
    compound_trend = pd.DataFrame(
        [
            {
                "compound": COMPOUND_BEHAVIOR,
                "deepep_vs_physical_corr": compound_by_iteration[
                    "deepep_tx_bw_GBps"
                ].corr(compound_by_iteration["mtlink_physical_rate_GBps"]),
                "payload_full_vs_physical_corr": compound_by_iteration[
                    "payload_full_bucket_GBps"
                ].corr(compound_by_iteration["mtlink_physical_rate_GBps"]),
                "deepep_iter_first_GBps": compound_by_iteration[
                    "deepep_tx_bw_GBps"
                ].iloc[0],
                "deepep_iter_last_GBps": compound_by_iteration[
                    "deepep_tx_bw_GBps"
                ].iloc[-1],
                "physical_iter_first_GBps": compound_by_iteration[
                    "mtlink_physical_rate_GBps"
                ].iloc[0],
                "physical_iter_last_GBps": compound_by_iteration[
                    "mtlink_physical_rate_GBps"
                ].iloc[-1],
            }
        ]
    )
    lines = [
        f"# {case_id} EP active-time BW audit",
        "",
        f"Status: **{validation['status']}**",
        "",
        "## Phase summary",
        "",
        markdown_table(phase_summary),
        "",
        "## Confidence cells",
        "",
        markdown_table(confidence.reset_index()),
        "",
        "## Iteration-trend diagnostic",
        "",
        markdown_table(pd.DataFrame(trend_rows)),
        "",
        "## Compound burst",
        "",
        markdown_table(
            compound_rows.groupby("compound_behavior", as_index=False).agg(
                event_count=("event_count", "sum"),
                deepep_send_bytes=("deepep_send_bytes", "sum"),
                sum_of_cell_active_full_bucket_ns=(
                    "mtlink_active_time_full_bucket_ns",
                    "sum",
                ),
                pair_envelope_p50_ns=("pair_envelope_duration_p50_ns", "median"),
                resolution_ratio_median=(
                    "event_to_sample_resolution_ratio",
                    "median",
                ),
                shared_fraction_median=("shared_sample_fraction", "median"),
            )
        ),
        "",
        markdown_table(compound_trend),
        "",
        "## Interpretation",
        "",
        "- DeepEP TX BW is logical payload / DeepEP elapsed time.",
        "- Payload/full-bucket time is an interval-censored reference, not physical MTLink BW.",
        "- `sum_of_cell_active_*` is a sum of per-cell unions; shared buckets can appear in more than one phase cell.",
        "- LOW cells must not be presented without the shared-sample and resolution warnings.",
        "- The recompute-fwd-combine + bwd-dispatch compound removes their sub-5-ms split claim.",
        "- Trend correlation is Pearson correlation across the 20 per-iteration rank medians; it is a consistency diagnostic, not a bytes-domain identity.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repository_root = args.case_root.resolve().parent
    if args.audit_only or args.refresh_quality_only:
        output_dir = args.output_dir
        if args.refresh_quality_only:
            for name in (
                "ep_phase_call_gpu.csv",
                "ep_phase_iter_gpu.csv",
                "ep_compound_burst_iter_gpu.csv",
            ):
                path = output_dir / name
                atomic_csv(refresh_quality_columns(pd.read_csv(path)), path)
        phase_rows = pd.read_csv(output_dir / "ep_phase_iter_gpu.csv")
        compound_rows = pd.read_csv(output_dir / "ep_compound_burst_iter_gpu.csv")
        validation_path = output_dir / "ep_active_time_bw_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["confidence_counts"] = clean_json(
            phase_rows["confidence"].value_counts().to_dict()
        )
        validation["confidence_by_phase"] = clean_json(
            phase_rows.groupby(["phase", "confidence"])
            .size()
            .unstack(fill_value=0)
            .to_dict("index")
        )
        validation["compound_confidence_counts"] = clean_json(
            compound_rows["confidence"].value_counts().to_dict()
        )
        atomic_text(
            validation_path,
            json.dumps(clean_json(validation), ensure_ascii=False, indent=2) + "\n",
        )
        atomic_text(
            output_dir / "ep_active_time_bw_audit.md",
            audit_markdown(
                args.case_root.resolve().name,
                validation,
                phase_rows,
                compound_rows,
            ),
        )
        print(
            json.dumps(
                {
                    "status": validation["status"],
                    "case_id": validation["case_id"],
                    "phase_row_count": len(phase_rows),
                    "compound_row_count": len(compound_rows),
                    "compound_confidence_counts": validation[
                        "compound_confidence_counts"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if str(validation["status"]).startswith("PASS") else 1
    event_columns = [
        "event_id",
        "source",
        "behavior",
        "iteration",
        "rank",
        "host",
        "gpu_id",
        "start_ns",
        "end_ns",
        "duration_ns",
        "logical_input_bytes",
        "expected_tx_bytes",
        "expected_rx_bytes",
        "source_path",
        "source_row",
    ]
    all_events = pd.read_parquet(args.event_parquet, columns=event_columns)
    deep = all_events[
        all_events["source"].eq("deepep")
        & all_events["behavior"].isin(BEHAVIOR_TO_PHASE)
    ].copy()
    other = all_events[
        ~all_events["source"].eq("deepep")
        & ~all_events["behavior"].isin(BEHAVIOR_TO_PHASE)
    ][["rank", "start_ns", "end_ns"]].copy()
    topology = pd.read_csv(args.rank_topology)
    events = build_event_facts(deep, topology)
    iterations = tuple(sorted(int(value) for value in events["iteration"].unique()))
    ranks = parse_rank_subset(args.ranks, events["rank"].unique())
    events = events[events["rank"].isin(ranks)].copy()
    other = other[other["rank"].isin(ranks)].copy()
    paths = resolve_manifest_paths(args.mtlink_manifest, args.hardware_root)
    tasks: list[RankTask] = []
    for rank in ranks:
        group = events[events["rank"].eq(rank)].copy()
        host = str(group["host"].iloc[0])
        gpu_id = int(group["gpu_id"].iloc[0])
        key = (host, gpu_id)
        if key not in paths:
            raise ValueError(f"missing MTLink files for {key}")
        tasks.append(
            RankTask(
                rank=rank,
                event_frame=group,
                other_frame=other[other["rank"].eq(rank)].copy(),
                mtlink_paths=paths[key],
                iterations=iterations,
                min_link_tx_rate_gbps=args.min_link_tx_rate_gbps,
                long_gap_ns=round(args.mtlink_long_gap_ms * 1_000_000),
            )
        )
    call_parts: list[pd.DataFrame] = []
    cell_parts: list[pd.DataFrame] = []
    compound_parts: list[pd.DataFrame] = []
    rank_checks: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            call, cell, compound, check = process_rank(task)
            call_parts.append(call)
            cell_parts.append(cell)
            compound_parts.append(compound)
            rank_checks.append(check)
            print(f"processed rank {task.rank}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_rank, task): task.rank for task in tasks}
            for future in as_completed(futures):
                rank = futures[future]
                call, cell, compound, check = future.result()
                call_parts.append(call)
                cell_parts.append(cell)
                compound_parts.append(compound)
                rank_checks.append(check)
                print(f"processed rank {rank}", flush=True)
    call_rows = pd.concat(call_parts, ignore_index=True).sort_values(
        ["rank", "start_ns", "end_ns"]
    )
    phase_rows = pd.concat(cell_parts, ignore_index=True).sort_values(
        ["phase", "iteration", "rank"]
    )
    compound_rows = pd.concat(compound_parts, ignore_index=True).sort_values(
        ["compound_behavior", "iteration", "rank"]
    )
    case_id = args.case_root.resolve().name
    for frame in (call_rows, phase_rows, compound_rows):
        frame.insert(0, "case_id", case_id)
        frame.insert(1, "min_link_tx_rate_GBps", args.min_link_tx_rate_gbps)
    output_dir = args.output_dir
    call_output = output_dir / "ep_phase_call_gpu.csv"
    phase_output = output_dir / "ep_phase_iter_gpu.csv"
    compound_output = output_dir / "ep_compound_burst_iter_gpu.csv"
    validation_output = output_dir / "ep_active_time_bw_validation.json"
    audit_output = output_dir / "ep_active_time_bw_audit.md"
    atomic_csv(call_rows, call_output)
    atomic_csv(phase_rows, phase_output)
    atomic_csv(compound_rows, compound_output)

    source = pd.read_csv(args.source_cell_csv)
    source = source[
        source["behavior"].isin(BEHAVIOR_TO_PHASE)
        & source["rank"].isin(ranks)
    ].copy()
    compare = phase_rows.merge(
        source[
            [
                "behavior",
                "iteration",
                "rank",
                "event_count",
                "duration_ns_sum",
                "expected_tx_bytes_total",
            ]
        ],
        on=["behavior", "iteration", "rank"],
        how="outer",
        suffixes=("_active", "_source"),
        validate="one_to_one",
    )
    deep_equal = bool(
        len(compare) == len(phase_rows)
        and compare["event_count_active"].eq(compare["event_count_source"]).all()
        and np.allclose(
            compare["deepep_elapsed_ns"],
            compare["duration_ns_sum"],
            rtol=0.0,
            atol=0.0,
        )
        and np.allclose(
            compare["deepep_send_bytes"],
            compare["expected_tx_bytes_total"],
            rtol=0.0,
            atol=0.0,
        )
    )
    expected_full_rank_count = int(topology["rank"].nunique())
    full_run = len(ranks) == expected_full_rank_count
    expected_call_count = 453_120 if expected_full_rank_count == 256 else 293_760
    expected_phase_rows = len(PHASES) * len(iterations) * len(ranks)
    expected_compound_rows = len(iterations) * len(ranks)
    bwd_calls = call_rows[call_rows["phase"].eq("bwd_dispatch")]
    validation = {
        "status": "PASS" if full_run else "PASS_PARTIAL_RANK_AUDIT",
        "schema_version": "ep-active-time-bw-v1",
        "case_id": case_id,
        "full_run": full_run,
        "rank_count": len(ranks),
        "expected_full_rank_count": expected_full_rank_count,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "phase_count": len(PHASES),
        "phases": PHASES,
        "event_count": len(call_rows),
        "expected_event_count_if_full": expected_call_count,
        "phase_row_count": len(phase_rows),
        "expected_phase_row_count": expected_phase_rows,
        "compound_row_count": len(compound_rows),
        "expected_compound_row_count": expected_compound_rows,
        "deep_ep_sums_equal_core_csv": deep_equal,
        "bwd_dispatch_call_count": len(bwd_calls),
        "bwd_dispatch_fixed_80MiB_fraction": float(
            bwd_calls["deepep_send_bytes"].eq(80 * 1024 * 1024).mean()
        ),
        "bwd_dispatch_predecessor_counts": clean_json(
            bwd_calls["previous_behavior"].value_counts().to_dict()
        ),
        "compound_pair_count": int(
            call_rows.loc[call_rows["compound_member"], "compound_pair_id"].nunique()
        ),
        "bwd_dispatch_resolution_ratio_median": float(
            phase_rows.loc[
                phase_rows["phase"].eq("bwd_dispatch"),
                "event_to_sample_resolution_ratio",
            ].median()
        ),
        "compound_resolution_ratio_median": float(
            compound_rows["event_to_sample_resolution_ratio"].median()
        ),
        "bwd_dispatch_shared_sample_fraction_median": float(
            phase_rows.loc[
                phase_rows["phase"].eq("bwd_dispatch"),
                "shared_sample_fraction",
            ].median()
        ),
        "compound_shared_sample_fraction_median": float(
            compound_rows["shared_sample_fraction"].median()
        ),
        "min_link_tx_rate_GBps": args.min_link_tx_rate_gbps,
        "mtlink_long_gap_ms": args.mtlink_long_gap_ms,
        "confidence_counts": clean_json(
            phase_rows["confidence"].value_counts().to_dict()
        ),
        "confidence_by_phase": clean_json(
            phase_rows.groupby(["phase", "confidence"]).size().unstack(fill_value=0).to_dict("index")
        ),
        "compound_confidence_counts": clean_json(
            compound_rows["confidence"].value_counts().to_dict()
        ),
        "maximum_ep_proportional_weight_sum": max(
            float(item["maximum_ep_proportional_weight_sum"]) for item in rank_checks
        ),
        "minimum_active_clipped_le_full_bucket": bool(
            phase_rows["mtlink_active_time_clipped_ns"]
            .le(phase_rows["mtlink_active_time_full_bucket_ns"])
            .all()
        ),
        "outputs": {
            "call_csv": portable(call_output, repository_root),
            "phase_iter_gpu_csv": portable(phase_output, repository_root),
            "compound_iter_gpu_csv": portable(compound_output, repository_root),
            "audit_markdown": portable(audit_output, repository_root),
        },
        "rank_checks": sorted(rank_checks, key=lambda item: int(item["rank"])),
        "interpretation": [
            "DeepEP TX BW is logical send payload divided by DeepEP elapsed time.",
            "Payload/MTLink-active-time is interval-censored cross-domain reference, not physical MTLink bandwidth.",
            "Full-bucket and clipped durations are sampling conventions, not exact true activity bounds.",
            "LOW confidence cells must retain shared-sample and resolution warnings.",
        ],
    }
    checks = [
        len(phase_rows) == expected_phase_rows,
        len(compound_rows) == expected_compound_rows,
        deep_equal,
        validation["bwd_dispatch_fixed_80MiB_fraction"] == 1.0,
        set(validation["bwd_dispatch_predecessor_counts"]) == {"ep_fwd_combine"},
        validation["compound_pair_count"] == len(bwd_calls),
        int(compound_rows["event_count"].sum()) == 2 * len(bwd_calls),
        validation["compound_resolution_ratio_median"]
        > validation["bwd_dispatch_resolution_ratio_median"],
        validation["compound_shared_sample_fraction_median"]
        < validation["bwd_dispatch_shared_sample_fraction_median"],
        validation["maximum_ep_proportional_weight_sum"] <= 1.0 + 1e-12,
        validation["minimum_active_clipped_le_full_bucket"],
    ]
    if full_run:
        checks.append(len(call_rows) == expected_call_count)
    if not all(checks):
        validation["status"] = "FAIL"
    atomic_text(
        validation_output,
        json.dumps(clean_json(validation), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_text(
        audit_output,
        audit_markdown(case_id, validation, phase_rows, compound_rows),
    )
    console_summary = {
        key: validation[key]
        for key in (
            "status",
            "case_id",
            "rank_count",
            "iteration_count",
            "event_count",
            "phase_row_count",
            "compound_row_count",
            "deep_ep_sums_equal_core_csv",
            "bwd_dispatch_resolution_ratio_median",
            "compound_resolution_ratio_median",
            "bwd_dispatch_shared_sample_fraction_median",
            "compound_shared_sample_fraction_median",
            "confidence_counts",
            "compound_confidence_counts",
            "outputs",
        )
    }
    print(json.dumps(clean_json(console_summary), ensure_ascii=False, indent=2))
    return 0 if validation["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

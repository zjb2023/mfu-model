#!/usr/bin/env python3
"""Build DP MTLink/NIC physical rates with domain-native active time.

The existing collective table intentionally divides MTLink bytes by the full
Profiler Trace collective duration.  That is an effective rate which includes
collective synchronization wait.  This script keeps that interpretation and
adds physical active-time views for the two DP collectives:

* MTLink active bytes / union of active MTLink sample intervals;
* mapped-NIC active bytes / its own mapped-NIC active union;
* mapped-NIC bytes inside MTLink-active windows as a cross-domain audit.

The displayed physical rates use each collector's own active-time denominator.
MTLink and NIC bytes are never added because the same logical payload may
traverse both.
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


BEHAVIORS = (
    {
        "behavior": "dp_grad_reduce_scatter",
        "order": 0,
        "label": "DP-reduce-scatter",
        "expected_events": 1,
    },
    {
        "behavior": "dp_param_allgather",
        "order": 1,
        "label": "DP-all-gather",
        "expected_events": 2,
    },
)
BEHAVIOR_NAMES = tuple(str(item["behavior"]) for item in BEHAVIORS)
BEHAVIOR_META = {str(item["behavior"]): item for item in BEHAVIORS}
ITERATIONS = tuple(range(5, 101, 5))
RANKS = tuple(range(256))
MAX_I64 = np.uint64(2**63 - 1)
PCIE_LINK_ID = 2**32 - 1


@dataclass(frozen=True)
class RankTask:
    rank: int
    host: str
    gpu_id: int
    nic_device: str
    target_starts: np.ndarray
    target_ends: np.ndarray
    target_cells: np.ndarray
    mtlink_paths: tuple[Path, ...]
    nic_paths: tuple[Path, ...]
    mtlink_min_rate_GBps: float
    nic_min_rate_Gbps: float
    long_gap_ns: int


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    result_dir = case_root / "results" / "collective_bw_no_pp"
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, default=case_root)
    parser.add_argument(
        "--event-parquet",
        type=Path,
        default=result_dir / "collective_events_no_pp_pre_mtlink.parquet",
    )
    parser.add_argument(
        "--source-cell-csv",
        type=Path,
        default=result_dir
        / "collective_iter_rank_bandwidth_ep_phase_constrained_no_pp.csv",
    )
    parser.add_argument(
        "--rank-topology",
        type=Path,
        default=case_root / "results" / "topology" / "rank_topology.csv",
    )
    parser.add_argument(
        "--rank-nic-mapping",
        type=Path,
        default=case_root
        / "results"
        / "dashboard_reference"
        / "dashboard_rank_nic_mapping.csv",
    )
    parser.add_argument(
        "--mtlink-manifest",
        type=Path,
        default=case_root / "results" / "readiness" / "mtlink_files.csv",
    )
    parser.add_argument(
        "--nic-manifest",
        type=Path,
        default=case_root / "results" / "readiness" / "nic_files.csv",
    )
    parser.add_argument(
        "--hardware-root", type=Path, default=case_root / ".raw_view"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=result_dir / "dp_active_physical_iteration_rank.csv",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=result_dir / "dp_active_physical_validation.json",
    )
    parser.add_argument(
        "--mtlink-min-rate-GBps",
        type=float,
        default=0.05,
        help="Per-link MTLink activity floor in decimal GB/s.",
    )
    parser.add_argument(
        "--nic-min-rate-Gbps",
        type=float,
        default=1.0,
        help="Mapped-NIC activity floor in decimal Gb/s.",
    )
    parser.add_argument("--long-gap-ms", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--ranks",
        help="Optional comma-separated rank subset for development validation.",
    )
    args = parser.parse_args()
    if args.mtlink_min_rate_GBps < 0 or args.nic_min_rate_Gbps < 0:
        parser.error("activity thresholds must be non-negative")
    if args.long_gap_ms <= 0:
        parser.error("--long-gap-ms must be positive")
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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def portable(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        return str(resolved)


def parse_rank_subset(text: str | None) -> list[int]:
    if not text:
        return list(RANKS)
    selected = sorted(
        {int(piece.strip()) for piece in text.split(",") if piece.strip()}
    )
    unknown = sorted(set(selected) - set(RANKS))
    if unknown:
        raise ValueError(f"unknown ranks: {unknown}")
    return selected


def merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([int(start), int(end)])
        else:
            merged[-1][1] = max(merged[-1][1], int(end))
    return [(start, end) for start, end in merged]


def grouped_interval_union(
    groups: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    group_count: int,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    duration = np.zeros(group_count, dtype=np.int64)
    first = np.full(group_count, -1, dtype=np.int64)
    last = np.full(group_count, -1, dtype=np.int64)
    segment_count = np.zeros(group_count, dtype=np.int64)
    valid = mask & (groups >= 0) & (starts < ends)
    groups = groups[valid].astype(np.int64, copy=False)
    starts = starts[valid].astype(np.int64, copy=False)
    ends = ends[valid].astype(np.int64, copy=False)
    if len(groups) == 0:
        return duration, first, last, segment_count
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
        segment_count[current_group] += 1
        last[current_group] = current_end
        current_group = group_i
        current_start = start_i
        current_end = end_i
        if first[current_group] < 0:
            first[current_group] = current_start
    duration[current_group] += current_end - current_start
    segment_count[current_group] += 1
    last[current_group] = current_end
    return duration, first, last, segment_count


def grouped_interval_lists(
    groups: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    group_count: int,
    mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """Return one merged interval list per cell for the selected pieces."""
    result: list[list[tuple[int, int]]] = [[] for _ in range(group_count)]
    valid = mask & (groups >= 0) & (starts < ends)
    for group, start, end in zip(
        groups[valid].astype(np.int64, copy=False),
        starts[valid].astype(np.int64, copy=False),
        ends[valid].astype(np.int64, copy=False),
    ):
        result[int(group)].append((int(start), int(end)))
    return [merge_intervals(intervals) for intervals in result]


def resolve_mtlink_paths(
    manifest_path: Path, hardware_root: Path
) -> dict[tuple[str, int], tuple[Path, ...]]:
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("error"):
                continue
            grouped[(str(row["host"]), int(row["gpu_id"]))].append(
                hardware_root / str(row["path"])
            )
    return {key: tuple(sorted(paths)) for key, paths in grouped.items()}


def resolve_nic_paths(
    manifest_path: Path, hardware_root: Path
) -> dict[tuple[str, str], tuple[Path, ...]]:
    grouped: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("error"):
                continue
            grouped[(str(row["host"]), str(row["device"]))].append(
                (int(row["sequence"]), hardware_root / str(row["path"]))
            )
    return {
        key: tuple(path for _sequence, path in sorted(items))
        for key, items in grouped.items()
    }


def build_cells(
    event_path: Path,
    source_cell_path: Path,
    topology_path: Path,
    mapping_path: Path,
    ranks: list[int],
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    events = pd.read_parquet(
        event_path,
        columns=[
            "behavior",
            "iteration",
            "rank",
            "host",
            "gpu_id",
            "start_ns",
            "end_ns",
            "duration_ns",
        ],
    )
    events = events[
        events["behavior"].isin(BEHAVIOR_NAMES)
        & events["rank"].isin(ranks)
    ].copy()
    source = pd.read_csv(source_cell_path)
    source = source[
        source["behavior"].isin(BEHAVIOR_NAMES)
        & source["rank"].isin(ranks)
    ].copy()
    topology = pd.read_csv(topology_path)
    mapping = pd.read_csv(mapping_path)
    identity = topology[
        ["rank", "host", "local_rank", "gpu_id", "pp_stage"]
    ].merge(
        mapping[
            ["rank", "nic_device", "mapping_status", "relative_path"]
        ],
        on="rank",
        validate="one_to_one",
    )
    identity = identity[identity["rank"].isin(ranks)].set_index("rank")
    source_index = source.set_index(["behavior", "iteration", "rank"])
    event_groups = {
        (str(behavior), int(iteration), int(rank)): group
        for (behavior, iteration, rank), group in events.groupby(
            ["behavior", "iteration", "rank"]
        )
    }

    rows: list[dict[str, Any]] = []
    targets_by_rank: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    target_interval_count = 0
    cross_cell_overlap_ns = 0
    for rank in ranks:
        target_starts: list[int] = []
        target_ends: list[int] = []
        target_cells: list[int] = []
        previous_targets: list[tuple[int, int]] = []
        meta = identity.loc[rank]
        local_cell = 0
        for item in BEHAVIORS:
            behavior = str(item["behavior"])
            expected_events = int(item["expected_events"])
            for iteration in ITERATIONS:
                group = event_groups.get((behavior, iteration, rank))
                if group is None or group.empty:
                    intervals: list[tuple[int, int]] = []
                    event_count = 0
                    duration_sum = 0
                    data_status = "missing_source_event"
                else:
                    intervals = merge_intervals(
                        zip(
                            group["start_ns"].astype(int),
                            group["end_ns"].astype(int),
                        )
                    )
                    event_count = int(len(group))
                    duration_sum = int(group["duration_ns"].sum())
                    data_status = (
                        "observed"
                        if event_count == expected_events
                        else "partial_source_event"
                    )
                union_duration = sum(end - start for start, end in intervals)
                source_key = (behavior, iteration, rank)
                if source_key not in source_index.index:
                    raise ValueError(f"source cell missing: {source_key}")
                source_row = source_index.loc[source_key]
                rows.append(
                    {
                        "local_cell": local_cell,
                        "behavior_order": int(item["order"]),
                        "behavior_family": "DP",
                        "behavior": behavior,
                        "behavior_label": str(item["label"]),
                        "iteration": iteration,
                        "rank": rank,
                        "node_index": rank // 8,
                        "host": str(meta["host"]),
                        "local_rank": int(meta["local_rank"]),
                        "gpu_id": int(meta["gpu_id"]),
                        "pp_stage": int(meta["pp_stage"]),
                        "nic_device": str(meta["nic_device"]),
                        "nic_entity_id": (
                            f"{meta['host']} / r{rank} / GPU{int(meta['gpu_id'])} / "
                            f"{meta['nic_device']}"
                        ),
                        "nic_mapping_status": str(meta["mapping_status"]),
                        "nic_mapping_source": str(meta["relative_path"]),
                        "event_count": event_count,
                        "expected_event_count": expected_events,
                        "data_status": data_status,
                        "trace_union_segment_count": len(intervals),
                        "trace_window_start_ns": (
                            min(start for start, _end in intervals)
                            if intervals
                            else pd.NA
                        ),
                        "trace_window_end_ns": (
                            max(end for _start, end in intervals)
                            if intervals
                            else pd.NA
                        ),
                        "trace_event_duration_sum_ns": (
                            duration_sum if intervals else pd.NA
                        ),
                        "trace_event_union_duration_ns": (
                            union_duration if intervals else pd.NA
                        ),
                        "expected_tx_bytes_total": float(
                            source_row["expected_tx_bytes_total"]
                        ),
                        "expected_rx_bytes_total": float(
                            source_row["expected_rx_bytes_total"]
                        ),
                        "canonical_mtlink_attributed_tx_bytes_total": float(
                            source_row["mtlink_attributed_tx_bytes_total"]
                        ),
                        "canonical_mtlink_attributed_rx_bytes_total": float(
                            source_row["mtlink_attributed_rx_bytes_total"]
                        ),
                        "canonical_mtlink_normalized_overlap_count": int(
                            source_row["mtlink_normalized_overlap_count"]
                        ),
                    }
                )
                for start, end in intervals:
                    target_starts.append(start)
                    target_ends.append(end)
                    target_cells.append(local_cell)
                    previous_targets.append((start, end))
                    target_interval_count += 1
                local_cell += 1
        order = np.argsort(np.asarray(target_starts, dtype=np.int64))
        starts = np.asarray(target_starts, dtype=np.int64)[order]
        ends = np.asarray(target_ends, dtype=np.int64)[order]
        cells = np.asarray(target_cells, dtype=np.int64)[order]
        if len(starts) > 1:
            cross_cell_overlap_ns += int(
                np.maximum(0, ends[:-1] - starts[1:]).sum()
            )
        targets_by_rank[rank] = (starts, ends, cells)
    return (
        pd.DataFrame(rows),
        targets_by_rank,
        {
            "source_event_count": int(len(events)),
            "target_interval_count": target_interval_count,
            "cross_cell_overlap_ns": cross_cell_overlap_ns,
        },
    )


def read_mtlink(paths: tuple[Path, ...]) -> pd.DataFrame:
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
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    frame = frame[
        frame["link_id"].between(0, 13)
        & frame["link_id"].ne(PCIE_LINK_ID)
    ].copy()
    duration = (
        frame["mt_timestamp_end_ns"].to_numpy(dtype=np.int64)
        - frame["mt_timestamp_begin_ns"].to_numpy(dtype=np.int64)
    )
    tx = frame["tx_delta_bytes"].to_numpy(dtype=np.uint64)
    rx = frame["rx_delta_bytes"].to_numpy(dtype=np.uint64)
    valid = (duration > 0) & (tx <= MAX_I64) & (rx <= MAX_I64)
    frame = frame.loc[valid].copy()
    duration = duration[valid]
    frame["sample_duration_ns"] = duration
    frame["sample_end_ns"] = frame["host_realtime_ns"].to_numpy(
        dtype=np.int64
    )
    frame["sample_start_ns"] = (
        frame["sample_end_ns"].to_numpy(dtype=np.int64) - duration
    )
    frame["tx_bytes"] = frame["tx_delta_bytes"].astype(np.float64)
    frame["rx_bytes"] = frame["rx_delta_bytes"].astype(np.float64)
    return frame.sort_values("sample_end_ns", kind="stable").reset_index(
        drop=True
    )


def read_nic(paths: tuple[Path, ...]) -> pd.DataFrame:
    usecols = ["timestamp_ns", "sample_interval_us", "xmit_bytes", "recv_bytes"]
    frames = [
        pd.read_csv(
            path,
            usecols=usecols,
            dtype={
                "timestamp_ns": "int64",
                "sample_interval_us": "float64",
                "xmit_bytes": "int64",
                "recv_bytes": "int64",
            },
        )
        for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    duration = np.rint(
        frame["sample_interval_us"].to_numpy(dtype=np.float64) * 1000.0
    ).astype(np.int64)
    tx = frame["xmit_bytes"].to_numpy(dtype=np.int64)
    rx = frame["recv_bytes"].to_numpy(dtype=np.int64)
    valid = (duration > 0) & (tx >= 0) & (rx >= 0)
    frame = frame.loc[valid].copy()
    duration = duration[valid]
    frame["sample_duration_ns"] = duration
    frame["sample_end_ns"] = frame["timestamp_ns"].to_numpy(dtype=np.int64)
    frame["sample_start_ns"] = (
        frame["sample_end_ns"].to_numpy(dtype=np.int64) - duration
    )
    frame["tx_bytes"] = frame["xmit_bytes"].astype(np.float64)
    frame["rx_bytes"] = frame["recv_bytes"].astype(np.float64)
    return frame.sort_values("sample_end_ns", kind="stable").reset_index(
        drop=True
    )


def build_pairs(
    samples: pd.DataFrame,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = samples["sample_start_ns"].to_numpy(dtype=np.int64)
    ends = samples["sample_end_ns"].to_numpy(dtype=np.int64)
    first = np.searchsorted(target_ends, starts, side="right")
    stop = np.searchsorted(target_starts, ends, side="left")
    candidate_count = stop - first
    single_rows = np.flatnonzero(candidate_count == 1)
    pair_samples: list[np.ndarray] = []
    pair_targets: list[np.ndarray] = []
    pair_starts: list[np.ndarray] = []
    pair_ends: list[np.ndarray] = []
    if len(single_rows):
        target_index = first[single_rows]
        clipped_start = np.maximum(starts[single_rows], target_starts[target_index])
        clipped_end = np.minimum(ends[single_rows], target_ends[target_index])
        keep = clipped_start < clipped_end
        pair_samples.append(single_rows[keep])
        pair_targets.append(target_index[keep])
        pair_starts.append(clipped_start[keep])
        pair_ends.append(clipped_end[keep])
    for sample_index in np.flatnonzero(candidate_count > 1):
        indices = np.arange(first[sample_index], stop[sample_index], dtype=np.int64)
        clipped_start = np.maximum(starts[sample_index], target_starts[indices])
        clipped_end = np.minimum(ends[sample_index], target_ends[indices])
        keep = clipped_start < clipped_end
        if np.any(keep):
            pair_samples.append(
                np.full(np.count_nonzero(keep), sample_index, dtype=np.int64)
            )
            pair_targets.append(indices[keep])
            pair_starts.append(clipped_start[keep])
            pair_ends.append(clipped_end[keep])
    if not pair_samples:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty
    return (
        np.concatenate(pair_samples),
        np.concatenate(pair_targets),
        np.concatenate(pair_starts),
        np.concatenate(pair_ends),
    )


def build_mtlink_active_windows(
    samples: pd.DataFrame,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
    target_cells: np.ndarray,
    cell_count: int,
    rate_floor_GBps: float,
) -> dict[str, list[list[tuple[int, int]]]]:
    """Build direction-specific MTLink-active wall-time windows per DP cell."""
    sample_index, target_index, clipped_start, clipped_end = build_pairs(
        samples, target_starts, target_ends
    )
    if len(sample_index) == 0:
        raise RuntimeError("no MTLink sample/DP intersections for active windows")
    cells = target_cells[target_index]
    duration = samples["sample_duration_ns"].to_numpy(dtype=np.int64)
    tx = samples["tx_bytes"].to_numpy(dtype=np.float64)
    rx = samples["rx_bytes"].to_numpy(dtype=np.float64)
    tx_active = (
        (tx[sample_index] > 0)
        & ((tx[sample_index] / duration[sample_index]) >= rate_floor_GBps)
    )
    rx_active = (
        (rx[sample_index] > 0)
        & ((rx[sample_index] / duration[sample_index]) >= rate_floor_GBps)
    )
    return {
        "tx": grouped_interval_lists(
            cells, clipped_start, clipped_end, cell_count, tx_active
        ),
        "rx": grouped_interval_lists(
            cells, clipped_start, clipped_end, cell_count, rx_active
        ),
        "bidir": grouped_interval_lists(
            cells, clipped_start, clipped_end, cell_count, tx_active | rx_active
        ),
    }


def flatten_cell_windows(
    windows: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pieces = sorted(
        (start, end, cell)
        for cell, intervals in enumerate(windows)
        for start, end in intervals
    )
    if not pieces:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    starts = np.asarray([piece[0] for piece in pieces], dtype=np.int64)
    ends = np.asarray([piece[1] for piece in pieces], dtype=np.int64)
    cells = np.asarray([piece[2] for piece in pieces], dtype=np.int64)
    if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
        raise RuntimeError("MTLink-active windows overlap across DP cells")
    return starts, ends, cells


def attribute_nic_to_mtlink_active_windows(
    nic_samples: pd.DataFrame,
    active_windows: dict[str, list[list[tuple[int, int]]]],
    cell_count: int,
    long_gap_ns: int,
) -> pd.DataFrame:
    """Integrate mapped-NIC deltas over the same MTLink-active windows."""
    duration = nic_samples["sample_duration_ns"].to_numpy(dtype=np.int64)
    tx = nic_samples["tx_bytes"].to_numpy(dtype=np.float64)
    rx = nic_samples["rx_bytes"].to_numpy(dtype=np.float64)
    output: dict[str, np.ndarray] = {}
    for direction in ("tx", "rx", "bidir"):
        starts, ends, target_cells = flatten_cell_windows(
            active_windows[direction]
        )
        target_duration = np.bincount(
            target_cells,
            weights=(ends - starts),
            minlength=cell_count,
        ).astype(np.int64)
        target_segments = np.bincount(
            target_cells, minlength=cell_count
        ).astype(np.int64)
        sample_index, target_index, clipped_start, clipped_end = build_pairs(
            nic_samples, starts, ends
        )
        if len(sample_index) == 0:
            raise RuntimeError(
                f"no mapped-NIC intersections with MTLink-active {direction} windows"
            )
        cells = target_cells[target_index]
        overlap = (clipped_end - clipped_start).astype(np.float64)
        ratio = overlap / duration[sample_index]
        if direction == "tx":
            sample_bytes = tx[sample_index]
        elif direction == "rx":
            sample_bytes = rx[sample_index]
        else:
            sample_bytes = tx[sample_index] + rx[sample_index]
        attributed = sample_bytes * ratio
        long_gap = duration[sample_index] > long_gap_ns
        output[f"nic_mtlink_active_window_{direction}_bytes"] = np.bincount(
            cells, weights=attributed, minlength=cell_count
        )
        output[
            f"nic_mtlink_active_window_{direction}_uncertain_long_gap_bytes"
        ] = np.bincount(
            cells[long_gap],
            weights=attributed[long_gap],
            minlength=cell_count,
        )
        output[f"nic_mtlink_active_window_{direction}_duration_ns"] = (
            target_duration
        )
        output[f"nic_mtlink_active_window_{direction}_segment_count"] = (
            target_segments
        )
        unique_key = cells.astype(np.int64) * len(nic_samples) + sample_index
        _, unique_position = np.unique(unique_key, return_index=True)
        output[f"nic_mtlink_active_window_{direction}_touched_sample_count"] = (
            np.bincount(cells[unique_position], minlength=cell_count)
        )
        covered, _first, _last, _segments = grouped_interval_union(
            cells,
            clipped_start,
            clipped_end,
            cell_count,
            np.ones(len(cells), dtype=bool),
        )
        output[f"nic_mtlink_active_window_{direction}_coverage_pct"] = (
            covered / np.where(target_duration > 0, target_duration, np.nan) * 100.0
        )
    frame = pd.DataFrame(output)
    frame.insert(0, "local_cell", np.arange(cell_count, dtype=np.int64))
    return frame


def attribute_domain(
    samples: pd.DataFrame,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
    target_cells: np.ndarray,
    cell_count: int,
    tx_rate_floor: float,
    rx_rate_floor: float,
    rate_factor: float,
    long_gap_ns: int,
    prefix: str,
) -> pd.DataFrame:
    sample_index, target_index, clipped_start, clipped_end = build_pairs(
        samples, target_starts, target_ends
    )
    if len(sample_index) == 0:
        raise RuntimeError(f"no {prefix} sample/DP intersections")
    cells = target_cells[target_index]
    overlap = (clipped_end - clipped_start).astype(np.float64)
    duration = samples["sample_duration_ns"].to_numpy(dtype=np.int64)
    tx = samples["tx_bytes"].to_numpy(dtype=np.float64)
    rx = samples["rx_bytes"].to_numpy(dtype=np.float64)
    ratio = overlap / duration[sample_index]
    tx_rate = tx / duration * rate_factor
    rx_rate = rx / duration * rate_factor
    tx_active_sample = (tx > 0) & (tx_rate >= tx_rate_floor)
    rx_active_sample = (rx > 0) & (rx_rate >= rx_rate_floor)
    tx_active = tx_active_sample[sample_index]
    rx_active = rx_active_sample[sample_index]
    bidir_active = tx_active | rx_active
    long_gap = duration[sample_index] > long_gap_ns
    precise = ~long_gap

    output: dict[str, np.ndarray] = {}
    output[f"{prefix}_window_attributed_tx_bytes"] = np.bincount(
        cells, weights=tx[sample_index] * ratio, minlength=cell_count
    )
    output[f"{prefix}_window_attributed_rx_bytes"] = np.bincount(
        cells, weights=rx[sample_index] * ratio, minlength=cell_count
    )
    output[f"{prefix}_active_tx_bytes"] = np.bincount(
        cells[tx_active],
        weights=tx[sample_index[tx_active]] * ratio[tx_active],
        minlength=cell_count,
    )
    output[f"{prefix}_active_rx_bytes"] = np.bincount(
        cells[rx_active],
        weights=rx[sample_index[rx_active]] * ratio[rx_active],
        minlength=cell_count,
    )
    output[f"{prefix}_active_bidir_bytes"] = np.bincount(
        cells[bidir_active],
        weights=(tx[sample_index[bidir_active]] + rx[sample_index[bidir_active]])
        * ratio[bidir_active],
        minlength=cell_count,
    )
    output[f"{prefix}_precise_active_tx_bytes"] = np.bincount(
        cells[tx_active & precise],
        weights=tx[sample_index[tx_active & precise]]
        * ratio[tx_active & precise],
        minlength=cell_count,
    )
    output[f"{prefix}_precise_active_rx_bytes"] = np.bincount(
        cells[rx_active & precise],
        weights=rx[sample_index[rx_active & precise]]
        * ratio[rx_active & precise],
        minlength=cell_count,
    )
    output[f"{prefix}_precise_active_bidir_bytes"] = np.bincount(
        cells[bidir_active & precise],
        weights=(
            tx[sample_index[bidir_active & precise]]
            + rx[sample_index[bidir_active & precise]]
        )
        * ratio[bidir_active & precise],
        minlength=cell_count,
    )
    output[f"{prefix}_uncertain_long_gap_tx_bytes"] = np.bincount(
        cells[long_gap],
        weights=tx[sample_index[long_gap]] * ratio[long_gap],
        minlength=cell_count,
    )
    output[f"{prefix}_uncertain_long_gap_rx_bytes"] = np.bincount(
        cells[long_gap],
        weights=rx[sample_index[long_gap]] * ratio[long_gap],
        minlength=cell_count,
    )
    for direction, mask in (
        ("tx", tx_active),
        ("rx", rx_active),
        ("bidir", bidir_active),
    ):
        active_duration, active_start, active_end, segment_count = (
            grouped_interval_union(
                cells,
                clipped_start,
                clipped_end,
                cell_count,
                mask,
            )
        )
        precise_duration, _start, _end, precise_segments = grouped_interval_union(
            cells,
            clipped_start,
            clipped_end,
            cell_count,
            mask & precise,
        )
        output[f"{prefix}_active_{direction}_duration_ns"] = active_duration
        output[f"{prefix}_active_{direction}_start_ns"] = active_start
        output[f"{prefix}_active_{direction}_end_ns"] = active_end
        output[f"{prefix}_active_{direction}_segment_count"] = segment_count
        output[f"{prefix}_precise_active_{direction}_duration_ns"] = (
            precise_duration
        )
        output[f"{prefix}_precise_active_{direction}_segment_count"] = (
            precise_segments
        )
    unique_key = cells.astype(np.int64) * len(samples) + sample_index
    _, unique_position = np.unique(unique_key, return_index=True)
    unique_cells = cells[unique_position]
    output[f"{prefix}_touched_sample_count"] = np.bincount(
        unique_cells, minlength=cell_count
    )
    output[f"{prefix}_long_gap_sample_count"] = np.bincount(
        unique_cells,
        weights=long_gap[unique_position],
        minlength=cell_count,
    )
    frame = pd.DataFrame(output)
    frame.insert(0, "local_cell", np.arange(cell_count, dtype=np.int64))
    return frame


def add_rates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    trace_duration = pd.to_numeric(
        result["trace_event_union_duration_ns"], errors="coerce"
    )
    for prefix, factor, unit in (
        ("mtlink", 1.0, "GBps"),
        ("nic", 8.0, "Gbps"),
    ):
        result[f"{prefix}_window_attributed_bidir_bytes"] = (
            result[f"{prefix}_window_attributed_tx_bytes"]
            + result[f"{prefix}_window_attributed_rx_bytes"]
        )
        result[f"{prefix}_trace_window_tx_bandwidth_{unit}"] = (
            result[f"{prefix}_window_attributed_tx_bytes"]
            / trace_duration
            * factor
        )
        result[f"{prefix}_trace_window_rx_bandwidth_{unit}"] = (
            result[f"{prefix}_window_attributed_rx_bytes"]
            / trace_duration
            * factor
        )
        result[f"{prefix}_trace_window_bidir_bandwidth_{unit}"] = (
            result[f"{prefix}_window_attributed_bidir_bytes"]
            / trace_duration
            * factor
        )
        for direction in ("tx", "rx", "bidir"):
            active_duration = result[
                f"{prefix}_active_{direction}_duration_ns"
            ].replace(0, np.nan)
            precise_duration = result[
                f"{prefix}_precise_active_{direction}_duration_ns"
            ].replace(0, np.nan)
            result[f"{prefix}_active_{direction}_bandwidth_{unit}"] = (
                result[f"{prefix}_active_{direction}_bytes"]
                / active_duration
                * factor
            )
            result[
                f"{prefix}_precise_active_{direction}_bandwidth_{unit}"
            ] = (
                result[f"{prefix}_precise_active_{direction}_bytes"]
                / precise_duration
                * factor
            )
            result[f"{prefix}_active_{direction}_fraction_of_trace_pct"] = (
                result[f"{prefix}_active_{direction}_duration_ns"]
                / trace_duration
                * 100.0
            )
        uncertain = (
            result[f"{prefix}_uncertain_long_gap_tx_bytes"]
            + result[f"{prefix}_uncertain_long_gap_rx_bytes"]
        )
        result[f"{prefix}_long_gap_bidir_pct"] = (
            uncertain
            / result[f"{prefix}_window_attributed_bidir_bytes"].replace(
                0, np.nan
            )
            * 100.0
        )
    for direction in ("tx", "rx", "bidir"):
        name = f"nic_mtlink_active_window_{direction}"
        denominator = result[f"{name}_duration_ns"].replace(0, np.nan)
        result[f"{name}_bandwidth_Gbps"] = (
            result[f"{name}_bytes"] / denominator * 8.0
        )
        result[f"{name}_long_gap_pct"] = (
            result[f"{name}_uncertain_long_gap_bytes"]
            / result[f"{name}_bytes"].replace(0, np.nan)
            * 100.0
        )
    return result


def process_rank(task: RankTask) -> pd.DataFrame:
    cell_count = len(BEHAVIORS) * len(ITERATIONS)
    mtlink = read_mtlink(task.mtlink_paths)
    nic = read_nic(task.nic_paths)
    mtlink_metrics = attribute_domain(
        mtlink,
        task.target_starts,
        task.target_ends,
        task.target_cells,
        cell_count,
        task.mtlink_min_rate_GBps,
        task.mtlink_min_rate_GBps,
        1.0,
        task.long_gap_ns,
        "mtlink",
    )
    nic_metrics = attribute_domain(
        nic,
        task.target_starts,
        task.target_ends,
        task.target_cells,
        cell_count,
        task.nic_min_rate_Gbps,
        task.nic_min_rate_Gbps,
        8.0,
        task.long_gap_ns,
        "nic",
    )
    mtlink_active_windows = build_mtlink_active_windows(
        mtlink,
        task.target_starts,
        task.target_ends,
        task.target_cells,
        cell_count,
        task.mtlink_min_rate_GBps,
    )
    nic_mtlink_window_metrics = attribute_nic_to_mtlink_active_windows(
        nic,
        mtlink_active_windows,
        cell_count,
        task.long_gap_ns,
    )
    output = mtlink_metrics.merge(
        nic_metrics, on="local_cell", validate="one_to_one"
    ).merge(
        nic_mtlink_window_metrics,
        on="local_cell",
        validate="one_to_one",
    )
    output.insert(0, "rank", task.rank)
    return output


def main() -> int:
    args = parse_args()
    case_root = args.case_root.resolve()
    repository_root = case_root.parent
    ranks = parse_rank_subset(args.ranks)
    cells, targets_by_rank, build_validation = build_cells(
        args.event_parquet.resolve(),
        args.source_cell_csv.resolve(),
        args.rank_topology.resolve(),
        args.rank_nic_mapping.resolve(),
        ranks,
    )
    mtlink_paths = resolve_mtlink_paths(
        args.mtlink_manifest.resolve(), args.hardware_root.resolve()
    )
    nic_paths = resolve_nic_paths(
        args.nic_manifest.resolve(), args.hardware_root.resolve()
    )
    identity = cells.drop_duplicates("rank").set_index("rank")
    tasks = []
    for rank in ranks:
        item = identity.loc[rank]
        starts, ends, target_cells = targets_by_rank[rank]
        task = RankTask(
            rank=rank,
            host=str(item["host"]),
            gpu_id=int(item["gpu_id"]),
            nic_device=str(item["nic_device"]),
            target_starts=starts,
            target_ends=ends,
            target_cells=target_cells,
            mtlink_paths=mtlink_paths[(str(item["host"]), int(item["gpu_id"]))],
            nic_paths=nic_paths[(str(item["host"]), str(item["nic_device"]))],
            mtlink_min_rate_GBps=args.mtlink_min_rate_GBps,
            nic_min_rate_Gbps=args.nic_min_rate_Gbps,
            long_gap_ns=round(args.long_gap_ms * 1_000_000),
        )
        if any(not path.is_file() for path in (*task.mtlink_paths, *task.nic_paths)):
            raise FileNotFoundError(task)
        tasks.append(task)

    results: list[pd.DataFrame] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(process_rank, task): task.rank for task in tasks}
        for future in as_completed(future_map):
            rank = future_map[future]
            results.append(future.result())
            completed += 1
            if completed % 8 == 0 or completed == len(tasks):
                print(
                    f"processed {completed}/{len(tasks)} ranks; latest r{rank}",
                    flush=True,
                )
    metrics = pd.concat(results, ignore_index=True)
    output = cells.merge(
        metrics, on=["rank", "local_cell"], validate="one_to_one"
    )
    output = add_rates(output).drop(columns="local_cell")
    output = output.sort_values(["behavior_order", "iteration", "rank"])

    expected_rows = len(ranks) * len(BEHAVIORS) * len(ITERATIONS)
    observed = output[output["data_status"].ne("missing_source_event")]
    canonical_differences: dict[str, dict[str, Any]] = {}
    canonical_checks: list[bool] = []
    for direction in ("tx", "rx"):
        rebuilt = output[
            f"mtlink_window_attributed_{direction}_bytes"
        ].to_numpy(dtype=np.float64)
        canonical = output[
            f"canonical_mtlink_attributed_{direction}_bytes_total"
        ].to_numpy(dtype=np.float64)
        absolute = np.abs(rebuilt - canonical)
        relative = absolute / np.maximum(np.abs(canonical), 1.0)
        mismatch = absolute > 1.0
        normalized_only = output.loc[
            mismatch, "canonical_mtlink_normalized_overlap_count"
        ].gt(0).all()
        canonical_differences[direction] = {
            "cell_count_abs_difference_gt_1_byte": int(mismatch.sum()),
            "maximum_absolute_difference_bytes": float(absolute.max()),
            "maximum_relative_difference": float(relative.max()),
            "all_gt_1_byte_differences_have_canonical_normalization": bool(
                normalized_only
            ),
        }
        canonical_checks.extend(
            [bool(normalized_only), bool(relative.max() <= 1e-3)]
        )
    behavior_summary: dict[str, dict[str, Any]] = {}
    for behavior, group in output.groupby("behavior"):
        behavior_summary[str(behavior)] = {
            "cell_count": int(len(group)),
            "mtlink_active_tx_bandwidth_GBps_p50": float(
                group["mtlink_active_tx_bandwidth_GBps"].median()
            ),
            "nic_mtlink_active_window_tx_bandwidth_Gbps_p50": float(
                group[
                    "nic_mtlink_active_window_tx_bandwidth_Gbps"
                ].median()
            ),
            "nic_own_active_tx_bandwidth_Gbps_p50": float(
                group["nic_active_tx_bandwidth_Gbps"].median()
            ),
            "nic_mtlink_active_window_tx_bytes_vs_expected_tx_pct": float(
                group["nic_mtlink_active_window_tx_bytes"].sum()
                / group["expected_tx_bytes_total"].sum()
                * 100.0
            ),
        }
    validation: dict[str, Any] = {
        "status": "PASS",
        "schema_version": "256gpu-dp-active-physical-v3",
        "rank_subset": ranks,
        "full_rank_run": ranks == list(RANKS),
        "behaviors": list(BEHAVIOR_NAMES),
        "row_count": int(len(output)),
        "expected_row_count": expected_rows,
        "rank_count": int(output["rank"].nunique()),
        "host_count": int(output["host"].nunique()),
        "iteration_count": int(output["iteration"].nunique()),
        "source_event_count": build_validation["source_event_count"],
        "target_interval_count": build_validation["target_interval_count"],
        "cross_cell_overlap_ns": build_validation["cross_cell_overlap_ns"],
        "data_status_counts": {
            str(key): int(value)
            for key, value in output["data_status"].value_counts().items()
        },
        "mtlink_activity_threshold_GBps_per_link": args.mtlink_min_rate_GBps,
        "nic_activity_threshold_Gbps": args.nic_min_rate_Gbps,
        "long_gap_ms": args.long_gap_ms,
        "mtlink_active_formula": (
            "active-threshold attributed bytes / union of clipped active "
            "MTLink sample intervals across links 0..13"
        ),
        "nic_active_formula": (
            "active-threshold mapped-NIC attributed bytes / union of clipped "
            "mapped-NIC active sample intervals"
        ),
        "nic_mtlink_active_window_formula": (
            "mapped-NIC delta bytes attributed inside direction-matched MTLink "
            "active windows / the same MTLink active-window union duration"
        ),
        "trace_window_formula": (
            "all valid attributed bytes / DP Trace event union duration"
        ),
        "canonical_trace_window_rebuild_comparison": canonical_differences,
        "behavior_summary": behavior_summary,
        "minimum_mtlink_active_bidir_fraction_of_trace_pct": float(
            observed["mtlink_active_bidir_fraction_of_trace_pct"].min()
        ),
        "minimum_nic_active_bidir_fraction_of_trace_pct": float(
            observed["nic_active_bidir_fraction_of_trace_pct"].min()
        ),
        "zero_mtlink_active_tx_cell_count": int(
            observed["mtlink_active_tx_duration_ns"].eq(0).sum()
        ),
        "zero_nic_active_tx_cell_count": int(
            observed["nic_active_tx_duration_ns"].eq(0).sum()
        ),
        "zero_nic_mtlink_active_window_tx_byte_cell_count": int(
            observed["nic_mtlink_active_window_tx_bytes"].eq(0).sum()
        ),
        "minimum_nic_mtlink_active_window_tx_coverage_pct": float(
            observed[
                "nic_mtlink_active_window_tx_coverage_pct"
            ].min()
        ),
        "nic_mapping_status_counts": {
            str(key): int(value)
            for key, value in output["nic_mapping_status"].value_counts().items()
        },
        "output_csv": portable(args.output_csv, repository_root),
        "notes": [
            "Trace expected/effective metrics are not changed by this output.",
            "Displayed MTLink rates use MTLink-active interval unions from MTLink CSV.",
            "Displayed mapped-NIC rates use NIC-active interval unions from RDMA NIC CSV.",
            "MTLink-gated common-window NIC metrics remain only as a cross-domain audit.",
            "Active sample intervals are interval-censored upper bounds on active time.",
            "MTLink and NIC bytes must not be added.",
            "The rank/NIC mapping is the documented 224-case carry-over mapping.",
        ],
    }
    checks = [
        len(output) == expected_rows,
        validation["rank_count"] == len(ranks),
        validation["iteration_count"] == len(ITERATIONS),
        validation["cross_cell_overlap_ns"] == 0,
        output.duplicated(["behavior", "iteration", "rank"]).sum() == 0,
        observed["mtlink_active_bidir_duration_ns"]
        .le(observed["trace_event_union_duration_ns"])
        .all(),
        observed["nic_active_bidir_duration_ns"]
        .le(observed["trace_event_union_duration_ns"])
        .all(),
        observed["nic_mtlink_active_window_tx_duration_ns"]
        .eq(observed["mtlink_active_tx_duration_ns"])
        .all(),
        observed["nic_mtlink_active_window_rx_duration_ns"]
        .eq(observed["mtlink_active_rx_duration_ns"])
        .all(),
        observed["nic_mtlink_active_window_bidir_duration_ns"]
        .eq(observed["mtlink_active_bidir_duration_ns"])
        .all(),
        observed["nic_mtlink_active_window_tx_coverage_pct"].ge(99.0).all(),
        *canonical_checks,
    ]
    if not all(checks):
        validation["status"] = "FAIL"
    atomic_csv(output, args.output_csv)
    atomic_json(validation, args.validation_output)
    print(json.dumps(clean_json(validation), ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

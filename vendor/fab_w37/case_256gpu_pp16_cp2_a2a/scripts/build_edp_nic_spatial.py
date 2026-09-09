#!/usr/bin/env python3
"""Attribute host NIC counters to merged expert-DP communication windows.

NIC telemetry is collected per host/device and there is no validated
rank-to-NIC-rail map.  This script therefore deliberately stops at
behavior x iteration x host.  The eight ranks' expert-DP event intervals
are merged before attribution, so concurrent rank events do not duplicate
physical NIC bytes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ITERATIONS = tuple(range(5, 101, 5))
EDP_BEHAVIORS = (
    {
        "behavior": "expert_dp_grad_reduce_scatter",
        "behavior_order": 0,
        "behavior_family": "Expert-DP",
        "behavior_label": "Expert-DP-reduce-scatter",
        "behavior_description": "Expert gradient reduce-scatter",
        "expected_events_per_rank": 1,
    },
    {
        "behavior": "expert_dp_param_allgather",
        "behavior_order": 1,
        "behavior_family": "Expert-DP",
        "behavior_label": "Expert-DP-all-gather",
        "behavior_description": "Expert parameter all-gather",
        "expected_events_per_rank": 2,
    },
)
BEHAVIOR_BY_NAME = {str(item["behavior"]): item for item in EDP_BEHAVIORS}


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    case_root = Path(__file__).resolve().parents[1]
    collective_dir = case_root / "results" / "collective_bw_no_pp"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events-parquet",
        type=Path,
        default=collective_dir / "collective_events_no_pp_pre_mtlink.parquet",
    )
    parser.add_argument(
        "--hardware-root",
        type=Path,
        default=case_root / ".raw_view",
    )
    parser.add_argument(
        "--nic-manifest",
        type=Path,
        default=case_root / "results" / "readiness" / "nic_files.csv",
    )
    parser.add_argument(
        "--event-windows",
        type=Path,
        default=case_root / "results" / "readiness" / "event_windows.csv",
    )
    parser.add_argument(
        "--rank-topology",
        type=Path,
        default=case_root / "results" / "topology" / "rank_topology.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=collective_dir / "edp_nic_spatial_heatmap_data.csv",
    )
    parser.add_argument(
        "--device-audit-output",
        type=Path,
        default=collective_dir / "edp_nic_iteration_host_device.csv",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=collective_dir / "edp_nic_spatial_validation.json",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--nic-long-gap-ms", type=float, default=10.0)
    return parser.parse_args()


def portable_path(path: Path) -> str:
    repository_root = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root))
    except ValueError:
        return str(resolved)


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


def intersection_duration(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(
            0, min(left_end, right_end) - max(left_start, right_start)
        )
        if left_end < right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def build_cells(
    events_path: Path,
    event_windows_path: Path,
    topology_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, list[tuple[int, int, int]]],
    dict[str, Any],
]:
    columns = [
        "behavior",
        "iteration",
        "rank",
        "host",
        "start_ns",
        "end_ns",
    ]
    events = pd.read_parquet(events_path, columns=columns)
    topology = pd.read_csv(topology_path)
    windows = pd.read_csv(event_windows_path)
    behavior_names = set(BEHAVIOR_BY_NAME)
    edp = events[events["behavior"].isin(behavior_names)].copy()
    other = events[~events["behavior"].isin(behavior_names)].copy()

    if (
        len(topology) != 256
        or topology["rank"].nunique() != 256
        or topology["host"].nunique() != 32
    ):
        raise ValueError("rank topology is not a complete 256-rank/32-host map")
    if set(edp["iteration"].unique()) != set(ITERATIONS):
        raise ValueError("expert-DP iterations do not match the frozen grid")
    if set(edp["host"].unique()) != set(topology["host"].unique()):
        raise ValueError("expert-DP hosts do not match rank topology")

    host_meta = (
        topology.groupby("host", as_index=False)
        .agg(
            first_rank=("rank", "min"),
            last_rank=("rank", "max"),
            topology_rank_count=("rank", "nunique"),
        )
        .sort_values("first_rank")
        .reset_index(drop=True)
    )
    host_meta["node_index"] = np.arange(len(host_meta), dtype=np.int64)
    host_order = host_meta["host"].tolist()
    if (
        not host_meta["topology_rank_count"].eq(8).all()
        or not host_meta["last_rank"].sub(host_meta["first_rank"]).eq(7).all()
    ):
        raise ValueError("each host must contain eight contiguous global ranks")
    host_meta_by_name = host_meta.set_index("host").to_dict("index")

    host_windows = (
        windows.groupby(["host", "iteration"], as_index=False)
        .agg(
            profiler_step_start_ns=("start_ns", "min"),
            profiler_step_end_ns=("end_ns", "max"),
            nic_devices_covering=("nic_devices_covering", "min"),
        )
    )
    host_windows["profiler_step_duration_ns"] = (
        host_windows["profiler_step_end_ns"]
        - host_windows["profiler_step_start_ns"]
    )
    if (
        len(host_windows) != 32 * len(ITERATIONS)
        or not host_windows["nic_devices_covering"].eq(8).all()
    ):
        raise ValueError("host ProfilerStep/NIC coverage grid is incomplete")
    host_window_map = host_windows.set_index(["host", "iteration"]).to_dict(
        "index"
    )

    edp_groups = {
        (str(behavior), int(iteration), str(host)): group
        for (behavior, iteration, host), group in edp.groupby(
            ["behavior", "iteration", "host"]
        )
    }
    other_groups = {
        (int(iteration), str(host)): merge_intervals(
            zip(group["start_ns"].astype(int), group["end_ns"].astype(int))
        )
        for (iteration, host), group in other.groupby(["iteration", "host"])
    }

    cell_rows: list[dict[str, Any]] = []
    target_by_host: dict[str, list[tuple[int, int, int]]] = {
        host: [] for host in host_order
    }
    all_cell_intervals: dict[int, list[tuple[int, int]]] = {}
    for behavior_item in EDP_BEHAVIORS:
        behavior = str(behavior_item["behavior"])
        for iteration in ITERATIONS:
            for host in host_order:
                group = edp_groups.get((behavior, iteration, host))
                if group is None or group.empty:
                    raise ValueError(
                        f"missing complete host cell: {behavior}/{iteration}/{host}"
                    )
                intervals = merge_intervals(
                    zip(
                        group["start_ns"].astype(int),
                        group["end_ns"].astype(int),
                    )
                )
                union_duration_ns = sum(end - start for start, end in intervals)
                if union_duration_ns <= 0:
                    raise ValueError("expert-DP union duration is not positive")
                window = host_window_map[(host, iteration)]
                if any(
                    start < int(window["profiler_step_start_ns"])
                    or end > int(window["profiler_step_end_ns"])
                    for start, end in intervals
                ):
                    raise ValueError("expert-DP event is outside ProfilerStep")
                expected_events = (
                    int(behavior_item["expected_events_per_rank"]) * 8
                )
                event_count = int(len(group))
                data_status = (
                    "observed"
                    if event_count == expected_events
                    else "partial_source_event"
                )
                other_overlap_ns = intersection_duration(
                    intervals, other_groups.get((iteration, host), [])
                )
                meta = host_meta_by_name[host]
                cell_id = len(cell_rows)
                cell_rows.append(
                    {
                        "cell_id": cell_id,
                        "behavior_order": int(
                            behavior_item["behavior_order"]
                        ),
                        "behavior_family": behavior_item["behavior_family"],
                        "behavior": behavior,
                        "behavior_label": behavior_item["behavior_label"],
                        "behavior_description": behavior_item[
                            "behavior_description"
                        ],
                        "iteration": iteration,
                        "node_index": int(meta["node_index"]),
                        "host": host,
                        "first_rank": int(meta["first_rank"]),
                        "last_rank": int(meta["last_rank"]),
                        "event_count": event_count,
                        "expected_event_count": expected_events,
                        "event_count_delta": event_count - expected_events,
                        "observed_rank_count": int(group["rank"].nunique()),
                        "expected_rank_count": 8,
                        "data_status": data_status,
                        "edp_union_segment_count": len(intervals),
                        "edp_window_start_ns": min(x[0] for x in intervals),
                        "edp_window_end_ns": max(x[1] for x in intervals),
                        "edp_union_duration_ns": union_duration_ns,
                        "profiler_step_start_ns": int(
                            window["profiler_step_start_ns"]
                        ),
                        "profiler_step_end_ns": int(
                            window["profiler_step_end_ns"]
                        ),
                        "profiler_step_duration_ns": int(
                            window["profiler_step_duration_ns"]
                        ),
                        "other_collective_overlap_ns": other_overlap_ns,
                        "other_collective_overlap_pct": (
                            other_overlap_ns / union_duration_ns * 100.0
                        ),
                    }
                )
                all_cell_intervals[cell_id] = intervals
                for start, end in intervals:
                    target_by_host[host].append((start, end, cell_id))

    cross_cell_overlap_ns = 0
    for host, targets in target_by_host.items():
        targets.sort()
        for left, right in zip(targets, targets[1:]):
            cross_cell_overlap_ns += max(0, left[1] - right[0])
        if any(end <= start for start, end, _ in targets):
            raise ValueError(f"invalid target interval for {host}")
    if cross_cell_overlap_ns:
        raise ValueError(
            "expert-DP target windows overlap; independent attribution "
            "would duplicate NIC bytes"
        )

    cells = pd.DataFrame(cell_rows)
    build_validation = {
        "edp_source_event_count": int(len(edp)),
        "other_collective_source_event_count": int(len(other)),
        "target_interval_count": int(
            sum(len(value) for value in all_cell_intervals.values())
        ),
        "cross_cell_overlap_ns": int(cross_cell_overlap_ns),
    }
    return cells, target_by_host, build_validation


def attribute_device(
    path: Path | list[Path],
    host: str,
    device: str,
    targets: list[tuple[int, int, int]],
    cell_count: int,
    chunk_size: int,
    long_gap_ns: int,
) -> list[dict[str, Any]]:
    target_starts = np.array([item[0] for item in targets], dtype=np.int64)
    target_ends = np.array([item[1] for item in targets], dtype=np.int64)
    target_cells = np.array([item[2] for item in targets], dtype=np.int64)
    tx = np.zeros(cell_count, dtype=np.float64)
    rx = np.zeros(cell_count, dtype=np.float64)
    uncertain_tx = np.zeros(cell_count, dtype=np.float64)
    uncertain_rx = np.zeros(cell_count, dtype=np.float64)
    sampled_overlap_ns = np.zeros(cell_count, dtype=np.int64)
    overlap_piece_count = np.zeros(cell_count, dtype=np.int64)
    invalid_overlap_piece_count = np.zeros(cell_count, dtype=np.int64)

    usecols = ["timestamp_ns", "sample_interval_us", "xmit_bytes", "recv_bytes"]
    input_paths = [path] if isinstance(path, Path) else path
    chunks = (
        chunk
        for input_path in input_paths
        for chunk in pd.read_csv(
            input_path,
            usecols=usecols,
            chunksize=chunk_size,
            dtype={
                "timestamp_ns": "int64",
                "sample_interval_us": "float64",
                "xmit_bytes": "int64",
                "recv_bytes": "int64",
            },
        )
    )
    for chunk in chunks:
        ends = chunk["timestamp_ns"].to_numpy(dtype=np.int64)
        durations = np.rint(
            chunk["sample_interval_us"].to_numpy(dtype=np.float64) * 1000.0
        ).astype(np.int64)
        starts = ends - durations
        raw_tx = chunk["xmit_bytes"].to_numpy(dtype=np.int64)
        raw_rx = chunk["recv_bytes"].to_numpy(dtype=np.int64)
        first = np.searchsorted(target_ends, starts, side="right")
        stop = np.searchsorted(target_starts, ends, side="left")
        candidate_count = stop - first
        valid_sample = (durations > 0) & (raw_tx >= 0) & (raw_rx >= 0)
        overlaps = candidate_count > 0

        invalid_rows = np.flatnonzero(overlaps & ~valid_sample)
        for row_index in invalid_rows:
            for target_index in range(first[row_index], stop[row_index]):
                invalid_overlap_piece_count[target_cells[target_index]] += 1

        single_rows = np.flatnonzero(
            overlaps & valid_sample & candidate_count.__eq__(1)
        )
        if len(single_rows):
            target_index = first[single_rows]
            overlap_ns = np.minimum(
                ends[single_rows], target_ends[target_index]
            ) - np.maximum(starts[single_rows], target_starts[target_index])
            usable = overlap_ns > 0
            row_index = single_rows[usable]
            target_index = target_index[usable]
            overlap_ns = overlap_ns[usable]
            cell_index = target_cells[target_index]
            ratio = overlap_ns / durations[row_index]
            np.add.at(tx, cell_index, raw_tx[row_index] * ratio)
            np.add.at(rx, cell_index, raw_rx[row_index] * ratio)
            np.add.at(sampled_overlap_ns, cell_index, overlap_ns)
            np.add.at(overlap_piece_count, cell_index, 1)
            long_gap = durations[row_index] > long_gap_ns
            np.add.at(
                uncertain_tx,
                cell_index[long_gap],
                raw_tx[row_index[long_gap]] * ratio[long_gap],
            )
            np.add.at(
                uncertain_rx,
                cell_index[long_gap],
                raw_rx[row_index[long_gap]] * ratio[long_gap],
            )

        multiple_rows = np.flatnonzero(
            overlaps & valid_sample & (candidate_count > 1)
        )
        for row_index in multiple_rows:
            for target_index in range(first[row_index], stop[row_index]):
                overlap_ns = max(
                    0,
                    min(int(ends[row_index]), int(target_ends[target_index]))
                    - max(
                        int(starts[row_index]),
                        int(target_starts[target_index]),
                    ),
                )
                if overlap_ns == 0:
                    continue
                cell_index = int(target_cells[target_index])
                ratio = overlap_ns / int(durations[row_index])
                tx[cell_index] += int(raw_tx[row_index]) * ratio
                rx[cell_index] += int(raw_rx[row_index]) * ratio
                sampled_overlap_ns[cell_index] += overlap_ns
                overlap_piece_count[cell_index] += 1
                if durations[row_index] > long_gap_ns:
                    uncertain_tx[cell_index] += int(raw_tx[row_index]) * ratio
                    uncertain_rx[cell_index] += int(raw_rx[row_index]) * ratio

    observed_cells = sorted(set(target_cells.tolist()))
    return [
        {
            "cell_id": cell_id,
            "host": host,
            "device": device,
            "nic_attributed_tx_bytes": tx[cell_id],
            "nic_attributed_rx_bytes": rx[cell_id],
            "nic_attributed_bidir_bytes": tx[cell_id] + rx[cell_id],
            "nic_uncertain_long_gap_tx_bytes": uncertain_tx[cell_id],
            "nic_uncertain_long_gap_rx_bytes": uncertain_rx[cell_id],
            "nic_sampled_overlap_ns": int(sampled_overlap_ns[cell_id]),
            "nic_overlap_piece_count": int(overlap_piece_count[cell_id]),
            "nic_invalid_overlap_piece_count": int(
                invalid_overlap_piece_count[cell_id]
            ),
        }
        for cell_id in observed_cells
    ]


def main() -> int:
    args = parse_args()
    events_path = args.events_parquet.resolve()
    hardware_root = args.hardware_root.resolve()
    manifest_path = args.nic_manifest.resolve()
    cells, targets_by_host, build_validation = build_cells(
        events_path,
        args.event_windows.resolve(),
        args.rank_topology.resolve(),
    )
    manifest = pd.read_csv(manifest_path)
    expected_devices = {
        "mlx5_0",
        "mlx5_1",
        "mlx5_2",
        "mlx5_3",
        "mlx5_6",
        "mlx5_7",
        "mlx5_8",
        "mlx5_9",
    }
    if (
        manifest[["host", "device"]].drop_duplicates().shape[0] != 256
        or manifest["host"].nunique() != 32
        or set(manifest["device"].unique()) != expected_devices
        or not manifest.groupby("host")["device"].nunique().eq(8).all()
    ):
        raise ValueError("NIC manifest is not a complete 32-host/8-device grid")

    device_rows: list[dict[str, Any]] = []
    long_gap_ns = round(args.nic_long_gap_ms * 1_000_000)
    for (host, device_name), group in manifest.sort_values(
        ["host", "device", "sequence"]
    ).groupby(["host", "device"]):
        paths = [hardware_root / str(value) for value in group["path"]]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(paths)
        device_rows.extend(
            attribute_device(
                path=paths,
                host=str(host),
                device=str(device_name),
                targets=targets_by_host[str(host)],
                cell_count=len(cells),
                chunk_size=args.chunk_size,
                long_gap_ns=long_gap_ns,
            )
        )

    device = pd.DataFrame(device_rows).merge(
        cells[
            [
                "cell_id",
                "behavior_order",
                "behavior",
                "iteration",
                "edp_union_duration_ns",
            ]
        ],
        on="cell_id",
        how="left",
        validate="many_to_one",
    )
    device["nic_active_window_tx_bandwidth_GBps"] = (
        device["nic_attributed_tx_bytes"] / device["edp_union_duration_ns"]
    )
    device["nic_active_window_rx_bandwidth_GBps"] = (
        device["nic_attributed_rx_bytes"] / device["edp_union_duration_ns"]
    )
    device["nic_active_window_bidir_bandwidth_GBps"] = (
        device["nic_attributed_bidir_bytes"]
        / device["edp_union_duration_ns"]
    )
    device["nic_window_coverage_pct"] = (
        device["nic_sampled_overlap_ns"]
        / device["edp_union_duration_ns"]
        * 100.0
    )
    device = device.sort_values(
        ["behavior_order", "iteration", "host", "device"]
    )

    host_nic = (
        device.groupby("cell_id", as_index=False)
        .agg(
            nic_device_count=("device", "nunique"),
            nic_attributed_tx_bytes=(
                "nic_attributed_tx_bytes",
                "sum",
            ),
            nic_attributed_rx_bytes=(
                "nic_attributed_rx_bytes",
                "sum",
            ),
            nic_attributed_bidir_bytes=(
                "nic_attributed_bidir_bytes",
                "sum",
            ),
            nic_uncertain_long_gap_tx_bytes=(
                "nic_uncertain_long_gap_tx_bytes",
                "sum",
            ),
            nic_uncertain_long_gap_rx_bytes=(
                "nic_uncertain_long_gap_rx_bytes",
                "sum",
            ),
            nic_overlap_piece_count=("nic_overlap_piece_count", "sum"),
            nic_invalid_overlap_piece_count=(
                "nic_invalid_overlap_piece_count",
                "sum",
            ),
            nic_min_device_window_coverage_pct=(
                "nic_window_coverage_pct",
                "min",
            ),
            nic_max_device_window_coverage_pct=(
                "nic_window_coverage_pct",
                "max",
            ),
        )
    )
    result = cells.merge(
        host_nic, on="cell_id", how="left", validate="one_to_one"
    )
    active_duration = result["edp_union_duration_ns"]
    step_duration = result["profiler_step_duration_ns"]
    result["nic_active_window_tx_bandwidth_GBps"] = (
        result["nic_attributed_tx_bytes"] / active_duration
    )
    result["nic_active_window_rx_bandwidth_GBps"] = (
        result["nic_attributed_rx_bytes"] / active_duration
    )
    result["nic_active_window_bidir_bandwidth_GBps"] = (
        result["nic_attributed_bidir_bytes"] / active_duration
    )
    result["nic_profiler_step_tx_bandwidth_GBps"] = (
        result["nic_attributed_tx_bytes"] / step_duration
    )
    result["nic_profiler_step_rx_bandwidth_GBps"] = (
        result["nic_attributed_rx_bytes"] / step_duration
    )
    result["nic_profiler_step_bidir_bandwidth_GBps"] = (
        result["nic_attributed_bidir_bytes"] / step_duration
    )
    result["nic_long_gap_bidir_pct"] = (
        (
            result["nic_uncertain_long_gap_tx_bytes"]
            + result["nic_uncertain_long_gap_rx_bytes"]
        )
        / result["nic_attributed_bidir_bytes"].replace(0, np.nan)
        * 100.0
    )
    result["nic_long_gap_warning"] = (
        result["nic_long_gap_bidir_pct"].fillna(0).gt(5.0)
    )
    result = result.sort_values(
        ["behavior_order", "iteration", "node_index"]
    ).drop(columns="cell_id")

    attributed_bidir = float(result["nic_attributed_bidir_bytes"].sum())
    uncertain_bidir = float(
        result["nic_uncertain_long_gap_tx_bytes"].sum()
        + result["nic_uncertain_long_gap_rx_bytes"].sum()
    )
    partial = result[result["data_status"].eq("partial_source_event")]
    overlap_stats = (
        result.groupby("behavior")["other_collective_overlap_pct"]
        .agg(["median", "max"])
        .reset_index()
        .to_dict("records")
    )
    validation = {
        "status": "PASS",
        "schema_version": "256gpu-edp-nic-host-spatial-v1",
        "attribution_granularity": "behavior x iteration x host",
        "rank_to_nic_attribution_performed": False,
        "host_rank_event_intervals_merged_before_attribution": True,
        "nic_byte_attribution": (
            "sample_delta_bytes * overlap_ns / sample_interval_ns"
        ),
        "active_window_bandwidth_formula": (
            "sum(attributed NIC bytes across 8 devices) / "
            "merged EDP union duration_ns"
        ),
        "profiler_step_bandwidth_formula": (
            "sum(attributed NIC bytes across 8 devices) / "
            "host ProfilerStep duration_ns"
        ),
        "bandwidth_unit": "decimal GB/s",
        "events_parquet": portable_path(events_path),
        "nic_manifest": portable_path(manifest_path),
        "output_csv": portable_path(args.output_csv),
        "device_audit_output": portable_path(args.device_audit_output),
        "row_count": int(len(result)),
        "expected_row_count": 2 * len(ITERATIONS) * 32,
        "device_audit_row_count": int(len(device)),
        "expected_device_audit_row_count": 2 * len(ITERATIONS) * 32 * 8,
        "behavior_count": int(result["behavior"].nunique()),
        "iteration_count": int(result["iteration"].nunique()),
        "host_count": int(result["host"].nunique()),
        "nic_device_count": int(device["device"].nunique()),
        "partial_source_host_cell_count": int(len(partial)),
        "partial_source_host_cells": partial[
            [
                "behavior",
                "iteration",
                "host",
                "event_count",
                "expected_event_count",
            ]
        ].to_dict("records"),
        "cross_cell_overlap_ns": build_validation["cross_cell_overlap_ns"],
        "target_interval_count": build_validation["target_interval_count"],
        "edp_source_event_count": build_validation["edp_source_event_count"],
        "nic_manifest_sample_row_count": int(manifest["row_count"].sum()),
        "attributed_nic_bidir_bytes": attributed_bidir,
        "long_gap_attributed_bidir_fraction": (
            uncertain_bidir / attributed_bidir
            if attributed_bidir
            else None
        ),
        "minimum_device_window_coverage_pct": float(
            result["nic_min_device_window_coverage_pct"].min()
        ),
        "maximum_device_window_coverage_pct": float(
            result["nic_max_device_window_coverage_pct"].max()
        ),
        "invalid_overlap_piece_count": int(
            result["nic_invalid_overlap_piece_count"].sum()
        ),
        "other_collective_overlap_pct_by_behavior": overlap_stats,
        "interpretation_limits": [
            (
                "NIC values are host-level window-aligned physical counters, "
                "not rank- or rail-specific measurements."
            ),
            (
                "Window alignment is correlation, not proof that every byte "
                "was caused exclusively by expert-DP."
            ),
            (
                "TX+RX is full-duplex activity and must not be added to "
                "Trace or MTLink byte domains."
            ),
            (
                "Bytes from NIC sampling intervals longer than 10 ms remain "
                "included and are separately reported."
            ),
        ],
    }
    checks = [
        len(result) == validation["expected_row_count"],
        len(device) == validation["expected_device_audit_row_count"],
        validation["behavior_count"] == 2,
        validation["iteration_count"] == len(ITERATIONS),
        validation["host_count"] == 32,
        validation["nic_device_count"] == 8,
        result["nic_device_count"].eq(8).all(),
        result["nic_attributed_bidir_bytes"].notna().all(),
        validation["cross_cell_overlap_ns"] == 0,
        validation["invalid_overlap_piece_count"] == 0,
        validation["minimum_device_window_coverage_pct"] >= 99.0,
    ]
    if not all(checks):
        validation["status"] = "FAIL"
        raise RuntimeError(
            json.dumps(clean_json(validation), ensure_ascii=False, indent=2)
        )

    atomic_csv(result, args.output_csv)
    atomic_csv(device, args.device_audit_output)
    atomic_json(validation, args.validation_output)
    print(json.dumps(clean_json(validation), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

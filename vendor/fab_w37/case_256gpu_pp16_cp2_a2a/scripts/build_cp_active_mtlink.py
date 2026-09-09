#!/usr/bin/env python3
"""Build CP MTLink TX/RX rates with MTLink-native active time.

The page-facing physical denominator is the union of active MTLink sample
intervals clipped to CP all-to-all kernel windows.  NIC is intentionally not
read: every validated CP size-2 group is contained in one host.

In addition to uint64 underflow rows, the immediately following sample on the
same link is quarantined.  The raw data contains reset-recovery deltas below
2**63 that are still physically impossible (about 137 GB in about 5 ms).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_dp_active_physical import (
    ITERATIONS,
    MAX_I64,
    RANKS,
    atomic_csv,
    atomic_json,
    attribute_domain,
    clean_json,
    merge_intervals,
    parse_rank_subset,
    portable,
    resolve_mtlink_paths,
)


BEHAVIOR = "cp_all_to_all"
LABEL = "CP-all-to-all"


@dataclass(frozen=True)
class RankTask:
    rank: int
    target_starts: np.ndarray
    target_ends: np.ndarray
    target_cells: np.ndarray
    mtlink_paths: tuple[Path, ...]
    mtlink_min_rate_GBps: float
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
        "--mtlink-manifest",
        type=Path,
        default=case_root / "results" / "readiness" / "mtlink_files.csv",
    )
    parser.add_argument(
        "--hardware-root", type=Path, default=case_root / ".raw_view"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=result_dir / "cp_active_mtlink_iteration_rank.csv",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=result_dir / "cp_active_mtlink_validation.json",
    )
    parser.add_argument("--mtlink-min-rate-GBps", type=float, default=0.05)
    parser.add_argument("--long-gap-ms", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ranks")
    args = parser.parse_args()
    if args.mtlink_min_rate_GBps < 0:
        parser.error("--mtlink-min-rate-GBps must be non-negative")
    if args.long_gap_ms <= 0 or args.workers < 1:
        parser.error("long-gap and workers must be positive")
    return args


def expected_event_count(pp_stage: int) -> int:
    return 104 if pp_stage in (0, 15) else 208


def build_cells(
    event_path: Path,
    source_path: Path,
    topology_path: Path,
    ranks: list[int],
) -> tuple[
    pd.DataFrame,
    dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, Any],
]:
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
        events["behavior"].eq(BEHAVIOR) & events["rank"].isin(ranks)
    ].copy()
    source = pd.read_csv(source_path)
    source = source[
        source["behavior"].eq(BEHAVIOR) & source["rank"].isin(ranks)
    ].set_index(["iteration", "rank"])
    topology = (
        pd.read_csv(topology_path)
        .loc[lambda frame: frame["rank"].isin(ranks)]
        .set_index("rank")
    )
    grouped = {
        (int(iteration), int(rank)): group
        for (iteration, rank), group in events.groupby(["iteration", "rank"])
    }

    rows: list[dict[str, Any]] = []
    targets: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cross_cell_overlap_ns = 0
    interval_count = 0
    for rank in ranks:
        meta = topology.loc[rank]
        starts: list[int] = []
        ends: list[int] = []
        cells: list[int] = []
        for cell, iteration in enumerate(ITERATIONS):
            group = grouped.get((iteration, rank))
            intervals = (
                merge_intervals(
                    zip(
                        group["start_ns"].astype(int),
                        group["end_ns"].astype(int),
                    )
                )
                if group is not None
                else []
            )
            event_count = 0 if group is None else int(len(group))
            expected = expected_event_count(int(meta["pp_stage"]))
            union_ns = sum(end - start for start, end in intervals)
            duration_sum = (
                0 if group is None else int(group["duration_ns"].sum())
            )
            span_ns = (
                0
                if not intervals
                else max(end for _start, end in intervals)
                - min(start for start, _end in intervals)
            )
            source_row = source.loc[(iteration, rank)]
            rows.append(
                {
                    "local_cell": cell,
                    "behavior": BEHAVIOR,
                    "behavior_label": LABEL,
                    "iteration": iteration,
                    "rank": rank,
                    "node_index": rank // 8,
                    "host": str(meta["host"]),
                    "local_rank": int(meta["local_rank"]),
                    "gpu_id": int(meta["gpu_id"]),
                    "pp_stage": int(meta["pp_stage"]),
                    "cp_group": int(meta["cp_group"]),
                    "cp_rank": int(meta["cp_rank"]),
                    "kernel_name": (
                        "mcclKernel_SendRecv_RING_SIMPLE_Sum_int8_t"
                        "(mcclWorkElem)"
                    ),
                    "event_count": event_count,
                    "expected_event_count": expected,
                    "data_status": (
                        "observed"
                        if event_count == expected
                        else "partial_source_event"
                    ),
                    "kernel_duration_sum_ns": duration_sum,
                    "kernel_union_duration_ns": union_ns,
                    "kernel_span_ns": span_ns,
                    "kernel_union_segment_count": len(intervals),
                    "expected_tx_bytes_total": float(
                        source_row["expected_tx_bytes_total"]
                    ),
                    "expected_rx_bytes_total": float(
                        source_row["expected_rx_bytes_total"]
                    ),
                    "canonical_mtlink_tx_bytes_total": float(
                        source_row["mtlink_attributed_tx_bytes_total"]
                    ),
                    "canonical_mtlink_rx_bytes_total": float(
                        source_row["mtlink_attributed_rx_bytes_total"]
                    ),
                }
            )
            for start, end in intervals:
                starts.append(start)
                ends.append(end)
                cells.append(cell)
                interval_count += 1
        order = np.argsort(np.asarray(starts, dtype=np.int64))
        start_array = np.asarray(starts, dtype=np.int64)[order]
        end_array = np.asarray(ends, dtype=np.int64)[order]
        cell_array = np.asarray(cells, dtype=np.int64)[order]
        if len(start_array) > 1:
            cross_cell_overlap_ns += int(
                np.maximum(0, end_array[:-1] - start_array[1:]).sum()
            )
        targets[rank] = (start_array, end_array, cell_array)
    return (
        pd.DataFrame(rows),
        targets,
        {
            "source_event_count": int(len(events)),
            "target_interval_count": interval_count,
            "cross_cell_overlap_ns": cross_cell_overlap_ns,
        },
    )


def read_mtlink_reset_clean(
    paths: tuple[Path, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
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
    frame = frame[frame["link_id"].between(0, 13)].copy()
    frame = frame.sort_values(
        ["link_id", "host_realtime_ns"], kind="stable"
    ).reset_index(drop=True)
    duration = (
        frame["mt_timestamp_end_ns"].to_numpy(dtype=np.int64)
        - frame["mt_timestamp_begin_ns"].to_numpy(dtype=np.int64)
    )
    tx = frame["tx_delta_bytes"].to_numpy(dtype=np.uint64)
    rx = frame["rx_delta_bytes"].to_numpy(dtype=np.uint64)
    reset = (tx > MAX_I64) | (rx > MAX_I64)
    recovery = (
        pd.Series(reset)
        .groupby(frame["link_id"], sort=False)
        .shift(1, fill_value=False)
        .to_numpy(dtype=bool)
    )
    valid = (duration > 0) & ~reset & ~recovery
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
    return (
        frame.sort_values("sample_end_ns", kind="stable").reset_index(drop=True),
        {
            "reset_underflow_row_count": int(reset.sum()),
            "reset_recovery_row_count": int(recovery.sum()),
        },
    )


def process_rank(task: RankTask) -> tuple[pd.DataFrame, dict[str, int]]:
    samples, cleaning = read_mtlink_reset_clean(task.mtlink_paths)
    metrics = attribute_domain(
        samples,
        task.target_starts,
        task.target_ends,
        task.target_cells,
        len(ITERATIONS),
        task.mtlink_min_rate_GBps,
        task.mtlink_min_rate_GBps,
        1.0,
        task.long_gap_ns,
        "mtlink",
    )
    metrics.insert(0, "rank", task.rank)
    return metrics, cleaning


def add_cp_rates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    kernel_duration = result["kernel_union_duration_ns"].replace(0, np.nan)
    result["mtlink_kernel_window_tx_bandwidth_GBps"] = (
        result["mtlink_window_attributed_tx_bytes"] / kernel_duration
    )
    result["mtlink_kernel_window_rx_bandwidth_GBps"] = (
        result["mtlink_window_attributed_rx_bytes"] / kernel_duration
    )
    for direction in ("tx", "rx"):
        active_duration = result[
            f"mtlink_active_{direction}_duration_ns"
        ].replace(0, np.nan)
        result[f"mtlink_active_{direction}_bandwidth_GBps"] = (
            result[f"mtlink_active_{direction}_bytes"] / active_duration
        )
        result[f"mtlink_active_{direction}_fraction_of_kernel_pct"] = (
            result[f"mtlink_active_{direction}_duration_ns"]
            / kernel_duration
            * 100.0
        )
    uncertain = (
        result["mtlink_uncertain_long_gap_tx_bytes"]
        + result["mtlink_uncertain_long_gap_rx_bytes"]
    )
    window_bidir = (
        result["mtlink_window_attributed_tx_bytes"]
        + result["mtlink_window_attributed_rx_bytes"]
    )
    result["mtlink_long_gap_bidir_pct"] = (
        uncertain / window_bidir.replace(0, np.nan) * 100.0
    )
    return result


def main() -> int:
    args = parse_args()
    case_root = args.case_root.resolve()
    ranks = parse_rank_subset(args.ranks)
    cells, targets, build_validation = build_cells(
        args.event_parquet.resolve(),
        args.source_cell_csv.resolve(),
        args.rank_topology.resolve(),
        ranks,
    )
    paths = resolve_mtlink_paths(
        args.mtlink_manifest.resolve(), args.hardware_root.resolve()
    )
    identities = cells.drop_duplicates("rank").set_index("rank")
    tasks: list[RankTask] = []
    for rank in ranks:
        identity = identities.loc[rank]
        starts, ends, target_cells = targets[rank]
        task = RankTask(
            rank=rank,
            target_starts=starts,
            target_ends=ends,
            target_cells=target_cells,
            mtlink_paths=paths[(str(identity["host"]), int(identity["gpu_id"]))],
            mtlink_min_rate_GBps=args.mtlink_min_rate_GBps,
            long_gap_ns=round(args.long_gap_ms * 1_000_000),
        )
        if any(not path.is_file() for path in task.mtlink_paths):
            raise FileNotFoundError(task)
        tasks.append(task)

    frames: list[pd.DataFrame] = []
    cleaning_totals = {
        "reset_underflow_row_count": 0,
        "reset_recovery_row_count": 0,
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(process_rank, task): task.rank for task in tasks}
        for completed, future in enumerate(as_completed(future_map), start=1):
            metrics, cleaning = future.result()
            frames.append(metrics)
            for key in cleaning_totals:
                cleaning_totals[key] += cleaning[key]
            if completed % 8 == 0 or completed == len(tasks):
                print(f"processed {completed}/{len(tasks)} ranks", flush=True)
    metrics = pd.concat(frames, ignore_index=True)
    output = cells.merge(
        metrics, on=["rank", "local_cell"], validate="one_to_one"
    )
    output = add_cp_rates(output).drop(columns="local_cell")
    output = output.sort_values(["iteration", "rank"])
    observed = output[output["data_status"].ne("missing_source_event")]
    cp_rank_meta = output[
        ["rank", "host", "cp_group", "cp_rank"]
    ].drop_duplicates()
    cp_group_shape = cp_rank_meta.groupby("cp_group").agg(
        rank_count=("rank", "nunique"),
        host_count=("host", "nunique"),
        cp_rank_count=("cp_rank", "nunique"),
    )
    cp_groups_two_ranks_one_host = bool(
        cp_group_shape["rank_count"].eq(2).all()
        and cp_group_shape["host_count"].eq(1).all()
        and cp_group_shape["cp_rank_count"].eq(2).all()
    )

    expected_rows = len(ranks) * len(ITERATIONS)
    validation: dict[str, Any] = {
        "status": "PASS",
        "schema_version": "256gpu-cp-active-mtlink-v1",
        "full_rank_run": ranks == list(RANKS),
        "row_count": int(len(output)),
        "expected_row_count": expected_rows,
        "rank_count": int(output["rank"].nunique()),
        "host_count": int(output["host"].nunique()),
        "iteration_count": int(output["iteration"].nunique()),
        "cp_group_count": int(len(cp_group_shape)),
        "cp_groups_two_ranks_one_host": cp_groups_two_ranks_one_host,
        "source_event_count": build_validation["source_event_count"],
        "target_interval_count": build_validation["target_interval_count"],
        "cross_cell_overlap_ns": build_validation["cross_cell_overlap_ns"],
        "data_status_counts": {
            str(key): int(value)
            for key, value in output["data_status"].value_counts().items()
        },
        "kernel_name": output["kernel_name"].iloc[0],
        "kernel_time_definition": "union of CP all_to_all GPU kernel intervals",
        "active_time_definition": (
            "union of active MTLink sample intervals clipped to CP kernel windows"
        ),
        "mtlink_activity_threshold_GBps_per_link": args.mtlink_min_rate_GBps,
        "nic_read_or_attributed": False,
        **cleaning_totals,
        "kernel_duration_ms_p50": float(
            observed["kernel_union_duration_ns"].median() / 1e6
        ),
        "active_tx_duration_ms_p50": float(
            observed["mtlink_active_tx_duration_ns"].median() / 1e6
        ),
        "active_rx_duration_ms_p50": float(
            observed["mtlink_active_rx_duration_ns"].median() / 1e6
        ),
        "active_tx_fraction_of_kernel_pct_p50": float(
            observed["mtlink_active_tx_fraction_of_kernel_pct"].median()
        ),
        "kernel_window_tx_bandwidth_GBps_p50": float(
            observed["mtlink_kernel_window_tx_bandwidth_GBps"].median()
        ),
        "active_tx_bandwidth_GBps_p50": float(
            observed["mtlink_active_tx_bandwidth_GBps"].median()
        ),
        "active_rx_bandwidth_GBps_p50": float(
            observed["mtlink_active_rx_bandwidth_GBps"].median()
        ),
        "active_rx_bandwidth_GBps_max": float(
            observed["mtlink_active_rx_bandwidth_GBps"].max()
        ),
        "output_csv": portable(args.output_csv, case_root.parent),
        "notes": [
            "CP has no NIC output because all validated CP groups are intra-host.",
            "Kernel-time and MTLink-active-time are separate denominator views.",
            "Reset underflow rows and their same-link successor samples are quarantined.",
        ],
    }
    checks = [
        len(output) == expected_rows,
        validation["rank_count"] == len(ranks),
        validation["iteration_count"] == len(ITERATIONS),
        validation["cross_cell_overlap_ns"] == 0,
        output.duplicated(["iteration", "rank"]).sum() == 0,
        observed["mtlink_active_tx_duration_ns"].gt(0).all(),
        observed["mtlink_active_rx_duration_ns"].gt(0).all(),
        observed["mtlink_active_tx_duration_ns"]
        .le(observed["kernel_union_duration_ns"])
        .all(),
        observed["mtlink_active_rx_duration_ns"]
        .le(observed["kernel_union_duration_ns"])
        .all(),
        not validation["full_rank_run"]
        or (
            validation["cp_group_count"] == 128
            and validation["cp_groups_two_ranks_one_host"]
        ),
    ]
    if not all(checks):
        validation["status"] = "FAIL"
    atomic_csv(output, args.output_csv)
    atomic_json(validation, args.validation_output)
    print(json.dumps(clean_json(validation), ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

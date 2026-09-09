#!/usr/bin/env python3
"""Build 256-rank collective bandwidth CSVs using the 16-GPU case formulas.

PP Send/Recv and PP/model-parallel control groups are deliberately excluded.
Logical/expected bytes and MTLink physical counters remain separate domains.
"""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


COLLECTIVE_MARKER = b'"Collective name"'
DTYPE_BYTES = {
    "Byte": 1,
    "Char": 1,
    "Int8": 1,
    "Half": 2,
    "BFloat16": 2,
    "Float": 4,
    "Int": 4,
    "Long": 8,
    "Double": 8,
    "Bool": 1,
}
EP_BEHAVIOR = {
    "dispatch": "ep_fwd_dispatch",
    "combine": "ep_fwd_combine",
    "combine_backward_dispatch": "ep_bwd_combine_backward_dispatch",
    "dispatch_backward_combine": "ep_bwd_dispatch_backward_combine",
}
PP_DESCRIPTIONS = {
    "PIPELINE_MODEL_PARALLEL_GROUP",
    "MODEL_PARALLEL_GROUP",
    "EMBEDDING_GROUP",
    "POSITION_EMBEDDING_GROUP",
}
EXCLUDED_COLLECTIVES = {"send", "recv", "wait"}
MAX_I64 = np.uint64(2**63 - 1)
PCIE_LINK_ID = 2**32 - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--hardware-root", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--trace-metadata", type=Path, required=True)
    parser.add_argument("--event-windows", type=Path, required=True)
    parser.add_argument("--rank-topology", type=Path, required=True)
    parser.add_argument("--mtlink-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mtlink-long-gap-ms", type=float, default=10.0)
    parser.add_argument("--world-size", type=int, default=256)
    parser.add_argument(
        "--expected-deepep-events", type=int, default=453_120
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def classify_trace(collective: str, pg: str) -> str:
    op = slug(collective)
    if pg == "CONTEXT_PARALLEL_GROUP" and collective == "all_to_all":
        return "cp_all_to_all"
    if pg == "TENSOR_AND_CONTEXT_PARALLEL_GROUP":
        return f"cp_control_{op}"
    if pg == "DATA_PARALLEL_GROUP_WITH_CP":
        if collective == "reduce_scatter_tensor_coalesced":
            return "dp_grad_reduce_scatter"
        if collective == "allgather_into_tensor_coalesced":
            return "dp_param_allgather"
        return f"dp_with_cp_{op}"
    if pg == "DATA_PARALLEL_GROUP":
        return f"dp_{op}"
    if pg == "EXPERT_DATA_PARALLEL_GROUP":
        if collective == "reduce_scatter_tensor_coalesced":
            return "expert_dp_grad_reduce_scatter"
        if collective == "allgather_into_tensor_coalesced":
            return "expert_dp_param_allgather"
        return f"expert_dp_{op}"
    if pg == "default_pg":
        return f"global_control_{op}"
    return f"{slug(pg)}__{op}"


def expected_trace_bytes(
    collective: str, input_bytes: int, group_size: int
) -> tuple[float, float, str]:
    n = int(group_size)
    if n <= 1:
        return 0.0, 0.0, "group_size<=1"
    if collective in {"allreduce"}:
        value = 2.0 * (n - 1) / n * input_bytes
        return value, value, "ring allreduce 2*(N-1)/N*input"
    if collective == "reduce_scatter_tensor_coalesced":
        value = (n - 1) / n * input_bytes
        return value, value, "ring reduce-scatter (N-1)/N*input"
    if collective in {"allgather_into_tensor_coalesced", "_allgather_base"}:
        value = (n - 1) * input_bytes
        return value, value, "ring all-gather (N-1)*local_input"
    if collective == "all_to_all":
        value = (n - 1) / n * input_bytes
        return value, value, "equal-split all-to-all excludes local share"
    if collective == "barrier":
        return 0.0, 0.0, "barrier has no payload"
    return np.nan, np.nan, "unavailable"


def window_maps(
    path: Path,
) -> tuple[dict[int, pd.DataFrame], dict[int, tuple[np.ndarray, np.ndarray]]]:
    frame = pd.read_csv(path).sort_values(["rank", "iteration"])
    tables: dict[int, pd.DataFrame] = {}
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rank, group in frame.groupby("rank"):
        table = group[["iteration", "start_ns", "end_ns"]].reset_index(drop=True)
        tables[int(rank)] = table
        arrays[int(rank)] = (
            table["start_ns"].to_numpy(dtype=np.int64),
            table["end_ns"].to_numpy(dtype=np.int64),
        )
    return tables, arrays


def assign_window(
    start_ns: int,
    end_ns: int,
    table: pd.DataFrame,
    arrays: tuple[np.ndarray, np.ndarray],
) -> int | None:
    starts, ends = arrays
    index = int(np.searchsorted(ends, start_ns, side="right"))
    if index < len(starts) and end_ns > starts[index] and start_ns < ends[index]:
        return int(table.iloc[index]["iteration"])
    return None


def parse_deepep(
    framework_root: Path,
    topology: pd.DataFrame,
    tables: dict[int, pd.DataFrame],
    arrays: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    rank_info = topology.set_index("rank").to_dict("index")
    rows: list[dict[str, object]] = []
    for path in sorted(framework_root.glob("**/deepep_trace/*/rank_*.log")):
        rank = int(path.stem.split("_", 1)[1])
        info = rank_info[rank]
        path_host = path.parents[2].name
        if path_host != info["host"]:
            raise ValueError(f"DeepEP host mismatch: {path}")
        current_call = None
        with path.open(errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if doc.get("type") != "bw" or doc.get("op") not in EP_BEHAVIOR:
                    continue
                if doc["op"] == "dispatch":
                    current_call = int(doc["iter"])
                end_ns = int(doc["timestamp_ns"])
                duration_ns = round(float(doc["elapsed_ms"]) * 1_000_000)
                start_ns = end_ns - duration_ns
                iteration = assign_window(
                    start_ns, end_ns, tables[rank], arrays[rank]
                )
                if iteration is None:
                    continue
                send_bytes = int(doc.get("send_bytes", 0) or 0)
                recv_bytes = int(doc.get("recv_bytes", 0) or 0)
                comm_bytes = int(doc.get("comm_bytes", 0) or 0)
                rows.append(
                    {
                        "source": "deepep",
                        "behavior": EP_BEHAVIOR[str(doc["op"])],
                        "iteration": iteration,
                        "rank": rank,
                        "host": info["host"],
                        "gpu_id": int(info["gpu_id"]),
                        "collective": "alltoall_semantic",
                        "pg_description": "EXPERT_MODEL_PARALLEL_GROUP",
                        "pg_name": "",
                        "group_size": int(doc["world_size"]),
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ns": duration_ns,
                        "logical_input_bytes": comm_bytes,
                        "expected_tx_bytes": float(send_bytes),
                        "expected_rx_bytes": float(recv_bytes),
                        "expected_byte_definition": "DeepEP direct send/recv bytes",
                        "dtype": "",
                        "external_id": "",
                        "call_index": (
                            str(current_call) if current_call is not None else ""
                        ),
                        "source_path": str(path.relative_to(framework_root)),
                        "source_row": str(line_number),
                    }
                )
    return rows


def trace_event_blobs(path: Path):
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            position = 0
            while True:
                marker = mapped.find(COLLECTIVE_MARKER, position)
                if marker < 0:
                    return
                start = mapped.rfind(b"\n  {", 0, marker)
                end = mapped.find(b"\n  },", marker)
                if start < 0 or end < 0:
                    raise ValueError(f"cannot isolate collective event in {path}")
                blob = mapped[start + 3 : end + 4].rstrip(b",")
                position = end + 4
                if b'"cat": "kernel"' in blob:
                    yield blob


def parse_traces(
    framework_root: Path,
    manifest_path: Path,
    metadata_path: Path,
    topology: pd.DataFrame,
) -> list[dict[str, object]]:
    metadata = {
        row["relative_path"]: int(row["base_time_ns"])
        for row in read_csv(metadata_path)
    }
    rank_info = topology.set_index("rank").to_dict("index")
    def parse_one(item: dict[str, str]) -> list[dict[str, object]]:
        file_rows: list[dict[str, object]] = []
        rel = item["relative_path"]
        rank = int(item["rank"])
        iteration = int(item["iter"])
        info = rank_info[rank]
        for blob in trace_event_blobs(framework_root / rel):
            event = json.loads(blob)
            args = event.get("args") or {}
            collective = str(args.get("Collective name", ""))
            pg = str(args.get("Process Group Description", ""))
            if (
                collective in EXCLUDED_COLLECTIVES
                or pg in PP_DESCRIPTIONS
                or "PIPELINE" in pg
            ):
                continue
            duration_ns = round(float(event["dur"]) * 1000)
            start_ns = metadata[rel] + round(float(event["ts"]) * 1000)
            dtype = str(args.get("dtype", ""))
            input_nelems = int(args.get("In msg nelems", 0) or 0)
            logical_bytes = input_nelems * DTYPE_BYTES.get(dtype, 1)
            group_size = int(args.get("Group size", 0) or 0)
            expected_tx, expected_rx, definition = expected_trace_bytes(
                collective, logical_bytes, group_size
            )
            file_rows.append(
                {
                    "source": "profiler",
                    "behavior": classify_trace(collective, pg),
                    "iteration": iteration,
                    "rank": rank,
                    "host": info["host"],
                    "gpu_id": int(info["gpu_id"]),
                    "collective": collective,
                    "pg_description": pg,
                    "pg_name": str(args.get("Process Group Name", "")),
                    "group_size": group_size,
                    "start_ns": start_ns,
                    "end_ns": start_ns + duration_ns,
                    "duration_ns": duration_ns,
                    "logical_input_bytes": logical_bytes,
                    "expected_tx_bytes": expected_tx,
                    "expected_rx_bytes": expected_rx,
                    "expected_byte_definition": definition,
                    "dtype": dtype,
                    "external_id": str(args.get("External id", "")),
                    "call_index": "",
                    "source_path": rel,
                    "source_row": "",
                }
            )
        return file_rows

    rows: list[dict[str, object]] = []
    # The corpus is on a rotational disk; concurrent mmaps amplify seeks.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for file_rows in pool.map(parse_one, read_csv(manifest_path)):
            rows.extend(file_rows)
    return rows


def attribute_rank_mtlink(
    events: pd.DataFrame,
    paths: list[Path],
    long_gap_ns: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
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
    # Collector v1.1.1 writes PCIe as special link_id 0xffffffff in the same
    # file.  It is a different physical domain and must not enter MTLink BW.
    samples = samples[
        samples["link_id"].between(0, 13)
        & samples["link_id"].ne(PCIE_LINK_ID)
    ].drop_duplicates().sort_values("host_realtime_ns", kind="stable")
    sample_end = samples["host_realtime_ns"].to_numpy(dtype=np.int64)
    sample_duration = (
        samples["mt_timestamp_end_ns"].to_numpy(dtype=np.int64)
        - samples["mt_timestamp_begin_ns"].to_numpy(dtype=np.int64)
    )
    sample_start = sample_end - sample_duration
    tx = samples["tx_delta_bytes"].to_numpy(dtype=np.uint64)
    rx = samples["rx_delta_bytes"].to_numpy(dtype=np.uint64)
    valid_sample = (
        (sample_duration > 0) & (tx <= MAX_I64) & (rx <= MAX_I64)
    )
    maximum_duration = int(sample_duration[valid_sample].max())
    # Keep rare long intervals from expanding every event's candidate slice.
    # Each bin remains sorted by sample_end because the parent frame is sorted.
    bin_edges = (10_000_000, 25_000_000, 50_000_000, 100_000_000)
    duration_bins: list[tuple[np.ndarray, int]] = []
    lower = 0
    for upper in (*bin_edges, maximum_duration):
        if upper < lower:
            continue
        indices = np.flatnonzero(
            valid_sample
            & (sample_duration > lower)
            & (sample_duration <= upper)
        )
        if len(indices):
            duration_bins.append((indices, int(upper)))
        lower = int(upper)

    pair_samples: list[np.ndarray] = []
    pair_events: list[np.ndarray] = []
    pair_weights: list[np.ndarray] = []
    event_start = events["start_ns"].to_numpy(dtype=np.int64)
    event_end = events["end_ns"].to_numpy(dtype=np.int64)
    for event_index, (start_ns, end_ns) in enumerate(
        zip(event_start, event_end)
    ):
        for bin_indices, upper in duration_bins:
            bin_ends = sample_end[bin_indices]
            left = int(np.searchsorted(bin_ends, start_ns, side="right"))
            right = int(
                np.searchsorted(bin_ends, end_ns + upper, side="right")
            )
            if right <= left:
                continue
            indices = bin_indices[left:right]
            overlap = np.minimum(sample_end[indices], end_ns) - np.maximum(
                sample_start[indices], start_ns
            )
            keep = overlap > 0
            indices = indices[keep]
            if len(indices) == 0:
                continue
            pair_samples.append(indices)
            pair_events.append(
                np.full(len(indices), event_index, dtype=np.int64)
            )
            pair_weights.append(overlap[keep] / sample_duration[indices])

    result = events.copy()
    for column in (
        "mtlink_attributed_tx_bytes",
        "mtlink_attributed_rx_bytes",
        "mtlink_uncertain_tx_bytes",
        "mtlink_uncertain_rx_bytes",
        "mtlink_sample_count",
        "mtlink_observed_link_count",
        "mtlink_normalized_overlap_count",
    ):
        result[column] = 0.0
    if not pair_samples:
        return result, {"pair_count": 0, "max_attribution_sum": 0.0}
    sample_index = np.concatenate(pair_samples)
    event_index = np.concatenate(pair_events)
    raw_weight = np.concatenate(pair_weights)
    raw_sum = np.bincount(
        sample_index, weights=raw_weight, minlength=len(samples)
    )
    normalization = np.maximum(raw_sum[sample_index], 1.0)
    weight = raw_weight / normalization
    count = len(result)
    tx_float = tx[sample_index].astype(np.float64)
    rx_float = rx[sample_index].astype(np.float64)
    long_gap = sample_duration[sample_index] > long_gap_ns
    result["mtlink_attributed_tx_bytes"] = np.bincount(
        event_index, weights=tx_float * weight, minlength=count
    )
    result["mtlink_attributed_rx_bytes"] = np.bincount(
        event_index, weights=rx_float * weight, minlength=count
    )
    result["mtlink_uncertain_tx_bytes"] = np.bincount(
        event_index,
        weights=tx_float * weight * long_gap,
        minlength=count,
    )
    result["mtlink_uncertain_rx_bytes"] = np.bincount(
        event_index,
        weights=rx_float * weight * long_gap,
        minlength=count,
    )
    result["mtlink_sample_count"] = np.bincount(
        event_index, minlength=count
    )
    normalized = normalization > 1.0
    result["mtlink_normalized_overlap_count"] = np.bincount(
        event_index, weights=normalized, minlength=count
    )
    event_link = np.unique(
        event_index * 14
        + samples["link_id"].to_numpy(dtype=np.int64)[sample_index]
    )
    result["mtlink_observed_link_count"] = np.bincount(
        event_link // 14, minlength=count
    )
    attributed_by_sample = np.bincount(
        sample_index, weights=weight, minlength=len(samples)
    )
    return result, {
        "pair_count": int(len(weight)),
        "max_attribution_sum": float(attributed_by_sample.max()),
        "normalized_pair_count": int(np.count_nonzero(normalized)),
    }


def aggregate_cells(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["mtlink_attributed_bidir_bytes"] = (
        frame["mtlink_attributed_tx_bytes"]
        + frame["mtlink_attributed_rx_bytes"]
    )
    grouped = (
        frame.groupby(
            [
                "behavior",
                "source",
                "iteration",
                "rank",
                "host",
                "gpu_id",
                "collective",
                "pg_description",
                "group_size",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            event_count=("duration_ns", "size"),
            duration_ns_sum=("duration_ns", "sum"),
            logical_input_bytes_total=("logical_input_bytes", "sum"),
            expected_tx_bytes_total=("expected_tx_bytes", "sum"),
            expected_rx_bytes_total=("expected_rx_bytes", "sum"),
            mtlink_attributed_tx_bytes_total=(
                "mtlink_attributed_tx_bytes",
                "sum",
            ),
            mtlink_attributed_rx_bytes_total=(
                "mtlink_attributed_rx_bytes",
                "sum",
            ),
            mtlink_attributed_bidir_bytes_total=(
                "mtlink_attributed_bidir_bytes",
                "sum",
            ),
            mtlink_uncertain_tx_bytes_total=("mtlink_uncertain_tx_bytes", "sum"),
            mtlink_uncertain_rx_bytes_total=("mtlink_uncertain_rx_bytes", "sum"),
            mtlink_sample_count=("mtlink_sample_count", "sum"),
            mtlink_observed_link_count_min=("mtlink_observed_link_count", "min"),
            mtlink_normalized_overlap_count=(
                "mtlink_normalized_overlap_count",
                "sum",
            ),
        )
        .sort_values(["behavior", "iteration", "rank"])
    )
    duration = grouped["duration_ns_sum"].replace(0, np.nan)
    grouped["trace_logical_input_bandwidth_GBps"] = (
        grouped["logical_input_bytes_total"] / duration
    )
    grouped["trace_expected_tx_bandwidth_GBps"] = (
        grouped["expected_tx_bytes_total"] / duration
    )
    grouped["trace_expected_rx_bandwidth_GBps"] = (
        grouped["expected_rx_bytes_total"] / duration
    )
    grouped["mtlink_14link_tx_bandwidth_GBps"] = (
        grouped["mtlink_attributed_tx_bytes_total"] / duration
    )
    grouped["mtlink_14link_rx_bandwidth_GBps"] = (
        grouped["mtlink_attributed_rx_bytes_total"] / duration
    )
    grouped["mtlink_14link_bidir_bandwidth_GBps"] = (
        grouped["mtlink_attributed_bidir_bytes_total"] / duration
    )
    grouped["mtlink_long_gap_bidir_pct"] = (
        (
            grouped["mtlink_uncertain_tx_bytes_total"]
            + grouped["mtlink_uncertain_rx_bytes_total"]
        )
        / grouped["mtlink_attributed_bidir_bytes_total"].replace(0, np.nan)
        * 100
    ).fillna(0.0)
    return grouped


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology = pd.read_csv(args.rank_topology)
    tables, arrays = window_maps(args.event_windows)
    pre_mtlink = args.output_dir / "collective_events_no_pp_pre_mtlink.parquet"
    if pre_mtlink.exists():
        events = pd.read_parquet(pre_mtlink)
        ep_count = int(events["source"].eq("deepep").sum())
        trace_count = int(events["source"].eq("profiler").sum())
        if ep_count != args.expected_deepep_events:
            ep_rows = parse_deepep(
                args.framework_root, topology, tables, arrays
            )
            profiler = events[events["source"].eq("profiler")].copy()
            events = pd.concat(
                [pd.DataFrame(ep_rows), profiler.drop(columns="event_id")],
                ignore_index=True,
            ).sort_values(
                ["rank", "start_ns", "end_ns", "behavior"]
            ).reset_index(drop=True)
            events.insert(
                0,
                "event_id",
                [
                    f"collective_{index:07d}"
                    for index in range(len(events))
                ],
            )
            ep_count = len(ep_rows)
            trace_count = len(profiler)
            events.to_parquet(pre_mtlink, index=False)
    else:
        ep_rows = parse_deepep(
            args.framework_root, topology, tables, arrays
        )
        trace_rows = parse_traces(
            args.framework_root,
            args.trace_manifest,
            args.trace_metadata,
            topology,
        )
        ep_count = len(ep_rows)
        trace_count = len(trace_rows)
        events = pd.DataFrame(ep_rows + trace_rows).sort_values(
            ["rank", "start_ns", "end_ns", "behavior"]
        ).reset_index(drop=True)
        events.insert(
            0,
            "event_id",
            [f"collective_{index:07d}" for index in range(len(events))],
        )
        events.to_parquet(pre_mtlink, index=False)

    mt_paths: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for row in read_csv(args.mtlink_manifest):
        mt_paths[(row["host"], int(row["gpu_id"]))].append(
            args.hardware_root / row["path"]
        )
    for paths in mt_paths.values():
        paths.sort()
    attributed = []
    attribution_checks = []
    long_gap_ns = round(args.mtlink_long_gap_ms * 1_000_000)
    for rank, group in events.groupby("rank", sort=True):
        host = str(group["host"].iloc[0])
        gpu_id = int(group["gpu_id"].iloc[0])
        result, check = attribute_rank_mtlink(
            group.reset_index(drop=True),
            mt_paths[(host, gpu_id)],
            long_gap_ns,
        )
        attributed.append(result)
        attribution_checks.append({"rank": int(rank), **check})
    events = pd.concat(attributed, ignore_index=True)
    events["mtlink_attributed_bidir_bytes"] = (
        events["mtlink_attributed_tx_bytes"]
        + events["mtlink_attributed_rx_bytes"]
    )
    events["logical_input_bandwidth_GBps"] = (
        events["logical_input_bytes"] / events["duration_ns"]
    )
    events["expected_tx_bandwidth_GBps"] = (
        events["expected_tx_bytes"] / events["duration_ns"]
    )
    events["expected_rx_bandwidth_GBps"] = (
        events["expected_rx_bytes"] / events["duration_ns"]
    )
    events["mtlink_tx_bandwidth_GBps"] = (
        events["mtlink_attributed_tx_bytes"] / events["duration_ns"]
    )
    events["mtlink_rx_bandwidth_GBps"] = (
        events["mtlink_attributed_rx_bytes"] / events["duration_ns"]
    )
    cells = aggregate_cells(events)
    summary = (
        cells.groupby(
            [
                "behavior",
                "source",
                "collective",
                "pg_description",
                "group_size",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            cell_count=("rank", "size"),
            profiled_iteration_count=("iteration", "nunique"),
            rank_count=("rank", "nunique"),
            event_count=("event_count", "sum"),
            duration_ns_sum=("duration_ns_sum", "sum"),
            logical_input_bytes_total=("logical_input_bytes_total", "sum"),
            expected_tx_bytes_total=("expected_tx_bytes_total", "sum"),
            expected_rx_bytes_total=("expected_rx_bytes_total", "sum"),
            mtlink_attributed_tx_bytes_total=(
                "mtlink_attributed_tx_bytes_total",
                "sum",
            ),
            mtlink_attributed_rx_bytes_total=(
                "mtlink_attributed_rx_bytes_total",
                "sum",
            ),
        )
        .sort_values("behavior")
    )
    duration = summary["duration_ns_sum"].replace(0, np.nan)
    summary["trace_logical_input_bandwidth_GBps"] = (
        summary["logical_input_bytes_total"] / duration
    )
    summary["trace_expected_tx_bandwidth_GBps"] = (
        summary["expected_tx_bytes_total"] / duration
    )
    summary["trace_expected_rx_bandwidth_GBps"] = (
        summary["expected_rx_bytes_total"] / duration
    )
    summary["mtlink_14link_tx_bandwidth_GBps"] = (
        summary["mtlink_attributed_tx_bytes_total"] / duration
    )
    summary["mtlink_14link_rx_bandwidth_GBps"] = (
        summary["mtlink_attributed_rx_bytes_total"] / duration
    )

    events.to_parquet(
        args.output_dir / "collective_event_bandwidth_no_pp.parquet",
        index=False,
    )
    cells.to_csv(
        args.output_dir / "collective_iter_rank_bandwidth_no_pp.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "collective_type_bandwidth_summary_no_pp.csv",
        index=False,
    )
    behavior_counts = Counter(events["behavior"])
    validation = {
        "status": "PASS",
        "pp_included": False,
        "event_count": len(events),
        "deepep_event_count": ep_count,
        "profiler_event_count": trace_count,
        "cell_count": len(cells),
        "behavior_count": len(behavior_counts),
        "behavior_event_counts": dict(sorted(behavior_counts.items())),
        "iterations": sorted(int(x) for x in events["iteration"].unique()),
        "rank_count": int(events["rank"].nunique()),
        "max_mtlink_attribution_sum": max(
            float(row["max_attribution_sum"]) for row in attribution_checks
        ),
        "normalized_mtlink_pair_count": sum(
            int(row["normalized_pair_count"]) for row in attribution_checks
        ),
        "bandwidth_unit": "decimal GB/s; multiply by 8 for Gb/s",
        "cell_bandwidth_formula": "sum(bytes)/sum(event_duration_ns)",
        "outputs": [
            "collective_event_bandwidth_no_pp.parquet",
            "collective_iter_rank_bandwidth_no_pp.csv",
            "collective_type_bandwidth_summary_no_pp.csv",
        ],
    }
    if (
        ep_count != args.expected_deepep_events
        or validation["rank_count"] != args.world_size
        or validation["max_mtlink_attribution_sum"] > 1.0 + 1e-12
        or any(
            "pipeline" in value.lower()
            for value in events["pg_description"].unique()
        )
    ):
        validation["status"] = "FAIL"
    (args.output_dir / "collective_bandwidth_validation_no_pp.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

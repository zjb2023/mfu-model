from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .intervals import Interval, LabeledInterval, allocate_bytes_by_overlap, overlap_ns


_NIC_FILE_RE = re.compile(r"^host_([^_]+)_(mlx5_bond_\d+_port\d+)_")
_MTLINK_FILE_RE = re.compile(r"^host_([^_]+)_(gpu\d+)_")


def _identity(path: Path, pattern: re.Pattern[str], host_map: Mapping[str, str]) -> tuple[str, str, str]:
    match = pattern.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized hardware filename: {path.name}")
    host_id, device = match.groups()
    if host_id not in host_map:
        raise ValueError(f"Missing host mapping for {host_id}")
    return host_map[host_id], host_id, device


def _clip_window(frame: pd.DataFrame, window: tuple[int, int] | None) -> pd.DataFrame:
    if window is None:
        return frame
    start_ns, end_ns = window
    return frame[(frame["end_ns"] > start_ns) & (frame["start_ns"] < end_ns)].copy()


def normalize_nic_frame(
    raw: pd.DataFrame,
    path: Path,
    host_map: Mapping[str, str],
    long_gap_ns: int,
    window: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Normalize cumulative-NIC delta rows into exact previous-to-current intervals."""
    host, host_id, device = _identity(path, _NIC_FILE_RE, host_map)
    required = {"timestamp_ns", "xmit_bytes", "recv_bytes"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"NIC CSV missing columns: {missing}")

    frame = raw.copy()
    frame["end_ns"] = pd.to_numeric(frame["timestamp_ns"], errors="raise").astype("int64")
    frame["start_ns"] = frame["end_ns"].shift(1)
    frame = frame.dropna(subset=["start_ns"]).copy()
    frame["start_ns"] = frame["start_ns"].astype("int64")
    frame["duration_ns"] = frame["end_ns"] - frame["start_ns"]
    if (frame["duration_ns"] <= 0).any():
        raise ValueError(f"NIC timestamps are not strictly increasing in {path}")
    frame["tx_bytes"] = pd.to_numeric(frame["xmit_bytes"], errors="raise").astype("int64")
    frame["rx_bytes"] = pd.to_numeric(frame["recv_bytes"], errors="raise").astype("int64")
    if ((frame["tx_bytes"] < 0) | (frame["rx_bytes"] < 0)).any():
        raise ValueError(f"NIC byte deltas must be non-negative in {path}")
    frame["is_long_gap"] = frame["duration_ns"] > int(long_gap_ns)
    has_bytes = (frame["tx_bytes"] + frame["rx_bytes"]) > 0
    frame["allocation_status"] = "allocatable"
    frame.loc[frame["is_long_gap"] & has_bytes, "allocation_status"] = "unallocatable_long_gap"
    frame["source"] = "nic"
    frame["host"] = host
    frame["host_id"] = host_id
    frame["device"] = device
    frame["provenance"] = "direct_hardware_deepep"
    if "sample_interval_us" in frame:
        frame["reported_sample_interval_ns"] = (
            pd.to_numeric(frame["sample_interval_us"], errors="coerce") * 1000
        ).round().astype("Int64")
    else:
        frame["reported_sample_interval_ns"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    columns = [
        "source",
        "host",
        "host_id",
        "device",
        "start_ns",
        "end_ns",
        "duration_ns",
        "tx_bytes",
        "rx_bytes",
        "is_long_gap",
        "allocation_status",
        "reported_sample_interval_ns",
        "provenance",
    ]
    return _clip_window(frame[columns], window).reset_index(drop=True)


def normalize_mtlink_frame(
    raw: pd.DataFrame,
    path: Path,
    host_map: Mapping[str, str],
    window: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Map MTLink device-duration samples onto host realtime interval ends."""
    host, host_id, device = _identity(path, _MTLINK_FILE_RE, host_map)
    required = {
        "host_realtime_ns",
        "gpu_id",
        "link_id",
        "link_state",
        "ret",
        "tx_delta_bytes",
        "rx_delta_bytes",
        "mt_dt_ns",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"MTLink CSV missing columns: {missing}")

    frame = raw.copy()
    frame["end_ns"] = pd.to_numeric(frame["host_realtime_ns"], errors="raise").astype("int64")
    frame["duration_ns"] = pd.to_numeric(frame["mt_dt_ns"], errors="raise").astype("int64")
    frame["start_ns"] = frame["end_ns"] - frame["duration_ns"]
    if (frame["duration_ns"] <= 0).any():
        raise ValueError(f"MTLink duration must be positive in {path}")
    frame["tx_bytes"] = pd.to_numeric(frame["tx_delta_bytes"], errors="raise").astype("int64")
    frame["rx_bytes"] = pd.to_numeric(frame["rx_delta_bytes"], errors="raise").astype("int64")
    if ((frame["tx_bytes"] < 0) | (frame["rx_bytes"] < 0)).any():
        raise ValueError(f"MTLink byte deltas must be non-negative in {path}")
    frame["link_id"] = pd.to_numeric(frame["link_id"], errors="raise").astype("int16")
    frame["gpu_id"] = pd.to_numeric(frame["gpu_id"], errors="raise").astype("int16")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="raise").astype("int32")
    frame["source"] = "mtlink"
    frame["host"] = host
    frame["host_id"] = host_id
    frame["device"] = device
    frame["allocation_status"] = "allocatable"
    frame.loc[(frame["ret"] != 0) | (frame["link_state"] != "UP"), "allocation_status"] = "invalid_link_sample"
    frame["provenance"] = "direct_hardware_deepep"

    columns = [
        "source",
        "host",
        "host_id",
        "device",
        "gpu_id",
        "link_id",
        "link_state",
        "ret",
        "start_ns",
        "end_ns",
        "duration_ns",
        "tx_bytes",
        "rx_bytes",
        "allocation_status",
        "provenance",
    ]
    return _clip_window(frame[columns], window).reset_index(drop=True)


def _write_parquet_part(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def _clear_existing_parts(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.glob("part-*.parquet"):
        path.unlink()
    for path in directory.glob("part-*.parquet.tmp"):
        path.unlink()


def parse_nic_corpus(
    paths: Iterable[Path],
    output_root: Path,
    host_map: Mapping[str, str],
    long_gap_ns: int,
    window: tuple[int, int] | None,
    chunk_size: int = 250_000,
) -> dict[str, int]:
    """Chunk-parse NIC CSVs while preserving intervals across chunk boundaries."""
    manifest_rows: list[dict[str, object]] = []
    total_rows = total_tx = total_rx = 0
    for path in sorted(paths):
        host, host_id, device = _identity(path, _NIC_FILE_RE, host_map)
        output_dir = output_root / "counter_nic" / f"host={host}" / f"device={device}"
        _clear_existing_parts(output_dir)
        previous: pd.DataFrame | None = None
        part_index = 0
        file_rows = file_tx = file_rx = 0
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            if previous is not None:
                chunk = pd.concat([previous, chunk], ignore_index=True)
            previous = chunk.iloc[[-1]].copy()
            normalized = normalize_nic_frame(
                chunk,
                path=path,
                host_map=host_map,
                long_gap_ns=long_gap_ns,
                window=window,
            )
            if normalized.empty:
                continue
            _write_parquet_part(normalized, output_dir / f"part-{part_index:05d}.parquet")
            part_index += 1
            file_rows += len(normalized)
            file_tx += int(normalized["tx_bytes"].sum())
            file_rx += int(normalized["rx_bytes"].sum())
        manifest_rows.append(
            {
                "source": "nic",
                "host": host,
                "host_id": host_id,
                "device": device,
                "input_path": str(path),
                "row_count": file_rows,
                "tx_bytes": file_tx,
                "rx_bytes": file_rx,
                "part_count": part_index,
            }
        )
        total_rows += file_rows
        total_tx += file_tx
        total_rx += file_rx

    manifest_path = output_root / "nic_corpus_manifest_v5.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return {"row_count": total_rows, "tx_bytes": total_tx, "rx_bytes": total_rx}


def parse_mtlink_corpus(
    paths: Iterable[Path],
    output_root: Path,
    host_map: Mapping[str, str],
    window: tuple[int, int] | None,
    chunk_size: int = 250_000,
) -> dict[str, int]:
    """Chunk-parse MTLink CSVs into host-epoch interval facts."""
    manifest_rows: list[dict[str, object]] = []
    total_rows = total_tx = total_rx = 0
    for path in sorted(paths):
        host, host_id, device = _identity(path, _MTLINK_FILE_RE, host_map)
        output_dir = output_root / "counter_mtlink" / f"host={host}" / f"device={device}"
        _clear_existing_parts(output_dir)
        part_index = 0
        file_rows = file_tx = file_rx = 0
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            normalized = normalize_mtlink_frame(
                chunk,
                path=path,
                host_map=host_map,
                window=window,
            )
            if normalized.empty:
                continue
            _write_parquet_part(normalized, output_dir / f"part-{part_index:05d}.parquet")
            part_index += 1
            file_rows += len(normalized)
            file_tx += int(normalized["tx_bytes"].sum())
            file_rx += int(normalized["rx_bytes"].sum())
        manifest_rows.append(
            {
                "source": "mtlink",
                "host": host,
                "host_id": host_id,
                "device": device,
                "input_path": str(path),
                "row_count": file_rows,
                "tx_bytes": file_tx,
                "rx_bytes": file_rx,
                "part_count": part_index,
            }
        )
        total_rows += file_rows
        total_tx += file_tx
        total_rx += file_rx

    manifest_path = output_root / "mtlink_corpus_manifest_v5.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return {"row_count": total_rows, "tx_bytes": total_tx, "rx_bytes": total_rx}


def allocate_counter_frame_to_iter_windows(
    counters: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Split counter intervals across non-overlapping iter windows with exact bytes."""
    ordered_windows = windows.sort_values("start_ns").reset_index(drop=True)
    if ordered_windows.empty:
        raise ValueError("iter windows must not be empty")
    starts = ordered_windows["start_ns"].to_numpy(dtype=np.int64)
    ends = ordered_windows["end_ns"].to_numpy(dtype=np.int64)
    if np.any(ends <= starts) or np.any(starts[1:] < ends[:-1]):
        raise ValueError("iter windows must be positive and non-overlapping")

    frame = counters.copy().reset_index(drop=True)
    row_starts = frame["start_ns"].to_numpy(dtype=np.int64)
    row_ends = frame["end_ns"].to_numpy(dtype=np.int64)
    indices = np.searchsorted(ends, row_starts, side="right")
    valid = indices < len(ordered_windows)
    safe_indices = np.minimum(indices, len(ordered_windows) - 1)
    contained = np.logical_and.reduce(
        (valid, row_starts >= starts[safe_indices], row_ends <= ends[safe_indices])
    )

    output_frames: list[pd.DataFrame] = []
    if contained.any():
        direct = frame.loc[contained].copy()
        direct_indices = indices[contained]
        direct["iter"] = ordered_windows.iloc[direct_indices]["iter"].to_numpy()
        direct["overlap_ns"] = direct["duration_ns"].astype("int64")
        direct["active_ns"] = np.where(
            (direct["tx_bytes"] + direct["rx_bytes"]) > 0,
            direct["overlap_ns"],
            0,
        )
        direct["unallocated_tx_bytes"] = 0
        direct["unallocated_rx_bytes"] = 0
        output_frames.append(direct)

    for row_index in np.flatnonzero(~contained):
        row = frame.iloc[int(row_index)]
        counter = Interval(int(row.start_ns), int(row.end_ns))
        target_intervals: list[LabeledInterval] = []
        target_ids: list[int] = []
        first = max(0, int(np.searchsorted(ends, counter.start_ns, side="right")))
        last = min(
            len(ordered_windows),
            int(np.searchsorted(starts, counter.end_ns, side="left")) + 1,
        )
        for window_index in range(first, last):
            target = Interval(int(starts[window_index]), int(ends[window_index]))
            if overlap_ns(counter, target) > 0:
                iter_id = int(ordered_windows.iloc[window_index]["iter"])
                target_intervals.append(
                    LabeledInterval(target.start_ns, target.end_ns, str(iter_id))
                )
                target_ids.append(iter_id)
        allocation = allocate_bytes_by_overlap(
            counter,
            tx_bytes=int(row.tx_bytes),
            rx_bytes=int(row.rx_bytes),
            targets=target_intervals,
        )
        emitted = False
        for target, iter_id in zip(target_intervals, target_ids):
            overlap = overlap_ns(counter, target)
            if overlap <= 0:
                continue
            record = row.to_dict()
            record["iter"] = iter_id
            record["tx_bytes"] = allocation.tx_by_label[str(iter_id)]
            record["rx_bytes"] = allocation.rx_by_label[str(iter_id)]
            record["overlap_ns"] = overlap
            record["active_ns"] = overlap if record["tx_bytes"] + record["rx_bytes"] > 0 else 0
            record["unallocated_tx_bytes"] = allocation.unallocated_tx_bytes if not emitted else 0
            record["unallocated_rx_bytes"] = allocation.unallocated_rx_bytes if not emitted else 0
            output_frames.append(pd.DataFrame([record]))
            emitted = True
        if not emitted:
            record = row.to_dict()
            record["iter"] = -1
            record["tx_bytes"] = 0
            record["rx_bytes"] = 0
            record["overlap_ns"] = 0
            record["active_ns"] = 0
            record["unallocated_tx_bytes"] = int(row.tx_bytes)
            record["unallocated_rx_bytes"] = int(row.rx_bytes)
            output_frames.append(pd.DataFrame([record]))

    if not output_frames:
        return pd.DataFrame(columns=[*frame.columns, "iter", "overlap_ns", "active_ns"])
    result = pd.concat(output_frames, ignore_index=True)
    return result.sort_values(["iter", "start_ns", "end_ns"]).reset_index(drop=True)


def aggregate_counter_partitions_by_iter(
    part_paths: Iterable[Path],
    windows: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Aggregate partitioned counter facts by iter while retaining uncertainty bytes."""
    partials: list[pd.DataFrame] = []
    keys = ["iter", *group_columns]
    value_columns = [
        "tx_bytes",
        "rx_bytes",
        "overlap_ns",
        "active_ns",
        "uncertain_tx_bytes",
        "uncertain_rx_bytes",
        "sample_fragments",
    ]
    for path in sorted(part_paths):
        frame = pd.read_parquet(path)
        allocated = allocate_counter_frame_to_iter_windows(frame, windows)
        allocated = allocated[allocated["iter"] >= 0].copy()
        if allocated.empty:
            continue
        uncertain = allocated["allocation_status"] != "allocatable"
        allocated["uncertain_tx_bytes"] = np.where(uncertain, allocated["tx_bytes"], 0) + allocated[
            "unallocated_tx_bytes"
        ]
        allocated["uncertain_rx_bytes"] = np.where(uncertain, allocated["rx_bytes"], 0) + allocated[
            "unallocated_rx_bytes"
        ]
        allocated["sample_fragments"] = 1
        partials.append(
            allocated.groupby(keys, dropna=False)[value_columns].sum().reset_index()
        )
    if not partials:
        return pd.DataFrame(columns=[*keys, *value_columns])
    combined = pd.concat(partials, ignore_index=True)
    return (
        combined.groupby(keys, dropna=False)[value_columns]
        .sum()
        .reset_index()
        .sort_values(keys)
        .reset_index(drop=True)
    )

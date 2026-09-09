#!/usr/bin/env python3
"""Validate NIC/MTLink entity grids and CSV semantics for the 256-GPU run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


MT_RE = re.compile(r"^host_(worker\d+)_gpu(\d+)_.*_seq(\d+)\.csv$")
NIC_RE = re.compile(r"^host_(worker\d+)_(mlx5_\d+)_.*_seq(\d+)\.csv$")
EXPECTED_GPUS = set(range(8))
EXPECTED_NICS = {
    "mlx5_0",
    "mlx5_1",
    "mlx5_2",
    "mlx5_3",
    "mlx5_6",
    "mlx5_7",
    "mlx5_8",
    "mlx5_9",
}
EXPECTED_LINKS = set(range(14))
MAX_SIGNED_INT64 = np.uint64(2**63 - 1)
MT_COLUMNS = [
    "host_realtime_ns",
    "link_id",
    "tx_delta_bytes",
    "rx_delta_bytes",
    "mt_timestamp_begin_ns",
    "mt_timestamp_end_ns",
]
NIC_REQUIRED_COLUMNS = [
    "row",
    "timestamp_ns",
    "timestamp_sec",
    "sample_interval_us",
    "xmit_words",
    "recv_words",
    "xmit_delta_words",
    "recv_delta_words",
    "xmit_bytes",
    "recv_bytes",
    "xmit_gbps",
    "recv_gbps",
]
PCIE_LINK_ID = 2**32 - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank-host-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": float(np.quantile(clean, 0.50)),
        "p95": float(np.quantile(clean, 0.95)),
        "p99": float(np.quantile(clean, 0.99)),
        "max": float(np.max(clean)),
    }


def inspect_mt(path: Path, root: Path) -> dict[str, object]:
    parent_host = path.parent.parent.name
    match = MT_RE.match(path.name)
    if match is None:
        return {"path": path.relative_to(root).as_posix(), "error": "bad_filename"}
    file_host, gpu_text, sequence = match.groups()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != MT_COLUMNS:
        return {
            "path": path.relative_to(root).as_posix(),
            "error": "schema_mismatch",
            "columns": header,
        }
    try:
        frame = pd.read_csv(
            path,
            usecols=MT_COLUMNS,
            dtype={
                "host_realtime_ns": "int64",
                "link_id": "int64",
                "tx_delta_bytes": "uint64",
                "rx_delta_bytes": "uint64",
                "mt_timestamp_begin_ns": "int64",
                "mt_timestamp_end_ns": "int64",
            },
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "path": path.relative_to(root).as_posix(),
            "error": f"numeric_parse_error:{type(exc).__name__}",
        }
    realtime = frame["host_realtime_ns"].to_numpy(dtype=np.int64)
    begin = frame["mt_timestamp_begin_ns"].to_numpy(dtype=np.int64)
    end = frame["mt_timestamp_end_ns"].to_numpy(dtype=np.int64)
    tx = frame["tx_delta_bytes"].to_numpy(dtype=np.uint64)
    rx = frame["rx_delta_bytes"].to_numpy(dtype=np.uint64)
    dt = end - begin
    link_id = frame["link_id"].to_numpy(dtype=np.int64)
    links = set(link_id[link_id != PCIE_LINK_ID].tolist())
    # A reset can be subtracted in unsigned arithmetic by the collector, producing
    # 2^64 - N. Such values previously wrapped to negative int64 in this script.
    # They are invalid delta samples, not negative traffic and not real exabytes.
    tx_underflow = tx > MAX_SIGNED_INT64
    rx_underflow = rx > MAX_SIGNED_INT64
    valid = ~(tx_underflow | rx_underflow)
    return {
        "path": path.relative_to(root).as_posix(),
        "host": file_host,
        "parent_host": parent_host,
        "gpu_id": int(gpu_text),
        "sequence": int(sequence),
        "row_count": int(len(frame)),
        "start_ns": int(realtime.min()) if len(frame) else None,
        "end_ns": int(realtime.max()) if len(frame) else None,
        "strict_time_order": bool(
            frame.groupby("link_id", sort=False)["host_realtime_ns"]
            .apply(lambda values: bool(np.all(np.diff(values.to_numpy()) >= 0)))
            .all()
        ),
        "link_ids": ",".join(map(str, sorted(links))),
        "complete_link_universe": links == EXPECTED_LINKS,
        "pcie_record_count": int(np.count_nonzero(link_id == PCIE_LINK_ID)),
        "invalid_dt_count": int(np.count_nonzero(dt <= 0)),
        "unsigned_underflow_tx_count": int(np.count_nonzero(tx_underflow)),
        "unsigned_underflow_rx_count": int(np.count_nonzero(rx_underflow)),
        "invalid_delta_count": int(np.count_nonzero(~valid)),
        "zero_bidir_fraction": (
            float(np.mean((tx[valid] == 0) & (rx[valid] == 0)))
            if np.any(valid)
            else None
        ),
        "sample_dt_ns": quantiles(dt.astype(np.float64)),
        "error": "",
    }


def inspect_nic(path: Path, root: Path) -> dict[str, object]:
    parent_host = path.parent.parent.name
    match = NIC_RE.match(path.name)
    if match is None:
        return {"path": path.relative_to(root).as_posix(), "error": "bad_filename"}
    file_host, device, sequence = match.groups()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if not set(NIC_REQUIRED_COLUMNS).issubset(header):
        return {
            "path": path.relative_to(root).as_posix(),
            "error": "schema_mismatch",
            "columns": header,
            "missing_columns": sorted(set(NIC_REQUIRED_COLUMNS) - set(header)),
        }
    usecols = [
        "timestamp_ns",
        "sample_interval_us",
        "xmit_delta_words",
        "recv_delta_words",
        "xmit_bytes",
        "recv_bytes",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    timestamp = frame["timestamp_ns"].to_numpy(dtype=np.int64)
    interval_us = frame["sample_interval_us"].to_numpy(dtype=np.float64)
    xmit_words = frame["xmit_delta_words"].to_numpy(dtype=np.int64)
    recv_words = frame["recv_delta_words"].to_numpy(dtype=np.int64)
    xmit_bytes = frame["xmit_bytes"].to_numpy(dtype=np.int64)
    recv_bytes = frame["recv_bytes"].to_numpy(dtype=np.int64)
    return {
        "path": path.relative_to(root).as_posix(),
        "host": file_host,
        "parent_host": parent_host,
        "device": device,
        "sequence": int(sequence),
        "row_count": int(len(frame)),
        "start_ns": int(timestamp.min()) if len(frame) else None,
        "end_ns": int(timestamp.max()) if len(frame) else None,
        "strict_time_order": bool(np.all(np.diff(timestamp) > 0)),
        "negative_xmit_count": int(np.count_nonzero(xmit_bytes < 0)),
        "negative_recv_count": int(np.count_nonzero(recv_bytes < 0)),
        "xmit_word_byte_mismatch": int(
            np.count_nonzero(xmit_words * 4 != xmit_bytes)
        ),
        "recv_word_byte_mismatch": int(
            np.count_nonzero(recv_words * 4 != recv_bytes)
        ),
        "zero_bidir_fraction": float(
            np.mean((xmit_bytes == 0) & (recv_bytes == 0))
        ),
        "sample_interval_us": quantiles(interval_us),
        "long_gap_over_10ms_fraction": float(np.mean(interval_us > 10_000.0)),
        "error": "",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    flattened: list[dict[str, object]] = []
    for row in rows:
        copy = dict(row)
        for key in ("sample_dt_ns", "sample_interval_us"):
            nested = copy.pop(key, None)
            if isinstance(nested, dict):
                for nested_key, value in nested.items():
                    copy[f"{key}_{nested_key}"] = value
        flattened.append(copy)
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"hardware root is not a directory: {root}")
    with args.rank_host_map.open(newline="", encoding="utf-8") as handle:
        expected_hosts = {
            row["host"] for row in csv.DictReader(handle) if row.get("host")
        }

    mt_paths = sorted(root.glob("worker*/mtlink/*.csv"))
    nic_paths = sorted(root.glob("worker*/nic/*.csv"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        mt_rows = list(pool.map(lambda path: inspect_mt(path, root), mt_paths))
        nic_rows = list(pool.map(lambda path: inspect_nic(path, root), nic_paths))

    blockers: list[str] = []
    warnings: list[str] = []
    if len(mt_rows) < len(expected_hosts) * 8:
        blockers.append("mtlink_file_count")
    if len(nic_rows) < len(expected_hosts) * 8:
        blockers.append("nic_file_count")
    if any(row.get("error") for row in mt_rows):
        blockers.append("mtlink_schema_or_filename")
    if any(row.get("error") for row in nic_rows):
        blockers.append("nic_schema_or_filename")

    mt_entities = {
        (str(row.get("host")), int(row.get("gpu_id", -1)))
        for row in mt_rows
        if not row.get("error")
    }
    nic_entities = {
        (str(row.get("host")), str(row.get("device")))
        for row in nic_rows
        if not row.get("error")
    }
    expected_mt = {(host, gpu) for host in expected_hosts for gpu in EXPECTED_GPUS}
    expected_nic = {
        (host, device) for host in expected_hosts for device in EXPECTED_NICS
    }
    if mt_entities != expected_mt:
        blockers.append("mtlink_entity_grid")
    if nic_entities != expected_nic:
        blockers.append("nic_entity_grid")

    for name, rows, entity_column in (
        ("mtlink", mt_rows, "gpu_id"),
        ("nic", nic_rows, "device"),
    ):
        sequence_map: dict[tuple[str, object], list[int]] = {}
        for row in rows:
            if row.get("error"):
                continue
            key = (str(row["host"]), row[entity_column])
            sequence_map.setdefault(key, []).append(int(row["sequence"]))
        if any(
            sorted(values) != list(range(max(values) + 1))
            for values in sequence_map.values()
        ):
            blockers.append(f"{name}_sequence_grid")

    for name, rows in (("mtlink", mt_rows), ("nic", nic_rows)):
        if any(row.get("host") != row.get("parent_host") for row in rows):
            blockers.append(f"{name}_host_identity")
        if any(not row.get("strict_time_order", False) for row in rows):
            blockers.append(f"{name}_time_order")
    entity_links: dict[tuple[str, int], set[int]] = {}
    for row in mt_rows:
        if row.get("error"):
            continue
        key = (str(row["host"]), int(row["gpu_id"]))
        observed = {
            int(value)
            for value in str(row.get("link_ids", "")).split(",")
            if value
        }
        entity_links.setdefault(key, set()).update(observed)
    if any(links != EXPECTED_LINKS for links in entity_links.values()):
        blockers.append("mtlink_entity_link_universe")
    if any(int(row.get("invalid_dt_count", 0)) for row in mt_rows):
        blockers.append("mtlink_invalid_dt")
    if any(int(row.get("invalid_delta_count", 0)) for row in mt_rows):
        warnings.append("mtlink_unsigned_underflow_reset_samples")
    if any(
        int(row.get(key, 0))
        for row in nic_rows
        for key in (
            "negative_xmit_count",
            "negative_recv_count",
            "xmit_word_byte_mismatch",
            "recv_word_byte_mismatch",
        )
    ):
        blockers.append("nic_counter_semantics")
    if any(float(row.get("long_gap_over_10ms_fraction", 0.0)) > 0 for row in nic_rows):
        warnings.append("nic_sparse_or_long_gap_emission")

    mt_starts = [int(row["start_ns"]) for row in mt_rows if row.get("start_ns")]
    mt_ends = [int(row["end_ns"]) for row in mt_rows if row.get("end_ns")]
    nic_starts = [int(row["start_ns"]) for row in nic_rows if row.get("start_ns")]
    nic_ends = [int(row["end_ns"]) for row in nic_rows if row.get("end_ns")]
    report = {
        "status": "PASS_ENTITY_SCHEMA" if not blockers else "BLOCKED",
        "capability": "hardware_file_and_schema",
        "root": str(root),
        "expected_host_count": len(expected_hosts),
        "mtlink": {
            "file_count": len(mt_rows),
            "entity_count": len(mt_entities),
            "multi_sequence_entity_count": len(mt_rows) - len(mt_entities),
            "pcie_record_count_excluded_from_mtlink": sum(
                int(row.get("pcie_record_count", 0)) for row in mt_rows
            ),
            "start_ns_min": min(mt_starts) if mt_starts else None,
            "end_ns_max": max(mt_ends) if mt_ends else None,
            "row_count": sum(int(row.get("row_count", 0)) for row in mt_rows),
            "minimum_zero_bidir_fraction": min(
                float(row.get("zero_bidir_fraction", 0.0)) for row in mt_rows
            ),
        },
        "nic": {
            "file_count": len(nic_rows),
            "entity_count": len(nic_entities),
            "multi_sequence_entity_count": len(nic_rows) - len(nic_entities),
            "start_ns_min": min(nic_starts) if nic_starts else None,
            "end_ns_max": max(nic_ends) if nic_ends else None,
            "row_count": sum(int(row.get("row_count", 0)) for row in nic_rows),
            "minimum_zero_bidir_fraction": min(
                float(row.get("zero_bidir_fraction", 0.0)) for row in nic_rows
            ),
            "maximum_long_gap_over_10ms_fraction": max(
                float(row.get("long_gap_over_10ms_fraction", 0.0))
                for row in nic_rows
            ),
        },
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "limitations": [
            "Event-window coverage is not yet evaluated.",
            "Sparse emission reconstruction is not performed at inventory stage.",
            "Rank-to-NIC uses the separately documented GPU-slot mapping assumption.",
            "MTLink unsigned-underflow/reset rows must be excluded as NA.",
            "MTLink link_id=0xffffffff is PCIe and is excluded from MTLink totals.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mtlink_files.csv", mt_rows)
    write_csv(args.output_dir / "nic_files.csv", nic_rows)
    (args.output_dir / "hardware_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())

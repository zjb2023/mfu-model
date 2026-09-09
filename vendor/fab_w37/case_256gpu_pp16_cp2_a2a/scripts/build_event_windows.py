#!/usr/bin/env python3
"""Extract ProfilerStep windows and validate per-entity hardware coverage."""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import re
from pathlib import Path


STEP_RE = re.compile(
    rb'"name":\s*"ProfilerStep#(\d+)".*?'
    rb'"ts":\s*([0-9.]+),\s*"dur":\s*([0-9.]+)',
    re.DOTALL,
)
STEP_MARKER = b'"name": "ProfilerStep#'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trace-metadata", type=Path, required=True)
    parser.add_argument("--rank-topology", type=Path, required=True)
    parser.add_argument("--mtlink-files", type=Path, required=True)
    parser.add_argument("--nic-files", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    topology = {
        int(row["rank"]): row for row in read_csv(args.rank_topology)
    }
    mt_ranges: dict[tuple[str, int], tuple[int, int]] = {}
    for row in read_csv(args.mtlink_files):
        if row.get("error"):
            continue
        key = (row["host"], int(row["gpu_id"]))
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])
        previous = mt_ranges.get(key)
        mt_ranges[key] = (
            min(start_ns, previous[0]) if previous else start_ns,
            max(end_ns, previous[1]) if previous else end_ns,
        )
    nic_ranges: dict[str, list[tuple[int, int]]] = {}
    for row in read_csv(args.nic_files):
        if not row.get("error"):
            nic_ranges.setdefault(row["host"], []).append(
                (int(row["start_ns"]), int(row["end_ns"]))
            )

    output_rows = []
    errors: dict[str, int] = {}
    for meta in read_csv(args.trace_metadata):
        rank = int(meta["path_rank"])
        iteration = int(meta["iteration"])
        host = meta["host"]
        gpu_id = int(topology[rank]["gpu_id"])
        with (root / meta["relative_path"]).open("rb") as handle:
            with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
                marker_offset = mapped.find(STEP_MARKER)
                if marker_offset < 0:
                    event_bytes = b""
                else:
                    event_start = max(0, mapped.rfind(b"{", 0, marker_offset))
                    event_bytes = mapped[event_start : marker_offset + 1024]
        match = STEP_RE.search(event_bytes)
        error = ""
        if match is None:
            error = "missing_profiler_step"
            step = None
            start_ns = None
            end_ns = None
        else:
            step = int(match.group(1))
            ts_us = float(match.group(2))
            dur_us = float(match.group(3))
            start_ns = int(meta["base_time_ns"]) + round(ts_us * 1000)
            end_ns = start_ns + round(dur_us * 1000)
            if step + 1 != iteration:
                error = "iteration_step_mismatch"
        mt_range = mt_ranges.get((host, gpu_id))
        mt_covered = bool(
            start_ns is not None
            and mt_range is not None
            and mt_range[0] <= start_ns
            and mt_range[1] >= end_ns
        )
        host_nic_ranges = nic_ranges.get(host, [])
        nic_devices_covering = (
            sum(
                range_start <= start_ns and range_end >= end_ns
                for range_start, range_end in host_nic_ranges
            )
            if start_ns is not None
            else 0
        )
        if not error and not mt_covered:
            error = "mtlink_window_not_covered"
        if not error and nic_devices_covering != 8:
            error = "nic_window_not_covered_by_all_devices"
        if error:
            errors[error] = errors.get(error, 0) + 1
        output_rows.append(
            {
                "iteration": iteration,
                "profiler_step": step,
                "rank": rank,
                "host": host,
                "gpu_id": gpu_id,
                "pp_stage": topology[rank]["pp_stage"],
                "pp_lane": topology[rank]["pp_lane"],
                "cp_group": topology[rank]["cp_group"],
                "cp_rank": topology[rank]["cp_rank"],
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": (
                    end_ns - start_ns
                    if start_ns is not None and end_ns is not None
                    else None
                ),
                "mtlink_covered": mt_covered,
                "nic_devices_covering": nic_devices_covering,
                "error": error,
                "relative_path": meta["relative_path"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "event_windows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "status": "PASS_EVENT_WINDOW_COVERAGE" if not errors else "BLOCKED",
        "event_window_count": len(output_rows),
        "mtlink_covered_count": sum(
            bool(row["mtlink_covered"]) for row in output_rows
        ),
        "nic_all_devices_covered_count": sum(
            int(row["nic_devices_covering"]) == 8 for row in output_rows
        ),
        "event_start_ns_min": min(
            int(row["start_ns"])
            for row in output_rows
            if row["start_ns"] is not None
        ),
        "event_end_ns_max": max(
            int(row["end_ns"])
            for row in output_rows
            if row["end_ns"] is not None
        ),
        "errors": errors,
        "limitations": [
            "Coverage means file time range contains the event; sample overlap "
            "and gap-quality metrics are computed during aggregation.",
            "NIC coverage is host/device level; rank-to-NIC rail mapping is unknown.",
        ],
    }
    (args.output_dir / "event_window_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

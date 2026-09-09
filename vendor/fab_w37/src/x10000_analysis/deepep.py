from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeepEPParseResult:
    events: pd.DataFrame
    iter_windows: pd.DataFrame


def _milliseconds_to_ns(value: object) -> int:
    decimal = Decimal(str(value))
    return int((decimal * 1_000_000).to_integral_value(rounding=ROUND_HALF_EVEN))


def _log_identity(path: Path) -> tuple[str, int]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 2 or not parts[0].startswith("r") or not parts[1].startswith("rank"):
        raise ValueError(f"Unrecognized DeepEP log filename: {path.name}")
    return parts[0], int(parts[1].removeprefix("rank"))


def _stage_hint(op: str) -> str:
    return {
        "dispatch": "deepep_forward_scaleup",
        "combine": "deepep_forward_scaleup",
        "combine_backward_dispatch": "deepep_backward_scaleup",
        "dispatch_backward_combine": "deepep_backward_scaleup",
    }.get(op, "other_communication")


def parse_deepep_logs(
    paths: Iterable[Path],
    host_ids: dict[str, tuple[str, str]],
    host_split_rank: int,
    calls_per_training_iter: int,
    analysis_iters: tuple[int, ...],
) -> DeepEPParseResult:
    """Parse DeepEP JSONL logs into exact active intervals and training-iter windows."""
    all_rows: list[dict[str, object]] = []
    group_starts_by_file: dict[str, list[tuple[int, int]]] = {}

    for path in sorted(paths):
        host_side, global_rank = _log_identity(path)
        if host_side not in host_ids:
            raise ValueError(f"Missing host mapping for {host_side}")
        host, host_id = host_ids[host_side]
        local_rank = global_rank if host == "A" else global_rank - host_split_rank
        if local_rank < 0 or local_rank >= host_split_rank:
            raise ValueError(
                f"DeepEP rank {global_rank} is inconsistent with host side {host_side}"
            )
        raw_rows: list[dict[str, object]] = []
        with path.open() as fh:
            for line_number, line in enumerate(fh, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("type") != "bw" or not row.get("op"):
                    continue
                if "timestamp_ns" not in row or "elapsed_ms" not in row:
                    continue
                row["_line_number"] = line_number
                raw_rows.append(row)

        dispatch = sorted(
            (
                int(row["iter"]),
                int(row["timestamp_ns"]) - _milliseconds_to_ns(row["elapsed_ms"]),
                int(row["timestamp_ns"]),
            )
            for row in raw_rows
            if row.get("op") == "dispatch" and row.get("iter") is not None
        )
        if not dispatch:
            raise ValueError(f"No dispatch events with iter in {path}")
        dispatch_call_iters = np.array([item[0] for item in dispatch], dtype=np.int64)
        dispatch_end_ns = np.array([item[2] for item in dispatch], dtype=np.int64)
        expected_calls = np.arange(1, len(dispatch) + 1, dtype=np.int64)
        if not np.array_equal(dispatch_call_iters, expected_calls):
            raise ValueError(f"Non-contiguous DeepEP dispatch iter sequence in {path}")

        group_starts: list[tuple[int, int]] = []
        for call_iter, start_ns, _ in dispatch:
            if (call_iter - 1) % calls_per_training_iter == 0:
                training_iter = (call_iter - 1) // calls_per_training_iter + 1
                group_starts.append((training_iter, start_ns))
        group_starts_by_file[str(path)] = group_starts
        group_start_ns = np.array([item[1] for item in group_starts], dtype=np.int64)

        for row in raw_rows:
            end_ns = int(row["timestamp_ns"])
            start_ns = end_ns - _milliseconds_to_ns(row["elapsed_ms"])
            group_index = int(np.searchsorted(group_start_ns, end_ns, side="right") - 1)
            if group_index < 0:
                raise ValueError(f"DeepEP event precedes first training iter in {path}")
            training_iter = group_starts[group_index][0]
            if training_iter not in analysis_iters:
                continue
            dispatch_index = int(np.searchsorted(dispatch_end_ns, end_ns, side="right") - 1)
            call_iter = int(dispatch_call_iters[max(dispatch_index, 0)])
            op = str(row["op"])
            all_rows.append(
                {
                    "iter": training_iter,
                    "call_iter": call_iter,
                    "rank": global_rank,
                    "host": host,
                    "host_id": host_id,
                    "local_rank": local_rank,
                    "source": "deepep",
                    "event_type": "deepep_op",
                    "stage_hint": _stage_hint(op),
                    "op": op,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": end_ns - start_ns,
                    "send_bytes": int(row.get("send_bytes", 0) or 0),
                    "recv_bytes": int(row.get("recv_bytes", 0) or 0),
                    "comm_bytes": int(row.get("comm_bytes", 0) or 0),
                    "logged_bw_field_value": row.get("bw_gbps"),
                    "logged_bw_field_unit": "misnamed_2x_bytes_per_second_decimal_GBps_like",
                    "instrumented": True,
                    "device_synchronize": True,
                    "overlap_config": False,
                    "trace_path": str(path),
                    "line_number": int(row["_line_number"]),
                    "provenance": "direct_hardware_deepep",
                }
            )

    if not all_rows:
        raise ValueError("No DeepEP events in requested analysis_iters")
    events = pd.DataFrame(all_rows).sort_values(["iter", "rank", "start_ns", "end_ns"]).reset_index(drop=True)

    global_starts: dict[int, int] = {}
    for starts in group_starts_by_file.values():
        for iter_id, start_ns in starts:
            global_starts[iter_id] = min(global_starts.get(iter_id, start_ns), start_ns)
    event_max_end = int(events["end_ns"].max())
    window_rows: list[dict[str, int | str]] = []
    for iter_id in analysis_iters:
        if iter_id not in global_starts:
            raise ValueError(f"Missing DeepEP group start for training iter {iter_id}")
        start_ns = global_starts[iter_id]
        end_ns = global_starts.get(iter_id + 1, event_max_end)
        if end_ns <= start_ns:
            raise ValueError(f"Invalid DeepEP iter window for iter {iter_id}")
        window_rows.append(
            {
                "iter": iter_id,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "wall_ns": end_ns - start_ns,
                "provenance": "direct_hardware_deepep",
            }
        )
    iter_windows = pd.DataFrame(window_rows)
    return DeepEPParseResult(events=events, iter_windows=iter_windows)

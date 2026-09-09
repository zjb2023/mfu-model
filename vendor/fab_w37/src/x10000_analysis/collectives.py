from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd


class CallAlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class CallAlignmentResult:
    calls: pd.DataFrame
    per_rank: pd.DataFrame
    failures: pd.DataFrame


def _parse_ranks(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return tuple(int(rank) for rank in parsed)


def _canonical_ranks(ranks: tuple[int, ...] | list[int]) -> str:
    return json.dumps([int(rank) for rank in ranks], separators=(", ", ": "))


def _pg_map(pg_config: pd.DataFrame) -> dict[tuple[int, str], tuple[int, ...]]:
    mapping: dict[tuple[int, str], tuple[int, ...]] = {}
    for row in pg_config.itertuples(index=False):
        key = (int(row.iter), str(row.pg_name))
        ranks = _parse_ranks(row.global_ranks)
        if key in mapping and mapping[key] != ranks:
            raise CallAlignmentError(
                f"conflicting pg_config for iter={key[0]}, pg={key[1]}: "
                f"{mapping[key]} vs {ranks}"
            )
        mapping[key] = ranks
    return mapping


def align_collective_calls(
    events: pd.DataFrame,
    pg_config: pd.DataFrame,
    fail_closed: bool = True,
) -> CallAlignmentResult:
    """Align per-rank MCCL kernels into global calls using PG topology and occurrence."""
    required = {
        "iter",
        "rank",
        "host",
        "local_rank",
        "event_type",
        "collective",
        "pg_name",
        "size_bytes",
        "start_ns",
        "end_ns",
        "duration_ns",
        "algo",
        "protocol",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"collective events missing columns: {missing}")

    mccl = events[events["event_type"] == "mccl_kernel"].copy()
    if mccl.empty:
        raise ValueError("no mccl_kernel events to align")
    mccl["pg_name"] = mccl["pg_name"].astype(str)
    mccl["collective"] = mccl["collective"].astype(str)
    mccl = mccl.sort_values(
        ["iter", "rank", "pg_name", "collective", "size_bytes", "start_ns", "end_ns"]
    ).reset_index(drop=True)
    occurrence_group = ["iter", "rank", "pg_name", "collective", "size_bytes"]
    mccl["occurrence_index"] = mccl.groupby(occurrence_group, dropna=False).cumcount()

    topology = _pg_map(pg_config)
    call_group = ["iter", "pg_name", "collective", "size_bytes", "occurrence_index"]
    call_rows: list[dict[str, object]] = []
    per_rank_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for key, group in mccl.groupby(call_group, sort=False, dropna=False):
        iter_id, pg_name, collective, size_bytes, occurrence_index = key
        pg_key = (int(iter_id), str(pg_name))
        expected = topology.get(pg_key)
        observed = tuple(sorted(int(rank) for rank in group["rank"].tolist()))
        reason: str | None = None
        if expected is None:
            reason = "missing pg_config"
        elif len(observed) != len(set(observed)):
            reason = "duplicate observed rank"
        elif set(observed) != set(expected):
            reason = f"observed ranks {observed} do not match expected ranks {expected}"
        elif group["algo"].nunique(dropna=False) != 1:
            reason = "inconsistent algorithm"
        elif group["protocol"].nunique(dropna=False) != 1:
            reason = "inconsistent protocol"
        elif (group["end_ns"] <= group["start_ns"]).any():
            reason = "non-positive event interval"

        if reason is not None:
            failure_rows.append(
                {
                    "iter": int(iter_id),
                    "pg_name": str(pg_name),
                    "collective": str(collective),
                    "size_bytes": int(size_bytes),
                    "occurrence_index": int(occurrence_index),
                    "expected_ranks": _canonical_ranks(expected or ()),
                    "observed_ranks": _canonical_ranks(observed),
                    "reason": reason,
                }
            )
            continue

        earliest_start = int(group["start_ns"].min())
        latest_start = int(group["start_ns"].max())
        earliest_end = int(group["end_ns"].min())
        latest_end = int(group["end_ns"].max())
        call_id = (
            f"iter{int(iter_id)}:pg{pg_name}:{collective}:"
            f"size{int(size_bytes)}:occ{int(occurrence_index)}"
        )
        call_rows.append(
            {
                "call_id": call_id,
                "iter": int(iter_id),
                "pg_name": str(pg_name),
                "pg_description": group["pg_description"].dropna().iloc[0]
                if "pg_description" in group and group["pg_description"].notna().any()
                else None,
                "collective": str(collective),
                "size_bytes": int(size_bytes),
                "occurrence_index": int(occurrence_index),
                "expected_ranks": _canonical_ranks(expected or ()),
                "rank_count": len(group),
                "algo": group["algo"].iloc[0],
                "protocol": group["protocol"].iloc[0],
                "earliest_start_ns": earliest_start,
                "latest_start_ns": latest_start,
                "earliest_end_ns": earliest_end,
                "latest_end_ns": latest_end,
                "arrival_skew_ns": latest_start - earliest_start,
                "service_after_last_arrival_ns": latest_end - latest_start,
                "finish_spread_ns": latest_end - earliest_end,
                "critical_exposure_ns": latest_end - earliest_start,
                "provenance": "direct_profiled",
            }
        )
        for row in group.itertuples(index=False):
            per_rank_rows.append(
                {
                    "call_id": call_id,
                    "iter": int(iter_id),
                    "pg_name": str(pg_name),
                    "collective": str(collective),
                    "size_bytes": int(size_bytes),
                    "occurrence_index": int(occurrence_index),
                    "rank": int(row.rank),
                    "host": row.host,
                    "local_rank": int(row.local_rank),
                    "start_ns": int(row.start_ns),
                    "end_ns": int(row.end_ns),
                    "duration_ns": int(row.duration_ns),
                    "arrival_offset_ns": int(row.start_ns) - earliest_start,
                    "finish_offset_ns": int(row.end_ns) - earliest_end,
                    "trace_path": getattr(row, "trace_path", None),
                    "provenance": "direct_profiled",
                }
            )

    calls = pd.DataFrame(call_rows)
    if not calls.empty:
        calls = calls.sort_values(["iter", "earliest_start_ns", "pg_name", "occurrence_index"]).reset_index(drop=True)
    per_rank = pd.DataFrame(per_rank_rows)
    if not per_rank.empty:
        per_rank = per_rank.sort_values(["iter", "start_ns", "pg_name", "rank"]).reset_index(drop=True)
    failures = pd.DataFrame(failure_rows)
    if fail_closed and not failures.empty:
        first = failures.iloc[0]
        raise CallAlignmentError(
            f"collective call alignment failed: iter={first['iter']} pg={first['pg_name']} "
            f"collective={first['collective']}: {first['reason']}"
        )
    return CallAlignmentResult(calls=calls, per_rank=per_rank, failures=failures)

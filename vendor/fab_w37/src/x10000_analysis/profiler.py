from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig


DTYPE_BYTES = {
    "Int": 4,
    "Float": 4,
    "BFloat16": 2,
    "Int8": 1,
    "int8": 1,
    "Half": 2,
    "Double": 8,
}


@dataclass(frozen=True)
class ParsedTrace:
    events: pd.DataFrame
    pg_config: pd.DataFrame
    metadata: dict[str, Any]


def _microseconds_to_ns(value: object) -> int:
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    return int((decimal * 1000).to_integral_value(rounding=ROUND_HALF_EVEN))


def _canonical_list(value: object) -> str:
    if value is None or value == "":
        return "[]"
    if isinstance(value, str):
        return value
    return json.dumps([int(x) for x in value], separators=(", ", ": "))


def _host_of_rank(rank: int, host_split_rank: int) -> str:
    return "A" if rank < host_split_rank else "B"


def _event_type(category: str, name: str) -> str:
    if category == "kernel":
        return "mccl_kernel" if "mcclKernel_" in name else "compute_kernel"
    if category == "gpu_user_annotation":
        return "gpu_annotation"
    if category == "user_annotation":
        return "user_annotation"
    if category in {"privateuse1_runtime", "cuda_runtime", "musa_runtime"}:
        return "runtime_sync" if "Synchronize" in name else "runtime"
    if category == "cpu_op":
        return "cpu_op"
    return "other"


def _algo_protocol(name: str) -> tuple[str | None, str | None]:
    if "mcclKernel_" not in name:
        return None, None
    algo = "RING" if "_RING_" in name else "TREE" if "_TREE_" in name else "OTHER"
    if "_SIMPLE_" in name:
        protocol = "SIMPLE"
    elif "_LL128_" in name:
        protocol = "LL128"
    elif "_LL_" in name:
        protocol = "LL"
    else:
        protocol = "OTHER"
    return algo, protocol


def parse_trace_file(
    path: Path,
    iter_id: int,
    expected_rank: int,
    host_split_rank: int,
) -> ParsedTrace:
    """Parse one profiler trace into event and process-group fact tables."""
    with path.open() as fh:
        doc = json.load(fh, parse_float=Decimal)

    distributed = doc.get("distributedInfo") or {}
    rank = int(distributed.get("rank", -1))
    if rank != expected_rank:
        raise ValueError(
            f"rank mismatch for {path}: expected={expected_rank}, distributedInfo.rank={rank}"
        )
    base_time_ns = int(doc.get("baseTimeNanoseconds", 0))
    if base_time_ns <= 0:
        raise ValueError(f"missing baseTimeNanoseconds in {path}")

    host = _host_of_rank(rank, host_split_rank)
    local_rank = rank if host == "A" else rank - host_split_rank
    event_rows: list[dict[str, Any]] = []
    for event in doc.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        duration_us = event.get("dur", 0)
        if duration_us is None or Decimal(str(duration_us)) <= 0:
            continue
        category = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        args = event.get("args") or {}
        start_ns = base_time_ns + _microseconds_to_ns(event.get("ts", 0))
        duration_ns = _microseconds_to_ns(duration_us)
        dtype = str(args.get("dtype", ""))
        dtype_bytes = DTYPE_BYTES.get(dtype, 4)
        in_nelems = int(args.get("In msg nelems", 0) or 0)
        algo, protocol = _algo_protocol(name)
        stream = args.get("stream", event.get("tid") if category in {"kernel", "gpu_user_annotation"} else None)
        event_rows.append(
            {
                "iter": int(iter_id),
                "rank": rank,
                "host": host,
                "local_rank": local_rank,
                "source": "profiler",
                "event_type": _event_type(category, name),
                "category": category,
                "name": name,
                "start_ns": start_ns,
                "end_ns": start_ns + duration_ns,
                "duration_ns": duration_ns,
                "pid": str(event.get("pid", "")),
                "tid": str(event.get("tid", "")),
                "stream": stream,
                "external_id": args.get("External id"),
                "collective": args.get("Collective name"),
                "pg_name": str(args.get("Process Group Name", "")),
                "pg_description": args.get("Process Group Description"),
                "pg_ranks_event": _canonical_list(args.get("Process Group Ranks")),
                "group_size": int(args.get("Group size", 0) or 0),
                "dtype": dtype,
                "dtype_bytes": dtype_bytes,
                "in_msg_nelems": in_nelems,
                "out_msg_nelems": int(args.get("Out msg nelems", 0) or 0),
                "size_bytes": in_nelems * dtype_bytes,
                "algo": algo,
                "protocol": protocol,
                "trace_path": str(path),
                "provenance": "direct_profiled",
            }
        )

    pg_rows: list[dict[str, Any]] = []
    for group in distributed.get("pg_config", []):
        ranks = [int(x) for x in group.get("ranks", [])]
        hosts = sorted({_host_of_rank(value, host_split_rank) for value in ranks})
        pg_rows.append(
            {
                "iter": int(iter_id),
                "reporting_rank": rank,
                "pg_name": str(group.get("pg_name", "")),
                "pg_description": group.get("pg_desc"),
                "backend_config": group.get("backend_config"),
                "pg_size": int(group.get("pg_size", len(ranks))),
                "global_ranks": _canonical_list(ranks),
                "global_rank_hosts": "".join(hosts),
                "cross_host": len(hosts) > 1,
                "trace_path": str(path),
            }
        )

    events = pd.DataFrame(event_rows)
    pg_config = pd.DataFrame(pg_rows)
    metadata = {
        "iter": int(iter_id),
        "rank": rank,
        "base_time_ns": base_time_ns,
        "event_count": len(events),
        "pg_count": len(pg_config),
        "trace_path": str(path),
    }
    return ParsedTrace(events=events, pg_config=pg_config, metadata=metadata)


def write_parsed_trace(parsed: ParsedTrace, output_root: Path) -> dict[str, Path]:
    """Write one parsed trace into iter/rank partition files."""
    iter_id = int(parsed.metadata["iter"])
    rank = int(parsed.metadata["rank"])
    paths = {
        "events": output_root / "events_profiler" / f"iter={iter_id}" / f"rank={rank}.parquet",
        "pg_config": output_root / "pg_config_profiler" / f"iter={iter_id}" / f"rank={rank}.parquet",
        "metadata": output_root / "metadata_profiler" / f"iter={iter_id}" / f"rank={rank}.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    for key, frame in (("events", parsed.events), ("pg_config", parsed.pg_config)):
        path = paths[key]
        temp = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temp, index=False)
        temp.replace(path)

    metadata_path = paths["metadata"]
    metadata_temp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temp.write_text(json.dumps(parsed.metadata, ensure_ascii=False, indent=2) + "\n")
    metadata_temp.replace(metadata_path)
    return paths


def parse_profiler_corpus(cfg: RunConfig, output_root: Path) -> dict[str, int]:
    """Parse configured profiler traces sequentially and write partitioned facts."""
    manifest_rows: list[dict[str, Any]] = []
    total_events = 0
    total_pg_rows = 0
    trace_count = 0
    for iter_id in cfg.profiler_iters:
        iter_dir = cfg.profiler_dir / f"iteration_{iter_id}"
        if not iter_dir.is_dir():
            raise FileNotFoundError(f"Missing profiler iteration directory: {iter_dir}")
        for rank in range(cfg.world_size):
            candidates = sorted(iter_dir.glob(f"rank{rank}.*.pt.trace.json"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one trace for iter={iter_id}, rank={rank}; "
                    f"found={len(candidates)}"
                )
            parsed = parse_trace_file(
                candidates[0],
                iter_id=iter_id,
                expected_rank=rank,
                host_split_rank=cfg.host_split_rank,
            )
            paths = write_parsed_trace(parsed, output_root)
            total_events += len(parsed.events)
            total_pg_rows += len(parsed.pg_config)
            trace_count += 1
            manifest_rows.append(
                {
                    **parsed.metadata,
                    "events_path": str(paths["events"]),
                    "pg_config_path": str(paths["pg_config"]),
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(["iter", "rank"])
    manifest_path = output_root / "profiler_trace_manifest_v5.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest.to_csv(temp, index=False)
    temp.replace(manifest_path)
    return {
        "trace_count": trace_count,
        "event_count": total_events,
        "pg_row_count": total_pg_rows,
    }

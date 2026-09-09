#!/usr/bin/env python3
"""Validate identity, PG topology and epoch metadata without loading full traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


BASE_TIME_RE = re.compile(rb'"baseTimeNanoseconds"\s*:\s*(\d+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def expected_groups(rank: int) -> dict[str, list[int]]:
    stage_base = (rank // 16) * 16
    cp_base = stage_base + ((rank % 16) // 2) * 2
    ep_base = (rank // 8) * 8
    return {
        "PIPELINE_MODEL_PARALLEL_GROUP": [
            (rank % 16) + stage * 16 for stage in range(16)
        ],
        "CONTEXT_PARALLEL_GROUP": [cp_base, cp_base + 1],
        "EXPERT_MODEL_PARALLEL_GROUP": list(range(ep_base, ep_base + 8)),
        "DATA_PARALLEL_GROUP": [
            stage_base + (rank % 2) + offset for offset in range(0, 16, 2)
        ],
    }


def inspect_one(root: Path, item: dict[str, str]) -> dict[str, object]:
    rel = item["relative_path"]
    path = root / rel
    expected_rank = int(item["rank"])
    row: dict[str, object] = {
        "relative_path": rel,
        "host": item["host"],
        "iteration": int(item["iter"]),
        "path_rank": expected_rank,
        "size_bytes": int(item["size_bytes"]),
        "error": "",
    }
    try:
        distributed = None
        schema_version = None
        with path.open("rb") as handle:
            for _ in range(128):
                line = handle.readline()
                if not line:
                    break
                if b'"schemaVersion"' in line:
                    schema_version = int(line.split(b":", 1)[1].rstrip(b",\r\n "))
                if b'"distributedInfo"' in line:
                    value = line.split(b":", 1)[1].rstrip(b",\r\n ")
                    distributed = json.loads(value)
                    break
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 1_048_576))
            tail = handle.read()
        if distributed is None:
            raise ValueError("missing_distributedInfo")
        base_matches = BASE_TIME_RE.findall(tail)
        if not base_matches:
            raise ValueError("missing_baseTimeNanoseconds")
        base_time_ns = int(base_matches[-1])
        last_nonspace = tail.rstrip()[-1:] if tail.rstrip() else b""
        pg_by_desc = {
            pg["pg_desc"]: pg["ranks"] for pg in distributed.get("pg_config", [])
        }
        mismatches = []
        for desc, expected in expected_groups(expected_rank).items():
            if pg_by_desc.get(desc) != expected:
                mismatches.append(desc)
        row.update(
            {
                "schema_version": schema_version,
                "json_closing_brace": last_nonspace == b"}",
                "distributed_rank": int(distributed["rank"]),
                "world_size": int(distributed["world_size"]),
                "pg_count": int(distributed["pg_count"]),
                "base_time_ns": base_time_ns,
                "pg_mismatches": ",".join(mismatches),
            }
        )
        if schema_version != 1:
            row["error"] = "schema_version"
        elif last_nonspace != b"}":
            row["error"] = "truncated_json"
        elif int(distributed["rank"]) != expected_rank:
            row["error"] = "rank_identity"
        elif int(distributed["world_size"]) != 256:
            row["error"] = "world_size"
        elif mismatches:
            row["error"] = "pg_topology"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        row["error"] = str(exc)
    return row


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        items = list(csv.DictReader(handle))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda item: inspect_one(root, item), items))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "trace_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    errors: dict[str, int] = {}
    for row in rows:
        error = str(row.get("error", ""))
        if error:
            errors[error] = errors.get(error, 0) + 1
    ranks = {int(row["distributed_rank"]) for row in rows if not row.get("error")}
    iterations = {
        int(row["iteration"]) for row in rows if not row.get("error")
    }
    bases = [int(row["base_time_ns"]) for row in rows if not row.get("error")]
    report = {
        "status": "PASS_TRACE_METADATA" if not errors else "BLOCKED",
        "file_count": len(rows),
        "valid_file_count": len(rows) - sum(errors.values()),
        "rank_count": len(ranks),
        "iteration_count": len(iterations),
        "iterations": sorted(iterations),
        "base_time_ns_min": min(bases) if bases else None,
        "base_time_ns_max": max(bases) if bases else None,
        "errors": errors,
        "validated_pg_formulas": [
            "PIPELINE_MODEL_PARALLEL_GROUP",
            "CONTEXT_PARALLEL_GROUP",
            "EXPERT_MODEL_PARALLEL_GROUP",
            "DATA_PARALLEL_GROUP",
        ],
        "limitations": [
            "This is a metadata/tail-integrity pass, not a full JSON event parse.",
            "Cross-host clock synchronization precision remains unverified.",
        ],
    }
    (args.output_dir / "trace_metadata_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build fail-closed file-level inventory for the 256-GPU framework corpus."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TRACE_RE = re.compile(
    r"^(?P<run>[^/]+)/(?P<host>worker\d+)/profiler/"
    r"iteration_(?P<iter>\d+)/rank(?P<rank>\d+)\."
    r"(?P<stamp>\d+)\.pt\.trace\.json$"
)
EXPECTED_RANKS = set(range(256))
EXPECTED_HOST_COUNT = 32
EXPECTED_RANKS_PER_HOST = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def classify(relative: str) -> str:
    parts = Path(relative).parts
    if "profiler" in parts and relative.endswith(".pt.trace.json"):
        return "profiler_trace"
    if "deepep_trace" in parts:
        return "deepep"
    if "tf_logs" in parts or ".tfevents." in relative:
        return "tf"
    if "wandb" in parts:
        return "wandb"
    if "checkpoints" in parts:
        return "checkpoint"
    if relative.endswith(".log"):
        return "training_log"
    return "other"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root is not a directory: {root}")

    file_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    malformed_trace_paths: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        category = classify(relative)
        file_rows.append(
            {
                "relative_path": relative,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "category": category,
            }
        )
        if category != "profiler_trace":
            continue
        match = TRACE_RE.match(relative)
        if match is None:
            malformed_trace_paths.append(relative)
            continue
        trace_rows.append(
            {
                "relative_path": relative,
                "run": match["run"],
                "host": match["host"],
                "iter": int(match["iter"]),
                "rank": int(match["rank"]),
                "stamp": int(match["stamp"]),
                "size_bytes": stat.st_size,
            }
        )

    key_counts = Counter((row["iter"], row["rank"]) for row in trace_rows)
    duplicate_keys = [
        {"iter": iteration, "rank": rank, "count": count}
        for (iteration, rank), count in sorted(key_counts.items())
        if count != 1
    ]
    iterations = sorted({int(row["iter"]) for row in trace_rows})
    ranks = {int(row["rank"]) for row in trace_rows}
    hosts = sorted({str(row["host"]) for row in trace_rows})
    missing_by_iter: dict[str, list[int]] = {}
    extra_by_iter: dict[str, list[int]] = {}
    for iteration in iterations:
        observed = {
            int(row["rank"]) for row in trace_rows if row["iter"] == iteration
        }
        if observed != EXPECTED_RANKS:
            missing_by_iter[str(iteration)] = sorted(EXPECTED_RANKS - observed)
            extra_by_iter[str(iteration)] = sorted(observed - EXPECTED_RANKS)

    host_ranks: dict[str, set[int]] = defaultdict(set)
    rank_hosts: dict[int, set[str]] = defaultdict(set)
    for row in trace_rows:
        host = str(row["host"])
        rank = int(row["rank"])
        host_ranks[host].add(rank)
        rank_hosts[rank].add(host)
    bad_host_rank_counts = {
        host: sorted(values)
        for host, values in sorted(host_ranks.items())
        if len(values) != EXPECTED_RANKS_PER_HOST
    }
    ambiguous_rank_hosts = {
        str(rank): sorted(values)
        for rank, values in sorted(rank_hosts.items())
        if len(values) != 1
    }

    expected_trace_count = len(iterations) * len(EXPECTED_RANKS)
    category_counts = Counter(str(row["category"]) for row in file_rows)
    blockers: list[str] = []
    if len(hosts) != EXPECTED_HOST_COUNT:
        blockers.append("host_count")
    if ranks != EXPECTED_RANKS:
        blockers.append("rank_set")
    if len(trace_rows) != expected_trace_count:
        blockers.append("trace_count")
    if duplicate_keys:
        blockers.append("duplicate_iter_rank")
    if missing_by_iter or extra_by_iter:
        blockers.append("iteration_rank_grid")
    if malformed_trace_paths:
        blockers.append("malformed_trace_path")
    if bad_host_rank_counts:
        blockers.append("host_rank_count")
    if ambiguous_rank_hosts:
        blockers.append("rank_host_ambiguity")

    report = {
        "status": "PASS_FILE_GRID" if not blockers else "BLOCKED",
        "capability": "file_identity_only",
        "root": str(root),
        "file_count": len(file_rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        "category_counts": dict(sorted(category_counts.items())),
        "profiler": {
            "trace_count": len(trace_rows),
            "expected_trace_count_from_discovered_iterations": expected_trace_count,
            "iterations": iterations,
            "iteration_count": len(iterations),
            "rank_count": len(ranks),
            "rank_min": min(ranks) if ranks else None,
            "rank_max": max(ranks) if ranks else None,
            "host_count": len(hosts),
            "hosts": hosts,
            "host_ranks": {
                host: sorted(values) for host, values in sorted(host_ranks.items())
            },
            "missing_by_iter": missing_by_iter,
            "extra_by_iter": extra_by_iter,
            "duplicate_keys": duplicate_keys,
            "malformed_trace_paths": malformed_trace_paths,
            "bad_host_rank_counts": bad_host_rank_counts,
            "ambiguous_rank_hosts": ambiguous_rank_hosts,
        },
        "blockers": blockers,
        "limitations": [
            "This report validates file paths and entity grids only.",
            "Trace JSON, distributedInfo, PG config and event semantics are not yet parsed.",
            "Hardware CSV coverage is validated separately after RAR extraction.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "framework_files.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(file_rows[0]))
        writer.writeheader()
        writer.writerows(file_rows)
    with (args.output_dir / "profiler_trace_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    with (args.output_dir / "rank_host_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "host"])
        writer.writeheader()
        for rank, values in sorted(rank_hosts.items()):
            for host in sorted(values):
                writer.writerow({"rank": rank, "host": host})
    (args.output_dir / "framework_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())

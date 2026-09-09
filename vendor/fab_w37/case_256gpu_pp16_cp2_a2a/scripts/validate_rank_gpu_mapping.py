#!/usr/bin/env python3
"""Validate rank→GPU mapping from kernel event pid for one complete iteration."""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


KERNEL_MARKER = b'"cat": "kernel"'
PID_RE = re.compile(rb'"cat":\s*"kernel".*?"pid":\s*(\d+)', re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rank-topology", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def kernel_pid(path: Path) -> int:
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            offset = mapped.find(KERNEL_MARKER)
            if offset < 0:
                raise ValueError("missing_kernel_event")
            match = PID_RE.search(mapped[offset : offset + 1024])
            if match is None:
                raise ValueError("missing_kernel_pid")
            return int(match.group(1))


def main() -> int:
    args = parse_args()
    with args.rank_topology.open(newline="", encoding="utf-8") as handle:
        topology = {
            int(row["rank"]): row for row in csv.DictReader(handle)
        }
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if int(row["iter"]) == args.iteration
        ]
    if len(selected) != 256:
        raise SystemExit(
            f"iteration {args.iteration} must contain 256 traces; found {len(selected)}"
        )

    def inspect(item: dict[str, str]) -> dict[str, object]:
        rank = int(item["rank"])
        expected = int(topology[rank]["gpu_id"])
        try:
            observed = kernel_pid(args.root / item["relative_path"])
            error = "" if observed == expected else "gpu_id_mismatch"
        except (OSError, ValueError) as exc:
            observed = None
            error = str(exc)
        return {
            "iteration": args.iteration,
            "rank": rank,
            "host": item["host"],
            "expected_gpu_id": expected,
            "kernel_pid_gpu_id": observed,
            "error": error,
            "relative_path": item["relative_path"],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(inspect, selected))
    rows.sort(key=lambda row: int(row["rank"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rank_gpu_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = [row for row in rows if row["error"]]
    report = {
        "status": "PASS_RANK_GPU_MAPPING" if not errors else "BLOCKED",
        "iteration": args.iteration,
        "checked_rank_count": len(rows),
        "valid_count": len(rows) - len(errors),
        "error_count": len(errors),
        "mapping": "gpu_id=local rank within each 8-rank host block",
    }
    (args.output_dir / "rank_gpu_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

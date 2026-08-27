#!/usr/bin/env python3
"""Freeze PP send/recv CPU annotations used by the v2 software calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


CASE_ROOT = Path(__file__).resolve().parents[1]
PP_API_NAMES = {
    "recv_forward",
    "send_forward",
    "recv_backward",
    "send_backward",
    "send_forward_recv_backward",
    "send_backward_recv_forward",
}
_ANNOTATION = re.compile(
    rb'"ph": "X", "cat": "user_annotation", "name": "([^"]+)"[^\n]*'
    rb'\n\s*"ts": ([0-9.]+), "dur": ([0-9.]+),'
)
_BASE_TIME = re.compile(rb'"baseTimeNanoseconds"\s*:\s*([0-9]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PP P2P API annotations.")
    parser.add_argument("--iteration", type=int, default=55)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=CASE_ROOT / "data/pp_api_trace_events.csv",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=CASE_ROOT / "data/pp_api_trace_provenance.json",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_one(values: tuple[Path, str, int, int, int, int]) -> list[dict[str, object]]:
    trace_root, relative_path, iteration, rank, stage, lane = values
    path = trace_root / relative_path
    data = path.read_bytes()
    base_match = _BASE_TIME.search(data)
    if not base_match:
        raise ValueError(f"missing baseTimeNanoseconds in {path}")
    base_time_ns = int(base_match.group(1))
    found: list[tuple[int, str, int]] = []
    for match in _ANNOTATION.finditer(data):
        raw_name, raw_start_us, raw_duration_us = match.groups()
        name = raw_name.decode("utf-8", errors="replace")
        if name not in PP_API_NAMES:
            continue
        start_ns = base_time_ns + int(round(float(raw_start_us) * 1000.0))
        duration_ns = int(round(float(raw_duration_us) * 1000.0))
        found.append((start_ns, name, duration_ns))
    occurrence = Counter()
    rows = []
    for start_ns, name, duration_ns in sorted(found):
        rows.append(
            {
                "iteration": iteration,
                "rank": rank,
                "pp_stage": stage,
                "pp_lane": lane,
                "api_name": name,
                "occurrence": occurrence[name],
                "observed_start_ns": start_ns,
                "observed_end_ns": start_ns + duration_ns,
                "duration_ns": duration_ns,
            }
        )
        occurrence[name] += 1
    return rows


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(document: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest_path = args.source_case / "results/readiness/profiler_trace_manifest.csv"
    topology_path = args.source_case / "results/topology/rank_topology.csv"
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["iter"].eq(args.iteration)]
    topology = pd.read_csv(topology_path)[["rank", "pp_stage", "pp_lane"]]
    selected = manifest.merge(topology, on="rank", validate="one_to_one")
    if len(selected) != 256:
        raise ValueError(f"iteration {args.iteration} must contain 256 traces")
    tasks = [
        (
            args.trace_root,
            str(row.relative_path),
            args.iteration,
            int(row.rank),
            int(row.pp_stage),
            int(row.pp_lane),
        )
        for row in selected.itertuples(index=False)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = [row for parsed in executor.map(_parse_one, tasks) for row in parsed]
    frame = pd.DataFrame(rows).sort_values(
        ["pp_lane", "pp_stage", "observed_start_ns"]
    ).reset_index(drop=True)
    if set(frame["api_name"]) != PP_API_NAMES:
        raise ValueError("PP API annotation set is incomplete")
    _atomic_csv(frame, args.output)
    _atomic_json(
        {
            "schema": "pp-api-trace-events-v1",
            "status": "PASS",
            "iteration": args.iteration,
            "ranks": int(frame["rank"].nunique()),
            "events": len(frame),
            "counts": {
                str(key): int(value)
                for key, value in frame["api_name"].value_counts().sort_index().items()
            },
            "manifest_sha256": _sha256(manifest_path),
            "topology_sha256": _sha256(topology_path),
            "output_sha256": _sha256(args.output),
        },
        args.provenance,
    )
    print(f"wrote {len(frame)} PP API events for iteration {args.iteration} to {args.output}")


if __name__ == "__main__":
    main()

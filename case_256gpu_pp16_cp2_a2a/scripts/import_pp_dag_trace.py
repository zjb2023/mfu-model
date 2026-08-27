#!/usr/bin/env python3
"""Freeze one 256-GPU iteration's FWD/BWD annotations for the PP DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x10000_analysis.pp_dag import non_interleaved_1f1b_schedule  # noqa: E402


_ANNOTATION = re.compile(
    rb'"ph": "X", "cat": "user_annotation", "name": "(forward_step|backward_step)"[^\n]*'
    rb'\n\s*"ts": ([0-9.]+), "dur": ([0-9.]+),'
)
_BASE_TIME = re.compile(rb'"baseTimeNanoseconds"\s*:\s*([0-9]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the eight per-rank FWD/BWD annotations for one PP-DAG iteration."
    )
    parser.add_argument("--iteration", type=int, default=55)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=CASE_ROOT / "data/pp_dag_trace_events.csv",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=CASE_ROOT / "data/pp_dag_trace_provenance.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _parse_one(
    trace_root: Path,
    relative_path: str,
    iteration: int,
    rank: int,
    pp_stage: int,
    pp_lane: int,
) -> list[dict[str, object]]:
    path = trace_root / relative_path
    data = path.read_bytes()
    base_match = _BASE_TIME.search(data)
    if not base_match:
        raise ValueError(f"missing baseTimeNanoseconds in {path}")
    base_time_ns = int(base_match.group(1))
    found: list[tuple[str, int, int]] = []
    for match in _ANNOTATION.finditer(data):
        raw_name, raw_start_us, raw_duration_us = match.groups()
        phase = "forward" if raw_name == b"forward_step" else "backward"
        start_ns = base_time_ns + int(round(float(raw_start_us) * 1000.0))
        duration_ns = int(round(float(raw_duration_us) * 1000.0))
        found.append((phase, start_ns, duration_ns))
    if len(found) != 8:
        raise ValueError(f"expected eight FWD/BWD annotations in {path}, found {len(found)}")

    occurrence = {"forward": 0, "backward": 0}
    rows: list[dict[str, object]] = []
    symbols: list[str] = []
    for phase, start_ns, duration_ns in sorted(found, key=lambda item: item[1]):
        microbatch = occurrence[phase]
        occurrence[phase] += 1
        symbols.append(f"{'F' if phase == 'forward' else 'B'}{microbatch}")
        rows.append(
            {
                "iteration": iteration,
                "rank": rank,
                "pp_stage": pp_stage,
                "pp_lane": pp_lane,
                "phase": phase,
                "microbatch": microbatch,
                "observed_start_ns": start_ns,
                "observed_end_ns": start_ns + duration_ns,
                "duration_ns": duration_ns,
            }
        )
    expected = [
        f"{'F' if row.phase == 'forward' else 'B'}{int(row.microbatch)}"
        for row in non_interleaved_1f1b_schedule(16, 4, pp_stage).itertuples(index=False)
    ]
    if symbols != expected:
        raise ValueError(
            f"rank {rank} stage {pp_stage} schedule mismatch: observed={symbols}, expected={expected}"
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    manifest_path = args.source_case / "results/readiness/profiler_trace_manifest.csv"
    topology_path = args.source_case / "results/topology/rank_topology.csv"
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["iter"].eq(args.iteration)].copy()
    topology = pd.read_csv(topology_path)[["rank", "pp_stage", "pp_lane"]]
    selected = manifest.merge(topology, on="rank", validate="one_to_one")
    if len(selected) != 256 or selected["rank"].nunique() != 256:
        raise ValueError(f"iteration {args.iteration} does not contain 256 unique ranks")

    tasks = [
        (
            args.trace_root,
            str(row.relative_path),
            int(args.iteration),
            int(row.rank),
            int(row.pp_stage),
            int(row.pp_lane),
        )
        for row in selected.itertuples(index=False)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        parsed = list(executor.map(lambda values: _parse_one(*values), tasks))
    rows = [row for event_rows in parsed for row in event_rows]
    frame = pd.DataFrame(rows).sort_values(
        ["pp_lane", "pp_stage", "observed_start_ns"]
    ).reset_index(drop=True)
    _atomic_csv(frame, args.output)
    _atomic_json(
        {
            "schema": "pp-dag-trace-events-v1",
            "status": "PASS",
            "iteration": args.iteration,
            "ranks": int(frame["rank"].nunique()),
            "events": len(frame),
            "forward_events": int(frame["phase"].eq("forward").sum()),
            "backward_events": int(frame["phase"].eq("backward").sum()),
            "schedule_validation": "all ranks match Megatron non-interleaved 1F1B",
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "topology": {"path": str(topology_path), "sha256": _sha256(topology_path)},
            "trace_root": str(args.trace_root),
            "trace_files_read": len(parsed),
            "trace_bytes_read": int(selected["size_bytes"].sum()),
            "output": {"path": str(args.output), "sha256": _sha256(args.output)},
        },
        args.provenance,
    )
    print(
        f"wrote {len(frame)} events from {frame['rank'].nunique()} ranks "
        f"for iteration {args.iteration} to {args.output}"
    )


if __name__ == "__main__":
    main()

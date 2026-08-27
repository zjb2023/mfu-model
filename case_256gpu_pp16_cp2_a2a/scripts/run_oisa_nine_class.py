#!/usr/bin/env python3
"""Run or resume all frozen nine-class OISA requests and collect results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    result_root = case_root / "results/oisa_s5000_256gpu_nine_class"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=result_root / "inputs")
    parser.add_argument("--output", type=Path, default=result_root / "runs")
    parser.add_argument("--oisa-root", type=Path, default=Path("/tmp/oisa-rank-release-fct"))
    parser.add_argument(
        "--simulator",
        type=Path,
        help="Exact OISA binary; defaults to <oisa-root>/bin/OISA_simulator.",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path(
            "/tmp/oisa-rank-release-fct/topologies/"
            "s5000_256gpu_32host_1spine"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--result-ready-grace-seconds", type=float, default=3.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _write_summary(output: Path, results: list[dict[str, object]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        "kind", "request_id", "collective_elapsed_ns", "arrival_span_ns",
        "tail_after_last_release_ns", "runner_lifecycle", "simulator_returncode",
        "runner_elapsed_seconds", "simulator_commit", "topology_hash",
        "simulator_worktree_dirty", "simulator_binary_sha256",
        "runner_source_sha256",
    ]
    with (output.parent / "oisa_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({column: result.get(column, "") for column in columns})
    document = {
        "schema": "mfu-oisa-nine-class-results-v1",
        "status": "PASS" if len(results) == 9 else "INCOMPLETE",
        "completed_requests": len(results),
        "expected_requests": 9,
        "all_timing_identities_hold": all(
            int(result["collective_elapsed_ns"])
            == int(result["arrival_span_ns"])
            + int(result["tail_after_last_release_ns"])
            for result in results
        ),
        "results": results,
    }
    (output.parent / "batch_summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enrich_provenance(
    result: dict[str, object], oisa_root: Path, simulator: Path
) -> dict[str, object]:
    enriched = dict(result)
    enriched.setdefault(
        "simulator_worktree_dirty",
        bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=oisa_root,
                text=True,
            ).strip()
        ),
    )
    enriched.setdefault("simulator_binary_sha256", _sha256(simulator.resolve()))
    enriched.setdefault(
        "runner_source_sha256",
        _sha256((oisa_root / "oisa_sim/collective_runner.py").resolve()),
    )
    return enriched


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.inputs / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    simulator = args.simulator or args.oisa_root / "bin/OISA_simulator"
    results: list[dict[str, object]] = []
    for relative_request in manifest["requests"]:
        request_path = args.inputs / relative_request
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result_path = args.output / request["request_id"] / "collective_result.json"
        if result_path.is_file() and not args.no_resume:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            command = [
                sys.executable,
                str(args.oisa_root / "scripts/run_collective_fct.py"),
                "--request", str(request_path),
                "--topology", str(args.topology),
                "--output-root", str(args.output),
                "--simulator", str(simulator),
                "--timeout-seconds", str(args.timeout_seconds),
                "--result-ready-grace-seconds", str(args.result_ready_grace_seconds),
            ]
            subprocess.run(command, cwd=args.oisa_root, check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        result = _enrich_provenance(result, args.oisa_root, simulator)
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result["kind"] = request["kind"]
        results.append(result)
        _write_summary(args.output, results)
        print(
            f"[{len(results)}/9] {request['kind']}: "
            f"tail={int(result['tail_after_last_release_ns']) / 1e6:.6f} ms, "
            f"lifecycle={result.get('runner_lifecycle', 'legacy')}",
            flush=True,
        )
    _write_summary(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

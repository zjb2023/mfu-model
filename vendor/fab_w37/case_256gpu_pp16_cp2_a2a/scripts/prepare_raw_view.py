#!/usr/bin/env python3
"""Create a no-copy canonical hardware view for the 256-GPU analysis.

The collectors store files below independent ``machines/host_*/data/run_*``
trees.  The analysis pipeline consumes a stable
``worker*/{mtlink,nic}/*.csv`` layout.  This script bridges the layouts with
absolute symbolic links; it never copies or changes the raw 200-GB corpus.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


MT_RE = re.compile(r"^host_(worker\d+)_gpu(\d+)_.*_seq(\d+)\.csv$")
NIC_RE = re.compile(r"^host_(worker\d+)_(mlx5_\d+)_.*_seq(\d+)\.csv$")
EXPECTED_GPUS = set(range(8))
EXPECTED_NICS = {
    "mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3",
    "mlx5_6", "mlx5_7", "mlx5_8", "mlx5_9",
}


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mtlink-root",
        type=Path,
        default=Path("/home/zjb/gbs64/mtlink_256_gbs64/machines"),
    )
    parser.add_argument(
        "--nic-root",
        type=Path,
        default=Path("/home/zjb/gbs64/rdma_256_gbs64/machines"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=case_root / ".raw_view"
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=case_root / "results" / "readiness" / "raw_view.json",
    )
    return parser.parse_args()


def discover(
    root: Path, pattern: str, regex: re.Pattern[str], kind: str
) -> list[tuple[Path, str, str, int]]:
    rows: list[tuple[Path, str, str, int]] = []
    for path in sorted(root.glob(pattern)):
        match = regex.match(path.name)
        if match is None:
            continue
        host, entity, sequence = match.groups()
        parent_host = path.parents[3].name.removeprefix("host_")
        if parent_host != host:
            raise ValueError(f"{kind} host mismatch: {path}")
        rows.append((path.resolve(), host, entity, int(sequence)))
    return rows


def link_rows(
    rows: list[tuple[Path, str, str, int]], output_root: Path, kind: str
) -> None:
    for source, host, _entity, _sequence in rows:
        target_dir = output_root / host / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"existing link points elsewhere: {target}")
            continue
        if target.exists():
            raise FileExistsError(target)
        target.symlink_to(source)


def entity_sequences(
    rows: list[tuple[Path, str, str, int]]
) -> dict[tuple[str, str], list[int]]:
    result: dict[tuple[str, str], list[int]] = defaultdict(list)
    for _path, host, entity, sequence in rows:
        result[(host, entity)].append(sequence)
    return {key: sorted(value) for key, value in sorted(result.items())}


def main() -> int:
    args = parse_args()
    mt_rows = discover(
        args.mtlink_root.resolve(),
        "host_worker*/data/run_*/csv/host_worker*_gpu*_seq*.csv",
        MT_RE,
        "mtlink",
    )
    nic_rows = discover(
        args.nic_root.resolve(),
        "host_worker*/data/run_*/csv/host_worker*_mlx5_*_seq*.csv",
        NIC_RE,
        "nic",
    )
    mt_sequences = entity_sequences(mt_rows)
    nic_sequences = entity_sequences(nic_rows)
    hosts = sorted({host for _path, host, _entity, _seq in mt_rows})
    blockers: list[str] = []
    if len(hosts) != 32:
        blockers.append("host_count")
    if {host for _path, host, _entity, _seq in nic_rows} != set(hosts):
        blockers.append("host_set_mismatch")
    expected_mt = {(host, str(gpu)) for host in hosts for gpu in EXPECTED_GPUS}
    expected_nic = {
        (host, device) for host in hosts for device in EXPECTED_NICS
    }
    if set(mt_sequences) != expected_mt:
        blockers.append("mtlink_entity_grid")
    if set(nic_sequences) != expected_nic:
        blockers.append("nic_entity_grid")
    if any(values[0] != 0 or values != list(range(values[-1] + 1))
           for values in mt_sequences.values()):
        blockers.append("mtlink_sequence_gap")
    if any(values[0] != 0 or values != list(range(values[-1] + 1))
           for values in nic_sequences.values()):
        blockers.append("nic_sequence_gap")
    if blockers:
        raise RuntimeError(f"raw view blocked: {blockers}")

    link_rows(mt_rows, args.output_root.resolve(), "mtlink")
    link_rows(nic_rows, args.output_root.resolve(), "nic")
    report = {
        "status": "PASS",
        "schema_version": "256gpu-raw-view-v1",
        "copy_performed": False,
        "output_root": str(args.output_root.resolve()),
        "host_count": len(hosts),
        "hosts": hosts,
        "mtlink_file_count": len(mt_rows),
        "mtlink_entity_count": len(mt_sequences),
        "mtlink_multi_sequence_entity_count": sum(
            len(value) > 1 for value in mt_sequences.values()
        ),
        "nic_file_count": len(nic_rows),
        "nic_entity_count": len(nic_sequences),
        "nic_multi_sequence_entity_count": sum(
            len(value) > 1 for value in nic_sequences.values()
        ),
        "blockers": blockers,
    }
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

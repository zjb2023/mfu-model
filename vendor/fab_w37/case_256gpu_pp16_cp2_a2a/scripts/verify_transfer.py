#!/usr/bin/env python3
"""Verify local transferred files against frozen remote inventory sizes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory: dict[str, int] = {}
    with args.inventory.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            inventory[row["relative_path"]] = int(row["size_bytes"])

    requested = [
        line.strip()
        for line in args.file_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("file list contains duplicate paths")

    missing_from_inventory: list[str] = []
    missing_local: list[str] = []
    size_mismatch: list[dict[str, object]] = []
    verified_bytes = 0

    local_root = args.local_root.resolve()
    for relative_path in requested:
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe path in file list: {relative_path!r}")
        expected = inventory.get(relative_path)
        if expected is None:
            missing_from_inventory.append(relative_path)
            continue
        local_path = local_root.joinpath(*pure.parts)
        if not local_path.is_file():
            missing_local.append(relative_path)
            continue
        actual = local_path.stat().st_size
        if actual != expected:
            size_mismatch.append(
                {
                    "relative_path": relative_path,
                    "expected_size_bytes": expected,
                    "actual_size_bytes": actual,
                }
            )
            continue
        verified_bytes += actual

    status = (
        "PASS"
        if not missing_from_inventory and not missing_local and not size_mismatch
        else "FAIL"
    )
    report = {
        "status": status,
        "verification": "path_existence_and_exact_size",
        "requested_file_count": len(requested),
        "verified_file_count": (
            len(requested)
            - len(missing_from_inventory)
            - len(missing_local)
            - len(size_mismatch)
        ),
        "verified_bytes": verified_bytes,
        "missing_from_inventory": missing_from_inventory,
        "missing_local": missing_local,
        "size_mismatch": size_mismatch,
        "note": (
            "PASS confirms exact paths and sizes only. Run remote rsync checksum "
            "verification before admitting raw inputs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

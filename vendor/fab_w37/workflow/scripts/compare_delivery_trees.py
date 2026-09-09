#!/usr/bin/env python3
"""Compare two delivery trees, allowing only declared path/timestamp differences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".csv", ".html", ".md", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tree",
        action="append",
        nargs=4,
        metavar=("NAME", "REFERENCE", "CANDIDATE", "EXPECTED_FILES"),
        default=[],
    )
    parser.add_argument(
        "--file",
        action="append",
        nargs=3,
        metavar=("NAME", "REFERENCE", "CANDIDATE"),
        default=[],
    )
    parser.add_argument(
        "--path-map",
        action="append",
        nargs=2,
        metavar=("CANDIDATE_PREFIX", "REFERENCE_PREFIX"),
        default=[],
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_mappings(raw_mappings: list[list[str]]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for candidate, reference in raw_mappings:
        pairs = [
            (candidate, reference),
            (str(Path(candidate).resolve()), str(Path(reference).resolve())),
        ]
        for pair in pairs:
            if pair not in mappings:
                mappings.append(pair)
    return sorted(mappings, key=lambda pair: len(pair[0]), reverse=True)


def normalize_text(value: str, mappings: list[tuple[str, str]]) -> str:
    for candidate, reference in mappings:
        value = value.replace(candidate, reference)
    return value


def normalize_json(
    value: Any,
    mappings: list[tuple[str, str]],
    drop_checked_at: bool,
) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_json(child, mappings, drop_checked_at)
            for key, child in value.items()
            if not (drop_checked_at and key == "checked_at")
        }
    if isinstance(value, list):
        return [normalize_json(child, mappings, drop_checked_at) for child in value]
    if isinstance(value, str):
        return normalize_text(value, mappings)
    return value


def normalized_bytes(
    path: Path,
    mappings: list[tuple[str, str]],
    drop_checked_at: bool,
) -> tuple[bytes, str]:
    data = path.read_bytes()
    if path.suffix == ".json":
        payload = json.loads(data.decode("utf-8"))
        normalized = normalize_json(payload, mappings, drop_checked_at)
        comparison = "json-structure/path"
        if drop_checked_at:
            comparison += "/checked_at"
        return (
            json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            comparison,
        )
    if path.suffix in TEXT_SUFFIXES:
        text = normalize_text(data.decode("utf-8"), mappings)
        return text.encode("utf-8"), "text/path"
    return data, "byte-exact"


def compare_file(
    name: str,
    reference: Path,
    candidate: Path,
    mappings: list[tuple[str, str]],
) -> dict[str, Any]:
    if not reference.is_file():
        raise FileNotFoundError(f"missing reference file: {reference}")
    if not candidate.is_file():
        raise FileNotFoundError(f"missing candidate file: {candidate}")

    reference_raw = reference.read_bytes()
    candidate_raw = candidate.read_bytes()
    drop_checked_at = name.endswith("_gate")
    reference_data, comparison = normalized_bytes(reference, [], drop_checked_at)
    candidate_data, _ = normalized_bytes(candidate, mappings, drop_checked_at)
    if reference_data != candidate_data:
        raise ValueError(
            f"{name} differs after allowed normalization: "
            f"{reference} != {candidate}"
        )

    return {
        "name": name,
        "reference": str(reference),
        "candidate": str(candidate),
        "comparison": comparison,
        "byte_exact": reference_raw == candidate_raw,
        "reference_sha256": sha256(reference_raw),
        "candidate_sha256": sha256(candidate_raw),
        "normalized_sha256": sha256(reference_data),
        "bytes": len(reference_raw),
    }


def main() -> int:
    args = parse_args()
    mappings = path_mappings(args.path_map)
    compared: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for name, reference_raw, candidate_raw, expected_raw in args.tree:
        reference_root = Path(reference_raw)
        candidate_root = Path(candidate_raw)
        expected = int(expected_raw)
        candidate_files = sorted(
            path for path in candidate_root.rglob("*") if path.is_file()
        )
        if len(candidate_files) != expected:
            raise SystemExit(
                f"{name}: expected {expected} candidate files, found {len(candidate_files)}"
            )
        start = len(compared)
        for candidate in candidate_files:
            relative = candidate.relative_to(candidate_root)
            compared.append(
                compare_file(
                    f"{name}/{relative.as_posix()}",
                    reference_root / relative,
                    candidate,
                    mappings,
                )
            )
        groups.append({"name": name, "file_count": len(compared) - start})

    for name, reference, candidate in args.file:
        compared.append(compare_file(name, Path(reference), Path(candidate), mappings))

    report = {
        "status": "PASS",
        "contract": (
            "All candidate artifacts match the reference after normalizing only declared "
            "output-path prefixes and validation-gate checked_at timestamps."
        ),
        "file_count": len(compared),
        "byte_exact_count": sum(item["byte_exact"] for item in compared),
        "groups": groups,
        "files": compared,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

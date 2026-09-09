#!/usr/bin/env python3
"""Classify a read-only remote inventory into reviewable transfer lists."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath


ARCHIVE_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.zst",
    ".tzst",
    ".zip",
    ".rar",
    ".7z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def classify(relative_path: str) -> str:
    lowered = relative_path.lower()
    components = set(PurePosixPath(lowered).parts)
    if "mtlink" in lowered or any(part.startswith("mtlink") for part in components):
        return "mtlink"
    if (
        "rdma" in lowered
        or "mlx5" in lowered
        or re.search(r"(^|[/_.-])nic([/_.-]|$)", lowered)
    ):
        return "nic"
    if "deepep" in lowered:
        return "deepep"
    if (
        "profiler" in lowered
        or ".pt.trace.json" in lowered
        or re.search(r"(^|[/_.-])trace([/_.-]|$)", lowered)
    ):
        return "trace"
    return "other"


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path in inventory: {value!r}")
    if "\n" in value or "\t" in value:
        raise ValueError(f"unsupported newline/tab in path: {value!r}")


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with args.inventory.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"relative_path", "size_bytes", "mtime_epoch"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(
                f"inventory columns must be {sorted(required)}, got {reader.fieldnames}"
            )
        for source in reader:
            relative_path = source["relative_path"]
            validate_relative_path(relative_path)
            if relative_path in seen:
                raise ValueError(f"duplicate inventory path: {relative_path}")
            seen.add(relative_path)
            size_bytes = int(source["size_bytes"])
            if size_bytes < 0:
                raise ValueError(f"negative file size: {relative_path}")
            rows.append(
                {
                    "relative_path": relative_path,
                    "size_bytes": size_bytes,
                    "mtime_epoch": source["mtime_epoch"],
                    "category": classify(relative_path),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_category = {
        "nic": [],
        "mtlink": [],
        "trace": [],
        "deepep": [],
        "other": [],
    }
    for row in rows:
        by_category[str(row["category"])].append(row)

    for category, category_rows in by_category.items():
        output = args.output_dir / f"{category}.files"
        output.write_text(
            "".join(f"{row['relative_path']}\n" for row in category_rows),
            encoding="utf-8",
        )
    (args.output_dir / "all.files").write_text(
        "".join(f"{row['relative_path']}\n" for row in rows),
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "inventory": str(args.inventory),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "total_size_human": human_bytes(
            sum(int(row["size_bytes"]) for row in rows)
        ),
        "categories": {},
        "archive_suffix_counts": dict(
            sorted(
                Counter(
                    suffix
                    for row in rows
                    for suffix in ARCHIVE_SUFFIXES
                    if str(row["relative_path"]).lower().endswith(suffix)
                ).items()
            )
        ),
        "warning": (
            "Path classification is heuristic. Review attempt/time/host identity "
            "before approving any transfer list."
        ),
    }
    categories = summary["categories"]
    assert isinstance(categories, dict)
    for category, category_rows in by_category.items():
        size = sum(int(row["size_bytes"]) for row in category_rows)
        categories[category] = {
            "file_count": len(category_rows),
            "size_bytes": size,
            "size_human": human_bytes(size),
        }

    summary_path = args.output_dir / "transfer_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

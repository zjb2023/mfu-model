#!/usr/bin/env python3
"""Fail unless every supplied validation JSON reports a PASS status."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("validations", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked: list[dict[str, str]] = []
    failures: list[str] = []

    for path in args.validations:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: unreadable validation JSON: {exc}")
            continue

        status = payload.get("status")
        if not isinstance(status, str):
            failures.append(f"{path}: missing string status")
            continue
        if not status.upper().startswith("PASS"):
            failures.append(f"{path}: status={status!r}")
            continue
        checked.append({"path": str(path), "status": status})

    if failures:
        raise SystemExit("validation gate failed:\n- " + "\n- ".join(failures))

    report = {
        "status": "PASS",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "validation_count": len(checked),
        "validations": checked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Attach test evidence and refresh hashes for the DAG v6 structure bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_v60_operator_ir_256_to_224"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    junit = run / "logs/pytest.xml"
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] <= 0 or totals["failures"] or totals["errors"]:
        raise RuntimeError(f"tests did not pass: {totals}")
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["tests"] = {
        "status": "PASS",
        **totals,
        "junit_xml": {"path": str(junit), "sha256": digest(junit)},
    }
    dump(provenance_path, provenance)
    manifest_path = run / "artifact_manifest.json"
    artifacts = sorted(
        path for path in run.rglob("*")
        if path.is_file() and path != manifest_path
    )
    dump(
        manifest_path,
        {
            "status": provenance["status"],
            "tests": provenance["tests"],
            "artifacts": [
                {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
                for path in artifacts
            ],
        },
    )
    print(json.dumps({"run": str(run), "tests": totals, "artifacts": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

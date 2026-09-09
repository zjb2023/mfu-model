#!/usr/bin/env python3
"""Finalize DAG v6.2 test evidence, provenance and artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v62_cost_bound_prediction_2026w36.toml"


def configured_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    output = configured_path(config["outputs"]["output_dir"])
    pytest_path = output / "logs/pytest.xml"
    if not pytest_path.is_file():
        raise FileNotFoundError(pytest_path)

    root = ElementTree.parse(pytest_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        name: sum(int(float(suite.attrib.get(name, "0"))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if totals["failures"] or totals["errors"]:
        raise ValueError(f"pytest did not pass: {totals}")
    test_results_path = output / "test_results.json"
    atomic_json(
        test_results_path,
        {
            "schema": "dag-v6.2-test-results-v1",
            "status": "PASS",
            **totals,
            "pytest_xml": {
                "path": str(pytest_path.resolve()),
                "size_bytes": pytest_path.stat().st_size,
                "sha256": sha256(pytest_path),
            },
        },
    )

    builder = REPO / "case_224gpu_pp14_cp2_a2a/scripts/build_dag_v62_cost_bound_prediction.py"
    finalizer = Path(__file__).resolve()
    test_source = REPO / "case_224gpu_pp14_cp2_a2a/tests/test_dag_v62_cost_bound_prediction.py"
    code_paths = [config_path, builder, finalizer, test_source]
    tree_digest = hashlib.sha256()
    code_entries = []
    for path in code_paths:
        digest = sha256(path)
        tree_digest.update(str(path.resolve()).encode())
        tree_digest.update(digest.encode())
        code_entries.append(
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest}
        )
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_code_tree_hash"] = tree_digest.hexdigest()
    provenance["code_and_config"] = code_entries
    provenance["verification"] = {
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "test_results": str(test_results_path.resolve()),
    }
    atomic_json(provenance_path, provenance)

    reproduction_path = output / "reproduction_command.txt"
    atomic_text(
        reproduction_path,
        f"cd {REPO}\n"
        f"python {builder} --config {config_path}\n"
        f"pytest -q case_224gpu_pp14_cp2_a2a/tests/test_dag_v60_operator_ir.py "
        f"case_224gpu_pp14_cp2_a2a/tests/test_dag_v61_kernel_calibration.py "
        f"case_224gpu_pp14_cp2_a2a/tests/test_dag_v62_cost_bound_prediction.py "
        f"--junitxml={pytest_path}\n"
        f"python {finalizer} --config {config_path}\n",
    )

    manifest_path = output / "artifact_manifest.json"
    artifacts = sorted(
        path.resolve()
        for path in output.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve() and not path.name.endswith(".tmp")
    )
    atomic_json(
        manifest_path,
        {
            "schema": "dag-v6.2-artifact-manifest-v1",
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in artifacts
            ],
        },
    )
    print(json.dumps({"status": "PASS", **totals, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

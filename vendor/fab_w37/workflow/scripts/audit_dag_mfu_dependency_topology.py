#!/usr/bin/env python3
"""Fail closed when a parameter-only DAG candidate changes dependency edges."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EDGE_COLUMNS = (
    "case_id", "src", "dst", "edge_type", "tensor_key", "dependency_source",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def topology_fingerprint(edges: pd.DataFrame) -> str:
    missing = set(EDGE_COLUMNS) - set(edges.columns)
    if missing:
        raise ValueError(f"edge schema missing: {sorted(missing)}")
    frame = edges[list(EDGE_COLUMNS)].fillna("").astype(str).sort_values(
        list(EDGE_COLUMNS), kind="stable"
    )
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit(lock: dict[str, Any], edges: pd.DataFrame) -> dict[str, Any]:
    expected = str(lock.get("future_parameter_search_must_match_sha256", ""))
    if len(expected) != 64:
        raise ValueError("dependency lock has no future candidate fingerprint")
    actual = topology_fingerprint(edges)
    passed = actual == expected
    return {
        "schema": "dag-mfu-dependency-topology-guard-v1",
        "status": "PASS_PARAMETER_ONLY_TOPOLOGY_UNCHANGED" if passed else "FAIL_DEPENDENCY_TOPOLOGY_CHANGED",
        "passed": passed,
        "expected_topology_sha256": expected,
        "actual_topology_sha256": actual,
        "candidate_edge_rows": len(edges),
        "allowed_candidate_changes": [
            "node_duration", "cost_component", "source_only_static_scaling",
            "network_service_fct", "software_launch_cost",
        ],
        "forbidden_candidate_changes": [
            "edge_add", "edge_delete", "edge_direction", "edge_type",
            "dependency_source",
        ],
        "topology_change_authority": "new reviewed static schedule or training-code evidence requires a new semantic version and a new lock",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--candidate-edges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lock_path = args.lock.resolve()
    edges_path = args.candidate_edges.resolve()
    for path in (lock_path, edges_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    edges = pd.read_csv(edges_path, low_memory=False)
    result = audit(lock, edges)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "dependency_topology_audit.json"
    report_path = output / "DEPENDENCY_TOPOLOGY_GUARD.md"
    provenance_path = output / "provenance.json"
    manifest_path = output / "manifest.json"
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(f"""# DAG MFU依赖拓扑门禁

状态：**{result['status']}**。

- 锁定指纹：`{result['expected_topology_sha256']}`。
- 候选指纹：`{result['actual_topology_sha256']}`。
- 候选边数：`{result['candidate_edge_rows']}`。

该门禁只服务于参数优化版本：允许修改节点耗时、成本分量、静态缩放与FCT，不允许增加、删除、改向或重命名依赖边。若训练代码或静态schedule提供了新的结构证据，必须新建语义版本、单独审阅并生成新锁，不能在参数搜索中绕过本门禁。
""", encoding="utf-8")
    provenance_path.write_text(json.dumps({
        "schema": "dag-mfu-dependency-topology-guard-provenance-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [artifact(lock_path), artifact(edges_path)],
        "script": artifact(Path(__file__)),
        "outputs": [artifact(audit_path), artifact(report_path)],
        "target_timing_files_read": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "schema": "dag-mfu-dependency-topology-guard-manifest-v1",
        "status": result["status"],
        "artifacts": [artifact(path) for path in (audit_path, report_path, provenance_path)],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

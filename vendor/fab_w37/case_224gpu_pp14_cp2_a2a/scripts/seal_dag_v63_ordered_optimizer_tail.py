#!/usr/bin/env python3
"""Seal DAG v6.3 before evaluator-only target timing access."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    REPO / "case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36"
    / "dag_v63_ordered_optimizer_tail_256_to_224"
)
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v63_ordered_optimizer_tail_2026w36.toml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = args.config.resolve()
    builder = REPO / "case_224gpu_pp14_cp2_a2a/scripts/build_dag_v63_ordered_optimizer_tail.py"
    artifacts = [
        run / "predictions/method_predictions.csv",
        run / "predictions/dag_v63_nodes.csv.gz",
        run / "predictions/dag_v63_edges.csv.gz",
        run / "predictions/dag_v63_critical_path.csv",
        run / "calibration/optimizer_tail_program_gaps.json",
        run / "optimizer_tail_order_audit.json",
        run / "prediction_contract.json",
        run / "input_access_audit.json",
        run / "provenance.json",
        run / "DAG_V63_ORDERED_OPTIMIZER_TAIL_REPORT.md",
        run / "dag_v63_optimizer_tail_order.html",
        config,
        builder,
    ]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    access = json.loads((run / "input_access_audit.json").read_text(encoding="utf-8"))
    contract = json.loads((run / "prediction_contract.json").read_text(encoding="utf-8"))
    order = json.loads((run / "optimizer_tail_order_audit.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run / "predictions/method_predictions.csv")
    if access.get("status") != "PASS_NO_TARGET_TIMING" or int(access.get("target_timing_files_read", -1)) != 0:
        raise ValueError("v6.3 input-access audit is not target-safe")
    if contract.get("target_timing_read") is not False or contract.get("accuracy_claim_allowed") is not False:
        raise ValueError("v6.3 contract violates pre-evaluation boundary")
    if order.get("status") != "PASS" or int(order.get("failure_count", -1)) != 0:
        raise ValueError("v6.3 optimizer-tail ordering is not valid")
    if not predictions["target_timing_read"].eq(False).all():
        raise ValueError("v6.3 prediction rows report target access")

    seal_path = run / "predictions/prediction_seal.json"
    payload = {
        "schema": "dag-v6.3-prediction-seal-v1",
        "status": "SEALED_BEFORE_V63_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_status": contract["status"],
        "target_timing_opened_by_sealer": False,
        "parameter_mutation_after_seal": "FORBIDDEN",
        "optimizer_tail_order_audit": "PASS",
        "artifacts": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifacts
        ],
    }
    atomic_json(seal_path, payload)
    print(json.dumps({"status": payload["status"], "seal": str(seal_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

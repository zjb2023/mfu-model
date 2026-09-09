#!/usr/bin/env python3
"""Seal the frozen DAG v6.2 prediction before evaluator-only target access."""

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
    REPO
    / "case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36"
    / "dag_v62_cost_bound_256_to_224"
)
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v62_cost_bound_prediction_2026w36.toml"


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
    prediction = run / "predictions/method_predictions.csv"
    access_path = run / "input_access_audit.json"
    contract_path = run / "prediction_contract.json"
    artifacts = [
        prediction,
        run / "predictions/dag_v62_nodes.csv.gz",
        run / "predictions/dag_v62_edges.csv.gz",
        run / "predictions/dag_v62_critical_path.csv",
        run / "model_inputs/cost_binding_table.csv.gz",
        contract_path,
        access_path,
        run / "provenance.json",
        config,
        REPO / "case_224gpu_pp14_cp2_a2a/scripts/build_dag_v62_cost_bound_prediction.py",
    ]
    transfer = run / "model_inputs/local_noncompute_layer_transfer.csv"
    if transfer.is_file():
        artifacts.append(transfer)
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    access = json.loads(access_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction)
    if access.get("target_timing_files_read") != 0:
        raise ValueError("prediction process accessed target timing")
    if access.get("status") != "PASS_NO_TARGET_TIMING":
        raise ValueError("prediction input-access audit is not target-safe")
    if contract.get("target_timing_read") is not False:
        raise ValueError("prediction contract does not exclude target timing")
    if contract.get("accuracy_claim_allowed") is not False:
        raise ValueError("unevaluated prediction already claims accuracy")
    if not predictions["target_timing_read"].eq(False).all():
        raise ValueError("prediction rows are not target-safe")
    if int(contract["binding_stats"]["used_physical_parameter_count"]) != int(
        contract["binding_stats"]["physical_parameter_count"]
    ):
        raise ValueError("not all physical parameters were bound")

    seal_path = run / "predictions/prediction_seal.json"
    payload = {
        "schema": "dag-v6.2-prediction-seal-v1",
        "status": "SEALED_BEFORE_V62_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_status": contract["status"],
        "target_timing_opened_by_sealer": False,
        "parameter_mutation_after_seal": "FORBIDDEN",
        "artifacts": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }
    atomic_json(seal_path, payload)
    print(json.dumps({"status": payload["status"], "seal": str(seal_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

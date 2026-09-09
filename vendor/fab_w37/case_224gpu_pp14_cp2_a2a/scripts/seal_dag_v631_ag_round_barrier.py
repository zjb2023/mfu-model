#!/usr/bin/env python3
"""Seal DAG v6.3.1 before evaluator-only target access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from build_dag_v63_ordered_optimizer_tail import atomic_json, sha256


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    REPO / "case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36"
    / "dag_v631_ag_round_barrier_256_to_224"
)
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v631_ag_round_barrier_2026w36.toml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = args.config.resolve()
    builder = REPO / "case_224gpu_pp14_cp2_a2a/scripts/build_dag_v631_ag_round_barrier.py"
    artifacts = [
        run / "predictions/method_predictions.csv", run / "predictions/dag_v631_nodes.csv.gz",
        run / "predictions/dag_v631_edges.csv.gz", run / "predictions/dag_v631_critical_path.csv",
        run / "calibration/ag_round_barrier_parameters.json", run / "ag_round_barrier_audit.json",
        run / "prediction_contract.json", run / "input_access_audit.json", run / "provenance.json",
        run / "DAG_V631_AG_ROUND_BARRIER_REPORT.md", config, builder,
    ]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    access = json.loads((run / "input_access_audit.json").read_text())
    contract = json.loads((run / "prediction_contract.json").read_text())
    audit = json.loads((run / "ag_round_barrier_audit.json").read_text())
    predictions = pd.read_csv(run / "predictions/method_predictions.csv")
    if access.get("status") != "PASS_NO_TARGET_TIMING" or int(access.get("target_timing_files_read", -1)) != 0:
        raise ValueError("v6.3.1 model input was not target-safe")
    if contract.get("target_timing_read") is not False or contract.get("accuracy_claim_allowed") is not False:
        raise ValueError("v6.3.1 prediction contract violates seal boundary")
    if audit.get("status") != "PASS" or int(audit.get("barrier_count", -1)) != 14:
        raise ValueError("v6.3.1 barrier audit failed")
    if not predictions["target_timing_read"].eq(False).all():
        raise ValueError("v6.3.1 prediction rows report target access")
    payload = {
        "schema": "dag-v6.3.1-prediction-seal-v1",
        "status": "SEALED_BEFORE_V631_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_status": contract["status"], "target_timing_opened_by_sealer": False,
        "parameter_mutation_after_seal": "FORBIDDEN", "ag_round_barrier_audit": "PASS",
        "artifacts": [{"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts],
    }
    seal = run / "predictions/prediction_seal.json"
    atomic_json(seal, payload)
    print(json.dumps({"status": payload["status"], "seal": str(seal)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

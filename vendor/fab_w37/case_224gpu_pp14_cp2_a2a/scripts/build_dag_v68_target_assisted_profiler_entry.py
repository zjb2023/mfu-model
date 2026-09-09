#!/usr/bin/env python3
"""Build target-assisted DAG v6.8 with an explicit pre-F/B profiler-entry node."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_dag_v67_microbatch_runtime_shape import replay


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v68_target_assisted_profiler_entry_2026w36.toml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def configured_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def verify_v67_seal(path: Path) -> dict[str, Any]:
    seal = json.loads(checked(path).read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_BEFORE_V67_EVALUATOR_ACCESS":
        raise ValueError("v6.7 prediction seal status changed")
    for artifact in seal["artifacts"]:
        source = checked(Path(artifact["path"]))
        if source.stat().st_size != int(artifact["size_bytes"]) or sha256(source) != artifact["sha256"]:
            raise ValueError(f"sealed v6.7 artifact changed: {source}")
    return seal


def metric(frame: pd.DataFrame, predicted: str, actual: str) -> dict[str, float]:
    error = frame[predicted] - frame[actual]
    ape = error.abs() / frame[actual] * 100.0
    return {
        "count": int(len(frame)),
        "mae_ms": float(error.abs().mean()),
        "bias_ms": float(error.mean()),
        "mape_pct": float(ape.mean()),
        "ape_p50_pct": float(ape.median()),
        "max_ape_pct": float(ape.max()),
    }


def build_entry_node(columns: list[str], template: pd.Series, node_id: str, duration_ns: int) -> dict[str, Any]:
    row = {column: template.get(column, "") for column in columns}
    row.update({
        "node_id": node_id,
        "kind": "framework_entry",
        "rank": -1,
        "pp_stage": -1,
        "pp_lane": -1,
        "phase": "ITER",
        "microbatch": -1,
        "schedule_region": "profiler_entry",
        "operation_sequence": -1,
        "layer_id": -1,
        "stage_local_layer": -1,
        "layer_type": "framework_boundary",
        "autograd_phase": "none",
        "op_name": "profiler_entry_to_first_phase",
        "op_family": "framework_orchestration",
        "stream": "host",
        "resource": "framework_scheduler",
        "parallelism": "world",
        "dtype": "none",
        "shape_key": "",
        "payload_key": "",
        "flops_formula": "0",
        "bytes_formula": "0",
        "cost_status": "TARGET_ASSISTED_CALIBRATED",
        "cost_source": "target224_iterations_60_80_median",
        "code_anchor": "ProfilerStep_to_first_forward_step_boundary",
        "template_index": -1,
        "duration_ns": duration_ns,
        "compute_exposed_ns_model": 0,
        "compute_overlap_ns_model": 0,
        "network_service_ns_model": 0,
        "software_sync_ns_model": 0,
        "framework_residual_ns_model": duration_ns,
        "timing_component": "profiler_entry_to_first_phase",
        "timing_source": "target224_calibration_median",
        "semantic_slot": "PROFILER_ENTRY",
        "source_parameter_key": "target224:profiler_entry_to_phase:median",
        "source_v54_node_id": "",
        "predicted_start_ns": 0,
        "predicted_end_ns": duration_ns,
        "critical_predecessor": "",
        "on_critical_path": False,
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    calibration_iterations = tuple(int(value) for value in config["split"]["target_calibration"])
    pseudo_iterations = tuple(int(value) for value in config["split"]["retrospective_pseudo_holdout"])
    if calibration_iterations != (60, 65, 70, 75, 80) or pseudo_iterations != (85, 90, 95, 100):
        raise ValueError("v6.8 target-assisted split changed")

    v67 = configured_path(config["inputs"]["v67_run_dir"])
    v67_seal_path = checked(v67 / "predictions/prediction_seal.json")
    verify_v67_seal(v67_seal_path)
    v67_nodes_path = checked(v67 / "predictions/dag_v67_nodes.csv.gz")
    v67_edges_path = checked(v67 / "predictions/dag_v67_edges.csv.gz")
    v67_contract_path = checked(v67 / "prediction_contract.json")
    parameter_path = checked(configured_path(config["inputs"]["profiler_entry_parameter"]))
    calibration_access_path = checked(configured_path(config["inputs"]["calibration_access_audit"]))
    calibration_samples_path = checked(configured_path(config["inputs"]["calibration_samples"]))
    parameter = json.loads(parameter_path.read_text(encoding="utf-8"))
    calibration_access = json.loads(calibration_access_path.read_text(encoding="utf-8"))
    samples = pd.read_csv(calibration_samples_path)
    if tuple(parameter["calibration_iterations"]) != calibration_iterations:
        raise ValueError("profiler-entry parameter calibration split changed")
    if parameter.get("not_final_step_residual") is not True:
        raise ValueError("v6.8 parameter is not a physical-boundary measurement")
    if calibration_access.get("status") != "PASS_TARGET_CALIBRATION_ROWS_ONLY":
        raise ValueError("v6.8 calibration access audit failed")
    if calibration_access.get("retrospective_pseudo_holdout_iterations_used_for_parameter") != []:
        raise ValueError("pseudo-holdout leaked into v6.8 parameter")
    if tuple(samples["iteration"].astype(int)) != calibration_iterations:
        raise ValueError("v6.8 calibration samples changed")
    entry_ms = float(parameter["profiler_entry_to_phase_ms"])
    if abs(entry_ms - float(samples["actual_profiler_entry_to_phase_ms"].median())) > 1e-9:
        raise ValueError("v6.8 profiler-entry parameter is not the calibration median")
    entry_ns = int(round(entry_ms * 1e6))

    nodes = pd.read_csv(v67_nodes_path, low_memory=False)
    edges = pd.read_csv(v67_edges_path, low_memory=False)
    entry_id = str(config["model"]["entry_node_id"])
    if entry_id in set(nodes["node_id"].astype(str)):
        raise ValueError("v6.8 entry node already exists")
    roots = nodes[~nodes["node_id"].isin(set(edges["dst"].astype(str)))].copy()
    if len(roots) != 16 or not roots["node_id"].str.match(r"^r(?:[0-9]|1[0-5]):s0:l(?:[0-9]|1[0-5]):fwd0:q0:start$").all():
        raise ValueError(f"v6.7 PP0 root contract changed: {len(roots)}")
    entry = build_entry_node(list(nodes.columns), roots.iloc[0], entry_id, entry_ns)
    nodes = pd.concat([nodes, pd.DataFrame([entry], columns=nodes.columns)], ignore_index=True)
    entry_edges = pd.DataFrame([
        {
            "case_id": str(config["target"]["case_id"]),
            "src": entry_id,
            "dst": str(root.node_id),
            "edge_type": "profiler_entry_dependency",
            "tensor_key": "none",
            "dependency_source": "target_assisted_trace_boundary",
        }
        for root in roots.itertuples(index=False)
    ], columns=edges.columns)
    edges = pd.concat([edges, entry_edges], ignore_index=True)
    nodes, critical = replay(nodes, edges, "iteration:completion_join")
    completion_ns = int(nodes.loc[
        nodes["node_id"].eq("iteration:completion_join"), "predicted_end_ns"
    ].item())
    if completion_ns <= entry_ns:
        raise ValueError("v6.8 completion did not include profiler-entry node")
    if not bool(nodes.loc[nodes["node_id"].eq(entry_id), "on_critical_path"].item()):
        raise ValueError("v6.8 profiler-entry node is not on the critical path")
    components = nodes[[
        "compute_exposed_ns_model", "compute_overlap_ns_model", "network_service_ns_model",
        "software_sync_ns_model", "framework_residual_ns_model",
    ]].fillna(0).sum(axis=1).round().astype("int64")
    if not components.eq(nodes["duration_ns"].astype("int64")).all():
        raise ValueError("v6.8 component conservation failed")

    v67_contract = json.loads(v67_contract_path.read_text(encoding="utf-8"))
    v67_prediction = v67_contract["prediction"]
    v67_raw_ms = float(v67_prediction["raw_graph_ms"])
    raw_ms = completion_ns / 1e6
    if abs(raw_ms - v67_raw_ms - entry_ms) > 0.002:
        raise ValueError("v6.8 graph delta is not exactly the profiler-entry node")
    post_graph_reconciliation_ms = (
        float(v67_prediction["profiler_step_ms"]) - v67_raw_ms
    )
    profiler_ms = raw_ms + post_graph_reconciliation_ms
    outer_framework_ms = float(v67_prediction["outer_framework_ms"])
    training_ms = profiler_ms + outer_framework_ms
    mfu_pct = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"])
        * 1e12
        * training_ms / 1000.0
    )

    output = configured_path(config["outputs"]["output_dir"])
    predictions_dir = output / "predictions"
    evaluator = output / "evaluator_only"
    calibration = output / "calibration"
    logs = output / "logs"
    for directory in (predictions_dir, evaluator, calibration, logs):
        directory.mkdir(parents=True, exist_ok=True)
    node_path = predictions_dir / "dag_v68_nodes.csv.gz"
    edge_path = predictions_dir / "dag_v68_edges.csv.gz"
    critical_path = predictions_dir / "dag_v68_critical_path.csv"
    prediction_path = predictions_dir / "method_predictions.csv"
    contract_path = output / "prediction_contract.json"
    access_path = output / "input_access_audit.json"
    calibration_copy = calibration / "profiler_entry_parameter.json"
    calibration_samples_copy = calibration / "profiler_entry_samples_60_80.csv"
    atomic_csv(node_path, nodes, compression="gzip")
    atomic_csv(edge_path, edges, compression="gzip")
    atomic_csv(critical_path, critical)
    prediction_rows = pd.DataFrame([
        {
            "iteration": iteration,
            "split": "target_calibration" if iteration in calibration_iterations else "retrospective_pseudo_holdout",
            "method": "DAG v6.8 target-assisted profiler entry",
            "predicted_profiler_step_ms": profiler_ms,
            "predicted_outer_framework_ms": outer_framework_ms,
            "predicted_training_step_ms": training_ms,
            "predicted_mfu_pct": mfu_pct,
            "target_timing_read": True,
            "target_parameter_scope": "profiler_entry_to_first_phase_only",
        }
        for iteration in (*calibration_iterations, *pseudo_iterations)
    ])
    atomic_csv(prediction_path, prediction_rows)
    atomic_json(contract_path, {
        "schema": "dag-v6.8-target-assisted-profiler-entry-contract-v1",
        "status": "TARGET_ASSISTED_PREDICTION_UNEVALUATED",
        "accuracy_claim_allowed": False,
        "blind_extrapolation_claim_allowed": False,
        "target_timing_read": True,
        "target_timing_scope": "iterations 60-80 profiler-entry boundary only",
        "final_step_residual_fit": False,
        "entry_node": {
            "node_id": entry_id,
            "duration_ms": entry_ms,
            "root_edge_count": int(len(entry_edges)),
            "timing_component": "profiler_entry_to_first_phase",
        },
        "prediction": {
            "raw_graph_ms": raw_ms,
            "post_graph_reconciliation_ms": post_graph_reconciliation_ms,
            "profiler_step_ms": profiler_ms,
            "outer_framework_ms": outer_framework_ms,
            "training_step_ms": training_ms,
            "mfu_pct": mfu_pct,
        },
    })
    atomic_json(access_path, {
        "schema": "dag-v6.8-target-assisted-input-access-v1",
        "status": "PASS_DECLARED_TARGET_ASSISTED_CALIBRATION",
        "source_model": "sealed DAG v6.7",
        "target_timing_read": True,
        "allowed_target_iterations": list(calibration_iterations),
        "used_target_iterations": list(calibration_iterations),
        "pseudo_holdout_iterations_read_before_seal": [],
        "target_fields_read": ["profiler_start_ns", "first_F/B_phase_start_ns"],
        "final_step_residual_read": False,
    })
    atomic_json(calibration_copy, parameter)
    atomic_csv(calibration_samples_copy, samples)

    seal_path = predictions_dir / "prediction_seal.json"
    sealed_artifacts = [
        node_path, edge_path, critical_path, prediction_path, contract_path,
        access_path, calibration_copy, calibration_samples_copy,
    ]
    atomic_json(seal_path, {
        "schema": "dag-v6.8-target-assisted-prediction-seal-v1",
        "status": "TARGET_ASSISTED_SEALED_BEFORE_RETROSPECTIVE_PSEUDO_HOLDOUT_EVALUATION",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sealed_artifacts
        ],
    })

    # Retrospective evaluator access begins only after the target-assisted prediction seal.
    truth_path = checked(configured_path(config["inputs"]["retrospective_truth"]))
    truth = pd.read_csv(truth_path)
    evaluation = prediction_rows.merge(
        truth[["iteration", "actual_profiler_step_ms", "actual_training_step_ms"]],
        on="iteration",
        validate="one_to_one",
    )
    evaluation["profiler_error_ms"] = (
        evaluation["predicted_profiler_step_ms"] - evaluation["actual_profiler_step_ms"]
    )
    evaluation["profiler_abs_error_pct"] = (
        evaluation["profiler_error_ms"].abs() / evaluation["actual_profiler_step_ms"] * 100.0
    )
    evaluation["training_error_ms"] = (
        evaluation["predicted_training_step_ms"] - evaluation["actual_training_step_ms"]
    )
    evaluation["training_abs_error_pct"] = (
        evaluation["training_error_ms"].abs() / evaluation["actual_training_step_ms"] * 100.0
    )
    evaluation_path = evaluator / "iteration_evaluation.csv"
    metrics_path = evaluator / "metrics.json"
    evaluation_access_path = evaluator / "evaluation_access_audit.json"
    atomic_csv(evaluation_path, evaluation)
    pseudo = evaluation[evaluation["split"].eq("retrospective_pseudo_holdout")]
    calibration_eval = evaluation[evaluation["split"].eq("target_calibration")]
    metrics = {
        "schema": "dag-v6.8-target-assisted-retrospective-evaluation-v1",
        "status": "PASS_RETROSPECTIVE_PSEUDO_HOLDOUT",
        "scope": "target-assisted development metric; target case was already inspected by humans",
        "blind_extrapolation_claim_allowed": False,
        "parameter_updates_during_evaluation": 0,
        "target_calibration": {
            "profiler": metric(calibration_eval, "predicted_profiler_step_ms", "actual_profiler_step_ms"),
            "training": metric(calibration_eval, "predicted_training_step_ms", "actual_training_step_ms"),
        },
        "retrospective_pseudo_holdout": {
            "profiler": metric(pseudo, "predicted_profiler_step_ms", "actual_profiler_step_ms"),
            "training": metric(pseudo, "predicted_training_step_ms", "actual_training_step_ms"),
        },
        "comparison": {
            "v67_validation_profiler_mape_pct": 11.14314621034797,
            "v68_pseudo_holdout_profiler_mape_pct": metric(
                pseudo, "predicted_profiler_step_ms", "actual_profiler_step_ms"
            )["mape_pct"],
            "comparison_limitation": "different retrospective subsets; diagnostic direction only",
        },
    }
    atomic_json(metrics_path, metrics)
    atomic_json(evaluation_access_path, {
        "schema": "dag-v6.8-retrospective-evaluation-access-v1",
        "status": "PSEUDO_HOLDOUT_OPENED_AFTER_TARGET_ASSISTED_SEAL",
        "seal_path": str(seal_path.resolve()),
        "pseudo_holdout_iterations": list(pseudo_iterations),
        "parameter_updates": 0,
    })

    report_path = output / "DAG_V68_TARGET_ASSISTED_PROFILER_ENTRY.md"
    reproduction_path = output / "reproduction_command.txt"
    build_log_path = logs / "build.log"
    atomic_text(report_path, f"""# DAG v6.8 target-assisted Profiler入口节点

- 状态：`{metrics['status']}`
- 目标辅助校准：224卡iterations `{list(calibration_iterations)}`
- 回顾性伪留出：iterations `{list(pseudo_iterations)}`
- 显式入口节点：`{entry_ms:.6f} ms`
- v6.7 Profiler预测：`{float(v67_prediction['profiler_step_ms']):.6f} ms`
- v6.8 Profiler预测：`{profiler_ms:.6f} ms`
- 伪留出Profiler MAPE：`{metrics['retrospective_pseudo_holdout']['profiler']['mape_pct']:.6f}%`

v6.8没有在最终Step上追加误差残差。它测量Profiler全rank最早开始到首个F/B
16-rank包络开始的边界时间，并在16个PP0 lane根节点之前加入一个可见DAG节点。

这是目标场景辅助校准后的开发结果，不是256到224的盲外推。下一步应处理仍然欠估的
F/B 16-rank包络和DP RS rank到达传播，而不是继续增大OISA service。
""")
    command = (
        ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
        "build_dag_v68_target_assisted_profiler_entry.py --config "
        "case_224gpu_pp14_cp2_a2a/config/dag_v68_target_assisted_profiler_entry_2026w36.toml\n"
    )
    atomic_text(reproduction_path, command)
    atomic_text(build_log_path, "\n".join([
        "status=PASS_RETROSPECTIVE_PSEUDO_HOLDOUT",
        f"entry_ms={entry_ms:.6f}",
        f"v67_profiler_ms={float(v67_prediction['profiler_step_ms']):.6f}",
        f"v68_profiler_ms={profiler_ms:.6f}",
        f"pseudo_holdout_profiler_mape_pct={metrics['retrospective_pseudo_holdout']['profiler']['mape_pct']:.6f}",
        "blind_extrapolation_claim_allowed=false",
        "final_step_residual_fit=false",
    ]) + "\n")

    provenance_path = output / "provenance.json"
    inputs = [
        config_path, v67_seal_path, v67_nodes_path, v67_edges_path, v67_contract_path,
        parameter_path, calibration_access_path, calibration_samples_path, truth_path,
    ]
    outputs = [
        *sealed_artifacts, seal_path, evaluation_path, metrics_path,
        evaluation_access_path, report_path, reproduction_path, build_log_path,
    ]
    atomic_json(provenance_path, {
        "schema": "dag-v6.8-target-assisted-provenance-v1",
        "status": metrics["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
    })
    print(json.dumps({
        "status": metrics["status"],
        "output_dir": str(output),
        "entry_ms": entry_ms,
        "predicted_profiler_ms": profiler_ms,
        "pseudo_holdout_profiler_mape_pct": metrics["retrospective_pseudo_holdout"]["profiler"]["mape_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

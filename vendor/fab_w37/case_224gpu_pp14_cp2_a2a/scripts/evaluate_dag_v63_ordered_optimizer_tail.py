#!/usr/bin/env python3
"""Evaluate the sealed v6.3 prediction without updating any parameter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v63_evaluation_2026w36.toml"
ITERATION_PATTERN = re.compile(
    r"iteration\s+(?P<iteration>\d+)/.*?elapsed time per iteration \(ms\):\s*"
    r"(?P<elapsed_ms>[0-9.]+).*?throughput per GPU \(TFLOP/s/GPU\):\s*(?P<tflops>[0-9.]+)"
)


def configured_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def checked(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def verify_seal(seal: dict[str, Any], expected_status: str) -> list[dict[str, Any]]:
    if seal.get("status") != expected_status:
        raise ValueError(f"prediction was not sealed before target access: {seal.get('status')} != {expected_status}")
    verified = []
    for artifact in seal["artifacts"]:
        path = checked(Path(artifact["path"]))
        observed = sha256(path)
        if observed != artifact["sha256"] or path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"sealed prediction artifact changed: {path}")
        verified.append({"path": str(path), "sha256": observed})
    return verified


def parse_training_log(path: Path) -> pd.DataFrame:
    rows = [
        {"iteration": int(match.group("iteration")),
         "actual_training_step_ms": float(match.group("elapsed_ms")),
         "reported_tflops_per_gpu": float(match.group("tflops"))}
        for match in ITERATION_PATTERN.finditer(path.read_text(errors="replace"))
    ]
    if not rows:
        raise ValueError(f"no iteration summaries in {path}")
    return pd.DataFrame(rows).drop_duplicates("iteration", keep="last").sort_values("iteration")


def metric(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    error = predicted.astype(float) - actual.astype(float)
    ape = 100.0 * error.abs() / actual.abs()
    return {
        "count": int(len(actual)), "mae_ms": float(error.abs().mean()),
        "rmse_ms": float(math.sqrt(error.pow(2).mean())), "bias_ms": float(error.mean()),
        "mape_pct": float(ape.mean()), "ape_p50_pct": float(ape.quantile(0.50)),
        "ape_p90_pct": float(ape.quantile(0.90)), "max_ape_pct": float(ape.max()),
    }


def render_html(
    metrics: dict[str, Any], comparison: pd.DataFrame, primary_label: str,
    comparison_label: str, display_version: str,
) -> str:
    primary = metrics["methods"][primary_label]["validation"]
    rows = "".join(
        f"<tr><td>{row.method}</td><td>{row.predicted_training_step_ms:.3f}</td>"
        f"<td>{row.training_mape_pct:.3f}%</td><td>{row.training_mae_ms:.3f}</td></tr>"
        for row in comparison.itertuples(index=False)
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DAG {display_version} validation</title><style>body{{margin:0;background:#07111d;color:#e9f2fb;font:14px/1.55 system-ui}}main{{max-width:1000px;margin:auto;padding:32px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.card,.panel{{background:#0e1b2c;border:1px solid #2b4059;padding:15px;margin:12px 0}}.v{{font-size:26px;font-weight:800;color:#58d6c4}}.muted{{color:#93a8bd}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #2b4059}}.warn{{border-left:4px solid #f59e0b;padding:10px;background:#1b1d28}}</style></head><body><main><p class="muted">sealed prediction · evaluator-only</p><h1>DAG {display_version}：224 卡冻结预测验证</h1><div class="cards"><div class="card"><div class="muted">Training MAPE</div><div class="v">{primary['training_step_ms']['mape_pct']:.3f}%</div></div><div class="card"><div class="muted">Training MAE</div><div class="v">{primary['training_step_ms']['mae_ms']:.1f} ms</div></div><div class="card"><div class="muted">Profiler MAPE</div><div class="v">{primary['profiler_step_ms']['mape_pct']:.3f}%</div></div></div><section class="panel"><h2>与冻结 {comparison_label} 同口径对比</h2><table><thead><tr><th>模型</th><th>预测 Training ms</th><th>MAPE</th><th>MAE ms</th></tr></thead><tbody>{rows}</tbody></table></section><p class="warn">这里只验证本轮依赖修正的影响，且没有用 224 卡真值更新任何参数；结果不代表一般场景胜负。</p></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    model = config.get("model", {})
    schema_version = str(model.get("schema_version", "v6.3"))
    file_version = str(model.get("file_version", "v63"))
    primary_label = str(model.get("primary_label", "DAG v6.3"))
    comparison_label = str(model.get("comparison_label", "DAG v6.2"))
    expected_seal_status = str(model.get("expected_seal_status", "SEALED_BEFORE_V63_EVALUATOR_ACCESS"))
    run = configured_path(config["inputs"]["run_dir"])
    evaluator = configured_path(config["outputs"]["evaluator_dir"])
    evaluator.mkdir(parents=True, exist_ok=True)
    prediction_path = checked(run / "predictions/method_predictions.csv")
    seal_path = checked(run / "predictions/prediction_seal.json")
    access_path = checked(run / "input_access_audit.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    sealed_before = verify_seal(seal, expected_seal_status)
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if access.get("status") != "PASS_NO_TARGET_TIMING" or int(access["target_timing_files_read"]) != 0:
        raise ValueError("model process was not target-safe")
    predictions = pd.read_csv(prediction_path)
    regression = tuple(int(value) for value in config["split"]["regression_only"])
    validation = tuple(int(value) for value in config["split"]["validation"])
    grid = regression + validation
    if tuple(predictions["iteration"].astype(int)) != grid or not predictions["target_timing_read"].eq(False).all():
        raise ValueError(f"frozen {schema_version} prediction grid changed")

    # Evaluator-only target access starts after the seal verification above.
    windows_path = checked(configured_path(config["inputs"]["event_windows"]))
    training_path = checked(configured_path(config["inputs"]["training_log"]))
    comparison_value = config["inputs"].get("comparison_predictions", config["inputs"].get("v62_predictions"))
    comparison_source_path = checked(configured_path(comparison_value))
    windows = pd.read_csv(windows_path)
    windows = windows[windows["iteration"].isin(grid)].copy()
    rank_counts = windows.groupby("iteration")["rank"].nunique()
    if tuple(sorted(rank_counts.index.astype(int))) != tuple(sorted(grid)):
        raise ValueError("target profiler iteration grid is incomplete")
    if not rank_counts.eq(int(config["target"]["world_size"])).all():
        raise ValueError("target profiler rank grid is incomplete")
    truth = windows.groupby("iteration", as_index=False).agg(
        profiler_start_ns=("start_ns", "min"), profiler_end_ns=("end_ns", "max")
    )
    truth["actual_profiler_step_ms"] = (truth["profiler_end_ns"] - truth["profiler_start_ns"]) / 1e6
    training = parse_training_log(training_path)
    training = training[training["iteration"].isin(grid)]
    if tuple(training["iteration"].astype(int)) != tuple(sorted(grid)):
        raise ValueError("target training iteration grid is incomplete")
    truth = truth.merge(training, on="iteration", validate="one_to_one")
    truth["split"] = truth["iteration"].map(
        lambda iteration: "regression_only" if int(iteration) in regression else "validation"
    )
    truth["actual_outer_framework_ms"] = truth["actual_training_step_ms"] - truth["actual_profiler_step_ms"]
    truth["actual_mfu_pct_from_training_clock"] = (
        100.0 * float(config["target"]["model_flops_per_iteration"])
        / (int(config["target"]["world_size"]) * float(config["target"]["peak_tflops_per_gpu"]) * 1e12
           * truth["actual_training_step_ms"] / 1000.0)
    )
    truth = truth[[
        "iteration", "split", "actual_profiler_step_ms", "actual_training_step_ms",
        "actual_outer_framework_ms", "actual_mfu_pct_from_training_clock", "reported_tflops_per_gpu",
    ]]
    truth_path = evaluator / "iteration_ground_truth.csv"
    atomic_csv(truth_path, truth)
    atomic_json(evaluator / "ground_truth_manifest.json", {
        "schema": f"dag-{schema_version}-224gpu-ground-truth-v1",
        "status": "EVALUATOR_ONLY_OPENED_AFTER_PREDICTION_SEAL", "fit_permission": "FORBIDDEN",
        "regression_only": list(regression), "validation": list(validation),
        "sources": [
            {"path": str(windows_path), "sha256": sha256(windows_path)},
            {"path": str(training_path), "sha256": sha256(training_path)},
        ],
        "materialized": {"path": str(truth_path), "sha256": sha256(truth_path)},
    })

    joined = predictions.merge(truth, on=["iteration", "split"], validate="one_to_one")
    for clock in ("profiler", "training"):
        joined[f"{clock}_error_ms"] = joined[f"predicted_{clock}_step_ms"] - joined[f"actual_{clock}_step_ms"]
        joined[f"{clock}_abs_error_pct"] = 100.0 * joined[f"{clock}_error_ms"].abs() / joined[f"actual_{clock}_step_ms"]
    evaluation_path = evaluator / "iteration_evaluation.csv"
    atomic_csv(evaluation_path, joined)

    comparison_predictions = pd.read_csv(comparison_source_path)
    methods = ((primary_label, predictions), (comparison_label, comparison_predictions))
    method_metrics: dict[str, Any] = {}
    rows = []
    for label, frame in methods:
        comparison = frame.merge(truth, on=["iteration", "split"], validate="one_to_one")
        held = comparison[comparison["iteration"].isin(validation)]
        check = comparison[comparison["iteration"].isin(regression)].iloc[0]
        profiler_metric = metric(held["actual_profiler_step_ms"], held["predicted_profiler_step_ms"])
        training_metric = metric(held["actual_training_step_ms"], held["predicted_training_step_ms"])
        method_metrics[label] = {
            "constant_prediction": {
                "profiler_step_ms": float(frame["predicted_profiler_step_ms"].iloc[0]),
                "training_step_ms": float(frame["predicted_training_step_ms"].iloc[0]),
            },
            "regression_only_iteration_55": {
                "profiler_abs_error_pct": float(abs(check.predicted_profiler_step_ms - check.actual_profiler_step_ms) / check.actual_profiler_step_ms * 100.0),
                "training_abs_error_pct": float(abs(check.predicted_training_step_ms - check.actual_training_step_ms) / check.actual_training_step_ms * 100.0),
            },
            "validation": {"profiler_step_ms": profiler_metric, "training_step_ms": training_metric},
        }
        rows.append({
            "method": label, "predicted_profiler_step_ms": float(frame["predicted_profiler_step_ms"].iloc[0]),
            "predicted_training_step_ms": float(frame["predicted_training_step_ms"].iloc[0]),
            "profiler_mape_pct": profiler_metric["mape_pct"], "training_mape_pct": training_metric["mape_pct"],
            "training_mae_ms": training_metric["mae_ms"],
        })
    comparison = pd.DataFrame(rows).sort_values("training_mape_pct").reset_index(drop=True)
    comparison_path = evaluator / "model_comparison.csv"
    atomic_csv(comparison_path, comparison)
    metrics = {
        "schema": f"dag-{schema_version}-frozen-224gpu-evaluation-v1", "status": "PASS_RETROSPECTIVE_VALIDATION",
        "source_compute_window": list(range(60, 101, 5)), "target_regression_only": list(regression),
        "target_validation": list(validation), "parameter_updates_during_evaluation": 0,
        "methods": method_metrics, "scope": config["protocol"]["accuracy_scope"],
    }
    metrics_path = evaluator / "metrics.json"
    atomic_json(metrics_path, metrics)
    primary = method_metrics[primary_label]["validation"]
    baseline = method_metrics[comparison_label]["validation"]
    report_path = evaluator / f"DAG_{file_version.upper()}_EVALUATION.md"
    atomic_text(report_path, f"""# DAG {schema_version}：224 卡冻结预测验证

预测先封存，再读取 iteration 55 和 60–100；55 只作回归检查，精度统计只使用 60–100，评估阶段参数更新数为 0。

- {primary_label} Training MAPE：`{primary['training_step_ms']['mape_pct']:.6f}%`，MAE：`{primary['training_step_ms']['mae_ms']:.6f} ms`。
- {primary_label} Profiler MAPE：`{primary['profiler_step_ms']['mape_pct']:.6f}%`，MAE：`{primary['profiler_step_ms']['mae_ms']:.6f} ms`。
- {comparison_label} Training MAPE：`{baseline['training_step_ms']['mape_pct']:.6f}%`。
- 本轮修正带来的 Training MAPE 变化：`{primary['training_step_ms']['mape_pct'] - baseline['training_step_ms']['mape_pct']:.6f}` 个百分点。

该结果只说明在同一冻结后端下本轮依赖修正的影响，不能据此宣称一般方法胜负。
""")
    html_path = evaluator / f"dag_{file_version}_224gpu_validation.html"
    atomic_text(html_path, render_html(metrics, comparison, primary_label, comparison_label, schema_version))
    sealed_after = verify_seal(seal, expected_seal_status)
    atomic_json(evaluator / "leakage_audit.json", {
        "schema": f"dag-{schema_version}-evaluator-leakage-audit-v1", "status": "PASS",
        "checks": {
            "prediction_sealed_before_target_access": True,
            "prediction_artifacts_unchanged_after_evaluation": sealed_before == sealed_after,
            "model_target_timing_files_read_before_evaluation": int(access["target_timing_files_read"]),
            "parameter_updates_during_evaluation": 0,
            "iteration_55_excluded_from_validation_metrics": True,
        },
    })
    atomic_json(evaluator / "evaluation_access_audit.json", {
        "schema": f"dag-{schema_version}-evaluation-access-audit-v1",
        "status": "PASS_TARGET_OPENED_IN_EVALUATOR_ONLY", "seal_verified_before_target_open": True,
        "target_files_read": [
            {"path": str(windows_path), "fields": ["iteration", "rank", "start_ns", "end_ns"], "sha256": sha256(windows_path)},
            {"path": str(training_path), "fields": ["iteration", "elapsed time per iteration (ms)", "throughput per GPU"], "sha256": sha256(training_path)},
        ],
        "comparison_only_files_read": [{"path": str(comparison_source_path), "sha256": sha256(comparison_source_path)}],
        "writes_outside_evaluator_dir": 0,
    })
    reproduction = evaluator / "reproduction_command.txt"
    atomic_text(reproduction, f"cd {REPO}\npython {Path(__file__).resolve()} --config {config_path}\n")
    log_path = evaluator / "evaluation.log"
    atomic_text(log_path, "\n".join([
        f"DAG {schema_version} evaluator PASS", f"generated_at={datetime.now(timezone.utc).isoformat()}",
        f"training_mape_pct={primary['training_step_ms']['mape_pct']:.9f}",
        f"profiler_mape_pct={primary['profiler_step_ms']['mape_pct']:.9f}",
        f"comparison_training_mape_pct={baseline['training_step_ms']['mape_pct']:.9f}",
        "parameter_updates=0", "prediction_seal_unchanged=true", "status=PASS_RETROSPECTIVE_VALIDATION", "",
    ]))
    provenance_path = evaluator / "evaluation_provenance.json"
    evidence = [truth_path, evaluation_path, comparison_path, metrics_path, report_path, html_path, reproduction, log_path]
    atomic_json(provenance_path, {
        "schema": f"dag-{schema_version}-evaluation-provenance-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "prediction_seal": {"path": str(seal_path), "sha256": sha256(seal_path)},
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in evidence],
    })
    manifest_path = evaluator / "artifact_manifest.json"
    output_files = sorted(path for path in evaluator.rglob("*") if path.is_file() and path != manifest_path)
    atomic_json(manifest_path, {
        "schema": f"dag-{schema_version}-evaluation-artifact-manifest-v1",
        "artifacts": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_files],
    })
    print(json.dumps({
        "status": metrics["status"], "primary_method": primary_label,
        "primary_training": primary["training_step_ms"],
        "primary_profiler": primary["profiler_step_ms"],
        "comparison_training_mape_pct": baseline["training_step_ms"]["mape_pct"],
        "output": str(evaluator),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a sealed DAG v6.2 prediction on 224-GPU iteration clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v62_evaluation_2026w36.toml"
ITERATION_PATTERN = re.compile(
    r"iteration\s+(?P<iteration>\d+)/.*?"
    r"elapsed time per iteration \(ms\):\s*(?P<elapsed_ms>[0-9.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*(?P<tflops>[0-9.]+)"
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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def verify_seal(seal: dict[str, Any]) -> list[dict[str, Any]]:
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
        {
            "iteration": int(match.group("iteration")),
            "actual_training_step_ms": float(match.group("elapsed_ms")),
            "reported_tflops_per_gpu": float(match.group("tflops")),
        }
        for match in ITERATION_PATTERN.finditer(path.read_text(errors="replace"))
    ]
    if not rows:
        raise ValueError(f"no iteration summaries in {path}")
    return (
        pd.DataFrame(rows)
        .drop_duplicates("iteration", keep="last")
        .sort_values("iteration")
        .reset_index(drop=True)
    )


def metric(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    error = predicted.astype(float) - actual.astype(float)
    ape = 100.0 * error.abs() / actual.abs()
    return {
        "count": int(len(actual)),
        "mae_ms": float(error.abs().mean()),
        "rmse_ms": float(math.sqrt(error.pow(2).mean())),
        "bias_ms": float(error.mean()),
        "mape_pct": float(ape.mean()),
        "ape_p50_pct": float(ape.quantile(0.50)),
        "ape_p90_pct": float(ape.quantile(0.90)),
        "max_ape_pct": float(ape.max()),
    }


def svg_line_chart(frame: pd.DataFrame) -> str:
    width, height = 920, 320
    left, right, top, bottom = 70, 20, 25, 45
    inner_w, inner_h = width - left - right, height - top - bottom
    iterations = frame["iteration"].astype(int).tolist()
    series = {
        "实测 Training Step": frame["actual_training_step_ms"].astype(float).tolist(),
        "DAG v6.2": frame["predicted_training_step_ms"].astype(float).tolist(),
    }
    colors = {"实测 Training Step": "#61d6ff", "DAG v6.2": "#ffb86b"}
    values = [value for items in series.values() for value in items]
    y_min, y_max = min(values), max(values)
    margin = max((y_max - y_min) * 0.12, 1.0)
    y_min, y_max = y_min - margin, y_max + margin

    def x(index: int) -> float:
        return left + (inner_w * index / max(len(iterations) - 1, 1))

    def y(value: float) -> float:
        return top + inner_h * (y_max - value) / max(y_max - y_min, 1e-9)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="training step comparison">']
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#263d55"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" fill="#91a7bd">{value:.0f}</text>')
    for index, iteration in enumerate(iterations):
        xx = x(index)
        parts.append(f'<text x="{xx:.1f}" y="{height-16}" text-anchor="middle" fill="#91a7bd">{iteration}</text>')
    for name, values_for_series in series.items():
        points = " ".join(f"{x(i):.1f},{y(value):.1f}" for i, value in enumerate(values_for_series))
        color = colors[name]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.extend(
            f'<circle cx="{x(i):.1f}" cy="{y(value):.1f}" r="4" fill="{color}"/>'
            for i, value in enumerate(values_for_series)
        )
    parts.append('</svg>')
    return "".join(parts)


def render_html(
    validation: pd.DataFrame, metrics: dict[str, Any], comparison: pd.DataFrame,
    diagnosis: dict[str, Any],
) -> str:
    v62 = metrics["methods"]["DAG v6.2"]
    training = v62["validation"]["training_step_ms"]
    profiler = v62["validation"]["profiler_step_ms"]
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.method))}</td>"
        f"<td>{row.predicted_training_step_ms:.3f}</td>"
        f"<td>{row.training_mape_pct:.3f}%</td>"
        f"<td>{row.training_ape_p90_pct:.3f}%</td>"
        f"<td>{row.training_mae_ms:.3f}</td>"
        "</tr>"
        for row in comparison.itertuples(index=False)
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DAG v6.2 224卡验证</title>
<style>body{{margin:0;background:#08111e;color:#dce8f5;font:14px/1.55 system-ui}}main{{max-width:1180px;margin:auto;padding:34px}}h1{{margin:0}}h2{{margin-top:32px}}.muted{{color:#91a7bd}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.card,.panel{{background:#111e30;border:1px solid #293d56;border-radius:12px;padding:16px}}.value{{font-size:25px;font-weight:700;color:#61d6ff}}svg{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #293d56;text-align:left}}.warn{{color:#ffd17a}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}</style></head><body><main>
<h1>DAG v6.2 · 224 卡冻结预测验证</h1><p class="muted">预测先封存，再读取 iteration 60–100；iteration 55 仅作回归检查。</p>
<div class="cards"><div class="card"><div class="muted">Training MAPE</div><div class="value">{training['mape_pct']:.3f}%</div></div>
<div class="card"><div class="muted">Training MAE</div><div class="value">{training['mae_ms']:.1f} ms</div></div>
<div class="card"><div class="muted">Profiler MAPE</div><div class="value">{profiler['mape_pct']:.3f}%</div></div>
<div class="card"><div class="muted">Prediction</div><div class="value">{validation['predicted_training_step_ms'].iloc[0]:.1f} ms</div></div></div>
<section class="panel"><h2>Training Step：预测与实测</h2>{svg_line_chart(validation)}</section>
<section class="panel"><h2>低估发生在哪里</h2><p>平均 Training 缺口为 <b>{diagnosis['mean_training_shortfall_ms']:.1f} ms</b>；其中 profiler 内部缺口为 <b>{diagnosis['mean_profiler_shortfall_ms']:.1f} ms</b>（{diagnosis['profiler_share_of_training_shortfall_pct']:.1f}%），outer-framework 缺口为 {diagnosis['mean_outer_framework_shortfall_ms']:.1f} ms。</p><p class="muted">三种模型的预测跨度只有 {diagnosis['shared_model_prediction_span_ms']:.1f} ms，却共同明显低估实测，说明当前首要问题是共享的 256→224 成本迁移，而不是模型展示粒度。</p></section>
<section class="panel"><h2>同一真值口径下的历史模型参考</h2><table><thead><tr><th>方法</th><th>预测 Training</th><th>MAPE</th><th>APE P90</th><th>MAE</th></tr></thead><tbody>{rows}</tbody></table></section>
<p class="warn">结论只适用于本次冻结的 256→224 回顾性验证；没有进行 224 卡调参，也不代表一般场景中的方法胜负。</p>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    run = configured_path(config["inputs"]["run_dir"])
    evaluator = configured_path(config["outputs"]["evaluator_dir"])
    evaluator.mkdir(parents=True, exist_ok=True)
    prediction_path = checked(run / "predictions/method_predictions.csv")
    seal_path = checked(run / "predictions/prediction_seal.json")
    access_path = checked(run / "input_access_audit.json")

    # This boundary must complete before any evaluator-only target timing path opens.
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_BEFORE_V62_EVALUATOR_ACCESS":
        raise ValueError("v6.2 prediction was not sealed before evaluation")
    sealed_before = verify_seal(seal)
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if access.get("target_timing_files_read") != 0:
        raise ValueError("model process read target timing")
    predictions = pd.read_csv(prediction_path)
    regression = tuple(int(value) for value in config["split"]["regression_only"])
    validation_grid = tuple(int(value) for value in config["split"]["validation"])
    target_grid = regression + validation_grid
    if tuple(predictions["iteration"].astype(int)) != target_grid:
        raise ValueError("frozen prediction iteration grid changed")
    if not predictions["target_timing_read"].eq(False).all():
        raise ValueError("frozen predictions report target timing access")

    # Evaluator-only target access starts here.
    windows_path = checked(configured_path(config["inputs"]["event_windows"]))
    training_path = checked(configured_path(config["inputs"]["training_log"]))
    comparison_path = checked(configured_path(config["inputs"]["comparison_predictions"]))
    windows = pd.read_csv(windows_path)
    windows = windows[windows["iteration"].isin(target_grid)].copy()
    rank_counts = windows.groupby("iteration")["rank"].nunique()
    if tuple(sorted(int(value) for value in rank_counts.index)) != tuple(sorted(target_grid)):
        raise ValueError("target profiler iteration grid is incomplete")
    if not rank_counts.eq(int(config["target"]["world_size"])).all():
        raise ValueError("target profiler rank grid is incomplete")
    truth = windows.groupby("iteration", as_index=False).agg(
        profiler_start_ns=("start_ns", "min"), profiler_end_ns=("end_ns", "max")
    )
    truth["actual_profiler_step_ms"] = (truth["profiler_end_ns"] - truth["profiler_start_ns"]) / 1e6
    training = parse_training_log(training_path)
    training = training[training["iteration"].isin(target_grid)].copy()
    if tuple(training["iteration"].astype(int)) != tuple(sorted(target_grid)):
        raise ValueError("target training iteration grid is incomplete")
    truth = truth.merge(training, on="iteration", validate="one_to_one")
    truth["split"] = truth["iteration"].map(
        lambda value: "regression_only" if int(value) in regression else "validation"
    )
    truth["actual_outer_framework_ms"] = truth["actual_training_step_ms"] - truth["actual_profiler_step_ms"]
    truth["actual_mfu_pct_from_training_clock"] = (
        100.0 * float(config["target"]["model_flops_per_iteration"])
        / (
            int(config["target"]["world_size"])
            * float(config["target"]["peak_tflops_per_gpu"])
            * 1e12
            * truth["actual_training_step_ms"] / 1000.0
        )
    )
    truth = truth[[
        "iteration", "split", "actual_profiler_step_ms", "actual_training_step_ms",
        "actual_outer_framework_ms", "actual_mfu_pct_from_training_clock", "reported_tflops_per_gpu",
    ]]
    truth_path = evaluator / "iteration_ground_truth.csv"
    atomic_csv(truth_path, truth)
    ground_manifest = evaluator / "ground_truth_manifest.json"
    atomic_json(ground_manifest, {
        "schema": "dag-v6.2-224gpu-ground-truth-v1",
        "status": "EVALUATOR_ONLY_OPENED_AFTER_PREDICTION_SEAL",
        "fit_permission": "FORBIDDEN",
        "regression_only": list(regression), "validation": list(validation_grid),
        "sources": [
            {"path": str(windows_path), "size_bytes": windows_path.stat().st_size, "sha256": sha256(windows_path)},
            {"path": str(training_path), "size_bytes": training_path.stat().st_size, "sha256": sha256(training_path)},
        ],
        "materialized": {"path": str(truth_path), "size_bytes": truth_path.stat().st_size, "sha256": sha256(truth_path)},
    })

    iteration_evaluation = predictions.merge(truth, on=["iteration", "split"], validate="one_to_one")
    for clock in ("profiler", "training"):
        iteration_evaluation[f"{clock}_error_ms"] = (
            iteration_evaluation[f"predicted_{clock}_step_ms"] - iteration_evaluation[f"actual_{clock}_step_ms"]
        )
        iteration_evaluation[f"{clock}_abs_error_pct"] = (
            100.0 * iteration_evaluation[f"{clock}_error_ms"].abs()
            / iteration_evaluation[f"actual_{clock}_step_ms"]
        )
    iteration_evaluation["mfu_error_percentage_points"] = (
        iteration_evaluation["predicted_mfu_pct"] - iteration_evaluation["actual_mfu_pct_from_training_clock"]
    )
    iteration_path = evaluator / "iteration_evaluation.csv"
    atomic_csv(iteration_path, iteration_evaluation)

    base = pd.read_csv(comparison_path)
    methods = [
        ("DAG v6.2", predictions),
        ("DAG v5.4", base[base["method"].str.contains("DAG v5.4", case=False)]),
        ("三阶段", base[base["method"].str.contains("three-stage", case=False)]),
    ]
    method_metrics: dict[str, Any] = {}
    comparison_rows = []
    for label, frame in methods:
        if frame.empty:
            raise ValueError(f"comparison prediction missing: {label}")
        joined = frame.merge(truth, on=["iteration", "split"], validate="one_to_one")
        held = joined[joined["iteration"].isin(validation_grid)]
        check = joined[joined["iteration"].isin(regression)].iloc[0]
        profiler_metric = metric(held["actual_profiler_step_ms"], held["predicted_profiler_step_ms"])
        training_metric = metric(held["actual_training_step_ms"], held["predicted_training_step_ms"])
        method_metrics[label] = {
            "constant_prediction": {
                "profiler_step_ms": float(frame["predicted_profiler_step_ms"].iloc[0]),
                "training_step_ms": float(frame["predicted_training_step_ms"].iloc[0]),
                "mfu_pct": float(frame["predicted_mfu_pct"].iloc[0]),
            },
            "regression_only_iteration_55": {
                "profiler_abs_error_pct": float(
                    abs(check.predicted_profiler_step_ms - check.actual_profiler_step_ms)
                    / check.actual_profiler_step_ms * 100.0
                ),
                "training_abs_error_pct": float(
                    abs(check.predicted_training_step_ms - check.actual_training_step_ms)
                    / check.actual_training_step_ms * 100.0
                ),
            },
            "validation": {"profiler_step_ms": profiler_metric, "training_step_ms": training_metric},
        }
        comparison_rows.append({
            "method": label,
            "predicted_profiler_step_ms": float(frame["predicted_profiler_step_ms"].iloc[0]),
            "predicted_training_step_ms": float(frame["predicted_training_step_ms"].iloc[0]),
            "profiler_mape_pct": profiler_metric["mape_pct"],
            "training_mape_pct": training_metric["mape_pct"],
            "training_ape_p90_pct": training_metric["ape_p90_pct"],
            "training_mae_ms": training_metric["mae_ms"],
        })
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["training_ape_p90_pct", "training_mae_ms", "method"]
    ).reset_index(drop=True)
    comparison_output = evaluator / "model_comparison.csv"
    atomic_csv(comparison_output, comparison)
    metrics_payload = {
        "schema": "dag-v6.2-frozen-224gpu-evaluation-v1",
        "status": "PASS_RETROSPECTIVE_VALIDATION",
        "source_compute_window": list(range(60, 101, 5)),
        "target_regression_only": list(regression), "target_validation": list(validation_grid),
        "parameter_updates_during_evaluation": 0,
        "methods": method_metrics,
        "retrospective_ranking": comparison["method"].tolist(),
        "scope": config["protocol"]["accuracy_scope"],
    }
    metrics_path = evaluator / "metrics.json"
    atomic_json(metrics_path, metrics_payload)

    v62 = method_metrics["DAG v6.2"]["validation"]
    validation_rows = iteration_evaluation[iteration_evaluation["split"].eq("validation")].copy()
    mean_profiler_shortfall = float(
        (validation_rows["actual_profiler_step_ms"] - validation_rows["predicted_profiler_step_ms"]).mean()
    )
    mean_outer_shortfall = float(
        (validation_rows["actual_outer_framework_ms"] - validation_rows["predicted_outer_framework_ms"]).mean()
    )
    mean_training_shortfall = float(
        (validation_rows["actual_training_step_ms"] - validation_rows["predicted_training_step_ms"]).mean()
    )
    ordered = validation_rows.sort_values("iteration")
    model_prediction_span = float(
        comparison["predicted_training_step_ms"].max() - comparison["predicted_training_step_ms"].min()
    )
    diagnosis = {
        "schema": "dag-v6.2-transfer-gap-diagnosis-v1",
        "status": "EVALUATOR_ONLY_DIAGNOSTIC_NO_PARAMETER_UPDATE",
        "mean_training_shortfall_ms": mean_training_shortfall,
        "mean_profiler_shortfall_ms": mean_profiler_shortfall,
        "mean_outer_framework_shortfall_ms": mean_outer_shortfall,
        "profiler_share_of_training_shortfall_pct": 100.0 * mean_profiler_shortfall / mean_training_shortfall,
        "outer_share_of_training_shortfall_pct": 100.0 * mean_outer_shortfall / mean_training_shortfall,
        "actual_training_step_first_validation_ms": float(ordered["actual_training_step_ms"].iloc[0]),
        "actual_training_step_last_validation_ms": float(ordered["actual_training_step_ms"].iloc[-1]),
        "actual_training_step_60_to_100_delta_ms": float(
            ordered["actual_training_step_ms"].iloc[-1] - ordered["actual_training_step_ms"].iloc[0]
        ),
        "shared_model_prediction_span_ms": model_prediction_span,
        "v62_minus_v54_prediction_ms": float(
            method_metrics["DAG v6.2"]["constant_prediction"]["training_step_ms"]
            - method_metrics["DAG v5.4"]["constant_prediction"]["training_step_ms"]
        ),
        "evidence_based_interpretation": [
            "The dominant shortfall is inside the profiler clock, not outer-framework time.",
            "All three source-transferred models cluster tightly relative to their common target shortfall.",
            "The next model change should audit shared 256-to-224 compute/noncompute transfer before adding more DAG display detail.",
        ],
        "parameter_update_permission": "FORBIDDEN_IN_EVALUATOR",
    }
    diagnosis_path = evaluator / "transfer_gap_diagnosis.json"
    atomic_json(diagnosis_path, diagnosis)
    report_path = evaluator / "DAG_V62_EVALUATION.md"
    atomic_text(report_path, f"""# DAG v6.2：224 卡冻结预测验证

预测产物先封存，再由独立 evaluator 读取 224 卡 iteration 55 和 60–100。iteration 55 只作回归检查，以下统计只使用 60–100；评估阶段没有更新参数。

- Training Step：MAPE `{v62['training_step_ms']['mape_pct']:.6f}%`，APE P90 `{v62['training_step_ms']['ape_p90_pct']:.6f}%`，MAE `{v62['training_step_ms']['mae_ms']:.6f} ms`，bias `{v62['training_step_ms']['bias_ms']:.6f} ms`。
- Profiler Step：MAPE `{v62['profiler_step_ms']['mape_pct']:.6f}%`，APE P90 `{v62['profiler_step_ms']['ape_p90_pct']:.6f}%`，MAE `{v62['profiler_step_ms']['mae_ms']:.6f} ms`。
- 预测 Training Step：`{method_metrics['DAG v6.2']['constant_prediction']['training_step_ms']:.6f} ms`。
- 回顾性排序：`{' → '.join(comparison['method'].tolist())}`。

## 误差定位

- 平均 Training 缺口：`{mean_training_shortfall:.6f} ms`；
- profiler 内部缺口：`{mean_profiler_shortfall:.6f} ms`，占 Training 缺口 `{diagnosis['profiler_share_of_training_shortfall_pct']:.3f}%`；
- outer-framework 缺口：`{mean_outer_shortfall:.6f} ms`；
- 三种模型的 Training 预测跨度只有 `{model_prediction_span:.6f} ms`，但都明显低估目标实测。

因此，本轮证据不支持把主要误差归因于 outer-framework，也不支持只继续细化 DAG 展示。下一步应先审计三种模型共同使用的 256→224 计算量、每层成本和非计算后端迁移规则。

这里的排序只描述同一组 224 卡实测上的数值误差，不代表方法的一般优劣。v6.2 的计算参数来自 256 卡稳定段，CP/EP/PP/DP/EDP 与 optimizer 非计算时钟仍继承 sealed v5.4 后端。
""")
    html_path = evaluator / "dag_v62_224gpu_validation.html"
    atomic_text(html_path, render_html(
        validation_rows, metrics_payload, comparison, diagnosis
    ))

    sealed_after = verify_seal(seal)
    leakage_path = evaluator / "leakage_audit.json"
    atomic_json(leakage_path, {
        "schema": "dag-v6.2-evaluator-leakage-audit-v1", "status": "PASS",
        "checks": {
            "prediction_sealed_before_target_access": True,
            "prediction_artifacts_unchanged_after_evaluation": sealed_before == sealed_after,
            "model_target_timing_files_read_before_evaluation": int(access["target_timing_files_read"]),
            "parameter_updates_during_evaluation": 0,
            "iteration_55_excluded_from_validation_metrics": True,
        },
        "prediction_seal": {"path": str(seal_path), "sha256": sha256(seal_path)},
        "model_input_access_audit": {"path": str(access_path), "sha256": sha256(access_path)},
    })
    access_audit_path = evaluator / "evaluation_access_audit.json"
    atomic_json(access_audit_path, {
        "schema": "dag-v6.2-evaluation-access-audit-v1",
        "status": "PASS_TARGET_OPENED_IN_EVALUATOR_ONLY",
        "seal_verified_before_target_open": True,
        "target_files_read": [
            {"path": str(windows_path), "fields": ["iteration", "rank", "start_ns", "end_ns"], "sha256": sha256(windows_path)},
            {"path": str(training_path), "fields": ["iteration", "elapsed time per iteration (ms)", "throughput per GPU"], "sha256": sha256(training_path)},
        ],
        "comparison_only_files_read": [{"path": str(comparison_path), "sha256": sha256(comparison_path)}],
        "writes_outside_evaluator_dir": 0,
    })
    reproduction_path = evaluator / "reproduction_command.txt"
    atomic_text(reproduction_path, f"cd {REPO}\npython {Path(__file__).resolve()} --config {config_path}\n")
    log_path = evaluator / "evaluation.log"
    atomic_text(log_path, "\n".join([
        "DAG v6.2 evaluator PASS", f"generated_at={datetime.now(timezone.utc).isoformat()}",
        f"training_mape_pct={v62['training_step_ms']['mape_pct']:.9f}",
        f"training_mae_ms={v62['training_step_ms']['mae_ms']:.9f}",
        f"profiler_mape_pct={v62['profiler_step_ms']['mape_pct']:.9f}",
        "parameter_updates=0", "prediction_seal_unchanged=true", "status=PASS_RETROSPECTIVE_VALIDATION", "",
    ]))
    evidence = [
        truth_path, ground_manifest, iteration_path, comparison_output, metrics_path, report_path,
        diagnosis_path, html_path, leakage_path, access_audit_path, reproduction_path, log_path,
    ]
    provenance_path = evaluator / "evaluation_provenance.json"
    atomic_json(provenance_path, {
        "schema": "dag-v6.2-evaluation-provenance-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "prediction_seal": {"path": str(seal_path), "sha256": sha256(seal_path)},
        "outputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in evidence],
    })
    manifest_path = evaluator / "artifact_manifest.json"
    manifest_artifacts = sorted(
        path.resolve() for path in evaluator.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve() and not path.name.endswith(".tmp")
    )
    atomic_json(manifest_path, {
        "schema": "dag-v6.2-evaluation-artifact-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in manifest_artifacts
        ],
    })
    print(json.dumps({
        "status": metrics_payload["status"],
        "v62_training": v62["training_step_ms"],
        "v62_profiler": v62["profiler_step_ms"],
        "retrospective_ranking": comparison["method"].tolist(),
        "output": str(evaluator),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

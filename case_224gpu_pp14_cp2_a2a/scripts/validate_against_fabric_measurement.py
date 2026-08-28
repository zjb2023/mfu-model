#!/usr/bin/env python3
"""Validate the source-only PP14 extrapolation against the Fabric 224-GPU run."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x10000_analysis.model_flops import (  # noqa: E402
    ModelArchitecture,
    analytical_iteration_flops,
    scale_calibrated_model_flops,
)


ITERATION_PATTERN = re.compile(
    r"iteration\s+(?P<iteration>\d+)/.*?"
    r"elapsed time per iteration \(ms\):\s*(?P<elapsed_ms>[0-9.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*(?P<tflops>[0-9.]+)"
)
PROFILED_ITERATIONS = tuple(range(5, 101, 5))
ARCHITECTURE_FIELDS = tuple(ModelArchitecture.__dataclass_fields__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the PP14 extrapolation with Fabric 224-GPU measurements."
    )
    parser.add_argument(
        "--fabric-root",
        type=Path,
        default=REPO_ROOT.parent / "fabric-data-analysis",
    )
    parser.add_argument(
        "--prediction-summary",
        type=Path,
        default=CASE_ROOT / "results/pp14_extrapolation_v1/summary.json",
    )
    parser.add_argument(
        "--scenario-sweep",
        type=Path,
        default=CASE_ROOT / "results/pp14_extrapolation_v1/scenario_sweep.csv",
    )
    parser.add_argument("--validation-iteration", type=int, default=55)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            CASE_ROOT
            / "results/pp14_extrapolation_v1/fabric_measurement_validation"
        ),
    )
    return parser.parse_args()


def _atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(document: dict[str, object], path: Path) -> None:
    _atomic_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        path,
    )


def _portable_path(path: Path) -> str:
    """Record inputs relative to this repository instead of leaking host paths."""
    return Path(os.path.relpath(path.resolve(), REPO_ROOT)).as_posix()


def _toml_int(path: Path, section: str, key: str) -> int:
    current_section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue
        if current_section != section or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if name == key:
            return int(value)
    raise ValueError(f"missing integer [{section}] {key} in {path}")


def _architecture(path: Path) -> ModelArchitecture:
    values = {
        field: _toml_int(path, "architecture", field)
        for field in ARCHITECTURE_FIELDS
    }
    return ModelArchitecture(**values)


def _parse_training_log(path: Path) -> pd.DataFrame:
    rows = [
        {
            "iteration": int(match.group("iteration")),
            "elapsed_time_ms": float(match.group("elapsed_ms")),
            "reported_tflops_per_gpu": float(match.group("tflops")),
        }
        for match in ITERATION_PATTERN.finditer(path.read_text(errors="replace"))
    ]
    if not rows:
        raise ValueError(f"no training iteration summaries in {path}")
    return (
        pd.DataFrame(rows)
        .drop_duplicates("iteration", keep="last")
        .sort_values("iteration")
        .reset_index(drop=True)
    )


def _find_training_log(framework_root: Path) -> tuple[Path, pd.DataFrame]:
    candidates: list[tuple[Path, pd.DataFrame]] = []
    for path in framework_root.rglob("*.log"):
        if "deepep_trace" in path.parts:
            continue
        try:
            frame = _parse_training_log(path)
        except ValueError:
            continue
        candidates.append((path, frame))
    if len(candidates) != 1:
        found = [(str(path), len(frame)) for path, frame in candidates]
        raise ValueError(f"expected one performance log, found {found}")
    return candidates[0]


def _quantiles(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype="float64")
    return {
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _report(summary: dict[str, object]) -> str:
    point = summary["point_validation"]
    all_iterations = summary["all_profiled_iterations"]
    work = summary["model_work"]
    workload = summary["workload"]
    return f"""# Fabric 224-GPU 实测校验

## 结论

当前 v1 在对齐的 iteration {point['iteration']} 上，training-log Step 低估
`{abs(point['training_log_error_pct']):.3f}%`；用目标配置 FLOPs 重算后，MFU
预测 `{point['predicted_mfu_pct_config_matched']:.4f}%`，实测
`{point['actual_mfu_pct']:.4f}%`，相对高估
`{point['config_matched_mfu_relative_error_pct']:.3f}%`（绝对
`{point['config_matched_mfu_error_percentage_points']:.4f}` 个百分点）。

修正前/源 FLOPs 误用口径会输出
`{point['legacy_source_flops_mfu_pct']:.4f}%`。它沿用了源任务 60 层/GBS64 的
iteration FLOPs，和实测 52 层/GBS48 不是同一分子；若直接比较会相对高估
`{point['legacy_source_flops_mfu_relative_error_pct']:.3f}%`，只作为缺陷审计项保留。

## 对齐口径

| 项目 | 256 卡源任务 | 224 卡实测任务 |
| --- | ---: | ---: |
| 层数 | {workload['source_num_layers']} | {workload['target_num_layers']} |
| Global batch | {workload['source_global_batch_size']} | {workload['target_global_batch_size']} |
| Microbatch 数 | {workload['source_num_micro_batches']} | {workload['target_num_micro_batches']} |
| World size | 256 | 224 |

源日志校准 FLOPs 为 `{work['source_calibrated_flops_per_iteration']:.9e}`；按已审计
MLA/MoE 解析式缩放到目标配置后为
`{work['target_calibrated_flops_per_iteration']:.9e}`，缩放比
`{work['target_to_source_analytical_ratio']:.9f}`。224 日志用“报告 TFLOP/s ×
Step × 224”反推的中位数为 `{work['target_log_inferred_flops_median']:.9e}`，
两者只差 `{work['target_log_vs_scaled_flops_pct']:.4f}%`，说明目标 MFU 分子已
对齐。

## Iteration {point['iteration']} 误差分解

| 指标 | 当前预测 | Fabric 实测 | 误差 |
| --- | ---: | ---: | ---: |
| ProfilerStep cluster envelope | {point['predicted_profiler_step_ms']:.3f} ms | {point['actual_profiler_step_ms']:.3f} ms | {point['profiler_step_error_pct']:+.3f}% |
| ProfilerStep 外残差 | {point['predicted_outer_residual_ms']:.3f} ms | {point['actual_outer_residual_ms']:.3f} ms | {point['outer_residual_error_pct']:+.3f}% |
| training-log Step | {point['predicted_training_log_ms']:.3f} ms | {point['actual_training_log_ms']:.3f} ms | {point['training_log_error_pct']:+.3f}% |
| 配置匹配 MFU | {point['predicted_mfu_pct_config_matched']:.4f}% | {point['actual_mfu_pct']:.4f}% | {point['config_matched_mfu_relative_error_pct']:+.3f}% relative |

ProfilerStep 实测取 224 Rank 的 `min(start_ns) → max(end_ns)` cluster envelope，
和 256 卡 `iteration_clocks.csv` 的构造口径一致。训练日志的额外误差主要来自
沿用源 iteration 55 的 outer residual：目标实测比沿用值多
`{point['actual_outer_residual_ms'] - point['predicted_outer_residual_ms']:.3f} ms`。

## 20 个 profiler iteration 的稳定性

固定使用当前单点预测时，20 个 iteration 的 Step 绝对相对误差 P50 为
`{all_iterations['training_log_absolute_error_pct']['p50']:.3f}%`、P90 为
`{all_iterations['training_log_absolute_error_pct']['p90']:.3f}%`；配置匹配 MFU
相对误差绝对值 P90 为
`{all_iterations['config_matched_mfu_absolute_relative_error_pct']['p90']:.3f}%`。
iteration 55 单点通过 3% 门槛，但全时段 P90 未通过 5% 门槛，后半段明显变慢；
因此当前结果是“单点外推接近、时间漂移泛化失败”，不能据此宣布 224 卡模型已
完成校准。

逐 iteration 明细见 `fabric_validation_by_iteration.csv`，机器可读结论见
`fabric_validation_summary.json`。
"""


def main() -> None:
    args = parse_args()
    fabric_case = args.fabric_root / "case_224gpu_pp14_cp2_a2a"
    source_case = args.fabric_root / "case_256gpu_pp16_cp2_a2a"
    raw_framework = (
        args.fabric_root
        / "fabric-data-analysis-raw/0731_224gpu_pp14_cp2_a2a/extracted/framework/2026-07-30-09:38"
    )
    event_windows_path = fabric_case / "results/readiness/event_windows.csv"
    rank_steps_path = (
        fabric_case / "results/derived/profiler_step_by_iteration_rank.csv"
    )
    source_layer_map_path = source_case / "results/topology/layer_stage_map.csv"
    target_layer_map_path = fabric_case / "results/topology/layer_stage_map.csv"
    source_model_config = source_case / "config/mfu_model.toml"
    target_run_contract = fabric_case / "config/run_contract.toml"

    prediction = json.loads(args.prediction_summary.read_text(encoding="utf-8"))
    predicted = prediction["representative_target_prediction"]
    world_size = _toml_int(target_run_contract, "topology", "world_size")
    target_num_layers = _toml_int(target_run_contract, "topology", "num_layers")
    target_global_batch = _toml_int(
        target_run_contract, "topology", "global_batch_size"
    )
    target_micro_batches = _toml_int(
        target_run_contract, "topology", "num_micro_batches"
    )
    source_layers = pd.read_csv(source_layer_map_path)
    target_layers = pd.read_csv(target_layer_map_path)
    source_num_layers = int(source_layers["layer_id"].nunique())
    observed_target_layers = int(target_layers["layer_id"].nunique())
    if observed_target_layers != target_num_layers:
        raise ValueError(
            f"target layer map has {observed_target_layers}, expected {target_num_layers}"
        )
    source_architecture = _architecture(source_model_config)
    target_architecture = source_architecture.with_global_batch_size(
        target_global_batch
    )
    source_flops = float(
        prediction["calibration_reused"]["model_flops_per_iteration"]
    )
    source_analytical_flops = analytical_iteration_flops(
        source_architecture, source_num_layers
    )
    target_analytical_flops = analytical_iteration_flops(
        target_architecture, target_num_layers
    )
    target_flops = scale_calibrated_model_flops(
        source_flops,
        source_architecture,
        source_num_layers,
        target_architecture,
        target_num_layers,
    )
    peak_tflops = float(prediction["calibration_reused"]["peak_tflops_per_gpu"])

    log_path, training = _find_training_log(raw_framework)
    training["log_inferred_flops"] = (
        training["reported_tflops_per_gpu"]
        * 1e12
        * training["elapsed_time_ms"]
        / 1000.0
        * world_size
    )
    inferred_flops = float(training["log_inferred_flops"].median())

    windows = pd.read_csv(event_windows_path)
    expected_cells = len(PROFILED_ITERATIONS) * world_size
    if len(windows) != expected_cells or windows["error"].notna().any():
        raise ValueError("Fabric event-window grid is incomplete or contains errors")
    if set(windows["iteration"].astype(int)) != set(PROFILED_ITERATIONS):
        raise ValueError("Fabric event-window iterations do not match the frozen set")
    envelope = (
        windows.groupby("iteration")
        .agg(step_start_ns=("start_ns", "min"), step_end_ns=("end_ns", "max"))
        .reset_index()
    )
    envelope["actual_profiler_step_ms"] = (
        envelope["step_end_ns"] - envelope["step_start_ns"]
    ) / 1e6
    rank_steps = pd.read_csv(rank_steps_path)
    rank_summary = (
        rank_steps.groupby("iteration")["duration_s"]
        .agg(
            actual_rank_profiler_step_p50_s="median",
            actual_rank_profiler_step_max_s="max",
        )
        .reset_index()
    )
    rank_summary["actual_rank_profiler_step_p95_s"] = (
        rank_steps.groupby("iteration")["duration_s"].quantile(0.95).values
    )

    measured = (
        training[training["iteration"].isin(PROFILED_ITERATIONS)]
        .merge(envelope, on="iteration", validate="one_to_one")
        .merge(rank_summary, on="iteration", validate="one_to_one")
        .sort_values("iteration")
        .reset_index(drop=True)
    )
    if len(measured) != len(PROFILED_ITERATIONS):
        raise ValueError("training log does not cover every profiled iteration")

    predicted_profiler_ms = float(predicted["profiler_step_ms"])
    predicted_training_ms = float(predicted["training_log_ms"])
    predicted_outer_ms = predicted_training_ms - predicted_profiler_ms
    predicted_mfu_current = float(predicted["mfu_pct"])
    predicted_mfu_config_matched = (
        target_flops
        / (world_size * peak_tflops * 1e12 * predicted_training_ms / 1000.0)
        * 100.0
    )
    legacy_source_flops_mfu = (
        source_flops
        / (world_size * peak_tflops * 1e12 * predicted_training_ms / 1000.0)
        * 100.0
    )
    measured["actual_outer_residual_ms"] = (
        measured["elapsed_time_ms"] - measured["actual_profiler_step_ms"]
    )
    measured["actual_mfu_pct"] = (
        target_flops
        / (
            world_size
            * peak_tflops
            * 1e12
            * measured["elapsed_time_ms"]
            / 1000.0
        )
        * 100.0
    )
    measured["predicted_profiler_step_ms"] = predicted_profiler_ms
    measured["predicted_training_log_ms"] = predicted_training_ms
    measured["predicted_outer_residual_ms"] = predicted_outer_ms
    measured["predicted_mfu_pct_current"] = predicted_mfu_current
    measured["legacy_source_flops_mfu_pct_config_mismatch"] = (
        legacy_source_flops_mfu
    )
    measured["predicted_mfu_pct_config_matched"] = predicted_mfu_config_matched
    for prefix, predicted_column, actual_column in (
        ("profiler_step", "predicted_profiler_step_ms", "actual_profiler_step_ms"),
        ("training_log", "predicted_training_log_ms", "elapsed_time_ms"),
        ("outer_residual", "predicted_outer_residual_ms", "actual_outer_residual_ms"),
    ):
        measured[f"{prefix}_error_ms"] = (
            measured[predicted_column] - measured[actual_column]
        )
        measured[f"{prefix}_error_pct"] = (
            measured[f"{prefix}_error_ms"] / measured[actual_column] * 100.0
        )
    measured["config_matched_mfu_error_percentage_points"] = (
        predicted_mfu_config_matched - measured["actual_mfu_pct"]
    )
    measured["config_matched_mfu_relative_error_pct"] = (
        measured["config_matched_mfu_error_percentage_points"]
        / measured["actual_mfu_pct"]
        * 100.0
    )
    measured["legacy_source_flops_mfu_relative_error_pct"] = (
        (legacy_source_flops_mfu / measured["actual_mfu_pct"] - 1.0) * 100.0
    )

    point_rows = measured[measured["iteration"].eq(args.validation_iteration)]
    if len(point_rows) != 1:
        raise ValueError("validation iteration is not present exactly once")
    point_row = point_rows.iloc[0]
    point = {
        "iteration": args.validation_iteration,
        "predicted_profiler_step_ms": predicted_profiler_ms,
        "actual_profiler_step_ms": float(point_row["actual_profiler_step_ms"]),
        "profiler_step_error_pct": float(point_row["profiler_step_error_pct"]),
        "predicted_outer_residual_ms": predicted_outer_ms,
        "actual_outer_residual_ms": float(point_row["actual_outer_residual_ms"]),
        "outer_residual_error_pct": float(point_row["outer_residual_error_pct"]),
        "predicted_training_log_ms": predicted_training_ms,
        "actual_training_log_ms": float(point_row["elapsed_time_ms"]),
        "training_log_error_pct": float(point_row["training_log_error_pct"]),
        "predicted_mfu_pct_current": predicted_mfu_current,
        "legacy_source_flops_mfu_pct": legacy_source_flops_mfu,
        "legacy_source_flops_mfu_relative_error_pct": float(
            point_row["legacy_source_flops_mfu_relative_error_pct"]
        ),
        "predicted_mfu_pct_config_matched": predicted_mfu_config_matched,
        "actual_mfu_pct": float(point_row["actual_mfu_pct"]),
        "config_matched_mfu_error_percentage_points": float(
            point_row["config_matched_mfu_error_percentage_points"]
        ),
        "config_matched_mfu_relative_error_pct": float(
            point_row["config_matched_mfu_relative_error_pct"]
        ),
    }

    sweep = pd.read_csv(args.scenario_sweep)
    train_low = float(sweep["predicted_training_log_ms"].quantile(0.1))
    train_high = float(sweep["predicted_training_log_ms"].quantile(0.9))
    profiler_low = float(sweep["predicted_profiler_step_ms"].quantile(0.1))
    profiler_high = float(sweep["predicted_profiler_step_ms"].quantile(0.9))
    full_validation = {
        "iteration_count": len(measured),
        "actual_training_log_ms": _quantiles(measured["elapsed_time_ms"]),
        "actual_mfu_pct": _quantiles(measured["actual_mfu_pct"]),
        "training_log_signed_error_pct": _quantiles(
            measured["training_log_error_pct"]
        ),
        "training_log_absolute_error_pct": _quantiles(
            measured["training_log_error_pct"].abs()
        ),
        "profiler_step_absolute_error_pct": _quantiles(
            measured["profiler_step_error_pct"].abs()
        ),
        "config_matched_mfu_absolute_relative_error_pct": _quantiles(
            measured["config_matched_mfu_relative_error_pct"].abs()
        ),
        "scenario_p10_p90_training_log_coverage_count": int(
            measured["elapsed_time_ms"].between(train_low, train_high).sum()
        ),
        "scenario_p10_p90_profiler_step_coverage_count": int(
            measured["actual_profiler_step_ms"]
            .between(profiler_low, profiler_high)
            .sum()
        ),
    }
    checks = {
        "fabric_event_windows_are_20_by_224": len(windows) == expected_cells,
        "target_layer_map_matches_run_contract": (
            observed_target_layers == target_num_layers
        ),
        "scaled_target_flops_match_log_inference_within_1pct": abs(
            inferred_flops / target_flops - 1.0
        )
        <= 0.01,
        "iteration55_training_log_error_within_3pct": abs(
            point["training_log_error_pct"]
        )
        <= 3.0,
        "iteration55_config_matched_mfu_error_within_3pct": abs(
            point["config_matched_mfu_relative_error_pct"]
        )
        <= 3.0,
        "main_prediction_uses_target_config_flops": abs(
            predicted_mfu_current / predicted_mfu_config_matched - 1.0
        )
        <= 1e-12,
        "all_iteration_training_log_absolute_error_p90_within_5pct": (
            full_validation["training_log_absolute_error_pct"]["p90"] <= 5.0
        ),
    }
    result = {
        "schema": "pp14-224gpu-fabric-measurement-validation-v1",
        "status": "PASS_POINT_FAIL_TEMPORAL_GENERALIZATION"
        if all(value for key, value in checks.items() if "all_iteration" not in key)
        and not checks["all_iteration_training_log_absolute_error_p90_within_5pct"]
        else "CHECK_RESULTS",
        "workload": {
            "source_num_layers": source_num_layers,
            "source_global_batch_size": source_architecture.global_batch_size,
            "source_num_micro_batches": int(
                pd.read_csv(REPO_ROOT / "case_256gpu_pp16_cp2_a2a/data/pp_dag_trace_events.csv")[
                    "microbatch"
                ].nunique()
            ),
            "target_num_layers": target_num_layers,
            "target_global_batch_size": target_global_batch,
            "target_num_micro_batches": target_micro_batches,
        },
        "model_work": {
            "source_calibrated_flops_per_iteration": source_flops,
            "source_analytical_flops_per_iteration": source_analytical_flops,
            "target_analytical_flops_per_iteration": target_analytical_flops,
            "target_to_source_analytical_ratio": (
                target_analytical_flops / source_analytical_flops
            ),
            "target_calibrated_flops_per_iteration": target_flops,
            "target_log_inferred_flops_median": inferred_flops,
            "target_log_vs_scaled_flops_pct": (
                (inferred_flops / target_flops - 1.0) * 100.0
            ),
        },
        "point_validation": point,
        "all_profiled_iterations": full_validation,
        "checks": checks,
        "inputs": {
            "prediction_summary": _portable_path(args.prediction_summary),
            "fabric_training_log": _portable_path(log_path),
            "fabric_event_windows": _portable_path(event_windows_path),
            "fabric_rank_profiler_steps": _portable_path(rank_steps_path),
            "source_layer_map": _portable_path(source_layer_map_path),
            "target_layer_map": _portable_path(target_layer_map_path),
        },
        "interpretation": (
            "The main MFU numerator is target-configuration matched. The "
            "corrected MFU passes the aligned iteration-55 point gate, while "
            "one fixed prediction fails the 20-iteration temporal P90 gate."
        ),
    }

    output_columns = [
        "iteration",
        "elapsed_time_ms",
        "reported_tflops_per_gpu",
        "log_inferred_flops",
        "actual_profiler_step_ms",
        "actual_rank_profiler_step_p50_s",
        "actual_rank_profiler_step_p95_s",
        "actual_rank_profiler_step_max_s",
        "actual_outer_residual_ms",
        "actual_mfu_pct",
        "predicted_profiler_step_ms",
        "profiler_step_error_ms",
        "profiler_step_error_pct",
        "predicted_training_log_ms",
        "training_log_error_ms",
        "training_log_error_pct",
        "predicted_outer_residual_ms",
        "outer_residual_error_ms",
        "outer_residual_error_pct",
        "predicted_mfu_pct_current",
        "legacy_source_flops_mfu_pct_config_mismatch",
        "legacy_source_flops_mfu_relative_error_pct",
        "predicted_mfu_pct_config_matched",
        "config_matched_mfu_error_percentage_points",
        "config_matched_mfu_relative_error_pct",
    ]
    _atomic_text(
        measured[output_columns].to_csv(index=False),
        args.output_dir / "fabric_validation_by_iteration.csv",
    )
    _atomic_json(result, args.output_dir / "fabric_validation_summary.json")
    _atomic_text(_report(result), args.output_dir / "FABRIC_MEASUREMENT_VALIDATION.md")
    print(
        f"{result['status']}: iter {args.validation_iteration} "
        f"step_error={point['training_log_error_pct']:+.3f}%, "
        f"corrected_mfu_error={point['config_matched_mfu_relative_error_pct']:+.3f}% "
        f"-> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the structurally validated 224-GPU PP14 extrapolation sweep."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CASE = REPO_ROOT / "case_256gpu_pp16_cp2_a2a"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x10000_analysis.parallel_extrapolation import (  # noqa: E402
    RepartitionedCase,
    build_repartitioned_case,
    enumerate_legal_strategies,
    stage_omission_scenarios,
)
from x10000_analysis.model_flops import (  # noqa: E402
    ModelArchitecture,
    scale_calibrated_model_flops,
)
from x10000_analysis.unified_mfu_dag import (  # noqa: E402
    UnifiedDagResult,
    build_unified_mfu_dag,
)


SOURCE_ARCHITECTURE = ModelArchitecture(
    hidden_size=5120,
    sequence_length=8192,
    global_batch_size=64,
    num_attention_heads=128,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_head_dim=128,
    qk_pos_emb_head_dim=64,
    v_head_dim=128,
    dense_ffn_hidden_size=12288,
    moe_ffn_hidden_size=1536,
    shared_expert_intermediate_size=3072,
    num_experts=160,
    moe_router_topk=6,
    vocab_size=163840,
    first_dense_layers=1,
)
SOURCE_NUM_LAYERS = 60
TARGET_ARCHITECTURE = SOURCE_ARCHITECTURE.with_global_batch_size(48)
TARGET_NUM_LAYERS = 52


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrapolate PP16/CP2/DP8/EP8 on 256 GPUs to PP14 on 224 GPUs."
    )
    parser.add_argument(
        "--trace-events",
        type=Path,
        default=SOURCE_CASE / "data/pp_dag_trace_events.csv",
    )
    parser.add_argument(
        "--pp-api-events",
        type=Path,
        default=SOURCE_CASE / "data/pp_api_trace_events.csv",
    )
    parser.add_argument(
        "--optimizer-calls",
        type=Path,
        default=SOURCE_CASE / "data/optimizer_timeline_calls.csv",
    )
    parser.add_argument(
        "--iteration-clocks",
        type=Path,
        default=SOURCE_CASE / "data/iteration_clocks.csv",
    )
    parser.add_argument(
        "--pp-service-cells",
        type=Path,
        default=SOURCE_CASE / "data/pp_framework_sync_cells.csv",
    )
    parser.add_argument(
        "--optimizer-summary",
        type=Path,
        default=SOURCE_CASE / "results/mfu_timeline/mfu_timeline_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_ROOT / "results/pp14_extrapolation_v1",
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


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_text(frame.to_csv(index=False), path)


def _atomic_json(document: dict[str, object], path: Path) -> None:
    _atomic_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        path,
    )


def _pp_calibration(api: pd.DataFrame, cells: pd.DataFrame) -> tuple[int, int]:
    network_service_ns = int(round(cells["service_reference_fct_ns"].median()))
    selected = api[
        api["api_name"].eq("send_backward") & api["pp_stage"].between(1, 14)
    ]
    if len(selected) < 800:
        raise ValueError("insufficient backward PP software calibration observations")
    observed_completion_ns = int(round(selected["duration_ns"].median()))
    return network_service_ns, max(observed_completion_ns - network_service_ns, 0)


def _run(
    case: RepartitionedCase,
    *,
    pp_network_service_ns: int,
    pp_backward_software_ns: int,
    model_flops: float,
    peak_tflops: float,
    calculate_service_marginals: bool,
) -> UnifiedDagResult:
    return build_unified_mfu_dag(
        trace_events=case.pipeline.trace_events,
        pp_service_ns=pp_network_service_ns,
        pp_software_completion_ns={
            "forward": 0,
            "backward": pp_backward_software_ns,
        },
        optimizer_calls=case.optimizer.calls,
        clocks=case.clocks,
        model_flops_per_iteration=model_flops,
        world_size=case.target_world_size,
        peak_tflops_per_gpu=peak_tflops,
        optimizer_service_scales=case.optimizer.service_scales,
        calculate_optimizer_service_marginals=calculate_service_marginals,
    )


def _source_baseline(
    trace: pd.DataFrame,
    calls: pd.DataFrame,
    clocks: pd.DataFrame,
    *,
    pp_network_service_ns: int,
    pp_backward_software_ns: int,
    model_flops: float,
    peak_tflops: float,
) -> UnifiedDagResult:
    return build_unified_mfu_dag(
        trace_events=trace,
        pp_service_ns=pp_network_service_ns,
        pp_software_completion_ns={
            "forward": 0,
            "backward": pp_backward_software_ns,
        },
        optimizer_calls=calls,
        clocks=clocks,
        model_flops_per_iteration=model_flops,
        world_size=256,
        peak_tflops_per_gpu=peak_tflops,
        calculate_optimizer_service_marginals=False,
    )


def _sweep_row(
    retained: tuple[int, ...],
    case: RepartitionedCase,
    result: UnifiedDagResult,
    source_metrics: pd.Series,
) -> dict[str, object]:
    metrics = result.iteration.iloc[0]
    optimizer_metrics = result.optimizer.iterations.iloc[0]
    omitted = tuple(sorted(set(range(16)) - set(retained)))
    scales = case.pipeline.compute_conservation["scale"]
    row: dict[str, object] = {
        "scenario_id": "omit_" + "_".join(f"{stage:02d}" for stage in omitted),
        "omitted_source_stages": ",".join(map(str, omitted)),
        "retained_source_stages": ",".join(map(str, retained)),
        "target_world_size": case.target_world_size,
        "target_pp_size": int(case.pipeline.trace_events["pp_stage"].nunique()),
        "target_lanes": int(case.pipeline.trace_events["pp_lane"].nunique()),
        "predicted_profiler_step_ms": float(metrics["predicted_profiler_step_ms"]),
        "predicted_training_log_ms": float(metrics["predicted_training_log_ms"]),
        "predicted_mfu_pct": float(metrics["predicted_mfu_pct"]),
        "profiler_step_delta_vs_256_ms": float(
            metrics["predicted_profiler_step_ms"]
            - source_metrics["predicted_profiler_step_ms"]
        ),
        "profiler_step_delta_vs_256_pct": 100.0
        * (
            float(metrics["predicted_profiler_step_ms"])
            / float(source_metrics["predicted_profiler_step_ms"])
            - 1.0
        ),
        "mfu_relative_delta_vs_256_pct": 100.0
        * (
            float(metrics["predicted_mfu_pct"])
            / float(source_metrics["predicted_mfu_pct"])
            - 1.0
        ),
        "pp_front_predicted_ms_p50": float(metrics["pp_front_predicted_ms_p50"]),
        "first_rs_offset_ms": float(metrics["first_rs_offset_ms"]),
        "last_ag_offset_ms": float(metrics["last_ag_offset_ms"]),
        "compute_route_scale_min": float(scales.min()),
        "compute_route_scale_p50": float(scales.median()),
        "compute_route_scale_max": float(scales.max()),
        "clamped_negative_dependency_lags": int(
            optimizer_metrics["clamped_negative_dependency_lags"]
        ),
        "fallback_dependencies": int(optimizer_metrics["fallback_dependencies"]),
    }
    for kind, scale in sorted(case.optimizer.service_scales.items()):
        row[f"{kind}_payload_service_scale"] = float(scale)
    return row


def _quantiles(sweep: pd.DataFrame, column: str) -> dict[str, float]:
    values = sweep[column].to_numpy(dtype="float64")
    return {
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.1)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def _source_stage_profile(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, stage_events in trace.groupby("pp_stage", sort=True):
        row: dict[str, object] = {
            "source_pp_stage": int(stage),
            "rank_count": int(stage_events["rank"].nunique()),
            "compute_total_ms_all_ranks": float(
                stage_events["duration_ns"].sum() / 1e6
            ),
        }
        for phase, phase_events in stage_events.groupby("phase"):
            rank_totals = phase_events.groupby("rank")["duration_ns"].sum() / 1e6
            route_totals = (
                phase_events.groupby(["pp_lane", "microbatch"])["duration_ns"].sum()
                / 1e6
            )
            row[f"{phase}_rank_total_ms_p50"] = float(rank_totals.median())
            row[f"{phase}_route_ms_p50"] = float(route_totals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def _stage_omission_sensitivity(sweep: pd.DataFrame) -> pd.DataFrame:
    omitted_sets = sweep["omitted_source_stages"].map(
        lambda value: {int(stage) for stage in str(value).split(",")}
    )
    rows = []
    for stage in range(16):
        omitted = omitted_sets.map(lambda values: stage in values)
        omitted_step = sweep.loc[omitted, "predicted_profiler_step_ms"]
        retained_step = sweep.loc[~omitted, "predicted_profiler_step_ms"]
        rows.append(
            {
                "source_pp_stage": stage,
                "omitted_scenarios": int(omitted.sum()),
                "step_ms_p50_when_omitted": float(omitted_step.median()),
                "step_ms_p50_when_retained": float(retained_step.median()),
                "omission_step_penalty_ms": float(
                    omitted_step.median() - retained_step.median()
                ),
                "mfu_pct_p50_when_omitted": float(
                    sweep.loc[omitted, "predicted_mfu_pct"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "omission_step_penalty_ms", ascending=False
    ).reset_index(drop=True)


def _comparison(
    source: UnifiedDagResult, target: UnifiedDagResult, scenario_id: str
) -> pd.DataFrame:
    source_metrics = source.iteration.iloc[0]
    target_metrics = target.iteration.iloc[0]
    return pd.DataFrame(
        [
            {
                "scenario": "256gpu_pp16_trace_replay",
                "evidence_level": "in_sample_iteration55",
                "world_size": 256,
                "pipeline_parallel_size": 16,
                "predicted_profiler_step_ms": source_metrics[
                    "predicted_profiler_step_ms"
                ],
                "predicted_training_log_ms": source_metrics[
                    "predicted_training_log_ms"
                ],
                "predicted_mfu_pct": source_metrics["predicted_mfu_pct"],
            },
            {
                "scenario": f"224gpu_pp14_{scenario_id}",
                "evidence_level": "structural_extrapolation_with_fabric_validation",
                "world_size": 224,
                "pipeline_parallel_size": 14,
                "predicted_profiler_step_ms": target_metrics[
                    "predicted_profiler_step_ms"
                ],
                "predicted_training_log_ms": target_metrics[
                    "predicted_training_log_ms"
                ],
                "predicted_mfu_pct": target_metrics["predicted_mfu_pct"],
            },
        ]
    )


def _report(
    source_metrics: pd.Series,
    target_metrics: pd.Series,
    optimizer_metrics: pd.Series,
    sweep: pd.DataFrame,
    stage_profile: pd.DataFrame,
    omission_sensitivity: pd.DataFrame,
    representative_id: str,
    distributions: dict[str, dict[str, float]],
    source_model_flops: float,
    target_model_flops: float,
) -> str:
    step = distributions["predicted_profiler_step_ms"]
    mfu = distributions["predicted_mfu_pct"]
    stage0_compute = float(
        stage_profile.loc[
            stage_profile["source_pp_stage"].eq(0), "compute_total_ms_all_ranks"
        ].iloc[0]
    )
    stage15_compute = float(
        stage_profile.loc[
            stage_profile["source_pp_stage"].eq(15), "compute_total_ms_all_ranks"
        ].iloc[0]
    )
    interior_compute = float(
        stage_profile.loc[
            stage_profile["source_pp_stage"].between(1, 14),
            "compute_total_ms_all_ranks",
        ].median()
    )
    stage0_penalty = float(
        omission_sensitivity.loc[
            omission_sensitivity["source_pp_stage"].eq(0),
            "omission_step_penalty_ms",
        ].iloc[0]
    )
    return f"""# 224-GPU PP14 MFU 外推 v1

> 结构校验通过；Fabric 224 卡实测已完成单点与 20-iteration 校验。iteration 55
> 单点接近，但时间漂移 P90 未通过，且 v1 时间 DAG 仍是源工作量守恒模型，不能
> 标记为 224 卡已完成校准。

## 策略与口径

- 源策略：`PP16 / CP2 / DP8 / EP8 / TP1 = 256 GPU`；
- 目标策略：`PP14 / CP2 / DP8 / EP8 / TP1 = 224 GPU`；
- 保持 16 条 pipeline lane、dense communicator 16 Rank 和 expert communicator 2 Rank；
- 时间 DAG 仍保持源 Trace 的 4 个 microbatch 和 60 层工作量；真实目标是 3 个 microbatch、52 层，因此该时间外推是诊断基线，不是配置完整的 v2；
- MFU 分子已从源 `{source_model_flops:.9e}` FLOPs 按 60 层/GBS64 → 52 层/GBS48 的解析工作比修正为目标 `{target_model_flops:.9e}` FLOPs；
- 对每条 `lane × phase × microbatch` 路径严格守恒 16-stage Trace 的计算总量；
- 对每类 DP/EDP RS/AG 严格守恒 aggregate payload，网络 service 按 payload 比例缩放；
- PP 网络 FCT `4.712346 ms` 和 backward 软件 completion `93.463844 ms` 暂沿用 256 卡校准。

## 120 场结构不确定性

PP16 压到 PP14 时，v1 紧凑输入没有带入 layer-to-stage 映射，因此枚举省略两个源 stage 模板的全部 `C(16,2)=120` 种组合，并将剩余模板按原顺序重映射到 PP14。Fabric 工程现已确认实际映射为 stage0/13 各 2 层、中间 stage 各 4 层；后续 v2 应直接使用该映射和 3-microbatch schedule 替换 ensemble。

| 指标 | min | P10 | P50 | P90 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| ProfilerStep (ms) | {step['min']:.3f} | {step['p10']:.3f} | {step['p50']:.3f} | {step['p90']:.3f} | {step['max']:.3f} |
| MFU (%) | {mfu['min']:.4f} | {mfu['p10']:.4f} | {mfu['p50']:.4f} | {mfu['p90']:.4f} | {mfu['max']:.4f} |

上尾主要来自 endpoint 模板不确定性：源 stage 0/15 的全 Rank compute 总量为
{stage0_compute:.1f}/{stage15_compute:.1f} ms，内部 stage 中位为
{interior_compute:.1f} ms；省略 stage 0 的场景中位 Step 比保留它高
{stage0_penalty:.1f} ms。实际 PP14 layer map 到位后应直接替换这项 ensemble，
区间会明显收窄。

代表场景 `{representative_id}` 是 MFU 最接近 120 场中位数的实际场景：

| 指标 | 256 卡回放 | 224 卡代表外推 |
| --- | ---: | ---: |
| ProfilerStep | {source_metrics['predicted_profiler_step_ms']:.3f} ms | {target_metrics['predicted_profiler_step_ms']:.3f} ms |
| training-log Step | {source_metrics['predicted_training_log_ms']:.3f} ms | {target_metrics['predicted_training_log_ms']:.3f} ms |
| 配置匹配固定 FLOPs MFU | {source_metrics['predicted_mfu_pct']:.4f}% | {target_metrics['predicted_mfu_pct']:.4f}% |
| DP/EDP service 最终暴露 | N/A（sweep 关闭反事实） | {optimizer_metrics['all_dp_service_exposed_ms']:.3f} ms |

## 校验边界

已校验：224 Rank 完整网格、PP14×lane16、每条 compute route 纳秒级精确守恒、六类 payload 取整误差受控、group size 保持 dense16/EDP2、120 场输出有限且可复现。Fabric iteration 55 上，training-log Step 误差 `-2.117%`，目标 FLOPs MFU 相对误差 `+2.162%`；详见 `fabric_measurement_validation/FABRIC_MEASUREMENT_VALIDATION.md`。

尚未建模：52 层/3-microbatch 的真实 PP schedule、PP14 的 P2P 软件 completion、目标拓扑上的六类 collective FCT，以及运行后半程的性能退化。20 个 iteration 的 Step 绝对相对误差 P90 为 `8.929%`，未通过 5% 门槛。
"""


def main() -> None:
    args = parse_args()
    trace = pd.read_csv(args.trace_events)
    api = pd.read_csv(args.pp_api_events)
    calls = pd.read_csv(args.optimizer_calls)
    clocks = pd.read_csv(args.iteration_clocks)
    cells = pd.read_csv(args.pp_service_cells)
    summary = json.loads(args.optimizer_summary.read_text())
    source_model_flops = float(summary["model_flops_per_iteration"])
    target_model_flops = scale_calibrated_model_flops(
        source_model_flops,
        SOURCE_ARCHITECTURE,
        SOURCE_NUM_LAYERS,
        TARGET_ARCHITECTURE,
        TARGET_NUM_LAYERS,
    )
    peak_tflops = float(summary["peak_tflops_per_gpu"])
    network_ns, backward_software_ns = _pp_calibration(api, cells)

    source = _source_baseline(
        trace,
        calls,
        clocks,
        pp_network_service_ns=network_ns,
        pp_backward_software_ns=backward_software_ns,
        model_flops=source_model_flops,
        peak_tflops=peak_tflops,
    )
    source_metrics = source.iteration.iloc[0]
    scenarios = stage_omission_scenarios(16, 14)
    rows: list[dict[str, object]] = []
    print(f"running {len(scenarios)} PP14 stage-template scenarios", flush=True)
    for index, retained in enumerate(scenarios, start=1):
        case = build_repartitioned_case(trace, calls, clocks, retained)
        result = _run(
            case,
            pp_network_service_ns=network_ns,
            pp_backward_software_ns=backward_software_ns,
            model_flops=target_model_flops,
            peak_tflops=peak_tflops,
            calculate_service_marginals=False,
        )
        rows.append(_sweep_row(retained, case, result, source_metrics))
        if index % 10 == 0 or index == len(scenarios):
            print(f"completed {index}/{len(scenarios)} scenarios", flush=True)
    sweep = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)

    median_mfu = float(sweep["predicted_mfu_pct"].median())
    representative_row = sweep.loc[
        (sweep["predicted_mfu_pct"] - median_mfu).abs().sort_values().index[0]
    ]
    representative_id = str(representative_row["scenario_id"])
    retained = tuple(
        int(stage) for stage in str(representative_row["retained_source_stages"]).split(",")
    )
    representative_case = build_repartitioned_case(trace, calls, clocks, retained)
    representative = _run(
        representative_case,
        pp_network_service_ns=network_ns,
        pp_backward_software_ns=backward_software_ns,
        model_flops=target_model_flops,
        peak_tflops=peak_tflops,
        calculate_service_marginals=True,
    )
    target_metrics = representative.iteration.iloc[0]
    optimizer_metrics = representative.optimizer.iterations.iloc[0]

    strategies = enumerate_legal_strategies(
        world_size=224,
        expert_parallel_size=8,
        source_cp_size=2,
        source_dp_size=8,
        source_microbatches=3,
        pp_sizes=(7, 14, 28),
        cp_sizes=(1, 2, 4),
        captured_microbatches=4,
    )
    distributions = {
        column: _quantiles(sweep, column)
        for column in (
            "predicted_profiler_step_ms",
            "predicted_training_log_ms",
            "predicted_mfu_pct",
        )
    }
    stage_profile = _source_stage_profile(trace)
    omission_sensitivity = _stage_omission_sensitivity(sweep)
    group_sizes = {
        kind: sorted(
            map(
                int,
                representative_case.optimizer.calls.loc[
                    representative_case.optimizer.calls["kind"].eq(kind),
                    "group_size",
                ].unique(),
            )
        )
        for kind in sorted(representative_case.optimizer.calls["kind"].unique())
    }
    payload_error = int(
        representative_case.optimizer.payload_conservation[
            "payload_conservation_error_bytes"
        ].abs().max()
    )
    structural_checks = {
        "scenario_count_is_120": len(sweep) == 120,
        "target_world_size_is_224": representative_case.target_world_size == 224,
        "target_pp_size_is_14": representative_case.pipeline.trace_events[
            "pp_stage"
        ].nunique()
        == 14,
        "target_lane_count_is_16": representative_case.pipeline.trace_events[
            "pp_lane"
        ].nunique()
        == 16,
        "compute_routes_conserve_exactly": bool(
            representative_case.pipeline.compute_conservation[
                "conservation_error_ns"
            ].eq(0).all()
        ),
        "payload_rounding_error_at_most_one_byte_per_group": payload_error
        <= int(
            representative_case.optimizer.payload_conservation["target_groups"].max()
        ),
        "dense_groups_are_16_rank": all(
            group_sizes[kind] == [16] for kind in ("dp_rs", "dp_ag0", "dp_ag1")
        ),
        "expert_groups_are_2_rank": all(
            group_sizes[kind] == [2]
            for kind in ("edp_rs", "edp_ag0", "edp_ag1")
        ),
        "representative_metrics_are_finite": bool(
            np.isfinite(
                target_metrics[
                    [
                        "predicted_profiler_step_ms",
                        "predicted_training_log_ms",
                        "predicted_mfu_pct",
                    ]
                ].astype(float)
            ).all()
        ),
    }
    validation = {
        "schema": "pp14-224gpu-extrapolation-v1",
        "status": "PASS_STRUCTURAL"
        if all(structural_checks.values())
        else "FAIL",
        "predictive_validation": "AVAILABLE_SEPARATE_FABRIC_VALIDATION",
        "predictive_validation_path": (
            "fabric_measurement_validation/fabric_validation_summary.json"
        ),
        "checks": structural_checks,
        "representative_scenario": representative_id,
        "group_sizes": group_sizes,
        "max_payload_conservation_rounding_error_bytes": payload_error,
    }
    output = args.output_dir
    _atomic_csv(sweep, output / "scenario_sweep.csv")
    _atomic_csv(strategies, output / "legal_224gpu_strategies.csv")
    _atomic_csv(stage_profile, output / "source_stage_profile.csv")
    _atomic_csv(
        omission_sensitivity, output / "stage_omission_sensitivity.csv"
    )
    _atomic_csv(
        representative_case.pipeline.stage_mapping,
        output / "representative_stage_mapping.csv",
    )
    _atomic_csv(
        representative_case.pipeline.compute_conservation,
        output / "compute_conservation.csv",
    )
    _atomic_csv(
        representative_case.optimizer.payload_conservation,
        output / "optimizer_payload_conservation.csv",
    )
    _atomic_csv(
        representative.front_anchors, output / "representative_front_anchors.csv"
    )
    _atomic_csv(
        representative.optimizer.groups,
        output / "representative_optimizer_groups.csv",
    )
    _atomic_csv(
        _comparison(source, representative, representative_id),
        output / "iteration_comparison.csv",
    )
    _atomic_json(validation, output / "validation.json")
    _atomic_json(
        {
            "schema": "pp14-224gpu-extrapolation-summary-v1",
            "source_strategy": {
                "world_size": 256,
                "pipeline_parallel_size": 16,
                "context_parallel_size": 2,
                "data_parallel_size": 8,
                "expert_parallel_size": 8,
            },
            "target_strategy": {
                "world_size": 224,
                "pipeline_parallel_size": 14,
                "context_parallel_size": 2,
                "data_parallel_size": 8,
                "expert_parallel_size": 8,
            },
            "calibration_reused": {
                "pp_network_service_ns": network_ns,
                "pp_backward_software_completion_ns": backward_software_ns,
                "model_flops_per_iteration": source_model_flops,
                "source_model_flops_per_iteration": source_model_flops,
                "target_model_flops_per_iteration": target_model_flops,
                "target_to_source_model_flops_ratio": (
                    target_model_flops / source_model_flops
                ),
                "peak_tflops_per_gpu": peak_tflops,
                "outer_residual_source": "iteration 55 measured outer residual",
                "post_ag_source": "iteration 55 measured post-AG residual",
            },
            "scenario_count": len(sweep),
            "representative_scenario": representative_id,
            "distributions": distributions,
            "source_prediction": {
                "profiler_step_ms": float(
                    source_metrics["predicted_profiler_step_ms"]
                ),
                "training_log_ms": float(
                    source_metrics["predicted_training_log_ms"]
                ),
                "mfu_pct": float(source_metrics["predicted_mfu_pct"]),
            },
            "representative_target_prediction": {
                "profiler_step_ms": float(
                    target_metrics["predicted_profiler_step_ms"]
                ),
                "training_log_ms": float(
                    target_metrics["predicted_training_log_ms"]
                ),
                "mfu_pct": float(target_metrics["predicted_mfu_pct"]),
                "all_dp_service_exposed_ms": float(
                    optimizer_metrics["all_dp_service_exposed_ms"]
                ),
            },
            "target_workload": {
                "num_layers": TARGET_NUM_LAYERS,
                "global_batch_size": TARGET_ARCHITECTURE.global_batch_size,
                "num_micro_batches": 3,
                "timing_model_workload_match": False,
                "mfu_numerator_workload_match": True,
            },
            "evidence_level": "STRUCTURAL_EXTRAPOLATION_WITH_FABRIC_VALIDATION",
        },
        output / "summary.json",
    )
    _atomic_text(
        _report(
            source_metrics,
            target_metrics,
            optimizer_metrics,
            sweep,
            stage_profile,
            omission_sensitivity,
            representative_id,
            distributions,
            source_model_flops,
            target_model_flops,
        ),
        output / "PP14_EXTRAPOLATION_V1.md",
    )
    print(
        f"{validation['status']}: representative={representative_id}, "
        f"step={target_metrics['predicted_profiler_step_ms']:.3f} ms, "
        f"MFU={target_metrics['predicted_mfu_pct']:.4f}% -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build v2 with separate PP network service and Trace software completion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_pp_optimizer_dag_v1 import _atomic_csv, _atomic_json, _atomic_text, _html  # noqa: E402
from x10000_analysis.unified_mfu_dag import build_unified_mfu_dag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PP+RS/AG v2 with cluster-specific P2P software completion."
    )
    parser.add_argument(
        "--trace-events", type=Path, default=CASE_ROOT / "data/pp_dag_trace_events.csv"
    )
    parser.add_argument(
        "--pp-api-events", type=Path, default=CASE_ROOT / "data/pp_api_trace_events.csv"
    )
    parser.add_argument(
        "--optimizer-calls",
        type=Path,
        default=CASE_ROOT / "data/optimizer_timeline_calls.csv",
    )
    parser.add_argument(
        "--iteration-clocks",
        type=Path,
        default=CASE_ROOT / "data/iteration_clocks.csv",
    )
    parser.add_argument(
        "--pp-service-cells",
        type=Path,
        default=CASE_ROOT / "data/pp_framework_sync_cells.csv",
    )
    parser.add_argument(
        "--optimizer-summary",
        type=Path,
        default=CASE_ROOT / "results/mfu_timeline/mfu_timeline_summary.json",
    )
    parser.add_argument("--pp-network-service-ns", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_ROOT / "results/pp_optimizer_dag_v2",
    )
    return parser.parse_args()


def _calibration(api: pd.DataFrame, network_service_ns: int) -> tuple[pd.DataFrame, int]:
    # Stage 0 send_backward is a no-op endpoint; stage 15's final send contains
    # an additional drain tail.  Stages 1-14 provide 848 stable observations of
    # the repeated backward completion path used at every pipeline boundary.
    selected = api[
        api["api_name"].eq("send_backward")
        & api["pp_stage"].between(1, 14)
    ]
    if len(selected) < 800:
        raise ValueError("insufficient stable send_backward calibration observations")
    observed_completion_ns = int(round(selected["duration_ns"].median()))
    software_completion_ns = max(observed_completion_ns - network_service_ns, 0)
    frame = pd.DataFrame(
        [
            {
                "direction": "forward",
                "calibration_source": "network_service_only",
                "stage_filter": "all",
                "observations": 0,
                "observed_api_completion_ns_p10": network_service_ns,
                "observed_api_completion_ns_p50": network_service_ns,
                "observed_api_completion_ns_p90": network_service_ns,
                "network_service_ns": network_service_ns,
                "software_completion_ns": 0,
                "modeled_message_completion_ns": network_service_ns,
            },
            {
                "direction": "backward",
                "calibration_source": "send_backward_user_annotation",
                "stage_filter": "1-14",
                "observations": len(selected),
                "observed_api_completion_ns_p10": int(
                    round(selected["duration_ns"].quantile(0.1))
                ),
                "observed_api_completion_ns_p50": observed_completion_ns,
                "observed_api_completion_ns_p90": int(
                    round(selected["duration_ns"].quantile(0.9))
                ),
                "network_service_ns": network_service_ns,
                "software_completion_ns": software_completion_ns,
                "modeled_message_completion_ns": network_service_ns
                + software_completion_ns,
            },
        ]
    )
    return frame, software_completion_ns


def _report(
    metrics: pd.Series,
    calibration: pd.DataFrame,
    no_software_step_ms: float,
    no_network_step_ms: float,
) -> str:
    backward = calibration[calibration["direction"].eq("backward")].iloc[0]
    relative_mfu_error = 100.0 * (
        metrics["predicted_mfu_pct"] / metrics["actual_mfu_pct_fixed_flops"] - 1.0
    )
    return f"""# PP + Optimizer RS/AG 校准 DAG v2

> v0 `pp_dag_minimal` 与 v1 `pp_optimizer_dag_v1` 全部保留。v2 只新增集群软件 completion 校准，不修改网络 FCT 的语义。

## 精度结果

| 项目 | v2 |
| --- | ---: |
| 实测 ProfilerStep | {metrics['observed_profiler_step_ms']:.3f} ms |
| 预测 ProfilerStep | {metrics['predicted_profiler_step_ms']:.3f} ms |
| Step 误差 | {metrics['profiler_replay_error_ms']:+.3f} ms / {metrics['profiler_replay_error_pct']:+.4f}% |
| 实测 training-log Step | {metrics['observed_training_log_ms']:.3f} ms |
| 预测 training-log Step | {metrics['predicted_training_log_ms']:.3f} ms |
| 固定 FLOPs 实测 MFU | {metrics['actual_mfu_pct_fixed_flops']:.4f}% |
| v2 MFU | {metrics['predicted_mfu_pct']:.4f}% |
| MFU 相对误差 | {relative_mfu_error:+.4f}% |

## 校准拆分

backward message 不再等同于纯网络 FCT：

```text
modeled backward completion
  = network service  {backward['network_service_ns'] / 1e6:.6f} ms
  + software completion {backward['software_completion_ns'] / 1e6:.6f} ms
  = {backward['modeled_message_completion_ns'] / 1e6:.6f} ms
```

软件项来自 iteration 55、PP stage 1–14 的 {int(backward['observations'])} 个 `send_backward` annotation，其中位为 {backward['observed_api_completion_ns_p50'] / 1e6:.6f} ms。stage 0 no-op 和 stage 15 drain tail 没有混入校准。该参数属于当前硬件集群的软件栈，不应作为 OISA/NS-3 网络 FCT。

| 对最终 ProfilerStep 的边际 | 时间 |
| --- | ---: |
| backward 软件 completion | {metrics['predicted_profiler_step_ms'] - no_software_step_ms:.3f} ms |
| PP 网络 service | {metrics['predicted_profiler_step_ms'] - no_network_step_ms:.3f} ms |
| DP/EDP RS/AG service | {metrics['all_dp_service_exposed_ms']:.3f} ms |

## 使用边界

v2 达到的是 iteration 55 的样本内回放精度。它比直接拟合 100 ms 更保守：采用独立 Trace API 的 98.176 ms 中位数，没有把最终 Step 误差调成零。下一步应在其余 19 个 iteration 上做留出验证，再决定软件 completion 使用单一分布、按方向分布还是按 stage/调用类型分层。
"""


def main() -> None:
    args = parse_args()
    trace = pd.read_csv(args.trace_events)
    api = pd.read_csv(args.pp_api_events)
    calls = pd.read_csv(args.optimizer_calls)
    clocks = pd.read_csv(args.iteration_clocks)
    summary = json.loads(args.optimizer_summary.read_text())
    if args.pp_network_service_ns is None:
        cells = pd.read_csv(args.pp_service_cells)
        network_service_ns = int(round(cells["service_reference_fct_ns"].median()))
    else:
        network_service_ns = int(round(args.pp_network_service_ns))
    calibration, backward_software_ns = _calibration(api, network_service_ns)
    base = {
        "trace_events": trace,
        "optimizer_calls": calls,
        "clocks": clocks,
        "model_flops_per_iteration": float(summary["model_flops_per_iteration"]),
        "world_size": int(summary["parallel"]["world_size"]),
        "peak_tflops_per_gpu": float(summary["peak_tflops_per_gpu"]),
    }
    completion = {"forward": 0, "backward": backward_software_ns}
    result = build_unified_mfu_dag(
        pp_service_ns=network_service_ns,
        pp_software_completion_ns=completion,
        **base,
    )
    no_software = build_unified_mfu_dag(
        pp_service_ns=network_service_ns,
        pp_software_completion_ns={"forward": 0, "backward": 0},
        **base,
    )
    no_network = build_unified_mfu_dag(
        pp_service_ns=0,
        pp_software_completion_ns=completion,
        **base,
    )
    metrics = result.iteration.iloc[0]
    no_software_step_ms = float(
        no_software.iteration.iloc[0]["predicted_profiler_step_ms"]
    )
    no_network_step_ms = float(
        no_network.iteration.iloc[0]["predicted_profiler_step_ms"]
    )
    relative_mfu_error = 100.0 * (
        float(metrics["predicted_mfu_pct"])
        / float(metrics["actual_mfu_pct_fixed_flops"])
        - 1.0
    )

    output = args.output_dir
    _atomic_csv(calibration, output / "pp_software_completion_calibration.csv")
    _atomic_csv(result.pp_nodes_absolute, output / "pp_nodes_absolute.csv")
    _atomic_csv(result.pipeline.edges, output / "pp_edges.csv")
    _atomic_csv(result.front_anchors, output / "front_anchors.csv")
    _atomic_csv(result.optimizer.calls, output / "optimizer_calls.csv")
    _atomic_csv(result.optimizer.groups, output / "optimizer_groups.csv")
    _atomic_csv(result.combined_timeline, output / "combined_timeline.csv")
    _atomic_csv(result.dependency_edges, output / "dependency_edges.csv")
    _atomic_csv(result.iteration, output / "iteration_summary.csv")
    validation = {
        "schema": "pp-optimizer-dag-v2",
        "status": "PASS"
        if abs(float(metrics["profiler_replay_error_pct"])) < 1.0
        and abs(relative_mfu_error) < 1.0
        else "FAIL",
        "iteration": int(metrics["iteration"]),
        "preserved_versions": ["results/pp_dag_minimal", "results/pp_optimizer_dag_v1"],
        "calibration": {
            "network_service_ns": network_service_ns,
            "backward_software_completion_ns": backward_software_ns,
            "backward_total_completion_ns": network_service_ns
            + backward_software_ns,
            "observations": int(calibration.iloc[1]["observations"]),
            "source": "send_backward, stages 1-14, iteration 55 median",
        },
        "metrics": {
            "observed_profiler_step_ms": float(metrics["observed_profiler_step_ms"]),
            "predicted_profiler_step_ms": float(metrics["predicted_profiler_step_ms"]),
            "profiler_replay_error_ms": float(metrics["profiler_replay_error_ms"]),
            "profiler_replay_error_pct": float(metrics["profiler_replay_error_pct"]),
            "actual_mfu_pct_fixed_flops": float(metrics["actual_mfu_pct_fixed_flops"]),
            "predicted_mfu_pct": float(metrics["predicted_mfu_pct"]),
            "mfu_relative_error_pct": relative_mfu_error,
            "software_completion_final_step_marginal_ms": float(
                metrics["predicted_profiler_step_ms"] - no_software_step_ms
            ),
            "network_service_final_step_marginal_ms": float(
                metrics["predicted_profiler_step_ms"] - no_network_step_ms
            ),
            "optimizer_service_final_step_marginal_ms": float(
                metrics["all_dp_service_exposed_ms"]
            ),
        },
    }
    _atomic_json(validation, output / "validation.json")
    _atomic_text(
        _report(metrics, calibration, no_software_step_ms, no_network_step_ms),
        output / "PP_OPTIMIZER_DAG_V2.md",
    )
    _atomic_text(
        _html(result, metrics, page_title="PP + RS/AG 校准 DAG v2"),
        output / "pp_optimizer_dag_v2.html",
    )
    print(
        f"v2 iteration {int(metrics['iteration'])}: "
        f"step error={float(metrics['profiler_replay_error_pct']):+.4f}%, "
        f"MFU relative error={relative_mfu_error:+.4f}% -> {output}"
    )


if __name__ == "__main__":
    main()

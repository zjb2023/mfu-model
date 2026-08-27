#!/usr/bin/env python3
"""Exercise the OISA FCT boundary with deterministic mock responses."""

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

from build_pp_optimizer_dag_v1 import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _html,
)
from build_pp_optimizer_dag_v2 import _calibration  # noqa: E402
from x10000_analysis.mfu_timeline import build_collective_slack_audit  # noqa: E402
from x10000_analysis.oisa_fct import (  # noqa: E402
    RecordingFctProvider,
    ScaledMockOisaFctProvider,
    TraceReferenceFctProvider,
)
from x10000_analysis.unified_mfu_dag import build_unified_mfu_dag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v3 using mock OISA collective-elapsed responses."
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
    parser.add_argument(
        "--mock-fct-scale",
        type=float,
        default=1.20,
        help="Scale only OISA tail-after-last-release; arrival span is unchanged.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_ROOT / "results/pp_optimizer_dag_v3_oisa_mock",
    )
    return parser.parse_args()


def _report(
    baseline: pd.Series,
    simulated: pd.Series,
    scale: float,
    audit: pd.DataFrame,
) -> str:
    step_drag = float(simulated["predicted_profiler_step_ms"]) - float(
        baseline["predicted_profiler_step_ms"]
    )
    mfu_change = float(simulated["predicted_mfu_pct"]) - float(
        baseline["predicted_mfu_pct"]
    )
    exposed = audit[audit["iteration_drag_ns"].gt(0)]
    fully_hidden = audit[audit["fully_hidden"]]
    partial = exposed[exposed["hidden_by_slack_ns"].gt(0)]
    top = audit.sort_values(
        ["iteration_drag_ns", "group_finish_shift_ns"], ascending=False
    ).head(8)
    top_rows = "\n".join(
        "| {kind} | PP{stage} | `{group}` | {tail:.3f} | {drag:.3f} | "
        "{hidden:.3f} | {ratio:.1%} |".format(
            kind=row.kind,
            stage=int(row.pp_stage),
            group=str(row.group_key).rsplit(":", 1)[-1],
            tail=row.network_tail_shift_ns / 1e6,
            drag=row.iteration_drag_ns / 1e6,
            hidden=row.hidden_by_slack_ns / 1e6,
            ratio=float(row.exposure_ratio) if pd.notna(row.exposure_ratio) else 0.0,
        )
        for row in top.itertuples(index=False)
    )
    return f"""# PP + Optimizer DAG v3：OISA FCT 接口演练

> 本目录保留 v0/v1/v2，v3 不声称预测了真实网络。它使用 `Mock OISA = Trace 网络尾部 × {scale:.3f}` 验证接口、依赖传播和 slack 计算，等真实 OISA 返回后只替换 Provider。

## 本次接入的准确语义

```text
MFU DAG 计算每个 rank 的 arrival
  → release_offset[r] = arrival[r] - min(arrival)
  → OISA(rank_release_offsets, op, payload, group)
  → collective_elapsed = arrival_span + tail_after_last_release
  → group_end = min(arrival) + collective_elapsed
```

模型不会使用 `max(arrival) + collective_elapsed`，因此 arrival span 不会被重复计算。网络 FCT 只替换 service；每个 rank 相对 group end 的 completion skew 仍来自 Trace，并与网络缩放解耦。

## Mock 场景结果

| 指标 | Trace-reference FCT | Mock OISA | 变化 |
| --- | ---: | ---: | ---: |
| ProfilerStep | {baseline['predicted_profiler_step_ms']:.3f} ms | {simulated['predicted_profiler_step_ms']:.3f} ms | {step_drag:+.3f} ms |
| 固定 FLOP MFU | {baseline['predicted_mfu_pct']:.4f}% | {simulated['predicted_mfu_pct']:.4f}% | {mfu_change:+.4f} pp |
| 相对实测 Step 误差 | {baseline['profiler_replay_error_pct']:+.4f}% | {simulated['profiler_replay_error_pct']:+.4f}% | — |

这里把六类已显式建模的 optimizer collective 网络尾部统一增加 {(scale - 1.0) * 100:.1f}%。这是接口测试，不是 OISA 性能结论。

## Slack 审计

本次对 {len(audit):,} 个 collective group 计算“只替换该组 FCT”的独立反事实敏感性。实现上只回放一次基线 DAG，再通过反向 max-plus 传播得到每组到 iteration 终点的 downstream slack；它与逐组重跑的正向 slowdown 结果等价：

- 完全隐藏：{len(fully_hidden):,} 组；
- 至少拖累 iteration：{len(exposed):,} 组；
- 先消耗部分 slack、随后拖累：{len(partial):,} 组。

单组字段定义：

```text
group_finish_shift = candidate_group_end - reference_group_end
iteration_drag     = candidate_step_end  - reference_step_end
hidden_by_slack    = group_finish_shift  - iteration_drag
exposure_ratio     = iteration_drag / group_finish_shift
```

各组 `iteration_drag` 是独立敏感性实验，不能直接相加；多个通信同时变慢时必须像本页 Mock 场景一样重新跑完整 DAG。

| kind | stage | group | 网络尾部增加 ms | Step 拖累 ms | slack 隐藏 ms | 暴露比例 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
{top_rows}

## 可替换边界

- `oisa_mock_requests.csv`：MFU 发给 OISA 的请求形状，包含各 rank release offsets；
- `oisa_mock_responses.csv`：符合约定的 OISA 返回形状；
- `collective_slack_audit.csv`：逐组独立 slowdown 的隐藏/暴露结果；
- `optimizer_groups.csv`：完整 Mock 场景传播后的 group 时间；
- `pp_optimizer_dag_v3_oisa_mock.html`：Mock 场景的完整关键 lane 时间线。

真实 OISA 上线后，Provider 必须返回 `collective_elapsed_ns`、`arrival_span_ns` 和 `tail_after_last_release_ns`，并满足：

```text
collective_elapsed_ns
  = arrival_span_ns + tail_after_last_release_ns
```
"""


def main() -> None:
    args = parse_args()
    if args.mock_fct_scale < 0:
        raise ValueError("--mock-fct-scale must be non-negative")
    trace = pd.read_csv(args.trace_events)
    api = pd.read_csv(args.pp_api_events)
    calls = pd.read_csv(args.optimizer_calls)
    clocks = pd.read_csv(args.iteration_clocks)
    summary = json.loads(args.optimizer_summary.read_text())
    cells = pd.read_csv(args.pp_service_cells)
    network_service_ns = int(round(cells["service_reference_fct_ns"].median()))
    calibration, backward_software_ns = _calibration(api, network_service_ns)
    completion = {"forward": 0, "backward": backward_software_ns}
    common = {
        "trace_events": trace,
        "pp_service_ns": network_service_ns,
        "pp_software_completion_ns": completion,
        "optimizer_calls": calls,
        "clocks": clocks,
        "model_flops_per_iteration": float(summary["model_flops_per_iteration"]),
        "world_size": int(summary["parallel"]["world_size"]),
        "peak_tflops_per_gpu": float(summary["peak_tflops_per_gpu"]),
    }

    reference_provider = TraceReferenceFctProvider()
    baseline_result = build_unified_mfu_dag(
        optimizer_fct_provider=reference_provider,
        **common,
    )
    recording_provider = RecordingFctProvider(
        ScaledMockOisaFctProvider(default_scale=args.mock_fct_scale)
    )
    simulated_result = build_unified_mfu_dag(
        optimizer_fct_provider=recording_provider,
        **common,
    )
    iteration = int(trace["iteration"].iloc[0])
    selected_calls = calls[calls["iteration"].eq(iteration)]
    selected_clocks = clocks[clocks["iteration"].eq(iteration)]
    audit = build_collective_slack_audit(
        selected_calls,
        selected_clocks,
        reference_provider=reference_provider,
        candidate_provider=ScaledMockOisaFctProvider(
            default_scale=args.mock_fct_scale
        ),
        front_anchors=baseline_result.front_anchors[
            ["iteration", "rank", "predicted_start_ns"]
        ],
    )

    baseline = baseline_result.iteration.iloc[0]
    simulated = simulated_result.iteration.iloc[0]
    baseline_ids = set(simulated_result.optimizer.groups["fct_request_id"])
    requests = pd.DataFrame(recording_provider.requests)
    responses = pd.DataFrame(recording_provider.responses)
    requests = requests[requests["request_id"].isin(baseline_ids)].drop_duplicates(
        "request_id"
    )
    responses = responses[responses["request_id"].isin(baseline_ids)].drop_duplicates(
        "request_id"
    )
    if len(requests) != len(simulated_result.optimizer.groups) or len(responses) != len(
        simulated_result.optimizer.groups
    ):
        raise ValueError("recorded baseline OISA request/response coverage is incomplete")

    output = args.output_dir
    _atomic_csv(calibration, output / "pp_software_completion_calibration.csv")
    _atomic_csv(requests, output / "oisa_mock_requests.csv")
    _atomic_csv(responses, output / "oisa_mock_responses.csv")
    _atomic_csv(audit, output / "collective_slack_audit.csv")
    _atomic_csv(simulated_result.pp_nodes_absolute, output / "pp_nodes_absolute.csv")
    _atomic_csv(simulated_result.pipeline.edges, output / "pp_edges.csv")
    _atomic_csv(simulated_result.front_anchors, output / "front_anchors.csv")
    _atomic_csv(simulated_result.optimizer.calls, output / "optimizer_calls.csv")
    _atomic_csv(simulated_result.optimizer.groups, output / "optimizer_groups.csv")
    _atomic_csv(simulated_result.combined_timeline, output / "combined_timeline.csv")
    _atomic_csv(simulated_result.dependency_edges, output / "dependency_edges.csv")
    comparison = pd.DataFrame(
        [
            {"scenario": "trace_reference", **baseline.to_dict()},
            {"scenario": f"mock_oisa_x{args.mock_fct_scale:g}", **simulated.to_dict()},
        ]
    )
    _atomic_csv(comparison, output / "iteration_comparison.csv")

    step_drag = float(simulated["predicted_profiler_step_ms"]) - float(
        baseline["predicted_profiler_step_ms"]
    )
    validation = {
        "schema": "pp-optimizer-dag-v3-oisa-mock",
        "status": "PASS",
        "iteration": iteration,
        "mock_only_not_network_prediction": True,
        "mock_tail_scale": args.mock_fct_scale,
        "oisa_groups": len(requests),
        "slack_audit_groups": len(audit),
        "identity": {
            "all_response_elapsed_conserved": bool(
                (
                    responses["collective_elapsed_ns"]
                    == responses["arrival_span_ns"]
                    + responses["tail_after_last_release_ns"]
                ).all()
            ),
            "all_requests_have_normalized_release": bool(
                requests["rank_release_offsets_ns"].str.contains(r"(?:^|;)\d+:0(?:;|$)")
                .all()
            ),
            "baseline_replay_error_pct": float(
                baseline["profiler_replay_error_pct"]
            ),
        },
        "counterfactual": {
            "profiler_step_drag_ms": step_drag,
            "mfu_change_percentage_points": float(simulated["predicted_mfu_pct"])
            - float(baseline["predicted_mfu_pct"]),
            "fully_hidden_groups": int(audit["fully_hidden"].sum()),
            "iteration_exposed_groups": int(audit["iteration_drag_ns"].gt(0).sum()),
        },
    }
    if not validation["identity"]["all_response_elapsed_conserved"]:
        validation["status"] = "FAIL"
    if not validation["identity"]["all_requests_have_normalized_release"]:
        validation["status"] = "FAIL"
    _atomic_json(validation, output / "validation.json")
    _atomic_text(
        _report(baseline, simulated, args.mock_fct_scale, audit),
        output / "PP_OPTIMIZER_DAG_V3_OISA_MOCK.md",
    )
    _atomic_text(
        _html(
            simulated_result,
            simulated,
            page_title=f"PP + RS/AG DAG v3 · Mock OISA ×{args.mock_fct_scale:g}",
            focus_critical_lane=True,
        ),
        output / "pp_optimizer_dag_v3_oisa_mock.html",
    )
    print(
        f"v3 mock OISA iteration {iteration}: {len(requests)} groups, "
        f"Step {step_drag:+.3f} ms, "
        f"{int(audit['fully_hidden'].sum())} fully hidden -> {output}"
    )


if __name__ == "__main__":
    main()

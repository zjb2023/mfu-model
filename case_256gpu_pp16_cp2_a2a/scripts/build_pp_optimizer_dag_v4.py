#!/usr/bin/env python3
"""Backfill nine representative OISA service points into the MFU DAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
OISA_SOURCE_SNAPSHOT_COMMIT = "2bd0e53d0e572b434490877d6aa3911973d1d63b"
OISA_SOURCE_SNAPSHOT_BRANCH = "feat/rank-release-fct"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_pp_optimizer_dag_v1 import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _html,
)
from build_pp_optimizer_dag_v2 import _calibration  # noqa: E402
from x10000_analysis.oisa_fct import (  # noqa: E402
    RecordingFctProvider,
    RepresentativeOisaFctProvider,
    TraceReferenceFctProvider,
)
from x10000_analysis.mfu_timeline import build_collective_slack_audit  # noqa: E402
from x10000_analysis.unified_mfu_dag import build_unified_mfu_dag  # noqa: E402


def parse_args() -> argparse.Namespace:
    backfill = CASE_ROOT / "results/oisa_s5000_256gpu_nine_class/backfill"
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--collective-calibration",
        type=Path,
        default=backfill / "collective_service_calibration.csv",
    )
    parser.add_argument(
        "--pipeline-adjustments",
        type=Path,
        default=backfill / "pipeline_node_adjustments.csv",
    )
    parser.add_argument(
        "--pipeline-adjustments-network-only",
        type=Path,
        default=backfill / "pipeline_node_adjustments_network_only.csv",
    )
    parser.add_argument(
        "--optimizer-calibration",
        type=Path,
        default=backfill / "optimizer_kind_calibration.csv",
    )
    parser.add_argument(
        "--backfill-validation",
        type=Path,
        default=backfill / "validation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_ROOT / "results/pp_optimizer_dag_v4_oisa_s5000",
    )
    return parser.parse_args()


def _metric_row(name: str, result: object) -> dict[str, object]:
    metrics = result.iteration.iloc[0]
    return {"scenario": name, **metrics.to_dict()}


def _report(
    comparison: pd.DataFrame,
    collective_calibration: pd.DataFrame,
    pp_calibration: pd.DataFrame,
    pipeline_adjustments: pd.DataFrame,
    requests: pd.DataFrame,
    slack_audit: pd.DataFrame,
    validation: dict[str, object],
) -> str:
    metric = comparison.set_index("scenario")
    baseline = metric.loc["trace_reference"]
    network_pipeline = metric.loc["oisa_network_pipeline_only"]
    network_optimizer = metric.loc["oisa_network_optimizer_only"]
    network_combined = metric.loc["oisa_network_all_nine_classes"]
    calibrated_pipeline = metric.loc["oisa_calibrated_pipeline_only"]
    calibrated_optimizer = metric.loc["oisa_calibrated_optimizer_only"]
    combined = metric.loc["oisa_calibrated_all_nine_classes"]
    delta = lambda row: float(row["predicted_profiler_step_ms"]) - float(  # noqa: E731
        baseline["predicted_profiler_step_ms"]
    )
    class_rows = "\n".join(
        "| {behavior} | {group_size} | {payload:.3f} | {trace:.3f} | "
        "{oisa:.3f} | {ratio:.3f}× |".format(
            behavior=row.behavior,
            group_size=int(row.group_size),
            payload=int(row.reference_payload_bytes) / 2**20,
            trace=int(row.observed_service_ns) / 1e6,
            oisa=int(row.tail_after_last_release_ns) / 1e6,
            ratio=float(row.service_ratio_oisa_to_trace),
        )
        for row in collective_calibration.itertuples(index=False)
    )
    backward = pp_calibration[pp_calibration["direction"].eq("backward")].iloc[0]
    topology_hashes = sorted(set(collective_calibration["topology_hash"].astype(str)))
    simulator_commits = sorted(
        set(collective_calibration["simulator_commit"].astype(str))
    )
    simulator_binaries = sorted(
        set(collective_calibration["simulator_binary_sha256"].astype(str))
    )
    worktree_dirty = bool(collective_calibration["simulator_worktree_dirty"].any())
    critical_groups = int(slack_audit["downstream_slack_ns"].eq(0).sum())
    fully_slack_groups = int(slack_audit["downstream_slack_ns"].gt(0).sum())
    return f"""# MFU DAG v4：256-GPU S5000 九类集合通信回填

> v4 保留 v2 的 PP send/recv 网络与软件 completion 校准，只替换 PP 之外九类集合通信的网络 service。它是“每类一个代表点 + 按字节线性缩放”的首版接入，不等同于多尺寸、多并发条件下的最终 OISA 性能曲线。

## 回放结果

| 场景 | ProfilerStep | 相对实测误差 | 固定 FLOPs MFU | 相对 v2 Step 变化 |
| --- | ---: | ---: | ---: | ---: |
| Trace-reference（v2 语义） | {baseline['predicted_profiler_step_ms']:.3f} ms | {baseline['profiler_replay_error_pct']:+.4f}% | {baseline['predicted_mfu_pct']:.4f}% | — |
| 诊断：仅 OISA 网络，FWD/BWD 五类 | {network_pipeline['predicted_profiler_step_ms']:.3f} ms | {network_pipeline['profiler_replay_error_pct']:+.4f}% | {network_pipeline['predicted_mfu_pct']:.4f}% | {delta(network_pipeline):+.3f} ms |
| 诊断：仅 OISA 网络，optimizer 四类 | {network_optimizer['predicted_profiler_step_ms']:.3f} ms | {network_optimizer['profiler_replay_error_pct']:+.4f}% | {network_optimizer['predicted_mfu_pct']:.4f}% | {delta(network_optimizer):+.3f} ms |
| 诊断：仅 OISA 网络，九类合并 | {network_combined['predicted_profiler_step_ms']:.3f} ms | {network_combined['profiler_replay_error_pct']:+.4f}% | {network_combined['predicted_mfu_pct']:.4f}% | {delta(network_combined):+.3f} ms |
| 校准：OISA 网络 + Trace 有符号残差，FWD/BWD 五类 | {calibrated_pipeline['predicted_profiler_step_ms']:.3f} ms | {calibrated_pipeline['profiler_replay_error_pct']:+.4f}% | {calibrated_pipeline['predicted_mfu_pct']:.4f}% | {delta(calibrated_pipeline):+.3f} ms |
| 校准：OISA 网络 + Trace 有符号残差，optimizer 四类 | {calibrated_optimizer['predicted_profiler_step_ms']:.3f} ms | {calibrated_optimizer['profiler_replay_error_pct']:+.4f}% | {calibrated_optimizer['predicted_mfu_pct']:.4f}% | {delta(calibrated_optimizer):+.3f} ms |
| 最终：九类合并 | {combined['predicted_profiler_step_ms']:.3f} ms | {combined['profiler_replay_error_pct']:+.4f}% | {combined['predicted_mfu_pct']:.4f}% | {delta(combined):+.3f} ms |

两个单独变化不能相加；九类合并后会重新执行完整 max-plus DAG，通信变慢先消耗 slack，只有越过 slack 的部分才拖累 iteration。

当前 baseline topology 与 target topology 相同，所以校准场景的网络 delta 为 0；432 个 optimizer group 中有 {critical_groups} 个 downstream slack 为 0，{fully_slack_groups} 个有正 slack。未来 target OISA tail 变慢且逐 Rank completion 结构不变时，单组拖累按 `max(0, target_network_delta - downstream_slack)` 计算；多个组同时变化仍必须完整重放 DAG，不能把单组拖累直接相加。

## 接入逻辑

```text
五类 EP/CP A2A:
  保留 Trace 的 rank arrival wait 与 completion advance
  → network = OISA tail
  → calibration residual = Trace group service - baseline OISA network
  → target service = target OISA network + signed calibration residual
  → service 差额分配回对应 rank 的 FWD/BWD compute node

DP/Expert-DP RS/AG:
  DAG 实时产生各 rank arrival
  → arrival offsets 交给 RepresentativeOisaFctProvider
  → 按 payload/reference_payload 缩放 baseline/target OISA network tail
  → 保留每个 Trace request 的有符号基线校准残差
  → group_end = first_arrival + arrival_span + target_network + calibration_residual
```

五类 A2A/CP 暂时嵌入 FWD/BWD node，是因为当前 PP DAG 没有把这些框架通信拆成独立 node；这不会把实测同步等待再次相加，但尚不能表达同一 phase 内更细粒度的 compute/communication overlap。DP/EDP 六种 optimizer kind（RS、AG0、AG1）本来就是独立 DAG node，使用四类 OISA 代表值覆盖。有符号残差的正值可解释为软件/同步残差；负值表示当前 OISA 算法、payload 语义或重叠行为相对 Trace 的系统偏差，不能误称为负软件开销。

PP 通信不在这九类中，仍沿用 v2：网络 service {int(backward['network_service_ns']) / 1e6:.6f} ms，backward 软件 completion {int(backward['software_completion_ns']) / 1e6:.6f} ms。

## 九类代表点

| behavior | group size | payload MiB | Trace service ms | OISA tail ms | OISA/Trace |
| --- | ---: | ---: | ---: | ---: | ---: |
{class_rows}

## 可审计边界

- OISA simulator commit：`{', '.join(simulator_commits)}`；
- 仿真运行时 worktree dirty：`{str(worktree_dirty).lower()}`；对应完整源码快照已提交到 `{OISA_SOURCE_SNAPSHOT_BRANCH}` / `{OISA_SOURCE_SNAPSHOT_COMMIT}`；
- simulator binary SHA256：`{', '.join(simulator_binaries)}`；
- topology hash：`{', '.join(topology_hashes)}`；
- network-only pipeline node 调整总量：{int(validation['pipeline_network_only_adjustment_ns_total']) / 1e6:+.3f} ms；
- 保留有符号基线校准残差后的 pipeline node 调整总量：{int(pipeline_adjustments['adjustment_ns'].sum()) / 1e6:+.3f} ms；
- optimizer 动态请求数：{len(requests):,}；
- pipeline 对齐覆盖：{float(validation['pipeline_event_alignment_fraction']):.2%}；
- 最终状态：`{validation['status']}`。

下一步若要承担并行策略外推，需对每类补充 payload、group size、跨机比例和并发度的多点 OISA 曲线；当前线性单点模型首先用于验证 service 回填、slack 隐藏和关键路径传播是否正确。
"""


def main() -> int:
    args = parse_args()
    trace = pd.read_csv(args.trace_events)
    api = pd.read_csv(args.pp_api_events)
    calls = pd.read_csv(args.optimizer_calls)
    clocks = pd.read_csv(args.iteration_clocks)
    summary = json.loads(args.optimizer_summary.read_text(encoding="utf-8"))
    collective_calibration = pd.read_csv(args.collective_calibration)
    pipeline_adjustments = pd.read_csv(args.pipeline_adjustments)
    pipeline_adjustments_network_only = pd.read_csv(
        args.pipeline_adjustments_network_only
    )
    optimizer_calibration = pd.read_csv(args.optimizer_calibration)
    backfill_validation = json.loads(
        args.backfill_validation.read_text(encoding="utf-8")
    )

    cells = pd.read_csv(args.pp_service_cells)
    pp_network_service_ns = int(round(cells["service_reference_fct_ns"].median()))
    pp_calibration, backward_software_ns = _calibration(api, pp_network_service_ns)
    completion = {"forward": 0, "backward": backward_software_ns}
    common = {
        "trace_events": trace,
        "pp_service_ns": pp_network_service_ns,
        "pp_software_completion_ns": completion,
        "optimizer_calls": calls,
        "clocks": clocks,
        "model_flops_per_iteration": float(summary["model_flops_per_iteration"]),
        "world_size": int(summary["parallel"]["world_size"]),
        "peak_tflops_per_gpu": float(summary["peak_tflops_per_gpu"]),
        # Seven comparison builds are sufficient here; the v3 slack audit owns
        # the additional no-RS/no-AG counterfactuals.
        "optimizer_calculate_service_marginals": False,
    }

    reference = TraceReferenceFctProvider()
    baseline = build_unified_mfu_dag(optimizer_fct_provider=reference, **common)
    network_pipeline_only = build_unified_mfu_dag(
        optimizer_fct_provider=reference,
        pipeline_compute_duration_adjustments=pipeline_adjustments_network_only,
        **common,
    )
    network_optimizer_provider = RepresentativeOisaFctProvider(
        optimizer_calibration.to_dict(orient="records")
    )
    network_optimizer_only = build_unified_mfu_dag(
        optimizer_fct_provider=network_optimizer_provider,
        **common,
    )
    network_combined = build_unified_mfu_dag(
        optimizer_fct_provider=network_optimizer_provider,
        pipeline_compute_duration_adjustments=pipeline_adjustments_network_only,
        **common,
    )
    calibrated_pipeline_only = build_unified_mfu_dag(
        optimizer_fct_provider=reference,
        pipeline_compute_duration_adjustments=pipeline_adjustments,
        **common,
    )
    calibrated_optimizer_provider = RepresentativeOisaFctProvider(
        optimizer_calibration.to_dict(orient="records"),
        preserve_trace_calibration_residual=True,
    )
    calibrated_optimizer_only = build_unified_mfu_dag(
        optimizer_fct_provider=calibrated_optimizer_provider,
        **common,
    )
    recording = RecordingFctProvider(
        RepresentativeOisaFctProvider(
            optimizer_calibration.to_dict(orient="records"),
            preserve_trace_calibration_residual=True,
        )
    )
    combined = build_unified_mfu_dag(
        optimizer_fct_provider=recording,
        pipeline_compute_duration_adjustments=pipeline_adjustments,
        **common,
    )
    modeled_iteration = int(trace["iteration"].iloc[0])
    slack_audit = build_collective_slack_audit(
        calls[calls["iteration"].eq(modeled_iteration)],
        clocks[clocks["iteration"].eq(modeled_iteration)],
        reference_provider=reference,
        candidate_provider=RepresentativeOisaFctProvider(
            optimizer_calibration.to_dict(orient="records"),
            preserve_trace_calibration_residual=True,
        ),
        front_anchors=combined.front_anchors[
            ["iteration", "rank", "predicted_start_ns"]
        ],
    )

    requests = pd.DataFrame(recording.requests).drop_duplicates("request_id")
    responses = pd.DataFrame(recording.responses).drop_duplicates("request_id")
    if len(requests) != len(combined.optimizer.groups) or len(responses) != len(
        combined.optimizer.groups
    ):
        raise ValueError("recorded OISA request/response coverage is incomplete")

    comparison = pd.DataFrame(
        [
            _metric_row("trace_reference", baseline),
            _metric_row("oisa_network_pipeline_only", network_pipeline_only),
            _metric_row("oisa_network_optimizer_only", network_optimizer_only),
            _metric_row("oisa_network_all_nine_classes", network_combined),
            _metric_row("oisa_calibrated_pipeline_only", calibrated_pipeline_only),
            _metric_row("oisa_calibrated_optimizer_only", calibrated_optimizer_only),
            _metric_row("oisa_calibrated_all_nine_classes", combined),
        ]
    )
    metrics = combined.iteration.iloc[0]
    relative_mfu_error = 100.0 * (
        float(metrics["predicted_mfu_pct"])
        / float(metrics["actual_mfu_pct_fixed_flops"])
        - 1.0
    )
    elapsed_conserved = bool(
        (
            responses["collective_elapsed_ns"]
            == responses["arrival_span_ns"]
            + responses["tail_after_last_release_ns"]
        ).all()
    )
    topology_hashes = sorted(
        set(collective_calibration["topology_hash"].dropna().astype(str))
    )
    commits = sorted(
        set(collective_calibration["simulator_commit"].dropna().astype(str))
    )
    status = "PASS"
    failures: list[str] = []
    if backfill_validation.get("status") != "PASS":
        failures.append("backfill validation failed")
    if not elapsed_conserved:
        failures.append("optimizer elapsed conservation failed")
    if len(collective_calibration) != 9:
        failures.append("nine-class calibration coverage is incomplete")
    if len(slack_audit) != len(combined.optimizer.groups):
        failures.append("optimizer slack coverage is incomplete")
    if abs(float(metrics["profiler_replay_error_pct"])) >= 1.0:
        failures.append("combined replay error is at least 1%")
    if failures:
        status = "FAIL"
    validation = {
        "schema": "pp-optimizer-dag-v4-oisa-s5000",
        "status": status,
        "failures": failures,
        "iteration": int(metrics["iteration"]),
        "collective_classes": len(collective_calibration),
        "optimizer_dynamic_groups": len(requests),
        "optimizer_slack_groups": len(slack_audit),
        "optimizer_zero_downstream_slack_groups": int(
            slack_audit["downstream_slack_ns"].eq(0).sum()
        ),
        "optimizer_positive_downstream_slack_groups": int(
            slack_audit["downstream_slack_ns"].gt(0).sum()
        ),
        "pipeline_event_alignment_fraction": float(
            backfill_validation["pipeline_event_alignment_fraction"]
        ),
        "pipeline_network_only_adjustment_ns_total": int(
            backfill_validation["pipeline_network_only_adjustment_ns_total"]
        ),
        "pipeline_calibrated_adjustment_ns_total": int(
            backfill_validation["pipeline_adjustment_ns_total"]
        ),
        "oisa_simulator_commits": commits,
        "topology_hashes": topology_hashes,
        "identity": {
            "all_optimizer_elapsed_conserved": elapsed_conserved,
            "pipeline_adjustment_conservation_error_ns_max": int(
                backfill_validation["pipeline_adjustment_conservation_error_ns_max"]
            ),
        },
        "metrics": {
            "observed_profiler_step_ms": float(metrics["observed_profiler_step_ms"]),
            "predicted_profiler_step_ms": float(metrics["predicted_profiler_step_ms"]),
            "profiler_replay_error_ms": float(metrics["profiler_replay_error_ms"]),
            "profiler_replay_error_pct": float(metrics["profiler_replay_error_pct"]),
            "actual_mfu_pct_fixed_flops": float(
                metrics["actual_mfu_pct_fixed_flops"]
            ),
            "predicted_mfu_pct": float(metrics["predicted_mfu_pct"]),
            "mfu_relative_error_pct": relative_mfu_error,
            "network_only_profiler_replay_error_pct": float(
                comparison.set_index("scenario").loc[
                    "oisa_network_all_nine_classes", "profiler_replay_error_pct"
                ]
            ),
        },
        "model_boundary": {
            "pp": "unchanged v2 network service plus trace software completion",
            "pipeline_five_classes": (
                "preserve trace arrival/completion skew and signed baseline "
                "calibration residual; replace network service and embed the net delta in "
                "FWD/BWD nodes"
            ),
            "optimizer_four_classes": (
                "dynamic DAG arrivals plus representative OISA network tail scaled "
                "by bytes and per-request signed Trace calibration residual"
            ),
        },
    }

    output = args.output_dir
    _atomic_csv(pp_calibration, output / "pp_software_completion_calibration.csv")
    _atomic_csv(
        collective_calibration, output / "collective_service_calibration.csv"
    )
    _atomic_csv(pipeline_adjustments, output / "pipeline_node_adjustments.csv")
    _atomic_csv(
        pipeline_adjustments_network_only,
        output / "pipeline_node_adjustments_network_only.csv",
    )
    _atomic_csv(optimizer_calibration, output / "optimizer_kind_calibration.csv")
    _atomic_csv(requests, output / "oisa_optimizer_requests.csv")
    _atomic_csv(responses, output / "oisa_optimizer_responses.csv")
    _atomic_csv(slack_audit, output / "collective_slack_audit.csv")
    _atomic_csv(comparison, output / "iteration_comparison.csv")
    _atomic_csv(combined.pp_nodes_absolute, output / "pp_nodes_absolute.csv")
    _atomic_csv(combined.pipeline.edges, output / "pp_edges.csv")
    _atomic_csv(combined.front_anchors, output / "front_anchors.csv")
    _atomic_csv(combined.optimizer.calls, output / "optimizer_calls.csv")
    _atomic_csv(combined.optimizer.groups, output / "optimizer_groups.csv")
    _atomic_csv(combined.combined_timeline, output / "combined_timeline.csv")
    _atomic_csv(combined.dependency_edges, output / "dependency_edges.csv")
    _atomic_csv(combined.iteration, output / "iteration_summary.csv")
    _atomic_json(validation, output / "validation.json")
    _atomic_text(
        _report(
            comparison,
            collective_calibration,
            pp_calibration,
            pipeline_adjustments,
            requests,
            slack_audit,
            validation,
        ),
        output / "PP_OPTIMIZER_DAG_V4_OISA_S5000.md",
    )
    _atomic_text(
        _html(
            combined,
            metrics,
            page_title="PP + 九类集合通信 DAG v4 · OISA S5000 256-GPU",
            focus_critical_lane=True,
        ),
        output / "pp_optimizer_dag_v4_oisa_s5000.html",
    )
    print(
        f"v4 iteration {int(metrics['iteration'])}: status={status}, "
        f"step error={float(metrics['profiler_replay_error_pct']):+.4f}%, "
        f"MFU relative error={relative_mfu_error:+.4f}% -> {output}"
    )
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

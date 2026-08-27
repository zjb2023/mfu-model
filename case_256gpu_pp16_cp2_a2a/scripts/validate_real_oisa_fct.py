#!/usr/bin/env python3
"""Run one measured MFU collective through real OISA and replay the DAG."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
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
from x10000_analysis.oisa_fct import (  # noqa: E402
    RecordedOisaFctProvider,
    SelectiveFctProvider,
    TraceReferenceFctProvider,
    make_collective_request,
)
from x10000_analysis.unified_mfu_dag import build_unified_mfu_dag  # noqa: E402


DEFAULT_OISA_COMMIT = "faa3c789549d542484ca9aab132b7d4727afd14a"
DEFAULT_TARGET = (
    "iter55:pp14:edp_ag0:round0:expert_dp_param_allgather:pg=1279"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oisa-root", type=Path, default=Path("/tmp/oisa-rank-release-fct")
    )
    parser.add_argument("--expected-oisa-commit", default=DEFAULT_OISA_COMMIT)
    parser.add_argument("--target-group-id", default=DEFAULT_TARGET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_ROOT / "results/oisa_real_validation_faa3c78",
    )
    return parser.parse_args()


def _load_oisa(oisa_root: Path):
    if str(oisa_root) not in sys.path:
        sys.path.insert(0, str(oisa_root))
    from oisa_sim.collective_fct import CollectiveRequest  # type: ignore
    from oisa_sim.topology import parse_topology_file  # type: ignore

    smoke_path = oisa_root / "scripts/run_rank_release_smoke.py"
    spec = importlib.util.spec_from_file_location("oisa_rank_release_smoke", smoke_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load OISA smoke entry: {smoke_path}")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    return CollectiveRequest, parse_topology_file, smoke


def _report(
    *,
    target_group_id: str,
    original_ranks: list[int],
    rank_mapping: pd.DataFrame,
    oisa_commit: str,
    topology_hash: str,
    topology_gpu_count: int,
    payload_bytes: int,
    interface_summary: dict,
    sync: dict,
    staggered: dict,
    baseline_metric: pd.Series,
    tail_only_metric: pd.Series,
    hybrid_metric: pd.Series,
    reference_tail_ns: int,
    oisa_tail_ns: int,
    target_completion_models: list[str],
    dp_group_size: int,
) -> str:
    tail_step_delta = float(tail_only_metric["predicted_profiler_step_ms"]) - float(
        baseline_metric["predicted_profiler_step_ms"]
    )
    rank_done_step_delta = float(hybrid_metric["predicted_profiler_step_ms"]) - float(
        tail_only_metric["predicted_profiler_step_ms"]
    )
    step_delta = tail_step_delta + rank_done_step_delta
    mfu_delta = float(hybrid_metric["predicted_mfu_pct"]) - float(
        baseline_metric["predicted_mfu_pct"]
    )
    mapping_text = ", ".join(
        f"{int(row.original_rank)}→{int(row.oisa_local_rank)}"
        for row in rank_mapping.itertuples(index=False)
    )
    return f"""# 真实 OISA FCT → MFU DAG 单组联调

> 这是一项接口和传播验证，不是目标 256-GPU 集群的网络校准。OISA 使用本地提交 `{oisa_commit}` 和固定 8-GPU 验证拓扑。

## 做了什么

1. 从 iteration 55 的 MFU 基线 DAG 选择零 downstream slack 的关键通信组：
   `{target_group_id}`；
2. 保留实测 payload `{payload_bytes:,}` bytes 和 Rank arrival offsets；
3. 因 OISA smoke 拓扑只有 {topology_gpu_count} Rank，把实测 Rank `{original_ranks}` 按 `{mapping_text}` 映射到拓扑本地 Rank；
4. 分别运行同步到达和实测错峰到达的 OISA/ns-3；
5. 校验 flow release/dependency gate、FCT 恒等式、commit 和 topology hash；
6. 将错峰 OISA 返回通过 `RecordedOisaFctProvider` 只替换这个通信组，其他 431 组仍使用 Trace-reference FCT；
7. 重新执行完整 PP→RS/AG max-plus DAG，并使用 OISA 的逐 Rank network-done 时间替换该组的 Trace completion skew。

## 通用错峰接口 smoke

在同一 OISA commit/topology 上额外运行了 4-Rank、8 MiB AllToAll：

| 指标 | 同步到达 | `[0,20,50,30] ms` 错峰到达 |
| --- | ---: | ---: |
| flow start times | `{interface_summary['synchronized']['observed_flow_start_times_ns']}` | `{interface_summary['staggered']['observed_flow_start_times_ns']}` |
| collective elapsed | {interface_summary['synchronized']['collective_elapsed_ns'] / 1e6:.6f} ms | {interface_summary['staggered']['collective_elapsed_ns'] / 1e6:.6f} ms |
| tail after last release | {interface_summary['synchronized']['tail_after_last_release_ns'] / 1e6:.6f} ms | {interface_summary['staggered']['tail_after_last_release_ns'] / 1e6:.6f} ms |

12 条 flow 的 start time 全部由 0 变成 20/30/50 ms 三个 release 波次，且 12 条 flow 的 FCT 全部变化，说明错峰不是在最终结果上简单加 offset，而是真正改变了 ns-3 中的流量竞争。

## OISA 结果

| 指标 | 同步 Rank | 实测错峰 Rank |
| --- | ---: | ---: |
| arrival span | {sync['arrival_span_ns'] / 1e6:.6f} ms | {staggered['arrival_span_ns'] / 1e6:.6f} ms |
| collective elapsed | {sync['collective_elapsed_ns'] / 1e6:.6f} ms | {staggered['collective_elapsed_ns'] / 1e6:.6f} ms |
| tail after last release | {sync['tail_after_last_release_ns'] / 1e6:.6f} ms | {staggered['tail_after_last_release_ns'] / 1e6:.6f} ms |
| dependency gate | {sync['dependency_gate_valid']} | {staggered['dependency_gate_valid']} |

本例的实测 arrival span 是 {staggered['arrival_span_ns'] / 1e3:.3f} μs。错峰结果满足：

```text
collective_elapsed
  = arrival_span + tail_after_last_release
  = {staggered['arrival_span_ns']} + {staggered['tail_after_last_release_ns']}
  = {staggered['collective_elapsed_ns']} ns
```

## 放回 MFU DAG 后

| 指标 | Trace-reference | 只换 group FCT | 再换 Rank done |
| --- | ---: | ---: | ---: |
| 该组网络尾部 | {reference_tail_ns / 1e6:.6f} ms | {oisa_tail_ns / 1e6:.6f} ms | {oisa_tail_ns / 1e6:.6f} ms |
| ProfilerStep | {baseline_metric['predicted_profiler_step_ms']:.6f} ms | {tail_only_metric['predicted_profiler_step_ms']:.6f} ms | {hybrid_metric['predicted_profiler_step_ms']:.6f} ms |
| 相对基线 Step 增量 | — | {tail_step_delta:+.6f} ms | {step_delta:+.6f} ms |
| 固定 FLOP MFU | {baseline_metric['predicted_mfu_pct']:.6f}% | {tail_only_metric['predicted_mfu_pct']:.6f}% | {hybrid_metric['predicted_mfu_pct']:.6f}% |

目标组在基线关键路径上没有 downstream slack，因此该组网络尾部增加量 {(oisa_tail_ns-reference_tail_ns) / 1e6:.6f} ms 完整传播到 Step。进一步使用 OISA 的逐 Rank network-done 后，Step 又变化 {rank_done_step_delta * 1e3:+.3f} μs；这部分不是额外 group FCT，而是原 Trace Rank completion skew 被替换后的依赖传播。目标组 Rank completion model 为 `{', '.join(target_completion_models)}`。

## 不能据此下的结论

- 当前 OISA 拓扑是 8-GPU 验证拓扑，实测 DP communicator 是 {dp_group_size} Rank，不能直接仿真 DP16；
- 实测 Rank 到本地 Rank 的映射只是接口验证映射，不等价于 256-GPU 目标集群物理放置；
- `n_channels=1`、collective 算法和 payload 口径仍需与训练通信库对齐；
- 因此 {oisa_tail_ns / 1e6:.3f} ms 只能证明真实 OISA 返回可正确进入 MFU DAG，不能作为最终 MFU 校准参数。

拓扑 hash：`{topology_hash}`。
"""


def main() -> None:
    args = parse_args()
    oisa_root = args.oisa_root.resolve()
    oisa_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=oisa_root, text=True
    ).strip()
    if oisa_commit != args.expected_oisa_commit:
        raise ValueError(
            f"OISA commit mismatch: {oisa_commit} != {args.expected_oisa_commit}"
        )
    CollectiveRequest, parse_topology_file, smoke = _load_oisa(oisa_root)
    simulator = oisa_root / "bin/OISA_simulator"
    if not simulator.exists():
        raise ValueError(f"OISA simulator binary is missing: {simulator}")
    topology_path = oisa_root / "topologies/oisa_8gpu_8switch"
    topology = parse_topology_file(topology_path)

    trace = pd.read_csv(CASE_ROOT / "data/pp_dag_trace_events.csv")
    calls = pd.read_csv(CASE_ROOT / "data/optimizer_timeline_calls.csv")
    clocks = pd.read_csv(CASE_ROOT / "data/iteration_clocks.csv")
    api = pd.read_csv(CASE_ROOT / "data/pp_api_trace_events.csv")
    cells = pd.read_csv(CASE_ROOT / "data/pp_framework_sync_cells.csv")
    summary = json.loads(
        (CASE_ROOT / "results/mfu_timeline/mfu_timeline_summary.json").read_text()
    )
    pp_network_ns = int(round(cells["service_reference_fct_ns"].median()))
    calibration, backward_software_ns = _calibration(api, pp_network_ns)
    common = {
        "trace_events": trace,
        "pp_service_ns": pp_network_ns,
        "pp_software_completion_ns": {
            "forward": 0,
            "backward": backward_software_ns,
        },
        "optimizer_calls": calls,
        "clocks": clocks,
        "model_flops_per_iteration": float(summary["model_flops_per_iteration"]),
        "world_size": int(summary["parallel"]["world_size"]),
        "peak_tflops_per_gpu": float(summary["peak_tflops_per_gpu"]),
    }
    reference = TraceReferenceFctProvider()
    baseline = build_unified_mfu_dag(
        optimizer_fct_provider=reference,
        **common,
    )
    target_calls = baseline.optimizer.calls[
        baseline.optimizer.calls["fct_group_id"].eq(args.target_group_id)
    ].copy()
    if target_calls.empty:
        raise ValueError(f"MFU target group not found: {args.target_group_id}")
    mfu_request = make_collective_request(target_calls)
    original_ranks = list(mfu_request.ranks)
    if len(original_ranks) > topology.gpu_count:
        raise ValueError("target collective is larger than the OISA validation topology")
    local_for_original = {
        original_rank: local_rank
        for local_rank, original_rank in enumerate(original_ranks)
    }
    releases = dict(mfu_request.rank_release_offsets_ns)
    rank_mapping = pd.DataFrame(
        [
            {
                "original_rank": rank,
                "oisa_local_rank": local_for_original[rank],
                "release_offset_ns": releases[rank],
            }
            for rank in original_ranks
        ]
    )
    request_base = {
        "iteration": mfu_request.iteration,
        "kind": mfu_request.kind,
        "round": mfu_request.round,
        "group_key": mfu_request.group_key,
        "op": mfu_request.op,
        "group_ranks": list(range(len(original_ranks))),
        "payload_bytes": mfu_request.payload_bytes,
        "topology_id": topology_path.name,
        "n_channels": 1,
    }
    sync_request = CollectiveRequest.from_mapping(
        {
            **request_base,
            "request_id": "mfu_iter55_edp_ag0_pg1279_sync",
            "rank_release_offsets_ns": {
                str(local_rank): 0 for local_rank in range(len(original_ranks))
            },
        }
    )
    staggered_request = CollectiveRequest.from_mapping(
        {
            **request_base,
            "request_id": "mfu_iter55_edp_ag0_pg1279_staggered",
            "rank_release_offsets_ns": {
                str(local_for_original[rank]): releases[rank]
                for rank in original_ranks
            },
        }
    )
    with tempfile.TemporaryDirectory(prefix="mfu-oisa-real-validation-") as temp:
        run_root = Path(temp)
        interface_sync = smoke.run_case(
            run_root,
            simulator,
            smoke.request(
                "interface_a2a_sync",
                [0, 0, 0, 0],
                payload_bytes=8 * 1024 * 1024,
            ),
        )
        interface_staggered = smoke.run_case(
            run_root,
            simulator,
            smoke.request(
                "interface_a2a_staggered",
                [0, 20_000_000, 50_000_000, 30_000_000],
                payload_bytes=8 * 1024 * 1024,
            ),
        )
        sync_result = smoke.run_case(run_root, simulator, sync_request)
        staggered_result = smoke.run_case(run_root, simulator, staggered_request)
        flow_frame = pd.read_csv(
            run_root / staggered_request.request_id / "p2p_flow_fct.csv"
        )

    interface_summary = {
        "synchronized": interface_sync,
        "staggered": interface_staggered,
        "start_times_differ": interface_sync["observed_flow_start_times_ns"]
        != interface_staggered["observed_flow_start_times_ns"],
        "competition_changed": interface_sync["observed_flow_fct_ns"]
        != interface_staggered["observed_flow_fct_ns"],
        "changed_flow_fct_count": sum(
            interface_sync["observed_flow_fct_ns"].get(flow_id)
            != interface_staggered["observed_flow_fct_ns"].get(flow_id)
            for flow_id in set(interface_sync["observed_flow_fct_ns"])
            | set(interface_staggered["observed_flow_fct_ns"])
        ),
    }

    local_done = {
        int(rank): int(offset)
        for rank, offset in staggered_result["rank_network_done_offsets_ns"].items()
    }
    original_done = {
        original_rank: local_done[local_for_original[original_rank]]
        for original_rank in original_ranks
    }
    adapted_response = {
        "request_id": mfu_request.request_id,
        "first_release_ns": int(staggered_result["first_release_ns"]),
        "last_release_ns": int(staggered_result["last_release_ns"]),
        "last_flow_end_ns": int(staggered_result["last_flow_end_ns"]),
        "collective_elapsed_ns": int(staggered_result["collective_elapsed_ns"]),
        "arrival_span_ns": int(staggered_result["arrival_span_ns"]),
        "tail_after_last_release_ns": int(
            staggered_result["tail_after_last_release_ns"]
        ),
        "rank_network_done_offsets_ns": original_done,
        "source": "oisa_ns3_rank_remapped_validation",
        "simulator_commit": staggered_result["simulator_commit"],
        "topology_hash": staggered_result["topology_hash"],
    }
    candidate = RecordedOisaFctProvider([adapted_response])
    tail_only_response = dict(adapted_response)
    tail_only_response.pop("rank_network_done_offsets_ns")
    tail_only_candidate = RecordedOisaFctProvider([tail_only_response])
    tail_only_provider = SelectiveFctProvider(
        args.target_group_id,
        reference,
        tail_only_candidate,
    )
    tail_only = build_unified_mfu_dag(
        optimizer_fct_provider=tail_only_provider,
        **common,
    )
    hybrid_provider = SelectiveFctProvider(
        args.target_group_id,
        reference,
        candidate,
    )
    hybrid = build_unified_mfu_dag(
        optimizer_fct_provider=hybrid_provider,
        **common,
    )
    baseline_metric = baseline.iteration.iloc[0]
    tail_only_metric = tail_only.iteration.iloc[0]
    hybrid_metric = hybrid.iteration.iloc[0]
    baseline_group = baseline.optimizer.groups[
        baseline.optimizer.groups["fct_group_id"].eq(args.target_group_id)
    ].iloc[0]
    hybrid_group = hybrid.optimizer.groups[
        hybrid.optimizer.groups["fct_group_id"].eq(args.target_group_id)
    ].iloc[0]
    hybrid_target_calls = hybrid.optimizer.calls[
        hybrid.optimizer.calls["fct_group_id"].eq(args.target_group_id)
    ]
    completion_models = sorted(set(hybrid_target_calls["completion_model"]))
    reference_tail_ns = int(baseline_group["predicted_service_ns"])
    oisa_tail_ns = int(hybrid_group["predicted_service_ns"])
    tail_delta_ns = oisa_tail_ns - reference_tail_ns
    tail_only_step_delta_ns = int(
        round(
            (
                float(tail_only_metric["predicted_profiler_step_ms"])
                - float(baseline_metric["predicted_profiler_step_ms"])
            )
            * 1e6
        )
    )
    rank_done_step_delta_ns = int(
        round(
            (
                float(hybrid_metric["predicted_profiler_step_ms"])
                - float(tail_only_metric["predicted_profiler_step_ms"])
            )
            * 1e6
        )
    )
    step_delta_ns = tail_only_step_delta_ns + rank_done_step_delta_ns
    dp_group_size = int(
        baseline.optimizer.groups.loc[
            baseline.optimizer.groups["kind"].eq("dp_rs"), "group_size"
        ].max()
    )

    output = args.output_dir
    _atomic_csv(rank_mapping, output / "rank_mapping.csv")
    _atomic_csv(flow_frame, output / "staggered_p2p_flow_fct.csv")
    _atomic_csv(calibration, output / "pp_software_completion_calibration.csv")
    _atomic_csv(hybrid.optimizer.groups, output / "hybrid_optimizer_groups.csv")
    _atomic_json(sync_request.to_dict(), output / "oisa_sync_request.json")
    _atomic_json(staggered_request.to_dict(), output / "oisa_staggered_request.json")
    _atomic_json(sync_result, output / "oisa_sync_result.json")
    _atomic_json(staggered_result, output / "oisa_staggered_result.json")
    _atomic_json(adapted_response, output / "adapted_mfu_response.json")
    _atomic_json(interface_summary, output / "interface_a2a_comparison.json")
    comparison = pd.DataFrame(
        [
            {
                "scenario": "trace_reference",
                "target_tail_ns": reference_tail_ns,
                **baseline_metric.to_dict(),
            },
            {
                "scenario": "single_group_oisa_tail_only",
                "target_tail_ns": oisa_tail_ns,
                **tail_only_metric.to_dict(),
            },
            {
                "scenario": "single_group_real_oisa",
                "target_tail_ns": oisa_tail_ns,
                **hybrid_metric.to_dict(),
            },
        ]
    )
    _atomic_csv(comparison, output / "iteration_comparison.csv")

    identity_valid = bool(
        int(staggered_result["collective_elapsed_ns"])
        == int(staggered_result["arrival_span_ns"])
        + int(staggered_result["tail_after_last_release_ns"])
    )
    validation = {
        "schema": "mfu-real-oisa-single-group-validation-v1",
        "status": "PASS",
        "oisa_commit": oisa_commit,
        "topology_hash": staggered_result["topology_hash"],
        "topology_gpu_count": topology.gpu_count,
        "target_group_id": args.target_group_id,
        "target_original_ranks": original_ranks,
        "rank_remapping_is_validation_only": True,
        "target_payload_bytes": mfu_request.payload_bytes,
        "checks": {
            "oisa_commit_matches": oisa_commit == args.expected_oisa_commit,
            "collective_elapsed_identity": identity_valid,
            "sync_dependency_gate_valid": bool(sync_result["dependency_gate_valid"]),
            "staggered_dependency_gate_valid": bool(
                staggered_result["dependency_gate_valid"]
            ),
            "interface_start_times_differ": bool(
                interface_summary["start_times_differ"]
            ),
            "interface_competition_changed": bool(
                interface_summary["competition_changed"]
            ),
            "all_interface_flow_fcts_changed": int(
                interface_summary["changed_flow_fct_count"]
            )
            == 12,
            "provider_used_oisa_rank_done": completion_models
            == ["oisa_rank_network_done"],
            "critical_group_tail_delta_reaches_step": tail_only_step_delta_ns
            == tail_delta_ns,
            "rank_done_effect_is_separately_conserved": step_delta_ns
            == tail_delta_ns + rank_done_step_delta_ns,
            "dp16_exceeds_current_topology": dp_group_size > topology.gpu_count,
        },
        "metrics": {
            "trace_reference_tail_ns": reference_tail_ns,
            "oisa_tail_after_last_release_ns": oisa_tail_ns,
            "tail_delta_ns": tail_delta_ns,
            "tail_only_profiler_step_delta_ns": tail_only_step_delta_ns,
            "rank_done_profiler_step_delta_ns": rank_done_step_delta_ns,
            "profiler_step_delta_ns": step_delta_ns,
            "baseline_profiler_step_ms": float(
                baseline_metric["predicted_profiler_step_ms"]
            ),
            "hybrid_profiler_step_ms": float(
                hybrid_metric["predicted_profiler_step_ms"]
            ),
            "baseline_mfu_pct": float(baseline_metric["predicted_mfu_pct"]),
            "hybrid_mfu_pct": float(hybrid_metric["predicted_mfu_pct"]),
        },
        "limitations": {
            "current_oisa_topology_ranks": topology.gpu_count,
            "measured_dp_group_size": dp_group_size,
            "n_channels": 1,
            "performance_calibration_valid": False,
        },
    }
    if not all(validation["checks"].values()):
        validation["status"] = "FAIL"
    _atomic_json(validation, output / "validation.json")
    _atomic_text(
        _report(
            target_group_id=args.target_group_id,
            original_ranks=original_ranks,
            rank_mapping=rank_mapping,
            oisa_commit=oisa_commit,
            topology_hash=str(staggered_result["topology_hash"]),
            topology_gpu_count=topology.gpu_count,
            payload_bytes=int(mfu_request.payload_bytes or 0),
            interface_summary=interface_summary,
            sync=sync_result,
            staggered=staggered_result,
            baseline_metric=baseline_metric,
            tail_only_metric=tail_only_metric,
            hybrid_metric=hybrid_metric,
            reference_tail_ns=reference_tail_ns,
            oisa_tail_ns=oisa_tail_ns,
            target_completion_models=completion_models,
            dp_group_size=dp_group_size,
        ),
        output / "MFU_OISA_REAL_VALIDATION.md",
    )
    _atomic_text(
        _html(
            hybrid,
            hybrid_metric,
            page_title="PP + RS/AG DAG · 单组真实 OISA FCT 验证",
            focus_critical_lane=True,
        ),
        output / "pp_optimizer_dag_oisa_real_single_group.html",
    )
    print(
        f"real OISA {args.target_group_id}: tail {reference_tail_ns / 1e6:.3f} "
        f"-> {oisa_tail_ns / 1e6:.3f} ms, Step {step_delta_ns / 1e6:+.3f} ms, "
        f"status={validation['status']} -> {output}"
    )


if __name__ == "__main__":
    main()

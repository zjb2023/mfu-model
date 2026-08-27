#!/usr/bin/env python3
"""Build the separate v1 DAG spanning PP compute through optimizer RS/AG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x10000_analysis.mfu_timeline import TIMELINE_KINDS  # noqa: E402
from x10000_analysis.unified_mfu_dag import build_unified_mfu_dag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join the trace-driven PP DAG to dynamic DP/EDP RS/AG rendezvous."
    )
    parser.add_argument(
        "--trace-events",
        type=Path,
        default=CASE_ROOT / "data/pp_dag_trace_events.csv",
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
    parser.add_argument("--pp-service-ns", type=float)
    parser.add_argument(
        "--optimizer-summary",
        type=Path,
        default=CASE_ROOT / "results/mfu_timeline/mfu_timeline_summary.json",
    )
    parser.add_argument(
        "--optimizer-service-scale",
        action="append",
        default=[],
        metavar="KIND=VALUE",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_ROOT / "results/pp_optimizer_dag_v1",
    )
    return parser.parse_args()


def _scales(items: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        try:
            kind, raw = item.split("=", 1)
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid optimizer service scale {item!r}") from exc
        if kind not in TIMELINE_KINDS:
            raise ValueError(f"unknown optimizer kind: {kind}")
        if not np.isfinite(value) or value < 0:
            raise ValueError("optimizer service scales must be finite and non-negative")
        result[kind] = value
    return result


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _atomic_json(document: dict[str, object], path: Path) -> None:
    _atomic_text(json.dumps(document, indent=2, sort_keys=True) + "\n", path)


def _report(
    metrics: pd.Series,
    pp_service_ns: int,
    result: object,
    pp_zero_step_ms: float,
) -> str:
    calls = result.optimizer.calls
    groups = result.optimizer.groups
    pp_marginal = float(metrics["predicted_profiler_step_ms"]) - pp_zero_step_ms
    return f"""# PP + Optimizer RS/AG 合并 DAG v1

> `results/pp_dag_minimal/` 的 v0 页面、CSV 和说明全部保留。本目录是独立的合并版本。

## 合并结果

iteration {int(metrics['iteration'])} 已从 FWD/BWD、双向 PP message 一直连接到 DP/Expert-DP 的 RS 与两轮 AG。首个 DP-RS 不再使用固定实测开始时间，而由每个 rank 的 `预测B3完成 + 实测软件尾部lag` 产生；后续 collective 使用 `max(组内rank到达) + service` 重新计算。

| 项目 | 结果 |
| --- | ---: |
| PP 计算节点 | {int(result.pipeline.nodes['kind'].eq('compute').sum()):,} |
| PP message 节点 | {int(result.pipeline.nodes['kind'].eq('pp_message').sum()):,} |
| optimizer rank collective 调用 | {len(calls):,} |
| optimizer collective group | {len(groups):,} |
| 合并依赖边 | {len(result.dependency_edges):,} |
| 单条 PP service | {pp_service_ns / 1e6:.6f} ms |
| PP service 对最终 Step 的边际 | {pp_marginal:.3f} ms |
| 实测 ProfilerStep | {metrics['observed_profiler_step_ms']:.3f} ms |
| 合并 DAG ProfilerStep | {metrics['predicted_profiler_step_ms']:.3f} ms |
| ProfilerStep 回放误差 | {metrics['profiler_replay_error_ms']:+.3f} ms / {metrics['profiler_replay_error_pct']:+.3f}% |
| 实测 training-log Step | {metrics['observed_training_log_ms']:.3f} ms |
| 合并 DAG training-log Step | {metrics['predicted_training_log_ms']:.3f} ms |
| 固定 FLOPs 实测 MFU | {metrics['actual_mfu_pct_fixed_flops']:.4f}% |
| 合并 DAG MFU | {metrics['predicted_mfu_pct']:.4f}% |
| DP/EDP service 对最终 Step 的边际 | {metrics['all_dp_service_exposed_ms']:.3f} ms |

## 完整依赖

```text
FWD(s,m) → PP forward message → FWD(s+1,m)
                                      ↓
BWD(s,m) ← PP backward message ← BWD(s+1,m)
    ↓
rank B3 + measured software-tail lag
    ↓
DP-RS → Expert-DP RS → world RS done
    ↓
DP-AG0 → Expert-DP AG0 → DP-AG1 → Expert-DP AG1
    ↓
post-AG residual → ProfilerStep → outer residual → training-log Step → MFU
```

## 当前解释

本次已经完成“传播关系”的合并：修改 PP FCT 会改变各 rank 的 B3、RS 到达波次、RS/AG 完成时间、最终 Step 和 MFU；修改 RS/AG service 也会沿同一条链传播。

当前 {metrics['profiler_replay_error_pct']:+.3f}% 的端到端回放误差主要继承自 v0 PP 前段少建模的软件/P2P API 等待，而不是 RS/AG 未连接。下一步校准目标仍是把 `recv/send/send_recv` 的 post 与完成节点加入 PP 图，不能通过放大纯链路 FCT 隐藏该残差。

详细分解见 `ERROR_ANALYSIS.md`：B0 backward 波逐 stage 少算约 93.6 ms/hop，与 `send_backward` P50 98.1 ms 减去 4.7 ms service 基本一致。
"""


def _html(
    result: object,
    metrics: pd.Series,
    page_title: str = "PP + Optimizer RS/AG 合并 DAG v1",
) -> str:
    timeline = result.combined_timeline[
        result.combined_timeline["pp_lane"].ge(0)
    ].copy()
    columns = [
        "node_id",
        "source_model",
        "kind",
        "category",
        "rank",
        "pp_stage",
        "pp_lane",
        "microbatch",
        "predicted_start_ns",
        "predicted_end_ns",
        "duration_ns",
        "network_service_ns",
        "software_completion_ns",
        "group_key",
        "dependency",
    ]
    payload = json.dumps(
        {
            "iteration": int(metrics["iteration"]),
            "stepStart": int(metrics["step_start_ns"]),
            "predictedProfilerMs": float(metrics["predicted_profiler_step_ms"]),
            "observedProfilerMs": float(metrics["observed_profiler_step_ms"]),
            "predictedMfu": float(metrics["predicted_mfu_pct"]),
            "actualMfu": float(metrics["actual_mfu_pct_fixed_flops"]),
            "events": timeline[columns].to_dict(orient="records"),
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title}</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f5f7fb;color:#172033}}main{{padding:22px;max-width:1540px;margin:auto}}h1{{margin:0 0 8px}}.note{{color:#536179;margin-bottom:14px}}
.toolbar{{display:flex;gap:18px;align-items:center;margin:12px 0;flex-wrap:wrap}}.metric{{background:#fff;border:1px solid #dce2ed;border-radius:8px;padding:7px 10px}}
.card{{background:#fff;border:1px solid #dce2ed;border-radius:12px;padding:14px;box-shadow:0 2px 8px #14213d0d}}#chart{{position:relative;height:736px;margin-left:58px;border-left:1px solid #aab4c5;border-bottom:1px solid #aab4c5;overflow:hidden}}
.row{{position:absolute;left:0;right:0;height:42px;border-top:1px solid #eef1f6}}.label{{position:absolute;right:calc(100% + 8px);width:46px;text-align:right;font-size:12px;color:#5c677d}}
.node{{position:absolute;height:24px;top:8px;border-radius:4px;min-width:2px;opacity:.92}}.compute.forward{{background:#2f80ed}}.compute.backward{{background:#9b51e0}}.pp_message{{height:5px;top:35px;border-radius:0}}.pp_message.forward{{background:#f2994a}}.pp_message.backward{{background:#eb5757}}
.software_tail{{height:8px;top:29px;background:#7f8c8d}}.optimizer_collective.dp_rs{{background:#d63031}}.optimizer_collective.edp_rs{{background:#e17055}}.optimizer_collective.dp_ag0{{background:#00a86b}}.optimizer_collective.edp_ag0{{background:#00b894}}.optimizer_collective.dp_ag1{{background:#0984a3}}.optimizer_collective.edp_ag1{{background:#00cec9}}
.axis{{display:flex;justify-content:space-between;margin-left:58px;color:#69758a;font-size:12px}}.legend span{{display:inline-flex;align-items:center;margin-right:10px;font-size:12px}}.sw{{width:12px;height:8px;margin-right:4px;border-radius:2px}}#tip{{position:fixed;display:none;background:#101726;color:white;padding:8px 10px;border-radius:7px;font-size:12px;pointer-events:none;z-index:5;white-space:pre}}
</style></head><body><main><h1>{page_title}</h1><div class="note">iteration {int(metrics['iteration'])}；选择 PP lane 后，每行对应一个 stage。RS/AG 已接在该 rank 的 B3 之后。</div>
<div class="toolbar"><label>PP lane <select id="lane"></select></label><span class="metric" id="step"></span><span class="metric" id="mfu"></span></div>
<div class="card"><div class="legend"><span><i class="sw" style="background:#2f80ed"></i>FWD</span><span><i class="sw" style="background:#9b51e0"></i>BWD</span><span><i class="sw" style="background:#f2994a"></i>PP msg</span><span><i class="sw" style="background:#7f8c8d"></i>tail lag</span><span><i class="sw" style="background:#d63031"></i>DP-RS</span><span><i class="sw" style="background:#e17055"></i>EDP-RS</span><span><i class="sw" style="background:#00a86b"></i>AG0</span><span><i class="sw" style="background:#0984a3"></i>AG1</span></div><div id="chart"></div><div class="axis"><span>ProfilerStep 0</span><span id="axisEnd"></span></div></div><div id="tip"></div>
</main><script>const DATA={payload};const sel=document.querySelector('#lane'),chart=document.querySelector('#chart'),tip=document.querySelector('#tip');for(let i=0;i<16;i++){{const o=document.createElement('option');o.value=i;o.textContent=i;sel.appendChild(o)}}
document.querySelector('#step').textContent='ProfilerStep: DAG '+DATA.predictedProfilerMs.toFixed(1)+' ms / Trace '+DATA.observedProfilerMs.toFixed(1)+' ms';document.querySelector('#mfu').textContent='MFU: DAG '+DATA.predictedMfu.toFixed(4)+'% / Trace '+DATA.actualMfu.toFixed(4)+'%';
function render(){{const lane=+sel.value,end=DATA.predictedProfilerMs*1e6,events=DATA.events.filter(e=>e.pp_lane===lane);chart.innerHTML='';for(let s=15;s>=0;s--){{const row=document.createElement('div');row.className='row';row.style.top=((15-s)*45)+'px';const lab=document.createElement('span');lab.className='label';lab.textContent='PP'+s;row.appendChild(lab);chart.appendChild(row);events.filter(e=>e.pp_stage===s).forEach(e=>{{const start=e.predicted_start_ns-DATA.stepStart,finish=e.predicted_end_ns-DATA.stepStart;if(finish<0||start>end)return;const el=document.createElement('div');el.className='node '+e.kind+' '+e.category;el.style.left=(100*Math.max(0,start)/end)+'%';el.style.width=Math.max(.12,100*Math.max(0,finish-start)/end)+'%';el.onmousemove=x=>{{tip.style.display='block';tip.style.left=(x.clientX+12)+'px';tip.style.top=(x.clientY+12)+'px';const split=e.kind==='pp_message'?'\\nnetwork '+(e.network_service_ns/1e6).toFixed(3)+' ms + software '+(e.software_completion_ns/1e6).toFixed(3)+' ms':'';tip.textContent=e.node_id+'\\n'+e.category+'  PP'+e.pp_stage+' rank '+e.rank+'\\n'+(start/1e6).toFixed(3)+' → '+(finish/1e6).toFixed(3)+' ms'+split+'\\ndep: '+e.dependency;}};el.onmouseleave=()=>tip.style.display='none';row.appendChild(el)}})}}document.querySelector('#axisEnd').textContent=DATA.predictedProfilerMs.toFixed(1)+' ms';}}sel.onchange=render;render();</script></body></html>"""


def main() -> None:
    args = parse_args()
    trace_events = pd.read_csv(args.trace_events)
    optimizer_calls = pd.read_csv(args.optimizer_calls)
    clocks = pd.read_csv(args.iteration_clocks)
    optimizer_summary = json.loads(args.optimizer_summary.read_text())
    if args.pp_service_ns is None:
        cells = pd.read_csv(args.pp_service_cells)
        pp_service_ns = int(round(cells["service_reference_fct_ns"].median()))
    else:
        pp_service_ns = int(round(args.pp_service_ns))
    scales = _scales(args.optimizer_service_scale)
    common = {
        "trace_events": trace_events,
        "optimizer_calls": optimizer_calls,
        "clocks": clocks,
        "model_flops_per_iteration": float(
            optimizer_summary["model_flops_per_iteration"]
        ),
        "world_size": int(optimizer_summary["parallel"]["world_size"]),
        "peak_tflops_per_gpu": float(optimizer_summary["peak_tflops_per_gpu"]),
        "optimizer_service_scales": scales or None,
    }
    result = build_unified_mfu_dag(pp_service_ns=pp_service_ns, **common)
    no_pp = build_unified_mfu_dag(pp_service_ns=0, **common)
    metrics = result.iteration.iloc[0]
    pp_zero_step_ms = float(no_pp.iteration.iloc[0]["predicted_profiler_step_ms"])
    output = args.output_dir
    _atomic_csv(result.pp_nodes_absolute, output / "pp_nodes_absolute.csv")
    _atomic_csv(result.pipeline.edges, output / "pp_edges.csv")
    _atomic_csv(result.front_anchors, output / "front_anchors.csv")
    _atomic_csv(result.optimizer.calls, output / "optimizer_calls.csv")
    _atomic_csv(result.optimizer.groups, output / "optimizer_groups.csv")
    _atomic_csv(result.combined_timeline, output / "combined_timeline.csv")
    _atomic_csv(result.dependency_edges, output / "dependency_edges.csv")
    _atomic_csv(result.iteration, output / "iteration_summary.csv")

    known_nodes = set(result.combined_timeline["node_id"])
    unknown_edges = set(result.dependency_edges["src"]).union(
        result.dependency_edges["dst"]
    ) - known_nodes
    validation = {
        "schema": "pp-optimizer-dag-v1",
        "status": "PASS" if not unknown_edges else "FAIL",
        "iteration": int(metrics["iteration"]),
        "v0_preserved_at": "case_256gpu_pp16_cp2_a2a/results/pp_dag_minimal",
        "pp_service_ns": pp_service_ns,
        "optimizer_service_scales": {
            kind: float(scales.get(kind, 1.0)) for kind in TIMELINE_KINDS
        },
        "counts": {
            "combined_timeline_nodes": len(result.combined_timeline),
            "dependency_edges": len(result.dependency_edges),
            "front_anchor_ranks": len(result.front_anchors),
            "optimizer_rank_calls": len(result.optimizer.calls),
            "optimizer_groups": len(result.optimizer.groups),
        },
        "validation": {
            "all_dense_rs_use_pp_frontier": bool(
                result.optimizer.calls.loc[
                    result.optimizer.calls["kind"].eq("dp_rs"), "dependency"
                ].eq("pp_frontier").all()
            ),
            "all_dependency_edge_nodes_exist": not unknown_edges,
            "unknown_dependency_nodes": sorted(unknown_edges),
            "pp_fct_changes_final_step": bool(
                float(metrics["predicted_profiler_step_ms"]) > pp_zero_step_ms
            ),
            "all_rs_finish_before_ag0": bool(
                metrics["first_ag0_offset_ms"] >= metrics["gradient_rs_done_offset_ms"]
            ),
        },
        "metrics": {
            key: float(metrics[key])
            for key in [
                "observed_profiler_step_ms",
                "predicted_profiler_step_ms",
                "profiler_replay_error_ms",
                "profiler_replay_error_pct",
                "observed_training_log_ms",
                "predicted_training_log_ms",
                "actual_mfu_pct_fixed_flops",
                "predicted_mfu_pct",
                "all_dp_service_exposed_ms",
            ]
        },
        "pp_service_final_step_marginal_ms": float(
            metrics["predicted_profiler_step_ms"] - pp_zero_step_ms
        ),
    }
    _atomic_json(validation, output / "validation.json")
    _atomic_text(
        _report(metrics, pp_service_ns, result, pp_zero_step_ms),
        output / "PP_OPTIMIZER_DAG_V1.md",
    )
    _atomic_text(
        _html(result, metrics), output / "pp_optimizer_dag_v1.html"
    )
    print(
        f"built unified PP+RS/AG DAG for iteration {int(metrics['iteration'])}: "
        f"{len(result.combined_timeline)} timeline nodes, "
        f"{len(result.dependency_edges)} dependency edges -> {output}"
    )


if __name__ == "__main__":
    main()

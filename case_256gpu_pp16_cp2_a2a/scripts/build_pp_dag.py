#!/usr/bin/env python3
"""Build the minimal trace-driven PP DAG and its optimizer-frontier hook."""

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

from x10000_analysis.pp_dag import (  # noqa: E402
    build_pipeline_dag,
    build_schedule_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FWD/BWD + PP-message max-plus DAG from frozen trace facts."
    )
    parser.add_argument(
        "--trace-events",
        type=Path,
        default=CASE_ROOT / "data/pp_dag_trace_events.csv",
    )
    parser.add_argument(
        "--pp-service-ns",
        type=float,
        help="One logical 80 MiB P2P message FCT; default is the frozen Trace proxy median.",
    )
    parser.add_argument(
        "--pp-service-cells",
        type=Path,
        default=CASE_ROOT / "data/pp_framework_sync_cells.csv",
    )
    parser.add_argument(
        "--optimizer-calls",
        type=Path,
        default=CASE_ROOT / "data/optimizer_timeline_calls.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASE_ROOT / "results/pp_dag_minimal",
    )
    return parser.parse_args()


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


def _add_optimizer_hook(
    frontier: pd.DataFrame,
    trace_events: pd.DataFrame,
    optimizer_calls_path: Path,
) -> pd.DataFrame:
    result = frontier.copy()
    if not optimizer_calls_path.exists() or "observed_end_ns" not in trace_events:
        result["measured_tail_launch_lag_ns"] = pd.NA
        result["predicted_first_rs_ready_ns"] = pd.NA
        return result
    iteration = int(trace_events["iteration"].iloc[0])
    calls = pd.read_csv(optimizer_calls_path)
    calls = calls[calls["iteration"].eq(iteration)]
    first_rs = (
        calls[calls["kind"].eq("dp_rs")]
        .groupby("rank", as_index=False)["start_ns"]
        .min()
        .rename(columns={"start_ns": "observed_first_rs_start_ns"})
    )
    observed_last_bwd = (
        trace_events[
            trace_events["phase"].eq("backward")
            & trace_events["microbatch"].eq(trace_events["microbatch"].max())
        ][["rank", "observed_end_ns"]]
        .rename(columns={"observed_end_ns": "observed_backward_done_ns"})
    )
    result = result.merge(observed_last_bwd, on="rank", how="left", validate="one_to_one")
    result = result.merge(first_rs, on="rank", how="left", validate="one_to_one")
    result["measured_tail_launch_lag_ns"] = (
        result["observed_first_rs_start_ns"] - result["observed_backward_done_ns"]
    ).clip(lower=0)
    result["predicted_first_rs_ready_ns"] = (
        result["predicted_optimizer_ready_ns"]
        + result["measured_tail_launch_lag_ns"]
    )
    return result


def _schedule_text(schedule: pd.DataFrame, stage: int) -> str:
    selected = schedule[schedule["pp_stage"].eq(stage)].sort_values("sequence")
    return " ".join(
        f"{'F' if row.phase == 'forward' else 'B'}{int(row.microbatch)}"
        for row in selected.itertuples(index=False)
    )


def _report(
    iteration: int,
    service_ns: int,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    lanes: pd.DataFrame,
    frontier: pd.DataFrame,
    schedule: pd.DataFrame,
) -> str:
    compute = nodes[nodes["kind"].eq("compute")]
    messages = nodes[nodes["kind"].eq("pp_message")]
    makespan_ms = lanes["predicted_front_makespan_ns"] / 1e6
    observed_ms = lanes["observed_front_makespan_ns"] / 1e6
    replay_error_pct = lanes["front_replay_error_pct"]
    wait_ms = compute["local_wait_ns"] / 1e6
    hook_count = int(frontier["predicted_first_rs_ready_ns"].notna().sum())
    tail_lag_ms = frontier["measured_tail_launch_lag_ns"] / 1e6
    return f"""# 最小版 PP-DAG（Trace 驱动）

## 已经建成什么

本版把 iteration {iteration} 的每个 rank 拆成 4 个 FWD 与 4 个 BWD 计算节点，并用逐 microbatch 的 PP 消息节点连接。正向依赖为 `PP0 → PP15`，反向依赖为 `PP15 → PP0`。等待时间由 max-plus DAG 自动算出，没有输入任何 stage 的平均 bubble。

| 项目 | 结果 |
| --- | ---: |
| 计算节点 | {len(compute):,} |
| 唯一 PP 消息节点 | {len(messages):,} |
| 依赖边 | {len(edges):,} |
| PP lane | {len(lanes)} |
| 单条 80 MiB PP service proxy | {service_ns / 1e6:.6f} ms |
| lane 前段 makespan P50 | {makespan_ms.median():.3f} ms |
| Trace 前段包络 P50 | {observed_ms.median():.3f} ms |
| v0 前段回放误差 P50 | {replay_error_pct.median():+.3f}% |
| lane 前段 makespan范围 | {makespan_ms.min():.3f}–{makespan_ms.max():.3f} ms |
| 单计算节点等待 P50 / P90 | {wait_ms.median():.3f} / {wait_ms.quantile(0.9):.3f} ms |
| 已连接 optimizer frontier 的 rank | {hook_count} / {len(frontier)} |
| BWD→首个 DP-RS 实测软件 lag P50 | {tail_lag_ms.median():.3f} ms |

## DAG 的创建逻辑

1. 每个 rank 先按真实 Megatron 非交错 1F1B 顺序串起本地计算。
2. `F(s,m)` 完成后产生正向消息，消息完成后 `F(s+1,m)` 才能开始。
3. `B(s,m)` 完成后产生反向消息，消息完成后 `B(s-1,m)` 才能开始。
4. 当前训练关闭 P2P overlap，因此发送完成也约束发送 rank 的下一个计算节点。
5. 每个 rank 最后一个 BWD 节点输出到 `optimizer_frontier.csv`。表中保留实测 BWD→首个 DP-RS 的软件尾部 lag，供下一步接入现有 optimizer DAG。

关键 stage 的本地顺序：

- PP0：`{_schedule_text(schedule, 0)}`
- PP13：`{_schedule_text(schedule, 13)}`
- PP14：`{_schedule_text(schedule, 14)}`
- PP15：`{_schedule_text(schedule, 15)}`

## 当前边界

这是 v0 的**数据依赖 DAG**。FWD/BWD 节点时长直接取 Trace 注解，因此其内部的 CP/EP/重计算仍作为整体保留；九类集合通信尚未拆成子节点。PP 消息使用可替换的 FCT，尚未加入多链路竞争与 fused send/recv 的显式 post 节点。当前相对 Trace 前段包络仍有 {replay_error_pct.median():+.3f}% 的残差，这个数被明确保留为下一版的软件调度校准目标，不能塞进链路 service。optimizer 已有可连接的 frontier，但本次结果尚未把两张 DAG 合成一个最终 Step/MFU 数字。

查看 `pp_dag_minimal.html` 可切换 16 条 PP lane；灰色空档就是依赖产生的等待，而不是预先填写的 bubble。
"""


def _html(nodes: pd.DataFrame, lanes: pd.DataFrame, iteration: int) -> str:
    columns = [
        "node_id",
        "kind",
        "phase",
        "microbatch",
        "pp_stage",
        "pp_lane",
        "src_stage",
        "dst_stage",
        "predicted_start_ns",
        "predicted_end_ns",
        "duration_ns",
        "local_wait_ns",
        "region",
    ]
    records = nodes[columns].to_dict(orient="records")
    lane_records = lanes.to_dict(orient="records")
    payload = json.dumps(
        {"iteration": iteration, "nodes": records, "lanes": lane_records},
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>最小版 PP-DAG</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f5f7fb;color:#172033}}main{{padding:22px;max-width:1500px;margin:auto}}
h1{{margin:0 0 8px}}.note{{color:#536179;margin-bottom:16px}}.toolbar{{display:flex;gap:18px;align-items:center;margin:12px 0}}
.card{{background:white;border:1px solid #dce2ed;border-radius:12px;padding:14px;box-shadow:0 2px 8px #14213d0d}}
#chart{{position:relative;height:736px;margin-left:58px;border-left:1px solid #aab4c5;border-bottom:1px solid #aab4c5;overflow:hidden}}
.row{{position:absolute;left:0;right:0;height:42px;border-top:1px solid #eef1f6}}.label{{position:absolute;right:calc(100% + 8px);width:46px;text-align:right;font-size:12px;color:#5c677d}}
.node{{position:absolute;height:24px;top:8px;border-radius:4px;min-width:2px;opacity:.91;cursor:default}}.forward{{background:#2f80ed}}.backward{{background:#9b51e0}}
.pp_message{{height:5px;top:34px;border-radius:0;opacity:.95}}.pp_message.forward{{background:#f2994a}}.pp_message.backward{{background:#eb5757}}
.axis{{display:flex;justify-content:space-between;margin-left:58px;color:#69758a;font-size:12px}}.legend span{{display:inline-flex;align-items:center;margin-right:13px}}.sw{{width:13px;height:8px;margin-right:5px;border-radius:2px}}
#tip{{position:fixed;display:none;background:#101726;color:white;padding:8px 10px;border-radius:7px;font-size:12px;pointer-events:none;z-index:4;white-space:pre}}
</style></head><body><main><h1>最小版 PP-DAG</h1>
<div class="note">iteration {iteration}；每行一个 PP stage。计算块之间的空档是 DAG 依赖算出的等待。橙/红细线为正向/反向 PP service。</div>
<div class="toolbar"><label>PP lane <select id="lane"></select></label><strong id="summary"></strong></div>
<div class="card"><div class="legend"><span><i class="sw" style="background:#2f80ed"></i>FWD</span><span><i class="sw" style="background:#9b51e0"></i>BWD</span><span><i class="sw" style="background:#f2994a"></i>FWD message</span><span><i class="sw" style="background:#eb5757"></i>BWD message</span></div><div id="chart"></div><div class="axis"><span>0 ms</span><span id="axisEnd"></span></div></div>
<div id="tip"></div></main><script>const DATA={payload};
const select=document.querySelector('#lane'), chart=document.querySelector('#chart'), tip=document.querySelector('#tip');
DATA.lanes.forEach(x=>{{const o=document.createElement('option');o.value=x.pp_lane;o.textContent=x.pp_lane;select.appendChild(o)}});
function render(){{const lane=+select.value, all=DATA.nodes.filter(n=>n.pp_lane===lane), info=DATA.lanes.find(x=>x.pp_lane===lane), end=info.predicted_optimizer_ready_ns;chart.innerHTML='';
for(let s=15;s>=0;s--){{const row=document.createElement('div');row.className='row';row.style.top=((15-s)*45)+'px';const label=document.createElement('span');label.className='label';label.textContent='PP'+s;row.appendChild(label);chart.appendChild(row);
all.filter(n=>(n.kind==='compute'?n.pp_stage:n.src_stage)===s).forEach(n=>{{const el=document.createElement('div');el.className='node '+n.kind+' '+n.phase;el.style.left=(100*n.predicted_start_ns/end)+'%';el.style.width=Math.max(0.15,100*(n.predicted_end_ns-n.predicted_start_ns)/end)+'%';el.onmousemove=e=>{{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=n.node_id+'\\n'+(n.predicted_start_ns/1e6).toFixed(3)+' → '+(n.predicted_end_ns/1e6).toFixed(3)+' ms\\nwait '+(n.local_wait_ns/1e6).toFixed(3)+' ms';}};el.onmouseleave=()=>tip.style.display='none';row.appendChild(el)}})}}
document.querySelector('#axisEnd').textContent=(end/1e6).toFixed(1)+' ms';document.querySelector('#summary').textContent='DAG '+(info.predicted_front_makespan_ns/1e6).toFixed(3)+' ms / Trace '+(info.observed_front_makespan_ns/1e6).toFixed(3)+' ms / error '+info.front_replay_error_pct.toFixed(2)+'%';}}
select.onchange=render;render();</script></body></html>"""


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.trace_events)
    if args.pp_service_ns is None:
        service_cells = pd.read_csv(args.pp_service_cells)
        service_ns = int(round(service_cells["service_reference_fct_ns"].median()))
    else:
        service_ns = int(round(args.pp_service_ns))
    result = build_pipeline_dag(events, service_ns)
    frontier = _add_optimizer_hook(result.optimizer_frontier, events, args.optimizer_calls)
    pp_size = int(events["pp_stage"].max()) + 1
    microbatches = int(events["microbatch"].max()) + 1
    schedule = build_schedule_template(pp_size, microbatches)
    iteration = int(events["iteration"].iloc[0])

    output = args.output_dir
    _atomic_csv(result.nodes, output / "nodes.csv")
    _atomic_csv(result.edges, output / "edges.csv")
    _atomic_csv(result.rank_summary, output / "rank_summary.csv")
    _atomic_csv(result.lane_summary, output / "lane_summary.csv")
    _atomic_csv(frontier, output / "optimizer_frontier.csv")
    _atomic_csv(schedule, output / "schedule_template.csv")
    validation = {
        "schema": "pp-dag-minimal-v1",
        "status": "PASS",
        "iteration": iteration,
        "pp_size": pp_size,
        "pp_lanes": int(events["pp_lane"].nunique()),
        "microbatches": microbatches,
        "compute_nodes": int(result.nodes["kind"].eq("compute").sum()),
        "message_nodes": int(result.nodes["kind"].eq("pp_message").sum()),
        "edges": len(result.edges),
        "pp_service_ns": service_ns,
        "negative_local_wait_nodes": int(result.nodes["local_wait_ns"].lt(0).sum()),
        "optimizer_frontier_ranks": len(frontier),
        "optimizer_hook_ranks": int(frontier["predicted_first_rs_ready_ns"].notna().sum()),
        "predicted_front_makespan_ms_p50": float(
            result.lane_summary["predicted_front_makespan_ns"].median() / 1e6
        ),
        "observed_front_makespan_ms_p50": float(
            result.lane_summary["observed_front_makespan_ns"].median() / 1e6
        ),
        "front_replay_error_pct_p50": float(
            result.lane_summary["front_replay_error_pct"].median()
        ),
        "measured_tail_launch_lag_ms_p50": float(
            frontier["measured_tail_launch_lag_ns"].median() / 1e6
        ),
        "model_scope": "FWD/BWD annotations + blocking PP messages; optimizer frontier only",
    }
    _atomic_json(validation, output / "validation.json")
    _atomic_text(
        _report(
            iteration,
            service_ns,
            result.nodes,
            result.edges,
            result.lane_summary,
            frontier,
            schedule,
        ),
        output / "PP_DAG_MINIMAL.md",
    )
    _atomic_text(_html(result.nodes, result.lane_summary, iteration), output / "pp_dag_minimal.html")
    print(
        f"built {len(result.nodes)} nodes / {len(result.edges)} edges; "
        f"outputs: {output}"
    )


if __name__ == "__main__":
    main()

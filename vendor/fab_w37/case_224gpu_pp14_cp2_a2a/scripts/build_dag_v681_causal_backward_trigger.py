#!/usr/bin/env python3
"""Patch v6.8 so non-terminal backward work is causally gated by PP gradients.

The source Trace F->B interval is an observed outcome of downstream pipeline
work.  It must not be replayed as an independent local wait in parallel with
the explicit PP-gradient dependency.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_dag_v62_trace_timeline_comparison import (
    predicted_collective_bars,
    semantic_dependencies,
)
from build_dag_v68_profiler_entry_visualization import build_critical_path_payload
from build_dag_v68_target_assisted_profiler_entry import (
    REPO,
    atomic_csv,
    atomic_json,
    atomic_text,
    checked,
    configured_path,
    metric,
    replay,
    sha256,
)


DEFAULT_CONFIG = REPO / (
    "case_224gpu_pp14_cp2_a2a/config/"
    "dag_v681_causal_backward_trigger_2026w36.toml"
)
TARGET_COMPONENTS = (
    "compute_exposed_ns_model",
    "compute_overlap_ns_model",
    "network_service_ns_model",
    "software_sync_ns_model",
    "framework_residual_ns_model",
)
SOURCE_COMPONENTS = (
    "compute_work_ns",
    "network_service_ns",
    "software_sync_ns",
    "unclassified_calibration_ns",
)


def verify_seal(path: Path, expected_status: str) -> dict[str, Any]:
    seal = json.loads(checked(path).read_text(encoding="utf-8"))
    if seal.get("status") != expected_status:
        raise ValueError(f"unexpected seal status: {seal.get('status')}")
    for artifact in seal["artifacts"]:
        source = checked(Path(artifact["path"]))
        if source.stat().st_size != int(artifact["size_bytes"]) or sha256(source) != artifact["sha256"]:
            raise ValueError(f"sealed input changed: {source}")
    return seal


def zero_nonterminal_f2b(
    nodes: pd.DataFrame,
    transfer: pd.DataFrame,
    *,
    pp: int,
    component_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = transfer[
        transfer["transition_category"].eq("F2B")
        & transfer["pp_stage"].lt(pp - 1)
    ].copy()
    node_ids = set(selected["node_id"].astype(str))
    output = nodes.copy()
    mask = output["node_id"].astype(str).isin(node_ids)
    if int(mask.sum()) != len(node_ids):
        raise ValueError(f"F2B correction node mismatch: {int(mask.sum())} != {len(node_ids)}")
    old = output.loc[mask, ["node_id", "duration_ns", "pp_stage", "pp_lane"]].copy()
    old = old.rename(columns={"duration_ns": "old_observed_f2b_gap_ns"})
    selected = selected.merge(old, on=["node_id", "pp_stage", "pp_lane"], validate="one_to_one")
    selected["new_program_order_cost_ns"] = 0
    selected["semantic_action"] = "diagnostic_only_not_a_duration"
    selected["required_trigger"] = "downstream_pp_gradient_receive_complete"

    output.loc[mask, "duration_ns"] = 0
    for column in component_columns:
        if column in output.columns:
            output.loc[mask, column] = 0
    # Keep the structural kind so existing timeline tooling can still locate
    # every schedule handoff.  Its zero duration and timing fields carry the
    # corrected semantics; it is no longer a local wait cost.
    output.loc[mask, "op_name"] = "phase_handoff_f2b_program_order"
    output.loc[mask, "cost_status"] = "CAUSAL_SEMANTICS_PATCH"
    output.loc[mask, "cost_source"] = "no_independent_f2b_wait_cost"
    output.loc[mask, "timing_component"] = "program_order_zero_duration"
    output.loc[mask, "timing_source"] = "backward_requires_downstream_gradient"
    return output, selected.sort_values(["pp_stage", "pp_lane", "sequence_index"])


def audit_gradient_dependencies(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    correction: pd.DataFrame,
) -> pd.DataFrame:
    node_ids = set(nodes["node_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for item in correction.itertuples(index=False):
        handoff_id = str(item.node_id)
        outgoing = edges[
            edges["src"].astype(str).eq(handoff_id)
            & edges["edge_type"].eq("phase_handoff_complete")
        ]
        if len(outgoing) != 1:
            raise ValueError(f"bad F2B handoff output: {handoff_id}")
        backward_start = str(outgoing.iloc[0]["dst"])
        incoming = edges[edges["dst"].astype(str).eq(backward_start)]
        gradient = incoming[incoming["edge_type"].eq("pp_gradient_recv")]
        activation = incoming[incoming["edge_type"].eq("autograd_activation_dependency")]
        if len(gradient) != 1 or len(activation) != 1:
            raise ValueError(f"backward prerequisite grid incomplete: {backward_start}")
        gradient_node = str(gradient.iloc[0]["src"])
        if gradient_node not in node_ids:
            raise ValueError(f"gradient node missing: {gradient_node}")
        rows.append({
            "rank": int(item.rank),
            "pp_stage": int(item.pp_stage),
            "pp_lane": int(item.pp_lane),
            "microbatch": int(item.to_microbatch),
            "program_order_node": handoff_id,
            "backward_start_node": backward_start,
            "gradient_receive_node": gradient_node,
            "program_order_cost_ns": 0,
            "gradient_is_mandatory_predecessor": True,
            "join_semantics": "all_prerequisites_required_max_is_not_priority",
        })
    return pd.DataFrame(rows).sort_values(["pp_stage", "pp_lane", "microbatch"])


def component_conserved(nodes: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    available = [column for column in columns if column in nodes.columns]
    totals = nodes[available].fillna(0).sum(axis=1).round().astype("int64")
    return bool(totals.eq(nodes["duration_ns"].astype("int64")).all())


def phase_bars(nodes: pd.DataFrame) -> list[dict[str, Any]]:
    phase_nodes = nodes[
        nodes["rank"].ge(0) & nodes["phase"].isin(["FWD", "BWD"])
    ].copy()
    ranks = phase_nodes.groupby(
        ["rank", "pp_stage", "pp_lane", "phase", "microbatch"], as_index=False
    ).agg(
        start_ns=("predicted_start_ns", "min"),
        end_ns=("predicted_end_ns", "max"),
        critical=("on_critical_path", "any"),
    )
    rows: list[dict[str, Any]] = []
    for (stage, phase, microbatch), group in ranks.groupby(
        ["pp_stage", "phase", "microbatch"], sort=True
    ):
        if group["rank"].nunique() != 16:
            raise ValueError(f"phase envelope is not 16-rank complete: {stage}/{phase}/{microbatch}")
        start_ns = int(group["start_ns"].min())
        end_ns = int(group["end_ns"].max())
        symbol = "F" if phase == "FWD" else "B"
        rows.append({
            "id": f"s{int(stage)}:{symbol}{int(microbatch)}",
            "stage": int(stage),
            "phase": "forward" if phase == "FWD" else "backward",
            "microbatch": int(microbatch),
            "start_ms": start_ns / 1e6,
            "end_ms": end_ns / 1e6,
            "duration_ms": (end_ns - start_ns) / 1e6,
            "rank_count": 16,
            "critical": bool(group["critical"].any()),
        })
    if len(rows) != 14 * 2 * 3:
        raise ValueError(f"target phase grid incomplete: {len(rows)}")
    return rows


def pp_dependencies() -> list[dict[str, Any]]:
    return [
        row for row in semantic_dependencies(pp=14, microbatches=3)
        if row["kind"] in {"pp_activation", "pp_gradient"}
    ]


def branch_snapshot(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    backward_start: str,
) -> dict[str, Any]:
    lookup = nodes.set_index("node_id", drop=False)
    target = lookup.loc[backward_start]
    incoming = edges[edges["dst"].astype(str).eq(backward_start)]
    candidates = []
    for edge in incoming.itertuples(index=False):
        row = lookup.loc[str(edge.src)]
        candidates.append({
            "node_id": str(edge.src),
            "edge_type": str(edge.edge_type),
            "arrival_ms": float(row.predicted_end_ns) / 1e6,
            "duration_ms": float(row.duration_ns) / 1e6,
            "selected_as_critical_predecessor": str(edge.src) == str(target.critical_predecessor),
        })
    return {
        "backward_start_node": backward_start,
        "start_ms": float(target.predicted_start_ns) / 1e6,
        "critical_predecessor": str(target.critical_predecessor),
        "prerequisites": sorted(candidates, key=lambda row: row["arrival_ms"], reverse=True),
    }


def causal_critical_path_payload(
    nodes_path: Path,
    edges_path: Path,
    profiler_ms: float,
    node_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Reuse max-plus reconstruction but replace the obsolete wait labels."""
    payload = build_critical_path_payload(nodes_path, edges_path, profiler_ms)
    nodes = node_frame.set_index("node_id")

    def label(node_id: str, fallback: str) -> str:
        if node_id not in nodes.index:
            return fallback
        row = nodes.loc[node_id]
        if str(row.get("timing_component", "")) == "program_order_zero_duration":
            return f"PP{int(row.pp_stage)} 程序顺序（0ms）"
        if (
            str(row.get("op_name", "")) == "phase_handoff_f2b"
            and int(row.pp_stage) == 13
        ):
            return "PP13 loss反向启动"
        return fallback

    for span in payload["spans"]:
        span["label"] = label(str(span["first_node_id"]), str(span["label"]))
    for decision in payload["join_decisions"]:
        decision["winner"] = label(str(decision["winner_node_id"]), str(decision["winner"]))
        decision["runner_up"] = label(
            str(decision["runner_up_node_id"]), str(decision["runner_up"])
        )
    dominant = payload["dominant_join_decision"]
    dominant["winner"] = label(str(dominant["winner_node_id"]), str(dominant["winner"]))
    dominant["runner_up"] = label(
        str(dominant["runner_up_node_id"]), str(dominant["runner_up"])
    )
    return payload


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    old = payload["prediction"]["v68_profiler_ms"]
    new = payload["prediction"]["v681_profiler_ms"]
    mape = payload["evaluation_summary"]["all_60_100_profiler_mape_pct"]
    late = payload["evaluation_summary"]["late_85_100_profiler_mape_pct"]
    corrected = payload["correction"]["target_node_count"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAG v6.8.1 · 反向梯度因果修复</title>
<style>
:root{{--bg:#07111d;--panel:#0e1b2c;--line:#2b4059;--text:#e9f2fb;--muted:#93a8bd;--f:#24c8ff;--b:#ff725c;--entry:#f5a524;--grad:#ef79ff;--act:#55d6be;--critical:#ffe45e;--dp:#27d17f;--edp:#b78cff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,sans-serif}}main{{max-width:1580px;margin:auto;padding:26px}}h1{{margin:4px 0 7px;font-size:31px}}h2{{margin:0 0 7px;font-size:19px}}p{{margin:5px 0}}.muted{{color:var(--muted)}}.eyebrow{{color:var(--act);font-size:12px;font-weight:800;letter-spacing:.12em}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:14px 0}}.card,.panel{{background:var(--panel);border:1px solid var(--line)}}.card{{padding:13px}}.card .k{{color:var(--muted);font-size:12px}}.card .v{{font-size:22px;font-weight:800;margin-top:4px}}.panel{{padding:15px;margin:11px 0}}.rule{{display:flex;gap:8px;align-items:center;overflow-x:auto;padding:8px 0}}.node{{min-width:170px;padding:11px;border:1px solid var(--line);background:#091827}}.node.hot{{border-color:var(--grad)}}.arrow{{font-size:20px;color:var(--muted)}}.formula{{padding:10px 12px;border-left:4px solid var(--grad);background:#171629;font:600 15px ui-monospace,monospace}}.controls{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}}button{{border:1px solid var(--line);background:#0b1828;color:var(--muted);padding:6px 10px;border-radius:6px;cursor:pointer}}button.active{{border-color:var(--act);color:#fff;background:#14383e}}.timeline{{overflow-x:auto;background:#081523;border:1px solid #22384f;padding:5px}}.timeline svg{{display:block;min-width:1350px;width:100%;height:auto}}.legend{{display:flex;gap:13px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:8px 0}}.sw{{display:inline-block;width:17px;height:8px;margin-right:5px}}.note{{border-left:4px solid var(--critical);background:#1b1b25;padding:10px 12px;margin:10px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:7px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{font-size:11px;color:var(--muted)}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<div class="eyebrow">DAG MFU · v6.8.1 CAUSAL PATCH</div><h1>反向只在下一级梯度到达后启动</h1>
<p class="muted">这是v6.8的语义修正版，不占用仍处于候选研究状态的v6.9版本号。</p>
<div class="cards"><div class="card"><div class="k">旧 v6.8 Profiler</div><div class="v">{old/1000:.3f}s</div></div><div class="card"><div class="k">修正后 v6.8.1</div><div class="v">{new/1000:.3f}s</div></div><div class="card"><div class="k">60–100开发集 MAPE</div><div class="v">{mape:.3f}%</div></div><div class="card"><div class="k">85–100开发集 MAPE</div><div class="v">{late:.3f}%</div></div></div>
<section class="panel"><h2>核心依赖</h2><div class="rule"><div class="node">本stage程序顺序<br><span class="muted">零时长约束</span></div><div class="arrow">＋</div><div class="node">本microbatch激活<br><span class="muted">F已完成</span></div><div class="arrow">＋</div><div class="node hot">下一级反向梯度<br><span class="muted">PP接收完成，必需</span></div><div class="arrow">→</div><div class="node">本级 B 开始</div></div><div class="formula">B(s,m).start = max(program_order, activation_ready, gradient_recv(s+1→s,m))</div><p class="muted">max表示三个条件必须全部满足，不表示“本地等待”和“梯度传输”二选一。Trace里的完整F→B间隔只是结果，不再作为本地耗时；共修正 {corrected} 个非末级节点。末级没有下一级梯度，只保留亚毫秒级loss/backward启动间隔。</p></section>
<div class="controls" id="iterations"></div><div class="legend"><span><i class="sw" style="background:var(--entry)"></i>ENTRY</span><span><i class="sw" style="background:var(--f)"></i>FWD</span><span><i class="sw" style="background:var(--b)"></i>BWD</span><span><i class="sw" style="background:var(--act)"></i>PP激活</span><span><i class="sw" style="background:var(--grad)"></i>PP梯度</span><span><i class="sw" style="background:var(--critical)"></i>关键路径节点</span></div>
<section class="panel"><h2>A · v6.8.1 预测</h2><p class="muted">黄色边表示回放后的最长路径；非末级B必须经过紫色PP梯度依赖。</p><div class="timeline" id="pred"></div></section>
<section class="panel"><h2>B · 224卡 Trace</h2><p class="muted">只用于封存后的开发评估，不回填本次语义修复。</p><div class="timeline" id="target"></div></section>
<section class="panel"><h2>C · 256卡源 Trace</h2><p class="muted">用于来源对照；旧版F→B长间隔正是这条完整流水线的观测结果。</p><div class="timeline" id="source"></div></section>
<div class="note"><b>精度说明：</b>因果修正使预测比v6.8缩短 {(old-new):.3f}ms。开发集误差可能变大，这说明旧长边曾偶然补偿其他缺失成本；本版没有读取224卡结果重新调参。</div>
<section class="panel"><h2>逐iteration开发评估</h2><table id="eval"></table></section>
<script>const D={data};const ns='http://www.w3.org/2000/svg';let selected=String(D.default_iteration);const $=id=>document.getElementById(id),fmt=(v,n=2)=>Number(v).toFixed(n);function add(p,t,a={{}},x=''){{const e=document.createElementNS(ns,t);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);if(x)e.textContent=x;p.appendChild(e);return e}}function color(c){{if(c.group_type==='dp')return c.collective==='rs'?'#27d17f':'#ffb54a';return c.collective==='rs'?'#00b8a9':'#b78cff'}}
function draw(target,base,barShift,entryWidth,profiler,trace=false,stages=14){{const W=1480,rowH=42,m={{l:62,r:24,t:55,b:36}},H=m.t+stages*rowH+m.b,total=Math.ceil(Math.max(D.prediction.v681_profiler_ms,...D.evaluation.map(r=>r.actual_profiler_ms))/5000)*5000,x=v=>m.l+(W-m.l-m.r)*v/total,svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);const er=add(svg,'rect',{{x:m.l,y:18,width:Math.max(2,x(entryWidth)-x(0)),height:16,rx:2,fill:'#f5a524'}});add(er,'title',{{}},`ENTRY · 0–${{fmt(entryWidth,3)}}ms`);add(svg,'text',{{x:m.l-7,y:31,'text-anchor':'end',fill:'#93a8bd','font-size':10}},'ENTRY');if(x(entryWidth)-x(0)>48)add(svg,'text',{{x:m.l+5,y:30,fill:'#241803','font-size':9,'font-weight':800}},`${{fmt(entryWidth/1000,4)}}s`);for(let s=0;s<stages;s++){{const y=m.t+s*rowH;add(svg,'rect',{{x:m.l,y,width:W-m.l-m.r,height:rowH,fill:s%2?'#0a1827':'#0c1b2c'}});add(svg,'text',{{x:m.l-7,y:y+24,'text-anchor':'end',fill:'#93a8bd','font-size':11}},`PP${{s}}`)}}for(let t=0;t<=total;t+=5000){{const xx=x(t);add(svg,'line',{{x1:xx,y1:13,x2:xx,y2:H-m.b,stroke:'#294057','stroke-dasharray':'3 4'}});add(svg,'text',{{x:xx,y:H-12,'text-anchor':'middle',fill:'#93a8bd','font-size':10}},`${{t/1000}}s`)}}const pos=new Map();for(const d of base.bars){{const start=d.start_ms+barShift,end=d.end_ms+barShift,y=m.t+d.stage*rowH+(d.phase==='forward'?3:17),r=add(svg,'rect',{{x:x(start),y,width:Math.max(1.5,x(end)-x(start)),height:9,rx:2,fill:d.phase==='forward'?'#24c8ff':'#ff725c',stroke:!trace&&d.critical?'#ffe45e':'none','stroke-width':!trace&&d.critical?2:0}});pos.set(d.id,{{start:x(start),end:x(end),y:y+4.5}});add(r,'title',{{}},`PP${{d.stage}} ${{d.phase==='forward'?'F':'B'}}${{d.microbatch}} · ${{fmt(start/1000,3)}}–${{fmt(end/1000,3)}}s`)}}for(const d of base.collectives||[]){{if(d.group_type==='handoff')continue;const start=d.start_ms+barShift,end=d.end_ms+barShift,y=m.t+d.stage*rowH+(d.group_type==='dp'?30:36),r=add(svg,'rect',{{x:x(start),y,width:Math.max(1.3,x(end)-x(start)),height:5,fill:color(d),stroke:!trace&&d.critical?'#ffe45e':'none'}});add(r,'title',{{}},`PP${{d.stage}} ${{d.group_type}} ${{d.collective}} · ${{fmt(d.duration_ms,2)}}ms`)}}if(!trace)for(const d of base.dependencies){{const a=pos.get(d.source),b=pos.get(d.target);if(!a||!b)continue;const c=d.kind==='pp_gradient'?'#ef79ff':'#55d6be',mid=(a.end+b.start)/2,p=add(svg,'path',{{d:`M ${{a.end}} ${{a.y}} C ${{mid}} ${{a.y}},${{mid}} ${{b.y}},${{b.start}} ${{b.y}}`,fill:'none',stroke:c,'stroke-width':1.3,'stroke-opacity':.62}});add(p,'title',{{}},d.label)}}add(svg,'line',{{x1:x(profiler),y1:13,x2:x(profiler),y2:H-m.b,stroke:'#fff','stroke-width':1.3}});target.replaceChildren(svg)}}
function table(){{$('eval').innerHTML='<thead><tr><th>Iteration</th><th>Trace Profiler</th><th>v6.8.1预测</th><th>APE</th></tr></thead><tbody>'+D.evaluation.map(r=>`<tr><td>iter ${{r.iteration}}</td><td>${{fmt(r.actual_profiler_ms/1000,3)}}s</td><td>${{fmt(r.predicted_profiler_ms/1000,3)}}s</td><td>${{fmt(r.ape_pct,3)}}%</td></tr>`).join('')+'</tbody>'}}function render(){{const tr=D.target_trace[selected],sr=D.source_trace[selected],ev=D.evaluation.find(r=>String(r.iteration)===selected),sb=D.source_boundaries[selected];draw($('pred'),D.predicted_timeline,0,D.entry_node.duration_ms,D.prediction.v681_profiler_ms,false,14);draw($('target'),tr,ev.actual_entry_ms,ev.actual_entry_ms,ev.actual_profiler_ms,true,14);draw($('source'),sr,sb.actual_entry_ms,sb.actual_entry_ms,sb.actual_profiler_ms,true,16);for(const b of $('iterations').querySelectorAll('button'))b.classList.toggle('active',b.dataset.i===selected)}}$('iterations').innerHTML=D.iterations.map(i=>`<button data-i="${{i}}">iter ${{i}}</button>`).join('');$('iterations').onclick=e=>{{if(e.target.dataset.i){{selected=e.target.dataset.i;render()}}}};table();render();</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    v68 = configured_path(config["inputs"]["v68_run_dir"])
    v67 = configured_path(config["inputs"]["v67_run_dir"])
    verify_seal(v68 / "predictions/prediction_seal.json", "SEALED_BEFORE_V68_TARGET_EVALUATOR_ACCESS")
    verify_seal(v67 / "predictions/prediction_seal.json", "SEALED_BEFORE_V67_EVALUATOR_ACCESS")
    v68_contract_path = checked(v68 / "prediction_contract.json")
    v68_nodes_path = checked(v68 / "predictions/dag_v68_nodes.csv.gz")
    v68_edges_path = checked(v68 / "predictions/dag_v68_edges.csv.gz")
    target_transfer_path = checked(v67 / "calibration/target_release_gap_transfer.csv")
    source_nodes_path = checked(v67 / "source_replay/dag_v67_source_nodes.csv.gz")
    source_edges_path = checked(v67 / "source_replay/dag_v67_source_edges.csv.gz")
    source_transfer_path = checked(v67 / "source_replay/source_release_gap_transfer.csv")
    source_contract_path = checked(v67 / "source_replay/source_replay_contract.json")
    timeline_path = checked(configured_path(config["inputs"]["v67_timeline_payload"]))
    v68_visualization_path = checked(configured_path(config["inputs"]["v68_visualization_payload"]))

    # Source-side causal audit.  No target timing is opened before the prediction seal.
    source_nodes = pd.read_csv(source_nodes_path, low_memory=False)
    source_edges = pd.read_csv(source_edges_path, low_memory=False)
    source_transfer = pd.read_csv(source_transfer_path)
    source_old_end_ns = int(source_nodes.loc[
        source_nodes["node_id"].eq(config["model"]["source_completion_node_id"]),
        "predicted_end_ns",
    ].item())
    source_nodes, source_correction = zero_nonterminal_f2b(
        source_nodes,
        source_transfer,
        pp=int(config["source"]["pp"]),
        component_columns=SOURCE_COMPONENTS,
    )
    source_nodes, _ = replay(source_nodes, source_edges, config["model"]["source_completion_node_id"])
    source_new_end_ns = int(source_nodes.loc[
        source_nodes["node_id"].eq(config["model"]["source_completion_node_id"]),
        "predicted_end_ns",
    ].item())
    if source_old_end_ns != source_new_end_ns:
        raise ValueError("source256 completion changed; F2B wall gap was unexpectedly causal")

    nodes = pd.read_csv(v68_nodes_path, low_memory=False)
    edges = pd.read_csv(v68_edges_path, low_memory=False)
    transfer = pd.read_csv(target_transfer_path)
    old_nodes = nodes.copy()
    nodes, correction = zero_nonterminal_f2b(
        nodes,
        transfer,
        pp=int(config["target"]["pp"]),
        component_columns=TARGET_COMPONENTS,
    )
    dependency_audit = audit_gradient_dependencies(nodes, edges, correction)
    nodes, critical = replay(nodes, edges, config["model"]["completion_node_id"])
    if not component_conserved(nodes, TARGET_COMPONENTS):
        raise ValueError("target component conservation failed")
    raw_ms = float(nodes.loc[
        nodes["node_id"].eq(config["model"]["completion_node_id"]), "predicted_end_ns"
    ].item()) / 1e6
    v68_contract = json.loads(v68_contract_path.read_text(encoding="utf-8"))
    v68_prediction = v68_contract["prediction"]
    reconciliation_ms = float(v68_prediction["post_graph_reconciliation_ms"])
    profiler_ms = raw_ms + reconciliation_ms
    outer_ms = float(v68_prediction["outer_framework_ms"])
    training_ms = profiler_ms + outer_ms
    mfu_pct = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"])
        * 1e12
        * training_ms / 1000.0
    )

    output = configured_path(config["outputs"]["output_dir"])
    predictions = output / "predictions"
    calibration = output / "calibration"
    evaluator = output / "evaluator_only"
    logs = output / "logs"
    for directory in (predictions, calibration, evaluator, logs):
        directory.mkdir(parents=True, exist_ok=True)
    nodes_out = predictions / "dag_v681_nodes.csv.gz"
    edges_out = predictions / "dag_v681_edges.csv.gz"
    critical_out = predictions / "dag_v681_critical_path.csv"
    methods_out = predictions / "method_predictions.csv"
    correction_out = calibration / "nonterminal_f2b_semantic_correction.csv"
    dependency_out = calibration / "backward_gradient_dependency_audit.csv"
    source_audit_out = calibration / "source256_f2b_causal_audit.json"
    contract_out = output / "prediction_contract.json"
    access_out = output / "input_access_audit.json"
    command_out = output / "reproduction_command.txt"
    atomic_csv(nodes_out, nodes, compression="gzip")
    atomic_csv(edges_out, edges, compression="gzip")
    atomic_csv(critical_out, critical)
    atomic_csv(correction_out, correction)
    atomic_csv(dependency_out, dependency_audit)
    iterations = [int(value) for value in config["evaluation"]["iterations"]]
    methods = pd.DataFrame([{
        "iteration": iteration,
        "method": "DAG v6.8.1 causal backward trigger",
        "predicted_raw_graph_ms": raw_ms,
        "predicted_profiler_step_ms": profiler_ms,
        "predicted_outer_framework_ms": outer_ms,
        "predicted_training_step_ms": training_ms,
        "predicted_mfu_pct": mfu_pct,
        "target_timing_read": False,
    } for iteration in iterations])
    atomic_csv(methods_out, methods)
    atomic_json(source_audit_out, {
        "schema": "dag-v6.8.1-source256-f2b-causal-audit-v1",
        "status": "PASS_SOURCE_COMPLETION_UNCHANGED",
        "corrected_nonterminal_f2b_nodes": len(source_correction),
        "source_raw_graph_before_ms": source_old_end_ns / 1e6,
        "source_raw_graph_after_ms": source_new_end_ns / 1e6,
        "delta_ms": (source_new_end_ns - source_old_end_ns) / 1e6,
        "interpretation": "explicit PP-gradient dependencies already determine source F-to-B timing",
    })
    atomic_json(contract_out, {
        "schema": "dag-v6.8.1-causal-backward-trigger-contract-v1",
        "status": "SOURCE256_ONLY_CAUSAL_PATCH_PREDICTION",
        "parent_version": "v6.8",
        "version": "v6.8.1",
        "target_timing_read_before_seal": False,
        "formal_blind_claim_allowed": False,
        "causal_rule": {
            "nonterminal_backward_start": "max(program_order_zero_cost, activation_ready, downstream_gradient_receive_complete)",
            "max_semantics": "all prerequisites required; not priority arbitration",
            "observed_f2b_gap_role": "diagnostic_only_not_a_duration",
            "terminal_stage": "keeps source256-derived local loss/backward launch gap",
        },
        "corrected_target_nodes": len(correction),
        "mandatory_gradient_dependency_rows": len(dependency_audit),
        "prediction": {
            "v68_raw_graph_ms": float(v68_prediction["raw_graph_ms"]),
            "raw_graph_ms": raw_ms,
            "causal_patch_delta_ms": raw_ms - float(v68_prediction["raw_graph_ms"]),
            "post_graph_reconciliation_ms": reconciliation_ms,
            "profiler_step_ms": profiler_ms,
            "outer_framework_ms": outer_ms,
            "training_step_ms": training_ms,
            "mfu_pct": mfu_pct,
        },
    })
    atomic_json(access_out, {
        "schema": "dag-v6.8.1-input-access-v1",
        "status": "PASS_NO_TARGET_TIMING_BEFORE_SEAL",
        "source_and_parent_files_read": [
            str(path.resolve()) for path in (
                v68_contract_path, v68_nodes_path, v68_edges_path, target_transfer_path,
                source_nodes_path, source_edges_path, source_transfer_path, source_contract_path,
            )
        ],
        "target_timing_files_read_before_seal": [],
        "target_parameter_updates": 0,
    })
    atomic_text(command_out, (
        ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
        "build_dag_v681_causal_backward_trigger.py --config "
        "case_224gpu_pp14_cp2_a2a/config/"
        "dag_v681_causal_backward_trigger_2026w36.toml\n"
    ))
    seal_out = predictions / "prediction_seal.json"
    sealed = [
        nodes_out, edges_out, critical_out, methods_out, correction_out,
        dependency_out, source_audit_out, contract_out, access_out, command_out,
    ]
    atomic_json(seal_out, {
        "schema": "dag-v6.8.1-prediction-seal-v1",
        "status": "SEALED_BEFORE_V681_TARGET_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sealed
        ],
    })

    # Target truth is evaluator-only and opened after the prediction seal.
    truth_path = checked(configured_path(config["inputs"]["target_ground_truth"]))
    truth = pd.read_csv(truth_path)
    truth = truth[truth["iteration"].isin(iterations)][
        ["iteration", "actual_profiler_step_ms", "actual_training_step_ms"]
    ]
    evaluation = methods.merge(truth, on="iteration", validate="one_to_one")
    evaluation["profiler_error_ms"] = evaluation["predicted_profiler_step_ms"] - evaluation["actual_profiler_step_ms"]
    evaluation["profiler_abs_error_pct"] = evaluation["profiler_error_ms"].abs() / evaluation["actual_profiler_step_ms"] * 100.0
    evaluation["training_error_ms"] = evaluation["predicted_training_step_ms"] - evaluation["actual_training_step_ms"]
    evaluation["training_abs_error_pct"] = evaluation["training_error_ms"].abs() / evaluation["actual_training_step_ms"] * 100.0
    evaluation_out = evaluator / "iteration_evaluation.csv"
    metrics_out = evaluator / "metrics.json"
    atomic_csv(evaluation_out, evaluation)
    late_iterations = [int(value) for value in config["evaluation"]["late_development"]]
    late = evaluation[evaluation["iteration"].isin(late_iterations)]
    metrics = {
        "schema": "dag-v6.8.1-target-development-evaluation-v1",
        "status": "PASS_CAUSAL_PATCH_TARGET_DEVELOPMENT_EVALUATION",
        "target_parameter_updates": 0,
        "formal_blind_claim_allowed": False,
        "all_60_100": {
            "profiler": metric(evaluation, "predicted_profiler_step_ms", "actual_profiler_step_ms"),
            "training": metric(evaluation, "predicted_training_step_ms", "actual_training_step_ms"),
        },
        "late_85_100": {
            "profiler": metric(late, "predicted_profiler_step_ms", "actual_profiler_step_ms"),
            "training": metric(late, "predicted_training_step_ms", "actual_training_step_ms"),
        },
    }
    atomic_json(metrics_out, metrics)

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    v68_visualization = json.loads(v68_visualization_path.read_text(encoding="utf-8"))
    predicted_bars = phase_bars(nodes)
    critical_lanes = nodes.loc[
        nodes["on_critical_path"].astype(bool) & nodes["pp_lane"].ge(0), "pp_lane"
    ].astype(int)
    representative_lane = int(critical_lanes.mode().iloc[0])
    predicted_collectives = predicted_collective_bars(nodes, representative_lane, 0)
    evaluation_rows = [{
        "iteration": int(row.iteration),
        "actual_entry_ms": float(v68_visualization["evaluation"][index]["actual_entry_ms"]),
        "actual_profiler_ms": float(row.actual_profiler_step_ms),
        "predicted_profiler_ms": float(row.predicted_profiler_step_ms),
        "ape_pct": float(row.profiler_abs_error_pct),
    } for index, row in enumerate(evaluation.itertuples(index=False))]
    old_snapshot = branch_snapshot(old_nodes, edges, "r50:s3:l2:bwd0:q3:start")
    new_snapshot = branch_snapshot(nodes, edges, "r50:s3:l2:bwd0:q3:start")
    critical_payload = causal_critical_path_payload(nodes_out, edges_out, profiler_ms, nodes)
    payload = {
        "schema": "dag-v6.8.1-causal-backward-trigger-visualization-v1",
        "status": "PASS_CAUSAL_PATCH_VISUALIZATION",
        "iterations": iterations,
        "default_iteration": int(config["evaluation"]["default_iteration"]),
        "entry_node": {
            "duration_ms": float(v68_visualization["entry_node"]["duration_ms"]),
            "source_case": "256gpu_pp16_cp2_a2a",
            "source_iteration": int(v68_visualization["entry_node"]["source_iteration"]),
        },
        "prediction": {
            "v68_profiler_ms": float(v68_prediction["profiler_step_ms"]),
            "v681_profiler_ms": profiler_ms,
            "delta_ms": profiler_ms - float(v68_prediction["profiler_step_ms"]),
        },
        "correction": {
            "target_node_count": len(correction),
            "source_node_count": len(source_correction),
            "source_completion_delta_ms": (source_new_end_ns - source_old_end_ns) / 1e6,
            "pp3_lane2_before": old_snapshot,
            "pp3_lane2_after": new_snapshot,
        },
        "predicted_timeline": {
            "bars": predicted_bars,
            "collectives": predicted_collectives,
            "dependencies": pp_dependencies(),
            "critical_path": critical_payload,
        },
        "target_trace": timeline["trace"],
        "source_trace": timeline["source_256_trace"],
        "source_boundaries": v68_visualization["source_256_profiler_boundaries"],
        "evaluation": evaluation_rows,
        "evaluation_summary": {
            "all_60_100_profiler_mape_pct": metrics["all_60_100"]["profiler"]["mape_pct"],
            "late_85_100_profiler_mape_pct": metrics["late_85_100"]["profiler"]["mape_pct"],
            "target_parameter_updates": 0,
        },
    }
    payload_out = evaluator / "dag_v681_causal_backward_trigger_payload.json"
    html_out = evaluator / "dag_v681_causal_backward_trigger.html"
    report_out = output / "DAG_V681_CAUSAL_BACKWARD_TRIGGER.md"
    provenance_out = output / "provenance.json"
    log_out = logs / "build.log"
    atomic_json(payload_out, payload)
    atomic_text(html_out, render_html(payload))
    atomic_text(report_out, f"""# DAG v6.8.1：反向梯度因果修复

- 非末级反向启动：`max(零时长程序顺序, 激活就绪, 下一级PP梯度接收完成)`。
- `max` 表示所有前置条件都要满足，不是本地等待和梯度传输二选一。
- 256卡Trace完整F→B间隔只保留为诊断数据，不再作为非末级本地耗时。
- 修正目标节点：`{len(correction)}`；修正源节点：`{len(source_correction)}`。
- 256卡源图完成时间变化：`{(source_new_end_ns-source_old_end_ns)/1e6:.6f} ms`。
- v6.8.1 Profiler预测：`{profiler_ms:.6f} ms`。
- 224卡开发集60–100 MAPE：`{metrics['all_60_100']['profiler']['mape_pct']:.6f}%`。

v6.8仍保留为冻结版本；本修复使用v6.8.1补丁号，v6.9继续保持未发布候选状态。精度评估在预测seal后进行，目标参数更新为0。
""")
    atomic_text(log_out, "\n".join([
        "status=PASS_CAUSAL_PATCH_TARGET_DEVELOPMENT_EVALUATION",
        f"corrected_target_f2b_nodes={len(correction)}",
        f"corrected_source_f2b_nodes={len(source_correction)}",
        f"source_completion_delta_ms={(source_new_end_ns-source_old_end_ns)/1e6:.6f}",
        f"v68_profiler_ms={float(v68_prediction['profiler_step_ms']):.6f}",
        f"v681_profiler_ms={profiler_ms:.6f}",
        f"target_development_60_100_mape_pct={metrics['all_60_100']['profiler']['mape_pct']:.6f}",
        "target_parameter_updates=0",
    ]) + "\n")
    input_paths = [
        config_path, Path(__file__).resolve(), v68_contract_path, v68_nodes_path,
        v68_edges_path, target_transfer_path, source_nodes_path, source_edges_path,
        source_transfer_path, source_contract_path, timeline_path,
        v68_visualization_path, truth_path,
    ]
    output_paths = [
        nodes_out, edges_out, critical_out, methods_out, correction_out,
        dependency_out, source_audit_out, contract_out, access_out, command_out,
        seal_out, evaluation_out, metrics_out, payload_out, html_out, report_out, log_out,
    ]
    atomic_json(provenance_out, {
        "schema": "dag-v6.8.1-causal-backward-trigger-provenance-v1",
        "status": metrics["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_parameter_updates": 0,
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in input_paths
        ],
        "outputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_paths
        ],
    })
    print(json.dumps({
        "status": metrics["status"],
        "html": str(html_out.resolve()),
        "v68_profiler_ms": float(v68_prediction["profiler_step_ms"]),
        "v681_profiler_ms": profiler_ms,
        "corrected_target_nodes": len(correction),
        "source_completion_delta_ms": (source_new_end_ns - source_old_end_ns) / 1e6,
        "target_development_mape_pct": metrics["all_60_100"]["profiler"]["mape_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

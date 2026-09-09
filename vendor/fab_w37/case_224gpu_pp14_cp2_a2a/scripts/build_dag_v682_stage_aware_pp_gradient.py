#!/usr/bin/env python3
"""Build v6.8.2 with source256 stage/lane/microbatch-aware PP gradients."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_dag_v62_trace_timeline_comparison import predicted_collective_bars
from build_dag_v681_causal_backward_trigger import (
    REPO,
    atomic_csv,
    atomic_json,
    atomic_text,
    component_conserved,
    phase_bars,
    pp_dependencies,
    sha256,
    verify_seal,
    zero_nonterminal_f2b,
)
from build_dag_v68_target_assisted_profiler_entry import (
    checked,
    configured_path,
    metric,
    replay,
)


DEFAULT_CONFIG = REPO / (
    "case_224gpu_pp14_cp2_a2a/config/dag_v682_stage_aware_pp_gradient_2026w36.toml"
)
TARGET_PP_PATTERN = re.compile(r"^pp:lane(\d+):bwd(\d+):s(\d+)_to_s(\d+)$")
SOURCE_PP_PATTERN = re.compile(r"^lane(\d+):B(\d+):s(\d+)->s(\d+)$")
TARGET_COMPONENTS = (
    "compute_exposed_ns_model", "compute_overlap_ns_model", "network_service_ns_model",
    "software_sync_ns_model", "framework_residual_ns_model",
)
SOURCE_COMPONENTS = (
    "compute_work_ns", "network_service_ns_model", "software_sync_ns_model",
    "unclassified_calibration_ns",
)


def fit_gradient_wall(events: pd.DataFrame, iterations: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch",
        "observed_start_ns", "observed_end_ns",
    }
    if not required.issubset(events.columns):
        raise ValueError("source PP event schema changed")
    selected = events[
        events["phase"].eq("backward") & events["iteration"].isin(iterations)
    ].copy()
    if set(selected["iteration"].astype(int)) != set(iterations):
        raise ValueError("source iteration grid incomplete")
    if selected["rank"].nunique() != 256:
        raise ValueError("source rank grid incomplete")
    receiver = selected.rename(columns={
        "pp_stage": "source_receiver_stage",
        "observed_start_ns": "receiver_bwd_start_ns",
    })[[
        "iteration", "pp_lane", "source_receiver_stage", "microbatch",
        "receiver_bwd_start_ns",
    ]]
    downstream = selected.rename(columns={
        "pp_stage": "source_downstream_stage",
        "observed_end_ns": "downstream_bwd_end_ns",
    })[[
        "iteration", "pp_lane", "source_downstream_stage", "microbatch",
        "downstream_bwd_end_ns",
    ]]
    samples = receiver.merge(
        downstream, on=["iteration", "pp_lane", "microbatch"], validate="many_to_many"
    )
    samples = samples[
        samples["source_downstream_stage"].eq(samples["source_receiver_stage"] + 1)
    ].copy()
    samples["gradient_wall_ns"] = (
        samples["receiver_bwd_start_ns"] - samples["downstream_bwd_end_ns"]
    )
    expected = len(iterations) * 15 * 16 * 4
    if len(samples) != expected or not samples["gradient_wall_ns"].gt(0).all():
        raise ValueError(f"source PP-gradient sample grid invalid: {len(samples)} != {expected}")
    parameters = samples.groupby(
        ["source_receiver_stage", "pp_lane", "microbatch"], as_index=False
    ).agg(
        gradient_wall_p10_ns=("gradient_wall_ns", lambda values: values.quantile(0.10)),
        gradient_wall_median_ns=("gradient_wall_ns", "median"),
        gradient_wall_p90_ns=("gradient_wall_ns", lambda values: values.quantile(0.90)),
        gradient_wall_min_ns=("gradient_wall_ns", "min"),
        gradient_wall_max_ns=("gradient_wall_ns", "max"),
        sample_count=("gradient_wall_ns", "size"),
    )
    for column in ("gradient_wall_p10_ns", "gradient_wall_median_ns", "gradient_wall_p90_ns"):
        parameters[column] = parameters[column].round().astype("int64")
    parameters["parameter_source"] = "source256_trace_iterations_60_100"
    if len(parameters) != 15 * 16 * 4 or not parameters["sample_count"].eq(len(iterations)).all():
        raise ValueError("source PP-gradient parameter grid incomplete")
    return parameters, samples.sort_values(
        ["source_receiver_stage", "pp_lane", "microbatch", "iteration"]
    )


def parameter_lookup(parameters: pd.DataFrame) -> dict[tuple[int, int, int], Any]:
    return {
        (int(row.source_receiver_stage), int(row.pp_lane), int(row.microbatch)): row
        for row in parameters.itertuples(index=False)
    }


def apply_target_gradient_wall(
    nodes: pd.DataFrame,
    parameters: pd.DataFrame,
    source_stage_map: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = nodes.copy()
    lookup = parameter_lookup(parameters)
    rows: list[dict[str, Any]] = []
    for index, node in output[output["kind"].eq("pp_p2p")].iterrows():
        match = TARGET_PP_PATTERN.fullmatch(str(node["node_id"]))
        if match is None:
            continue
        lane, microbatch, downstream_stage, receiver_stage = map(int, match.groups())
        if downstream_stage != receiver_stage + 1:
            raise ValueError(f"bad target PP gradient direction: {node['node_id']}")
        source_receiver_stage = source_stage_map[receiver_stage]
        parameter = lookup[(source_receiver_stage, lane, microbatch)]
        wall_ns = int(parameter.gradient_wall_median_ns)
        network_ns = int(node["network_service_ns_model"])
        if wall_ns < network_ns:
            raise ValueError(f"PP wall is shorter than network service: {node['node_id']}")
        old_wall_ns = int(node["duration_ns"])
        output.at[index, "duration_ns"] = wall_ns
        output.at[index, "software_sync_ns_model"] = wall_ns - network_ns
        output.at[index, "timing_source"] = "source256_receiver_stage_lane_microbatch_gradient_wall_median"
        output.at[index, "cost_source"] = "source256_trace_interstage_backward_boundary"
        output.at[index, "source_parameter_key"] = (
            f"pp_gradient:receiver{source_receiver_stage}:lane{lane}:mb{microbatch}"
        )
        rows.append({
            "target_node_id": str(node["node_id"]),
            "target_receiver_stage": receiver_stage,
            "target_downstream_stage": downstream_stage,
            "pp_lane": lane,
            "microbatch": microbatch,
            "source_receiver_stage": source_receiver_stage,
            "source_downstream_stage": source_receiver_stage + 1,
            "old_uniform_wall_ns": old_wall_ns,
            "new_stage_aware_wall_ns": wall_ns,
            "network_service_ns": network_ns,
            "software_completion_ns": wall_ns - network_ns,
            "source_sample_count": int(parameter.sample_count),
        })
    transfer = pd.DataFrame(rows).sort_values(
        ["target_receiver_stage", "pp_lane", "microbatch"]
    )
    if len(transfer) != 13 * 16 * 3:
        raise ValueError(f"target PP-gradient transfer grid incomplete: {len(transfer)}")
    return output, transfer


def apply_source_gradient_wall(
    nodes: pd.DataFrame, parameters: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = nodes.copy()
    lookup = parameter_lookup(parameters)
    rows: list[dict[str, Any]] = []
    for index, node in output[output["kind"].eq("pp_message")].iterrows():
        match = SOURCE_PP_PATTERN.fullmatch(str(node["node_id"]))
        if match is None:
            continue
        lane, microbatch, downstream_stage, receiver_stage = map(int, match.groups())
        parameter = lookup[(receiver_stage, lane, microbatch)]
        wall_ns = int(parameter.gradient_wall_median_ns)
        network_ns = int(node["network_service_ns_model"])
        old_wall_ns = int(node["duration_ns"])
        output.at[index, "duration_ns"] = wall_ns
        output.at[index, "software_sync_ns_model"] = wall_ns - network_ns
        rows.append({
            "source_node_id": str(node["node_id"]),
            "source_receiver_stage": receiver_stage,
            "source_downstream_stage": downstream_stage,
            "pp_lane": lane,
            "microbatch": microbatch,
            "old_uniform_wall_ns": old_wall_ns,
            "new_stage_aware_wall_ns": wall_ns,
        })
    transfer = pd.DataFrame(rows).sort_values(
        ["source_receiver_stage", "pp_lane", "microbatch"]
    )
    if len(transfer) != 15 * 16 * 4:
        raise ValueError(f"source PP-gradient replay grid incomplete: {len(transfer)}")
    return output, transfer


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    prediction = payload["prediction"]
    metrics = payload["metrics"]
    error = payload["error_analysis"]
    ledger = error["additive_wall_clock_ledger"]
    ledger_rows = "".join(
        "<tr>"
        f"<td>{component['label']}</td>"
        f"<td>{component['predicted_ms'] / 1000:.3f}s</td>"
        f"<td>{component['actual_mean_ms'] / 1000:.3f}s</td>"
        f"<td><b>+{component['underprediction_ms'] / 1000:.3f}s</b></td>"
        f"<td>{component['error_share_pct']:.1f}%</td>"
        f"<td>{component['meaning']}</td>"
        "</tr>"
        for component in ledger["components"]
    )
    tail = payload["gradient_summary"]["source_receiver_stage14_target_stage12_median_ms"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DAG v6.8.2 · 分位置PP梯度</title><style>
:root{{--bg:#07111d;--panel:#0e1b2c;--line:#2b4059;--text:#e9f2fb;--muted:#93a8bd;--f:#24c8ff;--b:#ff725c;--entry:#f5a524;--grad:#ef79ff;--act:#55d6be;--critical:#ffe45e}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,sans-serif}}main{{max-width:1580px;margin:auto;padding:25px}}h1{{font-size:31px;margin:4px 0 7px}}h2{{font-size:19px;margin:0 0 6px}}h3{{font-size:15px;margin:0 0 6px}}p{{margin:5px 0}}.muted{{color:var(--muted)}}.eyebrow{{color:var(--act);font-size:12px;font-weight:800;letter-spacing:.12em}}.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:14px 0}}.card,.panel{{background:var(--panel);border:1px solid var(--line)}}.card{{padding:12px}}.k{{font-size:12px;color:var(--muted)}}.v{{font-size:21px;font-weight:800;margin-top:4px}}.panel{{padding:15px;margin:11px 0}}.formula{{border-left:4px solid var(--grad);padding:10px 12px;background:#171629;font:600 15px ui-monospace,monospace}}.note{{border-left:4px solid var(--critical);padding:10px 12px;background:#1b1b25}}.error-summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:12px 0}}.error-item{{padding:12px;background:#091827;border:1px solid #294057;border-radius:5px}}.error-item.primary{{border-color:#ff725c;background:#22151a}}.error-item .n{{font-size:22px;font-weight:850;margin:3px 0}}.diagnosis{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:12px}}.diagnosis article{{padding:12px;background:#0a1726;border-top:3px solid #4f6882}}.diagnosis article:first-child{{border-top-color:#ff725c}}.diagnosis article:nth-child(2){{border-top-color:#f5a524}}.diagnosis article:nth-child(3){{border-top-color:#ef79ff}}.diagnosis article:nth-child(4){{border-top-color:#24c8ff}}.tag{{display:inline-block;padding:2px 7px;border-radius:10px;background:#26374a;color:#d6e6f5;font-size:11px;font-weight:750}}.controls{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}}button{{border:1px solid var(--line);background:#0b1828;color:var(--muted);padding:6px 10px;border-radius:6px}}button.active{{border-color:var(--act);color:white;background:#14383e}}.timeline{{overflow-x:auto;background:#081523;border:1px solid #22384f;padding:5px}}.timeline svg{{display:block;min-width:1350px;width:100%;height:auto}}.legend{{display:flex;gap:13px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:8px 0}}.sw{{display:inline-block;width:17px;height:8px;margin-right:5px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:7px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-size:11px}}@media(max-width:850px){{.cards,.error-summary,.diagnosis{{grid-template-columns:1fr 1fr}}}}@media(max-width:520px){{.error-summary,.diagnosis{{grid-template-columns:1fr}}}}</style></head><body><main>
<div class="eyebrow">DAG MFU · v6.8.2 SOURCE256-ONLY PP GRADIENT</div><h1>保留正确依赖，再补齐PP梯度位置差异</h1><p class="muted">完整F→B间隔仍不作为本地等待；本版只更新显式PP梯度边的墙钟。</p>
<div class="cards"><div class="card"><div class="k">v6.8（旧错误长边）</div><div class="v">{prediction['v68_profiler_ms']/1000:.3f}s</div></div><div class="card"><div class="k">v6.8.1（因果修复）</div><div class="v">{prediction['v681_profiler_ms']/1000:.3f}s</div></div><div class="card"><div class="k">v6.8.2（分位置梯度）</div><div class="v">{prediction['v682_profiler_ms']/1000:.3f}s</div></div><div class="card"><div class="k">60–100 MAPE</div><div class="v">{metrics['all_mape_pct']:.3f}%</div></div><div class="card"><div class="k">85–100 MAPE</div><div class="v">{metrics['late_mape_pct']:.3f}%</div></div></div>
<section class="panel"><h2>新增逻辑</h2><div class="formula">B(s,m).start = max(program_order, activation_ready, PP_gradient_done(s+1→s,m))</div><p>PP梯度完成时间不再全部固定为98.328ms，而由256卡Trace按接收stage、lane和microbatch取中位数。普通位置约98–99ms；映射到目标PP12的源PP15→PP14边中位数约 <b>{tail:.3f}ms</b>。</p><p class="muted">这段时间属于梯度传输与软件完成，不是本地F→B等待。ENTRY仍完整显示为1.2522秒。</p></section>
<div class="controls" id="iterations"></div><div class="legend"><span><i class="sw" style="background:var(--entry)"></i>ENTRY</span><span><i class="sw" style="background:var(--f)"></i>FWD</span><span><i class="sw" style="background:var(--b)"></i>BWD</span><span><i class="sw" style="background:var(--act)"></i>PP激活</span><span><i class="sw" style="background:var(--grad)"></i>PP梯度</span></div>
<section class="panel"><h2>A · v6.8.2 预测</h2><div class="timeline" id="v682"></div></section><section class="panel"><h2>B · v6.8.1 预测</h2><div class="timeline" id="v681"></div></section><section class="panel"><h2>C · 224卡 Trace</h2><div class="timeline" id="target"></div></section><section class="panel"><h2>D · 256卡源 Trace</h2><div class="timeline" id="source"></div></section>
<section class="panel"><h2>10.3%误差来源拆解</h2><p>按第一次前向计算与最后一次反向计算划分三个连续时间区间。三段互不重叠，差值严格相加为 <b>{error['mean_underprediction_ms']/1000:.3f}s</b>；测试集 <b>{error['underpredicted_iterations']}/{error['iteration_count']}</b> 个训练轮次全部低估。</p><div class="error-summary"><div class="error-item"><div class="k">v6.8.2预测</div><div class="n">{error['predicted_profiler_ms']/1000:.3f}s</div><div class="muted">冻结后每个测试点使用同一预测</div></div><div class="error-item"><div class="k">224卡实测均值</div><div class="n">{error['actual_profiler_mean_ms']/1000:.3f}s</div><div class="muted">训练轮次60–100，共{error['iteration_count']}点</div></div><div class="error-item primary"><div class="k">平均少算</div><div class="n">{error['mean_underprediction_ms']/1000:.3f}s</div><div class="muted">平均绝对百分比误差 {metrics['all_mape_pct']:.3f}%</div></div><div class="error-item"><div class="k">严格重构误差</div><div class="n">{ledger['reconstruction_error_ms']:.3e}ms</div><div class="muted">三段差值之和 − 总差值</div></div></div><table><thead><tr><th>训练阶段</th><th>模型预测</th><th>224卡实测均值</th><th>少算</th><th>占2.452秒</th><th>阶段包含内容</th></tr></thead><tbody>{ledger_rows}</tbody><tfoot><tr><th>完整训练轮次</th><th>{error['predicted_profiler_ms']/1000:.3f}s</th><th>{error['actual_profiler_mean_ms']/1000:.3f}s</th><th>+{error['mean_underprediction_ms']/1000:.3f}s</th><th>100.0%</th><th>采样范围内的一次训练轮次</th></tr></tfoot></table><div class="diagnosis"><article><span class="tag">75.8% · 最大项</span><h3>前向与反向流水线执行阶段 +1.860s</h3><p>这是第一次前向计算到最后一次反向计算的全局时间范围，包含计算、流水线传播、同步与运行时空档；目前不能全部称为纯计算。</p></article><article><span class="tag">13.3%</span><h3>训练迭代准备阶段 +0.327s</h3><p>224卡进入第一次前向计算之前比模型更慢，属于训练轮次入口和运行时准备成本。</p></article><article><span class="tag">10.8%</span><h3>参数更新与通信收尾阶段 +0.266s</h3><p>包含优化器、数据并行与专家并行各进程到齐、同步和完成尾部，不能只按网络通信量缩放。</p></article><article><span class="tag">下一层拆解</span><h3>网络传输不是独立时间段</h3><p>网络仿真返回时间嵌在后两段的依赖路径中；被并行计算隐藏的传输时间不能再次从总误差中扣除。</p></article></div><div class="note"><b>归因边界：</b>0.327 + 1.860 + 0.266秒是严格时间分账；但1.860秒内部的计算、调度、同步和通信仍会重叠，不能在没有更细关键路径对齐前强行继续按比例分摊。本页为封存后的224卡开发集诊断，目标参数更新仍为0。</div></section>
<div class="note"><b>版本边界：</b>v6.8.2相对v6.8.1只恢复 {prediction['v682_minus_v681_ms']:.3f}ms，MAPE改善 {metrics['v681_minus_v682_mape_pp']:.3f}个百分点，尚未回到v6.8的表面精度。剩余差异不能用旧F→B长边填充，需要继续补真实运行时成本。</div><section class="panel"><h2>逐iteration开发评估</h2><table id="eval"></table></section>
<script>const D={data},ns='http://www.w3.org/2000/svg';let selected=String(D.default_iteration);const $=x=>document.getElementById(x),fmt=(x,n=2)=>Number(x).toFixed(n);function add(p,t,a={{}},s=''){{const e=document.createElementNS(ns,t);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);if(s)e.textContent=s;p.appendChild(e);return e}}function cc(d){{if(d.group_type==='dp')return d.collective==='rs'?'#27d17f':'#ffb54a';return d.collective==='rs'?'#00b8a9':'#b78cff'}}function draw(el,base,shift,entry,prof,trace=false,stages=14){{const W=1480,rh=42,m={{l:62,r:24,t:55,b:36}},H=m.t+stages*rh+m.b,total=Math.ceil(Math.max(D.prediction.v68_profiler_ms,...D.evaluation.map(x=>x.actual_profiler_ms))/5000)*5000,x=v=>m.l+(W-m.l-m.r)*v/total,svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);const er=add(svg,'rect',{{x:m.l,y:18,width:Math.max(2,x(entry)-x(0)),height:16,rx:2,fill:'#f5a524'}});add(er,'title',{{}},`ENTRY 0–${{fmt(entry,3)}}ms`);add(svg,'text',{{x:m.l-7,y:31,'text-anchor':'end',fill:'#93a8bd','font-size':10}},'ENTRY');if(x(entry)-x(0)>48)add(svg,'text',{{x:m.l+5,y:30,fill:'#241803','font-size':9,'font-weight':800}},`${{fmt(entry/1000,4)}}s`);for(let s=0;s<stages;s++){{const y=m.t+s*rh;add(svg,'rect',{{x:m.l,y,width:W-m.l-m.r,height:rh,fill:s%2?'#0a1827':'#0c1b2c'}});add(svg,'text',{{x:m.l-7,y:y+24,'text-anchor':'end',fill:'#93a8bd','font-size':11}},`PP${{s}}`)}}for(let t=0;t<=total;t+=5000){{const xx=x(t);add(svg,'line',{{x1:xx,y1:13,x2:xx,y2:H-m.b,stroke:'#294057','stroke-dasharray':'3 4'}});add(svg,'text',{{x:xx,y:H-12,'text-anchor':'middle',fill:'#93a8bd','font-size':10}},`${{t/1000}}s`)}}const pos=new Map();for(const d of base.bars){{const a=d.start_ms+shift,b=d.end_ms+shift,y=m.t+d.stage*rh+(d.phase==='forward'?3:17),r=add(svg,'rect',{{x:x(a),y,width:Math.max(1.5,x(b)-x(a)),height:9,rx:2,fill:d.phase==='forward'?'#24c8ff':'#ff725c',stroke:!trace&&d.critical?'#ffe45e':'none','stroke-width':2}});pos.set(d.id,{{a:x(a),b:x(b),y:y+4.5}});add(r,'title',{{}},`PP${{d.stage}} ${{d.phase==='forward'?'F':'B'}}${{d.microbatch}} · ${{fmt(a/1000,3)}}–${{fmt(b/1000,3)}}s`)}}for(const d of base.collectives||[]){{if(d.group_type==='handoff')continue;const a=d.start_ms+shift,b=d.end_ms+shift,y=m.t+d.stage*rh+(d.group_type==='dp'?30:36);add(svg,'rect',{{x:x(a),y,width:Math.max(1.3,x(b)-x(a)),height:5,fill:cc(d)}})}}if(!trace)for(const d of base.dependencies){{const a=pos.get(d.source),b=pos.get(d.target);if(!a||!b)continue;const c=d.kind==='pp_gradient'?'#ef79ff':'#55d6be',mid=(a.b+b.a)/2;add(svg,'path',{{d:`M ${{a.b}} ${{a.y}} C ${{mid}} ${{a.y}},${{mid}} ${{b.y}},${{b.a}} ${{b.y}}`,fill:'none',stroke:c,'stroke-width':1.3,'stroke-opacity':.62}})}}add(svg,'line',{{x1:x(prof),y1:13,x2:x(prof),y2:H-m.b,stroke:'#fff','stroke-width':1.3}});el.replaceChildren(svg)}}function table(){{$('eval').innerHTML='<thead><tr><th>Iteration</th><th>Trace</th><th>v6.8.1 APE</th><th>v6.8.2 APE</th></tr></thead><tbody>'+D.evaluation.map(r=>`<tr><td>iter ${{r.iteration}}</td><td>${{fmt(r.actual_profiler_ms/1000,3)}}s</td><td>${{fmt(r.v681_ape_pct,3)}}%</td><td>${{fmt(r.v682_ape_pct,3)}}%</td></tr>`).join('')+'</tbody>'}}function render(){{const e=D.evaluation.find(x=>String(x.iteration)===selected),t=D.target_trace[selected],s=D.source_trace[selected],sb=D.source_boundaries[selected],en=D.entry_ms;draw($('v682'),D.v682_timeline,0,en,D.prediction.v682_profiler_ms);draw($('v681'),D.v681_timeline,0,en,D.prediction.v681_profiler_ms);draw($('target'),t,e.actual_entry_ms,e.actual_entry_ms,e.actual_profiler_ms,true);draw($('source'),s,sb.actual_entry_ms,sb.actual_entry_ms,sb.actual_profiler_ms,true,16);for(const b of $('iterations').querySelectorAll('button'))b.classList.toggle('active',b.dataset.i===selected)}}$('iterations').innerHTML=D.iterations.map(i=>`<button data-i="${{i}}">iter ${{i}}</button>`).join('');$('iterations').onclick=e=>{{if(e.target.dataset.i){{selected=e.target.dataset.i;render()}}}};table();render();</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    v681 = configured_path(config["inputs"]["v681_run_dir"])
    v67 = configured_path(config["inputs"]["v67_run_dir"])
    verify_seal(v681 / "predictions/prediction_seal.json", "SEALED_BEFORE_V681_TARGET_EVALUATOR_ACCESS")
    verify_seal(v67 / "predictions/prediction_seal.json", "SEALED_BEFORE_V67_EVALUATOR_ACCESS")
    parent_contract_path = checked(v681 / "prediction_contract.json")
    parent_nodes_path = checked(v681 / "predictions/dag_v681_nodes.csv.gz")
    parent_edges_path = checked(v681 / "predictions/dag_v681_edges.csv.gz")
    source_trace_path = checked(configured_path(config["inputs"]["source_trace_events"]))
    source_nodes_path = checked(v67 / "source_replay/dag_v67_source_nodes.csv.gz")
    source_edges_path = checked(v67 / "source_replay/dag_v67_source_edges.csv.gz")
    source_release_path = checked(v67 / "source_replay/source_release_gap_transfer.csv")
    target_release_path = checked(v67 / "calibration/target_release_gap_transfer.csv")
    source_contract_path = checked(v67 / "source_replay/source_replay_contract.json")
    v67_contract_path = checked(v67 / "prediction_contract.json")
    iterations = tuple(int(value) for value in config["source"]["iterations"])
    events = pd.read_csv(source_trace_path)
    parameters, samples = fit_gradient_wall(events, iterations)

    source_nodes = pd.read_csv(source_nodes_path, low_memory=False)
    source_edges = pd.read_csv(source_edges_path, low_memory=False)
    source_release = pd.read_csv(source_release_path)
    source_nodes, _ = zero_nonterminal_f2b(
        source_nodes, source_release, pp=int(config["source"]["pp"]),
        component_columns=SOURCE_COMPONENTS,
    )
    source_nodes, source_transfer = apply_source_gradient_wall(source_nodes, parameters)
    source_nodes, _ = replay(source_nodes, source_edges, config["model"]["source_completion_node_id"])
    source_raw_ms = float(source_nodes.loc[
        source_nodes["node_id"].eq(config["model"]["source_completion_node_id"]),
        "predicted_end_ns",
    ].item()) / 1e6
    if not component_conserved(source_nodes, SOURCE_COMPONENTS):
        raise ValueError("source component conservation failed")
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_profiler_ms = float(source_contract["source_profiler_median_ms"])
    source_reconciliation_ms = source_profiler_ms - source_raw_ms
    if source_reconciliation_ms < 0:
        raise ValueError("stage-aware source graph exceeds source Profiler median")
    target_reconciliation_ms = source_reconciliation_ms * (
        int(config["target"]["microbatches"]) / int(config["source"]["microbatches"])
    )

    nodes = pd.read_csv(parent_nodes_path, low_memory=False)
    edges = pd.read_csv(parent_edges_path, low_memory=False)
    v67_contract = json.loads(v67_contract_path.read_text(encoding="utf-8"))
    source_stage_map = [int(value) for value in config["model"]["source_receiver_stage_for_target"]]
    if source_stage_map != [int(value) for value in v67_contract["target_release_gap_stage_to_source_stage"]]:
        raise ValueError("source receiver-stage map differs from sealed v6.7 role map")
    nodes, target_transfer = apply_target_gradient_wall(nodes, parameters, source_stage_map)
    nodes, critical = replay(nodes, edges, config["model"]["target_completion_node_id"])
    if not component_conserved(nodes, TARGET_COMPONENTS):
        raise ValueError("target component conservation failed")
    raw_ms = float(nodes.loc[
        nodes["node_id"].eq(config["model"]["target_completion_node_id"]), "predicted_end_ns"
    ].item()) / 1e6
    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
    parent_prediction = parent_contract["prediction"]
    profiler_ms = raw_ms + target_reconciliation_ms
    outer_ms = float(parent_prediction["outer_framework_ms"])
    training_ms = profiler_ms + outer_ms
    mfu_pct = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        int(config["target"]["world_size"]) * float(config["target"]["peak_tflops_per_gpu"])
        * 1e12 * training_ms / 1000.0
    )

    output = configured_path(config["outputs"]["output_dir"])
    predictions, calibration, source_replay, evaluator, logs = (
        output / "predictions", output / "calibration", output / "source_replay",
        output / "evaluator_only", output / "logs",
    )
    for directory in (predictions, calibration, source_replay, evaluator, logs):
        directory.mkdir(parents=True, exist_ok=True)
    nodes_out = predictions / "dag_v682_nodes.csv.gz"
    edges_out = predictions / "dag_v682_edges.csv.gz"
    critical_out = predictions / "dag_v682_critical_path.csv"
    methods_out = predictions / "method_predictions.csv"
    parameters_out = calibration / "source256_pp_gradient_wall_parameters.csv"
    samples_out = calibration / "source256_pp_gradient_wall_samples.csv"
    target_transfer_out = calibration / "target_pp_gradient_wall_transfer.csv"
    source_transfer_out = source_replay / "source_pp_gradient_wall_transfer.csv"
    source_contract_out = source_replay / "source_replay_contract.json"
    contract_out = output / "prediction_contract.json"
    access_out = output / "input_access_audit.json"
    command_out = output / "reproduction_command.txt"
    atomic_csv(nodes_out, nodes, compression="gzip")
    atomic_csv(edges_out, edges, compression="gzip")
    atomic_csv(critical_out, critical)
    atomic_csv(parameters_out, parameters)
    atomic_csv(samples_out, samples)
    atomic_csv(target_transfer_out, target_transfer)
    atomic_csv(source_transfer_out, source_transfer)
    atomic_json(source_contract_out, {
        "schema": "dag-v6.8.2-source256-stage-aware-pp-gradient-replay-v1",
        "status": "PASS_SOURCE256_REPLAY",
        "source_raw_graph_ms": source_raw_ms,
        "source_profiler_median_ms": source_profiler_ms,
        "source_reconciliation_ms": source_reconciliation_ms,
        "target_reconciliation_scale": 0.75,
        "target_reconciliation_ms": target_reconciliation_ms,
        "parameter_rows": len(parameters),
        "sample_rows": len(samples),
        "target_timing_files_read": 0,
    })
    evaluation_iterations = [int(value) for value in config["evaluation"]["iterations"]]
    methods = pd.DataFrame([{
        "iteration": iteration,
        "method": "DAG v6.8.2 stage-aware PP gradient",
        "predicted_raw_graph_ms": raw_ms,
        "predicted_profiler_step_ms": profiler_ms,
        "predicted_outer_framework_ms": outer_ms,
        "predicted_training_step_ms": training_ms,
        "predicted_mfu_pct": mfu_pct,
        "target_timing_read": False,
    } for iteration in evaluation_iterations])
    atomic_csv(methods_out, methods)
    atomic_json(contract_out, {
        "schema": "dag-v6.8.2-stage-aware-pp-gradient-contract-v1",
        "status": "SOURCE256_ONLY_PREDICTION",
        "parent_version": "v6.8.1",
        "target_timing_read_before_seal": False,
        "target_parameter_updates": 0,
        "causal_rule_preserved": parent_contract["causal_rule"],
        "pp_gradient_parameterization": "source receiver stage x lane x microbatch median wall",
        "prediction": {
            "v68_profiler_ms": float(parent_prediction["profiler_step_ms"] - parent_prediction["causal_patch_delta_ms"]),
            "v681_profiler_ms": float(parent_prediction["profiler_step_ms"]),
            "raw_graph_ms": raw_ms,
            "source_reconciliation_origin_ms": source_reconciliation_ms,
            "target_reconciliation_ms": target_reconciliation_ms,
            "profiler_step_ms": profiler_ms,
            "outer_framework_ms": outer_ms,
            "training_step_ms": training_ms,
            "mfu_pct": mfu_pct,
        },
    })
    atomic_json(access_out, {
        "schema": "dag-v6.8.2-input-access-v1",
        "status": "PASS_SOURCE256_ONLY_BEFORE_SEAL",
        "source_trace_files_read": [str(source_trace_path.resolve())],
        "source_trace_fields_read": [
            "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch",
            "observed_start_ns", "observed_end_ns",
        ],
        "target_timing_files_read_before_seal": [],
        "target_parameter_updates": 0,
    })
    atomic_text(command_out, (
        ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
        "build_dag_v682_stage_aware_pp_gradient.py --config "
        "case_224gpu_pp14_cp2_a2a/config/dag_v682_stage_aware_pp_gradient_2026w36.toml\n"
    ))
    seal_out = predictions / "prediction_seal.json"
    sealed = [
        nodes_out, edges_out, critical_out, methods_out, parameters_out, samples_out,
        target_transfer_out, source_transfer_out, source_contract_out, contract_out,
        access_out, command_out,
    ]
    atomic_json(seal_out, {
        "schema": "dag-v6.8.2-prediction-seal-v1",
        "status": "SEALED_BEFORE_V682_TARGET_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sealed
        ],
    })

    truth_path = checked(configured_path(config["inputs"]["target_ground_truth"]))
    truth = pd.read_csv(truth_path)
    truth = truth[truth["iteration"].isin(evaluation_iterations)][
        ["iteration", "actual_profiler_step_ms", "actual_training_step_ms"]
    ]
    evaluation = methods.merge(truth, on="iteration", validate="one_to_one")
    evaluation["profiler_error_ms"] = evaluation["predicted_profiler_step_ms"] - evaluation["actual_profiler_step_ms"]
    evaluation["profiler_abs_error_pct"] = evaluation["profiler_error_ms"].abs() / evaluation["actual_profiler_step_ms"] * 100.0
    parent_eval = pd.read_csv(v681 / "evaluator_only/iteration_evaluation.csv")[
        ["iteration", "predicted_profiler_step_ms", "profiler_abs_error_pct"]
    ].rename(columns={
        "predicted_profiler_step_ms": "v681_predicted_profiler_step_ms",
        "profiler_abs_error_pct": "v681_profiler_abs_error_pct",
    })
    evaluation = evaluation.merge(parent_eval, on="iteration", validate="one_to_one")
    evaluation_out = evaluator / "iteration_evaluation.csv"
    metrics_out = evaluator / "metrics.json"
    atomic_csv(evaluation_out, evaluation)
    late_iterations = [int(value) for value in config["evaluation"]["late_development"]]
    late = evaluation[evaluation["iteration"].isin(late_iterations)]
    all_metric = metric(evaluation, "predicted_profiler_step_ms", "actual_profiler_step_ms")
    late_metric = metric(late, "predicted_profiler_step_ms", "actual_profiler_step_ms")
    v681_mape = float(evaluation["v681_profiler_abs_error_pct"].mean())
    metrics = {
        "schema": "dag-v6.8.2-target-development-evaluation-v1",
        "status": "PASS_SOURCE256_ONLY_PP_GRADIENT_TARGET_DEVELOPMENT",
        "formal_blind_claim_allowed": False,
        "target_parameter_updates": 0,
        "all_60_100": {"profiler": all_metric},
        "late_85_100": {"profiler": late_metric},
        "comparison": {
            "v681_all_mape_pct": v681_mape,
            "v682_all_mape_pct": all_metric["mape_pct"],
            "improvement_pp": v681_mape - all_metric["mape_pct"],
        },
    }
    atomic_json(metrics_out, metrics)

    parent_payload_path = checked(configured_path(config["inputs"]["v681_visualization_payload"]))
    parent_payload = json.loads(parent_payload_path.read_text(encoding="utf-8"))
    representative_lane = int(nodes.loc[
        nodes["on_critical_path"].astype(bool) & nodes["pp_lane"].ge(0), "pp_lane"
    ].astype(int).mode().iloc[0])
    v682_timeline = {
        "bars": phase_bars(nodes),
        "collectives": predicted_collective_bars(nodes, representative_lane, 0),
        "dependencies": pp_dependencies(),
    }
    evaluation_rows = []
    entry_by_iteration = {
        int(row["iteration"]): float(row["actual_entry_ms"])
        for row in parent_payload["evaluation"]
    }
    for row in evaluation.itertuples(index=False):
        evaluation_rows.append({
            "iteration": int(row.iteration),
            "actual_entry_ms": entry_by_iteration[int(row.iteration)],
            "actual_profiler_ms": float(row.actual_profiler_step_ms),
            "v681_ape_pct": float(row.v681_profiler_abs_error_pct),
            "v682_ape_pct": float(row.profiler_abs_error_pct),
        })
    tail_transfer = target_transfer[target_transfer["target_receiver_stage"].eq(12)]
    predicted_fb_start_ms = min(float(bar["start_ms"]) for bar in v682_timeline["bars"])
    predicted_fb_end_ms = max(float(bar["end_ms"]) for bar in v682_timeline["bars"])
    predicted_ledger = {
        "entry": predicted_fb_start_ms,
        "forward_backward_envelope": predicted_fb_end_ms - predicted_fb_start_ms,
        "post_forward_backward_tail": profiler_ms - predicted_fb_end_ms,
    }
    actual_ledger_rows = []
    actual_profiler_by_iteration = {
        int(row.iteration): float(row.actual_profiler_step_ms)
        for row in evaluation.itertuples(index=False)
    }
    for evaluation_row in evaluation_rows:
        iteration = int(evaluation_row["iteration"])
        trace_bars = parent_payload["target_trace"][str(iteration)]["bars"]
        trace_fb_start_ms = min(float(bar["start_ms"]) for bar in trace_bars)
        trace_fb_end_ms = max(float(bar["end_ms"]) for bar in trace_bars)
        absolute_fb_start_ms = float(evaluation_row["actual_entry_ms"]) + trace_fb_start_ms
        absolute_fb_end_ms = float(evaluation_row["actual_entry_ms"]) + trace_fb_end_ms
        actual_ledger_rows.append({
            "iteration": iteration,
            "entry": absolute_fb_start_ms,
            "forward_backward_envelope": trace_fb_end_ms - trace_fb_start_ms,
            "post_forward_backward_tail": (
                actual_profiler_by_iteration[iteration] - absolute_fb_end_ms
            ),
        })
    actual_ledger_means = {
        key: sum(float(row[key]) for row in actual_ledger_rows) / len(actual_ledger_rows)
        for key in predicted_ledger
    }
    actual_profiler_mean_ms = float(evaluation["actual_profiler_step_ms"].mean())
    mean_underprediction_ms = float(
        (evaluation["actual_profiler_step_ms"] - evaluation["predicted_profiler_step_ms"]).mean()
    )
    late_mean_underprediction_ms = float(
        (late["actual_profiler_step_ms"] - late["predicted_profiler_step_ms"]).mean()
    )
    underpredicted_iterations = int(evaluation["profiler_error_ms"].lt(0).sum())
    component_metadata = {
        "entry": (
            "训练迭代准备阶段",
            "采样开始到第一次前向计算；包含训练轮次入口与运行时准备",
        ),
        "forward_backward_envelope": (
            "前向与反向流水线执行阶段",
            "第一次前向到最后一次反向；包含计算、流水线传播、同步与运行时空档",
        ),
        "post_forward_backward_tail": (
            "参数更新与通信收尾阶段",
            "最后一次反向到采样结束；包含优化器、数据并行与专家并行到齐和完成",
        ),
    }
    additive_components = []
    for key in predicted_ledger:
        underprediction_ms = actual_ledger_means[key] - predicted_ledger[key]
        label, meaning = component_metadata[key]
        additive_components.append({
            "component": key,
            "label": label,
            "predicted_ms": predicted_ledger[key],
            "actual_mean_ms": actual_ledger_means[key],
            "underprediction_ms": underprediction_ms,
            "error_share_pct": 100.0 * underprediction_ms / mean_underprediction_ms,
            "meaning": meaning,
        })
    reconstructed_underprediction_ms = sum(
        float(component["underprediction_ms"]) for component in additive_components
    )
    reconstruction_error_ms = reconstructed_underprediction_ms - mean_underprediction_ms
    if abs(reconstruction_error_ms) > 1e-8:
        raise ValueError(
            "wall-clock error decomposition does not reconstruct profiler error: "
            f"{reconstruction_error_ms} ms"
        )
    payload = {
        "schema": "dag-v6.8.2-stage-aware-pp-gradient-visualization-v1",
        "status": "PASS_V682_VISUALIZATION",
        "iterations": evaluation_iterations,
        "default_iteration": int(config["evaluation"]["default_iteration"]),
        "entry_ms": float(parent_payload["entry_node"]["duration_ms"]),
        "prediction": {
            "v68_profiler_ms": float(parent_payload["prediction"]["v68_profiler_ms"]),
            "v681_profiler_ms": float(parent_payload["prediction"]["v681_profiler_ms"]),
            "v682_profiler_ms": profiler_ms,
            "v682_minus_v681_ms": profiler_ms - float(parent_payload["prediction"]["v681_profiler_ms"]),
        },
        "metrics": {
            "all_mape_pct": all_metric["mape_pct"],
            "late_mape_pct": late_metric["mape_pct"],
            "v681_minus_v682_mape_pp": v681_mape - all_metric["mape_pct"],
        },
        "error_analysis": {
            "status": "PASS_POST_SEAL_EXACT_WALL_CLOCK_BOUNDARY_DECOMPOSITION",
            "predicted_profiler_ms": profiler_ms,
            "actual_profiler_mean_ms": actual_profiler_mean_ms,
            "mean_underprediction_ms": mean_underprediction_ms,
            "late_mean_underprediction_ms": late_mean_underprediction_ms,
            "underpredicted_iterations": underpredicted_iterations,
            "iteration_count": len(evaluation),
            "target_parameter_updates": 0,
            "additive_wall_clock_ledger": {
                "boundary_rule": (
                    "profiler_start_to_first_f; first_f_to_last_b; "
                    "last_b_to_profiler_end"
                ),
                "components": additive_components,
                "reconstructed_underprediction_ms": reconstructed_underprediction_ms,
                "reconstruction_error_ms": reconstruction_error_ms,
                "component_internal_causal_attribution_complete": False,
            },
        },
        "gradient_summary": {
            "source_parameter_rows": len(parameters),
            "target_transfer_rows": len(target_transfer),
            "source_receiver_stage14_target_stage12_median_ms": float(
                tail_transfer["new_stage_aware_wall_ns"].median() / 1e6
            ),
        },
        "v682_timeline": v682_timeline,
        "v681_timeline": parent_payload["predicted_timeline"],
        "target_trace": parent_payload["target_trace"],
        "source_trace": parent_payload["source_trace"],
        "source_boundaries": parent_payload["source_boundaries"],
        "evaluation": evaluation_rows,
    }
    payload_out = evaluator / "dag_v682_stage_aware_pp_gradient_payload.json"
    html_out = evaluator / "dag_v682_stage_aware_pp_gradient.html"
    report_out = output / "DAG_V682_STAGE_AWARE_PP_GRADIENT.md"
    log_out = logs / "build.log"
    provenance_out = output / "provenance.json"
    atomic_json(payload_out, payload)
    atomic_text(html_out, render_html(payload))
    atomic_text(report_out, f"""# DAG v6.8.2：分位置PP梯度墙钟

- 保留v6.8.1反向因果关系，不恢复完整F→B本地长等待。
- 256卡参数：接收stage × lane × microbatch，共`{len(parameters)}`行，每行9个iteration样本。
- 目标PP梯度边：`{len(target_transfer)}`条。
- v6.8.1预测：`{float(parent_prediction['profiler_step_ms']):.6f} ms`。
- v6.8.2预测：`{profiler_ms:.6f} ms`，增加`{profiler_ms-float(parent_prediction['profiler_step_ms']):.6f} ms`。
- 224卡开发集60–100 MAPE：`{all_metric['mape_pct']:.6f}%`，比v6.8.1改善`{v681_mape-all_metric['mape_pct']:.6f}`个百分点。
- 224卡Profiler均值：`{actual_profiler_mean_ms:.6f} ms`；v6.8.2平均低估`{mean_underprediction_ms:.6f} ms`，`{underpredicted_iterations}/{len(evaluation)}`个评估点全部低估。
- 严格时间分账：训练迭代准备阶段少算`{additive_components[0]['underprediction_ms']:.6f} ms`，前向与反向流水线执行阶段少算`{additive_components[1]['underprediction_ms']:.6f} ms`，参数更新与通信收尾阶段少算`{additive_components[2]['underprediction_ms']:.6f} ms`；重构误差`{reconstruction_error_ms:.3e} ms`。

源PP15→PP14在不同microbatch上明显慢于普通PP边；本版把这部分归入显式梯度传输与软件完成。预测封存前未读取224卡时序，目标参数更新为0。当前改善有限，剩余差异不能用已删除的错误F→B长边补齐。

上述三段是互不重叠、可相加的时间分账。前向与反向流水线执行阶段内部仍同时包含计算、流水线传播、调度、同步和通信，不能在尚未完成更细关键路径对齐时把整段差值叫作纯计算。
""")
    atomic_text(log_out, "\n".join([
        f"status={metrics['status']}", f"parameter_rows={len(parameters)}",
        f"target_transfer_rows={len(target_transfer)}", f"source_raw_graph_ms={source_raw_ms:.6f}",
        f"target_profiler_ms={profiler_ms:.6f}", f"all_60_100_mape_pct={all_metric['mape_pct']:.6f}",
        "target_parameter_updates=0",
    ]) + "\n")
    input_paths = [
        config_path, Path(__file__).resolve(), parent_contract_path, parent_nodes_path,
        parent_edges_path, source_trace_path, source_nodes_path, source_edges_path,
        source_release_path, target_release_path, source_contract_path, v67_contract_path,
        truth_path, parent_payload_path,
    ]
    output_paths = [
        nodes_out, edges_out, critical_out, methods_out, parameters_out, samples_out,
        target_transfer_out, source_transfer_out, source_contract_out, contract_out,
        access_out, command_out, seal_out, evaluation_out, metrics_out, payload_out,
        html_out, report_out, log_out,
    ]
    atomic_json(provenance_out, {
        "schema": "dag-v6.8.2-stage-aware-pp-gradient-provenance-v1",
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
        "status": metrics["status"], "html": str(html_out.resolve()),
        "v681_profiler_ms": float(parent_prediction["profiler_step_ms"]),
        "v682_profiler_ms": profiler_ms, "all_60_100_mape_pct": all_metric["mape_pct"],
        "mape_improvement_pp": v681_mape - all_metric["mape_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

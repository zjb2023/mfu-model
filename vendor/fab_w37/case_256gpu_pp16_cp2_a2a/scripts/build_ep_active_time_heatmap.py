#!/usr/bin/env python3
"""Build the audited per-GPU EP active-time spatial heatmap."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


PHASES = (
    ("fwd_dispatch", "EP forward dispatch", "forward expert dispatch"),
    ("fwd_combine", "EP forward combine", "forward expert combine"),
    (
        "bwd_dispatch",
        "EP backward dispatch",
        "combine_backward_dispatch; sub-5-ms split is LOW confidence",
    ),
    (
        "bwd_combine",
        "EP backward combine",
        "dispatch_backward_combine",
    ),
)
COMPOUND = "recompute_fwd_combine_plus_bwd_dispatch"
METRICS = (
    {
        "column": "deepep_tx_bw_GBps",
        "label": "DeepEP TX logical BW",
        "unit": "GB/s",
        "description": "Σ DeepEP send_bytes / Σ DeepEP elapsed_ns",
        "numerator": "deepep_send_bytes",
        "denominator": "deepep_elapsed_ns",
        "multiplier": 1.0,
    },
    {
        "column": "deepep_payload_bw_mtlink_clipped_GBps",
        "label": "Payload / MTLink-active (clipped)",
        "unit": "GB/s",
        "description": "DeepEP payload / union(active sample ∩ phase windows)",
        "numerator": "deepep_send_bytes",
        "denominator": "mtlink_active_time_clipped_ns",
        "multiplier": 1.0,
    },
    {
        "column": "deepep_payload_bw_mtlink_full_bucket_GBps",
        "label": "Payload / MTLink-active (full bucket)",
        "unit": "GB/s",
        "description": "DeepEP payload / union(full active buckets touching phase)",
        "numerator": "deepep_send_bytes",
        "denominator": "mtlink_active_time_full_bucket_ns",
        "multiplier": 1.0,
    },
    {
        "column": "mtlink_proportional_tx_rate_GBps",
        "label": "MTLink physical TX rate",
        "unit": "GB/s",
        "description": "overlap-proportional MTLink TX bytes / clipped active time",
        "numerator": "mtlink_proportional_tx_bytes",
        "denominator": "mtlink_active_time_clipped_ns",
        "multiplier": 1.0,
    },
    {
        "column": "shared_sample_fraction",
        "label": "Shared MTLink samples",
        "unit": "%",
        "description": "touched samples shared with another EP event",
        "numerator": None,
        "denominator": None,
        "multiplier": 100.0,
    },
    {
        "column": "long_gap_sample_fraction",
        "label": "Long-gap samples (>10 ms)",
        "unit": "%",
        "description": "fraction of touched samples with duration above 10 ms",
        "numerator": None,
        "denominator": None,
        "multiplier": 100.0,
    },
    {
        "column": "event_to_sample_resolution_ratio",
        "label": "Event/sample resolution",
        "unit": "×",
        "description": "phase elapsed or compound envelope / MTLink sample duration",
        "numerator": None,
        "denominator": None,
        "multiplier": 1.0,
    },
)


def parse_args() -> argparse.Namespace:
    default_case = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, default=default_case)
    parser.add_argument("--phase-csv", type=Path)
    parser.add_argument("--compound-csv", type=Path)
    parser.add_argument("--source-validation", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    args = parser.parse_args()
    case_root = args.case_root.resolve()
    result_dir = case_root / "results" / "ep_active_time_bw"
    args.phase_csv = args.phase_csv or (result_dir / "ep_phase_iter_gpu.csv")
    args.compound_csv = args.compound_csv or (
        result_dir / "ep_compound_burst_iter_gpu.csv"
    )
    args.source_validation = args.source_validation or (
        result_dir / "ep_active_time_bw_validation.json"
    )
    args.html_output = args.html_output or (
        case_root / "figures" / "communication_spatial_heatmaps_ep_active_time.html"
    )
    args.validation_output = args.validation_output or (
        result_dir / "ep_active_time_heatmap_validation.json"
    )
    return args


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def portable(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def matrix(
    frame: pd.DataFrame,
    value_column: str,
    ranks: tuple[int, ...],
    iterations: tuple[int, ...],
) -> list[list[Any]]:
    values = frame.pivot(index="rank", columns="iteration", values=value_column)
    values = values.reindex(index=ranks, columns=iterations)
    return clean_json(values.to_numpy().tolist())


def load_data(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = pd.read_csv(args.phase_csv)
    compound = pd.read_csv(args.compound_csv)
    source_validation = json.loads(
        args.source_validation.read_text(encoding="utf-8")
    )
    required = {
        "case_id",
        "iteration",
        "rank",
        "host",
        "gpu_id",
        "pp_stage",
        "event_count",
        "deepep_send_bytes",
        "deepep_elapsed_ns",
        "mtlink_active_time_clipped_ns",
        "mtlink_active_time_full_bucket_ns",
        "mtlink_proportional_tx_bytes",
        "shared_sample_fraction",
        "other_collective_sample_fraction",
        "long_gap_sample_fraction",
        "event_to_sample_resolution_ratio",
        "confidence",
        *(item["column"] for item in METRICS),
    }
    for name, frame in (("phase", phase), ("compound", compound)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    case_values = set(phase["case_id"]) | set(compound["case_id"])
    if len(case_values) != 1:
        raise ValueError(f"case_id mismatch: {sorted(case_values)}")
    case_id = str(next(iter(case_values)))
    ranks = tuple(sorted(int(value) for value in phase["rank"].unique()))
    iterations = tuple(
        sorted(int(value) for value in phase["iteration"].unique())
    )
    rank_meta = (
        phase[["rank", "host", "gpu_id", "local_rank", "pp_stage"]]
        .drop_duplicates()
        .sort_values("rank")
    )
    if len(rank_meta) != len(ranks):
        raise ValueError("rank metadata is not one-to-one")

    payload: dict[str, Any] = {
        "schema_version": "ep-active-time-heatmap-v1",
        "case_id": case_id,
        "world_size": len(ranks),
        "iterations": list(iterations),
        "ranks": list(ranks),
        "rank_meta": clean_json(rank_meta.to_dict("records")),
        "plot_height": max(980, round(len(ranks) * 5.8)),
        "metrics": list(METRICS),
        "behaviors": [],
        "source_status": source_validation.get("status"),
    }
    detail_columns = (
        "event_count",
        "deepep_send_bytes",
        "deepep_elapsed_ns",
        "mtlink_active_time_clipped_ns",
        "mtlink_active_time_full_bucket_ns",
        "mtlink_proportional_tx_bytes",
        "shared_sample_fraction",
        "other_collective_sample_fraction",
        "long_gap_sample_fraction",
        "event_to_sample_resolution_ratio",
        "confidence",
    )
    for phase_id, label, description in PHASES:
        group = phase[phase["phase"].eq(phase_id)]
        payload["behaviors"].append(
            {
                "id": phase_id,
                "label": label,
                "description": description,
                "kind": "phase",
                "metrics": {
                    item["column"]: matrix(
                        group, item["column"], ranks, iterations
                    )
                    for item in METRICS
                },
                "details": {
                    column: matrix(group, column, ranks, iterations)
                    for column in detail_columns
                },
            }
        )
    payload["behaviors"].append(
        {
            "id": COMPOUND,
            "label": "Recompute fwd-combine + bwd-dispatch",
            "description": "causal compound burst; preferred MTLink cross-domain view",
            "kind": "compound",
            "metrics": {
                item["column"]: matrix(
                    compound, item["column"], ranks, iterations
                )
                for item in METRICS
            },
            "details": {
                column: matrix(compound, column, ranks, iterations)
                for column in detail_columns
            },
        }
    )

    phase_key_duplicates = int(
        phase.duplicated(["case_id", "iteration", "rank", "phase"]).sum()
    )
    compound_key_duplicates = int(
        compound.duplicated(
            ["case_id", "iteration", "rank", "compound_behavior"]
        ).sum()
    )
    expected_phase_rows = len(PHASES) * len(ranks) * len(iterations)
    expected_compound_rows = len(ranks) * len(iterations)
    validation: dict[str, Any] = {
        "status": "PASS",
        "schema_version": payload["schema_version"],
        "case_id": case_id,
        "source_validation_status": source_validation.get("status"),
        "world_size": len(ranks),
        "iteration_count": len(iterations),
        "behavior_count": len(payload["behaviors"]),
        "metric_count": len(METRICS),
        "phase_row_count": len(phase),
        "expected_phase_row_count": expected_phase_rows,
        "compound_row_count": len(compound),
        "expected_compound_row_count": expected_compound_rows,
        "phase_key_duplicate_count": phase_key_duplicates,
        "compound_key_duplicate_count": compound_key_duplicates,
        "clipped_time_le_full_bucket": bool(
            phase["mtlink_active_time_clipped_ns"]
            .le(phase["mtlink_active_time_full_bucket_ns"])
            .all()
            and compound["mtlink_active_time_clipped_ns"]
            .le(compound["mtlink_active_time_full_bucket_ns"])
            .all()
        ),
        "fraction_bounds_valid": bool(
            phase[
                [
                    "shared_sample_fraction",
                    "other_collective_sample_fraction",
                    "long_gap_sample_fraction",
                ]
            ]
            .apply(lambda column: column.between(0, 1))
            .all()
            .all()
            and compound[
                [
                    "shared_sample_fraction",
                    "other_collective_sample_fraction",
                    "long_gap_sample_fraction",
                ]
            ]
            .apply(lambda column: column.between(0, 1))
            .all()
            .all()
        ),
        "bwd_dispatch_confidence_counts": clean_json(
            phase.loc[phase["phase"].eq("bwd_dispatch"), "confidence"]
            .value_counts()
            .to_dict()
        ),
        "compound_confidence_counts": clean_json(
            compound["confidence"].value_counts().to_dict()
        ),
        "metric_definitions": list(METRICS),
    }
    checks = (
        str(validation["source_validation_status"]).startswith("PASS"),
        len(phase) == expected_phase_rows,
        len(compound) == expected_compound_rows,
        phase_key_duplicates == 0,
        compound_key_duplicates == 0,
        validation["clipped_time_le_full_bucket"],
        validation["fraction_bounds_valid"],
        set(phase["phase"]) == {item[0] for item in PHASES},
        set(phase["iteration"]) == set(iterations),
        set(compound["iteration"]) == set(iterations),
        set(phase["rank"]) == set(ranks),
        set(compound["rank"]) == set(ranks),
    )
    if not all(checks):
        validation["status"] = "FAIL"
        raise RuntimeError(
            json.dumps(clean_json(validation), ensure_ascii=False, indent=2)
        )
    return payload, validation


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<script>__PLOTLY_JS__</script>
<style>
:root { --ink:#172331; --muted:#596775; --panel:#fff; --line:#d8e1e7; --accent:#0f766e; --warn:#9a3412; --bg:#f2f5f4; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:"IBM Plex Sans","Noto Sans CJK SC",sans-serif; }
main { max-width:1740px; margin:0 auto; padding:22px; }
h1 { margin:0 0 6px; font-size:29px; }
.subtitle { color:var(--muted); line-height:1.55; }
.back { color:var(--accent); text-decoration:none; font-weight:650; display:inline-block; margin-top:7px; }
.controls { display:grid; grid-template-columns:repeat(5,minmax(190px,1fr)); gap:12px; margin:16px 0; padding:15px; background:var(--panel); border:1px solid var(--line); border-radius:12px; }
label { display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }
select { width:100%; padding:9px; border:1px solid #b9c5cc; border-radius:8px; background:#fff; color:var(--ink); }
.warning { display:none; margin:0 0 14px; padding:12px 15px; color:#7c2d12; background:#fff7ed; border:1px solid #fdba74; border-radius:10px; line-height:1.55; }
.cards { display:grid; grid-template-columns:repeat(7,minmax(135px,1fr)); gap:9px; margin-bottom:13px; }
.card { min-height:78px; padding:12px; background:var(--panel); border:1px solid var(--line); border-radius:10px; }
.card .name { color:var(--muted); font-size:11px; }
.card .value { margin-top:7px; font-size:18px; font-weight:700; }
.plot-panel,.notes { background:var(--panel); border:1px solid var(--line); border-radius:12px; }
.plot-panel { padding:7px; }
#heatmap { width:100%; }
.notes { margin-top:13px; padding:15px 18px; color:#43515e; line-height:1.7; }
.notes code { background:#edf3f1; padding:2px 5px; border-radius:4px; }
@media(max-width:1000px){ .controls{grid-template-columns:1fr 1fr}.cards{grid-template-columns:1fr 1fr}main{padding:11px} }
</style>
</head>
<body><main>
<h1>__PAGE_HEADING__</h1>
<div class="subtitle">Per-GPU EP · 4 phases + causal compound · DeepEP payload 与 MTLink activity/physical 分域展示</div>
<a class="back" href="communication_spatial_heatmaps_ep_phase_constrained.html">← 返回上一版严格 phase 页面</a>
<section class="controls">
  <div><label for="behavior">EP phase / compound</label><select id="behavior"></select></div>
  <div><label for="metric">显示口径</label><select id="metric"></select></div>
  <div><label for="window">Profiler iteration</label><select id="window"></select></div>
  <div><label for="quality">置信度筛选</label><select id="quality"><option value="all">全部（保留 LOW 警告）</option><option value="medium">MEDIUM + HIGH</option><option value="high">仅 HIGH</option></select></div>
  <div><label for="scale">颜色上限</label><select id="scale"><option value="p99">P99（突出空间差异）</option><option value="max">实际最大值</option></select></div>
</section>
<div id="warning" class="warning"></div>
<section class="cards">
 <div class="card"><div class="name">显示单元</div><div class="value" id="cells">—</div></div>
 <div class="card"><div class="name">汇总/中位值</div><div class="value" id="aggregate">—</div></div>
 <div class="card"><div class="name">P95</div><div class="value" id="p95">—</div></div>
 <div class="card"><div class="name">LOW / NO_ACTIVITY</div><div class="value" id="low">—</div></div>
 <div class="card"><div class="name">shared sample 中位</div><div class="value" id="shared">—</div></div>
 <div class="card"><div class="name">分辨率比中位</div><div class="value" id="resolution">—</div></div>
 <div class="card"><div class="name">long-gap 中位</div><div class="value" id="longgap">—</div></div>
</section>
<section class="plot-panel"><div id="heatmap"></div></section>
<section class="notes">
 <b>字段边界：</b><code>DeepEP TX logical BW</code> 是 payload/DeepEP elapsed；
 两个 <code>Payload / MTLink-active</code> 是 interval-censored 参考值；
 只有 <code>MTLink physical TX rate</code> 使用 MTLink delta bytes。
 单独 bwd-dispatch 的 MTLink 视图受 5 ms bucket 与相邻 recompute fwd-combine 共享影响，
 应优先查看 causal compound。LOW 单元不会静默删除，除非主动使用置信度筛选。
</section>
</main><script>
const DATA=__PAYLOAD_JSON__;
const behaviorSelect=document.getElementById("behavior"),metricSelect=document.getElementById("metric"),windowSelect=document.getElementById("window"),qualitySelect=document.getElementById("quality"),scaleSelect=document.getElementById("scale");
document.getElementById("heatmap").style.height=`${DATA.plot_height}px`;
for(const item of DATA.behaviors){const o=document.createElement("option");o.value=item.id;o.textContent=`${item.label} — ${item.description}`;behaviorSelect.appendChild(o)}
for(const item of DATA.metrics){const o=document.createElement("option");o.value=item.column;o.textContent=`${item.label} (${item.unit})`;metricSelect.appendChild(o)}
let o=document.createElement("option");o.value="all";o.textContent="全部20个采集 iteration";windowSelect.appendChild(o);
for(const iteration of DATA.iterations){o=document.createElement("option");o.value=String(iteration);o.textContent=`iteration ${iteration}`;windowSelect.appendChild(o)}
function behavior(){return DATA.behaviors.find(x=>x.id===behaviorSelect.value)}
function metric(){return DATA.metrics.find(x=>x.column===metricSelect.value)}
function indices(){return windowSelect.value==="all"?DATA.iterations.map((_,i)=>i):[DATA.iterations.indexOf(Number(windowSelect.value))]}
function percentile(values,q){if(!values.length)return null;const s=[...values].sort((a,b)=>a-b),p=(s.length-1)*q,l=Math.floor(p),u=Math.ceil(p);return l===u?s[l]:s[l]+(s[u]-s[l])*(p-l)}
function format(value,unit){if(value===null||!Number.isFinite(value))return "—";if(unit==="%")return `${value.toFixed(2)}%`;if(unit==="×")return `${value.toFixed(3)}×`;return `${value.toFixed(3)} GB/s`}
function allowed(conf){if(qualitySelect.value==="all")return true;if(qualitySelect.value==="high")return conf==="HIGH";return conf==="HIGH"||conf==="MEDIUM"}
function render(){
 const b=behavior(),m=metric(),ix=indices(),iterations=ix.map(i=>DATA.iterations[i]),details=b.details;
 const z=b.metrics[m.column].map((row,r)=>ix.map(i=>allowed(details.confidence[r][i])&&row[i]!==null?row[i]*m.multiplier:null));
 const finite=[],shared=[],resolution=[],longgap=[];let numerator=0,denominator=0,low=0,total=0;
 let max=-Infinity,maxRank=null,maxIteration=null;
 const custom=DATA.ranks.map((rank,r)=>ix.map((sourceIndex,j)=>{const meta=DATA.rank_meta[r],conf=details.confidence[r][sourceIndex],value=z[r][j];total++;if(conf==="LOW"||conf==="NO_ACTIVITY")low++;if(value!==null&&Number.isFinite(value)){finite.push(value);shared.push(details.shared_sample_fraction[r][sourceIndex]*100);resolution.push(details.event_to_sample_resolution_ratio[r][sourceIndex]);longgap.push(details.long_gap_sample_fraction[r][sourceIndex]*100);if(m.numerator){numerator+=details[m.numerator][r][sourceIndex];denominator+=details[m.denominator][r][sourceIndex]}if(value>max){max=value;maxRank=rank;maxIteration=DATA.iterations[sourceIndex]}}
 return [meta.host,rank,meta.gpu_id,meta.pp_stage,DATA.iterations[sourceIndex],details.event_count[r][sourceIndex],details.deepep_send_bytes[r][sourceIndex]/1e9,details.deepep_elapsed_ns[r][sourceIndex]/1e6,details.mtlink_active_time_clipped_ns[r][sourceIndex]/1e6,details.mtlink_active_time_full_bucket_ns[r][sourceIndex]/1e6,conf,details.shared_sample_fraction[r][sourceIndex]*100,details.other_collective_sample_fraction[r][sourceIndex]*100,details.long_gap_sample_fraction[r][sourceIndex]*100,details.event_to_sample_resolution_ratio[r][sourceIndex]]}))
 const p99=percentile(finite,.99),actualMax=finite.length?Math.max(...finite):1,zmax=scaleSelect.value==="p99"?p99:actualMax;
 const tickvals=[],ticktext=[];for(let i=0;i<DATA.rank_meta.length;i+=8){tickvals.push(i+3.5);ticktext.push(`${DATA.rank_meta[i].host}<br>r${i}–${Math.min(i+7,DATA.world_size-1)}`)}
 const shapes=[];for(let r=7.5;r<DATA.world_size-1;r+=8)shapes.push({type:"line",xref:"paper",x0:0,x1:1,y0:r,y1:r,line:{color:"rgba(38,66,73,.28)",width:1}});
 const trace={type:"heatmap",x:iterations,y:DATA.ranks,z,customdata:custom,zmin:0,zmax:zmax>0?zmax:1,colorscale:[[0,"#f7fbf9"],[.18,"#d7eee6"],[.42,"#84cbbb"],[.67,"#26958a"],[.84,"#ef9c5d"],[1,"#b83245"]],hoverongaps:false,xgap:ix.length===1?10:0,colorbar:{title:{text:m.unit},thickness:18,len:.82},hovertemplate:[`${b.label}`,`Host: %{customdata[0]}`,`Rank: %{customdata[1]} / GPU%{customdata[2]} / PP%{customdata[3]}`,`Iteration: %{customdata[4]}`,`Calls: %{customdata[5]}`,`DeepEP send: %{customdata[6]:.3f} GB`,`DeepEP elapsed: %{customdata[7]:.3f} ms`,`active clipped/full: %{customdata[8]:.3f} / %{customdata[9]:.3f} ms`,`Confidence: %{customdata[10]}`,`shared/other/long-gap: %{customdata[11]:.2f}% / %{customdata[12]:.2f}% / %{customdata[13]:.2f}%`,`resolution: %{customdata[14]:.3f}×`,`${m.label}: %{z:.3f} ${m.unit}`,"<extra></extra>"].join("<br>")};
 const layout={title:{text:`${b.label}<br><sup>${m.label} · ${m.description}</sup>`,x:.01,xanchor:"left"},margin:{l:170,r:90,t:92,b:68},paper_bgcolor:"#fff",plot_bgcolor:"#d9dee3",font:{family:"IBM Plex Sans, Noto Sans CJK SC, sans-serif",color:"#172331"},xaxis:{title:"Profiled iteration",tickmode:"array",tickvals:iterations,ticktext:iterations.map(String),side:"top",showgrid:false},yaxis:{title:"Global rank / host",tickmode:"array",tickvals,ticktext,autorange:"reversed",showgrid:false},shapes,hoverlabel:{bgcolor:"#fff",bordercolor:"#27364a",font:{size:12}}};
 Plotly.react("heatmap",[trace],layout,{responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:["lasso2d","select2d"]});
 const aggregate=m.numerator&&denominator>0?numerator/denominator*m.multiplier:percentile(finite,.5);
 document.getElementById("cells").textContent=`${finite.length.toLocaleString()} / ${total.toLocaleString()}`;document.getElementById("aggregate").textContent=format(aggregate,m.unit);document.getElementById("p95").textContent=format(percentile(finite,.95),m.unit);document.getElementById("low").textContent=`${low.toLocaleString()} / ${total.toLocaleString()}`;document.getElementById("shared").textContent=format(percentile(shared,.5),"%");document.getElementById("resolution").textContent=format(percentile(resolution,.5),"×");document.getElementById("longgap").textContent=format(percentile(longgap,.5),"%");
 const warning=document.getElementById("warning"),parts=[];if(b.id==="bwd_dispatch")parts.push("bwd-dispatch 的独立 MTLink bucket 拆分在本 case 中为 LOW confidence；请优先切换到 causal compound。");if(m.column.includes("payload_bw_mtlink"))parts.push("当前值使用 DeepEP payload 和 MTLink 活跃时间，属于跨域参考值，不是 MTLink physical BW。");if(m.column==="mtlink_proportional_tx_rate_GBps"&&b.kind==="phase")parts.push("phase physical rate 含 sample-overlap 分摊；shared 比例高时不能声称该 phase 独占这些 bytes。");if(low>0)parts.push(`当前窗口有 ${low}/${total} 个 LOW 或 NO_ACTIVITY 单元。`);warning.textContent=parts.join(" ");warning.style.display=parts.length?"block":"none";
 }
behaviorSelect.value=DATA.behaviors[0].id;metricSelect.value=DATA.metrics[0].column;windowSelect.value="all";for(const element of [behaviorSelect,metricSelect,windowSelect,qualitySelect,scaleSelect])element.addEventListener("change",render);render();
</script></body></html>"""


def main() -> int:
    args = parse_args()
    repository_root = args.case_root.resolve().parent
    payload, validation = load_data(args)
    payload_json = json.dumps(
        clean_json(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    heading = f"{payload['world_size']}-GPU EP active-time 空间热力图"
    html = (
        HTML_TEMPLATE.replace("__PAGE_TITLE__", heading)
        .replace("__PAGE_HEADING__", heading)
        .replace("__PLOTLY_JS__", get_plotlyjs())
        .replace("__PAYLOAD_JSON__", payload_json)
    )
    atomic_text(args.html_output, html)
    validation.update(
        {
            "phase_csv": portable(args.phase_csv, repository_root),
            "compound_csv": portable(args.compound_csv, repository_root),
            "html_output": portable(args.html_output, repository_root),
            "html_size_bytes": args.html_output.stat().st_size,
            "html_self_contained_plotly": True,
            "quality_filter_options": ["all", "medium", "high"],
        }
    )
    atomic_text(
        args.validation_output,
        json.dumps(clean_json(validation), ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(clean_json(validation), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

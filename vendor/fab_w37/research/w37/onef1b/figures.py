"""Deterministic standalone SVG figures, HTML explorer, and exact projection tables."""
from html import escape
import json
import pandas as pd
from worker import csv, phases, schedule, WINDOW

F='#2374ab'; B='#ca5546'; PP='#8f50aa'; WAIT='#dde3ea'


def svg(width,height,title,subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{escape(title)}</title>',
            '<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#76568b"/></marker></defs>',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="22" y="29" font-family="sans-serif" font-size="20" fill="#182b40">{escape(title)}</text>',
            f'<text x="22" y="51" font-family="sans-serif" font-size="12" fill="#516378">{escape(subtitle)}</text>']


def text(s,x,y,t,size=11,color='#253b50'):
    s.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="{size}" fill="{color}">{escape(str(t))}</text>')


def rect(s,x,y,w,h,color,title='',stroke='none',dash=''):
    s.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{max(w,0.15):.3f}" height="{h}" fill="{color}" stroke="{stroke}" stroke-dasharray="{dash}"><title>{escape(title)}</title></rect>')


def finish(s,path):
    path.write_text(''.join(s)+'</svg>')


def figures(out,nodes,edges,payload):
    dest=out/'figures';dest.mkdir(exist_ok=True)
    phase=phases(nodes); lane=phase[phase.pp_lane.eq(0)]
    region={r.node_id:(r.schedule_region,r.operation_sequence) for r in nodes[nodes.op_name.isin(['fwd_start','bwd_start'])].itertuples()}
    region_by_key={(r.pp_stage,r.phase,r.microbatch):r.schedule_region for r in nodes[nodes.pp_lane.eq(0) & nodes.op_name.isin(['fwd_start','bwd_start'])].itertuples()}
    lane=lane.copy();lane['region']=[region_by_key[(r.pp_stage,r.phase,r.microbatch)] for r in lane.itertuples()]
    csv(out,'graph/lane0_phase_projection.csv',lane)
    W=1600;H=14*56+135;x=lambda ms:95+ms/25000*1470
    s=svg(W,H,'W37 1F1B — frozen v6.8.5 prediction, lane 0',
          'F blue / B red; W warmup / S steady / C cooldown. Purple: PP completion interval. Grey: local F/B gap, not GPU idle proof. Times in seconds.')
    waits=[]
    for stage,g in lane.groupby('pp_stage'):
        y=80+int(stage)*56;text(s,15,y+20,f'PP {stage}')
        rect(s,95,y,1470,45,'#f4f7fa')
        previous=None
        for r in g.sort_values('start_ns').itertuples():
            start=r.start_ns/1e6;end=r.end_ns/1e6
            if previous is not None and start>previous:
                rect(s,x(previous),y+8,x(start)-x(previous),22,WAIT,f'Local phase gap {start-previous:.6f} ms; dependencies/communication/runtime may overlap')
                waits.append({'pp_stage':int(stage),'pp_lane':0,'start_ms':previous,'end_ms':start,'duration_ms':start-previous,
                              'semantics':'complement between F/B phase envelopes; not an inserted DAG node or independent extra cost'})
            color=F if r.phase=='FWD' else B
            rect(s,x(start),y+8,x(end)-x(start),22,color,f'Predicted PP{stage} {r.phase} MB{r.microbatch}, {r.region}: {start:.6f}–{end:.6f}ms')
            text(s,x(start)+2,y+23,f'{r.phase[0]}{r.microbatch}',9,'white')
            text(s,x(start)+2,y+6,r.region[0].upper(),8)
            previous=end
    comm=nodes[nodes.kind.eq('pp_p2p') & nodes.pp_lane.eq(0)]
    csv(out,'graph/lane0_communication_nodes.csv',comm)
    csv(out,'graph/lane0_wait_intervals.csv',pd.DataFrame(waits))
    for r in comm.itertuples():
        start=r.predicted_start_ns/1e6;end=r.predicted_end_ns/1e6
        sender=int(r.node_id.split(':s')[-1].split('_')[0]);receiver=int(r.node_id.rsplit('_s',1)[1])
        sy=80+sender*56+38;dy=80+receiver*56+38
        rect(s,x(start),sy,x(end)-x(start),3,PP,f'{r.node_id}; service {r.network_service_ns_model/1e6:.4f}ms + software/wall {r.software_sync_ns_model/1e6:.4f}ms')
        s.append(f'<path d="M{x(end):.2f},{sy} L{x(end):.2f},{dy}" stroke="{PP}" stroke-opacity=".4" fill="none" marker-end="url(#arrow)"><title>{escape(r.node_id)}</title></path>')
    for ms in range(0,25001,2500):
        s.append(f'<path d="M{x(ms)},69 V{H-45}" stroke="#8ca0b8" stroke-dasharray="2 4" stroke-opacity=".45"/>');text(s,x(ms)-10,H-24,ms/1000)
    finish(s,dest/'model_overview.svg')
    # The unit-cost factory is a structural explanatory view, explicitly distinct from milliseconds.
    s=svg(1240,735,'Non-interleaved 1F1B schedule semantics — PP14, MB3',
          'Local program order left to right. Only PP11–13 have steady operations because warmup = min(PP-stage-1, MB). No millisecond costs in this view.')
    for st in range(14):
        y=78+st*43;text(s,20,y+20,f'PP{st}')
        warm=min(13-st,3);remaining=3-warm
        for q,(ph,mb) in enumerate(schedule(st,14,3)):
            reg='warmup' if q<warm else ('steady' if q<warm+2*remaining else 'cooldown')
            xx=100+q*178;rect(s,xx,y,158,29,F if ph=='F' else B,reg)
            text(s,xx+10,y+19,f'{ph}{mb} / {reg}',12,'white')
            if q<5:s.append(f'<path d="M{xx+158},{y+14} h18" stroke="#76568b" marker-end="url(#arrow)"/>')
    text(s,20,710,'All 16 PP lanes follow the same static order. Full autograd/PP and operator dependencies are in graph/edges.csv.gz.',12)
    finish(s,dest/'schedule_semantics.svg')
    # Comparable envelope view: both sides aggregate min start / max finish across 16 ranks.
    grouped=phase.groupby(['pp_stage','phase','microbatch'],as_index=False).agg(start_ns=('start_ns','min'),end_ns=('end_ns','max'))
    for it in WINDOW:
        e=next(r for r in payload['evaluation'] if r['iteration']==it)
        s=svg(1600,14*55+135,f'Prediction and trace-derived phase envelopes — target224 iteration {it}',
              'Solid = model, dashed outline = observed CPU F/B phase envelope (16 ranks per stage). Both include internal communication/runtime; neither is kernel-active time.')
        for st in range(14):
            y=80+st*55;text(s,15,y+23,f'PP{st}');rect(s,95,y,1470,46,'#f4f7fa')
        for r in grouped.itertuples():
            y=80+r.pp_stage*55;start=r.start_ns/1e6;end=r.end_ns/1e6
            rect(s,x(start),y+2,x(end)-x(start),15,F if r.phase=='FWD' else B,f'Model {r.phase}{r.microbatch}');text(s,x(start)+2,y+13,f'{r.phase[0]}{r.microbatch}',8,'white')
        for r in payload['target_trace'][str(it)]['bars']:
            y=80+r['stage']*55;start=r['start_ms']+e['actual_entry_ms'];end=r['end_ms']+e['actual_entry_ms'];c=F if r['phase']=='forward' else B
            rect(s,x(start),y+24,x(end)-x(start),16,'white',f'Observed {r["id"]}: {start:.5f}–{end:.5f}ms',c,'3 2');text(s,x(start)+2,y+36,r['id'].split(':')[-1],8,c)
        for ms in range(0,25001,2500):text(s,x(ms)-10,14*55+112,ms/1000)
        finish(s,dest/f'observed_comparison_{it}.svg')
    # Exact neighborhood of a late-stage backward start; retain native node and edge IDs.
    focus=nodes[nodes.pp_stage.eq(12)&nodes.pp_lane.eq(0)&nodes.microbatch.eq(0)&nodes.op_name.eq('bwd_start')].iloc[0].node_id
    incoming={k:list(g.src) for k,g in edges.groupby('dst',sort=False)}
    depth={focus:0};front=[focus]
    for d in range(1,4):
        next_front=[]
        for n in front:
            for parent in incoming.get(n,[]):
                if parent not in depth:depth[parent]=d;next_front.append(parent)
        front=next_front
    subset=nodes[nodes.node_id.isin(depth)].copy();se=edges[edges.src.isin(depth)&edges.dst.isin(depth)].copy()
    csv(out,'graph/detail_nodes.csv',subset);csv(out,'graph/detail_edges.csv',se)
    positions={};levels={d:sorted([n for n,v in depth.items() if v==d]) for d in set(depth.values())}
    height=max(len(v) for v in levels.values())*95+100
    s=svg(1720,height,'Exact dependency neighborhood — PP12 lane0 B0 start',
          'Three predecessor levels. Orange = critical predecessor edge; grey = another prerequisite. Boxes include inherited cost. No added edges or waits.')
    lookup=subset.set_index('node_id')
    for d,ids in levels.items():
        for q,n in enumerate(ids):positions[n]=(25+(3-d)*425,85+q*95)
    for e in se.itertuples():
        ax,ay=positions[e.src];bx,by=positions[e.dst];critical=lookup.loc[e.dst,'critical_predecessor']==e.src
        s.append(f'<path d="M{ax+390},{ay+32} C{ax+412},{ay+32} {bx-22},{by+32} {bx},{by+32}" stroke="{"#d68820" if critical else "#9aabbc"}" fill="none" marker-end="url(#arrow)"><title>{escape(e.edge_type+": "+str(e.dependency_source))}</title></path>')
    for n,(xx,yy) in positions.items():
        r=lookup.loc[n];rect(s,xx,yy,390,68,'#f1f5fa',n,'#8fa4b8')
        text(s,xx+7,yy+16,n[:57],10);text(s,xx+7,yy+31,n[57:112],10)
        text(s,xx+7,yy+47,f'{r["kind"]}; cost {r.duration_ns/1e6:.6f} ms',10)
        text(s,xx+7,yy+61,f'{r.predicted_start_ns/1e6:.3f}–{r.predicted_end_ns/1e6:.3f} ms',10)
    finish(s,dest/'dependency_detail.svg')
    # Physical cost view for the same phase: show original tokens, not decorative operators at zero cost.
    d=nodes[nodes.pp_stage.eq(12)&nodes.pp_lane.eq(0)&nodes.microbatch.eq(0)&nodes.phase.eq('BWD')&nodes.duration_ns.gt(0)].copy()
    csv(out,'graph/detail_physical_cost_nodes.csv',d)
    cp=d[d.on_critical_path].sort_values('predicted_start_ns').head(24)
    if cp.empty:cp=d.sort_values('predicted_start_ns').head(24)
    csv(out,'graph/detail_cost_window.csv',cp)
    lo=float(cp.predicted_start_ns.min())/1e6;hi=float(cp.predicted_end_ns.max())/1e6
    s=svg(1550,880,'Local physical cost tokens — PP12 lane0 B0',
          'First 24 positive critical tokens (fallback: time order). Component sum equals node duration; parallel rows must not be summed as elapsed time.')
    colors=['#2374ab','#79add0','#8f50aa','#d1a451','#727f8e'];cols=['compute_exposed_ns_model','compute_overlap_ns_model','network_service_ns_model','software_sync_ns_model','framework_residual_ns_model']
    xx=lambda ms:530+(ms-lo)/max(hi-lo,1)*990
    for q,r in enumerate(cp.itertuples()):
        y=85+q*30;text(s,10,y+14,r.node_id[-74:],9)
        at=r.predicted_start_ns/1e6
        for c,color in zip(cols,colors):
            end=at+getattr(r,c)/1e6
            if end>at:rect(s,xx(at),y,xx(end)-xx(at),18,color,c)
            at=end
    text(s,20,835,'Blue exposed compute | light blue overlap token | purple network | gold software/sync | grey framework residual',12)
    text(s,530,860,f'Window {lo:.6f}–{hi:.6f} ms, predicted Profiler origin',12)
    finish(s,dest/'physical_cost_detail.svg')
    options=''.join(f'<option value="observed_comparison_{i}.svg">Observed comparison {i}</option>' for i in WINDOW)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>W37 1F1B evidence explorer</title><style>body{font:16px/1.55 system-ui;margin:24px;color:#20354a}select{font:inherit;padding:8px}object{width:100%;height:1000px;border:1px solid #dce3ea}a{color:#17659d}</style><h1>W37 1F1B 计算图与证据</h1><p>预测、观测、静态语义分别展示。灰色间隙是 F/B 包络的补集；它不是新增等待成本，也不证明 GPU 空闲。图中所有毫秒成本来自既有 256 校准。</p><select id="view"><option value="model_overview.svg">Frozen model / lane0</option><option value="schedule_semantics.svg">Warmup / steady / cooldown</option><option value="dependency_detail.svg">Exact dependency detail</option><option value="physical_cost_detail.svg">Physical costs and overlap</option>'''+options+'''</select><object id="figure" type="image/svg+xml" data="model_overview.svg"></object><p><a href="../graph/nodes.csv.gz">全部节点</a> · <a href="../graph/edges.csv.gz">全部边</a> · <a href="../graph/detail_edges.csv">局部依赖表</a> · <a href="../evaluator_only/iteration_results.csv">逐轮结果</a> · <a href="../evaluator_only/phase_results.csv">分阶段结果</a> · <a href="../prediction_seal.json">预测封存</a></p><script>document.getElementById('view').onchange=e=>document.getElementById('figure').data=e.target.value</script></html>'''
    (dest/'index.html').write_text(page)

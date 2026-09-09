#!/usr/bin/env python3
"""Render review artifacts from sealed local runs; no original input or trace access."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/w37/A'
RUNS={'v686':BASE/'post685-candidate-causal-r3','v687':BASE/'post685-candidate-readiness-r1','v688':BASE/'post685-candidate-scenarios-r1'}
COLORS={'forward':'#168b73','backward':'#de744a'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'post685-delivery-round2')
    for version,run in RUNS.items():p.add_argument('--'+version+'-run',type=Path,default=run)
    a=p.parse_args();out=a.output.resolve()
    for version in RUNS:
        RUNS[version]=getattr(a,version+'_run').resolve();assert RUNS[version].is_relative_to(BASE)
    assert out.is_relative_to(BASE);out.mkdir(parents=True,exist_ok=False)
    for run in RUNS.values():
        manifest=json.loads((run/'run_manifest.json').read_text());assert manifest['exit_code']==0,run
        for f in json.loads((run/'prediction_seal.json').read_text())['files']:assert sha(run/f['path'])==f['sha256']
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=145,bbox_inches='tight');plt.close(fig)
    # Mechanism evidence: the observed receiver entry is only a local validation
    # input here, not an oracle timestamp fed into free-running target predictions.
    local=pd.read_csv(RUNS['v687']/'source_validation/v687_split_full_pp_local_results.csv.gz')
    s=local[local.both_endpoints_single_message&local.direction.eq('B')&local.iteration.isin([85,90,95,100])]
    fig,ax=plt.subplots(figsize=(10,4.5))
    for split,color in [('source_fit','#5476a3'),('source_incremental_validation','#c76327')]:
        x=s[s.split.eq(split)];ax.scatter(x.receiver_minus_sender_ms,x.actual_postpublication_ms,s=8,alpha=.35,label=split.replace('_',' '),color=color)
    x=np.linspace(-250,400,500);ax.plot(x,np.maximum(93.0369,x)-np.maximum(0,x)+5.2470,color='#262626',lw=2,label='v687: max(sender + ready, receiver) + completion')
    ax.axhline(98.03,color='#b71c37',ls='--',label='v686: one post-publication median')
    ax.set(xlim=(-250,400),ylim=(0,120),xlabel='Receiver API entry minus sender API entry (ms)',ylabel='Time remaining after both API entries (ms)',
           title='Source B-message evidence: sender head start hides effective readiness')
    ax.legend(fontsize=8,loc='upper right');fig.text(.12,-.02,'Single-message endpoints only. Latent readiness is not independently identified as GPU or CPU work.',fontsize=9)
    save(fig,'pp_readiness_evidence')
    # Local dependency contract, including max semantics and unverified physical origin.
    fig,ax=plt.subplots(figsize=(12,4.7));ax.set(xlim=(0,12),ylim=(0,4.7));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=color,ec='#506070'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    box(.1,3.1,2.1,.85,'Sender local F/B\nthen API entry','#cce5db');box(2.9,3.1,2.4,.85,'Effective sender ready\nB ~93 ms (source fit)','#e2d5f0')
    box(.1,1.4,2.1,.85,'Receiver local F/B\nthen API entry','#cce5db');box(6,2.25,1.5,.8,'MAX\nboth ready','#ededed')
    box(8,2.25,1.8,.8,'PP completion\nB ~5.25 ms','#d8e8f9');box(10.35,2.25,1.45,.8,'API return\nnext F/B','#fde6c8')
    for xy,xytext in [((2.8,3.52),(2.2,3.52)),((6,2.9),(5.3,3.52)),((6,2.4),(2.2,1.82)),((8,2.65),(7.5,2.65)),((10.35,2.65),(9.8,2.65))]:
        ax.annotate('',xy=xy,xytext=xytext,arrowprops={'arrowstyle':'->','color':'#465365','lw':1.8})
    ax.text(.1,4.4,'v687 independent research dependency contract',fontsize=15,weight='bold')
    ax.text(.1,.65,'Fused APIs wait for BOTH associated messages before returning; readiness may overlap receiver work.',fontsize=10)
    ax.text(.1,.2,'This reallocates measured PP wall time. It does not insert an error-fitted pause or replace the formal topology lock.',fontsize=10)
    save(fig,'causal_contract')
    graph=pd.read_csv(RUNS['v687']/'prediction/v687_split_full/target224_nodes.csv.gz')
    phase=graph[graph.kind.eq('phase_compute_communication_runtime')&graph.pp_lane.eq(0)].copy()
    origin=graph[graph.kind.eq('phase_compute_communication_runtime')].predicted_start_ns.min()
    # Timeline shows blocked API intervals separately from phase wall rectangles.
    waits=[];idx=graph.set_index('node_id')
    for r in graph[graph.kind.eq('api_publication')].itertuples():
        ret=idx.loc[r.action_id+':return'];waits.append(dict(rank=r.rank,pp_stage=r.pp_stage,pp_lane=r.pp_lane,action_id=r.action_id,api_name=r.api_name,
            start_ms=(r.predicted_start_ns-origin)/1e6,end_ms=(ret.predicted_end_ns-origin)/1e6,
            blocked_to_join_ms=(ret.predicted_start_ns-r.predicted_start_ns)/1e6,local_return_ms=ret.duration_ns/1e6,
            interpretation='API wall interval may overlap remote F/B and sender readiness; not additive with PP node durations'))
    pd.DataFrame(waits).to_csv(out/'predicted_api_intervals.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(15,9),sharex=True)
    obs=pd.read_csv(RUNS['v688']/'evaluator_only/target_rank_phase_nodes.csv.gz');o=obs[obs.iteration.eq(95)&obs.pp_lane.eq(0)]
    for ax,title,rows,observed in [(axes[0],'v687 source-only prediction | lane0 | F/B CPU wall + blocked API intervals',phase,False),
                                   (axes[1],'Trace observation | iteration 95 | lane0 | CPU F/B boundaries',o,True)]:
        if not observed:
            for w in waits:
                if w['pp_lane']==0:ax.broken_barh([(w['start_ms']/1000,(w['end_ms']-w['start_ms'])/1000)],(w['pp_stage']-.13,.26),facecolors='#c3c7ce')
        for r in rows.itertuples():
            start=r.start_ms if observed else (r.predicted_start_ns-origin)/1e6
            end=r.end_ms if observed else (r.predicted_end_ns-origin)/1e6
            ax.broken_barh([(start/1000,(end-start)/1000)],(r.pp_stage-.31,.62),facecolors=COLORS[r.phase],edgecolors='white',lw=.35)
            if end-start>250:ax.text((start+end)/2000,r.pp_stage,f'{r.phase[0].upper()}{r.microbatch}',ha='center',va='center',fontsize=7,color='white')
        ax.set(yticks=range(14),ylabel='PP stage',title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
    axes[1].set_xlabel('Seconds relative to first F/B among all ranks');fig.tight_layout();save(fig,'timeline_overview')
    # Show the schedule-region membership without equating different ranks' clocks.
    table=phase[['pp_stage','pp_lane','phase','microbatch','region','duration_ns','predicted_start_ns','predicted_end_ns','node_id']]
    table.to_csv(out/'lane0_phase_region_nodes.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,6))
    for r in phase.itertuples():
        sequence=phase[phase.pp_stage.eq(r.pp_stage)].sort_values('predicted_start_ns').node_id.tolist().index(r.node_id)
        regioncolor={'warmup':'#d9ebf7','steady':'#e2efcb','cooldown':'#f8e0d0'}[r.region]
        ax.add_patch(plt.Rectangle((sequence-.43,r.pp_stage-.4),.86,.8,color=regioncolor));ax.text(sequence,r.pp_stage,f'{r.phase[0].upper()}{r.microbatch}',ha='center',va='center',color=COLORS[r.phase])
    ax.set(xlim=(-.6,5.6),ylim=(13.6,-.6),xticks=range(6),yticks=range(14),xlabel='Local F/B action ordinal (not a global time axis)',ylabel='PP stage',
           title='PP14 / MB3: warmup (blue), steady (green), cooldown (orange)')
    save(fig,'schedule_regions')
    # Same metric, same target iterations. Separate Profiler, training, MFU axes.
    metrics=[];iterations=[];phase_metrics=[]
    selected={'v686':['v685_frozen','v686_split_full'],'v687':['v687_split_full'],'v688':['v688_split_ensemble','v688_full4_ensemble']}
    for version,names in selected.items():
        for file,dest in [('metrics.csv',metrics),('iteration_results.csv',iterations),('phase_metrics.csv',phase_metrics)]:
            d=pd.read_csv(RUNS[version]/'evaluator_only'/file);dest.append(d[d.variant.isin(names)])
    metrics=pd.concat(metrics,ignore_index=True);iterations=pd.concat(iterations,ignore_index=True);phase_metrics=pd.concat(phase_metrics,ignore_index=True)
    metrics.to_csv(out/'version_metrics.csv',index=False);iterations.to_csv(out/'version_iteration_results.csv',index=False);phase_metrics.to_csv(out/'version_phase_metrics.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for variant,g in iterations[iterations.split.eq('development_primary')].groupby('variant',sort=False):
        for ax,col in zip(axes,['profiler_ape_pct','training_ape_pct','mfu_relative_ape_pct']):ax.plot(g.iteration,g[col],marker='o',label=variant)
    for ax,title in zip(axes,['Profiler step APE','Training step APE','MFU relative APE']):ax.set(title=title,xlabel='Iteration',ylabel='Error (%)',xticks=[85,90,95,100]);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.tight_layout();save(fig,'iteration_regression')
    sh=pd.read_csv(RUNS['v688']/'evaluator_only/diagnostic_shapley_components.csv');ledger=pd.read_csv(RUNS['v688']/'evaluator_only/diagnostic_lane0_error_ledger.csv')
    names=['phase_noncommunication_only_ms','phase_communication_only_ms','phase_comm_noncommunication_overlap_ms','phase_gpu_idle_ms']
    means=sh.groupby('component').attributed_delta_ms.mean();values=[means[k] for k in names]+[ledger.remaining_schedule_pp_initial_skew_error_ms.mean()]
    fig,ax=plt.subplots(figsize=(10,4.7));ax.barh(['Non-communication only','Communication only','Compute/comm overlap','No traced GPU activity','Remaining schedule / PP / skew'],values,color=['#188977','#397cae','#729d81','#d29142','#8b8692'])
    ax.axvline(0,color='#444',lw=.8);ax.set(xlabel='Mean contribution to lane0 underprediction (ms)',title='Posthoc target-assisted lane0 attribution | NOT a prediction')
    ax.set_xlim(min(-450,min(values)*1.6),max(values)*1.15)
    for i,v in enumerate(values):ax.text(v+(20 if v>=0 else -20),i,f'{v:.1f}',ha='left' if v>=0 else 'right',va='center')
    fig.text(.12,-.025,'Exact Shapley effects across 16 phase-component subsets; remaining error closes the lane0 ledger. No full-rank claim.',fontsize=9)
    save(fig,'internal_attribution')
    # Inspectable local node/edge slice with PP readiness and fused returns.
    edges=pd.read_csv(RUNS['v687']/'prediction/v687_split_full/target224_edges.csv.gz')
    seeds=set(graph[graph.pp_lane.eq(0)&graph.pp_stage.isin([11,12,13])&graph.microbatch.le(1)].node_id)
    sub=edges[edges.src.isin(seeds)|edges.dst.isin(seeds)];ids=set(sub.src)|set(sub.dst)
    graph[graph.node_id.isin(ids)].to_csv(out/'detail_nodes.csv',index=False);sub.to_csv(out/'detail_edges.csv',index=False)
    # A small browser explorer compares one actual rank with the same modeled rank.
    records=[]
    for r in phase.itertuples():
        rows=obs[obs.pp_lane.eq(0)&obs.pp_stage.eq(r.pp_stage)&obs.phase.eq(r.phase)&obs.microbatch.eq(r.microbatch)&obs.iteration.isin([85,90,95,100])]
        records.append(dict(stage=r.pp_stage,phase=r.phase,mb=r.microbatch,region=r.region,node_id=r.node_id,predicted_duration_ms=r.duration_ns/1e6,
            predicted_start_ms=(r.predicted_start_ns-origin)/1e6,observed=[dict(iteration=int(x.iteration),duration_ms=x.duration_ns/1e6,start_ms=x.start_ms) for x in rows.itertuples()]))
    gallery=''.join(f'<figure><img src="{name}.svg"><figcaption>{caption}</figcaption></figure>' for name,caption in [
        ('pp_readiness_evidence','源侧 PP 就绪机制：观测发布时刻只用于局部检验，自由运行使用模型预测发布。'),
        ('causal_contract','独立候选依赖合同；潜在就绪尚未细分为 CPU/GPU/transport 内部动作。'),
        ('schedule_regions','训练调度推导的 warmup / steady / cooldown。局部顺序轴不是全局时间。'),
        ('timeline_overview','总览：模型预测与 trace 观测分开；灰色 API wall 不能再与 PP 节点相加。'),
        ('iteration_regression','目标四轮同口径开发回归。MFU 使用继承 FLOPs/peak 和 training 时钟。'),
        ('internal_attribution','lane0 事后反事实分账，不是可部署预测、全 rank 误差贡献或盲测。')])
    page='''<!doctype html><meta charset="utf-8"><title>W37 post-v685 1F1B evidence</title><style>body{max-width:1150px;margin:30px auto;font:16px system-ui;color:#24303b;line-height:1.65}figure{margin:30px 0}img{width:100%;background:white}figcaption{color:#586673}select{margin:8px;padding:6px}pre{background:#f1f4f7;padding:18px;overflow:auto}a{color:#2268a7}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccd5dd}</style>
<h1>W37：1F1B v6.8.6–v6.8.8 研究证据</h1><p>正式模型仍为 v6.8.5；新模型属于独立研究候选。v6.8.7 改善源侧 PP 机制与自由运行误差，224 卡整体预测没有改善。所有目标结果均为开发评估。</p>
<p><a href="version_iteration_results.csv">逐迭代结果</a> · <a href="version_phase_metrics.csv">阶段回归</a> · <a href="detail_nodes.csv">局部节点</a> · <a href="detail_edges.csv">局部边</a> · <a href="predicted_api_intervals.csv">API 等待区间</a> · <a href="lane0_phase_region_nodes.csv">调度阶段节点</a></p>
<h2>同 rank / PP / MB / F-B 边界查看器（lane0）</h2><label>PP<select id="stage"></select></label><label>MB<select id="mb"><option>0</option><option>1</option><option>2</option></select></label><label>方向<select id="phase"><option>forward</option><option>backward</option></select></label><pre id="detail"></pre>'''+gallery+'''
<script>const rows=DATA;const stage=document.querySelector('#stage');for(let i=0;i<14;i++)stage.add(new Option(i,i));function show(){const row=rows.find(r=>r.stage===+stage.value&&r.mb===+document.querySelector('#mb').value&&r.phase===document.querySelector('#phase').value);document.querySelector('#detail').textContent=JSON.stringify(row,null,2)}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',show));show();</script>'''
    (out/'index.html').write_text(page.replace('const rows=DATA','const rows='+json.dumps(records)))
    (out/'render_manifest.json').write_text(json.dumps({'status':'SEALED_RUNS_VERIFIED_AND_RENDERED','renderer':str(Path(__file__).resolve()),'renderer_sha256':sha(Path(__file__)),
        'runs':{k:str(v) for k,v in RUNS.items()},'artifacts':[dict(path=str(p.relative_to(out)),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()

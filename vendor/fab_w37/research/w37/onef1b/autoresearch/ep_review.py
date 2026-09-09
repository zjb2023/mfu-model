"""Post-seal figures and acceptance evidence for a completed EP prediction wave."""
import json
from pathlib import Path
import pandas as pd
from guards import checked_stage
from smoke_worker import dump,sha
from worker import csv


def diagnose(out,paths,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    run=Path(plan['sealed_reference']['run_root']);primary=plan['sealed_reference']['variant'];variants=plan['review_variants'];seals=[];iterations=[];phases=[];metrics=[]
    for name in variants:
        for folder in ['models','seals','evaluations']:checked_stage(run/folder/name)
        sealed=json.loads((run/'seals'/name/'prediction_seal.json').read_text())
        for item in sealed['files']:assert sha(run/'models'/name/item['path'])==item['sha256']
        seals.append(dict(variant=name,sha256=sha(run/'seals'/name/'prediction_seal.json')))
        ev=run/'evaluations'/name/'evaluator_only'
        for filename,collection in [('iteration_results.csv',iterations),('phase_results.csv',phases),('metrics.csv',metrics)]:
            d=pd.read_csv(ev/filename);collection.append(d[d.variant.eq(name)|(d.variant.eq('v685_frozen')&(name==variants[0]))])
    it=pd.concat(iterations,ignore_index=True);ph=pd.concat(phases,ignore_index=True);mt=pd.concat(metrics,ignore_index=True)
    csv(out,'version_iteration_results.csv',it);csv(out,'version_phase_results.csv',ph);csv(out,'version_metrics.csv',mt)
    model=run/'models'/primary/'prediction'/primary;span=pd.read_csv(model/'target224_phase_spans.csv.gz')
    origin=span.start_ms.min();span['start_ms']-=origin;span['end_ms']-=origin
    pred=span.groupby(['pp_stage','phase','microbatch','region']).agg(start_ms=('start_ms','min'),end_ms=('end_ms','max')).reset_index()
    obs=pd.read_csv(run/'evaluations'/primary/'evaluator_only/observed_phase_envelopes.csv');obs=obs[obs.iteration.eq(85)].rename(columns={'actual_start_ms':'start_ms','actual_end_ms':'end_ms'})
    regions=pred.set_index(['pp_stage','phase','microbatch']).region.to_dict()
    obs['region']=[regions[(r.pp_stage,r.phase,r.microbatch)] for r in obs.itertuples()]
    csv(out,'overview_model_bars.csv',pred);csv(out,'overview_observed85_bars.csv',obs)
    fig,axes=plt.subplots(2,1,figsize=(15,8),sharex=True)
    colors={'forward':'#188876','backward':'#d57642'};short={'warmup':'W','steady':'S','cooldown':'C'}
    for ax,data,title in [(axes[0],pred,'Model prediction: v6.8.10 CPU EP graph'),(axes[1],obs,'Observed trace: iteration85 CPU phase envelope across16 ranks / stage')]:
        for r in data.itertuples():
            ax.broken_barh([(r.start_ms/1000,(r.end_ms-r.start_ms)/1000)],(r.pp_stage-.36,.72),facecolors=colors[r.phase],edgecolor='white',linewidth=.25)
            ax.text((r.start_ms+r.end_ms)/2000,r.pp_stage,r.phase[0].upper()+str(r.microbatch)+short[r.region],ha='center',va='center',fontsize=6,color='white')
        ax.set_yticks(range(14));ax.invert_yaxis();ax.set_ylabel('PP stage');ax.set_title(title,loc='left')
    axes[-1].set_xlabel('Time since first F/B (seconds); W/S/C are static schedule regions')
    fig.tight_layout();fig.savefig(out/'overview.svg',bbox_inches='tight');fig.savefig(out/'overview.png',dpi=160,bbox_inches='tight');plt.close(fig)
    waits=pd.read_csv(model/'target224_ep_waits.csv.gz')
    detail=waits[waits.pp_stage.eq(6)&waits.phase.eq('backward')&waits.microbatch.eq(0)&waits.execution_layer.eq(0)&waits.semantic_region.eq('ep_recompute_dispatch_wall')&waits.pp_lane.lt(8)].sort_values('pp_lane')
    assert len(detail)==8;csv(out,'local_ep_wait_detail.csv',detail)
    n=pd.read_csv(model/'target224_nodes.csv.gz');e=pd.read_csv(model/'target224_edges.csv.gz')
    ids=set(detail.entry_node)|set(detail.group_node)|set(detail.return_node)
    local=e[e.src.isin(ids)|e.dst.isin(ids)];ids|=set(local.src)|set(local.dst)
    csv(out,'local_ep_nodes.csv',n[n.node_id.isin(ids)]);csv(out,'local_ep_edges.csv',local)
    fig,ax=plt.subplots(figsize=(12,4));origin=int(detail.entry_ns.min())
    for r in detail.itertuples():
        for left,right,color in [(r.entry_ns,r.all_entered_ns,'#9ba8b8'),(r.all_entered_ns,r.first_return_ns,'#d99c3d'),(r.first_return_ns,r.rank_return_ns,'#76589c')]:
            ax.broken_barh([((left-origin)/1e6,(right-left)/1e6)],(r.pp_lane-.32,.64),facecolors=color)
        ax.text((r.rank_return_ns-origin)/1e6+1,r.pp_lane,f'wait {r.wait_for_other_entries_ns/1e6:.3f} / tail {r.rank_tail_ns/1e6:.3f} ms',va='center',fontsize=8)
    ax.set_xlim(-1,(detail.rank_return_ns.max()-origin)/1e6+55);ax.set_yticks(range(8));ax.set_ylabel('PP lane / rank within EP group')
    ax.set_xlabel('Modeled CPU time relative to earliest wrapper entry (ms)');ax.set_title('Local dependency detail: stage6 / B0 / first recompute dispatch')
    ax.legend(handles=[Patch(color=c,label=t) for c,t in [('#9ba8b8','wait for other entries'),('#d99c3d','shared CPU completion interval'),('#76589c','rank return tail')]],fontsize=8,loc='upper right')
    fig.tight_layout();fig.savefig(out/'local_ep_detail.svg',bbox_inches='tight');fig.savefig(out/'local_ep_detail.png',dpi=160,bbox_inches='tight');plt.close(fig)
    ledger=pd.read_csv(run/'models'/primary/'prediction/critical_path_summary.csv');csv(out,'critical_path_summary.csv',ledger)
    observed=ph[ph.variant.eq('v685_frozen')&ph.phase.eq('onef1b')].sort_values('iteration')
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(observed.iteration,observed.actual_ms/1000,'ko-',label='observed target224')
    for name,g in ph[ph.phase.eq('onef1b')].groupby('variant'):
        ax.plot(g.iteration,g.predicted_ms/1000,label=name)
    ax.axvspan(85,100,color='#deeced',alpha=.5);ax.set_xlabel('Iteration (85/90/95/100: primary development;60-80: posthoc)');ax.set_ylabel('1F1B envelope (s)');ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(out/'iteration_regression.svg',bbox_inches='tight');fig.savefig(out/'iteration_regression.png',dpi=160,bbox_inches='tight');plt.close(fig)
    source=pd.read_csv(paths['source256_rank_topology.csv']);target=pd.read_csv(paths['target224_rank_topology.csv'])
    hosts=dict(source_hosts=source.host.nunique(),target_hosts=target.host.nunique(),shared_hosts=len(set(source.host)&set(target.host)),
        target_unseen_hosts=len(set(target.host)-set(source.host)),same_stage_ep_group_host_matches=len(source[['pp_stage','ep_group','host']].drop_duplicates().merge(target[['pp_stage','ep_group','host']].drop_duplicates())),
        source_and_target_ep_groups_each_within_one_host=bool(source.groupby('ep_group').host.nunique().eq(1).all() and target.groupby('ep_group').host.nunique().eq(1).all()))
    dump(out/'host_structure_audit.json',hosts);dump(out/'reviewed_seals.json',seals)
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>W37 T07 EP graph review</title><style>body{max-width:1250px;margin:30px auto;font:16px system-ui;line-height:1.65;color:#24303b}img{width:100%}a{color:#17658a}</style><h1>v6.8.10 CPU EP 内部图</h1><p>目标1F1B MAPE 15.906607%，源增量0.791940%，均未优于v687；正式v685保留。group完成区间包括本地设备排队、计算、通信与runtime，不能称为纯EP网络FCT。所有图来自独立seal之后的复核。</p><h2>模型与观察分开</h2><img src="overview.svg"><h2>八rank入口、共同完成与返回尾部</h2><img src="local_ep_detail.svg"><p>等待由边计算，未新增耗时节点。共同完成区间在图中只有一个节点，多行显示用于对齐rank，不可重复求和。</p><h2>逐迭代回归</h2><img src="iteration_regression.svg"><p><a href="version_iteration_results.csv">逐迭代step/MFU</a> · <a href="version_phase_results.csv">分阶段</a> · <a href="local_ep_nodes.csv">局部节点</a> · <a href="local_ep_edges.csv">局部边</a> · <a href="critical_path_summary.csv">关键路径账本</a></p><p>目标全为已见开发/事后数据；MFU分子及部署flags未独立核验。source和target共享15台主机，13台目标主机无源数据，stage/EP位置上的主机均发生变化。</p>''')
    dump(out/'diagnostic.json',dict(status='SEALED_EP_PREDICTION_REVIEW_PASS',new_prediction=False,used_to_fit_model=False,diagnostic=plan['diagnostic'],
        reviewed_variants=variants,reviewed_seals=seals,host_structure=hosts,raw_trace_scanned=False,
        conclusion='EP CPU refinement improves attribution detail, but median component prediction regresses source and target; do not promote. Investigate covariance-preserving scenarios before host-conditioned transfer.'))

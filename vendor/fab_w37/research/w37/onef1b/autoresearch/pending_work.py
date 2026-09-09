"""Test the pending-backward hypothesis using independent source PP observations."""
import numpy as np
import pandas as pd
from ep_costs import EPCosts,GROUP_KEYS


def pair_dispatch(group,layout):
    """Join forward/backward by logical layer; endpoint shapes are not admitted."""
    rows=group[group.pp_stage.isin([s for s,layers in layout.items() if len(layers)==4])].copy()
    rows['layer_id']=[(layout[int(r.pp_stage)] if r.phase=='forward' else list(reversed(layout[int(r.pp_stage)])))[int(r.execution_layer)] for r in rows.itertuples()]
    keys=['iteration','pp_stage','ep_lane_group','microbatch','layer_id']
    f=rows[rows.semantic_region.eq('ep_dispatch_wall')][keys+['execution_layer','group_completion_ns']]
    b=rows[rows.semantic_region.eq('ep_recompute_dispatch_wall')][keys+['execution_layer','group_completion_ns']]
    paired=b.merge(f,on=keys,suffixes=('_b','_f'),validate='one_to_one')
    assert len(paired)==len(b) and (paired.execution_layer_b+paired.execution_layer_f).eq(3).all()
    paired['has_previous_b_layer']=paired.execution_layer_b.gt(0)
    paired['observed_extra_b_ns']=paired.group_completion_ns_b-paired.group_completion_ns_f
    return paired


def pp_group_proxy(parameters):
    """Predeclared median of the eight PP B readiness delays, source only."""
    return {group:float(np.median([parameters[('B',lane)][0] for lane in range(group*8,(group+1)*8)])) for group in [0,1]}


class PendingBWorkCosts(EPCosts):
    def __init__(self,*args,ppcost,zero_pending_proxy=False,**kwargs):
        super().__init__(*args,**kwargs);self.replacements={}
        proxy=pp_group_proxy(getattr(ppcost,'ep_proxy_ready_parameters',ppcost.ready_parameters))
        for key,old in list(self.group.items()):
            st,ep,phase,mb,execution,semantic=key
            if not (1<=st<=14 and phase=='backward' and semantic=='ep_recompute_dispatch_wall'):continue
            assert 0<=execution<4
            fkey=(st,ep,'forward',mb,3-execution,'ep_dispatch_wall');base=float(self.group[fkey])
            pending=proxy[ep] if execution>0 and not zero_pending_proxy else 0.
            self.group[key]=base+pending
            self.replacements[key]=dict(original_group_cost_ns=float(old),forward_basis_key=str(fkey),forward_basis_cost_ns=base,
                pending_previous_b_proxy_ns=pending,pp_source_fit=','.join(map(str,ppcost.fit)),
                source_pp_lanes=','.join(map(str,range(ep*8,(ep+1)*8))),has_previous_b_layer=execution>0,
                pending_proxy_zeroed=zero_pending_proxy,proxy_statistic='median of eight source PP B effective-readiness delays',
                scope='replaces the same CPU group interval; not added to observed group cost; latent pending-work proxy, not identified GPU compute')
    def value(self,component,row,execution_layer=-1,semantic_region=''):
        value=super().value(component,row,execution_layer,semantic_region)
        if component=='group':
            st,mb=self.mapped(row);key=(st,int(row['ep_lane_group']),row['phase'],mb,execution_layer,semantic_region)
            if key in self.replacements:
                self.bindings[-1].update(self.replacements[key]);self.bindings[-1]['cost_model']='forward_CPU_basis_plus_previous_B_proxy'
                self.last_key='source_pending_B_replacement:'+str(key)
                self.bindings[-1]['parameter_key']=self.last_key
        return value


def diagnose(out,paths,plan):
    from smoke_worker import dump
    from worker import csv
    from pp_semantics import align_observations
    from readiness import ReadinessCosts
    from ep_costs import components
    from ep_model import structures,ANCHOR_COLS
    assert plan['diagnostic_access']=='source_only'
    phase=pd.read_csv(paths['source_pp_trace_events_60_100.csv']);api=pd.read_csv(paths['source_pp_api_events_60_100.csv'])
    aligned,pairs=align_observations(phase,api);a=pd.read_csv(paths['ep_source_ep_rank_anchor_events.csv'],usecols=ANCHOR_COLS)
    rank,group,finish=components(a,phase);layout=structures(paths)['source256'];paired=pair_dispatch(group,layout)
    ppcost=ReadinessCosts(aligned,pairs,[85,90]);proxy=pp_group_proxy(ppcost.ready_parameters)
    paired['source_pp_proxy_ns']=paired.ep_lane_group.map(proxy).where(paired.has_previous_b_layer,0.)
    paired['proxy_error_ms']=(paired.source_pp_proxy_ns-paired.observed_extra_b_ns)/1e6
    paired['scope']='observed F basis isolates extra B interval for diagnosis; full forward prediction checked separately'
    paired['split']=np.where(paired.iteration.isin([85,90]),'source_fit_evidence',np.where(paired.iteration.isin([95,100]),'source_incremental_validation','historical_diagnostic'))
    direct=EPCosts(rank,group,finish,[85,90]);bridge=PendingBWorkCosts(rank,group,finish,[85,90],ppcost=ppcost)
    for model,cost in [('direct_group_median',direct),('pp_pending_bridge',bridge)]:
        paired[model+'_predicted_ms']=[float(cost.group[(r.pp_stage,r.ep_lane_group,'backward',r.microbatch,r.execution_layer_b,'ep_recompute_dispatch_wall')])/1e6 for r in paired.itertuples()]
        paired[model+'_error_ms']=paired[model+'_predicted_ms']-paired.group_completion_ns_b/1e6
    csv(out,'source_logical_layer_pairs.csv.gz',paired)
    rows=[]
    for (label,previous),g in paired.groupby(['split','has_previous_b_layer']):
        for model in ['direct_group_median','pp_pending_bridge']:
            rows.append(dict(split=label,has_previous_b_layer=bool(previous),model=model,calls=len(g),
                observed_group_mean_ms=float(g.group_completion_ns_b.mean()/1e6),observed_extra_b_mean_ms=float(g.observed_extra_b_ns.mean()/1e6),
                pp_proxy_mean_ms=float(g.source_pp_proxy_ns.mean()/1e6),proxy_extra_mae_ms=float(g.proxy_error_ms.abs().mean()),
                full_group_mae_ms=float(g[model+'_error_ms'].abs().mean()),full_group_bias_ms=float(g[model+'_error_ms'].mean())))
    metrics=pd.DataFrame(rows);csv(out,'source_pending_bridge_local_metrics.csv',metrics)
    csv(out,'source_PP_readiness_parameters.csv',pd.DataFrame(ppcost.parameter_rows))
    csv(out,'source_group_cost_replacements.csv',pd.DataFrame([dict(zip(GROUP_KEYS,k))|v for k,v in bridge.replacements.items()]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label,g in paired[paired.iteration.isin([85,90,95,100])].groupby('split'):
        means=g.groupby('execution_layer_b').observed_extra_b_ns.mean()/1e6
        axes[0].plot(means.index,means.values,'o-',label=label)
    axes[0].plot([0,1,2,3],[0]+[np.mean(list(proxy.values()))/1e6]*3,'k--',label='source PP readiness proxy')
    axes[0].set_xlabel('B execution ordinal (logical F layer matched)');axes[0].set_ylabel('B recompute minus F dispatch CPU interval (ms)');axes[0].legend(fontsize=7)
    val=metrics[metrics.split.eq('source_incremental_validation')].pivot(index='has_previous_b_layer',columns='model',values='full_group_mae_ms')
    val.plot.bar(ax=axes[1]);axes[1].set_xlabel('Has previous B layer');axes[1].set_ylabel('Full CPU group prediction MAE (ms)');axes[1].tick_params(axis='x',labelrotation=0);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'pending_work_crosscheck.svg',bbox_inches='tight');fig.savefig(out/'pending_work_crosscheck.png',dpi=160,bbox_inches='tight');plt.close(fig)
    dump(out/'diagnostic.json',dict(status='SOURCE_PENDING_B_CROSSCHECK_COMPLETE',new_prediction=False,new_target_timing_read=False,
        target_static_layout_checked=True,source_fit=[85,90],source_incremental_validation=[95,100],logical_layer_pairing='F ordinal + B ordinal == 3, same layer_id; four-layer interior stages only',
        pp_group_proxy_ns=proxy,raw_trace_scanned=False,used_to_fit_model=False,
        cost_rule='B recompute CPU completion := fitted F dispatch at same logical layer + source PP B readiness proxy if a previous B layer exists',
        caution='proxy association and code-consistent exposure, not independently identified GPU computation or unique cause; first observed-F comparison is diagnostic only',
        next='Review pure full-group prediction MAE and source evidence before free-running candidate; keep source-only PP parameter origin and replace, never add to old group wall.'))

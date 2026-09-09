"""Source-only evidence about the meaning and stability of CPU EP completion."""
from time import perf_counter
import json
import numpy as np
import pandas as pd
from smoke_worker import dump
from worker import csv

KEY=['pp_stage','ep_group','phase','microbatch','execution_layer','semantic_region']
SYNC=['stream_synchronize_union_ns','device_synchronize_union_ns','event_synchronize_union_ns']


def split(iterations):
    return np.where(iterations.isin([85,90]),'source_fit_evidence',np.where(iterations.isin([95,100]),'source_incremental_validation','historical_diagnostic'))


def diagnose(out,paths,plan):
    assert plan['diagnostic_access']=='source_only'
    begin=perf_counter();access=json.loads(paths['ep_input_access_audit.json'].read_text())
    assert access['target_profiler_or_training_timing_read'] is False
    cols=['iteration','rank','pp_lane','layer_id','arrival_ns','completion_ns','captured_anchor_wall_ns']+KEY+SYNC
    a=pd.read_csv(paths['ep_source_ep_rank_anchor_events.csv'],usecols=cols)
    assert len(a)==203904 and a.completion_ns.sub(a.arrival_ns).eq(a.captured_anchor_wall_ns).all()
    for col in SYNC:assert a[col].ge(0).all() and a[col].le(a.captured_anchor_wall_ns).all()
    # These are independently clipped unions over all matching runtime events,
    # without thread/correlation filtering in the pinned extractor. The sum is
    # not an approved disjoint partition and is never added as a graph cost.
    sync_check=dict(rank_wrappers=len(a),each_sync_union_within_wrapper=True,
        sum_of_three_unions_exceeds_wrapper_count=int(a[SYNC].sum(axis=1).gt(a.captured_anchor_wall_ns).sum()),
        scope='each named runtime union intersected with CPU wrapper wall; no thread or CPU/GPU correlation guarantee; do not add unions')
    keys=['iteration']+KEY
    agg=dict(first_entry_ns=('arrival_ns','min'),last_entry_ns=('arrival_ns','max'),first_return_ns=('completion_ns','min'),last_return_ns=('completion_ns','max'),ranks=('rank','nunique'))
    for col in SYNC:agg[col+'_rank_max']=(col,'max')
    group=a.groupby(keys).agg(**agg).reset_index();assert group.ranks.eq(8).all()
    group['completion_after_last_entry_ns']=group.first_return_ns-group.last_entry_ns
    group['entry_spread_ns']=group.last_entry_ns-group.first_entry_ns
    group['elapsed_to_first_return_ns']=group.first_return_ns-group.first_entry_ns
    group['rank_tail_spread_ns']=group.last_return_ns-group.first_return_ns
    assert group.completion_after_last_entry_ns.ge(0).all()
    for column,operation,label in [('arrival_ns','idxmax','last_entry_rank'),('completion_ns','idxmin','first_return_rank')]:
        idx=getattr(a.groupby(keys)[column],operation)()
        picked=a.loc[idx,keys+['rank']].rename(columns={'rank':label});group=group.merge(picked,on=keys,validate='one_to_one')
    ranks=pd.read_csv(paths['source256_rank_topology.csv']);host=ranks.groupby(['pp_stage','ep_group']).host.agg(['first','nunique']).reset_index()
    assert host['nunique'].eq(1).all();group=group.merge(host.rename(columns={'first':'host'})[['pp_stage','ep_group','host']],on=['pp_stage','ep_group'],validate='many_to_one')
    group['stage_role']=np.where(group.pp_stage.eq(0),'first',np.where(group.pp_stage.eq(15),'last','interior'))
    group['execution_position']=np.where(group.execution_layer.eq(0),'first_executed_moe_layer','later_executed_moe_layer')
    group['microbatch_role']=np.where(group.microbatch.eq(0),'first',np.where(group.microbatch.eq(3),'last','middle'))
    group['split']=split(group.iteration)
    for col in ['completion_after_last_entry_ns','entry_spread_ns','elapsed_to_first_return_ns','rank_tail_spread_ns']+[c+'_rank_max' for c in SYNC]:
        group[col.replace('_ns','_ms')]=group[col]/1e6
    csv(out,'source_ep_completion_features.csv.gz',group)
    ms=['completion_after_last_entry_ms','entry_spread_ms','elapsed_to_first_return_ms','rank_tail_spread_ms']+[c.replace('_ns','_ms')+'_rank_max' for c in SYNC]
    profile=group.groupby(['split','stage_role','phase','semantic_region','execution_position','microbatch_role'])[ms].agg(['median','mean','size'])
    profile.columns=['_'.join(c) for c in profile.columns];csv(out,'source_position_profile.csv',profile.reset_index())
    # Evaluate the existing per-key median CPU completion model on held iterations.
    fitted=group[group.iteration.isin([85,90])].groupby(KEY).completion_after_last_entry_ms.median().rename('predicted_completion_ms').reset_index()
    joined=group.merge(fitted,on=KEY,validate='many_to_one');joined['error_ms']=joined.predicted_completion_ms-joined.completion_after_last_entry_ms
    csv(out,'source_completion_local_validation.csv.gz',joined[['iteration','split']+KEY+['completion_after_last_entry_ms','predicted_completion_ms','error_ms']])
    local=joined.groupby(['split','stage_role','phase','semantic_region']).agg(calls=('error_ms','size'),mae_ms=('error_ms',lambda v:v.abs().mean()),bias_ms=('error_ms','mean')).reset_index()
    csv(out,'source_completion_local_metrics.csv',local)
    paired=[];correlations=[]
    for first,last,label in [(85,90,'source_fit_85_to90'),(95,100,'source_incremental_95_to100')]:
        columns=KEY+['host','completion_after_last_entry_ms','entry_spread_ms','elapsed_to_first_return_ms','last_entry_rank','first_return_rank']
        p=group[group.iteration.eq(first)][columns].merge(group[group.iteration.eq(last)][columns],on=KEY,suffixes=('_first','_last'),validate='one_to_one')
        p['pair']=label
        for col in ['completion_after_last_entry_ms','entry_spread_ms','elapsed_to_first_return_ms']:p['delta_'+col]=p[col+'_last']-p[col+'_first']
        p['last_entry_rank_changed']=p.last_entry_rank_first.ne(p.last_entry_rank_last)
        p['first_return_rank_changed']=p.first_return_rank_first.ne(p.first_return_rank_last)
        assert (p.delta_completion_after_last_entry_ms+p.delta_entry_spread_ms-p.delta_elapsed_to_first_return_ms).abs().max()<1e-9
        paired.append(p)
        for (phase,semantic),g in p.groupby(['phase','semantic_region']):
            correlations.append(dict(pair=label,phase=phase,semantic_region=semantic,calls=len(g),
                completion_vs_spread_delta_corr=g.delta_completion_after_last_entry_ms.corr(g.delta_entry_spread_ms),
                elapsed_vs_spread_delta_corr=g.delta_elapsed_to_first_return_ms.corr(g.delta_entry_spread_ms),
                last_entry_rank_change_fraction=float(g.last_entry_rank_changed.mean()),first_return_rank_change_fraction=float(g.first_return_rank_changed.mean()),
                median_abs_completion_delta_ms=float(g.delta_completion_after_last_entry_ms.abs().median()),
                caution='completion=elapsed-entry_spread; correlation partly shares timestamp terms, not causal readiness evidence'))
    pairs=pd.concat(paired,ignore_index=True);csv(out,'source_matched_iteration_deltas.csv.gz',pairs);csv(out,'source_delta_correlations.csv',pd.DataFrame(correlations))
    # Within-stage comparison removes layer/stage differences but does not
    # distinguish host hardware, data load, CP rank or other group conditions.
    group['ep_lane_group']=group.ep_group%2
    pk=['iteration','pp_stage','phase','microbatch','execution_layer','semantic_region']
    columns=pk+['host','completion_after_last_entry_ms']
    paired_host=group[group.ep_lane_group.eq(0)][columns].merge(group[group.ep_lane_group.eq(1)][columns],on=pk,suffixes=('_g0','_g1'),validate='one_to_one')
    paired_host['g1_minus_g0_ms']=paired_host.completion_after_last_entry_ms_g1-paired_host.completion_after_last_entry_ms_g0
    paired_host['split']=split(paired_host.iteration);csv(out,'source_within_stage_group_pairs.csv.gz',paired_host)
    hkeys=['pp_stage','phase','semantic_region','host_g0','host_g1']
    by=paired_host.groupby(['split']+hkeys).g1_minus_g0_ms.mean().reset_index()
    fit=by[by.split.eq('source_fit_evidence')].drop(columns='split');val=by[by.split.eq('source_incremental_validation')].drop(columns='split')
    stable=fit.merge(val,on=hkeys,suffixes=('_fit','_validation'),validate='one_to_one')
    stable['same_sign']=np.sign(stable.g1_minus_g0_ms_fit).eq(np.sign(stable.g1_minus_g0_ms_validation))
    stable['delta_ms']=stable.g1_minus_g0_ms_validation-stable.g1_minus_g0_ms_fit
    csv(out,'source_within_stage_group_stability.csv',stable)
    stability=[]
    for (phase,semantic),g in stable.groupby(['phase','semantic_region']):
        stability.append(dict(phase=phase,semantic_region=semantic,stage_pairs=len(g),fit_validation_corr=g.g1_minus_g0_ms_fit.corr(g.g1_minus_g0_ms_validation),
            same_sign_fraction=float(g.same_sign.mean()),median_abs_fit_difference_ms=float(g.g1_minus_g0_ms_fit.abs().median()),
            validation_difference_mae_ms=float(g.delta_ms.abs().mean()),scope='within-stage EP-group contrast; host vs group/data/CP effects confounded'))
    csv(out,'source_group_stability_metrics.csv',pd.DataFrame(stability))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    g=group[group.stage_role.eq('interior')&group.iteration.isin([85,90,95,100])]
    bars=g.groupby(['split','semantic_region','execution_position']).completion_after_last_entry_ms.mean().unstack(['split','execution_position'])
    fig,ax=plt.subplots(figsize=(12,5));bars.plot.bar(ax=ax);ax.set_ylabel('CPU first completion after last entry (ms)');ax.set_xlabel('Wrapper semantic (interior stages)')
    ax.tick_params(axis='x',labelrotation=18);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'source_completion_positions.svg',bbox_inches='tight');fig.savefig(out/'source_completion_positions.png',dpi=150,bbox_inches='tight');plt.close(fig)
    extractor=paths['ep_extract_dag_v53_ep_group_calibration.py'];(out/'extractor_evidence.py').write_text(extractor.read_text())
    dump(out/'runtime_union_scope.json',sync_check)
    dump(out/'diagnostic.json',dict(status='SOURCE_EP_COMPLETION_EVIDENCE_PASS',new_prediction=False,new_target_timing_read=False,used_to_fit_model=False,
        source_rank_rows=len(a),source_group_rows=len(group),source_dataframe_bytes=int(a.memory_usage(deep=True).sum()),analysis_seconds=perf_counter()-begin,
        raw_trace_scanned=False,source_calibration_evidence=[85,90],incremental_validation=[95,100],historical_diagnostic=[60,65,70,75,80],
        runtime_union_scope=sync_check,identification_limits=['runtime unions have no thread/correlation filter','entry-spread relation has shared timestamp algebra','host and EP/CP/data effects not independently separated'],
        next='Read source position/local-error/group-stability evidence before selecting a readiness or host-transfer predictor. No target correction or topology promotion authorized by correlations alone.'))

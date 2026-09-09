"""Snakemake source preparation and sealed candidate assembly for CPU EP costs."""
import json
import numpy as np
import pandas as pd
from smoke_worker import dump,sha
from worker import csv
from pp_graph import ObservedCosts
from readiness import ReadinessCosts
from candidate import baseline,source_scores,partition_prediction,KEY
from ep_costs import components,EPCosts,RANK_KEYS,GROUP_KEYS,PHASE_KEYS
import ep_predict

ANCHOR_COLS=['iteration','rank','pp_stage','pp_lane','ep_group','phase','microbatch','execution_layer','layer_id','semantic_region','arrival_ns','completion_ns']


def structures(paths):
    result={}
    for case,pp in [('source256',16),('target224',14)]:
        ranks=pd.read_csv(paths[case+'_rank_topology.csv']);layers=pd.read_csv(paths[case+'_layer_stage_map.csv'])
        assert sorted(ranks['rank'])==list(range(pp*16))
        assert ranks['rank'].eq(ranks.pp_stage*16+ranks.pp_lane).all()
        assert ranks.ep_group.eq(ranks.pp_stage*2+ranks.pp_lane//8).all()
        assert ranks.groupby('ep_group').size().eq(8).all()
        counts=layers.groupby('pp_stage').size();assert counts.to_list()==[2]+[4]*(pp-2)+[2]
        assert sorted(layers.layer_id)==list(range(4*pp-4))
        result[case]={int(st):g.sort_values('stage_local_layer').layer_id.astype(int).tolist() for st,g in layers.groupby('pp_stage')}
    return result


def prepare_source(out,paths,aligned,pairs,phase):
    # Raw observation admission was established by T05/T06; verify the source
    # provenance again and keep historically fitted parameter tables excluded.
    access=json.loads(paths['ep_input_access_audit.json'].read_text())
    assert access['target_profiler_or_training_timing_read'] is False
    for item in access['structured_inputs']:
        if item['path'] in [str(paths[k]) for k in ['ep_dag_v53_ep_group_2026w36.toml','source256_rank_topology.csv','source256_layer_stage_map.csv']]:
            assert sha(item['path'])==item['sha256']
    layout=structures(paths);dump(out/'ep_static_structure.json',layout)
    a=pd.read_csv(paths['ep_source_ep_rank_anchor_events.csv'],usecols=ANCHOR_COLS)
    rank,group,finish=components(a,phase)
    csv(out,'source_ep_rank_components.csv.gz',rank);csv(out,'source_ep_group_components.csv.gz',group);csv(out,'source_ep_finish_components.csv.gz',finish)
    # General factory must reproduce the already verified T06 topology and every
    # boundary at85 before fitted use; this is an observed-cost check, not a score.
    cost=EPCosts(rank,group,finish,[85]);ppcost=ObservedCosts(aligned,pairs,85)
    n,e,b,h=ep_predict.build(16,4,ppcost,cost,layout['source256']);by=n.set_index('node_id')
    assert h=='8d3e85b013bfb268507641d17d0625d1db8bb381d37e9340bd8bddbc5b46b2cf'
    obs=aligned[aligned.iteration.eq(85)]
    start=obs.apply(lambda r:r.action_id+(':enter' if r.kind=='phase' else ':post'),axis=1).map(by.predicted_start_ns)
    end=obs.apply(lambda r:r.action_id+('' if r.kind=='phase' else ':return'),axis=1).map(by.predicted_end_ns)
    se=int((start-(obs.start_ns-ppcost.origin)).abs().max());ee=int((end-(obs.end_ns-ppcost.origin)).abs().max())
    join=b.merge(a[a.iteration.eq(85)],on=['rank','pp_stage','pp_lane','phase','microbatch','execution_layer','layer_id','semantic_region'],validate='one_to_one')
    be=int((join.entry_node.map(by.predicted_start_ns)-(join.arrival_ns-ppcost.origin)).abs().max())
    re=int((join.return_node.map(by.predicted_end_ns)-(join.completion_ns-ppcost.origin)).abs().max())
    assert max(se,ee,be,re)==0
    dump(out/'ep_observed_factory_check.json',dict(status='PASS',source85_pp_phase_start_error_ns=se,source85_pp_phase_end_error_ns=ee,
        source85_ep_entry_error_ns=be,source85_ep_return_error_ns=re,matched_T06_topology=h,
        source_and_target_static_groups='two EP8 groups in each PP16-lane stage; endpoint2/interior4 layers',historical_fitted_costs_used=False))


def local_scores(out,rank,group,finish,cost,folder='source_validation'):
    frames=[]
    for component,data,keys,col,table in [('local_before_wrapper',rank,RANK_KEYS,'local_before_wrapper_ns',cost.local),
        ('rank_return_tail',rank,RANK_KEYS,'rank_return_tail_ns',cost.tail),('group_completion',group,GROUP_KEYS,'group_completion_ns',cost.group),
        ('local_finish',finish,PHASE_KEYS,'local_finish_ns',cost.finish)]:
        z=data[['iteration']+keys+[col]].copy();z['predicted_ms']=[float(table[tuple(v)])/1e6 for v in z[keys].itertuples(index=False,name=None)]
        if component=='rank_return_tail' and cost.zero_rank_tail:z['predicted_ms']=0.
        z['actual_ms']=z[col]/1e6;z['error_ms']=z.predicted_ms-z.actual_ms;z['component']=component
        z['split']=np.where(z.iteration.isin(cost.fit),'source_fit',np.where(z.iteration.isin([95,100]),'source_incremental_validation','historical_diagnostic'))
        frames.append(z.drop(columns=col))
    result=pd.concat(frames,ignore_index=True);csv(out,folder+'/ep_local_components.csv.gz',result)
    csv(out,folder+'/ep_local_metrics.csv',result.groupby(['split','phase','component']).agg(samples=('error_ms','size'),
        mae_ms=('error_ms',lambda v:v.abs().mean()),bias_ms=('error_ms','mean')).reset_index())


class ProportionView:
    """Diagnostic GPU proportions normalized to a modeled CPU span; not a cost."""
    def __init__(self,cost,frame):
        self.fit=cost.fit;self.role_transfer=cost.role_transfer;self.mapped_stage=cost.mapped_stage
        self.values=frame.set_index(KEY).duration_ns.to_dict()
    def phase(self,a):return self.values[(a['pp_stage'],a['pp_lane'],a['name'],a['microbatch'])]


def run_candidate(out,run,name,spec,paths):
    data=run/'source_audit'
    from guards import checked_stage
    checked_stage(data)
    aligned=pd.read_csv(data/'source_actions.csv.gz');pairs=pd.read_csv(data/'source_message_pairs.csv.gz');phase=pd.read_csv(data/'source_phases.csv.gz')
    rank=pd.read_csv(data/'source_ep_rank_components.csv.gz');group=pd.read_csv(data/'source_ep_group_components.csv.gz');finish=pd.read_csv(data/'source_ep_finish_components.csv.gz')
    gpu=pd.read_csv(data/'source_gpu_partition.csv');wrapper=pd.read_csv(data/'source_wrapper_partition.csv')
    layout={case:{int(k):v for k,v in rows.items()} for case,rows in json.loads((data/'ep_static_structure.json').read_text()).items()}
    br,bp,contract=baseline(paths);records=[br];phases=[bp];parts=[];ledger=[];sourcegraphs={};fit=spec['source_fit_iterations']
    parent_name=name;parent_fit=list(fit);scenario=spec['method']=='ep_cpu_scenarios'
    cp_tails=None
    if spec.get('pp_cost_model')=='cp_anchored_readiness':
        from cp_readiness import read_cp_tails
        cp_tails,missing=read_cp_tails(paths,pairs)
        csv(out,'source_validation/CP_coverage_exclusions.csv',missing)
    members=[(name+'_source'+str(it),[it]) for it in fit] if scenario else [(name,fit)]
    for name,fit in members:
        folder='source_validation/'+name if scenario else 'source_validation'
        for pp,mb,case in [(16,4,'source256'),(14,3,'target224')]:
            ppfit=spec.get('pp_fit_iterations',fit)
            if cp_tails is not None:
                from cp_readiness import CPReadinessCosts
                ppcost=CPReadinessCosts(aligned,pairs,ppfit,pp,mb,role_transfer=True,cp_tails=cp_tails,
                    zero_post_CP_remainder=spec.get('zero_post_CP_remainder',False))
            else:ppcost=ReadinessCosts(aligned,pairs,ppfit,pp,mb,role_transfer=True)
            kwargs=dict(statistic=spec.get('statistic','median'),zero_rank_tail=spec.get('zero_rank_tail',False))
            if spec.get('ep_cost_model')=='pp_pending_bridge':
                from pending_work import PendingBWorkCosts
                epcost=PendingBWorkCosts(rank,group,finish,fit,pp,mb,ppcost=ppcost,zero_pending_proxy=spec.get('zero_pending_proxy',False),**kwargs)
            else:epcost=EPCosts(rank,group,finish,fit,pp,mb,**kwargs)
            n,e,b,h=ep_predict.build(pp,mb,ppcost,epcost,layout[case]);span=ep_predict.spans(n);env=ep_predict.envelope(n)
            csv(out,f'prediction/{name}/{case}_nodes.csv.gz',n);csv(out,f'prediction/{name}/{case}_edges.csv.gz',e)
            csv(out,f'prediction/{name}/{case}_ep_bindings.csv.gz',b);csv(out,f'prediction/{name}/{case}_ep_cost_bindings.csv.gz',pd.DataFrame(epcost.bindings))
            if hasattr(epcost,'replacements'):
                csv(out,f'prediction/{name}/{case}_source_group_parameter_decomposition.csv',pd.DataFrame([dict(zip(GROUP_KEYS,k))|v for k,v in epcost.replacements.items()]))
            csv(out,f'prediction/{name}/{case}_phase_spans.csv.gz',span);csv(out,f'prediction/{name}/{case}_ep_waits.csv.gz',ep_predict.wait_table(n,b))
            csv(out,f'prediction/{name}/{case}_readiness_parameters.csv',pd.DataFrame(ppcost.parameter_rows))
            if cp_tails is not None:
                csv(out,f'prediction/{name}/{case}_CP_offset_parameters.csv',ppcost.cp_model.basis)
                csv(out,f'prediction/{name}/{case}_CP_ready_parameters.csv',pd.DataFrame(ppcost.cp_parameter_rows))
                csv(out,f'prediction/{name}/{case}_CP_ready_bindings.csv',pd.DataFrame(ppcost.cp_bindings))
                csv(out,f'prediction/{name}/{case}_frozen_EP_proxy_PP_parameters.csv',pd.DataFrame(ppcost.baseline_parameter_rows))
            dump(out/f'prediction/{name}/{case}_contract.json',dict(variant=name,case=case,spec=spec,source_fit=fit,pp_source_fit=ppfit,envelope=env,
                topology_sha256=h,node_count=len(n),edge_count=len(e),formal_topology_replaced=False,phase_parent_cost_added=False,
                target_runtime_semantic_status='source-supported CPU wrapper completion candidate; target deployment flags still require review',
                physical_scope='CPU intervals contain device queueing, computation, collectives and runtime; not a unique GPU or pure-network decomposition'))
            ledger+=ep_predict.critical_ledger(n,name,case)
            part=partition_prediction(gpu,wrapper,ProportionView(ppcost,span),pp,mb,name)
            for row in part:row['scope']='diagnostic source lane0 proportions normalized to modeled CPU span; EP CPU ledger is a separate view, not additional time'
            parts+=part
            if pp==16:
                sourcegraphs[name]=(ep_predict.metrics_view(n),fit);local_scores(out,rank,group,finish,epcost,folder=folder)
                csv(out,folder+'/pp_local_results.csv.gz',ppcost.local_validation(pairs))
            else:
                span['variant']=name;span['start_ms']-=env['first_phase_ns']/1e6;span['end_ms']-=env['first_phase_ns']/1e6;phases.append(span)
                rec={**br,'variant':name,'onef1b_ms':env['onef1b_ms']};rec['profiler_ms']=rec['entry_ms']+rec['onef1b_ms']+rec['tail_ms'];rec['training_ms']=rec['profiler_ms']+rec['outer_ms'];records.append(rec)
    source_scores(out,sourcegraphs,aligned,phase,pd.read_csv(paths['source256_profiler_entry_60_100.csv']))
    if scenario:
        from ep_scenarios import expected_step,expected_source,expected_phases,expected_partition
        member_names=[n for n,_ in members]
        records.append(expected_step(records,member_names,parent_name))
        phases.append(expected_phases(phases,member_names,parent_name))
        parts+=expected_partition(parts,member_names,parent_name)
        src=pd.read_csv(out/'source_validation/iteration_results.csv')
        extra=expected_source(src,member_names,parent_name,parent_fit)
        # A sample used by another scenario is not an independent validation sample.
        for member,member_fit in members:
            mask=src.variant.eq(member)&src.iteration.isin(set(parent_fit)-set(member_fit))
            src.loc[mask,'split']='source_other_fit_scenario_diagnostic'
        src=pd.concat([src,extra],ignore_index=True);csv(out,'source_validation/iteration_results.csv',src)
        csv(out,'source_validation/metrics.csv',src.groupby(['variant','split']).agg(count=('iteration','size'),
            onef1b_mape_pct=('ape_pct','mean'),bias_ms=('error_ms','mean')).reset_index())
        timeline=pd.read_csv(out/'source_validation/phase_results.csv.gz')
        keys=['iteration']+KEY
        expected=timeline.groupby(keys)[['actual_duration_ms','duration_ms','actual_start_ms','predicted_start_ms']].mean().reset_index()
        expected['variant']=parent_name;expected['duration_error_ms']=expected.duration_ms-expected.actual_duration_ms
        expected['start_error_ms']=expected.predicted_start_ms-expected.actual_start_ms
        expected['split']=np.where(expected.iteration.isin(parent_fit),'source_fit',np.where(expected.iteration.isin([95,100]),'source_incremental_validation','historical_diagnostic'))
        for member,member_fit in members:
            timeline.loc[timeline.variant.eq(member)&timeline.iteration.isin(set(parent_fit)-set(member_fit)),'split']='source_other_fit_scenario_diagnostic'
        csv(out,'source_validation/phase_results.csv.gz',pd.concat([timeline,expected],ignore_index=True))
        spans=[]
        for case in ['source256','target224']:
            values=[]
            for member in member_names:
                c=json.loads((out/f'prediction/{member}/{case}_contract.json').read_text());values.append(c['envelope']['onef1b_ms'])
            spans.append(dict(variant=parent_name,case=case,mean_onef1b_ms=float(np.mean(values)),scenario_min_ms=min(values),scenario_max_ms=max(values),
                members=','.join(member_names),interpretation='two exposed source-iteration scenarios; empirical span, NOT a prediction interval'))
        csv(out,'prediction/scenario_spans.csv',pd.DataFrame(spans))
        dump(out/'prediction/scenario_contract.json',dict(variant=parent_name,source_fit=parent_fit,members=member_names,
            weights='equal, source only',mean_of_replays_not_replay_of_mean_costs=True,phase_parent_cost_added=False,
            stage_envelope_aggregation='mean of per-member cross-rank envelopes; never envelope the mean rank timestamps',
            critical_path_ledger='member paths only; expected contributions in expected_critical_path_summary.csv do not imply a single replay path',
            physical_scope='CPU interval decomposition; group cost is not pure network service',unseen_regime_coverage='not established by two source samples'))
        expected_ledger=pd.DataFrame(ledger).groupby(['case','kind']).critical_contribution_ms.sum().div(len(members)).reset_index()
        expected_ledger['variant']=parent_name;csv(out,'prediction/expected_critical_path_summary.csv',expected_ledger)
    csv(out,'prediction/phase_nodes.csv.gz',pd.concat(phases,ignore_index=True)[['variant']+KEY+['start_ms','end_ms','duration_ms']])
    csv(out,'prediction/internal_partition.csv',pd.DataFrame(parts));csv(out,'prediction/critical_path_ledger.csv',pd.DataFrame(ledger))
    csv(out,'prediction/critical_path_summary.csv',pd.DataFrame(ledger).groupby(['variant','case','kind']).critical_contribution_ms.sum().reset_index())
    dump(out/'prediction/step_predictions.json',records);dump(out/'prediction/frozen_baseline_contract.json',contract)

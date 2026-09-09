"""v686 source-only causal PP candidate and sealed development evaluation."""
import json
import numpy as np
import pandas as pd
from pp_graph import build,ObservedCosts,FittedCosts,envelope,MAP
from readiness import ReadinessCosts
from pp_semantics import all_actions
from worker import csv
from smoke_worker import dump,sha

KEY=['pp_stage','pp_lane','phase','microbatch']
GPU=['phase_noncommunication_only_ms','phase_communication_only_ms','phase_comm_noncommunication_overlap_ms','phase_gpu_idle_ms']
WRAP=['fused_prelaunch_sum_ms','fused_postlaunch_sum_ms','outside_fused_wrapper_ms']
WINDOW=[85,90,95,100]


def phase_frame(n):
    p=n[n.kind.eq('phase_compute_communication_runtime')].copy()
    p['start_ms']=p.predicted_start_ns/1e6;p['end_ms']=p.predicted_end_ns/1e6
    p['duration_ms']=p.duration_ns/1e6
    return p


def critical_ledger(n,variant,case):
    by=n.set_index('node_id');ph=phase_frame(n);lo=int(ph.predicted_start_ns.min());last=ph.loc[ph.predicted_end_ns.idxmax(),'node_id']
    rows=[]
    while last:
        r=by.loc[last];ns=max(0,int(r.predicted_end_ns)-max(lo,int(r.predicted_start_ns)))
        if ns:rows.append(dict(variant=variant,case=case,node_id=last,kind=r.kind,pp_stage=int(r.pp_stage),pp_lane=int(r.pp_lane),
                               phase=r.phase,microbatch=int(r.microbatch),critical_contribution_ms=ns/1e6))
        last=r.critical_predecessor
    assert abs(sum(r['critical_contribution_ms'] for r in rows)-envelope(n)['onef1b_ms'])<1e-7
    return rows


def source_oracle(out,aligned,pairs):
    checks=[]
    for it in sorted(aligned.iteration.unique()):
        cost=ObservedCosts(aligned,pairs,it);n,e,h=build(16,4,cost);idx=n.set_index('node_id')
        a=aligned[aligned.iteration.eq(it)]
        start=a.action_id.where(a.kind.eq('phase'),a.action_id+':post').map(idx.predicted_start_ns).astype('int64')
        end=a.action_id.where(a.kind.eq('phase'),a.action_id+':return').map(idx.predicted_end_ns).astype('int64')
        se=(start-(a.start_ns-cost.origin)).abs();ee=(end-(a.end_ns-cost.origin)).abs()
        checks.append(dict(iteration=int(it),actions=len(a),max_start_error_ns=int(se.max()),max_end_error_ns=int(ee.max()),topology_sha256=h,
                           interpretation='observed-cost representation check, NOT out-of-sample prediction'))
        assert se.max()==0 and ee.max()==0,checks[-1]
    csv(out,'source_validation/observed_cost_reconstruction.csv',pd.DataFrame(checks))


def partition_prediction(gpu,wrapper,costs,pp,mb,variant):
    # Independent marginal medians are normalized to the source-fitted lane0 phase
    # wall so each view adds up. GPU and wrapper views must never be added together.
    tables=[(gpu,GPU,'gpu_disjoint'),(wrapper,WRAP,'wrapper_disjoint')];rows=[]
    for data,cols,view in tables:
        med=data[data.iteration.isin(costs.fit)].groupby(KEY)[cols].median()
        for a in all_actions(pp,mb):
            if a['kind']!='phase' or a['pp_lane']!=0:continue
            st=costs.mapped_stage(a);m=a['microbatch']
            if pp==14 and costs.role_transfer and m==mb-1:m=3
            v=med.loc[(st,0,a['name'],m)];total=costs.phase(a)/1e6
            for col in cols:
                rows.append(dict(variant=variant,case='source256' if pp==16 else 'target224',pp_stage=a['pp_stage'],pp_lane=0,
                    phase=a['name'],microbatch=a['microbatch'],view=view,component=col,predicted_ms=float(v[col]/v.sum()*total),
                    scope='source lane0 component proportions; CPU phase aggregate; no per-rank GPU observation'))
    return rows


def source_scores(out,graphs,aligned,phase,boundaries):
    env=phase.groupby('iteration').agg(first=('observed_start_ns','min'),last=('observed_end_ns','max'))
    rows=[];local=[];timelines=[]
    for variant,(n,fit) in graphs.items():
        p=phase_frame(n);p['variant']=variant;first=p.start_ms.min();obs=phase.copy()
        obs['actual_duration_ms']=obs.duration_ns/1e6
        obs['actual_start_ms']=(obs.observed_start_ns-obs.iteration.map(env['first']))/1e6
        j=obs.merge(p[KEY+['start_ms','duration_ms']],on=KEY,validate='many_to_one')
        j['predicted_start_ms']=j.start_ms-first;j['duration_error_ms']=j.duration_ms-j.actual_duration_ms
        j['start_error_ms']=j.predicted_start_ms-j.actual_start_ms;j['variant']=variant
        j['split']=np.where(j.iteration.isin(fit),'source_fit',np.where(j.iteration.isin(WINDOW),'source_incremental_validation','historical_diagnostic'))
        timelines.append(j[['variant','iteration','split']+KEY+['actual_duration_ms','duration_ms','duration_error_ms','actual_start_ms','predicted_start_ms','start_error_ms']])
        for b in boundaries.itertuples():
            it=int(b.iteration);actual=(int(env.loc[it,'last'])-int(env.loc[it,'first']))/1e6;pred=envelope(n)['onef1b_ms']
            rows.append(dict(variant=variant,iteration=it,split='source_fit' if it in fit else ('source_incremental_validation' if it in WINDOW else 'historical_diagnostic'),
                actual_onef1b_ms=actual,predicted_onef1b_ms=pred,error_ms=pred-actual,ape_pct=100*abs(pred-actual)/actual,
                actual_entry_ms=b.profiler_entry_to_phase_ms,actual_tail_ms=b.profiler_step_ms-b.profiler_entry_to_phase_ms-actual,
                predicted_program_ms=envelope(n)['program_ms'],scope='free-running; no per-iteration observed readiness'))
        # Return/post and gaps are checked as local costs, NOT summed PP API wall waits.
        for kind,col,prefix in [('local_prelaunch','local_prelaunch_gap_ns','gap:'),('api_return','api_postjoin_return_ns','')]:
            x=n[n.kind.eq(kind)][['action_id','duration_ns']]
            o=aligned if kind=='local_prelaunch' else aligned[aligned.kind.eq('api')]
            z=o.merge(x,on='action_id',suffixes=('_observed','_predicted'),validate='many_to_one')
            pc='duration_ns_predicted' if 'duration_ns_predicted' in z else 'duration_ns'
            z['error_ms']=(z[pc]-z[col])/1e6
            for it,g in z.groupby('iteration'):
                local.append(dict(variant=variant,iteration=int(it),component=kind,mae_ms=g.error_ms.abs().mean(),bias_ms=g.error_ms.mean(),
                                  actual_sum_ms=g[col].sum()/1e6,predicted_sum_ms=g[pc].sum()/1e6,
                                  scope='sum across rank actions, not step attribution'))
    csv(out,'source_validation/iteration_results.csv',pd.DataFrame(rows))
    csv(out,'source_validation/phase_results.csv.gz',pd.concat(timelines,ignore_index=True))
    csv(out,'source_validation/local_runtime_results.csv',pd.DataFrame(local))
    metrics=pd.DataFrame(rows).groupby(['variant','split']).agg(count=('iteration','size'),onef1b_mape_pct=('ape_pct','mean'),bias_ms=('error_ms','mean')).reset_index()
    csv(out,'source_validation/metrics.csv',metrics)
    return metrics


def baseline(paths):
    contract=json.loads(paths['prediction_contract.json'].read_text());c=contract['prediction']
    n=pd.read_csv(paths['dag_v685_nodes.csv.gz'],usecols=['rank','pp_stage','pp_lane','phase','microbatch','predicted_start_ns','predicted_end_ns'])
    p=n[n['rank'].ge(0)&n.phase.isin(['FWD','BWD'])&n.microbatch.ge(0)]
    p=p.groupby(KEY,as_index=False).agg(predicted_start_ns=('predicted_start_ns','min'),predicted_end_ns=('predicted_end_ns','max'))
    entry=float(p.predicted_start_ns.min())/1e6;last=float(p.predicted_end_ns.max())/1e6
    record=dict(variant='v685_frozen',entry_ms=entry,onef1b_ms=last-entry,tail_ms=c['profiler_step_ms']-last,outer_ms=c['outer_framework_ms'],
                profiler_ms=c['profiler_step_ms'],training_ms=c['training_step_ms'],legacy_reconciliation_ms=c['target_reconciliation_ms'])
    p['phase']=p.phase.replace({'FWD':'forward','BWD':'backward'});p['variant']='v685_frozen'
    p['start_ms']=p.predicted_start_ns/1e6-entry;p['end_ms']=p.predicted_end_ns/1e6-entry
    p['duration_ms']=p.end_ms-p.start_ms
    assert abs(record['onef1b_ms']-19115.772914)<1e-6
    return record,p,contract


def run_candidate(out,aligned,pairs,phase,gpu,wrapper,paths,guard):
    source_oracle(out,aligned,pairs)
    boundary=pd.read_csv(paths['source256_profiler_entry_60_100.csv'])
    baseline_record,bp,contract=baseline(paths)
    dump(out/'prediction/frozen_baseline_contract.json',contract)
    sourcegraphs={};targetgraphs={};partitions=[];critical=[];records=[baseline_record];phasepred=[bp]
    specs=[('v686_split_full',[85,90],'full',True,FittedCosts),
           ('v687_split_full',[85,90],'full',True,ReadinessCosts),
           ('v687_split_zero_ready',[85,90],'zero_sender_ready',True,ReadinessCosts),
           ('v687_split_ordinal',[85,90],'full',False,ReadinessCosts),
           ('v687_full4',[85,90,95,100],'full',True,ReadinessCosts)]
    for name,fit,kind,role,cls in specs:
        for pp,mb,case in [(16,4,'source256'),(14,3,'target224')]:
            cost=cls(aligned,pairs,fit,pp,mb,role_transfer=role)
            n,e,h=build(pp,mb,cost,variant=kind)
            csv(out,f'prediction/{name}/{case}_nodes.csv.gz',n);csv(out,f'prediction/{name}/{case}_edges.csv.gz',e)
            csv(out,f'prediction/{name}/{case}_runtime_bindings.csv.gz',pd.DataFrame(cost.bindings))
            if isinstance(cost,ReadinessCosts):
                csv(out,f'prediction/{name}/{case}_readiness_parameters.csv',pd.DataFrame(cost.parameter_rows))
                if pp==16:
                    lv=cost.local_validation(pairs)
                    csv(out,f'source_validation/{name}_pp_local_results.csv.gz',lv)
                    csv(out,f'source_validation/{name}_pp_local_metrics.csv',lv.groupby(['split','direction','both_endpoints_single_message']).agg(
                        samples=('message_id','size'),mae_ms=('error_ms',lambda x:x.abs().mean()),bias_ms=('error_ms','mean'),
                        v686_mae_ms=('v686_error_ms',lambda x:x.abs().mean()),v686_bias_ms=('v686_error_ms','mean')).reset_index())
            dump(out/f'prediction/{name}/{case}_contract.json',dict(variant=name,case=case,fit=fit,pp=pp,microbatches=mb,
                 model_topology_sha256=h,formal_topology_sha256=contract['dependency_topology_sha256'],formal_topology_replaced=False,
                 node_count=len(n),edge_count=len(e),envelope=envelope(n),pp_cost_ns={str(k):v for k,v in cost.ppcost.items()},
                 phase_policy='rank-stage-phase ordinal' if not role else 'rank-stage-phase first/middle/last role',
                 uncertainty='CPU PP completion upper bound; whole F/B phase costs still include internal runtime/collectives'))
            critical.extend(critical_ledger(n,name,case));partitions.extend(partition_prediction(gpu,wrapper,cost,pp,mb,name))
            if pp==16:sourcegraphs[name]=(n,fit)
            else:
                targetgraphs[name]=n;p=phase_frame(n);p['variant']=name;p['start_ms']-=envelope(n)['first_phase_ns']/1e6;p['end_ms']-=envelope(n)['first_phase_ns']/1e6;phasepred.append(p)
                rec={**baseline_record,'variant':name,'onef1b_ms':envelope(n)['onef1b_ms']}
                rec['profiler_ms']=rec['entry_ms']+rec['onef1b_ms']+rec['tail_ms'];rec['training_ms']=rec['profiler_ms']+rec['outer_ms'];records.append(rec)
                sensitivity={**rec,'variant':name+'_no_legacy_reconciliation_sensitivity'}
                for field in ['tail_ms','profiler_ms','training_ms']:sensitivity[field]-=rec['legacy_reconciliation_ms']
                records.append(sensitivity)
    sm=source_scores(out,sourcegraphs,aligned,phase,boundary)
    csv(out,'prediction/phase_nodes.csv.gz',pd.concat(phasepred,ignore_index=True)[['variant']+KEY+['start_ms','end_ms','duration_ms']])
    csv(out,'prediction/internal_partition.csv',pd.DataFrame(partitions))
    csv(out,'prediction/critical_path_ledger.csv',pd.DataFrame(critical))
    csv(out,'prediction/critical_path_summary.csv',pd.DataFrame(critical).groupby(['variant','case','kind']).critical_contribution_ms.sum().reset_index())
    dump(out/'prediction/step_predictions.json',records)
    files=[p for directory in ['prediction','source_validation'] for p in (out/directory).rglob('*') if p.is_file()]
    seal={'status':'SOURCE_ONLY_PREDICTIONS_SEALED_BEFORE_TARGET_TIMING_ACCESS','files':[dict(path=str(p.relative_to(out)),sha256=sha(p)) for p in sorted(files)],
          'target_parameter_updates':0,'target_timing_read_before_seal':False,'historical_exposure':'development data, never blind',
          'code':[dict(path=str(p),sha256=sha(p)) for p in sorted(__import__('pathlib').Path(__file__).parent.glob('*.py'))]}
    dump(out/'prediction_seal.json',seal)
    for item in seal['files']:assert sha(out/item['path'])==item['sha256']
    guard.phase='evaluator'
    from scoring import evaluate
    metrics=evaluate(out,records,paths,partitions)
    for item in seal['files']:assert sha(out/item['path'])==item['sha256']
    return dict(status='SEALED_AND_EVALUATED',source_metrics=sm.to_dict('records'),target_metrics=metrics)

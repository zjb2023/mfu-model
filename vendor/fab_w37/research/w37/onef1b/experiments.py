"""Pre-registered source-only runtime shape correction on the locked graph."""
import json
import math
import numpy as np
import pandas as pd
from worker import ROOT, csv, dump, phases, phase_summary, conserved, TARGET_COMPONENTS, SOURCE_COMPONENTS, LOCK, topology_fingerprint, replay, apply_runtime_shape

MAP=[0,1,2,3,4,5,6,7,8,9,10,11,12,15]  # inherited v67 runtime map, not PP-gradient map


def fit_shape(events, source, fit_iterations):
    keys=['rank','pp_stage','pp_lane','phase','microbatch']
    obs=events[events.iteration.isin(fit_iterations)].copy()
    assert set(obs.iteration.unique())==set(fit_iterations)
    assert obs['source_case'].eq('256gpu_pp16_cp2_a2a').all()
    observed=obs.groupby(keys,as_index=False).agg(trace_median_ns=('duration_ns','median'),
        trace_p10_ns=('duration_ns',lambda s:s.quantile(.1)),trace_p90_ns=('duration_ns',lambda s:s.quantile(.9)),trace_samples=('duration_ns','size'))
    sel=source[source['rank'].ge(0)&source.phase.isin(['forward','backward'])&source.microbatch.ge(0)].copy()
    sel['model_adjustable_ns']=sel.compute_work_ns+sel.unclassified_calibration_ns
    sel['model_fixed_ns']=sel.network_service_ns_model+sel.software_sync_ns_model
    modeled=sel.groupby(keys,as_index=False)[['model_adjustable_ns','model_fixed_ns','duration_ns']].sum().rename(columns={'duration_ns':'model_phase_ns'})
    p=observed.merge(modeled,on=keys,validate='one_to_one')
    assert len(p)==2048 and p.trace_samples.eq(len(fit_iterations)).all()
    g=keys[:-1]
    p['phase_trace_total_ns']=p.groupby(g).trace_median_ns.transform('sum')
    p['phase_model_total_ns']=p.groupby(g).model_phase_ns.transform('sum')
    p['shape_target_phase_ns']=p.trace_median_ns*p.phase_model_total_ns/p.phase_trace_total_ns
    p['raw_adjustable_shape_factor']=(p.shape_target_phase_ns-p.model_fixed_ns)/p.model_adjustable_ns
    assert p.raw_adjustable_shape_factor.map(math.isfinite).all() and p.raw_adjustable_shape_factor.gt(0).all()
    p['runtime_p10_over_median']=p.trace_p10_ns/p.trace_median_ns
    p['runtime_p90_over_median']=p.trace_p90_ns/p.trace_median_ns
    p['fit_case']='256gpu_pp16_cp2_a2a'
    p['fit_iterations']='|'.join(map(str,fit_iterations))
    p['base_provenance']='v685 source graph with legacy 60–100 calibration; correction alone uses fit_iterations'
    p['cost_semantics']='aggregate adjustable runtime shape, not pure compute or added waiting'
    return p


def incremental_score(events, before, after, out, variant):
    rows=[]
    for tag,nodes in [('baseline',before),(variant,after)]:
        p=phases(nodes,True);p['phase']=p.phase.replace({'FWD':'forward','BWD':'backward'})
        p['pred_share']=p.duration_ms/p.groupby(['rank','phase']).duration_ms.transform('sum')
        o=events.copy();o['actual_share']=o.duration_ns/o.groupby(['iteration','rank','phase']).duration_ns.transform('sum')
        joined=o.merge(p[['rank','phase','microbatch','pred_share']],on=['rank','phase','microbatch'],validate='many_to_one')
        for it,g in joined.groupby('iteration'):
            rows.append({'variant':tag,'iteration':int(it),'shape_share_mae_pp':100*float((g.pred_share-g.actual_share).abs().mean()),
                         'meaning':'per-rank F/B microbatch duration share; historical model is contaminated, not independent validation'})
    csv(out,f'parameters/{variant}_source_incremental_scores.csv',pd.DataFrame(rows))


def run_experiments(out,nodes,edges,source,source_edges,events,boundaries,contract):
    plan=json.loads((ROOT/'docs/w37/1f1b/experiment_plan.json').read_text())
    dump(out/'parameters/experiment_plan.json',plan)
    results=[]
    for variant,its in [('shape_split',[85,90]),('shape_same_window',[85,90,95,100])]:
        p=fit_shape(events,source,its)
        csv(out,f'parameters/{variant}_source_parameters.csv',p)
        sn,st=apply_runtime_shape(source,p,target=False,source_stage_for_target=MAP)
        tn,tt=apply_runtime_shape(nodes,p,target=True,source_stage_for_target=MAP)
        csv(out,f'parameters/{variant}_source_transfer.csv',st)
        csv(out,f'parameters/{variant}_target_transfer.csv',tt)
        conserved(sn,SOURCE_COMPONENTS);conserved(tn,TARGET_COMPONENTS)
        for old,new in [(source,sn),(nodes,tn)]:
            for col in ['network_service_ns_model','software_sync_ns_model']:
                assert old[col].equals(new[col])
            # Every non-phase node stays identical, including entry, PP and optimizer tail.
            mask=~(old['rank'].ge(0)&old.phase.isin(['FWD','BWD','forward','backward'])&old.microbatch.ge(0))
            assert old.loc[mask,'duration_ns'].equals(new.loc[mask,'duration_ns'])
        assert topology_fingerprint(edges)==LOCK
        sn,sc=replay(sn,source_edges,'iteration:collectives_complete')
        tn,tc=replay(tn,edges,'iteration:completion_join')
        sr=float(sn.loc[sn.node_id.eq('iteration:collectives_complete'),'predicted_end_ns'].item())/1e6
        tr=float(tn.loc[tn.node_id.eq('iteration:completion_join'),'predicted_end_ns'].item())/1e6
        recon=(float(contract['source_profiler_reference_median_ms'])-sr)*.75
        if recon<0:raise ValueError('source graph exceeds source profiler; do not clip or conceal reconciliation')
        f,b=phase_summary(tn);sf,sb=phase_summary(sn,True)
        incremental_score(events,source,sn,out,variant)
        csv(out,f'candidates/{variant}/nodes.csv.gz',tn)
        csv(out,f'candidates/{variant}/source_nodes.csv.gz',sn)
        csv(out,f'candidates/{variant}/critical.csv.gz',tc)
        csv(out,f'candidates/{variant}/source_critical.csv.gz',sc)
        for name,residual in [(variant,recon)]+([('shape_split_fixed_reconciliation',contract['target_reconciliation_ms'])] if variant=='shape_split' else []):
            results.append({'variant':name,'entry_ms':f,'onef1b_ms':b-f,'raw_graph_ms':tr,
                'reconciliation_ms':residual,'tail_ms':tr+residual-b,'profiler_ms':tr+residual,'outer_ms':contract['outer_framework_ms'],
                'training_ms':tr+residual+contract['outer_framework_ms'],'source_raw_ms':sr,
                'source_phase_start_ms':sf,'source_phase_end_ms':sb})
    dump(out/'parameters/invariants.json',{'target_topology_sha256':LOCK,'target_edges_changed':0,'source_edges_changed':0,
        'per_rank_phase_adjustable_ns_conserved':True,'network_and_software_ns_unchanged':True,
        'entry_pp_optimizer_node_costs_unchanged':True,'parameter_target_timing_reads':0,
        'shape_stage_map':MAP,'phase_model':'aggregate adjustable compute/runtime only; no attribution to pure compute'})
    return results

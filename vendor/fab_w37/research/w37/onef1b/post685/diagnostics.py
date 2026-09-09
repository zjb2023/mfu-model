"""Explicit posthoc target-assisted lane0 attribution, never a deployable predictor."""
import math
import pandas as pd
from worker import csv
from smoke_worker import dump
from pp_semantics import replay
from pp_graph import envelope

GPU=['phase_noncommunication_only_ms','phase_communication_only_ms','phase_comm_noncommunication_overlap_ms','phase_gpu_idle_ms']
KEY=['pp_stage','pp_lane','phase','microbatch']


def attribution(out,graph,partitions,paths):
    obs=pd.read_csv(paths['target_phase_rank_events_60_100.csv'])
    truth=pd.read_csv(paths['target224_outside_wrapper_ground_truth.csv.gz'])
    pred=pd.DataFrame(partitions);pred=pred[pred.variant.eq('v687_split_full')&pred.case.eq('target224')&pred.view.eq('gpu_disjoint')]
    pivot=pred.pivot(index=KEY,columns='component',values='predicted_ms')
    lane=graph[graph.pp_lane.eq(0)|graph.pp_lane.eq(-1)].copy();ids=set(lane.node_id)
    edges=pd.read_csv(out/'prediction/v687_split_full/target224_edges.csv.gz');edges=edges[edges.src.isin(ids)&edges.dst.isin(ids)]
    baseline=replay(lane.to_dict('records'),edges.to_dict('records'));base=envelope(baseline)['onef1b_ms']
    # Validate the per-rank observation table against the older frozen 16-rank
    # phase envelopes; this does not turn stage envelopes into rank observations.
    old=pd.read_csv(out/'evaluator_only/observed_phase_envelopes.csv')
    origin=obs.groupby('iteration').observed_start_ns.min();d=obs.copy()
    d['start_ms']=(d.observed_start_ns-d.iteration.map(origin))/1e6;d['end_ms']=(d.observed_end_ns-d.iteration.map(origin))/1e6
    env=d.groupby(['iteration','pp_stage','phase','microbatch']).agg(start_ms=('start_ms','min'),end_ms=('end_ms','max')).reset_index()
    joined=old.merge(env,on=['iteration','pp_stage','phase','microbatch'],validate='one_to_one')
    assert (joined.start_ms-joined.actual_start_ms).abs().max()<.001
    assert (joined.end_ms-joined.actual_end_ms).abs().max()<.001
    csv(out,'evaluator_only/target_rank_phase_nodes.csv.gz',d)
    costs=[];shapley=[];ledgers=[]
    for it in [85,90,95,100]:
        t=truth[truth.iteration.eq(it)].set_index(KEY);o=obs[obs.iteration.eq(it)&obs.pp_lane.eq(0)]
        cpu=o.set_index(KEY).duration_ns/1e6
        assert (t[GPU].sum(axis=1)-cpu).abs().max()<.001
        delta=t[GPU]-pivot[GPU];values={}
        for mask in range(16):
            n=lane.copy()
            for i,r in n[n.kind.eq('phase_compute_communication_runtime')].iterrows():
                key=tuple(r[k] for k in KEY)
                addition=sum(delta.loc[key,GPU[k]] for k in range(4) if mask&(1<<k))
                n.loc[i,'duration_ns']=round(int(r.duration_ns)+addition*1e6)
            assert n.duration_ns.ge(0).all()
            v=envelope(replay(n.to_dict('records'),edges.to_dict('records')))['onef1b_ms'];values[mask]=v
            costs.append(dict(iteration=it,subset_mask=mask,observed_components=','.join(GPU[k] for k in range(4) if mask&(1<<k)),
                lane0_onef1b_ms=v,delta_from_prediction_ms=v-base,interpretation='POSTHOC target lane0 phase component substitution, not prediction'))
        phi=[]
        for k,col in enumerate(GPU):
            effect=0.
            for mask in range(16):
                if mask&(1<<k):continue
                count=mask.bit_count();weight=math.factorial(count)*math.factorial(3-count)/math.factorial(4)
                effect+=weight*(values[mask|(1<<k)]-values[mask])
            phi.append(effect);shapley.append(dict(iteration=it,component=col,attributed_delta_ms=effect,
                scope='lane0 coarse-DAG phase-cost counterfactual; interactions averaged exactly; NOT full-rank error contribution'))
        assert abs(sum(phi)-(values[15]-base))<1e-7
        actual=(int(o.observed_end_ns.max())-int(o.observed_start_ns.min()))/1e6
        ledgers.append(dict(iteration=it,predicted_lane0_onef1b_ms=base,actual_lane0_onef1b_ms=actual,
            all_observed_phase_components_counterfactual_ms=values[15],phase_component_effect_ms=sum(phi),
            remaining_schedule_pp_initial_skew_error_ms=actual-values[15],total_underprediction_ms=actual-base,
            interpretation='remaining error is unresolved PP/local scheduling/initial skew or coupled collectives; no target timing added to predictor'))
    csv(out,'evaluator_only/diagnostic_subset_replays.csv',pd.DataFrame(costs))
    csv(out,'evaluator_only/diagnostic_shapley_components.csv',pd.DataFrame(shapley))
    csv(out,'evaluator_only/diagnostic_lane0_error_ledger.csv',pd.DataFrame(ledgers))
    dump(out/'evaluator_only/diagnostic_contract.json',{'status':'POSTHOC_TARGET_ASSISTED_NOT_CANDIDATE','scope':'lane0 only, four target iterations, exact 16 subsets',
        'formal_graph_modified':False,'predictor_parameters_modified':False,'global_error_attribution_claimed':False,
        'observed_GPU_idle_meaning':'no traced GPU activity inside CPU phase, not proven host work or optimizable stall',
        'GPU_communication_meaning':'active communication kernels can include waiting/polling; not pure payload transmission'})

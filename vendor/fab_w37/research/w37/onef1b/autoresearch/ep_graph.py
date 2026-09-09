"""Independent CPU-observable EP refinement of the v687 PP program graph.

The shared completion is a source-supported CPU wrapper abstraction, not a GPU
barrier or network service claim. Formal v684/v685 topology is never modified.
"""
import pandas as pd
from pp_graph import build as pp_build,ObservedCosts
from pp_semantics import replay,topology_sha,align_observations
from worker import csv
from smoke_worker import dump

F=['ep_dispatch_wall','ep_combine_wall']
B=['ep_recompute_dispatch_wall','ep_recompute_combine_wall','ep_combine_backward_wall','ep_dispatch_backward_wall']
GROUP_KEYS=['ep_group','pp_stage','phase','microbatch','layer_id','semantic_region']


def group_id(row):
    return f'ep:g{row.ep_group}:s{row.pp_stage}:{row.phase}:m{row.microbatch}:L{row.layer_id}:{row.semantic_region}'


def observed_graph(aligned,pairs,anchors,iteration):
    costs=ObservedCosts(aligned,pairs,iteration);base,edges,_=pp_build(16,4,costs,phase_override=lambda a,ns:0)
    phase=base[base.kind.eq('phase_compute_communication_runtime')].copy()
    base.loc[base.kind.eq('phase_compute_communication_runtime'),'kind']='phase_exit'
    nodes=base.drop(columns=['predicted_start_ns','predicted_end_ns','critical_predecessor']).to_dict('records')
    edges=edges.to_dict('records')
    for edge in edges:
        if edge['edge_type']=='phase_launch':edge['dst']+=':enter'
    aa=anchors[anchors.iteration.eq(iteration)].copy()
    groups=aa.groupby(GROUP_KEYS).agg(last_entry=('arrival_ns','max'),first_return=('completion_ns','min')).reset_index()
    groups['node_id']=[group_id(r) for r in groups.itertuples()]
    group_rows=groups.set_index('node_id').to_dict('index')
    source=aa.groupby(['rank','phase','microbatch'])
    observations=aligned[aligned.iteration.eq(iteration)].set_index('action_id')
    bindings=[]
    def node(node_id,kind,duration,record,**extra):
        assert duration>=0,(node_id,duration)
        nodes.append(dict(node_id=node_id,kind=kind,duration_ns=int(duration),rank=int(record['rank']),pp_stage=int(record['pp_stage']),
            pp_lane=int(record['pp_lane']),phase=record['phase'],microbatch=int(record['microbatch']),action_id=record.get('action_id',''),
            region=record.get('region','internal'),cost_key='observed_source_reassembly_NOT_prediction',**extra))
    def edge(src,dst,kind):
        edges.append(dict(src=src,dst=dst,edge_type=kind,dependency_source='pinned_full_block_recompute_per_layer_and_EP_wrapper_order; CPU_group_completion_candidate_T05'))
    for row in phase.to_dict('records'):
        aid=row['node_id'];observed=observations.loc[aid]
        node(aid+':enter','phase_enter',0,row)
        members=source.get_group((row['rank'],row['phase'],row['microbatch'])).sort_values('arrival_ns')
        expected=[(int(i),kind) for i in sorted(members.execution_layer.unique()) for kind in (F if row['phase']=='forward' else B)]
        actual=list(zip(members.execution_layer.astype(int),members.semantic_region))
        assert actual==expected,(iteration,aid,actual,expected)
        previous=aid+':enter';previous_time=int(observed.start_ns)
        for anchor in members.itertuples():
            gid=group_id(anchor);g=group_rows[gid]
            tag=f'{aid}:L{anchor.layer_id}:{anchor.semantic_region}'
            node(tag+':local','local_phase_work_and_unresolved_runtime',int(anchor.arrival_ns)-previous_time,row,layer_id=int(anchor.layer_id))
            node(tag+':enter','ep_wrapper_entry',0,row,layer_id=int(anchor.layer_id),semantic_region=anchor.semantic_region)
            node(tag+':return','ep_rank_return_tail',int(anchor.completion_ns)-int(g['first_return']),row,layer_id=int(anchor.layer_id),semantic_region=anchor.semantic_region)
            edge(previous,tag+':local','rank_local_wrapper_program_order');edge(tag+':local',tag+':enter','wrapper_cpu_entry')
            edge(tag+':enter',gid,'all_ep_rank_cpu_entries');edge(gid,tag+':return','earliest_cpu_group_return_then_rank_tail')
            bindings.append(dict(iteration=iteration,rank=int(anchor.rank),pp_stage=int(anchor.pp_stage),pp_lane=int(anchor.pp_lane),phase=anchor.phase,
                microbatch=int(anchor.microbatch),layer_id=int(anchor.layer_id),semantic_region=anchor.semantic_region,
                entry_node=tag+':enter',return_node=tag+':return',arrival_ns=int(anchor.arrival_ns),completion_ns=int(anchor.completion_ns)))
            previous=tag+':return';previous_time=int(anchor.completion_ns)
        node(aid+':finish','local_phase_work_and_unresolved_runtime',int(observed.end_ns)-previous_time,row)
        edge(previous,aid+':finish','rank_local_phase_finish');edge(aid+':finish',aid,'phase_completed_after_internal_work')
    for g in groups.itertuples():
        row=dict(rank=-1,pp_stage=g.pp_stage,pp_lane=-1,phase=g.phase,microbatch=g.microbatch)
        node(g.node_id,'ep_cpu_completion_after_last_entry',int(g.first_return)-int(g.last_entry),row,ep_group=int(g.ep_group),
             layer_id=int(g.layer_id),semantic_region=g.semantic_region)
    result=replay(nodes,edges)
    return result,pd.DataFrame(edges),pd.DataFrame(bindings),costs.origin,topology_sha(edges)


def source_oracle(out,paths,plan):
    phase=pd.read_csv(paths['source_pp_trace_events_60_100.csv']);api=pd.read_csv(paths['source_pp_api_events_60_100.csv'])
    aligned,pairs=align_observations(phase,api)
    cols=['iteration','rank','pp_stage','pp_lane','ep_group','phase','microbatch','execution_layer','layer_id','semantic_region','arrival_ns','completion_ns']
    anchors=pd.read_csv(paths['ep_source_ep_rank_anchor_events.csv'],usecols=cols)
    checks=[];hashes=[]
    for iteration in sorted(aligned.iteration.unique()):
        nodes,edges,bindings,origin,topology=observed_graph(aligned,pairs,anchors,int(iteration));by=nodes.set_index('node_id')
        obs=aligned[aligned.iteration.eq(iteration)].copy()
        starts=obs.apply(lambda r:r.action_id+(':enter' if r.kind=='phase' else ':post'),axis=1).map(by.predicted_start_ns).astype('int64')
        ends=obs.apply(lambda r:r.action_id+('' if r.kind=='phase' else ':return'),axis=1).map(by.predicted_end_ns).astype('int64')
        se=(starts-(obs.start_ns-origin)).abs();ee=(ends-(obs.end_ns-origin)).abs()
        be=(bindings.entry_node.map(by.predicted_start_ns)-(bindings.arrival_ns-origin)).abs()
        re=(bindings.return_node.map(by.predicted_end_ns)-(bindings.completion_ns-origin)).abs()
        check=dict(iteration=int(iteration),nodes=len(nodes),edges=len(edges),ep_rank_wrappers=len(bindings),
            max_pp_phase_start_error_ns=int(se.max()),max_pp_phase_end_error_ns=int(ee.max()),max_ep_entry_error_ns=int(be.max()),max_ep_return_error_ns=int(re.max()),topology_sha256=topology)
        assert max(check[k] for k in check if k.startswith('max_'))==0,check
        checks.append(check);hashes.append(topology)
        if iteration==85:
            csv(out,'source85_nodes.csv.gz',nodes);csv(out,'source85_edges.csv.gz',edges);csv(out,'source85_ep_bindings.csv.gz',bindings)
    assert len(set(hashes))==1
    csv(out,'source_observed_reassembly.csv',pd.DataFrame(checks))
    dump(out/'diagnostic.json',dict(status='SOURCE_PP_EP_CPU_OBSERVED_REASSEMBLY_PASS',new_prediction=False,used_to_fit_model=False,
        raw_trace_scanned=False,formal_topology_changed=False,source_iterations=list(map(int,sorted(aligned.iteration.unique()))),
        all_pp_phase_and_ep_entry_return_errors_ns=0,topology_sha256=hashes[0],diagnostic=plan['diagnostic'],sealed_reference=plan['sealed_reference'],
        boundary='Observed-cost representability only, not predictive validation. CPU EP completion joins still require target runtime/flag review; GPU-ready and pure network service are not identified.',
        next='Fit disjoint local intervals, earliest EP group CPU completion, per-rank return tails and PP readiness using source85/90 only; validate source95/100 before sealed target224 evaluation.'))

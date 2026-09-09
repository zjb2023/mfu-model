"""Static EP CPU program factory; costs and observation spans remain separate."""
import pandas as pd
from pp_graph import build as pp_build
from pp_semantics import replay,topology_sha

F=['ep_dispatch_wall','ep_combine_wall']
B=['ep_recompute_dispatch_wall','ep_recompute_combine_wall','ep_combine_backward_wall','ep_dispatch_backward_wall']
DEPENDENCY='pinned_full_block_recompute_per_layer_and_EP_wrapper_order; CPU_group_completion_candidate_T05'


def build(pp,mb,ppcosts,epcosts,layer_map):
    base,edges,_=pp_build(pp,mb,ppcosts,phase_override=lambda a,ns:0)
    phases=base[base.kind.eq('phase_compute_communication_runtime')].to_dict('records')
    base.loc[base.kind.eq('phase_compute_communication_runtime'),'kind']='phase_exit'
    nodes=base.drop(columns=['predicted_start_ns','predicted_end_ns','critical_predecessor','on_critical_path']).to_dict('records')
    edges=edges.to_dict('records');groups={};bindings=[]
    for e in edges:
        if e['edge_type']=='phase_launch':e['dst']+=':enter'
    def node(name,kind,duration,row,**extra):
        assert duration>=0
        nodes.append(dict(node_id=name,kind=kind,duration_ns=int(round(duration)),rank=int(row['rank']),pp_stage=int(row['pp_stage']),
            pp_lane=int(row['pp_lane']),phase=row['phase'],microbatch=int(row['microbatch']),action_id=row.get('action_id',''),
            region=row.get('region','internal'),cost_key=epcosts.last_key if duration else 'structural_zero_or_explicit_ablation',**extra))
    def edge(a,b,kind):edges.append(dict(src=a,dst=b,edge_type=kind,dependency_source=DEPENDENCY))
    for row in phases:
        aid=row['node_id'];node(aid+':enter','phase_enter',0,row);previous=aid+':enter'
        layers=layer_map[int(row['pp_stage'])];layers=layers if row['phase']=='forward' else list(reversed(layers))
        layers=[layer for layer in layers if layer>=1]
        for ex,layer in enumerate(layers):
            for semantic in (F if row['phase']=='forward' else B):
                group_index=int(row['pp_lane'])//8;gid=f'ep:g{int(row["pp_stage"])*2+group_index}:s{row["pp_stage"]}:{row["phase"]}:m{row["microbatch"]}:L{layer}:{semantic}'
                groups[gid]=dict(rank=-1,pp_stage=row['pp_stage'],pp_lane=-1,phase=row['phase'],microbatch=row['microbatch'],ep_lane_group=group_index,
                    ep_group=int(row['pp_stage'])*2+group_index,execution_layer=ex,layer_id=layer,semantic_region=semantic)
                tag=f'{aid}:L{layer}:{semantic}'
                node(tag+':local','local_phase_work_and_unresolved_runtime',epcosts.value('local',row,ex,semantic),row,layer_id=layer)
                node(tag+':enter','ep_wrapper_entry',0,row,layer_id=layer,semantic_region=semantic)
                node(tag+':return','ep_rank_return_tail',epcosts.value('tail',row,ex,semantic),row,layer_id=layer,semantic_region=semantic)
                edge(previous,tag+':local','rank_local_wrapper_program_order');edge(tag+':local',tag+':enter','wrapper_cpu_entry')
                edge(tag+':enter',gid,'all_ep_rank_cpu_entries');edge(gid,tag+':return','earliest_cpu_group_return_then_rank_tail')
                bindings.append(dict(rank=int(row['rank']),pp_stage=int(row['pp_stage']),pp_lane=int(row['pp_lane']),phase=row['phase'],microbatch=int(row['microbatch']),
                    execution_layer=ex,layer_id=layer,semantic_region=semantic,entry_node=tag+':enter',group_node=gid,return_node=tag+':return'))
                previous=tag+':return'
        node(aid+':finish','local_phase_work_and_unresolved_runtime',epcosts.value('finish',row),row)
        edge(previous,aid+':finish','rank_local_phase_finish');edge(aid+':finish',aid,'phase_completed_after_internal_work')
    for gid,row in groups.items():
        node(gid,'ep_cpu_completion_after_last_entry',epcosts.value('group',row,row['execution_layer'],row['semantic_region']),row,
            ep_group=row['ep_group'],layer_id=row['layer_id'],semantic_region=row['semantic_region'])
    result=replay(nodes,edges)
    return result,pd.DataFrame(edges),pd.DataFrame(bindings),topology_sha(edges)


def spans(nodes):
    """Non-executable macro spans for scoring; never add these to the real graph."""
    by=nodes.set_index('node_id');p=nodes[nodes.kind.eq('phase_exit')].copy()
    p['predicted_start_ns']=(p.node_id+':enter').map(by.predicted_start_ns).astype('int64')
    p['duration_ns']=p.predicted_end_ns-p.predicted_start_ns
    p['kind']='phase_compute_communication_runtime';p['view_scope']='derived macro span for scoring only; not an executable node'
    p['start_ms']=p.predicted_start_ns/1e6;p['end_ms']=p.predicted_end_ns/1e6;p['duration_ms']=p.duration_ns/1e6
    return p


def metrics_view(nodes):
    return pd.concat([nodes[~nodes.kind.eq('phase_exit')],spans(nodes)],ignore_index=True)


def envelope(nodes):
    p=spans(nodes)
    return dict(first_phase_ns=int(p.predicted_start_ns.min()),last_phase_ns=int(p.predicted_end_ns.max()),
        onef1b_ms=float(p.predicted_end_ns.max()-p.predicted_start_ns.min())/1e6,program_ms=float(nodes.predicted_end_ns.max())/1e6)


def critical_ledger(nodes,variant,case):
    by=nodes.set_index('node_id');p=spans(nodes);lo=int(p.predicted_start_ns.min());last=p.loc[p.predicted_end_ns.idxmax(),'node_id'];rows=[]
    while last:
        r=by.loc[last];ns=max(0,int(r.predicted_end_ns)-max(lo,int(r.predicted_start_ns)))
        if ns:rows.append(dict(variant=variant,case=case,node_id=last,kind=r.kind,pp_stage=int(r.pp_stage),pp_lane=int(r.pp_lane),phase=r.phase,
            microbatch=int(r.microbatch),critical_contribution_ms=ns/1e6))
        last=r.critical_predecessor
    assert abs(sum(r['critical_contribution_ms'] for r in rows)-envelope(nodes)['onef1b_ms'])<1e-7
    return rows


def wait_table(nodes,bindings):
    by=nodes.set_index('node_id');r=bindings.copy()
    r['entry_ns']=r.entry_node.map(by.predicted_start_ns);r['all_entered_ns']=r.group_node.map(by.predicted_start_ns)
    r['first_return_ns']=r.group_node.map(by.predicted_end_ns);r['rank_return_ns']=r.return_node.map(by.predicted_end_ns)
    r['wait_for_other_entries_ns']=r.all_entered_ns-r.entry_ns
    r['group_completion_ns']=r.first_return_ns-r.all_entered_ns;r['rank_tail_ns']=r.rank_return_ns-r.first_return_ns
    r['wrapper_wall_ns']=r.rank_return_ns-r.entry_ns
    assert r.wait_for_other_entries_ns.ge(0).all()
    assert (r.wait_for_other_entries_ns+r.group_completion_ns+r.rank_tail_ns).eq(r.wrapper_wall_ns).all()
    r['scope']='CPU wrapper modeled wait; waiting intervals are derived, not extra costs'
    return r

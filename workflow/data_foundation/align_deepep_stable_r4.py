"""Normalize trace partitions; audit conditional edges using source data only."""
import copy, json, re, statistics
from collections import defaultdict
from pathlib import Path
from deepep_stable_pipeline_r1 import OUT, C, save, load, sha
from build_b_cp_pair_r4 import run
from audit_source80_topology_r1 import project

def collapse_bridges(graph):
    g=copy.deepcopy(graph);groups=defaultdict(list)
    for k,n in g.items():
        if n.get('kind') in ['gradient_processing_exposed_union','uncovered_trace_residual']:
            assert re.fullmatch(r'U\d+:r\d+:bridge\d+',k),k
            groups[k.rsplit(':',1)[0]].append(k)
    for prefix,keys in groups.items():
        keys.sort(key=lambda k:int(k.rsplit('bridge',1)[1]));keyset=set(keys)
        for a,b in zip(keys,keys[1:]):assert g[b]['deps']==[a]
        for k,n in g.items():
            if k not in keyset and set(n['deps'])&set(keys[:-1]):
                assert keys[-1] in n['deps'],(k,'interior-only bridge consumer')
        active=prefix+':bridge_active';gap=prefix+':bridge_residual'
        base={k:v for k,v in g[keys[0]].items() if k not in ['deps','cost_ms','kind']}
        g[active]=dict(base,deps=g[keys[0]]['deps'],cost_ms=sum(g[k]['cost_ms'] for k in keys if g[k]['kind']=='gradient_processing_exposed_union'),kind='gradient_processing_exposed_union')
        g[gap]=dict(base,deps=[active],cost_ms=sum(g[k]['cost_ms'] for k in keys if g[k]['kind']=='uncovered_trace_residual'),kind='uncovered_trace_residual')
        for k,n in g.items():
            if k not in keyset:n['deps']=list(dict.fromkeys(gap if d in keyset else d for d in n['deps']))
        for k in keys:del g[k]
    return g

def align(graphs,phase):
    normalized=[];timings=[];projections=[];keep=None
    for original in graphs:
        g=collapse_bridges(original) if phase=='B' else copy.deepcopy(original)
        t=run(g);before=run(original)
        assert abs(t['END']['end_ms']-before['END']['end_ms'])<1e-5
        for k in g.keys()&original.keys():assert abs(t[k]['end_ms']-before[k]['end_ms'])<1e-5,(k,'collapse changed boundary')
        selected={k for k,n in g.items() if n.get('kind') not in ['trace_handoff','recompute_readiness']}
        if keep is None:keep=selected
        assert selected==keep,('semantic key mismatch',sorted(selected^keep))
        normalized.append(g);timings.append(t);projections.append(project(g,keep))
    common={k:sorted(set.intersection(*(set(p[k]) for p in projections))) for k in keep}
    removed=[]
    for k in sorted(keep):
        choices=[set(p[k]) for p in projections]
        if any(ds!=choices[0] for ds in choices):
            assert phase=='B' and k.startswith('R'),('unexpected dependency variation',k)
            removed.append(dict(node=k,kind=normalized[0][k].get('kind'),retained=common[k],per_iteration=[dict(iteration=it,removed=sorted(ds-set(common[k]))) for it,ds in zip(C['tail_iterations'],choices)]))
    bound=[];checks=[]
    for it,g,t in zip(C['tail_iterations'],normalized,timings):
        out={};reserved=0
        for k in sorted(keep):
            node=copy.deepcopy(g[k]);ready=max([t[d]['end_ms'] for d in common[k]] or [0]);residual=t[k]['start_ms']-ready
            assert residual>=-1e-5,(k,residual)
            slot='aligned_ready::'+k
            out[slot]=dict(deps=common[k],cost_ms=max(0,residual),kind='source_readiness_residual',policy='trace_reserved',basis='source residual after invariant producer dependencies; not pure idle or network service')
            node.update(deps=[slot]);out[k]=node;reserved+=max(0,residual)
        replay=run(out);err=max(abs(replay[k]['end_ms']-t[k]['end_ms']) for k in keep)
        assert err<1e-5,(it,phase,err)
        checks.append(dict(iteration=it,max_boundary_error_ms=err,window_ms=replay['END']['end_ms'],retained_nodes=len(keep)))
        bound.append(out)
    mean=copy.deepcopy(bound[0])
    for k,n in mean.items():
        assert all(g[k]['deps']==n['deps'] and g[k].get('kind')==n.get('kind') for g in bound)
        n['cost_ms']=statistics.mean(g[k]['cost_ms'] for g in bound)
        for field in ['observed_start_ms','observed_end_ms','events','device_events']:n.pop(field,None)
        n['source_cost_policy']='equal mean of ten source iterations after semantic partition alignment'
    return mean,dict(checks=checks,conditional_edge_changes=removed,nodes=len(mean),mean_window_ms=run(mean)['END']['end_ms'],limits=['Only invariant observed conditional edges retained; not event-handle proof.', 'Removed conditional timing constraints are represented by source readiness residuals; cost perturbations may require richer synchronization modeling.', 'Bridge active/residual sums retain exit timing; their interior partition timing is not a separate injectable operator.'])

def main():
    dest=OUT/'aligned-r4';dest.mkdir(exist_ok=True);audit={};inputs={}
    for ph,name in [('F','four-layer-f-r1'),('B','b-recompute-r12')]:
        graphs=[]
        for it in C['tail_iterations']:
            p=OUT/f'costs-r2/i{it}/{name}/graph.json';graphs.append(load(p));inputs[str(p)]=sha(p)
        mean,audit[ph]=align(graphs,ph);save(dest/f'{ph}.json',mean)
    save(dest/'audit.json',dict(status='PASS_SOURCE_ALIGNMENT_CONDITIONAL_DAG',phases=audit,inputs=inputs,no_target_read=True,code_sha256=sha(Path(__file__))))
    print(json.dumps({p:{k:v for k,v in a.items() if k not in ['checks','conditional_edge_changes']} for p,a in audit.items()}))
if __name__=='__main__':main()

"""Bind target costs onto EXACT frozen source edges, or report semantic gaps.

No timing scale factor, target-edge union, missing-node zero fill or clipping
of negative residuals is permitted.
"""
import argparse,copy,json,re
from pathlib import Path
from align_deepep_stable_r4 import collapse_bridges
from build_b_cp_pair_r4 import run
from calibrate256_fine_r1 import BASE,load,save,sha
def canonical(key,base):
    key=re.sub(r'(^|:)r(\d+)(?=:|$)',lambda m:m[1]+'r'+str(int(m[2])-base+8),key)
    return re.sub(r'(^|:)p(\d+)-(\d+)(?=:|$)',lambda m:m[1]+f'p{int(m[2])-base+8}-{int(m[3])-base+8}',key)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank-base',type=int,required=True);ap.add_argument('--mb',type=int,default=0);ap.add_argument('--model-dir',type=Path,default=BASE/'deepep-stable-20261016-r1/model');ap.add_argument('--tag',default='frozen-binding-r1');a=ap.parse_args()
    root=BASE/'calibrate256-fine-r1'/f'r{a.rank_base}-mb{a.mb}';dest=root/a.tag;dest.mkdir(exist_ok=False);report={};inputs={}
    for phase,name in [('F','four-layer-f-r1'),('B','b-recompute-r12')]:
        p=root/name/'graph.json'
        if not p.exists():report[phase]=dict(status='NOT_BUILT');continue
        original=load(p);original_keys={canonical(k,a.rank_base):k for k in original};target=collapse_bridges(original) if phase=='B' else original
        target={canonical(k,a.rank_base):dict(n,deps=[canonical(d,a.rank_base) for d in n['deps']]) for k,n in target.items()}
        times=run(target);sp=a.model_dir/f'{phase}.json';source=load(sp);inputs[str(p)]=sha(p);inputs[str(sp)]=sha(sp)
        keep={k for k in source if not k.startswith('aligned_ready::')};missing=sorted(keep-target.keys());extra=sorted(k for k,n in target.items() if k not in keep and n.get('kind') not in ['trace_handoff','recompute_readiness']);conflicts=[]
        kinds=[dict(node=k,source=source[k].get('kind'),target=target[k].get('kind')) for k in keep&target.keys() if source[k].get('kind')!=target[k].get('kind')]
        r=dict(missing=missing,extra=extra,kind_mismatch=kinds,target_ms=times['END']['end_ms'],source_ms=run(source)['END']['end_ms']);g=copy.deepcopy(source);bindings=[]
        if not missing and not extra and not kinds:
            for k in sorted(keep):
                slot='aligned_ready::'+k;ready=max([times[d]['end_ms'] for d in source[slot]['deps']] or [0]);res=times[k]['start_ms']-ready
                if res < -1e-5:conflicts.append(dict(node=k,actual_start_ms=times[k]['start_ms'],source_predecessor_completion_ms=ready,residual_ms=res,deps=source[slot]['deps']))
                g[k]['cost_ms']=target[k]['cost_ms'];g[slot]['cost_ms']=res
                bindings.append(dict(node=k,target_cost_ms=target[k]['cost_ms'],target_reserved_ms=res,actual_start_ms=times[k]['start_ms'],actual_end_ms=times[k]['end_ms'],source_cost_ms=source[k]['cost_ms'],evidence_graph=str(p),target_node=original_keys.get(k,'collapsed_bridge_union')))
            if not conflicts:
                replay=run(g);err=max(abs(replay[k]['end_ms']-times[k]['end_ms']) for k in keep);assert err<1e-5
                assert all(g[k]['deps']==source[k]['deps'] for k in source)
                r.update(status='PASS_EXACT_SOURCE_EDGES_CONDITIONAL_TARGET_REPLAY',replay_ms=replay['END']['end_ms'],max_endpoint_error_ms=err,nodes=len(g));save(dest/f'{phase}.json',g);save(dest/f'{phase}-replay.json',replay)
            else:r['status']='BLOCKED_SOURCE_DEPENDENCY_CONFLICT'
        else:r['status']='BLOCKED_SEMANTIC_PARTITION_MISMATCH'
        r['conflicts']=conflicts;report[phase]=r;save(dest/f'{phase}-bindings.json',bindings)
    save(dest/'report.json',dict(scope=dict(rank_base=a.rank_base,mb=a.mb,iter=60),phases=report,limits=['Target-calibrated observed readiness; not independent prediction.','Exact source edges checked separately from target builder conditional edges.']))
    save(dest/'manifest.json',dict(inputs=inputs,code_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    print(json.dumps({ph:{k:(len(v) if isinstance(v,list) else v) for k,v in r.items()} for ph,r in report.items()},ensure_ascii=False))
if __name__=='__main__':main()

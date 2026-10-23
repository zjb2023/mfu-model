"""Restore semantic zero-cost expert slots only for a proven empty subtree.

No missing nonzero work can enter this path. All original timing endpoints
remain unchanged; inserted nodes carry CPU-zero proof, never fake GPU IDs.
"""
import argparse,copy,re
from calibrate256_fine_r1 import BASE,load,save,sha
from build_b_cp_pair_r4 import run
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank-base',type=int,required=True);ap.add_argument('--mb',type=int,default=0);a=ap.parse_args()
    root=BASE/'calibrate256-zero-r2'/f'r{a.rank_base}-mb{a.mb}';p=root/'b-recompute-r12/graph.json';g=load(p);old=run(g);changes=[]
    candidates=[k for k,n in g.items() if re.fullmatch(r'R\d+:r\d+:s0:n20',k) and n.get('kind')=='recompute_expert_FC2' and n['cost_ms']==0]
    for k in candidates:
        pre=k.rsplit('n',1)[0]+'n';keys=[pre+str(i) for i in range(17,21)]
        assert [g[x]['kind'] for x in keys]==['recompute_expert_FC1','recompute_local_compute','recompute_local_reorder','recompute_expert_FC2']
        assert all(g[x]['cost_ms']==0 and g[x]['events'] and all(e is None for e in g[x]['events']) for x in keys)
        assert pre+'21' not in g and pre+'22' not in g
        # Target zero FC2 is node20 because the zero gate/cast slot emitted no
        # separate GPU block. Give it the shared-template semantic node21.
        mapping={k:pre+'21',k+':ready':pre+'21:ready'}
        g={mapping.get(x,x):dict(n,deps=[mapping.get(d,d) for d in n['deps']]) for x,n in g.items()}
        for i,dep in [(20,pre+'19'),(22,pre+'21')]:
            key=pre+str(i);g[key+':ready']=dict(deps=[dep],cost_ms=0,kind='recompute_readiness',policy='derived_zero_work_boundary')
            g[key]=dict(deps=[key+':ready'],cost_ms=0,kind='recompute_local_compute',policy='derived_zero_work',events=[],basis='zero expert-input subtree; no device activity, logical gate/cast or unpermute slot retained')
        n=g[pre+'21:ready'];n['deps']=[pre+'20' if d==pre+'19' else d for d in n['deps']]
        match=re.match(r'R(\d+):r(\d+):',k);u,r=match.groups()
        g[f'r{r}:u{u}:recompute']['deps'].append(pre+'22')
        g[f'R{u}:EP8:combine_effective']['deps'].append(pre+'22')
        changes.append(dict(rank=int(r),unit=int(u),old_FC2=k,new_FC2=pre+'21',zero_slots=[pre+'20',pre+'22']))
    t=run(g);mapping={c['old_FC2']:c['new_FC2'] for c in changes};mapping.update({k+':ready':v+':ready' for k,v in list(mapping.items())})
    err=max(abs(t[mapping.get(k,k)]['end_ms']-v['end_ms']) for k,v in old.items());assert err<1e-5
    dest=root/'b-recompute-r12-zero-normalized';dest.mkdir(exist_ok=False)
    save(dest/'graph.json',g);save(dest/'report.json',dict(status='PASS_PROVEN_ZERO_SUBTREE_NORMALIZATION',changes=changes,max_existing_endpoint_error_ms=err,B_ms=t['END']['end_ms'],limits=['Zero semantic slots are not observed GPU events.','Only zero-cost, virtual CPU-proven FC/activation subtree accepted.','Exact shared-edge audit still required.']))
    save(dest/'manifest.json',dict(inputs={str(p):sha(p)},code_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file()}));print('PASS zero normalization',len(changes),t['END']['end_ms'])
if __name__=='__main__':main()

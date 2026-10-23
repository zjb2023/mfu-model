"""One observed compact nonzero expert sequence; preserve every event/cost.

FC1 -> fused activation/casts -> FC2 -> unpermute. Optional standalone
reorder/cast slots are empty: their work must not be charged twice.
"""
import sys
from calibrate256_fine_r1 import BASE,load,save,sha
from build_b_cp_pair_r4 import run
from pathlib import Path
import audit256_fine_binding_r1 as audit
def main():
    root=BASE/'calibrate256-combined-r3/r232-mb0'
    p=root/'b-recompute-r12-zero-normalized/graph.json';g=load(p);old=run(g)
    pre='R3:r233:s0:n'
    assert [g[pre+str(i)]['kind'] for i in range(17,21)]==['recompute_expert_FC1','recompute_local_compute','recompute_expert_FC2','recompute_local_compute']
    assert len(g[pre+'17']['events'])==1 and len(g[pre+'19']['events'])==1
    assert any('SiluOp' in n for n in g[pre+'18']['names'])
    assert any('CastOp' in n for n in g[pre+'18']['names'])
    assert 'moe_unpermute' in g[pre+'20']['names'][0]
    mapping={pre+'19':pre+'21',pre+'20':pre+'22'}
    mapping.update({k+':ready':v+':ready' for k,v in list(mapping.items())})
    g={mapping.get(k,k):dict(n,deps=[mapping.get(d,d) for d in n['deps']]) for k,n in g.items()}
    for i,kind in [(19,'recompute_local_reorder'),(20,'recompute_local_compute')]:
        g[pre+str(i)]=dict(deps=[pre+str(i-1)],cost_ms=0,kind=kind,events=[],basis='no standalone device block; activation/casts accounted in n18; FC2 and unpermute retain all observed events')
    g[pre+'21:ready']['deps']=[pre+'20' if d==pre+'18' else d for d in g[pre+'21:ready']['deps']]
    now=run(g);err=max(abs(now[mapping.get(k,k)]['end_ms']-v['end_ms']) for k,v in old.items());assert err<1e-5
    dest=root/'b-recompute-r12-compact-normalized';dest.mkdir(exist_ok=False)
    save(dest/'graph.json',g);save(dest/'report.json',dict(max_existing_endpoint_error_ms=err,mapping=mapping,source=str(p),basis='observed compact nonzero expert sequence, no new cost or GPU events'))
    save(dest/'manifest.json',dict(input_sha256=sha(p),code_sha256=sha(Path(__file__))))
    code=Path(audit.__file__).read_text().replace("root=BASE/'calibrate256-fine-r1'","root=BASE/'calibrate256-combined-r3'").replace("('B','b-recompute-r12')","('B','b-recompute-r12-compact-normalized')")
    sys.argv=['audit','--rank-base','232','--mb','0','--model-dir',str(BASE/'iter60-shared-dag-r2'),'--tag','shared-binding-r4']
    ns=dict(audit.__dict__);ns['__name__']='compact_adapter';exec(compile(code,audit.__file__,'exec'),ns);ns['main']()
if __name__=='__main__':main()

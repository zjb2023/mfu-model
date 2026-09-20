"""Four-layer hybrid B: replace backward interiors, retain measured recompute/head/tail."""
import argparse,copy,hashlib,json,re
from pathlib import Path
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    sk=BASE/'b-four-unit-skeleton-r9';g=json.loads((sk/'graph.json').read_text());old=run(g)
    evidence=BASE/'b-cost-blocks-r1/evidence.json';xs=json.loads(evidence.read_text());origin=min(d['start_ms'] for x in xs for d in x['devices'])
    expected={};inputs=[sk/'graph.json',evidence];coverage=[]
    for unit in range(4):
        ep=BASE/('b-ep8-coupling-r7' if unit==0 else f'b-four-layer-r10-u{unit}-ep')
        moe=BASE/('b-moe-backward-r6-local-handoff' if unit==0 else f'b-four-layer-r10-u{unit}-moe')
        sub=json.loads((ep/'graph.json').read_text());st=run(sub);ev=json.loads((moe/'evidence.json').read_text());offset=min(d['start_us'] for x in ev for d in x['devices'])/1000-origin
        inputs.extend([ep/'graph.json',ep/'manifest.json',moe/'evidence.json'])
        def rankof(k):
            hit=re.search(r'(?:^|:)r(\d+):',k);return int(hit.group(1)) if hit else None
        for k,n in sub.items():
            key=f'U{unit}:{k}';g[key]={**n,'deps':[f'U{unit}:{d}' for d in n['deps']],'unit':unit,'rank':rankof(k)}
            if not n['deps']:
                rank=rankof(k);assert rank is not None
                start=old[f'r{rank}:u{unit}:backward']['start_ms']
                value=n['cost_ms']+offset-start;assert value>=-1e-5
                g[key].update(deps=[f'r{rank}:u{unit}:backward:handoff'],cost_ms=max(0,value),basis='per-stream backward head plus readiness; trace reserved')
            expected[key]=st[k]['end_ms']+offset
        for rank in range(8,16):
            ks=[f'U{unit}:{k}' for k in sub if rankof(k)==rank];end=max(expected[k] for k in ks)
            key=f'r{rank}:u{unit}:backward';tail=old[key]['end_ms']-end;assert tail>=-1e-5
            # Replace the old envelope; never retain its duration alongside detailed nodes.
            g[key].update(deps=ks,cost_ms=max(0,tail),policy='trace_reserved_backward_tail',kind='backward_boundary')
            coverage.append(dict(rank=rank,unit=unit,detail_start_ms=min(st[k]['start_ms']+offset for k in sub if rankof(k)==rank),detail_end_ms=end,tail_ms=tail,phase_start_ms=old[key]['start_ms'],phase_end_ms=old[key]['end_ms']))
    t=run(g)
    for k,v in expected.items():assert abs(t[k]['end_ms']-v)<1e-5,(k,t[k],v)
    for k,v in old.items():assert abs(t[k]['end_ms']-v['end_ms'])<1e-5
    tests=[]
    for unit in range(4):
        for rank in range(8,16):
            ks=[k for k,n in g.items() if k.startswith(f'U{unit}:moe:r{rank}:') and n.get('kind','').startswith('expert_grouped')]
            assert len(ks)==2
            for k in ks:
                c=copy.deepcopy(g);c[k]['cost_ms']+=10;s=run(c)
                shift=s['END']['end_ms']-t['END']['end_ms'];assert shift>=-1e-7
                downstream={f'r{r}:u{u}':s[f'r{r}:u{u}:backward']['end_ms']-t[f'r{r}:u{u}:backward']['end_ms'] for u in range(unit,4) for r in range(8,16)}
                assert min(downstream.values())>=-1e-7
                c[k]['cost_ms']+=40;large=run(c);large_shift=large['END']['end_ms']-t['END']['end_ms'];assert large_shift>0
                assert all(large[f'r{r}:u{u}:backward']['end_ms']>t[f'r{r}:u{u}:backward']['end_ms'] for u in range(unit,4) for r in range(8,16))
                tests.append(dict(node=k,increment_ms=10,B_shift_ms=shift,downstream=downstream,stress_increment_ms=50,stress_B_shift_ms=large_shift))
    summary=dict(status='PASS_HYBRID_REPLAY_PARTIAL_PREDICTOR',B_ms=t['END']['end_ms'],detailed_boundary_checks=len(expected),phase_boundary_checks=64,cost_tests=len(tests),graph_nodes=len(g),limitations=['Recompute envelopes and backward heads/tails remain trace-reserved.','EP8/CP2 joins and local handoffs remain conditional completion models.','Same-sample replay, not iter70 or 256 validation.'])
    a.out.mkdir(parents=True)
    for n,v in [('graph',g),('replay',t),('coverage',coverage),('tests',tests),('summary',summary)]: (a.out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs += [Path(__file__).resolve(),Path(__file__).with_name('build_b_cp_pair_r4.py')]
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=[rec(p) for p in inputs],outputs=[rec(p) for p in a.out.iterdir()]),indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()

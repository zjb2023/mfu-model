"""EP8 completion coupling candidate, explicitly not a measured physical barrier."""
import argparse,copy,hashlib,json
from pathlib import Path
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source',type=Path,default=BASE/'b-moe-backward-r6-local-handoff');a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    src=a.source;unit=json.loads((src/'summary.json').read_text()).get('unit',0);g=json.loads((src/'graph.json').read_text());baseline=run(g)
    assert baseline==json.loads((src/'replay.json').read_text())
    readiness={};outputs={};experts={};audit=[]
    for r in range(8,16):
        ns={k:n for k,n in g.items() if k.startswith(f'moe:r{r}:') and not k.endswith(':ready')}
        dispatch=next(k for k,n in ns.items() if n['kind']=='backward_dispatch_visible');commstream=dispatch.split(':')[2]
        out=next(k for k,n in ns.items() if n['kind']=='backward_combine_visible' and k.split(':')[2]==commstream)
        prep=next(k for k,n in ns.items() if n['kind']=='backward_combine_visible' and k!=out)
        ex=sorted([k for k,n in ns.items() if n['kind'].startswith('expert_grouped')],key=lambda k:baseline[k]['end_ms'])
        assert len(ex)==2
        readiness[r]=prep;outputs[r]=out;experts[r]=ex
    latest=max(baseline[k]['end_ms'] for k in readiness.values());first=min(baseline[k]['start_ms'] for k in outputs.values());residual=first-latest
    assert residual>=0
    gate='EP8:combine_effective_completion_gate'
    g[gate]=dict(deps=list(readiness.values()),cost_ms=residual,policy='external_effective_residual_replace',basis='conditional all-local-ready to earliest visible unpermute; NOT pure transport service',scope='EP8 ranks8..15, true backward unit0')
    for r,k in outputs.items():
        entry=k+':ready';offset=baseline[k]['start_ms']-first
        # Replace the old local readiness residual; never add both old and group costs.
        g[entry].update(deps=[gate],cost_ms=offset,policy='trace_reserved',basis='per-rank visible Combine start offset after conditional shared gate')
        audit.append(dict(rank=r,expert_end_ms=baseline[experts[r][-1]]['end_ms'],local_ready_ms=baseline[readiness[r]]['end_ms'],ready_slack_ms=latest-baseline[readiness[r]]['end_ms'],combine_start_ms=baseline[k]['start_ms'],combine_end_ms=baseline[k]['end_ms'],start_offset_ms=offset))
    replay=run(g)
    for k,t in baseline.items():assert abs(t['end_ms']-replay[k]['end_ms'])<1e-5,(k,t,replay[k])
    tests=[]
    for r in range(8,16):
        for expert in experts[r]:
            for extra in [1,10]:
                c=copy.deepcopy(g);c[expert]['cost_ms']+=extra;t=run(c)
                expected=max(latest,baseline[readiness[r]]['end_ms']+extra)-latest
                shifts={str(other):t[f'r{other}:combine_trace_anchor']['end_ms']-replay[f'r{other}:combine_trace_anchor']['end_ms'] for other in range(8,16)}
                assert all(abs(v-expected)<1e-5 for v in shifts.values()),(r,expert,extra,shifts,expected)
                attention={str(other):max(t[k]['end_ms'] for k,n in g.items() if f':r{other}:' in k and n.get('kind')=='attention_backward')-max(replay[k]['end_ms'] for k,n in g.items() if f':r{other}:' in k and n.get('kind')=='attention_backward') for other in range(8,16)}
                assert all(abs(v-expected)<1e-5 for v in attention.values())
                tests.append(dict(rank=r,node=expert,increment_ms=extra,expected_group_delay_ms=expected,combine_shift_ms=shifts,attention_shift_ms=attention))
    c=copy.deepcopy(g);c[gate]['cost_ms']+=1;t=run(c)
    assert all(abs(t[k]['end_ms']-replay[k]['end_ms']-1)<1e-5 for k in outputs.values())
    summary=dict(status='PARTIAL_CONDITIONAL_EP8_COUPLING',scope='32GPU iter60 PP1 B0 unit0 true backward; no recompute',baseline_checks='PASS all prior node completion times',single_rank_cost_tests=len(tests),cost_tests='PASS 32 tests across both grouped-linear blocks; combine and attention on all 8 ranks match slack-aware prediction',group_residual_plus1='PASS all 8 Combine endpoints +1ms',latest_ready_ms=latest,earliest_visible_combine_start_ms=first,effective_residual_ms=residual,latest_ready_rank=max(readiness,key=lambda r:baseline[readiness[r]]['end_ms']),limitations=['All-local-ready group join is a coarse completion model assumption, not proven EP transport barrier.','Visible preparation end is a proxy, not measured payload send-ready.','Effective residual may include synchronization/processing/transport; not interchangeable with backend pure service.','Expert perturbations use fixed routing/token distribution and reserved costs.','No independent prediction accuracy or complete B/256 validation.'])
    summary['scope']=f'32GPU iter60 PP1 B0 unit{unit} true backward; no recompute'
    g[gate]['scope']=summary['scope']
    a.out.mkdir(parents=True)
    for n,v in [('graph',g),('replay',replay),('pairwise-audit',audit),('perturbation-tests',tests),('summary',summary)]: (a.out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(dict(version='b-ep8-coupling-r7',inputs=[rec(p) for p in [src/'graph.json',src/'replay.json',src/'manifest.json',Path(__file__).resolve(),Path(__file__).with_name('build_b_cp_pair_r4.py')]],outputs=[rec(p) for p in a.out.iterdir()]),indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()

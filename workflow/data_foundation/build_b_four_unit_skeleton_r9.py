"""Four-unit bookkeeping gate: trace phase envelopes, not a cost prediction model."""
import argparse,hashlib,json
from pathlib import Path
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    src=BASE/'b-cost-blocks-r1/evidence.json';xs=json.loads(src.read_text());origin=min(d['start_ms'] for x in xs for d in x['devices']);g={};audit=[];outside=[]
    for x in xs:
        rank=x['rank'];cursor=0;prev=None;seen=set()
        for unit in range(4):
            for phase in ['recompute','backward']:
                ds=[d for d in x['devices'] if d['unit']==unit and d['phase']==phase];assert ds
                ids={d['event'] for d in ds};assert len(ids)==len(ds) and not seen.intersection(ids);seen.update(ids)
                lo=min(d['start_ms'] for d in ds)-origin;hi=max(d['end_ms'] for d in ds)-origin
                assert lo>=cursor-1e-5,(rank,unit,phase,'phase envelopes overlap',lo,cursor)
                key=f'r{rank}:u{unit}:{phase}';ready=key+':handoff'
                g[ready]=dict(deps=[prev] if prev else [],cost_ms=max(0,lo-cursor),policy='trace_reserved',basis='observed phase envelope handoff, not pure wait')
                g[key]=dict(deps=[ready],cost_ms=hi-lo,kind=phase,unit=unit,rank=rank,policy='trace_envelope_only_NOT_external_cost',observed_start_ms=lo,observed_end_ms=hi)
                audit.append(dict(rank=rank,unit=unit,phase=phase,events=len(ds),start_ms=lo,end_ms=hi,duration_ms=hi-lo,handoff_ms=max(0,lo-cursor)))
                cursor=hi;prev=key
        remaining=[d for d in x['devices'] if d['event'] not in seen]
        # Outside-unit events are explicit, not silently attributed to a layer.
        outside.append(dict(rank=rank,events=remaining))
        rank_end=max(d['end_ms'] for d in x['devices'])-origin
        assert rank_end>=cursor-1e-5
        g[f'r{rank}:end']=dict(deps=[prev],cost_ms=max(0,rank_end-cursor),policy='trace_reserved_outside_unit_tail')
    g['END']=dict(deps=[f'r{r}:end' for r in range(8,16)],cost_ms=0)
    t=run(g)
    for k,n in g.items():
        if 'observed_end_ms' in n:assert abs(t[k]['end_ms']-n['observed_end_ms'])<1e-5
    old=BASE/'b-cost-blocks-r1/summary.json';oldsummary=json.loads(old.read_text());assert abs(t['END']['end_ms']-oldsummary['B_device_window_ms'])<1e-5
    assert len(audit)==64
    summary=dict(status='PASS_PHASE_ASSEMBLY_GATE_NOT_PREDICTION',ranks=8,units_per_rank=4,phase_envelopes=64,B_window_ms=t['END']['end_ms'],boundary_checks='PASS 64 phase ends and prior B device window',order='u0 recompute/backward -> u1 recompute/backward -> u2 recompute/backward -> u3 recompute/backward',outside_unit_event_count=sum(len(x['events']) for x in outside),limitations=['Envelope graph is accounting only; do not add its costs on top of fine-grained modules.','No global barrier between ranks introduced; per-rank phase envelope order matches this trace only.','Unit-to-local-layer reverse mapping is framework-supported inference, not trace module-id proof.','r7 fine-grained unit0 has not yet been generalized/integrated for all four units.','No new 32/256 predictive accuracy.'])
    a.out.mkdir(parents=True)
    for n,v in [('graph',g),('replay',t),('phase-audit',audit),('outside-units',outside),('summary',summary)]: (a.out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    framework=Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/transformer_block.py')
    def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=[rec(p) for p in [src,old,framework,Path(__file__).resolve(),Path(__file__).with_name('build_b_cp_pair_r4.py')]],outputs=[rec(p) for p in a.out.iterdir()]),indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()

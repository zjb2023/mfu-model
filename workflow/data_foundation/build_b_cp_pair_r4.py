"""Conditional pair completion coupling, not inferred physical network service."""
import argparse,copy,graphlib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def run(g):
    out={}
    for k in graphlib.TopologicalSorter({k:n['deps'] for k,n in g.items()}).static_order():
        n=g[k];s=max([out[d]['end_ms'] for d in n['deps']] or [0]);out[k]=dict(start_ms=s,end_ms=s+n['cost_ms'])
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    source=BASE/'b-attention-replay-r3/graphs.json';structure=BASE/'b-attention-structure-r2/evidence.json'
    gs=json.loads(source.read_text());us=json.loads(structure.read_text());um={(u['rank'],u['unit']):u for u in us}
    result={};pairs=[];tests=[]
    for rank in [8,10,12,14]:
        for unit in range(4):
            xs=[gs[f'r{r}:u{unit}'] for r in [rank,rank+1]];origin=min(x['origin_us'] for x in xs);g={};cp={};flash={}
            for r,x in zip([rank,rank+1],xs):
                assert um[r,unit]['cp_group']==[rank,rank+1]
                offset=(x['origin_us']-origin)/1000;root=f'r{r}:origin'
                g[root]=dict(deps=[],cost_ms=offset,policy='trace_reserved')
                for k,n in x['nodes'].items():
                    key=f'r{r}:{k}';ready=key+':ready'
                    g[ready]=dict(deps=[f'r{r}:{d}' for d in n['deps']] or [root],cost_ms=n['trace_ready_ms'],policy='trace_reserved',basis='source local readiness proxy, not proven producer completion')
                    g[key]=dict(deps=[ready],cost_ms=n['cost_ms'],policy=n['policy'],kind=n['kind'],observed_end_ms=offset+n['end_ms'])
                    if n['kind'].startswith('CP'):cp[r,n['kind']]=(key,offset+n['start_ms'],offset+n['end_ms'])
                    if n['kind']=='attention_backward':flash[r]=key
            for j in range(5):
                kind=f'CP{j}';entries=[cp[r,kind] for r in [rank,rank+1]];latest=max(t[1] for t in entries);first_end=min(t[2] for t in entries)
                assert first_end>=latest
                shared=f'pair:{kind}'
                g[shared]=dict(deps=[t[0]+':ready' for t in entries],cost_ms=first_end-latest,policy='external_effective_pair_overlap_replace',basis='common device overlap proxy NOT pure service; conditional all-arrival completion model')
                for key,start,end in entries:
                    g[key].update(deps=[shared],cost_ms=end-first_end,policy='trace_reserved',basis='rank completion offset; old CP envelope removed, NOT added')
                pairs.append(dict(pair=[rank,rank+1],unit=unit,ordinal=j,start_skew_ms=abs(entries[0][1]-entries[1][1]),common_overlap_proxy_ms=first_end-latest,end_skew_ms=abs(entries[0][2]-entries[1][2])))
            baseline=run(g)
            for k,n in g.items():
                if 'observed_end_ms' in n:assert abs(baseline[k]['end_ms']-n['observed_end_ms'])<1e-5,(k,baseline[k],n)
            # Each common cost increment must reach both corresponding completions.
            for j in range(5):
                c=copy.deepcopy(g);c[f'pair:CP{j}']['cost_ms']+=1;t=run(c)
                for r in [rank,rank+1]:
                    k=cp[r,f'CP{j}'][0];assert abs(t[k]['end_ms']-baseline[k]['end_ms']-1)<1e-6
            # Delay either rank's attention enough to exceed existing skew.
            for slow in [rank,rank+1]:
                c=copy.deepcopy(g);c[flash[slow]]['cost_ms']+=2;t=run(c)
                shifts={str(r):t[cp[r,'CP2'][0]]['end_ms']-baseline[cp[r,'CP2'][0]]['end_ms'] for r in [rank,rank+1]}
                assert min(shifts.values())>0
                tests.append(dict(pair=[rank,rank+1],unit=unit,slowed_rank=slow,attention_increment_ms=2,CP2_completion_shift_ms=shifts))
            result[f'p{rank}-{rank+1}:u{unit}']=dict(origin_us=origin,graph=g,replay=baseline)
    summary=dict(status='PARTIAL_CONDITIONAL_PAIR_DAG',pair_units=16,matched_collectives=len(pairs),baseline='PASS all 352 original block ends',cost_tests='PASS 80 common-cost +1 tests, both ranks',coupling_tests='PASS 32 single-rank attention +2 tests affect partner CP2',max_start_skew_ms=max(p['start_skew_ms'] for p in pairs),max_end_skew_ms=max(p['end_skew_ms'] for p in pairs),limitations=['Kernel launch/start is a local entry proxy, not proven send-ready or all-arrival timestamp.','Pair overlap is not backend service and can include synchronization or processing.','Conditional pair coupling is an explicit model assumption, not recovered event handles.','No EP backward integration or independent prediction validation.'])
    a.out.mkdir(parents=True)
    for n,v in [('graphs',result),('pair-audit',pairs),('coupling-tests',tests),('summary',summary)]: (a.out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    def record(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(dict(version='b-cp-pair-r4',inputs=[record(p) for p in [source,structure,BASE/'b-attention-replay-r3/manifest.json',Path(__file__).resolve()]],outputs=[record(p) for p in a.out.iterdir()]),indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()

"""B0 stream-local cost blocks: executable scoped interface, not closed cross-rank DAG."""
import argparse,collections,graphlib,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--overrides',type=Path);a=ap.parse_args()
    xs=[];inputs=[]
    for r in range(8,16):
        p=BASE/f'pp32-f-split-r8/rank-{r}.json';meta=json.loads(p.read_text());raw=Path(meta['path']);blob=raw.read_bytes();digest=hashlib.sha256(blob).hexdigest();assert digest==meta['sha256'];es=json.loads(blob)['traceEvents'];del blob
        bi,b=next((i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name')=='backward_step')
        def inside(e,v):return e.get('pid')==v['pid'] and v['ts']<=e.get('ts',0) and e.get('ts',0)+e.get('dur',0)<=v['ts']+v.get('dur',0)+.01
        checkpoints=[(i,e) for i,e in enumerate(es) if e.get('name')=='CheckpointFunctionBackward' and e.get('cat')=='cpu_op' and inside(e,b)];checkpoints.sort(key=lambda z:z[1]['ts']);assert len(checkpoints)==4
        marks=[]
        for i,e in enumerate(es):
            if e.get('cat')!='cpu_op' or not inside(e,b):continue
            name=e.get('name');args=e.get('args',{})
            if name in ['FusedDispatch','FusedCombine','FusedCombineBackward','FusedDispatchBackward']:kind={'FusedDispatch':'EP_dispatch_recompute','FusedCombine':'EP_combine_recompute','FusedCombineBackward':'EP_dispatch_backward','FusedDispatchBackward':'EP_combine_backward'}[name]
            elif name=='record_param_comms':kind='COMM_'+str(args.get('Collective name','unknown'))
            else:continue
            marks.append(dict(event=i,kind=kind,e=e))
        marks.sort(key=lambda m:m['e']['ts']);runtime=collections.defaultdict(list)
        for i,e in enumerate(es):
            if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append((i,e))
        devices=[]
        for i,e in enumerate(es):
            if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
            calls=[(j,v) for j,v in runtime.get(e.get('args',{}).get('correlation'),[]) if inside(v,b)]
            if not calls:continue
            ri,call=min(calls,key=lambda z:z[1].get('dur',0))
            unit=next((j for j,(_,c) in enumerate(checkpoints) if inside(call,c)),None)
            hits=[m for m in marks if inside(call,m['e'])];hit=min(hits,key=lambda m:m['e']['dur']) if hits else None
            prior=[m for m in marks if m['e']['ts']+m['e']['dur']<=call['ts']]
            # Use CPU semantic marker boundaries to prevent merging unrelated computation.
            anchor=hit['event'] if hit else (prior[-1]['event'] if prior else 'entry')
            kind=hit['kind'] if hit else ('local_memory' if e['cat']!='kernel' else 'local_compute_envelope')
            args=e.get('args',{})
            if args.get('Process Group Description')=='CONTEXT_PARALLEL_GROUP':kind='CP_'+str(args.get('Collective name','unknown'))
            phase='backward_or_epilogue'
            if unit is not None:
                fcs=[m for m in marks if m['kind']=='EP_combine_recompute' and inside(m['e'],checkpoints[unit][1])]
                if fcs and call['ts']<=max(m['e']['ts']+m['e']['dur'] for m in fcs):phase='recompute'
                else:phase='backward'
            devices.append(dict(event=i,launch_event=ri,name=e['name'],start_ms=e['ts']/1000,end_ms=(e['ts']+e['dur'])/1000,stream=str(args.get('stream',e.get('tid'))),kind=kind,unit=unit,phase=phase,anchor=anchor,collective=args.get('Collective name'),group=args.get('Process Group Description')))
        assert devices
        counts=collections.Counter(m['kind'] for m in marks)
        for kind in ['EP_dispatch_recompute','EP_combine_recompute','EP_dispatch_backward','EP_combine_backward']:assert counts[kind]==4
        xs.append(dict(rank=r,path=str(raw),sha256=digest,cpu_B_event=bi,checkpoints=[i for i,_ in checkpoints],marker_counts=dict(counts),devices=devices));inputs.extend([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),dict(path=str(raw),sha256=digest)])
        print('B0 extracted rank',r,flush=True)
    origin=min(d['start_ms'] for x in xs for d in x['devices']);g={};bindings={};ends=[];stats=[]
    for x in xs:
        r=x['rank'];runs=[]
        for stream in sorted({d['stream'] for d in x['devices']}):
            ds=sorted([d for d in x['devices'] if d['stream']==stream],key=lambda d:d['start_ms']);prev=None;cursor=0
            groups=[]
            for d in ds:
                signature=(d['kind'],d['unit'],d['phase'],d['anchor'])
                if groups and groups[-1]['signature']==signature:groups[-1]['devices'].append(d)
                else:groups.append(dict(signature=signature,devices=[d]))
            for j,group in enumerate(groups):
                dev=group['devices'];lo=min(d['start_ms'] for d in dev)-origin;hi=max(d['end_ms'] for d in dev)-origin;k=f'r{r}:s{stream}:block{j}';kind,unit,phase,_=group['signature']
                # Same-stream overlap beyond precision would invalidate this construction.
                assert lo>=cursor-1e-5,(r,stream,lo,cursor)
                gap=k+':trace_ready';g[gap]=dict(deps=[prev] if prev else [],cost_ms=max(0,lo-cursor),kind='trace_ready',rank=r,stream=stream,unit=unit,phase=phase)
                g[k]=dict(deps=[gap],cost_ms=hi-lo,kind=kind,rank=r,stream=stream,unit=unit,phase=phase,device_events=[d['event'] for d in dev],names=sorted({d['name'] for d in dev}),observed_start_ms=lo,observed_end_ms=hi)
                g[k]['evidence']='same GPU stream order; CPU marker identity; envelope includes internal gaps'
                bindings[k]=dict(value_ms=hi-lo,policy='external_effective_replace' if kind.startswith(('CP_','EP_','COMM_','local_compute')) else 'trace_reserved',cost_basis='effective device envelope, NOT pure compute/transport service')
                bindings[gap]=dict(value_ms=max(0,lo-cursor),policy='trace_reserved',cost_basis='source trace stream readiness, may encode missing cross-stream dependency')
                prev=k;cursor=hi;runs.append(k)
            ends.append(prev)
        stats.append(dict(rank=r,blocks=len(runs),B_start_ms=min(d['start_ms'] for d in x['devices'])-origin,B_end_ms=max(d['end_ms'] for d in x['devices'])-origin,marker_counts=x['marker_counts']))
    g['END']=dict(deps=ends,cost_ms=0,kind='boundary',rank=None);bindings['END']=dict(value_ms=0,policy='trace_reserved')
    def schedule(graph):
        t={}
        for k in graphlib.TopologicalSorter({k:n['deps'] for k,n in graph.items()}).static_order():
            n=graph[k];s=max([t[d]['end_ms'] for d in n['deps']] or [0]);t[k]=dict(start_ms=s,end_ms=s+n['cost_ms'])
        return t
    baseline=schedule(g)
    for k,n in g.items():
        if 'observed_end_ms' in n:assert abs(baseline[k]['end_ms']-n['observed_end_ms'])<1e-5
    if a.overrides:
        o=json.loads(a.overrides.read_text());assert o['basis']=='effective_replacement' and o['unit']=='ms'
        for k,v in o['nodes'].items():
            if k not in bindings or bindings[k]['policy']!='external_effective_replace' or type(v) not in [int,float] or not math.isfinite(v) or v<0:raise ValueError('invalid override '+k)
            g[k]['cost_ms']=v
    replay=schedule(g);a.out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='PARTIAL_STREAM_LOCAL_COST_INTERFACE',scope='32GPU iter60 PP1 B0 ranks8..15',B_device_window_ms=baseline['END']['end_ms'],replay_ms=replay['END']['end_ms'],ranks=stats,checks='PASS same-stream baseline boundary replay',limitations=['Cross-stream and cross-rank CP/EP readiness edges NOT closed; stream ready costs preserve source trace only.','Do NOT insert this block graph into full 1F1B as a validated predictive B yet.','External replacements only test local stream cost sensitivity, not fully synchronized collective prediction.','No original aggregate B cost is added to this graph; pure service costs require decomposition.','Execution unit0..3 are checkpoint order, not verified model layer indices; recompute classification based on fused forward markers.'])
    for n,v in [('graph',g),('replay',replay),('baseline-replay',baseline),('cost-bindings',bindings),('evidence',xs),('summary',summary)]: (a.out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    framework=Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py')
    inputs.extend(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [Path(__file__).resolve(),framework])
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=inputs),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

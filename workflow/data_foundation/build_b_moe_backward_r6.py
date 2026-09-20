"""Join trace-conditioned MoE backward stream blocks to r5; no EP service fiction."""
import argparse,collections,copy,hashlib,json
from pathlib import Path
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--unit',type=int,choices=range(4),default=0);ap.add_argument('--bridge-dir',type=Path,default=BASE/'b-gradient-bridge-r5');a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    src=BASE/'b-cost-blocks-r1/evidence.json';bridge=a.bridge_dir/'bridges.json';oldgraph=a.bridge_dir/'graph.json'
    bs=json.loads(bridge.read_text());ranks=[];inputs=[]
    for x in json.loads(src.read_text()):
        p=Path(x['path']);blob=p.read_bytes();assert hashlib.sha256(blob).hexdigest()==x['sha256'];es=json.loads(blob)['traceEvents'];del blob
        inputs.append(dict(path=str(p),sha256=x['sha256']));ck=es[x['checkpoints'][a.unit]]
        def within(e,o):return e.get('pid')==o['pid'] and e.get('tid')==o['tid'] and o['ts']<=e.get('ts',0) and e.get('ts',0)+e.get('dur',0)<=o['ts']+o['dur']+.01
        owners=[(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and within(e,ck)]
        di,d=next((i,e) for i,e in owners if e['name']=='FusedCombineBackward');ci,c=next((i,e) for i,e in owners if e['name']=='FusedDispatchBackward')
        linears=[(i,e) for i,e in owners if e['name']=='_GroupedLinearBackward' and d['ts']<=e['ts']<c['ts']];assert len(linears)==2
        calls=collections.defaultdict(list)
        for i,e in enumerate(es):
            if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and e.get('pid')==d['pid'] and e.get('tid')==d['tid'] and d['ts']<=e.get('ts',0) and e.get('ts',0)+e.get('dur',0)<=c['ts']+c['dur']+.01:
                calls[e.get('args',{}).get('correlation')].append((i,e))
        devices=[]
        for i,e in enumerate(es):
            if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
            hits=calls.get(e.get('args',{}).get('correlation'),[])
            if not hits:continue
            _,call=min(hits,key=lambda z:z[1]['dur']);kind='local_autograd_processing'
            if within(call,d):kind='backward_dispatch_visible'
            elif within(call,c):kind='backward_combine_visible'
            else:
                for j,(_,op) in enumerate(linears):
                    if within(call,op):kind=f'expert_grouped_linear_backward_{j}'
            devices.append(dict(event=i,name=e['name'],start_us=e['ts'],end_us=e['ts']+e['dur'],stream=e['args'].get('stream',e['tid']),kind=kind,launch_events=[j for j,_ in hits]))
        assert devices
        combine_end=max(e['end_us'] for e in devices if e['kind']=='backward_combine_visible');b=next(b for b in bs if b['rank']==x['rank']);assert abs(combine_end-b['combine_visible_end_us'])<.01
        ranks.append(dict(rank=x['rank'],dispatch_cpu_event=di,combine_cpu_event=ci,grouped_linear_cpu_events=[i for i,_ in linears],devices=devices));print('PASS MoE unit0 rank',x['rank'],flush=True)
    origin=min(d['start_us'] for x in ranks for d in x['devices']);g={};expected={};combine_nodes={};counts=[]
    for x in ranks:
        r=x['rank'];combine_nodes[r]=[];streams=collections.defaultdict(list)
        for d in x['devices']:streams[d['stream']].append(d)
        blocks=0
        for stream,ds in streams.items():
            groups=[]
            for d in sorted(ds,key=lambda d:d['start_us']):
                if groups and groups[-1][0]==d['kind']:groups[-1][1].append(d)
                else:groups.append((d['kind'],[d]))
            prev=None;cursor=0
            for j,(kind,items) in enumerate(groups):
                start=(min(d['start_us'] for d in items)-origin)/1000;end=(max(d['end_us'] for d in items)-origin)/1000;assert start>=cursor-1e-5
                key=f'moe:r{r}:s{stream}:{j}';ready=key+':ready'
                g[ready]=dict(deps=[prev] if prev else [],cost_ms=max(0,start-cursor),policy='trace_reserved',basis='source stream readiness; can contain missing dependency, NOT pure idle')
                g[key]=dict(deps=[ready],cost_ms=end-start,policy='external_effective_replace' if kind.startswith('expert_grouped') else 'trace_reserved',kind=kind,events=[d['event'] for d in items],basis='same stream envelope; visible EP kernels are not full transport')
                expected[key]=end
                if kind=='backward_combine_visible':combine_nodes[r].append(key)
                prev=key;cursor=end;blocks+=1
        counts.append(dict(rank=r,blocks=blocks,devices=len(x['devices'])))
    # Explicit conditional local handoffs: preserve default starts by rebasing residuals.
    for x in ranks:
        r=x['rank'];local={k:n for k,n in g.items() if k.startswith(f'moe:r{r}:') and not k.endswith(':ready')}
        dispatch=next(k for k,n in local.items() if n['kind']=='backward_dispatch_visible')
        dispatch_stream=dispatch.split(':')[2]
        other=[k for k in local if k.split(':')[2]!=dispatch_stream]
        first=min(other,key=lambda k:expected[k]-g[k]['cost_ms'])
        combine=next(k for k in combine_nodes[r] if k.split(':')[2]==dispatch_stream)
        prep=max((k for k in combine_nodes[r] if k!=combine),key=lambda k:expected[k])
        for predecessor,target in [(dispatch,first),(prep,combine)]:
            ready=target+':ready';deps=g[ready]['deps']
            if predecessor not in deps:deps.append(predecessor)
            start=expected[target]-g[target]['cost_ms'];cost=start-max(expected[d] for d in deps)
            assert cost>=-1e-5
            g[ready].update(cost_ms=max(0,cost),basis='conditional local cross-stream handoff, not event-handle verified')
    oldorigin=min(b['combine_visible_end_us'] for b in bs);old=json.loads(oldgraph.read_text());oldreplay=run(old)
    for k,n in old.items():
        assert k not in g;g[k]=n
        if k.endswith(':combine_trace_anchor'):
            r=int(k.split(':')[0][1:]);g[k]={**n,'deps':combine_nodes[r],'cost_ms':0,'policy':'derived_visible_combine_exit','basis':'computed from visible MoE blocks; hidden EP transport still unresolved'}
        expected[k]=oldreplay[k]['end_ms']+(oldorigin-origin)/1000
    replay=run(g)
    for k,v in expected.items():assert abs(replay[k]['end_ms']-v)<1e-5,(k,replay[k]['end_ms'],v)
    sensitivity=[]
    for r in range(8,16):
        for k,n in g.items():
            if not k.startswith(f'moe:r{r}:') or not n.get('kind','').startswith('expert_grouped'):continue
            altered=copy.deepcopy(g);altered[k]['cost_ms']+=1;t=run(altered);anchor=f'r{r}:combine_trace_anchor'
            delta=t[anchor]['end_ms']-replay[anchor]['end_ms'];assert abs(delta-1)<1e-5
            sensitivity.append(dict(node=k,increment_ms=1,combine_completion_shift_ms=delta))
    summary=dict(status='PARTIAL_MOE_CONDITIONAL_REPLAY_CONNECTED',scope='iter60 PP1 B0 execution unit0 ranks8..15 TRUE backward only; recompute excluded',ranks=counts,replay='PASS visible MoE blocks and downstream r5 boundaries',sensitivity='PASS 16 grouped-linear +1 tests propagate to local Combine',limitations=['Local cross-stream handoffs are conditional, not event-handle verified.','EP8 transport/readiness edges not closed: visible permute/unpermute cannot stand for full service.','Stream readiness remains source trace residual; removing Combine timestamp anchor does NOT make model source-independent.','Grouped-linear costs cover associated device envelopes, not pure FLOP kernels.','No new 32/256 independent prediction accuracy.'])
    summary['scope']=f'iter60 PP1 B0 execution unit{a.unit} ranks8..15 TRUE backward only; recompute excluded'
    summary['unit']=a.unit
    a.out.mkdir(parents=True)
    for name,v in [('evidence',ranks),('graph',g),('replay',replay),('sensitivity',sensitivity),('summary',summary)]: (a.out/(name+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs.extend(rec(p) for p in [src,bridge,oldgraph,Path(__file__).resolve(),Path(__file__).with_name('build_b_cp_pair_r4.py')])
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,outputs=[rec(p) for p in a.out.iterdir()]),indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()

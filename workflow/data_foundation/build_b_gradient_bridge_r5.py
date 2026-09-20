"""Trace-anchored bridge into CP2 subgraphs; not a complete EP backward model."""
import argparse,collections,hashlib,json
from pathlib import Path
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--unit',type=int,choices=range(4),default=0);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    src=BASE/'b-cost-blocks-r1/evidence.json';pg=BASE/'b-cp-pair-r4/graphs.json';att=BASE/'b-attention-replay-r3/graphs.json'
    attgraphs=json.loads(att.read_text());bridges=[];inputs=[]
    for x in json.loads(src.read_text()):
        raw=Path(x['path']);blob=raw.read_bytes();assert hashlib.sha256(blob).hexdigest()==x['sha256'];es=json.loads(blob)['traceEvents'];del blob
        inputs.append(dict(path=str(raw),sha256=x['sha256']))
        checkpoint=es[x['checkpoints'][a.unit]]
        def within(e,o):return e.get('pid')==o['pid'] and e.get('tid')==o['tid'] and o['ts']<=e.get('ts',0) and e.get('ts',0)+e.get('dur',0)<=o['ts']+o['dur']+.01
        candidates=[(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e.get('name')=='FusedDispatchBackward' and within(e,checkpoint)]
        assert len(candidates)==1;oi,owner=candidates[0]
        calls=collections.defaultdict(list)
        for i,e in enumerate(es):
            if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}):calls[e['args']['correlation']].append((i,e))
        ds=[(i,e) for i,e in enumerate(es) if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']]
        combine=[(i,e) for i,e in ds if any(within(c,owner) for _,c in calls.get(e.get('args',{}).get('correlation'),[]))]
        assert combine
        lo=max(e['ts']+e['dur'] for _,e in combine);hi=attgraphs[f'r{x["rank"]}:u{a.unit}']['origin_us'];assert hi>=lo
        spans=[];evidence=[]
        for i,e in ds:
            if e['ts']>=hi or e['ts']+e['dur']<=lo:continue
            matched=calls.get(e.get('args',{}).get('correlation'),[])
            owned=any(within(c,checkpoint) for _,c in matched)
            evidence.append(dict(event=i,name=e['name'],start_us=e['ts'],end_us=e['ts']+e['dur'],stream=e['args'].get('stream'),checkpoint_owned=owned,launch_events=[j for j,_ in matched]))
            if owned:spans.append((max(lo,e['ts']),min(hi,e['ts']+e['dur'])))
        merged=[]
        for s,t in sorted(spans):
            if merged and s<=merged[-1][1]:merged[-1][1]=max(t,merged[-1][1])
            else:merged.append([s,t])
        bridges.append(dict(rank=x['rank'],combine_cpu_event=oi,combine_device_events=[i for i,_ in combine],combine_visible_end_us=lo,attention_entry_us=hi,bridge_ms=(hi-lo)/1000,active_union_ms=sum(t-s for s,t in merged)/1000,uncovered_ms=((hi-lo)-sum(t-s for s,t in merged))/1000,active_intervals_us=merged,events=evidence))
        print('bridge rank',x['rank'],'ms',(hi-lo)/1000,flush=True)
    origin=min(b['combine_visible_end_us'] for b in bridges);g={}
    for b in bridges:
        r=b['rank'];prev=f'r{r}:combine_trace_anchor';g[prev]=dict(deps=[],cost_ms=(b['combine_visible_end_us']-origin)/1000,policy='trace_reserved_anchor_not_predicted_EP')
        cursor=b['combine_visible_end_us'];segments=[]
        for s,t in b['active_intervals_us']:
            if s>cursor:segments.append(('uncovered_trace_residual',cursor,s))
            segments.append(('gradient_processing_exposed_union',s,t));cursor=t
        if cursor<b['attention_entry_us']:segments.append(('uncovered_trace_residual',cursor,b['attention_entry_us']))
        for j,(kind,s,t) in enumerate(segments):
            k=f'r{r}:bridge{j}';g[k]=dict(deps=[prev],cost_ms=(t-s)/1000,kind=kind,policy='trace_reserved',basis='clipped observed interval, not independently identified operator cost');prev=k
        b['entry_dep']=prev
    expected={}
    for p,x in json.loads(pg.read_text()).items():
        if not p.endswith(f':u{a.unit}'):continue
        for k,n in x['graph'].items():
            key=f'{p}:{k}';g[key]={**n,'deps':[f'{p}:{d}' for d in n['deps']]}
            if k.endswith(':origin'):
                r=int(k.split(':')[0][1:]);b=next(b for b in bridges if b['rank']==r);g[key].update(deps=[b['entry_dep']],cost_ms=0)
            if 'observed_end_ms' in n:expected[key]=n['observed_end_ms']+(x['origin_us']-origin)/1000
    replay=run(g)
    for k,v in expected.items():assert abs(replay[k]['end_ms']-v)<1e-5
    summary=dict(status='PARTIAL_TRACE_ANCHORED_EP8_TO_ATTENTION',ranks=8,unit=0,attention_block_boundary_checks=len(expected),replay_check='PASS',scope='visible backward Combine completion through local gradient bridge and Attention/CP, not entire B or complete MoE backward',limitations=['EP completion anchors are measured inputs, not computed cross-rank dependencies.','Exposed device unions preserve trace overlap; they are not pure compute costs.','Uncovered time is not automatically idle or communication.','CP pair coupling remains conditional. No iter70/256 data used.'])
    summary['unit']=a.unit
    a.out.mkdir(parents=True)
    for n,v in [('bridges',bridges),('graph',g),('replay',replay),('summary',summary)]: (a.out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs.extend(rec(p) for p in [src,pg,att,Path(__file__).resolve(),Path(__file__).with_name('build_b_cp_pair_r4.py')])
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,outputs=[rec(p) for p in a.out.iterdir()]),indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()

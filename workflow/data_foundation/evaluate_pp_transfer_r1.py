"""Compare frozen32 PP effective costs with bounded256 links; never fit target."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path
from pp32_rendezvous_r2 import message_ids
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';OLD=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def extract(world,rank,cache,meta):
    p=Path(meta['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==meta['sha256'];es=json.loads(raw)['traceEvents'];del raw
    step=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#'))
    if world==32:
        blockends={o['event']:o['absolute_end_ms'] for o in cache['ops'] if o['kind'] in ['F','B']};stage=cache['stage'];pp=4
    else:
        blockends={b['cpu_event']:b['end_absolute_ms'] for b in cache['blocks']};stage=cache['stage'];pp=16
        for b in cache['blocks']:
            e=es[b['last_device_event']];assert abs((e['ts']+e['dur'])/1000-b['end_absolute_ms'])<1e-5
    events=sorted([(i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and step['ts']<=e.get('ts',0)<step['ts']+step['dur'] and (i in blockends or any(t in e.get('name','') for t in ['send_forward','recv_forward','send_backward','recv_backward']))],key=lambda z:z[1]['ts'])
    comms=[(i,e) for i,e in enumerate(es) if e.get('name')=='record_param_comms' and e.get('args',{}).get('Collective name') in ['send','recv','isend','irecv'] and e['args'].get('Process Group Description')=='PIPELINE_MODEL_PARALLEL_GROUP']
    ready=step['ts']/1000;counter=collections.Counter();calls=[]
    for i,e in events:
        if i in blockends:ready=max(ready,blockends[i]);continue
        start=e['ts']/1000;end=(e['ts']+e['dur'])/1000;messages=message_ids(e['name'],stage,counter,pp)
        markers=[]
        for j,v in comms:
            if v.get('pid')==e['pid'] and e['ts']<=v['ts'] and v['ts']+v.get('dur',0)<=e['ts']+e['dur']+.01:
                a=v['args'];markers.append(dict(event=j,kind=a['Collective name'],nelems=a['In msg nelems'],dtype=a['dtype'],dims=a.get('Input Dims')))
        calls.append(dict(rank=rank,stage=stage,host=p.parts[-4],event=i,name=e['name'],start_ms=start,end_ms=end,ready_ms=max(start,ready),messages=messages,markers=markers))
        ready=max(ready,end)
    return dict(world=world,rank=rank,stage=stage,host=p.parts[-4],path=str(p),sha256=digest,calls=calls)
def pairs(xs):
    by=collections.defaultdict(list)
    for x in xs:
        for c in x['calls']:
            for msg,side in c['messages']:by[msg].append(dict(side=side,**c))
    result=[];unpaired=[]
    for msg,ends in by.items():
        if len(ends)!=2:unpaired.append(msg);continue
        assert sorted(e['side'] for e in ends)==['recv','send']
        start=max(e['ready_ms'] for e in ends);end=min(e['end_ms'] for e in ends)
        signatures=sorted({(m['nelems'],m['dtype']) for e in ends for m in e['markers']});assert signatures and all(dtype=='BFloat16' for _,dtype in signatures)
        result.append(dict(message=msg,phase=msg[0],ends=ends,standalone=all(len(e['messages'])==1 for e in ends),raw_overlap_ms=min(e['end_ms'] for e in ends)-max(e['start_ms'] for e in ends),ready_overlap_ms=end-start,completion_skew_ms=abs(ends[0]['end_ms']-ends[1]['end_ms']),payload_bytes=[n*2 for n,dtype in signatures],path_class='cross_host' if ends[0]['host']!=ends[1]['host'] else 'same_host'))
    return result,unpaired
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    params=BASE/'pp32-detailed-fb-assembly-r1/parameters.json';frozen=load(params)['pp_effective_overlap_ms']
    modelpaths=[BASE/'four-layer-f-r1/graph.json',BASE/'b-four-layers-r10/graph.json',params]
    seal={str(p):sha(p) for p in modelpaths}
    dump(out/'protocol.json',dict(source=32,target=256,iteration=60,source_ranks=[0,8,16,24],target_ranks=[0,16,112,128,224,240],target_links=[[0,16],[112,128],[224,240]],metric='min(endpoint CPU end) - max(endpoint call start, preceding associated GPU completion, preceding PP completion)',selection='both ends standalone only; combined calls reported but excluded from cost comparison',cost_basis='effective overlap proxy, NOT pure network service',target_fit=False,clock_assumption='cross-host timestamps assumed comparable; no clock correction fitted',frozen_costs_ms=frozen,sealed_models=seal))
    source=load(BASE/'pp32-gpu-boundary-r3b/source-observations.json');inputs=[params,BASE/'pp32-gpu-boundary-r3b/source-observations.json'];xs=[]
    for rank in [0,8,16,24]:
        mp=OLD/f'pp32-structure-r4/rank-{rank}.json';inputs.append(mp);x=extract(32,rank,next(x for x in source if x['rank']==rank),load(mp));xs.append(x);print('source checked',rank,flush=True)
    sp,_=pairs(xs)
    for ph in ['F','B']:assert abs(statistics.median(p['ready_overlap_ms'] for p in sp if p['phase']==ph and p['standalone'])-frozen[ph])<1e-5
    tx=[]
    for rank in [0,16,112,128,224,240]:
        mp=OLD/f'pp256-stage-audit-r7/rank-{rank}.json';inputs.append(mp);meta=load(mp);x=extract(256,rank,meta,meta);tx.append(x);print('target checked',rank,flush=True)
    tp,unpaired=pairs(tx);rows=[]
    for phase in ['F','B']:
        ss=[p for p in sp if p['phase']==phase and p['standalone']];ts=[p for p in tp if p['phase']==phase and p['standalone']]
        assert ts
        vals=[p['ready_overlap_ms'] for p in ts]
        rows.append(dict(phase=phase,source_count=len(ss),source_frozen_ms=frozen[phase],target_count=len(ts),target_median_ms=statistics.median(vals),target_min_ms=min(vals),target_max_ms=max(vals),negative_count=sum(v<0 for v in vals),cost_error_ms=frozen[phase]-statistics.median(vals),sample_MAE_ms=statistics.mean(abs(frozen[phase]-v) for v in vals)))
    edges=[]
    for edge in [0,7,14]:
        for phase in ['F','B']:
            ps=[p for p in tp if p['message'].startswith(f'{phase}:{edge}:') and p['standalone']]
            edges.append(dict(edge=edge,phase=phase,samples=len(ps),median_ms=statistics.median(p['ready_overlap_ms'] for p in ps) if ps else None,completion_skew_max_ms=max((p['completion_skew_ms'] for p in ps),default=None)))
    assert all(sha(Path(p))==s for p,s in seal.items())
    report=dict(status='PASS_EXTRACTION_PARTIAL_PHYSICAL_COST',rows=rows,edge_stats=edges,source_paired=len(sp),target_paired=len(tp),unpaired_outside_selected_links=unpaired,payload_bytes=sorted({n for p in sp+tp for n in p['payload_bytes']}),path_classes=sorted({p['path_class'] for p in sp+tp}),source_reproduction=True,models_unchanged=True,target_fitted=False,limitations=['six target ranks / three links only; not full-world concurrent PP validation','cross-host location proven by worker directory, exact NIC/switch route not resolved','overlap proxy may contain software, synchronization, contention and clock error','two-ended arrival waiting removed only through available GPU/call proxies','no new MFU or corrected prediction result'])
    for name,x in [('source-evidence',xs),('target-evidence',tx),('source-pairs',sp),('target-pairs',tp),('report',report)]:dump(out/(name+'.json'),x)
    def rec(p):return dict(path=str(p.resolve()),sha256=sha(p))
    dump(out/'manifest.json',dict(inputs=[rec(p) for p in inputs]+[dict(path=x['path'],sha256=x['sha256']) for x in xs+tx],code=[rec(Path(__file__)),rec(Path(__file__).with_name('pp32_rendezvous_r2.py'))],sealed_models=seal,outputs=[rec(p) for p in out.glob('*.json')]))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

"""Source32 PP1 F0 evidence split; service costs remain unidentifiable, not fitted."""
import argparse
import collections as C
import graphlib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
FRAME=Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/moe')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def union(intervals):
    total=0;end=-float('inf')
    for a,b in sorted(intervals):total+=max(0,b-max(a,end));end=max(end,b)
    return total

def extract(rank, path=None, iteration=60):
    meta=load(BASE/f'pp32-structure-r4/rank-{rank}.json');path=Path(path or meta['path']);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if iteration==60:assert digest==meta['sha256']
    data=json.loads(raw);es=data['traceEvents'];groups={str(g['pg_name']):g for g in meta['groups']}
    assert data['distributedInfo']['world_size']==32 and data['distributedInfo']['rank']==rank
    ep=next(g for g in meta['groups'] if g['pg_desc']=='EXPERT_MODEL_PARALLEL_GROUP' and rank in g['ranks']);assert ep['ranks']==list(range(8,16))
    fidx,f=next((i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name')=='forward_step')
    def inside(e,a):return e['pid']==a['pid'] and a['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=a['ts']+a['dur']+.01
    markers=[];counts=C.Counter()
    for i,e in sorted(enumerate(es),key=lambda z:z[1].get('ts',0)):
        if e.get('ph')!='X' or not inside(e,f):continue
        args=e.get('args',{});name=e.get('name','')
        if e.get('cat')=='cpu_op' and name=='record_param_comms' and args.get('Collective name')=='all_to_all':
            group=groups[str(args['Process Group Name'])];assert group['pg_desc']=='CONTEXT_PARALLEL_GROUP'
            kind='CP';members=group['ranks'];size=args['In msg nelems'];dtype=args['dtype']
        elif e.get('cat')=='cpu_op' and name in ['FusedDispatch','FusedCombine']:
            kind='EP_DISPATCH' if name=='FusedDispatch' else 'EP_COMBINE';members=ep['ranks'];size=None;dtype=None
        else:continue
        ordinal=counts[kind];counts[kind]+=1
        markers.append(dict(kind=kind,ordinal=ordinal,id=f'{kind}:{"-".join(map(str,members))}:{ordinal}',cpu_event=i,cpu_start=e['ts']/1000,cpu_end=(e['ts']+e['dur'])/1000,members=members,input_nelems=size,dtype=dtype,service_ms=None))
    assert counts=={'CP':16,'EP_DISPATCH':4,'EP_COMBINE':4},counts
    assert all(a['cpu_end']<=b['cpu_start'] for a,b in zip(markers,markers[1:]))
    runtime=C.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat')=='privateuse1_runtime' and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append((i,e))
    devices=[];cpcount=C.Counter()
    for i,e in enumerate(es):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        rs=runtime.get(e.get('args',{}).get('correlation'),[])
        if len(rs)!=1 or not inside(rs[0][1],f):continue
        ri,r=rs[0];args=e.get('args',{});hits=[m for m in markers if inside(r,es[m['cpu_event']])]
        if args.get('Process Group Description')=='CONTEXT_PARALLEL_GROUP':
            # Parent record_param_comms external id need not equal kernel external id.
            # Ordinal mapping checked against group/size signature below.
            pg=str(args['Process Group Name']);j=cpcount[pg];cpcount[pg]+=1
            candidates=[m for m in markers if m['kind']=='CP' and m['members']==groups[pg]['ranks']];m=candidates[j]
            assert args['Collective name']=='all_to_all' and args['In msg nelems']==m['input_nelems']
            semantic=m['id'];tag='CP'
        elif len(hits)==1:semantic=hits[0]['id'];tag=hits[0]['kind']
        else:
            prior=[m for m in markers if m['cpu_end']<=r['ts']/1000]
            slot=len(prior);semantic=f'r{rank}:LOCAL:{slot}';tag='LOCAL_MEMORY' if e['cat']!='kernel' else 'LOCAL_KERNEL'
        devices.append(dict(event=i,runtime_event=ri,name=e['name'],stream=args.get('stream',e.get('tid')),start_ms=e['ts']/1000,end_ms=(e['ts']+e['dur'])/1000,semantic=semantic,tag=tag))
    assert len([d for d in devices if d['tag']=='CP'])==16
    origin=min(d['start_ms'] for d in devices);end=max(d['end_ms'] for d in devices)
    for d in devices:d['start_ms']-=origin;d['end_ms']-=origin
    for m in markers:
        ds=[d for d in devices if d['semantic']==m['id']];assert ds
        m.update(device_events=[d['event'] for d in ds],device_active_union_ms=union((d['start_ms'],d['end_ms']) for d in ds),device_start_ms=min(d['start_ms'] for d in ds),device_end_ms=max(d['end_ms'] for d in ds),cpu_duration_ms=m['cpu_end']-m['cpu_start'])
        m['cpu_start']-=origin;m['cpu_end']-=origin
    # Exact exclusive wall-time partition; overlap is not counted twice.
    points=sorted({d[k] for d in devices for k in ['start_ms','end_ms']});partition=[];totals=C.Counter()
    for a,b in zip(points,points[1:]):
        active=sorted({d['tag'] for d in devices if d['start_ms']<b and d['end_ms']>a});tag='+'.join(active) or 'NO_OBSERVED_DEVICE'
        totals[tag]+=b-a
        if partition and partition[-1]['tag']==tag:partition[-1]['end_ms']=b
        else:partition.append(dict(start_ms=a,end_ms=b,tag=tag))
    assert abs(sum(totals.values())-(end-origin))<1e-7
    # GPU stream evidence: compress consecutive same-semantic events, not across streams.
    runs=[];edges=[]
    for stream in sorted({str(d['stream']) for d in devices}):
        ds=sorted([d for d in devices if str(d['stream'])==stream],key=lambda d:d['start_ms']);prev=None
        for d in ds:
            if prev is not None and prev['semantic']==d['semantic']:
                prev['end_ms']=max(prev['end_ms'],d['end_ms']);prev['events'].append(d['event']);prev['intervals'].append((d['start_ms'],d['end_ms']))
            else:
                run=dict(id=f'r{rank}:stream{stream}:run{len(runs)}',semantic=d['semantic'],stream=stream,start_ms=d['start_ms'],end_ms=d['end_ms'],events=[d['event']],intervals=[(d['start_ms'],d['end_ms'])])
                if prev is not None:edges.append(dict(source=prev['id'],target=run['id'],kind='same_GPU_stream'))
                runs.append(run);prev=run
    for run in runs:run['device_active_union_ms']=union(run.pop('intervals'))
    return dict(rank=rank,stage=1,iteration=iteration,microbatch=0,path=str(path),sha256=digest,cpu_F_event=fidx,origin_absolute_ms=origin,window_ms=end-origin,
        markers=markers,devices=devices,partition=partition,exclusive_time_ms=dict(totals),stream_runs=runs,stream_edges=edges,
        note='local kernels are non-identified-communication device work, not proven pure FLOPs; EP fused envelopes not pure communication')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    framework=[FRAME/'fused_a2a.py',FRAME/'token_dispatcher.py']
    dump(out/'protocol.json',dict(version='pp32-f-split-r8',scope='source32 iter60 PP1 F0 ranks8..15 only',target256_read=False,old_model_modified=False,
        framework=[dict(path=str(p),sha256=sha(p)) for p in framework],framework_exact_runtime_equivalence=False,
        gates=['exclusive time conservation','all CP group participants matched','EP fused semantics preserved','backend service slots remain null, no inferred pure transport'],
        warning='CPU-order skeleton is not complete GPU dependency graph; no invented collective readiness or service costs'))
    obs=[]
    for r in range(8,16):
        x=extract(r);dump(out/f'rank-{r}.json',x);obs.append(x);print('rank',r,'ms',x['window_ms'],'events',len(x['devices']),flush=True)
    collective=C.defaultdict(list)
    for x in obs:
        for m in x['markers']:collective[m['id']].append(dict(rank=x['rank'],**m))
    nodes={};edges=[]
    for k,participants in collective.items():
        assert sorted(p['rank'] for p in participants)==participants[0]['members']
        assert len({(p['input_nelems'],p['dtype']) for p in participants})==1
        nodes[k]=dict(id=k,kind=participants[0]['kind'],members=participants[0]['members'],service_ms=None,input_nelems=participants[0]['input_nelems'],dtype=participants[0]['dtype'],matching='CPU ordinal plus group and size; collective sequence id unavailable')
    # CPU submission order skeleton, useful identity map, explicitly not GPU-ready edges.
    for x in obs:
        prev=None
        for j in range(len(x['markers'])+1):
            k=f'r{x["rank"]}:LOCAL:{j}';ds=[d for d in x['devices'] if d['semantic']==k]
            nodes[k]=dict(id=k,kind='LOCAL_WORK',rank=x['rank'],device_active_union_ms=union((d['start_ms'],d['end_ms']) for d in ds),predictive_cost_ms=None)
            if prev:edges.append(dict(source=prev,target=k,kind='CPU_ORDER_ONLY'))
            if j<len(x['markers']):
                m=x['markers'][j];edges.append(dict(source=k,target=m['id'],kind='CPU_ORDER_ONLY'));prev=m['id']
    deps={k:[] for k in nodes}
    for e in edges:deps[e['target']].append(e['source'])
    list(graphlib.TopologicalSorter(deps).static_order())
    dump(out/'dependency-skeleton.json',dict(status='PARTIAL_NOT_EXECUTABLE_GPU_MODEL',nodes=list(nodes.values()),edges=edges))
    dump(out/'collective-inputs.json',list(nodes[k] for k in collective))
    report=dict(status='PARTIAL',evidence_checks='PASS',ranks=8,CP_calls_per_rank=16,EP_dispatch_per_rank=4,EP_combine_per_rank=4,
        matched_CP_collectives=sum(n['kind']=='CP' for n in nodes.values()),matched_EP_fused_operations=sum(n['kind'].startswith('EP_') for n in nodes.values()),
        exclusive_partition='PASS exact F GPU-window accounting, not predictive replay',service_identified=False,GPU_cross_stream_dependencies_complete=False,
        per_rank=[dict(rank=x['rank'],window_ms=x['window_ms'],exclusive_time_ms=x['exclusive_time_ms']) for x in obs],
        next='resolve FusedDispatch/Combine ACE transport and stream event waits before backend cost replacement')
    dump(out/'report.json',report)
    dump(out/'manifest.json',dict(code=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(out.glob('*.json'))]))
    print(json.dumps({k:v for k,v in report.items() if k!='per_rank'}),flush=True)
if __name__=='__main__':main()

"""9 training iterations: trace-anchored F layers + full EP8 dispatch count conservation.

Only representatives have direct FC trace checks each iteration. All ranks have
DeepEP sender/receiver counts. Never equate DeepEP call counter with training iter.
"""
import bisect
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
RAW=Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40')
OUT=BASE/'layer-iteration-grid-r1';ITERATIONS=list(range(40,81,5))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def process(task):
    stage,group,paths=task[:3];options=task[3] if len(task)>3 else {};first=stage*options.get('stage_width',16)+group*8;ranks=list(range(first,first+8));dest=Path(options.get('out',OUT))/f'pp{stage}-group{group}.json';microbatches=options.get('microbatches',4)
    if dest.exists():return dest.name,'CACHED'
    logs={};inputs={};trace_evidence=[];rows=[];rank_rows=[]
    for rank in ranks:
        worker=Path(paths['60'][str(rank)]).parents[2]
        events={}
        for p in sorted(worker.glob(f'deepep_trace/*/rank_{rank}.log')):
            used=False
            for lineno,line in enumerate(p.open(),1):
                e=json.loads(line)
                if e.get('event') not in ('dispatch_layout','after_dispatch'):continue
                # Includes all calls in the files; identity is established by CPU timestamps below.
                key=(e['iter'],e['event']);assert key not in events,(p,key)
                events[key]=dict(record=e,path=str(p),line=lineno);used=True
            if used:inputs[str(p)]=sha(p)
        assert events;logs[rank]=events
    anchor_layouts=sorted([e for (call,event),e in logs[first].items() if event=='dispatch_layout'],key=lambda e:e['record']['timestamp_ns'])
    times=[e['record']['timestamp_ns'] for e in anchor_layouts]
    for iteration in ITERATIONS:
        p=Path(paths[str(iteration)][str(first)]);buf=p.read_bytes();inputs[str(p)]=hashlib.sha256(buf).hexdigest();trace=json.loads(buf);del buf
        es=trace['traceEvents'];base=trace['baseTimeNanoseconds'];assert trace['distributedInfo']['rank']==first
        ep=next(g['ranks'] for g in trace['distributedInfo']['pg_config'] if g['pg_desc']=='EXPERT_MODEL_PARALLEL_GROUP');assert ep==ranks
        fs=sorted([(i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e['name']=='forward_step'],key=lambda x:x[1]['ts']);assert len(fs)==microbatches
        cpus=[(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e['name'] in ('CheckpointFunction','aten::split_with_sizes')]
        def inside(e,c):return e['pid']==c['pid'] and e['tid']==c['tid'] and c['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=c['ts']+c['dur']+.01
        used_calls=[]
        for mb,(fi,f) in enumerate(fs):
            cps=sorted([(i,e) for i,e in cpus if e['name']=='CheckpointFunction' and inside(e,f)],key=lambda x:x[1]['ts']);assert len(cps)==4
            for local,(ci,c) in enumerate(cps):
                splits=[]
                for i,e in cpus:
                    if e['name']!='aten::split_with_sizes' or not inside(e,c):continue
                    a=e.get('args',{});dims=a.get('Input Dims',[[]])[0]
                    if len(dims)!=2 or dims[1] not in (5120,1536):continue
                    counts=json.loads(a['Concrete Inputs'][1])
                    if len(counts)==20:splits.append((i,e,counts))
                splits.sort(key=lambda x:x[1]['ts']);assert len(splits)==2 and splits[0][2]==splits[1][2]
                assert sum(splits[0][2])==splits[0][1]['args']['Input Dims'][0][0]
                lo=base+c['ts']*1000;hi=lo+c['dur']*1000
                hits=anchor_layouts[bisect.bisect_left(times,lo):bisect.bisect_right(times,hi)];assert len(hits)==1,(stage,group,iteration,mb,local,len(hits))
                call=hits[0]['record']['iter'];used_calls.append(call)
                anchor_recv=logs[first][call,'after_dispatch'];assert lo<=anchor_recv['record']['timestamp_ns']<=hi
                assert anchor_recv['record']['num_recv_tokens_per_expert_list']==splits[0][2]
                layer=3+4*(stage-1)+local
                senders=[];receivers=[]
                for rank in ranks:
                    send=logs[rank][call,'dispatch_layout'];recv=logs[rank][call,'after_dispatch']
                    assert send['record']['rank']==recv['record']['rank']==rank
                    assert send['record']['x_shape']==[8192,5120]
                    sc=send['record']['num_tokens_per_expert'];rc=recv['record']['num_recv_tokens_per_expert_list']
                    assert len(sc)==160 and len(rc)==20 and all(0<=n<=615 for n in sc)
                    senders.append(sc);receivers.extend(rc)
                    rank_rows.append(dict(iteration=iteration,layer=layer,mb=mb,rank=rank,call=call,received=rc,send_path=send['path'],send_line=send['line'],recv_path=recv['path'],recv_line=recv['line']))
                outgoing=[sum(s[e] for s in senders) for e in range(160)];assert outgoing==receivers,(stage,group,iteration,mb,local)
                total=sum(receivers);loads=[sum(receivers[i*20:(i+1)*20]) for i in range(8)];mean=total/8
                rows.append(dict(iteration=iteration,stage=stage,group=group,layer=layer,mb=mb,retained=total,active_experts=sum(n>0 for n in receivers),offered=393216,not_entered=393216-total,cv=(sum((v-mean)**2 for v in loads)/8)**.5/mean,call=call))
                trace_evidence.append(dict(iteration=iteration,layer=layer,mb=mb,rank=first,path=str(p),forward_event=fi,checkpoint_event=ci,split_events=[x[0] for x in splits],dispatch_call=call,checkpoint_start_ns=lo,checkpoint_end_ns=hi))
        assert len(set(used_calls))==4*microbatches
        del trace,es,cpus
    dest.write_text(json.dumps(dict(stage=stage,group=group,rows=rows,rank_rows=rank_rows,trace_evidence=trace_evidence,inputs=inputs),separators=(',',':'))+'\n')
    return dest.name,'PASS'
def main():
    OUT.mkdir(exist_ok=True);paths={}
    for it in ITERATIONS:
        paths[str(it)]={}
        for p in RAW.glob(f'*/profiler/iteration_{it}/rank*.pt.trace.json'):
            rank=p.name.split('.')[0][4:];assert rank not in paths[str(it)];paths[str(it)][rank]=str(p)
        assert all(str(rank) in paths[str(it)] for rank in range(16,240))
    tasks=[(s,g,{it:{str(r):ps[str(r)] for r in range(s*16+g*8,s*16+g*8+8)} for it,ps in paths.items()}) for s in range(1,15) for g in range(2)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(process,t) for t in tasks]):print(*f.result(),flush=True)
    results=[json.loads((OUT/f'pp{s}-group{g}.json').read_text()) for s in range(1,15) for g in range(2)]
    rows=[r for d in results for r in d['rows']];assert len(rows)==4032
    inputs={k:v for d in results for k,v in d['inputs'].items()};inputs[str(Path(__file__).resolve())]=sha(Path(__file__))
    report=dict(status='PASS',iterations=ITERATIONS,scope='256 PP1..14 both EP8 groups; exact representative CPU checkpoint anchors; all rank send/recv logs; nonrepresentative FC traces not re-read',rows=rows,representative_trace_files=252,group_layer_mb=4032,expert_conservation_checks=len(rows)*160)
    (OUT/'report.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(inputs=inputs,outputs={p.name:sha(p) for p in OUT.glob('*.json') if p.name!='manifest.json'}),indent=2)+'\n');print('COMPLETE',len(rows),'group/layer/MB records')
if __name__=='__main__':main()

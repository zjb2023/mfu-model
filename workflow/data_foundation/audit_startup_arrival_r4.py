"""One-iteration all-rank AG arrival inventory; no clock alignment assumed."""
import argparse
import concurrent.futures
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40')

def extract(p):
    raw=p.read_bytes(); digest=hashlib.sha256(raw).hexdigest();d=json.loads(raw);del raw
    es=d['traceEvents'];rank=d['distributedInfo']['rank']
    f=min((e for e in es if e.get('name')=='forward_step' and e.get('cat')=='user_annotation'),key=lambda e:e['ts'])
    step=next(e for e in es if e.get('name','').startswith('ProfilerStep') and e['ts']<=f['ts']<e['ts']+e.get('dur',0))
    candidates=[(i,e) for i,e in enumerate(es) if e.get('cat')=='kernel' and e.get('args',{}).get('Collective name')=='_allgather_base' and step['ts']<=e['ts']<f['ts'] and e['args'].get('Process Group Name')=='0']
    assert len(candidates)==1,(rank,len(candidates));ki,k=candidates[0]
    assert k['args']['In msg nelems']==24 and k['args']['Out msg nelems']==6144 and k['args']['Group size']==256
    barriers=[(i,e) for i,e in enumerate(es) if e.get('cat')=='kernel' and e.get('args',{}).get('Collective name')=='barrier' and step['ts']<=e['ts']<k['ts'] and e['args'].get('Process Group Name')=='0']
    assert len(barriers)==2,(rank,len(barriers));bi,b=max(barriers,key=lambda x:x[1]['ts'])
    runtime=[(i,e) for i,e in enumerate(es) if e.get('cat')=='privateuse1_runtime' and e.get('args',{}).get('correlation')==k['args']['correlation']]
    assert len(runtime)==1;ri,rt=runtime[0]
    calls=[(i,e) for i,e in enumerate(es) if e.get('name')=='c10d::_allgather_base_' and e.get('pid')==rt.get('pid') and e.get('tid')==rt.get('tid') and e['ts']<=rt['ts'] and e['ts']+e['dur']>=rt['ts']+rt['dur']-.01]
    assert len(calls)==1;ci,c=calls[0]
    def small(i,e):return dict(index=i,name=e['name'],ts_us=e['ts'],duration_us=e['dur'],args=e.get('args',{}))
    return dict(rank=rank,worker=p.parts[-4],path=str(p),sha256=digest,bytes=p.stat().st_size,profiler_label=step['name'],baseTimeNanoseconds=d.get('baseTimeNanoseconds'),barrier=small(bi,b),ag=small(ki,k),launch=small(ri,rt),cpu_call=small(ci,c),ag_duration_ms=k['dur']/1000,barrier_duration_ms=b['dur']/1000,barrier_start_to_ag_start_ms=(k['ts']-b['ts'])/1000,barrier_end_to_ag_start_ms=(k['ts']-b['ts']-b['dur'])/1000,cpu_call_to_gpu_start_ms=(k['ts']-c['ts'])/1000,barrier_end_to_ag_end_ms=(k['ts']+k['dur']-b['ts']-b['dur'])/1000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    paths=sorted(DATA.glob('worker*/profiler/iteration_60/rank*.pt.trace.json'),key=lambda p:int(re.search(r'rank(\d+)\.',p.name)[1]))
    assert len(paths)==256
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(extract,paths):
            rows.append(row)
            if len(rows)%16==0:print('parsed',len(rows),'of 256',flush=True)
    assert sorted(r['rank'] for r in rows)==list(range(256))
    assert len(set(r['profiler_label'] for r in rows))==1
    longest=max(r['barrier_start_to_ag_start_ms'] for r in rows)
    # Barrier semantics: every rank entered the same preceding barrier before any
    # rank completes it. Thus max AG-start <= reference barrier-end + max(A_i-B_i).
    # Durations are local; cross-host absolute timestamps are not needed.
    for r in rows:
        bound=max(0,longest-r['barrier_end_to_ag_start_ms'])
        r['arrival_spread_upper_bound_ms']=bound
        r['residence_after_all_kernel_starts_lower_bound_ms']=max(0,r['ag_duration_ms']-bound)
    late=sorted(rows,key=lambda r:r['barrier_start_to_ag_start_ms'],reverse=True)
    stats=dict(min_duration_ms=min(r['ag_duration_ms'] for r in rows),max_duration_ms=max(r['ag_duration_ms'] for r in rows),max_barrier_start_to_ag_start_ms=longest,rank0_arrival_wait_upper_bound_ms=rows[0]['arrival_spread_upper_bound_ms'],rank0_after_all_starts_lower_bound_ms=rows[0]['residence_after_all_kernel_starts_lower_bound_ms'],latest_local_ranks=[r['rank'] for r in late[:12]],total_raw_bytes=sum(r['bytes'] for r in rows))
    report=dict(status='PASS_EXTRACT_PARTIAL_CAUSAL',scope='256 ranks, iteration60 only, first global 24-float AG and two preceding barriers',stats=stats,rows=rows,limits=['No calibrated cross-host timestamp subtraction.','Matching: same Profiler label, default group, first 24->6144 global AG, two preceding global barriers; no backend collective sequence ID available.','Clock-free bounds assume same barrier instance and standard barrier completion semantics; monotonic local CPU/GPU timestamps.','Kernel start is not proof of protocol/payload readiness. Remaining residence may include device/protocol readiness waits as well as transfer.','No controlled perturbation; not an independent prediction model.'])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(code={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},raw=[{k:r[k] for k in ['path','sha256','bytes']} for r in rows],outputs={str(out/'report.json'):hashlib.sha256((out/'report.json').read_bytes()).hexdigest()})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False),flush=True)
    print('latest',[(r['rank'],round(r['barrier_start_to_ag_start_ms'],3),round(r['ag_duration_ms'],3)) for r in late[:12]],flush=True)

if __name__=='__main__':main()

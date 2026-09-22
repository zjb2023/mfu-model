"""Select LAST global CPU timer AG, require exact GPU association; no guessed joins."""
import argparse,concurrent.futures,hashlib,json
from pathlib import Path
from audit_startup_cohort_r5 import save
ROOT=Path(__file__).resolve().parents[2]
REF=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/multi-strategy-r2/input_manifest.json')
def extract(job):
    p=Path(job['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();d=json.loads(raw);del raw;es=d['traceEvents']
    assert d['distributedInfo']['rank']==0 and d['distributedInfo']['world_size']==16
    steps=[e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep')];assert len(steps)==1;s=steps[0];lo=s['ts'];hi=lo+s['dur']
    calls=[(i,e) for i,e in enumerate(es) if e.get('name')=='record_param_comms' and lo<=e['ts']<hi and e.get('args',{}).get('Collective name')=='_allgather_base' and e['args'].get('Process Group Description')=='default_pg' and e['args'].get('Group size')==16 and e['args'].get('In msg nelems')==24 and e['args'].get('dtype')=='Float']
    calls.sort(key=lambda x:x[1]['ts']);gpu=[(i,e) for i,e in enumerate(es) if e.get('cat')=='kernel'];result=[]
    for ci,c in calls:
        ext=c['args']['External id'];matches=[(i,e) for i,e in gpu if e.get('args',{}).get('External id')==ext and 'AllGather' in e['name']]
        assert len(matches)<=1
        result.append(dict(cpu_event=ci,cpu_start_ms=(c['ts']-lo)/1000,external_id=ext,gpu_event=matches[0][0] if matches else None,gpu_start_ms=(matches[0][1]['ts']-lo)/1000 if matches else None,duration_ms=matches[0][1]['dur']/1000 if matches else None,method='exact External id + AllGather kernel' if matches else 'UNMATCHED: no exact device External id; do not guess correlation +/-1'))
    last=result[-1] if result else None
    return dict(**job,sha256=digest,bytes=p.stat().st_size,profiler_label=s['name'],profiler_ms=s['dur']/1000,status='PASS' if last and last['duration_ms'] is not None else 'PARTIAL_LAST_CPU_AG_NO_EXACT_GPU',startup_GPU_ms=None,timer_AG_count=len(result),timer_AGs=result,selected_AG=last,selected_AG_ms=last['duration_ms'] if last else None,all_timer_AG_sum_ms=sum(x['duration_ms'] for x in result) if result and all(x['duration_ms'] is not None for x in result) else None)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);m=json.loads(REF.read_text());jobs=[];inventory=[]
    cases=sorted({x['case'] for x in m['items']});assert len(cases)==14
    for case in cases:
        entry=next(x for x in m['items'] if x['case']==case and x['rank']==0);root=Path(entry['path']).parent.parent;chosen=[];missing=[]
        for it in range(8,104,5):
            p=root/f'iteration_{it}/gpu0.json'
            if not p.is_file():missing.append(it);continue
            chosen.append(it);jobs.append(dict(case='16-'+case,world=16,rank=0,iteration=it,path=str(p),resolved_path=str(p.resolve())))
        inventory.append(dict(case='16-'+case,world=16,root=str(root),selected=chosen,missing=missing))
    save(out/'inventory.json',inventory);rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for r in pool.map(extract,jobs):
            rows.append(r);save(out/f'{r["case"]}-iter{r["iteration"]}.json',r)
            if len(rows)%20==0:print('verified',len(rows),r['case'],flush=True)
    save(out/'report.json',dict(inventory=inventory,rows=rows,selection='last CPU-recorded default_pg16 Float24 AG, exact device External id only; not startup',limitations=['Unmatched last calls omitted from mean, never substituted with earlier AG.','Some runtime/kernel correlation IDs differ by one; no speculative correction.']))
    save(out/'manifest.json',dict(inputs={str(REF):hashlib.sha256(REF.read_bytes()).hexdigest()},code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),Path(__file__).with_name('audit_startup_cohort_r5.py')]},raw=[{k:r[k] for k in ['path','sha256','bytes']} for r in rows],outputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')}))
if __name__=='__main__':main()

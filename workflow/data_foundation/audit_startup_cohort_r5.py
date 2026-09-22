"""Rank0 cross-iteration startup and global timer-AG census, not world means."""
import argparse
import concurrent.futures
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def extract(item):
    try:
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();d=json.loads(raw);del raw
        assert d['distributedInfo']['rank']==0 and d['distributedInfo']['world_size']==item['world']
        es=d['traceEvents'];steps=[(i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep') and e.get('dur',0)>0]
        assert len(steps)==1,len(steps);si,s=steps[0];lo=s['ts'];hi=lo+s['dur']
        fs=[e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='forward_step' and lo<=e['ts']<hi]
        f=min(fs,key=lambda e:e['ts']) if fs else None
        loader=[e for e in es if e.get('cat')=='user_annotation' and 'DataLoader' in e.get('name','') and lo<=e['ts']<hi]
        gpu=[(i,e) for i,e in enumerate(es) if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']]
        startgpu=None
        if f:
            corr={e['args']['correlation'] for e in es if e.get('cat') in ['cuda_runtime','cuda_driver','privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}) and e.get('pid')==f['pid'] and f['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=f['ts']+f['dur']+.01}
            owned=[e for _,e in gpu if e.get('args',{}).get('correlation') in corr];assert owned
            startgpu=min(e['ts'] for e in owned)
        ags=[]
        for i,e in gpu:
            a=e.get('args',{})
            if not(lo<=e['ts']<hi and a.get('Collective name')=='_allgather_base' and int(a.get('Group size',0))==item['world'] and a.get('Process Group Description')=='default_pg' and a.get('In msg nelems')==24 and a.get('dtype')=='Float'):continue
            ags.append(dict(event=i,start_ms=(e['ts']-lo)/1000,duration_ms=e['dur']/1000,input_bytes=96,output_bytes=int(a['Out msg nelems'])*4,group=a.get('Process Group Name'),correlation=a.get('correlation'),position='before_first_F' if f and e['ts']<f['ts'] else ('after_first_F' if f else 'after_first_DataLoader' if loader and e['ts']>min(x['ts'] for x in loader) else 'UNRESOLVED')))
        ags.sort(key=lambda e:e['start_ms'])
        if f:eligible=[a for a in ags if a['position']=='before_first_F']
        else:eligible=ags[-1:]
        selected=eligible[0] if len(eligible)==1 else None
        barriers=[dict(event=i,start_ms=(e['ts']-lo)/1000,duration_ms=e['dur']/1000) for i,e in gpu if lo<=e['ts']<hi and e.get('args',{}).get('Collective name')=='barrier' and e.get('args',{}).get('Process Group Description')=='default_pg']
        return dict(**item,status='PASS' if selected else 'PARTIAL_NO_UNIQUE_MATCH',sha256=digest,bytes=p.stat().st_size,profiler_label=s['name'],profiler_ms=s['dur']/1000,first_F_CPU_ms=(f['ts']-lo)/1000 if f else None,startup_GPU_ms=(startgpu-lo)/1000 if startgpu is not None else None,first_DataLoader_CPU_ms=(min(e['ts'] for e in loader)-lo)/1000 if loader else None,timer_AG_count=len(ags),timer_AGs=ags,selected_AG=selected,selected_AG_ms=selected['duration_ms'] if selected else None,all_timer_AG_sum_ms=sum(e['duration_ms'] for e in ags) if ags else None,barriers=barriers,selection='unique pre-F global 24-float AG' if f else 'last global 24-float AG in ProfilerStep; not a startup boundary')
    except Exception as e:return dict(**item,status='FAIL',error=repr(e))

def stats(values):
    vs=[v for v in values if v is not None]
    return dict(n=len(vs),mean=statistics.mean(vs),median=statistics.median(vs),min=min(vs),max=max(vs),stdev=statistics.pstdev(vs)) if vs else dict(n=0,mean=None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    roots={
        '256':(256,Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40/worker33087/profiler')),
        '224':(224,Path('/home/zjb/Desktop/fabric-data-analysis/fabric-data-analysis-raw/0731_224gpu_pp14_cp2_a2a/extracted/framework/2026-07-30-09:38/worker33069/profiler')),
        '32':(32,Path('/home/zjb/Desktop/32/gpu32_gbs64_framework/2026-09-18-10:38/worker34095/profiler'))}
    for case in sorted(Path('/var/local/zjb-data/trace16node/0908').iterdir()):
        if case.is_dir():roots['16-'+case.name]=(16,case/'224d02205/profiler')
    assert len(roots)==17
    inventory=[];jobs=[]
    for case,(world,root) in roots.items():
        available=sorted(int(p.name.split('_')[1]) for p in root.glob('iteration_*') if p.is_dir())
        chosen=list(range(5,101,5)) if world!=16 else list(range(8,104,5))
        missing=[];valid=[]
        for it in chosen:
            paths=list((root/f'iteration_{it}').glob('gpu0.json' if world==16 else 'rank0.*.pt.trace.json'))
            if len(paths)!=1:missing.append(dict(iteration=it,matches=len(paths)));continue
            p=paths[0];jobs.append(dict(case=case,world=world,rank=0,iteration=it,path=str(p),resolved_path=str(p.resolve())));valid.append(it)
        inventory.append(dict(case=case,world=world,root=str(root),available=available,requested=chosen,selected=valid,missing=missing))
    save(out/'inventory.json',inventory)
    print('planned',len(jobs),'traces, rank0 only',flush=True)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(extract,jobs):
            rows.append(row);save(out/f'{row["case"]}-iter{row["iteration"]}.json',row)
            if len(rows)%10==0 or row['status']!='PASS':print(len(rows),row['case'],row['iteration'],row['status'],row.get('selected_AG_ms'),row.get('error',''),flush=True)
    summaries=[]
    for inv in inventory:
        rs=[r for r in rows if r['case']==inv['case']];good=[r for r in rs if r['status']!='FAIL']
        summaries.append(dict(case=inv['case'],world=inv['world'],n=len(rs),iterations=[r['iteration'] for r in rs],failures=[r for r in rs if r['status']=='FAIL'],startup_ms=stats([r.get('startup_GPU_ms') for r in good]),selected_AG_ms=stats([r.get('selected_AG_ms') for r in good]),all_AG_sum_ms=stats([r.get('all_timer_AG_sum_ms') for r in good]),AG_counts=sorted(set(r['timer_AG_count'] for r in good)),excluding_iter100_AG_ms=stats([r.get('selected_AG_ms') for r in good if r['iteration']!=100]),excluding_iter100_startup_ms=stats([r.get('startup_GPU_ms') for r in good if r['iteration']!=100])))
    report=dict(scope='Rank0 mean over iterations, not mean over ranks; large cases 5..100/5, 16-card cases8..103/5',inventory=inventory,summaries=summaries,rows=rows,limits=['16-card traces lack forward_step annotation: startup_GPU_ms is unavailable, not zero.','16-card last default-group 24-float AG is a statistics candidate at iteration tail; retain all candidates and count.','Iteration100 included for requested 20-sample mean; exclusion sensitivity separately provided.','Directory iteration labels retained separately from ProfilerStep labels.','No all-rank arrival attribution repeated across iterations; iter60 rank-arrival conclusion does not generalize automatically.','Means compare different platforms/configurations/logging; not controlled rank-scaling experiment.'])
    save(out/'report.json',report)
    save(out/'manifest.json',dict(code={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},raw=[{k:r[k] for k in ['path','resolved_path','sha256','bytes']} for r in rows if 'sha256' in r],outputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')}))
    print(json.dumps(summaries,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

"""Bounded local execution of target calibration; no agents or raw mutations."""
import argparse,concurrent.futures,datetime,json,subprocess,sys,time
from pathlib import Path
from calibrate256_fine_r1 import BASE,ROOT,save
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--skip',nargs='*',default=[]);a=ap.parse_args();assert 1<=a.workers<=16
    out=BASE/'calibrate256-fine-r1';out.mkdir(exist_ok=True);jobs=[(r,m) for r in range(16,240,8) for m in range(4) if f'{r}:{m}' not in a.skip]
    states={f'{r}:{m}':dict(status='QUEUED') for r,m in jobs};started=time.time()
    def task(r,m):
        dest=out/f'r{r}-mb{m}';dest.mkdir(exist_ok=True)
        with (dest/'batch.log').open('a') as log:
            ret=subprocess.run([sys.executable,str(ROOT/'workflow/data_foundation/calibrate256_fine_r1.py'),'--rank-base',str(r),'--mb',str(m)],stdout=log,stderr=log).returncode
            if ret:return dict(status='BUILD_FAILED',exit=ret,log=str(dest/'batch.log'))
            audit=dest/'shared-binding-r2/report.json'
            if not audit.exists():
                ret=subprocess.run([sys.executable,str(ROOT/'workflow/data_foundation/audit256_fine_binding_r1.py'),'--rank-base',str(r),'--mb',str(m),'--model-dir',str(BASE/'iter60-shared-dag-r2'),'--tag','shared-binding-r2'],stdout=log,stderr=log).returncode
            if ret:return dict(status='AUDIT_FAILED',exit=ret,log=str(dest/'batch.log'))
            phases=json.loads(audit.read_text())['phases'];ok=all(x['status'].startswith('PASS_') for x in phases.values())
            return dict(status='PASS' if ok else 'REVIEW_REQUIRED',phases={p:{k:(len(v) if isinstance(v,list) else v) for k,v in x.items()} for p,x in phases.items()},log=str(dest/'batch.log'))
    def publish():
        report=dict(started_utc=datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=a.workers,total=len(jobs),states=states,skipped=a.skip)
        tmp=out/'batch-progress.tmp';save(tmp,report);tmp.replace(out/'batch-progress.json')
    publish()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(task,r,m):(r,m) for r,m in jobs}
        for f in concurrent.futures.as_completed(futures):
            r,m=futures[f]
            try:states[f'{r}:{m}']=f.result()
            except Exception as e:states[f'{r}:{m}']=dict(status='RUNNER_ERROR',error=repr(e))
            publish();print(f'{r}:{m}',states[f'{r}:{m}']['status'],sum(s['status']!='QUEUED' for s in states.values()),'/',len(jobs),flush=True)
    print('BATCH FINISHED',flush=True)
if __name__=='__main__':main()

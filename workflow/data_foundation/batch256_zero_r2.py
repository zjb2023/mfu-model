"""Run the proven-zero candidate on r1 failures with the same missing-FC gate."""
import concurrent.futures,json,subprocess,sys
from pathlib import Path
from calibrate256_fine_r1 import BASE,ROOT,save,load,sha
def main():
    source=BASE/'calibrate256-fine-r1';out=BASE/'calibrate256-zero-r2';out.mkdir(exist_ok=True);jobs=[]
    for p in sorted(source.glob('r*-mb*')):
        logs=list(p.glob('*.log'))
        if any('assert len(ex)==2' in q.read_text() for q in logs):
            r,m=p.name[1:].split('-mb');jobs.append((int(r),int(m)))
    states={};inputs={str(p):sha(p) for p in [Path(__file__),ROOT/'workflow/data_foundation/calibrate256_zero_r2.py',ROOT/'workflow/data_foundation/zero_work_evidence_r2.py',ROOT/'workflow/data_foundation/normalize_zero_recompute_r2.py']}
    def task(r,m):
        dest=out/f'r{r}-mb{m}';dest.mkdir(exist_ok=True)
        cmds=[('scope.json',['calibrate256_zero_r2.py']),('b-recompute-r12-zero-normalized/report.json',['normalize_zero_recompute_r2.py']),('shared-binding-r4/report.json',['audit256_zero_r2.py','--model-dir',str(BASE/'iter60-shared-dag-r2'),'--tag','shared-binding-r4'])]
        with (dest/'zero-batch.log').open('a') as log:
            for result,args in cmds:
                if (dest/result).exists():continue
                cmd=[sys.executable,str(ROOT/'workflow/data_foundation'/args[0]),*args[1:],'--rank-base',str(r),'--mb',str(m)]
                ret=subprocess.run(cmd,stdout=log,stderr=log).returncode
                if ret:return dict(status='FAILED',stage=args[0],log=str(dest/'zero-batch.log'))
        d=load(dest/'shared-binding-r4/report.json');return dict(status='PASS' if all(v['status'].startswith('PASS') for v in d['phases'].values()) else 'REVIEW',report=str(dest/'shared-binding-r4/report.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        fs={pool.submit(task,r,m):(r,m) for r,m in jobs}
        for f in concurrent.futures.as_completed(fs):
            r,m=fs[f]
            try:states[f'{r}:{m}']=f.result()
            except Exception as e:states[f'{r}:{m}']=dict(status='ERROR',error=repr(e))
            save(out/'batch-progress.json',dict(total=len(jobs),states=states,inputs=inputs));print(r,m,states[f'{r}:{m}']['status'],len(states),'/',len(jobs),flush=True)
    print('ZERO BATCH FINISHED',flush=True)
if __name__=='__main__':main()

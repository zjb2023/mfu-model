"""Remaining twelve r1 zero-work cases discovered after the first r2 batch."""
import concurrent.futures,subprocess,sys
from calibrate256_fine_r1 import BASE,ROOT,load,save
JOBS=[(224,3),(232,0),(216,1),(232,1),(208,2),(216,3),(216,0),(232,3),(208,0),(224,1),(208,3),(216,2)]
def main():
    out=BASE/'calibrate256-combined-r3';out.mkdir(exist_ok=True);states={}
    def task(r,m):
        dest=out/f'r{r}-mb{m}';dest.mkdir(exist_ok=True)
        with (dest/'batch-late.log').open('a') as log:ret=subprocess.run([sys.executable,str(ROOT/'workflow/data_foundation/calibrate256_combined_r3.py'),'--rank-base',str(r),'--mb',str(m)],stdout=log,stderr=log).returncode
        if ret:return dict(status='FAILED',code=ret,log=str(dest/'batch-late.log'))
        d=load(dest/'shared-binding-r3/report.json');return dict(status='PASS' if all(v['status'].startswith('PASS') for v in d['phases'].values()) else 'REVIEW')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fs={pool.submit(task,r,m):(r,m) for r,m in JOBS}
        for f in concurrent.futures.as_completed(fs):
            r,m=fs[f];states[f'{r}:{m}']=f.result();save(out/'late-progress.json',dict(jobs=JOBS,states=states));print(r,m,states[f'{r}:{m}']['status'],len(states),'/',len(JOBS),flush=True)
if __name__=='__main__':main()

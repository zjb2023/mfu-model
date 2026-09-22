"""Resolve alternate rank0 hosts from the existing 14-case manifest."""
import argparse,concurrent.futures,hashlib,json
from pathlib import Path
from audit_startup_cohort_r5 import extract,save
ROOT=Path(__file__).resolve().parents[2]
REF=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/multi-strategy-r2/input_manifest.json')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    m=json.loads(REF.read_text());jobs=[];inventory=[]
    for case in ['2212','2411','2412','4212','4411']:
        entry=next(r for r in m['items'] if r['rank']==0 and r['case']==case)
        root=Path(entry['path']).parent.parent;available=sorted(int(p.name.split('_')[1]) for p in root.glob('iteration_*'))
        missing=[];selected=[]
        for it in range(8,104,5):
            p=root/f'iteration_{it}/gpu0.json'
            if not p.is_file():missing.append(it);continue
            selected.append(it);jobs.append(dict(case='16-'+case,world=16,rank=0,iteration=it,path=str(p),resolved_path=str(p.resolve())))
        inventory.append(dict(case='16-'+case,world=16,root=str(root),available=available,requested=list(range(8,104,5)),selected=selected,missing=missing))
    save(out/'inventory.json',inventory);rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for r in pool.map(extract,jobs):
            rows.append(r);save(out/f'{r["case"]}-iter{r["iteration"]}.json',r)
            if len(rows)%10==0 or r['status']!='PASS':print(len(rows),r['case'],r['iteration'],r['status'],r.get('error',''),flush=True)
    save(out/'report.json',dict(inventory=inventory,rows=rows))
    save(out/'manifest.json',dict(inputs={str(REF):hashlib.sha256(REF.read_bytes()).hexdigest()},code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),Path(__file__).with_name('audit_startup_cohort_r5.py')]},raw=[{k:r[k] for k in ['path','sha256','bytes']} for r in rows if 'sha256' in r],outputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')}))
if __name__=='__main__':main()

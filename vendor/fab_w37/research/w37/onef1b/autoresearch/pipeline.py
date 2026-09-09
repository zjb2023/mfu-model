#!/usr/bin/env python3
"""Launch only the dedicated Snakemake DAG, with a deadline and worktree-local I/O."""
import argparse
from datetime import datetime,timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path('/home/zjb/Desktop/fabric-data-analysis')


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--dag',action='store_true');a=p.parse_args()
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.run_id):p.error('simple run-id required')
    spec=a.spec.resolve();assert spec.is_relative_to(ROOT/'docs/w37/1f1b');plan=json.loads(spec.read_text())
    deadline=datetime.fromisoformat(plan['deadline_utc'].replace('Z','+00:00'));remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
    if remaining<=0:raise SystemExit('deadline reached; no new pipeline run')
    run=ROOT/'results/w37/A'/('autoresearch-'+a.run_id);run.mkdir(parents=True,exist_ok=True)
    controls=[p for directory in [Path(__file__).parent,ROOT/'research/w37/onef1b/post685'] for p in directory.glob('*.py')]
    controls+=[spec,ROOT/'workflow/w37/onef1b/Snakefile',ROOT/'scripts/w37/smoke_worker.py']
    controls+=[ROOT/p for p in ['docs/w37/coordination/code_manifest.json','docs/w37/coordination/external_inputs.json',
        'docs/w37/1f1b/additional_inputs.json','docs/w37/1f1b/post685/inputs.json']]
    if plan.get('extra_input_manifest'):
        extra=(ROOT/plan['extra_input_manifest']).resolve();assert extra.is_relative_to(ROOT/'docs/w37/1f1b');controls.append(extra)
    if plan.get('resource_review_file'):
        resource_review=(ROOT/plan['resource_review_file']).resolve();assert resource_review.is_relative_to(ROOT/'docs/w37/1f1b')
        assert file_sha(resource_review)==plan['resource_review_sha256'];controls.append(resource_review)
    contract={'files':[dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p)) for p in sorted(set(controls))]}
    frozen=run/'pipeline_contract.json'
    if frozen.exists():
        if json.loads(frozen.read_text())!=contract:raise SystemExit('run code/spec/input-manifest changed; use a new run id')
    else:
        frozen.write_text(json.dumps(contract,indent=2)+'\n')
        for p in controls:
            copy=run/'control_snapshot'/p.relative_to(ROOT);copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(p.read_bytes())
    for name in ['cache','tmp']: (run/name).mkdir(exist_ok=True)
    cmd=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--bind',str(run),str(run),
         '--dev','/dev','--proc','/proc','--tmpfs','/tmp','--unshare-net',
         '--chdir',str(ROOT),'--',str(SOURCE/'.snakemake-venv/bin/python'),'-B','-m','snakemake',
         '--snakefile',str(ROOT/'workflow/w37/onef1b/Snakefile'),'--directory',str(run),
         '--cores','2','--resources','mem_mb=4096','--rerun-incomplete','--printshellcmds',
         '--config','run_root='+str(run),'spec='+str(spec)]
    if a.dry_run:cmd+=['--dry-run']
    if a.dag:cmd+=['--dag','dot']
    label='dag' if a.dag else ('dry_run' if a.dry_run else 'pipeline')
    for suffix in ['_command.json','.dot' if a.dag else '.log']:
        old=run/(label+suffix)
        if old.exists():old.rename(old.with_name(old.name+'.prior-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
    (run/(label+'_command.json')).write_text(json.dumps({'argv':cmd,'deadline_utc':plan['deadline_utc'],
         'start_utc':datetime.now(timezone.utc).isoformat(),'snakemake_version':'9.23.1','default_global_workflow_used':False},indent=2)+'\n')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
         XDG_CACHE_HOME=str(run/'cache'),TMPDIR=str(run/'tmp'))
    logfile=run/(label+('.dot' if a.dag else '.log'))
    with logfile.open('w') as f:
        try:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        except subprocess.TimeoutExpired:
            (run/'deadline_reached.json').write_text(json.dumps({'status':'DEADLINE_REACHED','at_utc':datetime.now(timezone.utc).isoformat()})+'\n');raise
    print(logfile.read_text()[-10000:]);print(run);return r.returncode


if __name__=='__main__':raise SystemExit(main())

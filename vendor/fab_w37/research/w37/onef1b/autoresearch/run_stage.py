#!/usr/bin/env python3
"""One Snakemake stage, original/code read-only, current result directory writable."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/w37'))
from smoke_worker import sha,dump,SOURCE


def stage_path(root,stage,variant):
    if stage in ['model','seal','evaluate']:
        return root/{'model':'models','seal':'seals','evaluate':'evaluations'}[stage]/variant
    return root/stage


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','source_audit','model','seal','evaluate','diagnose','summary'],required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--variant',default='')
    a=p.parse_args();run=a.run_root.resolve();spec=a.spec.resolve();plan=json.loads(spec.read_text())
    assert run.is_relative_to(ROOT/'results/w37/A') and spec.is_relative_to(ROOT/'docs/w37/1f1b')
    for item in json.loads((run/'pipeline_contract.json').read_text())['files']:
        if sha(ROOT/item['path'])!=item['sha256']:raise SystemExit('pipeline control changed during run: '+item['path'])
    if a.stage in ['model','seal','evaluate']:assert a.variant in plan['variants']
    end=datetime.fromisoformat(plan['deadline_utc'].replace('Z','+00:00'));remaining=(end-datetime.now(timezone.utc)).total_seconds()
    if remaining<=0 and a.stage not in ['seal','evaluate','summary']:raise SystemExit('deadline reached: no new scientific job')
    out=stage_path(run,a.stage,a.variant)
    if out.exists():
        if (out/'complete.json').exists():raise SystemExit('immutable completed stage exists; use a new wave/run id for changed inputs or code')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        out.rename(out.with_name(out.name+'.failed-'+stamp))
    out.mkdir(parents=True)
    cmd=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--bind',str(out),str(out),'--tmpfs','/tmp','--unshare-net','--chdir',str(ROOT),'--',
         str(SOURCE/'.venv/bin/python'),'-B',str(Path(__file__).with_name('stage_impl.py')),
         '--stage',a.stage,'--run-root',str(run),'--spec',str(spec),'--variant',a.variant,'--output',str(out)]
    code=[p for directory in [Path(__file__).parent,ROOT/'research/w37/onef1b/post685'] for p in directory.glob('*.py')]
    dump(out/'command.json',{'argv':cmd,'spec':str(spec),'spec_sha256':sha(spec),'code':[dict(path=str(p),sha256=sha(p)) for p in code],
         'deadline_utc':plan['deadline_utc'],'start_utc':datetime.now(timezone.utc).isoformat(),
         'resource_policy':('one CPU; explicitly pinned target raw files parsed only by posthoc evaluator diagnose after verified seal, hash-only elsewhere; source/model access denied' if plan.get('target_raw_trace_intake') or plan.get('target_training_log_intake') else ('one CPU; explicitly pinned source raw files may be parsed only in bounded source diagnose, hash-only in other stages; no network or sibling writes' if plan.get('raw_trace_intake') or plan.get('source_training_log_intake') else 'one CPU thread; pinned derived inputs only; no raw traces, network or sibling writes'))})
    snapshots=out/'code_snapshot';snapshots.mkdir()
    for path in code:
        name=path.parent.name+'__'+path.name;(snapshots/name).write_bytes(path.read_bytes())
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mplconfig'))
    with (out/'execution.log').open('w') as f:
        try:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(30,min(3600,remaining)))
        except subprocess.TimeoutExpired:
            dump(out/'failure.json',{'status':'DEADLINE_OR_STAGE_TIMEOUT','stage':a.stage});raise
    artifacts=[dict(path=str(p.relative_to(out)),sha256=sha(p),size_bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file()]
    dump(out/'run_manifest.json',{'exit_code':r.returncode,'artifacts':artifacts,'end_utc':datetime.now(timezone.utc).isoformat()})
    if r.returncode:
        print((out/'execution.log').read_text()[-6000:]);return r.returncode
    dump(out/'complete.json',{'status':'PASS','stage':a.stage,'variant':a.variant,'manifest_sha256':sha(out/'run_manifest.json')})
    print(f'{a.stage} {a.variant}: PASS -> {out}')
    return 0


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only external inputs, worktree-local outputs; no original-trace scanner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/zjb/Desktop/fabric-data-analysis')
PYTHON = SOURCE / '.venv/bin/python'
CONTROL = ROOT/'docs/w37/coordination'

def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):d.update(block)
    return d.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['A','B'],required=True)
    p.add_argument('--mode',choices=['smoke','reproduce-v685'],default='smoke')
    p.add_argument('--run-id',default=None)
    a=p.parse_args()
    if ROOT==SOURCE:raise SystemExit('Use a prepared worktree, not the original repository.')
    if a.task=='B' and a.mode!='smoke':raise SystemExit('Task B must not run source256-calibrated v685.')
    tag=a.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in tag):
        raise SystemExit('run-id must be a simple filename')
    output=ROOT/'results/w37'/a.task/(a.mode+'-'+tag)
    if output.exists():raise SystemExit(f'Existing output preserved: {output}; choose another run-id.')
    output.mkdir(parents=True)
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    command=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/',
             '--bind',str(output),str(output),'--tmpfs','/tmp','--unshare-net',
             '--chdir',str(ROOT),'--',str(PYTHON),'-B',str(ROOT/'scripts/w37/smoke_worker.py'),
             '--task',a.task,'--mode',a.mode,'--output',str(output),'--commit',commit]
    (output/'command.json').write_text(json.dumps({'argv':command,'python':str(PYTHON),
        'python_sha256':sha(PYTHON.resolve()),'worktree':str(ROOT),'commit':commit,
        'filesystem':'/ read-only; only this run output and private /tmp writable; network unshared'},indent=2)+'\n')
    with (output/'execution.log').open('w') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    artifacts=[]
    for path in sorted(output.rglob('*')):
        if path.is_file():artifacts.append({'path':str(path),'sha256':sha(path),'size_bytes':path.stat().st_size})
    (output/'run_manifest.json').write_text(json.dumps({'exit_code':result.returncode,
        'task':a.task,'mode':a.mode,'worktree':str(ROOT),'commit':commit,'artifacts':artifacts},indent=2)+'\n')
    print((output/'execution.log').read_text())
    print(json.dumps({'status':'PASS' if result.returncode==0 else 'FAIL', 'output':str(output)}))
    return result.returncode

if __name__=='__main__':raise SystemExit(main())

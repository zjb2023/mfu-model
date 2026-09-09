#!/usr/bin/env python3
"""Protected post-v685 research; old model/runner/history remain unchanged."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/w37'))
from run_baseline import PYTHON,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['audit','candidate'],default='audit')
    p.add_argument('--study',choices=['v686','v687','v688'],default='v688')
    p.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    a=p.parse_args()
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.run_id):p.error('simple run-id required')
    out=ROOT/'results/w37/A'/f'post685-{a.mode}-{a.run_id}';out.mkdir(parents=True,exist_ok=False)
    cmd=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--bind',str(out),str(out),'--tmpfs','/tmp','--unshare-net',
        '--chdir',str(ROOT),'--',str(PYTHON),'-B',str(Path(__file__).with_name('worker.py')),'--mode',a.mode,'--study',a.study,'--output',str(out)]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mplconfig'))
    code=[*Path(__file__).parent.glob('*.py'),ROOT/'scripts/w37/smoke_worker.py',*list((ROOT/'docs/w37/1f1b/post685').glob('*.json'))]
    (out/'command.json').write_text(json.dumps({'argv':cmd,'code':[{'path':str(p),'sha256':sha(p)} for p in code],
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'resource_policy':'small pinned derived tables, single thread, no raw traces or external child process'},indent=2)+'\n')
    with (out/'execution.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    artifacts=[{'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()]
    (out/'run_manifest.json').write_text(json.dumps({'exit_code':r.returncode,'artifacts':artifacts},indent=2)+'\n')
    print((out/'execution.log').read_text());print(out);return r.returncode


if __name__=='__main__':raise SystemExit(main())

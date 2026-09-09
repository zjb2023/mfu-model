#!/usr/bin/env python3
"""Run bounded W37 Task A research with the baseline's read-only mount contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/w37'))
from run_baseline import PYTHON, SOURCE, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['audit', 'experiment'], default='audit')
    p.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    a = p.parse_args()
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.run_id):
        p.error('run-id must be a simple filename')
    out = ROOT / 'results/w37/A' / ('onef1b-' + a.mode + '-' + a.run_id)
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLCONFIGDIR=str(out / 'mplconfig'))
    cmd = ['/usr/bin/bwrap', '--die-with-parent', '--ro-bind', '/', '/', '--bind', str(out), str(out),
           '--tmpfs', '/tmp', '--unshare-net', '--chdir', str(ROOT), '--', str(PYTHON), '-B',
           str(Path(__file__).with_name('worker.py')), '--mode', a.mode, '--output', str(out)]
    code = [*Path(__file__).parent.glob('*.py'), ROOT/'scripts/w37/smoke_worker.py',
            ROOT/'docs/w37/1f1b/additional_inputs.json', ROOT/'docs/w37/coordination/external_inputs.json']
    (out/'command.json').write_text(json.dumps({'argv': cmd, 'commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code': [{'path': str(x), 'sha256': sha(x)} for x in code], 'python_sha256': sha(PYTHON.resolve()),
        'resource_policy': 'derived inputs only; single thread; no raw traces, subprocesses, network or unrelated simulation'},indent=2)+'\n')
    with (out/'execution.log').open('w') as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    artifacts = [{'path': str(x), 'sha256': sha(x), 'size_bytes': x.stat().st_size}
                 for x in sorted(out.rglob('*')) if x.is_file()]
    (out/'run_manifest.json').write_text(json.dumps({'exit_code': r.returncode, 'artifacts': artifacts},indent=2)+'\n')
    print((out/'execution.log').read_text())
    print(json.dumps({'status': 'PASS' if r.returncode == 0 else 'FAIL', 'output': str(out)}))
    return r.returncode


if __name__ == '__main__':
    raise SystemExit(main())

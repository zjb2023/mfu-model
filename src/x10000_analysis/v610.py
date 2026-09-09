"""Portable execution adapter for the byte-preserved W37 v6.10 algorithms.

Frozen code is vendored without editing its algorithms, defaults or old seals.
This module supplies relocated inputs and isolated stage I/O explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'vendor/fab_w37'
STAGES = {'release': ('model', 'evaluate', 'render'),
          'binding': ('audit', 'model', 'review', 'evaluate')}
SPECS = {'release': 'docs/w37/1f1b/v610/config.json',
         'binding': 'docs/w37/1f1b/binding/inputs.json'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def manifest():
    return json.loads((ROOT / 'docs/v610/artifact_manifest.json').read_text())


def artifact_root():
    env = os.environ.get('MFU_V610_ARTIFACT_ROOT')
    if env:
        return Path(env).resolve()
    local = ROOT / '.local/v610.json'
    if not local.exists():
        raise ValueError('Set MFU_V610_ARTIFACT_ROOT or create .local/v610.json from configs/v610/paths.example.json')
    return Path(json.loads(local.read_text())['artifact_root']).resolve()


def resolve_artifact(logical, m=None, base=None):
    m = m if m is not None else manifest()
    base = base if base is not None else artifact_root()
    if logical not in m['logical_paths']:
        raise ValueError('Unregistered frozen input: ' + logical)
    path = base / m['logical_paths'][logical]
    # A moved file may resolve outside base through a compatibility symlink.
    # Identity is still mandatory by SHA; this never edits the historical hash.
    return path


def mapped_spec(pipeline):
    spec = json.loads((VENDOR / SPECS[pipeline]).read_text())
    m, base = manifest(), artifact_root()
    for record in spec['inputs'].values():
        record['logical_path'] = record['path']
        record['path'] = str(resolve_artifact(record['path'], m, base))
    return spec


def checked_input(record):
    path = Path(record['path'])
    actual = sha(path) if path.is_file() else None
    if actual != record['sha256'] or path.stat().st_size != record['size_bytes']:
        raise ValueError(f"Frozen SHA mismatch: {path}; expected={record['sha256']}; actual={actual}")
    return dict(logical_path=record.get('logical_path', str(path)), path=str(path),
                resolved_path=str(path.resolve()), sha256=actual, role=record['role'])


def checked_run(path):
    run = Path(path).resolve()
    if not run.is_relative_to(ROOT / 'results/v610'):
        raise ValueError('Outputs must be inside this worktree/results/v610')
    return run


def stage_code(pipeline):
    return VENDOR / 'research/w37/onef1b' / ('v610' if pipeline == 'release' else 'binding')


def launch(pipeline, stage, run):
    run = checked_run(run)
    if stage not in STAGES[pipeline]:
        raise ValueError('Invalid stage')
    out = run / stage
    if out.exists() and any(out.iterdir()):
        raise ValueError('Nonempty frozen stage; choose a new run directory: ' + str(out))
    spec = mapped_spec(pipeline)
    verified = {k: checked_input(r) for k, r in spec['inputs'].items()}
    for rel, expected in spec['code_inputs'].items():
        if sha(VENDOR / rel) != expected:
            raise ValueError('Frozen code mismatch: ' + rel)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'resolved_inputs.json', verified)
    dump(out / 'execution_config.json', spec)
    # All existing projects are read-only AND old providers are hidden, proving
    # this run uses the relocated bundle rather than the former paths/venv.
    hidden = ['/home/zjb/Desktop/worktrees/fab-w37-1f1b',
              '/home/zjb/Desktop/fabric-data-analysis', '/home/zjb/Desktop/mfu-model']
    cmd = ['/usr/bin/bwrap', '--die-with-parent', '--ro-bind', '/', '/',
           '--bind', str(out), str(out), '--dev', '/dev', '--tmpfs', '/tmp', '--unshare-net']
    for path in hidden:
        if Path(path).exists() and Path(path).resolve() != ROOT:
            cmd += ['--tmpfs', path]
    cmd += ['--chdir', str(ROOT), '--', sys.executable, '-B',
            str(ROOT / 'scripts/v610/worker.py'), pipeline, stage, str(run)]
    tracked_code = [ROOT / 'src/x10000_analysis/v610.py', ROOT / 'scripts/v610/worker.py']
    tracked_code += [f for f in stage_code(pipeline).iterdir() if f.is_file()]
    snapshot = out / 'code_snapshot'
    for f in tracked_code:
        target = snapshot / f.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f.read_bytes())
    dump(out / 'command.json', dict(argv=cmd, pipeline=pipeline, stage=stage,
        vendor_algorithm_sha256={str(f.relative_to(ROOT)):sha(f) for f in tracked_code},
        hidden_original_projects=hidden, input_verification='hash only before role guard',
        python=sys.version, timeout_seconds=900, threads=1))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT/'src'),
               MFU_V610_ARTIFACT_ROOT=str(artifact_root()))
    start = time.monotonic()
    with (out / 'execution.log').open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        print((out / 'execution.log').read_text()[-7000:])
        raise RuntimeError(f'{pipeline}/{stage} failed: {result.returncode}')
    dump(out / 'complete.json', dict(status='PASS', elapsed_s=time.monotonic()-start,
        files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}))
    print(pipeline, stage, 'PASS', round(time.monotonic()-start, 2), flush=True)


def worker(pipeline, stage, run):
    run = checked_run(run); out = run / stage
    code = stage_code(pipeline)
    sys.path.insert(0, str(code))
    common = importlib.import_module('common')
    common.RUN = run
    common.SPEC = out / 'execution_config.json'
    spec = json.loads(common.SPEC.read_text())
    paths = {k:Path(r['path']) for k,r in spec['inputs'].items()}
    allowed = {str(p.resolve()) for k,p in paths.items()
               if stage in ('evaluate','render') or spec['inputs'][k]['role'] != 'evaluator'}
    previous = list(STAGES[pipeline][:STAGES[pipeline].index(stage)])
    permitted = [Path('/usr'),Path('/lib'),Path('/etc'),Path(sys.base_prefix),Path(sys.prefix),
                 VENDOR/'research/w37/onef1b',VENDOR/'case_224gpu_pp14_cp2_a2a/scripts',
                 VENDOR/'scripts/w37',VENDOR/'src',ROOT/'src']
    reads = set()
    def guard(event,args):
        if event in ('subprocess.Popen','os.system','socket.connect'):
            raise PermissionError(event)
        if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            if not p.is_relative_to(out):
                raise PermissionError('write outside current stage: ' + str(p))
            return
        if p.is_relative_to(out): return
        if str(p) in allowed: reads.add(str(p)); return
        if any(p.is_relative_to(d) for d in permitted): return
        if any(p.is_relative_to(run/s) for s in previous): return
        if str(p) in ('/dev/null','/dev/urandom','/proc/meminfo','/proc/cpuinfo'): return
        raise PermissionError('undeclared or evaluator-only input: ' + str(p))
    sys.addaudithook(guard)
    def setup(*args): return spec,paths,reads
    common.setup = setup
    sys.argv = [str(code/'worker.py'),stage]
    try:
        runpy.run_path(str(code/'worker.py'),run_name='__main__')
    finally:
        dump(out/'adapter_access.json',dict(pipeline=pipeline,stage=stage,interpreted_data_reads=sorted(reads),
            target_role_enabled=stage in ('evaluate','render'),raw_trace_scan=False,
            python_prefix=sys.prefix,original_providers_hidden=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pipeline',choices=STAGES,default='release')
    p.add_argument('--stage')
    p.add_argument('--run-root',type=Path,required=True)
    a=p.parse_args()
    for stage in ([a.stage] if a.stage else STAGES[a.pipeline]):
        launch(a.pipeline,stage,a.run_root)


if __name__=='__main__': main()

"""Run one immutable release stage with a read-only filesystem and no network."""
import argparse,os,subprocess,time
from pathlib import Path
from common import ROOT,RUN,SPEC,dump,sha
p=argparse.ArgumentParser();p.add_argument('stage',choices=['model','evaluate','render']);p.add_argument('--run-root',type=Path,default=RUN);a=p.parse_args()
run=a.run_root.resolve();assert run.is_relative_to(ROOT/'results/w37/A');out=run/a.stage
if out.exists() and any(out.iterdir()):raise SystemExit('Nonempty release stage: use a new run directory')
out.mkdir(parents=True,exist_ok=True)
code=Path(__file__).parent
code_files=[p for p in code.iterdir() if p.is_file() and (p.suffix in ['.py','.js'] or p.name=='Snakefile')]
cmd=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--bind',str(out),str(out),'--tmpfs','/tmp','--unshare-net','--chdir',str(ROOT),'--','/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python','-B',str(code/'worker.py'),a.stage]
dump(out/'command.json',dict(argv=cmd,spec_sha256=sha(SPEC),code={p.name:sha(p) for p in code_files},threads=1,timeout_seconds=900))
snap=out/'code_snapshot';snap.mkdir()
for f in code_files:(snap/f.name).write_bytes(f.read_bytes())
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TASK_V610_RUN=str(run))
t=time.monotonic()
with (out/'execution.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=900)
if r.returncode:print((out/'execution.log').read_text()[-6000:]);raise SystemExit(r.returncode)
dump(out/'complete.json',dict(status='PASS',elapsed_s=time.monotonic()-t,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
print(a.stage,'PASS')

"""Stage runner: read-only filesystem, private outputs, one CPU, no network."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,RUN,SPEC,dump,sha
p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','model','review','evaluate']);p.add_argument('--run-root',type=Path,default=RUN);a=p.parse_args()
RUN=a.run_root.resolve();assert RUN.is_relative_to(ROOT/'results/w37/A')
out=RUN/a.stage
if out.exists() and any(out.iterdir()):raise SystemExit('Preserve existing stage; choose a new run or archive failed stage.')
out.mkdir(parents=True,exist_ok=True)
cmd=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--bind',str(out),str(out),'--tmpfs','/tmp','--unshare-net','--chdir',str(ROOT),'--','/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python','-B',str(ROOT/'research/w37/onef1b/binding/worker.py'),a.stage]
dump(out/'command.json',{'argv':cmd,'spec_sha256':sha(SPEC),'code':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'limits':{'threads':1,'timeout_s':900,'raw_trace':False}})
snapshot=out/'code_snapshot';snapshot.mkdir()
for p in Path(__file__).parent.glob('*.py'):(snapshot/p.name).write_bytes(p.read_bytes())
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TASK_BINDING_RUN=str(RUN))
t=time.monotonic()
with (out/'execution.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
if r.returncode:print((out/'execution.log').read_text()[-6000:]);raise SystemExit(r.returncode)
files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
dump(out/'complete.json',{'status':'PASS','elapsed_s':time.monotonic()-t,'files':files});print(a.stage,'PASS',round(time.monotonic()-t,2))

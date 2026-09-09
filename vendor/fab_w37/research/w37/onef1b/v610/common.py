"""Frozen release inputs; target observations are evaluator-only."""
import csv,gzip,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
SPEC=ROOT/'docs/w37/1f1b/v610/config.json'
RUN=Path(os.environ.get('TASK_V610_RUN',ROOT/'results/w37/A/v610-release')).resolve()
assert RUN.is_relative_to(ROOT/'results/w37/A')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def verify(stage):
    p=RUN/stage;m=json.loads((p/'complete.json').read_text());assert m['status']=='PASS'
    for n,h in m['files'].items():assert sha(p/n)==h,n

def setup(stage):
    spec=json.loads(SPEC.read_text());paths={k:Path(v['path']) for k,v in spec['inputs'].items()}
    for k,p in paths.items():
        r=spec['inputs'][k];assert p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],k
    for p,h in spec['code_inputs'].items():assert sha(ROOT/p)==h,p
    allowed={str(p.resolve()) for k,p in paths.items() if stage!='model' or spec['inputs'][k]['role']!='evaluator'}
    reads=set();out=RUN/stage
    permitted=[Path('/usr'),Path('/lib'),Path('/etc'),Path(sys.base_prefix),Path('/home/zjb/Desktop/fabric-data-analysis/.venv'),ROOT/'research/w37/onef1b/v610',ROOT/'research/w37/onef1b/integration',ROOT/'case_224gpu_pp14_cp2_a2a/scripts',ROOT/'scripts/w37']
    previous={'model':[],'evaluate':['model'],'render':['model','evaluate']}.get(stage,[])
    def hook(event,args):
        if event in ['subprocess.Popen','os.system','socket.connect']:raise PermissionError(event)
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();writing=args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
        if writing:
            if not p.is_relative_to(out):raise PermissionError('write outside release stage: '+str(p))
            return
        if p.is_relative_to(out):return
        if str(p) in allowed:reads.add(str(p));return
        if any(p.is_relative_to(d) for d in permitted):return
        if any(p.is_relative_to(RUN/d) for d in previous):return
        if str(p) in ['/dev/null','/dev/urandom','/proc/meminfo','/proc/cpuinfo']:return
        raise PermissionError('undeclared or target-only input: '+str(p))
    sys.addaudithook(hook)
    return spec,paths,reads

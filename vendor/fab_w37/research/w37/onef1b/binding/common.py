"""Exact derived inputs only; target observations are restricted to scoring."""
import csv, gzip, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT/'docs/w37/1f1b/binding/inputs.json'
RUN = Path(os.environ.get('TASK_BINDING_RUN',ROOT/'results/w37/A/binding-20260908')).resolve()
assert RUN.is_relative_to(ROOT/'results/w37/A')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def write_csv(p,rows,fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt',newline='') if p.suffix=='.gz' else p.open('w',newline='')) as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def setup(stage,out):
    spec=json.loads(SPEC.read_text()); paths={k:Path(v['path']) for k,v in spec['inputs'].items()}
    for name,expected in spec.get('code_inputs',{}).items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen code mismatch: '+name)
    # Verify without interpreting target data before restricted stage execution.
    for k,p in paths.items():
        v=spec['inputs'][k]
        if p.stat().st_size!=v['size_bytes'] or sha(p)!=v['sha256']:raise ValueError('Frozen mismatch: '+k)
    allowed={str(p.resolve()):k for k,p in paths.items() if stage=='evaluate' or spec['inputs'][k]['role']!='evaluator'}
    reads=set()
    code_dirs=[ROOT/'research/w37/onef1b/binding',ROOT/'scripts/w37',ROOT/'case_224gpu_pp14_cp2_a2a/scripts']
    lib_dirs=[Path('/usr'),Path('/lib'),Path('/etc'),Path(sys.base_prefix),Path('/home/zjb/Desktop/fabric-data-analysis/.venv')]
    def hook(event,args):
        if event in ['subprocess.Popen','os.system','socket.connect']:raise PermissionError(event)
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();flags=args[2]
        writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing:
            if not p.is_relative_to(out):raise PermissionError('write outside stage: '+str(p))
            return
        if p.is_relative_to(out):return
        if str(p) in allowed:reads.add(str(p));return
        if any(p.is_relative_to(d) for d in lib_dirs+code_dirs):return
        allowed_stage={'audit':[], 'model':['audit'], 'review':['audit','model'], 'evaluate':['audit','model','review']}.get(stage,[])
        if any(p.is_relative_to(RUN/d) for d in allowed_stage):return
        if str(p) in ['/dev/null','/dev/urandom','/proc/meminfo','/proc/cpuinfo']:return
        raise PermissionError('undeclared or evaluator-only input: '+str(p))
    sys.addaudithook(hook)
    return spec,paths,reads

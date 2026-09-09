"""Read only the exact frozen inputs admitted for the integration report."""
from pathlib import Path
import csv, gzip, hashlib, json
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/w37/A/integration-20260907'
DOC=ROOT/'docs/w37/1f1b/integration'
SPEC=DOC/'inputs.json'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_spec():return json.loads(SPEC.read_text(encoding='utf-8'))
def path(key):
    record=load_spec()['inputs'][key]
    p=Path(record['path'])
    return p if p.is_absolute() else ROOT/p

def text(key):return path(key).read_text(encoding='utf-8')
def obj(key):return json.loads(text(key))
def iterrows(key):
    p=path(key)
    with (gzip.open(p,'rt',encoding='utf-8',newline='') if p.suffix=='.gz' else p.open(encoding='utf-8',newline='')) as f:
        yield from csv.DictReader(f)
def rows(key):return list(iterrows(key))
def dump(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_csv(name,data,fields=None):
    if not data and not fields:raise ValueError(name+' empty')
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)

def verify_inputs():
    records=[]
    for key,r in load_spec()['inputs'].items():
        p=path(key)
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
            raise ValueError('Frozen input changed: '+key+' '+str(p))
        records.append({'key':key,**r,'verified':True})
    return records

"""Second EP8 replica: same extraction and independent sender/receiver/FC checks."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import extract_layer_workload_curves_r1 as extraction
import audit_layer_filtering_r1 as audit
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
OUT=BASE/'layer-workload-edpB-r1'

def main():
    OUT.mkdir(exist_ok=True)
    rp=BASE/'b-position-curves-r2/raw-provenance.json'
    refs=[r for r in json.loads(rp.read_text()) if r['world']==256 and r['iteration']==60]
    root=Path(refs[0]['path']).parents[3]
    index={}
    for p in root.glob('*/profiler/iteration_60/rank*.pt.trace.json'):
        rank=int(p.name.split('.')[0][4:]);assert rank not in index;index[rank]=p
    todo=[dict(world=256,iteration=60,stage=r['stage'],path=str(index[rank])) for r in refs for rank in range(r['stage']*16+8,r['stage']*16+16)]
    assert len(todo)==112
    def work(ref):
        rank=int(Path(ref['path']).name.split('.')[0][4:])
        p=OUT/f'w256-i60-s{ref["stage"]}-r{rank}.json'
        if p.exists():
            d=json.loads(p.read_text())
        else:
            d=extraction.extract(ref)
            assert d['ep_group']==list(range(ref['stage']*16+8,ref['stage']*16+16))
            d['scope']='256 iter60 second EP8/EDP replica; no cross-host timestamp comparison'
            p.write_text(json.dumps(d,separators=(',',':'))+'\n')
        print('B extraction PASS',ref['stage'],rank,flush=True)
        return {k:d[k] for k in ('world','iteration','stage','rank','path','sha256')}
    with ThreadPoolExecutor(max_workers=4) as pool:done=list(pool.map(work,todo))
    codes=[Path(__file__).resolve(),Path(extraction.__file__),Path(audit.__file__),Path(extraction.__file__).with_name('extract_cohort14_expert_r1.py')]
    (OUT/'manifest.json').write_text(json.dumps(dict(raw=done,code={str(p):audit.digest(p) for p in codes},outputs={p.name:audit.digest(p) for p in OUT.glob('w*.json')}),indent=2)+'\n')
    audit.OUT=BASE/'layer-filtering-edpB-r1';audit.SOURCE=OUT
    audit.SCOPE='256 iter60 second EP8 replica, all middle stages and four microbatches'
    audit.main()

if __name__=='__main__':main()

"""Target F evidence using exact CPU marker/correlation, not file ordinal.

The generated adapter is preserved beside outputs. This leaves the frozen
legacy extractor and all in-flight r1 calibration jobs unchanged.
"""
import argparse,inspect,json,types
from pathlib import Path
import orjson
import pp32_f_split_r8 as old
from match_cp_marker_identity_r1 import match_marker
from calibrate256_fine_r1 import BASE,RAW,load,save,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank-base',type=int,required=True);ap.add_argument('--mb',type=int,required=True);a=ap.parse_args()
    out=BASE/'calibrate256-cp-identity-r2'/f'r{a.rank_base}-mb{a.mb}';out.mkdir(parents=True,exist_ok=False)
    code=inspect.getsource(old.extract)
    before="pg=str(args['Process Group Name']);j=cpcount[pg];cpcount[pg]+=1\n            candidates=[m for m in markers if m['kind']=='CP' and m['members']==groups[pg]['ranks']];m=candidates[j]"
    after="mi,marker,association=match_marker(es,r,e,f)\n            m=next(m for m in markers if m['cpu_event']==mi)\n            m['identity_association']=association"
    assert code.count(before)==1;code=code.replace(before,after)
    namespace=dict(old.__dict__);namespace['match_marker']=match_marker;namespace['json']=types.SimpleNamespace(loads=orjson.loads,dumps=json.dumps)
    exec(compile(code,str(out/'extract.adapter.py'),'exec'),namespace);(out/'extract.adapter.py').write_text(code)
    refs=[]
    for r in range(a.rank_base,a.rank_base+8):
        paths=list(RAW.glob(f'*/profiler/iteration_60/rank{r}.*.pt.trace.json'));assert len(paths)==1;p=paths[0];d=load(p);meta=dict(path=str(p),sha256=sha(p),groups=d['distributedInfo']['pg_config']);del d
        x=namespace['extract'](r,path=p,iteration=60,metadata=meta,expected_world_size=256,expected_ep_members=list(range(a.rank_base,a.rank_base+8)),microbatch=a.mb)
        x['stage']=a.rank_base//16;x['metadata']=meta;save(out/f'rank-{r}.json',x);refs.append(dict(path=str(p),sha256=meta['sha256']));print('PASS identity',r,a.mb,flush=True)
    save(out/'report.json',dict(status='PASS_EXACT_CP_IDENTITY',raw=refs,CP_events=128,iteration=60,mb=a.mb,rank_base=a.rank_base,limits=['F extraction only, no detailed replay or prediction claim.']))
    save(out/'manifest.json',dict(code={str(p):sha(p) for p in [Path(__file__),Path(old.__file__),Path(__file__).with_name('match_cp_marker_identity_r1.py')]},outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':main()

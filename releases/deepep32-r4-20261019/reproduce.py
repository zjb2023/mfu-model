"""Portable frozen-DAG replay. Python >=3.9; no raw traces or third-party packages."""
import graphlib,gzip,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def main():
    manifest=load(ROOT/'manifest.json')
    for name,h in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    with gzip.open(ROOT/'model/expanded-dag.json.gz','rt') as f:g=json.load(f)
    end={};start={}
    for k in graphlib.TopologicalSorter({k:n['deps'] for k,n in g.items()}).static_order():
        start[k]=max((end[d] for d in g[k]['deps']),default=0)
        end[k]=start[k]+g[k]['cost_ms']
    pred=end['s0:B3']-start['s0:F0']
    expected=load(ROOT/'model/model-seal.json')['predicted_ms'];assert abs(pred-expected)<1e-6
    scopes=load(ROOT/'prediction-overview-r3/scope-comparison.json')['scopes']
    assert abs(pred-scopes['onef1b']['predicted_ms'])<1e-6
    stages=load(ROOT/'prediction-overview-r3/last4-overview.json')['stages']
    result={'hashes':'PASS','expanded_nodes':len(g),'onef1b_ms':pred,'scopes':{}}
    for key,ms in [('onef1b',pred),('body',pred+stages['tail_ms']['predicted_ms']),('profiler',pred+stages['tail_ms']['predicted_ms']+stages['startup_ms']['predicted_ms'])]:
        s=scopes[key];assert abs(ms-s['predicted_ms'])<1e-6
        cohorts={}
        for c in ['all20','last4']:
            errs=[100*(ms/r['observed_ms']-1) for r in s[c]['rows']]
            mape=statistics.mean(map(abs,errs));assert abs(mape-s[c]['mape_percent'])<1e-8
            cohorts[c]={'mape_percent':mape,'signed_mean_percent':statistics.mean(errs)}
        result['scopes'][key]={'predicted_ms':ms,**cohorts}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

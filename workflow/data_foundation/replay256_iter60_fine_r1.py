"""Read-only stdlib verification of frozen detailed graph; no trace required."""
import argparse,gzip,hashlib,json,graphlib
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('model',type=Path);a=ap.parse_args();m=json.loads((a.model/'manifest.json').read_text())
    for p,h in m['outputs'].items():assert hashlib.sha256((a.model/p).read_bytes()).hexdigest()==h,p
    with gzip.open(a.model/'graph.json.gz','rt') as f:g=json.load(f)
    ends={};starts={}
    for k in graphlib.TopologicalSorter({k:n['deps'] for k,n in g.items()}).static_order():
        n=g[k];assert n['cost_ms']>=0;start=max([ends[d] for d in n['deps']] or [0]);starts[k]=start;ends[k]=start+n['cost_ms']
    r=json.loads((a.model/'report.json').read_text());one=ends['s0:B3']-starts['s0:F0'];body=ends['tail::END']-starts['s0:F0'];assert abs(one-r['onef1b']['replay_ms'])<1e-5 and abs(body-r['body']['replay_ms'])<1e-5
    print(json.dumps(dict(status='PASS',nodes=len(g),onef1b_ms=one,training_body_ms=body),ensure_ascii=False))
if __name__=='__main__':main()

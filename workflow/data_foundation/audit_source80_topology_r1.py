"""Compare semantic DAG dependencies separately from trace gap partition nodes."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';SRC=BASE/'source80-fine-dag-r1'
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def project(g,keep):
    cache={}
    def nearest(k):
        if k in keep:return {k}
        if k not in cache:cache[k]=set().union(*(nearest(d) for d in g[k]['deps'])) if g[k]['deps'] else set()
        return cache[k]
    deps={k:set().union(*(nearest(d) for d in g[k]['deps'])) if g[k]['deps'] else set() for k in keep}
    # Remove redundant edges; underlying reachability is what the DAG expresses.
    ancestors={}
    def all_anc(k):
        if k not in ancestors:ancestors[k]=deps[k]|set().union(*(all_anc(d) for d in deps[k])) if deps[k] else set()
        return ancestors[k]
    return {k:sorted(ds-set().union(*(all_anc(d) for d in ds))) if ds else [] for k,ds in deps.items()}
def main():
    global SRC
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=SRC);a=ap.parse_args();SRC=a.source
    result={};inputs=[]
    for ph,name in [('F','four-layer-f-r1'),('B','b-recompute-r11')]:
        a=BASE/name/'graph.json';b=SRC/name/'graph.json';old,new=load(a),load(b);inputs.extend([a,b])
        def selected(g):
            if ph=='F':return {k for k,n in g.items() if n.get('kind') not in ['trace_handoff','attention_input_reorder_tail']}
            return {k for k,n in g.items() if n.get('kind') not in [None,'uncovered_trace_residual','gradient_processing_exposed_union','recompute_readiness']}
        os,ns=selected(old),selected(new);op,np=project(old,os),project(new,ns)
        differences=[dict(node=k,old=op[k],new=np[k]) for k in sorted(os&ns) if op[k]!=np[k]]
        result[ph]=dict(old_nodes=len(old),new_nodes=len(new),semantic_old=len(os),semantic_new=len(ns),
            missing_semantic_nodes=sorted(os-ns),added_semantic_nodes=sorted(ns-os),changed_semantic_dependencies=differences,
            old_kinds=dict(Counter(str(n.get('kind')) for n in old.values())),new_kinds=dict(Counter(str(n.get('kind')) for n in new.values())),
            omitted_from_comparison='trace gap/readiness and gradient-processing partition only; these costs remain in both executable DAGs')
    report=dict(checks=result,inputs=[dict(path=str(p),sha256=digest(p)) for p in inputs],code=dict(path=str(Path(__file__).resolve()),sha256=digest(Path(__file__))))
    report['status']='PASS' if all(not r[k] for r in result.values() for k in ['missing_semantic_nodes','added_semantic_nodes','changed_semantic_dependencies']) else 'REVIEW_REQUIRED'
    (SRC/'semantic-topology-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({p:{k:(len(v) if isinstance(v,list) else v) for k,v in r.items() if k not in ['old_kinds','new_kinds']} for p,r in result.items()},indent=2))
    print(report['status'])
if __name__=='__main__':main()

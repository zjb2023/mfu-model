"""Integrate per-B spatial evidence and explicitly scoped B-only oracle diagnosis."""
import copy,graphlib,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    out=BASE/'b-evidence-index-r1';out.mkdir(parents=True,exist_ok=False)
    prev=BASE/'1f1b-overestimate-ui-r1/index.html';live=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')
    assert sha(prev)==sha(live/'index.html'),'Unexpected live revision'
    op=BASE/'detailed-fb-32to256-r1/outer-prediction.json';bp=BASE/'1f1b-overestimate-r1/blocks.json';rp=BASE/'1f1b-overestimate-r1/report.json';cp=BASE/'b-position-curves-r2/data.json'
    paths=[prev,op,bp,rp,cp];seals={str(p):sha(p) for p in paths};nodes=load(op)['nodes'];blocks=load(bp);report=load(rp);curves=load(cp);assert curves['iterations']==[40,60,80]
    order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
    def run(stages):
        replacements={r['node']:r['observed_duration_ms'] for r in blocks if r['phase']=='B' and r['stage'] in stages};ends={}
        for k in order:ends[k]=max((ends[d] for d in nodes[k]['dependencies']),default=0)+replacements.get(k,nodes[k]['duration_ms'])
        return ends['s0:B3']
    assert abs(run([])-report['baseline_ms'])<1e-6
    mid=run(range(1,15));assert abs(mid-report['scenarios']['middle_B']['window_ms'])<1e-6
    all_b=run(range(16));truth=report['truth_ms']
    d=dict(iteration=60,world=256,truth_ms=truth,baseline_ms=report['baseline_ms'],middle_b_ms=mid,middle_b_error_percent=100*(mid/truth-1),all_b_ms=all_b,all_b_error_percent=100*(all_b/truth-1),scope='target GPU durations replace B only; F, PP costs and dependency graph fixed; diagnostic oracle, not new predictive accuracy',curve_iterations=[40,60,80],target_fitted=False,prediction_changed=False)
    dump(out/'b-only-diagnosis.json',d)
    js=Path(__file__).with_name('b_evidence_index_r1.js')
    (out/'index.html').write_text(prev.read_text().replace('</body>','<script>const B_EVIDENCE='+json.dumps(d,ensure_ascii=False)+';\n'+js.read_text()+'</script></body>'))
    assert all(sha(Path(p))==h for p,h in seals.items())
    dump(out/'manifest.json',dict(inputs=seals,code={str(p.resolve()):sha(p) for p in [Path(__file__),js]},outputs={str(p.resolve()):sha(p) for p in out.iterdir() if p.is_file()}))
    for name in ['index.html','manifest.json']:
        backup=live/(Path(name).stem+'-before-b-evidence'+Path(name).suffix);assert not backup.exists();shutil.copyfile(live/name,backup)
    for name in ['index.html','manifest.json','b-only-diagnosis.json']:shutil.copyfile(out/name,live/name)
    print(json.dumps(d,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

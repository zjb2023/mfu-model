"""Descriptive layer curves, not a predictive fit. Run after both extractors finish."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'


def corr(a, b):
    am, bm = sum(a)/len(a), sum(b)/len(b)
    den = math.sqrt(sum((x-am)**2 for x in a)*sum((x-bm)**2 for x in b))
    return sum((x-am)*(y-bm) for x, y in zip(a,b))/den if den else None


def main():
    rows, inputs = [], {}
    for world in (224,256):
        folder = BASE / f'layer-workload-{world}-r1'
        manifest = folder / 'manifest.json'
        assert manifest.exists(), 'Extraction incomplete'
        inputs[str(manifest)] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for p in sorted(folder.glob('w*.json')):
            d = json.loads(p.read_text()); rows.extend(d['rows'])
    assert len(rows) == 2672
    representative = [r for r in rows if r['rank'] == r['stage']*16]
    checks = 0
    # Exact regression against previously frozen B FC ownership/costs.
    for stage in range(1,15):
        old = json.loads((BASE/f'fc-transfer-r1-verified/pp{stage}.json').read_text())
        lookup = {(r['mb'],r['backward_execution_order']):r for r in representative if r['world']==256 and r['iteration']==60 and r['stage']==stage}
        for b in old['blocks']:
            for op in b['operators']:
                r = lookup[b['mb'],op['unit']]
                key = f"{op['phase']}_FC{op['FC']}_gemm_ms"
                assert abs(r[key]-op['gemm_ms']) < 1e-8, (stage,b['mb'],key)
                checks += 1
    groups=collections.defaultdict(list)
    for r in rows:
        if r['world']==256 and r['iteration']==60:
            groups[r['layer'],r['mb']].append(r)
    ep=[]
    for (layer,mb),rs in sorted(groups.items()):
        assert len(rs)==8 and len({r['rank'] for r in rs})==8
        loads=[r['token_rows'] for r in rs]; mean=sum(loads)/8
        rep=next(r for r in rs if r['rank']==r['stage']*16)
        ep.append(dict(layer=layer,stage=rs[0]['stage'],mb=mb,total_rows=sum(loads),max_rows=max(loads),min_rows=min(loads),cv=math.sqrt(sum((x-mean)**2 for x in loads)/8)/mean,
                       representative_rows=rep['token_rows'],representative_share=rep['token_rows']/sum(loads),
                       max_B_FC_gemm_ms=max(r['B_FC_gemm_ms'] for r in rs),sum_B_FC_gemm_ms=sum(r['B_FC_gemm_ms'] for r in rs),ranks=[dict(rank=r['rank'],rows=r['token_rows'],B_FC_gemm_ms=r['B_FC_gemm_ms']) for r in rs]))
    stats=[]
    stability=[]
    for world in (224,256):
        for iteration in (40,60,80):
            rs=[r for r in representative if r['world']==world and r['iteration']==iteration]
            stats.append(dict(world=world,iteration=iteration,n=len(rs),rows_min=min(r['token_rows'] for r in rs),rows_max=max(r['token_rows'] for r in rs),
                              correlations={p:corr([r['token_rows'] for r in rs],[r[p+'_FC_gemm_ms'] for r in rs]) for p in ('F','recompute','backward','B')}))
        for mb in range(4 if world==256 else 3):
            sets={i:sorted([r for r in representative if r['world']==world and r['iteration']==i and r['mb']==mb],key=lambda r:r['layer']) for i in (40,60,80)}
            for a,b in ((40,60),(60,80),(40,80)):
                aa,bb=sets[a],sets[b]
                stability.append(dict(world=world,mb=mb,a=a,b=b,workload_r=corr([r['token_rows'] for r in aa],[r['token_rows'] for r in bb]),cost_r=corr([r['B_FC_gemm_ms'] for r in aa],[r['B_FC_gemm_ms'] for r in bb])))
    report=dict(scope='Descriptive evidence only; representative rank across iter40/60/80; first EP8 replica complete at256 iter60 only. GPU GEMM union sums are not DAG makespans.',regression_checks=checks,rows=len(rows),stats=stats,stability=stability,ep8=ep,representative=representative)
    out=BASE/'layer-workload-analysis-r1'; out.mkdir(exist_ok=False)
    (out/'data.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,code={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},outputs={'data.json':hashlib.sha256((out/'data.json').read_bytes()).hexdigest()},checks=checks),indent=2)+'\n')
    print(json.dumps(dict(regression_checks=checks,stats=stats,stability=stability,ep8_total_range=[min(r['total_rows'] for r in ep),max(r['total_rows'] for r in ep)],ep8_cv_range=[min(r['cv'] for r in ep),max(r['cv'] for r in ep)]),indent=2))


if __name__=='__main__':
    main()

"""Aggregate both independently validated EP8 replicas; never sum elapsed costs."""
import json
import math
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    reports=[BASE/'layer-filtering-r1/report.json',BASE/'layer-filtering-edpB-r1/report.json']
    a,b=[json.loads(p.read_text()) for p in reports]
    assert all(d['status']=='PASS' and len(d['rows'])==224 and d['expert_conservation_checks']==35840 for d in (a,b))
    aa={(r['layer'],r['mb']):r for r in a['rows']};bb={(r['layer'],r['mb']):r for r in b['rows']};assert aa.keys()==bb.keys()
    old=json.loads((BASE/'layer-workload-analysis-r1/data.json').read_text());costA={(r['layer'],r['mb']):r for r in old['ep8']}
    sourceB={}
    for p in (BASE/'layer-workload-edpB-r1').glob('w*.json'):
        for r in json.loads(p.read_text())['rows']:sourceB.setdefault((r['layer'],r['mb']),[]).append(r)
    rows=[]
    for key in sorted(aa):
        x,y=aa[key],bb[key];s=x['stage'];assert x['ranks']==list(range(s*16,s*16+8)) and y['ranks']==list(range(s*16+8,s*16+16))
        rs=sourceB[key];assert len(rs)==8
        loads=[r['token_rows'] for r in rs];mean=sum(loads)/8
        assert sum(loads)==y['retained_assignments'] and costA[key]['total_rows']==x['retained_assignments']
        rows.append(dict(layer=key[0],mb=key[1],stage=s,groupA=x['retained_assignments'],groupB=y['retained_assignments'],total=x['retained_assignments']+y['retained_assignments'],
                         offered=x['offered_assignments']+y['offered_assignments'],not_entered=x['inferred_removed_assignments']+y['inferred_removed_assignments'],ranksA=x['ranks'],ranksB=y['ranks'],
                         max_cost_A=costA[key]['max_B_FC_gemm_ms'],max_cost_B=max(r['B_FC_gemm_ms'] for r in rs),cv_A=costA[key]['cv'],cv_B=math.sqrt(sum((x-mean)**2 for x in loads)/8)/mean))
    for r in rows:assert r['offered']==786432 and r['total']+r['not_entered']==r['offered']
    comparisons=[]
    for mb in range(4):
        rs=[r for r in rows if r['mb']==mb];x,y=rs[0],rs[-1]
        comparisons.append(dict(mb=mb,first=x,last=y,total_decrease=x['total']-y['total'],total_decrease_pct=100*(x['total']-y['total'])/x['total']))
    out=BASE/'layer-edp-pair-r1';out.mkdir(exist_ok=False)
    report=dict(status='PASS',scope='256 iter60 PP1..14 both EP8 groups, 16 ranks/stage, four microbatches; not head/tail stages',rows=rows,comparisons=comparisons,expert_checks=71680,rank_layer_mb=3584,cost_policy='Only group-specific maximum FC GEMM costs; elapsed costs are never added.',causal_boundary='Inferred missing assignments include capacity/probability filtering; no measured pre-capacity perexpert histogram.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    files=reports+[BASE/'layer-filtering-r1/manifest.json',BASE/'layer-filtering-edpB-r1/manifest.json',BASE/'layer-workload-edpB-r1/manifest.json',BASE/'layer-workload-analysis-r1/data.json',Path(__file__).resolve()]
    (out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in files},outputs={'report.json':sha(out/'report.json')}),indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()

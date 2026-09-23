"""Separate depth-position patterns from iteration changes; no causal regression."""
import json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def corr(a,b):
    x=sum(a)/len(a);y=sum(b)/len(b);den=(sum((v-x)**2 for v in a)*sum((v-y)**2 for v in b))**.5
    return sum((u-x)*(v-y) for u,v in zip(a,b))/den if den else None
def main():
    source=BASE/'layer-iteration-grid-r1/report.json';d=json.loads(source.read_text());assert d['status']=='PASS'
    keyed={(r['iteration'],r['layer'],r['mb'],r['group']):r for r in d['rows']};assert len(keyed)==4032
    old=json.loads((BASE/'layer-edp-pair-r1/report.json').read_text());regression=0
    for r in old['rows']:
        for group,key in [(0,'groupA'),(1,'groupB')]:assert keyed[60,r['layer'],r['mb'],group]['retained']==r[key];regression+=1
    old_active={}
    for g,name in [(0,'layer-filtering-r1'),(1,'layer-filtering-edpB-r1')]:
        for r in json.loads((BASE/name/'report.json').read_text())['rows']:old_active[r['layer'],r['mb'],g]=r['nonzero_experts']
    for (layer,mb,g),n in old_active.items():assert keyed[60,layer,mb,g]['active_experts']==n;regression+=1
    rows=[]
    for it in d['iterations']:
        for mb in range(4):
            for layer in range(3,59):
                a,b=keyed[it,layer,mb,0],keyed[it,layer,mb,1]
                rows.append(dict(iteration=it,mb=mb,layer=layer,stage=a['stage'],retained_A=a['retained'],retained_B=b['retained'],retained_total=a['retained']+b['retained'],active_A=a['active_experts'],active_B=b['active_experts'],active_total=a['active_experts']+b['active_experts']))
    summary=[]
    for it in d['iterations']:
        for mb in range(4):
            rs=[r for r in rows if r['iteration']==it and r['mb']==mb];first,last=rs[0],rs[-1]
            summary.append(dict(iteration=it,mb=mb,first=first,last=last,endpoint_decrease_pct=100*(first['retained_total']-last['retained_total'])/first['retained_total'],depth_cost_workload_r=corr([r['layer'] for r in rs],[r['retained_total'] for r in rs]),adjacent_decreases=sum(x['retained_total']>y['retained_total'] for x,y in zip(rs,rs[1:])),adjacent_comparisons=55))
    temporal=[]
    for mb in range(4):
        for layer in range(3,59):
            rs=[r for r in rows if r['mb']==mb and r['layer']==layer]
            temporal.append(dict(mb=mb,layer=layer,first=rs[0]['retained_total'],last=rs[-1]['retained_total'],increase_steps=sum(x['retained_total']<y['retained_total'] for x,y in zip(rs,rs[1:])),steps=8,active_A_first=rs[0]['active_A'],active_A_last=rs[-1]['active_A'],active_B_first=rs[0]['active_B'],active_B_last=rs[-1]['active_B']))
    shape=[]
    for mb in range(4):
        for a,b in zip(d['iterations'],d['iterations'][1:]):
            aa=[r['retained_total'] for r in rows if r['mb']==mb and r['iteration']==a];bb=[r['retained_total'] for r in rows if r['mb']==mb and r['iteration']==b]
            shape.append(dict(mb=mb,a=a,b=b,r=corr(aa,bb)))
    report=dict(status='PASS',scope=d['scope'],iterations=d['iterations'],rows=rows,depth_summary=summary,temporal=temporal,adjacent_iteration_shape=shape,regression_checks=regression,causal_boundary='Different iterations change both weights/training state and input batch; post-filter observations cannot isolate router-parameter causality or capacity-only dropping.')
    out=BASE/'layer-iteration-analysis-r1';out.mkdir(exist_ok=False);(out/'report.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(inputs={str(source):sha(source),str(BASE/'layer-iteration-grid-r1/manifest.json'):sha(BASE/'layer-iteration-grid-r1/manifest.json'),str(Path(__file__).resolve()):sha(Path(__file__))},outputs={'report.json':sha(out/'report.json')}),indent=2)+'\n')
    print('PASS regression',regression)
    for r in summary:
        if r['mb']==0:print(r['iteration'],r['first']['retained_total'],r['last']['retained_total'],round(r['endpoint_decrease_pct'],2),'active A',r['first']['active_A'],r['last']['active_A'],'active B',r['first']['active_B'],r['last']['active_B'],'depthR',round(r['depth_cost_workload_r'],3),'down',r['adjacent_decreases'])
    print('temporal',sum(r['last']>r['first'] for r in temporal),'/',len(temporal),'increased 40->80; all8steps',sum(r['increase_steps']==8 for r in temporal))
if __name__=='__main__':main()

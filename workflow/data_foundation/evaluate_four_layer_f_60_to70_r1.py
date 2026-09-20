"""Source-only prediction sealed before target evidence extraction. Iter70 is not blind."""
import argparse,collections,hashlib,json,graphlib
from pathlib import Path
from pp32_f_split_r8 import extract
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    src=BASE/'four-layer-f-r1';g=json.loads((src/'graph.json').read_text());pred={}
    for k in graphlib.TopologicalSorter({k:n['deps'] for k,n in g.items()}).static_order():
        s=max([pred[d]['end_ms'] for d in g[k]['deps']] or [0]);pred[k]=dict(start_ms=s,end_ms=s+g[k]['cost_ms'])
    dump(out/'frozen-graph.json',g);dump(out/'sealed-prediction.json',pred)
    sealed={n:sha(out/n) for n in ['frozen-graph.json','sealed-prediction.json']};dump(out/'seal.json',sealed)
    xs=[];target=[];full={}
    for r in range(8,16):
        source=json.loads((BASE/f'pp32-f-split-r8/rank-{r}.json').read_text());folder=Path(source['path']).parent.parent/'iteration_70';paths=list(folder.glob(f'rank{r}.*.pt.trace.json'));assert len(paths)==1
        x=extract(r,paths[0],70)
        sig=lambda z:[(v['kind'],v['ordinal'],v['members'],v['input_nelems'],v['dtype']) for v in z['markers']]
        assert sig(x)==sig(source)
        dump(out/f'target-rank-{r}.json',x);xs.append(x);target.append(dict(path=str(paths[0]),sha256=x['sha256']))
        es=json.loads(paths[0].read_text())['traceEvents'];f=es[x['cpu_F_event']];calls=collections.defaultdict(list)
        for e in es:
            if e.get('cat') in ['privateuse1_driver','privateuse1_runtime'] and 'correlation' in e.get('args',{}):calls[e['args']['correlation']].append(e)
        ds=[e for e in es if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and any(v.get('pid')==f['pid'] and f['ts']<=v['ts'] and v['ts']+v.get('dur',0)<=f['ts']+f['dur']+.01 for v in calls.get(e.get('args',{}).get('correlation'),[]))]
        full[r]=(min(e['ts'] for e in ds)/1000,max(e['ts']+e['dur'] for e in ds)/1000)
        del es,calls,ds
        print('target verified rank',r,flush=True)
    origin=min(v[0] for v in full.values());rows=[];experts=[]
    for x in xs:
        r=x['rank'];off=x['origin_absolute_ms']-origin
        for l in range(4):
            pre=f'L{l}:r{r}:'
            for j,name in [(0,'A0'),(3,'A3'),(5,'expert')]:
                ds=[d for d in x['devices'] if d['semantic']==f'r{r}:LOCAL:{6*l+j}']
                if j==0 and l:
                    first=min(d['start_ms'] for d in ds if 'LayerNormGlobalKernel' in d['name']);ds=[d for d in ds if d['start_ms']>=first]
                start=min(d['start_ms'] for d in ds)+off;end=max(d['end_ms'] for d in ds)+off
                rows.append(dict(layer=l,rank=r,boundary=name+'_end',node=pre+name,trace_ms=end))
                if j==5:experts.append(dict(layer=l,rank=r,predicted_duration_ms=g[pre+name]['cost_ms'],trace_duration_ms=end-start))
            for kind,ordinal,name in [('CP',4*l+j,'CP'+str(j)) for j in range(4)]+[('EP_DISPATCH',l,'dispatch'),('EP_COMBINE',l,'unpermute')]:
                m=next(m for m in x['markers'] if m['kind']==kind and m['ordinal']==ordinal)
                rows.append(dict(layer=l,rank=r,boundary=name+'_end',node=pre+name,trace_ms=m['device_end_ms']+off))
            ds=sorted([d for d in x['devices'] if d['semantic']==f'r{r}:LOCAL:{6*l+6}'],key=lambda d:d['start_ms'])
            assert len(ds)>=2 and all('AddOp' in d['name'] for d in ds[:2])
            rows.append(dict(layer=l,rank=r,boundary='layer_tail_end',node=pre+'tail1',trace_ms=ds[1]['end_ms']+off))
    for f in rows:f.update(predicted_ms=pred[f['node']]['end_ms'],error_ms=pred[f['node']]['end_ms']-f['trace_ms'])
    assert len(rows)==320
    truth=max(v[1] for v in full.values())-origin;estimate=pred['END']['end_ms'];layer=[]
    for l in range(4):
        rs=[f for f in rows if f['layer']==l and f['boundary']=='layer_tail_end'];p=max(f['predicted_ms'] for f in rs);t=max(f['trace_ms'] for f in rs)
        ex=[v for v in experts if v['layer']==l]
        layer.append(dict(layer=l,predicted_tail_end_ms=p,trace_tail_end_ms=t,error_ms=p-t,expert_mean_predicted_ms=sum(v['predicted_duration_ms'] for v in ex)/8,expert_mean_trace_ms=sum(v['trace_duration_ms'] for v in ex)/8))
    assert all(sha(out/n)==h for n,h in sealed.items())
    report=dict(status='PASS_EVALUATION_EXECUTION',prediction_acceptance='NOT_PREDEFINED_NO_ACCURACY_PASS_CLAIM',scope='32GPU PP1 F0 four layers ranks8..15 iter60->70',predicted_F_ms=estimate,trace_F_ms=truth,error_ms=estimate-truth,absolute_relative_error_percent=abs(estimate-truth)/truth*100,boundary_MAE_ms=sum(abs(f['error_ms']) for f in rows)/len(rows),boundaries=320,layers=layer,target_costs_used=False,seal_verified=True,blind_holdout=False,limitations=['iter70 first unit ranks8/9 previously exposed; this is source-only evaluation, not blind research holdout','boundary MAE is cumulative endpoint error, not pure block duration error','effective costs and trace handoffs from iter60 frozen; no target retuning','only one stage and F0; not whole iteration MFU, B or 256GPU'])
    dump(out/'boundary-comparison.json',rows);dump(out/'expert-duration-comparison.json',experts);dump(out/'report.json',report)
    dump(out/'manifest.json',dict(source=[dict(path=str(src/n),sha256=sha(src/n)) for n in ['graph.json','manifest.json']],code=[dict(path=str(p),sha256=sha(p)) for p in [Path(__file__).resolve(),Path(__file__).with_name('pp32_f_split_r8.py')]],target=target,sealed=sealed))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

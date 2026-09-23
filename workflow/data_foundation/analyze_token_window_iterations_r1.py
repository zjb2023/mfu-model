"""Fixed source60, nine target iterations; diagnostic correlation, not blind prediction."""
import json,hashlib,statistics,math
from pathlib import Path
from pp32_to256_pipeline_r6 import target_window
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
OUT=BASE/'token-window-iterations-r1'
RAW=Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def fit(xs,ys):
    xm=statistics.mean(xs);ym=statistics.mean(ys);xx=sum((x-xm)**2 for x in xs);yy=sum((y-ym)**2 for y in ys);xy=sum((x-xm)*(y-ym) for x,y in zip(xs,ys));b=xy/xx;a=ym-b*xm
    return dict(intercept=a,slope=b,pearson=xy/math.sqrt(xx*yy),r2=xy**2/(xx*yy))
def main():
    OUT.mkdir(exist_ok=True)
    sp=BASE/'layer32-iteration-grid-r1/report.json';tp=BASE/'layer-iteration-grid-r1/report.json';bp=BASE/'detailed-fb-32to256-r1/report.json'
    s=json.loads(sp.read_text());t=json.loads(tp.read_text());baseline=json.loads(bp.read_text());pred=baseline['predicted_ms'];src=statistics.mean(r['retained'] for r in s['rows'] if r['iteration']==60 and r['stage']==1 and r['mb']==0)
    inputs={str(p):sha(p) for p in [sp,tp,bp]};rows=[]
    for it in t['iterations']:
        paths=list(RAW.glob(f'*/profiler/iteration_{it}/rank0.*.pt.trace.json'));assert len(paths)==1
        p=paths[0];cache=OUT/f'window-{it}.json'
        if cache.exists():w=json.loads(cache.read_text())
        else:
            w=target_window(p);w['iteration']=it # helper's original hardcoded label is not a trace identity
            w['iteration_identity']='iteration directory; exact rank0 path recorded'
            dump(cache,w)
        assert w['path']==str(p);inputs[str(p)]=w['sha256']
        rr=[r for r in t['rows'] if r['iteration']==it];assert len(rr)==448
        target=statistics.mean(r['retained'] for r in rr)
        row=dict(iteration=it,source_fixed_tokens=src,target_mean_tokens=target,token_gap=src-target,truth_ms=w['window_ms'],predicted_ms=pred,error_ms=pred-w['window_ms'],error_pct=100*(pred/w['window_ms']-1))
        rows.append(row);print(row,flush=True)
    assert abs(next(r for r in rows if r['iteration']==60)['truth_ms']-baseline['trace_ms'])<1e-6
    x=[r['token_gap'] for r in rows];y=[r['error_ms'] for r in rows];model=fit(x,y)
    loo=[]
    for i in range(len(rows)):
        train=[j for j in range(len(rows)) if j!=i];m=fit([x[j] for j in train],[y[j] for j in train]);e=y[i]-(m['intercept']+m['slope']*x[i]);loo.append(dict(iteration=rows[i]['iteration'],remaining_error_ms=e))
    time=fit([r['iteration'] for r in rows],y)
    diff=fit([x[i+1]-x[i] for i in range(8)],[y[i+1]-y[i] for i in range(8)])
    early=fit(x[:5],y[:5]);late=[y[i]-(early['intercept']+early['slope']*x[i]) for i in range(5,9)]
    report=dict(status='PASS_DIAGNOSTIC_ONLY',rows=rows,linear_gap_to_error=model,iteration_only_control=time,first_difference_correlation=diff,leave_one_out=loo,loo_mae_ms=statistics.mean(abs(r['remaining_error_ms']) for r in loo),early40to60_fit=early,later65to80_residual_ms=late,later_mae_ms=statistics.mean(map(abs,late)),baseline_mae_ms=statistics.mean(map(abs,y)),scope='F retained assignments averaged across 56 layers x 4 MB x 2 EP8 groups; rank0 first F GPU to last B GPU; fixed source60 prediction',limitations=['target token counts are evaluator data, unavailable for source-only prediction','9 serially correlated observations; not independent causal evidence','mean F counts do not directly measure B cost, group/rank stragglers or critical-path weighting','iteration also changes input batches and training state; rank0 window is not full-world makespan','no source parameters changed; no training or full simulation'])
    dump(OUT/'report.json',report)
    code=[Path(__file__).resolve(),Path(__file__).with_name('pp32_to256_pipeline_r6.py')];inputs.update({str(p):sha(p) for p in code})
    dump(OUT/'manifest.json',dict(inputs=inputs,outputs={p.name:sha(p) for p in OUT.glob('*.json') if p.name!='manifest.json'}))
    print('SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ['rows','leave_one_out','limitations']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

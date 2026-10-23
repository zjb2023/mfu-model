"""Assemble calibrated detailed F/B, existing PP schedule and explicit tail.

Primary metric retains the original rank0 reference-window convention. Both
EP8 cohorts are audited, but are NOT joined at every F/B by a fake barrier.
"""
import collections,gzip,json,graphlib
from pathlib import Path
from calibrate256_fine_r1 import BASE,load,save,sha
from build_b_cp_pair_r4 import run
VARIANTS=[('calibrate256-fine-r1','shared-binding-r2'),('calibrate256-zero-r2','shared-binding-r4'),('calibrate256-strict-r2','shared-binding-r3'),('calibrate256-stress-r2','shared-binding-r3'),('calibrate256-combined-r3','shared-binding-r3'),('calibrate256-overlap-r3','shared-binding-r3')]
def select():
    selected={};missing=[]
    for r in range(16,240,8):
        for mb in range(4):
            for name,tag in VARIANTS+[('calibrate256-combined-r3','shared-binding-r4')]:
                p=BASE/name/f'r{r}-mb{mb}'/tag/'report.json'
                if p.exists() and all(v['status'].startswith('PASS') for v in load(p)['phases'].values()):selected[r,mb]=p;break
            else:missing.append((r,mb))
    return selected,missing
def main():
    selected,missing=select();assert not missing,('Unbound units: no scalar fallback allowed',missing)
    out=BASE/'iter60-fine-model-r1';out.mkdir(exist_ok=False);inputs={};bindings=[]
    outer=BASE/'calibrate256-iter60-outer-r1/calibrated-graph.json';full=load(outer);inputs[str(outer)]=sha(outer)
    source=load(BASE/'deepep-stable-20261016-r1/model/outer-prediction.json')['nodes']
    g={k:dict(cost_ms=full[k]['duration_ms'],deps=full[k]['dependencies'],kind=full[k]['kind']) for k in source}
    checks=[];roundoff=[]
    for stage in range(1,15):
        for mb in range(4):
            r=16*stage;report=selected[r,mb];inputs[str(report)]=sha(report)
            for phase in ['F','B']:
                p=report.parent/(phase+'.json');sub=load(p);inputs[str(p)]=sha(p);key=f's{stage}:{phase}{mb}';pred=list(g[key]['deps']);prefix=key+'::'
                for k,n in sub.items():
                    cost=n['cost_ms'];assert cost>=-1e-7,(key,k,cost)
                    if cost<0:roundoff.append(dict(node=prefix+k,value_ms=cost));cost=0
                    g[prefix+k]=dict(deps=[prefix+d for d in n['deps']] if n['deps'] else pred,cost_ms=cost,kind=n.get('kind','UNSPECIFIED'),template_rank=n.get('rank'),physical_rank=r+n['rank']-8 if n.get('rank') is not None else None)
                g[key].update(deps=[prefix+'END'],cost_ms=0,kind=phase+'_zero_wrapper')
                expected=load(report)['phases'][phase]['replay_ms'];checks.append((key,prefix+'END',pred,expected));bindings.append(dict(stage=stage,mb=mb,phase=phase,rank_base=r,graph=str(p),sha256=sha(p),expected_ms=expected))
    tp=BASE/'iter60-tail-stream-r1/graph.json';tail=load(tp);inputs[str(tp)]=sha(tp)
    for k,n in tail.items():g['tail::'+k]=dict(cost_ms=n['cost_ms'],deps=['tail::'+d for d in n['deps']] if n['deps'] else ['s0:B3'],kind=n.get('kind'),event=n.get('event'))
    t=run(g)
    for key,end,pred,expected in checks:assert abs(t[end]['end_ms']-max(t[d]['end_ms'] for d in pred)-expected)<1e-5,key
    origin=t['s0:F0']['start_ms'];one=t['s0:B3']['end_ms']-origin;body=t['tail::END']['end_ms']-origin
    refp=BASE/'calibrate256-iter60-outer-r1/report.json';ref=load(refp);inputs[str(refp)]=sha(refp)
    param=BASE/'calibrate256-iter60-outer-r1/parameters.json';physics=load(param)['physics'];inputs[str(param)]=sha(param)
    factor=100*physics['model_flops_per_iteration']/(256*physics['peak_flops_per_gpu_s'])*1000
    coverage=[]
    for (r,mb),p in selected.items():inputs[str(p)]=sha(p);coverage.append(dict(rank_base=r,stage=r//16,group=(r%16)//8,mb=mb,report=str(p),phases=load(p)['phases']))
    report=dict(status='PASS_CONDITIONAL_SAME_ITERATION_FINE_MODEL_REPLAY',iteration=60,onef1b=dict(replay_ms=one,truth_ms=ref['onef1b']['truth_ms'],signed_error_percent=100*(one/ref['onef1b']['truth_ms']-1)),body=dict(replay_ms=body,truth_ms=ref['body']['truth_ms'],signed_error_percent=100*(body/ref['body']['truth_ms']-1),replay_mfu_percent=factor/body,truth_mfu_percent=factor/ref['body']['truth_ms'],mfu_signed_error_percent=100*(ref['body']['truth_ms']/body-1)),tail_ms=body-one,nodes=len(g),edges=sum(len(n['deps']) for n in g.values()),detailed_FB_bindings=len(bindings),validated_EP8_MB_units=len(coverage),primary_cohort='first EP8 at each middle stage; original rank0 window convention',roundoff_only_nonnegative_normalization=roundoff,physics=physics,limits=['Same-iteration target calibration with observed reserved readiness; NOT independent prediction.', 'New shared-DAG candidate removes15 conditional V-pack edges; frozen32→256 release untouched.', 'Second EP8 cohort audited independently, not inserted as a false per-FB global barrier.', 'First/last stages remain target whole-FB interfaces, as in original model.', 'PP costs are calibrated effective F/B medians, not pure service; cross-host clocks assumed comparable.', 'Tail is same-stream conditional parameter graph; cross-stream/all-rank causal closure remains unproven.', 'MFU inherits the existing effective FLOPs and peak convention.'])
    with gzip.open(out/'graph.json.gz','wt',encoding='utf-8') as f:json.dump(g,f,separators=(',',':'))
    display={k:dict(t[k]) for k in source}
    for key,end,pred,expected in checks:display[key]['start_ms']=max(t[d]['end_ms'] for d in pred)
    save(out/'report.json',report);save(out/'bindings.json',bindings);save(out/'coverage.json',coverage);save(out/'outer-timeline.json',display);save(out/'parameters.json',dict(physics=physics,PP=ref['pp_effective_ms'],scope=report['primary_cohort']))
    inputs[str(Path(__file__))]=sha(Path(__file__))
    save(out/'manifest.json',dict(inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()

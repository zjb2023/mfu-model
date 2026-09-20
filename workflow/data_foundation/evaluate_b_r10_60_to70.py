"""Read-only trace evaluation of sealed source60 B0; no target parameter fitting."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from build_b_cp_pair_r4 import run

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')
def inside(e, v):
    return (e.get('pid') == v['pid'] and v['ts'] <= e.get('ts', 0)
            and e.get('ts', 0)+e.get('dur', 0) <= v['ts']+v.get('dur', 0)+.01)

def extract(path, rank):
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    es = json.loads(raw)['traceEvents']; del raw
    b = next(e for e in es if e.get('cat') == 'user_annotation' and e.get('name') == 'backward_step')
    ck = sorted([e for e in es if e.get('cat') == 'cpu_op' and e.get('name') == 'CheckpointFunctionBackward' and inside(e,b)], key=lambda e:e['ts'])
    assert len(ck) == 4
    marks=[]
    for c in ck:
        ops=[e for e in es if e.get('cat')=='cpu_op' and inside(e,c)]
        combine=next(e for e in ops if e['name']=='FusedCombine')
        names={'FusedCombineBackward':'dispatch','FusedDispatchBackward':'combine'}
        selected=[(names[e['name']],e) for e in ops if e['name'] in names]
        linear=sorted([e for e in ops if e['name']=='_GroupedLinearBackward'],key=lambda e:e['ts'])
        assert len(linear)==2
        selected += [('FC2',linear[0]),('FC1',linear[1])]
        flashes=[e for e in ops if e['name']=='aten::_scaled_dot_product_attention_flash_musa_backward']
        assert len(flashes)==1
        selected += [('attention',flashes[0])]
        marks.append((combine['ts']+combine['dur'],selected))
    runtime=collections.defaultdict(list)
    for e in es:
        if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}):
            runtime[e['args']['correlation']].append(e)
    ds=[]
    for i,e in enumerate(es):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']: continue
        calls=[v for v in runtime.get(e.get('args',{}).get('correlation'),[]) if inside(v,b)]
        if not calls: continue
        call=min(calls,key=lambda v:v.get('dur',0))
        u=next((u for u,c in enumerate(ck) if inside(call,c)),None)
        phase='outside'; labels=[]
        if u is not None:
            end,ms=marks[u]; phase='recompute' if call['ts']<=end else 'backward'
            labels=[name for name,op in ms if inside(call,op)]
            if phase=='backward' and e.get('args',{}).get('Process Group Description')=='CONTEXT_PARALLEL_GROUP':
                labels.append('CP')
        ds.append(dict(event=i,unit=u,phase=phase,start_ms=e['ts']/1000,end_ms=(e['ts']+e['dur'])/1000,labels=labels))
    phases=[]; components=[]
    for u in range(4):
        for phase in ['recompute','backward']:
            members=[d for d in ds if d['unit']==u and d['phase']==phase]; assert members
            phases.append(dict(unit=u,phase=phase,start_ms=min(d['start_ms'] for d in members),end_ms=max(d['end_ms'] for d in members),events=len(members)))
        cp=sorted([d for d in ds if d['unit']==u and 'CP' in d['labels']],key=lambda d:d['start_ms'])
        assert len(cp)==5
        for j,d in enumerate(cp):d['labels'].append(f'CP{j}')
        for label in ['dispatch','FC2','FC1','combine','attention']+[f'CP{j}' for j in range(5)]:
            members=[d for d in ds if d['unit']==u and label in d['labels']]
            if members:
                components.append(dict(unit=u,component=label,start_ms=min(d['start_ms'] for d in members),end_ms=max(d['end_ms'] for d in members),events=len(members)))
    steps=[e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')]
    assert len(steps)==1
    # ProfilerStep is a CPU rank-local measurement, not global synchronized makespan.
    return dict(rank=rank,path=str(path),sha256=digest,start_ms=min(d['start_ms'] for d in ds),end_ms=max(d['end_ms'] for d in ds),phases=phases,components=components,profiler_step_ms=steps[0]['dur']/1000,profiler_step_name=steps[0]['name'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); out=a.out
    out.mkdir(parents=True,exist_ok=False)
    source=BASE/'b-four-layers-r10'; graph=json.loads((source/'graph.json').read_text()); pred=run(graph)
    assert pred==json.loads((source/'replay.json').read_text())
    dump(out/'frozen-graph.json',graph); dump(out/'sealed-prediction.json',pred)
    sealed={n:sha(out/n) for n in ['frozen-graph.json','sealed-prediction.json']}
    dump(out/'seal.json',dict(files=sealed,target_read=False,source_manifest_sha256=sha(source/'manifest.json')))
    inputs=json.loads((BASE/'b-cost-blocks-r1/evidence.json').read_text()); facts={}
    for it in [60,70]:
        facts[it]=[]
        for x in inputs:
            p=Path(x['path'])
            if it==70:
                paths=list((p.parent.parent/'iteration_70').glob(f'rank{x["rank"]}.*.pt.trace.json')); assert len(paths)==1; p=paths[0]
            f=extract(p,x['rank'])
            if it==60: assert f['sha256']==x['sha256']
            facts[it].append(f); print('extracted',it,x['rank'],flush=True)
        origin=min(f['start_ms'] for f in facts[it])
        for f in facts[it]:
            f['origin_absolute_ms']=origin
            for z in [f]+f['phases']+f['components']:
                z['start_ms']-=origin; z['end_ms']-=origin; z['duration_ms']=z['end_ms']-z['start_ms']
        dump(out/f'facts-{it}.json',facts[it])
    rows=[]; components=[]; ranks=[]
    for s,t in zip(facts[60],facts[70]):
        r=s['rank']; assert r==t['rank']
        for a,b in zip(s['phases'],t['phases']):
            u=a['unit']; phase=a['phase']; assert (u,phase)==(b['unit'],b['phase'])
            key=f'r{r}:u{u}:{phase}'; begin=pred[key]['start_ms'] if phase=='recompute' else pred[key+':handoff']['end_ms']
            finish=pred[key]['end_ms']
            assert abs(finish-a['end_ms'])<1e-5 and abs(begin-a['start_ms'])<1e-5
            rows.append(dict(rank=r,unit=u,phase=phase,predicted_start_ms=begin,predicted_end_ms=finish,predicted_duration_ms=finish-begin,trace_start_ms=b['start_ms'],trace_end_ms=b['end_ms'],trace_duration_ms=b['duration_ms'],duration_error_ms=finish-begin-b['duration_ms'],end_error_ms=finish-b['end_ms']))
        for c in s['components']:
            match=[v for v in t['components'] if (v['unit'],v['component'])==(c['unit'],c['component'])]; assert len(match)==1
            v=match[0]; components.append(dict(rank=r,unit=c['unit'],component=c['component'],source_ms=c['duration_ms'],target_ms=v['duration_ms'],change_ms=v['duration_ms']-c['duration_ms']))
        ranks.append(dict(rank=r,B60_ms=s['duration_ms'],B70_ms=t['duration_ms'],B_change_ms=t['duration_ms']-s['duration_ms'],step60_ms=s['profiler_step_ms'],step70_ms=t['profiler_step_ms'],step_change_ms=t['profiler_step_ms']-s['profiler_step_ms'],step_change_percent=100*(t['profiler_step_ms']/s['profiler_step_ms']-1)))
    source_ms=max(f['end_ms'] for f in facts[60]); target_ms=max(f['end_ms'] for f in facts[70]); estimate=pred['END']['end_ms']; assert abs(source_ms-estimate)<1e-5
    layers=[]
    for u in range(4):
        for phase in ['recompute','backward']:
            rs=[r for r in rows if r['unit']==u and r['phase']==phase]
            layers.append(dict(unit=u,phase=phase,source_mean_ms=sum(r['predicted_duration_ms'] for r in rs)/8,target_mean_ms=sum(r['trace_duration_ms'] for r in rs)/8,duration_MAE_ms=sum(abs(r['duration_error_ms']) for r in rs)/8,end_MAE_ms=sum(abs(r['end_error_ms']) for r in rs)/8))
    assert all(sha(out/n)==h for n,h in sealed.items())
    report=dict(status='PASS_EVALUATION_EXECUTION',accuracy_acceptance='NO_PREDEFINED_THRESHOLD',scope='32GPU PP1 B0 ranks8..15, four layers including recompute',source_B_ms=source_ms,target_B_ms=target_ms,measured_change_ms=target_ms-source_ms,measured_change_percent=100*(target_ms/source_ms-1),predicted_B_ms=estimate,error_ms=estimate-target_ms,ARE_percent=100*abs(estimate-target_ms)/target_ms,phase_duration_MAE_ms=sum(abs(r['duration_error_ms']) for r in rows)/64,phase_end_MAE_ms=sum(abs(r['end_error_ms']) for r in rows)/64,layers=layers,ranks=ranks,seal_verified=True,target_fitted=False,blind=False,limitations=['source-conditioned prediction retains source readiness and recompute envelopes','phase comparisons use CPU ownership and device envelopes; component intervals may overlap and must not be summed','ProfilerStep figures are per-rank CPU windows for PP1 only, not global Training Step','no all-microbatch B or full-world 1F1B prediction evaluation in this run','iteration70 previously exposed; not blind validation'])
    for name,v in [('report',report),('phase-comparison',rows),('component-variation',components)]: dump(out/(name+'.json'),v)
    dump(out/'manifest.json',dict(source=[dict(path=str(source/n),sha256=sha(source/n)) for n in ['graph.json','replay.json','manifest.json']],traces=[dict(path=f['path'],sha256=f['sha256']) for it in facts for f in facts[it]],code=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),sealed=sealed,outputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in out.glob('*.json')]))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__': main()

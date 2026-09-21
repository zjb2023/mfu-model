"""Evaluator-only oracle ablations on frozen contracted DAG; never fit costs."""
import argparse, copy, graphlib, hashlib, itertools, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
CACHE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/pp256-stage-audit-r7')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    paths=[BASE/'detailed-fb-32to256-r1/outer-prediction.json',BASE/'detailed-fb-32to256-r1/report.json']
    seals={str(p):sha(p) for p in paths};outer,old=map(load,paths);nodes=outer['nodes']
    order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
    def run(overrides):
        t={}
        for k in order:
            n=nodes[k];start=max((t[d]['end_ms'] for d in n['dependencies']),default=0)
            t[k]=dict(start_ms=start,end_ms=start+overrides.get(k,n['duration_ms']))
        return t,t['s0:B3']['end_ms']-t['s0:F0']['start_ms']
    baseline,window=run({});assert abs(window-old['predicted_ms'])<1e-6
    assert all(abs(baseline[k]['end_ms']-n['end_ms'])<1e-6 for k,n in nodes.items())
    groups={name:{} for name in ['middle_F','middle_B','first','last']};rows=[];gaps=[];raw_refs=[]
    for st in range(16):
        p=CACHE/f'rank-{st*16}.json';paths.append(p);c=load(p);assert c['stage']==st and c['rank']==st*16
        blocks=c['blocks'];assert len(blocks)==8 and {(b['phase'],b['mb']) for b in blocks}=={(ph,m) for ph in ['F','B'] for m in range(4)}
        raw_refs.append(dict(path=c['path'],sha256=c['sha256'],verification='cached provenance; raw trace not reread in this run'))
        for b in blocks:
            k=f's{st}:{b["phase"]}{b["mb"]}';assert abs(b['duration_ms']-(b['end_absolute_ms']-b['start_absolute_ms']))<1e-6
            group='first' if st==0 else 'last' if st==15 else 'middle_'+b['phase'];groups[group][k]=b['duration_ms']
            rows.append(dict(stage=st,rank=st*16,phase=b['phase'],mb=b['mb'],node=k,predicted_duration_ms=nodes[k]['duration_ms'],observed_duration_ms=b['duration_ms'],error_ms=nodes[k]['duration_ms']-b['duration_ms'],cpu_event=b['cpu_event']))
        for x,y in zip(blocks,blocks[1:]):
            xk=f's{st}:{x["phase"]}{x["mb"]}';yk=f's{st}:{y["phase"]}{y["mb"]}'
            gaps.append(dict(stage=st,from_node=xk,to_node=yk,observed_gap_ms=y['start_absolute_ms']-x['end_absolute_ms'],predicted_gap_ms=baseline[yk]['start_ms']-baseline[xk]['end_ms'],basis='local GPU envelope gaps; includes other-stage work/PP/wait; not additive global error'))
    stats=[]
    for st in range(16):
        for ph in ['F','B']:
            rs=[r for r in rows if r['stage']==st and r['phase']==ph]
            stats.append(dict(stage=st,rank=st*16,phase=ph,predicted_ms=rs[0]['predicted_duration_ms'],observed_mean_ms=statistics.mean(r['observed_duration_ms'] for r in rs),error_mean_ms=statistics.mean(r['error_ms'] for r in rs),samples=len(rs)))
    truth=old['trace_ms'];names=list(groups);scenarios={};times={}
    for bits in itertools.product([0,1],repeat=4):
        selected=tuple(n for n,b in zip(names,bits) if b);overrides={k:v for name in selected for k,v in groups[name].items()};t,w=run(overrides);times[selected]=w
        scenarios['+'.join(selected) or 'baseline']=dict(replaced_groups=list(selected),window_ms=w,error_ms=w-truth,error_percent=100*(w/truth-1),reduction_from_baseline_ms=window-w)
    contributions={n:0 for n in names}
    for permutation in itertools.permutations(names):
        chosen=set();prev=window
        for n in permutation:
            chosen.add(n);key=tuple(x for x in names if x in chosen);cur=times[key];contributions[n]+=(prev-cur)/24;prev=cur
    residual=times[tuple(names)]-truth
    assert abs(sum(contributions.values())+residual-(window-truth))<1e-6
    assert all(sha(Path(p))==h for p,h in seals.items())
    report=dict(status='PASS_DIAGNOSTIC_NOT_PREDICTION',baseline_ms=window,truth_ms=truth,baseline_error_ms=window-truth,stage_stats=stats,shapley_reduction_ms=contributions,residual_after_all_fb_ms=residual,scenarios=scenarios,
                attribution_method='four groups, 24 replacement orders averaged; oracle per-block target durations; positive means replacing reduces predicted window',
                scope='16 representative ranks, one per PP stage, 128 blocks, iteration60; not all256 ranks',
                constraints=['target durations are evaluator-only oracle substitutions; not promoted or calibrated parameters','residual includes dependency/PP/envelope/clock/cohort approximation; NOT pure PP cost','local gaps are not independent costs and cannot be summed across stages','duration-only comparisons use rank-local differences; no cross-host offset fitted','current detailed F/B contracts to frozen outer DAG exactly; no internal per-kernel attribution'],
                raw_reparsed=False,baseline_unchanged=True,target_fitted=False)
    dump(a.out/'report.json',report);dump(a.out/'blocks.json',rows);dump(a.out/'local-gaps.json',gaps)
    dump(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in paths},raw_provenance=raw_refs,code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
    print(json.dumps({k:report[k] for k in ['baseline_error_ms','shapley_reduction_ms','residual_after_all_fb_ms','scenarios']},indent=2))
if __name__=='__main__':main()

"""R10 phase-envelope transfer diagnostic, target costs evaluator-only."""
import argparse,collections,graphlib,hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
SOURCE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/b-cost-blocks-r1/evidence.json')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parts(ds,total):
 out={};units=[]
 for ph in ['recompute','backward']:
  durations=[]
  for u in range(4):
   es=[e for e in ds if e['unit']==u and e['phase']==ph];assert es
   duration=max(e['end_ms'] for e in es)-min(e['start_ms'] for e in es);durations.append(duration)
   units.append(dict(unit=u,phase=ph,duration_ms=duration))
  out[ph]=sum(durations)
 out['handoff_tail_group_offset']=total-sum(out.values());assert out['handoff_tail_group_offset']>=-1e-5
 return out,units
def extract(ref):
 raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];es=json.loads(raw)['traceEvents'];del raw
 def inside(e,b):return e.get('pid')==b['pid'] and b['ts']<=e.get('ts',0) and e['ts']+e.get('dur',0)<=b['ts']+b['dur']+.01
 bs=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='backward_step'],key=lambda e:e['ts']);assert len(bs)==4
 runtime=collections.defaultdict(list)
 for i,e in enumerate(es):
  if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append((i,e))
 owned=collections.defaultdict(list)
 for i,e in enumerate(es):
  if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
  calls=runtime.get(e.get('args',{}).get('correlation'),[])
  for mb,b in enumerate(bs):
   hits=[(j,c) for j,c in calls if inside(c,b)]
   if hits:owned[mb].append((i,e,min(hits,key=lambda x:x[1].get('dur',0))[1]))
 result=[]
 for mb,b in enumerate(bs):
  cks=sorted([e for e in es if e.get('cat')=='cpu_op' and e.get('name')=='CheckpointFunctionBackward' and inside(e,b)],key=lambda e:e['ts']);assert len(cks)==4
  cutoffs=[]
  for ck in cks:
   combine=[e for e in es if e.get('cat')=='cpu_op' and e.get('name')=='FusedCombine' and inside(e,ck)];assert len(combine)==1
   cutoffs.append(combine[0]['ts']+combine[0]['dur'])
  ds=[]
  for i,e,c in owned[mb]:
   u=next((j for j,ck in enumerate(cks) if inside(c,ck)),None)
   phase='outside' if u is None else ('recompute' if c['ts']<=cutoffs[u] else 'backward')
   ds.append(dict(event=i,unit=u,phase=phase,start_ms=e['ts']/1000,end_ms=(e['ts']+e['dur'])/1000))
  total=max(e['end_ms'] for e in ds)-min(e['start_ms'] for e in ds);p,units=parts(ds,total)
  result.append(dict(stage=ref['stage'],mb=mb,B_ms=total,components=p,units=units))
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 modelp=BASE/'b-four-layers-r10/summary.json';manifestp=BASE/'b-four-layers-r10/manifest.json'
 expected=next(x['sha256'] for x in load(manifestp)['inputs'] if x['path']==str(SOURCE));assert sha(SOURCE)==expected
 source=next(x for x in load(SOURCE) if x['rank']==8);total=load(modelp)['B_ms'];src,units=parts(source['devices'],total)
 # Source rank8 closes the EP8 group in this sealed sample; residual includes
 # the source group-to-rank8 entry offset, not a pure wait parameter.
 allsource=load(SOURCE);assert abs(max(e['end_ms'] for e in source['devices'])-max(e['end_ms'] for x in allsource for e in x['devices']))<1e-5
 refs_p=BASE/'b-position-curves-r2/raw-provenance.json';refs=[r for r in load(refs_p) if r['world']==256 and r['iteration']==60];rows=[]
 for ref in sorted(refs,key=lambda r:r['stage']):
  rows+=extract(ref);print('PP',ref['stage'],'PASS',flush=True)
 outerp=BASE/'detailed-fb-32to256-r1/outer-prediction.json';diagp=BASE/'1f1b-overestimate-r1/report.json';blockp=BASE/'1f1b-overestimate-r1/blocks.json'
 nodes=load(outerp)['nodes'];diag=load(diagp);obs={b['node']:b['observed_duration_ms'] for b in load(blockp)}
 for r in rows:assert abs(r['B_ms']-obs[f's{r["stage"]}:B{r["mb"]}'])<1e-4
 order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
 def run(selected):
  overrides={f's{r["stage"]}:B{r["mb"]}':total+sum(r['components'][g]-src[g] for g in selected) for r in rows};t={}
  assert min(overrides.values())>=0
  for k in order:
   n=nodes[k];s=max((t[d][1] for d in n['dependencies']),default=0);t[k]=(s,s+overrides.get(k,n['duration_ms']))
  return t['s0:B3'][1]-t['s0:F0'][0]
 groups=list(src);scenarios={}
 for bits in itertools.product([0,1],repeat=3):
  chosen=tuple(g for g,b in zip(groups,bits) if b);scenarios[chosen]=run(chosen)
 base=scenarios[()];allvalue=scenarios[tuple(groups)];assert abs(base-diag['baseline_ms'])<1e-6;assert abs(allvalue-diag['scenarios']['middle_B']['window_ms'])<1e-4
 shapley={g:0 for g in groups}
 for perm in itertools.permutations(groups):
  chosen=set();prev=base
  for g in perm:
   chosen.add(g);cur=scenarios[tuple(x for x in groups if x in chosen)];shapley[g]+=(prev-cur)/6;prev=cur
 assert abs(sum(shapley.values())-(base-allvalue))<1e-6
 report=dict(status='ORACLE_PHASE_DIAGNOSTIC_NOT_PREDICTION',source_components_ms=src,source_units=units,target_rows=rows,baseline_ms=base,middle_B_oracle_ms=allvalue,
  scenarios=[dict(replaced=list(k),window_ms=v,reduction_ms=base-v) for k,v in scenarios.items()],shapley_reduction_ms=shapley,
  limits=['Source rank8 selected because it closes source EP8 r10 END; target lane0 only, not group-complete target','Source residual includes EP8 entry offset; not purely phase handoff or tail','Phase envelopes include communication, compute and waits; not pure FLOP or bandwidth attribution','R10 recompute envelope and backward phase boundaries are replay bookkeeping, attribution does not replace inner causal graph','Target costs evaluator-only, no promotion; iter60 only'])
 (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 inputs=[SOURCE,modelp,manifestp,refs_p,outerp,diagp,blockp]
 (a.out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in inputs},raw=refs,code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str((a.out/'report.json').resolve()):sha(a.out/'report.json')}),indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['target_rows','source_units','limits']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

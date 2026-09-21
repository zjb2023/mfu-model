"""Source/target expert call work and envelope oracle; target-only evaluation."""
import argparse,bisect,collections,graphlib,hashlib,itertools,json
from pathlib import Path
from audit_b_operator_position_r1 import union
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
SRC=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/b-cost-blocks-r1/evidence.json')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(ref,world,stage,rank,all_mb=True):
 raw=Path(ref['path']).read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if ref.get('sha256'):assert digest==ref['sha256']
 es=json.loads(raw)['traceEvents'];del raw
 bs=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='backward_step'],key=lambda e:e['ts'])
 if not all_mb:bs=bs[:1]
 ops=[];threads=collections.defaultdict(list)
 for mb,b in enumerate(bs):
  selected=sorted([(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e.get('name') in ['_GroupedLinear','_GroupedLinearBackward'] and e['pid']==b['pid'] and b['ts']<=e['ts'] and e['ts']+e['dur']<=b['ts']+b['dur']+.01],key=lambda z:z[1]['ts']);assert len(selected)==16
  counters=collections.Counter()
  for i,e in selected:
   ph='recompute' if e['name']=='_GroupedLinear' else 'backward';j=counters[ph];counters[ph]+=1;fc=(j%2+1) if ph=='recompute' else (2-j%2)
   row=dict(event=i,mb=mb,phase=ph,unit=j//2,FC=fc,cpu_ms=e['dur']/1000,args=e.get('args',{}),devices=[])
   ops.append(row);threads[e['pid'],e['tid']].append((e['ts'],e,len(ops)-1))
 starts={}
 for k,v in threads.items():v.sort();starts[k]=[x[0] for x in v]
 runtime=collections.defaultdict(list)
 for i,e in enumerate(es):
  if e.get('cat')=='privateuse1_runtime' and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append(e)
 for i,e in enumerate(es):
  if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
  cs=runtime.get(e.get('args',{}).get('correlation'),[])
  if len(cs)!=1:continue
  c=cs[0];k=c['pid'],c['tid']
  if k not in threads:continue
  j=bisect.bisect_right(starts[k],c['ts'])-1
  if j<0:continue
  _,owner,index=threads[k][j]
  if c['ts']+c.get('dur',0)>owner['ts']+owner['dur']+.01:continue
  ops[index]['devices'].append(dict(event=i,name=e['name'],start_us=e['ts'],end_us=e['ts']+e['dur']))
 blocks=[]
 for mb,b in enumerate(bs):
  selected=[o for o in ops if o['mb']==mb]
  for op in selected:
   ds=op['devices'];assert ds;lo=min(d['start_us'] for d in ds);hi=max(d['end_us'] for d in ds);active=union([(d['start_us'],d['end_us']) for d in ds]);gemm=[d for d in ds if 'gemm' in d['name'].lower()]
   op.update(start_us=lo,end_us=hi,envelope_ms=(hi-lo)/1000,gpu_union_ms=active,internal_no_owned_ms=(hi-lo)/1000-active,gemm_union_ms=union([(d['start_us'],d['end_us']) for d in gemm]),gemm_count=len(gemm))
  span_sum=sum(o['envelope_ms'] for o in selected);span_union=union([(o['start_us'],o['end_us']) for o in selected]);assert abs(span_sum-span_union)<1e-5,'expert call envelopes overlap'
  blocks.append(dict(mb=mb,operators=selected,components={ph:sum(o['envelope_ms'] for o in selected if o['phase']==ph) for ph in ['recompute','backward']}))
 return dict(world=world,stage=stage,rank=rank,path=ref['path'],sha256=digest,blocks=blocks)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 sources=load(SRC);refs_p=BASE/'b-position-curves-r2/raw-provenance.json';refs=[r for r in load(refs_p) if r['world']==256 and r['iteration']==60];data=[]
 for x in sources:
  d=extract(x,32,1,x['rank'],False);data.append(d);print('source',x['rank'],'PASS',flush=True)
 for ref in sorted(refs,key=lambda r:r['stage']):
  d=extract(ref,256,ref['stage'],ref['stage']*16);data.append(d);print('target stage',ref['stage'],'PASS',flush=True)
 # Same EP8 cohort check at PP1: do not infer world-wide load from lane0 alone.
 anchor=next(r for r in refs if r['stage']==1)
 for rank in range(17,24):
  paths=list(Path(anchor['path']).parent.glob(f'rank{rank}.*.pt.trace.json'));assert len(paths)==1
  d=extract(dict(path=str(paths[0])),256,1,rank,False);data.append(d);print('target cohort',rank,'PASS',flush=True)
 for d in data:(a.out/f'w{d["world"]}-r{d["rank"]}.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 outerp=BASE/'prediction-r11-recompute-r1-verified/outer-prediction.json';blockp=BASE/'1f1b-overestimate-r1/blocks.json';oldreport=BASE/'1f1b-overestimate-r1/report.json'
 nodes=load(outerp)['nodes'];original=load(oldreport);cost=nodes['s1:B0']['duration_ms'];source=next(d for d in data if d['world']==32 and d['rank']==8)['blocks'][0]['components'];source=dict(source,rest=cost-sum(source.values()))
 observed={b['node']:b['observed_duration_ms'] for b in load(blockp)};rows=[]
 for d in data:
  if d['world']!=256 or d['rank']%16:continue
  for b in d['blocks']:
   k=f's{d["stage"]}:B{b["mb"]}';comp=dict(b['components'],rest=observed[k]-sum(b['components'].values()));assert comp['rest']>0
   rows.append(dict(stage=d['stage'],mb=b['mb'],node=k,components=comp))
 order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order());groups=list(source)
 def run(chosen):
  overrides={r['node']:cost+sum(r['components'][g]-source[g] for g in chosen) for r in rows};t={}
  for k in order:
   n=nodes[k];s=max((t[d][1] for d in n['dependencies']),default=0);t[k]=(s,s+overrides.get(k,n['duration_ms']))
  return t['s0:B3'][1]-t['s0:F0'][0]
 scenarios={}
 for bits in itertools.product([0,1],repeat=3):
  chosen=tuple(g for g,b in zip(groups,bits) if b);scenarios[chosen]=run(chosen)
 baseline=scenarios[()];oracle=scenarios[tuple(groups)];assert abs(oracle-original['scenarios']['middle_B']['window_ms'])<1e-5
 contribution={k:0 for k in groups}
 for perm in itertools.permutations(groups):
  chosen=set();prev=baseline
  for g in perm:
   chosen.add(g);v=scenarios[tuple(k for k in groups if k in chosen)];contribution[g]+=(prev-v)/6;prev=v
 cohorts=[]
 for w in [32,256]:
  ds=[d for d in data if d['world']==w and d['stage']==1]
  for u in range(4):
   os=[next(o for o in d['blocks'][0]['operators'] if o['phase']=='backward' and o['FC']==2 and o['unit']==u) for d in ds];assert len(os)==8
   shapes=[o['args'].get('Input Dims') for o in os]
   cohorts.append(dict(world=w,unit=u,ranks=[d['rank'] for d in ds],FC2_input_shapes=shapes,token_rows=[s[0][0] for s in shapes],sum_token_rows=sum(s[0][0] for s in shapes)))
 report=dict(status='EXPERT_ENVELOPE_ORACLE_NOT_PREDICTION',source_components_ms=source,target_rows=rows,baseline_ms=baseline,middle_B_oracle_ms=oracle,
   shapley_reduction_ms=contribution,scenarios=[dict(replaced=list(k),window_ms=v,reduction_ms=baseline-v) for k,v in scenarios.items()],cohorts=cohorts,
   limitations=['FC1/FC2 semantics from paired call order, not weight ID proof','Oracle group costs are call GPU envelopes incl internal gaps; not pure FLOPs attribution','No group-complete target B end; lane0 oracle retained; cohort tokens only PP1 B0','Expert envelopes disjoint on each sampled rank but rest includes communication overlap and waits','Target observations evaluator-only; no source cost calibration or new prediction claimed'])
 (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 inputs=[SRC,refs_p,outerp,blockp,oldreport];codes=[Path(__file__).resolve(),Path(__file__).with_name('audit_b_operator_position_r1.py')]
 (a.out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in inputs},raw=[{k:d[k] for k in ['path','sha256']} for d in data],code={str(p):sha(p) for p in codes},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}),indent=2)+'\n')
 print(json.dumps(dict(contribution=contribution,source=source,cohort_sums=[(c['world'],c['unit'],c['sum_token_rows']) for c in cohorts]),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

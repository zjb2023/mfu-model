"""Representative-lane FC GPU-cost substitution; outer-DAG oracle diagnostic."""
import argparse,bisect,collections,graphlib,hashlib,itertools,json
from pathlib import Path
from audit_b_operator_position_r1 import union
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
SRC=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/b-cost-blocks-r1/evidence.json')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def extract(ref,first=False):
 raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];es=json.loads(raw)['traceEvents'];del raw
 bs=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='backward_step'],key=lambda e:e['ts']);bs=bs[:1] if first else bs
 ops=[];threads=collections.defaultdict(list)
 for mb,b in enumerate(bs):
  selected=sorted([(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e.get('name') in ['_GroupedLinear','_GroupedLinearBackward'] and e['pid']==b['pid'] and b['ts']<=e['ts'] and e['ts']+e['dur']<=b['ts']+b['dur']+.01],key=lambda z:z[1]['ts']);assert len(selected)==16
  counts=collections.Counter()
  for i,e in selected:
   ph='recompute' if e['name']=='_GroupedLinear' else 'backward';n=counts[ph];counts[ph]+=1
   ops.append(dict(event=i,mb=mb,phase=ph,unit=n//2,FC=n%2+1 if ph=='recompute' else 2-n%2,args=e.get('args',{}),devices=[]))
   threads[e['pid'],e['tid']].append((e['ts'],e,len(ops)-1))
 starts={}
 for k,v in threads.items():v.sort(key=lambda x:x[0]);starts[k]=[x[0] for x in v]
 runtime=collections.defaultdict(list)
 for i,e in enumerate(es):
  if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append((i,e))
 ambiguous=[];duplicates=0
 for i,e in enumerate(es):
  if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
  hits=set();launches=[]
  for j,c in runtime.get(e.get('args',{}).get('correlation'),[]):
   k=c['pid'],c['tid']
   if k not in threads:continue
   pos=bisect.bisect_right(starts[k],c['ts'])-1
   if pos<0:continue
   _,owner,index=threads[k][pos]
   if c['ts']+c.get('dur',0)<=owner['ts']+owner['dur']+.01:hits.add(index);launches.append(j)
  if len(hits)>1:ambiguous.append(i);continue
  if not hits:continue
  duplicates+=int(len(launches)>1)
  ops[next(iter(hits))]['devices'].append(dict(event=i,launch_events=launches,name=e['name'],start_us=e['ts'],end_us=e['ts']+e['dur']))
 assert not ambiguous,('ambiguous ownership',ambiguous)
 blocks=[]
 for mb in range(len(bs)):
  selected=[o for o in ops if o['mb']==mb]
  for o in selected:
   ds=o['devices']
   if not ds:
    dims=o['args'].get('Input Dims',[])
    forward=next(x for x in selected if x['phase']=='recompute' and x['unit']==o['unit'] and x['FC']==1)
    counts=json.loads(forward['args']['Concrete Inputs'][1])
    assert dims and dims[0][0]==0 and len(counts)==20 and not any(counts),('unexplained missing devices',ref['path'],mb,o['event'])
    o.update(gpu_ms=0.,gemm_ms=0.,gemm_count=0,envelope_ms=0.,empty_expert_input_verified=True)
    continue
   iv=[(d['start_us'],d['end_us']) for d in ds];gi=[(d['start_us'],d['end_us']) for d in ds if 'gemm' in d['name'].lower()]
   o.update(gpu_ms=union(iv),gemm_ms=union(gi),gemm_count=len(gi),envelope_ms=(max(y for x,y in iv)-min(x for x,y in iv))/1000)
  # Disjoint expert activity makes representative-rank accounting unambiguous.
  assert abs(sum(o['gpu_ms'] for o in selected)-union([(d['start_us'],d['end_us']) for o in selected for d in o['devices']]))<1e-5
  blocks.append(dict(mb=mb,operators=selected,components={f'{ph}_FC{fc}':sum(o['gemm_ms'] for o in selected if o['phase']==ph and o['FC']==fc) for ph in ['recompute','backward'] for fc in [1,2]}))
 return dict(path=ref['path'],sha256=ref['sha256'],blocks=blocks,duplicate_launches_same_owner=duplicates)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 sr=next(r for r in load(SRC) if r['rank']==8);s=extract(sr,True);save(a.out/'source.json',s);src=s['blocks'][0]['components'];print('source PASS',flush=True)
 rp=BASE/'b-position-curves-r2/raw-provenance.json';refs=sorted([r for r in load(rp) if r['world']==256 and r['iteration']==60 and 1<=r['stage']<=14],key=lambda r:r['stage']);assert len(refs)==14
 rows=[]
 for ref in refs:
  d=extract(ref);save(a.out/f'pp{ref["stage"]}.json',d)
  assert len(d['blocks'])==4
  rows.extend(dict(stage=ref['stage'],mb=b['mb'],node=f's{ref["stage"]}:B{b["mb"]}',components=b['components']) for b in d['blocks']);print('PP',ref['stage'],'PASS',flush=True)
 op=BASE/'prediction-r11-recompute-r1-verified/outer-prediction.json';bp=BASE/'1f1b-overestimate-r1/blocks.json';dp=BASE/'1f1b-overestimate-r1/report.json'
 nodes=load(op)['nodes'];diag=load(dp);obs={b['node']:b['observed_duration_ms'] for b in load(bp)};cost=nodes['s1:B0']['duration_ms'];src['rest']=cost-sum(src.values())
 for r in rows:r['components']['rest']=obs[r['node']]-sum(r['components'].values());assert r['components']['rest']>0
 order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
 def run(groups):
  overrides={r['node']:cost+sum(r['components'][g]-src[g] for g in groups) for r in rows};t={}
  assert min(overrides.values())>0
  for k in order:
   n=nodes[k];t[k]=max((t[d] for d in n['dependencies']),default=0)+overrides.get(k,n['duration_ms'])
  return t['s0:B3']-(t['s0:F0']-nodes['s0:F0']['duration_ms'])
 groups=list(src);scenarios={}
 for bits in itertools.product([0,1],repeat=len(groups)):
  chosen=tuple(g for g,b in zip(groups,bits) if b);scenarios[chosen]=run(chosen)
 baseline=scenarios[()];allb=scenarios[tuple(groups)];assert abs(baseline-diag['baseline_ms'])<1e-5;assert abs(allb-diag['scenarios']['middle_B']['window_ms'])<1e-5
 shapley={g:0 for g in groups}
 for perm in itertools.permutations(groups):
  chosen=set();prev=baseline
  for g in perm:
   chosen.add(g);v=scenarios[tuple(x for x in groups if x in chosen)];shapley[g]+=(prev-v)/120;prev=v
 assert abs(sum(shapley.values())-(baseline-allb))<1e-5
 report=dict(status='PARTIAL_REPRESENTATIVE_LANE_ORACLE_NOT_PREDICTION',basis='Replace only GEMM GPU union within effective B cost, preserving residual and outer DAG; not an expanded EP8 inner-node rerun',baseline_ms=baseline,middle_B_oracle_ms=allb,source_components_ms=src,rows=rows,shapley_ms=shapley,
  scenarios=[dict(replaced=list(k),window_ms=v,reduction_ms=baseline-v,percent_of_middle_B_oracle=100*(baseline-v)/(baseline-allb)) for k,v in scenarios.items()],
  limitations=['One representative rank per stage, not all target EP8 ranks','Source rank8 closes source group; target rank stage*16 matches prior oracle convention','Frozen residual preserves local gaps but assumes GEMM union transfers additively into effective B duration','Cross-rank slack and changed inner critical paths require expanded eight-rank validation','FC labels inferred paired order, no weight identity proof','Target timing evaluator-only; no calibrated prediction or causal proof'])
 save(a.out/'report.json',report)
 inputs=[SRC,rp,op,bp,dp];codes=[Path(__file__).resolve(),Path(__file__).with_name('audit_b_operator_position_r1.py')]
 save(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in inputs},raw=[{k:r[k] for k in ['path','sha256']} for r in [sr]+refs],code={str(p):sha(p) for p in codes},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps({k:report[k] for k in ['baseline_ms','middle_B_oracle_ms','shapley_ms','scenarios']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

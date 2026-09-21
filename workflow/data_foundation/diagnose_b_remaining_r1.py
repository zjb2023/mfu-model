"""Exclusive trace-time accounting and conditional outer-DAG attribution."""
import argparse,bisect,collections,graphlib,hashlib,itertools,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
SRC=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/b-cost-blocks-r1/evidence.json')
GROUPS=['FC','CP','attention','EP','other','overlap','other_device','no_recorded_device','group_offset']
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def extract(ref,first=False):
 raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];es=json.loads(raw)['traceEvents'];del raw
 bs=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='backward_step'],key=lambda e:e['ts']);bs=bs[:1] if first else bs
 named={'_GroupedLinear','_GroupedLinearBackward','FusedDispatch','FusedCombine','FusedDispatchBackward','FusedCombineBackward','AttnFuncWithCPAndQKVOA2A','AttnFuncWithCPAndQKVOA2ABackward'}
 owners=collections.defaultdict(list)
 for i,e in enumerate(es):
  if e.get('cat')=='cpu_op' and e.get('name') in named:owners[e['pid'],e['tid']].append((e['ts'],e,i))
 starts={}
 for k,v in owners.items():v.sort(key=lambda x:x[0]);starts[k]=[x[0] for x in v]
 runtime=collections.defaultdict(list)
 for i,e in enumerate(es):
  if e.get('cat') not in ['privateuse1_runtime','privateuse1_driver'] or 'correlation' not in e.get('args',{}):continue
  hits=[m for m,b in enumerate(bs) if e['pid']==b['pid'] and b['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=b['ts']+b['dur']+.01]
  if len(hits)!=1:continue
  name=None;k=e['pid'],e['tid']
  if k in starts:
   p=bisect.bisect_right(starts[k],e['ts'])-1
   if p>=0:
    _,o,j=owners[k][p]
    if e['ts']+e.get('dur',0)<=o['ts']+o['dur']+.01:name=o['name']
  runtime[e['args']['correlation']].append((hits[0],name))
 assigned=collections.defaultdict(list);allgpu=[]
 for i,e in enumerate(es):
  if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
  allgpu.append((i,e));rs=runtime.get(e.get('args',{}).get('correlation'),[])
  if not rs:continue
  assert len({m for m,n in rs})==1
  names={n for m,n in rs if n};assert len(names)<=1
  name=next(iter(names),None);a=e.get('args',{});kn=e['name'].lower()
  if name in ['_GroupedLinear','_GroupedLinearBackward'] and 'gemm' in kn:g='FC'
  elif a.get('Process Group Description')=='CONTEXT_PARALLEL_GROUP':g='CP'
  elif name in ['AttnFuncWithCPAndQKVOA2A','AttnFuncWithCPAndQKVOA2ABackward']:g='attention'
  elif name in ['FusedDispatch','FusedCombine','FusedDispatchBackward','FusedCombineBackward']:g='EP'
  else:g='other'
  assigned[rs[0][0]].append((i,e,g,name))
 out=[]
 for mb in range(len(bs)):
  ds=assigned[mb];assert ds
  lo=min(e['ts'] for i,e,g,n in ds);hi=max(e['ts']+e['dur'] for i,e,g,n in ds);devs={e['args']['device'] for i,e,g,n in ds};assert len(devs)==1
  edges=[]
  for i,e,g,n in ds:edges.extend([(e['ts'],g,1),(e['ts']+e['dur'],g,-1)])
  for i,e in allgpu:
   if e.get('args',{}).get('device') not in devs:continue
   a=max(lo,e['ts']);b=min(hi,e['ts']+e['dur'])
   if b>a:edges.extend([(a,'all',1),(b,'all',-1)])
  totals={g:0. for g in GROUPS};active=collections.Counter();prev=lo;fc_overlap=0.
  for t,group,delta in sorted(edges):
   duration=(t-prev)/1000;present={k for k,v in active.items() if v>0 and k!='all'}
   if 'FC' in present:
    chosen='FC'
    if len(present)>1:fc_overlap+=duration
   elif len(present)>1:chosen='overlap'
   elif present:chosen=next(iter(present))
   elif active['all']>0:chosen='other_device'
   else:chosen='no_recorded_device'
   totals[chosen]+=duration;active[group]+=delta;prev=t
  assert abs(sum(totals.values())-(hi-lo)/1000)<1e-5
  out.append(dict(mb=mb,envelope_ms=(hi-lo)/1000,components=totals,FC_overlapping_other_ms=fc_overlap,
    devices=[dict(event=i,name=e['name'],group=g,owner=n,start_us=e['ts'],end_us=e['ts']+e['dur']) for i,e,g,n in ds]))
 return dict(path=ref['path'],sha256=ref['sha256'],blocks=out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 op=BASE/'prediction-r11-recompute-r1-verified/outer-prediction.json';bp=BASE/'1f1b-overestimate-r1/blocks.json';dp=BASE/'1f1b-overestimate-r1/report.json';fp=BASE/'fc-transfer-r1-verified/report.json';rp=BASE/'b-position-curves-r2/raw-provenance.json'
 nodes=load(op)['nodes'];cost=nodes['s1:B0']['duration_ms'];observed={b['node']:b['observed_duration_ms'] for b in load(bp)}
 sr=next(r for r in load(SRC) if r['rank']==8);s=extract(sr,True);src=s['blocks'][0]['components'];src['group_offset']=cost-s['blocks'][0]['envelope_ms'];assert src['group_offset']>=0;save(a.out/'source.json',s);print('source PASS',flush=True)
 refs=sorted([r for r in load(rp) if r['world']==256 and r['iteration']==60 and 1<=r['stage']<=14],key=lambda r:r['stage']);rows=[]
 for ref in refs:
  d=extract(ref);save(a.out/f'pp{ref["stage"]}.json',d)
  for b in d['blocks']:
   k=f's{ref["stage"]}:B{b["mb"]}';assert abs(b['envelope_ms']-observed[k])<1e-4
   rows.append(dict(node=k,stage=ref['stage'],mb=b['mb'],components=b['components'],FC_overlap_ms=b['FC_overlapping_other_ms']))
  print('PP',ref['stage'],'PASS',flush=True)
 order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
 def run(chosen):
  overrides={r['node']:cost+sum(r['components'][g]-src[g] for g in chosen) for r in rows};t={};assert min(overrides.values())>0
  for k in order:
   n=nodes[k];t[k]=max((t[d] for d in n['dependencies']),default=0)+overrides.get(k,n['duration_ms'])
  return t['s0:B3']-(t['s0:F0']-nodes['s0:F0']['duration_ms'])
 values={mask:run([g for i,g in enumerate(GROUPS) if mask>>i&1]) for mask in range(1<<len(GROUPS))};baseline=values[0];oracle=values[max(values)];diag=load(dp)
 assert abs(baseline-diag['baseline_ms'])<1e-5;assert abs(oracle-diag['scenarios']['middle_B']['window_ms'])<1e-4
 old=next(s for s in load(fp)['scenarios'] if len(s['replaced'])==4 and 'rest' not in s['replaced']);assert abs(values[1]-old['window_ms'])<1e-5
 n=len(GROUPS);shapley={}
 for i,g in enumerate(GROUPS):
  shapley[g]=sum(math.factorial(bin(mask).count('1'))*math.factorial(n-bin(mask).count('1')-1)/math.factorial(n)*(v-values[mask|1<<i]) for mask,v in values.items() if not mask>>i&1)
 assert abs(sum(shapley.values())-(baseline-oracle))<1e-5
 report=dict(status='PARTIAL_EXCLUSIVE_TIME_ORACLE',baseline_ms=baseline,oracle_ms=oracle,source=src,rows=rows,shapley_ms=shapley,
  single_reduction_ms={g:baseline-values[1<<i] for i,g in enumerate(GROUPS)},
  additional_after_FC_ms={g:values[1]-values[1|1<<i] for i,g in enumerate(GROUPS) if i},
  scenarios=[dict(replaced=[g for i,g in enumerate(GROUPS) if mask>>i&1],window_ms=v) for mask,v in values.items()],
  limitations=['Exclusive timeline accounting, not causal physical attribution','FC priority retains previous FC-only diagnostic; report FC overlapping other activities','Non-FC inter-category overlap retained separately, not allocated twice','CP kernel time includes wait; EP-associated GPU time is not complete transport service','No recorded device activity is not proven idle or avoidable wait','Representative lane effective-B replacement, not complete EP8 inner-node rescheduling','Other_device includes activity not associated to this B','Only iter60; target data evaluator-only'])
 save(a.out/'report.json',report);inputs=[SRC,op,bp,dp,fp,rp]
 save(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in inputs},code={str(Path(__file__).resolve()):sha(Path(__file__))},raw=[{k:r[k] for k in ['path','sha256']} for r in [sr]+refs],outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps({k:report[k] for k in ['shapley_ms','single_reduction_ms','additional_after_FC_ms']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

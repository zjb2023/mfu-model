"""Refine cached other-device accounting using launch CPU ancestors, FC held fixed."""
import argparse,collections,graphlib,hashlib,heapq,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
CACHE=BASE/'b-remaining-r1-verified'
GROUPS=['token_reorder','expert_activation','attention_known','router','unresolved_linear','unresolved_copy','other_unresolved','background']
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def refine(p):
 old=load(p);raw=Path(old['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==old['sha256'];es=json.loads(raw)['traceEvents'];del raw
 ids={d['event'] for b in old['blocks'] for d in b['devices']};corrs={es[i].get('args',{}).get('correlation') for i in ids}
 cp=collections.defaultdict(list);rs=collections.defaultdict(list)
 for i,e in enumerate(es):
  if e.get('cat')=='cpu_op':cp[e['pid'],e['tid']].append((i,e))
  elif e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and e.get('args',{}).get('correlation') in corrs:rs[e['pid'],e['tid']].append((i,e))
 chains=collections.defaultdict(list)
 for key,calls in rs.items():
  cpu=sorted(cp[key],key=lambda x:x[1]['ts']);pos=0;active={};ends=[]
  for i,r in sorted(calls,key=lambda x:x[1]['ts']):
   while pos<len(cpu) and cpu[pos][1]['ts']<=r['ts']:
    j,c=cpu[pos];active[j]=c;heapq.heappush(ends,(c['ts']+c['dur'],j));pos+=1
   while ends and ends[0][0]<r['ts']:
    _,j=heapq.heappop(ends);active.pop(j,None)
   parents=[dict(event=j,name=c['name'],start=c['ts'],end=c['ts']+c['dur']) for j,c in active.items() if r['ts']+r.get('dur',0)<=c['ts']+c['dur']+.01]
   chains[r['args']['correlation']].append(dict(runtime=i,ts=r['ts'],parents=parents))
 out=[]
 for b in old['blocks']:
  assert abs(b['components']['other_device'])<1e-9
  # Expert activation only when directly between paired expert linears on CPU.
  expert_windows=[]
  seen={}
  for d in b['devices']:
   if d['group']!='FC':continue
   for c in chains[es[d['event']]['args']['correlation']]:
    for par in c['parents']:
     if par['name'] in ['_GroupedLinear','_GroupedLinearBackward']:seen[par['event']]=par
  for name in ['_GroupedLinear','_GroupedLinearBackward']:
   ops=sorted([v for v in seen.values() if v['name']==name],key=lambda x:x['start'])
   # Empty expert layers have no FC devices; never infer a window across them.
   for left,right in zip(ops[::2],ops[1::2]):expert_windows.append((left['end'],right['start']))
  devices=[]
  for d in b['devices']:
   ch=chains.get(es[d['event']].get('args',{}).get('correlation'),[])
   names=sorted({x['name'] for c in ch for x in c['parents']});text=' '.join(names).lower();kind=None;reason=None
   if d['group']=='attention':kind='attention_known';reason='Attention autograd ancestor (excludes CP)'
   elif d['group']=='other':
    if any(x in text for x in ['_moe_permute','_moe_unpermute']):kind='token_reorder';reason='explicit MoE permute/unpermute ancestor'
    elif 'rotaryemb' in text:kind='attention_known';reason='explicit rotary embedding ancestor'
    elif 'routergating' in text or 'aten::topk' in names:kind='router';reason='router gating/topk ancestor; not entire routing implementation'
    elif any(x in text for x in ['silu','swiglu','mulbackward','aten::mul']) and any(lo<=c['ts']<=hi for c in ch for lo,hi in expert_windows):kind='expert_activation';reason='activation/multiply within paired expert linear CPU window'
    elif any(x in names for x in ['_Linear','_LinearBackward','_LayerNormLinear','_LayerNormLinearBackward']):kind='unresolved_linear';reason='linear ancestor, module identity unproven'
    elif any(x in names for x in ['aten::copy_','aten::_copy_from','aten::clone']):kind='unresolved_copy';reason='copy ancestor, module identity unproven'
    else:kind='other_unresolved';reason='unclassified other activity'
   else:kind='background';reason='existing CP/EP/FC/etc bucket retained separately'
   devices.append(dict(**d,semantic=kind,reason=reason,cpu_parent_names=names))
  edges=[]
  for j,d in enumerate(devices):edges.extend([(d['start_us'],j,1),(d['end_us'],j,-1)])
  active=set();totals={g:0. for g in GROUPS};fc=0.;last=min(d['start_us'] for d in devices)
  for t,j,sign in sorted(edges):
   dt=(t-last)/1000;og={devices[k]['group'] for k in active}
   if 'FC' in og:fc+=dt
   elif len(og)==1 and next(iter(og)) in ['other','attention']:
    sg={devices[k]['semantic'] for k in active};chosen=next(iter(sg)) if len(sg)==1 else 'other_unresolved';totals[chosen]+=dt
   else:totals['background']+=dt
   if sign==1:active.add(j)
   else:active.remove(j)
   last=t
  totals['background']+=b['components']['group_offset']
  assert abs(fc-b['components']['FC'])<1e-5
  assert abs(fc+sum(totals.values())-sum(b['components'].values()))<1e-5
  out.append(dict(mb=b['mb'],FC=fc,components=totals,devices=devices))
 return dict(path=old['path'],sha256=old['sha256'],blocks=out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 s=refine(CACHE/'source.json');save(a.out/'source.json',s);src=s['blocks'][0];rows=[];print('source PASS',flush=True)
 for stage in range(1,15):
  d=refine(CACHE/f'pp{stage}.json');save(a.out/f'pp{stage}.json',d)
  rows.extend(dict(stage=stage,node=f's{stage}:B{b["mb"]}',mb=b['mb'],FC=b['FC'],components=b['components']) for b in d['blocks']);print('PP',stage,'PASS',flush=True)
 op=BASE/'prediction-r11-recompute-r1-verified/outer-prediction.json';nodes=load(op)['nodes'];cost=nodes['s1:B0']['duration_ms'];order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
 def run(mask):
  overrides={r['node']:cost+r['FC']-src['FC']+sum(r['components'][g]-src['components'][g] for i,g in enumerate(GROUPS) if mask>>i&1) for r in rows};t={}
  for k in order:
   n=nodes[k];t[k]=max((t[d] for d in n['dependencies']),default=0)+overrides.get(k,n['duration_ms'])
  return t['s0:B3']-(t['s0:F0']-nodes['s0:F0']['duration_ms'])
 n=len(GROUPS);v={m:run(m) for m in range(1<<n)};old=load(CACHE/'report.json');expected=next(s['window_ms'] for s in old['scenarios'] if s['replaced']==['FC']);assert abs(v[0]-expected)<1e-5;assert abs(v[(1<<n)-1]-old['oracle_ms'])<1e-5
 shp={g:sum(math.factorial(bin(m).count('1'))*math.factorial(n-bin(m).count('1')-1)/math.factorial(n)*(x-v[m|1<<i]) for m,x in v.items() if not m>>i&1) for i,g in enumerate(GROUPS)}
 assert abs(sum(shp.values())-(v[0]-v[(1<<n)-1]))<1e-5
 truth=20396.01318693161
 report=dict(status='PARTIAL_CONFIRMED_SUBSET_ATTRIBUTION',FC_fixed_window_ms=v[0],remaining_error_ms=v[0]-truth,all_middle_B_window_ms=v[(1<<n)-1],source=src['components'],rows=rows,conditional_shapley_ms=shp,percent_of_426ms={g:x/(v[0]-truth)*100 for g,x in shp.items()},single_after_FC_ms={g:v[0]-v[1<<i] for i,g in enumerate(GROUPS)},
   scenarios=[dict(replaced=[g for i,g in enumerate(GROUPS) if m>>i&1],window_ms=x) for m,x in v.items()],limitations=['Semantic labels are confirmed subsets, not full module coverage','Attention projections and shared expert linears remain unresolved unless explicit ancestor identifies them','Token reorder excludes unidentifiable DeepEP transport packing','Activation window is inference bounded by expert calls, not weight-ID proof','Original inter-category overlap retained in background; inner semantic overlap kept unresolved','Representative-rank effective B substitution; not full EP8 causal graph','Shapley allocation depends on grouping; not hardware cause or measured speedup'])
 save(a.out/'report.json',report);inputs=[op,CACHE/'report.json',CACHE/'manifest.json']+[CACHE/f for f in ['source.json']+[f'pp{i}.json' for i in range(1,15)]]
 save(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in inputs},raw=[{k:d[k] for k in ['path','sha256']} for d in [s]+[load(a.out/f'pp{i}.json') for i in range(1,15)]],code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps({k:report[k] for k in ['conditional_shapley_ms','percent_of_426ms','single_after_FC_ms']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

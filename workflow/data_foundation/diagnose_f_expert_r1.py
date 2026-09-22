"""F-only expert compute substitution on top of prior B two-group oracle."""
import argparse,collections,graphlib,hashlib,heapq,json
from pathlib import Path
from audit_b_operator_position_r1 import union
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def extract(ref,first=False):
 raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];es=json.loads(raw)['traceEvents'];del raw
 fs=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='forward_step'],key=lambda e:e['ts']);fs=fs[:1] if first else fs
 assert len(fs)==(1 if first else 4)
 cpu=collections.defaultdict(list);runtime=collections.defaultdict(list);windows=[];opgroups={};synthetic={}
 for i,e in enumerate(es):
  if e.get('cat')=='cpu_op':cpu[e['pid'],e['tid']].append((i,e))
 for mb,f in enumerate(fs):
  ops=sorted([(i,e) for xs in cpu.values() for i,e in xs if e['name']=='_GroupedLinear' and e['pid']==f['pid'] and f['ts']<=e['ts'] and e['ts']+e['dur']<=f['ts']+f['dur']+.01],key=lambda x:x[1]['ts'])
  if not ops:
   local=[e for xs in cpu.values() for i,e in xs if e['pid']==f['pid'] and f['ts']<=e['ts'] and e['ts']+e['dur']<=f['ts']+f['dur']+.01]
   ps=sorted([e for e in local if e['name']=='_moe_permute_mask_map'],key=lambda e:e['ts']);us=sorted([e for e in local if e['name']=='_moe_unpermute_mask_map'],key=lambda e:e['ts']);assert len(ps)==len(us)==4
   for u,(p,z) in enumerate(zip(ps,us)):
    aa=[e for e in local if e['name']=='aten::silu' and p['ts']+p['dur']<=e['ts']<z['ts']];assert len(aa)==1;act=aa[0];assert p['tid']==z['tid']==act['tid']
    windows.append((mb,p['pid'],p['tid'],p['ts']+p['dur'],z['ts']))
    for fc,lo,hi in [(1,p['ts']+p['dur'],act['ts']),(2,act['ts'],z['ts'])]:
     i=-(mb*8+u*2+fc);e=dict(name='inferred_expert_FC',pid=p['pid'],tid=p['tid'],ts=lo,dur=hi-lo,args={'Input Dims':act['args']['Input Dims'],'boundary_basis':'permute/silu/unpermute CPU boundaries; only GEMM GPU counted'})
     synthetic[i]=e;ops.append((i,e));cpu[e['pid'],e['tid']].append((i,e))
  assert len(ops)==8,(mb,len(ops))
  for j,(i,e) in enumerate(ops):opgroups[i]=(mb,'FC1' if j%2==0 else 'FC2',j//2)
  for (i,l),(j,r) in zip(ops[::2],ops[1::2]):
   assert l['tid']==r['tid'];windows.append((mb,l['pid'],l['tid'],l['ts']+l['dur'],r['ts']))
 for i,e in enumerate(es):
  if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}):
   if any(e['pid']==f['pid'] and f['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=f['ts']+f['dur']+.01 for f in fs):runtime[e['pid'],e['tid']].append((i,e))
 ownership=collections.defaultdict(list)
 for key,calls in runtime.items():
  cps=sorted(cpu[key],key=lambda x:x[1]['ts']);pos=0;active={};ends=[]
  for i,c in sorted(calls,key=lambda x:x[1]['ts']):
   while pos<len(cps) and cps[pos][1]['ts']<=c['ts']:
    j,e=cps[pos];active[j]=e;heapq.heappush(ends,(e['ts']+e['dur'],j));pos+=1
   while ends and ends[0][0]<c['ts']:
    _,j=heapq.heappop(ends);active.pop(j,None)
   parents={j:e for j,e in active.items() if c['ts']+c.get('dur',0)<=e['ts']+e['dur']+.01};ops=[j for j in parents if j in opgroups];assert len(ops)<=1
   group=None;mb=None
   if ops and ops[0]>=0:mb,group,u=opgroups[ops[0]]
   else:
    names=' '.join(e['name'].lower() for e in parents.values())
    if any(x in names for x in ['silu','swiglu','aten::mul']):
     hits=[m for m,p,t,lo,hi in windows if (p,t)==key and lo<=c['ts'] and c['ts']+c.get('dur',0)<=hi+.01]
     assert len(hits)<=1
     hits=list(set(hits))
     if hits:mb=hits[0];group='activation'
    if group is None and ops:mb,group,u=opgroups[ops[0]]
   if group:ownership[c['args']['correlation']].append((mb,group,i,[e['name'] for e in parents.values()]))
 ds=collections.defaultdict(list)
 for i,e in enumerate(es):
  if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
  hits=ownership.get(e.get('args',{}).get('correlation'),[])
  if not hits:continue
  assert len({(m,g) for m,g,j,n in hits})==1
  m,g,_,_=hits[0]
  if g in ['FC1','FC2'] and 'gemm' not in e['name'].lower():continue
  ds[m].append(dict(event=i,name=e['name'],group=g,start_us=e['ts'],end_us=e['ts']+e['dur'],launch_events=[j for m,g,j,n in hits],parents=hits[0][3]))
 blocks=[]
 for mb in range(len(fs)):
  parts={g:union([(d['start_us'],d['end_us']) for d in ds[mb] if d['group']==g]) for g in ['FC1','FC2','activation']}
  total=union([(d['start_us'],d['end_us']) for d in ds[mb]]);assert abs(sum(parts.values())-total)<1e-5
  ops=[dict(event=i,group=g,unit=u,args=(synthetic[i] if i<0 else es[i]).get('args',{})) for i,(m,g,u) in opgroups.items() if m==mb]
  # Nonempty expert calls must have a correlated GEMM; empty-input calls retained explicitly.
  for o in ops:
   e=synthetic[o['event']] if o['event']<0 else es[o['event']];has=any(e['ts']<=es[j]['ts'] and es[j]['ts']+es[j].get('dur',0)<=e['ts']+e['dur']+.01 for d in ds[mb] if d['group']==o['group'] for j in d['launch_events'])
   assert has or o['args']['Input Dims'][0][0]==0,('unmatched nonempty FC',o['event'])
  blocks.append(dict(mb=mb,components=parts,total_ms=total,operators=ops,devices=ds[mb]))
 return dict(path=ref['path'],sha256=ref['sha256'],blocks=blocks)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 srp=BASE/'token-attention-r1/source.json';sr=load(srp);s=extract(sr,True);save(a.out/'source.json',s);print('source F0 PASS',flush=True)
 rp=BASE/'b-position-curves-r2/raw-provenance.json';refs=sorted([r for r in load(rp) if r['world']==256 and r['iteration']==60 and 1<=r['stage']<=14],key=lambda r:r['stage']);rows=[]
 for ref in refs:
  d=extract(ref);save(a.out/f'pp{ref["stage"]}.json',d);rows.extend(dict(node=f's{ref["stage"]}:F{b["mb"]}',stage=ref['stage'],mb=b['mb'],components=b['components'],total_ms=b['total_ms']) for b in d['blocks']);print('PP',ref['stage'],'PASS',flush=True)
 op=BASE/'prediction-r11-recompute-r1-verified/outer-prediction.json';bp=BASE/'moe-two-groups-r1/report.json';nodes=load(op)['nodes'];bdiag=load(bp);bc=bdiag['source_components_ms']
 fixed={r['node']:nodes[r['node']]['duration_ms']+sum(r['components'][g]-bc[g] for g in ['expert_compute','token_reorder_EP_visible']) for r in bdiag['rows']}
 order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
 def run(selected):
  ov=dict(fixed)
  for r in rows:ov[r['node']]=nodes[r['node']]['duration_ms']+sum(r['components'][g]-s['blocks'][0]['components'][g] for g in selected)
  assert min(ov.values())>=0;t={}
  for k in order:
   n=nodes[k];t[k]=max((t[d] for d in n['dependencies']),default=0)+ov.get(k,n['duration_ms'])
  return t['s0:B3']-(t['s0:F0']-nodes['s0:F0']['duration_ms'])
 scenarios=[]
 for selected in [[],['FC1','FC2'],['activation'],['FC1','FC2','activation']]:
  v=run(selected);scenarios.append(dict(F_replaced=selected,window_ms=v,error_ms=v-bdiag['truth_ms'],error_pct=100*(v/bdiag['truth_ms']-1)))
 expected=next(r['window_ms'] for r in bdiag['scenarios'] if set(r['replaced'])=={'expert_compute','token_reorder_EP_visible'});assert abs(scenarios[0]['window_ms']-expected)<1e-5
 report=dict(status='PARTIAL_F_EXPERT_ORACLE',source=s['blocks'][0]['components'],rows=rows,truth_ms=bdiag['truth_ms'],scenarios=scenarios,additional_reduction_ms=scenarios[0]['window_ms']-scenarios[-1]['window_ms'],limitations=['Representative rank8 source F0 and target stage*16; not complete EP8 inner graph','Only F GEMM and activation/gating CPU-window-associated device union replaced; other F effective costs fixed','B two-group oracle remains fixed in every scenario','Activation identity via CPU position and names, not full weight-ID proof','Target timing is evaluator-only, not independent prediction accuracy'])
 save(a.out/'report.json',report);inputs=[srp,rp,op,bp]
 save(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in inputs},code={str(Path(__file__).resolve()):sha(Path(__file__)),str(Path(__file__).with_name('audit_b_operator_position_r1.py')):sha(Path(__file__).with_name('audit_b_operator_position_r1.py'))},raw=[{k:r[k] for k in ['path','sha256']} for r in [sr]+refs],outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps({k:report[k] for k in ['source','scenarios','additional_reduction_ms']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

"""Replace r10 recompute envelopes with conditional compute/CP/EP subgraphs.

Same-sample replay is a regression gate, not independent predictive validation.
Communication ports are effective replacements, never raw backend additions.
"""
import argparse,collections,copy,hashlib,json,math
from pathlib import Path
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
SOURCE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/b-cost-blocks-r1/evidence.json')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 oldp=BASE/'b-four-layers-r10/graph.json';g=load(oldp);old=run(g)
 seal=load(BASE/'b-four-layers-r10/manifest.json');assert sha(SOURCE)==next(x['sha256'] for x in seal['inputs'] if x['path']==str(SOURCE))
 xs=load(SOURCE);origin=min(d['start_ms'] for x in xs for d in x['devices']);evidence=[];raws=[]
 # Recover the two forward expert calls in each checkpoint; no F0 durations reused.
 for x in xs:
  raw=Path(x['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==x['sha256'];es=json.loads(raw)['traceEvents'];del raw
  raws.append({k:x[k] for k in ['path','sha256']})
  def within(e,o):return e.get('pid')==o['pid'] and e.get('tid')==o['tid'] and o['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=o['ts']+o['dur']+.01
  for u,ckid in enumerate(x['checkpoints']):
   ck=es[ckid];ops=sorted([(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e.get('name')=='_GroupedLinear' and within(e,ck)],key=lambda z:z[1]['ts']);assert len(ops)==2
   for d in x['devices']:
    if d['unit']!=u or d['phase']!='recompute':continue
    c=es[d['launch_event']];kind='local_compute'
    if d['kind'].startswith('CP_'):kind='CP'
    elif 'ace_notify_dispatch' in d['name']:kind='EP_notify'
    elif d['kind']=='EP_dispatch_recompute':kind='EP_dispatch_local'
    elif d['kind']=='EP_combine_recompute':kind='EP_combine_visible'
    elif 'flash_atten' in d['name']:kind='attention_compute'
    elif d['kind'].startswith('COMM_'):kind='aux_communication'
    elif d['kind']=='local_memory' or 'copy' in d['name'].lower() or 'index_select' in d['name']:kind='local_reorder'
    for j,(oid,op) in enumerate(ops):
     if within(c,op):kind='expert_FC'+str(j+1)
    evidence.append(dict(**d,rank=x['rank'],semantic=kind,relative_start_ms=d['start_ms']-origin,relative_end_ms=d['end_ms']-origin))
  print('recompute evidence rank',x['rank'],'PASS',flush=True)
 expected={};blocks={};cp={};notify={};combine={};fc={};coupling=[]
 for u in range(4):
  for r in range(8,16):
   ds=[d for d in evidence if d['unit']==u and d['rank']==r];keys=[]
   for stream in sorted({d['stream'] for d in ds}):
    groups=[]
    for d in sorted([d for d in ds if d['stream']==stream],key=lambda d:d['relative_start_ms']):
     sig=(d['semantic'],d['anchor'])
     if groups and groups[-1][0]==sig:groups[-1][1].append(d)
     else:groups.append([sig,[d]])
    prev=f'r{r}:u{u}:recompute:handoff'
    for j,((kind,anchor),items) in enumerate(groups):
     k=f'R{u}:r{r}:s{stream}:n{j}';ready=k+':ready';lo=min(d['relative_start_ms'] for d in items);hi=max(d['relative_end_ms'] for d in items)
     before=expected.get(prev,old.get(prev,{}).get('end_ms'));assert lo>=before-1e-5
     g[ready]=dict(deps=[prev],cost_ms=max(0,lo-before),policy='trace_reserved',kind='recompute_readiness',rank=r,unit=u,basis='local source readiness residual; not pure idle');expected[ready]=lo
     policy='external_effective_replace' if kind not in ['local_reorder'] else 'trace_reserved'
     g[k]=dict(deps=[ready],cost_ms=hi-lo,policy=policy,kind='recompute_'+kind,rank=r,unit=u,stream=stream,events=[d['event'] for d in items],basis='source recompute same-stream block; effective replacement, not pure service',names=sorted({d['name'] for d in items}));expected[k]=hi
     blocks[k]=dict(start_ms=lo,end_ms=hi,kind=kind,rank=r,unit=u,stream=stream);keys.append(k);prev=k
     if kind=='CP':cp.setdefault((u,r),[]).append(k)
     if kind=='EP_notify':assert (u,r) not in notify;notify[u,r]=k
     if kind=='EP_combine_visible':assert (u,r) not in combine;combine[u,r]=k
     if kind.startswith('expert_FC'):fc.setdefault((u,r,kind),[]).append(k)
   key=f'r{r}:u{u}:recompute';assert abs(max(expected[k] for k in keys)-old[key]['end_ms'])<1e-5
   g[key].update(deps=keys,cost_ms=0,kind='recompute_exit',policy='derived_boundary',removed_envelope_ms=g[key]['cost_ms'],basis='zero-cost join replaces old recompute envelope; no double charging')
 def rebase_ready(k,deps):
  ready=k+':ready';deps=list(dict.fromkeys(g[ready]['deps']+deps));start=blocks[k]['start_ms'];end=max(expected.get(d,old.get(d,{}).get('end_ms')) for d in deps)
  assert start>=end-1e-5,(k,start,end)
  g[ready].update(deps=deps,cost_ms=max(0,start-end),basis='conditional local producer handoff plus source residual; not event-handle verified')
 # Local producer/consumer handoffs. Only explicit CP/Attention/EP relationships;
 # no generic all-stream barrier.
 for u in range(4):
  for r in range(8,16):
   local={k:v for k,v in blocks.items() if v['unit']==u and v['rank']==r}
   cps=sorted(cp[u,r],key=lambda k:blocks[k]['start_ms']);assert len(cps)==4;cp[u,r]=cps
   for k in cps:
    eligible=[j for j,v in local.items() if v['stream']=='0' and v['end_ms']<=blocks[k]['start_ms']+1e-5]
    if eligible:rebase_ready(k,[max(eligible,key=lambda j:expected[j])])
   for k,v in local.items():
    if v['kind'] in ['CP','EP_notify']:continue
    eligible=[j for j in cps if expected[j]<=v['start_ms']+1e-5]
    if eligible and v['kind'] in ['attention_compute','local_reorder','local_compute','EP_dispatch_local','expert_FC1','expert_FC2']:rebase_ready(k,[max(eligible,key=lambda j:expected[j])])
    if v['kind']=='attention_compute':
     eligible=[j for j,z in local.items() if z['stream']=='12' and z['end_ms']<=v['start_ms']+1e-5]
     if eligible:rebase_ready(k,[max(eligible,key=lambda j:expected[j])])
    if v['kind'].startswith('expert_FC'):
     eligible=[j for j,z in local.items() if z['kind'] in ['EP_notify','EP_dispatch_local'] and z['end_ms']<=v['start_ms']+1e-5]
     if eligible:rebase_ready(k,eligible)
 # CP2 common-overlap completion, matching the existing F/B effective-cost contract.
 for u in range(4):
  for p in range(8,16,2):
   for j in range(4):
    ks=[cp[u,r][j] for r in [p,p+1]];lo=max(blocks[k]['start_ms'] for k in ks);hi=min(expected[k] for k in ks);assert hi>=lo
    key=f'R{u}:p{p}-{p+1}:CP{j}';g[key]=dict(deps=[k+':ready' for k in ks],cost_ms=hi-lo,policy='external_effective_pair_overlap_replace',kind='recompute_CP_effective',rank=None,members=[p,p+1],unit=u,basis='conditional CP2 common-overlap proxy; not pure backend service');expected[key]=hi
    for k in ks:g[k].update(deps=[key],cost_ms=expected[k]-hi,policy='trace_reserved',basis='rank completion offset; full CP cost replaced, not added')
    coupling.append(dict(node=key,exits=ks,type='CP2'))
  ks=[notify[u,r] for r in range(8,16)];lo=max(blocks[k]['start_ms'] for k in ks);hi=min(expected[k] for k in ks);assert hi>=lo
  key=f'R{u}:EP8:dispatch_notify';g[key]=dict(deps=[k+':ready' for k in ks],cost_ms=hi-lo,policy='external_effective_replace',kind='recompute_EP_dispatch_notify_effective',rank=None,unit=u,basis='conditional EP8 notify overlap; dispatch service not separately identifiable');expected[key]=hi
  for k in ks:g[k].update(deps=[key],cost_ms=expected[k]-hi,policy='trace_reserved',basis='local notify completion offset')
  coupling.append(dict(node=key,exits=ks,type='EP8_notify'))
  # Consumers of local notify must see its coupled completion.
  for r in range(8,16):
   for k,v in blocks.items():
    if v['unit']==u and v['rank']==r and v['kind'] in ['EP_dispatch_local','expert_FC1'] and v['start_ms']>=expected[notify[u,r]]-1e-5:rebase_ready(k,[notify[u,r]])
  deps=[k for k,v in blocks.items() if v['unit']==u and v['kind']!='EP_combine_visible'];start=max(expected[k] for k in deps);end=min(blocks[combine[u,r]]['start_ms'] for r in range(8,16));assert end>=start-1e-5
  key=f'R{u}:EP8:combine_effective';g[key]=dict(deps=deps,cost_ms=max(0,end-start),policy='external_effective_replace',kind='recompute_EP_combine_effective',rank=None,unit=u,basis='max eight-rank local preparation plus residual to first visible combine; conditional, not physical barrier proof');expected[key]=end
  for r in range(8,16):rebase_ready(combine[u,r],[key])
  coupling.append(dict(node=key,exits=[combine[u,r] for r in range(8,16)],type='EP8_combine'))
 baseline=run(g)
 for k,v in expected.items():assert abs(baseline[k]['end_ms']-v)<1e-5,(k,baseline[k]['end_ms'],v)
 for k,v in old.items():assert abs(baseline[k]['end_ms']-v['end_ms'])<1e-5,(k,'old boundary changed')
 return g,baseline,evidence,coupling,fc,raws,oldp
def apply_overrides(graph,request):
 if request.get('unit')!='ms' or request.get('basis')!='effective_replacement':raise ValueError('unit/basis mismatch')
 out=copy.deepcopy(graph)
 for k,v in request.get('nodes',{}).items():
  if k not in out or not out[k].get('policy','').startswith('external') or type(v) not in [float,int] or not math.isfinite(v) or v<0:raise ValueError('invalid override '+k)
  out[k]['cost_ms']=v
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--overrides',type=Path);a=ap.parse_args();assert not a.out.exists()
 g,t,ev,couplings,fc,raws,oldp=build();tests=[]
 for c in couplings:
  changed=copy.deepcopy(g);changed[c['node']]['cost_ms']+=1;new=run(changed)
  assert all(abs(new[k]['end_ms']-t[k]['end_ms']-1)<1e-5 for k in c['exits'])
  tests.append(dict(node=c['node'],plus_ms=1,B_shift_ms=new['END']['end_ms']-t['END']['end_ms'],exits='PASS'))
 for (u,r,kind),ks in fc.items():
  assert len(ks)==1;changed=copy.deepcopy(g);changed[ks[0]]['cost_ms']+=100;new=run(changed)
  exits=next(c['exits'] for c in couplings if c['node']==f'R{u}:EP8:combine_effective')
  assert all(new[k]['end_ms']>t[k]['end_ms'] for k in exits) and new['END']['end_ms']>t['END']['end_ms']
  tests.append(dict(node=ks[0],plus_ms=100,B_shift_ms=new['END']['end_ms']-t['END']['end_ms'],all_EP8_combine_shift='PASS'))
 key=next(k for k,n in g.items() if n.get('policy','').startswith('external') and k.startswith('R'))
 for req in [dict(unit='s',basis='effective_replacement',nodes={}),dict(unit='ms',basis='effective_replacement',nodes={key:-1}),dict(unit='ms',basis='effective_replacement',nodes={'END':1}),dict(unit='ms',basis='effective_replacement',nodes={key:float('nan')})]:
  try:apply_overrides(g,req)
  except ValueError:pass
  else:raise AssertionError('invalid override accepted')
 bindings={k:dict(value_ms=n['cost_ms'],policy=n.get('policy','trace_reserved'),kind=n.get('kind'),unit=n.get('unit'),rank=n.get('rank'),basis=n.get('basis')) for k,n in g.items() if k.startswith('R')}
 summary=dict(status='PASS_REPLAY_PARTIAL_CONDITIONAL_RECOMPUTE_DAG',B_ms=t['END']['end_ms'],graph_nodes=len(g),replaced_recompute_envelopes=32,recompute_device_events=len(ev),cost_tests=len(tests),limits=['Same iter60 B0 calibration replay, not iter70/256 validation','CP2/EP8 and local handoffs are conditional; source residuals still empirical','Effective communication replacement only; raw bandwidth/FCT must be converted first','FC1/FC2 identity from paired forward order, not weight IDs','True backward r10 costs/dependencies unchanged except upstream recompute exit dependencies'])
 a.out.mkdir(parents=True)
 for name,obj in [('graph',g),('replay',t),('evidence',ev),('coupling',couplings),('cost-bindings',bindings),('tests',tests),('summary',summary)]:
  (a.out/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 if a.overrides:(a.out/'overridden-replay.json').write_text(json.dumps(run(apply_overrides(g,load(a.overrides))),indent=2)+'\n')
 inputs=[SOURCE,oldp,BASE/'b-four-layers-r10/manifest.json']+([a.overrides] if a.overrides else [])
 (a.out/'manifest.json').write_text(json.dumps(dict(version='b-recompute-r11',inputs={str(p):sha(p) for p in inputs},raw=raws,code={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('build_b_cp_pair_r4.py')]},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}),indent=2)+'\n')
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

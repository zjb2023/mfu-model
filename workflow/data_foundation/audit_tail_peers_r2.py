"""Resolve global EDP peers and distinct AG calls, bounded four target traces."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
DATA=Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 paths={rank:list(DATA.glob(f'worker*/profiler/iteration_60/rank{rank}.*.pt.trace.json')) for rank in [0,1,8,9]};assert all(len(v)==1 for v in paths.values());paths={k:v[0] for k,v in paths.items()};rows=[]
 for rank,p in paths.items():
  raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();d=json.loads(raw);del raw;es=d['traceEvents'];info=d['distributedInfo'];assert info['rank']==rank
  groups={g['pg_name']:g for g in info['pg_config']};step=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep'));f=next(e for e in sorted(es,key=lambda e:e.get('ts',0)) if e.get('cat')=='user_annotation' and e.get('name')=='forward_step')
  bs=[e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='backward_step'];last_b=max(e['ts'] for e in bs)
  calls=[];startup=[]
  for i,e in enumerate(es):
   arg=e.get('args',{});collective=arg.get('Collective name','')
   if e.get('cat')!='kernel' or not collective:continue
   is_tail=e['ts']>last_b and ('allgather_into_tensor_coalesced' in collective or 'reduce_scatter_tensor_coalesced' in collective)
   is_start=e['ts']<f['ts'] and collective=='_allgather_base'
   if not(is_tail or is_start):continue
   rs=[(j,x) for j,x in enumerate(es) if x.get('cat')=='privateuse1_runtime' and x.get('args',{}).get('correlation')==arg.get('correlation')];assert len(rs)==1;ri,r=rs[0]
   parents=[(j,x) for j,x in enumerate(es) if x.get('cat') in ['cpu_op','user_annotation'] and x.get('pid')==r['pid'] and x.get('tid')==r['tid'] and x['ts']<=r['ts'] and r['ts']+r.get('dur',0)<=x['ts']+x.get('dur',0)+.01]
   pg=groups[arg['Process Group Name']];width={'Float':4,'BFloat16':2}[arg['dtype']]
   record=dict(event=i,collective=collective,duration_ms=e['dur']/1000,start_from_step_ms=(e['ts']-step['ts'])/1000,runtime_event=ri,stream=arg['stream'],group=pg,event_ranks_as_recorded=arg.get('Process Group Ranks'),input_bytes=int(arg['In msg nelems'])*width,output_bytes=int(arg['Out msg nelems'])*width,
    parents=[dict(event=j,name=x['name'],duration_ms=x.get('dur',0)/1000,args=x.get('args',{})) for j,x in parents])
   (calls if is_tail else startup).append(record)
  rows.append(dict(rank=rank,path=str(p),sha256=digest,worker=p.parents[2].name,tail_calls=sorted(calls,key=lambda r:r['start_from_step_ms']),startup=startup,profiler_label=step['name']))
  print(rank,'PASS',flush=True)
 for row in rows:
  edp=[c for c in row['tail_calls'] if c['group']['pg_desc']=='EXPERT_DATA_PARALLEL_GROUP'];assert edp
  assert edp[0]['group']['ranks']==([0,8] if row['rank'] in [0,8] else [1,9])
  for desc in ['DATA_PARALLEL_GROUP_WITH_CP','EXPERT_DATA_PARALLEL_GROUP']:
   ag=[c for c in row['tail_calls'] if c['group']['pg_desc']==desc and 'allgather' in c['collective']];assert len(ag)==2
   ancestors=[next(p['event'] for p in c['parents'] if p['name']=='step_with_ready_grads') for c in ag];assert len(set(ancestors))==2
 rank0=rows[0];report=dict(status='PASS_GROUP_MAPPING_PARTIAL_COST_MODEL',rows=rows,
  findings=['Global pg_config resolves rank0 EDP peers to [0,8], not event-local [0,1]','rank0/1 on worker33087 and rank8/9 on worker33075: cross-worker EDP; DP16 includes both workers','Two AG calls per DP/EDP group under distinct step_with_ready_grads parents, not duplicate records of one kernel','Both AG calls expose a one-tensor input list of the same full size; not two half-size buckets','Startup small allgather timings are local to each trace; no calibrated interhost clock alignment'],
  limits=['Worker mapping is host-directory evidence, not NIC routing/rail assignment','Distinct calls do not prove redundant work; optimizer/model buffer ownership not established','Target 400Gbps, 60% and hierarchical algorithm remain explicit candidate assumptions','No global startup last-arrival identified from four ranks','Profiler marker is #59 while directory is iteration_60; preserve original labeling'])
 save(a.out/'report.json',report);save(a.out/'manifest.json',dict(raw=[{k:r[k] for k in ['path','sha256']} for r in rows],code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps([dict(rank=r['rank'],worker=r['worker'],startup_ms=[s['duration_ms'] for s in r['startup']],tail=[(c['collective'],c['group']['ranks'],c['duration_ms']) for c in r['tail_calls']]) for r in rows],ensure_ascii=False),flush=True)
if __name__=='__main__':main()

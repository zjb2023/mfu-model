"""Read two rank0 traces; freeze communication metadata and 60%-utilization scenarios."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
TOPO=Path('/home/zjb/Desktop/32/gpu32_gbs64_link_nic/20260918_103850/worker34095/nic/metadata/tcc_nic_topology.json')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 inputs=[BASE/'optimizer-tail-audit-r1'/f'{n}.json' for n in ['source','target']]+[TOPO];inventory=[];starts=[];rawrefs=[]
 for label,p in zip(['source','target'],inputs[:2]):
  prior=load(p);raw=Path(prior['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==prior['sha256'];trace=json.loads(raw);es=trace['traceEvents'];del raw
  rawrefs.append({k:prior[k] for k in ['path','sha256']})
  for row in prior['device_events']:
   if row['kind'] not in ['RS','AG']:continue
   e=es[row['event']];args=e['args'];n=int(args['Group size']);width={'Float':4,'BFloat16':2}[args['dtype']];ib=int(args['In msg nelems'])*width;ob=int(args['Out msg nelems'])*width
   assert (ib==n*ob if row['kind']=='RS' else ob==n*ib)
   full=ib if row['kind']=='RS' else ob
   inventory.append(dict(dataset=label,event=row['event'],kind=row['kind'],start_ms=row['start_ms'],end_ms=row['end_ms'],duration_ms=e['dur']/1000,stream=args['stream'],group=args['Process Group Description'],group_size=n,ranks_as_recorded=args['Process Group Ranks'],group_name=args['Process Group Name'],external_id=args['External id'],input_bytes=ib,output_bytes=ob,full_tensor_bytes=full,ring_per_rank_bytes=full*(n-1)/n,metadata=args))
  fs=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='forward_step'],key=lambda e:e['ts']);f=fs[0]
  steps=[e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep') and e['ts']<=f['ts']<e['ts']+e['dur']];assert len(steps)==1;step=steps[0]
  corrs={e['args']['correlation'] for e in es if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}) and e['pid']==f['pid'] and f['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=f['ts']+f['dur']+.01}
  dev=[e for e in es if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and e.get('args',{}).get('correlation') in corrs];assert dev
  first=min(e['ts'] for e in dev);startup=[dict(event=i,name=e['name'],start_from_step_ms=(e['ts']-step['ts'])/1000,duration_ms=e['dur']/1000,args=e.get('args',{})) for i,e in enumerate(es) if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and step['ts']<=e['ts']<first]
  starts.append(dict(dataset=label,step_to_first_F_GPU_ms=(first-step['ts'])/1000,step_to_first_F_CPU_ms=(f['ts']-step['ts'])/1000,device_events=startup,notice='rank0 window; events may straddle boundary; inventory not additive critical path'))
 topo=load(TOPO);ports=[e for e in topo['endpoints'] if e.get('active') and e.get('device_family')=='cx7' and e.get('rate_gbps',0)>=100];assert ports and all(e['rate_gbps']==400 for e in ports)
 eta=.6;gbps=400;effective=gbps*1e9/8*eta;scenarios=[]
 for group in ['DATA_PARALLEL_GROUP_WITH_CP','EXPERT_DATA_PARALLEL_GROUP']:
  for kind in ['RS','AG']:
   events=[e for e in inventory if e['dataset']=='target' and e['group']==group and e['kind']==kind];assert events
   assert len({e['full_tensor_bytes'] for e in events})==1
   e=events[0];n=e['group_size'];full=e['full_tensor_bytes'];flat=e['ring_per_rank_bytes']/effective*1000
   # Hypothesis: 2 hosts, 8 GPUs/NIC lanes each, node-local shard then two-host collective.
   cross_bytes=full/8/2 if n==16 else full/2
   scenarios.append(dict(group=group,kind=kind,device_event_count=len(events),full_tensor_bytes=full,cross_bytes_per_rank_hypothesis=cross_bytes,cross_service_ms_hypothesis=cross_bytes/effective*1000,flat_ring_all_bytes_at_cross_bandwidth_ms=flat,
    assumptions=['target uses same 400Gb/s links as source collection','one unshared effective NIC lane per rank at 60% utilization','DP16 hierarchical two-host/eight-shard algorithm; EDP2 interhost pair','kernel metadata total may be coalesced; do not multiply blindly by event count','no startup latency, local-phase cost, arrival wait or contention modeled']))
 report=dict(status='PARTIAL_METADATA_PASS_BANDWIDTH_SCENARIO',utilization=eta,source_nic_rate_gbps=gbps,effective_GBps=effective/1e9,source_active_cx7_ports=len(ports),inventory=inventory,startup=starts,scenarios=scenarios,
  limits=['EDP Process Group Ranks [0,1] may be group-local metadata; global placement unverified','No whole-tail prediction: intra-node rate, target NIC mapping and collective algorithm not yet independently verified','AG event multiplicity preserved, no unproven deduplication or sum','Startup CPU/launch/readiness work not scaled by bandwidth','Target-derived group/message metadata used for audit; not a blind source-only prediction'])
 save(a.out/'report.json',report);save(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in inputs},raw=rawrefs,code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps(dict(startup=[{k:v for k,v in s.items() if k!='device_events'} for s in starts],scenarios=scenarios),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

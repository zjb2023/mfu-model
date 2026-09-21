"""CP metadata decomposition and same-device coverage of B attribution gaps."""
import argparse,collections,hashlib,json
from pathlib import Path
from audit_b_operator_position_r1 import union
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def merged(xs):
 out=[]
 for a,b in sorted(xs):
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out
def clipped(es,a,b):return [(max(a,e['ts']),min(b,e['ts']+e['dur'])) for e in es if e['ts']<b and e['ts']+e.get('dur',0)>a]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
 allrows=[];inputs=[];raws=[]
 for w,s in [(224,5),(224,6),(256,1),(256,14)]:
  p=BASE/f'b-framework-position-r1/{w}-pp{s}.json';q=BASE/f'b-operator-position-r1/{w}-pp{s}.json';inputs += [p,q]
  fw=json.loads(p.read_text());prev=json.loads(q.read_text());raw=Path(prev['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==prev['sha256']
  es=json.loads(raw)['traceEvents'];del raw;raws.append({k:prev[k] for k in ['path','sha256']})
  runtime=[e for e in es if e.get('cat')=='privateuse1_runtime' and e.get('dur',0)>0]
  rows=[]
  for b,old in zip(fw['blocks'],prev['blocks']):
   ids={i for k in old['kernels'] for i in k['events']};owned=[es[i] for i in ids]
   devices={e.get('args',{}).get('device') for e in owned};assert len(devices)==1 and None not in devices
   device=next(iter(devices));allgpu=[e for e in es if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and e.get('args',{}).get('device')==device]
   spans=merged([(e['ts'],e['ts']+e['dur']) for e in owned]);gaps=[]
   for left,right in zip(spans,spans[1:]):
    a,z=left[1],right[0];busy=union(clipped(allgpu,a,z));calls=[e for e in runtime if e['ts']<z and e['ts']+e['dur']>a]
    names=collections.defaultdict(list)
    for e in calls:names[e['name']].append(e)
    gaps.append(dict(start_us=a,end_us=z,duration_ms=(z-a)/1000,other_device_coverage_ms=busy,no_recorded_device_ms=(z-a)/1000-busy,
      overlapping_runtime=[dict(name=n,overlap_ms=union(clipped(v,a,z))) for n,v in names.items()]))
   phases=[]
   for phase in ['recompute_attention_CP','backward_attention_CP']:
    indexes={i for op in b['operators'] if op['kind']==phase for i in op['device_events']};groups=collections.defaultdict(list)
    for i in indexes:
     e=es[i];a=e.get('args',{});name=e['name'].lower()
     if a.get('Process Group Description')=='CONTEXT_PARALLEL_GROUP':
      assert int(a['Group size'])==2;kind='CP2_communication'
     elif 'flash_atten' in name:kind='attention_flash_compute'
     elif 'copy' in name or e.get('cat') in ['gpu_memcpy','gpu_memset']:kind='local_copy_or_reorder'
     else:kind='other'
     groups[kind].append((i,e))
    assert len(groups['CP2_communication'])==(16 if phase.startswith('recompute') else 20)
    parts=[dict(kind=k,count=len(v),union_ms=union([(e['ts'],e['ts']+e['dur']) for _,e in v]),events=[i for i,_ in v]) for k,v in groups.items()]
    phases.append(dict(phase=phase,parts=parts))
   assert abs(sum(g['duration_ms'] for g in gaps)-old['no_owned_device_coverage_ms'])<1e-5
   row=dict(world=w,stage=s,mb=b['mb'],phases=phases,gap_ms=sum(g['duration_ms'] for g in gaps),other_device_coverage_ms=sum(g['other_device_coverage_ms'] for g in gaps),no_recorded_device_ms=sum(g['no_recorded_device_ms'] for g in gaps),gaps=sorted(gaps,key=lambda g:g['duration_ms'],reverse=True))
   rows.append(row);allrows.append(row)
  (args.out/f'{w}-pp{s}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print(w,s,'PASS',flush=True)
 report=dict(rows=[{k:v for k,v in r.items() if k!='gaps'} for r in allrows],limits=['Only iter60 four representative ranks, not all stages or EP8 peers','CP identified using process group metadata; CP kernel duration includes waiting, not pure bandwidth cost','No-recorded-device interval is a trace observation, not proof of physical GPU idle','Overlapping CPU runtime is correlation, not proof of causing gaps; runtime intervals can overlap','Per-category unions may overlap; not additive wall-time attribution; model unchanged'])
 (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 (args.out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in inputs},raw=raws,code={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('audit_b_operator_position_r1.py')]},outputs={str(p.resolve()):sha(p) for p in args.out.glob('*.json')}),indent=2)+'\n')
if __name__=='__main__':main()

"""Identify CPU parents of synchronization overlapping trace-visible GPU gaps."""
import argparse,collections,hashlib,json
from pathlib import Path
from audit_b_operator_position_r1 import union
from audit_b_cp_gaps_r1 import merged
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def intersections(xs,lo,hi):return [(max(a,lo),min(b,hi)) for a,b in xs if a<hi and b>lo]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 inputs=[];raws=[];reports=[]
 for stage in [1,14]:
  gp=BASE/f'b-cp-gaps-r1/256-pp{stage}.json';op=BASE/f'b-operator-position-r1/256-pp{stage}.json';fp=BASE/f'b-framework-position-r1/256-pp{stage}.json';inputs += [gp,op,fp]
  gaps=json.loads(gp.read_text())[1];meta=json.loads(op.read_text());framework=json.loads(fp.read_text())['blocks'][1]
  raw=Path(meta['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['sha256'];es=json.loads(raw)['traceEvents'];del raw
  raws.append({k:meta[k] for k in ['path','sha256']})
  ids={i for k in meta['blocks'][1]['kernels'] for i in k['events']};devs={es[i]['args']['device'] for i in ids};assert len(devs)==1
  gpu=[(i,e) for i,e in enumerate(es) if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and e.get('args',{}).get('device') in devs]
  occupied=merged([(e['ts'],e['ts']+e['dur']) for _,e in gpu]);idle=[]
  for g in gaps['gaps']:
   lo,hi=g['start_us'],g['end_us'];cursor=lo
   for x,y in intersections(occupied,lo,hi):
    if x>cursor:idle.append((cursor,x))
    cursor=max(cursor,y)
   if cursor<hi:idle.append((cursor,hi))
  assert abs(union(idle)-gaps['no_recorded_device_ms'])<1e-5
  cpu=[(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op'];calls=[];bykind=collections.defaultdict(list)
  for i,e in enumerate(es):
   if e.get('cat')!='privateuse1_runtime' or 'Synchronize' not in e.get('name',''):continue
   iv=intersections(idle,e['ts'],e['ts']+e['dur'])
   if not iv:continue
   owners=sorted([(j,c) for j,c in cpu if c['pid']==e['pid'] and c['tid']==e['tid'] and c['ts']<=e['ts'] and e['ts']+e['dur']<=c['ts']+c['dur']+.01],key=lambda v:v[1]['dur'])
   selected=[o for o in framework['operators'] if o['cpu_event'] in {j for j,_ in owners}]
   kind=selected[0]['kind'] if selected else ('other:'+owners[0][1]['name'] if owners else 'unattributed')
   bykind[kind]+=iv
   calls.append(dict(event=i,name=e['name'],duration_ms=e['dur']/1000,no_recorded_device_overlap_ms=union(iv),kind=kind,
     parents=[dict(event=j,name=c['name'],cpu_ms=c['dur']/1000) for j,c in owners],intervals_us=iv))
  covered=union([iv for c in calls for iv in c['intervals_us']]);total=union(idle)
  def eventrec(item):
   i,e=item;return dict(event=i,name=e['name'],stream=e.get('args',{}).get('stream'),start_us=e['ts'],end_us=e['ts']+e['dur'])
  largest=[]
  for lo,hi in sorted(idle,key=lambda x:x[1]-x[0],reverse=True)[:12]:
   before=max((x for x in gpu if x[1]['ts']+x[1]['dur']<=lo+.01),key=lambda x:x[1]['ts']+x[1]['dur'],default=None)
   after=min((x for x in gpu if x[1]['ts']>=hi-.01),key=lambda x:x[1]['ts'],default=None)
   largest.append(dict(start_us=lo,end_us=hi,duration_ms=(hi-lo)/1000,before=eventrec(before) if before else None,after=eventrec(after) if after else None,
      sync_events=[c['event'] for c in calls if intersections(c['intervals_us'],lo,hi)]))
  result=dict(stage=stage,world=256,mb=1,no_recorded_device_ms=total,sync_overlap_union_ms=covered,without_sync_overlap_ms=total-covered,
    groups=[dict(kind=k,overlap_ms=union(v)) for k,v in bykind.items()],calls=calls,largest_gaps=largest)
  (a.out/f'pp{stage}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');reports.append({k:v for k,v in result.items() if k not in ['calls','largest_gaps']});print(json.dumps(reports[-1],ensure_ascii=False),flush=True)
 (a.out/'report.json').write_text(json.dumps(dict(rows=reports,limits=['Same-thread CPU containment identifies call context, not physical cause','No-recorded-device intervals can include hidden transport','Overlaps with sync API do not prove host overhead or avoidable waits','Per-kind overlap unions can overlap each other; do not add blindly','Two rank traces B1 only, no independent prediction or full EP-group readiness verification']),ensure_ascii=False,indent=2)+'\n')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 (a.out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in inputs},raw=raws,code={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('audit_b_cp_gaps_r1.py'),Path(__file__).with_name('audit_b_operator_position_r1.py')]},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}),indent=2)+'\n')
if __name__=='__main__':main()

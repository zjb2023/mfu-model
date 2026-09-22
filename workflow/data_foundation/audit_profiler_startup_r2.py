"""Profiler boundary to first F CPU/GPU, with disjoint diagnostic subwindows."""
import argparse,hashlib,json
from pathlib import Path
from audit_b_operator_position_r1 import union
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);inp=BASE/'startup-scale-r1/report.json';refs=json.loads(inp.read_text())['rows'];rows=[]
 for ref in refs:
  raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];es=json.loads(raw)['traceEvents'];del raw
  f=min((e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='forward_step'),key=lambda e:e['ts']);step=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep') and e['ts']<=f['ts']<e['ts']+e['dur'])
  corrs={e['args']['correlation'] for e in es if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}) and e['pid']==f['pid'] and f['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=f['ts']+f['dur']+.01}
  gpu=[e for e in es if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']];owned=[e for e in gpu if e.get('args',{}).get('correlation') in corrs];assert owned;g=min(e['ts'] for e in owned);dev=owned[0]['args']['device'];ag=es[ref['event']]
  bounds=[step['ts'],ag['ts'],ag['ts']+ag['dur'],f['ts'],g];assert bounds==sorted(bounds)
  parts=[dict(name=n,ms=(y-x)/1000) for n,x,y in zip(['Profiler起点到统计候选AG开始','统计候选AG设备驻留','AG完成到首F CPU','首F CPU到首F GPU'],bounds,bounds[1:])]
  sync=[(max(step['ts'],e['ts']),min(g,e['ts']+e.get('dur',0))) for e in es if e.get('cat')=='privateuse1_runtime' and 'Synchronize' in e.get('name','') and e['ts']<g and e['ts']+e.get('dur',0)>step['ts']]
  busy=[(max(step['ts'],e['ts']),min(g,e['ts']+e['dur'])) for e in gpu if e.get('args',{}).get('device')==dev and e['ts']<g and e['ts']+e['dur']>step['ts']]
  total=(g-step['ts'])/1000;assert abs(sum(p['ms'] for p in parts)-total)<1e-5
  rows.append(dict(world=ref['world'],iteration=ref['iteration'],rank=0,path=ref['path'],sha256=ref['sha256'],profiler_label=step['name'],startup_GPU_ms=total,startup_CPU_ms=(f['ts']-step['ts'])/1000,parts=parts,device_union_ms=union(busy),no_recorded_device_ms=total-union(busy),runtime_sync_union_ms=union(sync),notice='sync runtime and device intervals overlap; do not add them'))
  print(ref['world'],ref['iteration'],'PASS',flush=True)
 report=dict(status='PARTIAL_PROFILER_STARTUP_BOUNDARY_AUDIT',definition='ProfilerStep start to first PP0 F-owned GPU; CPU start separately retained',rows=rows,limits=['AG used as diagnostic landmark only, not definition of startup','Statistics/timer origin unproven without actual source stack','Two sizes from separate collection runs do not identify world-size causal scaling','No cross-machine clock comparison; no last-arriving rank inferred','No fitting of target startup or update to prediction'])
 save(a.out/'report.json',report);save(a.out/'manifest.json',dict(inputs={str(inp):sha(inp)},raw=[{k:r[k] for k in ['path','sha256']} for r in refs],code={str(Path(__file__).resolve()):sha(Path(__file__)),str(Path(__file__).with_name('audit_b_operator_position_r1.py')):sha(Path(__file__).with_name('audit_b_operator_position_r1.py'))},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps([{k:v for k,v in r.items() if k not in ['path','sha256']} for r in rows],ensure_ascii=False),flush=True)
if __name__=='__main__':main()

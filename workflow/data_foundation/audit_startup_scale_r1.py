"""Bounded rank0 startup comparisons at 32/256 GPUs, iterations40/60/80."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
ROOTS={32:Path('/home/zjb/Desktop/32/gpu32_gbs64_framework/2026-09-18-10:38/worker34095/profiler'),256:Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40/worker33087/profiler')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
 for world,root in ROOTS.items():
  for iteration in [40,60,80]:
   paths=list((root/f'iteration_{iteration}').glob('rank0.*.pt.trace.json'));assert len(paths)==1;p=paths[0];raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();d=json.loads(raw);del raw;es=d['traceEvents']
   assert d['distributedInfo']['world_size']==world and d['distributedInfo']['rank']==0
   f=min((e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='forward_step'),key=lambda e:e['ts']);step=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep') and e['ts']<=f['ts']<e['ts']+e['dur'])
   ag=[(i,e) for i,e in enumerate(es) if e.get('cat')=='kernel' and e.get('args',{}).get('Collective name')=='_allgather_base' and step['ts']<=e['ts']<f['ts']];assert len(ag)==1;i,e=ag[0];args=e['args'];assert args['dtype']=='Float' and int(args['Group size'])==world
   rows.append(dict(world=world,iteration=iteration,rank=0,path=str(p),sha256=digest,event=i,profiler_label=step['name'],first_F_CPU_ms=(f['ts']-step['ts'])/1000,AG_start_ms=(e['ts']-step['ts'])/1000,AG_duration_ms=e['dur']/1000,input_bytes=int(args['In msg nelems'])*4,output_bytes=int(args['Out msg nelems'])*4))
 report=dict(status='PARTIAL_SCALE_ASSOCIATION_NOT_CAUSAL_MODEL',rows=rows,limits=['Two sizes from different collection runs, not controlled rank-scaling experiment','rank0 only; local clocks and profiler boundaries cannot identify last arriving rank','Small-message global collective duration includes progress/arrival/latency; not pure bandwidth','Timer statistics origin is code-shape hypothesis, not trace stack proof'])
 save(a.out/'report.json',report);save(a.out/'manifest.json',dict(raw=[{k:r[k] for k in ['path','sha256']} for r in rows],code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps([{k:v for k,v in r.items() if k not in ['path','sha256']} for r in rows],ensure_ascii=False),flush=True)
if __name__=='__main__':main()

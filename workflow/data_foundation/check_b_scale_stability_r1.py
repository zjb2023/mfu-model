"""Read-only cross-iteration CPU inventory plus nine bounded GPU trace checks."""
import argparse,collections,csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];FAB=Path('/home/zjb/Desktop/fabric-data-analysis')
FILES={256:FAB/'case_256gpu_pp16_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_v53_ep_group_source60_100/source_pp_trace_events_60_100.csv',224:FAB/'case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_v54_256_to_224_extrapolation/evaluator_only/target_lane0_pp_events_60_100.csv'}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def extract(p,world,rank):
    raw=p.read_bytes();d=json.loads(raw);es=d['traceEvents'];assert d['distributedInfo']['rank']==rank and d['distributedInfo']['world_size']==world
    step=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep'))
    phases=[(i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name')=='backward_step' and step['ts']<=e['ts']<step['ts']+step['dur']]
    phases.sort(key=lambda x:x[1]['ts']);runtime=collections.defaultdict(list);owned=collections.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat')=='privateuse1_runtime' and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append(e)
    for i,e in enumerate(es):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        rs=runtime.get(e.get('args',{}).get('correlation'),[])
        if len(rs)!=1:continue
        r=rs[0];hits=[j for j,a in phases if r['pid']==a['pid'] and a['ts']<=r['ts'] and r['ts']+r.get('dur',0)<=a['ts']+a['dur']+.01]
        if len(hits)==1:owned[hits[0]].append((i,e))
    result=[]
    for m,(j,e) in enumerate(phases):
        ds=owned[j];assert ds
        first=min(ds,key=lambda x:x[1]['ts']);last=max(ds,key=lambda x:x[1]['ts']+x[1]['dur'])
        result.append(dict(mb=m,cpu_event=j,first_gpu_event=first[0],last_gpu_event=last[0],gpu_ms=(last[1]['ts']+last[1]['dur']-first[1]['ts'])/1000,cpu_ms=e['dur']/1000))
    assert len(result)==(3 if world==224 else 4)
    return dict(path=str(p),resolved_path=str(p.resolve()),sha256=hashlib.sha256(raw).hexdigest(),blocks=result,gpu_mean_ms=statistics.mean(x['gpu_ms'] for x in result),cpu_mean_ms=statistics.mean(x['cpu_ms'] for x in result))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out;out.mkdir(parents=True,exist_ok=False)
    allrows={w:list(csv.DictReader(p.open())) for w,p in FILES.items()};cpu=[];stage_rows=[]
    for w,rows in allrows.items():
        for it in range(60,101,5):
            means=[]
            for st in range(1,w//16-1):
                rs=[r for r in rows if int(r['iteration'])==it and int(r['pp_lane'])==0 and int(r['pp_stage'])==st and r['phase']=='backward'];assert len(rs)==(3 if w==224 else 4)
                mean=statistics.mean(float(r['duration_ns'])/1e6 for r in rs);means.append(mean);stage_rows.append(dict(world=w,iteration=it,stage=st,cpu_mean_ms=mean))
            cpu.append(dict(world=w,iteration=it,cpu_mean_ms=statistics.mean(means),stage_min_ms=min(means),stage_max_ms=max(means)))
    checks=[]
    # Predetermined small samples: early/middle/late stage, beginning/end iterations.
    for w,its,stages in [(224,[60,100],[1,6,12]),(256,[100],[1,7,14])]:
        for it in its:
            for st in stages:
                r=next(r for r in allrows[w] if int(r['iteration'])==it and int(r['pp_stage'])==st and int(r['pp_lane'])==0 and r['phase']=='backward')
                x=extract(Path(r['source_path']),w,int(r['rank']));x.update(world=w,iteration=it,stage=st,rank=int(r['rank']));checks.append(x);dump(out/f'gpu-{w}-{it}-pp{st}.json',x);print(w,it,st,round(x['gpu_mean_ms'],3),flush=True)
    report=dict(cpu_iteration_summary=cpu,cpu_stage_means=stage_rows,gpu_checks=[{k:v for k,v in x.items() if k!='blocks'} for x in checks],scope='CPU lane0 9 iterations each world; GPU nine raw rank traces only',model_changed=False,warning='CPU duration is not GPU envelope; no numerical cross-basis comparison; no claim of all-iteration GPU stability')
    dump(out/'report.json',report)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    dump(out/'manifest.json',dict(inputs={str(p):sha(p) for p in FILES.values()},raw=[{k:x[k] for k in ['path','resolved_path','sha256']} for x in checks],code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in out.glob('*.json')}))
if __name__=='__main__':main()

"""Bounded rank0 source32/target256 tail facts; do not infer pure service costs."""
import argparse,collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';OLD=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def union(xs):
    merged=[]
    for a,b in sorted(xs):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)
def extract(world,meta,b_end):
    p=Path(meta['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==meta['sha256'];es=json.loads(raw)['traceEvents']
    step=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep'));end=(step['ts']+step['dur'])/1000
    runtime=collections.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat')=='privateuse1_runtime' and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append((i,e))
    anns=[(i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and (e['ts']+e.get('dur',0))/1000>=b_end and e['ts']/1000<=end and not e['name'].startswith('ProfilerStep')]
    records=[]
    for i,e in enumerate(es):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        start=e['ts']/1000;stop=(e['ts']+e['dur'])/1000
        if stop<=b_end or start>=end:continue
        rs=runtime.get(e.get('args',{}).get('correlation'),[]);owners=[]
        if len(rs)==1:
            ri,r=rs[0];owners=[(j,a['name']) for j,a in anns if a['pid']==r['pid'] and a['ts']<=r['ts'] and r['ts']+r.get('dur',0)<=a['ts']+a['dur']+.01]
        names=[name for _,name in owners]
        if 'ReduceScatter' in e['name']:kind='RS'
        elif 'AllGather' in e['name']:kind='AG'
        elif 'AllReduce' in e['name']:kind='AllReduce_sync_or_stats'
        elif any('Optimizer.step#' in name for name in names):kind='optimizer_associated_GPU'
        elif any(name.startswith(('get_grad_norm','clip_grad','prepare_grads')) for name in names):kind='gradient_processing_GPU'
        elif e['cat']!='kernel':kind='copy_or_memset'
        else:kind='other_GPU'
        records.append(dict(event=i,name=e['name'],kind=kind,start_ms=max(start,b_end)-b_end,end_ms=min(stop,end)-b_end,unclipped_start_ms=start-b_end,owners=owners,runtime_event=rs[0][0] if len(rs)==1 else None))
    stats=[]
    for kind in sorted({r['kind'] for r in records}):
        rs=[r for r in records if r['kind']==kind];stats.append(dict(kind=kind,count=len(rs),union_ms=union([(r['start_ms'],r['end_ms']) for r in rs]),first_ms=min(r['start_ms'] for r in rs),last_ms=max(r['end_ms'] for r in rs)))
    get=lambda k:[r for r in records if r['kind']==k]
    rs_end=max(r['end_ms'] for r in get('RS'));ag_end=max(r['end_ms'] for r in get('AG'));opt_start=min(r['start_ms'] for r in get('optimizer_associated_GPU'))
    assert 0<rs_end<=opt_start<=ag_end<=end-b_end
    windows=[dict(name='末B→最后RS完成',start_ms=0,end_ms=rs_end),dict(name='最后RS→首个优化器关联GPU',start_ms=rs_end,end_ms=opt_start),dict(name='首个优化器关联GPU→最后AG完成',start_ms=opt_start,end_ms=ag_end),dict(name='最后AG→ProfilerStep结束',start_ms=ag_end,end_ms=end-b_end)]
    for w in windows:w['duration_ms']=w['end_ms']-w['start_ms']
    active=union([(r['start_ms'],r['end_ms']) for r in records]);assert abs(sum(w['duration_ms'] for w in windows)-(end-b_end))<1e-6
    return dict(world=world,rank=0,iteration=60,path=str(p),resolved_path=str(p.resolve()),sha256=digest,tail_ms=end-b_end,windows=windows,activity_stats=stats,device_union_ms=active,no_device_event_coverage_ms=end-b_end-active,device_events=records,cpu_annotations=[dict(event=i,name=e['name'],start_ms=e['ts']/1000-b_end,end_ms=(e['ts']+e.get('dur',0))/1000-b_end) for i,e in anns],policy='window landmarks are observed brackets, not serial physical components; per-class unions may overlap; owner association conditional')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out;out.mkdir(parents=True,exist_ok=False)
    sp=OLD/'pp32-structure-r4/rank-0.json';tp=OLD/'pp256-stage-audit-r7/rank-0.json';obs=BASE/'pp32-gpu-boundary-r3b/source-observations.json';s=next(x for x in load(obs) if x['rank']==0);b_end=[o for o in s['ops'] if o['kind']=='B'][-1]['absolute_end_ms'];t=load(tp)
    source=extract(32,load(sp),b_end);target=extract(256,t,t['blocks'][-1]['end_absolute_ms'])
    rows=[dict(name=a['name'],source32_ms=a['duration_ms'],target256_ms=b['duration_ms'],target_minus_source_ms=b['duration_ms']-a['duration_ms']) for a,b in zip(source['windows'],target['windows'])]
    report=dict(status='PARTIAL_RANK0_TAIL_FACTS',source_tail_ms=source['tail_ms'],target_tail_ms=target['tail_ms'],underestimate_ms=target['tail_ms']-source['tail_ms'],windows=rows,limits=['one iteration,rank0 only; no full-stage optimizer barrier inferred','DP/EDP group identities not yet attributed per kernel','GPU completion union includes wait inside collective kernel, not pure service','no-device-event coverage is not automatically pure CPU overhead','1F1B and frozen B analysis unchanged; no new fitted tail model'])
    for name,obj in [('source',source),('target',target),('report',report)]:dump(out/(name+'.json'),obj)
    dump(out/'manifest.json',dict(inputs={str(p):sha(p) for p in [sp,tp,obs]},raw=[{k:x[k] for k in ['path','resolved_path','sha256']} for x in [source,target]],code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in out.glob('*.json')}))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

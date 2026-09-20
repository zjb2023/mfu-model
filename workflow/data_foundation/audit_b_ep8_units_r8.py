"""Audit EP8 completion structure across four units; not predictive evaluation."""
import argparse,collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    src=BASE/'b-cost-blocks-r1/evidence.json';rows=[];inputs=[]
    for x in json.loads(src.read_text()):
        p=Path(x['path']);blob=p.read_bytes();assert hashlib.sha256(blob).hexdigest()==x['sha256'];data=json.loads(blob);es=data['traceEvents'];del blob
        inputs.append(dict(path=str(p),sha256=x['sha256']))
        def inside(e,o):return e.get('pid')==o['pid'] and e.get('tid')==o['tid'] and o['ts']<=e.get('ts',0) and e.get('ts',0)+e.get('dur',0)<=o['ts']+o['dur']+.01
        device_by_corr=collections.defaultdict(list)
        for i,e in enumerate(es):
            if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']:device_by_corr[e.get('args',{}).get('correlation')].append((i,e))
        calls=[(i,e) for i,e in enumerate(es) if e.get('cat') in ['privateuse1_runtime','privateuse1_driver']]
        def owned(o):
            out={}
            for _,e in calls:
                if inside(e,o):
                    for i,d in device_by_corr.get(e.get('args',{}).get('correlation'),[]):out[i]=d
            return sorted(out.items())
        for unit,ci in enumerate(x['checkpoints']):
            ck=es[ci];owners=[(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and inside(e,ck)]
            dispatch=next(e for _,e in owners if e['name']=='FusedCombineBackward')
            oi,combine=next((i,e) for i,e in owners if e['name']=='FusedDispatchBackward')
            experts=[(i,e) for i,e in owners if e['name']=='_GroupedLinearBackward' and dispatch['ts']<=e['ts']<combine['ts']];assert len(experts)==2
            ds=owned(combine);visible=[(i,e) for i,e in ds if 'moe_unpermute' in e['name']];assert len(visible)==1
            vi,v=visible[0];prep=[(i,e) for i,e in ds if e.get('args',{}).get('stream')!=v['args']['stream']];assert prep
            expert_ds=[d for _,e in experts for _,d in owned(e)];assert expert_ds
            ready=max(e['ts']+e['dur'] for _,e in prep);expert_end=max(e['ts']+e['dur'] for e in expert_ds)
            rows.append(dict(rank=x['rank'],unit=unit,combine_cpu_event=oi,prep_device_events=[i for i,_ in prep],visible_event=vi,visible_name=v['name'],expert_cpu_events=[i for i,_ in experts],expert_end_us=expert_end,ready_us=ready,visible_start_us=v['ts'],visible_end_us=v['ts']+v['dur'],local_prepare_after_expert_ms=(ready-expert_end)/1000))
        print('PASS four-unit evidence rank',x['rank'],flush=True)
    units=[];ref=None
    for unit in range(4):
        rs=[r for r in rows if r['unit']==unit];assert len(rs)==8
        latest=max(r['ready_us'] for r in rs);first=min(r['visible_start_us'] for r in rs);residual=(first-latest)/1000
        if unit==0:ref=residual
        offsets={r['rank']:(r['visible_start_us']-first)/1000 for r in rs}
        units.append(dict(unit=unit,structure_compatible=residual>=0 and all(r['ready_us']>=r['expert_end_us'] for r in rs),latest_ready_rank=max(rs,key=lambda r:r['ready_us'])['rank'],effective_residual_ms=residual,visible_start_spread_ms=(max(r['visible_start_us'] for r in rs)-first)/1000,visible_end_spread_ms=(max(r['visible_end_us'] for r in rs)-min(r['visible_end_us'] for r in rs))/1000,unit0_residual_only_error_ms=ref-residual,rank_start_offsets_ms=offsets))
    summary=dict(status='PASS_STRUCTURE_AUDIT' if all(u['structure_compatible'] for u in units) else 'PARTIAL_STRUCTURE_CONTRADICTION',units=units,limits=['All-ready completion coupling remains a hypothesis; positive residual does not prove barrier.','Each unit uses its own measured readiness: residual transfer is an oracle diagnostic, NOT source-only prediction.','Checkpoint execution order is not verified forward layer index.','No iter70/256 input; no four-layer B prediction completed.'])
    a.out.mkdir(parents=True)
    for n,v in [('evidence',rows),('summary',summary)]: (a.out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs.extend(rec(p) for p in [src,Path(__file__).resolve()])
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,outputs=[rec(p) for p in a.out.iterdir()]),indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()

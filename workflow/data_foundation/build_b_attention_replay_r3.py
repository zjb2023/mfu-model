"""Conditional local Attention-backward replay; no global synchronization claim."""
import argparse, collections, graphlib, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'results/data-foundation'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    prior=BASE/'b-cost-blocks-r1/evidence.json'; struct=BASE/'b-attention-structure-r2/evidence.json';ids=BASE/'b-cp-identity-r1/events.json'
    units=json.loads(struct.read_text()); identities=json.loads(ids.read_text());graphs={};audit=[];inputs=[]
    for x in json.loads(prior.read_text()):
        raw=Path(x['path']);blob=raw.read_bytes();assert hashlib.sha256(blob).hexdigest()==x['sha256'];es=json.loads(blob)['traceEvents'];del blob
        inputs.append(dict(path=str(raw),sha256=x['sha256']))
        for u in [u for u in units if u['rank']==x['rank']]:
            identity=next(r for r in identities if r['rank']==x['rank'] and r['checkpoint_unit']==u['unit'] and r['phase']=='backward')
            owner=es[identity['cpu_owner_event']]
            def inside(e):return e.get('pid')==owner['pid'] and e.get('tid')==owner['tid'] and owner['ts']<=e.get('ts',0) and e.get('ts',0)+e.get('dur',0)<=owner['ts']+owner['dur']+.01
            calls=[(i,e) for i,e in enumerate(es) if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and inside(e)]
            corr={e.get('args',{}).get('correlation') for _,e in calls}-{None}
            devices=[(i,e) for i,e in enumerate(es) if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and e.get('args',{}).get('correlation') in corr]
            flashids={d['event'] for d in u['flash_device_events']}; cpids={c['device_event']:c['ordinal'] for c in u['cp']}
            origin=min(e['ts'] for _,e in devices); nodes={};streams=collections.defaultdict(list)
            for i,e in devices:streams[e['args'].get('stream',e['tid'])].append((i,e))
            for stream,ds in streams.items():
                groups=[]
                for i,e in sorted(ds,key=lambda z:z[1]['ts']):
                    kind=f'CP{cpids[i]}' if i in cpids else ('attention_backward' if i in flashids else ('index_select_reorder' if 'index_select' in e['name'] else 'local_processing'))
                    if groups and groups[-1][0]==kind:groups[-1][1].append((i,e))
                    else:groups.append((kind,[(i,e)]))
                prev=None
                for j,(kind,items) in enumerate(groups):
                    k=f's{stream}:{j}';lo=min(e['ts'] for _,e in items);hi=max(e['ts']+e['dur'] for _,e in items)
                    nodes[k]=dict(kind=kind,stream=stream,events=[i for i,_ in items],names=sorted({e['name'] for _,e in items}),start_ms=(lo-origin)/1000,end_ms=(hi-origin)/1000,cost_ms=(hi-lo)/1000,deps=[prev] if prev else [],edge_basis={prev:'same_stream'} if prev else {},policy='external_effective_replace' if kind.startswith('CP') or kind=='attention_backward' else 'trace_reserved')
                    prev=k
            flash=next(k for k,n in nodes.items() if n['kind']=='attention_backward')
            cp={n['kind']:k for k,n in nodes.items() if n['kind'].startswith('CP')}
            def candidate(src,dst):
                assert nodes[src]['end_ms']<=nodes[dst]['start_ms']+1e-5
                if src not in nodes[dst]['deps']:nodes[dst]['deps'].append(src);nodes[dst]['edge_basis'][src]='conditional_structure_not_event_handle_verified'
            candidate(cp['CP1'],flash);candidate(flash,cp['CP2'])
            reorders=[k for k,n in nodes.items() if n['kind']=='index_select_reorder' and n['end_ms']<=nodes[flash]['start_ms']+1e-5]
            if reorders:candidate(max(reorders,key=lambda k:nodes[k]['end_ms']),flash)
            order=list(graphlib.TopologicalSorter({k:n['deps'] for k,n in nodes.items()}).static_order())
            replay={}
            for k in order:
                n=nodes[k];ready=max([replay[d] for d in n['deps']] or [0]);n['trace_ready_ms']=max(0,n['start_ms']-ready)
                assert n['start_ms']>=ready-1e-5
                replay[k]=ready+n['trace_ready_ms']+n['cost_ms'];assert abs(replay[k]-n['end_ms'])<1e-5
            # Sensitivity is conditional, not predictive validation. No duration counted twice.
            altered={}
            for k in order:
                n=nodes[k];altered[k]=max([altered[d] for d in n['deps']] or [0])+n['trace_ready_ms']+n['cost_ms']+(1 if k==flash else 0)
            assert altered[cp['CP2']]-replay[cp['CP2']]>=1-1e-7
            lo=u['cp'][1]['end_us'];hi=min(d['start_us'] for d in u['flash_device_events'])
            spans=sorted((max(lo,e['ts']),min(hi,e['ts']+e['dur'])) for _,e in devices if 'index_select' in e['name'] and e['ts']<hi and e['ts']+e['dur']>lo)
            merged=[]
            for s,t in spans:
                if merged and s<=merged[-1][1]:merged[-1][1]=max(t,merged[-1][1])
                else:merged.append([s,t])
            waits=[dict(event=i,name=e['name'],args=e.get('args',{})) for i,e in calls if 'StreamWaitEvent' in e['name']]
            key=f'r{x["rank"]}:u{u["unit"]}';graphs[key]=dict(origin_us=origin,nodes=nodes,replay_end_ms=replay,attention_plus1_end_ms=altered)
            audit.append(dict(key=key,cp_group=u['cp_group'],blocks=len(nodes),CP1_to_flash_ms=(hi-lo)/1000,index_select_exposed_ms=sum(t-s for s,t in merged)/1000,wait_calls=waits,window_ms=max(replay.values())))
        print('PASS local replay rank',x['rank'],flush=True)
    summary=dict(status='PARTIAL_CONDITIONAL_LOCAL_DAG',units=len(audit),replay_checks='PASS all block boundaries',sensitivity_checks='PASS attention +1 propagates to CP2',limitations=['Cross-stream candidate edges lack event-handle pairing.','trace_ready parameters can encode missing dependencies; locked for this version.','No CP pair/global EP8 readiness or B-wide integration.','Same-sample reconstruction, not independent prediction.'])
    a.out.mkdir(parents=True)
    for n,v in [('graphs',graphs),('audit',audit),('summary',summary)]: (a.out/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    def record(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs.extend(record(p) for p in [prior,struct,ids,Path(__file__).resolve()])
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,outputs=[record(p) for p in a.out.iterdir()]),indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()

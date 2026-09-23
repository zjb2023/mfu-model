"""Read-only, full EP8 source/destination assignment conservation at iter60."""
import collections
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
OUT=BASE/'layer-filtering-r1'
SOURCE=BASE/'layer-workload-256-r1'
SCOPE='256 iter60 first EP8 replica, all middle stages and four microbatches'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    provenance={}; grouped=collections.defaultdict(list)
    def process(p):
        cached=OUT/p.name
        if cached.exists():
            old=json.loads(cached.read_text())
            provenance[old['trace']]=old['sha256']
            for r in old['records']:
                grouped[r['layer'],r['mb']].append(r)
                for key in ('send_log','recv_log'):
                    lp=Path(r[key]['path'])
                    if str(lp) not in provenance:provenance[str(lp)]=digest(lp)
            print('CACHED',p.name,flush=True)
            return
        d=json.loads(p.read_text()); raw=Path(d['path']); buf=raw.read_bytes()
        assert hashlib.sha256(buf).hexdigest()==d['sha256']
        trace=json.loads(buf); del buf
        es=trace['traceEvents']; base=trace['baseTimeNanoseconds']
        evidence=[(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e['name'] in ('aten::topk','aten::logical_and')]
        cps=[es[r['checkpoint_forward_event']] for r in d['rows']]
        lower=base+min(c['ts'] for c in cps)*1000;upper=base+max(c['ts']+c['dur'] for c in cps)*1000
        logs=[]
        for lp in sorted(raw.parents[2].glob(f'deepep_trace/*/rank_{d["rank"]}.log')):
            # Preserve exact file/line provenance, never infer training iteration from log iter counter.
            hit=False
            for lineno,line in enumerate(lp.open(),1):
                e=json.loads(line)
                if lower<=e.get('timestamp_ns',0)<=upper and e.get('event') in ('dispatch_layout','after_dispatch'):
                    logs.append(dict(path=str(lp),line=lineno,record=e));hit=True
            if hit:provenance[str(lp)]=digest(lp)
        records=[]
        for r,c in zip(d['rows'],cps):
            ev=[dict(event=i,name=e['name'],args=e.get('args',{})) for i,e in evidence if e['pid']==c['pid'] and e['tid']==c['tid'] and c['ts']<=e['ts']<c['ts']+c['dur']]
            tops=[e for e in ev if e['name']=='aten::topk' and e['args'].get('Input Dims',[None])[0]==[8192,160]]
            route=next(e for e in tops if e['args']['Concrete Inputs'][1:3]==['6','1'])
            cap=next(e for e in tops if e['args']['Concrete Inputs'][1:3]==['615','0'])
            assert any(e['name']=='aten::logical_and' for e in ev)
            lo=base+c['ts']*1000;hi=lo+c['dur']*1000
            hits=[e for e in logs if lo<=e['record']['timestamp_ns']<=hi]
            sends=[e for e in hits if e['record']['event']=='dispatch_layout'];recvs=[e for e in hits if e['record']['event']=='after_dispatch']
            assert len(sends)==len(recvs)==1,(p,r['layer'],r['mb'],len(sends),len(recvs))
            send,recv=sends[0],recvs[0]
            assert send['record']['iter']==recv['record']['iter']
            assert recv['record']['num_recv_tokens_per_expert_list']==r['tokens_per_expert']
            counts=send['record']['num_tokens_per_expert'];assert len(counts)==160 and all(0<=x<=615 for x in counts)
            assert send['record']['x_shape']==[8192,5120]
            entry=dict(layer=r['layer'],mb=r['mb'],stage=r['stage'],rank=r['rank'],ep_group=r['ep_group'],
                       offered_assignments=8192*6,post_capacity_send_counts=counts,received_counts=r['tokens_per_expert'],
                       send_log=send,recv_log=recv,route_event=route,capacity_event=cap,checkpoint_event=r['checkpoint_forward_event'])
            records.append(entry);grouped[r['layer'],r['mb']].append(entry)
        (OUT/p.name).write_text(json.dumps(dict(trace=str(raw),sha256=d['sha256'],records=records),separators=(',',':'))+'\n')
        provenance[str(raw)]=d['sha256']
        print('PASS',d['stage'],d['rank'],len(records),'windows',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(process,sorted(SOURCE.glob('w256-i60-*.json'))))
    summary=[];checks=0
    for (layer,mb),rs in sorted(grouped.items()):
        rs.sort(key=lambda r:r['rank']); assert len(rs)==8
        assert [r['rank'] for r in rs]==rs[0]['ep_group']
        outgoing=[sum(r['post_capacity_send_counts'][e] for r in rs) for e in range(160)]
        incoming=[n for r in rs for n in r['received_counts']]
        assert outgoing==incoming,(layer,mb);checks+=160
        offered=sum(r['offered_assignments'] for r in rs);retained=sum(outgoing)
        summary.append(dict(layer=layer,mb=mb,stage=rs[0]['stage'],ranks=[r['rank'] for r in rs],offered_assignments=offered,
                            retained_assignments=retained,inferred_removed_assignments=offered-retained,retention_fraction=retained/offered,
                            capped_sender_expert_pairs=sum(n==615 for r in rs for n in r['post_capacity_send_counts']),
                            nonzero_experts=sum(n>0 for n in outgoing)))
    report=dict(scope=SCOPE,status='PASS',
                method='Offered assignments derived from trace top6 input dimensions, not directly logged pre-capacity perexpert distribution. Retained counts independently matched sender logs -> receiver logs -> trace FC counts. Removed is semantic/accounting inference, not direct pre/post mask measurement. Reference DeepEP metadata additionally masks zero probabilities: cannot distinguish capacity masking from preexisting zero probabilities with these counts.',
                expert_conservation_checks=checks,rows=summary)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    provenance[str(Path(__file__).resolve())]=digest(Path(__file__))
    (OUT/'manifest.json').write_text(json.dumps(dict(inputs=provenance,outputs={p.name:digest(p) for p in OUT.iterdir() if p.is_file()}),indent=2)+'\n')
    print('COMPLETE',len(summary),'layer/MB groups',checks,'perexpert conservation checks')

if __name__=='__main__':main()

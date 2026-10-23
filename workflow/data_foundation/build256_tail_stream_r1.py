"""Target tail parameters on measured same-stream edges, explicitly conditional.

Does not claim event-handle or all-rank causality; keeps reserved stream-ready
costs visible. No measured scalar tail is added to the graph.
"""
import collections,json
from pathlib import Path
from calibrate256_fine_r1 import BASE,load,save,sha
from build_b_cp_pair_r4 import run
def main():
    out=BASE/'iter60-tail-stream-r1';out.mkdir(exist_ok=False)
    fact=BASE/'tail-source155-r1/target-60.json';x=load(fact);raw=Path(x['path']);assert sha(raw)==x['sha256'];data=load(raw);es=data['traceEvents'];groups={str(g['pg_name']):g for g in data['distributedInfo']['pg_config']}
    g={'ENTRY':dict(deps=[],cost_ms=0,kind='last_B_exit')};evidence=[];streams=collections.defaultdict(list)
    for d in x['device_events']:
        e=es[d['event']];args=e.get('args',{});key=(args.get('device'),args.get('context'),args.get('stream',e.get('tid')));kind=d['kind'];pg=None
        if kind in ['RS','AG','AllReduce_sync_or_stats']:
            pg=groups[str(args['Process Group Name'])];assert pg['pg_size']==args['Group size']
            if kind in ['RS','AG']:kind=('EDP_' if pg['pg_desc']=='EXPERT_DATA_PARALLEL_GROUP' else 'Dense_')+kind
            else:kind='sync_or_statistics_'+pg['pg_desc']
        streams[key].append(dict(d,kind=kind,group=pg,stream_identity=key))
    leaves=[];conflicts=[];expected={}
    for stream,ds in sorted(streams.items()):
        prev='ENTRY';cursor=0
        for d in sorted(ds,key=lambda v:v['start_ms']):
            k='event:'+str(d['event']);ready=k+':ready';gap=d['start_ms']-cursor
            if gap < -1e-5:conflicts.append(dict(event=d['event'],gap_ms=gap,stream=stream));continue
            g[ready]=dict(deps=[prev],cost_ms=max(0,gap),kind='reserved_stream_readiness',basis='measured interval, not pure idle; cross-stream causes not independently identified')
            g[k]=dict(deps=[ready],cost_ms=d['end_ms']-d['start_ms'],kind=d['kind'],policy='effective_replacement' if d['group'] or d['kind']=='optimizer_associated_GPU' else 'trace_reserved',event=d['event'],stream=stream)
            evidence.append(dict(node=k,**d));expected[k]=d['end_ms'];prev=k;cursor=d['end_ms']
        leaves.append(prev)
    assert not conflicts,conflicts
    g['END']=dict(deps=leaves,cost_ms=x['tail_ms']-max(expected.values()),kind='profiler_end_reserved')
    t=run(g);err=max(abs(t[k]['end_ms']-v) for k,v in expected.items());assert err<1e-5 and abs(t['END']['end_ms']-x['tail_ms'])<1e-5
    stats=collections.defaultdict(lambda:dict(events=0,sum_device_ms=0))
    for e in evidence:stats[e['kind']]['events']+=1;stats[e['kind']]['sum_device_ms']+=e['end_ms']-e['start_ms']
    report=dict(status='PASS_LOCAL_STREAM_CONDITIONAL_ACCOUNTING',tail_ms=t['END']['end_ms'],nodes=len(g),device_events=len(evidence),streams=len(streams),max_event_error_ms=err,activities=dict(stats),limits=['No tail scalar addition; END retains only post-last-device residual.','Per-stream order is explicit; cross-stream/event-handle and other-stage readiness remain reserved parameters.','Collective durations include waiting, not pure network transfer.','Activity sums can overlap; never sum them as total tail.', 'World group members use distributedInfo metadata, not kernel-local group index strings.'])
    for name,obj in [('graph',g),('replay',t),('evidence',evidence),('report',report)]:save(out/(name+'.json'),obj)
    save(out/'manifest.json',dict(inputs={str(fact):sha(fact),str(raw):sha(raw)},code_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()

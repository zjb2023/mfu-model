"""Target-calibrated same-iteration outer DAG replay, never a prediction claim.
Uses the current frozen outer graph, real target F/B envelopes and paired PP
effective costs. Tail retains explicit observed event/ready nodes, not a scalar.
"""
import argparse,collections,copy,graphlib,hashlib,json,statistics
from pathlib import Path
import orjson
from pp32_rendezvous_r2 import message_ids
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
CACHE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/pp256-stage-audit-r7')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def run(g):
    t={}
    for k in graphlib.TopologicalSorter({k:n['dependencies'] for k,n in g.items()}).static_order():
        s=max((t[d]['end_ms'] for d in g[k]['dependencies']),default=0)
        t[k]=dict(start_ms=s,end_ms=s+g[k]['duration_ms'])
    return t
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out;out.mkdir(parents=True,exist_ok=False)
    inputs={};messages=collections.defaultdict(list);blocks={};rawrefs=[]
    for stage in range(16):
        cpath=CACHE/f'rank-{stage*16}.json';c=load(cpath);inputs[str(cpath)]=sha(cpath)
        p=Path(c['path']);raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==c['sha256']
        es=orjson.loads(raw)['traceEvents'];del raw
        steps=[e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep')];assert len(steps)==1
        step=steps[0];origin=step['ts']/1000
        for b in c['blocks']:blocks[f's{stage}:{b["phase"]}{b["mb"]}']=b
        calls=[(i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name') in ['send_forward','recv_forward','send_backward','recv_backward','send_forward_recv_backward','send_backward_recv_forward'] and step['ts']<=e['ts']<step['ts']+step['dur']]
        counts=collections.Counter();ops=[]
        for i,e in sorted(calls,key=lambda x:x[1]['ts']):
            ids=message_ids(e['name'],stage,counts,16)
            prev=[b['end_absolute_ms'] for b in c['blocks'] if es[b['cpu_event']]['ts']<e['ts']]
            r=dict(stage=stage,rank=stage*16,event=i,name=e['name'],start_ms=e['ts']/1000,end_ms=(e['ts']+e['dur'])/1000,ready_ms=max([e['ts']/1000]+prev),messages=ids)
            ops.append(r)
            for msg,side in ids:messages[msg].append(dict(r,side=side))
        save(out/f'pp-stage-{stage}.json',ops);rawrefs.append(dict(path=str(p),sha256=h,rank=stage*16))
        if stage==0:
            tailpath=BASE/'tail-source155-r1/target-60.json';tail=load(tailpath);inputs[str(tailpath)]=sha(tailpath);assert tail['sha256']==h
            truth_one=blocks['s0:B3']['end_absolute_ms']-blocks['s0:F0']['start_absolute_ms']
            truth_body=origin+step['dur']/1000-blocks['s0:F0']['start_absolute_ms']
            assert abs(truth_body-truth_one-tail['tail_ms'])<1e-5
        del es
        print('EXTRACT',stage,flush=True)
    pairs=[]
    for msg,ends in messages.items():
        assert len(ends)==2 and sorted(r['side'] for r in ends)==['recv','send'],msg
        overlap=min(r['end_ms'] for r in ends)-max(r['ready_ms'] for r in ends)
        pairs.append(dict(message=msg,phase=msg[0],standalone=all(len(r['messages'])==1 for r in ends),effective_ms=overlap,ends=ends))
    samples={ph:[r['effective_ms'] for r in pairs if r['phase']==ph and r['standalone'] and r['effective_ms']>0] for ph in ['F','B']}
    assert all(samples.values());pp={ph:statistics.median(v) for ph,v in samples.items()}
    save(out/'pp-calibration.json',dict(costs=pp,samples=samples,pairs=pairs,excluded=[r['message'] for r in pairs if r['standalone'] and r['effective_ms']<=0]))
    source=BASE/'deepep-stable-20261016-r1/model/outer-prediction.json';outer=load(source);inputs[str(source)]=sha(source)
    g=copy.deepcopy(outer['nodes']);baseline=run(g);assert abs(baseline['s0:B3']['end_ms']-22119.790913276483)<1e-6
    for k,n in g.items():
        if k in blocks:n['duration_ms']=blocks[k]['duration_ms'];n['basis']='256iter60 measured whole FB GPU envelope'
        elif k.startswith('MSG:'):n['duration_ms']=pp[k.split(':')[1]];n['basis']='256iter60 standalone paired PP effective median'
    assert all(g[k]['dependencies']==outer['nodes'][k]['dependencies'] for k in outer['nodes'])
    one=run(g)['s0:B3']['end_ms']
    # Explicit tail event nodes. Observed-ready constraints are calibration
    # parameters, not inferred portable causal dependencies or pure idle time.
    events=sorted(tail['device_events'],key=lambda r:(r['start_ms'],r['end_ms']))
    built=[];costs=[]
    for i,e in enumerate(events):
        if e['end_ms']<=e['start_ms']:continue
        pred=max((x for x in built if x[1]<=e['start_ms']+1e-7),key=lambda x:x[1],default=('s0:B3',0))
        ready=f'tail:ready:{i}';key=f'tail:event:{i}';gap=max(0,e['start_ms']-pred[1])
        g[ready]=dict(duration_ms=gap,dependencies=[pred[0]],kind='target_observed_readiness',basis='same-sample reserved interval; not proven causal')
        g[key]=dict(duration_ms=e['end_ms']-e['start_ms'],dependencies=[ready],kind=e['kind'],event=e['event'],name=e['name'])
        built.append((key,e['end_ms']));costs.append(dict(node=key,kind=e['kind'],cost_ms=g[key]['duration_ms'],ready_ms=gap))
    last=max(built,key=lambda x:x[1]);g['ITER_END']=dict(duration_ms=tail['tail_ms']-last[1],dependencies=[k for k,_ in built],kind='profiler_end_reserved')
    assert g['ITER_END']['duration_ms']>=0
    t=run(g);body=t['ITER_END']['end_ms'];assert abs((body-one)-tail['tail_ms'])<1e-5
    max_event_error=max(abs(t[k]['end_ms']-one-end) for k,end in built);assert max_event_error<1e-5
    physics=load(BASE/'full-step-32to256-r1/sealed-prediction.json')['physics'];factor=100*physics['model_flops_per_iteration']/(256*physics['peak_flops_per_gpu_s'])*1000
    report=dict(status='PARTIAL_TARGET_CALIBRATED_OUTER_DAG_REPLAY',iteration=60,onef1b=dict(replay_ms=one,truth_ms=truth_one,error_percent=100*(one/truth_one-1)),body=dict(replay_ms=body,truth_ms=truth_body,error_percent=100*(body/truth_body-1),replay_mfu_percent=factor/body,truth_mfu_percent=factor/truth_body,mfu_error_percent=100*(truth_body/body-1)),pp_effective_ms=pp,tail_event_nodes=len(built),tail_max_event_error_ms=max_event_error,checks=dict(outer_dependencies_unchanged=True,current_baseline_reproduced=True,raw_sha_verified=16,fb_costs=128),limits=['Same-sample calibration, NOT prediction.','Current outer dependency graph preserved; fine F/B internal nodes are contracted, NOT individually256-calibrated.','PP median is effective host-endpoint overlap, not pure network service; clocks assumed comparable.','Tail observed-ready constraints make local replay exact by construction; not verified causal optimizer DAG.','Only rank0 tail; other-stage RS/EDP overlap and all-rank synchronization not validated.'])
    save(out/'calibrated-graph.json',g);save(out/'replay.json',t);save(out/'parameters.json',dict(fb={k:b['duration_ms'] for k,b in blocks.items()},pp=pp,tail=costs,physics=physics));save(out/'report.json',report)
    inputs[str(Path(__file__))]=sha(Path(__file__))
    save(out/'manifest.json',dict(inputs=inputs,raw=rawrefs,outputs={p.name:sha(p) for p in out.glob('*.json')}))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

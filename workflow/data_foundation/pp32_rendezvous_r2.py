"""Conditional CPU-envelope rendezvous baseline; not pure GPU/network service."""
import argparse
import collections as C
import graphlib
import hashlib
import json
from pathlib import Path
import statistics as S
from pp32_minimal_r1 import sequence

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/data-foundation/pp32-minimal-r1'
RANKS=[0,8,16,24]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def message_ids(name,s,counters,p=4):
    ids=[]
    for token,ph,direction,valid in [('send_forward','F','send',s<p-1),('recv_forward','F','recv',s>0),
                                    ('send_backward','B','send',s>0),('recv_backward','B','recv',s<p-1)]:
        if token not in name or not valid:continue
        mb=counters[token];counters[token]+=1
        edge=s if token in ['send_forward','recv_backward'] else s-1
        ids.append((f'{ph}:{edge}:{mb}',direction))
    return ids

def observed(x,origin):
    ops=[];counts=C.Counter()
    for r in x['phases']:
        ops.append(dict(kind=r['phase'],mb=r['microbatch'],id=f"s{x['stage']}:{r['phase']}{r['microbatch']}",
                        start=r['start_ms'],end=r['end_ms'],event=r['source_event'],messages=[]))
    for j,r in enumerate(x['pp_host_calls']):
        ops.append(dict(kind='PP',name=r['name'],id=f"s{x['stage']}:PP{j}",start=r['start_ms'],
                        end=r['start_ms']+r['duration_ms'],event=r['event']))
    ops.sort(key=lambda r:r['start'])
    for r in ops:
        if r['kind']=='PP':r['messages']=message_ids(r['name'],x['stage'],counts)
        r['absolute_start_ms']=origin+r['start'];r['absolute_end_ms']=origin+r['end']
    assert all(a['end']<=b['start']+1e-6 for a,b in zip(ops,ops[1:]))
    assert [(r['kind'],r['mb']) for r in ops if r['kind'] in ['F','B']]==sequence(4,8,x['stage'])
    return dict(rank=x['rank'],stage=x['stage'],origin_ms=origin,profiler_ms=x['profiler_ms'],ops=ops)

def timings(x):
    p=Path(x['path']);assert sha(p)==x['sha256']
    d=load(p);es=d['traceEvents'];step=next(e for e in es if e.get('cat')=='user_annotation' and e['name']==x['profiler_name'])
    # Audit broader runtime ownership without declaring cross-thread time containment causal.
    runtime=C.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat')=='privateuse1_runtime' and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append(e)
    phases=[(r,es[r['source_event']]) for r in x['phases']]
    assigned=C.Counter();unknown=0;cross=0;pp_inside=0
    for e in es:
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        rr=runtime.get(e.get('args',{}).get('correlation'),[])
        if len(rr)!=1:continue
        r=rr[0];hits=[(row,a) for row,a in phases if a['pid']==r['pid'] and a['ts']<=r['ts'] and r['ts']+r.get('dur',0)<=a['ts']+a['dur']+0.01]
        if len(hits)==1:
            row,a=hits[0];assigned[row['phase']]+=1;cross+=r['tid']!=a['tid']
            if e.get('args',{}).get('Process Group Description')=='PIPELINE_MODEL_PARALLEL_GROUP':pp_inside+=1
        else:unknown+=1
    audit=dict(rank=x['rank'],iteration=x['iteration'],r1_same_thread_assigned=x['owned_device_events'],
               cross_thread_temporal_candidates=dict(assigned),cross_thread_count=cross,unassigned_runtime_devices=unknown,
               pp_group_device_candidates_inside_fb=pp_inside,
               status='TEMPORAL_CANDIDATES_NOT_PROVEN_CROSS_THREAD_CAUSAL_OWNERSHIP')
    return step['ts']/1000,audit

def graph(observations,params,mode):
    nodes={};messages=C.defaultdict(list)
    def add(k,d,deps,kind,stage=None):
        nodes[k]=dict(duration_ms=d,dependencies=deps,kind=kind,stage=stage)
    add('ENTRY',params['entry_ms'],[],'entry')
    for x in observations:
        s=x['stage'];role='first' if s==0 else 'last' if s==3 else 'middle';prev='ENTRY'
        for op in x['ops']:
            k=op['id']
            if op['kind'] in ['F','B']:
                deps=[prev]
                if op['kind']=='B':deps.append(f's{s}:F{op["mb"]}')
                add(k,params['fb_ms'][role][op['kind']],deps,op['kind'],s);prev=k
            elif op['messages']:
                entry=k+':ready';add(entry,0,[prev],'pp_ready',s)
                for msg,side in op['messages']:messages[msg].append((entry,side))
                add(k+':done',0,[f'MSG:{m}' for m,_ in op['messages']],'pp_done',s);prev=k+':done'
        add(f'STAGE{s}_DONE',0,[prev],'stage_done',s)
    for msg,ends in messages.items():
        assert len(ends)==2 and sorted(side for _,side in ends)==['recv','send'], (msg,ends)
        cost=0 if mode=='zero_service' else params['pp_effective_overlap_ms'][msg[0]]
        add('MSG:'+msg,cost,[entry for entry,_ in ends],'communication')
    add('TAIL',params['post_pipeline_ms'],[f'STAGE{s}_DONE' for s in range(4)],'post_pipeline')
    order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
    for k in order:
        n=nodes[k];n['start_ms']=max([nodes[d]['end_ms'] for d in n['dependencies']] or [0]);n['end_ms']=n['start_ms']+n['duration_ms']
    return dict(mode=mode,nodes=nodes,order=order,total_ms=nodes['TAIL']['end_ms'],pp_messages=len(messages))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='pp32-rendezvous-r2',source_iteration=60,evaluation_iteration=70,ranks=RANKS,
        cost_basis='CPU F/B elapsed plus empirical PP rendezvous overlap; NOT pure GPU/network service',
        modes=['zero_service','effective_overlap'],target256_permission='none',
        assumptions=['source host trace timestamps comparable; not independently clock calibrated',
        'source PP call structure reused on70, no target time used for prediction',
        'one PP chain represents stage behavior, other ranks implicit in source CPU durations',
        'post-pipeline scalar includes DP/update/host overhead and is NOT transferable OPT cost'],
        planned_counts='6 FB + 2 effective PP + 1 entry + 1 post-pipeline = 10 empirical parameters',
        accuracy_gate='diagnostic only; no automatic model promotion or 256 extrapolation',
        source_code=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))
    dump(out/'protocol.json',protocol)
    obs=[];audits=[]
    for rank in RANKS:
        x=load(OLD/f'observation-60-{rank}.json');origin,audit=timings(x);audits.append(audit);obs.append(observed(x,origin));print('source',rank,flush=True)
    dump(out/'source-observations.json',obs)
    matched=C.defaultdict(list)
    for x in obs:
        for op in x['ops']:
            for msg,side in op['messages']:matched[msg].append(dict(rank=x['rank'],side=side,**op))
    pairs=[]
    for msg,ends in matched.items():
        assert len(ends)==2 and sorted(e['side'] for e in ends)==['recv','send']
        start=max(e['absolute_start_ms'] for e in ends);end=min(e['absolute_end_ms'] for e in ends)
        pairs.append(dict(id=msg,ends=ends,overlap_ms=end-start,standalone=all(len(e['messages'])==1 for e in ends)))
    dump(out/'source-pp-pairs.json',pairs)
    assert len(pairs)==48
    # Only standalone endpoint pairs calibrate communication; combined calls contain other-message waits.
    pp={ph:[p['overlap_ms'] for p in pairs if p['id'][0]==ph and p['standalone']] for ph in ['F','B']}
    assert all(pp[ph] and min(pp[ph])>0 for ph in pp), pp
    begin=min(x['origin_ms'] for x in obs)
    entry=min(op['absolute_start_ms'] for x in obs for op in x['ops'])-begin
    pipeline_end=max(op['absolute_end_ms'] for x in obs for op in x['ops'])
    post=max(x['origin_ms']+x['profiler_ms'] for x in obs)-pipeline_end
    params=dict(fb_ms=load(OLD/'parameters-source60.json')['costs'],pp_effective_overlap_ms={ph:S.median(pp[ph]) for ph in pp},
                pp_calibration_samples=pp,entry_ms=entry,post_pipeline_ms=post,count=10,
                meaning='conditional source empirical timings; not source-only portable physical model')
    dump(out/'parameters.json',params)
    predictions=[graph(obs,params,m) for m in protocol['modes']]
    dump(out/'predictions.json',predictions)
    dump(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.json'),parameters_sha256=sha(out/'parameters.json'),
        note='written before reading70 raw or observations; no256 accessed'))
    # Evaluation begins only after source parameters and predictions have been sealed.
    target=[]
    for rank in RANKS:
        x=load(OLD/f'observation-70-{rank}.json');origin,audit=timings(x);audits.append(audit);target.append(observed(x,origin));print('evaluation',rank,flush=True)
    dump(out/'target-observations.json',target);dump(out/'gpu-ownership-audit.json',audits)
    for a,b in zip(obs,target):
        assert [(o['kind'],o.get('mb'),o['messages']) for o in a['ops']]==[(o['kind'],o.get('mb'),o['messages']) for o in b['ops']]
    report=[]
    for it,rows in [(60,obs),(70,target)]:
        for pred in predictions:
            report.append(dict(iteration=it,mode=pred['mode'],predicted_ms=pred['total_ms'],
                measured_rank_local_profiler_mean_ms=S.mean(x['profiler_ms'] for x in rows),
                rank_errors=[dict(rank=x['rank'],measured_ms=x['profiler_ms'],error_ms=pred['total_ms']-x['profiler_ms'],
                                 ARE_percent=100*abs(pred['total_ms']-x['profiler_ms'])/x['profiler_ms']) for x in rows],
                boundary='one-chain surrogate versus rank-local CPU ProfilerStep; not Training Step/world GPU makespan'))
    dump(out/'report.json',report)
    dump(out/'check.json',dict(status='PARTIAL',message_pairing='PASS',source_target_structure='PASS',pp_messages=48,
        source_only_prediction_seal='PASS',target256_prediction=False,
        blockers=['cross-host clock calibration unverified','only4 of32 ranks','CPU rather than complete GPU ownership',
                  'effective PP overlap not pure transport','post-pipeline scalar not portable DP/OPT model']))
    dump(out/'manifest.json',dict(inputs=[dict(path=str(OLD/f'observation-{it}-{r}.json'),sha256=sha(OLD/f'observation-{it}-{r}.json')) for it in [60,70] for r in RANKS],
        implementation=protocol['source_code'],artifacts=[dict(path=str(f.resolve()),sha256=sha(f)) for f in sorted(out.glob('*.json'))]))
    print(json.dumps(params));print(json.dumps([{k:v for k,v in x.items() if k!='rank_errors'} for x in report]))

if __name__=='__main__':main()

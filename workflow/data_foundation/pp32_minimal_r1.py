"""Bounded 32-GPU evidence pilot and explicit 1F1B graph. No target timing fit."""
import argparse
import collections as C
import graphlib
import hashlib
import json
from pathlib import Path
import re
import statistics as S

SOURCE = Path('/home/zjb/Desktop/32/gpu32_gbs64_framework')
TARGET = Path('/home/zjb/gbs64/framework_256_gbs_64')
KEYS = '''world_size pipeline_model_parallel_size tensor_model_parallel_size context_parallel_size
expert_model_parallel_size data_parallel_size num_layers decoder_first_pipeline_num_layers
decoder_last_pipeline_num_layers num_experts hidden_size seq_length micro_batch_size global_batch_size
recompute_granularity recompute_method recompute_num_layers virtual_pipeline_model_parallel_size'''.split()

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def config(root):
    p = next(root.glob('*/worker*/*/*RANK0.*.log'))
    rows = []
    with p.open() as f:
        for line in f:
            if 'end of arguments' in line:
                break
            rows.append(line)
            if len(rows) > 650:
                break
    excerpt = ''.join(rows)
    values = {}
    for k in KEYS:
        m = re.search(r'^\s*'+k+r'\s+\.+\s+(\S+)', excerpt, re.M)
        if m:
            v = m[1]
            values[k] = int(v) if v.isdigit() else v
    values['microbatches_derived'] = values['global_batch_size']//(values['micro_batch_size']*values['data_parallel_size'])
    return {'path': str(p), 'read_scope': 'argument prefix only, no target timing',
            'prefix_sha256': hashlib.sha256(excerpt.encode()).hexdigest(), 'config': values}

def sequence(p, m, stage):
    w = min(p-1-stage, m)
    seq = [('F', i) for i in range(w)]
    for j in range(m-w):
        seq += [('F', w+j), ('B', j)]
    return seq + [('B', i) for i in range(m-w, m)]

def dag(p, m, costs, pp_delay=0):
    nodes = {}
    for s in range(p):
        prev = None
        for phase, mb in sequence(p, m, s):
            key = f'{s}:{phase}:{mb}'
            deps = [] if prev is None else [(prev, 0, 'local_order')]
            if phase == 'F' and s:
                deps.append((f'{s-1}:F:{mb}', pp_delay, 'activation'))
            if phase == 'B':
                deps.append((f'{s}:F:{mb}', 0, 'same_microbatch'))
                if s+1 < p:
                    deps.append((f'{s+1}:B:{mb}', pp_delay, 'gradient'))
            nodes[key] = dict(stage=s, phase=phase, microbatch=mb, duration_ms=costs[s][phase], dependencies=deps)
            prev = key
    order = graphlib.TopologicalSorter({k: [e[0] for e in v['dependencies']] for k,v in nodes.items()}).static_order()
    for k in order:
        n = nodes[k]
        n['start_ms'] = max([nodes[a]['end_ms']+lag for a,lag,_ in n['dependencies']] or [0])
        n['end_ms'] = n['start_ms']+n['duration_ms']
    return dict(nodes=nodes, total_ms=max(n['end_ms'] for n in nodes.values()))

def selftest():
    count=0
    for p in range(1,18):
        for m in range(1,18):
            d=dag(p,m,[{'F':1,'B':1} for _ in range(p)])
            assert len(d['nodes'])==2*p*m
            assert d['total_ms']==2*(m+p-1), (p,m,d['total_ms'])
            for s in range(p):
                q=sequence(p,m,s)
                assert sorted(i for ph,i in q if ph=='F')==list(range(m))
                assert sorted(i for ph,i in q if ph=='B')==list(range(m))
            for n in d['nodes'].values():
                assert all(n['start_ms']>=d['nodes'][a]['end_ms']+lag for a,lag,_ in n['dependencies'])
            count+=1
    return dict(status='PASS', combinations=count, meaning='synthetic scheduler tests, not trace accuracy')

def union(es):
    end=-float('inf'); total=0
    for a,b in sorted(es):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total

def extract(p, iteration, rank):
    raw=p.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    data=json.loads(raw); events=data['traceEvents']; info=data['distributedInfo']
    assert info['rank']==rank and info['world_size']==32
    groups=[g for g in info['pg_config'] if g['pg_desc']=='PIPELINE_MODEL_PARALLEL_GROUP' and rank in g['ranks']]
    assert len(groups)==1
    members=groups[0]['ranks']; stage=members.index(rank)
    steps=[(i,e) for i,e in enumerate(events) if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')]
    assert len(steps)==1
    idx,step=steps[0]; origin=step['ts']; stop=origin+step['dur']
    annotations=[(i,e) for i,e in enumerate(events) if e.get('cat')=='user_annotation' and e.get('ph')=='X' and origin<=e.get('ts',-1)<stop]
    phases=sorted([(i,e) for i,e in annotations if e['name'] in ['forward_step','backward_step']],key=lambda x:x[1]['ts'])
    assert len(phases)==16
    runtime=C.defaultdict(list)
    for i,e in enumerate(events):
        if e.get('cat')=='privateuse1_runtime' and 'correlation' in e.get('args',{}):
            runtime[e['args']['correlation']].append((i,e))
    owned=C.defaultdict(list); unmapped=0; devices=0
    for i,e in enumerate(events):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset'] or not(origin<=e.get('ts',-1)<stop):continue
        devices+=1; rs=runtime.get(e.get('args',{}).get('correlation'),[])
        if len(rs)!=1:unmapped+=1;continue
        _,r=rs[0]
        hits=[j for j,a in phases if (a['pid'],a['tid'])==(r['pid'],r['tid']) and a['ts']<=r['ts'] and r['ts']+r.get('dur',0)<=a['ts']+a['dur']+0.01]
        if len(hits)==1:owned[hits[0]].append((i,e))
    rows=[]; serial=C.Counter()
    for i,e in phases:
        ph='F' if e['name']=='forward_step' else 'B'; mb=serial[ph];serial[ph]+=1
        ds=owned[i]; intervals=[(d['ts'],d['ts']+d['dur']) for _,d in ds]
        rows.append(dict(id=f'i{iteration}-r{rank}-{ph}{mb}',phase=ph,microbatch=mb,source_event=i,
            start_ms=(e['ts']-origin)/1000,end_ms=(e['ts']+e['dur']-origin)/1000,cpu_ms=e['dur']/1000,
            device_events=[j for j,_ in ds],device_count=len(ds),
            device_span_ms=(max(b for a,b in intervals)-min(a for a,b in intervals))/1000 if intervals else None,
            device_union_ms=union(intervals)/1000,
            device_cost_status='DIAGNOSTIC_ONLY_same_thread_runtime_association_not_complete_semantic_cost'))
    expected=sequence(4,8,stage); actual=[(r['phase'],r['microbatch']) for r in rows]
    last_b=max(r['end_ms'] for r in rows if r['phase']=='B')
    pp=[]; tail=[]
    for i,e in annotations:
        row=dict(event=i,name=e['name'],start_ms=(e['ts']-origin)/1000,duration_ms=e['dur']/1000)
        if e['name'] in ['send_forward','recv_forward','send_backward','recv_backward','send_forward_recv_backward','send_backward_recv_forward']:pp.append(row)
        if e['name'] in ['finalize_model_grads','step','prepare_grads','step_with_ready_grads'] or e['name'].startswith('Optimizer.step#'):tail.append(row)
    return dict(iteration=iteration,rank=rank,stage=stage,pp_members=members,path=str(p),sha256=digest,
        profiler_name=step['name'],profiler_ms=step['dur']/1000,schedule_match=actual==expected,
        phases=rows,pp_host_calls=pp,pp_status='HOST_CALLS_INCLUDE_WAIT_NOT_TRANSFER_COST',tail_annotations=tail,
        last_B_to_profiler_end_ms=step['dur']/1000-last_b,tail_status='RESIDUAL_NOT_OPT_COST',
        device_events=devices,ambiguous_or_missing_runtime=unmapped,owned_device_events=sum(len(v) for v in owned.values()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='pp32-minimal-r1',scope='one complete PP chain, not all 32 ranks',ranks=[0,8,16,24],
        calibration_directory_iteration=60,validation_directory_iteration=70,raw_limit=8,
        target_permission='argument configuration only; no 256 timing, trace or v610 calibrated costs',
        accuracy_status='NOT_FULL_ITER_PREDICTOR',pp_service_ms=None,optimizer_service_ms=None,
        prediction_scope='six CPU-envelope means in explicit DAG, zero PP counterfactual only',
        model_source=str(Path(__file__).resolve()),model_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/'protocol.json',protocol);dump(out/'scheduler-check.json',selftest())
    configs={'source32':config(SOURCE),'target256':config(TARGET)};dump(out/'configs.json',configs)
    observations=[];costs={}
    for it in [60,70]:
        for rank in [0,8,16,24]:
            files=list(SOURCE.glob(f'*/worker*/profiler/iteration_{it}/rank{rank}.*.json'));assert len(files)==1
            row=extract(files[0],it,rank);observations.append(row);dump(out/f'observation-{it}-{rank}.json',row)
            print(it,rank,'schedule',row['schedule_match'],flush=True)
        if it==60:
            for role,stages in [('first',[0]),('middle',[1,2]),('last',[3])]:
                costs[role]={ph:S.mean(r['cpu_ms'] for x in observations if x['stage'] in stages for r in x['phases'] if r['phase']==ph) for ph in ['F','B']}
            dump(out/'parameters-source60.json',{'costs':costs,'count':6,'basis':'CPU annotation elapsed, includes internal sync; not pure GPU service'})
            d=dag(4,8,[costs['first'],costs['middle'],costs['middle'],costs['last']]);dump(out/'source-only-zero-pp-diagnostic.json',d)
    assert all(x['schedule_match'] for x in observations)
    metrics=[]
    for it in [60,70]:
        errors=[]
        for x in observations:
            if x['iteration']!=it:continue
            role='first' if x['stage']==0 else 'last' if x['stage']==3 else 'middle'
            for ph in ['F','B']:
                rs=[r['cpu_ms'] for r in x['phases'] if r['phase']==ph]; pred=costs[role][ph]
                errors.append(dict(rank=x['rank'],phase=ph,measured_mean_ms=S.mean(rs),predicted_ms=pred,ARE_mean_percent=100*S.mean(abs(pred-v)/v for v in rs)))
        metrics.append(dict(iteration=it,scope='CPU envelope duration only, NOT whole iteration prediction',rows=errors))
    dump(out/'block-cost-evaluation.json',metrics)
    dump(out/'check.json',dict(status='PARTIAL',schedule_status='PASS',traces=8,fb_blocks=128,
        ranks=[0,8,16,24],full32rank_coverage=False,target256_prediction=False,
        missing=['PP transfer/service versus wait separation','GPU semantic block completeness','DP and update-tail cost','remaining 28 ranks','source/target expert-DP and topology changes']))
    dump(out/'manifest.json',dict(inputs=[dict(path=x['path'],sha256=x['sha256']) for x in observations],
        artifacts=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.glob('*.json'))],
        source=protocol['model_source'],source_sha256=protocol['model_sha256']))
    print(json.dumps(costs))

if __name__=='__main__':main()

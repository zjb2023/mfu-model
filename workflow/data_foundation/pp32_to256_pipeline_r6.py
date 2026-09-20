"""Frozen source-only pipeline extrapolation; evaluate one rank-local GPU window."""
import argparse
import collections as C
import graphlib
import hashlib
import json
from pathlib import Path
from pp32_minimal_r1 import sequence
from pp32_rendezvous_r2 import message_ids

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def operations(p,m,s):
    out=[]; counters=C.Counter()
    def call(name):
        messages=message_ids(name,s,counters,p)
        if messages: out.append(dict(kind='PP',messages=messages))
    def fb(ph,i): out.append(dict(kind=ph,mb=i,messages=[]))
    w=min(p-s-1,m)
    for i in range(w):
        call('recv_forward');fb('F',i);call('send_forward')
    if m-w: call('recv_forward')
    for i in range(m-w):
        fb('F',w+i);call('send_forward_recv_backward');fb('B',i)
        call('send_backward' if i==m-w-1 else 'send_backward_recv_forward')
    for i in range(w):
        call('recv_backward');fb('B',m-w+i);call('send_backward')
    assert [(o['kind'],o['mb']) for o in out if o['kind']!='PP']==sequence(p,m,s)
    return out

def build(p,m,params):
    nodes={};messages=C.defaultdict(list)
    def add(k,cost,deps,kind,s=None): nodes[k]=dict(duration_ms=cost,dependencies=deps,kind=kind,stage=s)
    add('ENTRY',0,[],'ENTRY')
    for s in range(p):
        role='first' if s==0 else 'last' if s==p-1 else 'middle';prev='ENTRY'
        for i,o in enumerate(operations(p,m,s)):
            if o['kind']!='PP':
                k=f's{s}:{o["kind"]}{o["mb"]}';deps=[prev]
                if o['kind']=='B':deps.append(f's{s}:F{o["mb"]}')
                add(k,params['fb_ms'][role][o['kind']],deps,o['kind'],s);prev=k
            else:
                ready=f's{s}:call{i}:ready';done=f's{s}:call{i}:done'
                add(ready,0,[prev],'ready',s)
                for msg,side in o['messages']:messages[msg].append((ready,side))
                add(done,0,['MSG:'+msg for msg,_ in o['messages']],'done',s);prev=done
    for msg,ends in messages.items():
        assert len(ends)==2 and sorted(side for _,side in ends)==['recv','send']
        add('MSG:'+msg,params['pp_effective_overlap_ms'][msg[0]],[r for r,_ in ends],'PP')
    for k in graphlib.TopologicalSorter({k:v['dependencies'] for k,v in nodes.items()}).static_order():
        n=nodes[k];n['start_ms']=max([nodes[d]['end_ms'] for d in n['dependencies']] or [0]);n['end_ms']=n['start_ms']+n['duration_ms']
    end=nodes[f's0:B{m-1}']['end_ms'];start=nodes['s0:F0']['start_ms']
    assert end==max(n['end_ms'] for n in nodes.values())
    return dict(pp=p,microbatches=m,nodes=nodes,window_ms=end-start,paired_messages=len(messages))

def target_window(path):
    raw=path.read_bytes();d=json.loads(raw);es=d['traceEvents'];info=d['distributedInfo']
    assert info['rank']==0 and info['world_size']==256
    group=next(g for g in info['pg_config'] if g['pg_desc']=='PIPELINE_MODEL_PARALLEL_GROUP' and 0 in g['ranks'])
    assert len(group['ranks'])==16 and group['ranks'][0]==0
    steps=[e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')];assert len(steps)==1
    step=steps[0];phases=[(i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name') in ['forward_step','backward_step'] and step['ts']<=e['ts']<step['ts']+step['dur']]
    phases.sort(key=lambda v:v[1]['ts']);counts=C.Counter();runtime=C.defaultdict(list);owned=C.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat')=='privateuse1_runtime' and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append(e)
    for i,e in enumerate(es):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        rs=runtime.get(e.get('args',{}).get('correlation'),[])
        if len(rs)!=1:continue
        r=rs[0];hits=[j for j,a in phases if r['pid']==a['pid'] and a['ts']<=r['ts'] and r['ts']+r.get('dur',0)<=a['ts']+a['dur']+.01]
        if len(hits)==1:owned[hits[0]].append((i,e))
    blocks=[]
    for j,e in phases:
        ph='F' if e['name']=='forward_step' else 'B';mb=counts[ph];counts[ph]+=1
        ds=owned[j];assert ds
        blocks.append(dict(phase=ph,mb=mb,cpu_event=j,device_events=len(ds),
            start_ms=min(x['ts'] for _,x in ds)/1000,end_ms=max(x['ts']+x['dur'] for _,x in ds)/1000))
    assert [(b['phase'],b['mb']) for b in blocks]==sequence(16,4,0)
    assert all(a['end_ms']<=b['start_ms'] for a,b in zip(blocks,blocks[1:]))
    origin=blocks[0]['start_ms']
    for b in blocks:b['start_ms']-=origin;b['end_ms']-=origin
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),rank=0,iteration=60,pp_group=group['ranks'],blocks=blocks,window_ms=blocks[-1]['end_ms'],ownership='unique runtime correlation plus same-process temporal FB containment, cross-thread causality conditional')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    src=BASE/'pp32-gpu-boundary-r3b';cp=BASE/'pp32-minimal-r1/configs.json'
    dump(out/'protocol.json',dict(version='pp32-to256-pipeline-r6',source='32 iteration60; 4 representative ranks',evaluation='256 iteration60 rank0, predetermined before timing reads',
        window='PP0 first F associated GPU start -> last B associated GPU end, including fill/drain; excludes all tail RS/OPT/AG',
        assumptions=['one representative PP chain, not full world makespan','same role FB costs; internal CP/EP communication included','PP effective overlap reused across topology, not pure transport','framework snapshot is schedule reference, exact runtime equivalence unverified'],
        auxiliary_RS='deferred: r5 RS_PREP is aggregate, not an independently identified RS endpoint',
        no_target_fit=True,prior_target_history='256 data may have been seen in earlier v610 work; not blind holdout',code=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__)))) )
    params=load(src/'parameters.json');params={k:params[k] for k in ['fb_ms','pp_effective_overlap_ms']};dump(out/'parameters.json',params)
    configs=load(cp);dump(out/'configs.json',configs)
    for key in ['hidden_size','seq_length','micro_batch_size','num_experts','recompute_num_layers','recompute_method','recompute_granularity','context_parallel_size','expert_model_parallel_size']:
        assert configs['source32']['config'][key]==configs['target256']['config'][key],key
    # Generated framework calls must match the source observed rendezvous grouping.
    for x in load(src/'source-observations.json'):
        sig=lambda ops:[(o['kind'],o.get('mb'),tuple(tuple(m) for m in o['messages'])) for o in ops if o['kind']!='PP' or o['messages']]
        assert sig(operations(4,8,x['stage']))==sig(x['ops']),x['stage']
    tests=0
    for p in range(1,18):
        for m in range(1,10):build(p,m,params);tests+=1
    predictions={name:build(c['config']['pipeline_model_parallel_size'],c['config']['microbatches_derived'],params) for name,c in configs.items()}
    dump(out/'predictions.json',predictions)
    dump(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.json'),parameters_sha256=sha(out/'parameters.json'),target_timing_read=False,inputs=[dict(path=str(p),sha256=sha(p)) for p in [src/'parameters.json',src/'source-observations.json',cp]]))
    print('PREDICTION SEALED', {k:v['window_ms'] for k,v in predictions.items()},flush=True)
    # Target timing is only opened after the seal exists.
    target=Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40')
    paths=list(target.glob('worker*/profiler/iteration_60/rank0.*.pt.trace.json'));assert len(paths)==1
    truth=target_window(paths[0]);dump(out/'target-observation.json',truth)
    reports=[]
    for it,file in [(60,'source-observations.json'),(70,'target-observations.json')]:
        x=next(x for x in load(src/file) if x['rank']==0);fb=[o for o in x['ops'] if o['kind'] in ['F','B']]
        measured=fb[-1]['end']-fb[0]['start'];pred=predictions['source32']['window_ms']
        reports.append(dict(world=32,iteration=it,rank=0,predicted_ms=pred,measured_ms=measured,ARE_percent=100*abs(pred-measured)/measured))
    pred=predictions['target256']['window_ms'];measured=truth['window_ms']
    reports.append(dict(world=256,iteration=60,rank=0,predicted_ms=pred,measured_ms=measured,error_ms=pred-measured,ARE_percent=100*abs(pred-measured)/measured))
    dump(out/'report.json',reports)
    dump(out/'check.json',dict(status='PARTIAL',graph_checks='PASS',synthetic_cases=tests,source_call_structure='PASS',target_pp0_order='PASS',target_fit=False,full_world_accuracy=False,RS_evaluated=False))
    dump(out/'manifest.json',dict(files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(out.glob('*.json'))],implementation=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('pp32_minimal_r1.py'),Path(__file__).with_name('pp32_rendezvous_r2.py')]]))
    print(json.dumps(reports),flush=True)
if __name__=='__main__':main()

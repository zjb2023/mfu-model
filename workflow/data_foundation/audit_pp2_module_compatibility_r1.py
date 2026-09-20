"""Source60 structural compatibility, not PP2 detailed cost calibration."""
import argparse,collections,hashlib,json
from pathlib import Path
from evaluate_b_r10_60_to70 import inside
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def extract(rank):
    mp=BASE/f'pp32-structure-r4/rank-{rank}.json';meta=json.loads(mp.read_text());p=Path(meta['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['sha256'];d=json.loads(raw);es=d['traceEvents'];del raw
    runtime=collections.defaultdict(list)
    for e in es:
        if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append(e)
    result={}
    for phase,name in [('F','forward_step'),('B','backward_step')]:
        owner=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name')==name)
        ops=[e for e in es if e.get('cat')=='cpu_op' and inside(e,owner)]
        selected=['FusedDispatch','FusedCombine','FusedCombineBackward','FusedDispatchBackward','_GroupedLinear','_GroupedLinearBackward','CheckpointFunctionBackward']
        counts={n:sum(e['name']==n for e in ops) for n in selected}
        linear=[e for e in ops if e['name']=='_GroupedLinear']
        dims=[(e['args']['Input Dims'][0][1],tuple(e['args']['Input Dims'][17]),len(e['args']['Concrete Inputs'][1].strip('[]').split(','))) for e in linear]
        ds=[];cp=[]
        for i,e in enumerate(es):
            if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
            if not any(inside(v,owner) for v in runtime.get(e.get('args',{}).get('correlation'),[])):continue
            ds.append(dict(event=i,start_ms=e['ts']/1000,end_ms=(e['ts']+e['dur'])/1000))
            a=e.get('args',{})
            if a.get('Process Group Description')=='CONTEXT_PARALLEL_GROUP':cp.append(dict(event=i,start=e['ts'],signature=[a.get('Collective name'),a.get('In msg nelems'),a.get('Out msg nelems'),a.get('dtype')],args=a))
        cp.sort(key=lambda x:x['start'])
        result[phase]=dict(counts=counts,linear_shapes=dims,cp=cp,start_ms=min(x['start_ms'] for x in ds),end_ms=max(x['end_ms'] for x in ds))
        result[phase]['duration_ms']=result[phase]['end_ms']-result[phase]['start_ms']
        assert len(cp)==(16 if phase=='F' else 36)
        assert counts['FusedDispatch']==counts['FusedCombine']==4
        if phase=='B':assert counts['CheckpointFunctionBackward']==counts['FusedCombineBackward']==counts['FusedDispatchBackward']==4
    groups=[g for g in meta['groups'] if rank in g['ranks'] and g['pg_desc'] in ['CONTEXT_PARALLEL_GROUP','EXPERT_MODEL_PARALLEL_GROUP']]
    assert sorted(len(g['ranks']) for g in groups)==[2,8]
    return dict(rank=rank,stage=rank//8,path=str(p),sha256=meta['sha256'],groups=groups,phases=result)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    rows=[]
    for r in [8]+list(range(16,24)):
        x=extract(r);rows.append(x);print('structure checked',r,flush=True)
    ref=rows[0]
    for x in rows[1:]:
        for ph in ['F','B']:
            a1=ref['phases'][ph];b=x['phases'][ph]
            assert a1['counts']==b['counts'] and a1['linear_shapes']==b['linear_shapes']
            assert [c['signature'] for c in a1['cp']]==[c['signature'] for c in b['cp']]
    summary=dict(status='PASS_OBSERVED_SIGNATURE_PARTIAL_CAUSAL_EQUIVALENCE',source_iteration=60,reference_rank=8,pp2_ranks=list(range(16,24)),checks=['four layer EP/checkpoint/GroupedLinear counts','CP2/EP8 group sizes','F16 and B36 CP events with matching ordered message signatures','GroupedLinear input/weight dimensions'],PP2_cost_calibrated=False,limitations=['does not prove all cross-stream event edges or all microbatches equivalent','PP1 detailed costs reused unchanged; PP2 source durations diagnostic only'])
    for name,data in [('evidence',rows),('summary',summary)]:dump(a.out/(name+'.json'),data)
    def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    dump(a.out/'manifest.json',dict(inputs=[dict(path=x['path'],sha256=x['sha256']) for x in rows]+[rec(BASE/f'pp32-structure-r4/rank-{x["rank"]}.json') for x in rows],code=[rec(Path(__file__)),rec(Path(__file__).with_name('evaluate_b_r10_60_to70.py'))],outputs=[rec(p) for p in a.out.glob('*.json')]))
    print(json.dumps(summary))
if __name__=='__main__':main()

"""Audit last executed B layer; target facts never enter predictor parameters."""
import argparse,ast,collections,hashlib,json,statistics
from pathlib import Path
from evaluate_b_r10_60_to70 import inside
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(ds):
    if not ds:return None
    intervals=sorted((d['start_ms'],d['end_ms']) for d in ds);merged=[]
    for lo,hi in intervals:
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    lo=intervals[0][0];hi=max(v[1] for v in intervals);union=sum(b-a for a,b in merged)
    return dict(start_ms=lo,end_ms=hi,envelope_ms=hi-lo,union_ms=union,uncovered_ms=hi-lo-union,count=len(ds))
def extract(meta,it):
    p=Path(meta['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['sha256'];es=json.loads(raw)['traceEvents'];del raw
    b=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='backward_step')
    cks=sorted([e for e in es if e.get('cat')=='cpu_op' and e['name']=='CheckpointFunctionBackward' and inside(e,b)],key=lambda e:e['ts']);assert len(cks)==4;c=cks[3]
    ops=[(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and inside(e,c)]
    fs=sorted([(i,e) for i,e in ops if e['name']=='_GroupedLinear'],key=lambda z:z[1]['ts']);bs=sorted([(i,e) for i,e in ops if e['name']=='_GroupedLinearBackward'],key=lambda z:z[1]['ts']);assert len(fs)==len(bs)==2
    assert [e['args']['Sequence number'] for _,e in bs]==[e['args']['Sequence number'] for _,e in fs][::-1]
    tokens=ast.literal_eval(fs[0][1]['args']['Concrete Inputs'][1]);assert tokens==ast.literal_eval(fs[1][1]['args']['Concrete Inputs'][1]);assert len(tokens)==20
    assert sum(tokens)==fs[0][1]['args']['Input Dims'][0][0]
    markers=[('R_FC1',*fs[0]),('R_FC2',*fs[1]),('B_FC2',*bs[0]),('B_FC1',*bs[1])]
    names={'FusedDispatch':'R_dispatch','FusedCombine':'R_combine','FusedCombineBackward':'B_dispatch','FusedDispatchBackward':'B_combine'}
    markers += [(names[e['name']],i,e) for i,e in ops if e['name'] in names]
    end=next(e['ts']+e['dur'] for name,_,e in markers if name=='R_combine')
    runtime=collections.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append((i,e))
    ds=[]
    for i,e in enumerate(es):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        calls=[(j,v) for j,v in runtime.get(e.get('args',{}).get('correlation'),[]) if inside(v,c)]
        if not calls:continue
        j,call=min(calls,key=lambda z:z[1].get('dur',0));phase='R' if call['ts']<=end else 'B'
        hits=[(name,k,v) for name,k,v in markers if inside(call,v)]
        label=min(hits,key=lambda z:z[2]['dur'])[0] if hits else phase+'_other'
        if e.get('args',{}).get('Process Group Description')=='CONTEXT_PARALLEL_GROUP':label=phase+'_CP'
        ds.append(dict(event=i,launch_event=j,name=e['name'],stream=str(e.get('args',{}).get('stream',e.get('tid'))),phase=phase,label=label,start_ms=e['ts']/1000,end_ms=(e['ts']+e['dur'])/1000))
    origin=min(d['start_ms'] for d in ds)
    for d in ds:d['start_ms']-=origin;d['end_ms']-=origin
    groups={k:stats([d for d in ds if d['label']==k]) for k in sorted({d['label'] for d in ds})}
    phases={k:stats([d for d in ds if d['phase']==k]) for k in ['R','B']}
    combines={s:stats([d for d in ds if d['label']=='B_combine' and d['stream']==s]) for s in sorted({d['stream'] for d in ds if d['label']=='B_combine'})}
    dispatch_streams={d['stream'] for d in ds if d['label']=='B_dispatch'};assert len(dispatch_streams)==1
    comm=next(iter(dispatch_streams));assert comm in combines
    prep=[v for s,v in combines.items() if s!=comm];assert len(prep)==1
    handoff=combines[comm]['start_ms']-prep[0]['end_ms']
    # Decompose recompute elapsed window into disjoint timeline labels; overlapping
    # labels get their own bucket, never charge simultaneous streams twice.
    rd=[d for d in ds if d['phase']=='R'];points=sorted({v for d in rd for v in [d['start_ms'],d['end_ms']]});parts=collections.defaultdict(float)
    for a,z in zip(points,points[1:]):
        labels={d['label'] for d in rd if d['start_ms']<z and d['end_ms']>a}
        key='uncovered' if not labels else next(iter(labels)) if len(labels)==1 else 'overlap'
        parts[key]+=z-a
    assert abs(sum(parts.values())-phases['R']['envelope_ms'])<1e-5
    return dict(iteration=it,rank=meta['rank'],path=str(p),sha256=meta['sha256'],origin_absolute_ms=origin,checkpoint_sequence=c['args'].get('Sequence number'),tokens_per_expert=tokens,total_tokens=sum(tokens),token_cv=statistics.pstdev(tokens)/statistics.mean(tokens),max_tokens=max(tokens),zero_experts=tokens.count(0),groups=groups,phases=phases,combine_streams=combines,combine_comm_stream=comm,prepare_end_ms=prep[0]['end_ms'],combine_start_ms=combines[comm]['start_ms'],combine_handoff_ms=handoff,recompute_partition_ms=dict(parts),markers=[dict(label=n,event=i,sequence=e['args'].get('Sequence number'),input_dims=e['args'].get('Input Dims')) for n,i,e in markers],devices=ds)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    prior=BASE/'b-r10-60to70-eval-r2';facts={};inputs=[]
    for it in [60,70]:
        mp=prior/f'facts-{it}.json';inputs.append(mp);facts[it]=[]
        for m in json.loads(mp.read_text()):
            x=extract(m,it);facts[it].append(x);print('audit',it,m['rank'],flush=True)
        dump(a.out/f'evidence-{it}.json',facts[it])
    rows=[]
    for s,t in zip(facts[60],facts[70]):
        cs=s['combine_comm_stream'];ct=t['combine_comm_stream']
        rows.append(dict(rank=s['rank'],tokens60=s['total_tokens'],tokens70=t['total_tokens'],token_change_percent=100*(t['total_tokens']/s['total_tokens']-1),cv60=s['token_cv'],cv70=t['token_cv'],FC2_change_ms=t['groups']['B_FC2']['envelope_ms']-s['groups']['B_FC2']['envelope_ms'],FC1_change_ms=t['groups']['B_FC1']['envelope_ms']-s['groups']['B_FC1']['envelope_ms'],combine_envelope_change_ms=t['groups']['B_combine']['envelope_ms']-s['groups']['B_combine']['envelope_ms'],combine_union_change_ms=t['groups']['B_combine']['union_ms']-s['groups']['B_combine']['union_ms'],combine_uncovered_change_ms=t['groups']['B_combine']['uncovered_ms']-s['groups']['B_combine']['uncovered_ms'],visible_comm_tail_change_ms=t['combine_streams'][ct]['envelope_ms']-s['combine_streams'][cs]['envelope_ms'],handoff60_ms=s['combine_handoff_ms'],handoff70_ms=t['combine_handoff_ms'],recompute_change_ms=t['phases']['R']['envelope_ms']-s['phases']['R']['envelope_ms']))
    parts={k:sum(x['recompute_partition_ms'].get(k,0) for x in facts[70])/8-sum(x['recompute_partition_ms'].get(k,0) for x in facts[60])/8 for k in sorted({k for it in facts for x in facts[it] for k in x['recompute_partition_ms']})}
    aggregates={k:sum(r[k] for r in rows)/8 for k in rows[0] if k!='rank'}
    # Same time basis as prior stage evidence and unchanged predictor seal.
    for it in [60,70]:
        old=json.loads((prior/f'facts-{it}.json').read_text())
        for x,o in zip(facts[it],old):
            phase=next(v for v in o['phases'] if v['unit']==3 and v['phase']=='recompute')
            assert abs(x['phases']['R']['envelope_ms']-phase['duration_ms'])<1e-5
    report=dict(status='PASS_FACT_EXTRACTION_NOT_CAUSAL_PROOF',scope='32 PP1 B0 last executed layer, rank8-15 iter60/70',rows=rows,means=aggregates,recompute_disjoint_delta_ms=parts,limitations=['token distribution is observed target diagnostic input, not source-only prediction input','visible combine envelope spans streams; uncovered time is not proven pure wait or transport','cross-rank absolute clock alignment is inherited and conditional','CP/EP routing not inferred from token totals; no model costs updated'])
    dump(a.out/'report.json',report)
    def rec(p):return dict(path=str(p.resolve()),sha256=sha(p))
    dump(a.out/'manifest.json',dict(inputs=[rec(p) for p in inputs]+[dict(path=x['path'],sha256=x['sha256']) for it in facts for x in facts[it]],code=[rec(Path(__file__)),rec(Path(__file__).with_name('evaluate_b_r10_60_to70.py'))],outputs=[rec(p) for p in a.out.glob('*.json')]))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

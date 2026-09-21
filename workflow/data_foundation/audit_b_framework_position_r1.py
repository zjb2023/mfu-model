"""Framework marker timings, CPU and associated GPU separated, bounded four traces."""
import argparse, collections, hashlib, json
from pathlib import Path
from audit_b_operator_position_r1 import union
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
FRAME=Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer')
NAMES={'AttnFuncWithCPAndQKVOA2A':'recompute_attention_CP',
       'AttnFuncWithCPAndQKVOA2ABackward':'backward_attention_CP',
       'FusedDispatch':'recompute_EP_dispatch','FusedCombine':'recompute_EP_combine',
       'FusedCombineBackward':'backward_EP_dispatch','FusedDispatchBackward':'backward_EP_combine',
       '_GroupedLinear':'recompute_expert_linear','_GroupedLinearBackward':'backward_expert_linear'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inside(e,a):return e.get('pid')==a['pid'] and e.get('tid')==a['tid'] and a['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=a['ts']+a['dur']+.01
def extract(ref):
    p=Path(ref['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256']
    es=json.loads(raw)['traceEvents'];del raw
    bs=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='backward_step'],key=lambda e:e['ts'])
    runtimes=collections.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat')=='privateuse1_runtime':runtimes[e.get('args',{}).get('correlation')].append((i,e))
    devices=collections.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        calls=runtimes.get(e.get('args',{}).get('correlation'),[])
        if len(calls)==1:devices[calls[0][0]].append((i,e))
    blocks=[]
    for mb,b in enumerate(bs):
        ops=sorted([(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e.get('name') in NAMES and e['pid']==b['pid'] and b['ts']<=e['ts'] and e['ts']+e['dur']<=b['ts']+b['dur']+.01],key=lambda x:x[1]['ts'])
        assert len(ops)==40,(ref,mb,len(ops))
        counts=collections.Counter(); records=[]
        for i,op in ops:
            name=op['name'];ordinal=counts[name];counts[name]+=1
            kind=NAMES[name]
            if name=='_GroupedLinear':kind += '_FC1' if ordinal%2==0 else '_FC2'
            if name=='_GroupedLinearBackward':kind += '_FC2' if ordinal%2==0 else '_FC1'
            ds=[d for cid,rs in runtimes.items() if len(rs)==1 and inside(rs[0][1],op) for d in devices.get(rs[0][0],[])]
            iv=[(e['ts'],e['ts']+e['dur']) for _,e in ds]
            records.append(dict(cpu_event=i,name=name,kind=kind,ordinal=ordinal,cpu_ms=op['dur']/1000,
                gpu_union_ms=union(iv),gpu_envelope_ms=(max(y for x,y in iv)-min(x for x,y in iv))/1000 if iv else None,
                intervals=iv,device_events=[j for j,_ in ds]))
        sums=[]
        for kind in sorted({r['kind'] for r in records}):
            rs=[r for r in records if r['kind']==kind]
            sums.append(dict(kind=kind,count=len(rs),cpu_sum_ms=sum(r['cpu_ms'] for r in rs),
                gpu_union_ms=union([iv for r in rs for iv in r['intervals']]),
                call_envelopes_sum_ms=sum(r['gpu_envelope_ms'] or 0 for r in rs),
                calls_without_devices=sum(not r['device_events'] for r in rs)))
        blocks.append(dict(mb=mb,operators=records,summary=sums))
    return dict(world=ref['world'],stage=ref['stage'],iteration=60,blocks=blocks)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    previous=BASE/'b-operator-position-r1';data=[];refs=[]
    for w,s in [(224,5),(224,6),(256,1),(256,14)]:
        ref=json.loads((previous/f'{w}-pp{s}.json').read_text());refs.append({k:ref[k] for k in ['path','sha256','world','stage']})
        d=extract(ref);data.append(d);(a.out/f'{w}-pp{s}.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(w,s,'PASS',flush=True)
    comparisons=[]
    for left,right in [data[:2],data[2:]]:
        for l,r in zip(left['blocks'],right['blocks']):
            lm={x['kind']:x for x in l['summary']};rm={x['kind']:x for x in r['summary']}
            comparisons.append(dict(world=left['world'],stages=[left['stage'],right['stage']],mb=l['mb'],deltas=[dict(kind=k,cpu_ms=rm[k]['cpu_sum_ms']-lm[k]['cpu_sum_ms'],gpu_union_ms=rm[k]['gpu_union_ms']-lm[k]['gpu_union_ms']) for k in lm]))
    report=dict(status='PARTIAL_FRAMEWORK_MARKER_ATTRIBUTION',comparisons=comparisons,limits=['Reference236B is semantic evidence, not verified exact installed training revision','FC1/FC2 labels are inferred from paired forward/reverse order, not weight IDs','CPU duration includes dispatch/wait and is not GPU compute cost','GPU associations require same-thread runtime containment; hidden transport and unmatched work excluded','Operator unions/envelopes may overlap; never sum as B wall time','Only four representative ranks at iter60; no all-stage or cross-iteration claim'])
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    code=[Path(__file__).resolve(),Path(__file__).with_name('audit_b_operator_position_r1.py')]
    framework=[FRAME/'moe/fused_a2a.py',FRAME/'moe/experts.py',FRAME/'transformer_block.py']
    (a.out/'manifest.json').write_text(json.dumps(dict(raw=refs,reference_code={str(p):sha(p) for p in framework},code={str(p):sha(p) for p in code},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}),indent=2)+'\n')
if __name__=='__main__':main()

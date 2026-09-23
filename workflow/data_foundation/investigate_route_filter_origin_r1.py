"""Post-filter concentration and exact trace sequence; no invented pre-filter values."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=BASE/'route-filter-origin-r1';out.mkdir(exist_ok=False)
stats=[];inputs={}
for folder,label in [('layer-filtering-r1','A'),('layer-filtering-edpB-r1','B')]:
    p=BASE/folder/'report.json';d=json.loads(p.read_text());inputs[str(p)]=sha(p)
    records=[r for p in (BASE/folder).glob('w*.json') for r in json.loads(p.read_text())['records']]
    for mb in range(4):
        for layer in [3,58]:
            rs=[r for r in records if r['mb']==mb and r['layer']==layer];assert len(rs)==8
            counts=[n for r in rs for n in r['post_capacity_send_counts']]
            group=next(r for r in d['rows'] if r['mb']==mb and r['layer']==layer)
            assert sum(counts)==group['retained_assignments']
            stats.append(dict(group=label,mb=mb,layer=layer,active_experts=group['nonzero_experts'],total_experts=160,retained=sum(counts),sender_expert_pairs_at_cap=sum(n==615 for n in counts),assignments_at_cap=sum(n for n in counts if n==615),positive_sender_expert_pairs=sum(n>0 for n in counts)))
traces=[]
for stage,layer,rank in [(1,3,16),(14,58,224)]:
    p=BASE/f'layer-workload-256-r1/w256-i60-s{stage}-r{rank}.json';d=json.loads(p.read_text());raw=Path(d['path']);assert sha(raw)==d['sha256'];inputs[str(raw)]=d['sha256']
    es=json.loads(raw.read_text())['traceEvents'];r=next(r for r in d['rows'] if r['mb']==0 and r['layer']==layer);cp=es[r['checkpoint_forward_event']]
    events=[]
    for i,e in enumerate(es):
        if e.get('cat')=='cpu_op' and e.get('pid')==cp['pid'] and e.get('tid')==cp['tid'] and cp['ts']<=e['ts']<cp['ts']+cp['dur'] and e['name'] in ['aten::topk','aten::logical_and','aten::mul','aten::eq','aten::masked_fill','aten::masked_fill_']:
            a=e.get('args',{});dims=a.get('Input Dims',[])
            if dims and dims[0] in [[8192,160],[8192,6]]:events.append(dict(event=i,name=e['name'],dims=dims,concrete=a.get('Concrete Inputs'),ts=e['ts']))
    assert any(e['name']=='aten::eq' and e['dims'][0]==[8192,6] for e in events)
    assert any(e['name'].startswith('aten::masked_fill') and e['dims'][0]==[8192,6] for e in events)
    traces.append(dict(rank=rank,layer=layer,checkpoint=r['checkpoint_forward_event'],events=events))
code=Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/moe')
for p in [code/'router.py',code/'moe_utils.py',code/'token_dispatcher.py',Path(__file__).resolve()]:inputs[str(p)]=sha(p)
worker=Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40/worker33008')
searched={str(p):[str(x) for x in p.glob('**/*') if x.is_file()] for p in [worker/'2026-07-31_1340/logs/2026-07-31_1340_full_moments',worker/'tf_logs',worker/'wandb',worker/'checkpoints']}
report=dict(status='PASS for post-filter statistics and operator presence; causal attribution PARTIAL',stats=stats,trace_sequence=traces,searched_optional_artifacts=searched,
interpretation='Capacity sets selected probabilities to zero; downstream zero mask implements exclusion of these entries, not evidence of an independent second loss source. Preexisting zero probabilities remain unmeasured. Active-expert counts describe retained work only.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,outputs={'report.json':sha(out/'report.json')}),indent=2)+'\n')
print(json.dumps(dict(stats=[r for r in stats if r['mb']==0],trace_sequence=traces),indent=2))

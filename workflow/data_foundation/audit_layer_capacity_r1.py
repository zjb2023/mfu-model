"""Bounded trace evidence for capacity selection; no runtime-code identity claim."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
out=BASE/'layer-capacity-audit-r1'
out.mkdir(exist_ok=False)
records=[]
for world,stages in ((256,(1,7,14)),(224,(1,6,12))):
    for stage in stages:
        ref=BASE/f'layer-workload-{world}-r1/w{world}-i60-s{stage}-r{stage*16}.json'
        d=json.loads(ref.read_text()); raw=Path(d['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==d['sha256']
        es=json.loads(raw)['traceEvents']
        for r in d['rows']:
            cp=es[r['checkpoint_forward_event']]
            events=[dict(event=i,name=e['name'],dims=e.get('args',{}).get('Input Dims'),inputs=e.get('args',{}).get('Concrete Inputs')) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e.get('pid')==cp['pid'] and cp['ts']<=e['ts']<cp['ts']+cp['dur'] and e['name'] in ('aten::topk','aten::logical_and')]
            capacity=[e for e in events if e['name']=='aten::topk' and e['dims'][0]==[8192,160] and e['inputs'][1:3]==['615','0']]
            assert len(capacity)==1, (world,stage,r['layer'],events)
            assert any(e['name']=='aten::logical_and' and e['dims']==[[8192,160],[8192,160]] for e in events)
            records.append(dict(world=world,stage=stage,mb=r['mb'],layer=r['layer'],path=d['path'],sha256=d['sha256'],events=events))
code=Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/moe')
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (code/'router.py',code/'moe_utils.py',Path(__file__).resolve())}
(out/'report.json').write_text(json.dumps(dict(records=records,sources=sources,checks=len(records),status='PASS',interpretation='Trace shows top6 then expert-axis top615 and logical_and; consistent with reference capacity_factor2. Runtime source commit identity not established. EP8 perexpert ceiling=8*615=4920.'),indent=2)+'\n')
print('PASS',len(records),'layer/MB windows, 6 traces')

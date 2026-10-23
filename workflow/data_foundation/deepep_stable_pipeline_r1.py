"""Source-only DeepEP ingestion checks and token-stability stages.

Each stage writes to a new dataset namespace. Target timing is forbidden until
the model seal stage. This adapter deliberately fails on configuration drift.
"""
import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'workflow/deepep_stable/config.json'
C=json.loads(CONFIG.read_text());OUT=ROOT/C['output']
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def inventory():
    ingest=load(C['ingest_manifest']);assert ingest['status']=='PASS'
    paths={str(it):{} for it in C['sample_iterations']}
    for p in Path(C['data_root']).rglob('rank*.pt.trace.json'):
        m=re.fullmatch(r'iteration_(\d+)',p.parent.name)
        if not m or m[1] not in paths:continue
        rank=p.name.split('.')[0][4:];assert rank not in paths[m[1]]
        paths[m[1]][rank]=str(p)
    assert all(sorted(map(int,x))==list(range(32)) for x in paths.values()),{k:len(v) for k,v in paths.items()}
    raw=Path(paths['155']['8']).parents[3]
    keys=['moe_enable_deepep','moe_token_dispatcher_type','pipeline_model_parallel_size','tensor_model_parallel_size','expert_tensor_parallel_size','context_parallel_size','expert_model_parallel_size','data_parallel_size','num_layers','hidden_size','seq_length','num_experts','moe_router_topk','global_batch_size','micro_batch_size']
    configs=[]
    for p in raw.rglob('*.log'):
        if 'deepep_trace' in p.parts:continue
        with p.open(errors='replace') as f:
            lines=[]
            for n,line in enumerate(f):
                lines.append(line)
                if 'end of arguments' in line or n>=1000:break
        prefix=''.join(lines);values={}
        for k in keys:
            m=re.search(r'^\s*'+k+r'\s+\.+\s+(\S+)',prefix,re.M)
            if m:values[k]=m[1]
        if values:configs.append(dict(path=str(p),prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),values=values))
    assert configs,'No argument log'
    for x in configs:
        v=x['values']
        if 'moe_enable_deepep' in v:assert v['moe_enable_deepep']=='True',v
    v=next(x['values'] for x in configs if 'moe_enable_deepep' in x['values'])
    expected={'pipeline_model_parallel_size':'4','tensor_model_parallel_size':'1','context_parallel_size':'2','expert_model_parallel_size':'8','num_experts':'160','hidden_size':'5120','seq_length':'8192','global_batch_size':'64'}
    for k,val in expected.items():assert v.get(k)==val,(k,v.get(k),val)
    save(OUT/'inventory.json',dict(status='PASS',raw_root=str(raw),paths=paths,configuration=configs,ingest_sha256=sha(C['ingest_manifest']),config_sha256=sha(CONFIG)))
    print('INVENTORY PASS',raw,flush=True)
def tokens(stage):
    # Existing extractor anchors log call IDs to actual checkpoint timestamps,
    # verifies all EP8 sender/receiver counts and representative FC trace shapes.
    import extract_layer_iteration_grid_r1 as shared
    inv=load(OUT/'inventory.json');target=OUT/'tokens';target.mkdir(exist_ok=True)
    dest=target/f'pp{stage}-group0.json'
    assert not dest.exists(),'Existing token output: use Snakemake cache, do not overwrite'
    result=shared.process((stage,0,inv['paths'],dict(stage_width=8,microbatches=8,out=str(target),iterations=C['sample_iterations'])))
    print(result,flush=True)
def stability():
    parts=[load(OUT/f'tokens/pp{s}-group0.json') for s in [1,2]]
    rows=[r for p in parts for r in p['rows']];summary=[]
    for it in C['sample_iterations']:
        rr=[r for r in rows if r['iteration']==it];assert len(rr)==64
        summary.append(dict(iteration=it,rho=statistics.mean(r['retained']/r['offered'] for r in rr),tokens=statistics.mean(r['retained'] for r in rr),active_experts=statistics.mean(r['active_experts'] for r in rr)))
    checks=[]
    for stage in [1,2]:
        for layer in range(3+4*(stage-1),7+4*(stage-1)):
            vals=[statistics.mean(r['retained'] for r in rows if r['iteration']==it and r['stage']==stage and r['layer']==layer) for it in C['tail_iterations']]
            mean=statistics.mean(vals);cv=statistics.pstdev(vals)/mean;drift=abs(statistics.mean(vals[5:])-statistics.mean(vals[:5]))/mean
            checks.append(dict(stage=stage,layer=layer,mean_tokens=mean,cv=cv,half_drift=drift,pass_gate=cv<=C['stability_max_cv'] and drift<=C['stability_max_half_drift']))
    status='PASS' if all(x['pass_gate'] for x in checks) else 'PARTIAL_TAIL_NOT_STABLE'
    result=dict(status=status,summary=summary,layer_checks=checks,selected_iterations=C['tail_iterations'],selection='Predeclared155–200; no target data used. All8MB average per layer. Thresholds are engineering selection gates, not proof of stationarity.',thresholds={k:C[k] for k in ['stability_max_cv','stability_max_half_drift']},inputs={str(OUT/f'tokens/pp{s}-group0.json'):sha(OUT/f'tokens/pp{s}-group0.json') for s in [1,2]})
    save(OUT/'stability-review.json',result)
    print(status,checks,flush=True)
    assert status=='PASS','Tail gate failed; review source evidence before creating costs'
    save(OUT/'stability.json',result)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inventory','tokens','stability']);ap.add_argument('--pp',type=int);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='inventory':inventory()
    elif a.stage=='tokens':tokens(a.pp)
    else:stability()
if __name__=='__main__':main()

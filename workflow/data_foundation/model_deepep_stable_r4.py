"""New-source outer costs, guarded fine-DAG averaging, seal, and evaluation."""
import argparse
import collections
import copy
import gzip
import importlib.util
import json
import statistics
from pathlib import Path
from deepep_stable_pipeline_r1 import ROOT, C, OUT, load, save, sha
from build_b_cp_pair_r4 import run

BASE=ROOT/'results/data-foundation'
GPU_CODE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/workflow/data_foundation/pp32_gpu_boundary_r3.py')

def outer():
    from pp32_minimal_r1 import extract
    assert load(OUT/'stability.json')['status']=='PASS'
    inv=load(OUT/'inventory.json');dest=OUT/'outer';dest.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('newsource_gpu_boundary',GPU_CODE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.OLD=dest
    rows=[];raws={}
    for it in C['tail_iterations']:
        for rank in [0,8,16,24]:
            raw=Path(inv['paths'][str(it)][str(rank)]);out=dest/f'gpu-{it}-{rank}.json'
            if out.exists():
                result=load(out);assert result['raw_sha256']==sha(raw)
            else:
                obs=extract(raw,it,rank);assert obs['schedule_match'];save(dest/f'observation-{it}-{rank}.json',obs)
                result=mod.extract(it,rank);result.update(iteration=it,raw_path=str(raw),raw_sha256=obs['sha256']);save(out,result)
            rows.append(result);raws[str(raw)]=result['raw_sha256']
        print('OUTER',it,flush=True)
    roles={role:{ph:statistics.mean(o['end']-o['start'] for x in rows if x['stage']==stage for o in x['ops'] if o['kind']==ph) for ph in ['F','B']} for role,stage in [('first',0),('last',3)]}
    pairs=[]
    for it in C['tail_iterations']:
        messages=collections.defaultdict(list)
        for x in rows:
            if x['iteration']!=it:continue
            for o in x['ops']:
                for message,side in o['messages']:messages[message].append(dict(rank=x['rank'],side=side,**o))
        for message,ends in messages.items():
            assert len(ends)==2 and sorted(x['side'] for x in ends)==['recv','send']
            if not all(len(o['messages'])==1 for o in ends):continue
            ms=min(o['absolute_end_ms'] for o in ends)-max(o['device_ready_absolute_ms'] for o in ends)
            assert ms>0,(it,message,ms)
            pairs.append(dict(iteration=it,message=message,phase=message[0],ms=ms))
    params=dict(fb_ms=roles,pp_effective_overlap_ms={ph:statistics.median(x['ms'] for x in pairs if x['phase']==ph) for ph in ['F','B']},loss_tail_ms=0)
    save(OUT/'outer.json',dict(status='PASS',parameters=params,source_iterations=C['tail_iterations'],raw=raws,pp_pairs=pairs,code={str(GPU_CODE):sha(GPU_CODE)},limits=['Only representative ranks0/8/16/24, not world maximum.', 'PP effective overlap includes synchronization; not pure transmission.', 'Cross-host timestamp comparability remains assumed.', 'Last F already includes loss; zero additional loss slot avoids double accounting.']))

def seal():
    from extrapolate_detailed_fb_32to256_r1 import compose
    from pp_loss_slot_r1 import build
    assert load(OUT/'stability.json')['status']=='PASS'
    alignment=load(OUT/'aligned-r4/audit.json')
    assert alignment['status']=='PASS_SOURCE_ALIGNMENT_CONDITIONAL_DAG'
    for path,h in alignment['inputs'].items():assert sha(Path(path))==h
    models={p:load(OUT/f'aligned-r4/{p}.json') for p in ['F','B']}
    inputs={str(OUT/f'aligned-r4/{p}.json'):sha(OUT/f'aligned-r4/{p}.json') for p in ['F','B']}
    inputs[str(OUT/'aligned-r4/audit.json')]=sha(OUT/'aligned-r4/audit.json')
    params=copy.deepcopy(load(OUT/'outer.json')['parameters']);sizes={p:run(g)['END']['end_ms'] for p,g in models.items()};params['fb_ms']['middle']=sizes
    g,t,bindings=compose(C['target_pp'],C['target_microbatches'],params,models)
    predicted=t['s0:B3']['end_ms']-t['s0:F0']['start_ms'];contracted=build(16,4,params)
    diff=max(abs(t[k]['end_ms']-n['end_ms']) for k,n in contracted['nodes'].items());assert diff<1e-6
    assert len(bindings)==112 and all(g[b['node']]['cost_ms']==0 for b in bindings)
    out=OUT/'model';out.mkdir(exist_ok=True)
    for ph,graph in models.items():save(out/f'{ph}.json',graph)
    with gzip.open(out/'expanded-dag.json.gz','wt') as f:json.dump(g,f,separators=(',',':'))
    save(out/'outer-prediction.json',contracted);save(out/'parameters.json',params)
    perturb=[]
    for ph,kind in [('F','expert_envelope'),('F','CP_effective'),('B','expert_grouped_linear_backward_0'),('B','recompute_CP_effective')]:
        key=next(k for k,n in g.items() if k.startswith(f's1:{ph}0/') and n.get('kind')==kind)
        g[key]['cost_ms']+=1000;changed=run(g);g[key]['cost_ms']-=1000
        shift=changed['s0:B3']['end_ms']-t['s0:B3']['end_ms'];assert shift>0
        perturb.append(dict(node=key,increment_ms=1000,shift_ms=shift))
    save(OUT/'model-seal.json',dict(status='PASS_SOURCE_SEAL',predicted_ms=predicted,middle_ms=sizes,expanded_nodes=len(g),bindings=len(bindings),max_contraction_error_ms=diff,perturbation=perturb,target_timing_read=False,target_token_fit=False,source_inputs=inputs,files={str(p):sha(p) for p in out.iterdir() if p.is_file()}))

def evaluate():
    sealpath=OUT/'model-seal.json';seal=load(sealpath);digest=sha(sealpath)
    for p,h in seal['files'].items():assert sha(p)==h
    pred=seal['predicted_ms'];rows=[];inputs={}
    for it in C['target_iterations']:
        p=BASE/f'twenty-iteration-prediction-r1/window-{it}.json';w=load(p);assert w['rank']==0 and w['iteration']==it
        inputs[str(p)]=sha(p);rows.append(dict(iteration=it,truth_ms=w['window_ms'],predicted_ms=pred,error_percent=100*(pred/w['window_ms']-1)))
    def metrics(rs):
        e=[r['error_percent'] for r in rs];a=list(map(abs,e))
        return dict(n=len(e),signed_mean_percent=statistics.mean(e),signed_std_pp=statistics.pstdev(e),absolute_mean_percent=statistics.mean(a),absolute_std_pp=statistics.pstdev(a))
    assert sha(sealpath)==digest
    save(OUT/'evaluation.json',dict(status='PARTIAL_CONDITIONAL_SOURCE_TRANSFER_EVALUATED',prediction=seal,rows=rows,all20=metrics(rows),last4=metrics(rows[-4:]),target_inputs=inputs,source_seal_sha256=digest,limits=['1F1B rank0 GPU window incl fill/drain, not full-step MFU.', 'Historical target already seen; not blind evaluation.', 'Conditional cross-stream edges normalized using source-only invariance; omitted varying ordering retained as readiness costs, not event-verified causal edges.', 'Two target EP8 cohorts assumed equal.', 'Effective communication and reserved readiness costs, not pure network service.', 'Middle PP1 F0/B0 reused across stages/MB; does not predict unseen routing workloads.']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['outer','seal','evaluate']);a=ap.parse_args()
    {'outer':outer,'seal':seal,'evaluate':evaluate}[a.stage]()
if __name__=='__main__':main()

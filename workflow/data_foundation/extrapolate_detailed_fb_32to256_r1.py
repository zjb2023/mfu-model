"""Source-cost pipeline scaling; EP8 cohort symmetry is explicit, not measured."""
import argparse,copy,hashlib,json
from pathlib import Path
from pp_loss_slot_r1 import build as pipeline
from build_b_cp_pair_r4 import run
from pp32_to256_pipeline_r6 import target_window
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compose(pp,mb,params,templates):
    outer=pipeline(pp,mb,params)['nodes'];g={k:dict(cost_ms=n['duration_ms'],deps=n['dependencies'],kind=n['kind'],stage=n['stage']) for k,n in outer.items()};bindings=[]
    for s in range(1,pp-1):
        for m in range(mb):
            for ph in ['F','B']:
                key=f's{s}:{ph}{m}';deps=list(g[key]['deps']);prefix=key+'/'
                for k,n in templates[ph].items():
                    g[prefix+k]=dict(cost_ms=n['cost_ms'],deps=[prefix+d for d in n['deps']] if n['deps'] else deps,kind=n.get('kind','reserved'),stage=s,mb=m,phase=ph,source_node=k,template_rank=n.get('rank'),local_ep_rank=n['rank']-8 if n.get('rank') in range(8,16) else None)
                removed=g[key]['cost_ms'];g[key]=dict(cost_ms=0,deps=[prefix+'END'],kind=ph,stage=s)
                bindings.append(dict(node=key,entry_deps=deps,exit=prefix+'END',removed_scalar_ms=removed,wrapper_ms=0,scope='representative EP8 cohort; other cohort assumed identical'))
    return g,run(g),bindings
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    paths=[BASE/'four-layer-f-r1/graph.json',BASE/'b-four-layers-r10/graph.json',BASE/'pp32-detailed-fb-assembly-r1/parameters.json',BASE/'pp32-minimal-r1/configs.json']
    templates={'F':load(paths[0]),'B':load(paths[1])};params=load(paths[2]);configs=load(paths[3]);source=configs['source32']['config'];target=configs['target256']['config']
    same=['hidden_size','seq_length','micro_batch_size','num_experts','tensor_model_parallel_size','context_parallel_size','expert_model_parallel_size','recompute_num_layers','recompute_method','recompute_granularity','decoder_first_pipeline_num_layers','decoder_last_pipeline_num_layers']
    assert all(source[k]==target[k] for k in same)
    for c in [source,target]:assert (c['num_layers']-c['decoder_first_pipeline_num_layers']-c['decoder_last_pipeline_num_layers'])/(c['pipeline_model_parallel_size']-2)==4
    assert target['pipeline_model_parallel_size']==16 and target['microbatches_derived']==4
    edp=target['data_parallel_size']*target['context_parallel_size']//target['expert_model_parallel_size'];assert edp==2
    pp=16;mb=4;sizes={ph:run(g)['END']['end_ms'] for ph,g in templates.items()};scalar_params=copy.deepcopy(params);scalar_params['fb_ms']['middle']=sizes
    regression=pipeline(4,8,scalar_params)['window_ms'];old=load(BASE/'pp32-detailed-fb-assembly-r1/sealed-prediction.json')['window_ms'];assert abs(regression-old)<1e-6
    g,t,bindings=compose(pp,mb,params,templates);scalar=pipeline(pp,mb,scalar_params)
    for k,n in scalar['nodes'].items():assert abs(n['end_ms']-t[k]['end_ms'])<1e-6
    assert len(bindings)==112 and all(g[b['node']]['cost_ms']==0 for b in bindings)
    predicted=t['s0:B3']['end_ms']-t['s0:F0']['start_ms'];assert abs(predicted-scalar['window_ms'])<1e-6
    policy=dict(source_iteration=60,source_world=32,target_world=256,pp=16,microbatches=4,middle_stages=14,middle_layers=4,first_last_layers=2,target_EDP=edp,stage_ranks=16,modeled_cohort=8,cohort_policy='two EP8 cohorts assumed equal cost; only representative cohort expanded; no replica imbalance or full-world makespan claim',PP_policy='source32 effective rendezvous costs retained, no target timing fit',window='rank0 first F associated GPU start to last B associated GPU end; pipeline fill/drain included, RS/OPT/AG excluded',matched_config_keys=same)
    estimate=dict(window_ms=predicted,middle_cost_ms=sizes,source32_regression_ms=regression)
    for name,x in [('graph',g),('replay',t),('bindings',bindings),('parameters',params),('policy',policy),('config-snapshot',configs),('sealed-prediction',estimate),('outer-prediction',scalar)]:dump(out/(name+'.json'),x)
    sealed={n:sha(out/n) for n in ['graph.json','replay.json','parameters.json','policy.json','sealed-prediction.json','outer-prediction.json']};dump(out/'seal.json',dict(files=sealed,target_timing_read_in_this_run=False,target_has_prior_exposure=True))
    print('SEALED',predicted,flush=True)
    # Perturb an actual internal expert node; verify it reaches the outer chain.
    key=next(k for k,n in g.items() if k.startswith('s1:B0/') and n['kind'].startswith('expert_grouped'))
    baseline=g[key]['cost_ms'];g[key]['cost_ms']+=1000;changed=run(g);g[key]['cost_ms']=baseline
    assert changed['s0:B3']['end_ms']>t['s0:B3']['end_ms'];shift=changed['s0:B3']['end_ms']-t['s0:B3']['end_ms'];del changed
    # Evaluator opens only after seal: reproduce one existing rank0 trace window.
    cache=BASE/'pp32-to256-pipeline-r6/target-observation.json';cached=load(cache)
    truth=target_window(Path(cached['path']));assert truth['sha256']==cached['sha256']
    assert abs(truth['window_ms']-cached['window_ms'])<1e-6
    for x,y in zip(truth['blocks'],cached['blocks']):assert x==y
    comparisons=[]
    for b in truth['blocks']:
        key=f's0:{b["phase"]}{b["mb"]}';n=scalar['nodes'][key]
        comparisons.append(dict(node=key,phase=b['phase'],mb=b['mb'],predicted_start_ms=n['start_ms'],predicted_end_ms=n['end_ms'],trace_start_ms=b['start_ms'],trace_end_ms=b['end_ms'],end_error_ms=n['end_ms']-b['end_ms']))
    assert all(sha(out/n)==h for n,h in sealed.items())
    checks=dict(status='PASS_ASSEMBLY_AND_EVALUATOR',instances=len(bindings),nodes=len(g),source32_regression='PASS',contraction='PASS all outer endpoints',double_charge='PASS zero-cost replacements',loss_tail_ms=params['loss_tail_ms'],expert_plus1000_window_shift_ms=shift,seal_verified=True,target_cache_reproduced=True)
    report=dict(status='PARTIAL_CONDITIONAL_256_EXTRAPOLATION',scope=policy['window'],predicted_ms=predicted,trace_ms=truth['window_ms'],error_ms=predicted-truth['window_ms'],ARE_percent=100*abs(predicted-truth['window_ms'])/truth['window_ms'],prediction=estimate,checks=checks,target_fitted=False,blind=False,accuracy_acceptance='NO_PREDEFINED_THRESHOLD',limitations=['EP8 cohort symmetry excludes target EDP2 load imbalance','outer PP stage completion is a simplified collective boundary, not rank-resolved overlap','PP effective costs transferred from source32 without target topology calibration','same source F0/B0 costs and readiness reused in all middle stages/microbatches','first/last scalar role costs inherited from source32','one target iteration60 rank0 evaluated, not full256 rank makespan or MFU','existing target data previously exposed; not blind validation'])
    for name,x in [('target-observation',truth),('comparison',comparisons),('checks',checks),('report',report)]:dump(out/(name+'.json'),x)
    def rec(p):return dict(path=str(p.resolve()),sha256=sha(p))
    paths += [BASE/'pp32-detailed-fb-assembly-r1/sealed-prediction.json',cache]
    codes=[Path(__file__),Path(__file__).with_name('pp_loss_slot_r1.py'),Path(__file__).with_name('pp32_to256_pipeline_r6.py'),Path(__file__).with_name('build_b_cp_pair_r4.py'),Path(__file__).with_name('pp32_minimal_r1.py'),Path(__file__).with_name('pp32_rendezvous_r2.py')]
    dump(out/'manifest.json',dict(inputs=[rec(p) for p in paths]+[dict(path=truth['path'],sha256=truth['sha256'])],code=[rec(p) for p in codes],sealed=sealed,outputs=[rec(p) for p in out.glob('*.json')]))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

"""Replace middle-stage scalar envelopes by frozen eight-rank subgraphs."""
import argparse,copy,hashlib,json
from pathlib import Path
from pp_loss_slot_r1 import build as pipeline
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def compose(params,templates):
    skeleton=pipeline(4,8,params)['nodes'];g={};bindings=[]
    for k,n in skeleton.items():g[k]=dict(cost_ms=n['duration_ms'],deps=n['dependencies'],kind=n['kind'],stage=n['stage'])
    for stage in [1,2]:
        for mb in range(8):
            for phase in ['F','B']:
                key=f's{stage}:{phase}{mb}';prior=list(g[key]['deps']);template=templates[phase];prefix=key+'/'
                for tk,tn in template.items():
                    deps=[prefix+d for d in tn['deps']] if tn['deps'] else prior
                    g[prefix+tk]=dict(cost_ms=tn['cost_ms'],deps=deps,kind=tn.get('kind','reserved'),stage=stage,mb=mb,phase=phase,source_node=tk,rank=(stage*8+tn['rank']-8) if tn.get('rank') in range(8,16) else None)
                old=g[key]['cost_ms'];g[key]=dict(cost_ms=0,deps=[prefix+'END'],kind=phase,stage=stage,module_exit=True)
                bindings.append(dict(node=key,phase=phase,stage=stage,mb=mb,entry_dependencies=prior,exit=prefix+'END',removed_scalar_ms=old,new_wrapper_cost_ms=0))
    t=run(g)
    return g,t,bindings
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    pp2=BASE/'pp2-module-compatibility-r1';assert load(pp2/'summary.json')['status'].startswith('PASS')
    inputs=[BASE/'four-layer-f-r1/graph.json',BASE/'b-four-layers-r10/graph.json',BASE/'pp32-gpu-boundary-r3b/parameters.json',pp2/'summary.json',pp2/'manifest.json']
    templates={'F':load(inputs[0]),'B':load(inputs[1])};params=load(inputs[2]);params={k:params[k] for k in ['fb_ms','pp_effective_overlap_ms']};params['loss_tail_ms']=0
    g,t,bindings=compose(params,templates)
    sizes={ph:run(tg)['END']['end_ms'] for ph,tg in templates.items()}
    equivalent=copy.deepcopy(params);equivalent['fb_ms']['middle']=sizes
    scalar=pipeline(4,8,equivalent)
    for k,n in scalar['nodes'].items():assert abs(t[k]['end_ms']-n['end_ms'])<1e-6,(k,t[k],n)
    assert len(bindings)==32 and all(g[b['node']]['cost_ms']==0 for b in bindings)
    module_windows=[]
    for b in bindings:
        start=max(t[d]['end_ms'] for d in b['entry_dependencies']);end=t[b['node']]['end_ms']
        assert abs(end-start-sizes[b['phase']])<1e-6
        module_windows.append(dict(stage=b['stage'],mb=b['mb'],phase=b['phase'],start_ms=start,end_ms=end,duration_ms=end-start))
    estimates=dict(window_ms=t['s0:B7']['end_ms']-t['s0:F0']['start_ms'],middle_cost_ms=sizes)
    for n,v in [('graph',g),('replay',t),('bindings',bindings),('parameters',params),('module-windows',module_windows),('sealed-prediction',estimates)]:dump(out/(n+'.json'),v)
    seal={n:sha(out/n) for n in ['graph.json','replay.json','parameters.json','sealed-prediction.json']};dump(out/'seal.json',dict(files=seal,target_read_in_this_run=False))
    # Cost propagation smoke: no trace or target adjustment.
    key=next(k for k,n in g.items() if k.startswith('s1:B0/') and n['kind'].startswith('expert_grouped'))
    perturb=copy.deepcopy(g);perturb[key]['cost_ms']+=1000;pt=run(perturb)
    assert pt['s1:B0']['end_ms']>t['s1:B0']['end_ms'] and pt['s0:B7']['end_ms']>t['s0:B7']['end_ms']
    reports=[];timeline=[]
    for it,file in [(60,'source-observations.json'),(70,'target-observations.json')]:
        p=BASE/'pp32-gpu-boundary-r3b'/file;inputs.append(p);obs=load(p)
        first=next(x for x in obs if x['rank']==0);fb=[x for x in first['ops'] if x['kind'] in ['F','B']]
        truth=fb[-1]['end']-fb[0]['start'];pred=estimates['window_ms']
        reports.append(dict(iteration=it,rank=0,predicted_ms=pred,trace_ms=truth,error_ms=pred-truth,ARE_percent=100*abs(pred-truth)/truth))
        origin=fb[0]['absolute_start_ms']
        for x in obs:
            for op in x['ops']:
                if op['kind'] not in ['F','B']:continue
                node=op['id'];n=scalar['nodes'][node]
                timeline.append(dict(iteration=it,rank=x['rank'],stage=x['stage'],node=node,phase=op['kind'],predicted_start_ms=n['start_ms'],predicted_end_ms=n['end_ms'],trace_start_ms=op['absolute_start_ms']-origin,trace_end_ms=op['absolute_end_ms']-origin))
    assert all(sha(out/n)==h for n,h in seal.items())
    checks=dict(status='PASS_ASSEMBLY_PARTIAL_FULL_WORLD_MODEL',module_instances=32,graph_nodes=len(g),scalar_contraction='PASS every skeleton endpoint',double_charge='PASS zero-cost wrappers replace old scalars',cost_propagation=dict(node=key,plus_ms=1000,window_shift_ms=pt['s0:B7']['end_ms']-t['s0:B7']['end_ms']),seal_verified=True)
    report=dict(scope='32GPU PP4 m8; PP0 first F device start -> last B device end, including pipeline fill/drain; excludes RS/OPT/AG',prediction=estimates,evaluation=reports,checks=checks,target_fitted=False,blind=False,PP2_policy='PP1 detailed source60 templates reused after source60 PP2 signature checks; no PP2 fine-grained cost calibration',limitations=['stage-group completion and PP rendezvous are simplified, not full-world rank-resolved PP overlap','same F0/B0 source template reused across all microbatches and both middle stages','first/last stage keep source60 scalar costs; CP/EP already included inside FB','source reserved readiness remains empirical','target cached GPU windows use prior runtime-only association; current detailed modules use runtime+driver','cross-host timestamp alignment conditional; per-stage timeline comparison diagnostic only','no new256 prediction'])
    for n,v in [('report',report),('timeline',timeline),('checks',checks)]:dump(out/(n+'.json'),v)
    def rec(p):return dict(path=str(p.resolve()),sha256=sha(p))
    dump(out/'manifest.json',dict(inputs=[rec(p) for p in inputs],code=[rec(Path(__file__)),rec(Path(__file__).with_name('pp_loss_slot_r1.py')),rec(Path(__file__).with_name('pp32_to256_pipeline_r6.py')),rec(Path(__file__).with_name('build_b_cp_pair_r4.py'))],sealed=seal,outputs=[rec(p) for p in out.glob('*.json')]))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

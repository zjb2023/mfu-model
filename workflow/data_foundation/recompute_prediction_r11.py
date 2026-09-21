"""Expand r11 B into source32 and target256 pipelines; seal before evaluation.

Persist factorized templates + constructor + bindings instead of duplicating
hundreds of thousands of identical inner nodes in a monolithic JSON artifact.
"""
import argparse,copy,hashlib,json,gc
from pathlib import Path
from extrapolate_detailed_fb_32to256_r1 import compose
from pp_loss_slot_r1 import build as pipeline
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 paths=[BASE/'four-layer-f-r1/graph.json',BASE/'b-recompute-r11/graph.json',BASE/'pp32-detailed-fb-assembly-r1/parameters.json',BASE/'full-step-32to256-r1/sealed-prediction.json',BASE/'pp32-minimal-r1/configs.json']
 inputs={str(p):sha(p) for p in paths};templates={'F':load(paths[0]),'B':load(paths[1])};params=load(paths[2]);full_source=load(paths[3]);configs=load(paths[4])
 assert configs['target256']['config']['pipeline_model_parallel_size']==16 and configs['target256']['config']['microbatches_derived']==4
 sizes={ph:run(g)['END']['end_ms'] for ph,g in templates.items()};scalar_params=copy.deepcopy(params);scalar_params['fb_ms']['middle']=sizes
 expanded_checks=[];outer=None;target_bindings=None
 for pp,mb in [(4,8),(16,4)]:
  graph,t,bindings=compose(pp,mb,params,templates);scalar=pipeline(pp,mb,scalar_params)
  maxerr=max(abs(t[k]['end_ms']-v['end_ms']) for k,v in scalar['nodes'].items());assert maxerr<1e-6
  assert all(graph[b['node']]['cost_ms']==0 for b in bindings)
  window=t[f's0:B{mb-1}']['end_ms']-t['s0:F0']['start_ms'];assert abs(window-scalar['window_ms'])<1e-6
  row=dict(pp=pp,microbatches=mb,nodes=len(graph),instances=len(bindings),window_ms=window,max_contraction_error_ms=maxerr,zero_cost_wrappers=True)
  if pp==16:
   key=next(k for k,n in graph.items() if k.startswith('s1:B0/R0:') and n['kind']=='recompute_expert_FC1');graph[key]['cost_ms']+=100
   perturbed=run(graph);row['recompute_FC1_plus100_window_shift_ms']=perturbed['s0:B3']['end_ms']-t['s0:B3']['end_ms'];assert row['recompute_FC1_plus100_window_shift_ms']>=-1e-6
   del perturbed;graph[key]['cost_ms']+=900;perturbed=run(graph);row['recompute_FC1_plus1000_window_shift_ms']=perturbed['s0:B3']['end_ms']-t['s0:B3']['end_ms'];assert row['recompute_FC1_plus1000_window_shift_ms']>0
   del perturbed;outer=scalar;target_bindings=bindings
  expanded_checks.append(row);print('expanded PASS',row,flush=True);del graph,t;gc.collect()
 window=outer['window_ms'];cost=full_source['costs'];phys=full_source['physics'];prof=cost['entry_ms']+window+cost['post_last_b_ms'];train=prof+cost['training_minus_rank0_profiler_ms']
 def mfu(ms):return 100*phys['model_flops_per_iteration']/(phys['world_size']*phys['peak_flops_per_gpu_s']*ms/1000)
 prediction=copy.deepcopy(full_source);prediction.update(one_f_one_b_ms=window,profiler_step_ms=prof,training_step_ms=train,mfu_profiler_percent=mfu(prof),mfu_training_percent=mfu(train),B_model='b-recompute-r11',middle_cost_ms=sizes)
 dump(a.out/'sealed-prediction.json',prediction);dump(a.out/'outer-prediction.json',outer);dump(a.out/'bindings.json',target_bindings);dump(a.out/'expanded-checks.json',expanded_checks)
 seals={n:sha(a.out/n) for n in ['sealed-prediction.json','outer-prediction.json','bindings.json','expanded-checks.json']};dump(a.out/'seal.json',dict(files=seals,target_timing_read_in_this_run=False,prior_target_exposure=True))
 # Cached target evaluator: no trace rescan and no fitted parameters.
 oldfullp=BASE/'full-step-32to256-r1/report.json';olddagp=BASE/'detailed-fb-32to256-r1/report.json';old32p=BASE/'pp32-detailed-fb-assembly-r1/sealed-prediction.json'
 oldfull=load(oldfullp);olddag=load(olddagp);assert abs(expanded_checks[0]['window_ms']-load(old32p)['window_ms'])<1e-6
 def compare(p,t):return dict(predicted=p,observed=t,error=p-t,relative_error_percent=100*(p/t-1))
 report=copy.deepcopy(oldfull);report.update(status='PARTIAL_R11_REASSEMBLY_CACHED_EVALUATION',prediction=prediction,profiler=compare(prof,oldfull['profiler']['observed']),training=compare(train,oldfull['training']['observed']),mfu_profiler=compare(mfu(prof),oldfull['mfu_profiler']['observed']),mfu_training=compare(mfu(train),oldfull['mfu_training']['observed']),one_f_one_b=compare(window,olddag['trace_ms']),expanded_checks=expanded_checks)
 for row in report['components']:
  if row['name']=='1F1B':row['source_prediction_ms']=window
 report['difference_from_r10_ms']=window-olddag['predicted_ms'];assert abs(report['difference_from_r10_ms'])<1e-6
 report['limitations']+=['R11 expands recompute costs but default durations unchanged; no accuracy gain claimed','Cached evaluator reuses exposed iter60 rank0 target data, not blind or full256-rank validation','Factorized templates and compose function reproduce expanded graph; not stored monolithically','No new optimizer tail or expert token calibration applied']
 dump(a.out/'report.json',report)
 assert all(sha(a.out/n)==h for n,h in seals.items());assert all(sha(Path(p))==h for p,h in inputs.items())
 paths += [oldfullp,olddagp,old32p]
 codes=[Path(__file__).resolve()]+[Path(__file__).with_name(n) for n in ['extrapolate_detailed_fb_32to256_r1.py','pp_loss_slot_r1.py','pp32_to256_pipeline_r6.py','pp32_minimal_r1.py','pp32_rendezvous_r2.py','build_b_cp_pair_r4.py']]
 dump(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in paths},code={str(p):sha(p) for p in codes},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')},representation='factorized; graph templates + compose(pp,mb,params,templates); expanded and checked in memory'))
 print(json.dumps({k:report[k] for k in ['one_f_one_b','profiler','mfu_profiler','difference_from_r10_ms']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

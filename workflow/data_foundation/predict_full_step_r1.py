"""Transfer source32 entry/tail/log residual; seal before target evaluation."""
import argparse
import hashlib
import json
import re
import shutil
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
OLD=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation')
PHYSICS=Path('/home/zjb/Desktop/mfu-model/case_256gpu_pp16_cp2_a2a/results/mfu_timeline/mfu_timeline_summary.json')
CLOCKS=Path('/home/zjb/Desktop/mfu-model/case_256gpu_pp16_cp2_a2a/data/iteration_clocks.csv')
SOURCE_LOG=Path('/home/zjb/Desktop/32/gpu32_gbs64_framework/2026-09-18-10:38/worker33017/2026-09-18_1038/tp1_pp4_dp_mbs2_numbs_gbs64_gpus0_mtp1_forcelbfalse_pertensorfalse_NO_LOSS_REDUCE.RANK3.10.124.33.17.log')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--publish',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    obs_path=BASE/'pp32-gpu-boundary-r3b/source-observations.json'
    pred_path=BASE/'detailed-fb-32to256-r1/sealed-prediction.json'
    phys=load(PHYSICS)
    assert phys['parallel']['world_size']==256
    numerator=phys['model_flops_per_iteration'];peak=phys['peak_tflops_per_gpu']*1e12
    obs=next(x for x in load(obs_path) if x['rank']==0)
    blocks=[x for x in obs['ops'] if x['kind'] in ['F','B']]
    entry=blocks[0]['start'];tail=obs['profiler_ms']-blocks[-1]['end']
    meta_path=OLD/'pp32-structure-r4/rank-0.json';meta=load(meta_path);raw_path=Path(meta['path'])
    assert sha(raw_path)==meta['sha256']
    es=load(raw_path)['traceEvents'];step=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep'))
    assert abs(step['dur']/1000-obs['profiler_ms'])<1e-5
    assert abs((blocks[0]['absolute_start_ms']-step['ts']/1000)-entry)<1e-5
    matches=[(i+1,l) for i,l in enumerate(SOURCE_LOG.read_text().splitlines()) if re.search(r'iteration\s+60/',l)]
    assert len(matches)==1
    line_no,line=matches[0];training=float(re.search(r'elapsed time per iteration \(ms\):\s*([\d.]+)',line)[1])
    residual=training-obs['profiler_ms'];assert residual>=0
    window=load(pred_path)['window_ms'];profiler=entry+window+tail;train_pred=profiler+residual
    def mfu(t):return 100*numerator/(256*peak*t/1000)
    costs=dict(entry_ms=entry,post_last_b_ms=tail,training_minus_rank0_profiler_ms=residual,
               source_iteration=60,source_rank=0,source_profiler_label=step['name'],source_training_ms=training,
               source_profiler_ms=obs['profiler_ms'],source_log=str(SOURCE_LOG),source_log_line=line_no,
               policy='copy source32 scalar envelopes with coefficient1; no target timing used to fit; tail includes update and post-update sync; log residual is timing convention/outer overhead, not proven pure overhead')
    prediction=dict(one_f_one_b_ms=window,profiler_step_ms=profiler,training_step_ms=train_pred,
                    mfu_profiler_percent=mfu(profiler),mfu_training_percent=mfu(train_pred),
                    costs=costs,physics=dict(model_flops_per_iteration=numerator,world_size=256,peak_flops_per_gpu_s=peak,
                    source=str(PHYSICS),basis='legacy256 effective FLOPs inferred from training logs, not independently architectural; 500TFLOP/s convention, not new hardware validation'),
                    target_timing_fit=False,blind=False,scope='rank0 profiler proxy and source-log training clock; not full-world optimizer DAG',
                    source_training_record=line)
    dump(a.out/'sealed-prediction.json',prediction);seal=sha(a.out/'sealed-prediction.json')
    # Target only after seal; historical exposed target data are evaluator-only.
    overview_path=BASE/'prediction-overview-r1/data.json';overview=load(overview_path);target=overview['evidence']
    clock=next(r for r in csv.DictReader(CLOCKS.open()) if int(r['iteration'])==60)
    target_train=float(clock['training_log_ns'])/1e6
    target_prof=target['profiler_ms']
    def compare(p,t):return dict(predicted=p,observed=t,error=p-t,relative_error_percent=100*(p/t-1))
    result=dict(status='PARTIAL_SOURCE32_FULL_STEP_EXTRAPOLATION',prediction=prediction,
                profiler=compare(profiler,target_prof),training=compare(train_pred,target_train),
                mfu_profiler=compare(mfu(profiler),mfu(target_prof)),mfu_training=compare(mfu(train_pred),mfu(target_train)),
                components=[dict(name='启动',source_prediction_ms=entry,target_observed_ms=target['startup_ms']),
                            dict(name='1F1B',source_prediction_ms=window,target_observed_ms=target['one_f_one_b_ms']),
                            dict(name='末B后全部尾段',source_prediction_ms=tail,target_observed_ms=target['tail_ms']),
                            dict(name='日志时钟外层残余',source_prediction_ms=residual,target_observed_ms=target_train-target_prof)],
                clock_note='target historical iteration_clocks profiler is world-envelope; this report keeps rank0 profiler and derives target log residual against that same boundary',
                limitations=['DP4→8 / EDP1→2 update-tail cost copied without scaling; not physically validated',
                             'source training log clock copied from iter60 includes possible profiling overhead; not universal constant',
                             'startup and F/B biases may cancel; do not attribute low total error to accurate components',
                             'MFU uses legacy effective FLOPs convention, not independent physical validation',
                             '256 target previously exposed; single target iteration, no blind validation'])
    assert abs(sum(x['source_prediction_ms']-x['target_observed_ms'] for x in result['components'])-result['training']['error'])<1e-5
    assert sha(a.out/'sealed-prediction.json')==seal
    dump(a.out/'report.json',result);dump(a.out/'seal.json',dict(sha256=seal,target_timing_read_after_seal=True,prior_target_exposure=True))
    dump(a.out/'validation.json',dict(status='PASS_ACCOUNTING_AND_SEAL',source_raw_hash=True,source_clock_match=True,component_sum=True,seal_unchanged=True,independent_physical_validation=False))
    # Preserve overview r1; add a source-only full-step version with original diagrams intact.
    html=(BASE/'prediction-overview-r1/index.html').read_text()
    extra=Path(__file__).with_name('full_step_overview_r2.js')
    html=html.replace('</body>', '<script>const FULL='+json.dumps(result,ensure_ascii=False).replace('</','<\\/')+';\n'+extra.read_text()+'</script></body>')
    (a.out/'index.html').write_text(html)
    shutil.copyfile(overview_path,a.out/'data.json')
    inputs=[obs_path,pred_path,PHYSICS,CLOCKS,SOURCE_LOG,meta_path,raw_path,overview_path,BASE/'prediction-overview-r1/index.html']
    dump(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in inputs},code={str(p.resolve()):sha(p) for p in [Path(__file__),extra]},outputs={str(p.resolve()):sha(p) for p in a.out.iterdir() if p.is_file()}))
    if a.publish:shutil.copytree(a.out,a.publish)
    print(json.dumps({k:v for k,v in result.items() if k in ['status','profiler','training','mfu_training','components']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

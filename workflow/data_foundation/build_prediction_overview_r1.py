"""Assemble a trace-backed report; no prediction fitting or baseline mutation."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/data-foundation'
OLD = Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation')

def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--publish', type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    src = BASE/'detailed-fb-32to256-r1'
    paths = [src/'report.json', src/'outer-prediction.json', src/'parameters.json',
             src/'config-snapshot.json', OLD/'pp256-stage-audit-r7/rank-0.json',
             BASE/'pp-scale-candidate-r1-verified/report.json']
    seals = {str(p): sha(p) for p in paths}
    report, outer, params, configs, cached, candidate = map(load, paths)
    raw = Path(cached['path'])
    raw_bytes = raw.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == cached['sha256']
    es = json.loads(raw_bytes)['traceEvents']
    del raw_bytes
    start = cached['blocks'][0]['start_absolute_ms']
    finish = cached['blocks'][-1]['end_absolute_ms']
    step_i, step = next((i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep'))
    step_start, step_end = step['ts']/1000, (step['ts']+step['dur'])/1000
    gpu = [(i,e) for i,e in enumerate(es) if e.get('cat')=='kernel' and finish <= e.get('ts',0)/1000 <= step_end]
    ag = [(i,e) for i,e in gpu if 'AllGather' in e['name']]
    ag_i, ag_last = max(ag, key=lambda x:x[1]['ts']+x[1]['dur'])
    ag_end = (ag_last['ts']+ag_last['dur'])/1000
    truth = finish-start
    assert abs(truth-report['trace_ms']) < 1e-5
    startup, update, post = start-step_start, ag_end-finish, step_end-ag_end
    total = step_end-step_start
    assert min(startup,update,post)>0 and abs(startup+truth+update+post-total)<1e-5
    names = ['finalize_model_grads','should_run_forward_backward','step','prepare_grads',
             'get_grad_norm','step_with_ready_grads','Optimizer.step#FusedAdam.step',
             'logical_and_across_model_parallel_group','reduce_max_stat_across_model_parallel_group']
    annotations = [dict(event=i,name=e['name'],start_ms=e['ts']/1000-start,
                        end_ms=(e['ts']+e['dur'])/1000-start,basis='CPU envelope; not GPU completion')
                   for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name') in names and e.get('ts',0)/1000>finish-100]
    comm_events = [dict(event=i,name=e['name'],start_ms=e['ts']/1000-start,end_ms=(e['ts']+e['dur'])/1000-start)
                   for i,e in gpu if any(k in e['name'] for k in ['AllGather','ReduceScatter','AllReduce'])]
    evidence = dict(raw_path=str(raw),sha256=cached['sha256'],directory_iteration=60,
                    profiler_label=step['name'],profiler_event=step_i,last_ag_event=ag_i,
                    time_origin='rank0 first F associated GPU start; ms',
                    startup_ms=startup,one_f_one_b_ms=truth,to_last_ag_ms=update,
                    after_ag_ms=post,tail_ms=update+post,profiler_ms=total,
                    cpu_annotations=annotations,tail_gpu_collectives=comm_events,
                    scope='rank0 only; last AG observed, not proven full-world optimizer completion')
    hybrid = startup+report['predicted_ms']+update+post
    metrics = dict(full_prediction_ms=None,full_prediction_error_percent=None,
                   status='PARTIAL_FULL_ITERATION_NOT_PREDICTED',
                   hybrid_ms=hybrid,hybrid_error_percent=100*(hybrid/total-1),
                   hybrid_policy='target-observed startup and tail + frozen source-only 1F1B prediction; NOT independent full iteration prediction',
                   one_f_one_b_prediction_ms=report['predicted_ms'],one_f_one_b_error_percent=100*(report['predicted_ms']/truth-1))
    data = dict(metrics=metrics,evidence=evidence,outer=outer,params=params,configs=configs,candidate=candidate)
    dump(args.out/'data.json',data)
    template = Path(__file__).with_name('prediction_overview_r1.html')
    html = template.read_text().replace('__REPORT_DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (args.out/'index.html').write_text(html)
    assert all(sha(Path(p))==s for p,s in seals.items())
    dump(args.out/'validation.json',dict(status='PASS_DATA_ASSEMBLY',raw_sha_verified=True,
         interval_sum_verified=True,one_f_one_b_matches_frozen=True,frozen_inputs_unchanged=True,
         full_prediction_available=False,fit_performed=False))
    dump(args.out/'manifest.json',dict(inputs=seals,raw={str(raw):cached['sha256']},
         code={str(p):sha(p) for p in [Path(__file__).resolve(),template]},
         outputs={str(p.resolve()):sha(p) for p in args.out.iterdir() if p.is_file()}))
    if args.publish:
        shutil.copytree(args.out,args.publish)  # Refuse overwrite, preserve original pages.
        assert sha(args.out/'index.html')==sha(args.publish/'index.html')
    print(json.dumps(dict(metrics=metrics,evidence={k:v for k,v in evidence.items() if k not in ['cpu_annotations','tail_gpu_collectives']}),ensure_ascii=False,indent=2))

if __name__ == '__main__': main()

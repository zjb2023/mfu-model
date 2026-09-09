"""Frozen source device costs on target CPU-conditioned features; posthoc only."""
import json
import resource
from time import perf_counter

import pandas as pd
from guards import ROOT
from smoke_worker import dump,sha
from worker import csv
from device_queue import prepare,stream_audit,DTYPE
from device_queue_model import feature_frame,METHODS,FEATURES
from device_queue_reload import load_parameters


def score_events(predictions,truth):
    columns=['event_id','iteration','rank','start_ns','end_ns','duration_ns','family','category',
             'launch_FB_phase','launch_FB_window_id','split','observed_enqueue_remainder_ns']
    scored=predictions.merge(truth[columns],on=['event_id','iteration','rank'],validate='many_to_one')
    scored['start_error_ns']=scored.predicted_start_ns-scored.start_ns
    scored['duration_error_ns']=scored.predicted_duration_ns-scored.duration_ns
    scored['end_error_ns']=scored.predicted_end_ns-scored.end_ns
    assert scored.end_error_ns.eq(scored.start_error_ns+scored.duration_error_ns).all()
    return scored


def event_metrics(group):
    return pd.Series(dict(events=len(group),start_MAE_ms=group.start_error_ns.abs().mean()/1e6,
        end_MAE_ms=group.end_error_ns.abs().mean()/1e6,end_p95_AE_ms=group.end_error_ns.abs().quantile(.95)/1e6,
        start_bias_ms=group.start_error_ns.mean()/1e6,duration_bias_ms=group.duration_error_ns.mean()/1e6,
        end_bias_ms=group.end_error_ns.mean()/1e6,duration_MAE_ms=group.duration_error_ns.abs().mean()/1e6))


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='evaluator' and not plan['variants']
    review=json.loads((ROOT/plan['resource_review_file']).read_text())
    assert review['source_reload_gate']==plan['source_model_reload_gate'] and review['fixed_methods']==METHODS
    parameters=pd.read_csv(paths['source_device_queue_parameters.csv.gz'],dtype={'fit_iterations':'string','key':'string','cost_id':'string','level':'string'})
    model=load_parameters(parameters);parameter_hash=sha(paths['source_device_queue_parameters.csv.gz'])
    windows=pd.read_csv(paths['target_runtime_windows.csv.gz'],dtype=DTYPE,low_memory=False)
    windows=windows[windows['case'].eq('target224')]
    assert set(windows.iteration)=={85,90,95,100} and set(windows['rank'])=={16}
    data=[];audits=[];edges=[]
    for iteration in [85,90,95,100]:
        gpu=pd.read_csv(paths[f'target{iteration}_GPU.csv.gz'],dtype=DTYPE,low_memory=False)
        runtime=pd.read_csv(paths[f'target{iteration}_runtime.csv.gz'],dtype=DTYPE,low_memory=False)
        cpu=pd.read_csv(paths[f'target{iteration}_CPU.csv.gz'],dtype=DTYPE,low_memory=False)
        prepared=prepare(gpu,runtime,cpu,windows[windows.iteration.eq(iteration)])
        prepared['split']='target_development_posthoc'
        audit,ordered,edge=stream_audit(prepared)
        data.append(ordered);audits.append(audit);edges.append(edge)
    data=pd.concat(data,ignore_index=True);audits=pd.concat(audits,ignore_index=True);edges=pd.concat(edges,ignore_index=True)
    csv(out,'evaluator_only/target_stream_order_audit.csv',audits)
    csv(out,'evaluator_only/target_candidate_stream_edges.csv.gz',edges)
    negative=data[data.start_ns.lt(data.release_proxy_ns)].copy()
    negative['GPU_start_minus_release_proxy_ns']=negative.start_ns-negative.release_proxy_ns
    csv(out,'evaluator_only/target_release_proxy_violations.csv',negative[[
        'iteration','event_id','category','family','name','CPU_owner_name_checked','stream','unique_runtime_name',
        'runtime_start_ns','runtime_end_ns','start_ns','release_proxy_ns','GPU_start_minus_release_proxy_ns',
        'launch_FB_phase','launch_FB_window_id','launch_EP_semantic_region']])
    assert audits.CPU_order_unambiguous.all(),'Target CPU submission order ambiguous; no source-model stream remapping'
    features=feature_frame(data)
    assert set(features.columns)==set(FEATURES) and len(features)==33816
    # Event timings have already been observed for association and audit. They
    # must not alter the fixed prediction feature projection or model inputs.
    mutated=data.copy()
    for name in ['start_ns','end_ns','duration_ns','observed_enqueue_remainder_ns']:
        mutated[name]=mutated[name]+123456789
    pd.testing.assert_frame_equal(feature_frame(mutated),features,check_exact=True)
    del mutated
    unsupported=[];bindings=[]
    for row in features.to_dict('records'):
        try:p=model.lookup(row)
        except ValueError:
            unsupported.append({k:row[k] for k in ['iteration','event_id','category','stream','family','name','CPU_owner_input_dims']});continue
        bindings.append(dict(event_id=row['event_id'],iteration=row['iteration'],cost_id=p['cost_id'],cost_level=p['level']))
    csv(out,'evaluator_only/target_unsupported_cost_features.csv',pd.DataFrame(unsupported,
        columns=['iteration','event_id','category','stream','family','name','CPU_owner_input_dims']))
    assert not unsupported,'No source-supported target cost: see unsupported feature table; no zero-cost fallback'
    predictions=model.predict(features,METHODS)
    assert len(predictions)==4*len(features)
    destination=out/'evaluator_only'
    csv(destination,'target_device_queue_features.csv.gz',features)
    csv(destination,'target_device_queue_predictions_sealed.csv.gz',predictions)
    (destination/'source_device_queue_parameters_used.csv.gz').write_bytes(paths['source_device_queue_parameters.csv.gz'].read_bytes())
    assert sha(destination/'source_device_queue_parameters_used.csv.gz')==parameter_hash
    seal_files=['source_device_queue_parameters_used.csv.gz','target_device_queue_features.csv.gz','target_device_queue_predictions_sealed.csv.gz']
    dump(destination/'target_conditional_device_prediction_seal.json',dict(status='SEALED_TARGET_CPU_CONDITIONED_DEVICE_PREDICTION_POSTHOC',
        source_fit_iterations=[85,90],source_parameter_updates=0,target_parameter_updates=0,
        original_source_local_seal=plan['source_model_stages'][0]['local_prediction_seal_sha256'],
        source_reload_gate=plan['source_model_reload_gate'],global_prediction_reference=plan['sealed_reference'],
        files=[dict(path=n,sha256=sha(destination/n)) for n in seal_files],
        conditions='Observed target CPU API start/end, operator/shape/phase identity. Historical target timings physically read for association audit, excluded from prediction features. Not a blind test.',
        scope='Visible-device conditional completion only. No global CPU/F-B/1F1B/step/MFU prediction.',
        truth_scoring_join_after_this_seal=True))
    # Prediction file exists and is sealed before the metric/scoring join.
    scored=score_events(predictions,data)
    csv(destination,'target_device_queue_event_results.csv.gz',scored)
    for name,keys in [('metrics',['method']),('per_iteration',['method','iteration']),
                      ('per_family',['method','family']),('per_stream',['method','stream'])]:
        frame=scored.groupby(keys).apply(event_metrics,include_groups=False).reset_index()
        csv(destination,'target_device_queue_'+name+'.csv',frame)
    scopes=[]
    for scope,subset in [('all_visible_events',scored),('nonPP_visible_events',scored[~scored.family.eq('PP_candidate')]),
                         ('FB_owned_nonPP_visible_events',scored[~scored.family.eq('PP_candidate')&~scored.launch_FB_window_id.eq('outside_FB')])]:
        frame=subset.groupby('method').apply(event_metrics,include_groups=False).reset_index();frame['event_scope']=scope;scopes.append(frame)
    scope_metrics=pd.concat(scopes,ignore_index=True)
    csv(destination,'target_device_queue_metrics_by_scope.csv',scope_metrics)
    csv(destination,'target_device_queue_cost_bindings.csv.gz',pd.DataFrame(bindings))
    csv(destination,'target_device_queue_cost_coverage.csv',predictions.groupby(['method','iteration','cost_level']).size().rename('events').reset_index())
    local=scored[~scored.family.eq('PP_candidate')&~scored.launch_FB_window_id.eq('outside_FB')]
    keys=['method','iteration','rank','split','launch_FB_phase','launch_FB_window_id']
    endpoints=local.groupby(keys).agg(predicted_last_device_end_ns=('predicted_end_ns','max'),
        observed_last_device_end_ns=('end_ns','max'),events=('event_id','size')).reset_index()
    endpoints['endpoint_error_ns']=endpoints.predicted_last_device_end_ns-endpoints.observed_last_device_end_ns
    fb=windows[windows.window_type.eq('FB_annotation')][['window_id','microbatch','end_ns']].rename(
        columns={'window_id':'launch_FB_window_id','end_ns':'CPU_annotation_end_ns'})
    assert len(fb)==24 and len(endpoints)==96
    endpoints=endpoints.merge(fb,on='launch_FB_window_id',validate='many_to_one')
    endpoints['observed_device_tail_after_CPU_ns']=endpoints.observed_last_device_end_ns-endpoints.CPU_annotation_end_ns
    csv(destination,'target_CPU_owned_device_endpoints.csv',endpoints)
    selected=[]
    for key,g in local.groupby(keys):
        row=dict(zip(keys,key))
        for label,column in [('predicted','predicted_end_ns'),('observed','end_ns')]:
            picked=g.loc[g[column].idxmax()]
            for name in ['event_id','family','category','cost_id','cost_level','start_error_ns','duration_error_ns','end_error_ns']:
                row[label+'_endpoint_event_'+name]=picked[name]
        selected.append(row)
    csv(destination,'target_CPU_owned_endpoint_event_selection.csv',pd.DataFrame(selected))
    endpoint_metrics=endpoints.groupby(['method','launch_FB_phase']).endpoint_error_ns.agg(phases='size',
        endpoint_MAE_ms=lambda x:x.abs().mean()/1e6,endpoint_bias_ms=lambda x:x.mean()/1e6).reset_index()
    csv(destination,'target_device_queue_endpoint_metrics.csv',endpoint_metrics)
    source_metrics=pd.read_csv(paths['source_device_queue_endpoint_metrics.csv'])
    source_metrics=source_metrics[source_metrics.split.eq('source_incremental_validation')].copy();source_metrics['case']='source95/100'
    target_metrics=endpoint_metrics.copy();target_metrics['case']='target85/90/95/100'
    comparison=pd.concat([source_metrics,target_metrics],ignore_index=True)
    csv(destination,'source_target_conditional_endpoint_metrics.csv',comparison)
    for f in json.loads((destination/'target_conditional_device_prediction_seal.json').read_text())['files']:assert sha(destination/f['path'])==f['sha256']
    assert sha(paths['source_device_queue_parameters.csv.gz'])==parameter_hash
    draw(destination,comparison)
    dump(out/'field_contract.json',dict(scope='All target/mixed outputs evaluator_only, never model-fit inputs',
        inputs='Source85/90 original costs + target rank16 CPU submissions and observed event/shape/phase metadata',
        feature_projection=FEATURES,GPU_truth_mutation_after_preparation_leaves_features_exact=True,
        physical_truth_access='Cached GPU times used for existing runtime association and stream audit before features; not a blind or independent workload test',
        causal_limits='Same-stream recurrence uses previous predicted completion. Enqueue remainder includes hidden dependencies; not independently measured service.',
        accounting='For each device event: end_error = start_error + duration_error exactly. MAEs and overlapping event durations are not additive stage times.',
        endpoint='Maximum end of nonPP device work submitted inside each CPU F/B annotation; not the CPU annotation end or global1F1B envelope',
        family='Existing source/target heuristic GPU families retained, not independent FLOP/microbenchmark categories',
        stream='Original trace stream identifiers retained. No target-specific renaming or missing-cost zero fallback.',
        source_cost_updates=0,formal_topology_changed=False))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;elapsed=perf_counter()-begin
    assert peak<=review['resource']['maximum_peak_RSS_bytes'],'Target conditional evaluation exceeded resource gate'
    dump(out/'diagnostic.json',dict(status='TARGET_CPU_CONDITIONED_DEVICE_QUEUE_TRANSFER_PASS',new_prediction=True,
        conditional_device_prediction=True,new_global_prediction=False,used_to_fit_model=False,new_target_timing_read=True,
        source_cost_updates=0,raw_trace_scanned=False,raw_hardware_counter_scanned=False,raw_training_log_prefix_scanned=False,
        target_iterations=[85,90,95,100],target_ranks=[16],device_events=len(features),prediction_rows=len(predictions),
        CPU_owned_FB_endpoints=24,parameter_rows=len(parameters),source_parameter_bytes_unchanged=True,
        unsupported_cost_features=len(unsupported),CPU_order_unambiguous=bool(audits.CPU_order_unambiguous.all()),
        target_device_order_inversions=int(audits.observed_GPU_start_order_inversions.sum()),
        target_same_stream_device_overlaps=int(audits.observed_same_stream_GPU_overlaps.sum()),
        target_negative_release_proxy_latencies=int(audits.negative_release_proxy_latency.sum()),
        release_proxy_violation_min_ns=int(negative.GPU_start_minus_release_proxy_ns.min()) if len(negative) else None,
        release_proxy_violation_max_ns=int(negative.GPU_start_minus_release_proxy_ns.max()) if len(negative) else None,
        release_proxy_violation_categories=negative.category.value_counts().to_dict(),
        scoped_event_metrics=scope_metrics.to_dict('records'),
        GPU_truth_projection_mutation_gate=True,event_error_decomposition_exact=True,
        methods=METHODS,event_metrics=pd.read_csv(destination/'target_device_queue_metrics.csv').to_dict('records'),
        endpoint_metrics=endpoint_metrics.to_dict('records'),source_reference_endpoint_metrics=source_metrics.to_dict('records'),
        sealed_reference=plan['sealed_reference'],local_prediction_seal_sha256=sha(destination/'target_conditional_device_prediction_seal.json'),
        formal_topology_changed=False,peak_RSS_bytes=peak,analysis_seconds=elapsed,resource_gate_pass=True,
        within_target_seconds=elapsed<=review['resource']['target_analysis_seconds'],
        next='Inspect start-versus-duration and family/endpoint errors under fixed CPU conditions; only independently justified source modeling changes may become a global candidate.'))


def draw(out,comparison):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=['Independent\nlatency','Queue\nmedian','Queue\nmean','Queue\nzero remainder']
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,phase in zip(axes,['forward','backward']):
        for index,case in enumerate(['source95/100','target85/90/95/100']):
            rows=comparison[comparison['case'].eq(case)&comparison.launch_FB_phase.eq(phase)].set_index('method').loc[METHODS]
            ax.bar(np.arange(4)+(index-.5)*.36,rows.endpoint_MAE_ms,.36,label=case)
        ax.set_xticks(range(4),labels);ax.set_yscale('symlog',linthresh=.1)
        ax.set_ylabel('CPU-owned device endpoint MAE (ms, symlog)');ax.set_title(phase);ax.legend()
    fig.suptitle('Frozen source85/90 costs; observed CPU submissions and event metadata are conditions')
    fig.text(.5,.025,'All target results are development/posthoc. Device endpoint is not CPU F/B end or global1F1B.\n'
        'Four source-fixed methods retained; no target fitting, missing-cost zero fill, or new step/MFU claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94));fig.savefig(out/'conditional_device_queue_transfer.svg');fig.savefig(out/'conditional_device_queue_transfer.png',dpi=150);plt.close(fig)

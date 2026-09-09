"""Separate API-start release candidate; immutable T19 code remains the control."""
import json
import resource
from time import perf_counter

import pandas as pd
from guards import ROOT
from smoke_worker import dump,sha
from worker import csv
from device_queue import prepare,stream_audit,DTYPE
from device_queue_model import DeviceQueueModel,feature_frame,fit_predict_score,METHODS


def api_start_proxy(data):
    """CPU API entry is a condition, not a claim of effective device readiness."""
    result=data.copy()
    assert result.runtime_start_ns.dtype.kind=='i'
    result['release_proxy_ns']=result.runtime_start_ns
    result['release_proxy_kind']='observed_all_categories_API_start'
    return result


def diagnose(out,paths,plan):
    begin=perf_counter()
    assert plan['diagnostic_access']=='source_only' and not plan['variants']
    review=json.loads((ROOT/plan['resource_review_file']).read_text())
    assert review['fit_iterations']==[85,90] and review['incremental_validation']==[95,100]
    windows=pd.read_csv(paths['runtime_windows.csv.gz'],dtype=DTYPE,low_memory=False)
    prepared=[]
    for it in [85,90,95,100]:
        gpu=pd.read_csv(paths[f'source{it}_GPU.csv.gz'],dtype=DTYPE,low_memory=False)
        runtime=pd.read_csv(paths[f'source{it}_runtime.csv.gz'],dtype=DTYPE,low_memory=False)
        cpu=pd.read_csv(paths[f'source{it}_CPU.csv.gz'],dtype=DTYPE,low_memory=False)
        prepared.append(prepare(gpu,runtime,cpu,windows[windows.iteration.eq(it)]))
    prepared=pd.concat(prepared,ignore_index=True)
    assert len(prepared)==41764 and set(prepared.iteration)=={85,90,95,100}
    results={};parameters={};edge_reference=None;comparisons=[];event_comparisons=[]
    for candidate,data in [('legacy_API_end',prepared),('all_API_start',api_start_proxy(prepared))]:
        destination=out/candidate;destination.mkdir()
        audit,data,edges=stream_audit(data)
        assert audit.CPU_order_unambiguous.all()
        for column in ['observed_GPU_start_order_inversions','observed_same_stream_GPU_overlaps',
                       'negative_release_proxy_latency','negative_enqueue_remainder']:
            assert audit[column].eq(0).all(),(candidate,column)
        if edge_reference is None:edge_reference=edges
        else:pd.testing.assert_frame_equal(edges,edge_reference,check_exact=True)
        model=DeviceQueueModel(data)
        # Fitting and conditions must be invariant to held-out device truth.
        mutated=data.copy()
        for column in ['start_ns','end_ns','duration_ns','observed_enqueue_remainder_ns','observed_API_latency_ns']:
            mutated.loc[mutated.iteration.isin([95,100]),column]=-999999
        pd.testing.assert_frame_equal(model.parameters,DeviceQueueModel(mutated).parameters,check_exact=True)
        pd.testing.assert_frame_equal(feature_frame(data),feature_frame(mutated),check_exact=True)
        parameters[candidate]=model.parameters.copy()
        del mutated,model
        csv(destination,'source_stream_order_audit.csv',audit)
        csv(destination,'source_candidate_stream_edges.csv.gz',edges)
        csv(destination,'source_device_queue_observations.csv.gz',data)
        csv(destination,'source_API_release_summary.csv',data.groupby(['iteration','split','category']).agg(
            events=('event_id','size'),minimum_API_latency_ns=('observed_API_latency_ns','min'),
            mean_API_latency_ns=('observed_API_latency_ns','mean'),
            mean_enqueue_remainder_ns=('observed_enqueue_remainder_ns','mean')).reset_index())
        results[candidate]=fit_predict_score(destination,data,windows,plan)
        table=pd.read_csv(destination/'source_device_queue_endpoint_metrics.csv');table['release_candidate']=candidate;comparisons.append(table)
        table=pd.read_csv(destination/'source_device_queue_metrics.csv');table['release_candidate']=candidate;event_comparisons.append(table)
    original=parameters['legacy_API_end'];candidate=parameters['all_API_start']
    preserved=[c for c in original if c not in ['API_latency_median_ns','enqueue_remainder_median_ns','enqueue_remainder_mean_ns']]
    pd.testing.assert_frame_equal(original[preserved],candidate[preserved],check_exact=True)
    delta=original[['cost_id','level','key','samples']].copy()
    for column in ['API_latency_median_ns','enqueue_remainder_median_ns','enqueue_remainder_mean_ns']:
        delta[column+'_change']=candidate[column]-original[column]
    csv(out,'source_release_parameter_changes.csv.gz',delta)
    comparison=pd.concat(comparisons,ignore_index=True)
    csv(out,'source_release_endpoint_comparison.csv',comparison)
    csv(out,'source_release_event_comparison.csv',pd.concat(event_comparisons,ignore_index=True))
    draw(out,comparison)
    dump(out/'source_release_comparison_seal.json',dict(status='SEALED_SOURCE_ONLY_CONDITIONAL_RELEASE_COMPARISON',
        fit_iterations=[85,90],incremental_validation=[95,100],target_parameter_updates=0,
        candidates={key:dict(local_seal_path=key+'/source_device_queue_prediction_seal.json',
            sha256=sha(out/key/'source_device_queue_prediction_seal.json')) for key in results},
        only_model_change='CPU APIstart release proxy and associated source-fitted enqueue/API latency; duration costs, keys and edges unchanged'))
    dump(out/'field_contract.json',dict(scope='Source-only local device prediction conditioned on observed CPU API and event metadata',
        candidate='APIstart for all categories. It is an observable lower boundary; effective enqueue/release remains latent.',
        remainder='Source empirical release residual includes host enqueue, cross-stream and hidden work; not a measured service time.',
        immutable='T19 preparation/model/scorer remain unchanged; legacy control and candidate use same input, grouping and four methods.',
        accounting='Device residence may include communication waits; no GPU-duration sum is used as global1F1B. CPU-owned nonPP endpoints differ from CPU F/B end.',
        validation='Historically exposed source95/100 incremental, not blind. GPU timing mutation leaves fit and feature projection exact.',
        global_scope='No new224/global1F1B, step or MFU prediction; formal topology unchanged.'))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;elapsed=perf_counter()-begin
    assert peak<=review['resource']['maximum_peak_RSS_bytes']
    dump(out/'diagnostic.json',dict(status='SOURCE_API_START_RELEASE_CANDIDATE_PASS',new_prediction=True,
        new_global_prediction=False,used_to_fit_model=True,new_target_timing_read=False,raw_trace_scanned=False,
        source_iterations=[85,90,95,100],source_ranks=[16],source_fit_iterations=[85,90],device_events=len(prepared),
        prediction_rows=2*4*len(prepared),methods=METHODS,results=results,parameter_rows=len(original),
        duration_costs_and_keys_exact=True,candidate_edges_exact=True,validation_GPU_mutation_fit_features_exact=True,
        formal_topology_changed=False,comparison_seal_sha256=sha(out/'source_release_comparison_seal.json'),
        peak_RSS_bytes=peak,analysis_seconds=elapsed,resource_gate_pass=True,
        within_target_seconds=elapsed<=review['resource']['target_analysis_seconds'],
        next='Reproduce originalT19 control exactly and inspect source incremental release-proxy ablation; register separate target evaluation only after source acceptance.'))


def draw(out,comparison):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data=comparison[comparison.split.eq('source_incremental_validation')]
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    labels=['Independent\nlatency','Queue\nmedian','Queue\nmean','Queue\nzero remainder']
    for ax,phase in zip(axes,['forward','backward']):
        for index,candidate in enumerate(['legacy_API_end','all_API_start']):
            rows=data[data.launch_FB_phase.eq(phase)&data.release_candidate.eq(candidate)].set_index('method').loc[METHODS]
            ax.bar(np.arange(4)+(index-.5)*.36,rows.endpoint_MAE_ms,.36,label=candidate)
        ax.set_xticks(range(4),labels);ax.set_yscale('symlog',linthresh=.1)
        ax.set_ylabel('Device endpoint MAE (ms, symlog)');ax.set_title(phase);ax.legend()
    fig.suptitle('Source95/100 conditional validation, fit85/90: one release-proxy change')
    fig.text(.5,.025,'Same costs grouping, durations and candidate edges. Observed CPU API times are conditions.\n'
        'Device endpoint is not CPU F/B end or global1F1B; hidden constraints remain unresolved.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94));fig.savefig(out/'source_release_comparison.svg');fig.savefig(out/'source_release_comparison.png',dpi=150);plt.close(fig)

"""Sealed source release candidates on target CPU-conditioned device events."""
import gzip
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
from device_queue_transfer import score_events,event_metrics
from device_release import api_start_proxy


CANDIDATES=['legacy_API_end','all_API_start']


def decompressed_text(path):
    with gzip.open(path,'rt') as stream:return stream.read()


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='evaluator' and not plan['variants']
    review=json.loads((ROOT/plan['resource_review_file']).read_text())
    assert review['fixed_methods']==METHODS and review['source_stage']['variants']==CANDIDATES
    windows=pd.read_csv(paths['target_runtime_windows.csv.gz'],dtype=DTYPE,low_memory=False)
    windows=windows[windows['case'].eq('target224')]
    assert set(windows.iteration)=={85,90,95,100} and set(windows['rank'])=={16}
    prepared=[]
    for iteration in [85,90,95,100]:
        gpu=pd.read_csv(paths[f'target{iteration}_GPU.csv.gz'],dtype=DTYPE,low_memory=False)
        runtime=pd.read_csv(paths[f'target{iteration}_runtime.csv.gz'],dtype=DTYPE,low_memory=False)
        cpu=pd.read_csv(paths[f'target{iteration}_CPU.csv.gz'],dtype=DTYPE,low_memory=False)
        frame=prepare(gpu,runtime,cpu,windows[windows.iteration.eq(iteration)])
        frame['split']='target_development_posthoc';prepared.append(frame)
    prepared=pd.concat(prepared,ignore_index=True)
    assert len(prepared)==33816

    destination=out/'evaluator_only';candidate_data={};candidate_predictions={};audits=[];edges=[];violations=[]
    parameter_tables=[];unsupported=[];bindings=[]
    # Both predictions are created and sealed before any target truth scoring.
    for candidate in CANDIDATES:
        data=prepared if candidate=='legacy_API_end' else api_start_proxy(prepared)
        audit,ordered,edge=stream_audit(data)
        assert audit.CPU_order_unambiguous.all()
        audit['release_candidate']=candidate;edge['release_candidate']=candidate
        audits.append(audit);edges.append(edge)
        negative=ordered[ordered.start_ns.lt(ordered.release_proxy_ns)].copy()
        negative['GPU_start_minus_release_proxy_ns']=negative.start_ns-negative.release_proxy_ns
        negative['release_candidate']=candidate;violations.append(negative)
        features=feature_frame(ordered)
        assert set(features.columns)==set(FEATURES) and len(features)==33816
        mutated=ordered.copy()
        for name in ['start_ns','end_ns','duration_ns','observed_enqueue_remainder_ns']:
            mutated[name]=mutated[name]+123456789
        pd.testing.assert_frame_equal(feature_frame(mutated),features,check_exact=True)
        parameter_key=candidate+'_source_device_queue_parameters.csv.gz'
        parameters=pd.read_csv(paths[parameter_key],dtype={'fit_iterations':'string','key':'string','cost_id':'string','level':'string'})
        model=load_parameters(parameters);parameter_tables.append(parameters)
        local_unsupported=[];local_bindings=[]
        for row in features.to_dict('records'):
            try:cost=model.lookup(row)
            except ValueError:
                local_unsupported.append({k:row[k] for k in ['iteration','event_id','category','stream','family','name','CPU_owner_input_dims']});continue
            local_bindings.append(dict(release_candidate=candidate,event_id=row['event_id'],iteration=row['iteration'],
                cost_id=cost['cost_id'],cost_level=cost['level']))
        for row in local_unsupported:row['release_candidate']=candidate
        unsupported.extend(local_unsupported);bindings.extend(local_bindings)
        assert not local_unsupported,'No source-supported target cost; no zero fallback'
        predictions=model.predict(features,METHODS)
        assert len(predictions)==135264
        candidate_out=destination/candidate
        csv(candidate_out,'target_device_queue_features.csv.gz',features)
        csv(candidate_out,'target_device_queue_predictions_sealed.csv.gz',predictions)
        (candidate_out/'source_device_queue_parameters_used.csv.gz').write_bytes(paths[parameter_key].read_bytes())
        assert sha(candidate_out/'source_device_queue_parameters_used.csv.gz')==sha(paths[parameter_key])
        seal_files=['source_device_queue_parameters_used.csv.gz','target_device_queue_features.csv.gz','target_device_queue_predictions_sealed.csv.gz']
        dump(candidate_out/'target_conditional_device_prediction_seal.json',dict(
            status='SEALED_TARGET_CPU_CONDITIONED_RELEASE_CANDIDATE_POSTHOC',release_candidate=candidate,
            source_fit_iterations=[85,90],source_parameter_updates=0,target_parameter_updates=0,
            source_release_comparison_seal=plan['source_release_stage']['comparison_seal_sha256'],
            global_prediction_reference=plan['sealed_reference'],
            files=[dict(path=name,sha256=sha(candidate_out/name)) for name in seal_files],
            conditions='Observed target CPU API start/end and event/operator/shape/phase identity. Target GPU timing was historically observed for association but is excluded from prediction features.',
            scope='Visible-device conditional completion only; no CPU/F-B/global1F1B/step/MFU prediction.',truth_scoring_join_after_this_seal=True))
        candidate_data[candidate]=ordered;candidate_predictions[candidate]=predictions
    unchanged=[c for c in parameter_tables[0] if c not in ['API_latency_median_ns','enqueue_remainder_median_ns','enqueue_remainder_mean_ns']]
    pd.testing.assert_frame_equal(parameter_tables[0][unchanged],parameter_tables[1][unchanged],check_exact=True)
    csv(destination,'target_release_stream_order_audit.csv',pd.concat(audits,ignore_index=True))
    csv(destination,'target_release_candidate_stream_edges.csv.gz',pd.concat(edges,ignore_index=True))
    violation_frame=pd.concat(violations,ignore_index=True)
    columns=['release_candidate','iteration','event_id','category','family','name','CPU_owner_name_checked','stream','unique_runtime_name',
             'runtime_start_ns','runtime_end_ns','start_ns','release_proxy_ns','GPU_start_minus_release_proxy_ns',
             'launch_FB_phase','launch_FB_window_id','launch_EP_semantic_region']
    csv(destination,'target_release_proxy_violations.csv',violation_frame[columns])
    csv(destination,'target_release_unsupported_cost_features.csv',pd.DataFrame(unsupported,
        columns=['release_candidate','iteration','event_id','category','stream','family','name','CPU_owner_input_dims']))
    csv(destination,'target_release_cost_bindings.csv.gz',pd.DataFrame(bindings))

    legacy=destination/'legacy_API_end'
    assert decompressed_text(legacy/'target_device_queue_features.csv.gz')==decompressed_text(paths['target_control_features.csv.gz'])
    assert decompressed_text(legacy/'target_device_queue_predictions_sealed.csv.gz')==decompressed_text(paths['target_control_predictions.csv.gz'])

    all_metrics=[];all_iterations=[];all_families=[];all_streams=[];all_scopes=[];all_endpoints=[];all_endpoint_metrics=[];all_selections=[]
    for candidate in CANDIDATES:
        candidate_out=destination/candidate;scored=score_events(candidate_predictions[candidate],candidate_data[candidate])
        csv(candidate_out,'target_device_queue_event_results.csv.gz',scored)
        for name,keys,collector in [('metrics',['method'],all_metrics),('per_iteration',['method','iteration'],all_iterations),
                                    ('per_family',['method','family'],all_families),('per_stream',['method','stream'],all_streams)]:
            table=scored.groupby(keys).apply(event_metrics,include_groups=False).reset_index();table['release_candidate']=candidate
            csv(candidate_out,'target_device_queue_'+name+'.csv',table);collector.append(table)
        scopes=[]
        for scope,subset in [('all_visible_events',scored),('nonPP_visible_events',scored[~scored.family.eq('PP_candidate')]),
                             ('FB_owned_nonPP_visible_events',scored[~scored.family.eq('PP_candidate')&~scored.launch_FB_window_id.eq('outside_FB')])]:
            table=subset.groupby('method').apply(event_metrics,include_groups=False).reset_index()
            table['event_scope']=scope;table['release_candidate']=candidate;scopes.append(table)
        scope_table=pd.concat(scopes,ignore_index=True);csv(candidate_out,'target_device_queue_metrics_by_scope.csv',scope_table);all_scopes.append(scope_table)
        local=scored[~scored.family.eq('PP_candidate')&~scored.launch_FB_window_id.eq('outside_FB')]
        keys=['method','iteration','rank','split','launch_FB_phase','launch_FB_window_id']
        endpoints=local.groupby(keys).agg(predicted_last_device_end_ns=('predicted_end_ns','max'),
            observed_last_device_end_ns=('end_ns','max'),events=('event_id','size')).reset_index()
        endpoints['endpoint_error_ns']=endpoints.predicted_last_device_end_ns-endpoints.observed_last_device_end_ns
        fb=windows[windows.window_type.eq('FB_annotation')][['window_id','microbatch','end_ns']].rename(
            columns={'window_id':'launch_FB_window_id','end_ns':'CPU_annotation_end_ns'})
        endpoints=endpoints.merge(fb,on='launch_FB_window_id',validate='many_to_one')
        assert len(endpoints)==96
        endpoints['observed_device_tail_after_CPU_ns']=endpoints.observed_last_device_end_ns-endpoints.CPU_annotation_end_ns
        endpoints['release_candidate']=candidate;csv(candidate_out,'target_CPU_owned_device_endpoints.csv',endpoints);all_endpoints.append(endpoints)
        selected=[]
        for key,group in local.groupby(keys):
            row=dict(zip(keys,key));row['release_candidate']=candidate
            for label,column in [('predicted','predicted_end_ns'),('observed','end_ns')]:
                picked=group.loc[group[column].idxmax()]
                for name in ['event_id','family','category','cost_id','cost_level','start_error_ns','duration_error_ns','end_error_ns']:
                    row[label+'_endpoint_event_'+name]=picked[name]
            selected.append(row)
        selections=pd.DataFrame(selected);csv(candidate_out,'target_CPU_owned_endpoint_event_selection.csv',selections);all_selections.append(selections)
        endpoint_metrics=endpoints.groupby(['method','launch_FB_phase']).endpoint_error_ns.agg(phases='size',
            endpoint_MAE_ms=lambda x:x.abs().mean()/1e6,endpoint_bias_ms=lambda x:x.mean()/1e6).reset_index()
        csv(candidate_out,'target_device_queue_endpoint_metrics.csv',endpoint_metrics)
        endpoint_metrics['release_candidate']=candidate;all_endpoint_metrics.append(endpoint_metrics)
        for file in json.loads((candidate_out/'target_conditional_device_prediction_seal.json').read_text())['files']:
            assert sha(candidate_out/file['path'])==file['sha256']
    combined={
        'target_release_metrics.csv':all_metrics,'target_release_per_iteration.csv':all_iterations,
        'target_release_per_family.csv':all_families,'target_release_per_stream.csv':all_streams,
        'target_release_metrics_by_scope.csv':all_scopes,'target_release_CPU_owned_device_endpoints.csv':all_endpoints,
        'target_release_endpoint_metrics.csv':all_endpoint_metrics,'target_release_endpoint_event_selection.csv':all_selections,
    }
    for name,tables in combined.items():csv(destination,name,pd.concat(tables,ignore_index=True))
    assert (legacy/'target_device_queue_endpoint_metrics.csv').read_text()==paths['target_control_endpoint_metrics.csv'].read_text()
    comparison=[]
    for candidate in CANDIDATES:
        source=pd.read_csv(paths[candidate+'_source_device_queue_endpoint_metrics.csv'])
        source=source[source.split.eq('source_incremental_validation')].copy();source['case']='source95/100';source['release_candidate']=candidate
        target=next(table for table in all_endpoint_metrics if table.release_candidate.iat[0]==candidate).copy();target['case']='target85/90/95/100'
        comparison.extend([source,target])
    comparison=pd.concat(comparison,ignore_index=True);csv(destination,'source_target_release_endpoint_comparison.csv',comparison)
    draw(destination,comparison)
    dump(out/'field_contract.json',dict(scope='All target/mixed outputs evaluator_only; fixed source85/90 models and target observed CPU conditions',
        release='Legacy kernel APIend/copy APIstart versus all-category APIstart. APIstart is a lower boundary, not proven effective release.',
        prediction='Previous predicted same-stream completion only; no target observed predecessor completion, target fit, stream remapping or missing-cost zero fill.',
        truth='GPU timing already seen for association and stream audit; mutation after feature projection leaves model inputs exact. Scoring follows both seals.',
        accounting='Per event end error equals start plus duration error. PP/collective residence can include waiting and is not summed as step time.',
        endpoint='Maximum nonPP device end owned by a CPU F/B annotation, not CPU F/B/global1F1B end.',
        control='Legacy features/predictions decompressed text and endpoint metrics exactly reproduce T26B.',
        formal_topology_changed=False))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;elapsed=perf_counter()-begin
    assert peak<=review['resource']['maximum_peak_RSS_bytes']
    endpoint_table=pd.concat(all_endpoint_metrics,ignore_index=True);scope_table=pd.concat(all_scopes,ignore_index=True)
    violation_counts=violation_frame.groupby('release_candidate').size().to_dict()
    violation_counts['all_API_start']=int(violation_counts.get('all_API_start',0))
    dump(out/'diagnostic.json',dict(status='TARGET_API_START_RELEASE_ABLATION_PASS',conditional_device_prediction=True,
        new_prediction=True,new_global_prediction=False,used_to_fit_model=False,new_target_timing_read=True,
        source_cost_updates=0,target_parameter_updates=0,raw_trace_scanned=False,raw_hardware_counter_scanned=False,
        raw_training_log_prefix_scanned=False,target_iterations=[85,90,95,100],target_ranks=[16],device_events=len(prepared),
        prediction_rows=sum(len(frame) for frame in candidate_predictions.values()),parameter_rows_per_candidate=2772,
        unsupported_cost_features=len(unsupported),release_proxy_violations=violation_counts,
        duration_costs_keys_and_candidate_edges_exact=True,GPU_truth_projection_mutation_gate=True,
        event_error_decomposition_exact=True,legacy_T26B_features_and_predictions_text_exact=True,
        legacy_T26B_endpoint_metrics_exact=True,methods=METHODS,endpoint_metrics=endpoint_table.to_dict('records'),
        scoped_event_metrics=scope_table.to_dict('records'),sealed_reference=plan['sealed_reference'],
        release_candidate_seals={candidate:sha(destination/candidate/'target_conditional_device_prediction_seal.json') for candidate in CANDIDATES},
        formal_topology_changed=False,peak_RSS_bytes=peak,analysis_seconds=elapsed,resource_gate_pass=True,
        within_target_seconds=elapsed<=review['resource']['target_analysis_seconds'],
        next='Use fixed conditional ablation to decide whether release semantics explains target endpoints; any next model must be source-evidenced and independently sealed.'))


def draw(out,comparison):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(14,6));labels=['Independent\nlatency','Queue\nmedian','Queue\nmean','Queue\nzero remainder']
    target=comparison[comparison['case'].eq('target85/90/95/100')]
    for ax,phase in zip(axes,['forward','backward']):
        for index,candidate in enumerate(CANDIDATES):
            rows=target[target.launch_FB_phase.eq(phase)&target.release_candidate.eq(candidate)].set_index('method').loc[METHODS]
            ax.bar(np.arange(4)+(index-.5)*.36,rows.endpoint_MAE_ms,.36,label=candidate)
        ax.set_xticks(range(4),labels);ax.set_yscale('symlog',linthresh=.1)
        ax.set_ylabel('Conditional device endpoint MAE (ms, symlog)');ax.set_title(phase);ax.legend()
    fig.suptitle('Target224 posthoc: fixed source release-proxy ablation')
    fig.text(.5,.025,'Observed CPU API and event metadata are conditions; no target fit.\n'
        'Device endpoint is not CPU F/B end or global1F1B; formal topology unchanged.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94));fig.savefig(out/'target_release_comparison.svg');fig.savefig(out/'target_release_comparison.png',dpi=150);plt.close(fig)

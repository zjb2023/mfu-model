"""Transfer sealed source terminal rules onto sealed target conditional predictions."""
import json
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from guards import ROOT
from smoke_worker import dump, sha
from worker import csv
from device_queue_model import METHODS


GROUP = ['iteration', 'rank', 'split', 'launch_FB_phase', 'launch_FB_window_id']
IDENTITY = ['CPU_owner_name_checked', 'stream', 'family', 'category']
RELEASES = ['legacy_API_end', 'all_API_start']


def select_target_terminal_events(features, identities):
    data = features.copy();data['split'] = 'target_development_posthoc'
    rows = []
    for identity in identities.to_dict('records'):
        group = data[data.launch_FB_phase.eq(identity['launch_FB_phase']) & ~data.launch_FB_window_id.eq('outside_FB')]
        for name in IDENTITY:
            group = group[group[name].astype(str).eq(str(identity[name]))]
        group = group.sort_values(GROUP + ['runtime_start_ns', 'runtime_end_ns', 'event_id'])
        group = group.groupby(GROUP, sort=True, as_index=False).tail(1).copy()
        rows.append(group[GROUP + ['event_id'] + IDENTITY].rename(columns={'event_id':'selected_event_id'}))
    result = pd.concat(rows, ignore_index=True).sort_values(GROUP).reset_index(drop=True)
    assert len(result) == 24 and not result.duplicated(GROUP).any()
    return result


def predicted_maximum(features, predictions, release):
    meta = features[['event_id', 'iteration', 'rank', 'launch_FB_phase', 'launch_FB_window_id'] + IDENTITY].copy()
    meta['split'] = 'target_development_posthoc'
    joined = predictions.merge(meta, on=['event_id','iteration','rank'], validate='many_to_one')
    joined = joined[~joined.family.eq('PP_candidate') & ~joined.launch_FB_window_id.eq('outside_FB')]
    joined = joined.sort_values(['method'] + GROUP + ['predicted_end_ns','event_id'])
    result = joined.groupby(['method'] + GROUP, sort=True, as_index=False).tail(1).copy()
    result = result.rename(columns={'method':'component_method','event_id':'selected_event_id',
                                    'predicted_end_ns':'predicted_endpoint_ns'})
    result['release_candidate'] = release;result['endpoint_rule'] = 'predicted_maximum'
    return result[GROUP + ['release_candidate','endpoint_rule','component_method','selected_event_id','predicted_endpoint_ns']]


def semantic_independent(terminal_events, predictions, release):
    result = terminal_events.merge(predictions[predictions.method.eq('independent_latency')],
        left_on=['selected_event_id','iteration','rank'],right_on=['event_id','iteration','rank'],validate='one_to_one')
    result = result.rename(columns={'predicted_end_ns':'predicted_endpoint_ns'})
    result['release_candidate'] = release;result['endpoint_rule'] = 'fitted_terminal_event'
    result['component_method'] = 'independent_latency'
    return result[GROUP + ['release_candidate','endpoint_rule','component_method','selected_event_id','predicted_endpoint_ns']]


def source_selected(controls, semantic_events, selection, release):
    rows=[]
    for parameter in selection[selection.release_candidate.eq(release)].to_dict('records'):
        source = controls if parameter['endpoint_rule']=='predicted_maximum' else semantic_events
        group = source[source.launch_FB_phase.eq(parameter['launch_FB_phase']) &
            source.component_method.eq(parameter['component_method'])].copy()
        group['source_selected_component_rule']=parameter['endpoint_rule']
        group['source_selected_component_method']=parameter['component_method']
        group['endpoint_rule']='source_fit_selected';group['component_method']='phase_selected'
        rows.append(group)
    result=pd.concat(rows,ignore_index=True);assert len(result)==24
    return result


def tail_baseline(control_endpoints, tails, release):
    base=control_endpoints[control_endpoints.method.eq('independent_latency')][GROUP+['CPU_annotation_end_ns']].copy()
    result=base.merge(tails,on='launch_FB_phase',validate='many_to_one')
    result['predicted_endpoint_ns']=result.CPU_annotation_end_ns+result.phase_median_tail_ns
    result['release_candidate']=release;result['endpoint_rule']='CPU_annotation_phase_median_tail'
    result['component_method']='phase_median';result['selected_event_id']=''
    return result[GROUP+['release_candidate','endpoint_rule','component_method','selected_event_id','predicted_endpoint_ns']]


def metrics(group):
    applicable=group[group.endpoint_identity_applicable]
    return pd.Series(dict(windows=len(group),endpoint_MAE_ms=group.endpoint_error_ns.abs().mean()/1e6,
        endpoint_bias_ms=group.endpoint_error_ns.mean()/1e6,
        identity_applicable_windows=len(applicable),
        endpoint_event_identity_matches=int(applicable.endpoint_event_identity_match.fillna(False).sum()),
        endpoint_event_identity_match_rate=(float(applicable.endpoint_event_identity_match.fillna(False).mean()) if len(applicable) else np.nan)))


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='evaluator' and not plan['variants']
    review=json.loads((ROOT/plan['resource_review_file']).read_text())
    source=plan['source_terminal_stage'];target=plan['target_terminal_stage']
    identities=pd.read_csv(paths['source_terminal_identity_parameters.csv'],dtype={'stream':'string'})
    selection=pd.read_csv(paths['source_terminal_selection_parameters.csv'])
    tails=pd.read_csv(paths['source_phase_tail_parameters.csv'])
    assert len(identities)==2 and identities.fit_endpoint_windows.eq(8).all()
    assert identities.CPU_owner_name_checked.eq('aten::_copy_from').all() and identities.stream.eq('0').all()
    assert len(selection)==4 and set(selection.release_candidate)==set(RELEASES)
    assert len(tails)==2 and tails.fit_iterations.eq('85,90').all()

    candidates=[];terminal_reference=None;features_by_release={}
    for release in RELEASES:
        features=pd.read_csv(paths[f'{release}_target_features.csv.gz'],low_memory=False,dtype={'stream':'string'})
        predictions=pd.read_csv(paths[f'{release}_target_predictions.csv.gz'],low_memory=False,dtype={'stream':'string'})
        assert len(features)==33816 and len(predictions)==135264 and set(predictions.method)==set(METHODS)
        terminal=select_target_terminal_events(features,identities)
        if terminal_reference is None:terminal_reference=terminal
        else:pd.testing.assert_frame_equal(terminal_reference,terminal,check_exact=True)
        controls=predicted_maximum(features,predictions,release)
        semantic_all=[]
        for method in METHODS:
            group=terminal.merge(predictions[predictions.method.eq(method)],
                left_on=['selected_event_id','iteration','rank'],right_on=['event_id','iteration','rank'],validate='one_to_one')
            group=group.rename(columns={'predicted_end_ns':'predicted_endpoint_ns'});group['release_candidate']=release
            group['endpoint_rule']='fitted_terminal_event';group['component_method']=method
            semantic_all.append(group[GROUP+['release_candidate','endpoint_rule','component_method','selected_event_id','predicted_endpoint_ns']])
        semantic_all=pd.concat(semantic_all,ignore_index=True)
        chosen=source_selected(controls,semantic_all,selection,release)
        semantic=semantic_all[semantic_all.component_method.eq('independent_latency')]
        candidates += [controls,chosen,semantic]
        features_by_release[release]=features
    # CPU annotation end is a target condition already present in the sealed T27B endpoint table.
    # It is read before seal only as a prediction input; observed endpoint fields are not projected.
    for release in RELEASES:
        conditional=pd.read_csv(paths[f'{release}_target_endpoints.csv'])
        projected=tail_baseline(conditional,tails,release)
        mutated=conditional.copy()
        for column in ['observed_last_device_end_ns','observed_device_tail_after_CPU_ns','endpoint_error_ns']:
            mutated[column]=-999999
        pd.testing.assert_frame_equal(projected,tail_baseline(mutated,tails,release),check_exact=True)
        candidates.append(projected)
    predictions=pd.concat(candidates,ignore_index=True)
    predictions['prediction_scope']='target_CPU_submission_conditioned_visible_device_endpoint_NOT_global1F1B'
    predictions=predictions.sort_values(['release_candidate','endpoint_rule','component_method']+GROUP).reset_index(drop=True)
    assert len(predictions)==336
    csv(out,'evaluator_only/target_terminal_candidate_event_selection.csv.gz',terminal_reference)
    csv(out,'evaluator_only/target_terminal_endpoint_predictions_sealed.csv.gz',predictions)
    for name,key in [('source_terminal_identity_parameters.csv','source_terminal_identity_parameters.csv'),
                     ('source_terminal_selection_parameters.csv','source_terminal_selection_parameters.csv'),
                     ('source_phase_tail_parameters.csv','source_phase_tail_parameters.csv')]:
        (out/'evaluator_only'/name).write_bytes(paths[key].read_bytes())
    sealed=['target_terminal_candidate_event_selection.csv.gz','target_terminal_endpoint_predictions_sealed.csv.gz',
        'source_terminal_identity_parameters.csv','source_terminal_selection_parameters.csv','source_phase_tail_parameters.csv']
    destination=out/'evaluator_only'
    dump(destination/'target_terminal_endpoint_prediction_seal.json',dict(
        status='SEALED_TARGET_CPU_CONDITIONED_TERMINAL_ENDPOINT_PREDICTIONS_POSTHOC',
        source_fit_iterations=[85,90],source_parameter_updates=0,target_parameter_updates=0,
        source_terminal_seal_sha256=source['prediction_seal_sha256'],
        upstream_target_release_seals=target['release_candidate_seals'],
        candidates=review['pre_registered_target_candidates'],
        files=[dict(path=name,sha256=sha(destination/name)) for name in sealed],
        conditions='Observed target CPU submissions/event metadata and CPU F/B annotation end. No target endpoint time or identity selects a rule.',
        truth_scoring_join_after_this_seal=True,scope='Visible-device endpoint only; no global1F1B, Step or MFU prediction.'))

    # Attach target endpoint time and identity only after the prediction seal.
    truth_reference=None;selection_reference=None;control_tables=[]
    for release in RELEASES:
        old_endpoints=pd.read_csv(paths[f'{release}_target_endpoints.csv'])
        old_selection=pd.read_csv(paths[f'{release}_target_endpoint_selection.csv'])
        truth=old_endpoints[old_endpoints.method.eq('independent_latency')][GROUP+['observed_last_device_end_ns']].rename(
            columns={'observed_last_device_end_ns':'observed_endpoint_ns'})
        selected=old_selection[old_selection.method.eq('independent_latency')][GROUP+['observed_endpoint_event_event_id']].rename(
            columns={'observed_endpoint_event_event_id':'observed_endpoint_event_id'})
        if truth_reference is None:truth_reference=truth;selection_reference=selected
        else:
            pd.testing.assert_frame_equal(truth_reference,truth,check_exact=True)
            pd.testing.assert_frame_equal(selection_reference,selected,check_exact=True)
        old=old_endpoints[GROUP+['method','predicted_last_device_end_ns','observed_last_device_end_ns','endpoint_error_ns']].copy()
        old['release_candidate']=release;control_tables.append(old)
    truth=truth_reference.merge(selection_reference,on=GROUP,validate='one_to_one')
    identity_lookup=features_by_release['legacy_API_end'][['event_id','iteration','rank']+IDENTITY].rename(columns={
        'event_id':'observed_endpoint_event_id',**{name:'observed_'+name for name in IDENTITY}})
    truth=truth.merge(identity_lookup,on=['observed_endpoint_event_id','iteration','rank'],validate='one_to_one')
    scored=predictions.merge(truth,on=GROUP,validate='many_to_one')
    scored['endpoint_error_ns']=scored.predicted_endpoint_ns-scored.observed_endpoint_ns
    scored['endpoint_identity_applicable']=scored.selected_event_id.fillna('').ne('')
    scored['endpoint_event_identity_match']=np.where(scored.endpoint_identity_applicable,
        scored.selected_event_id.eq(scored.observed_endpoint_event_id),pd.NA)
    predicted_lookup=features_by_release['legacy_API_end'][['event_id','iteration','rank']+IDENTITY].rename(columns={
        'event_id':'selected_event_id',**{name:'predicted_'+name for name in IDENTITY}})
    scored=scored.merge(predicted_lookup,on=['selected_event_id','iteration','rank'],how='left',validate='many_to_one')
    csv(destination,'target_terminal_endpoint_results.csv.gz',scored)
    keys=['release_candidate','endpoint_rule','component_method','launch_FB_phase']
    metric_table=scored.groupby(keys).apply(metrics,include_groups=False).reset_index()
    per_iteration=scored.groupby(keys[:3]+['iteration','launch_FB_phase']).apply(metrics,include_groups=False).reset_index()
    csv(destination,'target_terminal_endpoint_metrics.csv',metric_table)
    csv(destination,'target_terminal_endpoint_per_iteration.csv',per_iteration)
    identity_counts=truth.groupby(['launch_FB_phase']+['observed_'+name for name in IDENTITY]).size().rename('windows').reset_index()
    csv(destination,'target_observed_terminal_identity_counts.csv',identity_counts)

    # Existing maximum controls must reproduce T27B exactly after the same truth is attached.
    controls=scored[scored.endpoint_rule.eq('predicted_maximum')][GROUP+['release_candidate','component_method','predicted_endpoint_ns','observed_endpoint_ns','endpoint_error_ns']].copy()
    controls=controls.rename(columns={'component_method':'method','predicted_endpoint_ns':'predicted_last_device_end_ns','observed_endpoint_ns':'observed_last_device_end_ns'})
    old=pd.concat(control_tables,ignore_index=True)
    order=['release_candidate','method']+GROUP
    controls=controls.sort_values(order).reset_index(drop=True);old=old[controls.columns].sort_values(order).reset_index(drop=True)
    pd.testing.assert_frame_equal(controls,old,check_exact=True)
    old_metrics=pd.read_csv(paths['target_release_endpoint_metrics.csv']).rename(columns={'method':'component_method'})
    new_metrics=metric_table[metric_table.endpoint_rule.eq('predicted_maximum')][
        ['release_candidate','component_method','launch_FB_phase','windows','endpoint_MAE_ms','endpoint_bias_ms']].rename(columns={'windows':'phases'})
    columns=['release_candidate','component_method','launch_FB_phase','phases','endpoint_MAE_ms','endpoint_bias_ms']
    new_metrics=new_metrics[columns].sort_values(columns[:3]).reset_index(drop=True)
    old_metrics=old_metrics[columns].sort_values(columns[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(new_metrics[columns[:4]],old_metrics[columns[:4]],check_exact=True,check_dtype=False)
    np.testing.assert_allclose(new_metrics[['endpoint_MAE_ms','endpoint_bias_ms']],
        old_metrics[['endpoint_MAE_ms','endpoint_bias_ms']],rtol=0,atol=5e-13)
    for file in json.loads((destination/'target_terminal_endpoint_prediction_seal.json').read_text())['files']:
        assert sha(destination/file['path'])==file['sha256']
    draw(destination,metric_table)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;elapsed=perf_counter()-begin
    assert peak<=review['resource']['maximum_peak_RSS_bytes'] and elapsed<=review['resource']['target_analysis_seconds']
    selected_metrics=metric_table[metric_table.endpoint_rule.eq('source_fit_selected')]
    semantic_metrics=metric_table[(metric_table.endpoint_rule.eq('fitted_terminal_event'))&metric_table.component_method.eq('independent_latency')]
    dump(out/'diagnostic.json',dict(status='TARGET_TERMINAL_ENDPOINT_TRANSFER_PASS',conditional_device_prediction=True,
        new_prediction=True,new_global_prediction=False,used_to_fit_model=False,new_target_timing_read=True,
        source_parameter_updates=0,target_parameter_updates=0,raw_trace_scanned=False,
        target_iterations=[85,90,95,100],target_ranks=[16],target_FB_windows=len(truth),prediction_rows=len(predictions),
        source_terminal_identity_coverage=int(truth.observed_CPU_owner_name_checked.eq('aten::_copy_from').sum()),
        selected_metrics=selected_metrics.to_dict('records'),semantic_independent_metrics=semantic_metrics.to_dict('records'),
        T27B_predicted_maximum_controls_exact=True,target_endpoint_truth_mutation_predictions_exact=True,
        truth_join_after_prediction_seal=True,
        formal_topology_changed=False,prediction_seal_sha256=sha(destination/'target_terminal_endpoint_prediction_seal.json'),
        peak_RSS_bytes=peak,analysis_seconds=elapsed,resource_gate_pass=True,
        next='Use the timing-versus-identity tradeoff to choose the next source-only missing dependency or cost hypothesis; do not promote a conditional endpoint to global1F1B.'))
    dump(out/'field_contract.json',dict(
        source='Terminal identity, phase selection and tail are byte-exact T28A source85/90 parameters.',
        target_prediction='T27B event predictions and target CPU submission metadata were already sealed; new endpoint rules are sealed before target endpoint truth joins.',
        truth='Target endpoint times and identities are development/posthoc and never select the rule.',
        accounting='Alternative endpoint estimates are compared, never summed or added to CPU F/B; no duplicate compute/communication/wait accounting.',
        global_scope='No graph edge, wait, CPU F/B, global1F1B, Step or MFU prediction.'))


def draw(out,metrics_table):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='w37-t28b-target-terminal-endpoint'
    labels=[('predicted_maximum','same_stream_median','max median'),('predicted_maximum','same_stream_mean','max mean'),
        ('source_fit_selected','phase_selected','source-fit selected'),('fitted_terminal_event','independent_latency','semantic independent'),
        ('CPU_annotation_phase_median_tail','phase_median','CPU tail median')]
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,phase in zip(axes,['forward','backward']):
        d=metrics_table[metrics_table.launch_FB_phase.eq(phase)]
        for index,release in enumerate(RELEASES):
            values=[]
            for rule,method,_ in labels:
                row=d[d.release_candidate.eq(release)&d.endpoint_rule.eq(rule)&d.component_method.eq(method)]
                values.append(float(row.endpoint_MAE_ms.iloc[0]))
            ax.bar(np.arange(len(labels))+(index-.5)*.36,values,.36,label=release)
        ax.set_xticks(range(len(labels)),[x[2].replace(' ','\n') for x in labels]);ax.set_yscale('symlog',linthresh=.1)
        ax.set_ylabel('Target endpoint MAE (ms, symlog)');ax.set_title(phase);ax.legend()
    fig.suptitle('Target224 conditional endpoint transfer; all rules fixed on source85/90')
    fig.text(.5,.02,'Endpoint timing and terminal-event identity are scored separately. Target data are development/posthoc.\n'
        'Observed CPU submissions are conditions; no graph edge, wait, global1F1B, Step or MFU prediction.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(out/'target_terminal_endpoint_transfer.svg',metadata={'Date':None})
    fig.savefig(out/'target_terminal_endpoint_transfer.png',dpi=150);plt.close(fig)

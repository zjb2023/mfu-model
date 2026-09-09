"""Source-only terminal-event semantics for CPU-conditioned visible-device endpoints."""
import json
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from guards import ROOT
from smoke_worker import dump, sha
from worker import csv
from device_queue import prepare, stream_audit, DTYPE
from device_queue_model import DeviceQueueModel, feature_frame, METHODS
from device_release import api_start_proxy


FIT = [85, 90]
VALIDATION = [95, 100]
GROUP = ['iteration', 'rank', 'split', 'launch_FB_phase', 'launch_FB_window_id']
IDENTITY = ['CPU_owner_name_checked', 'stream', 'family', 'category']


def observed_endpoints(data):
    """Choose the actual visible non-PP endpoint; callers control the allowed split."""
    local = data[~data.family.eq('PP_candidate') & ~data.launch_FB_window_id.eq('outside_FB')].copy()
    local = local.sort_values(GROUP + ['end_ns', 'event_id'])
    result = local.groupby(GROUP, sort=True, as_index=False).tail(1).copy()
    assert not result.duplicated(GROUP).any()
    result = result[GROUP + ['event_id', 'end_ns'] + IDENTITY].rename(columns={
        'event_id': 'observed_endpoint_event_id', 'end_ns': 'observed_endpoint_ns',
        **{name: 'observed_' + name for name in IDENTITY}})
    return result.sort_values(GROUP).reset_index(drop=True)


def fit_terminal_identity(fit_endpoints):
    """Require one exact endpoint identity to cover every fit window in each phase."""
    assert set(fit_endpoints.iteration) == set(FIT)
    rows = []
    for phase, group in fit_endpoints.groupby('launch_FB_phase', sort=True):
        counts = group.groupby(['observed_' + name for name in IDENTITY], dropna=False).size().rename('fit_endpoint_windows').reset_index()
        counts = counts.sort_values(['fit_endpoint_windows'] + ['observed_' + name for name in IDENTITY], ascending=[False, True, True, True, True])
        winner = counts.iloc[0]
        assert int(winner.fit_endpoint_windows) == len(group), f'No stable terminal identity for {phase}'
        row = {'launch_FB_phase': phase, 'fit_iterations': '85,90', 'fit_endpoint_windows': len(group)}
        row.update({name: winner['observed_' + name] for name in IDENTITY})
        rows.append(row)
    result = pd.DataFrame(rows)
    assert set(result.launch_FB_phase) == {'forward', 'backward'}
    return result


def select_terminal_events(features, identities):
    """Select the last CPU-submitted event with the frozen phase identity."""
    features = features.copy()
    features['split'] = np.where(features.iteration.isin(FIT), 'source_fit', 'source_incremental_validation')
    rows = []
    for identity in identities.to_dict('records'):
        group = features[features.launch_FB_phase.eq(identity['launch_FB_phase']) & ~features.launch_FB_window_id.eq('outside_FB')]
        for name in IDENTITY:
            group = group[group[name].astype(str).eq(str(identity[name]))]
        group = group.sort_values(GROUP + ['runtime_start_ns', 'runtime_end_ns', 'event_id'])
        group = group.groupby(GROUP, sort=True, as_index=False).tail(1).copy()
        rows.append(group[GROUP + ['event_id'] + IDENTITY].rename(columns={'event_id': 'selected_event_id'}))
    result = pd.concat(rows, ignore_index=True).sort_values(GROUP).reset_index(drop=True)
    assert len(result) == 32 and not result.duplicated(GROUP).any()
    return result


def endpoint_candidates(features, predictions, terminal_events, release_candidate):
    """Build prediction-only envelope and semantic-terminal candidates."""
    meta = features[['event_id'] + [name for name in GROUP if name != 'split'] + IDENTITY].copy()
    meta['split'] = np.where(meta.iteration.isin(FIT), 'source_fit', 'source_incremental_validation')
    joined = predictions.merge(meta, on=['event_id', 'iteration', 'rank'], validate='many_to_one')
    joined = joined[~joined.family.eq('PP_candidate') & ~joined.launch_FB_window_id.eq('outside_FB')]
    joined = joined.sort_values(['method'] + GROUP + ['predicted_end_ns', 'event_id'])
    envelope = joined.groupby(['method'] + GROUP, sort=True, as_index=False).tail(1).copy()
    envelope = envelope.rename(columns={'event_id': 'selected_event_id', 'method': 'component_method',
                                        'predicted_end_ns': 'predicted_endpoint_ns'})
    envelope['endpoint_rule'] = 'predicted_maximum'

    semantic = terminal_events.merge(predictions, left_on=['selected_event_id', 'iteration', 'rank'],
        right_on=['event_id', 'iteration', 'rank'], validate='one_to_many')
    semantic = semantic.rename(columns={'method': 'component_method', 'predicted_end_ns': 'predicted_endpoint_ns'})
    semantic['endpoint_rule'] = 'fitted_terminal_event'
    keep = GROUP + ['endpoint_rule', 'component_method', 'selected_event_id', 'predicted_endpoint_ns']
    result = pd.concat([envelope[keep], semantic[keep]], ignore_index=True)
    result['release_candidate'] = release_candidate
    result['prediction_scope'] = 'CPU_submission_conditioned_visible_device_endpoint_NOT_global1F1B'
    assert len(result) == 2 * len(METHODS) * 32
    return result.sort_values(['release_candidate', 'endpoint_rule', 'component_method'] + GROUP).reset_index(drop=True)


def fit_phase_selection(candidates, fit_truth):
    fit = candidates[candidates.iteration.isin(FIT)].merge(
        fit_truth[GROUP + ['observed_endpoint_ns']], on=GROUP, validate='many_to_one')
    fit['absolute_error_ns'] = (fit.predicted_endpoint_ns - fit.observed_endpoint_ns).abs()
    scores = fit.groupby(['release_candidate', 'launch_FB_phase', 'endpoint_rule', 'component_method']).agg(
        fit_endpoint_windows=('launch_FB_window_id', 'size'), fit_endpoint_MAE_ns=('absolute_error_ns', 'mean')).reset_index()
    scores['fit_endpoint_MAE_ns'] = scores.fit_endpoint_MAE_ns.round().astype('int64')
    scores = scores.sort_values(['release_candidate', 'launch_FB_phase', 'fit_endpoint_MAE_ns', 'endpoint_rule', 'component_method'])
    selected = scores.groupby(['release_candidate', 'launch_FB_phase'], sort=True, as_index=False).head(1).copy()
    selected['fit_iterations'] = '85,90'
    assert len(selected) == 4
    return selected.reset_index(drop=True), scores


def selected_candidate_rows(candidates, selection):
    rows = []
    for selected in selection.to_dict('records'):
        group = candidates[
            candidates.release_candidate.eq(selected['release_candidate']) &
            candidates.launch_FB_phase.eq(selected['launch_FB_phase']) &
            candidates.endpoint_rule.eq(selected['endpoint_rule']) &
            candidates.component_method.eq(selected['component_method'])].copy()
        group['source_selected_component_rule'] = group.endpoint_rule
        group['source_selected_component_method'] = group.component_method
        group['endpoint_rule'] = 'source_fit_selected'
        group['component_method'] = 'phase_selected'
        rows.append(group)
    result = pd.concat(rows, ignore_index=True)
    assert len(result) == 2 * 32
    return result


def fit_tail_baseline(windows, fit_truth, prediction_groups, releases):
    fb = windows[windows.window_type.eq('FB_annotation')][
        ['iteration', 'rank', 'phase', 'window_id', 'microbatch', 'end_ns']].copy()
    fb = fb.rename(columns={'phase': 'launch_FB_phase', 'window_id': 'launch_FB_window_id',
                            'end_ns': 'CPU_annotation_end_ns'})
    fb = fb.merge(prediction_groups, on=['iteration', 'rank', 'launch_FB_phase', 'launch_FB_window_id'], validate='one_to_one')
    train = fb[fb.iteration.isin(FIT)].merge(fit_truth[GROUP + ['observed_endpoint_ns']], on=GROUP, validate='one_to_one')
    train['observed_tail_ns'] = train.observed_endpoint_ns - train.CPU_annotation_end_ns
    parameters = train.groupby('launch_FB_phase').observed_tail_ns.median().round().astype('int64').rename('phase_median_tail_ns').reset_index()
    parameters['fit_iterations'] = '85,90'
    base = fb.merge(parameters, on='launch_FB_phase', validate='many_to_one')
    base['predicted_endpoint_ns'] = base.CPU_annotation_end_ns + base.phase_median_tail_ns
    base['endpoint_rule'] = 'CPU_annotation_phase_median_tail'
    base['component_method'] = 'phase_median'
    base['selected_event_id'] = ''
    base['prediction_scope'] = 'observed_CPU_FB_annotation_conditioned_tail_NOT_global1F1B'
    rows = []
    for release in releases:
        copy = base.copy(); copy['release_candidate'] = release; rows.append(copy)
    return pd.concat(rows, ignore_index=True), parameters


def build_prediction_state(prepared, windows, mutate_validation_truth=False):
    fit_truth = observed_endpoints(prepared[prepared.iteration.isin(FIT)])
    identities = fit_terminal_identity(fit_truth)
    candidates = []; parameters = []; edges = None; terminal_events = None
    for release, raw in [('legacy_API_end', prepared), ('all_API_start', api_start_proxy(prepared))]:
        audit, data, candidate_edges = stream_audit(raw)
        assert audit.CPU_order_unambiguous.all()
        assert audit.observed_GPU_start_order_inversions.eq(0).all() and audit.observed_same_stream_GPU_overlaps.eq(0).all()
        if edges is None: edges = candidate_edges
        else: pd.testing.assert_frame_equal(edges, candidate_edges, check_exact=True)
        if mutate_validation_truth:
            data = data.copy()
            for column in ['start_ns', 'end_ns', 'duration_ns', 'observed_enqueue_remainder_ns', 'observed_API_latency_ns']:
                data.loc[data.iteration.isin(VALIDATION), column] = -999999
        model = DeviceQueueModel(data)
        table = model.parameters.copy(); table['release_candidate'] = release; parameters.append(table)
        features = feature_frame(data)
        selected = select_terminal_events(features, identities)
        if terminal_events is None: terminal_events = selected
        else: pd.testing.assert_frame_equal(terminal_events, selected, check_exact=True)
        candidates.append(endpoint_candidates(features, model.predict(features), selected, release))
    candidates = pd.concat(candidates, ignore_index=True)
    selection, fit_scores = fit_phase_selection(candidates, fit_truth)
    chosen = selected_candidate_rows(candidates, selection)
    tails, tail_parameters = fit_tail_baseline(windows, fit_truth, candidates[GROUP].drop_duplicates(),
        ['legacy_API_end', 'all_API_start'])
    predictions = pd.concat([candidates, chosen, tails[candidates.columns.intersection(tails.columns).tolist()]], ignore_index=True, sort=False)
    required = ['release_candidate', 'endpoint_rule', 'component_method'] + GROUP + [
        'selected_event_id', 'predicted_endpoint_ns', 'prediction_scope']
    predictions = predictions[required].sort_values(['release_candidate', 'endpoint_rule', 'component_method'] + GROUP).reset_index(drop=True)
    return dict(identities=identities, terminal_events=terminal_events,
        cost_parameters=pd.concat(parameters, ignore_index=True), selection=selection,
        fit_scores=fit_scores, tail_parameters=tail_parameters, predictions=predictions, edges=edges)


def score_predictions(predictions, truth, observations):
    scored = predictions.merge(truth, on=GROUP, validate='many_to_one')
    scored['endpoint_error_ns'] = scored.predicted_endpoint_ns - scored.observed_endpoint_ns
    scored['endpoint_identity_applicable'] = scored.selected_event_id.fillna('').ne('')
    scored['endpoint_event_identity_match'] = np.where(scored.endpoint_identity_applicable,
        scored.selected_event_id.eq(scored.observed_endpoint_event_id), pd.NA)
    lookup = observations[['event_id', 'iteration', 'rank'] + IDENTITY].rename(columns={
        'event_id': 'selected_event_id', **{name: 'predicted_' + name for name in IDENTITY}})
    scored = scored.merge(lookup, on=['selected_event_id', 'iteration', 'rank'], how='left', validate='many_to_one')
    return scored


def endpoint_metrics(group):
    applicable = group[group.endpoint_identity_applicable]
    return pd.Series(dict(windows=len(group), endpoint_MAE_ms=group.endpoint_error_ns.abs().mean()/1e6,
        endpoint_bias_ms=group.endpoint_error_ns.mean()/1e6,
        identity_applicable_windows=len(applicable),
        endpoint_event_identity_matches=int(applicable.endpoint_event_identity_match.fillna(False).sum()),
        endpoint_event_identity_match_rate=(float(applicable.endpoint_event_identity_match.fillna(False).mean()) if len(applicable) else np.nan)))


def diagnose(out, paths, plan):
    begin = perf_counter()
    assert plan['diagnostic_access'] == 'source_only' and not plan['variants']
    review = json.loads((ROOT / plan['resource_review_file']).read_text())
    assert review['fit_iterations'] == FIT and review['incremental_validation'] == VALIDATION
    windows = pd.read_csv(paths['runtime_windows.csv.gz'], dtype=DTYPE, low_memory=False)
    prepared = []
    for iteration in FIT + VALIDATION:
        gpu = pd.read_csv(paths[f'source{iteration}_GPU.csv.gz'], dtype=DTYPE, low_memory=False)
        runtime = pd.read_csv(paths[f'source{iteration}_runtime.csv.gz'], dtype=DTYPE, low_memory=False)
        cpu = pd.read_csv(paths[f'source{iteration}_CPU.csv.gz'], dtype=DTYPE, low_memory=False)
        prepared.append(prepare(gpu, runtime, cpu, windows[windows.iteration.eq(iteration)]))
    prepared = pd.concat(prepared, ignore_index=True)
    assert len(prepared) == 41764 and set(prepared.iteration) == set(FIT + VALIDATION)

    state = build_prediction_state(prepared, windows)
    mutation_state = build_prediction_state(prepared, windows, mutate_validation_truth=True)
    for name in ['identities', 'terminal_events', 'cost_parameters', 'selection', 'fit_scores', 'tail_parameters', 'predictions', 'edges']:
        pd.testing.assert_frame_equal(state[name], mutation_state[name], check_exact=True)
    del mutation_state

    csv(out, 'source_terminal_identity_parameters.csv', state['identities'])
    csv(out, 'source_terminal_candidate_event_selection.csv.gz', state['terminal_events'])
    csv(out, 'source_terminal_endpoint_fit_candidate_scores.csv', state['fit_scores'])
    csv(out, 'source_terminal_endpoint_selection_parameters.csv', state['selection'])
    csv(out, 'source_phase_tail_parameters.csv', state['tail_parameters'])
    csv(out, 'source_terminal_device_cost_parameters.csv.gz', state['cost_parameters'])
    csv(out, 'source_terminal_endpoint_predictions_sealed.csv.gz', state['predictions'])
    sealed = ['source_terminal_identity_parameters.csv', 'source_terminal_candidate_event_selection.csv.gz',
        'source_terminal_endpoint_selection_parameters.csv', 'source_phase_tail_parameters.csv',
        'source_terminal_device_cost_parameters.csv.gz', 'source_terminal_endpoint_predictions_sealed.csv.gz']
    dump(out / 'source_terminal_endpoint_prediction_seal.json', dict(
        status='SEALED_SOURCE85_90_TERMINAL_ENDPOINT_RULES_AND_CPU_CONDITIONED_PREDICTIONS',
        fit_iterations=FIT, incremental_validation=VALIDATION, target_parameter_updates=0,
        files=[dict(path=name, sha256=sha(out/name)) for name in sealed],
        selection='Terminal identity, phase rule/method and CPU-annotation tail fitted only on source85/90 before source95/100 endpoint truth scoring.',
        conditions='Observed CPU API submissions, event metadata and CPU F/B annotation end. Not a free-running CPU or global224 prediction.',
        mutation_gate='Changing source95/100 GPU truth leaves identities, parameters, event selection and predictions exact.'))

    truth = observed_endpoints(prepared)
    scored = score_predictions(state['predictions'], truth, prepared)
    csv(out, 'source_terminal_endpoint_results.csv.gz', scored)
    keys = ['release_candidate', 'endpoint_rule', 'component_method', 'split', 'launch_FB_phase']
    metrics = scored.groupby(keys).apply(endpoint_metrics, include_groups=False).reset_index()
    per_iteration = scored.groupby(keys[:-2] + ['iteration', 'split', 'launch_FB_phase']).apply(
        endpoint_metrics, include_groups=False).reset_index()
    csv(out, 'source_terminal_endpoint_metrics.csv', metrics)
    csv(out, 'source_terminal_endpoint_per_iteration.csv', per_iteration)
    identity_counts = truth.groupby(['split', 'launch_FB_phase'] + ['observed_' + name for name in IDENTITY]).size().rename('windows').reset_index()
    csv(out, 'source_observed_terminal_identity_counts.csv', identity_counts)
    selection_columns = GROUP + ['release_candidate', 'endpoint_rule', 'component_method', 'selected_event_id',
        'observed_endpoint_event_id', 'endpoint_event_identity_match'] + ['predicted_' + name for name in IDENTITY] + ['observed_' + name for name in IDENTITY]
    csv(out, 'source_terminal_endpoint_event_selection_validation.csv.gz', scored[selection_columns])
    for file in json.loads((out/'source_terminal_endpoint_prediction_seal.json').read_text())['files']:
        assert sha(out/file['path']) == file['sha256']
    draw(out, metrics)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= review['resource']['maximum_peak_RSS_bytes'] and elapsed <= review['resource']['target_analysis_seconds']
    incremental = metrics[metrics.split.eq('source_incremental_validation')]
    selected_incremental = incremental[incremental.endpoint_rule.eq('source_fit_selected')]
    dump(out / 'field_contract.json', dict(
        endpoint='Last visible non-PP device event launched by a CPU operation inside each F/B annotation; not the CPU F/B boundary.',
        terminal_identity='Learned only from source85/90 actual endpoint owner/stream/family/category; source95/100 identity is validation truth.',
        prediction='Event choices use CPU submission order and frozen identity; completion uses source85/90 device costs. Validation GPU time is not a feature.',
        accounting='No device duration or tail is added to CPU F/B. Every candidate is an alternative endpoint estimate, so overlapping work is never summed.',
        boundary='Source95/100 is historically exposed incremental validation. No target semantic access, global1F1B, Step/MFU prediction or topology change.'))
    dump(out / 'diagnostic.json', dict(status='SOURCE_TERMINAL_ENDPOINT_SEMANTICS_PASS',
        new_prediction=True, new_global_prediction=False, used_to_fit_model=True, new_target_timing_read=False,
        raw_trace_scanned=False, source_iterations=FIT+VALIDATION, source_fit_iterations=FIT,
        source_incremental_validation=VALIDATION, source_ranks=[16], device_events=len(prepared),
        CPU_owned_FB_windows=len(truth), terminal_identity_fit_windows=len(truth[truth.iteration.isin(FIT)]),
        terminal_identity_validation_windows=len(truth[truth.iteration.isin(VALIDATION)]),
        prediction_rows=len(state['predictions']), parameter_rows=len(state['cost_parameters']),
        terminal_identity_parameters=state['identities'].to_dict('records'),
        source_fit_selected_parameters=state['selection'].to_dict('records'),
        incremental_selected_metrics=selected_incremental.to_dict('records'),
        validation_GPU_mutation_all_parameters_selections_predictions_exact=True,
        formal_topology_changed=False, prediction_seal_sha256=sha(out/'source_terminal_endpoint_prediction_seal.json'),
        peak_RSS_bytes=peak, analysis_seconds=elapsed, resource_gate_pass=True,
        next='Use source incremental identity and endpoint metrics to decide whether a separately sealed target conditional check is warranted; no global promotion.'))


def draw(out, metrics):
    import os
    os.environ['MPLCONFIGDIR'] = str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = metrics[metrics.split.eq('source_incremental_validation')].copy()
    labels = []
    for rule, method in [
        ('predicted_maximum', 'same_stream_median'),
        ('predicted_maximum', 'same_stream_mean'),
        ('fitted_terminal_event', 'independent_latency'),
        ('source_fit_selected', 'phase_selected'),
        ('CPU_annotation_phase_median_tail', 'phase_median')]:
        labels.append((rule, method))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, phase in zip(axes, ['forward', 'backward']):
        subset = data[data.launch_FB_phase.eq(phase)]
        for index, release in enumerate(['legacy_API_end', 'all_API_start']):
            values = []
            for rule, method in labels:
                row = subset[subset.release_candidate.eq(release) & subset.endpoint_rule.eq(rule) & subset.component_method.eq(method)]
                values.append(float(row.endpoint_MAE_ms.iloc[0]))
            x = np.arange(len(labels)) + (index-.5)*.36
            ax.bar(x, values, .36, label=release)
        ax.set_xticks(range(len(labels)), ['max\nmedian', 'max\nmean', 'terminal\nindependent', 'source-fit\nselected', 'CPU tail\nmedian'])
        ax.set_yscale('symlog', linthresh=.1); ax.set_ylabel('Endpoint MAE (ms, symlog)'); ax.set_title(phase); ax.legend()
    fig.suptitle('Source95/100 endpoint semantics; identity and rule selected only on source85/90')
    fig.text(.5, .02, 'Observed CPU submissions are conditions. Visible-device endpoint is not CPU F/B or global1F1B.\n'
        'Identity match and timing error are reported separately; no edge, wait or target cost is fitted.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95)); fig.savefig(out/'source_terminal_endpoint_semantics.svg')
    fig.savefig(out/'source_terminal_endpoint_semantics.png', dpi=150); plt.close(fig)

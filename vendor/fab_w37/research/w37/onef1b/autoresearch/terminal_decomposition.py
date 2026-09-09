"""Source-only free-running terminal timing and PP overlap audit."""
import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from guards import ROOT
from smoke_worker import dump, sha
from worker import csv


FIT = [85, 90]
VALIDATION = [95, 100]
GROUP = ['iteration', 'rank', 'split', 'launch_FB_phase', 'launch_FB_window_id']
METHODS = [
    'decomposed_role_submission_phase_post',
    'joint_endpoint_role_median',
    'joint_endpoint_phase_mean',
    'joint_endpoint_phase_median',
    'CPU_annotation_role_median_ablation',
]


def microbatch_role(value, maximum=3):
    value = int(value)
    if value == 0:
        return 'first'
    if value == maximum:
        return 'last'
    return 'middle'


def _rounded(values, estimator):
    value = values.mean() if estimator == 'mean' else values.median()
    return int(np.rint(value))


def fit_and_predict(observations):
    """Fit on 85/90; validation predictions use only F/B start, phase and MB role."""
    fit = observations[observations.iteration.isin(FIT)].copy()
    assert len(fit) == 16 and set(fit.split) == {'source_fit'}
    rows = []

    def fitted(method, component, keys, field, estimator):
        grouped = fit.groupby(keys, sort=True)[field]
        for labels, values in grouped:
            if not isinstance(labels, tuple):
                labels = (labels,)
            row = dict(method=method, component=component, estimator=estimator,
                       fit_iterations='85,90', samples=len(values), value_ns=_rounded(values, estimator),
                       launch_FB_phase='', microbatch_role='all')
            row.update(dict(zip(keys, labels)))
            rows.append(row)

    main = METHODS[0]
    fitted(main, 'terminal_submission_offset', ['launch_FB_phase', 'microbatch_role'],
           'terminal_API_start_from_FB_start_ns', 'median')
    fitted(main, 'post_submission_device_completion', ['launch_FB_phase'],
           'terminal_GPU_end_from_API_start_ns', 'median')
    fitted(METHODS[1], 'joint_terminal_endpoint_offset', ['launch_FB_phase', 'microbatch_role'],
           'terminal_GPU_end_from_FB_start_ns', 'median')
    fitted(METHODS[2], 'joint_terminal_endpoint_offset', ['launch_FB_phase'],
           'terminal_GPU_end_from_FB_start_ns', 'mean')
    fitted(METHODS[3], 'joint_terminal_endpoint_offset', ['launch_FB_phase'],
           'terminal_GPU_end_from_FB_start_ns', 'median')
    fitted(METHODS[4], 'CPU_annotation_duration', ['launch_FB_phase', 'microbatch_role'],
           'CPU_FB_wall_ns', 'median')
    parameters = pd.DataFrame(rows).sort_values(
        ['method', 'component', 'launch_FB_phase', 'microbatch_role']).reset_index(drop=True)

    def lookup(method, component, phase, role='all'):
        match = parameters[
            parameters.method.eq(method) & parameters.component.eq(component) &
            parameters.launch_FB_phase.eq(phase) & parameters.microbatch_role.eq(role)]
        assert len(match) == 1
        return int(match.value_ns.iloc[0])

    predictions = []
    features = observations[GROUP + ['microbatch', 'microbatch_role', 'FB_start_ns']].copy()
    for row in features.to_dict('records'):
        phase, role = row['launch_FB_phase'], row['microbatch_role']
        for method in METHODS:
            submission_offset = np.nan
            if method == main:
                submission_offset = lookup(method, 'terminal_submission_offset', phase, role)
                endpoint_offset = submission_offset + lookup(
                    method, 'post_submission_device_completion', phase)
            elif method == METHODS[1]:
                endpoint_offset = lookup(method, 'joint_terminal_endpoint_offset', phase, role)
            elif method in METHODS[2:4]:
                endpoint_offset = lookup(method, 'joint_terminal_endpoint_offset', phase)
            else:
                endpoint_offset = lookup(method, 'CPU_annotation_duration', phase, role)
            prediction = {key: row[key] for key in GROUP}
            prediction.update(method=method, microbatch=int(row['microbatch']), microbatch_role=role,
                predicted_submission_offset_ns=submission_offset,
                predicted_submission_ns=(row['FB_start_ns'] + submission_offset if method == main else np.nan),
                predicted_endpoint_offset_ns=int(endpoint_offset),
                predicted_endpoint_ns=int(row['FB_start_ns'] + endpoint_offset),
                prediction_features='FB_start_ns|launch_FB_phase|microbatch_role',
                prediction_scope='source_local_scheduled_FB_start_conditioned_NOT_global1F1B')
            predictions.append(prediction)
    predictions = pd.DataFrame(predictions).sort_values(['method'] + GROUP).reset_index(drop=True)
    assert len(predictions) == 32 * len(METHODS)
    return parameters, predictions


def _load_observations(paths):
    windows = pd.read_csv(paths['runtime_windows.csv.gz'], low_memory=False)
    fb = windows[windows.window_type.eq('FB_annotation')][
        ['iteration', 'rank', 'window_id', 'phase', 'microbatch', 'start_ns', 'end_ns',
         'CP_union_ns', 'non_PP_GPU_activity_union_ns', 'PP_candidate_union_ns',
         'no_non_PP_GPU_activity_visible_ns', 'all_thread_sync_union_ns']].copy()
    fb = fb.rename(columns={'window_id': 'launch_FB_window_id', 'phase': 'launch_FB_phase',
        'start_ns': 'FB_start_ns', 'end_ns': 'FB_end_ns'})
    fb['split'] = np.where(fb.iteration.isin(FIT), 'source_fit', 'source_incremental_validation')
    fb['microbatch_role'] = fb.microbatch.map(microbatch_role)
    selected = pd.read_csv(paths['source_terminal_candidate_event_selection.csv.gz'], dtype={'stream': 'string'})
    selected['stream'] = selected.stream.astype(str)
    fb = fb.merge(selected, on=GROUP, validate='one_to_one')
    assert len(fb) == 32 and fb.CPU_owner_name_checked.eq('aten::_copy_from').all()
    assert fb.stream.eq('0').all() and fb.family.eq('elementwise').all() and fb.category.eq('kernel').all()

    gpu_parts, runtime_parts, cpu_parts = [], [], []
    for iteration in FIT + VALIDATION:
        gpu = pd.read_csv(paths[f'source{iteration}_GPU.csv.gz'], low_memory=False)
        chosen = gpu[gpu.event_id.isin(fb[fb.iteration.eq(iteration)].selected_event_id)][
            ['iteration', 'rank', 'event_id', 'start_ns', 'end_ns', 'unique_runtime_event_id',
             'unique_CPU_owner_name', 'stream', 'family', 'category']].copy()
        gpu_parts.append(chosen)
        runtime_parts.append(pd.read_csv(paths[f'source{iteration}_runtime.csv.gz'], low_memory=False)[
            ['iteration', 'rank', 'event_id', 'start_ns', 'end_ns', 'name']])
        cpu_parts.append(pd.read_csv(paths[f'source{iteration}_CPU.csv.gz'], low_memory=False)[
            ['iteration', 'rank', 'event_id', 'name']])
    gpu = pd.concat(gpu_parts, ignore_index=True).rename(columns={'event_id': 'selected_event_id',
        'start_ns': 'terminal_GPU_start_ns', 'end_ns': 'terminal_GPU_end_ns',
        'unique_CPU_owner_name': 'terminal_CPU_owner_name', 'stream': 'terminal_stream',
        'family': 'terminal_family', 'category': 'terminal_category'})
    runtime = pd.concat(runtime_parts, ignore_index=True).rename(columns={'event_id': 'unique_runtime_event_id',
        'start_ns': 'terminal_API_start_ns', 'end_ns': 'terminal_API_end_ns', 'name': 'terminal_API_name'})
    fb = fb.merge(gpu, on=['iteration', 'rank', 'selected_event_id'], validate='one_to_one')
    fb = fb.merge(runtime, on=['iteration', 'rank', 'unique_runtime_event_id'], validate='one_to_one')
    assert fb.terminal_CPU_owner_name.eq(fb.CPU_owner_name_checked).all()
    assert fb.terminal_stream.astype(str).eq(fb.stream.astype(str)).all()
    assert fb.terminal_family.eq(fb.family).all() and fb.terminal_category.eq(fb.category).all()

    cpu = pd.concat(cpu_parts, ignore_index=True)
    pp = windows[windows.window_type.eq('PP_API')][
        ['iteration', 'rank', 'window_id', 'start_ns', 'end_ns']].copy()
    pp = pp.merge(cpu, left_on=['iteration', 'rank', 'window_id'],
                  right_on=['iteration', 'rank', 'event_id'], validate='one_to_one')
    pp = pp[pp.name.isin(['send_forward', 'send_backward'])]
    matched = []
    all_fb = fb.sort_values(['iteration', 'FB_start_ns'])
    for row in all_fb.to_dict('records'):
        expected = 'send_forward' if row['launch_FB_phase'] == 'forward' else 'send_backward'
        candidates = pp[pp.iteration.eq(row['iteration']) & pp['rank'].eq(row['rank']) &
                        pp.name.eq(expected) & pp.start_ns.ge(row['FB_end_ns'])].sort_values('start_ns')
        assert len(candidates) >= 1
        following = candidates.iloc[0]
        later_fb = all_fb[all_fb.iteration.eq(row['iteration']) & all_fb.FB_start_ns.gt(row['FB_end_ns'])]
        matched.append(dict(**row, following_PP_window_id=following.window_id,
            following_PP_name=following['name'], following_PP_start_ns=int(following.start_ns),
            following_PP_end_ns=int(following.end_ns),
            next_FB_start_ns=(int(later_fb.FB_start_ns.min()) if len(later_fb) else np.nan)))
    result = pd.DataFrame(matched).sort_values(GROUP).reset_index(drop=True)
    assert len(result) == 32 and not result.following_PP_window_id.duplicated().any()
    result['CPU_FB_wall_ns'] = result.FB_end_ns - result.FB_start_ns
    result['terminal_API_start_from_FB_start_ns'] = result.terminal_API_start_ns - result.FB_start_ns
    result['terminal_API_end_from_FB_start_ns'] = result.terminal_API_end_ns - result.FB_start_ns
    result['terminal_GPU_start_from_FB_start_ns'] = result.terminal_GPU_start_ns - result.FB_start_ns
    result['terminal_GPU_end_from_FB_start_ns'] = result.terminal_GPU_end_ns - result.FB_start_ns
    result['terminal_GPU_end_from_API_start_ns'] = result.terminal_GPU_end_ns - result.terminal_API_start_ns
    result['terminal_GPU_end_from_FB_end_ns'] = result.terminal_GPU_end_ns - result.FB_end_ns
    result['FB_end_to_PP_start_ns'] = result.following_PP_start_ns - result.FB_end_ns
    result['PP_start_to_terminal_GPU_end_ns'] = result.terminal_GPU_end_ns - result.following_PP_start_ns
    result['terminal_GPU_end_to_PP_end_ns'] = result.following_PP_end_ns - result.terminal_GPU_end_ns
    result['following_PP_duration_ns'] = result.following_PP_end_ns - result.following_PP_start_ns
    result['terminal_API_start_inside_FB'] = result.terminal_API_start_ns.between(result.FB_start_ns, result.FB_end_ns)
    result['terminal_GPU_end_inside_following_PP'] = result.terminal_GPU_end_ns.between(
        result.following_PP_start_ns, result.following_PP_end_ns)
    result['terminal_crosses_next_FB_start'] = result.next_FB_start_ns.notna() & result.terminal_GPU_end_ns.gt(result.next_FB_start_ns)
    positive_tail_start = np.maximum(result.FB_end_ns, result.following_PP_start_ns)
    positive_tail_end = np.minimum(result.terminal_GPU_end_ns, result.following_PP_end_ns)
    result['positive_terminal_tail_ns'] = np.maximum(0, result.terminal_GPU_end_ns - result.FB_end_ns)
    result['terminal_tail_overlap_following_PP_ns'] = np.maximum(0, positive_tail_end - positive_tail_start)
    result['decomposition_conservation_error_ns'] = (
        result.following_PP_end_ns - result.FB_start_ns -
        (result.CPU_FB_wall_ns + result.FB_end_to_PP_start_ns +
         result.PP_start_to_terminal_GPU_end_ns + result.terminal_GPU_end_to_PP_end_ns))
    assert result.decomposition_conservation_error_ns.eq(0).all()
    return result


def _metrics(group):
    applicable = group.predicted_submission_ns.notna()
    return pd.Series(dict(windows=len(group),
        endpoint_MAE_ms=group.endpoint_error_ns.abs().mean()/1e6,
        endpoint_bias_ms=group.endpoint_error_ns.mean()/1e6,
        submission_windows=int(applicable.sum()),
        submission_MAE_ms=(group.loc[applicable, 'submission_error_ns'].abs().mean()/1e6 if applicable.any() else np.nan),
        submission_bias_ms=(group.loc[applicable, 'submission_error_ns'].mean()/1e6 if applicable.any() else np.nan)))


def _summarize_intervals(observations):
    rows = []
    for (split, phase), group in observations.groupby(['split', 'launch_FB_phase'], sort=True):
        tail = group.positive_terminal_tail_ns.sum()
        overlap = group.terminal_tail_overlap_following_PP_ns.sum()
        rows.append(dict(split=split, launch_FB_phase=phase, windows=len(group),
            terminal_API_start_inside_FB_windows=int(group.terminal_API_start_inside_FB.sum()),
            terminal_GPU_end_inside_following_PP_windows=int(group.terminal_GPU_end_inside_following_PP.sum()),
            terminal_crosses_next_FB_windows=int(group.terminal_crosses_next_FB_start.sum()),
            CPU_FB_wall_mean_ms=group.CPU_FB_wall_ns.mean()/1e6,
            terminal_API_start_from_FB_start_mean_ms=group.terminal_API_start_from_FB_start_ns.mean()/1e6,
            terminal_GPU_end_from_FB_start_mean_ms=group.terminal_GPU_end_from_FB_start_ns.mean()/1e6,
            terminal_GPU_end_from_FB_end_mean_ms=group.terminal_GPU_end_from_FB_end_ns.mean()/1e6,
            FB_end_to_PP_start_mean_ms=group.FB_end_to_PP_start_ns.mean()/1e6,
            PP_start_to_terminal_GPU_end_mean_ms=group.PP_start_to_terminal_GPU_end_ns.mean()/1e6,
            terminal_GPU_end_to_PP_end_mean_ms=group.terminal_GPU_end_to_PP_end_ns.mean()/1e6,
            following_PP_duration_mean_ms=group.following_PP_duration_ns.mean()/1e6,
            positive_tail_overlap_following_PP_pct=(100*overlap/tail if tail else np.nan),
            maximum_absolute_conservation_error_ns=int(group.decomposition_conservation_error_ns.abs().max())))
    return pd.DataFrame(rows)


def _sealed_graph_PP_nodes(plan):
    ref = plan['sealed_reference']
    root = Path(ref['run_root'])/'models'/ref['variant']/'prediction'
    path = root/(ref['variant'] + '_source85')/'source256_nodes.csv.gz'
    nodes = pd.read_csv(path, low_memory=False)
    result = nodes[nodes.kind.isin(['pp_sender_effective_readiness', 'pp_postpublication_completion']) &
        nodes['rank'].eq(16) & nodes.pp_stage.eq(1) & nodes.pp_lane.eq(0) & nodes.direction.eq('B')].copy()
    assert len(result) == 8
    return result[['node_id', 'kind', 'duration_ns', 'rank', 'pp_stage', 'pp_lane', 'microbatch',
                   'api_name', 'cost_key']].sort_values(['microbatch', 'kind']).reset_index(drop=True)


def diagnose(out, paths, plan):
    begin = perf_counter()
    assert plan['diagnostic'] == 'source_terminal_decomposition'
    assert plan['diagnostic_access'] == 'source_only' and not plan['variants']
    review = json.loads((ROOT/plan['resource_review_file']).read_text())
    assert review['fit_iterations'] == FIT and review['incremental_validation'] == VALIDATION
    observations = _load_observations(paths)
    parameters, predictions = fit_and_predict(observations)
    mutated = observations.copy()
    truth_columns = ['FB_end_ns', 'terminal_API_start_ns', 'terminal_API_end_ns',
        'terminal_GPU_start_ns', 'terminal_GPU_end_ns', 'following_PP_start_ns', 'following_PP_end_ns']
    mutated.loc[mutated.iteration.isin(VALIDATION), truth_columns] = -999999
    mutation_parameters, mutation_predictions = fit_and_predict(mutated)
    pd.testing.assert_frame_equal(parameters, mutation_parameters, check_exact=True)
    pd.testing.assert_frame_equal(predictions, mutation_predictions, check_exact=True)

    csv(out, 'source_terminal_decomposition_parameters.csv', parameters)
    csv(out, 'source_terminal_decomposition_predictions_sealed.csv.gz', predictions)
    sealed = ['source_terminal_decomposition_parameters.csv',
              'source_terminal_decomposition_predictions_sealed.csv.gz']
    dump(out/'source_terminal_decomposition_prediction_seal.json', dict(
        status='SEALED_SOURCE85_90_FREE_RUNNING_TERMINAL_DECOMPOSITION',
        fit_iterations=FIT, incremental_validation=VALIDATION, target_parameter_updates=0,
        prediction_features=['FB_start_ns', 'launch_FB_phase', 'microbatch_role'],
        files=[dict(path=name, sha256=sha(out/name)) for name in sealed],
        mutation_gate='Changing source95/100 CPU end, terminal API/device time and PP boundaries leaves parameters and predictions exact.',
        global_condition='FB_start is observed only as a source-local origin; a global candidate must substitute a predicted schedule start.'))

    scored = predictions.merge(observations[GROUP + ['terminal_API_start_ns', 'terminal_GPU_end_ns']],
        on=GROUP, validate='many_to_one')
    scored['endpoint_error_ns'] = scored.predicted_endpoint_ns - scored.terminal_GPU_end_ns
    scored['submission_error_ns'] = scored.predicted_submission_ns - scored.terminal_API_start_ns
    csv(out, 'source_terminal_decomposition_results.csv.gz', scored)
    metrics = scored.groupby(['method', 'split', 'launch_FB_phase']).apply(_metrics, include_groups=False).reset_index()
    per_iteration = scored.groupby(['method', 'iteration', 'split', 'launch_FB_phase']).apply(
        _metrics, include_groups=False).reset_index()
    csv(out, 'source_terminal_decomposition_metrics.csv', metrics)
    csv(out, 'source_terminal_decomposition_per_iteration.csv', per_iteration)
    csv(out, 'source_terminal_PP_interval_ledger.csv.gz', observations)
    interval_summary = _summarize_intervals(observations)
    csv(out, 'source_terminal_PP_interval_summary.csv', interval_summary)
    graph_nodes = _sealed_graph_PP_nodes(plan)
    csv(out, 'sealed_v6813_source85_B_PP_readiness_nodes.csv', graph_nodes)
    for file in json.loads((out/'source_terminal_decomposition_prediction_seal.json').read_text())['files']:
        assert sha(out/file['path']) == file['sha256']

    incremental = metrics[metrics.split.eq('source_incremental_validation')]
    main = incremental[incremental.method.eq(METHODS[0])]
    controls = incremental[~incremental.method.eq(METHODS[0])]
    backward = interval_summary[interval_summary.launch_FB_phase.eq('backward')]
    backward_overlap_min = float(backward.positive_tail_overlap_following_PP_pct.min())
    source_free_running_gate = float(main.endpoint_MAE_ms.mean()) < float(controls.endpoint_MAE_ms.mean())
    additive_gate_rejected = backward_overlap_min > 99.0
    dump(out/'field_contract.json', dict(
        submission='Terminal runtime API start relative to F/B start; the source-local prediction uses phase and microbatch role, never observed CPU end.',
        endpoint='Semantic stream0 aten::_copy_from device completion selected by the sealed T28A identity.',
        PP_overlap='Following direction-matched send_forward/send_backward CPU API. Interval overlap is temporal accounting evidence, not a new causal edge.',
        graph='The sealed v6813 source85 B PP readiness nodes are copied for semantic comparison only; no duration or edge is edited.',
        integration='A terminal tail overlapping PP sender readiness cannot be added to F/B. A future candidate could only replace PP readiness after independent evidence.',
        boundary='Source85/90 fit; exposed source95/100 incremental validation; rank16 only. No target timing, global prediction, Step/MFU or topology change.'))
    draw(out, metrics, interval_summary)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    elapsed = perf_counter()-begin
    assert peak <= review['resource']['maximum_peak_RSS_bytes'] and elapsed <= review['resource']['target_analysis_seconds']
    dump(out/'diagnostic.json', dict(status='SOURCE_TERMINAL_SUBMISSION_DECOMPOSITION_PASS',
        new_prediction=True, new_global_prediction=False, used_to_fit_model=True,
        new_target_timing_read=False, raw_trace_scanned=False, source_iterations=FIT+VALIDATION,
        source_fit_iterations=FIT, source_incremental_validation=VALIDATION, source_ranks=[16],
        terminal_windows=len(observations), terminal_API_launch_inside_FB_windows=int(observations.terminal_API_start_inside_FB.sum()),
        backward_terminal_inside_following_PP_windows=int(observations[observations.launch_FB_phase.eq('backward')].terminal_GPU_end_inside_following_PP.sum()),
        forward_terminal_inside_following_PP_windows=int(observations[observations.launch_FB_phase.eq('forward')].terminal_GPU_end_inside_following_PP.sum()),
        terminal_crosses_next_FB_windows=int(observations.terminal_crosses_next_FB_start.sum()),
        maximum_absolute_interval_conservation_error_ns=int(observations.decomposition_conservation_error_ns.abs().max()),
        backward_minimum_positive_tail_PP_overlap_pct=backward_overlap_min,
        validation_truth_mutation_parameters_predictions_exact=True,
        prediction_seal_sha256=sha(out/'source_terminal_decomposition_prediction_seal.json'),
        source_incremental_main_metrics=main.to_dict('records'),
        source_free_running_diagnostic_gate_pass=source_free_running_gate,
        additive_integration_gate_rejected=additive_gate_rejected,
        promotion='REJECT_ADDITIVE_F_OR_B_TAIL; PP_REPLACEMENT_REQUIRES_INDEPENDENT_IMPROVEMENT',
        formal_topology_changed=False, target_parameter_updates=0,
        peak_RSS_bytes=peak, analysis_seconds=elapsed, resource_gate_pass=True,
        next='Do not run an additive T29B. Compare a sealed replacement of B PP sender readiness only if source evidence beats the existing PP local model; otherwise advance to a different source-supported hypothesis.'))


def draw(out, metrics, intervals):
    import os
    os.environ['MPLCONFIGDIR'] = str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'w37-t29a-terminal-decomposition'
    import matplotlib.pyplot as plt
    data = metrics[metrics.split.eq('source_incremental_validation')]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    short = ['decomposed', 'joint role', 'joint mean', 'joint median', 'CPU only']
    for ax, phase in zip(axes, ['forward', 'backward']):
        subset = data[data.launch_FB_phase.eq(phase)].set_index('method').reindex(METHODS)
        ax.bar(np.arange(len(METHODS)), subset.endpoint_MAE_ms, color='#4c78a8')
        ax.set_xticks(np.arange(len(METHODS)), short, rotation=22, ha='right')
        ax.set_ylabel('source95/100 endpoint MAE (ms)')
        interval = intervals[(intervals.split.eq('source_incremental_validation')) &
                             intervals.launch_FB_phase.eq(phase)].iloc[0]
        ax.set_title(f"{phase}: terminal-tail PP overlap {interval.positive_tail_overlap_following_PP_pct:.3f}%")
    fig.suptitle('Source-only terminal submission/completion prediction and PP overlap audit')
    fig.text(.5, .02, 'Predictions use F/B start, phase and microbatch role. Terminal tails are not added to F/B or PP.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out/'source_terminal_decomposition.svg', metadata={'Date': None})
    fig.savefig(out/'source_terminal_decomposition.png', dpi=150)
    plt.close(fig)

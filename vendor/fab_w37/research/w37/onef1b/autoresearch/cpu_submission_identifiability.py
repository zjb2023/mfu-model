"""Test whether offline CPU graph signatures identify terminal submission drift."""
import hashlib
import json
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from guards import ROOT
from smoke_worker import dump, sha
from worker import csv


FIT = [85, 90]
VALIDATION = [95, 100]
GROUP = ['iteration', 'rank', 'split', 'launch_FB_phase', 'launch_FB_window_id']
STATIC = ['operation_signature_sha256', 'root_operation', 'preceding_PP_API',
          'contained_CPU_operations', 'unique_CPU_operation_names', 'CPU_threads']
CONTROL = 'phase_role_median_control'
EXTENDED = 'phase_role_static_signature_median'


def microbatch_role(value, maximum=3):
    value = int(value)
    if value == 0:
        return 'first'
    if value == maximum:
        return 'last'
    return 'middle'


def signature_for_counts(counts):
    text = ''.join(f'{name}\t{int(count)}\n' for name, count in sorted(counts.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def fit_static_predictors(observations):
    """Use only fit-stable static labels; never validation-observed labels or timing."""
    fit = observations[observations.iteration.isin(FIT)].copy()
    assert len(fit) == 16 and set(fit.split) == {'source_fit'}
    expected = []
    for phase, group in fit.groupby('launch_FB_phase', sort=True):
        row = {'launch_FB_phase': phase, 'fit_iterations': '85,90', 'fit_windows': len(group)}
        for field in STATIC:
            assert group[field].nunique(dropna=False) == 1, f'fit-static field varies: {phase} {field}'
            row['expected_' + field] = group[field].iloc[0]
        expected.append(row)
    expected = pd.DataFrame(expected).sort_values('launch_FB_phase').reset_index(drop=True)

    parameters = []
    for method in [CONTROL, EXTENDED]:
        keys = ['launch_FB_phase', 'microbatch_role']
        if method == EXTENDED:
            local = fit.merge(expected, on='launch_FB_phase', validate='many_to_one')
            keys += ['expected_' + field for field in STATIC]
        else:
            local = fit
        for labels, group in local.groupby(keys, sort=True):
            if not isinstance(labels, tuple):
                labels = (labels,)
            row = dict(method=method, fit_iterations='85,90', samples=len(group),
                submission_offset_ns=int(np.rint(group.terminal_API_start_from_FB_start_ns.median())))
            row.update(dict(zip(keys, labels)))
            parameters.append(row)
    parameters = pd.DataFrame(parameters).sort_values(['method', 'launch_FB_phase', 'microbatch_role']).reset_index(drop=True)

    features = observations[GROUP + ['microbatch', 'microbatch_role', 'FB_start_ns']].merge(
        expected, on='launch_FB_phase', validate='many_to_one')
    predictions = []
    for feature in features.to_dict('records'):
        for method in [CONTROL, EXTENDED]:
            match = parameters[parameters.method.eq(method) &
                parameters.launch_FB_phase.eq(feature['launch_FB_phase']) &
                parameters.microbatch_role.eq(feature['microbatch_role'])]
            assert len(match) == 1
            offset = int(match.submission_offset_ns.iloc[0])
            row = {key: feature[key] for key in GROUP}
            row.update(method=method, microbatch=int(feature['microbatch']),
                microbatch_role=feature['microbatch_role'], predicted_submission_offset_ns=offset,
                predicted_submission_ns=int(feature['FB_start_ns'] + offset),
                prediction_scope='source_local_scheduled_FB_start_conditioned_NOT_global1F1B',
                prediction_features=('FB_start_ns|phase|microbatch_role' if method == CONTROL else
                    'FB_start_ns|phase|microbatch_role|source_fit_expected_static_signature'))
            for field in STATIC:
                row['expected_' + field] = feature['expected_' + field]
            predictions.append(row)
    predictions = pd.DataFrame(predictions).sort_values(['method'] + GROUP).reset_index(drop=True)
    assert len(predictions) == 64
    control = predictions[predictions.method.eq(CONTROL)].drop(columns=['method', 'prediction_features']).reset_index(drop=True)
    extended = predictions[predictions.method.eq(EXTENDED)].drop(columns=['method', 'prediction_features']).reset_index(drop=True)
    pd.testing.assert_frame_equal(control, extended, check_exact=True)
    return expected, parameters, predictions


def _load_observations(paths):
    windows = pd.read_csv(paths['runtime_windows.csv.gz'], low_memory=False)
    fb = windows[windows.window_type.eq('FB_annotation')][
        ['iteration', 'rank', 'window_id', 'phase', 'microbatch', 'start_ns', 'end_ns']].copy()
    fb = fb.rename(columns={'window_id': 'launch_FB_window_id', 'phase': 'launch_FB_phase',
        'start_ns': 'FB_start_ns', 'end_ns': 'FB_end_ns'})
    fb['split'] = np.where(fb.iteration.isin(FIT), 'source_fit', 'source_incremental_validation')
    fb['microbatch_role'] = fb.microbatch.map(microbatch_role)
    observations, count_rows = [], []
    for iteration in FIT + VALIDATION:
        local_fb = fb[fb.iteration.eq(iteration)]
        cpu = pd.read_csv(paths[f'source{iteration}_CPU.csv.gz'], low_memory=False)
        runtime = pd.read_csv(paths[f'source{iteration}_runtime.csv.gz'], low_memory=False)
        gpu = pd.read_csv(paths[f'source{iteration}_GPU.csv.gz'], low_memory=False)
        roots = cpu.set_index('event_id').name
        pp = windows[windows.iteration.eq(iteration) & windows.window_type.eq('PP_API')][
            ['window_id', 'start_ns', 'end_ns']].copy()
        pp['name'] = pp.window_id.map(roots)
        terminal_gpu = gpu[gpu.unique_CPU_owner_name.eq('aten::_copy_from') &
            gpu.stream.astype(str).eq('0') & gpu.family.eq('elementwise') & gpu.category.eq('kernel')][
            ['event_id', 'unique_runtime_event_id', 'end_ns']].rename(columns={
                'event_id': 'terminal_event_id', 'end_ns': 'terminal_GPU_end_ns'})
        terminal = terminal_gpu.merge(runtime[['event_id', 'start_ns', 'end_ns']],
            left_on='unique_runtime_event_id', right_on='event_id', validate='one_to_one').rename(columns={
                'start_ns': 'terminal_API_start_ns', 'end_ns': 'terminal_API_end_ns'})
        for window in local_fb.to_dict('records'):
            contained = cpu[cpu.start_ns.ge(window['FB_start_ns']) & cpu.end_ns.le(window['FB_end_ns'])]
            counts = contained.name.value_counts().to_dict()
            preceding = pp[pp.end_ns.le(window['FB_start_ns'])].sort_values('end_ns')
            assert len(preceding)
            candidates = terminal[terminal.terminal_API_start_ns.between(window['FB_start_ns'], window['FB_end_ns'])]
            candidates = candidates.sort_values(['terminal_API_start_ns', 'terminal_API_end_ns', 'terminal_event_id'])
            assert len(candidates)
            selected = candidates.iloc[-1]
            record = dict(**window,
                operation_signature_sha256=signature_for_counts(counts),
                root_operation=roots.loc[window['launch_FB_window_id']],
                preceding_PP_API=preceding.iloc[-1]['name'],
                contained_CPU_operations=len(contained), unique_CPU_operation_names=len(counts),
                CPU_threads=contained.tid.nunique(), terminal_event_id=selected.terminal_event_id,
                terminal_API_start_ns=int(selected.terminal_API_start_ns),
                terminal_API_end_ns=int(selected.terminal_API_end_ns),
                terminal_GPU_end_ns=int(selected.terminal_GPU_end_ns))
            record['terminal_API_start_from_FB_start_ns'] = record['terminal_API_start_ns'] - record['FB_start_ns']
            record['CPU_FB_wall_ns'] = record['FB_end_ns'] - record['FB_start_ns']
            observations.append(record)
            for name, count in sorted(counts.items()):
                count_rows.append({key: record[key] for key in GROUP} | dict(operation_name=name, count=int(count)))
    observations = pd.DataFrame(observations).sort_values(GROUP).reset_index(drop=True)
    counts = pd.DataFrame(count_rows).sort_values(GROUP + ['operation_name']).reset_index(drop=True)
    assert len(observations) == 32
    assert set(observations.root_operation) == {'forward_step', 'backward_step'}
    assert observations.groupby('launch_FB_phase').operation_signature_sha256.nunique().eq(1).all()
    return observations, counts


def _metrics(group):
    return pd.Series(dict(windows=len(group),
        submission_MAE_ms=group.submission_error_ns.abs().mean()/1e6,
        submission_bias_ms=group.submission_error_ns.mean()/1e6,
        maximum_absolute_submission_error_ms=group.submission_error_ns.abs().max()/1e6))


def diagnose(out, paths, plan):
    begin = perf_counter()
    assert plan['diagnostic'] == 'source_CPU_submission_identifiability'
    assert plan['diagnostic_access'] == 'source_only' and not plan['variants']
    review = json.loads((ROOT/plan['resource_review_file']).read_text())
    assert review['fit_iterations'] == FIT and review['incremental_validation'] == VALIDATION
    observations, counts = _load_observations(paths)
    expected, parameters, predictions = fit_static_predictors(observations)
    mutated = observations.copy()
    validation = mutated.iteration.isin(VALIDATION)
    mutated.loc[validation, ['FB_end_ns', 'terminal_API_start_ns', 'terminal_API_end_ns',
        'terminal_GPU_end_ns', 'terminal_API_start_from_FB_start_ns', 'CPU_FB_wall_ns']] = -999999
    mutated.loc[validation, STATIC] = ['mutated', 'mutated', 'mutated', -1, -1, -1]
    other_expected, other_parameters, other_predictions = fit_static_predictors(mutated)
    pd.testing.assert_frame_equal(expected, other_expected, check_exact=True)
    pd.testing.assert_frame_equal(parameters, other_parameters, check_exact=True)
    pd.testing.assert_frame_equal(predictions, other_predictions, check_exact=True)

    csv(out, 'source_CPU_expected_static_signatures.csv', expected)
    csv(out, 'source_CPU_submission_parameters.csv', parameters)
    csv(out, 'source_CPU_submission_predictions_sealed.csv.gz', predictions)
    sealed = ['source_CPU_expected_static_signatures.csv', 'source_CPU_submission_parameters.csv',
              'source_CPU_submission_predictions_sealed.csv.gz']
    dump(out/'source_CPU_submission_prediction_seal.json', dict(
        status='SEALED_SOURCE85_90_STATIC_CPU_SUBMISSION_CANDIDATES',
        fit_iterations=FIT, incremental_validation=VALIDATION, target_parameter_updates=0,
        files=[dict(path=name, sha256=sha(out/name)) for name in sealed],
        validation_feature_policy='Expected signatures come from source85/90. Source95/100 observed signatures are attached only after seal.',
        mutation_gate='Changing source95/100 CPU/API/GPU truth and observed static labels leaves parameters and predictions exact.'))

    scored = predictions.merge(observations[GROUP + ['terminal_API_start_ns'] + STATIC],
        on=GROUP, validate='many_to_one', suffixes=('', '_observed'))
    scored['submission_error_ns'] = scored.predicted_submission_ns - scored.terminal_API_start_ns
    for field in STATIC:
        scored[field + '_match'] = scored['expected_' + field].astype(str).eq(scored[field].astype(str))
    csv(out, 'source_CPU_submission_results.csv.gz', scored)
    metrics = scored.groupby(['method', 'split', 'launch_FB_phase']).apply(_metrics, include_groups=False).reset_index()
    per_iteration = scored.groupby(['method', 'iteration', 'split', 'launch_FB_phase']).apply(
        _metrics, include_groups=False).reset_index()
    csv(out, 'source_CPU_submission_metrics.csv', metrics)
    csv(out, 'source_CPU_submission_per_iteration.csv', per_iteration)
    csv(out, 'source_CPU_operation_signature_counts.csv.gz', counts)
    signature_validation = observations.merge(expected, on='launch_FB_phase', validate='many_to_one')
    for field in STATIC:
        signature_validation[field + '_match'] = signature_validation[field].astype(str).eq(
            signature_validation['expected_' + field].astype(str))
    csv(out, 'source_CPU_static_signature_validation.csv', signature_validation[
        GROUP + ['microbatch', 'microbatch_role'] + STATIC + [field + '_match' for field in STATIC]])
    for file in json.loads((out/'source_CPU_submission_prediction_seal.json').read_text())['files']:
        assert sha(out/file['path']) == file['sha256']

    base_classes = predictions[['launch_FB_phase', 'microbatch_role']].drop_duplicates()
    extended_classes = predictions[['launch_FB_phase', 'microbatch_role'] +
        ['expected_' + field for field in STATIC]].drop_duplicates()
    control = scored[scored.method.eq(CONTROL)].sort_values(GROUP).reset_index(drop=True)
    extended = scored[scored.method.eq(EXTENDED)].sort_values(GROUP).reset_index(drop=True)
    pd.testing.assert_series_equal(control.predicted_submission_ns, extended.predicted_submission_ns,
                                   check_names=False, check_exact=True)
    validation_rows = signature_validation[signature_validation.iteration.isin(VALIDATION)]
    all_static_match = bool(validation_rows[[field + '_match' for field in STATIC]].all().all())
    dump(out/'source_CPU_submission_identifiability.json', dict(
        status='NO_ADDITIONAL_STATIC_INFORMATION',
        phase_role_equivalence_classes=len(base_classes),
        extended_static_equivalence_classes=len(extended_classes),
        additional_equivalence_classes=0,
        control_extended_predictions_exact=True,
        source_validation_static_signature_matches=int(validation_rows.operation_signature_sha256_match.sum()),
        source_validation_windows=len(validation_rows),
        all_source_validation_static_fields_match=all_static_match,
        inference='Static graph identity fixes what work is submitted but does not identify iteration-specific CPU execution speed or runtime state.'))
    dump(out/'field_contract.json', dict(
        operation_signature='SHA256 of sorted CPU operation-name counts wholly inside an F/B annotation; no duration or timestamp is hashed.',
        static_use='Only source85/90 expected labels enter the sealed candidate. Source95/100 observed labels score stability after seal.',
        timing='The only numeric free-running inputs are scheduled F/B start plus a source-fit phase/role offset.',
        accounting='A future submission model would replace existing F/B local work up to terminal launch; it cannot be added to F/B or PP.',
        boundary='Source rank16 only; 85/90 fit, exposed 95/100 validation. No target timing, global1F1B, Step/MFU or topology change.'))
    draw(out, metrics, observations)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    elapsed = perf_counter()-begin
    assert peak <= review['resource']['maximum_peak_RSS_bytes'] and elapsed <= review['resource']['target_analysis_seconds']
    incremental = metrics[metrics.split.eq('source_incremental_validation') & metrics.method.eq(CONTROL)]
    dump(out/'diagnostic.json', dict(status='SOURCE_CPU_SUBMISSION_IDENTIFIABILITY_PASS',
        new_prediction=True, new_global_prediction=False, used_to_fit_model=True,
        new_target_timing_read=False, raw_trace_scanned=False, source_iterations=FIT+VALIDATION,
        source_fit_iterations=FIT, source_incremental_validation=VALIDATION, source_ranks=[16],
        FB_windows=len(observations), forward_contained_CPU_operations=int(expected[
            expected.launch_FB_phase.eq('forward')].expected_contained_CPU_operations.iloc[0]),
        backward_contained_CPU_operations=int(expected[
            expected.launch_FB_phase.eq('backward')].expected_contained_CPU_operations.iloc[0]),
        forward_unique_CPU_operation_names=int(expected[
            expected.launch_FB_phase.eq('forward')].expected_unique_CPU_operation_names.iloc[0]),
        backward_unique_CPU_operation_names=int(expected[
            expected.launch_FB_phase.eq('backward')].expected_unique_CPU_operation_names.iloc[0]),
        all_validation_static_fields_match=all_static_match,
        phase_role_equivalence_classes=len(base_classes), extended_static_equivalence_classes=len(extended_classes),
        static_feature_information_gain_classes=len(extended_classes)-len(base_classes),
        control_extended_predictions_exact=True,
        validation_truth_and_observed_signature_mutation_parameters_predictions_exact=True,
        source_incremental_control_metrics=incremental.to_dict('records'),
        prediction_seal_sha256=sha(out/'source_CPU_submission_prediction_seal.json'),
        promotion='REJECT_STATIC_SIGNATURE_EXTENSION_NO_INFORMATION_GAIN',
        formal_topology_changed=False, target_parameter_updates=0,
        peak_RSS_bytes=peak, analysis_seconds=elapsed, resource_gate_pass=True,
        next='Do not transfer static CPU signatures to target. Move to a different source-supported cost component or obtain an independently observable runtime-state feature.'))


def draw(out, metrics, observations):
    import os
    os.environ['MPLCONFIGDIR'] = str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'w37-t30a-cpu-submission-identifiability'
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    validation = metrics[metrics.split.eq('source_incremental_validation')]
    width = .36
    for index, method in enumerate([CONTROL, EXTENDED]):
        subset = validation[validation.method.eq(method)].set_index('launch_FB_phase').reindex(['forward', 'backward'])
        axes[0].bar(np.arange(2) + (index-.5)*width, subset.submission_MAE_ms, width,
                    label='phase/role' if method == CONTROL else '+ static signature')
    axes[0].set_xticks(range(2), ['forward', 'backward'])
    axes[0].set_ylabel('source95/100 submission MAE (ms)')
    axes[0].set_title('Static extension is prediction-identical')
    axes[0].legend()
    for phase, marker in [('forward', 'o'), ('backward', 's')]:
        local = observations[observations.launch_FB_phase.eq(phase)]
        axes[1].scatter(local.iteration, local.terminal_API_start_from_FB_start_ns/1e6,
                        label=phase, marker=marker)
    axes[1].set_xlabel('source iteration')
    axes[1].set_ylabel('terminal API start from F/B start (ms)')
    axes[1].set_title('Same operation signature, different runtime')
    axes[1].legend()
    fig.suptitle('Offline CPU graph-signature identifiability audit')
    fig.text(.5, .02, 'Operation names/counts are invariant by phase; observed durations are excluded from prediction.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out/'source_CPU_submission_identifiability.svg', metadata={'Date': None})
    fig.savefig(out/'source_CPU_submission_identifiability.png', dpi=150)
    plt.close(fig)

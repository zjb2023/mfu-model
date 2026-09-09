"""Source-only cross-checks; no target observations or candidate selection."""
import json
import pandas as pd
from common import RUN, dump, write_csv

def review(spec, paths, out):
    from worker import PHYSICAL
    observations = pd.read_csv(paths['observations'])
    physical = observations[
        observations.parameter_view.eq('window_split')
        & observations.cost_group.eq('__physical_slot_total__')
        & observations.autograd_phase.eq('physical_mixed')].copy()
    physical['aggregate_key'] = physical[PHYSICAL].astype(str).agg('|'.join, axis=1)
    aggregate = physical[physical.iteration.isin(spec['source_calibration'])].groupby('aggregate_key').active_union_ns.median().round()
    held = pd.read_csv(RUN/'audit/source_incremental_rows.csv.gz')
    held['aggregate_prediction_ns'] = held.aggregate_key.map(aggregate)
    matched = held[held.eligible & held.aggregate_prediction_ns.notna()].copy()
    comparisons = []
    for (iteration, scope), group in matched.groupby(['iteration', 'execution_scope']):
        for method, column in [('fine_global_layer_mb', 'predicted_ns'), ('aggregate_context', 'aggregate_prediction_ns')]:
            error = group[column]-group.active_union_ns
            comparisons.append(dict(iteration=int(iteration), scope=scope, method=method,
                rows=len(group), MAE_ms=float(error.abs().mean()/1e6), bias_ms=float(error.mean()/1e6),
                WMAPE_pct=float(error.abs().sum()/group.active_union_ns.sum()*100),
                observed_sum_ms=float(group.active_union_ns.sum()/1e6)))
    write_csv(out/'source_method_comparison.csv', comparisons)
    matched.to_csv(out/'source_common_rows.csv.gz', index=False, compression='gzip')

    meta = pd.read_csv(paths['v685_nodes'], usecols=['node_id','kind','phase','op_name','rank','pp_stage','microbatch']).set_index('node_id')
    baseline = pd.read_csv(RUN/'model/frozen_v685_timings.csv.gz').set_index('node_id').join(meta)
    predictions = pd.read_csv(RUN/'model/predictions.csv').set_index('variant')
    changes = pd.read_csv(RUN/'model/node_updates.csv.gz').join(meta[['kind','phase']], on='node_id')
    assert changes.phase.isin(['FWD','BWD']).all()
    assert ~changes.kind.isin(['pp_p2p','optimizer','collective_service']).any()
    tail_records, effect_records = [], []
    baseline_last_b = baseline[baseline.kind.eq('phase_boundary') & baseline.op_name.eq('bwd_end')].predicted_end_ns.max()
    for variant in spec['variants']:
        timing = pd.read_csv(RUN/'model'/f'{variant}_timings.csv.gz').set_index('node_id').join(meta)
        assert timing.index.equals(baseline.index)
        duration_delta = timing.duration_ns-baseline.duration_ns
        last_b = timing[timing.kind.eq('phase_boundary') & timing.op_name.eq('bwd_end')]
        last_id = last_b.predicted_end_ns.idxmax()
        last_ns = int(last_b.predicted_end_ns.max())
        completion = timing.loc['iteration:completion_join']
        changed_ids = set(duration_delta[duration_delta.ne(0)].index)
        declared = set(changes.loc[changes.variant.eq(variant),'node_id'])
        assert changed_ids <= declared
        opt = timing[timing.kind.eq('optimizer')]
        assert duration_delta.loc[opt.index].eq(0).all()
        # After-last-B time is a wall-clock difference, not a new tail cost.
        tail_raw_ms = (int(completion.predicted_end_ns)-last_ns)/1e6
        base_tail_raw_ms = (int(baseline.loc['iteration:completion_join','predicted_end_ns'])-baseline_last_b)/1e6
        tail_records.append(dict(variant=variant, last_b_node=last_id, last_b_ms=last_ns/1e6,
            last_b_delta_ms=(last_ns-baseline_last_b)/1e6,
            completion_ms=int(completion.predicted_end_ns)/1e6,
            completion_delta_ms=(int(completion.predicted_end_ns)-int(baseline.loc['iteration:completion_join','predicted_end_ns']))/1e6,
            raw_after_last_b_ms=tail_raw_ms, raw_after_last_b_delta_ms=tail_raw_ms-base_tail_raw_ms,
            optimizer_nodes=len(opt), optimizer_changed_duration_nodes=int(duration_delta.loc[opt.index].ne(0).sum()),
            optimizer_start_shift_min_ms=float((opt.predicted_start_ns-baseline.loc[opt.index,'predicted_start_ns']).min()/1e6),
            optimizer_start_shift_max_ms=float((opt.predicted_start_ns-baseline.loc[opt.index,'predicted_start_ns']).max()/1e6)))
        assert abs(tail_records[-1]['raw_after_last_b_delta_ms']-(predictions.loc[variant,'tail_ms']-predictions.loc['frozen_v685','tail_ms'])) < 1e-6
        for scope, group in changes[changes.variant.eq(variant)].groupby(changes.source_key.str.split('|').str[3]):
            d = group.new_duration_ns-group.old_duration_ns
            c = group.new_compute_ns-group.old_compute_ns
            effect_records.append(dict(variant=variant, scope=scope, updated_nodes=len(group),
                compute_increase_nodes=int(c.gt(0).sum()), compute_decrease_nodes=int(c.lt(0).sum()),
                duration_increase_nodes=int(d.gt(0).sum()), duration_decrease_nodes=int(d.lt(0).sum()),
                duration_unchanged_nodes=int(d.eq(0).sum()), summed_compute_delta_ms=float(c.sum()/1e6),
                summed_duration_delta_ms=float(d.sum()/1e6),
                note='Cross-node sums are not iteration gains; old wall floor prevents exposed duration decreases.'))
    write_csv(out/'tail_propagation.csv', tail_records)
    write_csv(out/'update_effect_summary.csv', effect_records)
    # Exercise the actual data-access guard, not a duplicate implementation.
    try:
        paths['version_iterations'].open().read(1)
    except PermissionError:
        evaluator_guard = 'PASS'
    else:
        raise AssertionError('Source review could read target observations')
    summary = dict(status='PASS', source_common_rows=len(matched), source_rows_without_fine_fit=int((~held.eligible).sum()),
        aggregate_fit_keys=len(aggregate), source_methods_same_fit=[85,90], source_methods_same_evaluation=[95,100],
        target_observation_guard=evaluator_guard, target_observation_read=False,
        cost_updates_only_in_F_B=True, optimizer_costs_unchanged=True,
        interpretation='Tail wall can change because last B and global completion shift differently, even when optimizer costs do not change.',
        source_full_iteration_validation=False, inherited_wall_floor_independently_validated=False)
    dump(out/'review.json', summary)
    print(json.dumps(summary))

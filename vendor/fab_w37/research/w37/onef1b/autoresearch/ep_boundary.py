"""Read-only EP evidence admission. Does not execute a legacy extractor or fit costs."""
import json
import pandas as pd
from smoke_worker import dump
from worker import csv


def diagnose(out,paths,plan):
    if plan['diagnostic']=='ep_source_graph_oracle':
        from ep_graph import source_oracle
        source_oracle(out,paths,plan);return
    d=pd.read_csv(paths['ep_source_ep_group_call_observations.csv'])
    params=pd.read_csv(paths['ep_source_ep_group_parameters.csv'])
    access=json.loads(paths['ep_input_access_audit.json'].read_text())
    assert access['target_profiler_or_training_timing_read'] is False
    dump(out/'schema.json',dict(rows=len(d),columns={k:str(v) for k,v in d.dtypes.items()},missing={k:int(v) for k,v in d.isna().sum().items()},historical_parameter_columns=list(params.columns)))
    csv(out,'sample_group_calls.csv',d.head(20));csv(out,'historical_parameter_sample_DIAGNOSTIC.csv',params.head(12))
    csv(out,'numeric_summary.csv',d.select_dtypes(include='number').describe(percentiles=[.05,.5,.95]).T.reset_index().rename(columns={'index':'column'}))
    keys=[c for c in ['iteration','phase','op_kind','operation','group_size'] if c in d]
    if keys:csv(out,'group_counts.csv',d.groupby(keys).size().rename('rows').reset_index())
    for key in ['ep_extract_dag_v53_ep_group_calibration.py','ep_dag_v53_ep_group_2026w36.toml']:
        p=paths[key];(out/p.name).write_text(p.read_text())
    if plan['diagnostic']=='ep_rank_publication':
        audit_rank_publication(out,paths,plan,d);return
    dump(out/'diagnostic.json',dict(status='EP_DERIVED_SCHEMA_INTAKE_PASS_FIT_ADMISSION_PENDING',source_group_calls=len(d),historical_parameter_rows=len(params),
        recorded_source_iterations=access['source_calibration_iterations'],upstream_extraction_target_timing_read=False,new_target_timing_read=False,
        raw_trace_scanned=False,rank_anchor_table_read=False,new_prediction=False,used_to_fit_model=False,
        diagnostic=plan['diagnostic'],sealed_reference=plan['sealed_reference'],
        next='Verify anchor definitions, group membership and arrival/completion algebra in frozen extractor text; split raw group observations85/90 vs95/100 before any new fit. Do not inherit historical all-nine-iteration parameters.'))


def audit_rank_publication(out,paths,plan,calls):
    from time import perf_counter
    import numpy as np
    begin=perf_counter()
    cols=['iteration','rank','pp_stage','pp_lane','ep_group','phase','microbatch','execution_layer','layer_id','semantic_region','behavior','arrival_ns','completion_ns','captured_anchor_wall_ns']
    rank=pd.read_csv(paths['ep_source_ep_rank_anchor_events.csv'],usecols=cols)
    assert len(rank)==203904 and (rank.completion_ns-rank.arrival_ns).eq(rank.captured_anchor_wall_ns).all()
    keys=['iteration','ep_group','pp_stage','phase','microbatch','layer_id','semantic_region','behavior']
    assert not rank.duplicated(keys+['rank']).any()
    grouped=rank.groupby(keys).agg(group_size=('rank','nunique'),first_arrival_ns=('arrival_ns','min'),last_arrival_ns=('arrival_ns','max'),
        first_completion_ns=('completion_ns','min'),group_end_ns=('completion_ns','max')).reset_index()
    assert grouped.group_size.eq(8).all()
    check=grouped.merge(calls,on=keys,suffixes=('','_saved'),validate='one_to_one');assert len(check)==len(calls)
    for c in ['group_size','first_arrival_ns','last_arrival_ns','group_end_ns']:assert check[c].eq(check[c+'_saved']).all(),c
    grouped['min_return_minus_last_entry_ns']=grouped.first_completion_ns-grouped.last_arrival_ns
    grouped['latest_return_minus_first_return_ns']=grouped.group_end_ns-grouped.first_completion_ns
    grouped['all_rank_entry_gate_feasible']=grouped.min_return_minus_last_entry_ns.ge(0)
    grouped['split']=np.where(grouped.iteration.isin([85,90]),'source_fit_candidate',np.where(grouped.iteration.isin([95,100]),'source_incremental_validation','historical_diagnostic'))
    csv(out,'group_publication_feasibility.csv.gz',grouped)
    csv(out,'group_publication_summary.csv',grouped.groupby(['split','phase','semantic_region']).agg(calls=('group_size','size'),feasible_calls=('all_rank_entry_gate_feasible','sum'),
        earliest_after_last_median_ns=('min_return_minus_last_entry_ns','median'),earliest_after_last_min_ns=('min_return_minus_last_entry_ns','min'),
        latest_return_after_first_median_ns=('latest_return_minus_first_return_ns','median')).reset_index())
    phases=pd.read_csv(paths['source_pp_trace_events_60_100.csv']);pk=['iteration','rank','pp_stage','pp_lane','phase','microbatch']
    z=rank.merge(phases[pk+['observed_start_ns','observed_end_ns']],on=pk,validate='many_to_one')
    assert z.arrival_ns.ge(z.observed_start_ns).all() and z.completion_ns.le(z.observed_end_ns+2).all()
    from intervals import preceding_end
    z=z.sort_values(pk+['arrival_ns']);previous=preceding_end(z,pk);z['local_gap_ns']=z.arrival_ns-previous
    overlap=z.local_gap_ns.lt(0).sum();assert overlap==0
    coverage=z.groupby(pk).size();assert len(coverage)==len(phases)
    member=rank[['ep_group','pp_stage','pp_lane','rank']].drop_duplicates().sort_values(['ep_group','rank'])
    assert len(member)==256 and member['rank'].nunique()==256
    assert member.groupby('ep_group').size().eq(8).all() and member.groupby('ep_group').pp_stage.nunique().eq(1).all()
    csv(out,'ep_group_members.csv',member)
    # Keep reduced machine-readable evidence in the local output, with no raw-trace paths.
    csv(out,'source_rank_anchors.csv.gz',rank)
    report=dict(status='SOURCE_RANK_ENTRY_GATE_FEASIBLE' if grouped.all_rank_entry_gate_feasible.all() else 'SOURCE_RANK_ENTRY_GATE_REJECTED',
        source_rank_anchors=len(rank),source_group_calls=len(grouped),group_calls_with_early_rank_return=int((~grouped.all_rank_entry_gate_feasible).sum()),
        min_first_return_after_last_entry_ns=int(grouped.min_return_minus_last_entry_ns.min()),source_phase_windows=len(phases),overlapping_local_wrapper_intervals=int(overlap),
        group_summary_exact_match=True,rank_group_mapping_stable=True,local_gap_timestamp_arithmetic='nullable int64; never epoch float',minimum_local_gap_ns=int(z.local_gap_ns.min()),source_dataframe_bytes=int(rank.memory_usage(deep=True).sum()),analysis_seconds=perf_counter()-begin,
        raw_trace_scanned=False,new_prediction=False,used_to_fit_model=False,diagnostic=plan['diagnostic'],sealed_reference=plan['sealed_reference'],
        boundary='CPU wrapper entry/return only, not GPU-ready or network-service timestamps. Source feasibility does not prove target deployed flags or validate a shared GPU barrier.',
        next='If feasible, independently model earliest group completion plus per-rank return tails and local between-wrapper work; first prove source observed-cost reassembly, then split fit85/90 and validate95/100. Keep service_fct and historical fitted parameters out of the new raw-cost fit.')
    dump(out/'diagnostic.json',report)

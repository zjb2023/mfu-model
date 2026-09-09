"""Admit source GPU CP observations without adding them to CPU graph costs."""
import json
import tomllib
from time import perf_counter
import numpy as np
import pandas as pd
from smoke_worker import dump,sha
from worker import csv
from ep_completion_audit import split

STEP=['iteration','rank','phase','microbatch']
PAIR=['iteration','cp_group','phase','microbatch']
CALL=['iteration','cp_group','pp_stage','phase','microbatch','layer_id','cp_slot_in_layer']
ITERS=[60,65,70,75,80,85,90,95,100]


def assert_same(left,right,keys,columns):
    a=left.set_index(keys).sort_index();b=right.set_index(keys).sort_index()
    assert a.index.is_unique and b.index.is_unique and a.index.equals(b.index), keys
    for col in columns:
        assert a[col].eq(b[col]).all(), ('mismatch',col)


def interval_union(intervals):
    """Integer endpoints; adjacent/overlapping GPU kernels are counted once."""
    merged=[]
    for start,end in sorted(intervals):
        start=int(start);end=int(end);assert end>=start
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return merged


def diagnose(out,paths,plan):
    assert plan['diagnostic_access']=='source_only';begin=perf_counter()
    access=json.loads(paths['cp_input_access_audit.json'].read_text())
    assert access['target_profiler_or_training_timing_read'] is False
    current={str(p):p for p in paths.values()}
    for item in access['structured_inputs']:
        assert item['path'] in current and sha(current[item['path']])==item['sha256']
    columns=['event_id','behavior','iteration','rank','group_size','start_ns','end_ns','duration_ns',
        'logical_input_bytes','expected_tx_bytes','expected_rx_bytes','pp_stage','pp_lane','cp_group','cp_rank',
        'phase','microbatch','pp_annotation_end_ns','outside_pp_annotation_end','rank_step_complete',
        'execution_index','layer_id','cp_slot_in_layer','pair_step_complete']
    a=pd.read_csv(paths['cp_source_cp_rank_events.csv'],usecols=columns)
    assert a.event_id.is_unique and a.group_size.eq(2).all() and a.behavior.eq('cp_all_to_all').all()
    assert a.end_ns.sub(a.start_ns).eq(a.duration_ns).all() and a.duration_ns.gt(0).all()
    base=['event_id','behavior','iteration','rank','group_size','start_ns','end_ns','duration_ns',
        'logical_input_bytes','expected_tx_bytes','expected_rx_bytes']
    upstream=pd.read_parquet(paths['cp_upstream_collective_events.parquet'],
        columns=base+['source','pg_description','collective'],
        filters=[('iteration','in',ITERS),('behavior','==','cp_all_to_all')])
    assert upstream.source.eq('profiler').all() and upstream.pg_description.eq('CONTEXT_PARALLEL_GROUP').all()
    assert upstream.collective.eq('all_to_all').all()
    assert_same(a,upstream,['event_id'],base[1:])
    dataframe_bytes=int(a.memory_usage(deep=True).sum()+upstream.memory_usage(deep=True).sum())
    del upstream
    static=[];layouts={}
    for case,stages,world,layers in [('source256',16,256,60),('target224',14,224,52)]:
        ranks=pd.read_csv(paths[case+'_rank_topology.csv']);layout=pd.read_csv(paths[case+'_layer_stage_map.csv'])
        assert len(ranks)==world and ranks.pp_stage.nunique()==stages and len(layout)==layers
        assert ranks['rank'].floordiv(2).eq(ranks.cp_group).all()
        groups=ranks.groupby('cp_group').agg(ranks=('rank','nunique'),hosts=('host','nunique'),ep_groups=('ep_group','nunique'),
            cp_ranks=('cp_rank',lambda v:','.join(map(str,sorted(v)))),host=('host','first'),pp_stage=('pp_stage','first'))
        assert groups.ranks.eq(2).all() and groups.hosts.eq(1).all() and groups.ep_groups.eq(1).all() and groups.cp_ranks.eq('0,1').all()
        groups['case']=case;static.append(groups.reset_index())
        layouts[case]={int(s):sorted(map(int,g.layer_id)) for s,g in layout.groupby('pp_stage')}
        assert [len(v) for v in layouts[case].values()]==[2]+[4]*(stages-2)+[2]
        if case=='source256':
            assert_same(a[['rank','pp_stage','pp_lane','cp_group','cp_rank']].drop_duplicates(),ranks,['rank'],['pp_stage','pp_lane','cp_group','cp_rank'])
    csv(out,'static_cp_pair_structure.csv',pd.concat(static,ignore_index=True))
    contract=tomllib.loads(paths['target224_run_contract.toml'].read_text())
    assert contract['topology']['num_micro_batches']==3 and contract['topology']['cp']==2
    assert contract['topology']['num_layers']==52
    phase=pd.read_csv(paths['source_pp_trace_events_60_100.csv'],usecols=STEP+['pp_stage','observed_start_ns','observed_end_ns'])
    assert sorted(phase.iteration.unique())==ITERS and sorted(phase.microbatch.unique())==[0,1,2,3]
    phase=phase.sort_values(['iteration','rank','observed_start_ns'])
    # Shift nullable Int64, never float: these absolute epochs are > 2**53.
    phase['next_phase_start_ns']=phase.observed_start_ns.astype('Int64').groupby([phase.iteration,phase['rank']]).shift(-1)
    a=a.merge(phase.rename(columns={'pp_stage':'phase_pp_stage'}),on=STEP,validate='many_to_one')
    assert a.pp_annotation_end_ns.eq(a.observed_end_ns).all() and a.pp_stage.eq(a.phase_pp_stage).all()
    assert a.start_ns.ge(a.observed_start_ns).all()
    assert (a.next_phase_start_ns.isna()|a.start_ns.lt(a.next_phase_start_ns)).all()
    assert a.outside_pp_annotation_end.eq(a.end_ns.gt(a.observed_end_ns+2)).all()
    a['crosses_next_FB_start']=a.end_ns.gt(a.next_phase_start_ns).fillna(False)
    rank_steps=a.groupby(STEP+['pp_stage','cp_group']).agg(observed_calls=('event_id','size'),
        calls_outside_pp_annotation_end=('outside_pp_annotation_end','sum')).reset_index()
    rank_steps['expected_calls']=[len(layouts['source256'][int(r.pp_stage)])*(4 if r.phase=='forward' else 9) for r in rank_steps.itertuples()]
    rank_steps['rank_step_complete']=rank_steps.observed_calls.eq(rank_steps.expected_calls)
    old_steps=pd.read_csv(paths['cp_source_cp_rank_step_audit.csv'])
    assert_same(rank_steps,old_steps,STEP,['pp_stage','cp_group','observed_calls','expected_calls','calls_outside_pp_annotation_end','rank_step_complete'])
    assert len(rank_steps)==len(phase)==9*256*8
    pair=rank_steps.groupby(PAIR).agg(participant_count=('rank','nunique'),all_rank_steps_complete=('rank_step_complete','all'),
        observed_rank_calls=('observed_calls','sum'),expected_rank_calls=('expected_calls','sum'),
        calls_outside_pp_annotation_end=('calls_outside_pp_annotation_end','sum')).reset_index()
    pair['pair_step_complete']=pair.participant_count.eq(2)&pair.all_rank_steps_complete&pair.observed_rank_calls.eq(pair.expected_rank_calls)
    invalid=pair[~pair.pair_step_complete].copy()
    old_invalid=pd.read_csv(paths['cp_source_cp_invalid_pair_steps.csv'])
    assert_same(invalid,old_invalid,PAIR,[c for c in old_invalid if c not in PAIR])
    assert_same(a[STEP+['rank_step_complete']].drop_duplicates(),rank_steps,STEP,['rank_step_complete'])
    assert_same(a[PAIR+['pair_step_complete']].drop_duplicates(),pair,PAIR,['pair_step_complete'])
    complete=a[a.rank_step_complete].sort_values(STEP+['start_ns','end_ns']).copy()
    complete['recomputed_execution_index']=complete.groupby(STEP).cumcount()
    assert complete.execution_index.eq(complete.recomputed_execution_index).all()
    expected_layer=[];expected_slot=[]
    for r in complete.itertuples():
        count=4 if r.phase=='forward' else 9
        order=layouts['source256'][r.pp_stage]
        if r.phase=='backward':order=list(reversed(order))
        expected_layer.append(order[r.execution_index//count]);expected_slot.append(r.execution_index%count)
    assert complete.layer_id.eq(expected_layer).all() and complete.cp_slot_in_layer.eq(expected_slot).all()
    assert a.loc[~a.rank_step_complete,['execution_index','layer_id','cp_slot_in_layer']].eq(-1).all().all()
    del complete
    valid=a[a.pair_step_complete].copy()
    calls=valid.groupby(CALL).agg(group_size=('rank','nunique'),first_arrival_ns=('start_ns','min'),last_arrival_ns=('start_ns','max'),
        group_end_ns=('end_ns','max'),first_end_ns=('end_ns','min'),duration_ns_rank_median=('duration_ns','median'),duration_ns_rank_max=('duration_ns','max'),
        logical_input_bytes=('logical_input_bytes','median'),expected_tx_bytes=('expected_tx_bytes','median')).reset_index()
    assert calls.group_size.eq(2).all()
    calls['arrival_span_ns']=calls.last_arrival_ns-calls.first_arrival_ns
    calls['trace_service_ns']=calls.group_end_ns-calls.last_arrival_ns
    calls['group_elapsed_ns']=calls.group_end_ns-calls.first_arrival_ns
    calls['service_fct_ns']=calls.trace_service_ns
    old_calls=pd.read_csv(paths['cp_source_cp_group_calls.csv'])
    assert_same(calls,old_calls,CALL,[c for c in old_calls if c not in CALL+['behavior','service_source']])
    assert old_calls.service_source.eq('source_trace_cp2_group_tail').all()
    calls['first_end_minus_last_start_ns']=calls.first_end_ns-calls.last_arrival_ns
    calls['split']=split(calls.iteration)
    csv(out,'source_cp_group_verified.csv.gz',calls)
    del old_calls
    invalid['split']=split(invalid.iteration);csv(out,'source_cp_invalid_pair_steps.csv',invalid)
    coverage=rank_steps.groupby('iteration').agg(rank_steps=('rank','size'),expected_rank_events=('expected_calls','sum'),observed_rank_events=('observed_calls','sum'),
        incomplete_rank_steps=('rank_step_complete',lambda v:int((~v).sum())),outside_annotation_events=('calls_outside_pp_annotation_end','sum')).reset_index()
    by_pair=pair.groupby('iteration').agg(pair_steps=('cp_group','size'),valid_pair_steps=('pair_step_complete','sum'))
    coverage=coverage.merge(by_pair,on='iteration',validate='one_to_one')
    coverage=coverage.merge(valid.groupby('iteration').size().rename('admitted_rank_events'),on='iteration',validate='one_to_one')
    coverage['missing_rank_events']=coverage.expected_rank_events-coverage.observed_rank_events
    coverage['excluded_observed_rank_events']=coverage.observed_rank_events-coverage.admitted_rank_events
    coverage['admitted_group_calls']=coverage.admitted_rank_events//2
    coverage['missing_or_excluded_group_calls']=coverage.expected_rank_events//2-coverage.admitted_group_calls
    coverage['split']=split(coverage.iteration);csv(out,'source_cp_coverage.csv',coverage)
    # A complete CPU annotation view and its GPU tail; each is a view of the
    # same timeline, never an additional delay on top of its CPU/PP duration.
    union_rows=[]
    for key,g in a.groupby(STEP,sort=False):
        start=int(g.observed_start_ns.iloc[0]);end=int(g.observed_end_ns.iloc[0])
        merged=interval_union(zip(g.start_ns,g.end_ns))
        total=sum(e-s for s,e in merged)
        inside=sum(max(0,min(e,end)-max(s,start)) for s,e in merged)
        after=sum(max(0,e-max(s,end)) for s,e in merged)
        assert total==inside+after
        union_rows.append(dict(zip(STEP,key))|dict(pp_stage=int(g.pp_stage.iloc[0]),cp_group=int(g.cp_group.iloc[0]),
            rank_step_complete=bool(g.rank_step_complete.iloc[0]),pair_step_complete=bool(g.pair_step_complete.iloc[0]),
            annotation_duration_ns=end-start,cp_kernel_duration_sum_ns=int(g.duration_ns.sum()),cp_kernel_union_ns=total,
            overlapping_duration_ns=int(g.duration_ns.sum())-total,cp_union_inside_annotation_ns=inside,cp_union_after_annotation_ns=after,
            cp_tail_envelope_ns=max(0,int(g.end_ns.max())-end),kernels_ending_after_annotation=int(g.outside_pp_annotation_end.sum()),
            kernels_crossing_next_FB_start=int(g.crosses_next_FB_start.sum())))
    union=pd.DataFrame(union_rows);union['split']=split(union.iteration)
    csv(out,'source_cp_annotation_union.csv.gz',union)
    ns=['cp_union_inside_annotation_ns','cp_union_after_annotation_ns','cp_tail_envelope_ns','overlapping_duration_ns']
    profile=union[union.pair_step_complete].groupby(['split','pp_stage','phase'])[ns+['kernels_ending_after_annotation']].agg(['mean','median','size'])
    profile.columns=['_'.join(c) for c in profile.columns];csv(out,'source_cp_annotation_profile.csv',profile.reset_index())
    csv(out,'source_cp_rank_verified.csv.gz',valid[columns+['observed_start_ns','observed_end_ns','next_phase_start_ns','crosses_next_FB_start']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label,g in union[union.phase.eq('backward')&union.pair_step_complete&union.iteration.isin([85,90,95,100])].groupby('split'):
        means=g.groupby('pp_stage')[ns].mean()/1e6
        axes[0].plot(means.index,means.cp_union_after_annotation_ns,'o-',label=label)
        axes[1].plot(means.index,means.cp_tail_envelope_ns,'o-',label=label)
    for ax in axes:ax.set_xlabel('PP stage');ax.legend(fontsize=7)
    axes[0].set_ylabel('GPU CP kernel union after CPU B annotation (ms)')
    axes[1].set_ylabel('CPU B annotation end to last CP kernel end (ms)')
    fig.suptitle('Source observations; CPU and GPU views must not be added as costs')
    fig.tight_layout();fig.savefig(out/'source_cp_tail_coverage.svg',bbox_inches='tight');fig.savefig(out/'source_cp_tail_coverage.png',dpi=150,bbox_inches='tight');plt.close(fig)
    for key in ['cp_extractor.py','cp_upstream_extractor.py']:(out/key).write_text(paths[key].read_text())
    dump(out/'field_contract.json',dict(timestamp_domain='profiler GPU kernel start/end; not CPU API entry/return or independent wire FCT',
        upstream='Pinned collector filters cat=kernel, CP process group and all_to_all; every selected derived row exactly matches pinned upstream parquet',
        annotation_assignment='[this CPU F/B start,next CPU F/B start); last phase has no upper start bound; NOT next PP operation start',
        layer_slot_assignment='Ordinal mapping under 4 forward / 9 backward kernels per layer; missing rank step invalidates whole CP pair-step; not independent operation-ID or deployed-code proof',
        service='max rank kernel end - max rank kernel start; service_fct aliases this observation, not independent network service',
        topology='CP pairs are within one host and one EP8 group in both pinned static layouts',
        target_contract='declared PP14/CP2/EP8/MB3/layers52; deployed geometry and cross-host clock synchronization unverified',
        cost_admission='No source-fit parameters or edges admitted; GPU overlap is diagnostic and cannot be added to inclusive CPU/PP graph intervals'))
    dump(out/'diagnostic.json',dict(status='SOURCE_CP_INTAKE_PASS_WITH_PARTIAL_COVERAGE',new_prediction=False,new_target_timing_read=False,used_to_fit_model=False,
        source_rank_rows=len(a),expected_rank_rows=int(coverage.expected_rank_events.sum()),missing_rank_rows=int(coverage.missing_rank_events.sum()),
        excluded_observed_rank_rows=int(coverage.excluded_observed_rank_events.sum()),admitted_group_calls=len(calls),invalid_pair_steps=len(invalid),
        upstream_exact_match=True,negative_first_end_minus_last_start_count=int(calls.first_end_minus_last_start_ns.lt(0).sum()),
        negative_first_end_minus_last_start_min_ns=int(calls.first_end_minus_last_start_ns.min()),
        events_ending_after_CPU_annotation=int(a.outside_pp_annotation_end.sum()),events_crossing_next_FB_start=int(a.crosses_next_FB_start.sum()),
        kernel_overlap_rank_steps=int(union.overlapping_duration_ns.gt(0).sum()),source_fit_evidence=[85,90],incremental_validation=[95,100],historical_diagnostic=[60,65,70,75,80],
        selected_rank_and_upstream_dataframe_bytes=dataframe_bytes,analysis_seconds=perf_counter()-begin,raw_trace_scanned=False,
        next='Align source GPU CP intervals with CPU EP wrapper completion and PP B readiness spans; preserve union and exclude incomplete pair-step fits before any new prediction decomposition.'))

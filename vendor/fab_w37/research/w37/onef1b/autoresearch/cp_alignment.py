"""Source GPU CP coverage inside CPU EP and PP windows, without added costs."""
from time import perf_counter
import numpy as np
import pandas as pd
from smoke_worker import dump
from worker import csv
from cp_intake import STEP,interval_union
from ep_completion_audit import split


def mask_partition(start,end,intervals):
    """Disjoint coverage by current/prior/other logical-layer GPU intervals.

    Mask bits 1/2/4 record membership, not additive durations. Absolute epochs
    stay integers even when several events share a start/end boundary.
    """
    start=int(start);end=int(end);assert end>=start
    changes={start:[],end:[]}
    for left,right,category in intervals:
        left=max(start,int(left));right=min(end,int(right));assert category in [0,1,2]
        if right<=left:continue
        changes.setdefault(left,[]).append((category,1));changes.setdefault(right,[]).append((category,-1))
    counts=[0,0,0];durations=[0]*8;previous=start
    for time in sorted(changes):
        mask=sum(1<<i for i,n in enumerate(counts) if n>0);durations[mask]+=time-previous
        for category,delta in changes[time]:counts[category]+=delta
        assert min(counts)>=0;previous=time
    assert sum(durations)==end-start and counts==[0,0,0]
    return durations


def clipped_union(intervals,start,end):
    return sum(max(0,min(e,int(end))-max(s,int(start))) for s,e in interval_union(intervals))


def diagnose(out,paths,plan):
    from pp_semantics import align_observations
    from readiness import ReadinessCosts
    assert plan['diagnostic_access']=='source_only';begin=perf_counter()
    columns=['event_id']+STEP+['pp_stage','cp_group','layer_id','cp_slot_in_layer','start_ns','end_ns','pair_step_complete','rank_step_complete','outside_pp_annotation_end']
    cp=pd.read_csv(paths['cp_source_cp_rank_events.csv'],usecols=columns)
    cp=cp[cp.pair_step_complete].copy();assert cp.rank_step_complete.all() and len(cp)==448896
    cp['split']=split(cp.iteration)
    ends=cp.groupby(['split','pp_stage','phase','cp_slot_in_layer']).agg(kernels=('event_id','size'),ending_after_CPU_annotation=('outside_pp_annotation_end','sum')).reset_index()
    csv(out,'source_cp_slot_tail_counts.csv',ends)
    w=pd.read_csv(paths['ep_source_ep_rank_anchor_events.csv'],usecols=STEP+['pp_stage','ep_group','layer_id','execution_layer','semantic_region','arrival_ns','completion_ns'])
    phase=pd.read_csv(paths['source_pp_trace_events_60_100.csv']);api=pd.read_csv(paths['source_pp_api_events_60_100.csv'])
    aligned,pairs=align_observations(phase,api);ppcost=ReadinessCosts(aligned,pairs,[85,90])
    layout=pd.read_csv(paths['source256_layer_stage_map.csv'])
    layers={int(s):sorted(map(int,g.layer_id)) for s,g in layout.groupby('pp_stage')}
    cp_lookup={key:list(g[['start_ns','end_ns','layer_id','cp_slot_in_layer']].itertuples(index=False,name=None)) for key,g in cp.groupby(STEP,sort=False)}
    rows=[];missing=0
    for r in w.itertuples():
        key=tuple(getattr(r,c) for c in STEP)
        if key not in cp_lookup:missing+=1;continue
        order=layers[r.pp_stage] if r.phase=='forward' else list(reversed(layers[r.pp_stage]))
        ordinal=order.index(r.layer_id);prior=order[ordinal-1] if ordinal>0 else -1
        intervals=[(s,e,0 if layer==r.layer_id else (1 if layer==prior else 2)) for s,e,layer,slot in cp_lookup[key]]
        parts=mask_partition(r.arrival_ns,r.completion_ns,intervals)
        row={c:getattr(r,c) for c in STEP+['pp_stage','ep_group','layer_id','execution_layer','semantic_region']}
        row.update(prior_executed_layer=prior,CPU_wrapper_duration_ns=int(r.completion_ns-r.arrival_ns),cp_union_ns=sum(parts[1:]),
            no_CP_kernel_visible_ns=parts[0],current_layer_CP_only_ns=parts[1],prior_layer_CP_only_ns=parts[2],other_layer_CP_only_ns=parts[4],
            multi_layer_CP_overlap_ns=sum(parts[i] for i in [3,5,6,7]),CPU_arrival_ns=int(r.arrival_ns),CPU_completion_ns=int(r.completion_ns))
        assert sum(row[c] for c in ['no_CP_kernel_visible_ns','current_layer_CP_only_ns','prior_layer_CP_only_ns','other_layer_CP_only_ns','multi_layer_CP_overlap_ns'])==row['CPU_wrapper_duration_ns']
        rows.append(row)
    wrappers=pd.DataFrame(rows);wrappers['split']=split(wrappers.iteration)
    csv(out,'source_EP_CPU_CP_disjoint_coverage.csv.gz',wrappers)
    counts=['CPU_wrapper_duration_ns','cp_union_ns','no_CP_kernel_visible_ns','current_layer_CP_only_ns','prior_layer_CP_only_ns','other_layer_CP_only_ns','multi_layer_CP_overlap_ns']
    profile=wrappers.groupby(['split','pp_stage','phase','semantic_region','execution_layer'])[counts].mean().reset_index()
    csv(out,'source_EP_CP_layer_profile.csv',profile)
    # Match F dispatch and B recompute by logical layer: compare two observed
    # windows, not a new fitted model or an independent GPU compute partition.
    keys=['iteration','rank','pp_stage','microbatch','layer_id']
    selected=wrappers[wrappers.pp_stage.between(1,14)]
    f=selected[selected.semantic_region.eq('ep_dispatch_wall')][keys+counts]
    b=selected[selected.semantic_region.eq('ep_recompute_dispatch_wall')][keys+['execution_layer','split']+counts]
    matched=b.merge(f,on=keys,suffixes=('_b','_f'),validate='one_to_one')
    matched['has_previous_B_layer']=matched.execution_layer.gt(0)
    for name in ['CPU_wrapper_duration_ns','cp_union_ns','no_CP_kernel_visible_ns']:
        matched['B_minus_F_'+name]=matched[name+'_b']-matched[name+'_f']
    assert (matched.B_minus_F_cp_union_ns+matched.B_minus_F_no_CP_kernel_visible_ns).eq(matched.B_minus_F_CPU_wrapper_duration_ns).all()
    csv(out,'source_logical_F_B_CP_coverage.csv.gz',matched)
    csv(out,'source_logical_F_B_CP_summary.csv',matched.groupby(['split','has_previous_B_layer'])[[c for c in matched if c.startswith('B_minus_F_')]].mean().reset_index())
    # PP API first return bounds message completion; only both-single pairs
    # with the receiver already posted by last CP end support the simple tail
    # comparison. Remaining time still mixes non-CP work and PP transport.
    bphase=phase[phase.phase.eq('backward')].set_index(['iteration','rank','microbatch'])
    pp_rows=[];missing_pp=0
    for r in pairs[pairs.direction.eq('B')].itertuples():
        rank=r.sender_stage*16+r.pp_lane;key=(r.iteration,rank,'backward',r.microbatch)
        if key not in cp_lookup:missing_pp+=1;continue
        p=bphase.loc[(r.iteration,rank,r.microbatch)];annotation_end=int(p.observed_end_ns)
        events=[(s,e) for s,e,layer,slot in cp_lookup[key]];last=max(e for s,e in events)
        sender=int(r.sender_post_ns);receiver=int(r.receiver_post_ns);first_return=int(r.first_api_return_ns)
        d,c=ppcost.ready_parameters[('B',r.pp_lane)]
        row=dict(iteration=r.iteration,rank=rank,pp_stage=r.sender_stage,pp_lane=r.pp_lane,microbatch=r.microbatch,
            both_endpoints_single_message=bool(r.both_endpoints_single_message),receiver_posted_by_last_CP_end=receiver<=last,
            CPU_B_end_ns=annotation_end,sender_API_entry_ns=sender,receiver_API_entry_ns=receiver,first_API_return_ns=first_return,last_CP_kernel_end_ns=last,
            last_CP_end_minus_CPU_B_end_ns=last-annotation_end,last_CP_end_minus_sender_API_entry_ns=last-sender,
            first_API_return_minus_last_CP_end_ns=first_return-last,source_fitted_PP_ready_delay_ns=d,source_fitted_PP_completion_ns=c,
            fitted_ready_delay_minus_CP_offset_ns=d-(last-sender),CP_union_after_CPU_B_ns=clipped_union(events,annotation_end,last),
            CP_union_after_B_before_sender_API_ns=clipped_union(events,annotation_end,sender),
            CP_union_inside_sender_API_to_first_return_ns=clipped_union(events,sender,first_return),
            CP_union_after_first_API_return_ns=clipped_union(events,first_return,last))
        assert sender>=annotation_end
        assert row['CP_union_after_CPU_B_ns']==sum(row[n] for n in ['CP_union_after_B_before_sender_API_ns','CP_union_inside_sender_API_to_first_return_ns','CP_union_after_first_API_return_ns'])
        pp_rows.append(row)
    pp=pd.DataFrame(pp_rows);pp['split']=split(pp.iteration)
    pp['admitted_local_completion_comparison']=pp.both_endpoints_single_message&pp.receiver_posted_by_last_CP_end
    csv(out,'source_PP_B_CP_tail_alignment.csv.gz',pp)
    cols=[c for c in pp if c.endswith('_ns') and c not in ['CPU_B_end_ns','sender_API_entry_ns','receiver_API_entry_ns','first_API_return_ns','last_CP_kernel_end_ns']]
    local=pp[pp.admitted_local_completion_comparison]
    csv(out,'source_PP_B_CP_tail_profile.csv',local.groupby(['split','pp_stage'])[cols].mean().reset_index())
    csv(out,'source_PP_readiness_parameters.csv',pd.DataFrame(ppcost.parameter_rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    val=wrappers[wrappers.iteration.isin([95,100])&wrappers.pp_stage.between(1,14)&wrappers.semantic_region.eq('ep_recompute_dispatch_wall')]
    bars=val.groupby('execution_layer')[['current_layer_CP_only_ns','prior_layer_CP_only_ns','other_layer_CP_only_ns','multi_layer_CP_overlap_ns','no_CP_kernel_visible_ns']].mean()/1e6
    bars.columns=['Current layer CP','Previous B layer CP','Other layer CP','Multiple layer CP overlap','No CP kernel visible']
    bars.plot.bar(stacked=True,ax=axes[0]);axes[0].set_xlabel('B execution ordinal');axes[0].set_ylabel('CPU recompute dispatch wall coverage (ms)');axes[0].tick_params(axis='x',labelrotation=0);axes[0].legend(fontsize=7)
    lv=local[local.iteration.isin([95,100])].groupby('pp_stage')[['last_CP_end_minus_sender_API_entry_ns','source_fitted_PP_ready_delay_ns','first_API_return_minus_last_CP_end_ns']].mean()/1e6
    for name,label in [('last_CP_end_minus_sender_API_entry_ns','API entry to last CP end'),('source_fitted_PP_ready_delay_ns','Fitted PP effective readiness'),('first_API_return_minus_last_CP_end_ns','Last CP end to first API return')]:axes[1].plot(lv.index,lv[name],'o-',label=label)
    axes[1].set_xlabel('PP sender stage');axes[1].set_ylabel('Source B local timing (ms)');axes[1].legend(fontsize=7)
    fig.suptitle('Incremental source95/100 observations; no-CP coverage is not pure compute')
    fig.tight_layout();fig.savefig(out/'source_CP_EP_PP_alignment.svg',bbox_inches='tight');fig.savefig(out/'source_CP_EP_PP_alignment.png',dpi=160,bbox_inches='tight');plt.close(fig)
    dump(out/'diagnostic.json',dict(status='SOURCE_CP_EP_PP_ALIGNMENT_PASS',new_prediction=False,new_target_timing_read=False,used_to_fit_model=False,
        source_fit_evidence=[85,90],incremental_validation=[95,100],historical_diagnostic=[60,65,70,75,80],raw_trace_scanned=False,
        included_CP_kernels=len(cp),included_EP_wrappers=len(wrappers),EP_wrappers_excluded_by_CP_pair_coverage=missing,
        PP_B_pairs=len(pp),PP_B_pairs_excluded_by_CP_coverage=missing_pp,PP_local_completion_pairs=len(local),
        CP_kernels_after_first_PP_API_return_pairs=int(pp.CP_union_after_first_API_return_ns.gt(0).sum()),
        wrapper_multilayer_CP_overlap_count=int(wrappers.multi_layer_CP_overlap_ns.gt(0).sum()),
        all_wrapper_and_PP_partitions_conserve=True,analysis_seconds=perf_counter()-begin,
        caution='GPU kernel coverage inside CPU windows is temporal association; no-CP-visible includes compute, EP, idle/runtime and unobserved activity. No network or GPU-ready causality claimed.',
        next='Review logical-layer matched CP and remaining visible-window differences, then test a predeclared source-only CP/non-CP cost replacement without adding inclusive intervals.'))

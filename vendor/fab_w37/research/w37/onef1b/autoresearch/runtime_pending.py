"""Source GPU launch timing within CPU wrappers; interval coverage, not new costs."""
from collections import Counter,defaultdict
import json
import resource
from time import perf_counter
import pandas as pd
from smoke_worker import dump
from worker import csv
from cp_intake import interval_union

KEY=['iteration','rank','phase','microbatch','layer_id','execution_layer','semantic_region','window_id']
PARTS=['CP_visible','nonCP_other_submitted_before_entry','nonCP_other_submitted_within_window',
       'nonCP_other_submission_spans_entry','nonCP_other_mixed_submission','nonCP_other_unknown_submission',
       'nonCP_EP_only','no_nonPP_device_event_visible']


def split_name(iteration):
    return 'source_fit' if iteration in [85,90] else 'source_incremental_validation'


def submission_role(start,end,entry):
    if pd.isna(start) or pd.isna(end):return 'unknown_submission'
    if int(end)<=entry:return 'submitted_before_entry'
    if int(start)<entry:return 'submission_spans_entry'
    return 'submitted_within_window'


def partition_window(start,end,records):
    """Sweep half-open integer intervals, preserve overlap and before/inside launch."""
    changes=defaultdict(list)
    for index,row in enumerate(records):
        s=max(start,int(row['start_ns']));e=min(end,int(row['end_ns']))
        if e>s:changes[s].append((index,1));changes[e].append((index,-1))
    changes[start];changes[end];active=set();previous=start;segments=[];parts=Counter()
    for point in sorted(changes):
        if point>previous:
            current=[records[i] for i in sorted(active) if records[i]['family']!='PP_candidate']
            cp=any(r['family']=='CP_collective' for r in current)
            ep=any(r['family']=='EP_kernel' for r in current)
            other=[r for r in current if r['family'] not in ['CP_collective','EP_kernel']]
            families=sorted({r['family'] for r in other});origins=sorted({r['submission_role'] for r in other})
            if cp:part='CP_visible'
            elif other:part='nonCP_other_'+(origins[0] if len(origins)==1 else 'mixed_submission')
            elif ep:part='nonCP_EP_only'
            else:part='no_nonPP_device_event_visible'
            assert part in PARTS
            duration=point-previous;parts[part]+=duration
            segments.append(dict(start_ns=previous,end_ns=point,duration_ns=duration,partition=part,CP_visible=cp,
                EP_visible=ep,other_families='|'.join(families),other_submission_roles='|'.join(origins),
                other_CPU_owners='|'.join(sorted({str(r.get('unique_CPU_owner_name','unknown')) for r in other})),
                active_device_events=len(current)))
        for index,delta in changes[point]:
            if delta==1:active.add(index)
            else:active.remove(index)
        previous=point
    assert not active and sum(parts.values())==end-start
    return {p+'_ns':parts[p] for p in PARTS},segments


def join_launches(gpu,runtime,cpu):
    assert runtime.event_id.is_unique and gpu.unique_runtime_event_id.notna().all()
    launch=runtime[['event_id','start_ns','end_ns','pid','tid']].rename(columns={'event_id':'unique_runtime_event_id',
        'start_ns':'runtime_start_ns','end_ns':'runtime_end_ns','pid':'runtime_pid','tid':'runtime_tid'})
    d=gpu.merge(launch,on='unique_runtime_event_id',how='left',validate='many_to_one',indicator=True)
    assert d['_merge'].eq('both').all();d=d.drop(columns='_merge')
    assert (d.start_ns-d.runtime_start_ns).eq(d.GPU_start_minus_runtime_launch_start_ns).all()
    assert (d.start_ns-d.runtime_end_ns).eq(d.GPU_start_minus_runtime_launch_end_ns).all()
    assert d.runtime_start_ns.le(d.start_ns).all()
    unique=cpu[cpu.external_id.notna()&~cpu.external_id.duplicated(keep=False)]
    owner=unique.set_index('external_id')
    d['CPU_owner_input_dims']=d.external_id.map(owner.input_dims)
    d['CPU_owner_name_checked']=d.external_id.map(owner.name)
    matched=d.CPU_external_id_matches.eq(1)
    assert d.loc[matched,'CPU_owner_name_checked'].eq(d.loc[matched,'unique_CPU_owner_name']).all()
    return d


def pair_dispatch(frame,columns):
    selected=frame[frame.semantic_region.isin(['ep_dispatch_wall','ep_recompute_dispatch_wall'])]
    keys=['iteration','rank','microbatch','layer_id','split']
    f=selected[selected.phase=='forward'][keys+columns]
    b=selected[selected.phase=='backward'][keys+['execution_layer']+columns]
    paired=b.merge(f,on=keys,suffixes=('_B','_F'),validate='one_to_one')
    paired['has_previous_B_layer']=paired.execution_layer.gt(0)
    for c in columns:paired['B_minus_F_'+c]=paired[c+'_B']-paired[c+'_F']
    return paired


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only'
    windows=pd.read_csv(paths['runtime_windows.csv.gz'],dtype={'pid':'string','tid':'string'},low_memory=False)
    windows=windows[windows.window_type=='EP_wrapper'].copy()
    assert set(windows.iteration)=={85,90,95,100} and set(windows['rank'])=={16}
    totals=[];segments=[];intersections=[];checks=[];kernel_profiles=[]
    for iteration in [85,90,95,100]:
        dtype={'correlation':'string','external_id':'string','pid':'string','tid':'string','stream':'string'}
        gpu=pd.read_csv(paths[f'source{iteration}_GPU.csv.gz'],dtype=dtype,low_memory=False)
        runtime=pd.read_csv(paths[f'source{iteration}_runtime.csv.gz'],dtype=dtype,low_memory=False)
        cpu=pd.read_csv(paths[f'source{iteration}_CPU.csv.gz'],dtype=dtype,low_memory=False)
        d=join_launches(gpu,runtime,cpu);w=windows[windows.iteration==iteration]
        # Same-thread EP CPU windows are disjoint; containment is temporal evidence,
        # not an inferred device dependency or a deployed logical operation ID.
        owner_ids=[];owner_regions=[];owner_layers=[]
        by_thread={k:x.sort_values('start_ns') for k,x in w.groupby(['pid','tid'])}
        for r in d.itertuples():
            subset=by_thread.get((r.runtime_pid,r.runtime_tid))
            hits=[] if subset is None else subset[(subset.start_ns<=r.runtime_start_ns)&(subset.end_ns>r.runtime_start_ns)].to_dict('records')
            assert len(hits)<=1
            owner_ids.append(hits[0]['window_id'] if hits else '')
            owner_regions.append(hits[0]['semantic_region'] if hits else 'outside_EP_CPU_wrapper')
            owner_layers.append(int(hits[0]['layer_id']) if hits else -1)
        d['launch_EP_wrapper_id']=owner_ids;d['launch_EP_semantic_region']=owner_regions;d['launch_EP_layer_id']=owner_layers
        for row in w.to_dict('records'):
            start=int(row['start_ns']);end=int(row['end_ns']);meta={k:row[k] for k in KEY};meta['split']=split_name(iteration)
            hit=d[(d.start_ns<end)&(d.end_ns>start)].copy()
            hit['submission_role']=[submission_role(s,e,start) for s,e in zip(hit.runtime_start_ns,hit.runtime_end_ns)]
            part,local=partition_window(start,end,hit.to_dict('records'))
            assert part['CP_visible_ns']==row['CP_union_ns']
            assert part['no_nonPP_device_event_visible_ns']==row['no_non_PP_GPU_activity_visible_ns']
            total=dict(**meta,window_duration_ns=end-start,**part);totals.append(total)
            for segment in local:segments.append(dict(**meta,**segment))
            for event in hit.to_dict('records'):
                intersections.append(dict(**meta,GPU_event_id=event['event_id'],family=event['family'],GPU_name=event['name'],
                    category=event['category'],GPU_stream=event['stream'],GPU_start_ns=int(event['start_ns']),GPU_end_ns=int(event['end_ns']),
                    clipped_start_ns=max(start,int(event['start_ns'])),clipped_end_ns=min(end,int(event['end_ns'])),
                    runtime_event_id=event['unique_runtime_event_id'],runtime_name=event['unique_runtime_name'],
                    runtime_start_ns=int(event['runtime_start_ns']),runtime_end_ns=int(event['runtime_end_ns']),
                    runtime_pid=event['runtime_pid'],runtime_tid=event['runtime_tid'],submission_role=event['submission_role'],
                    launch_EP_wrapper_id=event['launch_EP_wrapper_id'],launch_EP_semantic_region=event['launch_EP_semantic_region'],
                    launch_EP_layer_id=event['launch_EP_layer_id'],CPU_owner_name=event['unique_CPU_owner_name'],
                    CPU_owner_input_dims=event['CPU_owner_input_dims']))
        checks.append(dict(iteration=iteration,rank=16,split=split_name(iteration),CPU_windows=len(w),GPU_device_events=len(d),
            all_runtime_timestamp_joins_exact=True,all_unique_CPU_owner_names_exact=True,CP_and_no_device_coverage_match_T14=True))
    totals=pd.DataFrame(totals);segments=pd.DataFrame(segments);intersections=pd.DataFrame(intersections)
    for name,frame in [('source_CPU_window_submission_coverage.csv.gz',totals),('source_GPU_interval_segments.csv.gz',segments),
                       ('source_GPU_CPU_window_intersections.csv.gz',intersections)]:csv(out,name,frame)
    columns=['window_duration_ns']+[p+'_ns' for p in PARTS]
    paired=pair_dispatch(totals,columns);csv(out,'source_logical_F_B_submission_pairs.csv',paired)
    delta_cols=['B_minus_F_'+c for c in columns]
    summary=paired.groupby(['split','has_previous_B_layer'])[delta_cols].mean().reset_index()
    summary['pairs']=paired.groupby(['split','has_previous_B_layer']).size().values
    csv(out,'source_logical_F_B_submission_summary.csv',summary)
    # Non-CP visible family signatures are disjoint. Marginal family durations
    # would double-count overlaps, so retain a multi-family signature as one bin.
    seg=segments[~segments.CP_visible].copy()
    seg['family_signature']=seg.other_families.replace('', 'no_other_device_family')
    profile=seg.groupby(KEY+['split','family_signature'],dropna=False).duration_ns.sum().reset_index()
    csv(out,'source_nonCP_family_signature_by_window.csv.gz',profile)
    # Include zero coverage windows in family means and F/B paired deltas.
    wide=profile.pivot(index=KEY+['split'],columns='family_signature',values='duration_ns').fillna(0).astype('int64').reset_index()
    family_columns=[c for c in wide if c not in KEY+['split']]
    wide=totals[KEY+['split']].merge(wide,on=KEY+['split'],how='left').fillna({c:0 for c in family_columns})
    family_paired=pair_dispatch(wide,family_columns);csv(out,'source_logical_F_B_family_pairs.csv',family_paired)
    family_summary=family_paired.groupby(['split','has_previous_B_layer'])[['B_minus_F_'+c for c in family_columns]].mean().reset_index()
    csv(out,'source_logical_F_B_family_summary.csv',family_summary)
    # CPU owners are also retained as simultaneous sets, never summing marginal
    # kernel residence to infer a new compute cost.
    owned=segments[(~segments.CP_visible)&segments.other_submission_roles.eq('submitted_before_entry')]
    own=owned.groupby(['split','phase','execution_layer','semantic_region','other_CPU_owners'],dropna=False).duration_ns.sum().reset_index()
    csv(out,'source_preentry_nonCP_owner_signatures.csv.gz',own)
    draw(out,summary,family_summary,family_columns)
    dump(out/'field_contract.json',dict(time_domain='int64 absolute ns; half-open intervals',scope='source256 rank16 PPstage1 four-layer interior only',
        split={'fit':[85,90],'incremental_validation':[95,100],'target':'not parsed'},
        inference='Before-entry requires matched runtime API end <= CPU wrapper entry; it does not prove a prior logical layer dependency.',
        classification='CP exact process group, other families heuristic; owner External id and launch correlation checked. Multi-family/owner overlap retained as sets.',
        costs='No fitted costs or prediction; CPU wrapper partitions and GPU event residence are separate views, not additive.',
        missing='No event/stream synchronization handles or independently verified device dependency graph; GPU residence may include polling.',
        prior_stage=plan['source_evidence_stages'][0]))
    dump(out/'diagnostic.json',dict(status='SOURCE_RUNTIME_PENDING_COVERAGE_PASS',new_prediction=False,used_to_fit_model=False,new_target_timing_read=False,
        raw_trace_scanned=False,source_checks=checks,all_window_partitions_conserve=True,logical_pairs=len(paired),
        source_logical_pair_means=summary.to_dict('records'),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        analysis_seconds=perf_counter()-begin,formal_topology_changed=False,
        next='Review same-cohort launch timing and owner/family signatures, then assess evidence for transferable source GPU pending-work costs; no single-rank scaling or target fit.'))


def draw(out,summary,family_summary,families):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(17,6))
    data=summary[summary.has_previous_B_layer].set_index('split')
    colors=['#377eb8','#4daf4a','#ff7f00','#984ea3','#e41a1c','#a65628','#f781bf','#999999']
    labels=['CP visible','Other submitted before entry','Other submitted within wrapper','Submission spans entry','Mixed submission','Unknown submission','EP only outside CP','No non-PP event visible']
    positive=[0.,0.];negative=[0.,0.]
    for part,color,label in zip(PARTS,colors,labels):
        values=[float(data.loc[s,'B_minus_F_'+part+'_ns'])/1e6 for s in ['source_fit','source_incremental_validation']]
        bottom=[positive[i] if v>=0 else negative[i] for i,v in enumerate(values)]
        axes[0].bar([0,1],values,bottom=bottom,label=label,color=color)
        for i,v in enumerate(values):
            if v>=0:positive[i]+=v
            else:negative[i]+=v
    axes[0].set_xticks([0,1],['85/90 source fit evidence','95/100 incremental evidence'])
    axes[0].set_ylabel('B minus same logical F wrapper (ms)');axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.14),ncol=2)
    axes[0].set_title('Later B dispatch: disjoint temporal coverage')
    f=family_summary[family_summary.has_previous_B_layer].set_index('split')
    values=f.loc['source_incremental_validation', ['B_minus_F_'+c for c in families]].astype(float)/1e6
    values=values[values.abs()>.1].sort_values();names=[n.removeprefix('B_minus_F_').replace('|',' + ') for n in values.index]
    axes[1].barh(names,values);axes[1].set_xlabel('Non-CP family-signature B minus F (ms)')
    axes[1].set_title('95/100 source rank16; family names are heuristic')
    fig.suptitle('Observed interval coverage and launch timing; no new prediction or pure-compute attribution')
    fig.tight_layout();fig.savefig(out/'source_pending_submission.svg',bbox_inches='tight');fig.savefig(out/'source_pending_submission.png',dpi=160,bbox_inches='tight');plt.close(fig)

"""Exact device visibility gaps and CPU submission context; not device idle costs."""
from bisect import bisect_left
from collections import defaultdict
import resource
from time import perf_counter
import numpy as np
import pandas as pd
from cp_intake import interval_union
from runtime_pending import join_launches,split_name
from smoke_worker import dump
from worker import csv

SUBMISSION=['before_next_runtime_start','during_next_runtime_API','after_next_runtime_end',
            'tied_GPU_successors','missing_runtime_successor','outside_phase_successor','no_successor']
POSITION=['inside_EP_wrapper','outside_EP_inside_checkpoint','outside_checkpoint']
DTYPE={c:'string' for c in ['correlation','external_id','pid','tid','stream']}
ITERATIONS=[85,90,95,100]


def complement(start,end,intervals):
    """Complement of the clipped union, including leading and trailing gaps."""
    assert isinstance(start,int) and isinstance(end,int) and end>=start
    merged=interval_union((max(start,int(s)),min(end,int(e))) for s,e in intervals if e>start and s<end and e>s)
    gaps=[];cursor=start
    for s,e in merged:
        if s>cursor:gaps.append((cursor,s))
        cursor=e
    if cursor<end:gaps.append((cursor,end))
    assert sum(e-s for s,e in merged)+sum(e-s for s,e in gaps)==end-start
    return merged,gaps


def submission_parts(start,end,successors,phase_end):
    """Next observed GPU event is a temporal successor, never a proven blocker."""
    if not successors:return [(start,end,'no_successor')]
    if int(successors[0]['start_ns'])>=phase_end:return [(start,end,'outside_phase_successor')]
    if len(successors)>1:return [(start,end,'tied_GPU_successors')]
    r=successors[0]
    if pd.isna(r.get('runtime_start_ns')) or pd.isna(r.get('runtime_end_ns')):
        return [(start,end,'missing_runtime_successor')]
    rs,re=int(r['runtime_start_ns']),int(r['runtime_end_ns']);assert re>=rs
    cuts=sorted({start,end,*[t for t in [rs,re] if start<t<end]})
    return [(s,e,'before_next_runtime_start' if e<=rs else ('after_next_runtime_end' if s>=re else 'during_next_runtime_API'))
            for s,e in zip(cuts,cuts[1:])]


def overlay_segments(start,end,submission,contexts):
    """One disjoint cross-product view: submission, CPU location, runtime activity."""
    changes=defaultdict(list);active=set();records=[]
    for kind,rows in contexts.items():
        for r in rows:
            s=max(start,int(r['start_ns']));e=min(end,int(r['end_ns']))
            if e<=s:continue
            index=len(records);records.append((kind,r))
            changes[s].append((index,1));changes[e].append((index,-1))
    for s,e,label in submission:
        index=len(records);records.append(('submission',dict(name=label)))
        changes[s].append((index,1));changes[e].append((index,-1))
    changes[start];changes[end];previous=start;result=[]
    for point in sorted(changes):
        if point>previous:
            view=defaultdict(list)
            for index in sorted(active):
                kind,r=records[index];view[kind].append(r)
            assert len(view['submission'])==1
            ep=view['EP'];parent=view['checkpoint'];runtime=view['runtime'];same=view['same_thread_runtime']
            result.append(dict(start_ns=previous,end_ns=point,duration_ns=point-previous,
                submission=view['submission'][0]['name'],position='inside_EP_wrapper' if ep else ('outside_EP_inside_checkpoint' if parent else 'outside_checkpoint'),
                EP_semantic_signature='|'.join(sorted({r['semantic_region'] for r in ep})) or 'none',
                EP_window_ids='|'.join(sorted(r['window_id'] for r in ep)),
                checkpoint_names='|'.join(sorted({r['name'] for r in parent})) or 'none',
                checkpoint_ids='|'.join(sorted(r['event_id'] for r in parent)),
                runtime_signature='|'.join(sorted({r['name'] for r in runtime})) or 'no_visible_runtime_API',
                successor_thread_runtime_signature='|'.join(sorted({r['name'] for r in same})) or 'no_visible_runtime_API',
                runtime_sync_visible=any('Synchronize' in r['name'] for r in runtime),
                successor_thread_runtime_sync_visible=any('Synchronize' in r['name'] for r in same),
                PP_candidate_visible=bool(view['PP'])))
        for index,delta in changes[point]:
            if delta==1:active.add(index)
            else:active.remove(index)
        previous=point
    assert not active and sum(r['duration_ns'] for r in result)==end-start
    return result


def analyze_iteration(gpu,runtime,cpu,windows,case):
    """Same algorithm for source first and later sealed evaluator tables."""
    d=join_launches(gpu,runtime,cpu)
    nonpp=d[~d.family.eq('PP_candidate')&d.duration_ns.gt(0)].sort_values(['start_ns','event_id'])
    records=nonpp.to_dict('records');starts=[int(r['start_ns']) for r in records]
    parents=cpu[cpu.name.isin(['CheckpointFunction','CheckpointFunctionBackward'])]
    fb=windows[windows.window_type.eq('FB_annotation')].sort_values('start_ns')
    assert len(fb)==(8 if case=='source256' else 6)
    all_gaps=[];all_segments=[];phase_rows=[]
    for w in fb.to_dict('records'):
        start,end=int(w['start_ns']),int(w['end_ns'])
        iteration=int(w['iteration']);microbatch=int(w['microbatch']);last=3 if case=='source256' else 2
        meta=dict(case=case,iteration=iteration,rank=int(w['rank']),phase=w['phase'],microbatch=microbatch,
            MB_role='first' if microbatch==0 else ('last' if microbatch==last else 'middle'),phase_window_id=w['window_id'],
            split=split_name(iteration) if case=='source256' else 'target_development_posthoc')
        visible,gaps=complement(start,end,zip(nonpp.start_ns,nonpp.end_ns))
        gap_total=sum(e-s for s,e in gaps)
        assert gap_total==int(w['no_non_PP_GPU_activity_visible_ns'])
        # Select by time, never by a GPU event's launch phase. Queued work can
        # extend beyond its CPU owner or F/B annotation.
        contexts={'EP':windows[windows.window_type.eq('EP_wrapper')], 'checkpoint':parents,
                  'runtime':runtime,'PP':d[d.family.eq('PP_candidate')]}
        context_rows={k:x[(x.start_ns<end)&(x.end_ns>start)].to_dict('records') for k,x in contexts.items()}
        context_bounds={k:(np.array([int(r['start_ns']) for r in rows],dtype='int64'),
                           np.array([int(r['end_ns']) for r in rows],dtype='int64')) for k,rows in context_rows.items()}
        phase_segments=[]
        for index,(gs,ge) in enumerate(gaps):
            pos=bisect_left(starts,ge);successors=[]
            if pos<len(records):
                successor_start=starts[pos]
                while pos<len(records) and starts[pos]==successor_start:
                    successors.append(records[pos]);pos+=1
            parts=submission_parts(gs,ge,successors,end)
            gap_id=f"{w['window_id']}:gap{index}"
            gap=dict(**meta,gap_id=gap_id,start_ns=gs,end_ns=ge,duration_ns=ge-gs,
                leading_boundary=gs==start,trailing_boundary=ge==end,successor_count=len(successors),
                successor_GPU_ids='|'.join(r['event_id'] for r in successors),
                successor_GPU_families='|'.join(sorted({r['family'] for r in successors})),
                successor_GPU_names='|'.join(sorted({r['name'] for r in successors})),
                successor_GPU_streams='|'.join(sorted({str(r['stream']) for r in successors})),
                successor_runtime_ids='|'.join(sorted({r['unique_runtime_event_id'] for r in successors})),
                successor_runtime_names='|'.join(sorted({r['unique_runtime_name'] for r in successors})),
                successor_CPU_owners='|'.join(sorted({str(r['unique_CPU_owner_name']) for r in successors})))
            # Store nullable absolute timestamps as Python integers/strings,
            # never cast an epoch to float through a nullable DataFrame join.
            for column in ['start_ns','runtime_start_ns','runtime_end_ns']:
                gap['successor_'+column]=str(int(successors[0][column])) if len(successors)==1 else ''
            all_gaps.append(gap)
            local={k:[rows[i] for i in np.flatnonzero((context_bounds[k][0]<ge)&(context_bounds[k][1]>gs))]
                   for k,rows in context_rows.items()}
            local['same_thread_runtime']=[r for r in local['runtime'] if len(successors)==1 and
                r['pid']==successors[0]['runtime_pid'] and r['tid']==successors[0]['runtime_tid']]
            pieces=overlay_segments(gs,ge,parts,local)
            for piece in pieces:
                piece.update(**meta,gap_id=gap_id,leading_boundary=gs==start,trailing_boundary=ge==end)
            phase_segments.extend(pieces)
        assert sum(r['duration_ns'] for r in phase_segments)==gap_total
        all_segments.extend(phase_segments)
        phase_rows.append(dict(**meta,start_ns=start,end_ns=end,phase_duration_ns=end-start,
            nonPP_device_visible_ns=sum(e-s for s,e in visible),no_nonPP_device_event_visible_ns=gap_total,
            gap_count=len(gaps),segment_count=len(phase_segments),
            **{s+'_ns':sum(r['duration_ns'] for r in phase_segments if r['submission']==s) for s in SUBMISSION},
            **{p+'_ns':sum(r['duration_ns'] for r in phase_segments if r['position']==p) for p in POSITION},
            runtime_sync_overlap_ns=sum(r['duration_ns'] for r in phase_segments if r['runtime_sync_visible']),
            successor_thread_runtime_sync_overlap_ns=sum(r['duration_ns'] for r in phase_segments if r['successor_thread_runtime_sync_visible']),
            PP_candidate_overlap_ns=sum(r['duration_ns'] for r in phase_segments if r['PP_candidate_visible'])))
    return pd.DataFrame(all_gaps),pd.DataFrame(all_segments),pd.DataFrame(phase_rows)


def save_analysis(out,gaps,segments,phase_rows):
    csv(out,'device_gaps.csv.gz',gaps);csv(out,'gap_context_segments.csv.gz',segments);csv(out,'FB_gap_accounting.csv',phase_rows)
    columns=['phase_duration_ns','nonPP_device_visible_ns','no_nonPP_device_event_visible_ns']+[s+'_ns' for s in SUBMISSION]+[p+'_ns' for p in POSITION]
    keys=['case','iteration','split','phase']
    # First average microbatches within iteration; then equal-weight iterations.
    per_iteration=phase_rows.groupby(keys)[columns].mean().reset_index()
    csv(out,'FB_gap_per_iteration.csv',per_iteration)
    profile=per_iteration.groupby(['case','split','phase'])[columns].mean().reset_index()
    csv(out,'FB_gap_profile.csv',profile)
    csv(out,'FB_gap_by_MB_role.csv',phase_rows.groupby(['case','iteration','split','phase','MB_role'])[columns].mean().reset_index())
    for name,dimensions in [('position_submission',['position','submission']),
                            ('EP_submission',['EP_semantic_signature','submission']),
                            ('runtime_submission',['successor_thread_runtime_signature','submission'])]:
        value=segments.groupby(keys+dimensions,dropna=False).duration_ns.sum().reset_index()
        counts=phase_rows.groupby(keys).size().rename('phase_count').reset_index()
        value=value.merge(counts,on=keys,validate='many_to_one');value['mean_per_phase_ns']=value.duration_ns/value.phase_count
        csv(out,'gap_'+name+'_per_iteration.csv.gz',value)
    draw(out,profile)


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only'
    windows=pd.read_csv(paths['runtime_windows.csv.gz'],dtype=DTYPE,low_memory=False)
    outputs=[]
    for iteration in ITERATIONS:
        gpu=pd.read_csv(paths[f'source{iteration}_GPU.csv.gz'],dtype=DTYPE,low_memory=False)
        runtime=pd.read_csv(paths[f'source{iteration}_runtime.csv.gz'],dtype=DTYPE,low_memory=False)
        cpu=pd.read_csv(paths[f'source{iteration}_CPU.csv.gz'],dtype=DTYPE,low_memory=False)
        outputs.append(analyze_iteration(gpu,runtime,cpu,windows[windows.iteration.eq(iteration)],'source256'))
    frames=[pd.concat([r[i] for r in outputs],ignore_index=True) for i in range(3)]
    save_analysis(out,*frames)
    write_contract(out,plan)
    dump(out/'diagnostic.json',dict(status='SOURCE_DEVICE_GAP_CONTEXT_PASS',new_prediction=False,used_to_fit_model=False,
        new_target_timing_read=False,raw_trace_scanned=False,source_iterations=ITERATIONS,source_ranks=[16],
        phase_count=len(frames[2]),gap_count=len(frames[0]),segment_count=len(frames[1]),
        all_phase_visibility_and_gap_partitions_exact_ns=True,formal_topology_changed=False,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,analysis_seconds=perf_counter()-begin,
        next='Freeze source-tested classifier, then apply to cached target tables only after existing global prediction seal; do not infer a causal blocker or add gap costs.'))


def write_contract(out,plan):
    dump(out/'field_contract.json',dict(time='integer absolute ns, half-open intervals; no threshold or rounding in accounting',
        visible='Union of all non-PP kernel + copy/set events, independent of CPU launch ownership; PP overlay separate',
        gaps='Exact complement inside each CPU F/B annotation; leading/trailing boundary gaps retained',
        successor='First subsequent positive-duration non-PP GPU event by start time; ties, missing runtime, outside-phase and absent successor explicit',
        submission='Split at matched successor runtime API start/end; temporal context, not proof of the causal blocking task',
        position='EP wrapper priority, otherwise CPU CheckpointFunction or CheckpointFunctionBackward, otherwise outside checkpoint',
        runtime='All-thread and successor-launch-thread runtime views overlap gap duration, not additive costs; Synchronize name classifier only',
        limitations=['No visible event does not establish physical device idle or CPU work','All-thread runtime can include unrelated profiler waits',
            'Next event may be on another stream; no independently observed stream dependency graph','One rank and different hosts; no all-rank or host transfer claim',
            'source95/100 historically exposed incremental validation; target development/posthoc only'],sealed_reference=plan['sealed_reference']))


def draw(out,profile):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    colors=['#e69f00','#cc79a7','#0072b2','#009e73','#999999','#666666','#222222']
    labels=[f'{r.case}\n{r.split}\n{r.phase}' for r in profile.itertuples()]
    for ax,parts,title in zip(axes,[SUBMISSION,POSITION],['Gap split by next GPU event submission','Gap split by CPU annotation location']):
        bottom=np.zeros(len(profile))
        for name,color in zip(parts,colors):
            values=profile[name+'_ns'].to_numpy()/1e6
            ax.bar(range(len(profile)),values,bottom=bottom,label=name,color=color);bottom+=values
        ax.set_xticks(range(len(profile)),labels,fontsize=7);ax.set_ylabel('Mean gap per F/B annotation (ms)');ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle('Observed rank16 device visibility gaps; no physical-idle or causal-blocker claim')
    fig.tight_layout();fig.savefig(out/'device_gap_context.svg');fig.savefig(out/'device_gap_context.png',dpi=160);plt.close(fig)

"""Bounded, sealed target raw intake. All outputs are development/posthoc only."""
from pathlib import Path
import json
import resource
from time import perf_counter
import numpy as np
import pandas as pd
from guards import ROOT
from smoke_worker import dump,sha
from worker import csv
from runtime_intake import parse_one,PP,EP,union_in
from cp_alignment import mask_partition
from runtime_pending import join_launches
from checkpoint_tail import checkpoint_ownership,endpoint_rows


def target_intake_capabilities(plan,items,stage):
    intake=plan.get('target_raw_trace_intake')
    if not intake:return [],None
    assert not plan.get('raw_trace_intake') and not plan['variants']
    assert plan['diagnostic']=='target_runtime_intake' and plan['diagnostic_access']=='evaluator'
    review_path=ROOT/plan['resource_review_file'];assert sha(review_path)==plan['resource_review_sha256']
    review=json.loads(review_path.read_text());assert review['sealed_reference']==plan['sealed_reference']
    raw=[i for i in items if i.get('raw_trace')]
    assert {i['key'] for i in raw}==set(intake['keys'])
    assert len(raw)<=intake['max_files']<=review['max_files']
    assert sum(i['size_bytes'] for i in raw)<=intake['max_total_bytes']<=review['max_total_bytes']
    registered={r['path']:r for r in review['raw_files']}
    for item in raw:
        frozen=registered[item['path']]
        assert item['case_id']=='target224' and item['roles']==['evaluator224_raw_posthoc_NEVER_MODEL_FIT']
        assert all(item[k]==frozen[k] for k in ['size_bytes','rank','iteration'])
    probe=None
    if plan.get('target_first_file_probe_key'):
        item=next(i for i in items if i.get('key')==plan['target_first_file_probe_key'])
        assert not item.get('raw_trace') and item['roles']==['evaluator_target_raw_probe_gate_NEVER_MODEL_FIT']
        probe=Path(item['path']).resolve();assert probe.is_relative_to(ROOT/'results/w37/A')
    else:
        assert len(raw)==1 and raw[0]['iteration']==review['first_file']['iteration'] and raw[0]['rank']==review['first_file']['rank']
        assert raw[0]['size_bytes']<=review['first_file']['max_bytes']
    return ([i['path'] for i in raw] if stage=='diagnose' else []),probe


def verify_target_probe(plan,items,path):
    probe=json.loads(path.read_text())
    assert probe['status']=='TARGET_RAW_RUNTIME_POSTHOC_PASS' and probe['raw_files']==1 and probe['first_file_gate_pass']
    assert probe['new_prediction'] is False and probe['used_to_fit_model'] is False
    assert probe['sealed_reference']==plan['sealed_reference']
    assert probe['peak_RSS_bytes']<=1024*1024*1024
    first=probe['raw_input_identity'][0]
    raw=next(i for i in items if i.get('raw_trace') and i['iteration']==first['iteration'] and i['rank']==first['rank'])
    assert raw['sha256']==first['sha256']


def correlate(gpu,cpu,runtime):
    def compact(frame):
        frame=frame[frame.correlation.ne('')]
        x=frame.groupby('correlation').agg(matches=('event_id','size'),event_id=('event_id','first'),name=('name','first'),start_ns=('start_ns','first'),end_ns=('end_ns','first'))
        return {r.Index:(r.matches,r.event_id,r.name,int(r.start_ns),int(r.end_ns)) for r in x.itertuples()}
    launches=compact(runtime[runtime.name.str.contains('launch',case=False)]);all_runtime=compact(runtime)
    x=cpu[cpu.category.eq('cpu_op')&cpu.external_id.ne('')].groupby('external_id').agg(matches=('event_id','size'),name=('name','first'))
    owners={r.Index:(r.matches,r.name) for r in x.itertuples()};rows=[]
    for r in gpu.itertuples():
        launch=(launches if r.category=='kernel' else all_runtime).get(r.correlation);owner=owners.get(r.external_id)
        n=0 if launch is None else launch[0];count=0 if owner is None else owner[0]
        rows.append(dict(event_id=r.event_id,runtime_launch_matches=n if r.category=='kernel' else 0,runtime_event_matches=n,
            CPU_external_id_matches=count,unique_runtime_event_id=launch[1] if n==1 else '',unique_runtime_name=launch[2] if n==1 else '',
            GPU_start_minus_runtime_launch_start_ns=int(r.start_ns)-launch[3] if n==1 else None,
            GPU_start_minus_runtime_launch_end_ns=int(r.start_ns)-launch[4] if n==1 else None,unique_CPU_owner_name=owner[1] if count==1 else ''))
    links=pd.DataFrame(rows)
    for name in ['GPU_start_minus_runtime_launch_start_ns','GPU_start_minus_runtime_launch_end_ns']:links[name]=pd.array(links[name],dtype='Int64')
    return gpu.merge(links,on='event_id',validate='one_to_one')


def target_windows(cpu,ground,iteration,rank):
    rows=[];phases=cpu[cpu.category.eq('user_annotation')&cpu.name.isin(['forward_step','backward_step'])]
    for phase in ['forward','backward']:
        actual=phases[phases.name.eq(phase+'_step')].sort_values('start_ns');expected=ground[ground.phase.eq(phase)].sort_values('microbatch')
        assert len(actual)==len(expected)==3
        assert actual.start_ns.tolist()==expected.observed_start_ns.tolist() and actual.end_ns.tolist()==expected.observed_end_ns.tolist()
        for parent,ref in zip(actual.itertuples(),expected.itertuples()):
            meta=dict(iteration=iteration,rank=rank,phase=phase,microbatch=ref.microbatch)
            rows.append(dict(**meta,window_type='FB_annotation',window_id=parent.event_id,layer_id=-1,execution_layer=-1,semantic_region=phase,start_ns=int(parent.start_ns),end_ns=int(parent.end_ns),pid=parent.pid,tid=parent.tid))
            ops=cpu[cpu.category.eq('cpu_op')&cpu.name.isin(set(EP.values()))&cpu.start_ns.ge(parent.start_ns)&cpu.end_ns.le(parent.end_ns)]
            if phase=='forward':
                for semantic,name in [('ep_dispatch_wall','FusedDispatch'),('ep_combine_wall','FusedCombine')]:
                    group=ops[ops.name.eq(name)].sort_values('start_ns');assert len(group)==4
                    for ordinal,r in enumerate(group.itertuples()):rows.append(dict(**meta,window_type='EP_wrapper',window_id=r.event_id,layer_id=2+ordinal,execution_layer=ordinal,semantic_region=semantic,start_ns=int(r.start_ns),end_ns=int(r.end_ns),pid=r.pid,tid=r.tid))
                assert len(ops)==8
            else:
                parents=cpu[cpu.name.eq('CheckpointFunctionBackward')&cpu.start_ns.ge(parent.start_ns)&cpu.end_ns.le(parent.end_ns)].sort_values('start_ns');assert len(parents)==4
                names={'FusedDispatch':'ep_recompute_dispatch_wall','FusedCombine':'ep_recompute_combine_wall','FusedCombineBackward':'ep_combine_backward_wall','FusedDispatchBackward':'ep_dispatch_backward_wall'}
                for ordinal,p in enumerate(parents.itertuples()):
                    inside=ops[ops.pid.eq(p.pid)&ops.tid.eq(p.tid)&ops.start_ns.ge(p.start_ns)&ops.end_ns.le(p.end_ns)]
                    assert len(inside)==4 and set(inside.name)==set(names)
                    for r in inside.itertuples():rows.append(dict(**meta,window_type='EP_wrapper',window_id=r.event_id,layer_id=5-ordinal,execution_layer=ordinal,semantic_region=names[r.name],start_ns=int(r.start_ns),end_ns=int(r.end_ns),pid=r.pid,tid=r.tid))
                assert len(ops)==16
    pp=cpu[cpu.category.eq('user_annotation')&cpu.name.isin(PP)]
    assert len(pp)==12 and set(pp.name)=={'recv_forward','send_forward','recv_backward','send_backward'}
    for name,g in pp.groupby('name'):
        assert len(g)==3
        for mb,r in enumerate(g.sort_values('start_ns').itertuples()):rows.append(dict(iteration=iteration,rank=rank,phase='API',microbatch=mb,
            window_type='PP_API',window_id=r.event_id,layer_id=-1,execution_layer=-1,semantic_region=name,start_ns=int(r.start_ns),end_ns=int(r.end_ns),pid=r.pid,tid=r.tid))
    return pd.DataFrame(rows)


def check_legacy(windows,ground,legacy):
    actual=windows[windows.window_type.eq('EP_wrapper')].copy();actual['wrapper_name']=actual.semantic_region.map(EP)
    actual=actual.sort_values('start_ns');key=['iteration','rank','phase','microbatch','wrapper_name']
    actual['wrapper_occurrence']=actual.groupby(key).cumcount()
    phase=ground[['iteration','rank','phase','microbatch','observed_start_ns']]
    actual=actual.merge(phase,on=['iteration','rank','phase','microbatch'],validate='many_to_one')
    actual['start_offset_ns']=actual.start_ns-actual.observed_start_ns;actual['end_offset_ns']=actual.end_ns-actual.observed_start_ns
    joined=actual.merge(legacy,on=key+['wrapper_occurrence'],validate='one_to_one',suffixes=('','_legacy'))
    assert len(joined)==len(actual)==len(legacy)
    joined['start_difference_ns']=joined.start_offset_ns-joined.wrapper_start_offset_ms.mul(1e6).round().astype('int64')
    joined['end_difference_ns']=joined.end_offset_ns-joined.wrapper_end_offset_ms.mul(1e6).round().astype('int64')
    assert joined[['start_difference_ns','end_difference_ns']].abs().max().max()<=512
    return joined


def coverage(gpu,runtime,windows,case):
    rows=[];main=gpu[~gpu.family.eq('PP_candidate')].copy();main['activity_category']=np.where(main.family.eq('CP_collective'),0,np.where(main.family.eq('EP_kernel'),1,2))
    sync=runtime[runtime.name.isin(['musaStreamSynchronize','musaDeviceSynchronize','musaEventSynchronize'])]
    for w in windows.to_dict('records'):
        start,end=int(w['start_ns']),int(w['end_ns']);h=main[main.start_ns.lt(end)&main.end_ns.gt(start)]
        parts=mask_partition(start,end,h[['start_ns','end_ns','activity_category']].itertuples(index=False,name=None))
        all_gpu=gpu[gpu.start_ns.lt(end)&gpu.end_ns.gt(start)];all_union=union_in(zip(all_gpu.start_ns,all_gpu.end_ns),start,end)
        kernels=all_gpu[all_gpu.category.eq('kernel')];nonPP=kernels[~kernels.family.eq('PP_candidate')]
        same=sync[sync.pid.eq(w['pid'])&sync.tid.eq(w['tid'])]
        row={k:w[k] for k in ['iteration','rank','window_type','window_id','phase','microbatch','layer_id','execution_layer','semantic_region','start_ns','end_ns','pid','tid']}
        row.update(case=case,scope='development_posthoc_NEVER_MODEL_FIT',window_duration_ns=end-start,
            CP_only_ns=parts[1],EP_only_ns=parts[2],other_GPU_only_ns=parts[4],CP_EP_overlap_ns=parts[3],CP_other_overlap_ns=parts[5],EP_other_overlap_ns=parts[6],threeway_overlap_ns=parts[7],
            no_non_PP_GPU_activity_visible_ns=parts[0],CP_union_ns=sum(parts[m] for m in [1,3,5,7]),non_PP_GPU_activity_union_ns=sum(parts[1:]),
            no_non_PP_GPU_kernel_visible_ns=end-start-union_in(zip(nonPP.start_ns,nonPP.end_ns),start,end),no_any_GPU_activity_visible_ns=end-start-all_union,
            same_thread_sync_union_ns=union_in(zip(same.start_ns,same.end_ns),start,end),all_thread_sync_union_ns=union_in(zip(sync.start_ns,sync.end_ns),start,end))
        assert sum(row[n] for n in ['CP_only_ns','EP_only_ns','other_GPU_only_ns','CP_EP_overlap_ns','CP_other_overlap_ns','EP_other_overlap_ns','threeway_overlap_ns','no_non_PP_GPU_activity_visible_ns'])==end-start
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='evaluator';intake=plan['target_raw_trace_intake'];case='target224'
    resources=[];checks=[];all_coverage=[];all_parents=[];all_tails=[];raw_identity=[];legacy_matches=[]
    target_phase=pd.read_csv(paths['target_phase_rank_events_60_100.csv']);legacy=pd.read_csv(paths['wrapper_service_boundary_observations.csv.gz'])
    manifest=pd.read_csv(paths['target_profiler_manifest.csv']);source_windows=pd.read_csv(paths['runtime_windows.csv.gz'],dtype={'pid':'string','tid':'string'})
    input_items=json.loads((ROOT/plan['extra_input_manifest']).read_text())['files'];by_key={i.get('key'):i for i in input_items}
    for key in intake['keys']:
        item=by_key[key];iteration=item['iteration'];rank=item['rank'];path=paths[key]
        m=manifest[manifest['iter'].eq(iteration)&manifest['rank'].eq(rank)];assert len(m)==1
        assert str(path).endswith(m.relative_path.iloc[0]) and path.stat().st_size==int(m.size_bytes.iloc[0])
        gpu,cpu,runtime,resource_row,argkeys=parse_one(path,iteration,rank,case_prefix='target')
        assert resource_row['distributed_rank_metadata']['world_size']==224 and resource_row['distributed_rank_metadata']['rank']==rank
        gpu=correlate(gpu,cpu,runtime);folder=f'evaluator_only/target{iteration}_rank{rank}'
        csv(out,folder+'/GPU_device_events.csv.gz',gpu);csv(out,folder+'/CPU_operation_events.csv.gz',cpu);csv(out,folder+'/runtime_API_events.csv.gz',runtime)
        csv(out,folder+'/argument_key_counts.csv',pd.DataFrame([dict(category=c,key=k,count=v) for (c,k),v in argkeys.items()]))
        gt=target_phase[target_phase.iteration.eq(iteration)&target_phase['rank'].eq(rank)]
        windows=target_windows(cpu,gt,iteration,rank);l=legacy[legacy['case'].eq(case)&legacy.iteration.eq(iteration)&legacy['rank'].eq(rank)]
        matched=check_legacy(windows,gt,l);legacy_matches.append(matched)
        target_coverage=coverage(gpu,runtime,windows,case);all_coverage.append(target_coverage)
        # A complete linkage is checked, never assumed from the source trace.
        unique=bool(gpu.runtime_event_matches.eq(1).all());ownership=False
        if unique:
            linked=join_launches(gpu,runtime,cpu);parents,_,linked=checkpoint_ownership(cpu,windows,linked,microbatches=3)
            tails=endpoint_rows(parents,linked);parents['case']=case;parents['split']='target_development_posthoc';tails['case']=case;tails['split']='target_development_posthoc'
            all_parents.append(parents);all_tails.append(tails);csv(out,folder+'/GPU_checkpoint_ownership.csv.gz',linked);ownership=True
        dtype={'correlation':'string','external_id':'string','pid':'string','tid':'string','stream':'string'}
        sg=pd.read_csv(paths[f'source{iteration}_GPU.csv.gz'],dtype=dtype,low_memory=False);sr=pd.read_csv(paths[f'source{iteration}_runtime.csv.gz'],dtype=dtype,low_memory=False)
        sc=pd.read_csv(paths[f'source{iteration}_CPU.csv.gz'],dtype=dtype,low_memory=False);sw=source_windows[source_windows.iteration.eq(iteration)&source_windows['rank'].eq(rank)]
        source_coverage=coverage(sg,sr,sw,'source256')
        identity=source_coverage.merge(sw,on='window_id',suffixes=('','_T14'),validate='one_to_one')
        for column in ['window_duration_ns','CP_union_ns','no_non_PP_GPU_activity_visible_ns','same_thread_sync_union_ns']:
            assert identity[column].eq(identity[column+'_T14']).all()
        all_coverage.append(source_coverage)
        sl=join_launches(sg,sr,sc);sp,_,sl=checkpoint_ownership(sc,sw,sl);st=endpoint_rows(sp,sl)
        sp['case']='source256';sp['split']='source_observed_comparator_NOT_fit_in_this_run';st['case']='source256';st['split']=sp['split'].iloc[0]
        all_parents.append(sp);all_tails.append(st)
        kernels=gpu[gpu.category.eq('kernel')]
        row=dict(iteration=iteration,rank=rank,world_size=224,FB_annotations=len(gt),EP_wrappers=int(windows.window_type.eq('EP_wrapper').sum()),PP_annotations=int(windows.window_type.eq('PP_API').sum()),
            FB_boundaries_exact_ns=True,legacy_wrapper_max_difference_ns=int(matched[['start_difference_ns','end_difference_ns']].abs().max().max()),legacy_wrapper_tolerance_ns=512,
            GPU_kernels=len(kernels),GPU_device_events=len(gpu),GPU_CP_events=int(gpu.family.eq('CP_collective').sum()),
            unique_runtime_correlations=int(gpu.runtime_event_matches.eq(1).sum()),missing_runtime_correlations=int(gpu.runtime_event_matches.eq(0).sum()),ambiguous_runtime_correlations=int(gpu.runtime_event_matches.gt(1).sum()),
            GPU_starts_before_matched_runtime=int(gpu.GPU_start_minus_runtime_launch_start_ns.lt(0).sum()),checkpoint_ownership_complete=ownership,
            source_T14_coverage_reproduced_exact=True)
        checks.append(row);resource_row['peak_RSS_after_analysis_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert resource_row['peak_RSS_after_analysis_bytes']<=1024*1024*1024
        resources.append(resource_row);raw_identity.append(dict(iteration=iteration,rank=rank,path=str(path),sha256=item['sha256'],size_bytes=item['size_bytes']))
    view=pd.concat(all_coverage,ignore_index=True);csv(out,'evaluator_only/CPU_window_GPU_runtime_coverage.csv.gz',view)
    csv(out,'evaluator_only/legacy_wrapper_boundary_checks.csv',pd.concat(legacy_matches,ignore_index=True))
    parents=pd.concat(all_parents,ignore_index=True);tails=pd.concat(all_tails,ignore_index=True)
    csv(out,'evaluator_only/CPU_checkpoint_parents.csv',parents);csv(out,'evaluator_only/checkpoint_GPU_tails.csv',tails)
    columns=['window_duration_ns','CP_only_ns','EP_only_ns','other_GPU_only_ns','CP_EP_overlap_ns','CP_other_overlap_ns','EP_other_overlap_ns','threeway_overlap_ns','no_non_PP_GPU_activity_visible_ns','same_thread_sync_union_ns']
    dispatch=view[view.semantic_region.isin(['ep_dispatch_wall','ep_recompute_dispatch_wall'])]
    profile=dispatch.groupby(['case','iteration','phase','execution_layer'])[columns].mean().reset_index();csv(out,'evaluator_only/dispatch_activity_profile.csv',profile)
    phase=view[view.window_type.eq('FB_annotation')].groupby(['case','iteration','phase'])[columns].mean().reset_index();csv(out,'evaluator_only/FB_activity_profile.csv',phase)
    # Compare MB-role means separately; source has4 MB, target3. This table avoids
    # pretending ordinal MB2 has the same last/middle role in both runs.
    dispatch=dispatch.copy();dispatch['microbatch_role']=[('first' if r.microbatch==0 else 'last' if r.microbatch==(3 if r.case=='source256' else 2) else 'middle') for r in dispatch.itertuples()]
    csv(out,'evaluator_only/dispatch_by_MB_role.csv',dispatch.groupby(['case','iteration','phase','execution_layer','microbatch_role'])[columns].mean().reset_index())
    draw(out,profile,phase)
    dump(out/'raw_resource_measurement.json',resources)
    dump(out/'field_contract.json',dict(scope='target224 and same source rank16 only; target raw and all comparison tables are permanently evaluator-only development/posthoc',
        prediction='v6813 prediction and formal topology unchanged; no new fitted costs, target residuals or prediction',
        clock='integer absolute ns preserved inside each run, only durations/relative offsets compared across runs; cross-host sync unverified',
        boundaries='target F/B exactly match frozen full-rank phase file; EP local layer assignment from block checkpoint order; old wrapper offsets checked to512ns due float-epoch quantization',
        MB_comparison='Source4 and target3 microbatches; pooled per-wrapper descriptive means and first/middle/last role means separately reported, no paired ordinal transfer claim',
        GPU='Device kernel/copy/set and exact/heuristic family classification; no-event coverage is trace visibility, not necessarily physical idle or CPU work',
        CPU='Runtime synchronization is an overlapping CPU view, not additional cost; External id and GPU correlation are audited before ownership',
        raw_access='Explicit raw target capability only in diagnose/evaluator after verified prediction seal; no raw copying or broad scan',sealed_reference=plan['sealed_reference']))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    gate=all(c['FB_boundaries_exact_ns'] and c['checkpoint_ownership_complete'] and c['GPU_starts_before_matched_runtime']==0 for c in checks) and peak<=1024*1024*1024
    dump(out/'diagnostic.json',dict(status='TARGET_RAW_RUNTIME_POSTHOC_PASS',new_prediction=False,new_target_timing_read=True,used_to_fit_model=False,
        raw_trace_scanned=True,raw_files=len(resources),raw_input_bytes=sum(i['size_bytes'] for i in raw_identity),raw_input_identity=raw_identity,
        checks=checks,first_file_gate_pass=gate,peak_RSS_bytes=peak,analysis_seconds=perf_counter()-begin,all_CPU_partitions_conserve=True,
        sealed_reference=plan['sealed_reference'],formal_topology_changed=False,source_cost_updates=0,target_use='development_posthoc_NEVER_MODEL_FIT',
        next='Review first-file resource/correlation/boundary gates and local target/source visibility differences before any optional four-file expansion; no target cost fitting.'))


def draw(out,profile,phase):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns=['CP_only_ns','EP_only_ns','other_GPU_only_ns','CP_EP_overlap_ns','CP_other_overlap_ns','EP_other_overlap_ns','threeway_overlap_ns','no_non_PP_GPU_activity_visible_ns']
    labels=['CP only','EP only','Other GPU only','CP/EP overlap','CP/other overlap','EP/other overlap','Three-way overlap','No non-PP GPU event visible']
    colors=['#377eb8','#ff7f00','#4daf4a','#e41a1c','#984ea3','#a65628','#f781bf','#999999']
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,ph in zip(axes,['forward','backward']):
        rows=profile[profile.phase.eq(ph)].groupby(['case','execution_layer'])[columns].mean()/1e6
        rows.columns=labels;rows.plot.bar(stacked=True,color=colors,ax=ax,legend=False)
        ax.set_title(ph+' dispatch: source4 MB / target3 MB descriptive means');ax.set_ylabel('CPU wrapper duration partition (ms)');ax.set_xlabel('case / layer execution ordinal');ax.tick_params(axis='x',labelrotation=45)
    handles,_=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,-.06))
    fig.suptitle('Posthoc source256 vs target224 rank16 GPU visibility; no cost fitting or new prediction')
    fig.tight_layout();fig.savefig(out/'target_source_dispatch_visibility.svg',bbox_inches='tight');fig.savefig(out/'target_source_dispatch_visibility.png',dpi=160,bbox_inches='tight');plt.close(fig)

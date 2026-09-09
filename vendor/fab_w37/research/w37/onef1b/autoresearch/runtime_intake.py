"""Bounded source-only raw event intake with explicit timing and resource scope."""
from collections import Counter
import json
import resource
from time import perf_counter
import numpy as np
import pandas as pd
from smoke_worker import dump
from worker import csv
from cp_alignment import mask_partition
from cp_intake import interval_union

PP=['recv_forward','send_forward','recv_backward','send_backward','send_forward_recv_backward','send_backward_recv_forward']
EP={'ep_dispatch_wall':'FusedDispatch','ep_combine_wall':'FusedCombine','ep_recompute_dispatch_wall':'FusedDispatch',
    'ep_recompute_combine_wall':'FusedCombine','ep_combine_backward_wall':'FusedCombineBackward','ep_dispatch_backward_wall':'FusedDispatchBackward'}


def pick_id(args,keys):
    for key in keys:
        if key in args and args[key] is not None:return str(args[key])
    return ''


def kernel_family(name,pg,collective):
    lower=name.lower()
    if pg=='CONTEXT_PARALLEL_GROUP' and collective=='all_to_all':return 'CP_collective','exact_process_group_and_collective'
    if 'PIPELINE' in pg or ('mccl' in lower and 'sendrecv' in lower):return 'PP_candidate','pipeline_group_or_MCCL_SendRecv_name; residence_is_not_payload_service'
    if 'deep_ep' in lower or 'deepep' in lower:return 'EP_kernel','DeepEP_name; kernel_may_include_polling'
    if pg or 'mccl' in lower:return 'other_collective','process_group_or_MCCL_name'
    if 'flash_atten' in lower or 'flashatt' in lower:return ('attention_backward' if 'bwd' in lower or 'backward' in lower else 'attention_forward'),'kernel_name_heuristic'
    if any(s in lower for s in ['gemm','gemv','matmul']):return 'matrix_kernel','kernel_name_heuristic; no_MNK_claim'
    if 'norm' in lower:return 'normalization','kernel_name_heuristic'
    if any(s in lower for s in ['topk','index_select','scatter','permute','sort']):return 'routing','kernel_name_heuristic'
    if any(s in lower for s in ['memcpy','memset','kernelfill','copy']):return 'memory','kernel_name_heuristic'
    if any(s in lower for s in ['triton_poi','unary','binary','elementwise','pointwise']):return 'elementwise','kernel_name_heuristic'
    return 'unclassified_GPU_kernel','no_physical_compute_label_without_evidence'


def selected_csv(path,columns,iterations,ranks):
    selected=[]
    for part in pd.read_csv(path,usecols=columns,chunksize=50000):
        selected.append(part[part.iteration.isin(iterations)&part['rank'].isin(ranks)])
    return pd.concat(selected,ignore_index=True)


def parse_one(path,iteration,rank,case_prefix='source'):
    begin=perf_counter()
    with path.open() as f:doc=json.load(f)
    base=int(doc['baseTimeNanoseconds']);events=doc['traceEvents'];metadata=doc.get('distributedInfo',{}) or {}
    if 'rank' in metadata:assert int(metadata['rank'])==rank
    gpu=[];cpu=[];runtime=[];categories=Counter();argkeys=Counter()
    for index,event in enumerate(events):
        category=str(event.get('cat',''));categories[category]+=1
        if event.get('ph')!='X' or category not in ['kernel','gpu_memcpy','gpu_memset','cpu_op','user_annotation','privateuse1_runtime','privateuse1_driver']:continue
        args=event.get('args',{}) or {};argkeys.update((category,str(k)) for k in args)
        start=base+round(float(event['ts'])*1000);duration=round(float(event.get('dur',0))*1000);assert duration>=0
        name=str(event.get('name',''));row=dict(iteration=iteration,rank=rank,event_id=f'{case_prefix}{iteration}:r{rank}:event{index}',category=category,name=name,
            start_ns=start,end_ns=start+duration,duration_ns=duration,pid=str(event.get('pid','')),tid=str(event.get('tid','')),
            correlation=pick_id(args,['correlation','Correlation id','Correlation ID']),external_id=pick_id(args,['External id','external_id']))
        if category in ['kernel','gpu_memcpy','gpu_memset']:
            pg=str(args.get('Process Group Description',''));collective=str(args.get('Collective name',''))
            family,rule=kernel_family(name,pg,collective) if category=='kernel' else (category,'profiler_device_copy_or_set_event')
            row.update(family=family,classifier=rule,process_group=pg,collective=collective,stream=str(args.get('stream','')),device=str(args.get('device','')),
                dtype=str(args.get('dtype','')),group_size=args.get('Group size',None),input_nelems=args.get('In msg nelems',None))
            gpu.append(row)
        elif category in ['privateuse1_runtime','privateuse1_driver']:runtime.append(row)
        else:
            row['input_dims']=json.dumps(args.get('Input Dims',None),ensure_ascii=False,separators=(',',':'))
            cpu.append(row)
    count=len(events);del doc,events
    return pd.DataFrame(gpu),pd.DataFrame(cpu),pd.DataFrame(runtime),dict(iteration=iteration,rank=rank,source_path=str(path),input_bytes=path.stat().st_size,
        base_time_ns=base,raw_events=count,GPU_kernels=sum(r['category']=='kernel' for r in gpu),GPU_device_events=len(gpu),CPU_events=len(cpu),runtime_and_driver_events=len(runtime),parse_seconds=perf_counter()-begin,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,category_counts=dict(categories),
        distributed_rank_metadata={k:metadata[k] for k in ['rank','world_size','backend'] if k in metadata}),argkeys


def union_in(intervals,start,end):
    return sum(max(0,min(e,int(end))-max(s,int(start))) for s,e in interval_union(intervals))


def diagnose(out,paths,plan):
    assert plan['diagnostic_access']=='source_only';begin=perf_counter();intake=plan['raw_trace_intake']
    iterations=intake['allowed_iterations'];ranks=intake['allowed_source_ranks']
    phase=selected_csv(paths['source_pp_trace_events_60_100.csv'],['iteration','rank','phase','microbatch','observed_start_ns','observed_end_ns'],iterations,ranks)
    api=selected_csv(paths['source_pp_api_events_60_100.csv'],['iteration','rank','api_name','occurrence','observed_start_ns','observed_end_ns'],iterations,ranks)
    ep=selected_csv(paths['ep_source_ep_rank_anchor_events.csv'],['iteration','rank','phase','microbatch','layer_id','execution_layer','semantic_region','arrival_ns','completion_ns'],iterations,ranks)
    cp=selected_csv(paths['cp_source_cp_rank_events.csv'],['event_id','iteration','rank','phase','microbatch','layer_id','cp_slot_in_layer','start_ns','end_ns','pair_step_complete'],iterations,ranks)
    resources=[];windows=[];scope=[];matched_cp=[];family_tables=[]
    manifest=pd.read_csv(paths['source_profiler_trace_manifest.csv'])
    # Plan keys resolve only preflight-pinned raw paths; do not discover or open
    # adjacent files and do not import legacy raw scanners.
    for key in intake['keys']:
        path=paths[key];selected=manifest[manifest.relative_path.map(lambda v:str(path).endswith(v))]
        assert len(selected)==1;meta=selected.iloc[0];iteration=int(meta['iter']);rank=int(meta['rank'])
        assert iteration in iterations and rank in ranks and int(meta.size_bytes)==path.stat().st_size
        gpu,cpu,runtime,resource_row,argkeys=parse_one(path,iteration,rank)
        folder=f'source{iteration}_rank{rank}';csv(out,folder+'/argument_key_counts.csv',pd.DataFrame([dict(category=k[0],argument_key=k[1],events=n) for k,n in argkeys.items()]))
        xphase=phase[phase.iteration.eq(iteration)&phase['rank'].eq(rank)]
        xapi=api[api.iteration.eq(iteration)&api['rank'].eq(rank)]
        xep=ep[ep.iteration.eq(iteration)&ep['rank'].eq(rank)]
        xcp=cp[cp.iteration.eq(iteration)&cp['rank'].eq(rank)]
        annotations=cpu[cpu.category.eq('user_annotation')].copy();local=[]
        for ph in ['forward','backward']:
            actual=annotations[annotations.name.eq(ph+'_step')].sort_values('start_ns');expected=xphase[xphase.phase.eq(ph)].sort_values('microbatch')
            assert len(actual)==len(expected)==4
            assert actual.start_ns.to_list()==expected.observed_start_ns.to_list() and actual.end_ns.to_list()==expected.observed_end_ns.to_list()
            for (r,e) in zip(actual.itertuples(),expected.itertuples()):local.append(dict(window_type='FB_annotation',window_id=r.event_id,phase=ph,microbatch=e.microbatch,layer_id=-1,execution_layer=-1,semantic_region=ph,start_ns=r.start_ns,end_ns=r.end_ns,pid=r.pid,tid=r.tid))
        for name in PP:
            actual=annotations[annotations.name.eq(name)].sort_values('start_ns');expected=xapi[xapi.api_name.eq(name)].sort_values('occurrence')
            assert len(actual)==len(expected)
            assert actual.start_ns.to_list()==expected.observed_start_ns.to_list() and actual.end_ns.to_list()==expected.observed_end_ns.to_list()
            for r in actual.itertuples():local.append(dict(window_type='PP_API',window_id=r.event_id,phase='API',microbatch=-1,layer_id=-1,execution_layer=-1,semantic_region=name,start_ns=r.start_ns,end_ns=r.end_ns,pid=r.pid,tid=r.tid))
        actual=cpu[cpu.category.eq('cpu_op')&cpu.name.isin(set(EP.values()))]
        expected=xep.rename(columns={'arrival_ns':'start_ns','completion_ns':'end_ns'}).copy();expected['name']=expected.semantic_region.map(EP)
        matched=expected.merge(actual,on=['iteration','rank','name','start_ns','end_ns'],validate='one_to_one')
        assert len(matched)==len(expected)==len(actual)
        for r in matched.itertuples():local.append(dict(window_type='EP_wrapper',window_id=r.event_id,phase=r.phase,microbatch=r.microbatch,layer_id=r.layer_id,execution_layer=r.execution_layer,
            semantic_region=r.semantic_region,start_ns=r.start_ns,end_ns=r.end_ns,pid=r.pid,tid=r.tid))
        cpgpu=gpu[gpu.family.eq('CP_collective')]
        cpmatch=xcp.rename(columns={'event_id':'frozen_CP_event_id'}).merge(cpgpu[['event_id','start_ns','end_ns']],on=['start_ns','end_ns'],validate='one_to_one')
        assert len(cpmatch)==len(xcp)==len(cpgpu);matched_cp.append(cpmatch)
        csv(out,folder+'/matched_EP_CPU_anchors.csv',matched)
        # Correlation is audited as a trace linkage; one runtime launch may
        # own many GPU kernels. Empty or reused IDs are never forced to match.
        launch=runtime[runtime.name.str.contains('launch',case=False)&runtime.correlation.ne('')]
        def compact_runtime(frame):
            grouped=frame.groupby('correlation').agg(matches=('event_id','size'),event_id=('event_id','first'),name=('name','first'),start_ns=('start_ns','first'),end_ns=('end_ns','first'))
            return {r.Index:(r.matches,r.event_id,r.name,int(r.start_ns),int(r.end_ns)) for r in grouped.itertuples()}
        groups=compact_runtime(launch);all_runtime=compact_runtime(runtime[runtime.correlation.ne('')])
        owner_table=cpu[cpu.category.eq('cpu_op')&cpu.external_id.ne('')].groupby('external_id').agg(matches=('event_id','size'),name=('name','first'))
        owners={r.Index:(r.matches,r.name) for r in owner_table.itertuples()}
        linked=[]
        for r in gpu.itertuples():
            g=(groups if r.category=='kernel' else all_runtime).get(r.correlation);owner=owners.get(r.external_id)
            n=0 if g is None else g[0];no=0 if owner is None else owner[0]
            linked.append(dict(event_id=r.event_id,runtime_launch_matches=n if r.category=='kernel' else 0,runtime_event_matches=n,CPU_external_id_matches=no,
                unique_runtime_event_id=g[1] if n==1 else '',unique_runtime_name=g[2] if n==1 else '',
                GPU_start_minus_runtime_launch_start_ns=int(r.start_ns)-g[3] if n==1 else None,
                GPU_start_minus_runtime_launch_end_ns=int(r.start_ns)-g[4] if n==1 else None,
                unique_CPU_owner_name=owner[1] if no==1 else ''))
        gpu=gpu.merge(pd.DataFrame(linked),on='event_id',validate='one_to_one')
        csv(out,folder+'/GPU_device_events.csv.gz',gpu);csv(out,folder+'/GPU_kernel_events.csv.gz',gpu[gpu.category.eq('kernel')])
        csv(out,folder+'/CPU_operation_events.csv.gz',cpu);csv(out,folder+'/runtime_API_events.csv.gz',runtime)
        main=gpu[~gpu.family.eq('PP_candidate')].copy();main['activity_category']=np.where(main.family.eq('CP_collective'),0,np.where(main.family.eq('EP_kernel'),1,2))
        pp=gpu[gpu.family.eq('PP_candidate')];sync=runtime[runtime.name.isin(['musaStreamSynchronize','musaDeviceSynchronize','musaEventSynchronize'])]
        for window in local:
            start,end=int(window['start_ns']),int(window['end_ns'])
            g=main[main.start_ns.lt(end)&main.end_ns.gt(start)];parts=mask_partition(start,end,g[['start_ns','end_ns','activity_category']].itertuples(index=False,name=None))
            all_gpu=gpu[gpu.start_ns.lt(end)&gpu.end_ns.gt(start)];all_union=union_in(zip(all_gpu.start_ns,all_gpu.end_ns),start,end)
            kernels=all_gpu[all_gpu.category.eq('kernel')];kernel_union=union_in(zip(kernels.start_ns,kernels.end_ns),start,end)
            non_pp_kernels=kernels[~kernels.family.eq('PP_candidate')];non_pp_kernel_union=union_in(zip(non_pp_kernels.start_ns,non_pp_kernels.end_ns),start,end)
            pp_union=union_in(zip(pp.start_ns,pp.end_ns),start,end)
            visible=sum(parts[1:]);same_thread=sync[sync.pid.eq(window['pid'])&sync.tid.eq(window['tid'])]
            row=dict(iteration=iteration,rank=rank,**window,window_duration_ns=end-start,CP_only_ns=parts[1],EP_only_ns=parts[2],other_GPU_only_ns=parts[4],
                CP_EP_overlap_ns=parts[3],CP_other_overlap_ns=parts[5],EP_other_overlap_ns=parts[6],threeway_overlap_ns=parts[7],
                no_non_PP_GPU_activity_visible_ns=parts[0],no_non_PP_GPU_kernel_visible_ns=end-start-non_pp_kernel_union,
                CP_union_ns=sum(parts[m] for m in [1,3,5,7]),non_PP_GPU_activity_union_ns=visible,
                PP_candidate_union_ns=pp_union,no_non_PP_but_PP_candidate_visible_ns=all_union-visible,
                no_any_GPU_kernel_visible_ns=end-start-kernel_union,no_any_GPU_activity_visible_ns=end-start-all_union,
                all_thread_sync_union_ns=union_in(zip(sync.start_ns,sync.end_ns),start,end),same_thread_sync_union_ns=union_in(zip(same_thread.start_ns,same_thread.end_ns),start,end))
            assert visible+parts[0]==end-start and all_union<=end-start and all_union>=visible
            assert row['same_thread_sync_union_ns']<=row['all_thread_sync_union_ns']<=end-start
            windows.append(row)
        f=gpu.groupby('family').agg(kernels=('event_id','size'),duration_sum_ns=('duration_ns','sum'),correlated_launches=('runtime_launch_matches',lambda v:int(v.eq(1).sum())),
            external_owner_matches=('CPU_external_id_matches',lambda v:int(v.eq(1).sum()))).reset_index();f['iteration']=iteration;f['rank']=rank;family_tables.append(f)
        kernels=gpu[gpu.category.eq('kernel')];copy_set=gpu[~gpu.category.eq('kernel')]
        scope.append(dict(iteration=iteration,rank=rank,matched_FB_annotations=len(xphase),matched_PP_annotations=len(xapi),matched_EP_wrappers=len(matched),matched_CP_kernels=len(cpmatch),
            all_boundaries_exact_ns=True,GPU_kernels=len(kernels),GPU_device_events=len(gpu),unique_launch_correlation=int(kernels.runtime_launch_matches.eq(1).sum()),
            missing_launch_correlation=int(kernels.runtime_launch_matches.eq(0).sum()),ambiguous_launch_correlation=int(kernels.runtime_launch_matches.gt(1).sum()),
            device_copy_set_events=len(copy_set),device_copy_set_unique_runtime_correlation=int(copy_set.runtime_event_matches.eq(1).sum()),
            GPU_starts_before_unique_runtime_start=int(gpu.GPU_start_minus_runtime_launch_start_ns.lt(0).sum()),
            unique_CPU_external_id_owner=int(gpu.CPU_external_id_matches.eq(1).sum())))
        resource_row['peak_RSS_after_tables_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert resource_row['peak_RSS_after_tables_bytes']<=1024*1024*1024,resource_row
        resources.append(resource_row)
        del gpu,cpu,runtime,main,pp,launch,groups,all_runtime,owners,owner_table,linked
    windows=pd.DataFrame(windows);csv(out,'source_CPU_window_GPU_runtime_coverage.csv.gz',windows)
    csv(out,'source_CP_kernel_identity.csv',pd.concat(matched_cp,ignore_index=True));csv(out,'source_kernel_family_counts.csv',pd.concat(family_tables,ignore_index=True))
    csv(out,'source_boundary_and_correlation_checks.csv',pd.DataFrame(scope));dump(out/'source_raw_resource_measurement.json',resources)
    view=windows[windows.window_type.eq('EP_wrapper')&windows.semantic_region.isin(['ep_dispatch_wall','ep_recompute_dispatch_wall'])]
    metrics=['CP_only_ns','EP_only_ns','other_GPU_only_ns','CP_EP_overlap_ns','CP_other_overlap_ns','EP_other_overlap_ns','threeway_overlap_ns','no_non_PP_GPU_activity_visible_ns']
    profile=view.groupby(['iteration','phase','execution_layer'])[metrics+['window_duration_ns','same_thread_sync_union_ns']].mean().reset_index()
    csv(out,'source_dispatch_visible_activity_profile.csv',profile)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,ph in zip(axes,['forward','backward']):
        z=profile[profile.phase.eq(ph)].groupby('execution_layer')[metrics].mean()/1e6
        z.columns=['CP only','EP only','Other device activity only','CP/EP overlap','CP/other overlap','EP/other overlap','Three-way overlap','No non-PP device event visible']
        z.plot.bar(stacked=True,ax=ax);ax.set_xlabel('Layer execution ordinal');ax.set_ylabel('CPU dispatch wrapper coverage (ms)');ax.set_title(ph);ax.tick_params(axis='x',labelrotation=0);ax.legend(fontsize=6)
    fig.suptitle('Source rank16 sample; PP resident candidates separate; kernel wall may include waiting')
    fig.tight_layout();fig.savefig(out/'source_dispatch_GPU_coverage.svg',bbox_inches='tight');fig.savefig(out/'source_dispatch_GPU_coverage.png',dpi=160,bbox_inches='tight');plt.close(fig)
    dump(out/'field_contract.json',dict(time='baseTimeNanoseconds + round(ts_us*1000), end=start+round(dur_us*1000); CPU/GPU domains retained',
        CP='exact process-group+collective metadata and exact frozen event timestamp identity',EP='DeepEP name heuristic, possible device polling; not independent network FCT',
        other_GPU='includes name-inferred operators, unclassified kernels, gpu_memcpy and gpu_memset; no per-kernel FLOPs or complete operation dependency claim',
        PP='pipeline metadata or MCCL_SendRecv name; long kernel residence shown separately from non-PP GPU activity, not removed from raw event table',
        runtime='runtime plus driver events; all-thread and same-pid/tid synchronize union are separate views, never additional costs',correlation='exact correlation/external-ID match counts audited; missing/reused keys not forced; CPU owner may be broad CheckpointFunction; copy/set matches runtime API rather than a kernel launch',
        coverage='source rank16 only, no all-rank or target runtime claim',cost_admission='intake observations only, no costs fitted or promoted'))
    dump(out/'diagnostic.json',dict(status='SOURCE_RAW_RUNTIME_INTAKE_PASS',new_prediction=False,new_target_timing_read=False,used_to_fit_model=False,
        raw_trace_scanned=True,raw_files=len(resources),raw_input_bytes=sum(r['input_bytes'] for r in resources),raw_source_iterations=iterations,raw_source_ranks=ranks,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,analysis_seconds=perf_counter()-begin,checks=scope,
        all_CPU_window_partitions_conserve=True,source_only_no_raw_copy=True,formal_topology_changed=False,
        next='Review sample boundary/correlation/resource gates before expanding to the remaining registered source files; no CP-uncovered time relabeling as compute or CPU wait without evidence.'))

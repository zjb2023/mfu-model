"""Source device queue audit and explicitly CPU-conditioned local predictions."""
import json
import resource
from time import perf_counter
import numpy as np
import pandas as pd
from smoke_worker import dump,sha
from worker import csv
from runtime_pending import join_launches,split_name
from checkpoint_tail import containing_index
from runtime_intake import union_in

DTYPE={c:'string' for c in ['correlation','external_id','pid','tid','stream','device']}
STREAM=['iteration','rank','device','stream']
FIT=[85,90]


def prepare(gpu,runtime,cpu,windows):
    d=join_launches(gpu,runtime,cpu)
    assert d.stream.notna().all() and d.device.notna().all()
    fb=windows[windows.window_type.eq('FB_annotation')].reset_index(drop=True)
    ep=windows[windows.window_type.eq('EP_wrapper')].reset_index(drop=True)
    d['FB_index']=containing_index(d,fb,'runtime_start_ns',['iteration','rank'])
    for c in ['phase','microbatch','window_id']:
        d['launch_FB_'+c]=d.FB_index.map(fb[c]).fillna(-1 if c=='microbatch' else 'outside_FB')
    point=d.rename(columns={'pid':'GPU_pid','tid':'GPU_tid','runtime_pid':'pid','runtime_tid':'tid'})
    d['EP_index']=containing_index(point,ep,'runtime_start_ns',['pid','tid'])
    for c in ['semantic_region','layer_id','execution_layer','window_id']:
        d['launch_EP_'+c]=d.EP_index.map(ep[c]).fillna(-1 if c.endswith('layer') or c=='layer_id' else 'outside_EP')
    d['split']=[split_name(i) for i in d.iteration]
    # Pageable copies can start before API return. Category, not validation
    # device time, chooses the release proxy. Neither is a global CPU model.
    d['release_proxy_kind']=np.where(d.category.eq('kernel'),'observed_kernel_API_end','observed_copy_set_API_start')
    d['release_proxy_ns']=np.where(d.category.eq('kernel'),d.runtime_end_ns,d.runtime_start_ns).astype('int64')
    return d


def stream_audit(d):
    rows=[];ordered=[];edges=[]
    for key,group in d.groupby(STREAM):
        g=group.sort_values(['runtime_start_ns','runtime_end_ns','event_id']).copy()
        assert g.event_id.is_unique
        previous_end=g.end_ns.astype('Int64').shift()
        previous_start=g.start_ns.astype('Int64').shift()
        previous_API_end=g.runtime_end_ns.astype('Int64').shift()
        previous_ids=g.event_id.shift().fillna('')
        g['previous_CPU_order_GPU_id']=previous_ids
        g['previous_observed_GPU_end_ns']=previous_end
        ready=np.maximum(g.release_proxy_ns.to_numpy(dtype='int64'),previous_end.fillna(g.release_proxy_ns).to_numpy(dtype='int64'))
        g['observed_enqueue_remainder_ns']=g.start_ns-ready
        g['observed_API_latency_ns']=g.start_ns-g.release_proxy_ns
        info=dict(zip(STREAM,key));ties=g.runtime_start_ns.duplicated(keep=False)
        overlaps=g.runtime_start_ns.lt(previous_API_end).fillna(False)
        shared=g.unique_runtime_event_id.duplicated(keep=False)
        info.update(device_events=len(g),unique_runtime_events=g.unique_runtime_event_id.nunique(),CPU_threads=g.runtime_tid.nunique(),
            tied_runtime_starts=int(ties.sum()),overlapping_runtime_submissions=int(overlaps.sum()),shared_runtime_device_events=int(shared.sum()),
            observed_GPU_start_order_inversions=int(g.start_ns.lt(previous_start).fillna(False).sum()),
            observed_same_stream_GPU_overlaps=int(g.start_ns.lt(previous_end).fillna(False).sum()),
            GPU_starts_before_API_start=int(g.start_ns.lt(g.runtime_start_ns).sum()),GPU_starts_before_API_end=int(g.start_ns.lt(g.runtime_end_ns).sum()),
            negative_release_proxy_latency=int(g.observed_API_latency_ns.lt(0).sum()),negative_enqueue_remainder=int(g.observed_enqueue_remainder_ns.lt(0).sum()))
        info['CPU_order_unambiguous']=not(ties.any() or overlaps.any() or shared.any())
        rows.append(info);ordered.append(g)
        for r in g.itertuples():
            if not r.previous_CPU_order_GPU_id:continue
            edges.append(dict(**dict(zip(STREAM,key)),src=r.previous_CPU_order_GPU_id,dst=r.event_id,
                edge_type='candidate_same_stream_program_order',order_source='unique nonoverlapping CPU runtime submissions; never GPU timing sort',
                source_semantics='MUSA rc4.3 programming guide3.2.8.5; deployed flags/revision unverified',CPU_order_unambiguous=info['CPU_order_unambiguous']))
    return pd.DataFrame(rows),pd.concat(ordered,ignore_index=True),pd.DataFrame(edges)


def graph_audit(d,runtime,windows):
    graph=runtime[runtime.name.eq('musaGraphLaunch')].copy()
    ep=windows[windows.window_type.eq('EP_wrapper')].reset_index(drop=True)
    graph['EP_index']=containing_index(graph,ep,'start_ns',['pid','tid'])
    rows=[]
    for g in graph.itertuples():
        hits=d[d.unique_runtime_event_id.eq(g.event_id)]
        row=dict(iteration=g.iteration,rank=g.rank,split=split_name(g.iteration),graph_runtime_id=g.event_id,
            graph_API_start_ns=g.start_ns,graph_API_end_ns=g.end_ns,graph_API_duration_ns=g.duration_ns,
            directly_correlated_device_events=len(hits),CPU_EP_containment=g.EP_index>=0)
        if g.EP_index>=0:
            w=ep.iloc[g.EP_index]
            row.update(phase=w.phase,microbatch=int(w.microbatch),layer_id=int(w.layer_id),semantic_region=w.semantic_region,window_id=w.window_id)
            same=runtime[runtime.pid.eq(g.pid)&runtime.tid.eq(g.tid)&runtime.external_id.eq(g.external_id)&runtime.name.eq('musaDeviceSynchronize')
                &runtime.start_ns.ge(g.end_ns)&runtime.end_ns.le(w.end_ns)].sort_values('start_ns')
            row['post_graph_same_thread_owner_device_syncs']=len(same)
            if len(same):
                sync=same.iloc[0];start,end=int(g.start_ns),int(sync.end_ns)
                row.update(first_device_sync_id=sync.event_id,first_device_sync_start_ns=int(sync.start_ns),first_device_sync_end_ns=end,
                    graph_to_first_device_sync_return_ns=end-start)
                nonpp=d[~d.family.eq('PP_candidate')]
                visible=union_in(zip(nonpp.start_ns,nonpp.end_ns),start,end)
                row['nonPP_device_visible_in_graph_sync_span_ns']=visible
                row['no_nonPP_device_event_in_graph_sync_span_ns']=end-start-visible
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only'
    windows=pd.read_csv(paths['runtime_windows.csv.gz'],dtype=DTYPE,low_memory=False)
    data=[];audits=[];graphs=[];edges=[]
    for it in [85,90,95,100]:
        gpu=pd.read_csv(paths[f'source{it}_GPU.csv.gz'],dtype=DTYPE,low_memory=False)
        runtime=pd.read_csv(paths[f'source{it}_runtime.csv.gz'],dtype=DTYPE,low_memory=False)
        cpu=pd.read_csv(paths[f'source{it}_CPU.csv.gz'],dtype=DTYPE,low_memory=False)
        w=windows[windows.iteration.eq(it)]
        prepared=prepare(gpu,runtime,cpu,w);audit,ordered,edge=stream_audit(prepared)
        data.append(ordered);audits.append(audit);edges.append(edge);graphs.append(graph_audit(prepared,runtime,w))
    data=pd.concat(data,ignore_index=True);audits=pd.concat(audits,ignore_index=True);graphs=pd.concat(graphs,ignore_index=True);edges=pd.concat(edges,ignore_index=True)
    csv(out,'source_stream_order_audit.csv',audits);csv(out,'source_device_queue_observations.csv.gz',data)
    csv(out,'source_candidate_stream_edges.csv.gz',edges);csv(out,'source_graph_launch_visibility.csv',graphs)
    negative=data[data.start_ns.lt(data.runtime_end_ns)]
    csv(out,'source_GPU_before_API_return.csv.gz',negative)
    csv(out,'source_API_release_boundary_summary.csv',data.groupby(['iteration','split','category','unique_runtime_name']).agg(
        device_events=('event_id','size'),negative_API_end_latency=('GPU_start_minus_runtime_launch_end_ns',lambda x:int(x.lt(0).sum())),
        minimum_API_end_latency_ns=('GPU_start_minus_runtime_launch_end_ns','min'),minimum_release_proxy_latency_ns=('observed_API_latency_ns','min')).reset_index())
    graph_summary=graphs.groupby(['split','phase','semantic_region']).agg(calls=('graph_runtime_id','size'),
        correlated_device_events=('directly_correlated_device_events','sum'),mean_graph_sync_span_ns=('graph_to_first_device_sync_return_ns','mean'),
        mean_no_nonPP_device_event_ns=('no_nonPP_device_event_in_graph_sync_span_ns','mean')).reset_index()
    csv(out,'source_graph_launch_visibility_summary.csv',graph_summary)
    gate=bool(audits.CPU_order_unambiguous.all() and audits.observed_GPU_start_order_inversions.eq(0).all() and
        audits.observed_same_stream_GPU_overlaps.eq(0).all() and audits.negative_release_proxy_latency.eq(0).all())
    dump(out/'field_contract.json',dict(time='integer absolute ns; half-open intervals; nullable previous GPU end stays Int64',
        order='Sort unique nonoverlapping runtime API starts, not GPU execution timestamps. All edge evidence remains candidate; formal topology unchanged.',
        CPU_release='Kernel API end, copy/set API start, selected by category before scoring. Observed CPU endpoints are conditional inputs, not a global CPU model.',
        graph='Graph API -> first same-thread same-External-id DeviceSynchronize return is a descriptive bracket only, not pure graph/device service or causal completion proof.',
        missing='No graph executable/stream/event handles in cached runtime schema; cannot assign hidden graph nodes or cross-stream event edges.',
        source_fit=FIT,source_incremental_validation=[95,100],target='SHA only; no target costs or timing parse',semantics=plan.get('public_semantics',[])))
    local=None
    if plan.get('fit_local_device_queue'):
        assert gate,'visible queue structural prerequisites failed'
        from device_queue_model import fit_predict_score
        local=fit_predict_score(out,data,windows,plan)
    dump(out/'diagnostic.json',dict(status='SOURCE_DEVICE_QUEUE_LOCAL_PREDICTION_PASS' if local else 'SOURCE_DEVICE_QUEUE_AUDIT_PASS',
        new_prediction=bool(local),used_to_fit_model=bool(local),new_target_timing_read=False,local_prediction=local,
        raw_trace_scanned=False,source_iterations=[85,90,95,100],source_ranks=[16],device_events=len(data),stream_iterations=len(audits),candidate_edges=len(edges),
        visible_queue_structural_gate=gate,CPU_API_end_not_release_for_all_events=True,device_events_before_API_return=len(negative),
        graph_launch_calls=len(graphs),graph_directly_correlated_device_events=int(graphs.directly_correlated_device_events.sum()),
        graph_context_summary=graph_summary.to_dict('records'),formal_topology_changed=False,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,analysis_seconds=perf_counter()-begin,
        next='If source visible queue gate passes, fit only85/90 and seal local CPU-conditioned95/100 predictions; graph visibility gap remains outside causal device graph, not idle cost.'))

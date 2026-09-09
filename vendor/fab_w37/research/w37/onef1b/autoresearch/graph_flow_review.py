"""Resolve cached source flow endpoint identities; no new device execution edges."""
import json
import resource
from time import perf_counter
import pandas as pd
from smoke_worker import dump
from worker import csv

DTYPE={k:'string' for k in ['pid','tid','correlation','external_id','stream','device','flow_id_json','flow_id2_json','scope_json']}


def resolve_flows(gpu,runtime,flows):
    assert gpu.correlation.is_unique and runtime.correlation.is_unique
    assert flows.flow_id2_json.eq('null').all() and flows.scope_json.eq('null').all()
    by_runtime=runtime.set_index('correlation');by_GPU=gpu.set_index('correlation')
    rows=[];edges=[]
    for key,group in flows.groupby('flow_id_json',sort=False):
        decoded=json.loads(key);assert isinstance(decoded,int) and str(decoded) in by_runtime.index
        r=by_runtime.loc[str(decoded)];s=group[group.ph.eq('s')];f=group[group.ph.eq('f')]
        assert len(s)<=1 and len(f)<=1 and len(s)+len(f)==len(group)
        row=dict(correlation=str(decoded),runtime_id=r.event_id,runtime_name=r['name'],runtime_pid=r.pid,runtime_tid=r.tid,
            runtime_start_ns=int(r.start_ns),runtime_end_ns=int(r.end_ns),source_points=len(s),finish_points=len(f),
            device_matches=int(str(decoded) in by_GPU.index))
        if len(s):
            point=s.iloc[0];assert point.pid==r.pid and point.tid==r.tid and point.start_ns==r.start_ns
        if str(decoded) in by_GPU.index:
            device=by_GPU.loc[str(decoded)];assert len(s)==len(f)==1
            point=f.iloc[0];assert point.pid==device.pid and point.tid==device.tid and point.start_ns==device.start_ns
            kind='CPU_API_start_to_device_start'
            row.update(device_id=device.event_id,device_family=device.family,device_start_ns=str(int(device.start_ns)),device_end_ns=str(int(device.end_ns)))
            edges.append(dict(src=r.event_id,dst=device.event_id,edge_type='profiler_correlation_only',
                evidence=f'flow id={key}: s equalsCPUAPIstart(pid,tid,ns); f equalsGPUeventstart(pid,tid,ns)',
                relation='submission association, not complete device dependency or calibrated cost'))
        else:
            assert len(group)==1
            point=group.iloc[0];assert point.pid==r.pid and point.tid==r.tid and point.start_ns==r.start_ns
            kind='CPU_API_start_only_'+point.ph
        row['flow_kind']=kind;rows.append(row)
    assert len(rows)==len(runtime) and len(edges)==len(gpu)
    return pd.DataFrame(rows),pd.DataFrame(edges)


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only'
    gpu=pd.read_csv(paths['source85_GPU.csv.gz'],dtype=DTYPE,low_memory=False)
    runtime=pd.read_csv(paths['source85_runtime.csv.gz'],dtype=DTYPE,low_memory=False)
    cpu=pd.read_csv(paths['source85_CPU.csv.gz'],dtype=DTYPE,low_memory=False)
    flows=pd.read_csv(paths['source_extra_flow.csv.gz'],dtype=DTYPE,keep_default_na=False,low_memory=False)
    annotations=pd.read_csv(paths['source_extra_annotations.csv.gz'],dtype=DTYPE,keep_default_na=False,low_memory=False)
    resolved,edges=resolve_flows(gpu,runtime,flows)
    csv(out,'source_flow_endpoint_identities.csv.gz',resolved);csv(out,'source_CPU_GPU_correlation_edges.csv.gz',edges)
    profile=resolved.groupby(['runtime_name','flow_kind']).size().rename('runtime_calls').reset_index()
    csv(out,'source_flow_endpoint_by_API.csv',profile)
    graph=resolved[resolved.runtime_name.eq('musaGraphLaunch')]
    assert len(graph)==96 and graph.flow_kind.eq('CPU_API_start_only_s').all() and graph.device_matches.eq(0).all()
    csv(out,'source_GraphLaunch_without_device_endpoint.csv',graph)
    # External id associates annotations with CPU scopes; it does not turn
    # an annotation's elapsed range into traced execution on the device.
    owners=cpu[cpu.external_id.notna()&~cpu.external_id.duplicated(keep=False)].set_index('external_id')
    annotation_rows=[]
    for a in annotations.itertuples():
        assert a.external_id in owners.index
        owner=owners.loc[a.external_id];assert a.name==owner['name']
        annotation_rows.append(dict(annotation_id=a.event_id,annotation_name=a.name,CPU_scope_id=owner.event_id,
            GPU_annotation_start_ns=a.start_ns,GPU_annotation_end_ns=a.end_ns,CPU_scope_start_ns=int(owner.start_ns),CPU_scope_end_ns=int(owner.end_ns),
            GPU_annotation_pid=a.pid,GPU_annotation_tid=a.tid,external_id=a.external_id,
            relation='same External id and name; scope range only, not device-busy interval'))
    mapped=pd.DataFrame(annotation_rows);csv(out,'source_GPU_annotation_CPU_scope_mapping.csv',mapped)
    nodes=[]
    for kind,frame in [('runtime_API',runtime),('device_event',gpu),('GPU_annotation',annotations)]:
        subset=frame[['event_id','name','category','start_ns','end_ns','pid','tid']].copy();subset['node_kind']=kind;nodes.append(subset)
    csv(out,'source_flow_observation_nodes.csv.gz',pd.concat(nodes,ignore_index=True))
    draw(out,profile,gpu,runtime,cpu,annotations)
    dump(out/'field_contract.json',dict(time='Exact int64 timestamp equality within single source85 clock; nullable device endpoints stored as integer strings',
        paired='All10253 pairedac2g IDs identifyCPUAPIstart -> deviceeventstart with exactpid/tid/ns. This verifies existing correlations, not cross-stream dependencies.',
        singletons='Both s-only and f-only points are atCPUAPIstart. In this trace, f alone is not GPU completion or even GPU presence.',
        graph='All96GraphLaunch IDs have CPU s only. Omittedflows/annotations do not supply its graph-node execution endpoints.',
        annotations='All280 map toCPU scopes byExternal id+name; their ranges remain separate from traced kernel/copy/set busy union.',
        limits='No raw reads, no timing fit or new prediction, no accepted graph/event wait edge. Source85/rank16 only; do not assert all-rank profiler coverage.',
        scope='source-only evidence; target timingSHA only'))
    dump(out/'diagnostic.json',dict(status='SOURCE_GRAPH_FLOW_ENDPOINT_REVIEW_PASS',new_prediction=False,used_to_fit_model=False,new_target_timing_read=False,
        raw_trace_scanned=False,source_iterations=[85],source_ranks=[16],runtime_events=len(runtime),GPU_device_events=len(gpu),flow_points=len(flows),
        paired_CPU_GPU_correlations=len(edges),CPU_start_only_s=int(resolved.flow_kind.eq('CPU_API_start_only_s').sum()),
        CPU_start_only_f=int(resolved.flow_kind.eq('CPU_API_start_only_f').sum()),GraphLaunch_CPU_start_only=len(graph),
        GPU_annotations_with_exact_CPU_scope=len(mapped),all_endpoint_pid_tid_ns_exact=True,new_graph_device_endpoints=0,
        new_cross_stream_dependencies=0,formal_topology_changed=False,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        analysis_seconds=perf_counter()-begin,next='Do not expandraw merely to repeat an instrumentation gap. Seek source code or independent graph/runtime release evidence before assigning hidden activity costs.'))


def draw(out,profile,gpu,runtime,cpu,annotations):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cp_intake import interval_union
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    groups=profile.groupby('flow_kind').runtime_calls.sum()
    kinds=['CPU_API_start_to_device_start','CPU_API_start_only_s','CPU_API_start_only_f']
    labels=['CPU todevice\n10,253','CPU s only\n1,839','CPU f only\n8,942']
    axes[0].bar(range(3),[groups[k] for k in kinds],color=['#0072b2','#e69f00','#999999'])
    axes[0].set_xticks(range(3),labels);axes[0].set_ylabel('Runtime correlations');axes[0].set_title('Every paired flow matches an already extracted device event')
    graph=runtime[runtime.name.eq('musaGraphLaunch')].sort_values('start_ns').iloc[0]
    sync=runtime[runtime.pid.eq(graph.pid)&runtime.tid.eq(graph.tid)&runtime.external_id.eq(graph.external_id)&runtime.name.eq('musaDeviceSynchronize')&runtime.start_ns.ge(graph.end_ns)].sort_values('start_ns').iloc[0]
    start=int(graph.start_ns);end=int(sync.end_ns);ax=axes[1]
    ax.broken_barh([(0,(int(graph.end_ns)-start)/1e6)],(.1,.6),facecolors='#e69f00')
    ax.broken_barh([((int(sync.start_ns)-start)/1e6,(end-int(sync.start_ns))/1e6)],(1.1,.6),facecolors='#cc79a7')
    dev=gpu[(gpu.start_ns<end)&(gpu.end_ns>start)]
    if len(dev):ax.broken_barh([((max(start,int(r.start_ns))-start)/1e6,(min(end,int(r.end_ns))-max(start,int(r.start_ns)))/1e6) for r in dev.itertuples()],(2.1,.6),facecolors='#0072b2')
    ann=annotations[(annotations.start_ns<end)&(annotations.end_ns>start)]
    if len(ann):
        spans=interval_union((max(start,int(r.start_ns)),min(end,int(r.end_ns))) for r in ann.itertuples())
        ax.broken_barh([((s-start)/1e6,(e-s)/1e6) for s,e in spans],(3.1,.6),facecolors='none',edgecolors='#555555',hatch='//')
    ax.set_yticks([.4,1.4,2.4,3.4],['CPU GraphLaunch','CPU DeviceSynchronize','Visible device events','Overlapping GPU scopes'])
    ax.set_xlabel('ms since firstGraphLaunch APIstart');ax.set_title('An annotation can overlap a gap without proving device work')
    ax.set_ylim(-.2,4);ax.set_xlim(-.02,(end-start)/1e6*1.05)
    fig.suptitle('Source85/rank16: graph hasCPU flowstart only; omitted categories add no device endpoint')
    fig.text(.5,.02,'Paired flow is launch association, not a complete dependency graph. Hatched scope is not added to device busy time.\n'
        'Single source trace, no cost fitting or224 prediction.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(out/'source_graph_flow_coverage.svg');fig.savefig(out/'source_graph_flow_coverage.png',dpi=160);plt.close(fig)

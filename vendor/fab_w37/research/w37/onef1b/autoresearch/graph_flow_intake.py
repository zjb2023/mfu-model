"""One registered source trace: omitted GPU annotations and flow identity evidence."""
from collections import Counter,defaultdict
import json
import resource
from time import perf_counter
import pandas as pd
from smoke_worker import dump,sha
from worker import csv
from runtime_intake import pick_id

BASE=['iteration','rank','event_id','category','name','start_ns','end_ns','duration_ns','pid','tid','correlation','external_id']
RAW_GPU={'kernel','gpu_memcpy','gpu_memset'}
RAW_RUNTIME={'privateuse1_runtime','privateuse1_driver'}
RAW_CPU={'cpu_op','user_annotation'}


def compact_json(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=True)


def read_registered(path,iteration,rank):
    begin=perf_counter()
    with path.open() as f:doc=json.load(f)
    base=int(doc['baseTimeNanoseconds']);assert int(doc['distributedInfo']['rank'])==rank
    events=doc['traceEvents'];gpu=[];runtime=[];cpu=[];annotations=[];flows=[];special=[]
    args_keys=Counter();top_keys=Counter();categories=Counter();phases=Counter()
    for index,event in enumerate(events):
        cat=str(event.get('cat',''));ph=str(event.get('ph',''));name=str(event.get('name',''))
        categories[cat]+=1;phases[(cat,ph)]+=1
        args=event.get('args',{}) or {}
        is_special=cat in RAW_RUNTIME and any(word in name for word in ['Graph','StreamWaitEvent','EventRecord','Synchronize'])
        selected=cat in ['gpu_user_annotation','ac2g'] or is_special
        if selected:
            for k,v in event.items():top_keys[(cat,k,type(v).__name__)]+=1
            for k,v in args.items():args_keys[(cat,k,type(v).__name__)]+=1
        if not (selected or ph=='X' and cat in RAW_GPU|RAW_RUNTIME|RAW_CPU):continue
        assert 'ts' in event,(cat,ph,name)
        start=base+round(float(event['ts'])*1000);duration=round(float(event.get('dur',0))*1000)
        row=dict(iteration=iteration,rank=rank,event_id=f'source{iteration}:r{rank}:event{index}',category=cat,name=name,
            start_ns=start,end_ns=start+duration,duration_ns=duration,pid=str(event.get('pid','')),tid=str(event.get('tid','')),
            correlation=pick_id(args,['correlation','Correlation id','Correlation ID']),external_id=pick_id(args,['External id','external_id']))
        if ph=='X' and cat in RAW_GPU:
            row.update(stream=str(args.get('stream','')),device=str(args.get('device','')));gpu.append(row)
        elif ph=='X' and cat in RAW_RUNTIME:runtime.append(row)
        elif ph=='X' and cat in RAW_CPU:
            row['input_dims']=json.dumps(args.get('Input Dims',None),ensure_ascii=False,separators=(',',':'));cpu.append(row)
        if selected:
            expanded=dict(row,ph=ph,raw_event_index=index,flow_id_json=compact_json(event.get('id')),flow_id2_json=compact_json(event.get('id2')),
                bind_id_json=compact_json(event.get('bind_id')),binding_point=str(event.get('bp','')),scope_json=compact_json(event.get('scope')),
                args_json=compact_json(args),stream=pick_id(args,['stream','Stream','stream id']),device=pick_id(args,['device','Device']))
            if cat=='gpu_user_annotation':annotations.append(expanded)
            elif cat=='ac2g':flows.append(expanded)
            else:special.append(expanded)
    meta=dict(iteration=iteration,rank=rank,raw_path=str(path),raw_input_bytes=path.stat().st_size,raw_events=len(events),base_time_ns=base,
        distributedInfo=doc['distributedInfo'],category_counts=dict(categories),parse_seconds=perf_counter()-begin)
    del doc,events
    return [pd.DataFrame(x) for x in [gpu,runtime,cpu,annotations,flows,special]],meta,args_keys,top_keys,phases


def exact_core(actual,path,columns):
    old=pd.read_csv(path,dtype='string',keep_default_na=False,low_memory=False)
    left=actual[columns].astype('string').fillna('').set_index('event_id').sort_index()
    right=old[columns].set_index('event_id').sort_index()
    pd.testing.assert_frame_equal(left,right,check_dtype=False)
    return dict(path=str(path),rows=len(left),columns=columns,exact=True)


def identity_inventory(annotations,flows,special,runtime,cpu):
    correlations=defaultdict(list);owners=defaultdict(list)
    for r in runtime.itertuples():
        if r.correlation:correlations[r.correlation].append(r)
    for r in cpu.itertuples():
        if r.external_id:owners[r.external_id].append(r)
    rows=[]
    for a in annotations.itertuples():
        match=correlations.get(a.correlation,[]);owner=owners.get(a.external_id,[])
        rows.append(dict(annotation_id=a.event_id,annotation_name=a.name,correlation=a.correlation,external_id=a.external_id,
            runtime_matches=len(match),runtime_ids='|'.join(r.event_id for r in match),runtime_names='|'.join(r.name for r in match),
            CPU_external_id_matches=len(owner),CPU_owner_ids='|'.join(r.event_id for r in owner),CPU_owner_names='|'.join(r.name for r in owner)))
    flow_ids=defaultdict(list);at_point=defaultdict(list)
    for f in flows.itertuples():
        if f.flow_id_json!='null':flow_ids[f.flow_id_json].append(f)
        at_point[(f.pid,f.tid,int(f.start_ns))].append(f)
    graph_rows=[]
    for g in special[special.name.eq('musaGraphLaunch')].itertuples():
        # A numeric correlation ID equal to a flow ID is only a candidate
        # match; do not infer an edge before inspecting flow source/target.
        key=compact_json(int(g.correlation)) if g.correlation.isdigit() else compact_json(g.correlation)
        matched=flow_ids.get(key,[]);start=at_point.get((g.pid,g.tid,int(g.start_ns)),[]);end=at_point.get((g.pid,g.tid,int(g.end_ns)),[])
        graph_rows.append(dict(graph_runtime_id=g.event_id,correlation=g.correlation,external_id=g.external_id,
            same_numeric_correlation_flow_points=len(matched),same_correlation_flow_ids='|'.join(f.event_id for f in matched),
            exact_API_start_flow_points=len(start),exact_API_start_flow_ids='|'.join(f.event_id for f in start),
            exact_API_end_flow_points=len(end),exact_API_end_flow_ids='|'.join(f.event_id for f in end)))
    return pd.DataFrame(rows),pd.DataFrame(graph_rows)


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only' and plan['raw_trace_intake']['max_files']==1
    key=plan['raw_trace_intake']['keys'][0];iteration=plan['raw_trace_intake']['allowed_iterations'][0];rank=plan['raw_trace_intake']['allowed_source_ranks'][0]
    frames,meta,args_keys,top_keys,phases=read_registered(paths[key],iteration,rank)
    gpu,runtime,cpu,annotations,flows,special=frames
    checks=[exact_core(gpu,paths[f'source{iteration}_GPU.csv.gz'],BASE+['stream','device']),
            exact_core(runtime,paths[f'source{iteration}_runtime.csv.gz'],BASE),
            exact_core(cpu,paths[f'source{iteration}_CPU.csv.gz'],BASE+['input_dims'])]
    csv(out,'source_GPU_user_annotations.csv.gz',annotations);csv(out,'source_ac2g_flow_points.csv.gz',flows)
    csv(out,'source_graph_sync_runtime_fields.csv.gz',special)
    csv(out,'source_extra_argument_schema.csv',pd.DataFrame([dict(category=k[0],field=k[1],type=k[2],events=n) for k,n in sorted(args_keys.items())]))
    csv(out,'source_extra_top_level_schema.csv',pd.DataFrame([dict(category=k[0],field=k[1],type=k[2],events=n) for k,n in sorted(top_keys.items())]))
    csv(out,'source_trace_category_phase_counts.csv',pd.DataFrame([dict(category=k[0],ph=k[1],events=n) for k,n in sorted(phases.items())]))
    csv(out,'source_GPU_annotation_name_counts.csv',annotations.groupby(['name','ph','stream'],dropna=False).agg(
        annotations=('event_id','size'),minimum_duration_ns=('duration_ns','min'),maximum_duration_ns=('duration_ns','max')).reset_index())
    groups=[]
    for key,x in flows.groupby(['name','flow_id_json','flow_id2_json','scope_json'],dropna=False):
        groups.append(dict(name=key[0],flow_id_json=key[1],flow_id2_json=key[2],scope_json=key[3],points=len(x),
            phase_signature='|'.join(x.sort_values(['start_ns','raw_event_index']).ph.tolist()),timestamp_tied_points=int(x.start_ns.duplicated(keep=False).sum()),
            unique_pid_tids=x[['pid','tid']].drop_duplicates().shape[0]))
    groups=pd.DataFrame(groups);csv(out,'source_flow_identity_groups.csv.gz',groups)
    csv(out,'source_flow_shape_counts.csv',groups.groupby(['name','points','phase_signature','unique_pid_tids']).size().rename('flow_ids').reset_index())
    identity,graph=identity_inventory(annotations,flows,special,runtime,cpu)
    csv(out,'source_annotation_identity_candidates.csv',identity);csv(out,'source_graph_flow_identity_candidates.csv',graph)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    meta.update(peak_RSS_bytes=peak,analysis_seconds=perf_counter()-begin,core_checks=checks)
    dump(out/'source_graph_fields_resource_measurement.json',meta)
    dump(out/'field_contract.json',dict(time='T14-identical base + rounded raw offset, int64 ns; absent duration for flow point is zero, not a device span',
        annotations='gpu_user_annotation remains an annotation range, never added to kernel/copy/set busy union',
        flows='Preserve id/id2/scope and ph; numerical ID/temporal matches are candidates only, not accepted causal edges',
        runtime='Selected Graph/EventRecord/StreamWaitEvent/Synchronize args as present; do not invent missing handles',
        input_boundary='Single registered source85/rank16 raw plus exactT14 source comparison tables; target only SHA, no costs or prediction',
        next='Inspect schemas and graph flow candidates before inferring graph execution or extending any graph. No raw expansion by default.'))
    dump(out/'diagnostic.json',dict(status='SOURCE_GRAPH_FLOW_INTAKE_PASS',new_prediction=False,used_to_fit_model=False,new_target_timing_read=False,
        raw_trace_scanned=True,raw_source_iterations=[iteration],raw_source_ranks=[rank],raw_files=1,raw_bytes=meta['raw_input_bytes'],
        all_existing_T14_core_rows_exact=True,core_checks=checks,GPU_annotations=len(annotations),flow_points=len(flows),flow_identity_groups=len(groups),
        graph_launch_calls=len(graph),graph_with_correlation_flow_candidate=int(graph.same_numeric_correlation_flow_points.gt(0).sum()),
        graph_with_exact_start_flow_candidate=int(graph.exact_API_start_flow_points.gt(0).sum()),
        annotation_unique_runtime_candidates=int(identity.runtime_matches.eq(1).sum()),accepted_causal_edges=0,
        peak_RSS_bytes=peak,analysis_seconds=meta['analysis_seconds'],first_file_resource_gate_pass=peak<=1024*1024*1024,
        formal_topology_changed=False,next='Review omitted-category identity and graph candidates from this source-only intake; no timing cost or raw expansion before evidence review.'))

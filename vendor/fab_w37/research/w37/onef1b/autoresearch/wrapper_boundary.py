"""Post-seal inspection of existing derived wrapper boundaries; never a fitter."""
import json
import pandas as pd
from smoke_worker import dump,sha
from worker import csv


def diagnose(out,paths,plan):
    data=pd.read_csv(paths['wrapper_service_boundary_observations.csv.gz'])
    schema=dict(rows=len(data),columns={k:str(v) for k,v in data.dtypes.items()},
        missing={k:int(v) for k,v in data.isna().sum().items()},timing_scope='posthoc mixed source/target derived table; never source calibration')
    dump(out/'schema.json',schema);csv(out,'sample_rows.csv',data.head(24))
    groups=[c for c in ['case','phase','wrapper_name','wrapper_kind','kind','name','op'] if c in data]
    if groups:csv(out,'group_counts.csv',data.groupby(groups,dropna=False).size().rename('rows').reset_index())
    numbers=data.select_dtypes(include='number')
    csv(out,'numeric_summary.csv',numbers.describe(percentiles=[.05,.5,.95]).T.reset_index().rename(columns={'index':'column'}))
    for name in ['fused_a2a.py','token_dispatcher.py']:
        text=paths[name].read_text();(out/name).write_text(text)
        lines=[dict(file=name,line=i,text=line) for i,line in enumerate(text.splitlines(),1) if any(k in line.lower() for k in ['stream','synchronize','event','async','wait','dispatch','combine'])]
        csv(out,(name+'_semantic_lines.csv'),pd.DataFrame(lines))
    if plan['diagnostic']=='wrapper_boundary_partition':
        audit_boundaries(out,data,paths,plan);return
    manifest=json.loads(paths['wrapper_service_boundary_manifest'].read_text())
    dump(out/'diagnostic.json',dict(status='SCHEMA_AND_CODE_INTAKE_PASS',rows=len(data),diagnostic=plan['diagnostic'],
        source_and_target_mixed=True,used_to_fit_model=False,new_prediction=False,upstream_status=manifest['status'],
        input_sha256=sha(paths['wrapper_service_boundary_observations.csv.gz']),sealed_reference=plan['sealed_reference'],
        next='Use extracted schema and exact wrapper code to test interval identities and distinguish device-ready from CPU return; do not equate all wrapper wall with network service.'))


def interval_union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return sum(end-start for start,end in merged)


def audit_boundaries(out,data,paths,plan):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    tol=.001  # upstream epoch-ns float conversion; one microsecond, not a fit knob
    assert data.graph_launch_count.eq(1).all()
    wall=(data.prelaunch_duration_ms+data.postlaunch_to_wrapper_end_ms-data.wrapper_duration_ms).abs()
    assert wall.max()<tol
    for col in ['wrapper_duration_ms','prelaunch_duration_ms','postlaunch_to_wrapper_end_ms']:
        assert data[col].min()>=0
    data=data.copy()
    data['post_sync_call_duration_sum_ms']=data[['postlaunch_device_sync_ms','postlaunch_event_sync_ms','postlaunch_stream_sync_ms']].sum(axis=1)
    data['pre_sync_duration_exceeds_partition']=data.prelaunch_stream_sync_ms>data.prelaunch_duration_ms+tol
    data['post_sync_duration_exceeds_partition']=data.post_sync_call_duration_sum_ms>data.postlaunch_to_wrapper_end_ms+tol
    data['split']=np.where(data.case.eq('source256'),np.where(data.iteration.isin([85,90]),'source85_90_diagnostic',np.where(data.iteration.isin([95,100]),'source95_100_diagnostic','source60_80_diagnostic')),
        np.where(data.iteration.isin([85,90,95,100]),'target224_development_diagnostic','target224_60_80_diagnostic'))
    cols=['wrapper_duration_ms','prelaunch_duration_ms','postlaunch_to_wrapper_end_ms','prelaunch_stream_sync_ms',
          'postlaunch_device_sync_ms','postlaunch_event_sync_ms','postlaunch_stream_sync_ms']
    keys=['case','split','stage_role','phase','wrapper_name']
    group=data.groupby(keys)
    means=group[cols].mean().reset_index();means['wrappers']=group.size().to_numpy();csv(out,'wrapper_boundary_means.csv',means)
    stats=group[cols].quantile([.5,.95]).reset_index().rename(columns={'level_5':'quantile'});csv(out,'wrapper_boundary_quantiles.csv',stats)
    signatures=data.groupby(['case','split','phase','wrapper_name','boundary_signature_json']).size().rename('wrappers').reset_index();csv(out,'boundary_signatures.csv',signatures)
    pkeys=['case','iteration','rank','pp_stage','pp_lane','stage_role','phase','microbatch'];rows=[]
    for key,g in data.groupby(pkeys):
        union=interval_union(zip(g.wrapper_start_offset_ms,g.wrapper_end_offset_ms));total=g.wrapper_duration_ms.sum();phase=float(g.phase_duration_ms.iloc[0])
        assert abs(total-union)<tol and union<=phase+tol,(key,total,union,phase)
        assert g.wrapper_start_offset_ms.min()>=-tol and g.wrapper_end_offset_ms.max()<=phase+tol
        rows.append(dict(zip(pkeys,key))|dict(wrapper_count=len(g),phase_duration_ms=phase,wrapper_union_ms=union,wrapper_sum_ms=total,
            prelaunch_sum_ms=g.prelaunch_duration_ms.sum(),postlaunch_sum_ms=g.postlaunch_to_wrapper_end_ms.sum(),outside_wrapper_ms=phase-union,
            pre_stream_sync_sum_ms=g.prelaunch_stream_sync_ms.sum(),post_sync_call_sum_ms=g.post_sync_call_duration_sum_ms.sum()))
    phases=pd.DataFrame(rows);csv(out,'phase_wrapper_partition.csv.gz',phases)
    main=phases[phases.iteration.isin([85,90,95,100])]
    metrics=main.groupby(['case','phase']).mean(numeric_only=True).reset_index();csv(out,'primary_phase_boundary_means.csv',metrics)
    for label,path in [('source256','source256_window_partition.csv.gz'),('target224','target224_window_partition_ground_truth.csv.gz')]:
        previous=pd.read_csv(paths[path]);join=phases[phases.case.eq(label)].merge(previous,on=['iteration','pp_stage','pp_lane','phase','microbatch'],suffixes=('','_previous'),validate='one_to_one')
        assert len(join)==len(previous)
        for current,prior in [('prelaunch_sum_ms','fused_prelaunch_sum_ms'),('postlaunch_sum_ms','fused_postlaunch_sum_ms'),('outside_wrapper_ms','outside_fused_wrapper_ms')]:
            assert (join[current]-join[prior]).abs().max()<tol
    source=means[means.split.eq('source85_90_diagnostic')&means.stage_role.eq('interior')]
    target=means[means.split.eq('target224_development_diagnostic')&means.stage_role.eq('interior')]
    delta=source.merge(target,on=['stage_role','phase','wrapper_name'],suffixes=('_source','_target'),validate='one_to_one')
    for c in cols:delta[c+'_delta_ms']=delta[c+'_target']-delta[c+'_source']
    csv(out,'interior_wrapper_source_target_delta.csv',delta)
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(delta));width=.35
    for offset,label,suffix,color in [(-width/2,'source85/90 observed','source','#287a9e'),(width/2,'target224 85-100 observed','target','#d2763d')]:
        pre=delta['prelaunch_duration_ms_'+suffix].to_numpy();post=delta['postlaunch_to_wrapper_end_ms_'+suffix].to_numpy()
        ax.bar(x+offset,pre,width,color=color,label=label+' / before launch API')
        ax.bar(x+offset,post,width,bottom=pre,color=color,alpha=.45,hatch='//',label=label+' / after launch API')
    ax.set_xticks(x,[r.phase[0].upper()+' / '+r.wrapper_name.replace('Fused','') for r in delta.itertuples()],rotation=15,ha='right')
    ax.set_ylabel('Mean observed CPU wrapper wall (ms)');ax.set_title('Interior-stage wrapper boundaries: posthoc diagnosis, not prediction')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(out/'wrapper_boundary.svg',bbox_inches='tight');fig.savefig(out/'wrapper_boundary.png',dpi=160,bbox_inches='tight');plt.close(fig)
    script=paths['analyze_runtime_sync_service_boundary.py'];(out/script.name).write_text(script.read_text())
    report=dict(status='CPU_INTERVAL_IDENTITIES_PASS_GPU_EFFECTIVE_READY_NOT_OBSERVED',rows=len(data),phase_windows=len(phases),
        all_wrappers_have_one_graph_launch_api=True,max_cpu_pre_post_identity_error_ms=float(wall.max()),
        pre_sync_sum_exceeds_pre_partition_rows=int(data.pre_sync_duration_exceeds_partition.sum()),post_sync_sum_exceeds_post_partition_rows=int(data.post_sync_duration_exceeds_partition.sum()),
        cpu_wrapper_union_matches_sum=True,existing_phase_partition_reproduced=True,
        new_prediction=False,used_to_fit_model=False,formal_topology_changed=False,diagnostic=plan['diagnostic'],sealed_reference=plan['sealed_reference'],
        missing=['device graph start/end correlated to wrapper','payload-ready tensor event','stream and host thread IDs for sync-call interval union','communication request/group/payload linkage','deployed DeepEP/MUSA revision and flags'],
        interpretation='Launch API start partitions CPU wall only. Post-launch device/event waits may include queueing, local work and transport; their durations are subsets, never extra additive costs. Pinned Python wrapper provides event/dispatch semantics but does not identify MUSA backend implementation.')
    dump(out/'diagnostic.json',report)
    (out/'BOUNDARY_REPORT.md').write_text('# Wrapper 边界核验\n\n21,006 条 wrapper 记录均有一个 graph-launch API；pre/post 互斥分区与 wall、逐 phase wrapper union 及已有分区核对通过。同步调用耗时位于这些区间内部，不可再次累加。\n\n该表以 `musaGraphLaunch` 的 CPU API 入口切分，缺少 GPU 有效启动/完成与 payload-ready 事件，不能把这个切点直接当成网络 service 启点。Python wrapper 的 event/dispatch 代码只支持逻辑依赖，实际 MUSA 后端 revision 和 flags 未验证。图展示观察值的事后比较，不是新预测或通信 FCT。\n\n下一步必须使用可关联 request、stream、tensor-ready 与设备 graph 区间的源侧独立输入；在此之前不创建按目标差额拟合的等待节点。\n')

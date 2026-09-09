"""CPU checkpoint/GPU launch ownership and source-local tail prediction."""
from pathlib import Path
import json
import resource
from time import perf_counter
import numpy as np
import pandas as pd
from smoke_worker import dump,sha
from worker import csv
from runtime_pending import join_launches,split_name
from cp_alignment import mask_partition
from pp_semantics import align_observations
from readiness import ReadinessCosts

FIT=[85,90]
ENDS=['CP','stream0','all_nonPP']


def containing_index(points,intervals,time_column,groups):
    """Same-thread, disjoint half-open containment; preserve absolute int64 ns."""
    answer=pd.Series(-1,index=points.index,dtype='int64')
    indexed={k:g.sort_values('start_ns') for k,g in intervals.groupby(groups,dropna=False)}
    for key,group in points.groupby(groups,dropna=False):
        windows=indexed.get(key)
        if windows is None:continue
        starts=windows.start_ns.to_numpy(dtype='int64');ends=windows.end_ns.to_numpy(dtype='int64')
        assert np.all(ends>=starts) and np.all(starts[1:]>=ends[:-1])
        values=group[time_column].to_numpy(dtype='int64');where=np.searchsorted(starts,values,side='right')-1
        safe=np.maximum(where,0);valid=(where>=0)&(values<ends[safe])
        answer.loc[group.index[valid]]=windows.index.to_numpy()[safe[valid]]
    return answer


def checkpoint_ownership(cpu,windows,gpu,microbatches=4):
    parents=cpu[cpu.name.eq('CheckpointFunctionBackward')].copy().reset_index(drop=True)
    assert len(parents)==microbatches*4
    ep=windows[windows.window_type.eq('EP_wrapper')&windows.phase.eq('backward')].copy()
    ep['parent_index']=containing_index(ep,parents,'start_ns',['pid','tid'])
    assert ep.parent_index.ge(0).all() and len(ep)==microbatches*16
    expected={'ep_recompute_dispatch_wall','ep_recompute_combine_wall','ep_combine_backward_wall','ep_dispatch_backward_wall'}
    annotations=[]
    for index,p in parents.iterrows():
        w=ep[ep.parent_index==index].sort_values('start_ns')
        assert len(w)==4 and set(w.semantic_region)==expected and w.layer_id.nunique()==w.microbatch.nunique()==1
        assert w.end_ns.le(p.end_ns).all()
        row={k:p[k] for k in ['iteration','rank','event_id','start_ns','end_ns','pid','tid']}
        row.update(parent_index=index,parent_id=p.event_id,layer_id=int(w.layer_id.iloc[0]),execution_layer=int(w.execution_layer.iloc[0]),
            microbatch=int(w.microbatch.iloc[0]),split=split_name(int(p.iteration)),
            recompute_dispatch_start_ns=int(w.loc[w.semantic_region=='ep_recompute_dispatch_wall','start_ns'].iloc[0]),
            recompute_combine_end_ns=int(w.loc[w.semantic_region=='ep_recompute_combine_wall','end_ns'].iloc[0]),
            backward_combine_start_ns=int(w.loc[w.semantic_region=='ep_combine_backward_wall','start_ns'].iloc[0]))
        annotations.append(row)
    parents=pd.DataFrame(annotations)
    # Actual autograd::engine::evaluate_function child is a stronger CPU cut
    # than first EP-backward wrapper entry: attention/expert backward may start
    # earlier than that EP API. Identify its first post-recompute engine event.
    for index,p in parents.iterrows():
        engines=cpu[cpu.name.str.startswith('autograd::engine::evaluate_function:')&cpu.pid.eq(p.pid)&cpu.tid.eq(p.tid)
                    &cpu.start_ns.ge(p.recompute_combine_end_ns)&cpu.end_ns.le(p.end_ns)]
        assert len(engines)
        boundary=int(engines.start_ns.min());assert p.recompute_combine_end_ns<=boundary<=p.backward_combine_start_ns
        parents.loc[index,'autograd_start_offset_ns']=boundary-int(p.start_ns)
    parents['autograd_start_offset_ns']=parents.autograd_start_offset_ns.astype('int64')
    launch=gpu.rename(columns={'runtime_pid':'cpu_pid','runtime_tid':'cpu_tid'})
    point=launch.rename(columns={'pid':'gpu_pid','tid':'gpu_tid','cpu_pid':'pid','cpu_tid':'tid'})
    mapped=containing_index(point,parents,'runtime_start_ns',['pid','tid'])
    gpu=gpu.copy();gpu['parent_index']=mapped
    gpu['parent_id']=mapped.map(parents.parent_id).fillna('')
    for c in ['microbatch','layer_id','execution_layer']:
        gpu['parent_'+c]=mapped.map(parents[c]).fillna(-1).astype('int64')
    gpu['CPU_launch_region']='outside_checkpoint_parent'
    for index,p in parents.iterrows():
        mask=gpu.parent_index.eq(index)
        assert gpu.loc[mask,'runtime_end_ns'].le(p.end_ns).all()
        boundary=int(p.start_ns)+int(p.autograd_start_offset_ns)
        gpu.loc[mask,'CPU_launch_region']=np.where(gpu.loc[mask,'runtime_start_ns']<boundary,'checkpoint_recompute','checkpoint_autograd_backward')
    return parents,ep,gpu


def endpoint_rows(parents,gpu):
    rows=[]
    for p in parents.itertuples():
        own=gpu[gpu.parent_index.eq(p.parent_index)&~gpu.family.eq('PP_candidate')]
        row={k:getattr(p,k) for k in ['iteration','rank','microbatch','layer_id','execution_layer','parent_id','split','start_ns','end_ns']}
        row.update(mapped_GPU_events=len(own),mapped_CP_events=int(own.family.eq('CP_collective').sum()),
            recompute_GPU_events=int(own.CPU_launch_region.eq('checkpoint_recompute').sum()),
            autograd_GPU_events=int(own.CPU_launch_region.eq('checkpoint_autograd_backward').sum()))
        selections={'CP':own[own.family.eq('CP_collective')],'stream0':own[own.stream.eq('0')],'all_nonPP':own}
        for name,subset in selections.items():
            assert len(subset),(p.parent_id,name)
            end=int(subset.end_ns.max());last=subset[subset.end_ns.eq(end)].iloc[0]
            row[name+'_end_ns']=end;row[name+'_tail_after_parent_ns']=max(0,end-int(p.end_ns))
            row[name+'_end_after_parent_signed_ns']=end-int(p.end_ns)
            row[name+'_last_GPU_event_id']=last.event_id;row[name+'_last_family']=last.family
            row[name+'_last_CPU_owner']=last.unique_CPU_owner_name;row[name+'_last_launch_region']=last.CPU_launch_region
        tail_end=max(int(p.end_ns),row['all_nonPP_end_ns']);events=[]
        for g in own.itertuples():
            category=0 if g.family=='CP_collective' else (1 if g.stream=='0' else 2)
            events.append((int(g.start_ns),int(g.end_ns),category))
        parts=mask_partition(int(p.end_ns),tail_end,events)
        for mask,value in enumerate(parts):row[f'tail_mask{mask}_ns']=value
        assert sum(parts)==row['all_nonPP_tail_after_parent_ns']
        rows.append(row)
    return pd.DataFrame(rows)


class LocalTailModel:
    def __init__(self,tails,fit=FIT):
        self.fit=list(fit);self.data=tails[tails.iteration.isin(fit)].copy()
        assert len(self.data) and set(self.data.iteration)==set(fit)
        self.columns=[n+'_tail_after_parent_ns' for n in ENDS]
        self.parameters=self.data.groupby(['rank','layer_id'])[self.columns].median()
    def predict(self,keys):
        assert set(keys.columns)=={'iteration','rank','microbatch','layer_id','execution_layer'}
        out=keys.copy()
        for c in self.columns:out['predicted_'+c]=[float(self.parameters.loc[(r.rank,r.layer_id),c]) for r in keys.itertuples()]
        return out


def carryover_rows(parents,ep,gpu):
    rows=[];intersections=[]
    for w in ep[ep.semantic_region.eq('ep_recompute_dispatch_wall')].itertuples():
        hit=gpu[(gpu.start_ns<w.end_ns)&(gpu.end_ns>w.start_ns)&~gpu.family.eq('PP_candidate')].copy()
        start,end=int(w.start_ns),int(w.end_ns);records=[]
        for g in hit.itertuples():
            if g.family=='CP_collective':category=0
            elif g.parent_microbatch==w.microbatch and g.parent_execution_layer==w.execution_layer-1 and g.CPU_launch_region=='checkpoint_autograd_backward':category=1
            else:category=2
            records.append((int(g.start_ns),int(g.end_ns),category))
            intersections.append(dict(iteration=w.iteration,rank=w.rank,microbatch=w.microbatch,layer_id=w.layer_id,execution_layer=w.execution_layer,
                dispatch_window_id=w.window_id,GPU_event_id=g.event_id,source_parent_id=g.parent_id,source_parent_layer_id=g.parent_layer_id,
                source_parent_execution_layer=g.parent_execution_layer,source_parent_microbatch=g.parent_microbatch,source_launch_region=g.CPU_launch_region,
                family=g.family,category=category,clipped_start_ns=max(start,int(g.start_ns)),clipped_end_ns=min(end,int(g.end_ns))))
        parts=mask_partition(start,end,records)
        row=dict(iteration=w.iteration,rank=w.rank,microbatch=w.microbatch,layer_id=w.layer_id,execution_layer=w.execution_layer,split=split_name(w.iteration),duration_ns=end-start,
            CP_visible_ns=sum(parts[m] for m in [1,3,5,7]),previous_B_autograd_nonCP_only_ns=parts[2],previous_B_autograd_with_other_nonCP_ns=parts[6],
            other_nonCP_only_ns=parts[4],no_nonPP_device_event_visible_ns=parts[0])
        assert sum(row[c] for c in ['CP_visible_ns','previous_B_autograd_nonCP_only_ns','previous_B_autograd_with_other_nonCP_ns','other_nonCP_only_ns','no_nonPP_device_event_visible_ns'])==end-start
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(intersections)


def local_predictions(out,tails,pairs,aligned):
    keys=['iteration','rank','microbatch','layer_id','execution_layer']
    model=LocalTailModel(tails);parent_pred=model.predict(tails[keys]);csv(out,'source_parent_tail_parameters.csv',model.parameters.reset_index())
    csv(out,'source_parent_tail_predictions_sealed.csv',parent_pred)
    b=pairs[pairs.direction.eq('B')&pairs.pp_lane.eq(0)&pairs.sender_stage.eq(1)&pairs.iteration.isin([85,90,95,100])].copy()
    last=tails[tails.execution_layer.eq(3)].copy();last['rank']=16
    joined=b.merge(last,on=['iteration','microbatch'],validate='one_to_one')
    assert len(joined)==16 and joined.both_endpoints_single_message.all()
    for name in ENDS:
        joined[name+'_API_offset_ns']=joined[name+'_end_ns']-joined.sender_post_ns
        joined[name+'_to_first_API_return_ns']=joined.first_api_return_ns-np.maximum(joined[name+'_end_ns'],joined.receiver_post_ns)
    old=ReadinessCosts(aligned,pairs,FIT);d,c=old.ready_parameters[('B',0)]
    parameters=[dict(method='legacy_PP_lane0',ready_offset_ns=d,post_endpoint_completion_ns=c,fit_iterations='85,90',fit_scope='all source lane0 B single-message pairs',admitted=True)]
    fit=joined[joined.iteration.isin(FIT)]
    for name in ENDS:
        ready=float(fit[name+'_API_offset_ns'].median());residual=fit[name+'_to_first_API_return_ns']
        admitted=bool(ready>=0 and residual.ge(0).all())
        parameters.append(dict(method=name+'_GPU_endpoint',ready_offset_ns=ready,post_endpoint_completion_ns=float(residual.median()),
            fit_iterations='85,90',fit_scope='source rank16 last-layer8 messages',admitted=admitted,
            negative_fit_residual_count=int(residual.lt(0).sum()),interpretation='local temporal endpoint candidate; no all-rank transfer or pure service claim'))
    parameters=pd.DataFrame(parameters);csv(out,'source_PP_GPU_endpoint_parameters.csv',parameters)
    # Predict with API ENTRY inputs only, then seal before attaching completion
    # observations. This is local incremental validation, not a free-running PP graph.
    input_cols=['iteration','microbatch','sender_post_ns','receiver_post_ns']
    predictions=[]
    for p in parameters.itertuples():
        if not p.admitted:continue
        for r in joined[input_cols].itertuples():
            x=int(r.receiver_post_ns)-int(r.sender_post_ns)
            elapsed=max(float(p.ready_offset_ns),x)+float(p.post_endpoint_completion_ns)
            predictions.append(dict(iteration=r.iteration,microbatch=r.microbatch,method=p.method,predicted_first_return_after_sender_entry_ns=elapsed,
                receiver_minus_sender_API_entry_ns=x,prediction_scope='local observed API entries, no completion inputs'))
    predictions=pd.DataFrame(predictions);csv(out,'source_PP_GPU_endpoint_predictions_sealed.csv',predictions)
    names=['source_parent_tail_parameters.csv','source_parent_tail_predictions_sealed.csv','source_PP_GPU_endpoint_parameters.csv','source_PP_GPU_endpoint_predictions_sealed.csv']
    dump(out/'source_local_prediction_seal.json',dict(fit_iterations=FIT,incremental_validation=[95,100],target_access=False,
        scope='source256 rank16 local temporal tails and API-entry-conditioned PP completion; not end-to-end prediction',
        observation_access='source95/100 parsed for diagnostic ownership; only85/90 rows supplied to fitting, no blind claim',
        files=[dict(path=n,sha256=sha(out/n)) for n in names]))
    scored=parent_pred.merge(tails[keys+model.columns],on=keys,validate='one_to_one');scores=[]
    for name in ENDS:
        col=name+'_tail_after_parent_ns';scored[name+'_error_ns']=scored['predicted_'+col]-scored[col]
    scored['split']=scored.iteration.map(split_name);csv(out,'source_parent_tail_validation.csv',scored)
    for split,g in scored.groupby('split'):
        for name in ENDS:
            error=g[name+'_error_ns']/1e6;scores.append(dict(split=split,method=name,samples=len(g),mae_ms=float(error.abs().mean()),bias_ms=float(error.mean())))
    csv(out,'source_parent_tail_metrics.csv',pd.DataFrame(scores))
    observed=joined[['iteration','microbatch','first_api_return_ns','sender_post_ns']].copy();observed['actual_first_return_after_sender_entry_ns']=observed.first_api_return_ns-observed.sender_post_ns
    evaluated=predictions.merge(observed[['iteration','microbatch','actual_first_return_after_sender_entry_ns']],on=['iteration','microbatch'],validate='many_to_one')
    evaluated['error_ms']=(evaluated.predicted_first_return_after_sender_entry_ns-evaluated.actual_first_return_after_sender_entry_ns)/1e6
    evaluated['split']=evaluated.iteration.map(split_name);csv(out,'source_PP_GPU_endpoint_validation.csv',evaluated)
    metrics=evaluated.groupby(['split','method']).agg(samples=('error_ms','size'),mae_ms=('error_ms',lambda x:x.abs().mean()),bias_ms=('error_ms','mean')).reset_index();csv(out,'source_PP_GPU_endpoint_metrics.csv',metrics)
    csv(out,'source_PP_GPU_endpoint_observed_diagnostics.csv',joined)
    for item in json.loads((out/'source_local_prediction_seal.json').read_text())['files']:assert sha(out/item['path'])==item['sha256']
    return pd.DataFrame(scores),metrics,parameters,joined


def pp_GPU_alignment(out,gpu,joined,windows):
    rows=[]
    for r in joined.itertuples():
        api=windows[windows.iteration.eq(r.iteration)&windows.window_type.eq('PP_API')
            &windows.semantic_region.eq('send_backward')&windows.start_ns.eq(r.sender_post_ns)]
        assert len(api)==1
        a=api.iloc[0]
        kernels=gpu[gpu.iteration.eq(r.iteration)&gpu.family.eq('PP_candidate')&gpu.name.str.contains('mccl',case=False,regex=False)&gpu.name.str.contains('sendrecv',case=False,regex=False)
            &gpu.runtime_pid.eq(a.pid)&gpu.runtime_tid.eq(a.tid)&gpu.runtime_start_ns.ge(r.sender_post_ns)
            &gpu.runtime_end_ns.le(a.end_ns)]
        assert len(kernels)==1,'single B-send CPU API must contain one correlated SendRecv launch in this sample'
        k=kernels.iloc[0]
        before=int(k.start_ns)-int(r.stream0_end_ns);duration=int(k.end_ns)-int(k.start_ns)
        after=int(r.first_api_return_ns)-int(k.end_ns)
        assert min(before,duration,after)>=0
        assert before+duration+after==int(r.first_api_return_ns)-int(r.stream0_end_ns)
        rows.append(dict(iteration=r.iteration,rank=16,microbatch=r.microbatch,split=split_name(r.iteration),
            sender_API_id=a.window_id,sender_API_entry_ns=int(r.sender_post_ns),sender_API_return_ns=int(a.end_ns),
            last_parent_stream0_GPU_end_ns=int(r.stream0_end_ns),PP_GPU_event_id=k.event_id,PP_runtime_event_id=k.unique_runtime_event_id,
            PP_runtime_start_ns=int(k.runtime_start_ns),PP_runtime_end_ns=int(k.runtime_end_ns),PP_GPU_start_ns=int(k.start_ns),PP_GPU_end_ns=int(k.end_ns),
            PP_GPU_stream=k.stream,first_API_return_ns=int(r.first_api_return_ns),
            stream0_end_to_PP_GPU_start_ns=before,PP_GPU_residence_ns=duration,PP_GPU_end_to_first_API_return_ns=after,
            interpretation='observed same-thread API launch correlation and ordered boundaries; device residence not independent network service'))
    result=pd.DataFrame(rows);csv(out,'source_PP_GPU_start_end_alignment.csv',result)
    cols=['stream0_end_to_PP_GPU_start_ns','PP_GPU_residence_ns','PP_GPU_end_to_first_API_return_ns']
    summary=result.groupby('split')[cols].mean().reset_index();csv(out,'source_PP_GPU_alignment_summary.csv',summary)
    return summary


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only'
    windows=pd.read_csv(paths['runtime_windows.csv.gz'],dtype={'pid':'string','tid':'string'},low_memory=False)
    allparents=[];alltails=[];allgpu=[];carry=[];links=[];checks=[]
    for iteration in [85,90,95,100]:
        dtype={'correlation':'string','external_id':'string','pid':'string','tid':'string','stream':'string'}
        gpu=pd.read_csv(paths[f'source{iteration}_GPU.csv.gz'],dtype=dtype,low_memory=False)
        runtime=pd.read_csv(paths[f'source{iteration}_runtime.csv.gz'],dtype=dtype,low_memory=False)
        cpu=pd.read_csv(paths[f'source{iteration}_CPU.csv.gz'],dtype=dtype,low_memory=False)
        gpu=join_launches(gpu,runtime,cpu);w=windows[windows.iteration.eq(iteration)]
        parents,ep,gpu=checkpoint_ownership(cpu,w,gpu);tails=endpoint_rows(parents,gpu);cover,intersections=carryover_rows(parents,ep,gpu)
        allparents.append(parents);alltails.append(tails);allgpu.append(gpu);carry.append(cover);links.append(intersections)
        checks.append(dict(iteration=iteration,rank=16,parents=len(parents),backward_EP_wrappers=len(ep),GPU_events=len(gpu),
            GPU_launches_inside_checkpoint=int(gpu.parent_index.ge(0).sum()),GPU_launches_outside_checkpoint=int(gpu.parent_index.lt(0).sum()),
            all_EP_parents_unique=True,all_parent_submissions_return_before_parent=True,all_tail_partitions_conserve=True,
            CP_events_per_parent=sorted(tails.mapped_CP_events.unique().tolist())))
    parents=pd.concat(allparents,ignore_index=True);tails=pd.concat(alltails,ignore_index=True);gpu=pd.concat(allgpu,ignore_index=True);cover=pd.concat(carry,ignore_index=True)
    for name,frame in [('source_CPU_checkpoint_parents.csv',parents),('source_GPU_checkpoint_ownership.csv.gz',gpu),('source_checkpoint_GPU_tails.csv',tails),
        ('source_dispatch_previous_checkpoint_coverage.csv',cover),('source_dispatch_GPU_parent_intersections.csv.gz',pd.concat(links,ignore_index=True))]:csv(out,name,frame)
    csv(out,'source_previous_checkpoint_profile.csv',cover.groupby(['split','execution_layer'])[[c for c in cover if c.endswith('_ns')]].mean().reset_index())
    phases=pd.read_csv(paths['source_pp_trace_events_60_100.csv']);apis=pd.read_csv(paths['source_pp_api_events_60_100.csv']);aligned,pairs=align_observations(phases,apis)
    parent_metrics,pp_metrics,parameters,joined=local_predictions(out,tails,pairs,aligned)
    pp_alignment=pp_GPU_alignment(out,gpu,joined,windows)
    evidence=[]
    for key in ['training_checkpoint.py','training_transformer_block.py','training_fused_a2a.py']:
        path=paths[key];content=path.read_text();(out/key).write_text(content);evidence.append(dict(key=key,path=str(path),sha256=sha(path)))
    dump(out/'field_contract.json',dict(ownership='same-pid/tid runtime launch start inside one CPU checkpoint parent; GPU timestamps may extend past parent CPU return',
        logical_layer='each parent contains exactly4 frozen EP wrappers sharing MB/logical layer;16 parents per iteration',
        backward_cut='first nested autograd engine CPU event after recompute combine returns; temporal code-aligned boundary, not a device fence',
        PP_endpoint='stream0 is observed attention/matrix/default device stream; its completion is a tested local endpoint candidate, not independently proven payload-ready',
        async_collectives='all_nonPP means events whose runtime launch belongs to this checkpoint parent, not all events in the trace. Unassigned DP/EDP or later work is not included or made a PP gate.',
        PP_GPU_alignment='one correlated SendRecv launch inside each same-thread B-send CPU API; compare its GPU start/end to last parent stream0 end and first API return, do not fit it into predictions',
        source_code=evidence,deployed_revision_verified=False,topology='formal lock unchanged; ownership associations do not add scheduling edges',
        local_prediction='fit source85/90 only, parameter/prediction SHA seal before attaching validation completions;95/100 incremental with historical exposure',
        scope='single source rank16 only, not independent target or all-rank GPU model'))
    draw(out,parents,gpu,cover,parent_metrics,pp_metrics)
    dump(out/'diagnostic.json',dict(status='SOURCE_CHECKPOINT_GPU_TAIL_PASS',new_prediction='source_local_only',new_target_prediction=False,new_target_timing_read=False,
        used_to_fit_model='source85/90 local tail and endpoint parameters only',raw_trace_scanned=False,source_checks=checks,
        parent_tail_metrics=parent_metrics.to_dict('records'),PP_endpoint_metrics=pp_metrics.to_dict('records'),PP_endpoint_parameters=parameters.fillna('').to_dict('records'),
        observed_PP_endpoint_means=joined.groupby(joined.iteration.map(split_name).rename('split'))[[n+'_API_offset_ns' for n in ENDS]+[n+'_to_first_API_return_ns' for n in ENDS]].mean().reset_index().to_dict('records'),
        observed_PP_GPU_alignment=pp_alignment.to_dict('records'),
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,analysis_seconds=perf_counter()-begin,formal_topology_changed=False,
        next='Review previous logical-layer GPU ownership, source-only local tail errors and PP endpoint admissibility before broader source or target prediction.'))


def draw(out,parents,gpu,cover,parent_metrics,pp_metrics):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    p=parents[parents.iteration.eq(85)&parents.microbatch.eq(0)].sort_values('execution_layer');base=int(p.start_ns.min())
    ids=set(p.parent_id);g=gpu[gpu.iteration.eq(85)&gpu.parent_id.isin(ids)&~gpu.family.eq('PP_candidate')]
    streams=sorted(g.stream.unique(),key=int);labels=['CPU checkpoint']+['GPU stream '+s for s in streams];colors={i:plt.get_cmap('tab10')(i) for i in range(4)}
    fig,ax=plt.subplots(figsize=(15,5))
    for r in p.itertuples():
        x=(int(r.start_ns)-base)/1e6;width=(int(r.end_ns)-int(r.start_ns))/1e6
        ax.broken_barh([(x,width)],(-.35,.7),facecolors=[colors[r.execution_layer]],alpha=.55)
        ax.text(x+width/2,0,f'B layer {r.layer_id}',ha='center',va='center',fontsize=9)
    for r in g.itertuples():
        y=streams.index(r.stream)+1;ax.broken_barh([((int(r.start_ns)-base)/1e6,(int(r.end_ns)-int(r.start_ns))/1e6)],(y-.32,.64),facecolors=[colors[r.parent_execution_layer]])
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('ms from source85 rank16 MB0 first B checkpoint CPU entry')
    ax.set_title('Observed GPU activity colored by CPU launch checkpoint; ownership does not imply a completion gate')
    ax.legend(handles=[Patch(color=colors[i],label=f'CPU execution ordinal {i}') for i in range(4)],loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(out/'source_checkpoint_GPU_timeline.svg',bbox_inches='tight');fig.savefig(out/'source_checkpoint_GPU_timeline.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,frame,title in [(axes[0],parent_metrics,'Parent CPU-return to GPU-tail cost'),(axes[1],pp_metrics,'PP completion with observed API entries')]:
        d=frame.pivot(index='method',columns='split',values='mae_ms');d.plot.bar(ax=ax);ax.set_ylabel('MAE (ms)');ax.set_title(title);ax.tick_params(axis='x',labelrotation=15)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.35),fontsize=8)
    fig.suptitle('Source rank16 local predictions;85/90 fit,95/100 incremental, no target prediction')
    fig.tight_layout();fig.savefig(out/'source_GPU_tail_local_prediction.svg',bbox_inches='tight');fig.savefig(out/'source_GPU_tail_local_prediction.png',dpi=160,bbox_inches='tight');plt.close(fig)

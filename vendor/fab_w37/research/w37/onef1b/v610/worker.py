"""Materialize the approved method, replay it, seal, then evaluate and render."""
import json,sys
import numpy as np
import pandas as pd
from common import ROOT,RUN,dump,sha,setup,verify
COMP=['compute_exposed_ns_model','compute_overlap_ns_model','network_service_ns_model','software_sync_ns_model','framework_residual_ns_model']

def model(spec,paths,out):
    sys.path.insert(0,str(ROOT/'case_224gpu_pp14_cp2_a2a/scripts'))
    from build_dag_v67_microbatch_runtime_shape import replay
    from build_dag_v683_causal_program_order import topology_fingerprint
    seal=json.loads(paths['parent_seal'].read_text())
    for name in ['updates','expected_timings','sealed_predictions']:
        assert seal['files'][paths[name].name]==sha(paths[name])
    nodes=pd.read_csv(paths['v685_nodes'],low_memory=False)
    edges=pd.read_csv(paths['v685_edges'],low_memory=False)
    changes=pd.read_csv(paths['updates']);changes=changes[changes.variant.eq('fine_both')].copy()
    assert len(changes)==70192 and changes.node_id.is_unique
    idx=changes.node_id.map(pd.Series(nodes.index,index=nodes.node_id)).to_numpy()
    old=nodes.loc[idx].copy();overlap=changes.source_key.str.split('|').str[3].eq('overlap_with_collective').to_numpy()
    cost=changes.new_compute_ns.to_numpy(dtype='int64');duration=changes.new_duration_ns.to_numpy(dtype='int64')
    assert np.array_equal(old.duration_ns.to_numpy(),changes.old_duration_ns.to_numpy())
    assert np.array_equal((old.compute_exposed_ns_model+old.compute_overlap_ns_model).to_numpy(),changes.old_compute_ns.to_numpy())
    assert np.array_equal(duration,np.where(overlap,cost,np.maximum(old.duration_ns.to_numpy(),cost)))
    nodes.loc[idx,'compute_exposed_ns_model']=np.where(overlap,0,cost)
    nodes.loc[idx,'compute_overlap_ns_model']=np.where(overlap,cost,0)
    nodes.loc[idx,'framework_residual_ns_model']=np.where(overlap,0,duration-cost)
    nodes.loc[idx,'duration_ns']=duration
    nodes['model_version']='v6.10'
    nodes['inherited_timing_source']=nodes.timing_source
    nodes.loc[idx,'timing_source']='v610_source85_90_fine_cost_times_inherited_v67_shape'
    assert nodes[COMP].ge(0).all().all() and nodes[COMP].sum(axis=1).eq(nodes.duration_ns).all()
    assert topology_fingerprint(edges)==spec['topology_sha256']
    nodes,critical=replay(nodes,edges,'iteration:completion_join')
    expected=pd.read_csv(paths['expected_timings'])
    pd.testing.assert_frame_equal(nodes[expected.columns].reset_index(drop=True),expected,check_exact=True)
    p=pd.read_csv(paths['sealed_predictions']);p=p[p.variant.eq('fine_both')].iloc[0].to_dict();p['variant']='v6.10'
    assert abs(nodes.loc[nodes.node_id.eq('iteration:completion_join'),'predicted_end_ns'].iloc[0]/1e6-p['raw_ms'])<1e-6
    changes['variant']='v6.10';changes.to_csv(out/'node_parameter_bindings.csv.gz',index=False,compression='gzip')
    nodes.to_csv(out/'nodes.csv.gz',index=False,compression='gzip');(out/'edges.csv.gz').write_bytes(paths['v685_edges'].read_bytes())
    critical.to_csv(out/'critical_path.csv',index=False);pd.DataFrame([p]).to_csv(out/'predictions.csv',index=False)
    for key in ['fine_parameters','shape_factors']:(out/(key+'.csv')).write_bytes(paths[key].read_bytes())
    loc=changes.copy();loc['direction']=loc.source_key.str.split('|').str[0]
    loc.groupby(['pp_stage','direction']).agg(nodes=('node_id','size'),ranks=('rank','nunique'),microbatches=('microbatch','nunique')).reset_index().to_csv(out/'coverage.csv',index=False)
    # Use a genuine interior node, so the explanation no longer suggests only PP0/PP13 changed.
    example=changes[(changes.pp_stage==6)&(changes.microbatch==0)&(changes['rank']==96)&changes.source_key.str.contains('forward')].iloc[0]
    example_row={k:(None if pd.isna(v) else v) for k,v in nodes.loc[nodes.node_id.eq(example.node_id)].iloc[0].to_dict().items()}
    pp=nodes[nodes.kind.eq('pp_p2p')]
    pp_stats={phase:dict(nodes=len(g),mean_ms=float(g.duration_ns.mean()/1e6),min_ms=float(g.duration_ns.min()/1e6),max_ms=float(g.duration_ns.max()/1e6)) for phase,g in pp.groupby('phase')}
    summary=dict(pp_costs=pp_stats,version='v6.10',status='FROZEN_DEVELOPMENT_RELEASE',nodes=len(nodes),edges=len(edges),updated_nodes=len(changes),updated_ranks=int(changes['rank'].nunique()),bound_fine_keys=int(changes.source_key.nunique()),fit_parameters=len(pd.read_csv(paths['fine_parameters'])),prediction=p,example=example.to_dict(),example_node=example_row,topology=spec['topology_sha256'],source_full_iteration_validated=False,wall_floor_independently_validated=False,exact_prior_timings=True,selection=spec['selection'])
    dump(out/'release.json',summary)
    try:paths['version_iterations'].open().read(1)
    except PermissionError:guard=True
    else:raise AssertionError('Model could read target observations')
    files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['execution.log','command.json']}
    dump(out/'prediction_seal.json',dict(version='v6.10',status='SEALED_BEFORE_RELEASE_EVALUATOR',files=files,topology=spec['topology_sha256'],target_observation_read=False,guard_probe=guard,prior_research_selection_known=True,source_fit=[85,90],inherited_source=spec['inherited_source']))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['example_node','example']}))

def evaluate(spec,paths,out):
    verify('model')
    seal=json.loads((RUN/'model/prediction_seal.json').read_text())
    for name,h in seal['files'].items():assert sha(RUN/'model'/name)==h
    p=pd.read_csv(RUN/'model/predictions.csv').iloc[0]
    a=pd.read_csv(paths['version_iterations']);a=a[a.variant.eq('v685_frozen')&a.split.eq('development_primary')]
    ph=pd.read_csv(paths['v685_phase_rows']);ph=ph[ph.variant.eq('v685_frozen')&ph.split.eq('development_primary')]
    expected=pd.read_csv(paths['expected_evaluation']);expected=expected[expected.variant.eq('fine_both')].set_index('iteration')
    records=[];phase=[]
    assert set(a.iteration)=={85,90,95,100} and len(a)==4
    for r in a.itertuples():
        d=dict(version='v6.10',iteration=int(r.iteration))
        for key in ['profiler','training','onef1b']:
            predicted=float(p[key+'_ms'])
            observed=float(ph[(ph.iteration==r.iteration)&ph.phase.eq('onef1b')].actual_ms.iloc[0]) if key=='onef1b' else float(getattr(r,'actual_'+key+'_ms'))
            d.update({key+'_predicted_ms':predicted,key+'_actual_ms':observed,key+'_APE_pct':abs(predicted-observed)/observed*100})
            assert abs(d[key+'_APE_pct']-expected.loc[r.iteration,key+'_APE_pct'])<1e-9
        d.update(mfu_predicted_pct=float(p.mfu_pct),mfu_actual_pct=float(r.actual_mfu_pct_derived),mfu_relative_APE_pct=abs(p.mfu_pct-r.actual_mfu_pct_derived)/r.actual_mfu_pct_derived*100,mfu_absolute_error_pp=abs(p.mfu_pct-r.actual_mfu_pct_derived))
        assert abs(d['mfu_relative_APE_pct']-expected.loc[r.iteration,'mfu_relative_APE_pct'])<1e-9
        records.append(d)
        for q in ph[ph.iteration==r.iteration].itertuples():
            pred=float(p[q.phase+'_ms']);phase.append(dict(version='v6.10',iteration=int(r.iteration),phase=q.phase,predicted_ms=pred,actual_ms=float(q.actual_ms),error_ms=pred-q.actual_ms,APE_pct=abs(pred-q.actual_ms)/q.actual_ms*100))
        assert abs(sum(float(p[key+'_ms']) for key in ['entry','onef1b','tail','outer'])-p.training_ms)<1e-6
    data=pd.DataFrame(records);data.to_csv(out/'iteration_results.csv',index=False)
    phase=pd.DataFrame(phase);phase.to_csv(out/'phase_iteration_results.csv',index=False)
    phase.groupby('phase',sort=False).agg(predicted_ms=('predicted_ms','mean'),actual_ms=('actual_ms','mean'),MAPE_pct=('APE_pct','mean'),bias_ms=('error_ms','mean')).reset_index().to_csv(out/'phase_metrics.csv',index=False)
    metrics=dict(version='v6.10',iterations=4,**{key+'_MAPE_pct':float(data[key+'_APE_pct'].mean()) for key in ['onef1b','profiler','training','mfu_relative']},mfu_MAE_pp=float(data.mfu_absolute_error_pp.mean()),target_split='already exposed development',selection_known=True)
    dump(out/'metrics.json',metrics);print(json.dumps(metrics))

def render(spec,paths,out):
    verify('model');verify('evaluate')
    from render import build
    build(spec,paths,out)

if __name__=='__main__':
    stage=sys.argv[1];spec,paths,reads=setup(stage)
    try:globals()[stage](spec,paths,RUN/stage)
    finally:dump(RUN/stage/'input_access.json',dict(stage=stage,data_reads=sorted(reads),raw_trace_reads=0))

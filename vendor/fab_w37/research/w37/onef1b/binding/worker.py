"""Source cost schema audit, conservative candidate, then sealed target scoring."""
import argparse,collections,csv,gzip,json,sys
import numpy as np
import pandas as pd
from common import ROOT,RUN,dump,write_csv,sha,setup

PHYSICAL=['phase','layer_type','layer_context','execution_scope','semantic_slot']
FINE=['phase','microbatch','layer_id','execution_scope','semantic_slot']
COMP=['compute_exposed_ns_model','compute_overlap_ns_model','network_service_ns_model','software_sync_ns_model','framework_residual_ns_model']
PREFIXES=['compute_gap','compute_overlap','step_exit']

def verify_stage(stage):
    parent=RUN/stage;mark=json.loads((parent/'complete.json').read_text())
    assert mark['status']=='PASS'
    for p,d in mark['files'].items():assert sha(parent/p)==d,p

def audit(spec,paths,out):
    parameters=pd.read_csv(paths['compute_parameters'])
    p=parameters[parameters.parameter_view.eq('window_split')&parameters.cost_group.eq('__physical_slot_total__')&parameters.autograd_phase.eq('physical_mixed')].copy()
    p['key']=p[PHYSICAL].astype(str).agg('|'.join,axis=1);assert len(p)==58 and p.key.is_unique
    use=['node_id','kind','rank','pp_stage','pp_lane','phase','microbatch','layer_id','semantic_slot','timing_component','timing_source','source_parameter_key','duration_ns',*COMP]
    nodes=pd.read_csv(paths['v685_nodes'],usecols=use,keep_default_na=False)
    nodes['key_format']=nodes.source_parameter_key.map(lambda k:k.split('|')[0] if k else '(empty)')
    nodes['positive_compute']=nodes.compute_exposed_ns_model+nodes.compute_overlap_ns_model>0
    direct=nodes[nodes.source_parameter_key.isin(p.key)&nodes.positive_compute]
    assert len(direct)==144 and direct.source_parameter_key.nunique()==3
    direct.to_csv(out/'formal_direct_bindings.csv',index=False)
    cols=['iteration','rank','pp_stage','pp_lane','phase','microbatch','layer_id','layer_type','layer_context','parameter_view','execution_scope','semantic_slot','cost_group','autograd_phase','active_union_ns']
    observations=pd.read_csv(paths['observations'],usecols=cols)
    physical=observations[observations.parameter_view.eq('window_split')&observations.cost_group.eq('__physical_slot_total__')&observations.autograd_phase.eq('physical_mixed')].copy()
    physical['aggregate_key']=physical[PHYSICAL].astype(str).agg('|'.join,axis=1)
    physical['fine_key']=physical[FINE].astype(str).agg('|'.join,axis=1)
    fine=physical[physical.layer_id.ge(0)].copy()
    fits=fine[fine.iteration.isin(spec['source_calibration'])].groupby('fine_key',as_index=False).agg(value_ns=('active_union_ns','median'),fit_samples=('active_union_ns','size'),fit_iterations=('iteration','nunique'),aggregate_key=('aggregate_key','first'))
    fits.value_ns=fits.value_ns.round().astype('int64')
    aggregate_sets=fine.groupby('fine_key').aggregate_key.agg(lambda x:';'.join(sorted(set(x)))).to_dict()
    assert all(';' not in x for x in aggregate_sets.values()),'Fine key has changing semantic context'
    fits['source_fit']='85|90';fits.to_csv(out/'fine_parameters.csv',index=False)
    fit_lookup=fits.set_index('fine_key').value_ns.to_dict()
    old=fine[fine.iteration.isin([60,65,70,75,80,85,90,95,100])].groupby('fine_key').active_union_ns.median().round().astype('int64').to_dict()
    records=[]
    for n in nodes[nodes.key_format.isin(PREFIXES)].itertuples():
        prefix,key=n.source_parameter_key.split('|',1)
        parts=key.split('|');assert len(parts)==5
        phase,mb,layer,scope,slot=parts
        assert phase=={'FWD':'forward','BWD':'backward'}[n.phase] and int(mb)==n.microbatch
        if n.layer_id>=0:assert int(layer)==n.layer_id
        if prefix=='compute_overlap':assert scope=='overlap_with_collective'
        else:assert scope=='exposed_gap'
        c=n.compute_exposed_ns_model+n.compute_overlap_ns_model
        records.append(dict(node_id=n.node_id,rank=n.rank,pp_stage=n.pp_stage,microbatch=n.microbatch,source_layer_id=int(layer),schema=prefix,fine_key=key,aggregate_key=aggregate_sets.get(key,''),source_observation_present=key in aggregate_sets,new_fit_present=key in fit_lookup,old_positive_compute=bool(c>0),eligible=key in fit_lookup and c>0,old_compute_ns=c,old_duration_ns=n.duration_ns,old_residual_ns=n.framework_residual_ns_model,new_compute_ns=fit_lookup.get(key,0),old_all9_source_median_ns=old.get(key,0),old_matches_all9_median=(c==old.get(key,0))))
    bindings=pd.DataFrame(records);bindings.to_csv(out/'fine_node_bindings.csv.gz',index=False,compression='gzip')
    report=[]
    for r in p.itertuples():
        d=direct[direct.source_parameter_key.eq(r.key)]
        related=bindings[bindings.aggregate_key.eq(r.key)]
        reason='direct_boundary_key_retained' if len(d) else ('internal_nodes_use_fine_schema' if len(related) else 'no_same_context_fine_binding_requires_boundary_review')
        report.append(dict(key=r.key,formal_direct_nodes=len(d),related_fine_nodes=len(related),related_eligible_positive_nodes=int(related.eligible.sum()),stages='|'.join(map(str,sorted(set(d.pp_stage)|set(related.pp_stage)))),reason=reason,source_context_layers='|'.join(map(str,sorted(physical.loc[physical.aggregate_key.eq(r.key),'layer_id'].unique())))))
    write_csv(out/'all_58_key_audit.csv',report)
    schema=nodes.groupby(['key_format','timing_component','timing_source'],as_index=False).agg(nodes=('node_id','size'),positive_compute_nodes=('positive_compute','sum'))
    schema.to_csv(out/'node_key_schemas.csv',index=False)
    held=fine[fine.iteration.isin(spec['source_incremental_development'])].copy();held['predicted_ns']=held.fine_key.map(fit_lookup);held['legacy_all9_ns']=held.fine_key.map(old)
    held['eligible']=held.predicted_ns.notna();held['error_ns']=held.predicted_ns-held.active_union_ns
    held['absolute_error_ns']=held.error_ns.abs();held.to_csv(out/'source_incremental_rows.csv.gz',index=False,compression='gzip')
    metrics=[]
    for (iteration,scope),g in held[held.eligible].groupby(['iteration','execution_scope']):
        pos=g.active_union_ns>0
        metrics.append(dict(iteration=int(iteration),scope=scope,n=len(g),MAE_ms=float(g.absolute_error_ns.mean()/1e6),bias_ms=float(g.error_ns.mean()/1e6),positive_cost_MAPE_pct=float((g.loc[pos,'absolute_error_ns']/g.loc[pos,'active_union_ns']).mean()*100),zero_observations=int((~pos).sum()),meaning='local cost errors, not stage or iteration MAPE'))
    write_csv(out/'source_incremental_metrics.csv',metrics)
    src=pd.read_csv(paths['v685_source_nodes'],nrows=0)
    summary={'status':'PASS','target_nodes':len(nodes),'physical_keys':len(p),'direct_keys':3,'direct_nodes':144,'reason_counts':dict(collections.Counter(r['reason'] for r in report)),
             'fine_nodes':len(bindings),'fine_positive':int(bindings.old_positive_compute.sum()),'eligible_fine_nodes':int(bindings.eligible.sum()),'fine_unobserved_nodes':int((~bindings.source_observation_present).sum()),'old_fine_cost_matches_all9_source_median_nodes':int(bindings.old_matches_all9_median.sum()),'fine_fit_parameters':len(fits),'holdout_rows':len(held),'holdout_without_fit':int((~held.eligible).sum()),
             'source_full_graph_new_compute_key_column_present':'source_parameter_key' in src.columns,'source_full_graph_validation':'not established; source schema differs; local cost validation does not validate full iteration',
             'source_split':{'fit':[85,90],'incremental_development':[95,100],'historical_inheritance':[60,65,70,75,80,85,90,95,100]},'target_observation_read':False}
    dump(out/'summary.json',summary);print(json.dumps(summary))

def model(spec,paths,out):
    verify_stage('audit')
    sys.path.insert(0,str(ROOT/'case_224gpu_pp14_cp2_a2a/scripts'))
    from build_dag_v67_microbatch_runtime_shape import replay
    from build_dag_v683_causal_program_order import topology_fingerprint
    nodes=pd.read_csv(paths['v685_nodes'],low_memory=False);edges=pd.read_csv(paths['v685_edges'],low_memory=False)
    lock=topology_fingerprint(edges);assert lock=='f1d560daaa344a5980697b67e71b97140c9bfd38fe12a2c676daf05c53bd8d04'
    bindings=pd.read_csv(RUN/'audit/fine_node_bindings.csv.gz');eligible=bindings[bindings.eligible].copy()
    # v67 redistributes each rank/phase/microbatch's compute and residual costs.
    # Recover its existing effective multiplier; never overwrite it with raw medians.
    positive=bindings[bindings.old_positive_compute & bindings.old_all9_source_median_ns.gt(0)].copy()
    positive['phase']=positive.fine_key.str.split('|').str[0]
    group=['rank','phase','microbatch']
    shapes=positive.groupby(group)[['old_compute_ns','old_all9_source_median_ns']].sum()
    shapes['factor']=shapes.old_compute_ns/shapes.old_all9_source_median_ns
    positive=positive.join(shapes.factor,on=group)
    error=(positive.old_compute_ns-positive.old_all9_source_median_ns*positive.factor).abs()
    assert error.max()<1.0,'Legacy costs cannot be explained by one v67 group factor'
    eligible['phase']=eligible.fine_key.str.split('|').str[0]
    eligible=eligible.join(shapes.factor,on=group);assert eligible.factor.notna().all()
    shapes.to_csv(out/'inherited_runtime_shape_factors.csv')
    dump(out/'runtime_shape_check.json',{'groups':len(shapes),'nodes':len(positive),'max_reconstruction_error_ns':float(error.max()),'factor_min':float(shapes.factor.min()),'factor_max':float(shapes.factor.max()),'source':'v67 inherited redistribution recovered from source-fitted target prediction costs, not target observations'})
    contract=json.loads(paths['v685_contract'].read_text())['prediction']
    results=[];updated=[]
    for variant in spec['variants']:
        n=nodes.copy();index={id:i for i,id in enumerate(n.node_id)}
        selected=eligible[eligible.schema.eq('compute_overlap')] if variant=='fine_overlap_only' else eligible[~eligible.schema.eq('compute_overlap')] if variant=='fine_exposed_only' else eligible if variant=='fine_both' else eligible.iloc[:0]
        positions=selected.node_id.map(index).to_numpy()
        new=np.rint(selected.new_compute_ns.to_numpy()*selected.factor.to_numpy()).astype('int64')
        overlap=selected.schema.eq('compute_overlap').to_numpy()
        old_duration=n.loc[positions,'duration_ns'].to_numpy(dtype='int64')
        duration=np.where(overlap,new,np.maximum(old_duration,new))
        assert n.loc[positions,['network_service_ns_model','software_sync_ns_model']].to_numpy().sum()==0
        n.loc[positions,'compute_overlap_ns_model']=np.where(overlap,new,0)
        n.loc[positions,'compute_exposed_ns_model']=np.where(overlap,0,new)
        n.loc[positions,'framework_residual_ns_model']=np.where(overlap,0,duration-new)
        n.loc[positions,'duration_ns']=duration
        changes=selected[['node_id','fine_key','old_compute_ns','new_compute_ns','factor','old_duration_ns','rank','pp_stage','microbatch']].rename(columns={'fine_key':'source_key','new_compute_ns':'source_fit_compute_ns','factor':'inherited_shape_factor'}).copy()
        changes['variant']=variant;changes['new_compute_ns']=new;changes['new_duration_ns']=duration
        updated.extend(changes.to_dict('records'))
        assert ((n[COMP].sum(axis=1)-n.duration_ns)==0).all() and n[COMP].ge(0).all().all()
        n,critical=replay(n,edges,'iteration:completion_join')
        end=int(n.loc[n.node_id.eq('iteration:completion_join'),'predicted_end_ns'].iloc[0]);raw=end/1e6
        phase=n[n.kind.eq('phase_boundary')];f=phase[phase.phase.eq('FWD')&phase.op_name.eq('fwd_start')];b=phase[phase.phase.eq('BWD')&phase.op_name.eq('bwd_end')]
        entry=float(f.predicted_start_ns.min())/1e6;fb=float(b.predicted_end_ns.max())/1e6-entry
        profiler=raw+contract['target_reconciliation_ms'];training=profiler+contract['outer_framework_ms']
        mfu=100*8.436548311982576e16/(224*500e12*(training/1000))
        results.append(dict(variant=variant,raw_ms=raw,entry_ms=entry,onef1b_ms=fb,tail_ms=profiler-entry-fb,outer_ms=contract['outer_framework_ms'],profiler_ms=profiler,training_ms=training,mfu_pct=mfu,updated_nodes=len(selected),topology_sha256=lock))
        critical.to_csv(out/(variant+'_critical.csv'),index=False)
        n[['node_id','duration_ns','predicted_start_ns','predicted_end_ns']].to_csv(out/(variant+'_timings.csv.gz'),index=False,compression='gzip')
    assert abs(results[0]['raw_ms']-contract['target_raw_graph_ms'])<1e-6
    write_csv(out/'predictions.csv',results);write_csv(out/'node_updates.csv.gz',updated)
    dump(out/'prediction_seal.json',{'status':'SEALED_BEFORE_TARGET_EVALUATOR','topology':lock,'fit_iterations':[85,90],'inherited_source_iterations':'historical60–100; includes incremental validation','source_full_iteration_validated':False,'rule':spec['candidate_rule'],'files':{p.name:sha(p) for p in out.iterdir() if p.suffix in ['.csv','.gz']},'target_observation_read':False})
    print(json.dumps(results))

def review(spec,paths,out):
    verify_stage('audit');verify_stage('model')
    from review import review as run_review
    run_review(spec,paths,out)

def evaluate(spec,paths,out):
    verify_stage('model');verify_stage('review');seal=json.loads((RUN/'model/prediction_seal.json').read_text());assert seal['status']=='SEALED_BEFORE_TARGET_EVALUATOR'
    for p,d in seal['files'].items():assert sha(RUN/'model'/p)==d
    from scoring import score
    score(spec,paths,out)

if __name__=='__main__':
    stage=sys.argv[1];out=RUN/stage
    spec,paths,reads=setup(stage,out)
    try:globals()[stage](spec,paths,out)
    finally:dump(out/'input_access.json',{'stage':stage,'data_reads':sorted(reads),'raw_trace_reads':0})

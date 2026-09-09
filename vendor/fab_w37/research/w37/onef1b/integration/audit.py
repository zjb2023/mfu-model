"""Reconcile frozen graph counts, parameter updates and published scores."""
from collections import Counter, defaultdict
from statistics import mean, median
import hashlib, json, math, time
from common import *

started=time.monotonic()
verified=verify_inputs()
checks=[]
def check(name,condition,detail=''):
    if not condition:raise AssertionError(name+': '+str(detail))
    checks.append({'check':name,'passed':True,'detail':detail})
def close(name,a,b,tol=1e-6):check(name,abs(a-b)<=tol,{'actual':a,'expected':b,'tolerance':tol})

# Validate already sealed v685 artifacts used here against their original seal.
sealed={r['sha256'] for r in obj('v685_seal')['artifacts']}
for k in ['v685_nodes','v685_edges','compute_parameters','compute_updates','gradient_parameters','gradient_updates']:
    check('original_v685_seal:'+k,load_spec()['inputs'][k]['sha256'] in sealed)
contract=obj('v685_contract');pred=contract['prediction']
updates=rows('compute_updates');update_by={r['node_id']:r for r in updates}
node_types=[];edge_types=[];graph_summaries=[];update_positions=[]
for label,nkey,ekey in [('v685目标正式图','v685_nodes','v685_edges'),('v687目标阶段图','v687_nodes','v687_edges'),('T33源侧阶段图','T33_nodes','T33_edges'),('T34目标阶段图','T34_nodes','T34_edges')]:
    kinds=Counter();positive=Counter();ranks=set();lanes=set();stages=set();phase_count=0;total=0
    for r in iterrows(nkey):
        total+=1;kinds[r['kind']]+=1
        if int(r['duration_ns'])>0:positive[r['kind']]+=1
        for key,dest in [('rank',ranks),('pp_lane',lanes),('pp_stage',stages)]:
            value=int(r.get(key,'-1') or '-1')
            if value>=0:dest.add(value)
        if r['kind']=='phase_compute_communication_runtime':phase_count+=1
        if nkey=='v685_nodes' and r['node_id'] in update_by:
            u=update_by[r['node_id']]
            update_positions.append({**u,'rank':int(r['rank']),'pp_stage':int(r['pp_stage']),'pp_lane':int(r['pp_lane']),'kind':r['kind']})
            close('bound_compute_node:'+r['node_id'],int(r['duration_ns']),int(u['new_duration_ns']),0)
    edges=Counter();ntotal=0;tuples=[]
    for r in iterrows(ekey):
        ntotal+=1;edges[r['edge_type']]+=1
        if ekey=='v685_edges':tuples.append(tuple(r.get(k,'') for k in ['case_id','src','dst','edge_type','tensor_key','dependency_source']))
    if tuples:
        h=hashlib.sha256()
        for r in sorted(tuples):h.update(('\x1f'.join(r)+'\n').encode())
        check('formal_topology_exact',h.hexdigest()==contract['dependency_topology_sha256'],h.hexdigest())
        del tuples
    graph_summaries.append({'graph':label,'node_count':total,'edge_count':ntotal,'rank_count':len(ranks),'stage_count':len(stages),'lane_count':len(lanes),'phase_node_count':phase_count,'positive_duration_nodes':sum(positive.values()),'zero_duration_nodes':total-sum(positive.values())})
    node_types.extend({'graph':label,'kind':k,'count':v,'positive_duration':positive[k],'zero_duration':v-positive[k]} for k,v in kinds.items())
    edge_types.extend({'graph':label,'edge_type':k,'count':v} for k,v in edges.items())
check('v685_graph_count',graph_summaries[0]['node_count']==327746 and graph_summaries[0]['edge_count']==364784)
check('v687_all_ranks_not_lane0',graph_summaries[1]['rank_count']==224 and graph_summaries[1]['phase_node_count']==1344)
check('T33_all_ranks',graph_summaries[2]['rank_count']==256 and graph_summaries[2]['phase_node_count']==2048)
source_count=sum(1 for _ in iterrows('v685_source_nodes'))

parameters=rows('compute_parameters')
physical=[r for r in parameters if r['parameter_view']=='window_split' and r['cost_group']=='__physical_slot_total__' and r['autograd_phase']=='physical_mixed']
used_keys=sorted({r['source_parameter_key'] for r in updates})
check('compute_parameter_hierarchy',len(parameters)==360 and len(physical)==58 and len(used_keys)==3 and len(updates)==144)
check('compute_updates_only_first_last',set(r['pp_stage'] for r in update_positions)=={0,13})
compute_summary=[]
for key in used_keys:
    g=[r for r in update_positions if r['source_parameter_key']==key]
    new_values=sorted({int(r['steady_compute_ns']) for r in g})
    check('unique_fitted_cost:'+key,len(new_values)==1)
    for r in g:
        close('preserve_noncompute_floor:'+r['node_id'],int(r['new_duration_ns']),max(int(r['steady_compute_ns']),int(r['noncompute_floor_ns'])),0)
    compute_summary.append({'key':key,'nodes_updated':len(g),'stages':','.join(map(str,sorted({r['pp_stage'] for r in g}))),'rank_count':len({r['rank'] for r in g}),'old_compute_min_ms':min(int(r['old_compute_ns']) for r in g)/1e6,'old_compute_max_ms':max(int(r['old_compute_ns']) for r in g)/1e6,'new_compute_ms':new_values[0]/1e6,'summed_duration_delta_ms':sum(int(r['duration_delta_ns']) for r in g)/1e6})
gradients=rows('gradient_parameters');gupdates=rows('gradient_updates')
check('gradient_grid',len(gradients)==15*16*4 and len(gupdates)==13*16*3)
check('gradient_samples_four_each',all(int(r['sample_count'])==4 for r in gradients))
for r in gupdates:
    close('pp_time_account:'+r['target_node_id'],int(r['new_stage_aware_wall_ns']),int(r['network_service_ns'])+int(r['software_completion_ns']),0)
clock=rows('outer_clocks')
for r in clock:close('outer_clock:'+r['iteration'],int(r['training_log_ns'])-int(r['profiler_step_ns']),int(r['outer_residual_ns']),0)
close('outer_median',median(int(r['outer_residual_ns']) for r in clock)/1e6,pred['outer_framework_ms'])
close('reconciliation_source',pred['source_profiler_reference_median_ms']-pred['source_raw_graph_ms'],pred['source_reconciliation_ms'])
close('reconciliation_scale',pred['source_reconciliation_ms']*.75,pred['target_reconciliation_ms'])
close('profiler_total',pred['target_raw_graph_ms']+pred['target_reconciliation_ms'],pred['profiler_step_ms'])
close('training_total',pred['profiler_step_ms']+pred['outer_framework_ms'],pred['training_step_ms'])
close('mfu_formula',100*8.436548311982576e16/(224*500e12*(pred['training_step_ms']/1000)),pred['mfu_pct'])

iteration_rows=rows('version_iterations')
# The delivery has one baseline row per iteration; other variants must stay separate.
base=[r for r in iteration_rows if r['variant']=='v685_frozen' and r['split']=='development_primary']
check('baseline_four_iterations',len(base)==4 and {int(r['iteration']) for r in base}=={85,90,95,100})
for r in iteration_rows:
    for metric in ['profiler','training']:
        actual=float(r['actual_'+metric+'_ms']);prediction=float(r['predicted_'+metric+'_ms'])
        close('iteration_ape:'+r['variant']+r['iteration']+metric,abs(prediction-actual)/actual*100,float(r[metric+'_ape_pct']))
phase_rows=[r for r in rows('v685_phase_rows') if r['variant']=='v685_frozen' and r['split']=='development_primary']
phase_summary=[]
for phase in ['entry','onef1b','tail','outer']:
    g=[r for r in phase_rows if r['phase']==phase]
    phase_summary.append({'phase':phase,'predicted_ms':mean(float(r['predicted_ms']) for r in g),'actual_ms':mean(float(r['actual_ms']) for r in g),'underprediction_ms':mean(float(r['actual_ms'])-float(r['predicted_ms']) for r in g),'mape_pct':mean(float(r['ape_pct']) for r in g)})
close('phase_training_sum',sum(r['predicted_ms'] for r in phase_summary),pred['training_step_ms'])
for r in base:
    g=[x for x in phase_rows if x['iteration']==r['iteration']]
    close('phase_actual_sum:'+r['iteration'],sum(float(x['actual_ms']) for x in g),float(r['actual_training_ms']))

readiness=rows('readiness_parameters');lookup={(r['direction'],r['pp_lane']):r for r in readiness}
pp_rows=[r for r in iterrows('pp_local') if r['both_endpoints_single_message']=='True' and int(r['iteration']) in [85,90,95,100]]
pp_summary=[]
for r in pp_rows:
    p=lookup[r['direction'],r['pp_lane']];x=float(r['receiver_minus_sender_ms']);d=float(p['sender_effective_ready_ms']);c=float(p['post_ready_completion_ms'])
    if abs(max(d,x)-max(0,x)+c-float(r['predicted_postpublication_ms']))>1e-8:raise AssertionError('PP formula mismatch')
check('all_local_PP_predictions_verified',len(pp_rows)==6912)
for direction in ['F','B']:
    p=[r for r in readiness if r['direction']==direction]
    g=[r for r in pp_rows if r['direction']==direction and r['split']=='source_incremental_validation']
    pp_summary.append({'direction':direction,'parameter_lanes':len(p),'ready_mean_ms':mean(float(r['sender_effective_ready_ms']) for r in p),'completion_mean_ms':mean(float(r['post_ready_completion_ms']) for r in p),'validation_samples':len(g),'mae_ms':mean(abs(float(r['error_ms'])) for r in g),'v686_mae_ms':mean(abs(float(r['v686_error_ms'])) for r in g)})

lane=rows('lane_ledger');shapley=rows('lane_shapley')
for r in lane:
    effect=sum(float(x['attributed_delta_ms']) for x in shapley if x['iteration']==r['iteration'])
    close('lane_components:'+r['iteration'],effect,float(r['phase_component_effect_ms']))
    close('lane_total:'+r['iteration'],effect+float(r['remaining_schedule_pp_initial_skew_error_ms']),float(r['total_underprediction_ms']))
lane_summary={k:mean(float(r[k]) for r in lane) for k in lane[0] if k.endswith('_ms')}
lane_summary['remaining_mae_ms']=mean(abs(float(r['remaining_schedule_pp_initial_skew_error_ms'])) for r in lane)

online=rows('T38_iterations');online_metrics=rows('T38_metrics')
for summary in online_metrics:
    g=[r for r in online if r['method']==summary['method'] and (summary['split']=='all_including_initialization' or int(r['iteration'])!=85)]
    check('online_scope:'+summary['method']+summary['split'],len(g)==int(summary['iterations']))
    for metric,col in [('onef1b','onef1b_APE_pct'),('remaining','remaining_APE_pct'),('profiler','profiler_APE_pct'),('training','training_APE_pct'),('MFU_relative','MFU_relative_APE_pct')]:
        close('online_aggregate:'+summary['method']+summary['split']+metric,mean(float(r[col]) for r in g),float(summary[metric+'_MAPE_pct']))
for r in rows('T38_chain'):
    close('T38_addition:'+r['iteration'],float(r['base_predicted_onef1b_ms'])+float(r['lagged_base_residual_correction_ms']),float(r['predicted_onef1b_ms']))
    check('T38_previous_only:'+r['iteration'],not r['predecessor_iteration'] or float(r['predecessor_iteration'])<int(r['iteration']))
check('T38_other_phases_unchanged',all(abs(float(r['prediction_change_ms']))<1e-9 for r in rows('T38_ledger') if r['phase']!='onef1b'))
history=obj('historical_v682');historical_ledger=history['error_analysis']['additive_wall_clock_ledger'];components=historical_ledger['components']
fb=next(r for r in components if r['component']=='forward_backward_envelope')
close('historical_75_8_denominator',fb['underprediction_ms']/historical_ledger['reconstructed_underprediction_ms']*100,fb['error_share_pct'])

result={'graph_summaries':graph_summaries,'node_types':node_types,'edge_types':edge_types,'v685_source_node_count':source_count,'compute_summary':compute_summary,'compute_all_rows':len(parameters),'compute_physical_keys':len(physical),'compute_used_keys':len(used_keys),'compute_update_ranks':sorted({r['rank'] for r in update_positions}),'compute_total_node_delta_ms':sum(int(r['duration_delta_ns']) for r in updates)/1e6,'gradient_stats':{'rows':len(gradients),'updates':len(gupdates),'min_ms':min(int(r['gradient_wall_median_ns']) for r in gradients)/1e6,'max_ms':max(int(r['gradient_wall_median_ns']) for r in gradients)/1e6,'network_service_ms':sorted({int(r['network_service_ns'])/1e6 for r in gupdates})},'prediction':pred,'phase_summary':phase_summary,'pp_summary':pp_summary,'lane_summary':lane_summary,'historical_v684':obj('historical_v684'),'historical_v682_ledger':historical_ledger,'historical_v682_mape':history['metrics']['all_mape_pct'],'historical_v685_payload':json.loads(__import__('re').search(r'window.DAG_V685=(.*?);</script>',text('v685_html')).group(1)),'input_count':len(verified),'input_bytes':sum(r['bytes'] for r in verified),'check_count':len(checks)}
dump('audit.json',result);dump('input_verification.json',verified);dump('audit_checks.json',checks)
for name,data in [('graph_inventory.csv',graph_summaries),('graph_node_types.csv',node_types),('graph_edge_types.csv',edge_types),('compute_binding_summary.csv',compute_summary),('compute_node_updates.csv',update_positions),('compute_physical_parameters.csv',physical),('gradient_parameters.csv',gradients),('gradient_node_updates.csv',gupdates),('pp_direction_metrics.csv',pp_summary),('v685_phase_summary.csv',phase_summary),('version_iteration_results.csv',iteration_rows),('online_iteration_results.csv',online),('online_metrics.csv',online_metrics),('outer_source_clocks.csv',clock)]:write_csv(name,data)
print(json.dumps({'stage':'audit','status':'PASS','checks':len(checks),'inputs':len(verified),'input_bytes':sum(r['bytes'] for r in verified),'seconds':time.monotonic()-started}))

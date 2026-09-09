"""Explicit source-only fit, standalone seal, target evaluator and route stages."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
POST=ROOT/'research/w37/onef1b/post685'
sys.path[:0]=[str(POST),str(ROOT/'scripts/w37')]
from smoke_worker import SOURCE,CONTROL,InputGuard,verify,sha,dump
from worker import csv
from pp_semantics import align_observations
from pp_graph import build,envelope
from readiness import ReadinessCosts
from candidate import baseline,phase_frame,source_scores,partition_prediction,critical_ledger,KEY


from guards import StageGuard,checked_stage


def selected_items(plan):
    allitems=[]
    for p in [CONTROL/'external_inputs.json',ROOT/'docs/w37/1f1b/additional_inputs.json',ROOT/'docs/w37/1f1b/post685/inputs.json']:
        allitems+=json.loads(p.read_text())['files']
    names=['source_pp_trace_events_60_100.csv','source_pp_api_events_60_100.csv','source256_phase_noncommunication_windows.csv.gz',
           'source256_window_partition.csv.gz','source256_profiler_entry_60_100.csv','dag_v685_nodes.csv.gz','dag_v685_edges.csv.gz',
           'iteration_ground_truth.csv','dag_v682_stage_aware_pp_gradient_payload.json','target224_window_partition_ground_truth.csv.gz',
           'target224_outside_wrapper_ground_truth.csv.gz','target_phase_rank_events_60_100.csv','schedules.py','p2p_communication.py']
    items=[i for i in allitems if Path(i['path']).name in names or (Path(i['path']).name=='prediction_contract.json' and 'dag_v685_' in i['path'])]
    if plan.get('extra_input_manifest'):items+=json.loads((ROOT/plan['extra_input_manifest']).read_text())['files']
    if plan.get('resource_inputs_from_review'):
        review=json.loads((ROOT/plan['resource_review_file']).read_text())
        for key in plan['resource_inputs_from_review']:items+=review[key]
    assert len(items)==len({i['path'] for i in items});return items



def audit(out,paths):
    phase=pd.read_csv(paths['source_pp_trace_events_60_100.csv']);api=pd.read_csv(paths['source_pp_api_events_60_100.csv'])
    aligned,pairs=align_observations(phase,api)
    for name,data in [('source_actions.csv.gz',aligned),('source_message_pairs.csv.gz',pairs),('source_phases.csv.gz',phase)]:csv(out,name,data)
    gpu=pd.read_csv(paths['source256_phase_noncommunication_windows.csv.gz']);wrapper=pd.read_csv(paths['source256_window_partition.csv.gz'])
    cols=['phase_noncommunication_only_ms','phase_communication_only_ms','phase_comm_noncommunication_overlap_ms','phase_gpu_idle_ms']
    assert (gpu[cols].sum(axis=1)-gpu.phase_duration_ms).abs().max()<1e-6
    assert (wrapper.fused_prelaunch_sum_ms+wrapper.fused_postlaunch_sum_ms+wrapper.outside_fused_wrapper_ms-wrapper.phase_duration_ms).abs().max()<1e-6
    csv(out,'source_gpu_partition.csv',gpu);csv(out,'source_wrapper_partition.csv',wrapper)
    # Source evidence to guide later hypotheses; no target profile is read here.
    csv(out,'source_stage_cost_profile.csv',gpu[gpu.iteration.isin([85,90])].groupby(['pp_stage','phase'])[cols+['phase_duration_ms']].mean().reset_index())
    from models import source_evidence
    evidence,coverage=source_evidence(aligned)
    csv(out,'source_held_stage_results.csv.gz',evidence);csv(out,'source_held_stage_unsupported.csv',coverage)
    csv(out,'source_held_stage_metrics.csv',evidence.groupby(['method','phase']).agg(samples=('error_ms','size'),mae_ms=('error_ms',lambda x:x.abs().mean()),
        bias_ms=('error_ms','mean'),mape_pct=('ape_pct','mean')).reset_index())
    if 'ep_source_ep_rank_anchor_events.csv' in paths:
        from ep_model import prepare_source
        prepare_source(out,paths,aligned,pairs,phase)
    dump(out/'source_audit.json',{'status':'PASS','actions':len(aligned),'message_pairs':len(pairs),'gpu_rows':len(gpu),'raw_trace_scanned':False})


def model(out,run,name,spec,paths):
    if spec['method'].startswith('ep_'):
        from ep_model import run_candidate
        return run_candidate(out,run,name,spec,paths)
    data=run/'source_audit';checked_stage(data)
    aligned=pd.read_csv(data/'source_actions.csv.gz');pairs=pd.read_csv(data/'source_message_pairs.csv.gz');phase=pd.read_csv(data/'source_phases.csv.gz')
    gpu=pd.read_csv(data/'source_gpu_partition.csv');wrapper=pd.read_csv(data/'source_wrapper_partition.csv')
    boundaries=pd.read_csv(paths['source256_profiler_entry_60_100.csv'])
    br,bp,contract=baseline(paths);records=[br];phases=[bp];partitions=[];critical=[];sourcegraphs={}
    cls=ReadinessCosts
    if spec['method']!='v687_control':
        from models import cost_class
        cls=cost_class(spec)
    fit=spec['source_fit_iterations']
    for pp,mb,case in [(16,4,'source256'),(14,3,'target224')]:
        costs=cls(aligned,pairs,fit,pp,mb,role_transfer=spec.get('role_transfer',True))
        nodes,edges,topology=build(pp,mb,costs,variant=spec.get('graph_ablation','full'))
        csv(out,f'prediction/{name}/{case}_nodes.csv.gz',nodes);csv(out,f'prediction/{name}/{case}_edges.csv.gz',edges)
        csv(out,f'prediction/{name}/{case}_runtime_bindings.csv.gz',pd.DataFrame(costs.bindings))
        csv(out,f'prediction/{name}/{case}_readiness_parameters.csv',pd.DataFrame(costs.parameter_rows))
        dump(out/f'prediction/{name}/{case}_contract.json',dict(variant=name,method=spec['method'],case=case,source_fit=fit,
            topology_sha256=topology,formal_topology_replaced=False,envelope=envelope(nodes),spec=spec,
            data_boundary='source calibration only; target historical exposure remains development evaluation'))
        critical+=critical_ledger(nodes,name,case)
        if hasattr(costs,'phase_samples'):
            from models import partition_prediction as semantic_partition
            partitions+=semantic_partition(gpu,wrapper,costs,pp,mb,name)
            csv(out,f'prediction/{name}/{case}_phase_cost_bindings.csv',pd.DataFrame(costs.phase_bindings))
        else:partitions+=partition_prediction(gpu,wrapper,costs,pp,mb,name)
        if pp==16:
            sourcegraphs[name]=(nodes,fit);lv=costs.local_validation(pairs)
            csv(out,'source_validation/pp_local_results.csv.gz',lv)
        else:
            ph=phase_frame(nodes);ph['variant']=name;ph['start_ms']-=envelope(nodes)['first_phase_ns']/1e6;ph['end_ms']-=envelope(nodes)['first_phase_ns']/1e6;phases.append(ph)
            rec={**br,'variant':name,'onef1b_ms':envelope(nodes)['onef1b_ms']}
            rec['profiler_ms']=rec['entry_ms']+rec['onef1b_ms']+rec['tail_ms'];rec['training_ms']=rec['profiler_ms']+rec['outer_ms'];records.append(rec)
    source_scores(out,sourcegraphs,aligned,phase,boundaries)
    csv(out,'prediction/phase_nodes.csv.gz',pd.concat(phases,ignore_index=True)[['variant']+KEY+['start_ms','end_ms','duration_ms']])
    csv(out,'prediction/internal_partition.csv',pd.DataFrame(partitions));csv(out,'prediction/critical_path_ledger.csv',pd.DataFrame(critical))
    dump(out/'prediction/step_predictions.json',records);dump(out/'prediction/frozen_baseline_contract.json',contract)


def seal(out,modeldir):
    checked_stage(modeldir)
    files=[dict(path=str(p.relative_to(modeldir)),sha256=sha(p)) for folder in ['prediction','source_validation'] for p in (modeldir/folder).rglob('*') if p.is_file()]
    dump(out/'prediction_seal.json',{'status':'SEALED_SOURCE_ONLY_PREDICTION','model_dir':str(modeldir),'files':files,
        'model_manifest_sha256':sha(modeldir/'run_manifest.json'),'target_parameter_updates':0,'target_timing_read_before_seal':False,
        'sealed_utc':datetime.now(timezone.utc).isoformat(),'target_scope':'known development data; no independent blind claim'})


def evaluate(out,run,name,paths,guard):
    modeldir=run/'models'/name;sealdir=run/'seals'/name;checked_stage(sealdir)
    sealed=json.loads((sealdir/'prediction_seal.json').read_text());assert sha(modeldir/'run_manifest.json')==sealed['model_manifest_sha256']
    for item in sealed['files']:assert sha(modeldir/item['path'])==item['sha256']
    (out/'prediction').symlink_to(modeldir/'prediction',target_is_directory=True)
    records=json.loads((modeldir/'prediction/step_predictions.json').read_text());parts=pd.read_csv(modeldir/'prediction/internal_partition.csv').to_dict('records')
    guard.phase='evaluator'
    from scoring import evaluate as score
    metrics=score(out,records,paths,parts)
    for item in sealed['files']:assert sha(modeldir/item['path'])==item['sha256']
    dump(out/'evaluation.json',{'status':'PASS','metrics':metrics,'prediction_seal_sha256':sha(sealdir/'prediction_seal.json')})


def summarize(out,run,plan):
    rows=[];routes=[]
    for name,spec in plan['variants'].items():
        ev=run/'evaluations'/name;checked_stage(ev)
        metrics=pd.read_csv(ev/'evaluator_only/metrics.csv');phase=pd.read_csv(ev/'evaluator_only/phase_metrics.csv')
        source=pd.read_csv(run/'models'/name/'source_validation/metrics.csv')
        main=metrics[metrics.variant.eq(name)&metrics.split.eq('development_primary')].iloc[0]
        fb=phase[phase.variant.eq(name)&phase.split.eq('development_primary')&phase.phase.eq('onef1b')].iloc[0]
        base=phase[phase.variant.eq('v685_frozen')&phase.split.eq('development_primary')&phase.phase.eq('onef1b')].iloc[0]
        sv=source[source.variant.eq(name)&source.split.eq('source_incremental_validation')]
        row=dict(variant=name,method=spec['method'],target_onef1b_mape_pct=float(fb.mape_pct),baseline_onef1b_mape_pct=float(base.mape_pct),
                 delta_onef1b_mape_pp=float(fb.mape_pct-base.mape_pct),target_profiler_mape_pct=float(main.profiler_mape_pct),
                 target_training_mape_pct=float(main.training_mape_pct),mfu_relative_mape_pct=float(main.mfu_relative_mape_pct),mfu_bias_pp=float(main.mfu_bias_pp),
                 source_incremental_onef1b_mape_pct=float(sv.onef1b_mape_pct.mean()),
                 target_improved=bool(fb.mape_pct<base.mape_pct),promotion='NOT_PROMOTED_REQUIRES_EVIDENCE_REVIEW')
        rows.append(row);routes.append(dict(wave=plan['wave'],variant=name,hypothesis=spec['hypothesis'],source_fit=spec['source_fit_iterations'],
            evidence=spec['evidence'],result=row,sealed_prediction=str(run/'seals'/name/'prediction_seal.json'),evaluated_utc=datetime.now(timezone.utc).isoformat()))
    if plan.get('diagnostic'):
        checked_stage(run/'diagnose')
        result=json.loads((run/'diagnose/diagnostic.json').read_text())
        kind='source_only_local_prediction' if plan.get('diagnostic_access')=='source_only' and result.get('new_prediction') else ('source_only_evidence_NOT_prediction' if plan.get('diagnostic_access')=='source_only' else 'posthoc_diagnostic_NOT_prediction')
        if result.get('conditional_device_prediction'):kind='posthoc_CPU_conditioned_device_prediction_NOT_global1F1B'
        routes.append(dict(wave=plan['wave'],kind=kind,hypothesis=plan['hypothesis'],result=result,evidence=str(run/'diagnose')))
    csv(out,'leaderboard.csv',pd.DataFrame(rows) if rows else pd.DataFrame(columns=['variant','target_onef1b_mape_pct','promotion']));dump(out/'route.json',routes)
    lines=['# '+plan['wave'],'','Development evaluation only; formal topology unchanged.','']
    for row in rows:lines.append(f"- {row['variant']}: 224 1F1B MAPE {row['target_onef1b_mape_pct']:.6f}% (baseline {row['baseline_onef1b_mape_pct']:.6f}%; delta {row['delta_onef1b_mape_pp']:+.6f} pp); source incremental {row['source_incremental_onef1b_mape_pct']:.6f}%; {row['promotion']}.")
    if plan.get('diagnostic'):lines+=['- Evidence review: '+plan['diagnostic']+'. Prediction scope is recorded in diagnostic.json; no target cost fit.']
    (out/'ROUTE.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--variant',default='');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();run=a.run_root.resolve();plan=json.loads(a.spec.read_text());items=selected_items(plan)
    assert os.statvfs(SOURCE).f_flag&os.ST_RDONLY and os.statvfs(ROOT).f_flag&os.ST_RDONLY
    allowed=[run/'preflight']
    if a.stage=='model':allowed.append(run/'source_audit')
    if a.stage in ['seal','evaluate']:allowed.append(run/'models'/a.variant)
    if a.stage=='evaluate':allowed.append(run/'seals'/a.variant)
    if a.stage=='diagnose':
        ref=plan['sealed_reference'];prior=Path(ref['run_root']).resolve();assert prior.is_relative_to(ROOT/'results/w37/A')
        allowed+=[prior/'models'/ref['variant'],prior/'seals'/ref['variant']]
        for name in plan.get('review_variants',[]):allowed+=[prior/'models'/name,prior/'seals'/name,prior/'evaluations'/name]
        for ref in plan.get('sealed_diagnostic_stages',[]):
            stage=Path(ref['stage_root']).resolve();acceptance=Path(ref['acceptance_path']).resolve()
            assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
            assert acceptance.is_relative_to(ROOT/'results/w37/A') and acceptance.name=='acceptance.json'
            allowed += [stage,acceptance]
            if ref.get('pipeline_command_path'):
                command=Path(ref['pipeline_command_path']).resolve()
                assert command.is_relative_to(ROOT/'results/w37/A') and command.name=='pipeline_command.json'
                allowed.append(command)
    if a.stage=='summary':allowed=[run]
    raw_parse_paths=[];raw_probe_path=None
    if plan.get('raw_trace_intake'):
        intake=plan['raw_trace_intake'];assert plan['diagnostic'] in ['source_runtime_intake','source_graph_flow_intake','source_counter_interval_probe'] and plan['diagnostic_access']=='source_only'
        assert sha(ROOT/plan['resource_review_file'])==plan['resource_review_sha256']
        raw=[i for i in items if i.get('raw_trace')];assert {i['key'] for i in raw}==set(intake['keys'])
        assert len(raw)<=intake['max_files'] and sum(i['size_bytes'] for i in raw)<=intake['max_total_bytes']
        for i in raw:
            assert i['source_case']==intake['source_case']=='source256' and i['rank'] in intake['allowed_source_ranks'] and i['iteration'] in intake['allowed_iterations']
        if plan['diagnostic'] in ['source_graph_flow_intake','source_counter_interval_probe']:
            review=json.loads((ROOT/plan['resource_review_file']).read_text())
            assert len(raw)==intake['max_files']==review['max_files']==1
            assert intake['max_total_bytes']<=review['max_total_bytes'] and review['source_only'] and not review['raw_target_access']
            for i in raw:
                original=next(r for r in review['raw_files'] if r['path']==i['path'])
                assert all(i[k]==original[k] for k in ['size_bytes','sha256','rank','iteration'])
            if plan['diagnostic']=='source_counter_interval_probe':
                assert raw[0]['raw_kind']=='hardware_counter' and raw[0]['iteration']==85
                assert all(raw[0][k]==review['raw_files'][0][k] for k in ['host','gpu_id','selected_start_ns','selected_end_ns'])
                prerequisite=review['prerequisite']
                assert any(r['stage_root']==prerequisite['stage_root'] and r['manifest_sha256']==prerequisite['manifest_sha256']
                           and r.get('evidence_type')=='source_counter_admission' for r in plan['source_evidence_stages'])
                accepted=next(i for i in items if i.get('key')=='source_counter_prerequisite_acceptance.json')
                assert accepted['path']==prerequisite['acceptance_path'] and accepted['sha256']==prerequisite['acceptance_sha256']
                assert accepted['roles']==['source_only_counter_prerequisite_not_cost'] and not accepted.get('raw_trace')
                allowed.append(Path(accepted['path']))
        if len(raw)>1:
            probe_item=next(i for i in items if i.get('key')==plan['first_file_probe_key'])
            assert not probe_item.get('raw_trace') and probe_item['roles']==['source_only_raw_intake_probe_gate_not_cost_input']
            raw_probe_path=Path(probe_item['path']).resolve();assert raw_probe_path.is_relative_to(ROOT/'results/w37/A')
            allowed.append(raw_probe_path)
        if a.stage=='diagnose':raw_parse_paths=[i['path'] for i in raw]
    from deployment_intake import source_log_capabilities
    raw_parse_paths+=source_log_capabilities(plan,items,a.stage)
    from source_evidence import capabilities,verify_source_stages
    source_files,hash_only=capabilities(plan,items);allowed+=source_files
    from source_model_evidence import capabilities as model_evidence_capabilities,verify_source_models
    model_files,model_hash_only=model_evidence_capabilities(plan,items)
    allowed+=model_files;hash_only+=model_hash_only
    from evaluator_evidence import capabilities as evaluator_capabilities,verify_evaluator_stages
    evaluator_files,evaluator_hash_only=evaluator_capabilities(plan,items)
    allowed+=evaluator_files;hash_only+=evaluator_hash_only
    for ref in plan.get('sealed_diagnostic_stages',[]):
        hash_only += [Path(ref['stage_root']).resolve(),Path(ref['acceptance_path']).resolve()]
        if ref.get('pipeline_command_path'):hash_only.append(Path(ref['pipeline_command_path']).resolve())
    item_paths={str(Path(i['path']).resolve()) for i in items}
    for value in plan.get('additional_hash_only_result_paths',[]):
        path=Path(value).resolve()
        assert path.is_relative_to(ROOT/'results/w37/A') and str(path) in item_paths
        hash_only.append(path)
    from target_runtime import target_intake_capabilities,verify_target_probe
    target_raw_paths,target_probe=target_intake_capabilities(plan,items,a.stage)
    if target_probe:allowed.append(target_probe)
    from target_deployment import target_log_capabilities
    target_log_paths,target_log_prerequisites=target_log_capabilities(plan,items,a.stage)
    target_raw_paths+=target_log_paths;allowed+=target_log_prerequisites
    guard=StageGuard(out,items,a.stage,allowed,raw_parse_paths=raw_parse_paths,hash_only_results=hash_only,target_raw_parse_paths=target_raw_paths);sys.addaudithook(guard.event);guard.phase='hash_preflight';verify(items)
    paths={i.get('key',Path(i['path']).name):Path(i['path']) for i in items}
    for i in json.loads((CONTROL/'code_manifest.json').read_text())['files']:assert sha(ROOT/i['path'])==i['sha256']
    guard.phase='source_only' if a.stage!='summary' else 'summary_of_sealed_evaluations'
    verify_source_stages(plan,items,guard)
    verify_source_models(plan,items,guard)
    if plan.get('diagnostic')=='source_counter_interval_probe':
        accepted=json.loads(paths['source_counter_prerequisite_acceptance.json'].read_text())
        assert accepted['status']=='PASS_SOURCE_COUNTER_GRANULARITY_AND_PROVENANCE_REVIEW'
        assert accepted['prior_global_seal_unchanged'] and accepted['T19_graph_bracket_fields_exact_except_split_label']
    if raw_probe_path is not None:
        probe=json.loads(raw_probe_path.read_text())
        assert probe['status']=='SOURCE_RAW_RUNTIME_INTAKE_PASS' and probe['raw_files']==1 and probe['new_target_timing_read'] is False
        assert probe['peak_RSS_bytes']<=1024*1024*1024 and all(c['all_boundaries_exact_ns'] for c in probe['checks'])
        assert probe['raw_source_ranks']==intake['allowed_source_ranks']
        probe_iteration=probe['raw_source_iterations'][0]
        assert next(i for i in raw if i['iteration']==probe_iteration)['sha256']==plan['first_file_probe_input_sha256']
    if a.stage=='preflight':dump(out/'frozen_inputs.json',{'files':items,'target_access':'hash verification only','plan':plan})
    elif a.stage=='source_audit':checked_stage(run/'preflight');audit(out,paths)
    elif a.stage=='model':model(out,run,a.variant,plan['variants'][a.variant],paths)
    elif a.stage=='seal':seal(out,run/'models'/a.variant)
    elif a.stage=='evaluate':evaluate(out,run,a.variant,paths,guard)
    elif a.stage=='diagnose':
        checked_stage(run/'preflight');ref=plan['sealed_reference'];prior=Path(ref['run_root'])
        sealdir=prior/'seals'/ref['variant'];modeldir=prior/'models'/ref['variant']
        checked_stage(sealdir);checked_stage(modeldir)
        assert sha(sealdir/'prediction_seal.json')==ref['seal_sha256']
        for item in json.loads((sealdir/'prediction_seal.json').read_text())['files']:assert sha(modeldir/item['path'])==item['sha256']
        guard.sealed_reference_verified=True
        mode=plan.get('diagnostic_access','evaluator');assert mode in ['source_only','evaluator'];guard.phase=mode
        if target_probe:verify_target_probe(plan,items,target_probe)
        verify_evaluator_stages(plan,items,guard)
        if plan['diagnostic']=='final_evidence_chain_integrity':
            from final_integrity import diagnose
        elif plan['diagnostic']=='operational_decision_index':
            from operational_index import diagnose
        elif plan['diagnostic']=='cross_domain_state_tolerance_audit':
            from cross_domain_state import diagnose
        elif plan['diagnostic']=='fixed_state_robustness_audit':
            from state_robustness import diagnose
        elif plan['diagnostic']=='causal_lagged_target_update':
            from causal_lagged_update import diagnose
        elif plan['diagnostic']=='remaining_time_pareto_audit':
            from remaining_time_audit import diagnose
        elif plan['diagnostic']=='early_prefix_reliability_audit':
            from early_prefix import diagnose
        elif plan['diagnostic']=='online_prefix_runtime_audit':
            from online_prefix import diagnose
        elif plan['diagnostic']=='schedule_contraction_audit':
            from schedule_contraction import diagnose
        elif plan['diagnostic']=='source_phase_replay_audit':
            from source_phase_replay import diagnose
        elif plan['diagnostic']=='source_gap_bridge_audit':
            from source_gap_bridge import diagnose
        elif plan['diagnostic']=='source_fullrank_cost_split':
            from fullrank_cost_split import diagnose
        elif plan['diagnostic']=='target_terminal_endpoint_transfer':
            from target_terminal_endpoint import diagnose
        elif plan['diagnostic']=='source_CPU_submission_identifiability':
            from cpu_submission_identifiability import diagnose
        elif plan['diagnostic']=='source_terminal_decomposition':
            from terminal_decomposition import diagnose
        elif plan['diagnostic']=='source_terminal_endpoint':
            from terminal_endpoint import diagnose
        elif plan['diagnostic']=='target_device_release_transfer':
            from device_release_transfer import diagnose
        elif plan['diagnostic']=='source_API_start_release':
            from device_release import diagnose
        elif plan['diagnostic']=='target_device_queue_transfer':
            from device_queue_transfer import diagnose
        elif plan['diagnostic']=='source_device_queue_reload':
            from device_queue_reload import diagnose
        elif plan['diagnostic']=='target_deployment_intake':
            from target_deployment import diagnose
        elif plan['diagnostic']=='source_deployment_intake':
            from deployment_intake import diagnose
        elif plan['diagnostic']=='source_counter_clock_review':
            from counter_clock import diagnose
        elif plan['diagnostic']=='source_counter_interval_probe':
            from counter_intervals import diagnose
        elif plan['diagnostic']=='source_counter_admission':
            from counter_admission import diagnose
        elif plan['diagnostic']=='source_graph_flow_review':
            from graph_flow_review import diagnose
        elif plan['diagnostic']=='source_graph_flow_intake':
            from graph_flow_intake import diagnose
        elif plan['diagnostic']=='source_device_queue_audit':
            from device_queue import diagnose
        elif plan['diagnostic']=='target_device_gaps':
            from target_device_gaps import diagnose
        elif plan['diagnostic']=='source_device_gaps':
            from device_gaps import diagnose
        elif plan['diagnostic']=='target_runtime_intake':
            from target_runtime import diagnose
        elif plan['diagnostic']=='source_checkpoint_tail':
            from checkpoint_tail import diagnose
        elif plan['diagnostic']=='source_runtime_pending':
            from runtime_pending import diagnose
        elif plan['diagnostic']=='source_runtime_intake':
            from runtime_intake import diagnose
        elif plan['diagnostic']=='cp_source_readiness_audit':
            from cp_readiness import diagnose
        elif plan['diagnostic']=='cp_source_alignment':
            from cp_alignment import diagnose
        elif plan['diagnostic']=='cp_source_intake':
            from cp_intake import diagnose
        elif plan['diagnostic']=='ep_pending_work_audit':
            from pending_work import diagnose
        elif plan['diagnostic']=='ep_source_completion_audit':
            from ep_completion_audit import diagnose
        elif plan['diagnostic']=='ep_scenario_review':
            from ep_scenario_review import diagnose
        elif plan['diagnostic']=='ep_prediction_review':
            from ep_review import diagnose
        elif plan['diagnostic'].startswith('ep_'):
            from ep_boundary import diagnose
        else:
            from wrapper_boundary import diagnose
        diagnose(out,paths,plan)
    elif a.stage=='summary':summarize(out,run,plan)
    raw_parsed=[k for k,v in guard.reads.items() if guard.items[k].get('raw_trace') and set(v['phases'])-{'hash_preflight'}]
    raw_counters=[p for p in raw_parsed if guard.items[p].get('raw_kind')=='hardware_counter']
    raw_logs=[p for p in raw_parsed if guard.items[p].get('raw_kind')=='training_log']
    raw_profiler=[p for p in raw_parsed if p not in raw_counters and p not in raw_logs]
    dump(out/'input_access_audit.json',{'stage':a.stage,'raw_trace_scanned':bool(raw_profiler),'raw_trace_parsed_paths':raw_profiler,
        'raw_hardware_counter_scanned':bool(raw_counters),'raw_hardware_counter_parsed_paths':raw_counters,
        'raw_training_log_prefix_scanned':bool(raw_logs),'raw_training_log_prefix_parsed_paths':raw_logs,
        'sealed_reference_verified':guard.sealed_reference_verified,'reads':[dict(path=k,roles=v['roles'],phases=sorted(v['phases'])) for k,v in guard.reads.items()]})


if __name__=='__main__':main()

"""Admission of an explicitly sealed source-fitted local model, never target fit."""
import json
from pathlib import Path
import pandas as pd
from guards import ROOT,checked_stage
from smoke_worker import sha

ROLES={
    'complete.json':'source_only_local_model_provenance',
    'run_manifest.json':'source_only_local_model_provenance',
    'input_access_audit.json':'source_only_local_model_provenance',
    'diagnostic.json':'source_only_local_model_provenance',
    'source_device_queue_prediction_seal.json':'source_only_local_model_provenance',
    'source_device_queue_parameters.csv.gz':'source85_90_fitted_local_device_queue_parameters',
    'source_device_queue_features.csv.gz':'source_CPU_conditioned_model_features',
    'source_device_queue_predictions_sealed.csv.gz':'source_sealed_CPU_conditioned_predictions',
    'source_device_queue_endpoint_metrics.csv':'source_incremental_local_device_metrics',
}
RELOAD_FILES={
    'source_reload_complete.json':'complete.json',
    'source_reload_run_manifest.json':'run_manifest.json',
    'source_reload_input_access_audit.json':'input_access_audit.json',
    'source_reload_diagnostic.json':'diagnostic.json',
    'source_reload_seal.json':'source_device_queue_reload_seal.json',
}
RELEASE_FILES={
    'source_release_complete.json':('complete.json','source_only_local_model_provenance'),
    'source_release_run_manifest.json':('run_manifest.json','source_only_local_model_provenance'),
    'source_release_input_access_audit.json':('input_access_audit.json','source_only_local_model_provenance'),
    'source_release_diagnostic.json':('diagnostic.json','source_only_local_model_provenance'),
    'source_release_source_release_comparison_seal.json':('source_release_comparison_seal.json','source_only_local_model_provenance'),
    'source_release_acceptance.json':('../acceptance.json','source_only_local_model_provenance'),
    'legacy_API_end_source_device_queue_parameters.csv.gz':('legacy_API_end/source_device_queue_parameters.csv.gz','source85_90_fitted_release_candidate_parameters'),
    'legacy_API_end_source_device_queue_prediction_seal.json':('legacy_API_end/source_device_queue_prediction_seal.json','source_only_local_model_provenance'),
    'legacy_API_end_source_device_queue_endpoint_metrics.csv':('legacy_API_end/source_device_queue_endpoint_metrics.csv','source_incremental_local_device_metrics'),
    'all_API_start_source_device_queue_parameters.csv.gz':('all_API_start/source_device_queue_parameters.csv.gz','source85_90_fitted_release_candidate_parameters'),
    'all_API_start_source_device_queue_prediction_seal.json':('all_API_start/source_device_queue_prediction_seal.json','source_only_local_model_provenance'),
    'all_API_start_source_device_queue_endpoint_metrics.csv':('all_API_start/source_device_queue_endpoint_metrics.csv','source_incremental_local_device_metrics'),
}
TERMINAL_ROLE='source_only_terminal_model_provenance'
TERMINAL_PARAMETER_ROLE='source_only_terminal_model_parameters'
TERMINAL_FILES={
    'source_terminal_complete.json':('complete.json',TERMINAL_ROLE),
    'source_terminal_run_manifest.json':('run_manifest.json',TERMINAL_ROLE),
    'source_terminal_input_access_audit.json':('input_access_audit.json',TERMINAL_ROLE),
    'source_terminal_diagnostic.json':('diagnostic.json',TERMINAL_ROLE),
    'source_terminal_acceptance.json':('../acceptance.json',TERMINAL_ROLE),
    'source_terminal_seal.json':('source_terminal_endpoint_prediction_seal.json',TERMINAL_PARAMETER_ROLE),
    'source_terminal_identity_parameters.csv':('source_terminal_identity_parameters.csv',TERMINAL_PARAMETER_ROLE),
    'source_terminal_candidate_event_selection.csv.gz':('source_terminal_candidate_event_selection.csv.gz',TERMINAL_PARAMETER_ROLE),
    'source_terminal_selection_parameters.csv':('source_terminal_endpoint_selection_parameters.csv',TERMINAL_PARAMETER_ROLE),
    'source_phase_tail_parameters.csv':('source_phase_tail_parameters.csv',TERMINAL_PARAMETER_ROLE),
    'source_terminal_device_cost_parameters.csv.gz':('source_terminal_device_cost_parameters.csv.gz',TERMINAL_PARAMETER_ROLE),
    'source_terminal_predictions_sealed.csv.gz':('source_terminal_endpoint_predictions_sealed.csv.gz',TERMINAL_PARAMETER_ROLE),
}


def capabilities(plan,items):
    refs=plan.get('source_model_stages',[])
    release=plan.get('source_release_stage')
    terminal=plan.get('source_terminal_stage')
    if not refs and not release and not terminal:return [],[]
    assert sum(bool(x) for x in [refs,release,terminal])==1
    if terminal:
        assert (plan['diagnostic'],plan['diagnostic_access']) in [
            ('target_terminal_endpoint_transfer','evaluator'),
            ('source_terminal_decomposition','source_only'),
        ]
        assert not plan['variants']
        assert set(terminal['input_keys'])==set(TERMINAL_FILES)
        stage=Path(terminal['stage_root']).resolve()
        assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
        bykey={i.get('key'):i for i in items};allowed=[]
        for key,(relative,role) in TERMINAL_FILES.items():
            item=bykey[key];path=Path(item['path']).resolve()
            assert path==(stage/relative).resolve() and item['roles']==[role] and not item.get('raw_trace')
            allowed.append(path)
        return allowed,[stage]
    if release:
        assert not refs and plan['diagnostic']=='target_device_release_transfer'
        assert not plan['variants'] and plan['diagnostic_access']=='evaluator'
        assert set(release['input_keys'])==set(RELEASE_FILES)
        stage=Path(release['stage_root']).resolve()
        assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
        bykey={i.get('key'):i for i in items};allowed=[]
        for key,(relative,role) in RELEASE_FILES.items():
            item=bykey[key];path=Path(item['path']).resolve()
            assert path==(stage/relative).resolve() and item['roles']==[role] and not item.get('raw_trace')
            allowed.append(path)
        return allowed,[stage]
    assert plan['diagnostic'] in ['source_device_queue_reload','target_device_queue_transfer']
    assert not plan['variants'] and plan['diagnostic_access'] in ['source_only','evaluator']
    assert not any(i.get('raw_trace') for i in items),'This model reuse permits cached inputs only'
    allowed=[];hash_only=[];bykey={i.get('key'):i for i in items}
    for ref in refs:
        stage=Path(ref['stage_root']).resolve()
        assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
        assert ref['fit_iterations']==[85,90] and ref['incremental_validation_iterations']==[95,100]
        assert ref['source_case']=='source256' and ref['source_rank']==16
        assert set(ref['input_keys'])==set(ROLES)
        hash_only.append(stage)
        for key in ref['input_keys']:
            item=bykey[key];path=Path(item['path']).resolve()
            assert path==stage/key and item['roles']==[ROLES[key]] and not item.get('raw_trace')
            allowed.append(path)
    gate=plan.get('source_model_reload_gate')
    if gate:
        assert plan['diagnostic']=='target_device_queue_transfer' and plan['diagnostic_access']=='evaluator'
        stage=Path(gate['stage_root']).resolve();assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
        hash_only.append(stage)
        for key,name in RELOAD_FILES.items():
            item=bykey[key];path=Path(item['path']).resolve()
            assert path==stage/name and item['roles']==['source_only_local_model_provenance'] and not item.get('raw_trace')
            allowed.append(path)
        item=bykey['source_reload_acceptance.json'];path=Path(item['path']).resolve()
        assert str(path)==gate['acceptance_path'] and item['sha256']==gate['acceptance_sha256']
        assert path==stage.parent/'acceptance.json' and item['roles']==['source_only_local_model_provenance'] and not item.get('raw_trace')
        allowed.append(path)
    return allowed,hash_only


def verify_source_models(plan,items,guard):
    for ref in plan.get('source_model_stages',[]):
        stage=Path(ref['stage_root']);phase=guard.phase;guard.phase='hash_preflight'
        try:
            assert sha(stage/'run_manifest.json')==ref['manifest_sha256'];checked_stage(stage)
            for module in ref['source_modules']:
                p=Path(module['path']);snapshot=Path(module['source_snapshot'])
                assert not p.is_absolute() and '..' not in p.parts
                assert snapshot==stage/'code_snapshot'/(p.parent.name+'__'+p.name)
                assert sha(ROOT/p)==sha(snapshot)==module['sha256']
        finally:guard.phase=phase
        diagnostic=json.loads((stage/'diagnostic.json').read_text())
        assert diagnostic['status']=='SOURCE_DEVICE_QUEUE_LOCAL_PREDICTION_PASS'
        assert diagnostic['new_prediction'] and diagnostic['used_to_fit_model'] and not diagnostic['new_target_timing_read']
        assert diagnostic['source_iterations']==[85,90,95,100] and diagnostic['source_ranks']==[16]
        assert diagnostic['visible_queue_structural_gate'] and not diagnostic['formal_topology_changed']
        assert diagnostic['graph_directly_correlated_device_events']==0
        audit=json.loads((stage/'input_access_audit.json').read_text())
        assert audit['stage']=='diagnose' and not audit['raw_trace_scanned']
        for read in audit['reads']:
            if 'evaluator' in '|'.join(read['roles']) or '/evaluator_only/' in read['path']:
                assert set(read['phases'])=={'hash_preflight'},'Source model had target semantic access'
        sealpath=stage/'source_device_queue_prediction_seal.json'
        assert sha(sealpath)==ref['local_prediction_seal_sha256']==diagnostic['local_prediction']['prediction_seal_sha256']
        seal=json.loads(sealpath.read_text())
        assert seal['status']=='SEALED_SOURCE_FIT_CPU_CONDITIONED_LOCAL_PREDICTION'
        assert seal['fit_iterations']==[85,90] and seal['incremental_validation']==[95,100] and seal['target_parameter_updates']==0
        assert {f['path'] for f in seal['files']}=={'source_device_queue_parameters.csv.gz','source_device_queue_features.csv.gz','source_device_queue_predictions_sealed.csv.gz'}
        for f in seal['files']:assert sha(stage/f['path'])==f['sha256']
        hashes={str(stage/f['path']):f['sha256'] for f in json.loads((stage/'run_manifest.json').read_text())['artifacts']}
        for item in items:
            if item.get('key') in ref['input_keys'] and Path(item['path']).name not in ['complete.json','run_manifest.json']:
                assert hashes[item['path']]==item['sha256']
    gate=plan.get('source_model_reload_gate')
    if gate:
        stage=Path(gate['stage_root']);phase=guard.phase;guard.phase='hash_preflight'
        try:
            assert sha(stage/'run_manifest.json')==gate['manifest_sha256'];checked_stage(stage)
            module=Path(gate['loader_module']);assert not module.is_absolute() and '..' not in module.parts
            assert sha(ROOT/module)==sha(stage/'code_snapshot'/(module.parent.name+'__'+module.name))==gate['loader_sha256']
        finally:guard.phase=phase
        accepted=json.loads(Path(gate['acceptance_path']).read_text())
        assert sha(Path(gate['acceptance_path']))==gate['acceptance_sha256']
        assert accepted['status']=='PASS_EXACT_SOURCE_DEVICE_QUEUE_RELOAD' and accepted['tests_passed']==49
        assert accepted['old_source_prediction_decompressed_CSV_text_exact'] and accepted['source_parameters_bytes_unchanged']
        diagnostic=json.loads((stage/'diagnostic.json').read_text())
        assert diagnostic['status']=='SOURCE_SEALED_DEVICE_QUEUE_RELOAD_PASS' and diagnostic['all_fields_exact_to_original_source_prediction']
        assert not diagnostic['new_target_timing_read'] and not diagnostic['used_to_fit_model'] and not diagnostic['new_global_prediction']
        audit=json.loads((stage/'input_access_audit.json').read_text())
        for read in audit['reads']:
            if 'evaluator' in '|'.join(read['roles']) or '/evaluator_only/' in read['path']:assert set(read['phases'])=={'hash_preflight'}
        assert sha(stage/'source_device_queue_reload_seal.json')==gate['reload_seal_sha256']==accepted['reload_seal_sha256']
        seal=json.loads((stage/'source_device_queue_reload_seal.json').read_text())
        ref=plan['source_model_stages'][0]
        assert seal['source_model_manifest_sha256']==ref['manifest_sha256'] and seal['original_local_seal_sha256']==ref['local_prediction_seal_sha256']
        assert seal['fit_iterations']==[85,90] and seal['no_new_fit'] and seal['no_target_semantic_access']
    release=plan.get('source_release_stage')
    if release:
        stage=Path(release['stage_root']);phase=guard.phase;guard.phase='hash_preflight'
        try:
            assert sha(stage/'run_manifest.json')==release['manifest_sha256'];checked_stage(stage)
            for module in release['source_modules']:
                path=Path(module['path']);assert not path.is_absolute() and '..' not in path.parts
                snapshot=stage/'code_snapshot'/(path.parent.name+'__'+path.name)
                assert sha(ROOT/path)==sha(snapshot)==module['sha256']
        finally:guard.phase=phase
        accepted_path=Path(release['acceptance_path'])
        accepted=json.loads(accepted_path.read_text())
        assert sha(accepted_path)==release['acceptance_sha256']
        assert accepted['status']=='PASS_SOURCE_API_START_RELEASE_ABLATION' and accepted['tests_passed']==51
        assert accepted['old_source_prediction_rows_exact']==167056
        assert accepted['candidate_duration_costs_and_keys_exact'] and accepted['candidate_edges_exact']
        diagnostic=json.loads((stage/'diagnostic.json').read_text())
        assert diagnostic['status']=='SOURCE_API_START_RELEASE_CANDIDATE_PASS'
        assert diagnostic['used_to_fit_model'] and not diagnostic['new_target_timing_read'] and not diagnostic['new_global_prediction']
        assert diagnostic['source_fit_iterations']==[85,90] and diagnostic['source_iterations']==[85,90,95,100]
        assert diagnostic['validation_GPU_mutation_fit_features_exact'] and not diagnostic['formal_topology_changed']
        audit=json.loads((stage/'input_access_audit.json').read_text())
        assert audit['stage']=='diagnose' and not audit['raw_trace_scanned']
        for read in audit['reads']:
            if 'evaluator' in '|'.join(read['roles']) or '/evaluator_only/' in read['path']:
                assert set(read['phases'])=={'hash_preflight'},'Release model had target semantic access'
        comparison=stage/'source_release_comparison_seal.json'
        assert sha(comparison)==release['comparison_seal_sha256']==diagnostic['comparison_seal_sha256']
        comparison=json.loads(comparison.read_text())
        assert comparison['fit_iterations']==[85,90] and comparison['incremental_validation']==[95,100]
        assert comparison['target_parameter_updates']==0 and set(comparison['candidates'])=={'legacy_API_end','all_API_start'}
        for candidate,record in comparison['candidates'].items():
            sealpath=stage/record['local_seal_path'];assert sha(sealpath)==record['sha256']==release['local_seal_sha256'][candidate]
            seal=json.loads(sealpath.read_text());assert seal['fit_iterations']==[85,90] and seal['incremental_validation']==[95,100]
            assert seal['target_parameter_updates']==0
            phase=guard.phase;guard.phase='hash_preflight'
            try:
                for file in seal['files']:assert sha(sealpath.parent/file['path'])==file['sha256']
            finally:guard.phase=phase
        parameters=[]
        for candidate in ['legacy_API_end','all_API_start']:
            path=stage/candidate/'source_device_queue_parameters.csv.gz'
            table=pd.read_csv(path,dtype={'fit_iterations':'string','key':'string','cost_id':'string','level':'string'})
            assert len(table)==2772 and table.fit_iterations.eq('85,90').all();parameters.append(table)
        unchanged=[c for c in parameters[0] if c not in ['API_latency_median_ns','enqueue_remainder_median_ns','enqueue_remainder_mean_ns']]
        pd.testing.assert_frame_equal(parameters[0][unchanged],parameters[1][unchanged],check_exact=True)
        hashes={str((stage/file['path']).resolve()):file['sha256'] for file in json.loads((stage/'run_manifest.json').read_text())['artifacts']}
        for item in items:
            if item.get('key') not in release['input_keys'] or item['key'] in [
                'source_release_complete.json','source_release_run_manifest.json','source_release_acceptance.json']:continue
            assert hashes[item['path']]==item['sha256']
    terminal=plan.get('source_terminal_stage')
    if terminal:
        stage=Path(terminal['stage_root']);phase=guard.phase;guard.phase='hash_preflight'
        try:
            assert sha(stage/'run_manifest.json')==terminal['manifest_sha256'];checked_stage(stage)
            module=Path(terminal['source_module']);assert not module.is_absolute() and '..' not in module.parts
            snapshot=stage/'code_snapshot'/(module.parent.name+'__'+module.name)
            assert sha(ROOT/module)==sha(snapshot)==terminal['source_module_sha256']
        finally:guard.phase=phase
        accepted_path=Path(terminal['acceptance_path']);accepted=json.loads(accepted_path.read_text())
        assert sha(accepted_path)==terminal['acceptance_sha256']
        assert accepted['status']=='PASS_SOURCE_TERMINAL_ENDPOINT_SEMANTICS' and accepted['tests_passed']==53
        assert accepted['source_fit_iterations']==[85,90] and accepted['source_incremental_validation']==[95,100]
        assert accepted['validation_GPU_mutation_all_parameters_selections_predictions_exact']
        diagnostic=json.loads((stage/'diagnostic.json').read_text())
        assert diagnostic['status']=='SOURCE_TERMINAL_ENDPOINT_SEMANTICS_PASS'
        assert diagnostic['used_to_fit_model'] and not diagnostic['new_target_timing_read'] and not diagnostic['new_global_prediction']
        assert diagnostic['source_fit_iterations']==[85,90] and diagnostic['source_incremental_validation']==[95,100]
        assert diagnostic['validation_GPU_mutation_all_parameters_selections_predictions_exact'] and not diagnostic['formal_topology_changed']
        audit=json.loads((stage/'input_access_audit.json').read_text())
        assert audit['stage']=='diagnose' and not audit['raw_trace_scanned']
        for read in audit['reads']:
            if 'evaluator' in '|'.join(read['roles']) or '/evaluator_only/' in read['path']:
                assert set(read['phases'])=={'hash_preflight'},'Terminal source model had target semantic access'
        sealpath=stage/'source_terminal_endpoint_prediction_seal.json'
        assert sha(sealpath)==terminal['prediction_seal_sha256']==diagnostic['prediction_seal_sha256']
        seal=json.loads(sealpath.read_text())
        assert seal['status']=='SEALED_SOURCE85_90_TERMINAL_ENDPOINT_RULES_AND_CPU_CONDITIONED_PREDICTIONS'
        assert seal['fit_iterations']==[85,90] and seal['incremental_validation']==[95,100] and seal['target_parameter_updates']==0
        guard.phase='hash_preflight'
        try:
            for file in seal['files']:assert sha(stage/file['path'])==file['sha256']
        finally:guard.phase=phase
        identity=pd.read_csv(stage/'source_terminal_identity_parameters.csv',dtype={'stream':'string'})
        assert len(identity)==2 and identity.fit_endpoint_windows.eq(8).all()
        assert identity.CPU_owner_name_checked.eq('aten::_copy_from').all() and identity.stream.eq('0').all()
        selection=pd.read_csv(stage/'source_terminal_endpoint_selection_parameters.csv')
        assert len(selection)==4 and selection.fit_iterations.eq('85,90').all()
        hashes={str((stage/file['path']).resolve()):file['sha256'] for file in json.loads((stage/'run_manifest.json').read_text())['artifacts']}
        for item in items:
            if item.get('key') not in terminal['input_keys'] or item['key'] in [
                'source_terminal_complete.json','source_terminal_run_manifest.json','source_terminal_acceptance.json']:continue
            assert hashes[item['path']]==item['sha256']

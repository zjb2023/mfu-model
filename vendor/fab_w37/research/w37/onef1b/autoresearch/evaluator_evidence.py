"""Cached target runtime tables remain sealed evaluator-only evidence forever."""
import json
from pathlib import Path
from guards import ROOT,checked_stage
from smoke_worker import sha

ROLE='evaluator_cached_target_runtime_NEVER_MODEL_FIT'
CONTROL_ROLE='evaluator_previous_conditional_control_NEVER_MODEL_FIT'
CONTROL_FILES={
    'target_control_complete.json':'complete.json',
    'target_control_run_manifest.json':'run_manifest.json',
    'target_control_diagnostic.json':'diagnostic.json',
    'target_control_acceptance.json':'../acceptance.json',
    'target_control_seal.json':'evaluator_only/target_conditional_device_prediction_seal.json',
    'target_control_features.csv.gz':'evaluator_only/target_device_queue_features.csv.gz',
    'target_control_predictions.csv.gz':'evaluator_only/target_device_queue_predictions_sealed.csv.gz',
    'target_control_endpoint_metrics.csv':'evaluator_only/target_device_queue_endpoint_metrics.csv',
}
TERMINAL_ROLE='evaluator_prior_target_terminal_control_NEVER_MODEL_FIT'
TERMINAL_FILES={
    'target_terminal_complete.json':'complete.json',
    'target_terminal_run_manifest.json':'run_manifest.json',
    'target_terminal_input_access_audit.json':'input_access_audit.json',
    'target_terminal_diagnostic.json':'diagnostic.json',
    'target_terminal_acceptance.json':'../acceptance.json',
    'target_release_endpoint_metrics.csv':'evaluator_only/target_release_endpoint_metrics.csv',
    'legacy_API_end_target_seal.json':'evaluator_only/legacy_API_end/target_conditional_device_prediction_seal.json',
    'legacy_API_end_target_features.csv.gz':'evaluator_only/legacy_API_end/target_device_queue_features.csv.gz',
    'legacy_API_end_target_predictions.csv.gz':'evaluator_only/legacy_API_end/target_device_queue_predictions_sealed.csv.gz',
    'legacy_API_end_target_endpoints.csv':'evaluator_only/legacy_API_end/target_CPU_owned_device_endpoints.csv',
    'legacy_API_end_target_endpoint_selection.csv':'evaluator_only/legacy_API_end/target_CPU_owned_endpoint_event_selection.csv',
    'all_API_start_target_seal.json':'evaluator_only/all_API_start/target_conditional_device_prediction_seal.json',
    'all_API_start_target_features.csv.gz':'evaluator_only/all_API_start/target_device_queue_features.csv.gz',
    'all_API_start_target_predictions.csv.gz':'evaluator_only/all_API_start/target_device_queue_predictions_sealed.csv.gz',
    'all_API_start_target_endpoints.csv':'evaluator_only/all_API_start/target_CPU_owned_device_endpoints.csv',
    'all_API_start_target_endpoint_selection.csv':'evaluator_only/all_API_start/target_CPU_owned_endpoint_event_selection.csv',
}


def capabilities(plan,items):
    refs=plan.get('evaluator_evidence_stages',[])
    terminal=plan.get('target_terminal_stage')
    if not refs and not terminal:return [],[]
    assert plan['diagnostic_access']=='evaluator' and not plan['variants']
    assert plan['diagnostic'] in ['target_device_gaps','target_device_queue_transfer','target_device_release_transfer','target_terminal_endpoint_transfer'] and not plan.get('raw_trace_intake') and not plan.get('target_raw_trace_intake')
    if plan['diagnostic']=='target_device_queue_transfer':assert plan.get('source_model_stages') and plan.get('source_model_reload_gate')
    if plan['diagnostic']=='target_device_release_transfer':assert plan.get('source_release_stage') and plan.get('target_control_stage')
    bykey={i.get('key'):i for i in items};allowed=[];hash_only=[]
    for ref in refs:
        stage=Path(ref['stage_root']).resolve()
        assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
        hash_only.append(stage)
        for key in ref['input_keys']:
            item=bykey[key];path=Path(item['path']).resolve()
            assert path.is_relative_to(stage) and path!=stage and not item.get('raw_trace')
            assert item['roles']==[ROLE]
            allowed.append(path)
        assert all(stage/name in allowed for name in ['complete.json','run_manifest.json','input_access_audit.json','diagnostic.json'])
    control=plan.get('target_control_stage')
    if control:
        stage=Path(control['stage_root']).resolve()
        assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
        assert set(control['input_keys'])==set(CONTROL_FILES);hash_only.append(stage)
        for key,relative in CONTROL_FILES.items():
            item=bykey[key];path=Path(item['path']).resolve()
            assert path==(stage/relative).resolve() and item['roles']==[CONTROL_ROLE] and not item.get('raw_trace')
            allowed.append(path)
    if terminal:
        assert not refs and not control and plan.get('source_terminal_stage')
        stage=Path(terminal['stage_root']).resolve()
        assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
        assert set(terminal['input_keys'])==set(TERMINAL_FILES);hash_only.append(stage)
        for key,relative in TERMINAL_FILES.items():
            item=bykey[key];path=Path(item['path']).resolve()
            assert path==(stage/relative).resolve() and item['roles']==[TERMINAL_ROLE] and not item.get('raw_trace')
            allowed.append(path)
    return allowed,hash_only


def verify_evaluator_stages(plan,items,guard):
    if not plan.get('evaluator_evidence_stages') and not plan.get('target_terminal_stage'):return
    assert guard.stage=='diagnose' and guard.phase=='evaluator' and guard.sealed_reference_verified
    for ref in plan.get('evaluator_evidence_stages',[]):
        stage=Path(ref['stage_root']).resolve();guard.phase='hash_preflight'
        try:
            assert sha(stage/'run_manifest.json')==ref['manifest_sha256'];checked_stage(stage)
        finally:guard.phase='evaluator'
        diagnostic=json.loads((stage/'diagnostic.json').read_text())
        assert diagnostic['status']=='TARGET_RAW_RUNTIME_POSTHOC_PASS'
        assert diagnostic['sealed_reference']==plan['sealed_reference']
        assert diagnostic['used_to_fit_model'] is False and diagnostic['source_cost_updates']==0 and diagnostic['new_prediction'] is False
        assert diagnostic['all_CPU_partitions_conserve'] and diagnostic['first_file_gate_pass']
        assert sorted({r['iteration'] for r in diagnostic['checks']})==ref['target_iterations']
        assert sorted({r['rank'] for r in diagnostic['checks']})==ref['target_ranks']
        assert all(r['FB_boundaries_exact_ns'] and r['missing_runtime_correlations']==r['ambiguous_runtime_correlations']==0 for r in diagnostic['checks'])
        audit=json.loads((stage/'input_access_audit.json').read_text())
        assert audit['stage']=='diagnose' and audit['sealed_reference_verified']
        for read in audit['reads']:
            if 'evaluator' in '|'.join(read['roles']) or '/evaluator_only/' in read['path']:
                assert set(read['phases'])<={'hash_preflight','evaluator'},'prior target runtime evidence parsed outside evaluator'
        manifest=json.loads((stage/'run_manifest.json').read_text())
        hashes={str((stage/i['path']).resolve()):i['sha256'] for i in manifest['artifacts']}
        for item in items:
            if item.get('key') in ref['input_keys'] and Path(item['path']).name not in ['complete.json','run_manifest.json']:
                assert hashes[item['path']]==item['sha256']
    control=plan.get('target_control_stage')
    if control:
        stage=Path(control['stage_root']).resolve();guard.phase='hash_preflight'
        try:
            assert sha(stage/'run_manifest.json')==control['manifest_sha256'];checked_stage(stage)
        finally:guard.phase='evaluator'
        accepted_path=Path(control['acceptance_path']);accepted=json.loads(accepted_path.read_text())
        assert sha(accepted_path)==control['acceptance_sha256']
        assert accepted['status']=='PASS_FROZEN_SOURCE_TARGET_CONDITIONAL_TRANSFER' and accepted['tests_passed']==50
        diagnostic=json.loads((stage/'diagnostic.json').read_text())
        assert diagnostic['status']=='TARGET_CPU_CONDITIONED_DEVICE_QUEUE_TRANSFER_PASS'
        assert diagnostic['conditional_device_prediction'] and not diagnostic['used_to_fit_model'] and not diagnostic['new_global_prediction']
        sealpath=stage/'evaluator_only/target_conditional_device_prediction_seal.json'
        assert sha(sealpath)==control['local_prediction_seal_sha256']==diagnostic['local_prediction_seal_sha256']
        seal=json.loads(sealpath.read_text());assert seal['source_parameter_updates']==seal['target_parameter_updates']==0
        guard.phase='hash_preflight'
        try:
            for file in seal['files']:assert sha(sealpath.parent/file['path'])==file['sha256']
        finally:guard.phase='evaluator'
        hashes={str((stage/file['path']).resolve()):file['sha256'] for file in json.loads((stage/'run_manifest.json').read_text())['artifacts']}
        for item in items:
            if item.get('key') not in control['input_keys'] or item['key'] in [
                'target_control_complete.json','target_control_run_manifest.json','target_control_acceptance.json']:continue
            assert hashes[item['path']]==item['sha256']
    terminal=plan.get('target_terminal_stage')
    if terminal:
        stage=Path(terminal['stage_root']).resolve();guard.phase='hash_preflight'
        try:
            assert sha(stage/'run_manifest.json')==terminal['manifest_sha256'];checked_stage(stage)
        finally:guard.phase='evaluator'
        accepted_path=Path(terminal['acceptance_path']);accepted=json.loads(accepted_path.read_text())
        assert sha(accepted_path)==terminal['acceptance_sha256']
        assert accepted['status']=='PASS_TARGET_API_START_RELEASE_ABLATION' and accepted['tests_passed']==51
        assert accepted['source_parameter_updates']==accepted['target_parameter_updates']==0
        diagnostic=json.loads((stage/'diagnostic.json').read_text())
        assert diagnostic['status']=='TARGET_API_START_RELEASE_ABLATION_PASS'
        assert diagnostic['conditional_device_prediction'] and not diagnostic['used_to_fit_model'] and not diagnostic['new_global_prediction']
        assert diagnostic['target_iterations']==terminal['target_iterations'] and diagnostic['target_ranks']==terminal['target_ranks']
        audit=json.loads((stage/'input_access_audit.json').read_text())
        assert audit['stage']=='diagnose' and audit['sealed_reference_verified']
        for read in audit['reads']:
            if 'evaluator' in '|'.join(read['roles']) or '/evaluator_only/' in read['path']:
                assert set(read['phases'])<={'hash_preflight','evaluator'}
        for candidate,expected in terminal['release_candidate_seals'].items():
            sealpath=stage/f'evaluator_only/{candidate}/target_conditional_device_prediction_seal.json'
            assert sha(sealpath)==expected==diagnostic['release_candidate_seals'][candidate]
            seal=json.loads(sealpath.read_text())
            assert seal['source_parameter_updates']==seal['target_parameter_updates']==0
            guard.phase='hash_preflight'
            try:
                for file in seal['files']:assert sha(sealpath.parent/file['path'])==file['sha256']
            finally:guard.phase='evaluator'
        hashes={str((stage/file['path']).resolve()):file['sha256'] for file in json.loads((stage/'run_manifest.json').read_text())['artifacts']}
        for item in items:
            if item.get('key') not in terminal['input_keys'] or item['key'] in [
                'target_terminal_complete.json','target_terminal_run_manifest.json','target_terminal_acceptance.json']:continue
            assert hashes[item['path']]==item['sha256']

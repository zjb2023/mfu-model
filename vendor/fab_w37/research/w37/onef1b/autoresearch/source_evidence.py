"""Exact-file admission of completed source runtime evidence, never target costs."""
import json
from pathlib import Path
from guards import ROOT,checked_stage
from smoke_worker import sha


def capabilities(plan,items):
    refs=plan.get('source_evidence_stages',[])
    if not refs:return [],[]
    assert plan.get('diagnostic_access') in ['source_only','evaluator']
    if plan.get('raw_trace_intake'):
        assert plan['diagnostic_access']=='source_only' and plan['diagnostic'] in ['source_graph_flow_intake','source_counter_interval_probe']
    by_key={i.get('key'):i for i in items};allowed=[];hash_only=[]
    for ref in refs:
        stage=Path(ref['stage_root']).resolve()
        assert stage.is_relative_to(ROOT/'results/w37/A') and stage.name=='diagnose'
        hash_only.append(stage)
        for key in ref['input_keys']:
            item=by_key[key];path=Path(item['path']).resolve()
            assert path.is_relative_to(stage) and path!=stage and not item.get('raw_trace')
            assert item['roles'] in [['source_only_admitted_runtime_observations'],['source_only_stage_provenance_not_cost']]
            allowed.append(path)
        for name in ['complete.json','run_manifest.json','input_access_audit.json','diagnostic.json']:
            assert stage/name in allowed
    return allowed,hash_only


def verify_source_stages(plan,items,guard):
    """Hash all sealed artifacts, then parse only exact admitted metadata/data files."""
    for ref in plan.get('source_evidence_stages',[]):
        stage=Path(ref['stage_root']).resolve();phase=guard.phase
        guard.phase='hash_preflight'
        try:
            assert sha(stage/'run_manifest.json')==ref['manifest_sha256']
            checked_stage(stage)
        finally:guard.phase=phase
        diagnostic=json.loads((stage/'diagnostic.json').read_text())
        assert diagnostic['new_target_timing_read'] is False and diagnostic['used_to_fit_model'] is False
        if ref.get('evidence_type','source_runtime_intake')=='source_deployment_intake':
            assert diagnostic['status']=='SOURCE_DEPLOYMENT_INTAKE_PASS'
            assert diagnostic['source_case']=='source256' and diagnostic['source_host']=='worker33008' and diagnostic['source_node_rank']==2
            assert diagnostic['raw_trace_scanned'] is False and diagnostic['raw_hardware_counter_scanned'] is False
            assert diagnostic['raw_training_log_prefix_scanned'] and diagnostic['resource_gate_pass']
            assert diagnostic['physical_prefix_bytes_read']==262144 and diagnostic['static_observations']==384
            assert diagnostic['new_prediction'] is False and diagnostic['formal_topology_changed'] is False
            module=Path(ref['parser_module']);assert not module.is_absolute() and '..' not in module.parts
            snapshot=stage/'code_snapshot'/(module.parent.name+'__'+module.name)
            guard.phase='hash_preflight'
            try:assert sha(ROOT/module)==sha(snapshot)==ref['parser_sha256']
            finally:guard.phase=phase
        elif ref.get('evidence_type','source_runtime_intake')=='source_counter_interval_probe':
            assert diagnostic['status']=='SOURCE_COUNTER_INTERVAL_PROBE_PASS'
            assert diagnostic['source_iterations']==ref['source_iterations'] and diagnostic['source_ranks']==ref['source_ranks']
            assert diagnostic['raw_trace_scanned'] is False and diagnostic['raw_hardware_counter_scanned'] is True
            assert diagnostic['resource_gate_pass'] and diagnostic['new_causal_graph_edges']==0
            assert diagnostic['independent_graph_service_cost_identified'] is False
        elif ref.get('evidence_type','source_runtime_intake')=='source_counter_admission':
            assert diagnostic['status']=='SOURCE_COUNTER_ADMISSION_REVIEW_PASS'
            assert diagnostic['source_iterations']==ref['source_iterations'] and diagnostic['source_ranks']==ref['source_ranks']
            assert diagnostic['raw_trace_scanned'] is False and diagnostic['raw_counter_content_read'] is False
            assert diagnostic['max_absolute_old_CP_union_difference_ns']==0
        elif ref.get('evidence_type','source_runtime_intake')=='source_graph_flow_intake':
            assert diagnostic['status']=='SOURCE_GRAPH_FLOW_INTAKE_PASS'
            assert diagnostic['raw_source_iterations']==ref['source_iterations'] and diagnostic['raw_source_ranks']==ref['source_ranks']
            assert diagnostic['all_existing_T14_core_rows_exact'] and all(c['exact'] for c in diagnostic['core_checks'])
            assert diagnostic['first_file_resource_gate_pass'] and diagnostic['accepted_causal_edges']==0
        elif ref.get('evidence_type','source_runtime_intake')=='source_device_gap_context':
            assert diagnostic['status']=='SOURCE_DEVICE_GAP_CONTEXT_PASS'
            assert diagnostic['source_iterations']==ref['source_iterations'] and diagnostic['source_ranks']==ref['source_ranks']
            assert diagnostic['all_phase_visibility_and_gap_partitions_exact_ns'] and diagnostic['raw_trace_scanned'] is False
            module=Path(ref['classifier_module']);assert not module.is_absolute() and '..' not in module.parts
            snapshot=stage/'code_snapshot'/(module.parent.name+'__'+module.name)
            assert sha(ROOT/module)==sha(snapshot)==ref['classifier_sha256']
        else:
            assert diagnostic['status']=='SOURCE_RAW_RUNTIME_INTAKE_PASS'
            assert diagnostic['raw_source_iterations']==ref['source_iterations']
            assert diagnostic['raw_source_ranks']==ref['source_ranks']
            assert all(x['all_boundaries_exact_ns'] for x in diagnostic['checks'])
        audit=json.loads((stage/'input_access_audit.json').read_text());assert audit['stage']=='diagnose'
        for read in audit['reads']:
            if 'evaluator' in '|'.join(read['roles']) or '/evaluator_only/' in read['path']:
                assert set(read['phases'])=={'hash_preflight'},'prior source evidence parsed target timing'
        manifest=json.loads((stage/'run_manifest.json').read_text())
        hashes={str((stage/i['path']).resolve()):i['sha256'] for i in manifest['artifacts']}
        for item in items:
            if item.get('key') in ref['input_keys'] and Path(item['path']).name not in ['complete.json','run_manifest.json']:
                assert hashes[item['path']]==item['sha256']

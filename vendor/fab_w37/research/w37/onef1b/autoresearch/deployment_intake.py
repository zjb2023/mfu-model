"""Bounded startup evidence; declared shell defaults never imply deployment."""
import hashlib
import json
from pathlib import Path
import re
import resource
from time import perf_counter

import pandas as pd
from guards import ROOT
from smoke_worker import dump, sha
from worker import csv


ARGUMENTS = {
    'num_layers', 'hidden_size', 'num_attention_heads', 'seq_length',
    'max_position_embeddings', 'ffn_hidden_size', 'moe_ffn_hidden_size',
    'num_experts', 'moe_router_topk', 'moe_shared_expert_intermediate_size',
    'micro_batch_size', 'global_batch_size', 'world_size', 'rank',
    'tensor_model_parallel_size', 'pipeline_model_parallel_size',
    'context_parallel_size', 'expert_model_parallel_size', 'data_parallel_size',
    'decoder_first_pipeline_num_layers', 'decoder_last_pipeline_num_layers',
    'virtual_pipeline_model_parallel_size', 'recompute_granularity',
    'recompute_method', 'recompute_num_layers', 'mtp_num_layers',
    'moe_enable_deepep', 'moe_deepep_num_sms', 'moe_token_dispatcher_type',
    'overlap_moe_expert_parallel_comm', 'moe_shared_expert_overlap',
    'moe_grouped_gemm', 'moe_permute_fusion', 'moe_router_fusion',
    'moe_router_force_load_balancing', 'moe_router_load_balancing_type',
    'moe_expert_capacity_factor', 'moe_pad_expert_input_to_capacity',
    'moe_token_drop_policy', 'moe_router_dtype', 'moe_router_score_function',
    'bf16', 'fp16', 'fp8', 'params_dtype', 'transformer_impl',
    'distributed_backend', 'cp_comm_type', 'sequence_parallel',
    'use_precision_aware_optimizer', 'use_distributed_optimizer',
    'overlap_grad_reduce', 'overlap_param_gather', 'overlap_p2p_comm',
    'batch_p2p_comm', 'batch_p2p_sync', 'use_flash_attn',
    'bias_swiglu_fusion', 'bias_dropout_fusion', 'gradient_accumulation_fusion',
    'manual_gc', 'manual_gc_interval', 'q_lora_rank', 'kv_lora_rank',
    'qk_head_dim', 'qk_pos_emb_head_dim', 'v_head_dim', 'padded_vocab_size',
}
ENVIRONMENT = {
    'USE_DEEPEP_ACE', 'MUSA_LAUNCH_BLOCKING', 'CUDA_DEVICE_MAX_CONNECTIONS',
    'MUSA_BLOCK_SCHEDULE_MODE', 'TE_MULTI_STREAM_GROUPGEMM',
    'TORCH_MCCL_AVOID_RECORD_STREAMS', 'MCCL_BUFFSIZE', 'MCCL_ALGOS',
    'MCCL_PROTOS', 'MCCL_IB_QPS_PER_CONNECTION', 'MCCL_CROSS_NIC',
    'DEBUG_DEEPEP', 'MEASURE_DEEPEP_BW', 'DEEPEP_LOG_FREQ',
    'ENABLE_PROFILER', 'PROFILER_FREQ', 'PROFILER_WITH_STACK',
    'ENABLE_D2H_IN_PERMUTATION', 'USE_RECOMPUTE_VARIANCE',
    'NO_LOSS_REDUCE', 'EP_BALANCE_INFO', 'OMP_NUM_THREADS',
}
SCRIPT_DEFAULTS = {
    'TP_SIZE', 'PP_SIZE', 'EP_SIZE', 'CP_SIZE', 'FP8', 'TOKEN_DISPATCHER',
    'AC', 'SWIGLU_FUSION', 'SE_OVERLAP', 'USE_TOPK_ROUTER_FUSION',
    'MTP', 'FORCE_LB', 'MICRO_BATCH_SIZE', 'GLOBAL_BATCH_SIZE',
    'SEQ_LEN', 'LAYERS', 'FIRST_STAGE', 'LAST_STAGE', 'NUM_EXPERTS',
    'EXPERT_SIZE', 'FIRST_K_DENSE', 'DEEP_NUM_SM',
}
METRIC_START = re.compile(r'\biteration\s+\d+\s*/\s*\d+.*(?:elapsed|loss|throughput)', re.I)
ARG_LINE = re.compile(r'^\s*(?:\[rank\d+\]:\s*)?([a-z][a-z0-9_]*)\s+\.{2,}\s+(.+?)\s*$')
CONFIG_VALUE = r'(?:True|False|None|torch\.[a-z0-9_]+|[-+]?\d+(?:\.\d+)?|\[[^\]]*\]|\x27[^\x27]*\x27|"[^"]*")'
CONFIG_FIELD = re.compile(r'(?<!\w)('+'|'.join(sorted(ARGUMENTS))+r')=('+CONFIG_VALUE+r')(?=,|\))')
ENV_LINE = re.compile(r'^\s*(?:\+\s*)?(?:export\s+)?([A-Z][A-Z0-9_]*)=(.+?)\s*$')


def source_log_capabilities(plan, items, stage):
    intake = plan.get('source_training_log_intake')
    if not intake:
        return []
    assert plan['diagnostic'] == 'source_deployment_intake' and plan['diagnostic_access'] == 'source_only'
    assert not plan.get('raw_trace_intake') and not plan.get('target_raw_trace_intake')
    assert sha(ROOT / plan['resource_review_file']) == plan['resource_review_sha256']
    review = json.loads((ROOT / plan['resource_review_file']).read_text())
    assert review['source_only'] and not review['new_cost_fit'] and not review['new_target_timing_read']
    raw = [i for i in items if i.get('raw_trace')]
    assert len(raw) == intake['max_files'] == review['resource']['raw_files'] == 1
    item = raw[0]
    assert item['key'] == intake['key'] == 'source_training_log'
    assert item['source_case'] == 'source256' and item['raw_kind'] == 'training_log'
    assert item['roles'] == ['source_only_training_log_static_review_NEVER_COST_FIT']
    original = next(i for i in review['files'] if i['key'] == item['key'])
    assert all(item[k] == original[k] for k in ['path', 'size_bytes', 'sha256'])
    assert item['size_bytes'] <= intake['max_total_bytes'] == review['resource']['raw_file_bytes']
    assert intake['maximum_content_prefix_bytes'] == review['resource']['maximum_content_prefix_bytes']
    assert item['host'] == 'worker33008' and item['node_rank'] == 2
    assert item['associated_gpu_ranks'] == list(range(16, 24))
    return [item['path']] if stage == 'diagnose' else []


def parse_startup(text):
    """Whitelist literal static fields; stop semantic extraction at training metrics."""
    rows = []
    stop = None
    for number, line in enumerate(text.splitlines(), 1):
        if METRIC_START.search(line):
            stop = number
            break
        evidence = []
        m = ARG_LINE.match(line)
        if m and m[1] in ARGUMENTS:
            evidence.append(('effective_argument_dump', m[1], m[2]))
        if 'TransformerConfig(' in line:
            evidence += [('runtime_config_dump', m[1], m[2]) for m in CONFIG_FIELD.finditer(line)]
        m = ENV_LINE.match(line)
        if m and m[1] in ENVIRONMENT:
            evidence.append(('logged_environment_assignment', m[1], m[2]))
        if 'Successfully installed' in line:
            evidence += [('package_install_report', 'deep_ep_build', m.group(1))
                         for m in re.finditer(r'\bdeep[-_]ep-([0-9][A-Za-z0-9.+_-]*)', line)]
        for kind, key, value in evidence:
            rows.append(dict(line_number=number, evidence_kind=kind, field=key, value_literal=value,
                source_line_sha256=hashlib.sha256(line.encode()).hexdigest(), excerpt=f'{key} = {value}',
                deployment_revision_independently_verified=False))
    return rows, stop


def script_claims(text):
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        comment = stripped.startswith('#')
        clean = stripped.lstrip('#').strip() if comment else stripped
        m = ENV_LINE.match(clean)
        field, value = (m[1], m[2]) if m else ('', '')
        if field in ENVIRONMENT | SCRIPT_DEFAULTS:
            rows.append(dict(line_number=number, field=field, value_literal=value,
                evidence_kind='commented_script_assignment' if comment else 'script_assignment_not_execution_proof',
                source_line_sha256=hashlib.sha256(line.encode()).hexdigest(), excerpt=line,
                effective_at_runtime_verified=False))
        if 'pip install' in clean and 'deep_ep-' in clean:
            m = re.search(r'deep_ep-([^/\s]+)\.whl', clean)
            if m:
                rows.append(dict(line_number=number, field='deep_ep_wheel_requested', value_literal=m[1],
                    evidence_kind='commented_install' if comment else 'script_install_request_not_success',
                    source_line_sha256=hashlib.sha256(line.encode()).hexdigest(), excerpt=line,
                    effective_at_runtime_verified=False))
    return rows


def summarize_fields(rows):
    result = []
    for key in sorted(ARGUMENTS | ENVIRONMENT | {'deep_ep_build'}):
        seen = [r for r in rows if r['field'] == key]
        values = sorted({r['value_literal'] for r in seen})
        result.append(dict(field=key, observations=len(seen), distinct_values_json=json.dumps(values),
            status='NOT_VISIBLE_IN_BOUNDED_PREFIX' if not seen else ('MULTIPLE_LITERAL_VALUES_REVIEW_REQUIRED' if len(values)>1 else 'SINGLE_LITERAL_VALUE_OBSERVED'),
            independent_revision_proof=False))
    return result


def diagnose(out, paths, plan):
    begin = perf_counter()
    assert plan['diagnostic_access'] == 'source_only' and not plan['variants']
    review = json.loads((ROOT / plan['resource_review_file']).read_text())
    maximum = review['resource']['maximum_content_prefix_bytes']
    path = paths['source_training_log']
    with path.open('rb') as f:
        prefix = f.read(maximum)
    # Drop a partial final line; never fabricate a complete static value from it.
    complete = prefix if len(prefix) == path.stat().st_size else prefix[:prefix.rfind(b'\n')+1]
    decoded = complete.decode('utf-8', errors='replace')
    assert '\ufffd' not in decoded, 'Unverified log encoding; do not silently change source evidence'
    observations, stop = parse_startup(decoded)
    for row in observations:
        row.update(source_path=str(path), source_sha256=next(i['sha256'] for i in review['files'] if i['key']=='source_training_log'))
    assert observations, 'No allowlisted static startup fields: retain failed bounded attempt, do not expand automatically'
    claims = script_claims(paths['source_launch_script'].read_text())
    for row in claims:
        row.update(source_path=str(paths['source_launch_script']), source_sha256=next(i['sha256'] for i in review['files'] if i['key']=='source_launch_script'))
    scenario = json.loads(paths['legacy_source_scenario'].read_text())
    declared = {k:scenario[k] for k in ['schema', 'case_id', 'role', 'model', 'parallelism', 'workload']}
    dump(out/'source_declared_static_scenario.json', declared)
    csv(out, 'source_startup_static_observations.csv', pd.DataFrame(observations))
    csv(out, 'source_launch_script_claims.csv', pd.DataFrame(claims))
    summary = summarize_fields(observations)
    csv(out, 'source_deployment_field_readiness.csv', pd.DataFrame(summary))
    mapping = {'num_layers':('workload','num_layers'), 'hidden_size':('model','hidden_size'),
        'num_attention_heads':('model','num_attention_heads'), 'num_experts':('model','num_experts'),
        'moe_router_topk':('model','moe_router_topk'), 'micro_batch_size':('workload','micro_batch_size'),
        'global_batch_size':('workload','global_batch_size'), 'seq_length':('workload','sequence_length'),
        'world_size':('parallelism','world_size'), 'tensor_model_parallel_size':('parallelism','tp'),
        'pipeline_model_parallel_size':('parallelism','pp'), 'context_parallel_size':('parallelism','cp'),
        'expert_model_parallel_size':('parallelism','ep'), 'data_parallel_size':('parallelism','dp')}
    comparisons = []
    for field, (section, key) in mapping.items():
        values = sorted({r['value_literal'] for r in observations if r['field']==field})
        expected = str(declared[section][key])
        comparisons.append(dict(field=field, declared_value=expected, observed_values_json=json.dumps(values),
            status='NOT_OBSERVED' if not values else ('EXACT_LITERAL_MATCH' if values==[expected] else 'REVIEW_DIFFERENCE'),
            missing_is_not_default=True))
    csv(out, 'source_deployment_declared_comparison.csv', pd.DataFrame(comparisons))
    dump(out/'field_contract.json', dict(scope='Source256 single host training startup prefix; no iteration cost fit or target timing',
        physical_prefix_bytes_read=len(prefix), whole_file_hashed_bytes=path.stat().st_size,
        complete_prefix_lines=len(decoded.splitlines()), semantic_stop_before_training_metric_line=stop,
        prefix_may_physically_contain_timing_text=True, timing_values_extracted_or_used=False,
        decode_replacement_characters=decoded.count('\ufffd'),
        extraction='Whitelisted dotted argument fields, literal TransformerConfig fields, logged environment assignments and successful package-install reports only',
        identity='Pinned host worker33008/node2 file adjacent to source rank16 profiler; not independent proof of identical binary revision or all ranks',
        script='Static source assignments/defaults/comments, never executed. No variable expansion or conditional branch evaluation.',
        old_scenario='Only static model/parallelism/workload extracted. Existing timing services and historical fit iteration ranges excluded.',
        revision='Argument/config dumps are runtime observations; exact deployed code/build revision still unknown unless explicitly observed',
        no_new_prediction=True, no_cost_fit=True, formal_topology_changed=False))
    lines=['# Source256 startup evidence', '',
        'One bounded startup prefix; effective arguments and shell declarations remain separate.', '',
        '| Field | Declared | Observed literal values | Result |', '|---|---|---|---|']
    for r in comparisons:
        lines.append(f"| {r['field']} | {r['declared_value']} | {r['observed_values_json']} | {r['status']} |")
    lines += ['', 'Unknown environment flags/build revision cannot be replaced by script defaults. No new cost or prediction.']
    (out/'SOURCE_DEPLOYMENT_REVIEW.md').write_text('\n'.join(lines)+'\n')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    elapsed = perf_counter()-begin
    assert peak <= review['resource']['maximum_peak_RSS_bytes'], 'Single-file resource gate failed; no expansion'
    dump(out/'diagnostic.json', dict(status='SOURCE_DEPLOYMENT_INTAKE_PASS',
        new_prediction=False, used_to_fit_model=False, new_target_timing_read=False,
        raw_trace_scanned=False, raw_hardware_counter_scanned=False, raw_training_log_prefix_scanned=True,
        source_case='source256', source_host='worker33008', source_node_rank=2,
        physical_prefix_bytes_read=len(prefix), raw_file_bytes=path.stat().st_size,
        semantic_stop_before_training_metric_line=stop, static_observations=len(observations),
        visible_static_fields=sum(r['observations']>0 for r in summary),
        declared_exact_matches=sum(r['status']=='EXACT_LITERAL_MATCH' for r in comparisons),
        declared_differences=sum(r['status']=='REVIEW_DIFFERENCE' for r in comparisons),
        declared_unobserved=sum(r['status']=='NOT_OBSERVED' for r in comparisons),
        independent_deployment_revision_verified=False, formal_topology_changed=False,
        peak_RSS_bytes=peak, analysis_seconds=elapsed, resource_gate_pass=True,
        within_target_seconds=elapsed<=review['resource']['target_analysis_seconds'],
        next='Review exact static excerpts and identity, then separately decide target static posthoc intake. Do not fit deployment correction from target timing.'))

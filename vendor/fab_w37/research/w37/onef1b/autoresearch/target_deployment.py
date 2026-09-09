"""Sealed posthoc target startup comparison with a frozen source parser."""
import json
from pathlib import Path
import resource
from time import perf_counter

import pandas as pd
from guards import ROOT
from smoke_worker import dump, sha
from worker import csv
from deployment_intake import parse_startup, summarize_fields, ARGUMENTS, ENVIRONMENT

ROLE='evaluator224_training_log_static_posthoc_NEVER_MODEL_FIT'


def target_log_capabilities(plan,items,stage):
    intake=plan.get('target_training_log_intake')
    if not intake:return [],[]
    assert plan['diagnostic']=='target_deployment_intake' and plan['diagnostic_access']=='evaluator' and not plan['variants']
    assert not any(plan.get(k) for k in ['source_training_log_intake','raw_trace_intake','target_raw_trace_intake'])
    review_path=ROOT/plan['resource_review_file'];assert sha(review_path)==plan['resource_review_sha256']
    review=json.loads(review_path.read_text())
    assert review['diagnostic_access']=='evaluator' and not review['new_cost_fit'] and not review['new_prediction']
    assert review['sealed_reference']==plan['sealed_reference']
    raw=[i for i in items if i.get('raw_trace')]
    assert len(raw)==intake['max_files']==review['resource']['raw_files']==1
    item=raw[0];original=review['raw_file']
    assert item['key']==intake['key']=='target_training_log'
    assert item['roles']==[ROLE] and item['raw_kind']=='training_log' and item['source_case']=='target224'
    assert all(item[k]==original[k] for k in ['path','sha256','size_bytes','host','node_rank','associated_gpu_ranks'])
    assert item['host']=='worker33020' and item['node_rank']==2 and item['associated_gpu_ranks']==list(range(16,24))
    assert item['size_bytes']==intake['max_total_bytes']==review['resource']['raw_file_bytes']
    assert intake['maximum_content_prefix_bytes']==review['resource']['maximum_content_prefix_bytes']==262144
    prior=review['prerequisite']
    refs=[r for r in plan['source_evidence_stages'] if r.get('evidence_type')=='source_deployment_intake']
    assert len(refs)==1
    assert all(refs[0][k]==prior[k] for k in ['stage_root','manifest_sha256','parser_module','parser_sha256'])
    accepted=next(i for i in items if i.get('key')=='source_deployment_acceptance.json')
    assert accepted['path']==prior['acceptance_path'] and accepted['sha256']==prior['acceptance_sha256']
    assert accepted['roles']==['source_only_deployment_prerequisite_NOT_COST'] and not accepted.get('raw_trace')
    path=Path(accepted['path']).resolve();assert path.is_relative_to(ROOT/'results/w37/A')
    return ([item['path']] if stage=='diagnose' else []),[path]


def compare_literals(source,target):
    """Literal None is evidence. Missing observations never imply equality."""
    rows=[]
    for field in sorted(ARGUMENTS|ENVIRONMENT|{'deep_ep_build'}):
        left=sorted({r['value_literal'] for r in source if r['field']==field})
        right=sorted({r['value_literal'] for r in target if r['field']==field})
        if not left or not right:status='NOT_COMPARABLE_MISSING_OBSERVATION'
        elif len(left)!=1 or len(right)!=1:status='NOT_COMPARABLE_MULTIPLE_VALUES'
        else:status='SAME_OBSERVED_LITERAL' if left==right else 'DIFFERENT_OBSERVED_LITERAL'
        rows.append(dict(field=field,source_values_json=json.dumps(left),target_values_json=json.dumps(right),status=status,
            known_scenario_change=field in ['num_layers','pipeline_model_parallel_size'],
            used_as_model_feature=False,scope='evaluator_only_NEVER_MODEL_FIT'))
    return rows


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='evaluator' and not plan['variants']
    review=json.loads((ROOT/plan['resource_review_file']).read_text());prior=review['prerequisite']
    # The caller has checked the global seal and entire source evidence stage.
    assert sha(ROOT/prior['parser_module'])==prior['parser_sha256']
    accepted=json.loads(paths['source_deployment_acceptance.json'].read_text())
    assert accepted['status']=='PASS_SOURCE_DEPLOYMENT_STATIC_PREFIX_INTAKE' and accepted['prior_global_seal_unchanged']
    assert accepted['tests_passed']==45 and accepted['each_field_observed_8_times_with_same_literal']
    source=pd.read_csv(paths['source_static_observations.csv'],keep_default_na=False).to_dict('records')
    assert len(source)==384 and len({r['field'] for r in source})==48
    target_path=paths['target_training_log'];maximum=review['resource']['maximum_content_prefix_bytes']
    with target_path.open('rb') as f:prefix=f.read(maximum)
    complete=prefix if len(prefix)==target_path.stat().st_size else prefix[:prefix.rfind(b'\n')+1]
    decoded=complete.decode('utf-8',errors='strict');target,stop=parse_startup(decoded)
    assert target,'No static target observations in bounded prefix: do not expand automatically'
    for row in target:row.update(source_path=str(target_path),source_sha256=review['raw_file']['sha256'],scope='evaluator_only_NEVER_MODEL_FIT')
    csv(out,'evaluator_only/target_startup_static_observations.csv',pd.DataFrame(target))
    csv(out,'evaluator_only/target_deployment_field_readiness.csv',pd.DataFrame(summarize_fields(target)))
    comparisons=compare_literals(source,target)
    csv(out,'evaluator_only/source_target_static_field_comparison.csv',pd.DataFrame(comparisons))
    target_scenario=json.loads(paths['legacy_target_scenario'].read_text())
    declared={k:target_scenario[k] for k in ['schema','case_id','role','model','parallelism','workload']}
    dump(out/'evaluator_only/target_declared_static_scenario.json',declared)
    mapping={'num_layers':('workload','num_layers'),'hidden_size':('model','hidden_size'),
        'num_attention_heads':('model','num_attention_heads'),'num_experts':('model','num_experts'),
        'moe_router_topk':('model','moe_router_topk'),'micro_batch_size':('workload','micro_batch_size'),
        'global_batch_size':('workload','global_batch_size'),'seq_length':('workload','sequence_length'),
        'world_size':('parallelism','world_size'),'tensor_model_parallel_size':('parallelism','tp'),
        'pipeline_model_parallel_size':('parallelism','pp'),'context_parallel_size':('parallelism','cp'),
        'expert_model_parallel_size':('parallelism','ep'),'data_parallel_size':('parallelism','dp')}
    scenario_rows=[]
    for field,(section,key) in mapping.items():
        values=sorted({r['value_literal'] for r in target if r['field']==field});expected=str(declared[section][key])
        scenario_rows.append(dict(field=field,declared_value=expected,observed_values_json=json.dumps(values),
            status='NOT_OBSERVED' if not values else ('EXACT_LITERAL_MATCH' if values==[expected] else 'REVIEW_DIFFERENCE')))
    csv(out,'evaluator_only/target_declared_static_comparison.csv',pd.DataFrame(scenario_rows))
    changes=[r for r in comparisons if r['status']=='DIFFERENT_OBSERVED_LITERAL']
    unknown=[r for r in comparisons if r['status'].startswith('NOT_COMPARABLE')]
    same=[r for r in comparisons if r['status']=='SAME_OBSERVED_LITERAL']
    dump(out/'field_contract.json',dict(scope='Posthoc target startup configuration only; all target/mixed rows permanently evaluator_only',
        source_parser_sha256=prior['parser_sha256'],source_manifest_sha256=prior['manifest_sha256'],
        sealed_reference=plan['sealed_reference'],physical_target_prefix_bytes_read=len(prefix),
        target_whole_file_hashed_bytes=target_path.stat().st_size,semantic_stop_before_training_metric_line=stop,
        prefix_may_physically_contain_timing_text=True,timing_values_extracted_or_used=False,
        literal_None_preserved=True,missing_never_filled_from_script=True,
        rank_identity='File adjacent to target rank16 trace; each printed configuration occurrence is not independently rank-identified',
        observed_fields='Same fixed T24 whitelist; no target-specific aliases, field additions or parser changes',
        unchanged_fields_limit='Matching observed arguments do not establish matching GPU/CPU state, build revision, input routing or runtime service',
        formal_topology_changed=False,source_cost_updates=0))
    lines=['# Source/target startup configuration comparison','',
        'Posthoc only. Same frozen source parser; missing flags and build identifiers remain unknown.','',
        '| Field | Source literal | Target literal | Already represented by scenario |','|---|---|---|---|']
    for row in changes:lines.append(f"| {row['field']} | {row['source_values_json']} | {row['target_values_json']} | {row['known_scenario_change']} |")
    lines += ['',f'{len(same)} fields have the same observed literal; {len(unknown)} fields are not comparable.',
        'No cost correction is inferred from absence or equality of fields. Device state and custom backend version remain unverified.']
    (out/'evaluator_only/DEPLOYMENT_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;elapsed=perf_counter()-begin
    assert peak<=review['resource']['maximum_peak_RSS_bytes'],'Single target log prefix exceeded resource gate'
    dump(out/'diagnostic.json',dict(status='TARGET_DEPLOYMENT_STATIC_POSTHOC_PASS',new_prediction=False,used_to_fit_model=False,
        new_target_timing_read=False,target_raw_prefix_may_physically_contain_timing_text=True,
        source_cost_updates=0,raw_trace_scanned=False,raw_hardware_counter_scanned=False,raw_training_log_prefix_scanned=True,
        target_case='target224',target_host='worker33020',target_node_rank=2,raw_file_bytes=target_path.stat().st_size,
        physical_target_prefix_bytes_read=len(prefix),semantic_stop_before_training_metric_line=stop,
        target_static_observations=len(target),target_static_fields=len({r['field'] for r in target}),
        same_observed_fields=len(same),different_observed_fields=len(changes),uncomparable_fields=len(unknown),
        changed_fields=[r['field'] for r in changes],changes_outside_known_scenario=[r['field'] for r in changes if not r['known_scenario_change']],
        target_declared_matches=sum(r['status']=='EXACT_LITERAL_MATCH' for r in scenario_rows),
        target_declared_unobserved=sum(r['status']=='NOT_OBSERVED' for r in scenario_rows),
        target_declared_differences=sum(r['status']=='REVIEW_DIFFERENCE' for r in scenario_rows),
        source_parser_unchanged=True,sealed_reference=plan['sealed_reference'],formal_topology_changed=False,
        peak_RSS_bytes=peak,analysis_seconds=elapsed,resource_gate_pass=True,within_target_seconds=elapsed<=review['resource']['target_analysis_seconds'],
        next='Review configuration changes against known scenario; no deployment multiplier without independent state evidence. Return to a source-calibrated runtime/payload model hypothesis.'))

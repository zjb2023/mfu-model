#!/usr/bin/env python3
"""Snapshot an explicit code closure and hash external derived inputs, never raw trace."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/zjb/Desktop/fabric-data-analysis')
CONTROL = ROOT / 'docs/w37/coordination'
CASE = Path('case_224gpu_pp14_cp2_a2a')
RESULT = CASE / 'results/mfu_accuracy_comparison_2026w36'
V684 = RESULT / 'dag_v684_blocking_pp_program_order_256_to_224'
V685 = RESULT / 'dag_v685_source_steady_calibration_256_to_224'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args])

def original_state():
    index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute(): index = SOURCE / index
    tracked = git('diff', '--name-only', '-z').decode().split('\0')
    return {
        'head': git('rev-parse', 'HEAD').decode().strip(),
        'branch': git('branch', '--show-current').decode().strip(),
        'index_sha256': sha(index),
        'staged_diff_sha256': hashlib.sha256(git('diff', '--cached', '--binary')).hexdigest(),
        'unstaged_diff_sha256': hashlib.sha256(git('diff', '--binary')).hexdigest(),
        'tracked_dirty_files': {p: sha(SOURCE/p) if (SOURCE/p).is_file() else 'DELETED'
                                for p in tracked if p},
    }

def code_closure(seeds):
    roots = [SOURCE/CASE/'scripts', SOURCE/'case_256gpu_pp16_cp2_a2a/scripts',
             SOURCE/'workflow/scripts', SOURCE/'src', SOURCE/'scripts', SOURCE]
    selected = set(seeds)
    queue = list(seeds)
    while queue:
        rel = queue.pop()
        if rel.suffix != '.py': continue
        tree = ast.parse((SOURCE/rel).read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): modules.update(n.name for n in node.names)
            if isinstance(node, ast.ImportFrom) and node.module: modules.add(node.module)
        for module in modules:
            for directory in [SOURCE/rel.parent, *roots]:
                part = Path(*module.split('.'))
                candidates = [directory/part.with_suffix('.py'), directory/part/'__init__.py']
                found = next((p for p in candidates if p.is_file()), None)
                if found:
                    dep = found.relative_to(SOURCE)
                    if dep not in selected: selected.add(dep); queue.append(dep)
                    for parent in found.parents:
                        if parent == SOURCE: break
                        init = parent/'__init__.py'
                        if init.is_file(): selected.add(init.relative_to(SOURCE))
                    break
    return selected

def main():
    if ROOT == SOURCE: raise RuntimeError('preparation must use a separate worktree')
    if (CONTROL/'code_manifest.json').exists(): raise RuntimeError('snapshot exists; inspect before replacing')
    CONTROL.mkdir(parents=True, exist_ok=True)
    dump(CONTROL/'original_state_before.json', original_state())
    (CONTROL/'original_status_before.txt').write_bytes(git('status', '--porcelain=v1', '--untracked-files=all'))
    versions = tomllib.loads((SOURCE/'workflow/config/dag_mfu_versions.toml').read_text())
    seeds = {Path('workflow/scripts/build_dag_mfu_schedule_factorial.py'),
             Path('workflow/scripts/audit_dag_mfu_dependency_topology.py'),
             Path('workflow/scripts/manage_dag_mfu_versions.py'),
             CASE/'tests/test_dag_v685_source_steady_calibration.py'}
    configs = {Path('workflow/config/dag_mfu_versions.toml'), Path('workflow/DAG_MFU_VERSION_PIPELINE.md')}
    for version in versions['version']:
        for key in ['builder','evaluator','sealer','finalizer']:
            if key in version: seeds.add(Path(version[key]))
        for key in ['config','evaluation_config']:
            if key in version: configs.add(Path(version[key]))
    seeds.update(Path('case_16gpu_pp2')/n for n in ['build_canonical_v7.py'])
    selected = code_closure(seeds) | configs
    code_items = []
    for rel in sorted(selected):
        src, dst = SOURCE/rel, ROOT/rel
        if not src.is_file(): raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        code_items.append({'path':str(rel),'source_path':str(src), 'sha256':sha(src),
                           'size_bytes':src.stat().st_size,
                           'role':'legacy_case_bound_code_or_config' if str(rel).startswith(str(CASE)) else 'code_review_required_before_generalization'})
    dump(CONTROL/'code_manifest.json', {'schema':'w37-code-snapshot-v1','source_head':original_state()['head'],
        'files':code_items,'code_tree_sha256':hashlib.sha256(json.dumps(code_items,sort_keys=True).encode()).hexdigest(),
        'note':'Copied current working-tree bytes, not only HEAD; numeric source256 state is not approved for task B.'})
    external = {}
    def add(path, role, expected=None):
        path = Path(path).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        if path.suffix == '.py' or '/config/' in str(path): return
        if not path.is_relative_to(SOURCE/'case_224gpu_pp14_cp2_a2a/results') and not path.is_relative_to(SOURCE/'case_256gpu_pp16_cp2_a2a/results') and not path.is_relative_to(SOURCE/'case_16gpu_pp2/results'):
            raise ValueError(f'not a permitted derived input: {path}')
        actual = sha(path)
        if expected and actual != expected: raise ValueError(f'sealed hash changed: {path}')
        key = str(path)
        if key in external:
            if role not in external[key]['roles']: external[key]['roles'].append(role)
            return
        external[key] = {'path':key, 'sha256':actual,'size_bytes':path.stat().st_size,'roles':[role],
                         'raw_trace':False,'allowed_tasks':['A'] if role!='source16' else ['B']}
    def seal(rel):
        path=SOURCE/rel; data=json.loads(path.read_text())
        add(path,'historical_control')
        for item in data['artifacts']: add(item['path'],'source256_fitted_or_historical',item['sha256'])
        return {'path':str(path),'status':data['status'],'verified_artifacts':len(data['artifacts'])}
    seals=[seal(V684/'predictions/prediction_seal.json'),seal(V685/'predictions/prediction_seal.json')]
    prov=json.loads((SOURCE/V685/'provenance.json').read_text())
    for item in prov['inputs']+prov['outputs']:
        if Path(item['path']).suffix=='.py' or '/config/' in item['path']:
            if sha(item['path'])!=item['sha256']:raise ValueError(f'v685 source mismatch: {item["path"]}')
        else:add(item['path'],'source256_fitted_or_historical',item['sha256'])
    add(SOURCE/V685/'provenance.json','historical_control')
    add(SOURCE/V684/'evaluator_only/iteration_evaluation.csv','evaluator224_after_seal')
    truth=SOURCE/RESULT/'dag_v67_microbatch_runtime_shape_256_to_224/evaluator_only/iteration_ground_truth.csv'
    add(truth,'evaluator224_after_seal')
    for name in ['v69_candidate_design','v69_graph_input_bundle']:
        directory=SOURCE/RESULT/'autoresearch_224_vs_256_parameters'/name
        for path in sorted(directory.iterdir()):
            if path.is_file(): add(path,'candidate_diagnostic_not_released')
    for attempt in ['wo_mccllog_1116','w_mccllog_1129']:
        for name in ['attempt_contract','input_manifest','per_iteration_metrics','readiness_report','validation_report']:
            add(SOURCE/'case_16gpu_pp2/results'/attempt/'json'/f'{name}.json','source16')
    dump(CONTROL/'external_inputs.json', {'schema':'w37-external-derived-inputs-v1',
        'original_root':str(SOURCE),'files':list(external.values()),
        'raw_trace_policy':'Not copied or scanned. Raw archive hashes in source16 input_manifest are historical declarations, not reverified here.',
        'task_B_forbidden':'All source256 timings, cost-bound graphs, fitted parameters, 224 target timings, and metrics, even if stored under a model-code path.'})
    lock=json.loads((SOURCE/V684/'predictions/dependency_topology_lock.json').read_text())
    metrics=json.loads((SOURCE/V685/'evaluator_only/metrics.json').read_text())
    readiness=json.loads((SOURCE/RESULT/'autoresearch_224_vs_256_parameters/v69_graph_input_bundle/graph_input_readiness.json').read_text())
    dump(CONTROL/'baseline_audit.json', {'schema':'w37-baseline-audit-v1','status':'HASHES_VERIFIED_SMOKE_PENDING',
        'v685_metrics':metrics,'v684_topology_lock':lock,'seal_verifications':seals,
        'v69_registered_status':versions['candidate'][0]['status'],'v69_input_readiness':readiness,
        'limitations':['v685 is partial steady refit: noncompute floors and outer clock are inherited from earlier source256 fits',
                      'source_iterations_read in legacy audit denotes fit rows, but underlying source CSV files include 60-100',
                      '10.37% is Profiler step-time MAPE, not directly an MFU error or statistically defined accuracy',
                      'v682 75.8% error share belongs to 60-100, not v685 stable-window attribution']})
    configs16=[]
    for name in ['run_contract_wo_mccllog_v7.toml','run_contract_w_mccllog_v7.toml']:
        path=SOURCE/'case_16gpu_pp2/config'/name
        configs16.append({'path':str(path),'sha256':sha(path),'content':tomllib.loads(path.read_text())})
    dump(CONTROL/'source16_inventory.json',{'status':'DERIVED_INPUTS_AVAILABLE_MODEL_COMPARABILITY_PARTIAL',
        'attempts':configs16, 'unresolved':['num_layers','microbatches','global_batch','sequence_length','per_iteration_FLOPs','CP1_to_CP2_transfer','which_logging_attempt_to_calibrate'],
        'note':'Do not combine attempts. No capacity calibration claim is inherited from the L6 spatial-analysis admission.'})
    print(json.dumps({'code_files':len(code_items),'external_files':len(external),'external_total_bytes':sum(x['size_bytes'] for x in external.values()),'seals':seals}))

if __name__=='__main__':main()

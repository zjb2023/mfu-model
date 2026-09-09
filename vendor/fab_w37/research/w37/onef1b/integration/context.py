"""Verify committed historical evidence; never run either research model."""
import csv
import hashlib
import io
import json
import math
import subprocess
from statistics import mean
from common import ROOT, OUT, DOC, dump, write_csv, digest


def build():
    spec = json.loads((DOC / 'context_inputs.json').read_text())
    checks, files = [], {}

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)

    def close(name, actual, expected):
        check(name, math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-9))

    for r in spec['inputs']:
        repo = ROOT if r['repo'] == '.' else r['repo']
        ref = r['commit'] + ':' + r['git_path']
        blob = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', ref], text=True).strip()
        data = subprocess.check_output(['git', '-C', str(repo), 'show', ref])
        check(r['key'] + ':blob', blob == r['blob'])
        check(r['key'] + ':bytes', len(data) == r['bytes'])
        check(r['key'] + ':sha256', hashlib.sha256(data).hexdigest() == r['sha256'])
        p = OUT / r['output']
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[r['key']] = p
    points = list(csv.DictReader(io.StringIO(files['scaleout_points'].read_text())))
    summary = json.loads(files['scaleout_summary'].read_text())
    methods = {r['method'] for r in summary}
    check('112 rows / four methods / eight summaries', len(points) == 112 and len(methods) == 4 and len(summary) == 8)
    identities = [(r['case_id'], r['method'], int(r['iteration'])) for r in points]
    check('unique evaluation identities', len(identities) == len(set(identities)))
    for index, r in enumerate(points):
        a = lambda k: float(r[k])
        F, N, P = a('model_training_flops'), a('world_size'), a('peak_bf16_dense_tflops_per_gpu')
        close(f'{index}:peak', P, 500)
        close(f'{index}:FLOPs', F, 4.218697266757632e15 if r['case_id'] == 'source16_holdout' else 9.767709113843712e16)
        close(f'{index}:world', N, 16 if r['case_id'] == 'source16_holdout' else 256)
        for prefix in ['predicted', 'actual']:
            close(f'{index}:{prefix} MFU', a(prefix + '_mfu_pct'), 100 * F / (a(prefix + '_training_step_ms') / 1000 * N * P * 1e12))
        close(f'{index}:Training APE', a('training_ape_pct'), abs(a('predicted_training_step_ms') / a('actual_training_step_ms') - 1) * 100)
        close(f'{index}:Profiler APE', a('profiler_ape_pct'), abs(a('predicted_profiler_step_ms') / a('actual_profiler_step_ms') - 1) * 100)
        close(f'{index}:MFU relative APE', a('mfu_relative_ape_pct'), abs(a('predicted_mfu_pct') / a('actual_mfu_pct') - 1) * 100)
        close(f'{index}:MFU percentage points', a('absolute_mfu_error_percentage_points'), abs(a('predicted_mfu_pct') - a('actual_mfu_pct')))
        close(f'{index}:outer closure', a('predicted_training_step_ms'), a('predicted_profiler_step_ms') + a('predicted_outer_framework_ms'))
    for s in summary:
        group = [r for r in points if (r['case_id'], r['method']) == (s['case_id'], s['method'])]
        check(str((s['case_id'], s['method'])) + ':iteration grid', sorted(int(r['iteration']) for r in group) == list(range(65 if s['case_id'] == 'source16_holdout' else 5, 101, 5)))
        close('group count:' + s['method'] + s['case_id'], len(group), s['n'])
        for target, source in [('training_mape_pct', 'training_ape_pct'), ('profiler_mape_pct', 'profiler_ape_pct'), ('mfu_relative_mape_pct', 'mfu_relative_ape_pct'), ('mfu_mae_percentage_points', 'absolute_mfu_error_percentage_points'), ('actual_mfu_mean_pct', 'actual_mfu_pct'), ('predicted_training_step_ms', 'predicted_training_step_ms'), ('predicted_profiler_step_ms', 'predicted_profiler_step_ms'), ('predicted_mfu_pct', 'predicted_mfu_pct')]:
            close(s['case_id'] + s['method'] + ':' + target, s[target], mean(float(r[source]) for r in group))
    original_checks = json.loads(files['scaleout_checks'].read_text())
    check('prior status', original_checks['status'] == 'PASS_REAL_STEP_AND_ABSOLUTE_MFU')
    check('prior source-only declaration', original_checks['prediction_target_dynamic_reads'] == 0 and original_checks['source_only_time_calibration_preserved'])
    # Check claims against fixed report bytes, not moving branch names.
    for key, tokens in {
        'mfu_guide': ['16 个流水阶段', '4 个 microbatch', '三阶段分类'],
        'mfu_v1': ['7.7068', '19.970490'],
        'mfu_v2': ['93.463844', '21616.236', '0.1010'],
        'mfu_v4': ['27.6923', '21616.236'],
        'scaleout_final': ['45.7773', '84.8177', '不宣称新的独立盲测']
    }.items():
        for token in tokens:
            check(key + ':fact:' + token, token in files[key].read_text())
    target = [r for r in points if r['case_id'] == 'target256' and r['method'] == 'M2_source_only_linear']
    write_csv('scaleout_target_points.csv', target)
    write_csv('scaleout_summary.csv', summary)
    dump('context_audit.json', {'status': 'PASS', 'check_count': len(checks), 'checks': checks, 'inputs': spec['inputs'], 'input_bytes': sum(r['bytes'] for r in spec['inputs']), 'target': target, 'summary': summary, 'context_manifest_sha256': digest(DOC / 'context_inputs.json'), 'raw_trace_reads': 0, 'model_executions': 0, 'sibling_worktree_reads': 0, 'original_seals_reexecuted': False, 'scope': 'Recompute arithmetic from fixed committed outputs; prior provenance checks are attributed, not rerun.'})
    print(json.dumps({'stage': 'context', 'status': 'PASS', 'checks': len(checks), 'input_bytes': sum(r['bytes'] for r in spec['inputs'])}))


if __name__ == '__main__':
    build()

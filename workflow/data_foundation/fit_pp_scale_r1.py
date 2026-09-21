"""One-coefficient PP-scale candidate; target-calibrated, not held-out validation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from pp_loss_slot_r1 import build
from build_b_cp_pair_r4 import run

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    paths = [BASE/'pp-transfer-evaluation-r1/report.json',
             BASE/'pp32-detailed-fb-assembly-r1/parameters.json',
             BASE/'four-layer-f-r1/graph.json',
             BASE/'b-four-layers-r10/graph.json',
             BASE/'detailed-fb-32to256-r1/report.json']
    def load(p):
        return json.loads(p.read_text())
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    seals = {str(p): sha(p) for p in paths}
    evidence, params, fg, bg, frozen = map(load, paths)
    # Equal weight to F and B median costs: fit k minimizing sum((k*x-y)^2).
    rows = evidence['rows']
    k = sum(r['source_frozen_ms']*r['target_median_ms'] for r in rows) / sum(r['source_frozen_ms']**2 for r in rows)
    alpha = (k-1)/3  # PP16/PP4 - 1.
    scalar = copy.deepcopy(params)
    scalar['fb_ms']['middle'] = {'F': run(fg)['END']['end_ms'], 'B': run(bg)['END']['end_ms']}
    before = build(16, 4, scalar)['window_ms']
    assert abs(before-frozen['predicted_ms']) < 1e-6
    candidate = copy.deepcopy(scalar)
    candidate['pp_effective_overlap_ms'] = {ph: v*k for ph, v in scalar['pp_effective_overlap_ms'].items()}
    after = build(16, 4, candidate)['window_ms']
    assert after >= before
    assert all(sha(Path(p)) == s for p, s in seals.items())
    result = dict(status='PASS_CANDIDATE_REPLAY_NOT_VALIDATION',
                  formula='t_PP(phase,P)=t_PP_source(phase)*(1+alpha*(P/4-1))',
                  alpha=alpha, multiplier_at_PP16=k, calibrated_PP=[4,16],
                  scope='same 80MiB PP payload; effective cost, not pure network service; do not extrapolate beyond PP4..16 without verification',
                  fitted_parameter_count=1, target_used_for_calibration=True,
                  fit_objective='equal-weight least squares on F/B target medians, NOT total iteration error',
                  costs_ms=candidate['pp_effective_overlap_ms'],
                  original_prediction_ms=before, candidate_prediction_ms=after,
                  window_increase_ms=after-before, trace_ms=frozen['trace_ms'],
                  original_error_percent=100*(before/frozen['trace_ms']-1),
                  candidate_error_percent=100*(after/frozen['trace_ms']-1),
                  replay='exact scalar contraction of unchanged F/B templates; original frozen window reproduced within 1e-6 ms',
                  source_PP4_multiplier=1, frozen_inputs_unchanged=True,
                  promoted=False)
    def dump(name, obj):
        (out/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    dump('report.json', result)
    dump('candidate-parameters.json', candidate)
    code_paths = [Path(__file__), Path(__file__).with_name('pp_loss_slot_r1.py'),
                  Path(__file__).with_name('build_b_cp_pair_r4.py'),
                  Path(__file__).with_name('pp32_rendezvous_r2.py')]
    dump('manifest.json', dict(inputs=seals, code={str(p.resolve()):sha(p) for p in code_paths},
                               outputs={str(p.resolve()):sha(p) for p in out.glob('*.json')}))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

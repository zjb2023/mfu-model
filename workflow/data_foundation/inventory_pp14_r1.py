"""Inventory existing CPU PP evidence; no claim of transport cost calibration."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    base = Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/cpu-schedule-r1')
    paths = [base/'case_matrix.json', base/'pp_pairs.json']
    cases, pairs = [json.loads(p.read_text()) for p in paths]
    rows = []
    for case in cases:
        ps = [p for p in pairs if p['case'] == case['case']]
        assert len(ps) == case['pp_pairs']
        assert all(p['dtype'] == 'BFloat16' for p in ps)
        rows.append(dict(case=case['case'], pp=case['pp'], pairs=len(ps),
                         payload_bytes=sorted({p['nelems']*2 for p in ps}),
                         iterations=sorted({p['iteration'] for p in ps}),
                         status='CPU_PAIRING_ONLY_COST_NOT_READY' if ps else 'NOT_APPLICABLE'))
    report = dict(rows=rows, total_pairs=len(pairs), comparison_candidates=[
        ['2211','2411'], ['2212','2412'], ['4211','4411']],
        restrictions=['same payload alone does not control DP, layer count, topology or contention',
                      'existing CPU pair evidence is not transport completion; do not fit its envelope as network service',
                      '20MiB has no PP4 counterpart; PP1 is not a zero-cost calibration point',
                      'check GPU readiness and host clock basis before cross-case timing comparison'],
        next_step='start 2211 vs 2411, iter53 calibration / iter58 evaluation; verify path and readiness before fitting')
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in paths},
        code={str(Path(__file__).resolve()):sha(Path(__file__))},
        outputs={str((out/'report.json').resolve()):sha(out/'report.json')}), indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()

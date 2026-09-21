"""Source-only bandwidth-ratio scenarios; target is evaluation only, not a fit."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    base = ROOT / 'results/data-foundation/optimizer-tail-audit-r1'
    inputs = [base / (name + '.json') for name in ('source', 'target')]
    source, target = [json.loads(p.read_text()) for p in inputs]
    rs = [[e for e in d['device_events'] if e['kind'] == 'RS'] for d in (source, target)]
    source_rs = next(s['union_ms'] for s in source['activity_stats'] if s['kind'] == 'RS')
    gap = target['tail_ms'] - source['tail_ms']
    rows = []
    # Fixed sensitivity grid, not selected against the target. Equal volume and
    # all source RS duration bandwidth-sensitive are explicit provisional assumptions.
    for ratio in (1.0, 0.75, 0.5, 0.25):
        delta = source_rs * (1 / ratio - 1)
        predicted = source['tail_ms'] + delta
        rows.append(dict(mixed_to_intra_effective_bandwidth_ratio=ratio,
                         volume_ratio=1.0, bandwidth_sensitive_fraction=1.0,
                         added_ms=delta, candidate_tail_ms=predicted,
                         target_minus_candidate_ms=target['tail_ms']-predicted,
                         fraction_of_observed_gap=delta/gap))
    report = dict(status='SENSITIVITY_ONLY_NOT_VALIDATED_PREDICTION',
                  source_rs_union_ms=source_rs, source_tail_ms=source['tail_ms'],
                  target_tail_ms=target['tail_ms'], observed_gap_ms=gap,
                  rs_event_counts=[len(x) for x in rs], scenarios=rows,
                  formula='T_candidate = T_source + t_RS_source * f * (q / r - 1)',
                  parameters={'r':'mixed/intra effective bandwidth ratio',
                              'q':'target/source effective transferred-byte ratio',
                              'f':'source RS duration fraction treated as bandwidth-sensitive'},
                  limits=['rank0 iter60 only; source RS includes potential waiting',
                          'q=1 and f=1 are assumptions, not measured facts',
                          'extra target RS not attributed to DP/EDP; not modeled by r',
                          'AG, optimizer and all other windows unchanged',
                          'no absolute bandwidth or independent byte baseline available here',
                          'no scenario promoted to the frozen full-step prediction'])
    assert abs(rows[0]['candidate_tail_ms']-source['tail_ms']) < 1e-9
    assert all(a['candidate_tail_ms'] <= b['candidate_tail_ms'] for a,b in zip(rows,rows[1:]))
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out / 'report.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(inputs={str(p):digest(p) for p in inputs},
                    code={str(Path(__file__).resolve()):digest(Path(__file__))},
                    outputs={str(output.resolve()):digest(output)})
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()

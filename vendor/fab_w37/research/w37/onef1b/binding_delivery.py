"""Publish compact, hash-verified research evidence without changing model outputs."""
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'results/w37/A/binding-20260908-r2'
DOC = ROOT/'docs/w37/1f1b/binding'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def dump(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

def main():
    import pandas as pd
    records, stages = {}, {}
    for stage in ['audit','model','review','evaluate']:
        marker = RUN/stage/'complete.json'
        data = json.loads(marker.read_text())
        assert data['status'] == 'PASS'
        for name, expected in data['files'].items():
            p = RUN/stage/name
            assert sha(p) == expected, p
            records[str(p.relative_to(ROOT))] = expected
        stages[stage] = dict(complete_sha256=sha(marker), elapsed_s=data['elapsed_s'])
    evidence = DOC/'evidence'
    evidence.mkdir(exist_ok=True)
    copies = {}
    for stage, names in {
        'audit':['summary.json','all_58_key_audit.csv','node_key_schemas.csv'],
        'model':['predictions.csv','runtime_shape_check.json','prediction_seal.json'],
        'review':['review.json','source_method_comparison.csv','tail_propagation.csv','update_effect_summary.csv'],
        'evaluate':['summary.json','metrics.csv','iteration_results.csv','phase_iteration_results.csv','phase_metrics.csv','phase_regression.csv'],
    }.items():
        for name in names:
            source = RUN/stage/name
            dest = evidence/(stage+'_'+name)
            content = source.read_bytes()
            # Keep committed hashes stable under the repository's LF normalization.
            if source.suffix == '.csv': content = content.replace(b'\r\n', b'\n')
            dest.write_bytes(content)
            copies[str(dest.relative_to(DOC))] = dict(sha256=sha(dest), source=str(source.relative_to(ROOT)),
                source_sha256=sha(source), copy_transform='CSV CRLF to LF; values unchanged' if source.suffix=='.csv' else 'byte exact')
    dump(DOC/'delivery_inputs.json', dict(status='PASS', run=str(RUN.relative_to(ROOT)), files=copies))
    prior = ROOT/'results/w37/A/binding-20260908'
    pd.testing.assert_frame_equal(pd.read_csv(prior/'model/predictions.csv'), pd.read_csv(RUN/'model/predictions.csv'), check_exact=True)
    def updates(base):
        return pd.read_csv(base/'model/node_updates.csv.gz').sort_values(['variant','node_id']).sort_index(axis=1).reset_index(drop=True)
    old, new = updates(prior), updates(RUN)
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    selected = new[new.variant.eq('fine_both')]
    coverage = dict(status='PASS', exact_r1_r2_update_rows=len(new), exact_r1_r2_predictions=True,
        target_ranks=int(selected['rank'].nunique()), target_microbatches=[int(x) for x in sorted(selected.microbatch.unique())],
        updated_nodes_per_pp_stage={str(k):int(v) for k,v in selected.groupby('pp_stage').size().items()})
    dump(DOC/'coverage_and_reproduction.json', coverage)
    dump(DOC/'research_acceptance.json', dict(status='PASS', stages=stages, output_sha256=records,
        input_manifest_sha256=sha(DOC/'inputs.json'), delivery_manifest_sha256=sha(DOC/'delivery_inputs.json'),
        coverage_and_reproduction_sha256=sha(DOC/'coverage_and_reproduction.json'),
        formal_version='v685 unchanged', target_label='already exposed development evaluation',
        source_full_iteration_validation=False, wall_floor_independently_validated=False,
        result='Independent candidate reduces 1F1B error but tail regresses; do not promote.',
        resource='1 worker thread; 4 GB scheduler declaration, not an OS memory limit; 900 s timeout per stage; no raw scan'))
    print(json.dumps(dict(status='PASS', compact_files=len(copies), verified_output_files=len(records))))

if __name__ == '__main__': main()

"""Compare regenerated scientific content to immutable, relocated W37 evidence."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
from x10000_analysis.v610 import ROOT, artifact_root, manifest, resolve_artifact, sha, dump, STAGES

OLD = '/home/zjb/Desktop/worktrees/fab-w37-1f1b/results/w37/A/'


def content_hash(path):
    h = hashlib.sha256()
    with (gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')) as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def compare_table(actual, expected):
    # Identical decompressed bytes prove every field, including integer-ns costs,
    # zero-cost nodes, row order, bindings and dependency semantics.
    a, b = content_hash(actual), content_hash(expected)
    count = 0
    if a == b:
        for chunk in pd.read_csv(actual, chunksize=32768, low_memory=False):
            count += len(chunk)
        return dict(rows=count, comparison='identical_decompressed_bytes', content_sha256=a)
    from itertools import zip_longest
    for left, right in zip_longest(pd.read_csv(actual, chunksize=32768, low_memory=False),
                                  pd.read_csv(expected, chunksize=32768, low_memory=False)):
        assert left is not None and right is not None, 'Different row counts'
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        count += len(left)
    return dict(rows=count, comparison='all_cells_exact', actual_content_sha256=a, expected_content_sha256=b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    m, base = manifest(), artifact_root()
    result = dict(status='RUNNING', source_commit=m['source_commit'], audit_commit=m['audit_commit'],
                  artifact_root=str(base), release=str(args.release.resolve()), binding=str(args.binding.resolve()),
                  checks=[], raw_trace_reads=0)

    def check(name, action):
        try:
            detail = action()
            result['checks'].append(dict(name=name, status='PASS', detail=detail))
        except Exception as error:
            result['checks'].append(dict(name=name, status='FAIL', error=str(error)))
        print(name, result['checks'][-1]['status'], flush=True)

    def artifacts():
        for r in m['objects']:
            f = base/r['relative']
            assert f.is_file() and f.stat().st_size == r['size_bytes'] and sha(f) == r['sha256'], str(f)
        return dict(files=len(m['objects']), bytes=sum(r['size_bytes'] for r in m['objects']))
    check('all_frozen_artifact_hashes', artifacts)

    def code():
        records = json.loads((ROOT/'docs/v610/imported_code.json').read_text())['files']
        for r in records:
            assert sha(ROOT/r['path']) == r['sha256'], r['path']
        return dict(byte_preserved_files=len(records))
    check('all_vendored_code_hashes', code)

    for pipeline, run, prefix in [('release', args.release, 'v610-release/'),
                                  ('binding', args.binding, 'binding-20260908-r2/')]:
        for logical in m['logical_paths']:
            if not logical.startswith(OLD+prefix) or not logical.endswith(('.csv', '.csv.gz')):
                continue
            rel = logical.removeprefix(OLD+prefix)
            check(pipeline+':'+rel, lambda r=run/rel, s=logical: compare_table(r, resolve_artifact(s, m, base)))
        for stage in STAGES[pipeline]:
            def stage_evidence(run=run, stage=stage):
                directory = run/stage
                completion = json.loads((directory/'complete.json').read_text())
                assert completion['status'] == 'PASS'
                for rel, expected in completion['files'].items():
                    assert sha(directory/rel) == expected, str(directory/rel)
                access = json.loads((directory/'adapter_access.json').read_text())
                inputs = json.loads((directory/'resolved_inputs.json').read_text())
                evaluator = {r['resolved_path'] for r in inputs.values() if r['role'] == 'evaluator'}
                assert not access['raw_trace_scan']
                if stage not in ('evaluate', 'render'):
                    assert not evaluator.intersection(access['interpreted_data_reads']), 'Target data read before evaluator'
                command = json.loads((directory/'command.json').read_text())
                assert '--unshare-net' in command['argv'] and '--ro-bind' in command['argv']
                return dict(files=len(completion['files']), reads=access['interpreted_data_reads'],
                            original_providers_hidden=command['hidden_original_projects'])
            check(pipeline+':'+stage+':seal_and_input_boundary', stage_evidence)

    for rel in ['model/release.json', 'evaluate/metrics.json', 'render/full_iter_payload.json', 'render/view_checks.json']:
        def json_equal(rel=rel):
            actual = json.loads((args.release/rel).read_text())
            expected = json.loads(resolve_artifact(OLD+'v610-release/'+rel,m,base).read_text())
            assert actual == expected, 'JSON scientific content differs'
            return 'exact JSON content'
        check('release:'+rel, json_equal)

    def coverage():
        nodes = pd.read_csv(args.release/'model/nodes.csv.gz', low_memory=False)
        bindings = pd.read_csv(args.release/'model/node_parameter_bindings.csv.gz')
        components = ['compute_exposed_ns_model','compute_overlap_ns_model','network_service_ns_model',
                      'software_sync_ns_model','framework_residual_ns_model']
        assert len(nodes) == 327746 and nodes[components].sum(axis=1).eq(nodes.duration_ns).all()
        assert len(bindings) == bindings.node_id.nunique() == 70192
        assert bindings.source_key.nunique() == 4387
        assert bindings['rank'].nunique() == 224 and bindings.pp_stage.nunique() == 14 and bindings.microbatch.nunique() == 3
        assert len(pd.read_csv(args.release/'model/fine_parameters.csv')) == 6765
        predictions = pd.read_csv(args.release/'model/predictions.csv').iloc[0]
        assert abs(predictions.profiler_ms-predictions.raw_ms-604.439668124998) < 1e-8
        assert abs(predictions.training_ms-predictions.profiler_ms-1372.453390) < 1e-8
        assert abs(predictions.entry_ms-1265.388210) < 1e-8
        return dict(nodes=len(nodes), edges=364784, bound_nodes=len(bindings), fine_parameters=6765,
                    bound_keys=4387, ranks=224, pp_stages=14, microbatches=3,
                    zero_duration_nodes=int(nodes.duration_ns.eq(0).sum()),
                    component_totals_ns={c:int(nodes[c].sum()) for c in components},
                    prediction=predictions.to_dict())
    check('complete_cost_and_parameter_coverage', coverage)
    result['status'] = 'PASS' if all(r['status']=='PASS' for r in result['checks']) else 'FAIL'
    result['limitations'] = ['already exposed target development, not blind validation',
                            'source full-compatible replay and floor physical provenance remain unvalidated',
                            'raw-to-all-historical-cost regeneration not completed; raw data not relocated']
    dump(args.output, result)
    raise SystemExit(0 if result['status']=='PASS' else 1)


if __name__ == '__main__':
    main()

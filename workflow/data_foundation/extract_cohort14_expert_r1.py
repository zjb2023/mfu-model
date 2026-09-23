"""Trace facts only: sequential routed experts, no cost fitting or DAG mutation.

Ownership: CPU (pid,tid) runtime containment -> exact correlation -> device.
Identity: configured expert weight shapes + FC1/gate/FC2 order + MoE envelope,
then frozen unique name/sequence autograd pairs. No guessed correlation offsets.
"""
import argparse
import bisect
import collections
import csv
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation')


def load(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def union_ms(intervals):
    total = 0.
    end = float('-inf')
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total / 1000


def csv_write(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, list(rows[0]) if rows else ['empty'])
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def extract(item, cfg, pairs):
    raw = Path(item['path']).read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    del raw
    es = data['traceEvents']
    indexed = list(enumerate(es))
    step = [e for e in es if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('ProfilerStep#')]
    assert len(step) == 1
    origin = step[0]['ts']
    h, inter, etp, ep = [cfg[k] for k in ['hidden_size', 'moe_ffn_hidden_size', 'expert_tensor_parallel_size', 'expert_model_parallel_size']]
    local_experts = cfg['num_experts'] // ep
    shapes = {'FC1': [2 * inter // etp, h], 'FC2': [h, inter // etp]}
    forward_pairs = {int(p['forward_id'].split(':e')[1]): p for p in pairs}
    selected = []
    for i, e in indexed:
        dims = e.get('args', {}).get('Input Dims', [])
        if e.get('name') != '_Linear' or not dims:
            continue
        for kind, shape in shapes.items():
            if dims[0] == shape:
                assert i in forward_pairs, ('unpaired expert shape', item['key'], i)
                selected.append((i, e, kind, int(forward_pairs[i]['microbatch'])))
    selected.sort(key=lambda v: v[1]['ts'])
    gates = [(i, e) for i, e in indexed if e.get('name') == 'SwiGLUFunction']
    permutes = [(i, e) for i, e in indexed if e.get('name') == '_moe_permute_mask_map']
    unpermutes = [(i, e) for i, e in indexed if e.get('name') == '_moe_unpermute_mask_map']
    modules = {}
    ops = []
    assert len(selected) % 2 == 0
    for pos in range(0, len(selected), 2):
        fi, f, fk, mb = selected[pos]
        si, s, sk, smb = selected[pos + 1]
        assert (fk, sk, mb) == ('FC1', 'FC2', smb)
        fdim = f['args']['Input Dims'][1]
        sdim = s['args']['Input Dims'][1]
        assert len(fdim) == len(sdim) == 2 and fdim[0] == sdim[0]
        assert fdim[1] == h and sdim[1] == inter // etp
        before = [(i, e) for i, e in permutes if (e['pid'], e['tid']) == (f['pid'], f['tid']) and e['ts'] < f['ts']]
        after = [(i, e) for i, e in unpermutes if (e['pid'], e['tid']) == (s['pid'], s['tid']) and e['ts'] > s['ts']]
        assert before and after
        pi, p = max(before, key=lambda v: v[1]['ts'])
        ui, u = min(after, key=lambda v: v[1]['ts'])
        gate = [(i, e) for i, e in gates if (e['pid'], e['tid']) == (f['pid'], f['tid']) and f['ts'] + f['dur'] <= e['ts'] and e['ts'] + e['dur'] <= s['ts'] + .01]
        assert len(gate) == 1
        gi, g = gate[0]
        assert g['args']['Input Dims'][0] == [fdim[0], 2 * inter // etp]
        key = (mb, pi, ui)
        module = modules.setdefault(key, {'microbatch': mb, 'permute_event': pi, 'unpermute_event': ui, 'experts': []})
        ordinal = len(module['experts'])
        module['experts'].append({'local_expert_ordinal': ordinal, 'token_rows': fdim[0], 'FC1_event': fi, 'FC2_event': si, 'gate_event': gi})
        for kind, index, event in [('FC1', fi, f), ('gate', gi, g), ('FC2', si, s)]:
            pair = forward_pairs[index]
            bi = int(pair['backward_id'].split(':e')[1])
            b = es[bi]
            assert b['name'] == event['name'] + 'Backward'
            assert b['args']['Sequence number'] == event['args']['Sequence number']
            for phase, oi, op in [('F', index, event), ('B', bi, b)]:
                ops.append(dict(case=item['case'], iteration=item['iteration'], rank=item['rank'],
                                microbatch=mb, module_anchor=pi, local_expert_ordinal=ordinal,
                                expert_id_status='local ordinal from sequential order, not checkpoint ID',
                                phase=phase, component=kind, token_rows=fdim[0], hidden=h,
                                intermediate_local=inter // etp, etp=etp, ep=ep,
                                event=oi, forward_event=index, sequence=op['args']['Sequence number'],
                                input_dims=op['args'].get('Input Dims'), cpu_pid=op['pid'], cpu_tid=op['tid'],
                                cpu_start_us=op['ts'], cpu_end_us=op['ts'] + op['dur'], cpu_ms=op['dur'] / 1000,
                                devices=[]))
    module_list = sorted(modules.values(), key=lambda m: es[m['permute_event']]['ts'])
    layer_counts = collections.Counter()
    for module in module_list:
        mb = module['microbatch']
        module['moe_ordinal_in_microbatch'] = layer_counts[mb]
        layer_counts[mb] += 1
        assert len(module['experts']) == local_experts
        module['token_rows_sum'] = sum(x['token_rows'] for x in module['experts'])
    # Runtime lookup, only exact correlation. CPU events may be queued far before GPU execution.
    threads = collections.defaultdict(list)
    for oi, op in enumerate(ops):
        threads[op['cpu_pid'], op['cpu_tid']].append((op['cpu_start_us'], oi))
    starts = {}
    for key, values in threads.items():
        values.sort()
        starts[key] = [x[0] for x in values]
    launches = collections.defaultdict(list)
    for i, e in indexed:
        if e.get('cat') not in ['cuda_runtime', 'cuda_driver', 'privateuse1_runtime', 'privateuse1_driver']:
            continue
        corr = e.get('args', {}).get('correlation')
        key = (e.get('pid'), e.get('tid'))
        if corr is None or key not in threads:
            continue
        pos = bisect.bisect_right(starts[key], e['ts']) - 1
        if pos < 0:
            continue
        oi = threads[key][pos][1]
        if e['ts'] + e.get('dur', 0) <= ops[oi]['cpu_end_us'] + .01:
            launches[corr].append((i, oi))
    ambiguous = []
    for i, e in indexed:
        if e.get('cat') not in ['kernel', 'gpu_memcpy', 'gpu_memset']:
            continue
        matched = launches.get(e.get('args', {}).get('correlation'), [])
        owners = {v[1] for v in matched}
        if len(owners) > 1:
            ambiguous.append(i)
        if len(owners) != 1:
            continue
        owner = next(iter(owners))
        ops[owner]['devices'].append(dict(event=i, name=e['name'], category=e['cat'],
                                         start_us=e['ts'], end_us=e['ts'] + e['dur'], stream=e.get('args', {}).get('stream'),
                                         correlation=e.get('args', {}).get('correlation'),
                                         external_id=e.get('args', {}).get('External id'),
                                         runtime_events=[j for j, _ in matched]))
    assert not ambiguous, ('ambiguous device ownership', ambiguous)
    by_anchor = {m['permute_event']: m for m in module_list}
    for op in ops:
        module = by_anchor[op['module_anchor']]
        op['moe_ordinal_in_microbatch'] = module['moe_ordinal_in_microbatch']
        ds = op['devices']
        iv = [(d['start_us'], d['end_us']) for d in ds]
        gemm = [d for d in ds if 'gemm' in d['name'].lower()]
        op.update(device_count=len(ds), gemm_count=len(gemm),
                  gpu_union_ms=union_ms(iv),
                  gemm_union_ms=union_ms([(d['start_us'], d['end_us']) for d in gemm]),
                  gpu_envelope_ms=(max(y for x, y in iv) - min(x for x, y in iv)) / 1000 if iv else None,
                  start_from_profiler_ms=(min(x for x, y in iv) - origin) / 1000 if iv else None,
                  end_from_profiler_ms=(max(y for x, y in iv) - origin) / 1000 if iv else None,
                  attribution_status='EXACT_CORRELATION' if ds else 'NO_ASSOCIATED_DEVICE')
        op['envelope_without_owned_activity_ms'] = op['gpu_envelope_ms'] - op['gpu_union_ms'] if iv else None
    owned = [d['event'] for op in ops for d in op['devices']]
    assert len(owned) == len(set(owned))
    return dict(key=item['key'], path=item['path'], resolved=str(Path(item['path']).resolve()), sha256=raw_sha,
                profiler_label=step[0]['name'], profiler_ms=step[0]['dur'] / 1000,
                config=cfg, distributed_info=data.get('distributedInfo'),
                modules=module_list, operators=ops, ambiguous_devices=ambiguous,
                status='FACTS_EXTRACTED_IDENTITY_AND_GPU_AUDIT_REQUIRED')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='+', default=['2111', '2112'])
    ap.add_argument('--ranks', nargs='+', type=int, default=list(range(16)))
    ap.add_argument('--iterations', nargs='+', type=int, default=[53, 58])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest_path = SOURCE / 'multi-strategy-r2/input_manifest.json'
    config_path = SOURCE / 'shared-phase-r1/static_configs.json'
    pair_path = SOURCE / 'multi-strategy-r2/autograd_pairs.csv'
    configs = {c['case']: c['config'] for c in load(config_path)['cases']}
    items = [x for x in load(manifest_path)['items'] if x['case'] in args.cases and x['rank'] in args.ranks and x['iteration'] in args.iterations]
    keys = {x['key'] for x in items}
    pairs = collections.defaultdict(list)
    with pair_path.open() as f:
        for p in csv.DictReader(f):
            if p['key'] in keys:
                pairs[p['key']].append(p)
    flat, references, summaries = [], [], []
    start = time.monotonic()
    for item in items:
        result = extract(item, configs[item['case']], pairs[item['key']])
        path = args.out / (item['key'] + '.json')
        path.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
        references.append({k: result[k] for k in ['key', 'path', 'resolved', 'sha256']})
        flat.extend({k: v for k, v in op.items() if k != 'devices'} for op in result['operators'])
        row = dict(key=item['key'], modules=len(result['modules']), operators=len(result['operators']),
                   no_device=sum(not o['devices'] for o in result['operators']),
                   FC_gemm_counts=dict(collections.Counter((o['phase'] + ':' + str(o['gemm_count'])) for o in result['operators'] if o['component'] != 'gate')),
                   output_sha256=digest(path))
        summaries.append(row)
        print(json.dumps(row), flush=True)
    csv_write(args.out / 'operators.csv', flat)
    (args.out / 'summary.json').write_text(json.dumps(dict(status='EXTRACTION_ONLY', traces=len(items), elapsed_s=time.monotonic() - start, rows=summaries), indent=2) + '\n')
    (args.out / 'manifest.json').write_text(json.dumps(dict(code={str(Path(__file__).resolve()): digest(Path(__file__))},
        inputs={str(p): digest(p) for p in [manifest_path, config_path, pair_path]}, raw=references,
        outputs={p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()}), indent=2) + '\n')


if __name__ == '__main__':
    main()

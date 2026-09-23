"""Explain layer curves on 224/256: observed work/cost, never prediction fitting."""
import argparse
import bisect
import collections
import csv
import hashlib
import json
from pathlib import Path
from extract_cohort14_expert_r1 import digest, union_ms, csv_write

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'


def inside(e, container):
    return e.get('pid') == container['pid'] and container['ts'] <= e['ts'] and e['ts'] + e.get('dur', 0) <= container['ts'] + container['dur'] + .01


def extract(ref):
    raw = Path(ref['path']).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if ref.get('sha256'):
        assert sha == ref['sha256']
    data = json.loads(raw)
    del raw
    es = data['traceEvents']
    rank = data['distributedInfo']['rank']
    ep = next(g['ranks'] for g in data['distributedInfo']['pg_config'] if g['pg_desc'] == 'EXPERT_MODEL_PARALLEL_GROUP')
    assert len(ep) == 8 and rank in ep
    cpu = [(i, e) for i, e in enumerate(es) if e.get('cat') == 'cpu_op']
    fs = sorted([(i, e) for i, e in enumerate(es) if e.get('cat') == 'user_annotation' and e.get('name') == 'forward_step'], key=lambda x: x[1]['ts'])
    bs = sorted([(i, e) for i, e in enumerate(es) if e.get('cat') == 'user_annotation' and e.get('name') == 'backward_step'], key=lambda x: x[1]['ts'])
    assert len(fs) == len(bs) == (4 if ref['world'] == 256 else 3)
    windows, rows, checkpoint_map = [], [], {}

    def add_window(row, phase, part, index, event):
        windows.append(dict(row=row, phase=phase, component=part, event=index, pid=event['pid'], tid=event['tid'], start=event['ts'], end=event['ts'] + event['dur'], devices=[]))

    def synthetic(lo, hi, anchor):
        assert lo <= hi
        return dict(pid=anchor['pid'], tid=anchor['tid'], ts=lo, dur=hi - lo)

    for mb, (fi, f) in enumerate(fs):
        checkpoints = sorted([(i, e) for i, e in cpu if e['name'] == 'CheckpointFunction' and inside(e, f)], key=lambda x: x[1]['ts'])
        assert len(checkpoints) == 4
        for local, (ci, checkpoint) in enumerate(checkpoints):
            seq = checkpoint['args']['Sequence number']
            assert seq not in checkpoint_map
            local_ops = [(i, e) for i, e in cpu if inside(e, checkpoint)]
            splits = []
            for i, e in local_ops:
                a = e.get('args', {})
                if e['name'] == 'aten::split_with_sizes' and a.get('Input Dims', [[]])[0] and len(a['Input Dims'][0]) == 2:
                    counts = json.loads(a['Concrete Inputs'][1])
                    if len(counts) == 20:
                        splits.append((i, e, counts))
            assert len(splits) == 2, (ref['path'], mb, local, 'forward split count', len(splits))
            splits.sort(key=lambda x: x[1]['ts'])
            (_, split1, counts), (_, split2, counts2) = splits
            assert counts == counts2 and sum(counts) == split1['args']['Input Dims'][0][0]
            assert split1['args']['Input Dims'][0][1] == 5120 and split2['args']['Input Dims'][0][1] == 1536
            acts = [(i, e) for i, e in local_ops if e['name'] == 'aten::silu' and split1['ts'] < e['ts'] < split2['ts']]
            assert len(acts) == 1
            ai, act = acts[0]
            ends = [(i, e) for i, e in local_ops if e['name'] == '_moe_unpermute_mask_map' and e['ts'] > split2['ts']]
            assert len(ends) == 1
            ui, end = ends[0]
            row = len(rows)
            values = dict(world=ref['world'], iteration=ref['iteration'], stage=ref['stage'], rank=rank, ep_group=ep,
                          mb=mb, local_layer=local, layer=3 + (ref['stage'] - 1) * 4 + local,
                          tokens_per_expert=counts, token_rows=sum(counts), nonempty_experts=sum(x > 0 for x in counts),
                          max_expert_rows=max(counts), min_expert_rows=min(counts),
                          checkpoint_forward_event=ci, checkpoint_sequence=seq, forward_step_event=fi)
            rows.append(values)
            checkpoint_map[seq] = row
            add_window(row, 'F', 'FC1', ci, synthetic(split1['ts'], act['ts'], checkpoint))
            add_window(row, 'F', 'inter_FC_local', ai, synthetic(act['ts'], split2['ts'], checkpoint))
            add_window(row, 'F', 'FC2', ui, synthetic(split2['ts'], end['ts'], checkpoint))
    for mb, (bi, b) in enumerate(bs):
        checkpoints = sorted([(i, e) for i, e in cpu if e['name'] == 'CheckpointFunctionBackward' and inside(e, b)], key=lambda x: x[1]['ts'])
        assert len(checkpoints) == 4
        for order, (ci, c) in enumerate(checkpoints):
            row = checkpoint_map[c['args']['Sequence number']]
            assert rows[row]['mb'] == mb and rows[row]['local_layer'] == 3 - order
            rows[row].update(checkpoint_backward_event=ci, backward_execution_order=order, backward_step_event=bi)
            local_ops = [(i, e) for i, e in cpu if inside(e, c)]
            recompute = sorted([(i, e) for i, e in local_ops if e['name'] == '_GroupedLinear'], key=lambda x: x[1]['ts'])
            backward = [(i, e) for i, e in local_ops if e['name'] == '_GroupedLinearBackward']
            assert len(recompute) == len(backward) == 2
            paired = []
            for fc, (i, e) in enumerate(recompute, 1):
                counts = json.loads(e['args']['Concrete Inputs'][1])
                assert counts == rows[row]['tokens_per_expert'], 'F/recompute route mismatch'
                assert e['args']['Input Dims'][0][0] == sum(counts)
                matches = [(j, x) for j, x in backward if x['args']['Sequence number'] == e['args']['Sequence number']]
                assert len(matches) == 1
                j, x = matches[0]
                assert x['args']['Input Dims'][0][0] == sum(counts)
                add_window(row, 'recompute', 'FC' + str(fc), i, e)
                add_window(row, 'backward', 'FC' + str(fc), j, x)
                paired.append((j, x))
            (i, left), (j, right) = recompute
            add_window(row, 'recompute', 'inter_FC_local', i, synthetic(left['ts'] + left['dur'], right['ts'], left))
            (j, b1), (k, b2) = paired
            add_window(row, 'backward', 'inter_FC_local', k, synthetic(b2['ts'] + b2['dur'], b1['ts'], b2))
    threads = collections.defaultdict(list)
    for j, w in enumerate(windows):
        threads[w['pid'], w['tid']].append((w['start'], j))
    starts = {}
    for key, values in threads.items():
        values.sort()
        starts[key] = [x for x, _ in values]
        for (_, a), (_, b) in zip(values, values[1:]):
            assert windows[a]['end'] <= windows[b]['start'] + .01
    ownership = collections.defaultdict(list)
    for i, e in enumerate(es):
        if e.get('cat') not in ['privateuse1_runtime', 'privateuse1_driver'] or 'correlation' not in e.get('args', {}):
            continue
        key = e['pid'], e['tid']
        if key not in threads:
            continue
        j = bisect.bisect_right(starts[key], e['ts']) - 1
        if j < 0:
            continue
        wi = threads[key][j][1]
        if e['ts'] + e.get('dur', 0) <= windows[wi]['end'] + .01:
            ownership[e['args']['correlation']].append((wi, i))
    for i, e in enumerate(es):
        if e.get('cat') not in ['kernel', 'gpu_memcpy', 'gpu_memset']:
            continue
        hits = ownership.get(e.get('args', {}).get('correlation'), [])
        owners = {j for j, _ in hits}
        assert len(owners) <= 1
        if not owners:
            continue
        wi = next(iter(owners))
        windows[wi]['devices'].append(dict(event=i, name=e['name'], start_us=e['ts'], end_us=e['ts'] + e['dur'], runtime_events=[j for _, j in hits]))
    for w in windows:
        ds = w['devices']
        gemm = [d for d in ds if 'gemm' in d['name'].lower()]
        row = rows[w['row']]
        if w['component'] != 'inter_FC_local':
            assert gemm or row['token_rows'] == 0, ('nonempty FC without GEMM', ref['path'], row, w['phase'], w['component'])
        prefix = w['phase'] + '_' + w['component']
        row[prefix + '_gpu_ms'] = union_ms([(d['start_us'], d['end_us']) for d in ds])
        row[prefix + '_gemm_ms'] = union_ms([(d['start_us'], d['end_us']) for d in gemm])
        row[prefix + '_gemm_count'] = len(gemm)
        row[prefix + '_gap_ms'] = (max(d['end_us'] for d in ds) - min(d['start_us'] for d in ds)) / 1000 - row[prefix + '_gpu_ms'] if ds else 0.
    for row in rows:
        for phase in ['F', 'recompute', 'backward']:
            row[phase + '_FC_gemm_ms'] = row[phase + '_FC1_gemm_ms'] + row[phase + '_FC2_gemm_ms']
        row['B_FC_gemm_ms'] = row['recompute_FC_gemm_ms'] + row['backward_FC_gemm_ms']
    return dict(ref, sha256=sha, rank=rank, ep_group=ep, rows=rows, windows=windows,
                scope='First EP8 replica if rank belongs; no cross-host timestamps compared; inter-FC local activity is not pure activation FLOPs.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worlds', nargs='+', type=int, default=[256])
    ap.add_argument('--iterations', nargs='+', type=int, default=[40, 60, 80])
    ap.add_argument('--stages', nargs='+', type=int)
    ap.add_argument('--ep8-iter60', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rp = BASE / 'b-position-curves-r2/raw-provenance.json'
    refs = [r for r in json.loads(rp.read_text()) if r['world'] in a.worlds and r['iteration'] in a.iterations and (not a.stages or r['stage'] in a.stages)]
    extra = []
    if a.ep8_iter60:
        for ref in refs:
            if ref['iteration'] != 60:
                continue
            for rank in range(ref['stage'] * 16 + 1, ref['stage'] * 16 + 8):
                paths = list(Path(ref['path']).parent.glob(f'rank{rank}.*.pt.trace.json'))
                assert len(paths) == 1
                extra.append(dict(world=ref['world'], iteration=60, stage=ref['stage'], path=str(paths[0])))
    refs += extra
    all_rows, done = [], []
    for ref in refs:
        result = extract(ref)
        name = f'w{ref["world"]}-i{ref["iteration"]}-s{ref["stage"]}-r{result["rank"]}.json'
        (a.out / name).write_text(json.dumps(result, separators=(',', ':')) + '\n')
        all_rows.extend(result['rows'])
        done.append({k: result[k] for k in ['world', 'iteration', 'stage', 'rank', 'path', 'sha256']})
        print(name, len(result['rows']), 'layers PASS', flush=True)
    csv_write(a.out / 'layers.csv', all_rows)
    codes = [Path(__file__).resolve(), Path(__file__).with_name('extract_cohort14_expert_r1.py')]
    (a.out / 'manifest.json').write_text(json.dumps(dict(code={str(p): digest(p) for p in codes}, inputs={str(rp): digest(rp)}, raw=done,
        outputs={p.name: digest(p) for p in a.out.iterdir() if p.is_file()}), indent=2) + '\n')
    print('COMPLETE', len(done), 'traces', len(all_rows), 'layers')


if __name__ == '__main__':
    main()

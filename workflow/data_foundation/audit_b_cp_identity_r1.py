"""Audit B0 CP identities without inferring tensor roles from message sizes.

Uses eight explicitly listed traces, never discovers/scans a trace directory.
Output must be new; original data and predictive costs are read-only.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    source = BASE / 'b-cost-blocks-r1/evidence.json'
    rows, inputs, signatures = [], [], []
    for x in json.loads(source.read_text()):
        raw = Path(x['path'])
        blob = raw.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == x['sha256']
        events = json.loads(blob)['traceEvents']
        del blob
        parents = [(i, e) for i, e in enumerate(events)
                   if e.get('cat') == 'cpu_op' and e.get('name') in
                   ('AttnFuncWithCPAndQKVOA2A', 'AttnFuncWithCPAndQKVOA2ABackward')]
        for unit in range(4):
            for phase, expected in [('recompute', [384, 384, 256, 256]),
                                    ('backward', [256, 256, 384, 384, 256])]:
                ds = sorted((d for d in x['devices'] if d['unit'] == unit
                             and d['phase'] == phase and d['kind'] == 'CP_all_to_all'),
                            key=lambda d: events[d['launch_event']]['ts'])
                sizes = []
                for ordinal, d in enumerate(ds):
                    e, launch = events[d['event']], events[d['launch_event']]
                    args = e['args']
                    owners = [(i, p) for i, p in parents if p['pid'] == launch['pid']
                              and p['tid'] == launch['tid'] and p['ts'] <= launch['ts']
                              and p['ts'] + p['dur'] >= launch['ts'] + launch['dur']]
                    assert len(owners) == 1, (x['rank'], unit, phase, ordinal, owners)
                    owner_id, owner = owners[0]
                    expected_owner = 'AttnFuncWithCPAndQKVOA2A' + ('Backward' if phase == 'backward' else '')
                    assert owner['name'] == expected_owner
                    assert args['dtype'] == 'BFloat16' and args['Group size'] == 2
                    mib = int(args['In msg nelems']) * 2 / 2**20
                    sizes.append(mib)
                    rows.append(dict(rank=x['rank'], checkpoint_unit=unit, phase=phase,
                                     ordinal=ordinal, slot=f'{phase}:CP{ordinal}',
                                     input_mib=mib, input_bytes=int(args['In msg nelems']) * 2,
                                     group_size=2, group_name=args['Process Group Name'],
                                     trace_group_ranks=args.get('Process Group Ranks'),
                                     stream=args['stream'], device_event=d['event'],
                                     launch_event=d['launch_event'], cpu_owner_event=owner_id,
                                     cpu_owner=owner['name'], correlation=args['correlation'],
                                     start_us=e['ts'], duration_ms=e['dur'] / 1000,
                                     tensor_role='UNRESOLVED_INSTALLED_IMPLEMENTATION_REQUIRED',
                                     cost_basis='observed_device_envelope_not_transport_service'))
                assert sizes == expected, (x['rank'], unit, phase, sizes)
                signatures.append(dict(rank=x['rank'], checkpoint_unit=unit, phase=phase, input_mib=sizes))
        inputs.append(dict(path=str(raw), sha256=x['sha256']))
        print('PASS CP ancestry/signature rank', x['rank'], flush=True)
        del events
    assert len(rows) == 288 and len(signatures) == 64
    summary = dict(status='PARTIAL', trace_checks='PASS', ranks=list(range(8, 16)),
                   scope='32GPU iter60 PP1 B0; four checkpoint execution units per rank',
                   cp_events=288, recompute_events=128, backward_events=160,
                   signature_checks=64, mathematical_identity='NOT_CONFIRMED',
                   limitations=['Input bytes are tensor volume, not on-wire bytes.',
                                'Ordinal CP0 in backward is NOT an asserted match to forward CP0.',
                                'CPU nesting and same-stream order do not close cross-stream/rank edges.',
                                'Checkpoint execution unit is not a verified model layer index.',
                                'No target iter70 data used; no prediction parameters changed.'])
    a.out.mkdir(parents=True)
    for name, value in [('events', rows), ('signatures', signatures), ('summary', summary)]:
        (a.out / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    body = []
    for r in range(8, 16):
        body.append(f'<details{" open" if r == 8 else ""}><summary>rank {r} · 四个执行单元</summary><table><thead><tr><th>单元</th><th>阶段/顺序</th><th>输入 MiB</th><th>设备耗时 ms</th><th>设备事件 / CPU父事件</th></tr></thead><tbody>')
        for d in rows:
            if d['rank'] == r:
                body.append(f'<tr><td>{d["checkpoint_unit"]}</td><td>{html.escape(d["slot"])}</td><td>{d["input_mib"]:g}</td><td>{d["duration_ms"]:.6f}</td><td>{d["device_event"]} / {d["cpu_owner_event"]}</td></tr>')
        body.append('</tbody></table></details>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B CP 身份证据 r1</title>
<style>body{font:16px/1.6 "Noto Sans CJK SC",sans-serif;color:#24344a;background:#f3f6fa;margin:24px auto;padding:0 20px;max-width:1100px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:8px;border-bottom:1px solid #cbd5e0}details{background:white;padding:12px;margin:14px 0;overflow:auto}summary{cursor:pointer}a{color:#315d98}code{overflow-wrap:anywhere}</style>
<body><h1>B 中 CP：已证实的调用身份</h1><p>32卡 iter60 · PP1 B0 · ranks8–15。288个设备事件、64组签名检查通过。这里只列实测证据，不是预测精度。</p>
<p>重计算：384 → 384 → 256 → 256 MiB（4次）；真正反向：256 → 256 → 384 → 384 → 256 MiB（5次）。八卡四单元一致。</p>
<p>父函数分别为 <code>AttnFuncWithCPAndQKVOA2A</code> 与 <code>AttnFuncWithCPAndQKVOA2ABackward</code>。前两次反向通信的具体张量身份待核对安装版实现；不将消息量相同当作同一操作，不把反向CP0自动对应到前向CP0。</p>
<p>MiB为BF16输入张量字节数，并非链路实际传输量；设备耗时不是纯网络service。表中序号是阶段内提交顺序。</p>
<p><a href="../cp-ep8-structure-r1/#b-semantics">返回 F/B 主图</a> · <a href="events.json">完整事件证据</a> · <a href="manifest.json">输入版本与哈希</a></p>''' + ''.join(body) + '</body></html>'
    (a.out / 'index.html').write_text(page)
    framework = Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/extensions/transformer_engine.py')
    inputs += [dict(path=str(p), sha256=digest(p)) for p in [source, Path(__file__).resolve(), framework]]
    outputs = [dict(path=str(p.resolve()), sha256=digest(p)) for p in sorted(a.out.iterdir())]
    (a.out / 'manifest.json').write_text(json.dumps(dict(version='b-cp-identity-r1', inputs=inputs, outputs=outputs), indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()

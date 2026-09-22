"""Bounded startup inventory: six frozen rank0 traces, no model changes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from audit_b_operator_position_r1 import union

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'results/data-foundation/startup-scale-r1/report.json'
SRC = Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    out = ap.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for ref in json.loads(REF.read_text())['rows']:
        raw = Path(ref['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref['sha256']
        data = json.loads(raw)
        del raw
        es = data['traceEvents']
        f = min((e for e in es if e.get('cat') == 'user_annotation' and e.get('name') == 'forward_step'), key=lambda e: e['ts'])
        step = next(e for e in es if e.get('name', '').startswith('ProfilerStep') and e['ts'] <= f['ts'] < e['ts'] + e.get('dur', 0))
        start = step['ts']
        launch = {e['args']['correlation'] for e in es if e.get('cat') in ['privateuse1_runtime', 'privateuse1_driver'] and 'correlation' in e.get('args', {}) and e['pid'] == f['pid'] and f['ts'] <= e['ts'] and e['ts'] + e.get('dur', 0) <= f['ts'] + f['dur'] + .01}
        gpu = [(i, e) for i, e in enumerate(es) if e.get('cat') in ['kernel', 'gpu_memcpy', 'gpu_memset']]
        owned = [e for _, e in gpu if e.get('args', {}).get('correlation') in launch]
        end = min(e['ts'] for e in owned)
        device = owned[0]['args']['device']
        cpu = [(i, e) for i, e in enumerate(es) if e.get('cat') in ['cpu_op', 'privateuse1_runtime', 'privateuse1_driver', 'user_annotation'] and e.get('dur', 0) > 0 and e.get('ts', -1) < end and e.get('ts', 0) + e.get('dur', 0) > start and not e.get('name', '').startswith('ProfilerStep')]
        local_gpu = [(i, e) for i, e in gpu if e.get('args', {}).get('device') == device and e['ts'] < end and e['ts'] + e['dur'] > start]
        before = [(i, e) for i, e in cpu if e['ts'] < f['ts']]
        barriers = [e for _, e in before if e['name'] == 'c10d::barrier']
        waits = [e for _, e in before if e['name'] == 'record_param_comms' and e.get('args', {}).get('Collective name') == 'wait' and e.get('args', {}).get('Process Group Description') == 'default_pg' and e['ts'] < start + 100000]
        timer = next(e for _, e in before if e['name'] == 'aten::zero_' and e.get('args', {}).get('Input Dims') == [[ref['world'], 24]])
        timer_parent = max((e for _, e in before if e['name'] == 'aten::zeros' and e['ts'] <= timer['ts'] and e['ts'] + e['dur'] >= timer['ts'] + timer['dur']), key=lambda e: e['ts'])
        ag = es[ref['event']]
        zeros = [(i, e) for i, e in before if e['name'] == 'aten::zero_' and any(isinstance(d, list) and len(d) == 1 and isinstance(d[0], int) and d[0] > 1000000 for d in e.get('args', {}).get('Input Dims', []))]
        waits = [e for e in waits if e['ts'] < timer_parent['ts']]
        assert len(barriers) == 2 and len(waits) == 2 and len(zeros) == 2, (ref['world'], ref['iteration'], len(barriers),len(waits),len(zeros))
        bounds = [start, barriers[0]['ts'], max(e['ts'] + e['dur'] for e in waits), timer_parent['ts'], ag['ts'], ag['ts'] + ag['dur'], zeros[0][1]['ts'], end]
        assert bounds == sorted(bounds)
        labels = ['按层统计归约发起与本地处理', '两次全局barrier及前序工作排空', 'barrier后到计时表准备（含未记录主机工作）', '计时表准备与AG发起', '全局计时候选AG设备驻留', 'AG后统计处理与训练调度入口', '梯度缓冲清零发起至首F GPU（含CPU首F准备）']
        parts = []
        for name, a, b in zip(labels, bounds, bounds[1:]):
            def busy(events):
                return union([(max(a, e['ts']), min(b, e['ts'] + e['dur'])) for _, e in events if e['ts'] < b and e['ts'] + e['dur'] > a])
            parts.append(dict(name=name, start_ms=(a-start)/1000, end_ms=(b-start)/1000, elapsed_ms=(b-a)/1000, recorded_cpu_union_ms=busy(cpu), device_union_ms=busy(local_gpu)))
        def event(i, e):
            return dict(index=i, name=e['name'], cat=e.get('cat'), start_ms=(e['ts']-start)/1000, duration_ms=e['dur']/1000, args=e.get('args', {}))
        comm = [event(i, e) for i, e in local_gpu if e.get('args', {}).get('Collective name')]
        clear = []
        for i, e in zeros:
            ext = e['args']['External id']
            linked = [(j, g) for j, g in gpu if g.get('args', {}).get('External id') == ext and g.get('args', {}).get('device') == device]
            clear.append(dict(cpu=event(i, e), device=[event(j, g) for j, g in linked]))
        assert abs(sum(p['elapsed_ms'] for p in parts) - (end-start)/1000) < 1e-6
        names = collections.Counter(e['name'] for _, e in before if e.get('cat') == 'cpu_op')
        assert names['c10d::allreduce_'] == 5 and names['aten::nonzero'] == 24
        row = dict(world=ref['world'], iteration=ref['iteration'], rank=0, path=ref['path'], sha256=ref['sha256'], profiler_label=step['name'], startup_ms=(end-start)/1000, first_F_CPU_ms=(f['ts']-start)/1000, parts=parts, communication=comm, gradient_clear=clear, cpu_counts=names, events=[event(i,e) for i,e in cpu+local_gpu], process_groups=data.get('distributedInfo', {}).get('pg_config'), stack_events=sum(any('stack' in k.lower() for k in e.get('args', {})) for _,e in cpu))
        rows.append(row)
        print(ref['world'], ref['iteration'], 'PASS', [(p['name'], round(p['elapsed_ms'],3)) for p in parts], flush=True)
    report = dict(status='PASS_INVENTORY_PARTIAL_CAUSAL_ATTRIBUTION', scope='32/256 rank0 iter40/60/80; trace window only, no world makespan inference', caveats=['Disjoint parts are temporal windows, not exclusive physical work; CPU/GPU unions overlap.', 'Reference code is not verified installed acquisition code; shape/sequence matches are inferences.', 'Profiler-window attribution is not proof of logical training-iteration ownership.', 'Gradient clear can overlap CPU forward setup; do not add its full duration to the startup window.', 'No target calibration, no parameter-count/world-size fit.'], rows=rows)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    sources = [SRC/'core/timers.py', SRC/'core/transformer/moe/moe_utils.py', SRC/'training/training.py']
    manifest = dict(inputs={str(REF):sha(REF)}, raw=[{k:r[k] for k in ['path','sha256']} for r in rows], code={str(Path(__file__).resolve()):sha(Path(__file__)), str(Path(__file__).with_name('audit_b_operator_position_r1.py')):sha(Path(__file__).with_name('audit_b_operator_position_r1.py'))}, reference_code={str(p):sha(p) for p in sources}, outputs={str(out/'report.json'):sha(out/'report.json')})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__':
    main()

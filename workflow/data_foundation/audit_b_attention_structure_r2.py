"""Refine CP roles using shapes and flash-backward boundaries, no target fitting."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    src = BASE / 'b-cp-identity-r1/events.json'
    prior = BASE / 'b-cost-blocks-r1/evidence.json'
    rows = json.loads(src.read_text())
    result, inputs = [], []
    log = Path('/home/zjb/Desktop/32/gpu32_gbs64_framework/2026-09-18-10:38/worker34031/2026-09-18_1038/tp1_pp4_dp_mbs2_numbs_gbs64_gpus0_mtp1_forcelbfalse_pertensorfalse_NO_LOSS_REDUCE.RANK1.10.124.34.31.log')
    keys = ['num_attention_heads', 'qk_head_dim', 'qk_pos_emb_head_dim', 'v_head_dim', 'context_parallel_size', 'cp_comm_type', 'recompute_num_layers']
    config = {k: sorted(set(re.findall(r'\b' + k + r'=([^,\)]+)', log.read_text()))) for k in keys}
    assert config['num_attention_heads'] == ['128']
    assert config['qk_head_dim'] == ['128'] and config['qk_pos_emb_head_dim'] == ['64']
    assert config['v_head_dim'] == ['128'] and config['context_parallel_size'] == ['2']
    for x in json.loads(prior.read_text()):
        p = Path(x['path']); blob = p.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == x['sha256']
        data = json.loads(blob); es = data['traceEvents']; del blob
        inputs.append(dict(path=str(p), sha256=x['sha256']))
        for unit in range(4):
            rr = [r for r in rows if r['rank'] == x['rank'] and r['checkpoint_unit'] == unit and r['phase'] == 'backward']
            owner = es[rr[0]['cpu_owner_event']]
            def inside(e, parent):
                return e.get('pid') == parent['pid'] and e.get('tid') == parent['tid'] and parent['ts'] <= e.get('ts', 0) and e.get('ts', 0) + e.get('dur', 0) <= parent['ts'] + parent['dur'] + .01
            flashes = [(i, e) for i, e in enumerate(es) if e.get('name') == 'aten::_scaled_dot_product_attention_flash_musa_backward' and inside(e, owner)]
            assert len(flashes) == 1
            fi, flash = flashes[0]
            calls = [e for e in es if e.get('cat') in ['privateuse1_runtime','privateuse1_driver'] and inside(e, flash)]
            correlations = {e.get('args', {}).get('correlation') for e in calls} - {None}
            kernels = [(i,e) for i,e in enumerate(es) if e.get('cat') == 'kernel' and e.get('args', {}).get('correlation') in correlations]
            assert kernels
            lo = min(e['ts'] for _,e in kernels); hi = max(e['ts'] + e['dur'] for _,e in kernels)
            cp = []
            for r in rr:
                launch = es[r['launch_event']]
                markers = [(i,e) for i,e in enumerate(es) if e.get('name') == 'record_param_comms' and inside(launch,e)]
                assert len(markers) == 1
                mi, marker = markers[0]
                shape = marker['args']['Input Dims'][0]
                assert math.prod(shape) * 2 == r['input_bytes']
                ordinal = r['ordinal']
                cpu_side = 'before_attention_backward' if launch['ts'] < flash['ts'] else 'after_attention_backward'
                assert cpu_side == ('before_attention_backward' if ordinal < 2 else 'after_attention_backward')
                d = es[r['device_event']]
                cp.append(dict(ordinal=ordinal, shape=shape, input_mib=r['input_mib'], cpu_side=cpu_side,
                               marker_event=mi, device_event=r['device_event'], stream=d['args']['stream'],
                               start_us=d['ts'], end_us=d['ts']+d['dur'],
                               flash_start_minus_cp_end_ms=(lo-d['ts']-d['dur'])/1000,
                               cp_start_minus_flash_end_ms=(d['ts']-hi)/1000))
            groups = [g for g in data['distributedInfo']['pg_config'] if str(g['pg_name']) == rr[0]['group_name']]
            assert len(groups) == 1 and groups[0]['pg_size'] == 2 and x['rank'] in groups[0]['ranks']
            result.append(dict(rank=x['rank'],unit=unit,cp_group=groups[0]['ranks'],flash_cpu_event=fi,
                               flash_input_shapes=flash['args']['Input Dims'],
                               flash_device_events=[dict(event=i,name=e['name'],start_us=e['ts'],end_us=e['ts']+e['dur'],stream=e['args']['stream']) for i,e in kernels],cp=cp))
        print('PASS shapes and attention boundary rank',x['rank'],flush=True)
    summary = dict(status='PARTIAL_DAG',trace_structure_checks='PASS',units=len(result),config=config,
                   observed='two CP before flash backward; three CP after, in CPU submission order',
                   interpretation='O/dO input redistribution pair; dQ/dK output pair; dV output is a structural inference, not exact tensor lineage proof',
                   exact_order_within_equal_shape_pairs='unresolved',
                   before_device_order_checks=sum(all(c['flash_start_minus_cp_end_ms']>=-.001 for c in u['cp'][:2]) for u in result),
                   after_device_order_checks=sum(all(c['cp_start_minus_flash_end_ms']>=-.001 for c in u['cp'][2:]) for u in result),
                   limits=['Observed device order is not proof of event dependency or a global barrier.', 'No prediction costs changed. No iter70 read.'])
    a.out.mkdir(parents=True)
    for name,value in [('evidence',result),('summary',summary)]:
        (a.out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    framework=Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/megatron-lm-musa-patch/musa_patch/recomupte_variance/multi_latent_attention.py')
    def record(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs += [record(p) for p in [src,prior,log,framework,Path(__file__).resolve()]]
    (a.out/'manifest.json').write_text(json.dumps(dict(version='b-attention-structure-r2',inputs=inputs,outputs=[record(p) for p in a.out.iterdir()]),indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()

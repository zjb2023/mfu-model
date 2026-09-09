"""Pinned derived inputs -> graph audit -> sealed predictions -> development evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tomllib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/w37'), str(ROOT/'workflow/scripts'),
               str(ROOT/'case_224gpu_pp14_cp2_a2a/scripts')]
from smoke_worker import SOURCE, CONTROL, V684, V685, InputGuard, verify, sha, dump
from build_dag_mfu_schedule_factorial import schedule
from build_dag_v67_microbatch_runtime_shape import replay, apply_runtime_shape
from build_dag_v682_stage_aware_pp_gradient import TARGET_COMPONENTS, SOURCE_COMPONENTS
from build_dag_v683_causal_program_order import topology_fingerprint

LOCK = 'f1d560daaa344a5980697b67e71b97140c9bfd38fe12a2c676daf05c53bd8d04'
CASE = 'case_224gpu_pp14_cp2_a2a'
WINDOW = [85, 90, 95, 100]
KEYS = ['rank','pp_stage','pp_lane','phase','microbatch']


def is_target(item):
    return '/evaluator_only/' in item['path'] or 'historical_target' in '|'.join(item['roles'])


class ResearchGuard(InputGuard):
    def event(self, event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            item = self.items.get(str(Path(os.fsdecode(args[0])).resolve()))
            if item and is_target(item) and self.phase not in ['hash_preflight','evaluator']:
                raise PermissionError('target diagnostic/metric read before explicit prediction seal verification')
        super().event(event, args)


def csv(out, name, data):
    path = out/name
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, compression='gzip' if name.endswith('.gz') else None)


def conserved(nodes, cols):
    difference = nodes[list(cols)].fillna(0).sum(axis=1).to_numpy(dtype=np.int64) - nodes.duration_ns.to_numpy(dtype=np.int64)
    assert np.max(np.abs(difference)) == 0
    assert nodes[list(cols)].fillna(0).ge(0).all().all()


def phases(nodes, source=False):
    names = ['forward','backward'] if source else ['FWD','BWD']
    d = nodes[nodes['rank'].ge(0) & nodes.phase.isin(names) & nodes.microbatch.ge(0)]
    result = d.groupby(KEYS, as_index=False).agg(start_ns=('predicted_start_ns','min'),end_ns=('predicted_end_ns','max'))
    result['phase'] = result.phase.replace({'forward':'FWD','backward':'BWD'})
    result['duration_ms'] = (result.end_ns-result.start_ns)/1e6
    return result


def phase_summary(nodes, source=False):
    p = phases(nodes, source)
    return float(p.start_ns.min())/1e6, float(p.end_ns.max())/1e6


def ownership_samples(events):
    lookup = events.set_index(['iteration','pp_stage','pp_lane','phase','microbatch'])
    rows = []
    for r in events[events.phase.eq('backward') & events.pp_stage.lt(15)].itertuples():
        seq = schedule(r.pp_stage,16,4)
        ph, mb = seq[seq.index(('B',r.microbatch))-1]
        previous = lookup.loc[(r.iteration,r.pp_stage,r.pp_lane,'forward' if ph=='F' else 'backward',mb)]
        sender = lookup.loc[(r.iteration,r.pp_stage+1,r.pp_lane,'backward',r.microbatch)]
        ready = max(int(previous.observed_end_ns),int(sender.observed_end_ns))
        rows.append({'iteration':r.iteration,'receiver_stage':r.pp_stage,'pp_lane':r.pp_lane,'microbatch':r.microbatch,
                     'sender_end_ns':int(sender.observed_end_ns),'receiver_previous_end_ns':int(previous.observed_end_ns),
                     'receiver_start_ns':int(r.observed_start_ns),
                     'old_gradient_wall_ns':int(r.observed_start_ns)-int(sender.observed_end_ns),
                     'local_prerequisite_overlap_ns':ready-int(sender.observed_end_ns),
                     'after_both_ready_ns':int(r.observed_start_ns)-ready})
    frame = pd.DataFrame(rows)
    assert (frame.old_gradient_wall_ns == frame.local_prerequisite_overlap_ns+frame.after_both_ready_ns).all()
    assert frame.after_both_ready_ns.gt(0).all()
    return frame


def structure_audit(nodes, edges, out):
    assert topology_fingerprint(edges) == LOCK
    assert len(nodes) == 327746 and len(edges) == 364784 and nodes.node_id.is_unique
    n = nodes.set_index('node_id',drop=False)
    # Every replayed edge is causal, and every start is exactly max(predecessor finishes).
    src_end = edges.src.map(n.predicted_end_ns)
    dst_start = edges.dst.map(n.predicted_start_ns)
    assert src_end.notna().all() and dst_start.notna().all() and dst_start.ge(src_end).all()
    max_end = pd.DataFrame({'dst':edges.dst,'end':src_end}).groupby('dst').end.max()
    assert n.predicted_start_ns.eq(n.node_id.map(max_end).fillna(0)).all()
    starts = nodes[nodes.kind.eq('phase_boundary') & nodes.op_name.isin(['fwd_start','bwd_start'])]
    assert len(starts) == 1344
    incoming = {k:set(g.edge_type) for k,g in edges.groupby('dst',sort=False)}
    rows = []
    for r in starts.itertuples():
        types = incoming[r.node_id]
        assert ('pp_activation_recv' in types) == (r.phase == 'FWD' and r.pp_stage > 0)
        assert ('pp_gradient_recv' in types) == (r.phase == 'BWD' and r.pp_stage < 13)
        assert ('autograd_activation_dependency' in types) == (r.phase == 'BWD')
        seq = schedule(r.pp_stage,14,3)
        assert seq[int(r.operation_sequence)] == (r.phase[0],r.microbatch)
        warm = min(13-r.pp_stage,3); remain = 3-warm
        q = int(r.operation_sequence)
        region = 'warmup' if q < warm else ('steady' if q < warm+2*remain else 'cooldown')
        assert r.schedule_region == ('steady_or_drain' if region == 'steady' else region)
        rows.append({'node_id':r.node_id,'rank':r.rank,'pp_stage':r.pp_stage,'microbatch':r.microbatch,
                     'phase':r.phase,'region':region,'native_region':r.schedule_region,'incoming_edge_types':'|'.join(sorted(types)),'status':'PASS'})
    counts = edges.edge_type.value_counts()
    assert counts['blocking_forward_send'] == 624 and counts['blocking_backward_send'] == 416
    transition = nodes[nodes.kind.isin(['scheduler_handoff','microbatch_release_gap'])]
    assert len(transition) == 1120 and transition.duration_ns.eq(0).sum() == 1072
    assert transition[transition.duration_ns.gt(0)].pp_stage.eq(13).all()
    critical = nodes[nodes.on_critical_path]
    assert int(critical.duration_ns.sum()) == int(n.loc['iteration:completion_join','predicted_end_ns'])
    conserved(nodes,TARGET_COMPONENTS)
    csv(out,'audit/phase_prerequisites.csv',pd.DataFrame(rows))
    csv(out,'audit/edge_semantics_counts.csv',edges.groupby(['edge_type','dependency_source'],dropna=False).size().reset_index(name='edge_count'))
    # The full native schema retains source_parameter_key, code_anchor, resources and model components.
    csv(out,'graph/nodes.csv.gz',nodes)
    csv(out,'graph/edges.csv.gz',edges)
    csv(out,'graph/critical_path.csv.gz',critical)
    csv(out,'graph/phase_nodes.csv',phases(nodes))
    csv(out,'audit/critical_component_ledger.csv',critical.groupby('kind')[list(TARGET_COMPONENTS)+['duration_ns']].sum().reset_index())
    return {'status':'PASS','nodes':len(nodes),'edges':len(edges),'topology_sha256':LOCK,
            'phase_start_contracts':len(starts),'all_edges_causal':True,'all_max_plus_starts_exact':True,
            'zero_transition_nodes':1072,'terminal_launch_nodes':48,'blocking_send_edges':1040,
            'node_component_conservation':True,'critical_path_conservation':True,
            'authority':'local schedule formula + pinned training schedules.py / p2p_communication.py; matching deployed revision not established'}


def seal(out, paths, extra):
    files = [{'path':str(x.resolve()),'sha256':sha(x),'size_bytes':x.stat().st_size} for x in paths]
    dump(out/'prediction_seal.json',{'schema':'w37-1f1b-prediction-seal-v1','artifacts':files,**extra})
    for i in files:
        assert sha(i['path']) == i['sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    assert out.is_relative_to(ROOT/'results/w37/A')
    assert os.statvfs(SOURCE).f_flag & os.ST_RDONLY
    assert os.statvfs(ROOT).f_flag & os.ST_RDONLY
    assert not os.statvfs(out).f_flag & os.ST_RDONLY
    pins=json.loads((CONTROL/'external_inputs.json').read_text())['files']+json.loads((ROOT/'docs/w37/1f1b/additional_inputs.json').read_text())['files']
    cfg=tomllib.loads((ROOT/CASE/'config/dag_v685_source_steady_calibration_2026w36.toml').read_text())
    required={str(SOURCE/V685/x) for x in ['predictions/dag_v685_nodes.csv.gz','predictions/dag_v685_edges.csv.gz',
        'prediction_contract.json','source_replay/dag_v685_source_nodes.csv.gz']}
    required.update(str(SOURCE/cfg['inputs'][k]) for k in ['source_edges','source_pp_trace_events','source_profiler_boundaries','target_ground_truth'])
    required.update(i['path'] for i in pins if 'training_code_semantics' in i['roles'] or 'historical_target_diagnostic_after_seal' in i['roles'])
    items=[i for i in pins if i['path'] in required]
    assert set(i['path'] for i in items)==required
    # Preserve the original code snapshot, including all imported legacy modules.
    for i in json.loads((CONTROL/'code_manifest.json').read_text())['files']:
        assert sha(ROOT/i['path'])==i['sha256'],i['path']
    guard=ResearchGuard('A',out,items,'research');sys.addaudithook(guard.event)
    guard.phase='hash_preflight';verify(items)
    guard.phase='model'
    probes=[(SOURCE/'__forbidden_write__','w',os.O_WRONLY),(SOURCE/'new0729/__raw_trace__','r',os.O_RDONLY),
            (SOURCE/cfg['inputs']['target_ground_truth'],'r',os.O_RDONLY)]
    for path,mode,flags in probes:
        try:guard.event('open',(str(path),mode,flags))
        except PermissionError:pass
        else:raise AssertionError('boundary probe failed')
    nodes=pd.read_csv(SOURCE/V685/'predictions/dag_v685_nodes.csv.gz',low_memory=False)
    edges=pd.read_csv(SOURCE/V685/'predictions/dag_v685_edges.csv.gz',low_memory=False)
    source=pd.read_csv(SOURCE/V685/'source_replay/dag_v685_source_nodes.csv.gz',low_memory=False)
    source_edges=pd.read_csv(SOURCE/cfg['inputs']['source_edges'],low_memory=False)
    contract=json.loads((SOURCE/V685/'prediction_contract.json').read_text())['prediction']
    events=pd.read_csv(SOURCE/cfg['inputs']['source_pp_trace_events'])
    boundaries=pd.read_csv(SOURCE/cfg['inputs']['source_profiler_boundaries'])
    audit=structure_audit(nodes,edges,out);conserved(source,SOURCE_COMPONENTS)
    csv(out,'audit/pp_wait_ownership_samples.csv',ownership_samples(events))
    csv(out,'audit/source_phase_nodes.csv',phases(source,True))
    dump(out/'audit/structure_audit.json',audit)
    records=[]
    f,b=phase_summary(nodes);sf,sb=phase_summary(source,True)
    records.append({'variant':'baseline','entry_ms':f,'onef1b_ms':b-f,'raw_graph_ms':contract['target_raw_graph_ms'],
        'reconciliation_ms':contract['target_reconciliation_ms'],'tail_ms':contract['profiler_step_ms']-b,
        'profiler_ms':contract['profiler_step_ms'],'outer_ms':contract['outer_framework_ms'],'training_ms':contract['training_step_ms'],
        'source_raw_ms':contract['source_raw_graph_ms'],'source_phase_start_ms':sf,'source_phase_end_ms':sb})
    if a.mode=='experiment':
        from experiments import run_experiments
        records += run_experiments(out,nodes,edges,source,source_edges,events,boundaries,contract)
    csv(out,'predictions.csv',pd.DataFrame(records))
    seal(out,[out/'predictions.csv',out/'graph/nodes.csv.gz',out/'graph/edges.csv.gz',out/'audit/structure_audit.json']+
         sorted((out/'parameters').glob('*'))+sorted((out/'candidates').glob('**/*.gz')),
         {'target_used_for_fit':False,'formal_blind_claim_allowed':False,'baseline_topology_sha256':LOCK})
    guard.phase='evaluator'
    payload_path=next(Path(i['path']) for i in items if i['path'].endswith('dag_v682_stage_aware_pp_gradient_payload.json'))
    payload=json.loads(payload_path.read_text())
    historical=json.loads(next(Path(i['path']) for i in items if i['path'].endswith('three_stage_v54_visualization_payload.json')).read_text())
    truth=pd.read_csv(SOURCE/cfg['inputs']['target_ground_truth'])
    from evaluate import evaluate
    metrics=evaluate(out,records,truth,payload,historical,source,events,boundaries,contract)
    from figures import figures
    figures(out,nodes,edges,payload)
    dump(out/'input_access_audit.json',{'phase_order':['hash_preflight','model','prediction_seal','evaluator'],
         'hash_preflight_is_not_calibration':True,'raw_trace_scanned':False,'original_and_code_mount_readonly':True,
         'boundary_probes_passed':len(probes),'denied':guard.denied,
         'reads':[{'path':k,'roles':v['roles'],'phases':sorted(v['phases'])} for k,v in sorted(guard.reads.items())]})
    print(json.dumps({'structure':audit,'metrics':metrics},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

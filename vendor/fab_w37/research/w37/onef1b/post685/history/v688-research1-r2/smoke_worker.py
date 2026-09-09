#!/usr/bin/env python3
"""Guarded worker: hashes, DAG replay or source16 admission, never raw trace."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/zjb/Desktop/fabric-data-analysis')
CONTROL=ROOT/'docs/w37/coordination'
CASE='case_224gpu_pp14_cp2_a2a'
RESULT=f'{CASE}/results/mfu_accuracy_comparison_2026w36'
V684=f'{RESULT}/dag_v684_blocking_pp_program_order_256_to_224'
V685=f'{RESULT}/dag_v685_source_steady_calibration_256_to_224'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

class InputGuard:
    def __init__(self, task, output, items, mode):
        self.task,self.output,self.mode=task,output,mode
        self.items={i['path']:i for i in items if task in i['allowed_tasks']}
        self.reads={}
        self.phase='control_preflight'
        self.denied=[]
        self.import_roots=[Path(p).resolve() for p in sys.path if p and ('site-packages' in p or p.startswith('/usr/'))]
        self.import_roots += [Path('/usr'),Path('/lib'),Path('/etc'),SOURCE/'.venv',Path(sys.base_prefix)]
    def event(self,event,args):
        if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn','socket.connect'):
            raise PermissionError(f'child process/network disabled in baseline worker: {event}')
        if event!='open':return
        filename,mode,flags=args
        if not isinstance(filename,(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(filename)).resolve()
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing:
            if not path.is_relative_to(self.output):
                self.denied.append({'operation':'write','path':str(path)})
                raise PermissionError(f'write outside current result directory: {path}')
            return
        if path.is_relative_to(self.output):return
        item=self.items.get(str(path))
        if item:
            if self.mode=='reproduce-v685' and 'evaluator224_after_seal' in item['roles'] and self.phase!='evaluator':
                seal=self.output/'reproduced/predictions/prediction_seal.json'
                if not seal.is_file():raise PermissionError('target evaluation before prediction seal')
                self.phase='evaluator'
            self.reads.setdefault(str(path),{'roles':item['roles'],'phases':set()})['phases'].add(self.phase)
            return
        if path.is_relative_to(ROOT):
            if self.task=='A':return
            allowed=[ROOT/'scripts/w37',ROOT/'workflow/scripts/build_dag_mfu_schedule_factorial.py',
                     ROOT/'case_16gpu_pp2/config',CONTROL/'source16_inventory.json',CONTROL/'code_manifest.json']
            if any(path==p or path.is_relative_to(p) for p in allowed):return
        if any(path.is_relative_to(p) for p in self.import_roots):return
        if str(path) in ['/dev/null','/dev/urandom','/proc/cpuinfo','/proc/meminfo']:return
        self.denied.append({'operation':'read','path':str(path)})
        raise PermissionError(f'input is not allowed for task {self.task}: {path}')

def verify(items):
    for i in items:
        p=Path(i['path'])
        if p.stat().st_size!=i['size_bytes'] or sha(p)!=i['sha256']:
            raise ValueError(f'frozen input changed: {p}')

def generic_smoke():
    sys.path.insert(0,str(ROOT/'workflow/scripts'))
    from build_dag_mfu_schedule_factorial import build_graph,replay
    cases=[]
    for pp,mb in [(2,4),(14,3),(16,4)]:
        nodes,edges=build_graph(f'unit_pp{pp}_mb{mb}',pp,mb)
        result,critical=replay(nodes,edges)
        assert len(nodes)==2*pp*mb
        by_id=result.set_index('node_id')
        for e in edges.itertuples():
            assert by_id.loc[e.dst,'start_unit']>=by_id.loc[e.src,'end_unit']
        assert result.end_unit.max()==2*(pp+mb-1)
        for mb_index in range(mb):
            for stage in range(pp-1):
                assert ((edges.src==f's{stage+1}:B{mb_index}') & (edges.dst==f's{stage}:B{mb_index}')).any()
        cases.append({'pp':pp,'microbatches':mb,'nodes':len(nodes),'edges':len(edges),
                      'unit_cost_makespan':int(result.end_unit.max()),'critical_nodes':len(critical)})
    return cases

def replay_v685(output):
    import pandas as pd
    sys.path.insert(0,str(ROOT/CASE/'scripts'))
    from build_dag_v67_microbatch_runtime_shape import replay
    from build_dag_v683_causal_program_order import topology_fingerprint
    nodes=pd.read_csv(SOURCE/V685/'predictions/dag_v685_nodes.csv.gz',low_memory=False)
    edges=pd.read_csv(SOURCE/V685/'predictions/dag_v685_edges.csv.gz',low_memory=False)
    lock=json.loads((SOURCE/V684/'predictions/dependency_topology_lock.json').read_text())
    topology=topology_fingerprint(edges)
    assert topology==lock['child_topology_sha256']
    sealed=json.loads((SOURCE/V685/'prediction_contract.json').read_text())['prediction']
    predicted,critical=replay(nodes,edges,'iteration:completion_join')
    raw=float(predicted.loc[predicted.node_id.eq('iteration:completion_join'),'predicted_end_ns'].item())/1e6
    assert math.isclose(raw,sealed['target_raw_graph_ms'],abs_tol=1e-6)
    profiler=raw+sealed['target_reconciliation_ms']
    assert math.isclose(profiler,sealed['profiler_step_ms'],abs_tol=1e-6)
    critical.to_csv(output/'critical_path.csv',index=False)
    return {'nodes':len(nodes),'edges':len(edges),'topology_sha256':topology,
            'predicted_raw_graph_ms':raw,'predicted_profiler_step_ms':profiler,
            'predicted_training_step_ms':profiler+sealed['outer_framework_ms'],
            'meaning':'frozen graph replay reproducibility, not a new accuracy experiment'}

def reproduce(output,guard):
    sys.path.insert(0,str(ROOT/CASE/'scripts'))
    config=ROOT/CASE/'config/dag_v685_source_steady_calibration_2026w36.toml'
    content=config.read_text()
    original=tomllib.loads(content)
    for value in original['inputs'].values():
        content=content.replace(json.dumps(value),json.dumps(str(SOURCE/value)))
    content=content.replace(json.dumps(original['outputs']['output_dir']),json.dumps(str(output/'reproduced')))
    local_config=output/'v685.local.toml'
    local_config.write_text(content)
    import build_dag_v685_source_steady_calibration as builder
    guard.phase='model'
    sys.argv=[str(Path(builder.__file__)), '--config',str(local_config)]
    assert builder.REPO==ROOT
    assert builder.main()==0
    result=json.loads((output/'reproduced/evaluator_only/metrics.json').read_text())
    assert math.isclose(result['metrics']['profiler_mape_pct'],10.36952849443861,abs_tol=1e-8)
    assert math.isclose(result['prediction']['profiler_step_ms'],21385.696422125,abs_tol=1e-6)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['A','B'],required=True)
    p.add_argument('--mode',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args();output=args.output.resolve()
    if ROOT==SOURCE or not output.is_relative_to(ROOT/'results/w37'):raise ValueError('unsafe output')
    if not (os.statvfs(SOURCE).f_flag & os.ST_RDONLY):
        raise PermissionError('original repository must be mounted read-only')
    if os.statvfs(output).f_flag & os.ST_RDONLY:
        raise PermissionError('worktree result directory is not writable')
    items=json.loads((CONTROL/'external_inputs.json').read_text())['files']
    code=json.loads((CONTROL/'code_manifest.json').read_text())
    # B verifies only generic code and source16 data; 256 evidence stays unopened.
    generic='workflow/scripts/build_dag_mfu_schedule_factorial.py'
    code_items=code['files'] if args.task=='A' else [i for i in code['files'] if i['path']==generic]
    for i in code_items:
        if sha(ROOT/i['path'])!=i['sha256']:raise ValueError(f'code snapshot changed: {i["path"]}')
    if args.mode=='reproduce-v685':
        config=tomllib.loads((ROOT/CASE/'config/dag_v685_source_steady_calibration_2026w36.toml').read_text())
        required={str(SOURCE/v) for k,v in config['inputs'].items() if k!='v684_run_dir'}
        parent_seal=SOURCE/V684/'predictions/prediction_seal.json'
        required.add(str(parent_seal))
        required.update(i['path'] for i in json.loads(parent_seal.read_text())['artifacts'])
        required.add(str(SOURCE/V684/'evaluator_only/iteration_evaluation.csv'))
        items=[i for i in items if i['path'] in required]
    guard=InputGuard(args.task,output,items,args.mode)
    sys.addaudithook(guard.event)
    allowed=[i for i in items if args.task in i['allowed_tasks']]
    if args.mode=='reproduce-v685':
        allowed=[i for i in allowed if 'evaluator224_after_seal' not in i['roles']]
    verify(allowed)
    guard.phase='smoke'
    checks=generic_smoke()
    # Deliberate probes fail at policy check without touching original files.
    for path,mode,flags in [(SOURCE/'README.md','w',os.O_WRONLY),
                            (SOURCE/'new0729/__raw_trace_scan_forbidden__','r',os.O_RDONLY)]:
        try:guard.event('open',(str(path),mode,flags))
        except PermissionError:pass
        else:raise AssertionError('boundary guard accepted forbidden access')
    result={'status':'PASS','task':args.task,'mode':args.mode,'commit':args.commit,
            'generic_schedule_checks':checks,'external_files_verified':len(allowed),
            'environment':{'python':sys.version,'pandas':importlib.metadata.version('pandas'),
                           'numpy':importlib.metadata.version('numpy')},
            'raw_trace_scanned':False,'write_scope':str(output),'source256_parameter_access_allowed':args.task=='A',
            'original_mount_readonly_verified':True}
    if args.task=='A':
        result['baseline']=reproduce(output,guard) if args.mode=='reproduce-v685' else replay_v685(output)
        if args.mode=='reproduce-v685':
            verify([i for i in items if 'evaluator224_after_seal' in i['roles']])
    else:
        inventory=[]
        for i in allowed:
            if i['path'].endswith('per_iteration_metrics.json'):
                rows=json.loads(Path(i['path']).read_text())
                assert len(rows)==20 and {int(r['iter']) for r in rows}==set(range(5,101,5))
                inventory.append({'path':i['path'],'sample_count':len(rows),'iterations':list(range(5,101,5))})
        assert len(inventory)==2
        target=SOURCE/V685/'predictions/dag_v685_nodes.csv.gz'
        try:guard.event('open',(str(target),'r',os.O_RDONLY))
        except PermissionError:pass
        else:raise AssertionError('B accepted source256-fitted nodes')
        result['source16_inventory']=inventory
        result['research_status']='PARTIAL: source16 metadata exists; model/layer/MB/FLOPs comparability requires task B design'
    audit={'task':args.task,'mode':args.mode,'reads':[{'path':p,'roles':v['roles'],'phases':sorted(v['phases'])} for p,v in sorted(guard.reads.items())],
           'intentional_denial_probes':guard.denied,'raw_trace_scanned':False,
           'original_write_policy':'bubblewrap read-only mount and Python write guard',
           'note':'control_preflight hashes are not calibration reads; legacy files include source60-100 even when fit rows are85-100'}
    dump(output/'input_access_audit.json',audit)
    dump(output/'smoke_result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

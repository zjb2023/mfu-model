"""Audit → source fit/validation → candidate seal → target development evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/w37'))
from smoke_worker import InputGuard,verify,sha,dump,SOURCE,CONTROL
from pp_semantics import align_observations,all_actions,message_pairs


def csv(out,name,d):
    p=out/name;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False,compression='gzip' if name.endswith('.gz') else None)


class Guard(InputGuard):
    def event(self,event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            item=self.items.get(str(Path(os.fsdecode(args[0])).resolve()))
            if item and ('evaluator' in '|'.join(item['roles']) or '/evaluator_only/' in item['path']):
                if self.phase not in ['hash_preflight','evaluator']:raise PermissionError('target timing is evaluator-only after explicit candidate seal')
        super().event(event,args)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    assert out.is_relative_to(ROOT/'results/w37/A')
    assert os.statvfs(SOURCE).f_flag&os.ST_RDONLY and os.statvfs(ROOT).f_flag&os.ST_RDONLY
    assert not os.statvfs(out).f_flag&os.ST_RDONLY
    old=json.loads((CONTROL/'external_inputs.json').read_text())['files']
    new=json.loads((ROOT/'docs/w37/1f1b/post685/inputs.json').read_text())['files']
    first=json.loads((ROOT/'docs/w37/1f1b/additional_inputs.json').read_text())['files']
    names=['source_pp_trace_events_60_100.csv','source_pp_api_events_60_100.csv','source256_phase_noncommunication_windows.csv.gz',
           'source256_phase_kernel_family_windows.csv.gz','source256_window_partition.csv.gz','target224_window_partition_ground_truth.csv.gz',
           'source256_profiler_entry_60_100.csv','schedules.py','p2p_communication.py']
    items=[i for i in old+new+first if Path(i['path']).name in names]
    if a.mode=='candidate':
        names2=['dag_v685_nodes.csv.gz','dag_v685_edges.csv.gz','iteration_ground_truth.csv','dag_v682_stage_aware_pp_gradient_payload.json','target224_outside_wrapper_ground_truth.csv.gz']
        items += [i for i in old+new+first if Path(i['path']).name in names2]
        items += [i for i in old if Path(i['path']).name=='prediction_contract.json' and 'dag_v685_' in i['path']]
    assert len(items)==len({i['path'] for i in items})
    guard=Guard('A',out,items,'post685');sys.addaudithook(guard.event);guard.phase='hash_preflight';verify(items)
    for i in json.loads((CONTROL/'code_manifest.json').read_text())['files']:assert sha(ROOT/i['path'])==i['sha256'],i['path']
    paths={Path(i['path']).name:Path(i['path']) for i in items};guard.phase='source_diagnostic'
    denied=0
    for path,mode,flags in [(SOURCE/'new0729/__raw_forbidden__','r',os.O_RDONLY),(SOURCE/'__write_forbidden__','w',os.O_WRONLY),
                            (paths['target224_window_partition_ground_truth.csv.gz'],'r',os.O_RDONLY)]:
        try:guard.event('open',(str(path),mode,flags))
        except PermissionError:denied+=1
        else:raise AssertionError('boundary probe allowed forbidden access')
    phase=pd.read_csv(paths['source_pp_trace_events_60_100.csv']);api=pd.read_csv(paths['source_pp_api_events_60_100.csv'])
    aligned,pairs=align_observations(phase,api)
    csv(out,'audit/source_actions.csv.gz',aligned);csv(out,'audit/source_message_pairs.csv.gz',pairs)
    gpu=pd.read_csv(paths['source256_phase_noncommunication_windows.csv.gz']);wrapper=pd.read_csv(paths['source256_window_partition.csv.gz'])
    components=['phase_noncommunication_only_ms','phase_communication_only_ms','phase_comm_noncommunication_overlap_ms','phase_gpu_idle_ms']
    assert (gpu[components].sum(axis=1)-gpu.phase_duration_ms).abs().max()<1e-6
    assert (wrapper.fused_prelaunch_sum_ms+wrapper.fused_postlaunch_sum_ms+wrapper.outside_fused_wrapper_ms-wrapper.phase_duration_ms).abs().max()<1e-6
    cpu=phase[phase.pp_lane.eq(0)].copy();cpu['phase_duration_ms']=cpu.duration_ns/1e6
    check=cpu.merge(gpu,on=['iteration','rank','pp_stage','pp_lane','phase','microbatch'],suffixes=('_cpu','_gpu'),validate='one_to_one')
    assert (check.phase_duration_ms_cpu-check.phase_duration_ms_gpu).abs().max()<.001
    csv(out,'audit/source_gpu_partition.csv',gpu)
    csv(out,'audit/source_wrapper_partition.csv',wrapper)
    csv(out,'audit/source_gpu_partition_summary.csv',gpu.groupby(['iteration','phase'])[components+['phase_duration_ms']].mean().reset_index())
    simple=pairs[pairs.both_endpoints_single_message]
    csv(out,'audit/source_pp_publication_summary.csv',pairs.groupby(['iteration','direction','both_endpoints_single_message']).agg(
        samples=('message_id','size'),post_publication_median_ms=('post_publication_upper_bound_ns',lambda x:x.median()/1e6),
        sender_wait_mean_ms=('sender_wait_for_receiver_ns',lambda x:x.mean()/1e6),receiver_wait_mean_ms=('receiver_wait_for_sender_ns',lambda x:x.mean()/1e6)).reset_index())
    summary={'status':'PASS_SOURCE_STATIC_API_ORDER_AND_DISJOINT_PARTITIONS','source_actions':len(aligned),'source_api_rows':len(api),
        'source_phase_rows':len(phase),'source_message_pairs':len(pairs),'single_message_pair_samples':len(simple),
        'minimum_post_publication_ns':int(pairs.post_publication_upper_bound_ns.min()),
        'gpu_partition_rows':len(gpu),'gpu_scope':'lane0 only; four disjoint GPU categories, not additive with wrapper partition',
        'pp_completion_scope':'post-publication API completion upper bound, not pure NIC service',
        'first_round_formal_topology_unchanged':True,'target_timing_read':False}
    dump(out/'audit/summary.json',summary)
    if a.mode=='candidate':
        from candidate import run_candidate
        summary['candidate']=run_candidate(out,aligned,pairs,phase,gpu,wrapper,paths,guard)
    dump(out/'input_access_audit.json',{'raw_trace_scanned':False,'denial_probes_passed':denied,'original_and_code_readonly':True,
        'reads':[{'path':k,'roles':v['roles'],'phases':sorted(v['phases'])} for k,v in sorted(guard.reads.items())],
        'note':'hash_preflight byte access is not calibration; source tables contain 60–100; fitting masks declared separately'})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

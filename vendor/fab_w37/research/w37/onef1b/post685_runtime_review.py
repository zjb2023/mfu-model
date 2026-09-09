#!/usr/bin/env python3
"""Bounded read-only posthoc runtime visibility review; never updates a predictor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/w37'))
from smoke_worker import InputGuard,SOURCE,verify,sha,dump


def worker(out,prior):
    import numpy as np
    import pandas as pd
    assert os.statvfs(SOURCE).f_flag&os.ST_RDONLY
    names=['source256_outside_wrapper_windows.csv.gz','target224_outside_wrapper_ground_truth.csv.gz']
    items=[i for i in json.loads((ROOT/'docs/w37/1f1b/post685/inputs.json').read_text())['files'] if Path(i['path']).name in names]
    guard=InputGuard('A',out,items,'diagnostic');sys.addaudithook(guard.event);guard.phase='hash_preflight';verify(items)
    for f in json.loads((prior/'prediction_seal.json').read_text())['files']:assert sha(prior/f['path'])==f['sha256']
    guard.phase='posthoc_diagnostic_after_existing_prediction_seal';paths={Path(i['path']).name:Path(i['path']) for i in items}
    source=pd.read_csv(paths[names[0]]);target=pd.read_csv(paths[names[1]])
    cols=['outside_idle_device_synchronize_ms','outside_idle_stream_synchronize_ms','outside_idle_event_synchronize_ms',
          'outside_idle_other_long_runtime_call_ms','outside_idle_no_visible_runtime_wait_ms']
    for d in [source,target]:
        assert (d[cols].sum(axis=1)-d.outside_gpu_idle_ms).abs().max()<1e-6
        d['inside_wrapper_gpu_idle_ms']=d.phase_gpu_idle_ms-d.outside_gpu_idle_ms
        assert d.inside_wrapper_gpu_idle_ms.min()>-1e-6
    cols+=['inside_wrapper_gpu_idle_ms']
    base=pd.read_csv(prior/'prediction/internal_partition.csv')
    base=base[base.variant.eq('v687_split_full')&base.case.eq('target224')&base.component.eq('phase_gpu_idle_ms')]
    key=['pp_stage','pp_lane','phase','microbatch'];med=source[source.iteration.isin([85,90])].groupby(key)[cols].median()
    smap=[0,1,2,3,4,5,6,7,8,9,10,11,12,15];rows=[]
    for r in base.itertuples():
        m=3 if r.microbatch==2 else r.microbatch;v=med.loc[(smap[r.pp_stage],0,r.phase,m)];total=v.sum()
        for col in cols:rows.append(dict(pp_stage=r.pp_stage,pp_lane=0,phase=r.phase,microbatch=r.microbatch,component=col,source_prior_ms=float(v[col]/total*r.predicted_ms)))
    long=target.melt(id_vars=['iteration']+key,value_vars=cols,var_name='component',value_name='observed_ms')
    joined=long.merge(pd.DataFrame(rows),on=key+['component'],validate='many_to_one');joined['delta_ms']=joined.observed_ms-joined.source_prior_ms
    joined.to_csv(out/'runtime_visibility_rows.csv.gz',index=False,compression='gzip')
    summary=joined[joined.iteration.isin([85,90,95,100])].groupby(['phase','component']).agg(observed_mean_ms=('observed_ms','mean'),
        source_prior_mean_ms=('source_prior_ms','mean'),delta_mean_ms=('delta_ms','mean')).reset_index();summary.to_csv(out/'runtime_visibility_summary.csv',index=False)
    dump(out/'contract.json',{'status':'POSTHOC_ONLY_NOT_MODEL_FIT','source_artifact_status':'DIAGNOSTIC_VIEW_NOT_APPROVED_FOR_MODEL_FIT',
        'source_extraction_generation_had_target_access':True,'source_artifact_target_rows':0,'new_predictor_costs_changed':False,
        'meaning':'Mutually exclusive runtime-call visibility inside outside-wrapper GPU idle plus inside-wrapper idle. No visible runtime wait is not proof of host overhead or an optimizable idle interval.',
        'critical_path_attribution':'not claimed for these six local subcategories; use separately sealed four-category lane0 Shapley ledger',
        'source_prior':'diagnostic normalization to the already sealed v687 phase GPU-idle source prior; not a new prediction'})
    dump(out/'input_access_audit.json',{'raw_trace_scanned':False,'reads':[dict(path=k,roles=v['roles'],phases=sorted(v['phases'])) for k,v in guard.reads.items()]})
    print(summary.round(3).to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/w37/A/post685-runtime-review-r1');p.add_argument('--worker',action='store_true')
    p.add_argument('--input-run',type=Path,default=ROOT/'results/w37/A/post685-candidate-scenarios-r1');a=p.parse_args();out=a.output.resolve();prior=a.input_run.resolve()
    assert out.is_relative_to(ROOT/'results/w37/A')
    assert prior.is_relative_to(ROOT/'results/w37/A')
    if a.worker:return worker(out,prior)
    out.mkdir(parents=True,exist_ok=False)
    cmd=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--bind',str(out),str(out),'--tmpfs','/tmp','--unshare-net','--chdir',str(ROOT),'--',
         str(SOURCE/'.venv/bin/python'),'-B',str(Path(__file__).resolve()),'--worker','--output',str(out),'--input-run',str(prior)]
    dump(out/'command.json',{'argv':cmd,'code_sha256':sha(Path(__file__)),'resource_policy':'2 existing derived partitions, <200 KiB compressed, single CPU thread, no raw traces'})
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (out/'execution.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    dump(out/'run_manifest.json',{'exit_code':r.returncode,'artifacts':[dict(path=str(p.relative_to(out)),sha256=sha(p)) for p in out.iterdir() if p.is_file()]})
    print((out/'execution.log').read_text());return r.returncode


if __name__=='__main__':raise SystemExit(main())

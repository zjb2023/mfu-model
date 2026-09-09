#!/usr/bin/env python3
"""Verify a completed experiment and produce a bounded, guarded delivery review."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/w37'))
from run_baseline import PYTHON, SOURCE, sha


def worker(input_run,out):
    import tomllib
    import numpy as np
    import pandas as pd
    from smoke_worker import InputGuard
    assert os.statvfs(SOURCE).f_flag & os.ST_RDONLY
    assert os.statvfs(input_run).f_flag & os.ST_RDONLY
    assert not os.statvfs(out).f_flag & os.ST_RDONLY
    guard=InputGuard('A',out,[],'review');sys.addaudithook(guard.event)
    manifest=json.loads((input_run/'run_manifest.json').read_text())
    assert manifest['exit_code']==0
    for i in manifest['artifacts']:
        assert Path(i['path']).resolve().is_relative_to(input_run)
        assert sha(i['path'])==i['sha256'],i['path']
    command=json.loads((input_run/'command.json').read_text())
    for i in command['code']:assert sha(i['path'])==i['sha256'],i['path']
    seal=json.loads((input_run/'prediction_seal.json').read_text())
    for i in seal['artifacts']:assert sha(i['path'])==i['sha256'],i['path']
    nodes=pd.read_csv(input_run/'graph/nodes.csv.gz',low_memory=False)
    edges=pd.read_csv(input_run/'graph/edges.csv.gz',low_memory=False)
    overlap=nodes[nodes.kind.eq('compute_overlap_token')]
    assert len(overlap)==32352
    ins=edges[edges.edge_type.eq('compute_overlap_branch')]
    outs=edges[edges.edge_type.eq('compute_overlap_branch_join')]
    assert ins.dst.is_unique and outs.src.is_unique
    assert set(ins.dst)==set(overlap.node_id)==set(outs.src)
    # A timeline union, not a sum across parallel branches, closes each F/B envelope.
    rows=[]
    d=nodes[nodes['rank'].ge(0)&nodes.phase.isin(['FWD','BWD'])&nodes.microbatch.ge(0)]
    for key,g in d.groupby(['rank','pp_stage','pp_lane','phase','microbatch']):
        p=g[g.duration_ns.gt(0)].sort_values('predicted_start_ns')
        union=0;start=end=None
        for a,b in zip(p.predicted_start_ns,p.predicted_end_ns):
            a=int(a);b=int(b)
            if end is None:start,end=a,b
            elif a<=end:end=max(end,b)
            else:union+=end-start;start,end=a,b
        if end is not None:union+=end-start
        env=int(g.predicted_end_ns.max()-g.predicted_start_ns.min());total=int(p.duration_ns.sum())
        assert 0<=union<=env and union<=total
        rows.append(dict(zip(['rank','pp_stage','pp_lane','phase','microbatch'],key),phase_envelope_ns=env,
            positive_token_sum_ns=total,positive_token_interval_union_ns=union,
            parallel_token_overlap_ns=total-union,gap_without_positive_token_ns=env-union,
            zero_operator_nodes=int((g.kind.eq('operator')&g.duration_ns.eq(0)).sum()),
            meaning='model token coverage; includes mixed compute/communication/runtime, not observed GPU occupancy'))
    pd.DataFrame(rows).to_csv(out/'per_rank_phase_coverage.csv',index=False)
    metrics=pd.read_csv(input_run/'evaluator_only/metrics.csv')
    pred=pd.read_csv(input_run/'predictions.csv').set_index('variant')
    scores=pd.read_csv(input_run/'parameters/shape_split_source_incremental_scores.csv')
    score=scores[scores.iteration.isin([95,100])].groupby('variant').shape_share_mae_pp.mean().to_dict()
    source=pd.read_csv(input_run/'evaluator_only/source_iteration_ledger.csv')
    sg=source[source.iteration.isin([85,90,95,100])].groupby('variant',as_index=False).agg(
        onef1b_mape_pct=('onef1b_ape_pct','mean'),onef1b_bias_ms=('onef1b_error_ms','mean'),tail_mape_pct=('tail_ape_pct','mean'),
        signed_unlocated_after_entry_ms=('signed_unlocated_after_entry_ms','mean'))
    sg.to_csv(out/'source_regression_metrics.csv',index=False)
    delta=pred.subtract(pred.loc['baseline'])
    delta.to_csv(out/'prediction_deltas_ms.csv')
    decision={'status':'COMPLETE_NEGATIVE_PRIMARY_RESULT_BASELINE_RETAINED','promoted_variant':None,
        'incremental_validation_shape_share_mae_pp':score,
        'primary_candidate_target_profiler_mape_delta_pp':float(metrics[(metrics.variant=='shape_split')&(metrics.split=='development_primary')].profiler_mape_pct.iloc[0]-metrics[(metrics.variant=='baseline')&(metrics.split=='development_primary')].profiler_mape_pct.iloc[0]),
        'primary_candidate_delta_ms':delta.loc['shape_split'].to_dict(),
        'same_window_profiler_delta_ms':float(delta.loc['shape_same_window','profiler_ms']),
        'interpretation':'source85/90 correction overfits incremental phase shape; full-step regression with matched reconciliation. Fixed residual is an ablation, same-window gain is 3.319ms and has no independent holdout.',
        'all_historical_input_hashes_verified':True,'all_candidate_seal_hashes_verified':True,
        'overlap_fork_join_pairs_checked':32352,'generated_coverage_rows':len(rows)}
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    cfgpath=ROOT/'case_224gpu_pp14_cp2_a2a/config/dag_v60_operator_ir_2026w36.toml'
    baseline=json.loads((ROOT/'docs/w37/coordination/code_manifest.json').read_text())
    expected=next(i['sha256'] for i in baseline['files'] if i['path']==str(cfgpath.relative_to(ROOT)))
    assert sha(cfgpath)==expected
    cfg=tomllib.loads(cfgpath.read_text())
    placement=nodes[nodes.layer_id.ge(0)].groupby('pp_stage').layer_id.unique().map(lambda x:sorted(map(int,x))).to_dict()
    scenario={'case_id':cfg['target']['case_id'],'world_size':224,'pp':14,'dp':8,'cp':2,'ep':8,'pp_lanes':16,'microbatches':3,
        'layers':52,'layer_placement_from_frozen_graph':placement,'sequence_length':8192,'micro_batch_size':2,
        'global_batch_size':48,'gbs_evidence':'derived 2 micro_batch_size * 3 microbatches * 8 DP from inherited v60 static config',
        'model':'Kimi236B inherited run contract; full deployed revision not independently reverified',
        'topology_id':seal['baseline_topology_sha256'],'model_flops_per_iteration':8.436548311982576e16,'peak_tflops_per_gpu':500,
        'static_evidence_path':str(cfgpath),'static_evidence_sha256':expected,'status':'PARTIAL_DEPLOYED_REVISION_AND_MFU_NUMERATOR_NOT_INDEPENDENTLY_VERIFIED'}
    (out/'scenario.json').write_text(json.dumps(scenario,indent=2)+'\n')
    # Publication-friendly standalone plots, with no interactive dependency or external assets.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ev=pd.read_csv(input_run/'evaluator_only/iteration_results.csv');ph=pd.read_csv(input_run/'evaluator_only/phase_metrics.csv')
    names=['baseline','shape_split','shape_split_fixed_reconciliation','shape_same_window']
    colors=['#263c52','#bd5448','#a68b46','#2877a4']
    fig,axes=plt.subplots(1,2,figsize=(13.5,4.5),layout='constrained')
    for name,c in zip(names,colors):
        g=ev[(ev.variant==name)&ev.iteration.isin([85,90,95,100])]
        axes[0].plot(g.iteration,g.profiler_ape_pct,'o-',label=name,color=c,linewidth=1.5)
        axes[1].plot(g.iteration,g.mfu_relative_ape_pct,'o-',label=name,color=c,linewidth=1.5)
    for ax,title in zip(axes,['Profiler step absolute relative error','MFU relative error, derived from training clock']):
        ax.set_title(title);ax.set_ylabel('Error (%)');ax.set_xlabel('Target224 iteration — development evaluation');ax.set_xticks([85,90,95,100]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out/'iteration_errors.svg');fig.savefig(out/'iteration_errors.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained');xs=np.arange(4)
    for i,(name,c) in enumerate(zip(names,colors)):
        g=ph[(ph.variant==name)&(ph.split=='development_primary')].set_index('phase').loc[['entry','onef1b','tail','outer']]
        ax.bar(xs+(i-1.5)*.18,g.bias_ms/1000,width=.18,color=c,label=name)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(xs,['Entry','1F1B envelope','Tail incl. reconciliation','Outside Profiler'])
    ax.set_ylabel('Mean predicted − observed (s)');ax.set_title('Additive time ledger — four target224 development iterations');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'phase_error_ledger.svg');fig.savefig(out/'phase_error_ledger.png',dpi=150);plt.close(fig)
    print(json.dumps(decision,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-run',type=Path,required=True);p.add_argument('--run-id',default='delivery');p.add_argument('--worker',type=Path)
    a=p.parse_args();run=a.input_run.resolve()
    assert run.is_relative_to(ROOT/'results/w37/A')
    if a.worker:
        assert a.worker.resolve().is_relative_to(ROOT/'results/w37/A')
        worker(run,a.worker.resolve());return 0
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.run_id):p.error('simple run-id required')
    out=ROOT/'results/w37/A'/('onef1b-review-'+a.run_id);out.mkdir(parents=True,exist_ok=False)
    cmd=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--bind',str(out),str(out),'--tmpfs','/tmp','--unshare-net','--chdir',str(ROOT),'--',str(PYTHON),'-B',str(Path(__file__).resolve()),'--input-run',str(run),'--worker',str(out)]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mplconfig'))
    (out/'command.json').write_text(json.dumps({'argv':cmd,'script_sha256':sha(__file__),'input_manifest_sha256':sha(run/'run_manifest.json')},indent=2)+'\n')
    with (out/'execution.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    artifacts=[{'path':str(x),'sha256':sha(x),'size_bytes':x.stat().st_size} for x in out.rglob('*') if x.is_file()]
    (out/'run_manifest.json').write_text(json.dumps({'exit_code':r.returncode,'artifacts':artifacts},indent=2)+'\n')
    print((out/'execution.log').read_text());print(out);return r.returncode


if __name__=='__main__':raise SystemExit(main())

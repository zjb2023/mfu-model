"""Target values enter only after candidate seal; all target scores are development."""
import json
import numpy as np
import pandas as pd
from worker import csv
from smoke_worker import dump

WINDOW=[85,90,95,100]


def evaluate(out,records,paths,partitions):
    truth=pd.read_csv(paths['iteration_ground_truth.csv'])
    payload=json.loads(paths['dag_v682_stage_aware_pp_gradient_payload.json'].read_text())
    by={int(r['iteration']):r for r in payload['evaluation']}
    results=[];ledger=[];bars=[]
    for t in truth.itertuples():
        it=int(t.iteration)
        if str(it) not in payload['target_trace']:continue
        e=by[it];observed=payload['target_trace'][str(it)]['bars']
        lo=min(b['start_ms'] for b in observed);hi=max(b['end_ms'] for b in observed)
        entry=e['actual_entry_ms']+lo;fb=hi-lo;tail=t.actual_profiler_step_ms-entry-fb;outer=t.actual_training_step_ms-t.actual_profiler_step_ms
        assert abs(t.actual_profiler_step_ms-e['actual_profiler_ms'])<1e-6
        split='development_primary' if it in WINDOW else 'posthoc_60_80_diagnostic'
        for b in observed:
            bars.append(dict(iteration=it,pp_stage=b['stage'],microbatch=b['microbatch'],phase={'F':'forward','B':'backward','FWD':'forward','BWD':'backward'}.get(b['phase'],b['phase']),
                actual_start_ms=b['start_ms']-lo,actual_end_ms=b['end_ms']-lo,actual_duration_ms=b['end_ms']-b['start_ms'],split=split,
                scope='16-rank CPU phase envelope; no per-rank GPU active observation'))
        for r in records:
            ap=float(t.actual_profiler_step_ms);at=float(t.actual_training_step_ms)
            num=100*8.436548311982576e16/(224*500e12)*1000;pm=num/r['training_ms'];am=num/at
            results.append(dict(variant=r['variant'],iteration=it,split=split,
                predicted_profiler_ms=r['profiler_ms'],actual_profiler_ms=ap,profiler_error_ms=r['profiler_ms']-ap,profiler_ape_pct=100*abs(r['profiler_ms']-ap)/ap,
                predicted_training_ms=r['training_ms'],actual_training_ms=at,training_error_ms=r['training_ms']-at,training_ape_pct=100*abs(r['training_ms']-at)/at,
                predicted_mfu_pct=pm,actual_mfu_pct_derived=am,mfu_error_pp=pm-am,mfu_relative_ape_pct=100*abs(pm-am)/am,
                mfu_basis='inherited FLOPs/peak and training clock; numerator not independently verified'))
            total=0
            for name,actual in [('entry',entry),('onef1b',fb),('tail',tail),('outer',outer)]:
                pred=r[name+'_ms'];total+=pred-actual
                ledger.append(dict(variant=r['variant'],iteration=it,split=split,phase=name,actual_ms=actual,predicted_ms=pred,error_ms=pred-actual,
                                   ape_pct=100*abs(pred-actual)/actual))
            assert abs(total-(r['training_ms']-at))<1e-7
    ev=pd.DataFrame(results);ld=pd.DataFrame(ledger);obs=pd.DataFrame(bars)
    csv(out,'evaluator_only/iteration_results.csv',ev);csv(out,'evaluator_only/phase_results.csv',ld);csv(out,'evaluator_only/observed_phase_envelopes.csv',obs)
    metrics=ev.groupby(['variant','split']).agg(count=('iteration','size'),profiler_mape_pct=('profiler_ape_pct','mean'),profiler_bias_ms=('profiler_error_ms','mean'),
        training_mape_pct=('training_ape_pct','mean'),training_bias_ms=('training_error_ms','mean'),mfu_relative_mape_pct=('mfu_relative_ape_pct','mean'),mfu_bias_pp=('mfu_error_pp','mean')).reset_index()
    csv(out,'evaluator_only/metrics.csv',metrics)
    csv(out,'evaluator_only/phase_metrics.csv',ld.groupby(['variant','split','phase']).agg(actual_mean_ms=('actual_ms','mean'),predicted_mean_ms=('predicted_ms','mean'),
        bias_ms=('error_ms','mean'),mape_pct=('ape_pct','mean')).reset_index())
    pred=pd.read_csv(out/'prediction/phase_nodes.csv.gz')
    grouped=pred.groupby(['variant','pp_stage','phase','microbatch']).agg(predicted_start_ms=('start_ms','min'),predicted_end_ms=('end_ms','max')).reset_index()
    grouped['predicted_duration_ms']=grouped.predicted_end_ms-grouped.predicted_start_ms
    joined=obs.merge(grouped,on=['pp_stage','phase','microbatch'],validate='many_to_many')
    assert len(joined)==len(obs)*pred.variant.nunique()
    joined['duration_error_ms']=joined.predicted_duration_ms-joined.actual_duration_ms
    joined['start_error_ms']=joined.predicted_start_ms-joined.actual_start_ms
    csv(out,'evaluator_only/stage_mb_phase_results.csv',joined)
    csv(out,'evaluator_only/stage_phase_metrics.csv',joined.groupby(['variant','split','pp_stage','phase']).agg(duration_mae_ms=('duration_error_ms',lambda v:v.abs().mean()),
        duration_bias_ms=('duration_error_ms','mean'),start_mae_ms=('start_error_ms',lambda v:v.abs().mean())).reset_index())
    # Source-profile transfer decomposition for lane0 only. Two partitions describe
    # the same phase from different axes and are deliberately scored separately.
    wrapper=pd.read_csv(paths['target224_window_partition_ground_truth.csv.gz'])
    gpu=pd.read_csv(paths['target224_outside_wrapper_ground_truth.csv.gz'])
    dump(out/'evaluator_only/target_partition_schema.json',{'wrapper_columns':list(wrapper.columns),'gpu_columns':list(gpu.columns),'gpu_rows':len(gpu),
        'scope':'retrospective target lane0; never calibration or blind validation'})
    comps=pd.DataFrame(partitions);comps=comps[comps.case.eq('target224')];cr=[]
    for view,data in [('gpu_disjoint',gpu),('wrapper_disjoint',wrapper)]:
        cols=list(comps[comps.view.eq(view)].component.unique())
        assert set(cols).issubset(data.columns),list(data.columns)
        assert data.pp_lane.eq(0).all()
        key=['pp_stage','pp_lane','phase','microbatch']
        truthlong=data.melt(id_vars=['iteration']+key,value_vars=cols,var_name='component',value_name='actual_ms')
        z=truthlong.merge(comps[comps.view.eq(view)],on=key+['component'],validate='many_to_many')
        assert len(z)==len(truthlong)*comps.variant.nunique()
        z['error_ms']=z.predicted_ms-z.actual_ms;z['split']=np.where(z.iteration.isin(WINDOW),'development_primary','posthoc_60_80_diagnostic');cr.append(z)
    cp=pd.concat(cr,ignore_index=True);csv(out,'evaluator_only/lane0_partition_results.csv.gz',cp)
    csv(out,'evaluator_only/lane0_partition_metrics.csv',cp.groupby(['variant','split','view','phase','component']).agg(actual_mean_ms=('actual_ms','mean'),
        predicted_mean_ms=('predicted_ms','mean'),bias_ms=('error_ms','mean'),mae_ms=('error_ms',lambda x:x.abs().mean())).reset_index())
    main=metrics[metrics.split.eq('development_primary')]
    assert abs(main[main.variant.eq('v685_frozen')].profiler_mape_pct.iloc[0]-10.36952849443861)<1e-8
    return main.to_dict('records')

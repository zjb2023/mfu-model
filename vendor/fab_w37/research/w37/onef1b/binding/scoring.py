"""Only invoked by the post-seal evaluator; no model code imports target values."""
import json
import pandas as pd
from common import RUN,dump,sha,write_csv

def score(spec,paths,out):
    seal=json.loads((RUN/'model/prediction_seal.json').read_text())
    assert seal['status']=='SEALED_BEFORE_TARGET_EVALUATOR'
    for p,d in seal['files'].items():assert sha(RUN/'model'/p)==d
    actual=pd.read_csv(paths['version_iterations'])
    actual=actual[actual.variant.eq('v685_frozen')&actual.split.eq('development_primary')]
    phases=pd.read_csv(paths['v685_phase_rows'])
    phases=phases[phases.variant.eq('v685_frozen')&phases.split.eq('development_primary')]
    predictions=pd.read_csv(RUN/'model/predictions.csv')
    assert set(actual.iteration)=={85,90,95,100} and len(actual)==4 and len(phases)==16
    point=[];phase_rows=[];checks=[]
    for p in predictions.itertuples():
        for a in actual.itertuples():
            r={'variant':p.variant,'iteration':int(a.iteration)}
            for metric in ['profiler','training']:
                pred=getattr(p,metric+'_ms');obs=getattr(a,'actual_'+metric+'_ms')
                r.update({metric+'_predicted_ms':pred,metric+'_actual_ms':obs,metric+'_error_ms':pred-obs,metric+'_APE_pct':abs(pred-obs)/obs*100})
                if p.variant=='frozen_v685':assert abs(pred-getattr(a,'predicted_'+metric+'_ms'))<1e-6
            obs=a.actual_mfu_pct_derived
            r.update(mfu_predicted_pct=p.mfu_pct,mfu_actual_pct=obs,mfu_relative_APE_pct=abs(p.mfu_pct-obs)/obs*100,mfu_absolute_error_percentage_points=abs(p.mfu_pct-obs))
            if p.variant=='frozen_v685':assert abs(p.mfu_pct-a.predicted_mfu_pct)<1e-9
            iteration_phases=phases[phases.iteration.eq(a.iteration)]
            assert abs(iteration_phases.actual_ms.sum()-a.actual_training_ms)<1e-6
            assert abs(p.entry_ms+p.onef1b_ms+p.tail_ms+p.outer_ms-p.training_ms)<1e-6
            for ph in iteration_phases.itertuples():
                pred=getattr(p,ph.phase+'_ms');error=pred-ph.actual_ms
                phase_rows.append(dict(variant=p.variant,iteration=int(a.iteration),phase=ph.phase,predicted_ms=pred,actual_ms=ph.actual_ms,error_ms=error,APE_pct=abs(error)/ph.actual_ms*100))
                if ph.phase=='onef1b':r.update(onef1b_predicted_ms=pred,onef1b_actual_ms=ph.actual_ms,onef1b_APE_pct=abs(error)/ph.actual_ms*100)
                if p.variant=='frozen_v685':assert abs(pred-ph.predicted_ms)<1e-6
            point.append(r)
            checks.append({'variant':p.variant,'iteration':int(a.iteration),'phase_time_conservation':True,'baseline_reproduction_verified':p.variant=='frozen_v685'})
    points=pd.DataFrame(point);details=pd.DataFrame(phase_rows)
    points.to_csv(out/'iteration_results.csv',index=False);details.to_csv(out/'phase_iteration_results.csv',index=False)
    summary=points.groupby('variant',sort=False).agg(iterations=('iteration','size'),onef1b_MAPE_pct=('onef1b_APE_pct','mean'),profiler_MAPE_pct=('profiler_APE_pct','mean'),training_MAPE_pct=('training_APE_pct','mean'),mfu_relative_MAPE_pct=('mfu_relative_APE_pct','mean'),mfu_MAE_percentage_points=('mfu_absolute_error_percentage_points','mean')).reset_index()
    summary.to_csv(out/'metrics.csv',index=False)
    phase_summary=details.groupby(['variant','phase'],sort=False).agg(MAPE_pct=('APE_pct','mean'),bias_ms=('error_ms','mean'),predicted_ms=('predicted_ms','mean')).reset_index()
    phase_summary.to_csv(out/'phase_metrics.csv',index=False)
    baseline=phase_summary[phase_summary.variant.eq('frozen_v685')].set_index('phase')
    regress=[]
    for r in phase_summary.itertuples():
        old=baseline.loc[r.phase]
        regress.append(dict(variant=r.variant,phase=r.phase,MAPE_change_pp=r.MAPE_pct-old.MAPE_pct,predicted_change_ms=r.predicted_ms-old.predicted_ms,regression=r.MAPE_pct>old.MAPE_pct+1e-9))
    write_csv(out/'phase_regression.csv',regress)
    dump(out/'summary.json',{'status':'PASS','metrics':summary.to_dict('records'),'checks':checks,'decision':'CANDIDATE_SENSITIVITY_ONLY; source full-iteration binding remains unvalidated; no formal promotion','target_split':'development, already exposed','source_fit':[85,90],'inherited_source_data':'60–100 includes source incremental evaluation; not independent blind test','topology_sha256':seal['topology']})
    print(summary.to_string(index=False))

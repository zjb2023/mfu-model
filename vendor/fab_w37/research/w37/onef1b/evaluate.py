"""Evaluator-only additive boundaries and separate time / MFU errors."""
import json
import numpy as np
import pandas as pd
from worker import csv, dump, phases, WINDOW


def evaluate(out,records,truth,payload,historical,source,events,boundaries,contract):
    evaluation=[]; phase_rows=[]; observed=[]
    # The historical payload stores per-stage 16-rank F/B envelopes relative to first F.
    by_iter={int(r['iteration']):r for r in payload['evaluation']}
    for t in truth.itertuples():
        it=int(t.iteration)
        if str(it) not in payload['target_trace']:
            continue
        e=by_iter[it]; bars=payload['target_trace'][str(it)]['bars']
        first=min(b['start_ms'] for b in bars);last=max(b['end_ms'] for b in bars)
        actual_entry=e['actual_entry_ms']+first
        actual_fb=last-first
        actual_tail=float(t.actual_profiler_step_ms)-e['actual_entry_ms']-last
        assert abs(float(t.actual_profiler_step_ms)-e['actual_profiler_ms'])<1e-6
        for b in bars:
            observed.append({'iteration':it,'node_id':b['id'],'pp_stage':b['stage'],'phase':b['phase'],'microbatch':b['microbatch'],
                             'start_ms':b['start_ms']+e['actual_entry_ms'],'end_ms':b['end_ms']+e['actual_entry_ms'],
                             'observation_scope':'16-rank CPU F/B phase envelope; not kernel-active union',
                             'evidence':'pinned v682 historical target_trace bars'})
        for r in records:
            ap=float(t.actual_profiler_step_ms);at=float(t.actual_training_step_ms)
            # Inherited model configuration; this does not independently verify FLOPs or peak.
            numerator=100*8.436548311982576e16/(224*500e12)*1000
            pm=numerator/r['training_ms'];am=numerator/at
            evaluation.append({'variant':r['variant'],'iteration':it,'split':'development_primary' if it in WINDOW else 'posthoc_60_80_regression',
                'predicted_profiler_step_ms':r['profiler_ms'],'actual_profiler_step_ms':ap,
                'profiler_error_ms':r['profiler_ms']-ap,'profiler_ape_pct':100*abs(r['profiler_ms']-ap)/ap,
                'predicted_training_step_ms':r['training_ms'],'actual_training_step_ms':at,
                'training_error_ms':r['training_ms']-at,'training_ape_pct':100*abs(r['training_ms']-at)/at,
                'predicted_mfu_pct':pm,'actual_mfu_pct_derived':am,'mfu_error_pp':pm-am,
                'mfu_relative_ape_pct':100*abs(pm-am)/am,'mfu_basis':'inherited FLOPs/peak, training clock; no independently observed MFU'})
            phase_sum=0
            for name,actual,pred in [('entry',actual_entry,r['entry_ms']),('onef1b',actual_fb,r['onef1b_ms']),
                                     ('tail',actual_tail,r['tail_ms']),('outer',at-ap,r['outer_ms'])]:
                phase_rows.append({'variant':r['variant'],'iteration':it,'split':'development_primary' if it in WINDOW else 'posthoc_60_80_regression',
                    'phase':name,'actual_ms':actual,'predicted_ms':pred,'error_ms':pred-actual,'ape_pct':100*abs(pred-actual)/actual,
                    'boundary_semantics':'tail includes unlocalized graph reconciliation; outer outside profiler'})
                phase_sum+=pred-actual
            assert abs(phase_sum-(r['training_ms']-at))<1e-7
    ev=pd.DataFrame(evaluation);ph=pd.DataFrame(phase_rows)
    csv(out,'evaluator_only/iteration_results.csv',ev)
    csv(out,'evaluator_only/phase_results.csv',ph)
    csv(out,'evaluator_only/observed_phase_nodes.csv',pd.DataFrame(observed))
    summary=ev.groupby(['variant','split'],as_index=False).agg(count=('iteration','size'),
        profiler_mape_pct=('profiler_ape_pct','mean'),profiler_bias_ms=('profiler_error_ms','mean'),
        training_mape_pct=('training_ape_pct','mean'),training_bias_ms=('training_error_ms','mean'),
        mfu_relative_mape_pct=('mfu_relative_ape_pct','mean'),mfu_bias_pp=('mfu_error_pp','mean'))
    phase_summary=ph.groupby(['variant','split','phase'],as_index=False).agg(actual_mean_ms=('actual_ms','mean'),
        predicted_mean_ms=('predicted_ms','mean'),bias_ms=('error_ms','mean'),mape_pct=('ape_pct','mean'))
    csv(out,'evaluator_only/metrics.csv',summary);csv(out,'evaluator_only/phase_metrics.csv',phase_summary)
    p=summary[(summary.variant=='baseline') & (summary.split=='development_primary')].iloc[0]
    assert abs(p.profiler_mape_pct-10.36952849443861)<1e-8
    # Source graph time origin is first phase, not ProfilerStep. Show both conventions explicitly.
    sn=phases(source,True);source_start=float(sn.start_ns.min())/1e6;source_end=float(sn.end_ns.max())/1e6
    source_rows=[]
    source_obs=events.groupby('iteration').agg(first=('observed_start_ns','min'),last=('observed_end_ns','max'))
    for r in boundaries.itertuples():
        it=int(r.iteration);o=source_obs.loc[it]
        actual_fb=(int(o['last'])-int(o['first']))/1e6
        actual_tail=r.profiler_step_ms-r.profiler_entry_to_phase_ms-actual_fb
        for rec in records:
            pred_fb=rec.get('source_phase_end_ms',source_end)-rec.get('source_phase_start_ms',source_start)
            pred_tail=rec['source_raw_ms']-rec.get('source_phase_end_ms',source_end)
            explicit=float(r.profiler_entry_to_phase_ms)+rec['source_raw_ms']
            source_rows.append({'iteration':it,'variant':rec['variant'],'split':'legacy_fit_and_incremental_check' if it in WINDOW else 'historical_diagnostic',
                'actual_entry_ms':r.profiler_entry_to_phase_ms,'actual_onef1b_ms':actual_fb,'actual_tail_ms':actual_tail,
                'predicted_graph_entry_ms':rec.get('source_phase_start_ms',source_start),'predicted_onef1b_ms':pred_fb,'predicted_tail_ms':pred_tail,
                'onef1b_error_ms':pred_fb-actual_fb,'onef1b_ape_pct':100*abs(pred_fb-actual_fb)/actual_fb,
                'tail_error_ms':pred_tail-actual_tail,'tail_ape_pct':100*abs(pred_tail-actual_tail)/actual_tail,
                'actual_profiler_ms':r.profiler_step_ms,'raw_graph_ms':rec['source_raw_ms'],
                'legacy_unlocated_reconciliation_ms':r.profiler_step_ms-rec['source_raw_ms'],
                'explicit_entry_plus_graph_ms':explicit,'signed_unlocated_after_entry_ms':r.profiler_step_ms-explicit,
                'source_training_mfu_status':'not available in pinned source boundary input'})
            assert abs(r.profiler_entry_to_phase_ms+actual_fb+actual_tail-r.profiler_step_ms)<1e-7
    csv(out,'evaluator_only/source_iteration_ledger.csv',pd.DataFrame(source_rows))
    hist={'three_stage':{'target':'224gpu_pp14_cp2_a2a','window':historical['scope']['target_iterations'],
             'mape_pct':historical['metrics']['target224_profiler']['mape_pct']},
          'v682':{'window':payload['iterations'],'mape_pct':payload['metrics']['all_mape_pct'],
                  'ledger':payload['error_analysis']['additive_wall_clock_ledger'],
                  'denominator_ms':payload['error_analysis']['mean_underprediction_ms']},
          'v685':{'window':WINDOW,'mape_pct':float(p.profiler_mape_pct)},
          'conclusion':'16.823672% is nine target224 points; 10.369528% is four target224 points. Historical 10.3 is approximate with ambiguous exact version. 75.839644% is a signed mean error contribution, not phase MAPE.'}
    dump(out/'evaluator_only/historical_metric_audit.json',hist)
    return summary.to_dict('records')

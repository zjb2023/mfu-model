"""Whole-source-iteration cost scenarios preserve observed covariance across nodes."""
import numpy as np
import pandas as pd
from worker import csv
from smoke_worker import dump
from pp_graph import envelope


def aggregate(out,records,phasepred,sourcegraphs):
    source=pd.read_csv(out/'source_validation/iteration_results.csv');extra=[];spans=[]
    for name,fit in [('v688_split_ensemble',[85,90]),('v688_full4_ensemble',[85,90,95,100])]:
        names=[f'v688_member{it}' for it in fit];members=[r for r in records if r['variant'] in names]
        assert len(members)==len(fit)
        rec={k:float(np.mean([r[k] for r in members])) for k in members[0] if k!='variant'};rec['variant']=name;records.append(rec)
        sen={**rec,'variant':name+'_no_legacy_reconciliation_sensitivity'}
        for field in ['tail_ms','profiler_ms','training_ms']:sen[field]-=rec['legacy_reconciliation_ms']
        records.append(sen)
        srcvals=[envelope(sourcegraphs[n][0])['onef1b_ms'] for n in names]
        for case,vals in [('source256',srcvals),('target224',[r['onef1b_ms'] for r in members])]:
            spans.append(dict(variant=name,case=case,mean_onef1b_ms=float(np.mean(vals)),scenario_min_ms=min(vals),scenario_max_ms=max(vals),
                members=','.join(names),interpretation='equal-weight empirical scenario span, NOT confidence/prediction interval; 2 or 4 exposed source samples'))
        for r in source[source.variant.eq(names[0])].to_dict('records'):
            pred=float(np.mean(srcvals));actual=r['actual_onef1b_ms'];it=r['iteration']
            extra.append({**r,'variant':name,'predicted_onef1b_ms':pred,'error_ms':pred-actual,'ape_pct':100*abs(pred-actual)/actual,
                'predicted_program_ms':float(np.mean([envelope(sourcegraphs[n][0])['program_ms'] for n in names])),
                'split':'source_fit' if it in fit else ('source_incremental_validation' if it in [85,90,95,100] else 'historical_diagnostic'),
                'scope':'mean of independent free-running scenario envelopes; not one replayed mean-cost DAG'})
        frames=[p for p in phasepred if p.variant.iloc[0] in names]
        mean=pd.concat(frames).groupby(['pp_stage','pp_lane','phase','microbatch'])[['start_ms','end_ms','duration_ms']].mean().reset_index()
        mean['variant']=name;phasepred.append(mean)
    source=pd.concat([source,pd.DataFrame(extra)],ignore_index=True);csv(out,'source_validation/iteration_results.csv',source)
    metrics=source.groupby(['variant','split']).agg(count=('iteration','size'),onef1b_mape_pct=('ape_pct','mean'),bias_ms=('error_ms','mean')).reset_index()
    csv(out,'source_validation/metrics.csv',metrics);csv(out,'prediction/scenario_spans.csv',pd.DataFrame(spans))
    dump(out/'prediction/scenario_contract.json',{'mean_of_replays_not_replay_of_mean_costs':True,'weights':'equal, source only',
        'stage_envelope_aggregation':'mean of each member stage envelope; not min/max of expected per-rank timestamps',
        'unseen_regime_coverage':'not guaranteed by 2 or 4 source cost vectors','internal_partition_basis':'member source profiles; no averaged graph topology claimed'})
    return metrics

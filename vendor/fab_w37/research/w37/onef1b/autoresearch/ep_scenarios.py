"""Average independent source-iteration replays, never their node costs first."""
import numpy as np
import pandas as pd

KEY=['pp_stage','pp_lane','phase','microbatch']


def expected_step(records,members,name):
    selected=[r for r in records if r['variant'] in members]
    assert {r['variant'] for r in selected}==set(members) and len(selected)==len(members)
    return {'variant':name,**{k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k!='variant'}}


def expected_source(results,members,name,fit):
    selected=results[results.variant.isin(members)]
    assert selected.groupby('iteration').variant.nunique().eq(len(members)).all()
    means=selected.groupby('iteration')[['predicted_onef1b_ms','predicted_program_ms']].mean()
    result=selected[selected.variant.eq(members[0])].copy();result['variant']=name
    for col in means:result[col]=result.iteration.map(means[col])
    result['error_ms']=result.predicted_onef1b_ms-result.actual_onef1b_ms
    result['ape_pct']=100*result.error_ms.abs()/result.actual_onef1b_ms
    result['split']=np.where(result.iteration.isin(fit),'source_fit',np.where(result.iteration.isin([95,100]),'source_incremental_validation','historical_diagnostic'))
    result['scope']='equal mean of independent source-iteration CPU EP replays; not one mean-cost graph'
    return result


def expected_phases(frames,members,name):
    selected=pd.concat(frames,ignore_index=True);selected=selected[selected.variant.isin(members)]
    assert selected.groupby(KEY).variant.nunique().eq(len(members)).all()
    result=selected.groupby(KEY)[['start_ms','end_ms','duration_ms']].mean().reset_index();result['variant']=name
    return result


def expected_partition(rows,members,name):
    selected=pd.DataFrame(rows);selected=selected[selected.variant.isin(members)]
    keys=['case']+KEY+['view','component']
    assert selected.groupby(keys).variant.nunique().eq(len(members)).all()
    result=selected.groupby(keys).predicted_ms.mean().reset_index();result['variant']=name
    result['scope']='mean of member diagnostic partitions; not additional CPU graph costs'
    return result.to_dict('records')

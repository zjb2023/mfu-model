"""Source-only local queue model; observed CPU submissions are explicit conditions."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from smoke_worker import dump,sha
from worker import csv

BASE=['category','stream','family','name','CPU_owner_name_checked','launch_FB_phase','launch_EP_semantic_region']
LEVELS=[('shape_context',BASE+['CPU_owner_input_dims']),('operation_context',BASE),
        ('kernel_stream',['category','stream','name']),('family_stream',['category','stream','family'])]
FEATURES=['iteration','rank','device','event_id','runtime_start_ns','runtime_end_ns','release_proxy_ns','release_proxy_kind',
          'launch_FB_window_id','CPU_owner_input_dims']+BASE
METHODS=['independent_latency','same_stream_median','same_stream_mean','same_stream_zero_remainder']


def feature_frame(data):
    return data[FEATURES].copy()


class DeviceQueueModel:
    def __init__(self,data,fit=(85,90)):
        self.fit=list(fit);training=data[data.iteration.isin(fit)].copy()
        assert set(training.iteration)==set(fit)
        assert training.observed_enqueue_remainder_ns.ge(0).all() and training.duration_ns.ge(0).all()
        self.tables=[];records=[]
        for level,keys in LEVELS:
            normalized=training.copy()
            for c in keys:normalized[c]=normalized[c].fillna('<missing>').astype(str)
            table={}
            for key,group in normalized.groupby(keys,dropna=False,sort=True):
                if not isinstance(key,tuple):key=(key,)
                encoded=json.dumps(dict(zip(keys,key)),sort_keys=True,separators=(',',':'))
                row=dict(level=level,key=encoded,cost_id=hashlib.sha256((level+encoded).encode()).hexdigest()[:20],
                    fit_iterations=','.join(map(str,fit)),samples=len(group),
                    duration_median_ns=int(round(float(group.duration_ns.median()))),duration_mean_ns=int(round(float(group.duration_ns.mean()))),
                    API_latency_median_ns=int(round(float(group.observed_API_latency_ns.median()))),
                    enqueue_remainder_median_ns=int(round(float(group.observed_enqueue_remainder_ns.median()))),
                    enqueue_remainder_mean_ns=int(round(float(group.observed_enqueue_remainder_ns.mean()))))
                table[key]=row;records.append(row)
            self.tables.append((level,keys,table))
        self.parameters=pd.DataFrame(records)

    def lookup(self,row):
        for level,keys,table in self.tables:
            key=tuple('<missing>' if pd.isna(row[c]) else str(row[c]) for c in keys)
            if key in table:return table[key]
        raise ValueError('no source-supported operation/family cost; do not silently assign zero')

    def predict(self,features,methods=METHODS):
        assert set(features.columns)==set(FEATURES),'GPU truth is not a prediction feature'
        x=features.sort_values(['iteration','rank','device','stream','runtime_start_ns','runtime_end_ns','event_id'])
        assert not x.duplicated(['iteration','rank','device','stream','runtime_start_ns']).any(),'ambiguous CPU submission order'
        rows=[]
        for _,group in x.groupby(['iteration','rank','device','stream'],sort=False):
            assert np.all(group.runtime_start_ns.to_numpy(dtype='int64')[1:]>=group.runtime_end_ns.to_numpy(dtype='int64')[:-1])
            previous={m:None for m in methods};previous_id=''
            for row in group.to_dict('records'):
                p=self.lookup(row);release=int(row['release_proxy_ns'])
                for method in methods:
                    if method=='independent_latency':
                        start=release+p['API_latency_median_ns'];duration=p['duration_median_ns'];remainder=0
                    else:
                        mean=method=='same_stream_mean'
                        duration=p['duration_mean_ns' if mean else 'duration_median_ns']
                        remainder=0 if method=='same_stream_zero_remainder' else p['enqueue_remainder_mean_ns' if mean else 'enqueue_remainder_median_ns']
                        start=max(release,previous[method] if previous[method] is not None else release)+remainder
                    end=start+duration;previous[method]=end
                    rows.append(dict(event_id=row['event_id'],iteration=row['iteration'],rank=row['rank'],device=row['device'],stream=row['stream'],
                        method=method,cost_id=p['cost_id'],cost_level=p['level'],predicted_start_ns=start,predicted_end_ns=end,
                        predicted_duration_ns=duration,predicted_enqueue_remainder_ns=remainder,release_proxy_ns=release,
                        previous_predicted_event_id=previous_id if method!='independent_latency' else '',
                        prediction_scope='CPU_submission_conditioned_visible_device_only'))
                previous_id=row['event_id']
        return pd.DataFrame(rows)


def fit_predict_score(out,data,windows,plan):
    model=DeviceQueueModel(data);parameters=model.parameters
    features=feature_frame(data);predictions=model.predict(features)
    csv(out,'source_device_queue_parameters.csv.gz',parameters);csv(out,'source_device_queue_features.csv.gz',features)
    csv(out,'source_device_queue_predictions_sealed.csv.gz',predictions)
    sealed=['source_device_queue_parameters.csv.gz','source_device_queue_features.csv.gz','source_device_queue_predictions_sealed.csv.gz']
    dump(out/'source_device_queue_prediction_seal.json',dict(status='SEALED_SOURCE_FIT_CPU_CONDITIONED_LOCAL_PREDICTION',
        fit_iterations=[85,90],incremental_validation=[95,100],target_parameter_updates=0,
        files=[dict(path=n,sha256=sha(out/n)) for n in sealed],
        conditions='Observed CPU runtime start/end and structural device-event metadata; no validation GPU timing is a feature. Not free-running CPU or224 prediction.',
        hidden='Graph work and cross-stream dependencies absent. Enqueue remainder is empirical omitted release, not independently measured CPU/GPU service.'))
    # Only after the parameters/features/predictions seal attach GPU truth.
    truth=data[['event_id','iteration','rank','start_ns','end_ns','duration_ns','family','category','launch_FB_phase','launch_FB_window_id','split']]
    scored=predictions.merge(truth,on=['event_id','iteration','rank'],validate='many_to_one')
    scored['start_error_ns']=scored.predicted_start_ns-scored.start_ns
    scored['end_error_ns']=scored.predicted_end_ns-scored.end_ns
    scored['duration_error_ns']=scored.predicted_duration_ns-scored.duration_ns
    csv(out,'source_device_queue_event_validation.csv.gz',scored)
    def metrics(group):
        return pd.Series(dict(events=len(group),start_MAE_ms=group.start_error_ns.abs().mean()/1e6,
            end_MAE_ms=group.end_error_ns.abs().mean()/1e6,end_p95_AE_ms=group.end_error_ns.abs().quantile(.95)/1e6,
            end_bias_ms=group.end_error_ns.mean()/1e6,duration_MAE_ms=group.duration_error_ns.abs().mean()/1e6))
    for name,keys in [('metrics',['method','split']),('per_iteration',['method','iteration','split']),
                      ('per_family',['method','split','family']),('per_stream',['method','split','stream'])]:
        frame=scored.groupby(keys).apply(metrics,include_groups=False).reset_index();csv(out,'source_device_queue_'+name+'.csv',frame)
    # A CPU-owned device-work endpoint is NOT the CPU F/B boundary or 1F1B envelope.
    local=scored[~scored.family.eq('PP_candidate')&~scored.launch_FB_window_id.eq('outside_FB')]
    keys=['method','iteration','rank','split','launch_FB_phase','launch_FB_window_id']
    endpoints=local.groupby(keys).agg(predicted_last_device_end_ns=('predicted_end_ns','max'),observed_last_device_end_ns=('end_ns','max'),events=('event_id','size')).reset_index()
    endpoints['endpoint_error_ns']=endpoints.predicted_last_device_end_ns-endpoints.observed_last_device_end_ns
    endpoints=endpoints.merge(windows[windows.window_type.eq('FB_annotation')][['window_id','microbatch','end_ns']].rename(
        columns={'window_id':'launch_FB_window_id','end_ns':'CPU_annotation_end_ns'}),on='launch_FB_window_id',validate='many_to_one')
    endpoints['observed_device_tail_after_CPU_ns']=endpoints.observed_last_device_end_ns-endpoints.CPU_annotation_end_ns
    csv(out,'source_device_queue_CPU_owned_device_endpoints.csv',endpoints)
    endpoint_metrics=endpoints.groupby(['method','split','launch_FB_phase']).endpoint_error_ns.agg(
        phases='size',endpoint_MAE_ms=lambda x:x.abs().mean()/1e6,endpoint_bias_ms=lambda x:x.mean()/1e6).reset_index()
    csv(out,'source_device_queue_endpoint_metrics.csv',endpoint_metrics)
    fallback=predictions.groupby(['method','iteration','cost_level']).size().rename('events').reset_index()
    csv(out,'source_device_queue_cost_coverage.csv',fallback)
    for f in json.loads((out/'source_device_queue_prediction_seal.json').read_text())['files']:assert sha(out/f['path'])==f['sha256']
    draw(out,pd.read_csv(out/'source_device_queue_metrics.csv'),endpoint_metrics)
    return dict(methods=METHODS,parameter_rows=len(parameters),predicted_device_events=len(features),sealed_prediction_rows=len(predictions),
        incremental_event_metrics=pd.read_csv(out/'source_device_queue_metrics.csv').query("split == 'source_incremental_validation'").to_dict('records'),
        incremental_endpoint_metrics=endpoint_metrics[endpoint_metrics.split.eq('source_incremental_validation')].to_dict('records'),
        prediction_seal_sha256=sha(out/'source_device_queue_prediction_seal.json'))


def draw(out,metrics,endpoints):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'independent_latency':'Independent\nlatency','same_stream_median':'Queue\nmedian','same_stream_mean':'Queue\nmean','same_stream_zero_remainder':'Queue\nzero remainder'}
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    a=metrics[metrics.split.eq('source_incremental_validation')].set_index('method').loc[METHODS]
    axes[0].bar(range(4),a.end_MAE_ms);axes[0].set_ylabel('GPU event end MAE (ms)')
    b=endpoints[endpoints.split.eq('source_incremental_validation')]
    for i,phase in enumerate(['forward','backward']):
        v=b[b.launch_FB_phase.eq(phase)].set_index('method').loc[METHODS]
        axes[1].bar(np.arange(4)+(i-.5)*.35,v.endpoint_MAE_ms,width=.35,label=phase)
    axes[1].set_ylabel('CPU-owned device endpoint MAE (ms)');axes[1].legend()
    for ax in axes:ax.set_xticks(range(4),[names[m] for m in METHODS])
    fig.suptitle('Source95/100 local prediction, fit85/90: observed CPU submissions are conditions')
    fig.text(.5,.02,'Visible device events only. Enqueue remainder includes unresolved release constraints.\n'
        'Device endpoint is not CPU F/B or global1F1B; no224 prediction or target calibration.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(out/'source_device_queue_local_validation.svg');fig.savefig(out/'source_device_queue_local_validation.png',dpi=160);plt.close(fig)

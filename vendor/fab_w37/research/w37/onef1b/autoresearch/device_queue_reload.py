"""Reload exact source85/90 cost tables without fitting or timing inputs."""
import hashlib
import json
import resource
from time import perf_counter

import pandas as pd
from guards import ROOT
from smoke_worker import dump,sha
from worker import csv
from device_queue_model import DeviceQueueModel,LEVELS,FEATURES,METHODS

NUMERIC=['samples','duration_median_ns','duration_mean_ns','API_latency_median_ns',
         'enqueue_remainder_median_ns','enqueue_remainder_mean_ns']
PARAMETERS=['level','key','cost_id','fit_iterations']+NUMERIC
FEATURE_INTS=['iteration','rank','runtime_start_ns','runtime_end_ns','release_proxy_ns']
PREDICTION_INTS=['iteration','rank','predicted_start_ns','predicted_end_ns','predicted_duration_ns',
                 'predicted_enqueue_remainder_ns','release_proxy_ns']


def load_parameters(parameters):
    assert set(parameters.columns)==set(PARAMETERS)
    assert parameters.fit_iterations.eq('85,90').all(),'Only original source85/90 parameters admitted'
    assert not parameters.cost_id.duplicated().any()
    assert set(parameters.level)=={level for level,keys in LEVELS}
    for key in NUMERIC:
        assert pd.api.types.is_integer_dtype(parameters[key]) and parameters[key].ge(0).all()
    assert parameters.samples.gt(0).all()
    model=DeviceQueueModel.__new__(DeviceQueueModel)
    model.fit=[85,90];model.parameters=parameters.copy();model.tables=[]
    for level,keys in LEVELS:
        table={}
        for row in parameters[parameters.level.eq(level)].to_dict('records'):
            encoded=json.loads(row['key']);assert set(encoded)==set(keys) and all(isinstance(v,str) for v in encoded.values())
            assert json.dumps(encoded,sort_keys=True,separators=(',',':'))==row['key']
            assert hashlib.sha256((level+row['key']).encode()).hexdigest()[:20]==row['cost_id']
            key=tuple(encoded[k] for k in keys);assert key not in table
            table[key]=row
        assert table;model.tables.append((level,keys,table))
    return model


def read_features(path):
    return pd.read_csv(path,dtype={c:'int64' if c in FEATURE_INTS else 'string' for c in FEATURES},low_memory=False)


def read_predictions(path):
    columns=pd.read_csv(path,nrows=0).columns
    # Empty predecessor IDs are meaningful empty strings, not missing values.
    return pd.read_csv(path,dtype={c:'int64' if c in PREDICTION_INTS else 'string' for c in columns},keep_default_na=False)


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only' and not plan['variants']
    review=json.loads((ROOT/plan['resource_review_file']).read_text());ref=plan['source_model_stages'][0]
    parameters=pd.read_csv(paths['source_device_queue_parameters.csv.gz'],dtype={'fit_iterations':'string','key':'string','cost_id':'string','level':'string'})
    assert len(parameters)==2772
    model=load_parameters(parameters)
    features=read_features(paths['source_device_queue_features.csv.gz'])
    assert set(features.columns)==set(FEATURES) and len(features)==41764 and set(features.iteration)=={85,90,95,100}
    predictions=model.predict(features,METHODS)
    assert len(predictions)==167056
    csv(out,'source_device_queue_reloaded_predictions.csv.gz',predictions)
    actual=read_predictions(out/'source_device_queue_reloaded_predictions.csv.gz')
    expected=read_predictions(paths['source_device_queue_predictions_sealed.csv.gz'])
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    counts=actual.groupby(['iteration','method']).size().rename('prediction_rows').reset_index()
    counts['all_fields_equal_to_old_source_seal']=True
    csv(out,'source_device_queue_reload_per_iteration.csv',counts)
    # Archive the identical parameter bytes and provenance without recomputing fit.
    for key in ['source_device_queue_parameters.csv.gz','source_device_queue_endpoint_metrics.csv']:
        (out/key).write_bytes(paths[key].read_bytes())
        assert sha(out/key)==sha(paths[key])
    dump(out/'source_device_queue_reload_seal.json',dict(status='SEALED_EXACT_SOURCE_DEVICE_QUEUE_RELOAD',
        source_model_stage=ref['stage_root'],source_model_manifest_sha256=ref['manifest_sha256'],
        original_local_seal_sha256=ref['local_prediction_seal_sha256'],fit_iterations=[85,90],
        all_old_source_predictions_exact=True,source_modules=ref['source_modules'],
        files=[dict(path=n,sha256=sha(out/n)) for n in ['source_device_queue_parameters.csv.gz','source_device_queue_reloaded_predictions.csv.gz','source_device_queue_reload_per_iteration.csv']],
        no_new_fit=True,no_target_semantic_access=True,scope='CPU-submission-conditioned visible-device prediction, not global1F1B'))
    dump(out/'field_contract.json',dict(input='9 exact cached T19 source files; completed model manifest and local prediction seal verified',
        fit='Parameters are the original source85/90 local-device model;95/100 historically exposed incremental evaluation only',
        replay='41,764 visible device events x4 fixed methods; integer-ns times and predecessor/cost bindings match old sealed outputs exactly',
        features=FEATURES,methods=METHODS,
        condition='Observed CPU start/end, event metadata, owner shape and phase identity; no validation GPU timing is a prediction feature',
        omitted='Hidden graph work and cross-stream constraints remain unresolved; enqueue remainder is empirical, not pure service',
        next='Conditional target transfer requires a separate evaluator plan and source reload gate; do not call it global1F1B prediction'))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;elapsed=perf_counter()-begin
    assert peak<=review['resource']['maximum_peak_RSS_bytes']
    dump(out/'diagnostic.json',dict(status='SOURCE_SEALED_DEVICE_QUEUE_RELOAD_PASS',new_prediction=False,
        exact_reproduction_of_previous_local_prediction=True,new_global_prediction=False,used_to_fit_model=False,
        new_target_timing_read=False,raw_trace_scanned=False,raw_hardware_counter_scanned=False,raw_training_log_prefix_scanned=False,
        source_iterations=[85,90,95,100],source_ranks=[16],parameter_fit_iterations=[85,90],parameter_rows=len(parameters),
        feature_rows=len(features),prediction_rows=len(predictions),methods=METHODS,all_fields_exact_to_original_source_prediction=True,
        source_parameter_bytes_unchanged=True,formal_topology_changed=False,source_model_manifest_sha256=ref['manifest_sha256'],
        original_source_local_seal_sha256=ref['local_prediction_seal_sha256'],reload_seal_sha256=sha(out/'source_device_queue_reload_seal.json'),
        peak_RSS_bytes=peak,analysis_seconds=elapsed,resource_gate_pass=True,within_target_seconds=elapsed<=review['resource']['target_analysis_seconds'],
        next='Source reload gate passed. Freeze separate target cached conditional-feature and prediction plan; no target cost fitting.'))

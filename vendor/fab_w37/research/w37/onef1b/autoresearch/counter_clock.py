"""Cached source counter clock sensitivity, not clock calibration or model fit."""
import json
import resource
from time import perf_counter
import numpy as np
import pandas as pd
from smoke_worker import dump
from worker import csv
from cp_intake import interval_union


def containment(starts,ends,spans):
    spans=interval_union(spans)
    hit=np.zeros(len(starts),dtype=bool);full=hit.copy()
    for a,b in spans:
        hit |= (ends>a)&(starts<b)
        full |= (starts>=a)&(ends<=b)
    return hit,full


def offset_sweep(samples,spans,lo,hi):
    """Exact piecewise-constant lower-byte function over integer offset scenarios.

    Each sample contributes only for offsets putting its whole interval inside
    a window component. The minimum of this function is a conditional lower
    bound over the specified band, not proof the true clock offset is in it.
    """
    assert isinstance(lo,int) and isinstance(hi,int) and lo<=hi
    spans=interval_union(spans);events={lo:[0,0],hi+1:[0,0]};universal=[0,0];sample_ranges=[]
    for row in samples.itertuples():
        allowed=[]
        for a,b in spans:
            start=max(lo,int(a)-int(row.mt_timestamp_begin_ns))
            end=min(hi,int(b)-int(row.mt_timestamp_end_ns))
            if start<=end:allowed.append((start,end+1))
        allowed=interval_union(allowed)
        weights=[int(row.tx_delta_bytes),int(row.rx_delta_bytes)]
        for a,b in allowed:
            events.setdefault(a,[0,0]);events.setdefault(b,[0,0])
            for k in [0,1]:events[a][k]+=weights[k];events[b][k]-=weights[k]
            sample_ranges.append(dict(sample_id=row.sample_id,offset_start_ns=a,offset_end_exclusive_ns=b,
                tx_bytes=weights[0],rx_bytes=weights[1]))
        if allowed==[[lo,hi+1]]:
            for k in [0,1]:universal[k]+=weights[k]
    active=[0,0];curve=[];breaks=sorted(events)
    for index,a in enumerate(breaks[:-1]):
        for k in [0,1]:active[k]+=events[a][k]
        b=breaks[index+1]
        if a<=hi and b>a:curve.append(dict(offset_start_ns=a,offset_end_exclusive_ns=b,
            offset_start_minus_band_min_ns=a-lo,offset_end_minus_band_min_ns=b-lo,
            tx_contained_sample_lower_bytes=active[0],rx_contained_sample_lower_bytes=active[1]))
    assert curve and sum(r['offset_end_exclusive_ns']-r['offset_start_ns'] for r in curve)==hi-lo+1
    result={}
    for k,direction in enumerate(['tx','rx']):
        values=[r[f'{direction}_contained_sample_lower_bytes'] for r in curve]
        result.update({f'{direction}_minimum_lower_over_band_bytes':min(values),f'{direction}_maximum_lower_over_band_bytes':max(values),
                       f'{direction}_universal_sample_lower_bytes':universal[k]})
        assert min(values)>=universal[k]>=0
    return result,curve,sample_ranges


def device_order(samples):
    d=samples.copy();d['device_interval_overlap']=False;d['previous_device_end_ns']=pd.array([None]*len(d),dtype='Int64')
    profile=[]
    for link,g in d.groupby('link_id',sort=True):
        g=g.sort_values('raw_row_index',kind='stable')
        starts=g.mt_timestamp_begin_ns.to_numpy(dtype=np.int64);ends=g.mt_timestamp_end_ns.to_numpy(dtype=np.int64)
        inversion=int(np.count_nonzero(np.diff(starts)<0))+int(np.count_nonzero(np.diff(ends)<0))
        overlap=np.zeros(len(g),dtype=bool)
        if len(g)>1:
            overlap[1:] |= starts[1:]<np.maximum.accumulate(ends[:-1])
            overlap[:-1] |= ends[:-1]>starts[1:]
            d.loc[g.index[1:],'previous_device_end_ns']=pd.array(ends[:-1],dtype='Int64')
        d.loc[g.index,'device_interval_overlap']=overlap
        separation=starts[1:]-ends[:-1]
        offset=g.host_realtime_ns-g.mt_timestamp_end_ns
        profile.append(dict(link_id=int(link),samples=len(g),device_order_inversions=inversion,
            adjacent_device_counter_intervals=int(np.count_nonzero(separation==0)),
            positive_device_gaps=int(np.count_nonzero(separation>0)),negative_device_gaps=int(np.count_nonzero(separation<0)),
            device_overlap_samples=int(overlap.sum()),legacy_epoch_overlap_samples=int(g.epoch_same_link_overlap.sum()),
            observed_offset_min_ns=int(offset.min()),observed_offset_max_ns=int(offset.max()),
            observed_offset_range_ns=int(offset.max())-int(offset.min()),first_cached_predecessor_unknown=True))
    return d,pd.DataFrame(profile)


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only' and not plan.get('raw_trace_intake')
    samples=pd.read_csv(paths['counter_samples.csv.gz'],dtype={'tx_delta_bytes':'uint64','rx_delta_bytes':'uint64'})
    old=pd.read_csv(paths['counter_bounds.csv'])
    segments=pd.read_csv(paths['counter_gap_segments.csv'])
    physical=json.loads(paths['counter_physical_contract.json'].read_text())
    assert samples.clock_mapping.eq('legacy_host_end').all() and samples.sample_id.is_unique and len(samples)==2933
    assert physical['selected_source_iteration']==85 and physical['selected_source_rank']==16
    data,order=device_order(samples)
    assert order.device_order_inversions.eq(0).all(),'counter clock order ambiguous; no constant-offset remapping'
    data['valid_constant_offset_sample']=data.quality_valid & ~data.device_interval_overlap
    offset=data.host_realtime_ns-data.mt_timestamp_end_ns
    lo,hi=int(offset.min()),int(offset.max());width=hi-lo
    data['observed_host_end_minus_device_end_ns']=offset
    data['observed_offset_minus_min_ns']=offset-lo
    csv(out,'source85_device_counter_order.csv.gz',data)
    csv(out,'source85_device_counter_order_by_link.csv',order)
    spans_by={(key[0],key[1]):list(zip(g.start_ns,g.end_ns)) for key,g in segments.groupby(['graph_runtime_id','window_kind'])}
    rows=[];curves=[];sample_ranges=[]
    chosen=data[data.valid_constant_offset_sample]
    fixed_legacy_subset=data[data.valid_constant_offset_sample & data.valid_for_bounds]
    for r in old.itertuples():
        key=(r.graph_runtime_id,r.window_kind);spans=spans_by.get(key,[])
        hit,full=containment(data.start_ns.to_numpy(dtype=np.int64),data.end_ns.to_numpy(dtype=np.int64),spans)
        for direction in ['tx','rx']:
            lower=sum(int(x) for x in data.loc[full&data.valid_for_bounds,f'{direction}_delta_bytes'])
            cap=sum(int(x) for x in data.loc[hit&data.valid_for_bounds,f'{direction}_delta_bytes'])
            assert lower==getattr(r,f'{direction}_lower_bytes') and cap==getattr(r,f'{direction}_observed_overlap_cap_bytes')
        result,curve,ranges=offset_sweep(chosen,spans,lo,hi)
        fixed,fixed_curve,fixed_ranges=offset_sweep(fixed_legacy_subset,spans,lo,hi)
        result.update({'fixed_legacy_subset_'+k:v for k,v in fixed.items()})
        result.update(graph_runtime_id=r.graph_runtime_id,window_kind=r.window_kind,phase=r.phase,microbatch=r.microbatch,
            layer_id=r.layer_id,semantic_region=r.semantic_region,window_union_ns=r.window_union_ns,
            legacy_tx_lower_bytes=r.tx_lower_bytes,legacy_rx_lower_bytes=r.rx_lower_bytes,
            conditional_offset_band_min_ns=lo,conditional_offset_band_max_ns=hi,
            true_offset_in_band_verified=False,full_counter_coverage_claimed=False)
        if spans:
            # Margin checks protect this cached selection's query boundary, not
            # unknown raw samples or true offset/latency outside the chosen band.
            result['window_inside_old_selection_with_band_width_margin']=min(a for a,b in spans)>=physical['selected_start_ns']+width and max(b for a,b in spans)<=physical['selected_end_ns']-width
        else:result['window_inside_old_selection_with_band_width_margin']=True
        rows.append(result)
        for eligibility,collections in [('device_disjoint',[(curve,curves),(ranges,sample_ranges)]),
                                        ('fixed_legacy_subset',[(fixed_curve,curves),(fixed_ranges,sample_ranges)])]:
            for collection,target in collections:
                for item in collection:
                    item.update(graph_runtime_id=r.graph_runtime_id,window_kind=r.window_kind,sample_eligibility=eligibility);target.append(item)
    result=pd.DataFrame(rows)
    assert result.window_inside_old_selection_with_band_width_margin.all()
    csv(out,'source85_graph_counter_clock_sensitivity.csv',result)
    csv(out,'source85_graph_counter_offset_lower_curve.csv.gz',pd.DataFrame(curves))
    csv(out,'source85_counter_sample_containment_offset_ranges.csv.gz',pd.DataFrame(sample_ranges))
    summary=result.groupby(['phase','semantic_region','window_kind']).agg(windows=('graph_runtime_id','size'),
        legacy_positive_TX_lower_windows=('legacy_tx_lower_bytes',lambda v:int(v.gt(0).sum())),
        fixed_legacy_subset_positive_TX_lower_windows=('fixed_legacy_subset_tx_minimum_lower_over_band_bytes',lambda v:int(v.gt(0).sum())),
        constant_band_positive_TX_lower_windows=('tx_minimum_lower_over_band_bytes',lambda v:int(v.gt(0).sum())),
        mean_legacy_TX_lower_bytes=('legacy_tx_lower_bytes','mean'),
        mean_fixed_legacy_subset_minimum_TX_lower_over_band_bytes=('fixed_legacy_subset_tx_minimum_lower_over_band_bytes','mean'),
        mean_minimum_TX_lower_over_band_bytes=('tx_minimum_lower_over_band_bytes','mean'),
        mean_universal_sample_TX_lower_bytes=('tx_universal_sample_lower_bytes','mean')).reset_index()
    csv(out,'source85_counter_clock_sensitivity_summary.csv',summary)
    dump(out/'field_contract.json',dict(scope='OnlyT22source85cachedsamples/windows; nofit orrawread',
        offset_band=dict(min_ns=lo,max_ns=hi,width_ns=width,meaning='Observedhost_realtime-minus-mt_end range; sensitivityscenarios only, NOT calibratedclockinterval'),
        constancy='Singleconstantoffset appliedtoallretaineddeviceintervals. Actualclockdrift/querylatency andtrueoffset membership unverified.',
        lower='Exactminimum overinteger-ns offsetscenarios of sumofwhollycontainedsamplebytes; no sub-nanosecond clockclaim. Universal-samplelower separatelyretained.',
        coverage='Cachedsampleloweronly. Margincheckdoesnotprove omittedrawsamples absent; no completewindowupperbound or zero-traffic claim.',
        order='Rawfile roworder perlink audited; same-link deviceintervals andepochintervals reportedseparately. Firstcachedpredecessor unknown.',
        original='All288 legacy lowerbytes andintersecting-samplecaps exactlyreproduced. Oldepochoverlapexclusion remains inlegacycontrol.',
        ablation='Constant-offset sensitivity is run both on the fixedlegacyvalidsubset and on all device-disjoint samples; clockmapping effects and sample readmission are not conflated.',
        prohibited='No choosingoffset byGPU/targetboundaryfit; no byte-to-service conversion, graph edge orcost promotion.'))
    draw(out,order,result,lo,hi)
    gap=result[result.window_kind.eq('no_any_device_event')]
    dump(out/'diagnostic.json',dict(status='SOURCE_COUNTER_CLOCK_REVIEW_PASS',new_prediction=False,used_to_fit_model=False,new_target_timing_read=False,
        raw_trace_scanned=False,raw_hardware_counter_scanned=False,source_iterations=[85],source_ranks=[16],
        legacy_bounds_exact=True,device_counter_order_inversions=int(order.device_order_inversions.sum()),
        device_overlap_samples=int(data.device_interval_overlap.sum()),legacy_epoch_overlap_samples=int(data.epoch_same_link_overlap.sum()),
        observed_offset_band_width_ns=width,offset_band_is_calibrated=False,
        no_device_windows_with_legacy_positive_TX_lower=int(gap.legacy_tx_lower_bytes.gt(0).sum()),
        no_device_windows_with_fixed_legacy_subset_positive_band_TX_lower=int(gap.fixed_legacy_subset_tx_minimum_lower_over_band_bytes.gt(0).sum()),
        fixed_legacy_subset_samples=len(fixed_legacy_subset),device_disjoint_samples=len(chosen),
        no_device_windows_with_positive_TX_lower_across_sensitivity_band=int(gap.tx_minimum_lower_over_band_bytes.gt(0).sum()),
        sum_no_device_minimum_TX_lower_across_band_bytes=int(gap.tx_minimum_lower_over_band_bytes.sum()),
        legacy_positive_windows_losing_positive_band_lower=int((gap.legacy_tx_lower_bytes.gt(0)&gap.tx_minimum_lower_over_band_bytes.eq(0)).sum()),
        formal_topology_changed=False,independent_graph_service_cost_identified=False,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,analysis_seconds=perf_counter()-begin,
        next='Counterview supports onlyconditionaltraffic presence, not calibratedEPservice. Seek independent runtime/configuration or trainingbackend evidence before newcost transfer.'))


def draw(out,order,result,lo,hi):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,6));x=np.arange(len(order))
    axes[0].bar(x-.18,order.legacy_epoch_overlap_samples,.36,label='Legacy epoch overlap',color='#e69f00')
    axes[0].bar(x+.18,order.device_overlap_samples,.36,label='Device counter overlap',color='#0072b2')
    axes[0].set_xticks(x,order.link_id);axes[0].set_xlabel('MTLink link ID');axes[0].set_ylabel('Samples participating in overlap');axes[0].legend()
    axes[0].set_title('Device intervals and host-end mapping differ')
    gap=result[result.window_kind.eq('no_any_device_event')].reset_index(drop=True)
    axes[1].plot(range(len(gap)),gap.legacy_tx_lower_bytes/1e6,'o',label='Legacy conditional lower',markersize=4)
    axes[1].plot(range(len(gap)),gap.fixed_legacy_subset_tx_minimum_lower_over_band_bytes/1e6,'+',label='Offset-band minimum, same legacy subset',markersize=6)
    axes[1].plot(range(len(gap)),gap.tx_minimum_lower_over_band_bytes/1e6,'x',label='Offset-band minimum, all device-disjoint samples',markersize=5)
    axes[1].set_xlabel('Source85 graph window index');axes[1].set_ylabel('TX lower bytes (decimal MB)');axes[1].legend(fontsize=8)
    axes[1].set_title('No visible device event intervals; retained uncertainty')
    fig.suptitle(f'Source85/rank16: {((hi-lo)/1e6):.6f} ms observed offset spread is a sensitivity band, not a clock calibration')
    fig.text(.5,.02,'Clock sensitivity uses the same legacy subset first; readmitting device-disjoint samples is a separate view. True clock offset/drift remain unverified.\n'
        'All byte lower bounds remain conditional. No complete coverage, graph service duration, target prediction or causal edge is inferred.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(out/'source85_counter_clock_sensitivity.svg');fig.savefig(out/'source85_counter_clock_sensitivity.png',dpi=150);plt.close(fig)

"""Bounded raw MTLink interval probe. Counts are observations, never node costs."""
import json
import resource
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from smoke_worker import dump, sha
from worker import csv
from cp_intake import interval_union

REQUIRED = ['host_realtime_ns','link_id','tx_delta_bytes','rx_delta_bytes','mt_timestamp_begin_ns','mt_timestamp_end_ns']
OPTIONAL = ['host_mono_ns','gpu_id','ret','link_state','iter','mt_dt_ns']
MAX_I64 = 2**63-1


def prepare_samples(frame):
    """Keep invalid observations explicit; missing clock/status fields stay missing."""
    assert set(REQUIRED+['raw_row_index']) <= set(frame.columns)
    d = frame.copy()
    d['duplicate'] = d.duplicated([c for c in d if c != 'raw_row_index'], keep='first')
    d = d.sort_values(['link_id','host_realtime_ns','raw_row_index'], kind='stable').reset_index(drop=True)
    d['duration_ns'] = d.mt_timestamp_end_ns - d.mt_timestamp_begin_ns
    d['underflow'] = d.tx_delta_bytes.gt(MAX_I64) | d.rx_delta_bytes.gt(MAX_I64)
    # Duplicate copies do not consume the next physical counter sample.
    unique = d[~d.duplicate]
    recovery = unique.groupby('link_id', sort=False).underflow.shift(1, fill_value=False)
    d['recovery'] = False
    d.loc[unique.index, 'recovery'] = recovery
    d['invalid_duration'] = d.duration_ns.le(0)
    d['invalid_status'] = False
    if 'ret' in d: d['invalid_status'] |= d.ret.ne(0)
    if 'link_state' in d: d['invalid_status'] |= ~d.link_state.astype('string').eq('UP')
    d['quality_valid'] = ~(d.duplicate | d.underflow | d.recovery | d.invalid_duration | d.invalid_status)
    d['sample_id'] = d.raw_row_index.map(lambda i:f'source85:rank16:counter:row{int(i)}')
    mappings = {'legacy_host_end': (d.host_realtime_ns-d.duration_ns, d.host_realtime_ns)}
    if 'host_mono_ns' in d:
        offset = d.host_realtime_ns-d.host_mono_ns
        mappings['device_mono_epoch'] = (d.mt_timestamp_begin_ns+offset,d.mt_timestamp_end_ns+offset)
    rows = []
    for name,(start,end) in mappings.items():
        q=d.copy();q['clock_mapping']=name;q['start_ns']=start;q['end_ns']=end
        q['epoch_same_link_overlap']=False
        for _,g in q[q.quality_valid].groupby('link_id',sort=False):
            g=g.sort_values(['start_ns','end_ns','raw_row_index'],kind='stable')
            starts=g.start_ns.to_numpy(dtype=np.int64);ends=g.end_ns.to_numpy(dtype=np.int64)
            overlap=np.zeros(len(g),dtype=bool)
            if len(g)>1:
                overlap[1:] |= starts[1:] < np.maximum.accumulate(ends[:-1])
                overlap[:-1] |= ends[:-1] > starts[1:]
            q.loc[g.index,'epoch_same_link_overlap']=overlap
        q['valid_for_bounds']=q.quality_valid & ~q.epoch_same_link_overlap
        rows.append(q)
    return pd.concat(rows,ignore_index=True)


def complement(intervals,start,end):
    spans=interval_union((max(start,int(a)),min(end,int(b))) for a,b in intervals if int(b)>start and int(a)<end)
    result=[];cursor=int(start)
    for a,b in spans:
        if a>cursor:result.append((cursor,a))
        cursor=max(cursor,b)
    if cursor<end:result.append((cursor,int(end)))
    return result


def interval_bounds(samples,spans,links=range(14)):
    """Partial samples cannot be prorated into guaranteed bytes or busy time.

    Lower bound counts only wholly contained valid, nonoverlapping samples.
    The sum of intersecting sample bytes caps traffic from those samples only;
    it is a window upper bound only when all expected links are covered.
    """
    spans=interval_union(spans);duration=sum(b-a for a,b in spans)
    overlaps=np.zeros(len(samples),dtype=np.int64)
    starts=samples.start_ns.to_numpy(dtype=np.int64);ends=samples.end_ns.to_numpy(dtype=np.int64)
    for a,b in spans:overlaps+=np.maximum(0,np.minimum(ends,b)-np.maximum(starts,a))
    hit=samples.loc[overlaps>0].copy();hit['window_overlap_ns']=overlaps[overlaps>0]
    hit['fully_contained']=hit.window_overlap_ns.eq(hit.duration_ns)
    valid=hit[hit.valid_for_bounds]
    covered=0
    for link in links:
        g=valid[valid.link_id.eq(link)]
        for a,b in spans:
            covered+=sum(y-x for x,y in interval_union((max(a,int(r.start_ns)),min(b,int(r.end_ns)))
                for r in g.itertuples() if int(r.end_ns)>a and int(r.start_ns)<b))
    unknown=len(links)*duration-covered
    assert 0<=unknown<=len(links)*duration
    result=dict(window_union_ns=duration,observed_valid_link_ns=covered,unknown_link_ns=unknown,
        intersecting_samples=len(hit),valid_intersecting_samples=len(valid),fully_contained_valid_samples=int(valid.fully_contained.sum()),
        excluded_intersecting_samples=int((~hit.valid_for_bounds).sum()),all_expected_links_covered=unknown==0)
    for direction in ['tx','rx']:
        # Python integer sums preserve uint64 values without signed/float casts.
        lower=sum(int(x) for x in valid.loc[valid.fully_contained,f'{direction}_delta_bytes'])
        cap=sum(int(x) for x in valid[f'{direction}_delta_bytes'])
        result[f'{direction}_lower_bytes']=lower
        result[f'{direction}_observed_overlap_cap_bytes']=cap
        result[f'{direction}_complete_window_upper_bytes']=cap if unknown==0 else None
    return result,hit


def read_bounded(path,review):
    raw=review['raw_files'][0];assert str(path)==raw['path'] and path.stat().st_size==raw['size_bytes'] and sha(path)==raw['sha256']
    header=list(pd.read_csv(path,nrows=0).columns)
    assert set(REQUIRED)<=set(header),('missing_required_fields',header)
    columns=[c for c in REQUIRED+OPTIONAL if c in header]
    dtype={c:('uint64' if c in ['tx_delta_bytes','rx_delta_bytes'] else 'string' if c=='link_state' else 'int64') for c in columns}
    count=0;pcie=0;unexpected=0;frames=[]
    for chunk in pd.read_csv(path,usecols=columns,dtype=dtype,chunksize=review['resource']['chunk_rows']):
        chunk['raw_row_index']=np.arange(count,count+len(chunk),dtype=np.int64);count+=len(chunk)
        pcie+=int(chunk.link_id.eq(2**32-1).sum());unexpected+=int((~chunk.link_id.between(0,13)&~chunk.link_id.eq(2**32-1)).sum())
        frames.append(chunk[chunk.link_id.between(0,13)].copy())
    mt=pd.concat(frames,ignore_index=True)
    assert unexpected==0 and len(mt)+pcie==count
    if 'gpu_id' in mt:assert mt.gpu_id.eq(raw['gpu_id']).all()
    assert set(mt.link_id.unique())==set(range(14))
    mapped=prepare_samples(mt)
    start,end=raw['selected_start_ns'],raw['selected_end_ns']
    selected=mapped[(mapped.end_ns.gt(start)&mapped.start_ns.lt(end)) |
                    (~mapped.quality_valid & mapped.host_realtime_ns.between(start,end,inclusive='left'))].copy()
    assert len(selected)
    physical=dict(raw_path=str(path),raw_sha256=raw['sha256'],raw_size_bytes=raw['size_bytes'],columns=header,
        preserved_columns=columns,optional_fields_absent=[c for c in OPTIONAL if c not in header],
        physical_rows_parsed=count,physical_PCIe_rows_excluded=pcie,physical_MTLink_rows=len(mt),unexpected_links=unexpected,
        selected_source_iteration=85,selected_source_rank=16,selected_start_ns=start,selected_end_ns=end,
        selected_MTLink_raw_rows=int(selected.sample_id.nunique()),selected_clock_rows=len(selected),
        available_clock_mappings=sorted(selected.clock_mapping.unique()),
        field_identity='File path/manifest host and GPU0; explicitgpu_id field additionally checked if present',
        clock_limit='No calibrated offset fit. Inheritedhost-end mapping does not establish deployed timebase; mono correction available onlyif field exists.',
        memory_scope='Chunks of65536 rawrows, retainedMTLink rows fromonefile fororder/reset/overlap checks; onlysource85 samples outputandused forbounds. No otheriteration costfit.')
    return selected,physical


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='source_only'
    review=json.loads((Path(plan['resource_review_absolute'])).read_text())
    assert sha(Path(plan['resource_review_absolute']))==plan['resource_review_sha256']
    samples,physical=read_bounded(paths['source85_raw_counter.csv'],review)
    inventory=pd.read_csv(paths['counter_prior_file_inventory.csv'])
    row=inventory[inventory.sequence.eq(0)].iloc[0]
    assert physical['physical_rows_parsed']==int(row.row_count)
    assert physical['physical_PCIe_rows_excluded']==int(row.pcie_record_count)
    csv(out,'source85_MTLink_samples.csv.gz',samples)
    dump(out/'source85_counter_field_and_physical_read_contract.json',physical)
    profile=[]
    for (clock,link),g in samples.groupby(['clock_mapping','link_id']):
        pos=g[g.duration_ns.gt(0)]
        profile.append(dict(clock_mapping=clock,link_id=int(link),samples=len(g),quality_valid=int(g.quality_valid.sum()),
            bounds_valid=int(g.valid_for_bounds.sum()),duplicates=int(g.duplicate.sum()),underflow=int(g.underflow.sum()),recovery=int(g.recovery.sum()),
            nonpositive_duration=int(g.invalid_duration.sum()),epoch_overlaps=int(g.epoch_same_link_overlap.sum()),
            duration_p50_ns=float(pos.duration_ns.median()),duration_p95_ns=float(pos.duration_ns.quantile(.95)),duration_max_ns=int(pos.duration_ns.max())))
    csv(out,'source85_MTLink_sample_quality_by_link.csv',pd.DataFrame(profile))
    gpu=pd.read_csv(paths['source85_GPU.csv.gz'],dtype={c:'string' for c in ['pid','tid','correlation','external_id','stream','device']},low_memory=False)
    graphs=pd.read_csv(paths['counter_prior_graph_resolution.csv']);graphs=graphs[graphs.iteration.eq(85)]
    assert len(graphs)==96 and graphs['rank'].eq(16).all()
    windows=[];segments=[]
    for g in graphs.itertuples():
        a,b=int(g.graph_API_start_ns),int(g.first_device_sync_end_ns)
        nonpp=gpu[~gpu.family.eq('PP_candidate')]
        scopes={'graph_bracket':[(a,b)],'no_nonPP_device_event':complement(zip(nonpp.start_ns,nonpp.end_ns),a,b),
                'no_any_device_event':complement(zip(gpu.start_ns,gpu.end_ns),a,b)}
        assert sum(y-x for x,y in scopes['no_nonPP_device_event'])==g.no_nonPP_device_event_in_graph_sync_span_ns
        for kind,spans in scopes.items():
            for index,(x,y) in enumerate(spans):segments.append(dict(graph_runtime_id=g.graph_runtime_id,window_kind=kind,segment_index=index,start_ns=x,end_ns=y))
            for clock,s in samples.groupby('clock_mapping',sort=False):
                result,hit=interval_bounds(s,spans)
                result.update(graph_runtime_id=g.graph_runtime_id,iteration=85,rank=16,phase=g.phase,microbatch=int(g.microbatch),
                    layer_id=int(g.layer_id),semantic_region=g.semantic_region,window_kind=kind,clock_mapping=clock,
                    conditional_on_clock_mapping=True,EP_payload_attribution_verified=False)
                windows.append(result)
                if len(hit):
                    hit=hit[['sample_id','link_id','start_ns','end_ns','window_overlap_ns','fully_contained','valid_for_bounds']].copy()
                    hit['graph_runtime_id']=g.graph_runtime_id;hit['window_kind']=kind;hit['clock_mapping']=clock
                    # Accumulate small joins below; no repeated gzip append members.
                    result['_hits']=hit
    hits=[r.pop('_hits') for r in windows if '_hits' in r]
    table=pd.DataFrame(windows)
    for col in ['tx_complete_window_upper_bytes','rx_complete_window_upper_bytes']:table[col]=pd.array([r[col] for r in windows],dtype='Int64')
    csv(out,'source85_graph_counter_bounds.csv',table)
    csv(out,'source85_graph_device_gap_segments.csv',pd.DataFrame(segments))
    csv(out,'source85_graph_counter_sample_intersections.csv.gz',pd.concat(hits,ignore_index=True))
    summary=table.groupby(['clock_mapping','phase','semantic_region','window_kind']).agg(windows=('graph_runtime_id','size'),
        mean_window_union_ns=('window_union_ns','mean'),windows_with_positive_tx_lower=('tx_lower_bytes',lambda x:int(x.gt(0).sum())),
        mean_tx_lower_bytes=('tx_lower_bytes','mean'),mean_observed_overlap_cap_bytes=('tx_observed_overlap_cap_bytes','mean'),
        mean_unknown_link_ns=('unknown_link_ns','mean'),fully_covered_windows=('all_expected_links_covered','sum')).reset_index()
    csv(out,'source85_graph_counter_bounds_summary.csv',summary)
    dump(out/'field_contract.json',dict(scope='source85/rank16 singlefile diagnostic; no rawprofiler or targetcounter read',
        raw=physical,invalid='Duplicates, bad duration/status, uint64underflow and successor same-link recovery excluded. Epoch-overlapping samples excludedfrombounds conservatively, keptinrecords.',
        bounds='Wholly contained valid samples give conditional lower bytes. Intersecting-sample total is only an observed-sample cap; windowupper is null unlessall14links fullycovered.',
        unknown='Absent link/time coverage never equals zero traffic. Positive bytes do not prove continuouslybusy time, EPpayload attribution or graph causality.',
        nested='graph_bracket and two devicegap views overlap; bounds cannot be added. A boundarysample cancap morethanonewindow; uppercaps are not globallyadditive.',
        clock='Allbounds conditional on documentedmapping; unavailablehost_mono prevents independent devicetime calibration. No timeoffset fitted.',
        fit='No fittedcost, prediction or formalgraph modification. Source85only; source90/95/100 andtargetuncountered inthisprobe.'))
    draw(out,samples,table,float(row.sample_dt_ns_p50)/1e6)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    seconds=perf_counter()-begin
    assert peak<=review['resource']['max_peak_RSS_bytes'],'single file RSS gate failed; do not expand'
    dump(out/'source85_counter_resource_measurement.json',dict(peak_RSS_bytes=peak,analysis_seconds=seconds,raw_files=1,raw_bytes=physical['raw_size_bytes'],
        requested_chunk_rows=review['resource']['chunk_rows'],resource_gate_pass=True,target_analysis_seconds=review['resource']['target_analysis_seconds'],
        within_target_seconds=seconds<=review['resource']['target_analysis_seconds']))
    gap=table[table.window_kind.eq('no_any_device_event')]
    dump(out/'diagnostic.json',dict(status='SOURCE_COUNTER_INTERVAL_PROBE_PASS',new_prediction=False,used_to_fit_model=False,new_target_timing_read=False,
        raw_trace_scanned=False,raw_hardware_counter_scanned=True,source_iterations=[85],source_ranks=[16],raw_files=1,raw_bytes=physical['raw_size_bytes'],
        physical_counter_rows=physical['physical_rows_parsed'],selected_MTLink_raw_rows=physical['selected_MTLink_raw_rows'],
        available_clock_mappings=physical['available_clock_mappings'],graph_windows=len(graphs),bound_rows=len(table),
        no_any_device_windows_with_positive_tx_lower=int(gap.tx_lower_bytes.gt(0).sum()),
        new_causal_graph_edges=0,independent_graph_service_cost_identified=False,formal_topology_changed=False,
        resource_gate_pass=True,peak_RSS_bytes=peak,analysis_seconds=seconds,
        next='Review actualMTLink resolution/clock coverage and conditionalbounds. Do not promote counterbytestonodecost or extendrawscope withouta new evidence-backed plan.'))


def draw(out,samples,table,old_file_p50_ms):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    clocks=list(samples.clock_mapping.unique())
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    first=samples[samples.clock_mapping.eq(clocks[0])]
    stats=first[first.duration_ns.gt(0)].groupby('link_id').duration_ns.agg(['median','max'])/1e6
    axes[0].bar(stats.index,stats['median'],color='#0072b2',label='MTLink-only sample median')
    axes[0].set_xticks(range(14));axes[0].set_xlabel('MTLink link ID');axes[0].set_ylabel('Sample interval duration (ms)')
    axes[0].axhline(old_file_p50_ms,color='#d55e00',linestyle='--',label='Whole-file median incl.PCIe (T21)')
    axes[0].set_title('Actual source85 samples, separately from old file metadata');axes[0].legend(fontsize=8)
    view=table[table.clock_mapping.eq(clocks[0])&table.window_kind.eq('no_any_device_event')]
    labels=[];lower=[];caps=[]
    for semantic,g in view.groupby('semantic_region',sort=True):
        labels.append(semantic.removeprefix('ep_').removesuffix('_wall').replace('_','\n'))
        lower.append(g.tx_lower_bytes.mean()/1e6);caps.append(g.tx_observed_overlap_cap_bytes.mean()/1e6)
    x=np.arange(len(labels));axes[1].bar(x-.18,lower,.36,label='Conditional lower',color='#0072b2')
    axes[1].bar(x+.18,caps,.36,label='Intersecting-sample cap',color='#e69f00')
    axes[1].set_xticks(x,labels);axes[1].set_ylabel('Mean observed counter bytes (decimal MB)')
    axes[1].set_title('No visible device events: bound width is retained');axes[1].legend(fontsize=8)
    fig.suptitle('Source85/rank16 MTLink probe: interval observations, no graph service-cost claim')
    fig.text(.5,.015,'Bounds conditional on '+clocks[0]+'. A sample cap is not a total window upper bound when link/time coverage is missing.\n'
        'Missing rows remain unknown; sample byte increments do not establish continuous busy time, EP payload ownership or graph causality.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(out/'source85_counter_interval_probe.svg');fig.savefig(out/'source85_counter_interval_probe.png',dpi=150);plt.close(fig)

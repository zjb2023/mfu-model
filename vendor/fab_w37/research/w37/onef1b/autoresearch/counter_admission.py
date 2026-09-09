"""Source counter provenance and resolution review; no raw parse or fitted cost."""
import json
import resource
from pathlib import Path
from time import perf_counter
import pandas as pd
from smoke_worker import dump, sha
from worker import csv
from cp_intake import interval_union
from device_queue import graph_audit

ITERS = [85, 90, 95, 100]
DTYPE = {k: 'string' for k in ['pid', 'tid', 'correlation', 'external_id', 'stream', 'device']}


def diagnose(out, paths, plan):
    begin = perf_counter()
    assert plan['diagnostic_access'] == 'source_only' and not plan.get('raw_trace_intake')
    inventory = pd.read_csv(paths['source_mtlink_files.csv'], keep_default_na=False)
    topology = pd.read_csv(paths['source_counter_rank_topology.csv'])
    rank = topology[topology['rank'].eq(16)].iloc[0]
    selected = inventory[inventory.host.eq(rank.host) & inventory.gpu_id.eq(rank.gpu_id)].copy()
    assert len(selected) and selected.error.eq('').all()
    csv(out, 'source_rank16_counter_file_inventory.csv', selected)
    hardware = json.loads(paths['source_hardware_readiness.json'].read_text())
    validations = {name: json.loads(paths[f'source_{name}_validation.json'].read_text()) for name in ['cp_active', 'ep_phase']}
    assert all(v['status'] == 'PASS' for v in validations.values())
    # Archive source code as provenance, never execute its global/raw entry.
    sources = []
    for key in ['source_cp_counter_builder.py', 'source_ep_counter_builder.py', 'source_counter_domain_builder.py']:
        text = paths[key].read_text()
        (out / key).write_text(text)
        sources.append(dict(key=key, path=str(paths[key]), sha256=sha(paths[key])))
    cp = pd.read_csv(paths['source_cp_active_cells.csv'])
    ep = pd.read_csv(paths['source_ep_phase_cells.csv'])
    physical = [dict(key=key, rows=len(frame), iterations=sorted(map(int, frame.iteration.unique())),
                     ranks=int(frame['rank'].nunique()), used_for_fitting=False)
                for key, frame in [('source_cp_active_cells.csv', cp), ('source_ep_phase_cells.csv', ep)]]
    cp = cp[cp.iteration.isin(ITERS) & cp['rank'].eq(16)].copy()
    ep = ep[ep.iteration.isin(ITERS) & ep['rank'].eq(16) & ep.source.eq('deepep')].copy()
    assert len(cp) == 4 and len(ep) == 16
    assert not cp.duplicated(['iteration', 'rank']).any()
    assert not ep.duplicated(['iteration', 'rank', 'behavior']).any()
    for frame in [cp, ep]:
        assert frame.host.eq(rank.host).all() and frame.gpu_id.eq(rank.gpu_id).all()
        frame['split'] = frame.iteration.map(lambda i: 'source_fit_evidence' if i in [85, 90] else 'source_incremental_evidence')
    csv(out, 'source_rank16_CP_existing_counter_cells.csv', cp)
    csv(out, 'source_rank16_EP_existing_counter_cells.csv', ep)
    windows = pd.read_csv(paths['runtime_windows.csv.gz'], dtype=DTYPE, low_memory=False)
    scopes = []
    graphs = []
    for it in ITERS:
        gpu = pd.read_csv(paths[f'source{it}_GPU.csv.gz'], dtype=DTYPE, low_memory=False)
        runtime = pd.read_csv(paths[f'source{it}_runtime.csv.gz'], dtype=DTYPE, low_memory=False)
        assert gpu['rank'].eq(16).all() and gpu.device.astype(int).eq(rank.gpu_id).all()
        w = windows[windows.iteration.eq(it)]
        graph = graph_audit(gpu, runtime, w)
        assert len(graph) == 96 and graph.CPU_EP_containment.all()
        assert graph.first_device_sync_end_ns.notna().all()
        graph['split'] = cp.loc[cp.iteration.eq(it), 'split'].iloc[0]
        start = min(int(gpu.start_ns.min()), int(runtime.start_ns.min()))
        end = max(int(gpu.end_ns.max()), int(runtime.end_ns.max()))
        covering = selected[selected.start_ns.le(start) & selected.end_ns.ge(end)]
        assert len(covering) == 1, 'Need one complete metadata sequence; do not silently bridge file boundaries'
        file = covering.iloc[0]
        assert str(file.complete_link_universe).lower() == 'true'
        prospective = (Path(hardware['root']) / file.path).resolve()
        # stat only; content remains unadmitted raw input in this stage.
        assert prospective.is_file()
        visible_cp = gpu[gpu.process_group.eq('CONTEXT_PARALLEL_GROUP') & gpu.collective.eq('all_to_all')]
        assert visible_cp.family.eq('CP_collective').all()
        union = sum(b-a for a,b in interval_union(zip(visible_cp.start_ns, visible_cp.end_ns)))
        cell = cp[cp.iteration.eq(it)].iloc[0]
        assert len(visible_cp) == int(cell.event_count) == 208
        graph['inventory_sample_dt_p50_ns'] = float(file.sample_dt_ns_p50)
        graph['bracket_shorter_than_inventory_p50'] = graph.graph_to_first_device_sync_return_ns.lt(float(file.sample_dt_ns_p50))
        graph['raw_counter_file'] = str(prospective)
        scopes.append(dict(iteration=it, rank=16, host=rank.host, gpu_id=int(rank.gpu_id),
            split=cell['split'], observed_start_ns=start, observed_end_ns=end,
            metadata_file_start_ns=int(file.start_ns), metadata_file_end_ns=int(file.end_ns),
            raw_counter_file=str(prospective), raw_counter_file_size_bytes=prospective.stat().st_size,
            inventory_sample_dt_p50_ns=float(file.sample_dt_ns_p50), inventory_sample_dt_p95_ns=float(file.sample_dt_ns_p95),
            inventory_sample_dt_max_ns=float(file.sample_dt_ns_max),
            exact_visible_CP_events=len(visible_cp), visible_CP_union_ns=union,
            old_CP_union_ns=int(cell.kernel_union_duration_ns), old_minus_exact_CP_union_ns=int(cell.kernel_union_duration_ns)-union))
        graphs.append(graph)
    scopes = pd.DataFrame(scopes)
    graphs = pd.concat(graphs, ignore_index=True)
    csv(out, 'source_counter_runtime_scope_coverage.csv', scopes)
    csv(out, 'source_graph_counter_resolution.csv', graphs)
    summary = graphs.groupby(['split', 'phase', 'semantic_region']).agg(
        graph_calls=('graph_runtime_id', 'size'),
        mean_descriptive_bracket_ns=('graph_to_first_device_sync_return_ns', 'mean'),
        mean_no_nonPP_device_event_ns=('no_nonPP_device_event_in_graph_sync_span_ns', 'mean'),
        brackets_shorter_than_inventory_p50=('bracket_shorter_than_inventory_p50', 'sum')).reset_index()
    csv(out, 'source_graph_counter_resolution_summary.csv', summary)
    method = [
        dict(table='CP_active', granularity='iteration/rank union clipped to observed CP kernels',
             allocation='proportional sample-window overlap, active sample union above 0.05GB/s/link',
             reset='uint64 underflow and next same-link recovery sample quarantined',
             graph_gap_evidence=False, reason='No sample timeline outside CP kernel windows; active sample does not mean continuously active link'),
        dict(table='EP_phase', granularity='iteration/rank/behavior aggregate',
             allocation='Every byte of a sample intersecting EP allocated only among its overlapping EP phases',
             reset='Reject invalid duration and deltas above int64 max; this builder lacks same-link recovery quarantine',
             graph_gap_evidence=False, reason='Serial phase weighting assumes non-EP gaps carry no EP traffic; it cannot independently verify graph-gap traffic'),
        dict(table='MTLink_manifest', granularity='whole-file/entity metadata', allocation='No event attribution',
             reset='Reports counts only', graph_gap_evidence=False,
             reason='File-wide duration quantiles include PCIe rows. Typical5ms/max180ms are not MTLink-only or local resolution; exact coverage and timebase need raw sample evidence')]
    csv(out, 'source_existing_counter_method_review.csv', pd.DataFrame(method))
    dump(out/'field_contract.json', dict(source_code=sources, physical_reads=physical,
        selected_iterations=ITERS, selected_ranks=[16], source_costs_fitted=False,
        raw_counter_content_read=False, all_times='Integer ns within source clock; file-extent overlap is not timebase calibration',
        observations='Source GPU/CPU timestamps are diagnostic conditions only. Same-window bytes/duration are not independent predictor inputs.',
        inference='Existing aggregate tables cannot resolve graph-gap traffic. New raw sample review must retain interval censoring, invalid/reset/recovery and unknown intervals.',
        old_union='Old CP union comparison reported; no tolerance-based timing correction fitted', method=method))
    assert scopes.inventory_sample_dt_p50_ns.nunique() == 1
    draw(out, summary, float(scopes.inventory_sample_dt_p50_ns.iloc[0])/1e6)
    dump(out/'diagnostic.json', dict(status='SOURCE_COUNTER_ADMISSION_REVIEW_PASS', new_prediction=False, used_to_fit_model=False,
        new_target_timing_read=False, raw_trace_scanned=False, raw_counter_content_read=False,
        source_iterations=ITERS, source_ranks=[16], source_host=rank.host, source_gpu_id=int(rank.gpu_id),
        source_graph_brackets=len(graphs), brackets_shorter_than_inventory_p50=int(graphs.bracket_shorter_than_inventory_p50.sum()),
        unique_candidate_raw_files=int(scopes.raw_counter_file.nunique()),
        unique_candidate_raw_bytes=int(scopes.drop_duplicates('raw_counter_file').raw_counter_file_size_bytes.sum()),
        existing_counter_tables_sufficient_to_resolve_graph_gap=False,
        max_absolute_old_CP_union_difference_ns=int(scopes.old_minus_exact_CP_union_ns.abs().max()),
        formal_topology_changed=False, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        analysis_seconds=perf_counter()-begin, next='Register a bounded source85-only raw counter probe with exact SHA, chunking and sample-resolution/timebase checks; no model cost until independent evidence.'))


def draw(out, summary, typical_ms):
    import os
    os.environ['MPLCONFIGDIR'] = str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (split, frame) in zip(axes, summary.groupby('split', sort=True)):
        labels = [s.removeprefix('ep_').removesuffix('_wall').replace('_', '\n') for s in frame.semantic_region]
        ax.barh(range(len(frame)), frame.mean_descriptive_bracket_ns/1e6, color='#0072b2')
        ax.axvline(typical_ms, color='#d55e00', linestyle='--', label=f'All-link file median incl. PCIe: {typical_ms:.3f} ms')
        ax.set_yticks(range(len(frame)), labels)
        ax.set_xlabel('Mean GraphLaunch to same-owner DeviceSynchronize return (ms)')
        ax.set_title(split.replace('_', ' ')); ax.legend(fontsize=8)
        ax.tick_params(labelleft=True)
    fig.suptitle('Source rank16: counter resolution must be checked before interpreting graph visibility gaps')
    fig.text(.5, .02, 'Descriptive CPU bracket includes other activity; not pure graph service. File-wide quantiles include PCIe, not MTLink-only resolution.\n'
             'Existing CP/EP aggregates cannot independently locate traffic inside a sub-sample gap. No raw counters parsed or costs fitted.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(out/'source_counter_resolution_review.svg'); fig.savefig(out/'source_counter_resolution_review.png', dpi=150); plt.close(fig)

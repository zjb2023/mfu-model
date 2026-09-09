"""Apply the source-frozen gap classifier to admitted cached target observations."""
import gzip
import resource
from time import perf_counter
import pandas as pd
from device_gaps import DTYPE,ITERATIONS,analyze_iteration,save_analysis,write_contract
from smoke_worker import dump
from worker import csv


def diagnose(out,paths,plan):
    begin=perf_counter();assert plan['diagnostic_access']=='evaluator' and plan['evaluator_evidence_stages']
    source_windows=pd.read_csv(paths['runtime_windows.csv.gz'],dtype=DTYPE,low_memory=False)
    target_windows=pd.read_csv(paths['target_runtime_windows.csv.gz'],dtype=DTYPE,low_memory=False)
    target_windows=target_windows[target_windows['case'].eq('target224')]
    bycase={}
    for case,prefix,windows in [('source256','source',source_windows),('target224','target',target_windows)]:
        outputs=[]
        for iteration in ITERATIONS:
            gpu=pd.read_csv(paths[f'{prefix}{iteration}_GPU.csv.gz'],dtype=DTYPE,low_memory=False)
            runtime=pd.read_csv(paths[f'{prefix}{iteration}_runtime.csv.gz'],dtype=DTYPE,low_memory=False)
            cpu=pd.read_csv(paths[f'{prefix}{iteration}_CPU.csv.gz'],dtype=DTYPE,low_memory=False)
            outputs.append(analyze_iteration(gpu,runtime,cpu,windows[windows.iteration.eq(iteration)],case))
        bycase[case]=[pd.concat([r[i] for r in outputs],ignore_index=True) for i in range(3)]
        if case=='source256':
            for frame,key in zip(bycase[case],['source_gap_intervals.csv.gz','source_gap_segments.csv.gz','source_gap_accounting.csv']):
                opener=gzip.open if key.endswith('.gz') else open
                with opener(paths[key],'rt') as handle:assert frame.to_csv(index=False)==handle.read(),('source classifier regression',key)
    frames=[pd.concat([bycase[c][i] for c in ['source256','target224']],ignore_index=True) for i in range(3)]
    destination=out/'evaluator_only';destination.mkdir()
    save_analysis(destination,*frames);review_draw(destination);write_contract(out,plan)
    # Equal-weight source and target iteration means, despite different MB counts.
    phase=frames[2];columns=[c for c in phase if c.endswith('_ns') and c not in ['start_ns','end_ns']]
    average=phase.groupby(['case','iteration','phase'])[columns].mean().groupby(['case','phase']).mean()
    delta=average.loc['target224']-average.loc['source256']
    csv(destination,'target_minus_source_per_phase.csv',delta.reset_index())
    dump(out/'diagnostic.json',dict(status='TARGET_DEVICE_GAP_CONTEXT_POSTHOC_PASS',new_prediction=False,used_to_fit_model=False,
        new_target_timing_read=True,raw_trace_scanned=False,source_cost_updates=0,formal_topology_changed=False,
        sealed_reference=plan['sealed_reference'],source_classifier_reference=plan['source_evidence_stages'][-1],
        original_source_three_tables_text_exact=True,all_phase_visibility_and_gap_partitions_exact_ns=True,
        source_phase_count=len(bycase['source256'][2]),target_phase_count=len(bycase['target224'][2]),
        gap_count=len(frames[0]),segment_count=len(frames[1]),
        target_minus_source_per_phase_ns=delta.reset_index().to_dict('records'),
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,analysis_seconds=perf_counter()-begin,
        target_use='development_posthoc_NEVER_MODEL_FIT; mixed outputs remain evaluator-only, even filtered source rows',
        next='Inspect submission and CPU location signatures. Choose only a source-supported runtime mechanism, then independently fit85/90 and validate95/100; no arbitrary gap multiplier or PP tail uplift.'))


def review_draw(out):
    """Compact review labels and explicit bounds; source classifier stays frozen."""
    import numpy as np
    import matplotlib.pyplot as plt
    from device_gaps import SUBMISSION,POSITION
    profile=pd.read_csv(out/'FB_gap_profile.csv')
    labels=[('Source256\n85/90' if r.split=='source_fit' else ('Source256\n95/100' if r.case=='source256' else 'Target224\n85/90/95/100'))+
            ('\nBackward' if r.phase=='backward' else '\nForward') for r in profile.itertuples()]
    names={'before_next_runtime_start':'Before next runtime API','during_next_runtime_API':'During next runtime API',
           'after_next_runtime_end':'After next runtime API','tied_GPU_successors':'Tied GPU successors',
           'missing_runtime_successor':'Missing runtime match','outside_phase_successor':'Successor outside F/B',
           'no_successor':'No successor','inside_EP_wrapper':'Inside EP wrapper',
           'outside_EP_inside_checkpoint':'Outside EP, inside checkpoint','outside_checkpoint':'Outside checkpoint'}
    colors=['#e69f00','#cc79a7','#0072b2','#009e73','#999999','#666666','#222222']
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    for ax,parts,title in zip(axes,[SUBMISSION,POSITION],['CPU submission timing of next observed GPU event','CPU annotation containing the gap']):
        bottom=np.zeros(len(profile))
        for part,color in zip(parts,colors):
            values=profile[part+'_ns'].to_numpy()/1e6
            ax.bar(range(len(profile)),values,bottom=bottom,label=names[part],color=color);bottom+=values
        for i,v in enumerate(bottom):ax.text(i,v+3,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(bottom)*1.16);ax.set_xticks(range(len(profile)),labels,fontsize=8)
        ax.set_ylabel('Mean gap per F/B annotation (ms)');ax.set_title(title,fontsize=11)
        ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Rank16 observed device visibility gaps: two partitions of the SAME time')
    fig.text(.5,.02,'Source: fit 85/90, historically exposed validation 95/100. Target: development/posthoc only.\n'
             'No physical-idle or causal-blocker claim. Equal-weight iteration means; source4 vs target3 microbatches.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95))
    fig.savefig(out/'device_gap_context_review.svg');fig.savefig(out/'device_gap_context_review.png',dpi=160);plt.close(fig)

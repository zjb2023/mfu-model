"""Sealed scenario regression, mean-cost identity and explanatory figures."""
import json
from pathlib import Path
import pandas as pd
from guards import checked_stage
from smoke_worker import dump,sha
from worker import csv
from pp_semantics import stage_envelopes


def diagnose(out,paths,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    run=Path(plan['sealed_reference']['run_root']);names=plan['review_variants'];records={};tables={k:[] for k in ['iteration_results','phase_results','metrics','phase_metrics']}
    seals=[];checks=[];costs=[];spans=[]
    for name in names:
        for folder in ['models','seals','evaluations']:checked_stage(run/folder/name)
        model=run/'models'/name;seal=run/'seals'/name/'prediction_seal.json'
        for item in json.loads(seal.read_text())['files']:assert sha(model/item['path'])==item['sha256']
        seals.append(dict(variant=name,sha256=sha(seal)))
        records[name]=next(r for r in json.loads((model/'prediction/step_predictions.json').read_text()) if r['variant']==name)
        for key in tables:
            d=pd.read_csv(run/'evaluations'/name/'evaluator_only'/f'{key}.csv')
            tables[key].append(d[d.variant.eq(name)|(d.variant.eq('v685_frozen')&(name==names[0]))])
        if (model/'prediction/scenario_spans.csv').exists():
            s=pd.read_csv(model/'prediction/scenario_spans.csv');spans.append(s)
            expected=pd.read_csv(model/'prediction/expected_critical_path_summary.csv')
            for case,g in expected.groupby('case'):
                value=s[s['case'].eq(case)].mean_onef1b_ms.iloc[0]
                assert abs(g.critical_contribution_ms.sum()-value)<1e-6
            costs.append(expected)
            p=pd.read_csv(model/'prediction/phase_nodes.csv.gz');members=s[s['case'].eq('target224')].members.iloc[0].split(',')
            expected=stage_envelopes(p,{name:members});actual=pd.read_csv(run/'evaluations'/name/'evaluator_only/stage_mb_phase_results.csv')
            k=['pp_stage','phase','microbatch'];cols=['predicted_start_ms','predicted_end_ms','predicted_duration_ms']
            a=actual[actual.variant.eq(name)][k+cols].drop_duplicates().sort_values(k).reset_index(drop=True)
            b=expected[expected.variant.eq(name)][k+cols].sort_values(k).reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-8,rtol=0)
            checks.append(dict(variant=name,expected_critical_cost_conserves=True,stage_envelopes_use_member_max_before_mean=True))
        else:costs.append(pd.read_csv(model/'prediction/critical_path_summary.csv'))
        audit=json.loads((model/'input_access_audit.json').read_text())
        for item in audit['reads']:
            if 'evaluator' in '|'.join(item['roles']) or '/evaluator_only/' in item['path']:assert item['phases']==['hash_preflight']
    tables={k:pd.concat(v,ignore_index=True) for k,v in tables.items()}
    for key,data in tables.items():csv(out,'version_'+key+'.csv',data)
    csv(out,'scenario_spans.csv',pd.concat(spans,ignore_index=True));ledger=pd.concat(costs,ignore_index=True);csv(out,'critical_path_contributions.csv',ledger)
    # EP-only scenarios have identical non-EP costs and a mean EP cost equal to
    # the control up to integer nanosecond rounding. Their timing difference is
    # therefore not a hidden target correction or PP fit change.
    identity_name='mean_cost_identity.json'
    if plan.get('ep_only_variant'):
        control=names[0];isolated=plan['ep_only_variant'];identity=[]
        for case in ['source256','target224']:
            a=pd.read_csv(run/'models'/control/'prediction'/control/f'{case}_nodes.csv.gz',usecols=['node_id','kind','duration_ns']).set_index('node_id')
            s=pd.read_csv(run/'models'/isolated/'prediction/scenario_spans.csv');members=s[s['case'].eq(case)].members.iloc[0].split(',')
            child=[];edgehash=[]
            for name in members:
                folder=run/'models'/isolated/'prediction'/name
                n=pd.read_csv(folder/f'{case}_nodes.csv.gz',usecols=['node_id','duration_ns']).set_index('node_id');child.append(n)
                edgehash.append(json.loads((folder/f'{case}_contract.json').read_text())['topology_sha256'])
            merged=pd.concat(child).groupby('node_id').duration_ns.mean().reindex(a.index)
            error=(merged-a.duration_ns).abs();assert error.notna().all() and error.max()<=1
            c=json.loads((run/'models'/control/'prediction'/control/f'{case}_contract.json').read_text())
            assert edgehash==[c['topology_sha256']]*len(members)
            identity.append(dict(case=case,max_mean_node_cost_difference_ns=float(error.max()),edges_identical=True,
                mean_replay_minus_control_ms=float(s[s['case'].eq(case)].mean_onef1b_ms.iloc[0]-c['envelope']['onef1b_ms'])))
    elif plan.get('cp_readiness_variant'):
        from cp_readiness_review import cost_checks
        identity=cost_checks(out,run,plan);identity_name='CP_readiness_cost_checks.json'
    else:
        from pending_review import cost_checks
        identity=cost_checks(out,run,plan);identity_name='pending_cost_partition_checks.json'
    dump(out/identity_name,identity)
    phase=tables['phase_results'];metrics=tables['phase_metrics'];primary=metrics[metrics.split.eq('development_primary')&metrics.phase.eq('onef1b')]
    for ph in ['entry','tail','outer']:
        assert phase[phase.phase.eq(ph)].groupby('iteration').predicted_ms.nunique().eq(1).all()
    fig,axes=plt.subplots(1,2,figsize=(14,4),gridspec_kw={'width_ratios':[1,1.35]})
    for r in primary.itertuples():axes[0].barh(r.variant,r.mape_pct,color='#38897b' if r.variant=='v685_frozen' else '#d58e49')
    axes[0].set_xlabel('Target primary 1F1B MAPE (%)');axes[0].invert_yaxis()
    observed=phase[phase.variant.eq('v685_frozen')&phase.phase.eq('onef1b')].sort_values('iteration')
    axes[1].plot(observed.iteration,observed.actual_ms/1000,'ko-',label='observed target224')
    for name,g in phase[phase.phase.eq('onef1b')].groupby('variant'):axes[1].plot(g.iteration,g.predicted_ms/1000,label=name)
    axes[1].axvspan(85,100,color='#deeced',alpha=.5);axes[1].set_xlabel('Iteration;85-100 primary,60-80 posthoc');axes[1].set_ylabel('1F1B (s)');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'scenario_regression.svg',bbox_inches='tight');fig.savefig(out/'scenario_regression.png',dpi=160,bbox_inches='tight');plt.close(fig)
    links=''.join(f'<li>{name}: <a href="{run}/models/{name}/prediction/">sealed member graphs, waits and source cost keys</a></li>' for name in names)
    title=plan.get('review_title','T08：保留源迭代的共同波动')
    intro=plan.get('review_text','源85/90两个EP成本向量单独重放，再等权平均；95/100是增量验证，目标是已见开发数据。目标1F1B从15.906607%降为15.028381%，仍差于正式11.340786%。固定PP的EP场景为15.037738%。未采纳为正式升级。')
    extra='<img src="pending_cost_ledger.svg">' if plan.get('pending_bridge_variant') else ''
    if plan.get('cp_readiness_variant'):extra='<img src="CP_readiness_cost_ledger.svg">'
    (out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{max-width:1250px;margin:30px auto;font:16px system-ui;line-height:1.7}}img{{width:100%}}</style><h1>{title}</h1><p>{intro}</p><img src="scenario_regression.svg">{extra}<p>每个成员保留节点、边和路径；均值预测不伪装成一张单路径图。阶段包络先逐成员求max/min再平均，CPU共同完成不是纯网络耗时；场景最小/最大不是预测区间。</p><ul>{links}</ul><p><a href="version_iteration_results.csv">逐轮step/MFU</a> · <a href="version_phase_results.csv">逐轮阶段</a> · <a href="{identity_name}">成本与图边核验</a> · <a href="critical_path_contributions.csv">期望关键路径账</a></p><p>entry/tail/outer固定；MFU分子、部署flags、未见主机和目标运行时变化仍未验证。</p>''')
    dump(out/'diagnostic.json',dict(status='SEALED_EP_SCENARIO_REVIEW_PASS',new_prediction=False,used_to_fit_model=False,
        reviewed_seals=seals,**{identity_name.removesuffix('.json'):identity},checks=checks,other_phase_predictions_unchanged=True,raw_trace_scanned=False,
        conclusion=plan.get('review_conclusion','Joint EP costs recover most T07 aggregation regression; target transfer gap remains. Do not promote.')))

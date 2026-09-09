"""Post-seal accounting within the replaced CPU node; no physical GPU claim."""
import json
import pandas as pd
from worker import csv


def cost_checks(out,run,plan):
    control=plan['review_variants'][0];selected=[plan['pending_bridge_variant'],plan['zero_pending_variant']]
    checks=[];decomposition=[];expected=[];edge_cache={}
    for name in [control]+selected:
        root=run/'models'/name/'prediction';spans=pd.read_csv(root/'scenario_spans.csv');ledger=pd.read_csv(root/'critical_path_ledger.csv')
        for case in ['source256','target224']:
            members=spans[spans['case'].eq(case)].members.iloc[0].split(',');case_parts=[]
            for member in members:
                folder=root/member;fit=int(member.rsplit('source',1)[1]);c=json.loads((folder/f'{case}_contract.json').read_text())
                edges=pd.read_csv(folder/f'{case}_edges.csv.gz',low_memory=False)
                if name==control:edge_cache[(case,fit)]=edges;continue
                pd.testing.assert_frame_equal(edge_cache[(case,fit)],edges,check_exact=True)
                nodes=pd.read_csv(folder/f'{case}_nodes.csv.gz',usecols=['node_id','kind','cost_key','duration_ns','pp_stage','phase','microbatch'])
                assert nodes[nodes.kind.isin(['phase_exit','phase_enter'])].duration_ns.eq(0).all()
                costs=pd.read_csv(folder/f'{case}_ep_cost_bindings.csv.gz',low_memory=False)
                replaced=costs[costs.component.eq('group')&costs.forward_basis_cost_ns.notna()]
                n=nodes[nodes.kind.eq('ep_cpu_completion_after_last_entry')].merge(replaced,left_on='cost_key',right_on='parameter_key',suffixes=('','_binding'),validate='one_to_one')
                assert len(n)==len(replaced)
                assert (n.duration_ns-n.forward_basis_cost_ns-n.pending_previous_b_proxy_ns).abs().max()<=.5
                if name==plan['zero_pending_variant']:assert n.pending_previous_b_proxy_ns.eq(0).all()
                first=n.execution_layer.eq(0);assert n.loc[first,'pending_previous_b_proxy_ns'].eq(0).all()
                n['variant']=name;n['member']=member;n['case']=case
                decomposition.append(n[['variant','member','case','node_id','cost_key','pp_stage','phase','microbatch','execution_layer','duration_ns',
                    'original_group_cost_ns','forward_basis_cost_ns','pending_previous_b_proxy_ns','forward_basis_key','source_pp_lanes','pp_source_fit']])
                critical=ledger[ledger.variant.eq(member)&ledger['case'].eq(case)]
                joined=critical.merge(n[['node_id','forward_basis_cost_ns','pending_previous_b_proxy_ns']],on='node_id',how='left',validate='one_to_one')
                total=joined.forward_basis_cost_ns+joined.pending_previous_b_proxy_ns
                pending=(joined.critical_contribution_ms*joined.pending_previous_b_proxy_ns/total).fillna(0.).sum()
                forward=(joined.critical_contribution_ms*joined.forward_basis_cost_ns/total).fillna(0.).sum()
                rest=float(critical.critical_contribution_ms.sum()-pending-forward)
                assert abs(pending+forward+rest-c['envelope']['onef1b_ms'])<1e-6
                case_parts.append(dict(variant=name,member=member,case=case,forward_basis_ms=forward,pending_proxy_ms=pending,other_critical_ms=rest,
                    scope='accounting within modeled CPU nodes; no observed split timestamp or separately identified GPU cost'))
                checks.append(dict(variant=name,member=member,case=case,changed_group_nodes=len(n),edges_match_control=True,parent_phase_zero=True,
                    maximum_cost_conservation_error_ns=float((n.duration_ns-n.forward_basis_cost_ns-n.pending_previous_b_proxy_ns).abs().max()),
                    critical_cost_conserves=True,pending_zero_at_first_b_layer=True))
            expected+=case_parts
    csv(out,'pending_group_node_decomposition.csv.gz',pd.concat(decomposition,ignore_index=True))
    members=pd.DataFrame(expected);csv(out,'pending_critical_member_accounting.csv',members)
    mean=members.groupby(['variant','case'])[['forward_basis_ms','pending_proxy_ms','other_critical_ms']].mean().reset_index();csv(out,'pending_critical_expected_accounting.csv',mean)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,case in zip(axes,['source256','target224']):
        z=mean[mean['case'].eq(case)].set_index('variant')[['forward_basis_ms','pending_proxy_ms','other_critical_ms']]/1000
        z=z.rename(columns={'forward_basis_ms':'F CPU basis','pending_proxy_ms':'pending B proxy','other_critical_ms':'other critical CPU'})
        z.plot.barh(stacked=True,ax=ax,color=['#438d87','#ad7baf','#b7bcc4']);ax.set_xlabel('Expected critical CPU cost (s)');ax.set_ylabel('');ax.set_title(case)
        ax.legend(fontsize=7);ax.invert_yaxis()
    fig.suptitle('Cost accounting in candidate CPU graph; pending proxy is not identified GPU compute',fontsize=10)
    fig.tight_layout();fig.savefig(out/'pending_cost_ledger.svg',bbox_inches='tight');fig.savefig(out/'pending_cost_ledger.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return checks

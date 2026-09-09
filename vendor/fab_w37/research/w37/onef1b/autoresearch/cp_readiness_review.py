"""Post-seal checks for the isolated PP CP-anchor cost experiment."""
import json
import pandas as pd
from worker import csv


def cost_checks(out,run,plan):
    control=plan['review_variants'][0];full=plan['cp_readiness_variant'];zero=plan['zero_post_CP_variant']
    checks=[];decomposed=[];account=[];cached={};full_bindings={}
    for name in [control,full,zero]:
        root=run/'models'/name/'prediction';spans=pd.read_csv(root/'scenario_spans.csv');ledger=pd.read_csv(root/'critical_path_ledger.csv')
        for case in ['source256','target224']:
            members=spans[spans['case'].eq(case)].members.iloc[0].split(',')
            for member in members:
                folder=root/member;fit=int(member.rsplit('source',1)[1]);key=(case,fit)
                nodes=pd.read_csv(folder/f'{case}_nodes.csv.gz',usecols=['node_id','kind','direction','duration_ns'],low_memory=False,dtype={'direction':'string'}).set_index('node_id')
                edges=pd.read_csv(folder/f'{case}_edges.csv.gz',low_memory=False)
                if name==control:cached[key]=(nodes,edges);continue
                old,old_edges=cached[key];pd.testing.assert_frame_equal(edges,old_edges,check_exact=True)
                assert nodes.index.equals(old.index)
                allowed=nodes.kind.isin(['pp_sender_effective_readiness','pp_postpublication_completion'])&nodes.direction.eq('B')
                assert nodes.loc[~allowed,'duration_ns'].eq(old.loc[~allowed,'duration_ns']).all()
                assert nodes[nodes.kind.isin(['phase_exit','phase_enter'])].duration_ns.eq(0).all()
                b=pd.read_csv(folder/f'{case}_CP_ready_bindings.csv');b['node_id']=b.message_id+':sender_ready'
                b=b.merge(nodes.reset_index(),on='node_id',validate='one_to_one')
                assert len(b)==(960 if case=='source256' else 624)
                assert (b.predicted_CP_offset_ns+b.post_CP_latent_ready_ns-b.duration_ns).abs().max()<=.5
                assert b.replaces_existing_ready_cost.all() and not b.measured_target_or_validation_CP_end_used.any()
                service=nodes[nodes.kind.eq('pp_postpublication_completion')&nodes.direction.eq('B')].duration_ns
                if name==full:full_bindings[key]=(b[['message_id','predicted_CP_offset_ns']].copy(),service.copy())
                else:
                    reference,old_service=full_bindings[key]
                    pd.testing.assert_frame_equal(b[['message_id','predicted_CP_offset_ns']],reference,check_exact=True)
                    pd.testing.assert_series_equal(service,old_service,check_exact=True)
                    assert b.post_CP_latent_ready_ns.eq(0).all()
                b['variant']=name;b['member']=member;b['case']=case;decomposed.append(b)
                critical=ledger[ledger.variant.eq(member)&ledger['case'].eq(case)]
                joined=critical.merge(b[['node_id','predicted_CP_offset_ns','post_CP_latent_ready_ns','total_ready_ns']],on='node_id',how='left',validate='one_to_one')
                anchor=(joined.critical_contribution_ms*joined.predicted_CP_offset_ns/joined.total_ready_ns).fillna(0.).sum()
                remainder=(joined.critical_contribution_ms*joined.post_CP_latent_ready_ns/joined.total_ready_ns).fillna(0.).sum()
                total=critical.critical_contribution_ms.sum();other=total-anchor-remainder
                c=json.loads((folder/f'{case}_contract.json').read_text())
                assert abs(anchor+remainder+other-c['envelope']['onef1b_ms'])<1e-6
                account.append(dict(variant=name,member=member,case=case,predicted_API_to_CP_end_ms=anchor,post_CP_latent_ready_ms=remainder,other_critical_ms=other,
                    scope='cost accounting inside existing PP ready nodes; CP anchor includes gaps, not physical CP kernel sum'))
                checks.append(dict(variant=name,member=member,case=case,edges_match_control=True,non_B_PP_costs_including_EP_proxy_exact=True,
                    readiness_nodes=len(b),maximum_cost_conservation_error_ns=float((b.predicted_CP_offset_ns+b.post_CP_latent_ready_ns-b.duration_ns).abs().max()),
                    parent_phase_zero=True,critical_cost_conserves=True,ablation_preserves_CP_anchor_and_completion=name==zero,
                    measured_validation_or_target_CP_end_used=False))
    csv(out,'CP_readiness_node_decomposition.csv.gz',pd.concat(decomposed,ignore_index=True))
    member=pd.DataFrame(account);csv(out,'CP_readiness_critical_member_accounting.csv',member)
    mean=member.groupby(['variant','case'])[['predicted_API_to_CP_end_ms','post_CP_latent_ready_ms','other_critical_ms']].mean().reset_index()
    csv(out,'CP_readiness_critical_expected_accounting.csv',mean)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,case in zip(axes,['source256','target224']):
        z=mean[mean['case'].eq(case)].set_index('variant')[['predicted_API_to_CP_end_ms','post_CP_latent_ready_ms','other_critical_ms']]/1000
        z.columns=['Predicted API to CP-end span','Post-CP latent ready','Other critical CPU graph cost']
        z.plot.barh(stacked=True,ax=ax,color=['#4c9495','#ad7baf','#b7bcc4']);ax.set_xlabel('Expected critical graph cost (s)');ax.set_ylabel('');ax.set_title(case);ax.invert_yaxis();ax.legend(fontsize=7)
    fig.suptitle('Existing PP ready nodes split for accounting; CP anchor is not pure network time',fontsize=10)
    fig.tight_layout();fig.savefig(out/'CP_readiness_cost_ledger.svg',bbox_inches='tight');fig.savefig(out/'CP_readiness_cost_ledger.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return checks

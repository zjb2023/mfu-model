"""Source-only disjoint CPU EP cost samples and parameter transfer.

No historical all-iteration fitted costs, OISA service or target timing inputs.
"""
import pandas as pd
from pp_graph import MAP
from intervals import preceding_end

RANK_KEYS=['pp_stage','pp_lane','phase','microbatch','execution_layer','semantic_region']
GROUP_KEYS=['pp_stage','ep_lane_group','phase','microbatch','execution_layer','semantic_region']
PHASE_KEYS=['pp_stage','pp_lane','phase','microbatch']


def components(anchors,phases):
    rank=anchors.copy()
    rank['ep_lane_group']=rank.pp_lane//8
    keys=['iteration','ep_group','pp_stage','phase','microbatch','execution_layer','semantic_region']
    group=rank.groupby(keys).agg(first_return_ns=('completion_ns','min'),last_entry_ns=('arrival_ns','max'),group_size=('rank','nunique'),ep_lane_group=('ep_lane_group','first')).reset_index()
    assert group.group_size.eq(8).all()
    group['group_completion_ns']=group.first_return_ns-group.last_entry_ns
    assert group.group_completion_ns.ge(0).all()
    rank=rank.merge(group[keys+['first_return_ns']],on=keys,validate='many_to_one')
    rank['rank_return_tail_ns']=rank.completion_ns-rank.first_return_ns
    pk=['iteration','rank','pp_stage','pp_lane','phase','microbatch']
    rank=rank.merge(phases[pk+['observed_start_ns','observed_end_ns']],on=pk,validate='many_to_one').sort_values(pk+['arrival_ns'])
    previous=preceding_end(rank,pk).fillna(rank.observed_start_ns)
    rank['local_before_wrapper_ns']=(rank.arrival_ns-previous).astype('int64')
    finish=rank.groupby(pk).agg(last_return_ns=('completion_ns','max'),observed_end_ns=('observed_end_ns','first')).reset_index()
    finish['local_finish_ns']=finish.observed_end_ns-finish.last_return_ns
    assert rank.local_before_wrapper_ns.ge(0).all() and rank.rank_return_tail_ns.ge(0).all() and finish.local_finish_ns.ge(0).all()
    # Every CPU phase is an exact union of disjoint local gaps and wrapper walls.
    rank['wrapper_wall_ns']=rank.completion_ns-rank.arrival_ns
    totals=rank.groupby(pk).agg(local=('local_before_wrapper_ns','sum'),wrappers=('wrapper_wall_ns','sum'),start=('observed_start_ns','first'),end=('observed_end_ns','first')).reset_index()
    totals=totals.merge(finish[pk+['local_finish_ns']],on=pk,validate='one_to_one')
    assert (totals.local+totals.wrappers+totals.local_finish_ns).eq(totals.end-totals.start).all()
    return (rank[['iteration']+RANK_KEYS+['rank','ep_lane_group','layer_id','local_before_wrapper_ns','rank_return_tail_ns','wrapper_wall_ns']],
            group[['iteration']+GROUP_KEYS+['group_completion_ns']],finish[['iteration']+PHASE_KEYS+['local_finish_ns']])


class EPCosts:
    def __init__(self,rank,group,finish,fit,pp=16,mb=4,statistic='median',zero_rank_tail=False):
        self.fit=list(fit);self.pp=pp;self.mb=mb;self.zero_rank_tail=zero_rank_tail;self.last_key='';self.bindings=[]
        assert statistic in ['median','mean']
        self.local=rank[rank.iteration.isin(fit)].groupby(RANK_KEYS).local_before_wrapper_ns.agg(statistic).to_dict()
        self.tail=rank[rank.iteration.isin(fit)].groupby(RANK_KEYS).rank_return_tail_ns.agg(statistic).to_dict()
        self.group=group[group.iteration.isin(fit)].groupby(GROUP_KEYS).group_completion_ns.agg(statistic).to_dict()
        self.finish=finish[finish.iteration.isin(fit)].groupby(PHASE_KEYS).local_finish_ns.agg(statistic).to_dict()
    def mapped(self,row):
        st=MAP[int(row['pp_stage'])] if self.pp==14 else int(row['pp_stage'])
        m=int(row['microbatch']);m=3 if self.pp==14 and m==self.mb-1 else m
        return st,m
    def value(self,component,row,execution_layer=-1,semantic_region=''):
        st,m=self.mapped(row);phase=row['phase'];lane=int(row['pp_lane'])
        if component=='finish':key=(st,lane,phase,m);table=self.finish
        elif component=='group':key=(st,int(row['ep_lane_group']),phase,m,execution_layer,semantic_region);table=self.group
        else:key=(st,lane,phase,m,execution_layer,semantic_region);table=self.local if component=='local' else self.tail
        value=0. if component=='tail' and self.zero_rank_tail else float(table[key])
        self.last_key=f'source_ep_{component}:{key}'
        self.bindings.append(dict(component=component,parameter_key=self.last_key,source_fit_iterations=','.join(map(str,self.fit)),
            target_stage=int(row['pp_stage']),target_lane=lane,phase=phase,target_microbatch=int(row['microbatch']),execution_layer=execution_layer,
            semantic_region=semantic_region,source_stage=st,source_microbatch=m,value_ns=value,ablation_zeroed=component=='tail' and self.zero_rank_tail))
        return value

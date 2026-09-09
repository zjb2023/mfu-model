"""Source phase-cost transfer using code-derived scheduling roles, without new edges."""
import numpy as np
import pandas as pd
from readiness import ReadinessCosts
from pp_semantics import all_actions
from candidate import KEY,GPU,WRAP

CONTEXT=['stage_role','pp_lane','name','microbatch_role','region','previous_name']


class SemanticPhaseCosts(ReadinessCosts):
    policy='semantic_nearest'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.phase_data=self.data[self.data.kind.eq('phase')]
        self.groups={k:g for k,g in self.phase_data.groupby(CONTEXT)}
        self.phase_bindings=[]
        self.cache={}
    def phase_samples(self,a):
        if a['action_id'] in self.cache:return self.cache[a['action_id']]
        item={**a,'previous_name':self.previous_name[a['action_id']]}
        key=tuple(item[k] for k in CONTEXT)
        if key not in self.groups:raise ValueError(f'unsupported source phase context: {key}')
        rows=self.groups[key]
        if self.policy=='semantic_nearest':
            distance=(rows.pp_stage-self.mapped_stage(a)).abs()
            rows=rows[distance.eq(distance.min())]
            m=a['microbatch']
            if self.pp==14 and self.role_transfer and m==self.mb-1:m=3
            distance=(rows.microbatch-m).abs();rows=rows[distance.eq(distance.min())]
        self.cache[a['action_id']]=rows
        return rows
    def phase(self,a):
        rows=self.phase_samples(a);value=float(rows.duration_ns.median())
        self.last_key=f'{self.policy}:{a["action_id"]}'
        if not any(x['action_id']==a['action_id'] for x in self.phase_bindings):
            self.phase_bindings.append(dict(action_id=a['action_id'],target_pp=self.pp,pp_stage=a['pp_stage'],pp_lane=a['pp_lane'],phase=a['name'],
                microbatch=a['microbatch'],microbatch_role=a['microbatch_role'],region=a['region'],previous_name=self.previous_name[a['action_id']],
                source_stages=','.join(map(str,sorted(rows.pp_stage.unique()))),source_microbatches=','.join(map(str,sorted(rows.microbatch.unique()))),
                source_iterations=','.join(map(str,self.fit)),samples=len(rows),duration_ms=value/1e6,policy=self.policy))
        return value


class SemanticPoolCosts(SemanticPhaseCosts):
    policy='semantic_pool'


def cost_class(spec):
    return {'semantic_nearest':SemanticPhaseCosts,'semantic_pool':SemanticPoolCosts}[spec['method']]


def partition_prediction(gpu,wrapper,costs,pp,mb,variant):
    rows=[]
    for data,cols,view in [(gpu,GPU,'gpu_disjoint'),(wrapper,WRAP,'wrapper_disjoint')]:
        source=data[data.iteration.isin(costs.fit)]
        for a in all_actions(pp,mb):
            if a['kind']!='phase' or a['pp_lane']!=0:continue
            samples=costs.phase_samples(a)
            keys=samples[['iteration','pp_stage','pp_lane','name','microbatch']].rename(columns={'name':'phase'})
            matched=source.merge(keys,on=['iteration']+KEY,validate='one_to_one')
            assert len(matched)==len(keys)
            v=matched[cols].median();total=costs.phase(a)/1e6
            for col in cols:rows.append(dict(variant=variant,case='source256' if pp==16 else 'target224',pp_stage=a['pp_stage'],pp_lane=0,
                phase=a['name'],microbatch=a['microbatch'],view=view,component=col,predicted_ms=float(v[col]/v.sum()*total),
                scope='source lane0 proportions from matching schedule contexts, normalized to phase wall; views are not additive'))
    return rows


def source_evidence(aligned):
    """Leave one INTERIOR stage out; fit 85/90, score its 95/100 observations.

    All methods are evaluated on the same supported contexts. Unique endpoint
    roles are not fabricated by fitting interior costs. Unsupported roles are
    recorded as a coverage limitation rather than silently falling back.
    """
    fit=aligned[aligned.iteration.isin([85,90])&aligned.kind.eq('phase')]
    val=aligned[aligned.iteration.isin([95,100])&aligned.kind.eq('phase')]
    rows=[];coverage=[]
    for stage in range(1,15):
        train=fit[fit.stage_role.eq('interior')&~fit.pp_stage.eq(stage)]
        groups={k:g for k,g in train.groupby(CONTEXT)}
        indexed={k:g for k,g in train.groupby(['pp_lane','name','microbatch'])}
        for a in val[val.pp_stage.eq(stage)].itertuples():
            key=tuple(getattr(a,k) for k in CONTEXT)
            if key not in groups:
                coverage.append(dict(held_stage=stage,iteration=a.iteration,action_id=a.action_id,status='NO_SOURCE_SCHEDULE_CONTEXT'));continue
            sem=groups[key];d=(sem.pp_stage-stage).abs();nearest=sem[d.eq(d.min())]
            d=(nearest.microbatch-a.microbatch).abs();nearest=nearest[d.eq(d.min())]
            idx=indexed[(a.pp_lane,a.name,a.microbatch)];d=(idx.pp_stage-stage).abs();idx=idx[d.eq(d.min())]
            for method,g in [('indexed_neighbor',idx),('semantic_nearest',nearest),('semantic_pool',sem)]:
                prediction=float(g.duration_ns.median())/1e6;actual=a.duration_ns/1e6
                rows.append(dict(held_stage=stage,iteration=a.iteration,phase=a.name,pp_lane=a.pp_lane,microbatch=a.microbatch,region=a.region,
                    method=method,predicted_ms=prediction,actual_ms=actual,error_ms=prediction-actual,ape_pct=100*abs(prediction-actual)/actual,
                    source_samples=len(g),source_fit_iterations='85,90',split='95/100 temporal and held-stage incremental; historically seen'))
    return pd.DataFrame(rows),pd.DataFrame(coverage)

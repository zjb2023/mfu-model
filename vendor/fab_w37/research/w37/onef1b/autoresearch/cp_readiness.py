"""Source-predicted CP tail plus a separately fitted latent PP-ready remainder."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pp_semantics import stage_role,mb_role
from readiness import ReadinessCosts

CPKEY=['stage_role','pp_lane','microbatch_role']


def source_cp_tails(cp,pairs):
    events=cp[cp.phase.eq('backward')&cp.pair_step_complete]
    last=events.groupby(['iteration','rank','microbatch']).end_ns.max().rename('last_CP_end_ns').reset_index()
    # The left join has missing pair-steps. Nullable integers prevent nanosecond
    # epoch rounding in those rows before the coverage exclusions are applied.
    last['last_CP_end_ns']=last.last_CP_end_ns.astype('Int64')
    b=pairs[pairs.direction.eq('B')].copy();b['rank']=b.sender_stage*16+b.pp_lane
    b=b.merge(last,on=['iteration','rank','microbatch'],how='left',validate='one_to_one')
    b['CP_pair_covered']=b.last_CP_end_ns.notna()
    b['CP_end_offset_ns']=b.last_CP_end_ns-b.sender_post_ns
    valid=b[b.CP_pair_covered].copy()
    valid['CP_end_offset_ns']=valid.CP_end_offset_ns.astype('int64')
    assert valid.CP_end_offset_ns.ge(0).all()
    valid['stage_role']=[stage_role(int(s),16) for s in valid.sender_stage]
    valid['microbatch_role']=[mb_role(int(m),4) for m in valid.microbatch]
    return valid,b[~b.CP_pair_covered].drop(columns=['last_CP_end_ns','CP_end_offset_ns'])


class CPAnchorModel:
    """Fit only source iterations; prediction never consumes an observed CP end."""
    def __init__(self,tails,fit,initial=None):
        self.fit=list(fit);self.data=tails[tails.iteration.isin(fit)].copy()
        assert len(self.data) and self.data.iteration.isin(fit).all()
        grouped=self.data.groupby(CPKEY).CP_end_offset_ns
        self.offsets=grouped.median().to_dict()
        basis=grouped.agg(['median','size','min','max']).reset_index()
        basis['fit_iterations']=','.join(map(str,fit));basis['cost_domain']='source API entry to last GPU CP end; includes elapsed gaps, not CP kernel sum'
        self.basis=basis.rename(columns={'median':'predicted_CP_offset_ns','size':'source_samples','min':'observed_min_ns','max':'observed_max_ns'})
        self.parameters={};self.rows=[]
        simple=self.data[self.data.both_endpoints_single_message]
        for lane,g in simple.groupby('pp_lane'):
            anchor=np.array([self.offsets[tuple(v)] for v in g[CPKEY].itertuples(index=False,name=None)])/1e6
            x=(g.receiver_post_ns-g.sender_post_ns).to_numpy()/1e6
            y=g.post_publication_upper_bound_ns.to_numpy()/1e6
            old=(initial or {}).get(('B',int(lane)),(float(anchor.mean()+12)*1e6,5e6))
            start=[max(.000001,old[0]/1e6-float(anchor.mean())),max(.000001,old[1]/1e6)]
            fun=lambda z:np.maximum(anchor+z[0],x)-np.maximum(0,x)+z[1]-y
            fitted=least_squares(fun,start,bounds=([0,0],[np.inf,np.inf]),loss='soft_l1',f_scale=1.,
                xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=1000)
            assert fitted.success,fitted.message
            remainder,completion=map(float,fitted.x);self.parameters[int(lane)]=(remainder*1e6,completion*1e6)
            self.rows.append(dict(direction='B',pp_lane=int(lane),post_CP_latent_ready_ms=remainder,post_ready_completion_ms=completion,
                mean_fitted_CP_offset_ms=float(anchor.mean()),mean_total_ready_ms=float(anchor.mean()+remainder),
                fit_iterations=','.join(map(str,fit)),samples=len(g),fit_mae_ms=float(np.abs(fun(fitted.x)).mean()),
                receiver_before_predicted_CP_end_samples=int((x<anchor).sum()),receiver_after_total_ready_samples=int((x>anchor+remainder).sum()),
                interpretation='CP anchor is predicted from source context; remainder is latent, not identified pure GPU computation'))
    def offset(self,m,pp=16,mb=4):
        key=(stage_role(int(m['sender_stage']),pp),int(m['pp_lane']),mb_role(int(m['microbatch']),mb))
        return float(self.offsets[key]),key
    def delay(self,m,pp=16,mb=4,zero_remainder=False):
        offset,key=self.offset(m,pp,mb);remainder=self.parameters[int(m['pp_lane'])][0]
        return offset+(0. if zero_remainder else remainder)


class CPReadinessCosts(ReadinessCosts):
    def __init__(self,*args,cp_tails,zero_post_CP_remainder=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.ep_proxy_ready_parameters=dict(self.ready_parameters)
        self.baseline_parameter_rows=[dict(r) for r in self.parameter_rows]
        self.cp_model=CPAnchorModel(cp_tails,self.fit,initial=self.ready_parameters)
        self.zero_post_CP_remainder=zero_post_CP_remainder;self.cp_bindings=[]
        self.cp_parameter_rows=self.cp_model.rows
        for row in self.cp_parameter_rows:
            lane=row['pp_lane'];remainder,completion=self.cp_model.parameters[lane]
            mean_anchor=row['mean_fitted_CP_offset_ms']*1e6
            self.ready_parameters[('B',lane)]=(mean_anchor+(0. if zero_post_CP_remainder else remainder),completion)
        self.parameter_rows=[r for r in self.parameter_rows if r['direction']=='F']+[
            dict(direction='B',pp_lane=r['pp_lane'],sender_effective_ready_ms=self.ready_parameters[('B',r['pp_lane'])][0]/1e6,
                post_ready_completion_ms=r['post_ready_completion_ms'],fit_iterations=r['fit_iterations'],samples=r['samples'],
                fit_mae_ms=r['fit_mae_ms'],interpretation='mean context-ready for description only; each graph message uses its CP context key',
                post_CP_remainder_zeroed=zero_post_CP_remainder) for r in self.cp_parameter_rows]
    def sender_ready(self,m):
        if m['direction']!='B':return super().sender_ready(m)
        offset,key=self.cp_model.offset(m,self.pp,self.mb);raw=self.cp_model.parameters[m['pp_lane']][0]
        remainder=0. if self.zero_post_CP_remainder else raw
        self.last_key='source_CP_context_plus_post_CP_ready:'+str(key)
        self.cp_bindings.append(dict(message_id=m['message_id'],sender_stage=m['sender_stage'],pp_lane=m['pp_lane'],microbatch=m['microbatch'],
            predicted_CP_offset_ns=offset,post_CP_latent_ready_ns=remainder,unablated_post_CP_latent_ready_ns=raw,total_ready_ns=offset+remainder,
            source_CP_key=str(key),fit_iterations=','.join(map(str,self.fit)),post_CP_remainder_zeroed=self.zero_post_CP_remainder,
            replaces_existing_ready_cost=True,measured_target_or_validation_CP_end_used=False))
        return offset+remainder
    def service(self,m):
        if m['direction']!='B':return super().service(m)
        self.last_key='source_CP_anchored_post_ready_completion:'+str(m['pp_lane'])
        return self.cp_model.parameters[m['pp_lane']][1]
    def local_validation(self,pairs):
        d=super().local_validation(pairs)
        x=d.receiver_minus_sender_ms.to_numpy();delays=[];services=[];baseline=[];anchors=[];remainders=[]
        for r in d.to_dict('records'):
            direction,lane=r['direction'],r['pp_lane'];old=self.ep_proxy_ready_parameters[(direction,lane)]
            baseline.append(max(old[0]/1e6,r['receiver_minus_sender_ms'])-max(0.,r['receiver_minus_sender_ms'])+old[1]/1e6)
            if direction=='B':
                anchor,_=self.cp_model.offset(r,16,4);rem=self.cp_model.parameters[lane][0]
                if self.zero_post_CP_remainder:rem=0.
                delays.append((anchor+rem)/1e6);services.append(self.cp_model.parameters[lane][1]/1e6);anchors.append(anchor/1e6);remainders.append(rem/1e6)
            else:delays.append(old[0]/1e6);services.append(old[1]/1e6);anchors.append(np.nan);remainders.append(np.nan)
        d['predicted_postpublication_ms']=np.maximum(delays,x)-np.maximum(0,x)+np.array(services)
        d['error_ms']=d.predicted_postpublication_ms-d.actual_postpublication_ms
        d['baseline_readiness_predicted_postpublication_ms']=baseline
        d['baseline_readiness_error_ms']=d.baseline_readiness_predicted_postpublication_ms-d.actual_postpublication_ms
        d['predicted_CP_offset_ms']=anchors;d['post_CP_latent_ready_ms']=remainders
        d['prediction_scope']='local source validation with observed API entries only; CP end is predicted from fit-only source context, not observed validation CP'
        return d


def read_cp_tails(paths,pairs):
    cp=pd.read_csv(paths['cp_source_cp_rank_events.csv'],usecols=['iteration','rank','phase','microbatch','pair_step_complete','end_ns'])
    return source_cp_tails(cp,pairs)


def diagnose(out,paths,plan):
    from time import perf_counter
    from smoke_worker import dump
    from worker import csv
    from pp_semantics import align_observations
    from ep_completion_audit import split
    assert plan['diagnostic_access']=='source_only';start=perf_counter()
    phase=pd.read_csv(paths['source_pp_trace_events_60_100.csv']);api=pd.read_csv(paths['source_pp_api_events_60_100.csv'])
    aligned,pairs=align_observations(phase,api);tails,missing=read_cp_tails(paths,pairs)
    csv(out,'source_CP_tail_missing_pair_steps.csv',missing)
    metrics=[];parameters=[];basis=[]
    for fit in [[85,90],[85],[90]]:
        label='source'+('_'.join(map(str,fit)))
        cost=CPReadinessCosts(aligned,pairs,fit,cp_tails=tails)
        pred=cost.local_validation(pairs);pred['model_fit']=label
        # Join observed CP only after predictions; this output is a local
        # evaluation view and never feeds back into the frozen model object.
        pred=pred.merge(tails[['iteration','message_id','CP_end_offset_ns']],on=['iteration','message_id'],how='left',validate='one_to_one')
        pred['CP_offset_error_ms']=pred.predicted_CP_offset_ms-pred.CP_end_offset_ns/1e6
        pred['split']=np.where(pred.iteration.isin(fit),'source_fit',np.where(pred.iteration.isin([95,100]),'source_incremental_validation',
            np.where(pred.iteration.isin([85,90]),'source_other_fit_scenario_diagnostic','historical_diagnostic')))
        csv(out,label+'/source_local_prediction.csv.gz',pred)
        for (sp,direction,single),g in pred.groupby(['split','direction','both_endpoints_single_message']):
            metrics.append(dict(model_fit=label,split=sp,direction=direction,both_single=bool(single),samples=len(g),
                baseline_PP_mae_ms=float(g.baseline_readiness_error_ms.abs().mean()),CP_anchored_PP_mae_ms=float(g.error_ms.abs().mean()),CP_anchored_PP_bias_ms=float(g.error_ms.mean()),
                CP_offset_covered_samples=int(g.CP_end_offset_ns.notna().sum()),CP_offset_mae_ms=float(g.CP_offset_error_ms.abs().mean()) if g.CP_end_offset_ns.notna().any() else None))
        parameters.extend([dict(model_fit=label,**r) for r in cost.cp_parameter_rows])
        basis.append(cost.cp_model.basis.assign(model_fit=label))
    metrics=pd.DataFrame(metrics);csv(out,'source_CP_readiness_local_metrics.csv',metrics)
    csv(out,'source_CP_readiness_parameters.csv',pd.DataFrame(parameters));csv(out,'source_CP_offset_parameters.csv',pd.concat(basis,ignore_index=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    v=metrics[metrics.split.eq('source_incremental_validation')&metrics.direction.eq('B')&metrics.both_single]
    fig,ax=plt.subplots(figsize=(8,4));v.set_index('model_fit')[['baseline_PP_mae_ms','CP_anchored_PP_mae_ms']].plot.bar(ax=ax)
    ax.set_ylabel('Source95/100 single-message PP local MAE (ms)');ax.set_xlabel('Source fit iterations');ax.tick_params(axis='x',labelrotation=0)
    ax.legend(['Latent PP readiness baseline','Predicted CP anchor + latent remainder'],fontsize=8)
    fig.tight_layout();fig.savefig(out/'source_CP_readiness_validation.svg',bbox_inches='tight');fig.savefig(out/'source_CP_readiness_validation.png',dpi=150,bbox_inches='tight');plt.close(fig)
    dump(out/'diagnostic.json',dict(status='SOURCE_CP_READINESS_LOCAL_EVALUATION_COMPLETE',new_prediction=True,new_target_timing_read=False,
        prediction_scope='source local only, observed API entries; source CP end predicted from fit context; no target free-running model yet',used_to_fit_model=False,
        prediction_parameters_fitted_here=True,output_diagnostic_tables_used_as_cost_input=False,
        source_fit=[85,90],incremental_validation=[95,100],historical_diagnostic=[60,65,70,75,80],CP_pair_coverage=len(tails),missing_CP_pair_steps=len(missing),
        cost_model='ready = median source CP tail by stage role/lane/MB role + nonnegative per-lane latent remainder; completion=max(ready,receiver entry)+fitted post-ready duration',
        EP_pending_proxy='preserve baseline readiness parameters for EP pending proxy; isolate PP cost change',raw_trace_scanned=False,analysis_seconds=perf_counter()-start,
        next='Inspect pure source local errors and parameter evidence before graph prediction; freeze the same CP model and remainder ablation, keep formal topology and EP proxy unchanged.'))

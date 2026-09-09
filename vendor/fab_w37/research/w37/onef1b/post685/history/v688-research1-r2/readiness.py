"""Source-only identifiable PP release model; latent readiness is not pure GPU work."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pp_graph import FittedCosts


class ReadinessCosts(FittedCosts):
    def __init__(self,aligned,pairs,fit,pp=16,mb=4,role_transfer=True,**kwargs):
        super().__init__(aligned,pairs,fit,pp,mb,role_transfer,**kwargs)
        self.ready_parameters={};self.parameter_rows=[]
        # Only single-message endpoints identify a completion upper bound without
        # the other message in a fused API obscuring the matching completion.
        simple=pairs[pairs.iteration.isin(fit)&pairs.both_endpoints_single_message]
        for key,g in simple.groupby(['direction','pp_lane']):
            x=(g.receiver_post_ns-g.sender_post_ns).to_numpy()/1e6
            y=g.post_publication_upper_bound_ns.to_numpy()/1e6
            c=max(.000001,float(np.quantile(y,.05)))
            d=max(.000001,float(np.median(y[x<0]))-c)
            fun=lambda z:np.maximum(z[0],x)-np.maximum(0,x)+z[1]-y
            fitresult=least_squares(fun,[d,c],bounds=([0,0],[np.inf,np.inf]),loss='soft_l1',f_scale=1.,
                                    xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=1000)
            assert fitresult.success,fitresult.message
            delay,completion=map(float,fitresult.x)
            self.ready_parameters[key]=(delay*1e6,completion*1e6)
            self.parameter_rows.append(dict(direction=key[0],pp_lane=int(key[1]),sender_effective_ready_ms=delay,post_ready_completion_ms=completion,
                fit_iterations=','.join(map(str,fit)),samples=len(g),fit_mae_ms=float(np.abs(fun(fitresult.x)).mean()),
                receiver_first_samples=int((x<0).sum()),sender_first_samples=int((x>0).sum()),
                interpretation='API-entry-to-effective-send-ready latent delay; CPU vs GPU vs transport internals unidentifiable'))
    def sender_ready(self,m):
        key=(m['direction'],m['pp_lane']);self.last_key=f'source_ready_delay:{key}';return self.ready_parameters[key][0]
    def service(self,m):
        key=(m['direction'],m['pp_lane']);self.last_key=f'source_post_ready_completion:{key}';return self.ready_parameters[key][1]

    def local_validation(self,pairs):
        d=pairs.copy()
        d['receiver_minus_sender_ms']=(d.receiver_post_ns-d.sender_post_ns)/1e6
        d['actual_postpublication_ms']=d.post_publication_upper_bound_ns/1e6
        param=[self.ready_parameters[(r.direction,r.pp_lane)] for r in d.itertuples()]
        delay=np.array([x[0]/1e6 for x in param]);service=np.array([x[1]/1e6 for x in param]);x=d.receiver_minus_sender_ms.to_numpy()
        d['predicted_postpublication_ms']=np.maximum(delay,x)-np.maximum(0,x)+service
        d['v686_predicted_postpublication_ms']=[self.ppcost[(r.direction,r.pp_lane)]/1e6 for r in d.itertuples()]
        d['error_ms']=d.predicted_postpublication_ms-d.actual_postpublication_ms
        d['v686_error_ms']=d.v686_predicted_postpublication_ms-d.actual_postpublication_ms
        d['split']=np.where(d.iteration.isin(self.fit),'source_fit',np.where(d.iteration.isin([85,90,95,100]),'source_incremental_validation','historical_diagnostic'))
        d['prediction_scope']='local validation with OBSERVED API entry inputs; free-running schedule scored separately'
        return d

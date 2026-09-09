"""Independent v686 research DAG: phase work, local gaps, PP publication and API return."""
import pandas as pd
from pp_semantics import all_actions,message_pairs,replay,topology_sha,stage_role,mb_role

MAP=[0,1,2,3,4,5,6,7,8,9,10,11,12,15]


def build(pp,mb,costs,lanes=16,variant='full',phase_override=None):
    actions=all_actions(pp,mb,lanes);messages=message_pairs(actions)
    nodes=[];edges=[];previous={};phase_nodes={};lookup={a['action_id']:a for a in actions}
    def node(name,kind,ns,action=None,**extra):
        assert ns>=0,(name,ns)
        a=action or {}
        nodes.append({'node_id':name,'kind':kind,'duration_ns':int(round(ns)),
            'rank':a.get('rank',-1),'pp_stage':a.get('pp_stage',-1),'pp_lane':a.get('pp_lane',-1),
            'phase':a.get('name','') if a.get('kind')=='phase' else 'API','microbatch':a.get('microbatch',-1),
            'region':a.get('region','global'),'action_id':a.get('action_id',''),'api_name':a.get('name','') if a.get('kind')=='api' else '',
            **extra})
    def edge(src,dst,kind):
        edges.append({'src':src,'dst':dst,'edge_type':kind,'dependency_source':'pinned_schedules.py_non_interleaved_and_blocking_pp_api'})
    node('iteration:start','iteration_boundary',0)
    for a in actions:
        r=a['rank'];aid=a['action_id']
        if r not in previous:
            launch=f'r{r}:initial_api_ready';node(launch,'initial_api_readiness',costs.initial(a),a)
            edge('iteration:start',launch,'rank_initial_readiness');previous[r]=launch
        gap='gap:'+aid;ns=0 if variant=='zero_runtime' else costs.gap(a)
        node(gap,'local_prelaunch',ns,a,cost_key=costs.last_key);edge(previous[r],gap,'local_program_order')
        if a['kind']=='phase':
            ns=costs.phase(a)
            if phase_override is not None:ns=phase_override(a,ns)
            node(aid,'phase_compute_communication_runtime',ns,a,cost_key=costs.last_key)
            edge(gap,aid,'phase_launch')
            if a['name']=='backward':edge(phase_nodes[(r,'forward',a['microbatch'])],gap,'autograd_saved_activation')
            phase_nodes[(r,a['name'],a['microbatch'])]=aid;previous[r]=aid
        else:
            node(aid+':post','api_publication',0,a);edge(gap,aid+':post','api_publish_after_local_work')
            ns=0 if variant=='zero_runtime' else costs.post(a)
            node(aid+':return','api_return',ns,a,cost_key=costs.last_key)
            edge(aid+':post',aid+':return','api_local_control')
            previous[r]=aid+':return'
    for m in messages:
        aid=lookup[m['sender']];ns=costs.service(m)
        node(m['message_id'],'pp_postpublication_completion',ns,aid,cost_key=costs.last_key,
             sender_stage=m['sender_stage'],receiver_stage=m['receiver_stage'],direction=m['direction'])
        sender=m['sender']+':post'
        if hasattr(costs,'sender_ready'):
            ready=m['message_id']+':sender_ready'
            delay=costs.sender_ready(m) if variant!='zero_sender_ready' else 0
            node(ready,'pp_sender_effective_readiness',delay,aid,cost_key=costs.last_key,
                 sender_stage=m['sender_stage'],receiver_stage=m['receiver_stage'],direction=m['direction'])
            edge(sender,ready,'sender_api_entry_to_effective_readiness');sender=ready
        edge(sender,m['message_id'],'pp_sender_publication')
        if variant!='sender_only':edge(m['receiver']+':post',m['message_id'],'pp_receiver_publication')
        edge(m['message_id'],m['sender']+':return','blocking_send_completion')
        edge(m['message_id'],m['receiver']+':return','blocking_recv_completion')
    node('iteration:end','iteration_boundary',0)
    for last in previous.values():edge(last,'iteration:end','all_ranks_program_complete')
    frame=replay(nodes,edges)
    return frame,pd.DataFrame(edges),topology_sha(edges)


class ObservedCosts:
    """Exact source-event reassembly, solely a diagnostic representability check."""
    def __init__(self,aligned,pairs,iteration):
        self.a=aligned[aligned.iteration.eq(iteration)].set_index('action_id')
        self.p=pairs[pairs.iteration.eq(iteration)].set_index('message_id')
        self.origin=int(self.a.start_ns.min());self.last_key='observed_source_diagnostic_NOT_prediction'
    def initial(self,a):return int(self.a.loc[a['action_id'],'start_ns'])-self.origin
    def gap(self,a):return int(self.a.loc[a['action_id'],'local_prelaunch_gap_ns'])
    def phase(self,a):return int(self.a.loc[a['action_id'],'duration_ns'])
    def post(self,a):return int(self.a.loc[a['action_id'],'api_postjoin_return_ns'])
    def service(self,m):return int(self.p.loc[m['message_id'],'post_publication_upper_bound_ns'])


class FittedCosts:
    def __init__(self,aligned,pairs,fit,pp=16,mb=4,role_transfer=True,statistic='median'):
        self.fit=fit;self.pp=pp;self.mb=mb;self.role_transfer=role_transfer;self.last_key='';self.bindings=[]
        self.data=aligned[aligned.iteration.isin(fit)].copy()
        self.source_phase=self.data[self.data.kind.eq('phase')].groupby(['pp_stage','pp_lane','name','microbatch']).duration_ns.agg(statistic).to_dict()
        simple=pairs[pairs.iteration.isin(fit)&pairs.both_endpoints_single_message]
        self.ppcost=simple.groupby(['direction','pp_lane']).post_publication_upper_bound_ns.agg(statistic).to_dict()
        assert len(self.ppcost)==32
        first=self.data[self.data.sequence_index.eq(0)].copy()
        first['offset']=first.start_ns-first.groupby('iteration').start_ns.transform('min')
        self.initial_cost=first.groupby(['pp_stage','pp_lane']).offset.agg(statistic).to_dict()
        self.tables={}
        groups=[['pp_stage','pp_lane','name','previous_name','microbatch_role'],
                ['stage_role','pp_lane','name','previous_name','microbatch_role'],
                ['stage_role','name','previous_name','microbatch_role'],
                ['stage_role','name','previous_name'],['stage_role','name'],['name']]
        for component,col in [('gap','local_prelaunch_gap_ns'),('post','api_postjoin_return_ns')]:
            source=self.data if component=='gap' else self.data[self.data.kind.eq('api')]
            self.tables[component]=[(keys,source.groupby(keys)[col].agg(statistic).to_dict()) for keys in groups]
        self.previous_name={}
        for a in all_actions(pp,mb):
            a_id=a['action_id'];self.previous_name[a_id]=self.previous_name.get(a['rank'],'iteration_entry');self.previous_name[a['rank']]=a['name']
    def mapped_stage(self,a):return MAP[a['pp_stage']] if self.pp==14 else a['pp_stage']
    def initial(self,a):
        key=(self.mapped_stage(a),a['pp_lane']);self.last_key=f'initial:{key}';return self.initial_cost[key]
    def local(self,a,component):
        item={**a,'pp_stage':self.mapped_stage(a),'previous_name':self.previous_name[a['action_id']]}
        for level,(keys,table) in enumerate(self.tables[component]):
            key=tuple(item[k] for k in keys);key=key[0] if len(keys)==1 else key
            if key in table:
                self.last_key=f'{component}:level{level}:{key}'
                self.bindings.append({'action_id':a['action_id'],'component':component,'fallback_level':level,'parameter_key':self.last_key,'value_ns':table[key]})
                return table[key]
        raise KeyError(item)
    def gap(self,a):return self.local(a,'gap')
    def post(self,a):return self.local(a,'post')
    def phase(self,a):
        st=self.mapped_stage(a);m=a['microbatch']
        if self.pp==14 and self.role_transfer and m==self.mb-1:m=3
        key=(st,a['pp_lane'],a['name'],m);self.last_key=f'phase:{key}'
        return self.source_phase[key]
    def service(self,m):
        key=(m['direction'],m['pp_lane']);self.last_key=f'pp_single_pair_postpublication:{key}';return self.ppcost[key]


def envelope(nodes):
    p=nodes[nodes.kind.eq('phase_compute_communication_runtime')]
    return {'first_phase_ns':int(p.predicted_start_ns.min()),'last_phase_ns':int(p.predicted_end_ns.max()),
            'onef1b_ms':float(p.predicted_end_ns.max()-p.predicted_start_ns.min())/1e6,
            'program_ms':float(nodes.predicted_end_ns.max())/1e6}

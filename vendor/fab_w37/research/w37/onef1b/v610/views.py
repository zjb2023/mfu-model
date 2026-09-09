"""Full-iteration display from released nodes; cached observations stay unchanged."""
import json,re
from collections import defaultdict
import pandas as pd
from common import RUN,dump

def build(paths,out,release):
    nodes=pd.read_csv(RUN/'model/nodes.csv.gz',low_memory=False)
    rank_phases={};opts=defaultdict(list);services=[];release_points=defaultdict(list)
    for r in nodes.itertuples():
        rank=int(r.rank);stage=int(r.pp_stage);start=int(r.predicted_start_ns);end=int(r.predicted_end_ns)
        if rank>=0 and r.phase in ('FWD','BWD'):
            key=(rank,stage,r.phase,int(r.microbatch));old=rank_phases.get(key,(start,end));rank_phases[key]=(min(old[0],start),max(old[1],end))
        if r.kind=='optimizer' and r.op_name in ('optimizer_update','optimizer_pre_ag0','optimizer_post_ag0'):opts[stage,r.op_name].append((rank,start,end))
        if re.fullmatch(r'tail:dp_with_cp_stage\d+:rs_arrival_join::release:r\d+',r.node_id):release_points[stage].append((rank,end))
        if re.fullmatch(r'tail:dp_with_cp_stage\d+:(rs|ag0|ag1)_service',r.node_id) or re.fullmatch(r'tail:expert_dp_stage\d+_lane2:(rs|ag0|ag1)_service',r.node_id):services.append(r)
    groups=defaultdict(list)
    for (rank,stage,phase,mb),(start,end) in rank_phases.items():groups[stage,phase,mb].append((rank,start,end))
    bars=[]
    for (stage,phase,mb),items in sorted(groups.items()):
        assert len(items)==16 and len({x[0] for x in items})==16
        bars.append(dict(id=f's{stage}:{phase}{mb}',stage=stage,phase='forward' if phase=='FWD' else 'backward',microbatch=mb,start_ms=min(x[1] for x in items)/1e6,end_ms=max(x[2] for x in items)/1e6,rank_count=16,scope='同stage全部16rank的预测时间包络'))
    collectives=[]
    for r in services:
        stage=int(r.pp_stage);group='expert_dp' if ':expert_dp_' in r.node_id else 'dp';coll=r.node_id.split(':')[-1].removesuffix('_service');start=int(r.predicted_start_ns)/1e6;last=None
        if group=='dp' and coll=='rs':
            rel=release_points[stage];assert len(rel)==16 and abs(max(x[1] for x in rel)/1e6-start)<1e-6;last=start;start=min(x[1] for x in rel)/1e6
        collectives.append(dict(id=r.node_id,stage=stage,group_type=group,collective=coll,start_ms=start,end_ms=int(r.predicted_end_ns)/1e6,last_release_ms=last,scope='DP组全部16rank到达至完成' if last is not None else 'DP组服务节点' if group=='dp' else 'EDP中lane2所在组服务节点'))
    for (stage,op),items in sorted(opts.items()):
        assert len(items)==16
        collectives.append(dict(id=f'optimizer:s{stage}:{op}',stage=stage,group_type='optimizer',collective=op,start_ms=min(x[1] for x in items)/1e6,end_ms=max(x[2] for x in items)/1e6,scope='同stage全部16rank的模型参数更新区间'))
    p=release['prediction'];start=min(b['start_ms'] for b in bars);end=max(b['end_ms'] for b in bars)
    assert len(bars)==84 and len(collectives)==126
    assert abs(start-p['entry_ms'])<1e-6 and abs(end-start-p['onef1b_ms'])<1e-6
    pred=dict(id='prediction',label='A · v6.10 预测｜224卡，PP14 / MB3',stages=14,microbatches=3,bars=bars,collectives=collectives,entry_ms=start,fb_end_ms=end,profiler_ms=p['profiler_ms'],training_ms=p['training_ms'],outer_ms=p['outer_ms'],raw_graph_ms=p['raw_ms'],reconciliation_ms=p['profiler_ms']-p['raw_ms'],optimizer_observed=False)
    previous=json.loads(paths['historical_view'].read_text());trace=previous['trace']
    ledger=[];events=[]
    for i in [85,90,95,100]:
        for v in [pred,trace[str(i)]['target'],trace[str(i)]['source']]:
            ledger.append(dict(view=v['id'],iteration=i,entry_ms=v['entry_ms'],onef1b_ms=v['fb_end_ms']-v['entry_ms'],tail_ms=v['profiler_ms']-v['fb_end_ms'],profiler_ms=v['profiler_ms'],outer_ms=v['outer_ms'],training_ms=v['training_ms']))
    for v,it in [(pred,'fixed')]+[(v,i) for i,pair in trace.items() for v in pair.values()]:
        for b in v['bars']+v['collectives']:events.append(dict(view=v['id'],iteration=it,stage=b['stage'],kind=b.get('phase',b.get('group_type')),operation=b.get('collective',b.get('microbatch')),start_ms=b['start_ms'],end_ms=b['end_ms'],scope=b['scope']))
    data=dict(schema='w37-complete-iteration-view-v1',version='v6.10',default_iteration=85,iterations=[85,90,95,100],prediction=pred,trace=trace,ledger=ledger)
    dump(out/'full_iter_payload.json',data);pd.DataFrame(ledger).to_csv(out/'full_iter_ledger.csv',index=False);pd.DataFrame(events).to_csv(out/'full_iter_events.csv',index=False)
    assert data['trace']==previous['trace'] and data['prediction']!=previous['prediction']
    dump(out/'view_checks.json',dict(status='PASS',prediction_version='v6.10',trace_byte_semantics_unchanged=True,new_prediction_timing_used=True,full_rank_phase_groups=len(rank_phases),stage_phase_bars=84,raw_trace_reads=0))
    return data

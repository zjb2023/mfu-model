"""Code-derived non-interleaved PP API sequence, independent of timing observations."""
from collections import Counter, defaultdict, deque
import hashlib
import pandas as pd


def mb_role(mb,n):
    return 'only' if n==1 else ('first' if mb==0 else ('last' if mb==n-1 else 'middle'))


def stage_role(s,pp):
    return 'entry' if s==0 else ('terminal' if s==pp-1 else 'interior')


def action_sequence(stage,pp,microbatches,lane=0,lanes=16):
    """schedules.py non-interleaved training: warmup, fused steady, cooldown."""
    warm=min(pp-stage-1,microbatches);rem=microbatches-warm
    actions=[];occ=Counter()
    def add(kind,name,mb,region,**messages):
        seq=len(actions);o=occ[name];occ[name]+=1
        actions.append({'action_id':f'r{stage*lanes+lane}:a{seq}:{name}','rank':stage*lanes+lane,
            'pp_stage':stage,'pp_lane':lane,'stage_role':stage_role(stage,pp),'kind':kind,'name':name,
            'occurrence':o,'sequence_index':seq,'microbatch':mb,'microbatch_role':mb_role(mb,microbatches),
            'region':region,'messages':messages})
    for mb in range(warm):
        add('api','recv_forward',mb,'warmup',recv_f=mb)
        add('phase','forward',mb,'warmup')
        add('api','send_forward',mb,'warmup',send_f=mb)
    if rem:add('api','recv_forward',warm,'steady',recv_f=warm)
    for i in range(rem):
        add('phase','forward',warm+i,'steady')
        add('api','send_forward_recv_backward',i,'steady',send_f=warm+i,recv_b=i)
        add('phase','backward',i,'steady')
        if i==rem-1:add('api','send_backward',i,'steady',send_b=i)
        else:add('api','send_backward_recv_forward',i,'steady',send_b=i,recv_f=warm+i+1)
    for i in range(rem,microbatches):
        add('api','recv_backward',i,'cooldown',recv_b=i)
        add('phase','backward',i,'cooldown')
        add('api','send_backward',i,'cooldown',send_b=i)
    for a in actions:
        effective={}
        for direction,mb in a['messages'].items():
            if direction in ['send_f','recv_b'] and stage==pp-1:continue
            if direction in ['send_b','recv_f'] and stage==0:continue
            effective[direction]=mb
        a['messages']=effective;a['message_count']=len(effective)
    return actions


def all_actions(pp,mb,lanes=16):
    return [a for s in range(pp) for lane in range(lanes) for a in action_sequence(s,pp,mb,lane,lanes)]


def message_pairs(actions):
    pairs={}
    for a in actions:
        s,lane=a['pp_stage'],a['pp_lane']
        for role,mb in a['messages'].items():
            if role=='send_f':direction,src,dst,endpoint='F',s,s+1,'sender'
            elif role=='recv_f':direction,src,dst,endpoint='F',s-1,s,'receiver'
            elif role=='send_b':direction,src,dst,endpoint='B',s,s-1,'sender'
            else:direction,src,dst,endpoint='B',s+1,s,'receiver'
            key=f'pp:l{lane}:{direction}{mb}:s{src}_s{dst}'
            item=pairs.setdefault(key,{'message_id':key,'direction':direction,'pp_lane':lane,'sender_stage':src,'receiver_stage':dst,'microbatch':mb})
            assert endpoint not in item,(key,endpoint)
            item[endpoint]=a['action_id']
    assert all('sender' in p and 'receiver' in p for p in pairs.values())
    return list(pairs.values())


def align_observations(phases,apis,pp=16,mb=4,lanes=16):
    actions=all_actions(pp,mb,lanes);pairs=message_pairs(actions)
    phase_lookup=phases.set_index(['iteration','rank','phase','microbatch'])
    api_lookup=apis.set_index(['iteration','rank','api_name','occurrence'])
    assert phase_lookup.index.is_unique and api_lookup.index.is_unique
    rows=[]
    for it in sorted(phases.iteration.unique()):
        previous_end={};previous_name={}
        for a in actions:
            if a['kind']=='phase':r=phase_lookup.loc[(it,a['rank'],a['name'],a['microbatch'])]
            else:r=api_lookup.loc[(it,a['rank'],a['name'],a['occurrence'])]
            start,end=int(r.observed_start_ns),int(r.observed_end_ns)
            prev=previous_end.get(a['rank'])
            gap=start-prev if prev is not None else 0
            assert end>=start
            if gap<0:raise ValueError(f'phase/API overlap violates static sequence: iteration {it}, {a["action_id"]}, gap={gap}')
            rows.append({**{k:v for k,v in a.items() if k!='messages'},'iteration':int(it),
                'start_ns':start,'end_ns':end,'duration_ns':end-start,'local_prelaunch_gap_ns':gap,
                'previous_name':previous_name.get(a['rank'],'iteration_entry')})
            previous_end[a['rank']]=end;previous_name[a['rank']]=a['name']
    frame=pd.DataFrame(rows)
    assert len(frame[frame.kind.eq('api')])==len(apis)
    assert len(frame[frame.kind.eq('phase')])==len(phases)
    obs=frame.set_index(['iteration','action_id']);lookup={a['action_id']:a for a in actions}
    transfers=[]
    for it in sorted(phases.iteration.unique()):
        for p in pairs:
            send=obs.loc[(it,p['sender'])];recv=obs.loc[(it,p['receiver'])]
            release=max(int(send.start_ns),int(recv.start_ns));first=min(int(send.end_ns),int(recv.end_ns))
            # API completion is an upper bound on a message's completion when fused.
            both_single=lookup[p['sender']]['message_count']==lookup[p['receiver']]['message_count']==1
            transfers.append({**p,'iteration':int(it),'sender_post_ns':int(send.start_ns),'receiver_post_ns':int(recv.start_ns),
                'both_published_ns':release,'first_api_return_ns':first,'last_api_return_ns':max(int(send.end_ns),int(recv.end_ns)),
                'post_publication_upper_bound_ns':first-release,'both_endpoints_single_message':both_single,
                'sender_api_message_count':lookup[p['sender']]['message_count'],'receiver_api_message_count':lookup[p['receiver']]['message_count'],
                'sender_wait_for_receiver_ns':max(0,int(recv.start_ns)-int(send.start_ns)),
                'receiver_wait_for_sender_ns':max(0,int(send.start_ns)-int(recv.start_ns))})
    transfer=pd.DataFrame(transfers)
    assert transfer.post_publication_upper_bound_ns.ge(0).all()
    completions={}
    for r in transfer.itertuples():
        for action in [r.sender,r.receiver]:
            key=(r.iteration,action);completions[key]=max(completions.get(key,0),r.first_api_return_ns)
    frame['api_postjoin_return_ns']=[int(r.end_ns)-completions.get((r.iteration,r.action_id),int(r.start_ns)) if r.kind=='api' else 0 for r in frame.itertuples()]
    assert frame.api_postjoin_return_ns.ge(0).all()
    return frame,transfer


def replay(nodes,edges,completion='iteration:end'):
    """Max-plus replay with provenance-bearing edges, rejecting cyclic candidates."""
    by={r['node_id']:r for r in nodes};assert len(by)==len(nodes)
    succ=defaultdict(list);pred=defaultdict(list);ind={key:0 for key in by}
    for e in edges:
        a,b=e['src'],e['dst'];assert a in by and b in by
        succ[a].append(b);pred[b].append(a);ind[b]+=1
    ready=deque(sorted(n for n in by if ind[n]==0));ends={};starts={};parents={}
    while ready:
        n=ready.popleft();parent=max(pred[n],key=lambda p:(ends[p],p)) if pred[n] else ''
        starts[n]=ends[parent] if parent else 0;ends[n]=starts[n]+int(by[n]['duration_ns']);parents[n]=parent
        for child in succ[n]:
            ind[child]-=1
            if ind[child]==0:ready.append(child)
    if len(ends)!=len(by):raise ValueError(f'candidate graph cyclic: replayed {len(ends)} / {len(by)}')
    chain=set();n=completion
    while n:chain.add(n);n=parents[n]
    output=pd.DataFrame([{**r,'predicted_start_ns':starts[r['node_id']],'predicted_end_ns':ends[r['node_id']],
                           'critical_predecessor':parents[r['node_id']],'on_critical_path':r['node_id'] in chain} for r in nodes])
    assert int(output.loc[output.on_critical_path,'duration_ns'].sum())==ends[completion]
    return output


def topology_sha(edges):
    rows=sorted((e['src'],e['dst'],e['edge_type'],e['dependency_source']) for e in edges)
    return hashlib.sha256(('\n'.join('\x1f'.join(r) for r in rows)+'\n').encode()).hexdigest()


def stage_envelopes(predictions,scenario_members=None):
    """Average scenario stage envelopes, never envelope the averaged rank times."""
    keys=['pp_stage','phase','microbatch']
    grouped=predictions.groupby(['variant']+keys).agg(predicted_start_ms=('start_ms','min'),predicted_end_ms=('end_ms','max')).reset_index()
    for ensemble,members in (scenario_members or {}).items():
        selected=grouped[grouped.variant.isin(members)]
        assert selected.variant.nunique()==len(members)
        expected=selected.groupby(keys)[['predicted_start_ms','predicted_end_ms']].mean().reset_index();expected['variant']=ensemble
        grouped=pd.concat([grouped[~grouped.variant.eq(ensemble)],expected],ignore_index=True)
    grouped['predicted_duration_ms']=grouped.predicted_end_ms-grouped.predicted_start_ms
    return grouped

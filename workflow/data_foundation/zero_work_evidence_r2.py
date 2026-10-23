"""Explicit zero-work CPU operations; never turn unknown work into zero.

Only CPU calls with a zero tensor dimension and no associated device activity
qualify. Synthetic boundary records have event=None and carry CPU evidence.
"""
def zero_input(op):
    shapes=op.get('args',{}).get('Input Dims',[])
    return bool(shapes and isinstance(shapes[0],list) and 0 in shapes[0])
def add_backward_zero(devices,linears,events):
    out=list(devices)
    for j,(oid,op) in enumerate(linears):
        kind=f'expert_grouped_linear_backward_{j}'
        if any(d['kind']==kind for d in devices):continue
        assert zero_input(op),('missing nonempty expert GPU work',oid,op.get('args'))
        prior=[d for d in out if str(d['stream'])=='0' and all(events[i]['ts']<op['ts'] for i in d['launch_events'])]
        # With zero expert input even the pre-FC local processing can emit no
        # kernels. Its zero-cost boundary follows dispatch, not a fake GPU event.
        if not prior:prior=[d for d in out if d['kind']=='backward_dispatch_visible']
        assert prior,('no zero-work entry',oid)
        t=max(d['end_us'] for d in prior)
        out.append(dict(event=None,cpu_zero_event=oid,virtual_zero_work=True,name='EMPTY_LOCAL_INTERVAL_BEFORE_ZERO_FC',start_us=t,end_us=t,stream=0,kind='local_autograd_processing',launch_events=[oid]))
        out.append(dict(event=None,cpu_zero_event=oid,virtual_zero_work=True,name='CPU_ZERO_WORK_GroupedLinearBackward',start_us=t,end_us=t,stream=0,kind=kind,launch_events=[oid]))
    return out
def add_recompute_zero(devices,checkpoints,events):
    out=list(devices)
    for u,(_,ck) in enumerate(checkpoints):
        ops=[(i,e) for i,e in enumerate(events) if e.get('cat')=='cpu_op' and e.get('pid')==ck['pid'] and e.get('tid')==ck['tid'] and ck['ts']<=e.get('ts',0) and e['ts']+e.get('dur',0)<=ck['ts']+ck['dur']+.01]
        fc=[(i,e) for i,e in ops if e['name']=='_GroupedLinear'];assert len(fc)==2
        for oid,op in sorted(ops,key=lambda z:z[1]['ts']):
            if op['name'] not in ['_GroupedLinear','aten::silu','aten::mul','aten::copy_'] or not zero_input(op):continue
            if not fc[0][1]['ts']<=op['ts']<=fc[1][1]['ts']:continue
            owned=[d for d in devices if d['unit']==u and op['ts']<=events[d['launch_event']]['ts'] and events[d['launch_event']]['ts']+events[d['launch_event']].get('dur',0)<=op['ts']+op['dur']+.01]
            if owned:continue
            prior=[d for d in out if d['unit']==u and str(d['stream'])=='0' and events[d['launch_event']]['ts']<op['ts']]
            assert prior,('no recompute zero predecessor',oid)
            t=max(d['end_ms'] for d in prior);prev=max(prior,key=lambda d:d['end_ms'])
            out.append(dict(event=None,cpu_zero_event=oid,virtual_zero_work=True,launch_event=oid,name='CPU_ZERO_WORK_'+op['name'],start_ms=t,end_ms=t,stream='0',kind='local_compute_envelope',unit=u,phase='recompute',anchor=prev['anchor'],collective=None,group=None))
    return out
def adapt(module,text):
    if module=='build_b_four_unit_skeleton_r9':
        text=text.replace("ids={d['event'] for d in ds}","ids={('cpu_zero',d['cpu_zero_event']) if d.get('virtual_zero_work') else ('device',d['event']) for d in ds}")
        text=text.replace("if d['event'] not in seen", "if (('cpu_zero',d['cpu_zero_event']) if d.get('virtual_zero_work') else ('device',d['event'])) not in seen")
    if module=='build_b_cost_blocks_r1':
        before='        assert devices\n'
        assert text.count(before)==1
        text=text.replace(before,'        from zero_work_evidence_r2 import add_recompute_zero\n        devices=add_recompute_zero(devices,checkpoints,es)\n'+before)
    if module=='build_b_moe_backward_r6':
        before='        assert devices\n'
        assert text.count(before)==1
        text=text.replace(before,'        from zero_work_evidence_r2 import add_backward_zero\n        devices=add_backward_zero(devices,linears,es)\n'+before)
    return text

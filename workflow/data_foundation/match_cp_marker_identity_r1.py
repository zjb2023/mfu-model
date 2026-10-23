"""Strict identity fallback for profiler CPU marker/launch tail overhang.

Never edit timestamps, durations, correlations, groups, or graph dependencies.
"""
def within(e,p):
    return e.get('pid')==p.get('pid') and e.get('tid')==p.get('tid') and p['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=p['ts']+p.get('dur',0)+.01

def match_marker(events,launch,device,owner):
    markers=[(i,e) for i,e in enumerate(events) if e.get('cat')=='cpu_op' and e.get('name')=='record_param_comms' and within(launch,e)]
    mode='strict_temporal_containment'
    if len(markers)!=1:
        # Do not resolve ambiguous enclosing markers by choosing one arbitrarily.
        assert not markers,('ambiguous temporal markers',len(markers))
        ext=launch.get('args',{}).get('External id')
        assert ext is not None and device['args'].get('External id')==ext
        assert launch['args'].get('correlation')==device['args'].get('correlation')
        markers=[(i,e) for i,e in enumerate(events) if e.get('cat')=='cpu_op' and e.get('name')=='record_param_comms'
                 and e.get('pid')==launch.get('pid') and e.get('tid')==launch.get('tid')
                 and e.get('args',{}).get('External id')==ext]
        assert len(markers)==1,('nonunique exact identity',ext,len(markers))
        mode='exact_external_id_and_correlation'
    i,marker=markers[0]
    assert within(marker,owner) and within(launch,owner)
    assert marker['ts']<=launch['ts']<=marker['ts']+marker['dur'], 'launch must begin inside matched marker'
    ma,da=marker['args'],device['args']
    assert ma.get('Collective name')==da.get('Collective name')=='all_to_all'
    assert ma.get('Process Group Description')==da.get('Process Group Description')=='CONTEXT_PARALLEL_GROUP'
    for field in ['Process Group Name','Group size','In msg nelems','Out msg nelems','dtype']:
        assert str(ma[field])==str(da[field]),(field,ma[field],da[field])
    return i,marker,dict(method=mode,external_id=launch['args'].get('External id'),correlation=launch['args'].get('correlation'),launch_end_over_marker_end_us=max(0,launch['ts']+launch['dur']-marker['ts']-marker['dur']))

"""Four layers share topology, NOT durations. Same-sample conditional accounting."""
import argparse,collections,copy,graphlib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    paths=[BASE/f'pp32-f-split-r8/rank-{r}.json' for r in range(8,16)]
    xs=[json.loads(p.read_text()) for p in paths];origin=min(x['origin_absolute_ms'] for x in xs)
    np=BASE/'ep-notify-order-audit-r1/report.json';cp=BASE/'ep-combine-audit-r1/report.json'
    notify=json.loads(np.read_text());combine=json.loads(cp.read_text())
    assert abs(notify['time_origin_absolute_ms']-origin)<1e-6 and abs(combine['origin_absolute_ms']-origin)<1e-6
    local={};comms={};tails={};reorders=[];raws=[];fends={};starts={}
    for x in xs:
        r=x['rank'];off=x['origin_absolute_ms']-origin;p=Path(x['path']);blob=p.read_bytes();sha=hashlib.sha256(blob).hexdigest();assert sha==x['sha256']
        es=json.loads(blob)['traceEvents'];del blob
        f=es[x['cpu_F_event']];launch=collections.defaultdict(list)
        for i,e in enumerate(es):
            if e.get('cat') in ['privateuse1_driver','privateuse1_runtime'] and 'correlation' in e.get('args',{}):launch[e['args']['correlation']].append(e)
        dev=[]
        for i,e in enumerate(es):
            if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
            calls=launch.get(e.get('args',{}).get('correlation'),[])
            if any(v.get('pid')==f['pid'] and f['ts']<=v['ts'] and v['ts']+v.get('dur',0)<=f['ts']+f['dur']+.01 for v in calls):
                dev.append(dict(event=i,name=e['name'],start_ms=e['ts']/1000-origin,end_ms=(e['ts']+e['dur'])/1000-origin,stream=e.get('args',{}).get('stream'),launch_categories=sorted({v['cat'] for v in calls})))
        fends[r]=max(d['end_ms'] for d in dev);starts[r]=min(d['start_ms'] for d in dev)
        for l in range(4):
            for j in range(6):
                ds=[d for d in x['devices'] if d['semantic']==f'r{r}:LOCAL:{6*l+j}']
                if j==0 and l:
                    first=min(d['start_ms'] for d in ds if 'LayerNormGlobalKernel' in d['name']);ds=[d for d in ds if d['start_ms']>=first]
                local[l,r,j]=(min(d['start_ms'] for d in ds)+off,max(d['end_ms'] for d in ds)+off)
            for j in range(4):
                m=next(m for m in x['markers'] if m['kind']=='CP' and m['ordinal']==4*l+j);comms[l,r,j]=m['device_end_ms']+off
            ce=next(m for m in x['markers'] if m['kind']=='EP_COMBINE' and m['ordinal']==l)['device_end_ms']+off
            if l<3:
                ds=[d for d in x['devices'] if d['semantic']==f'r{r}:LOCAL:{6*l+6}'];stop=min(d['start_ms'] for d in ds if 'LayerNormGlobalKernel' in d['name'])+off
            else:stop=fends[r]+1e-5
            tail=[d for d in dev if d['start_ms']>=ce-1e-6 and d['start_ms']<stop-1e-6]
            tail.sort(key=lambda d:d['start_ms']);assert len(tail)>=2 and all('AddOp' in d['name'] for d in tail[:2])
            if l<3:assert len(tail)==2
            tails[l,r]=tail
            lo=comms[l,r,2];hi=local[l,r,3][0]
            ks=[d for d in dev if 'index_select' in d['name'] and d['start_ms']<hi and d['end_ms']>lo]
            assert len(ks)==2
            reorders.append(dict(layer=l,rank=r,tail_ms=hi-lo,kernels=ks))
        raws.append(dict(path=str(p),sha256=sha));print('audited rank',r,flush=True)
    nodes={};facts=[];prior={r:None for r in range(8,16)}
    def add(k,deps,start,end,kind,l,r=None):
        deps=[d for d in deps if d];ready=max([nodes[d]['observed_end_ms'] for d in deps] or [0]);assert start>=ready-1e-6,(k,start,ready)
        if start-ready>1e-9:
            gap=k+':trace_handoff';gkind='attention_input_reorder_tail' if k.endswith(':A3') else 'trace_handoff'
            nodes[gap]=dict(deps=deps,cost_ms=start-ready,observed_end_ms=start,kind=gkind,rank=r,layer=l);deps=[gap]
        nodes[k]=dict(deps=deps,cost_ms=end-start,observed_start_ms=start,observed_end_ms=end,kind=kind,rank=r,layer=l);assert end>=start
    for l in range(4):
        pre=f'L{l}:'
        ns={v['rank']:v for v in notify['rows'] if v['layer']==l};cs={v['rank']:v for v in combine['rows'] if v['layer']==l}
        for r in range(8,16):
            for j in range(3):add(f'{pre}r{r}:A{j}',[prior[r]] if j==0 else [f'{pre}r{r}:A{j-1}'],*local[l,r,j],'local_envelope',l,r)
        for pair in range(4):
            members=[8+pair*2,9+pair*2]
            for j in range(4):
                if j==3:
                    for r in members:add(f'{pre}r{r}:A3',[f'{pre}r{r}:A2',f'{pre}r{r}:CP2'],*local[l,r,3],'local_envelope',l,r)
                deps=[f'{pre}r{r}:A{j}' for r in members]+([f'{pre}r{r}:CP{j-1}' for r in members] if j else [])
                ready=max(nodes[d]['observed_end_ms'] for d in deps)
                for r in members:add(f'{pre}r{r}:CP{j}',deps,ready,comms[l,r,j],'CP_effective',l,r)
        for r in ns:
            add(f'{pre}r{r}:router',[f'{pre}r{r}:CP3'],*local[l,r,4],'local_envelope',l,r)
            add(f'{pre}r{r}:prepare',[f'{pre}r{r}:router'],local[l,r,4][1],ns[r]['notify_start_ms'],'preparation_envelope',l,r)
        ne=min(v['notify_end_ms'] for v in ns.values());add(pre+'notify',[f'{pre}r{r}:prepare' for r in ns],max(v['notify_start_ms'] for v in ns.values()),ne,'notify',l)
        for r in ns:
            add(f'{pre}r{r}:dispatch',[pre+'notify'],ne,cs[r]['dispatch_device_end_ms'],'EP_effective',l,r)
            add(f'{pre}r{r}:expert',[f'{pre}r{r}:dispatch'],*local[l,r,5],'expert_envelope',l,r)
        add(pre+'combine',[f'{pre}r{r}:expert' for r in ns],max(v['expert_related_end_ms'] for v in cs.values()),min(v['combine_visible_start_ms'] for v in cs.values()),'EP_effective',l)
        for r in ns:
            add(f'{pre}r{r}:unpermute',[pre+'combine'],cs[r]['combine_visible_start_ms'],cs[r]['combine_visible_end_ms'],'unpermute',l,r)
            prev=f'{pre}r{r}:unpermute'
            for j,d in enumerate(tails[l,r]):
                key=f'{pre}r{r}:tail{j}';add(key,[prev],d['start_ms'],d['end_ms'],'tail_add' if j<2 else 'F_epilogue',l,r);nodes[key]['event']=d['event'];nodes[key]['name']=d['name'];prev=key
                if j==1:layer_end=key
            prior[r]=prev
            for boundary,key in [('A0_end',f'{pre}r{r}:A0'),('A3_end',f'{pre}r{r}:A3'),('dispatch_end',f'{pre}r{r}:dispatch'),('expert_end',f'{pre}r{r}:expert'),('combine_end',f'{pre}r{r}:unpermute'),('layer_tail_end',layer_end)]+[(f'CP{j}_end',f'{pre}r{r}:CP{j}') for j in range(4)]:
                facts.append(dict(layer=l,rank=r,boundary=boundary,node=key,trace_ms=nodes[key]['observed_end_ms']))
    end=max(fends.values());add('END',list(prior.values()),end,end,'boundary',3)
    def schedule(g):
        t={}
        for k in graphlib.TopologicalSorter({k:n['deps'] for k,n in g.items()}).static_order():
            n=g[k];s=max([t[d]['end_ms'] for d in n['deps']] or [0]);t[k]=dict(start_ms=s,end_ms=s+n['cost_ms'])
        return t
    replay=schedule(nodes)
    for f in facts:f['replay_ms']=replay[f['node']]['end_ms'];f['error_ms']=f['replay_ms']-f['trace_ms']
    assert len(facts)==320 and max(abs(f['error_ms']) for f in facts)<1e-6
    assert abs(replay['END']['end_ms']-end)<1e-6
    for r in fends:assert abs(replay[prior[r]]['end_ms']-fends[r])<1e-6
    checks={}
    for l in range(4):
        g=copy.deepcopy(nodes);g[f'L{l}:combine']['cost_ms']+=1;delta=schedule(g)['END']['end_ms']-end;assert abs(delta-1)<1e-6;checks[f'L{l}_combine_plus1']=delta
    cache={}
    def rec(k):
        if k not in cache:cache[k]=max([rec(d) for d in nodes[k]['deps']] or [0])+nodes[k]['cost_ms']
        return cache[k]
    assert all(abs(rec(k)-v['end_ms'])<1e-7 for k,v in replay.items())
    summaries=[dict(layer=l,tail_end_ms=max(f['trace_ms'] for f in facts if f['layer']==l and f['boundary']=='layer_tail_end'),combine_ms=nodes[f'L{l}:combine']['cost_ms']) for l in range(4)]
    summary=dict(status='PASS_SAME_SAMPLE_ACCOUNTING',physical_prediction='PARTIAL_NOT_VALIDATED',scope='32GPU iter60 PP1 F0 ranks8..15 four layers',F_device_window_ms=end,rank_F_end_ms=fends,rank_F_start_ms=starts,layers=summaries,boundaries=len(facts),max_boundary_error_ms=max(abs(f['error_ms']) for f in facts),checks=checks,notes=['Same-sample costs and handoffs, not independent prediction.','Layer entry starts at LayerNorm, excludes prior tail Add; interlayer handoff is charged once.','Layer4 tail Add is separate from F epilogue kernels; endpoint is last F-associated device event, not CPU forward envelope.','Other stages, microbatches, backward and iteration time excluded.','CP/EP effective costs and conservative readiness assumptions inherited, not pure service.'])
    bindings={k:dict(value_ms=n['cost_ms'],policy='external_effective_replace' if n['kind'] in ['local_envelope','expert_envelope','CP_effective','EP_effective','tail_add'] else 'trace_reserved') for k,n in nodes.items()}
    a.out.mkdir(parents=True,exist_ok=False)
    for name,obj in [('graph',nodes),('replay',replay),('summary',summary),('cost-bindings',bindings),('boundary-comparison',facts),('reorder-evidence',reorders),('tail-evidence',[dict(layer=l,rank=r,events=d) for (l,r),d in tails.items()])]:
        (a.out/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    (a.out/'manifest.json').write_text(json.dumps(dict(inputs=[dict(path=str(p),sha256=digest(p)) for p in paths+[np,cp,Path(__file__).resolve()]],raw_inputs=raws,outputs=[dict(path=str(p.resolve()),sha256=digest(p)) for p in sorted(a.out.glob('*.json'))]),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

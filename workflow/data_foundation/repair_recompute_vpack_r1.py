"""Separate V input packing from unrelated in-flight Q/K CP transfers.

Produces a new candidate; never edits frozen source files. Preserves source
endpoints by explicitly rebasing source reserved readiness, not compute costs.
"""
import copy,json,re
from pathlib import Path
from calibrate256_fine_r1 import BASE,load,save,sha
from build_b_cp_pair_r4 import run
def main():
    out=BASE/'iter60-shared-dag-r2';out.mkdir(exist_ok=False)
    source=BASE/'deepep-stable-20261016-r1/model/B.json';g=load(source);old=run(g);changes=[];evidence=[]
    root=BASE/'calibrate256-fine-r1/r16-mb0';facts=load(root/'b-cost-blocks-r1/evidence.json');tg=load(root/'b-recompute-r12/graph.json')
    for rank,unit in [(16,1),(21,3)]:
        x=next(x for x in facts if x['rank']==rank);es=load(Path(x['path']))['traceEvents'];key=f'R{unit}:r{rank}:s0:n5';n=tg[key];assert n['kind']=='recompute_local_reorder' and len(n['events'])==1
        event=n['events'][0];d=next(d for d in x['devices'] if d['event']==event);call=es[d['launch_event']]
        parents=[(i,e) for i,e in enumerate(es) if e.get('cat')=='cpu_op' and e.get('pid')==call['pid'] and e.get('tid')==call['tid'] and e['ts']<=call['ts'] and e['ts']+e['dur']>=call['ts']+call['dur']]
        pack=next((i,e) for i,e in parents if e['name']=='aten::contiguous');assert pack[1]['args']['Input Dims'][0]==[2,2,4096,64,128]
        cpkey=f'R{unit}:r{rank}:s21:n0';cpe=tg[cpkey]['events'][0];overlap=(min(es[event]['ts']+es[event]['dur'],es[cpe]['ts']+es[cpe]['dur'])-max(es[event]['ts'],es[cpe]['ts']))/1000;assert overlap>0
        evidence.append(dict(rank=rank,unit=unit,path=x['path'],sha256=x['sha256'],copy_event=event,launch_event=d['launch_event'],contiguous_event=pack[0],shape=pack[1]['args']['Input Dims'][0],cp_event=cpe,overlap_ms=overlap,interpretation='V-width input packing overlaps prior CP transfer; full CP completion cannot be its prerequisite'))
    for slot,n in g.items():
        m=re.fullmatch(r'aligned_ready::(R\d+:r\d+):s0:n5',slot)
        if not m:continue
        dep=m[1]+':s21:n0';key=slot.removeprefix('aligned_ready::')
        assert g[key]['kind']=='recompute_local_reorder'
        if dep not in n['deps']:continue
        assert g[dep]['kind']=='recompute_CP';before=list(n['deps']);n['deps']=[d for d in n['deps'] if d!=dep]
        ready=max([old[d]['end_ms'] for d in n['deps']] or [0]);n['cost_ms']=old[slot]['end_ms']-ready
        changes.append(dict(node=slot,old_deps=before,new_deps=n['deps'],old_reserved_ms=load(source)[slot]['cost_ms'],new_reserved_ms=n['cost_ms']))
    replay=run(g);err=max(abs(replay[k]['end_ms']-old[k]['end_ms']) for k in g);assert err<1e-5
    save(out/'B.json',g);save(out/'F.json',load(BASE/'deepep-stable-20261016-r1/model/F.json'))
    save(out/'report.json',dict(status='CANDIDATE_GENERALIZED_VPACK_RULE',changes=changes,evidence=evidence,source_max_endpoint_error_ms=err,source_B_ms=replay['END']['end_ms'],limits=['Candidate differs from frozen graph; old release untouched.','Trace proves these completion prerequisites impossible at two instances. V identity follows tensor width, not installed code exact lineage.','All structurally corresponding V packing nodes use the same rule; not a per-error fitted correction.','Other conditional edges remain subject to validation.']))
    save(out/'manifest.json',dict(inputs={str(source):sha(source)},code_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('PASS candidate source endpoint regression; removed conditional edges:',len(changes))
if __name__=='__main__':main()

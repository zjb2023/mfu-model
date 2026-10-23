"""Isolated target calibration using the existing detailed builders.

Adapters only select physical rank cohort/MB and actual target configuration.
Generated adapter snapshots, logs and exact inputs are retained per stage.
No source artifacts or raw traces are overwritten.
"""
import argparse,contextlib,hashlib,importlib.util,json,re,sys,traceback,types
from datetime import datetime,timezone
from pathlib import Path
import orjson
import pp32_f_split_r8 as split
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
RAW=Path('/var/local/zjb-data/gbs64/framework_256_gbs_64/2026-07-31-13_40')
FAST=types.SimpleNamespace(loads=orjson.loads,dumps=json.dumps)
def load(p):return orjson.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def stamp():return datetime.now(timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank-base',type=int,required=True);ap.add_argument('--mb',type=int,default=0);ap.add_argument('--through',choices=['F','B'],default='B');a=ap.parse_args()
    rb=a.rank_base;assert rb%8==0 and 16<=rb<240 and 0<=a.mb<4
    out=BASE/'calibrate256-fine-r1'/f'r{rb}-mb{a.mb}';out.mkdir(parents=True,exist_ok=True)
    ranks=list(range(rb,rb+8));raws={}
    for r in ranks:
        ps=list(RAW.glob(f'*/profiler/iteration_60/rank{r}.*.pt.trace.json'));assert len(ps)==1;raws[r]=ps[0]
    logs=sorted(RAW.glob('worker33008/*/*.log'));assert logs;configlog=logs[0]
    def stage(name,func,inputs):
        marker=out/(name+'.complete.json');records={str(p):sha(p) for p in inputs}
        if marker.exists():
            m=load(marker);assert m['inputs']==records,('changed stage inputs',name)
            assert all(sha(Path(p))==h for p,h in m['outputs'].items()),name
            print('CACHE',name,flush=True);return
        dest=out/name;assert not dest.exists(),('incomplete retained',dest)
        print('START',name,stamp(),flush=True);started=stamp()
        with (out/(name+'.log')).open('w') as log:
            try:
                with contextlib.redirect_stdout(log):func(dest)
            except Exception:
                traceback.print_exc(file=log);raise
        save(marker,dict(started=started,finished=stamp(),inputs=records,outputs={str(p):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}))
        print('PASS',name,stamp(),flush=True)
    def extract(dest):
        dest.mkdir();split.json=FAST
        for r,p in raws.items():
            data=load(p);meta=dict(path=str(p),sha256=sha(p),groups=data['distributedInfo']['pg_config']);del data
            x=split.extract(r,path=p,iteration=60,metadata=meta,expected_world_size=256,expected_ep_members=ranks,microbatch=a.mb)
            x.update(stage=rb//16,metadata=meta);save(dest/f'rank-{r}.json',x)
    stage('pp32-f-split-r8',extract,[Path(split.__file__),*raws.values()])
    def f_audits(dest):
        xs=[load(out/f'pp32-f-split-r8/rank-{r}.json') for r in ranks];origin=min(x['origin_absolute_ms'] for x in xs);ns=[];cs=[]
        for x in xs:
            r=x['rank'];off=x['origin_absolute_ms']-origin
            for l in range(4):
                d=next(m for m in x['markers'] if m['kind']=='EP_DISPATCH' and m['ordinal']==l);c=next(m for m in x['markers'] if m['kind']=='EP_COMBINE' and m['ordinal']==l)
                n=[v for v in x['devices'] if v['semantic']==d['id'] and 'ace_notify_dispatch' in v['name']];assert len(n)==1
                ex=[v for v in x['devices'] if v['semantic']==f'r{r}:LOCAL:{6*l+5}']
                ns.append(dict(rank=r,layer=l,notify_start_ms=n[0]['start_ms']+off,notify_end_ms=n[0]['end_ms']+off))
                cs.append(dict(rank=r,layer=l,dispatch_device_end_ms=d['device_end_ms']+off,expert_related_end_ms=max(v['end_ms'] for v in ex)+off,combine_visible_start_ms=c['device_start_ms']+off,combine_visible_end_ms=c['device_end_ms']+off))
        dest.mkdir();save(dest/'notify.json',dict(time_origin_absolute_ms=origin,rows=ns));save(dest/'combine.json',dict(origin_absolute_ms=origin,rows=cs))
    stage('f-audit-inputs',f_audits,[Path(__file__),out/'pp32-f-split-r8.complete.json'])
    for name,file in [('ep-notify-order-audit-r1','notify.json'),('ep-combine-audit-r1','combine.json')]:
        p=out/name;p.mkdir(exist_ok=True);expected=load(out/'f-audit-inputs'/file)
        if (p/'report.json').exists():assert load(p/'report.json')==expected
        else:save(p/'report.json',expected)
    def builder(module,name,args=()):
        code=Path(__file__).with_name(module+'.py');text=code.read_text()
        text=re.sub(r'range\(8,\s*16(,\s*2)?\)',lambda m:f'range({rb},{rb+8}'+(',2)' if m[1] else ')'),text)
        if module=='build_four_layer_f_r1':text=text.replace('members=[8+pair*2,9+pair*2]',f'members=[{rb}+pair*2,{rb+1}+pair*2]')
        if module=='build_b_cost_blocks_r1':text=text.replace("next((i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name')=='backward_step')",f"sorted(((i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name')=='backward_step'),key=lambda z:z[1]['ts'])[{a.mb}]")
        adapter=out/(name+'.adapter.py')
        if adapter.exists():assert adapter.read_text()==text
        else:adapter.write_text(text)
        def execute(dest):
            spec=importlib.util.spec_from_file_location('target_'+module,code);mod=importlib.util.module_from_spec(spec);exec(compile(text,str(code),'exec'),mod.__dict__)
            mod.BASE=out;mod.json=FAST
            if module=='audit_b_attention_structure_identity_r3':mod.CONFIG_LOG=configlog
            if module=='build_b_recompute_cp_identity_r12':mod.SOURCE=out/'b-cost-blocks-r1/evidence.json'
            previous=sys.argv
            try:sys.argv=[str(code),'--out',str(dest),*map(str,args)];mod.main()
            finally:sys.argv=previous
        # Stage-specific frozen input list remains stable on resume.
        marker=out/(name+'.complete.json')
        inputs=[Path(p) for p in load(marker)['inputs']] if marker.exists() else [code,adapter,configlog,*sorted(out.glob('*.complete.json'))]
        stage(name,execute,inputs)
    builder('build_four_layer_f_r1','four-layer-f-r1')
    if a.through=='F':return
    builder('build_b_cost_blocks_r1','b-cost-blocks-r1')
    for module,name in [('audit_b_cp_identity_r1','b-cp-identity-r1'),('audit_b_attention_structure_identity_r3','b-attention-structure-r2'),('build_b_attention_replay_r3','b-attention-replay-r3'),('build_b_cp_pair_r4','b-cp-pair-r4'),('build_b_four_unit_skeleton_r9','b-four-unit-skeleton-r9')]:builder(module,name)
    for u in range(4):
        bridge=f'b-unit{u}-bridge';moe='b-moe-backward-r6-local-handoff' if u==0 else f'b-four-layer-r10-u{u}-moe';ep='b-ep8-coupling-r7' if u==0 else f'b-four-layer-r10-u{u}-ep'
        builder('build_b_gradient_bridge_r5',bridge,['--unit',u]);builder('build_b_moe_backward_r6',moe,['--unit',u,'--bridge-dir',out/bridge]);builder('build_b_ep8_coupling_r7',ep,['--source',out/moe])
    builder('build_b_four_layers_r10','b-four-layers-r10');builder('build_b_recompute_cp_identity_r12','b-recompute-r12')
    save(out/'scope.json',dict(iteration=60,world=256,stage=rb//16,mb=a.mb,ranks=ranks,status='CONDITIONAL_TARGET_GRAPHS_BUILT_TOPOLOGY_AUDIT_PENDING',config_log=str(configlog),adapter='Only rank loops, F CP member constants, B annotation selection; costs and dependency algorithms unchanged',limits=['Builder historic display labels may mention32/iter60; this scope and raw metadata are authoritative.','Generated target topology must be audited against frozen source before replacement.']))
if __name__=='__main__':main()

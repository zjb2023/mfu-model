"""Bounded PP1 B0 comparison; observations only, no calibration changes."""
import argparse, collections, hashlib, json
from pathlib import Path
from diagnose_expert_workload_r1 import extract, union

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
SRC=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/b-cost-blocks-r1/evidence.json')
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    refs=load(SRC);rp=BASE/'b-position-curves-r2/raw-provenance.json'
    target=next(r for r in load(rp) if (r['world'],r['stage'],r['iteration'])==(256,1,60))
    data=[];rawrefs=[]
    for world in [32,256]:
        for rank in (range(8,16) if world==32 else range(16,24)):
            if world==32: ref=next(r for r in refs if r['rank']==rank)
            else:
                paths=list(Path(target['path']).parent.glob(f'rank{rank}.*.pt.trace.json'))
                assert len(paths)==1,(rank,paths)
                ref=target if rank==16 else {'path':str(paths[0])}
            d=extract(ref,world,1,rank,False)
            save(a.out/f'experts-w{world}-rank{rank}.json',d);data.append(d)
            rawrefs.append({k:d[k] for k in ['path','sha256']})
            print(world,rank,'expert B0 PASS',flush=True)
    cohorts=[]
    for world in [32,256]:
        ds=[d for d in data if d['world']==world]
        for unit in range(4):
            ops=[next(o for o in d['blocks'][0]['operators'] if o['phase']=='backward' and o['FC']==2 and o['unit']==unit) for d in ds]
            shapes=[o['args'].get('Input Dims') for o in ops]
            cohorts.append(dict(world=world,unit=unit,ranks=[d['rank'] for d in ds],shapes=shapes,
                token_rows=[s[0][0] for s in shapes],sum_token_rows=sum(s[0][0] for s in shapes),
                gemm_counts=[o['gemm_count'] for o in ops]))
    # Reuse verified operator associations, enrich with original kernel names and shapes.
    module_sets=[];inputs=[SRC,rp]
    for world,rank,ref,cache in [(32,8,refs[0],BASE/'b-r10-source-framework-r1/source32-rank8.json'),
                               (256,16,target,BASE/'b-framework-position-r1/256-pp1.json')]:
        if world==32:ref=next(r for r in refs if r['rank']==8)
        raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256']
        events=json.loads(raw)['traceEvents'];del raw
        inputs.append(cache);block=load(cache)['blocks'][0];rows=[]
        for op in block['operators']:
            cpu=events[op['cpu_event']];assert cpu['name']==op['name']
            ds=[events[i] for i in op['device_events']];ks=collections.defaultdict(list)
            for e in ds:ks[e['name']].append((e['ts'],e['ts']+e['dur']))
            rows.append(dict(**op,input_dims=cpu.get('args',{}).get('Input Dims'),
                kernels=[dict(name=k,count=len(v),union_ms=union(v),sum_ms=sum(y-x for x,y in v)/1000) for k,v in ks.items()]))
        save(a.out/f'modules-w{world}-rank{rank}.json',rows)
        module_sets.append(rows);del events
    comparison=[]
    for kind in sorted({r['kind'] for r in module_sets[0]}):
        sides=[]
        for rows in module_sets:
            ops=[r for r in rows if r['kind']==kind];ivs=[v for o in ops for v in o['intervals']]
            sides.append(dict(gpu_union_ms=union(ivs),call_envelopes_ms=sum(o['gpu_envelope_ms'] or 0 for o in ops),
                internal_uncovered_ms=sum((o['gpu_envelope_ms'] or 0)-o['gpu_union_ms'] for o in ops),
                cpu_sum_ms=sum(o['cpu_ms'] for o in ops),kernel_count=sum(len(o['device_events']) for o in ops)))
        comparison.append(dict(kind=kind,source=sides[0],target=sides[1],
            source_minus_target_gpu_ms=sides[0]['gpu_union_ms']-sides[1]['gpu_union_ms']))
    blocks=BASE/'1f1b-overestimate-r1/blocks.json';inputs.append(blocks)
    stage_check=[]
    for mb in range(4):
        rows=[b for b in load(blocks) if b['phase']=='B' and b['mb']==mb and 1<=b['stage']<=14]
        rows.sort(key=lambda b:abs(b['error_ms']))
        stage_check.append(dict(mb=mb,ordered=[dict(stage=b['stage'],B_ms=b['observed_duration_ms'],error_ms=b['error_ms']) for b in rows]))
    report=dict(status='PP1_B0_OBSERVATIONAL_COMPARISON',iteration=60,modules=comparison,cohorts=cohorts,stage_error_ranking=stage_check,
        limitations=['FC1/FC2 inferred from forward/reverse paired order; exact weight IDs not established',
        'Source rank8 and target rank16 detailed operators; expert shapes additionally all eight ranks in one EP8 cohort',
        'Four execution-order units, not equal global layer IDs between different models',
        'Internal uncovered intervals are not proven pure waiting; other GPU work or hidden transport may overlap',
        'Module unions cannot be summed as wall time or contributions to 1F1B error',
        'Target second EP8 cohort and cross-iteration stability not covered; no cost model modified'])
    save(a.out/'report.json',report)
    save(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in inputs},raw=rawrefs,
        code={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('diagnose_expert_workload_r1.py'),Path(__file__).with_name('audit_b_operator_position_r1.py')]},
        outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
    print(json.dumps(dict(modules=comparison,cohorts=cohorts),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

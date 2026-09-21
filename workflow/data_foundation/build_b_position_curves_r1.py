"""Per-microbatch spatial curves. No stage/microbatch mean replaces a point."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
from check_b_scale_stability_r1 import extract,FILES
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
OLD=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/pp256-stage-audit-r7')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--publish',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=[];inputs=set(FILES.values());raw_refs=[];read_count=0
    for world,file in FILES.items():
        records=list(csv.DictReader(file.open()))
        for it in [60,80,100]:
            for st in range(1,world//16-1):
                cached=BASE/f'b-scale-stability-r1/gpu-{world}-{it}-pp{st}.json'
                cpu={int(r['microbatch']):r for r in records if int(r['iteration'])==it and int(r['pp_stage'])==st and int(r['pp_lane'])==0 and r['phase']=='backward'}
                expected=3 if world==224 else 4;assert len(cpu)==expected
                if world==256 and it==60:
                    p=OLD/f'rank-{st*16}.json';inputs.add(p);x=load(p)
                    bs=[dict(mb=b['mb'],gpu_ms=b['duration_ms'],cpu_event=b['cpu_event'],first_gpu_event=b['first_device_event'],last_gpu_event=b['last_device_event']) for b in x['blocks'] if b['phase']=='B']
                else:
                    if cached.exists():inputs.add(cached);x=load(cached)
                    else:
                        x=extract(Path(cpu[0]['source_path']),world,st*16);read_count+=1
                    bs=x['blocks']
                raw_refs.append(dict(world=world,iteration=it,stage=st,path=x['path'],resolved_path=str(Path(x['path']).resolve()),sha256=x['sha256']))
                assert len(bs)==expected
                for b in bs:
                    if 'cpu_ms' in b:assert abs(b['cpu_ms']-float(cpu[b['mb']]['duration_ns'])/1e6)<1e-3
                    rows.append(dict(world=world,iteration=it,stage=st,rank=st*16,mb=b['mb'],gpu_ms=b['gpu_ms'],cpu_ms=float(cpu[b['mb']]['duration_ns'])/1e6,cpu_event=b['cpu_event'],first_gpu_event=b['first_gpu_event'],last_gpu_event=b['last_gpu_event']))
                print(world,it,st,flush=True)
    trends=[]
    for world in [224,256]:
        for it in [60,80,100]:
            for mb in range(3 if world==224 else 4):
                rs=sorted([r for r in rows if (r['world'],r['iteration'],r['mb'])==(world,it,mb)],key=lambda r:r['stage'])
                vals=[r['gpu_ms'] for r in rs];ds=[b-a for a,b in zip(vals,vals[1:])]
                trends.append(dict(world=world,iteration=it,mb=mb,start_ms=vals[0],end_ms=vals[-1],end_minus_start_ms=vals[-1]-vals[0],decreasing_edges=sum(x<0 for x in ds),increasing_edges=sum(x>0 for x in ds),edge_count=len(ds),strict_decrease=all(x<0 for x in ds),adjacent_deltas_ms=ds))
    flips=[]
    for world in [224,256]:
        for mb in range(3 if world==224 else 4):
            ts=[x for x in trends if x['world']==world and x['mb']==mb]
            flips.append(dict(world=world,mb=mb,always_decreasing_edges=[i+1 for i in range(len(ts[0]['adjacent_deltas_ms'])) if all(x['adjacent_deltas_ms'][i]<0 for x in ts)],always_increasing_edges=[i+1 for i in range(len(ts[0]['adjacent_deltas_ms'])) if all(x['adjacent_deltas_ms'][i]>0 for x in ts)],edge_notation='i denotes PPi -> PP(i+1)'))
    data=dict(rows=rows,trends=trends,repeat_edges=flips,iterations=[60,80,100],scope='GPU first-to-last device envelope attributed to CPU B via unique runtime correlation; lane0 one rank per stage; no averaging across microbatches',new_raw_files_read=read_count,trace_records=len(raw_refs),points=len(rows),models_changed=False)
    assert len(rows)==276 and len(trends)==21
    dump(a.out/'data.json',data);dump(a.out/'raw-provenance.json',raw_refs)
    template=Path(__file__).with_name('b_position_curves_r1.html')
    (a.out/'index.html').write_text(template.read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False)))
    dump(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in sorted(inputs)},code={str(p.resolve()):sha(p) for p in [Path(__file__),template,Path(__file__).with_name('check_b_scale_stability_r1.py')]},outputs={str(p.resolve()):sha(p) for p in a.out.iterdir() if p.is_file()}))
    if a.publish:shutil.copytree(a.out,a.publish)
    print(json.dumps(dict(new_raw_files_read=read_count,trends=trends,repeat_edges=flips),ensure_ascii=False))
if __name__=='__main__':main()

"""Replace iteration100 with40; reuse60/80 and retain archived r1 evidence."""
import argparse,hashlib,json,shutil
from pathlib import Path
from check_b_scale_stability_r1 import extract
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--publish',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    previous=BASE/'b-position-curves-r1';old=load(previous/'data.json');refs=load(previous/'raw-provenance.json');rows=[r for r in old['rows'] if r['iteration'] in [60,80]];raws=[r for r in refs if r['iteration'] in [60,80]]
    planned=[]
    for ref in refs:
        if ref['iteration']!=60:continue
        rank=ref['stage']*16;folder=Path(ref['path']).parent.parent/'iteration_40';paths=list(folder.glob(f'rank{rank}.*.pt.trace.json'));assert len(paths)==1,(folder,rank,paths);planned.append((ref,rank,paths[0]))
    assert len(planned)==26
    dump(a.out/'protocol.json',dict(iterations=[40,60,80],removed_iteration=100,reason='user-selected exclusion of final iteration; not independently diagnosed cause',new_rank_traces=26,reuse_iterations=[60,80],model_unchanged=True))
    for ref,rank,p in planned:
        x=extract(p,ref['world'],rank);dump(a.out/f'gpu-{ref["world"]}-40-pp{ref["stage"]}.json',x)
        raws.append(dict(world=ref['world'],iteration=40,stage=ref['stage'],path=str(p),resolved_path=str(p.resolve()),sha256=x['sha256']))
        for b in x['blocks']:rows.append(dict(world=ref['world'],iteration=40,stage=ref['stage'],rank=rank,**{k:v for k,v in b.items()}))
        print(ref['world'],40,ref['stage'],flush=True)
    rows.sort(key=lambda r:(r['world'],r['iteration'],r['stage'],r['mb']));trends=[];repeats=[]
    for w in [224,256]:
        for it in [40,60,80]:
            for mb in range(3 if w==224 else 4):
                rs=[r for r in rows if (r['world'],r['iteration'],r['mb'])==(w,it,mb)];assert len(rs)==w//16-2
                vs=[r['gpu_ms'] for r in rs];ds=[b-a for a,b in zip(vs,vs[1:])]
                trends.append(dict(world=w,iteration=it,mb=mb,start_ms=vs[0],end_ms=vs[-1],end_minus_start_ms=vs[-1]-vs[0],decreasing_edges=sum(d<0 for d in ds),increasing_edges=sum(d>0 for d in ds),edge_count=len(ds),strict_decrease=all(d<0 for d in ds),adjacent_deltas_ms=ds))
        for mb in range(3 if w==224 else 4):
            ts=[t for t in trends if t['world']==w and t['mb']==mb]
            repeats.append(dict(world=w,mb=mb,always_decreasing_edges=[i+1 for i in range(w//16-3) if all(t['adjacent_deltas_ms'][i]<0 for t in ts)],always_increasing_edges=[i+1 for i in range(w//16-3) if all(t['adjacent_deltas_ms'][i]>0 for t in ts)],edge_notation='i denotes PPi -> PP(i+1)'))
    data=dict(rows=rows,trends=trends,repeat_edges=repeats,iterations=[40,60,80],scope=old['scope'],new_raw_files_read=26,trace_records=len(raws),points=len(rows),models_changed=False)
    assert len(rows)==276 and len(trends)==21 and all(r['iteration']!=100 for r in rows)
    dump(a.out/'data.json',data);dump(a.out/'raw-provenance.json',raws)
    template=Path(__file__).with_name('b_position_curves_r1.html');html=template.read_text()
    html=html.replace('>iter60</span>','>iter40</span>').replace('>iter80</span>','>iter60</span>').replace('>iter100</span>','>iter80</span>')
    html=html.replace('iter60/80/100','iter40/60/80').replace('复用已有23份记录，新读取55份trace','复用已有52份记录，新读取26份trace')
    html=html.replace('</h1>','</h1><p class="small">r2：按用户选择，将iter100替换为iter40；当前仅比较40、60、80。历史100数据保留归档，不参与当前曲线与统计。</p>',1)
    (a.out/'index.html').write_text(html.replace('__DATA__',json.dumps(data,ensure_ascii=False)))
    dump(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in [previous/'data.json',previous/'raw-provenance.json']},code={str(p.resolve()):sha(p) for p in [Path(__file__),template,Path(__file__).with_name('check_b_scale_stability_r1.py')]},outputs={str(p.resolve()):sha(p) for p in a.out.iterdir() if p.is_file()}))
    if a.publish:
        assert sha(a.publish/'index.html')==sha(previous/'index.html'),'Unexpected live version'
        assert not (a.publish/'index-60-80-100.html').exists()
        for name in ['index.html','data.json','raw-provenance.json','manifest.json']:
            shutil.copyfile(a.publish/name,a.publish/(Path(name).stem+'-60-80-100'+Path(name).suffix))
            shutil.copyfile(a.out/name,a.publish/name)
    print(json.dumps(repeats,ensure_ascii=False))
if __name__=='__main__':main()

"""Bounded within-world B kernel comparison; no costs fitted or replaced."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'

def union(intervals):
    end = -float('inf'); total = 0
    for a,b in sorted(intervals):
        total += max(0, b-max(a,end)); end=max(end,b)
    return total/1000

def extract(ref):
    p=Path(ref['path']); raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==ref['sha256']
    es=json.loads(raw)['traceEvents']; del raw
    step=next(e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep'))
    bs=sorted([(i,e) for i,e in enumerate(es) if e.get('cat')=='user_annotation' and e.get('name')=='backward_step' and step['ts']<=e['ts']<step['ts']+step['dur']],key=lambda x:x[1]['ts'])
    runtime=collections.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat')=='privateuse1_runtime' and 'correlation' in e.get('args',{}):runtime[e['args']['correlation']].append(e)
    owned=collections.defaultdict(list)
    for i,e in enumerate(es):
        if e.get('cat') not in ['kernel','gpu_memcpy','gpu_memset']:continue
        calls=runtime.get(e.get('args',{}).get('correlation'),[])
        if len(calls)!=1:continue
        r=calls[0]
        hits=[m for m,(_,b) in enumerate(bs) if b['pid']==r['pid'] and b['ts']<=r['ts'] and r['ts']+r.get('dur',0)<=b['ts']+b['dur']+.01]
        if len(hits)==1:owned[hits[0]].append((i,e))
    rows=[]
    for mb,(_,cpu) in enumerate(bs):
        ds=owned[mb]; assert ds
        start=min(e['ts'] for _,e in ds); end=max(e['ts']+e['dur'] for _,e in ds)
        groups=collections.defaultdict(list)
        for i,e in ds:groups[e['name']].append((i,e))
        kernels=[]
        for name,items in groups.items():
            kernels.append(dict(name=name,count=len(items),sum_ms=sum(e['dur'] for _,e in items)/1000,
                                union_ms=union([(e['ts'],e['ts']+e['dur']) for _,e in items]),events=[i for i,_ in items]))
        active=union([(e['ts'],e['ts']+e['dur']) for _,e in ds]); envelope=(end-start)/1000
        rows.append(dict(mb=mb,cpu_ms=cpu['dur']/1000,envelope_ms=envelope,owned_device_union_ms=active,
                         no_owned_device_coverage_ms=envelope-active,kernels=kernels))
    return dict(**ref,blocks=rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    refs_path=BASE/'b-position-curves-r2/raw-provenance.json'
    refs=json.loads(refs_path.read_text())
    # 224's repeated PP5->6 drop; 256 early/late contrast, not a monotonicity claim.
    chosen=[next(r for r in refs if (r['world'],r['iteration'],r['stage'])==(w,60,s)) for w,s in [(224,5),(224,6),(256,1),(256,14)]]
    args.out.mkdir(parents=True,exist_ok=False)
    data=[]
    for ref in chosen:
        row=extract(ref); data.append(row)
        (args.out/f"{ref['world']}-pp{ref['stage']}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(ref['world'],ref['stage'],'done',flush=True)
    comparisons=[]
    for a,b in [data[:2],data[2:]]:
        for left,right in zip(a['blocks'],b['blocks']):
            aa={k['name']:k for k in left['kernels']};bb={k['name']:k for k in right['kernels']}
            deltas=[dict(name=n,left_count=aa.get(n,{}).get('count',0),right_count=bb.get(n,{}).get('count',0),
                         right_minus_left_sum_ms=bb.get(n,{}).get('sum_ms',0)-aa.get(n,{}).get('sum_ms',0)) for n in aa.keys()|bb.keys()]
            comparisons.append(dict(world=a['world'],stages=[a['stage'],b['stage']],mb=left['mb'],
                                    envelope_delta_ms=right['envelope_ms']-left['envelope_ms'],
                                    coverage_gap_delta_ms=right['no_owned_device_coverage_ms']-left['no_owned_device_coverage_ms'],
                                    top_kernel_deltas=sorted(deltas,key=lambda x:abs(x['right_minus_left_sum_ms']),reverse=True)[:15]))
    report=dict(scope='iter60, lane0 representative rank only; 224 PP5/6, 256 PP1/14; all B microbatches',
                comparisons=comparisons,limitations=['kernel duration sums overlap and are not additive wall-time attribution',
                'same name does not establish same tensor shape, token work or phase identity',
                'no-owned-device coverage is not proof of GPU idle or CPU delay',
                'communication kernels can contain waiting; visible EP kernels may omit hidden transport',
                '224 and 256 compared within each run only, not pooled; no source32 used'])
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (args.out/'manifest.json').write_text(json.dumps(dict(inputs={str(refs_path):sha(refs_path)},raw=chosen,
        code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in args.out.glob('*.json')}),indent=2)+'\n')

if __name__=='__main__':main()

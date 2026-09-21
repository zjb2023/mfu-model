"""Source32 framework evidence versus existing target samples; no fitting."""
import argparse,hashlib,json
from pathlib import Path
from audit_b_framework_position_r1 import extract
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 p=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/b-cost-blocks-r1/evidence.json');ref=next(x for x in json.loads(p.read_text()) if x['rank']==8);ref.update(world=32,stage=1)
 source=extract(ref);(a.out/'source32-rank8.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
 s={x['kind']:x for x in source['blocks'][0]['summary']};rows=[];inputs=[p]
 for stage in [1,14]:
  tp=BASE/f'b-framework-position-r1/256-pp{stage}.json';inputs.append(tp)
  for b in json.loads(tp.read_text())['blocks']:
   rows.append(dict(stage=stage,mb=b['mb'],modules=[dict(kind=t['kind'],source_B0_gpu_ms=s[t['kind']]['gpu_union_ms'],target_gpu_ms=t['gpu_union_ms'],source_minus_target_gpu_ms=s[t['kind']]['gpu_union_ms']-t['gpu_union_ms']) for t in b['summary']]))
 (a.out/'report.json').write_text(json.dumps(dict(rows=rows,scope='source32 rank8 B0 used by r10 vs target256 PP1/14 all B; module GPU unions, not DAG contributions',limitations=['CPU-marker containment is conditional association','FC1/FC2 identity inferred from paired call order','CP included inside Attention; module unions may overlap','source B1..7 extracted only as extra evidence, not calibration or pooled costs']),ensure_ascii=False,indent=2)+'\n')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 (a.out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in inputs},raw_source={k:ref[k] for k in ['path','sha256']},code={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('audit_b_framework_position_r1.py'),Path(__file__).with_name('audit_b_operator_position_r1.py')]},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}),indent=2)+'\n')
 print(json.dumps(rows[0],ensure_ascii=False))
if __name__=='__main__':main()

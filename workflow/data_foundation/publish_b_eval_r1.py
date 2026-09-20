"""Publish a new immutable local evaluation page, without touching old pages."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
src=BASE/'b-r10-60to70-eval-r2';out=BASE/'2111-blocks-ui-r1/df-v001/b-r10-60to70-r2'
out.mkdir(parents=True,exist_ok=False)
payload={k:json.loads((src/f).read_text()) for k,f in [('report','report.json'),('phases','phase-comparison.json'),('components','component-variation.json')]}
template=Path(__file__).with_name('ui_b_eval_r1.html')
(out/'index.html').write_text(template.read_text().replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')))
for name in ['report.json','manifest.json','phase-comparison.json','component-variation.json','seal.json']:shutil.copy2(src/name,out/name)
def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(out/'ui-manifest.json').write_text(json.dumps(dict(version='b-r10-60to70-ui2',code=[rec(template),rec(Path(__file__))],files=[rec(p) for p in out.iterdir()]),indent=2)+'\n')
print(out)

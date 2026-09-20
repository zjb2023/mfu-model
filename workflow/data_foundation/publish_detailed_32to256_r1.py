import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';src=BASE/'detailed-fb-32to256-r1';out=BASE/'2111-blocks-ui-r1/df-v001/detailed-fb-32to256-r1';out.mkdir(exist_ok=False)
template=Path(__file__).with_name('ui_detailed_32to256_r1.html');data={k:json.loads((src/(file+'.json')).read_text()) for k,file in [('report','report'),('comparison','comparison'),('outer','outer-prediction')]}
(out/'index.html').write_text(template.read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False)))
for n in ['report','manifest','policy','comparison','checks','parameters','seal']:shutil.copy2(src/(n+'.json'),out/(n+'.json'))
def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(out/'ui-manifest.json').write_text(json.dumps(dict(version='detailed-fb-32to256-ui1',inputs=[rec(template),rec(Path(__file__))]+[rec(src/(n+'.json')) for n in ['report','comparison','outer-prediction']],outputs=[rec(p) for p in out.iterdir()]),indent=2)+'\n')
print(out)

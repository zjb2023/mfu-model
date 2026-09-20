import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';src=BASE/'pp32-detailed-fb-assembly-r1';out=BASE/'2111-blocks-ui-r1/df-v001/pp32-detailed-fb-r1';out.mkdir(exist_ok=False)
template=Path(__file__).with_name('ui_pp32_detailed_fb_r1.html');data={k:json.loads((src/(k+'.json')).read_text()) for k in ['report','timeline']}
(out/'index.html').write_text(template.read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False)))
for n in ['report','manifest','checks','bindings','parameters','seal']:shutil.copy2(src/(n+'.json'),out/(n+'.json'))
shutil.copy2(BASE/'pp2-module-compatibility-r1/summary.json',out/'pp2-summary.json')
def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(out/'ui-manifest.json').write_text(json.dumps(dict(version='pp32-detailed-fb-ui1',inputs=[rec(template),rec(Path(__file__)),rec(src/'timeline.json'),rec(src/'report.json')],outputs=[rec(p) for p in out.iterdir()]),indent=2)+'\n')
print(out)

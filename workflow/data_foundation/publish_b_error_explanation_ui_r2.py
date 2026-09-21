import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
old=BASE/'b-evidence-index-r1';out=BASE/'b-evidence-index-ui-r2'
live=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')
js=Path(__file__).with_name('b_error_explanation_ui_r2.js')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old/'index.html')==sha(live/'index.html')
out.mkdir(parents=True,exist_ok=False)
(out/'index.html').write_text((old/'index.html').read_text().replace('</body>','<script>'+js.read_text()+'</script></body>'))
d=json.loads((old/'b-only-diagnosis.json').read_text());original=d['baseline_ms']-d['truth_ms']
metrics=dict(middle_B_explained_percent=100*(d['baseline_ms']-d['middle_b_ms'])/original,all_B_explained_percent=100*(d['baseline_ms']-d['all_b_ms'])/original)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
m=dict(inputs={str(p):sha(p) for p in [old/'index.html',old/'b-only-diagnosis.json']},code={str(p.resolve()):sha(p) for p in [Path(__file__),js]},outputs={str(p.resolve()):sha(p) for p in out.iterdir()})
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','manifest.json']:
    dest=live/(Path(name).stem+'-before-explanation-r2'+Path(name).suffix);assert not dest.exists();shutil.copyfile(live/name,dest);shutil.copyfile(out/name,live/name)
print(json.dumps(metrics))

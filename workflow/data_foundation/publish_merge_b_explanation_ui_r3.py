import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
old=BASE/'b-evidence-index-ui-r2';out=BASE/'b-evidence-index-ui-r3'
live=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')
js=Path(__file__).with_name('merge_b_explanation_ui_r3.js')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old/'index.html')==sha(live/'index.html')
out.mkdir(parents=True,exist_ok=False)
(out/'index.html').write_text((old/'index.html').read_text().replace('</body>','<script>'+js.read_text()+'</script></body>'))
m=dict(inputs={str(old/'index.html'):sha(old/'index.html')},code={str(p.resolve()):sha(p) for p in [Path(__file__),js]},outputs={str(out/'index.html'):sha(out/'index.html')},prediction_changed=False)
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
for name in ['index.html','manifest.json']:
    dest=live/(Path(name).stem+'-before-merge-r3'+Path(name).suffix);assert not dest.exists();shutil.copyfile(live/name,dest);shutil.copyfile(out/name,live/name)
print('PASS published merged explanation')

"""Versioned presentation update at the existing URL; frozen results unchanged."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
source=BASE/'full-step-32to256-r1'
out=BASE/'full-step-32to256-ui-r3'
live=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')
script=Path(__file__).with_name('profiler_only_ui_r3.js')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
assert not out.exists()
assert sha(live/'index.html')==sha(source/'index.html'), 'Unexpected live revision; refuse overwrite'
seals={str(p):sha(p) for p in source.iterdir() if p.is_file()}
out.mkdir()
for name in ['report.json','data.json','validation.json','sealed-prediction.json','seal.json']:
    shutil.copyfile(source/name,out/name)
html=(source/'index.html').read_text().replace('</body>','<script>'+script.read_text()+'</script></body>')
(out/'index.html').write_text(html)
dump(out/'display-policy.json',dict(version='UI-r3',primary_clock='ProfilerStep',primary_mfu='mfu_profiler',secondary_clock='TrainingStep',secondary_display='collapsed appendix, excluded from current fitting',prediction_modified=False,source_model='full-step-32to256-r1'))
dump(out/'manifest.json',dict(inputs=seals,code={str(p.resolve()):sha(p) for p in [Path(__file__),script]},outputs={str(p.resolve()):sha(p) for p in out.iterdir() if p.is_file()}))
assert all(sha(Path(p))==h for p,h in seals.items())
assert not (live/'index-ui-r2.html').exists()
shutil.copyfile(live/'index.html',live/'index-ui-r2.html')
shutil.copyfile(live/'manifest.json',live/'manifest-ui-r2.json')
for name in ['index.html','display-policy.json','manifest.json']:
    shutil.copyfile(out/name,live/name)
assert sha(out/'index.html')==sha(live/'index.html')
print('PASS published UI-r3; frozen source intact; previous page retained as index-ui-r2.html')

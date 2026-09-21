import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
live=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')
source=BASE/'full-step-32to256-ui-r3';out=BASE/'1f1b-overestimate-ui-r1'
report=BASE/'1f1b-overestimate-r1/report.json';js=Path(__file__).with_name('overestimate_diagnostic_ui_r1.js')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(live/'index.html')==sha(source/'index.html')
assert not out.exists() and not (live/'index-ui-r3.html').exists()
out.mkdir()
html=(source/'index.html').read_text().replace('</body>','<script>const DIAG='+report.read_text()+';\n'+js.read_text()+'</script></body>')
(out/'index.html').write_text(html);shutil.copyfile(report,out/'diagnostic-report.json')
manifest=dict(inputs={str(p):sha(p) for p in [source/'index.html',report]},code={str(p.resolve()):sha(p) for p in [Path(__file__),js]},outputs={str(p.resolve()):sha(p) for p in out.iterdir()})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copyfile(live/'index.html',live/'index-ui-r3.html');shutil.copyfile(live/'manifest.json',live/'manifest-ui-r3.json')
for name in ['index.html','diagnostic-report.json','manifest.json']:shutil.copyfile(out/name,live/name)
print('PASS published diagnostic panel; original prediction unchanged')

import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
src=BASE/'b-four-layers-r10';out=BASE/'2111-blocks-ui-r1/df-v001/b-four-layers-r10'
template=Path(__file__).with_name('ui_b_four_layers_r10.html')
controller=Path(__file__).with_name('ui_b_dag_r2.js')
payload={k:json.loads((src/(k+'.json')).read_text()) for k in ['graph','replay','summary']}
if '--refresh' in sys.argv:
    previous=json.loads((out/'manifest.json').read_text())['version']
    assert previous in ['b-four-layers-r10-ui1','b-four-layers-r10-ui2','b-four-layers-r10-ui3']
    archive=out.with_name(previous+'-archive')
    if previous!='b-four-layers-r10-ui3':
        assert not archive.exists()
        shutil.copytree(out,archive)
else:out.mkdir(exist_ok=False)
(out/'index.html').write_text(template.read_text().replace('__DAG_CONTROLLER__',controller.read_text()).replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False)))
for n in ['graph','replay','summary','tests','coverage']:shutil.copyfile(src/(n+'.json'),out/(n+'.json'))
def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(dict(version='b-four-layers-r10-ui3',inputs=[rec(p) for p in [src/'manifest.json',template,controller,Path(__file__).resolve()]],outputs=[rec(p) for p in out.iterdir() if p.name!='manifest.json']),indent=2)+'\n')
print(out)

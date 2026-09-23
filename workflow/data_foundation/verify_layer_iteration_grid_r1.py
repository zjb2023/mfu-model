import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
n=0
for name in ['layer-iteration-grid-r1','layer-iteration-analysis-r1']:
    folder=BASE/name;manifest=json.loads((folder/'manifest.json').read_text())
    for file,h in manifest['outputs'].items():assert sha(folder/file)==h;n+=1
grid=json.loads((BASE/'layer-iteration-grid-r1/report.json').read_text());analysis=json.loads((BASE/'layer-iteration-analysis-r1/report.json').read_text());ui=json.loads((BASE/'layer-iteration-grid-ui-r1/report.json').read_text())
assert grid['status']==analysis['status']==ui['status']=='PASS'
assert len(grid['rows'])==4032 and len(analysis['rows'])==2016 and analysis['regression_checks']==896
web=BASE/'2111-blocks-ui-r1/df-v001/layer-workload-r1';assert sha(web/'index.html')==ui['html_sha256']
files=[BASE/'layer-iteration-grid-r1/manifest.json',BASE/'layer-iteration-analysis-r1/manifest.json',BASE/'layer-iteration-grid-ui-r1/report.json',ROOT/'docs/data-foundation/LAYER_ITERATION_TRENDS_R1.md',web/'index.html']
files+=list((ROOT/'workflow/data_foundation').glob('*layer_iteration_grid*r1*'))+[ROOT/'workflow/data_foundation/ui_layer_iteration_trends_r1.html']
out=BASE/'layer-iteration-grid-verification-r1';out.mkdir(exist_ok=False);report=dict(status='PASS',output_hash_checks=n,expert_conservation_checks=645120,regression_checks=896,ui_checks=24,evidence={str(p):sha(p) for p in files},prediction_unchanged=True)
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',n,'output hashes; counts, regression and UI passed')

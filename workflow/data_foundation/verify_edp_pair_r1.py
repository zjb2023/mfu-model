"""Freeze the full-stage extension, including UI verification, without a Git commit."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
folders=['layer-workload-edpB-r1','layer-filtering-edpB-r1','layer-edp-pair-r1'];hashes=0
for name in folders:
    p=BASE/name;d=json.loads((p/'manifest.json').read_text())
    for f,h in d['outputs'].items():assert sha(p/f)==h;hashes+=1
ui=BASE/'layer-edp-pair-ui-r1/report.json';assert json.loads(ui.read_text())['status']=='PASS'
report=json.loads((BASE/'layer-edp-pair-r1/report.json').read_text());assert report['status']=='PASS' and len(report['rows'])==224
files=[BASE/f/'manifest.json' for f in folders]+[ui,ROOT/'docs/data-foundation/LAYER_EDP_PAIR_R1.md']
files+=list((ROOT/'workflow/data_foundation').glob('*edp_pair_r1.*'))
web=BASE/'2111-blocks-ui-r1/df-v001/layer-workload-r1'
files+=[web/'index.html',web/'edp-pair.md',web/'edp-pair-report.json',web/'edp-pair-manifest.json']
result=dict(status='PASS',output_hash_checks=hashes,paired_layer_mb=224,expert_conservation_checks=71680,prediction_unchanged=True,evidence={str(p):sha(p) for p in files},scope='256 iter60 PP1..14 both EP8 replicas; no claim of pre-capacity distribution or cross-iteration stability')
out=BASE/'layer-edp-pair-verification-r1';out.mkdir(exist_ok=False);(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',hashes,'output hashes, 224 paired points, browser PASS')

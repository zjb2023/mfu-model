"""Freeze derived evidence and publication, not raw data or a model release."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'results/data-foundation'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifests=[base/x/'manifest.json' for x in ['layer-workload-224-r1','layer-workload-256-r1','layer-workload-analysis-r1']]
checks=0
for m in manifests[:2]:
    for f,h in json.loads(m.read_text())['outputs'].items():
        assert hashfile(m.parent/f)==h
        checks+=1
files=manifests+[base/'layer-capacity-audit-r1/report.json',base/'layer-workload-ui-checks-r1/report.json',ROOT/'docs/data-foundation/LAYER_WORKLOAD_CURVES_R1.md']
files+=list((ROOT/'workflow/data_foundation').glob('*layer*workload*r1.*'))
files+=[ROOT/'workflow/data_foundation/audit_layer_capacity_r1.py']
files+=[Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40/worker33008/2026-07-31_1340/tp1_pp16_dp_mbs2_numbs_gbs64_gpus0_mtp1_forcelbfalse_pertensorfalse_NO_LOSS_REDUCE.RANK2.10.124.33.8.log')]
web=base/'2111-blocks-ui-r1/df-v001/layer-workload-r1'
files+=[web/'index.html',web/'analysis.md']
out=base/'layer-workload-session-r1';out.mkdir(exist_ok=False)
d=dict(status='PASS within stated scope',branch='32to256',base_commit='9f83971d36382c665dfd00c1ace86e28c97e9c10',prediction_unchanged=True,output_hash_checks=checks,evidence={str(p):hashfile(p) for p in files},limitations=['EP8 complete only256 iter60 first replica','No runtime source commit identity','No before-capacity full route reconstruction','Not new prediction accuracy'])
(out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
(web/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
print('PASS',checks,'output hashes;',len(files),'evidence files')

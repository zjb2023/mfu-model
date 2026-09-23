"""Verify derived counts against frozen curves and hash the new evidence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
out=BASE/'layer-filtering-verification-r1';out.mkdir(exist_ok=False)
folder=BASE/'layer-filtering-r1';report=json.loads((folder/'report.json').read_text())
old=json.loads((BASE/'layer-workload-analysis-r1/data.json').read_text())
lookup={(r['layer'],r['mb']):r['total_rows'] for r in old['ep8']}
assert len(report['rows'])==224 and report['expert_conservation_checks']==35840
for r in report['rows']:
    assert lookup[r['layer'],r['mb']]==r['retained_assignments']
    assert r['offered_assignments']==393216
    assert r['retained_assignments']+r['inferred_removed_assignments']==393216
manifest=json.loads((folder/'manifest.json').read_text());hashes=0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for f,h in manifest['outputs'].items():
    assert sha(folder/f)==h;hashes+=1
code=Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/moe')
files=[code/'router.py',code/'moe_utils.py',code/'token_dispatcher.py',ROOT/'docs/data-foundation/LAYER_FILTERING_R1.md',Path(__file__).resolve(),ROOT/'workflow/data_foundation/publish_layer_workload_r1.py',folder/'manifest.json']
result=dict(status='PASS',layer_mb_matches=224,assignment_identities=224,output_hash_checks=hashes,expert_conservation_checks=35840,evidence={str(p):sha(p) for p in files},causality='No direct pre-capacity perexpert histogram or probability values; capacity-only removed count unavailable.')
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))

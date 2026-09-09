#!/usr/bin/env python3
"""Record only completed preparation checks; no training or simulation."""
from pathlib import Path
import json
import shutil
from prepare_baseline import CONTROL, ROOT, SOURCE, dump, original_state, sha

def main():
    before=json.loads((CONTROL/'original_state_before.json').read_text())
    after=original_state()
    if before!=after:raise RuntimeError('original index or tracked working changes shifted; inspect before proceeding')
    evidence=[]
    for role, rel in [('A_replay','results/w37/A/smoke-preflight2'),
                     ('B_boundary','results/w37/B/smoke-readonlycheck'),
                     ('A_full_reproduction','results/w37/A/reproduce-v685-preflight1')]:
        source=ROOT/rel
        result=json.loads((source/'smoke_result.json').read_text())
        if result['status']!='PASS':raise RuntimeError(f'{role} did not pass')
        for name in ['smoke_result.json','input_access_audit.json','execution.log','run_manifest.json']:
            dest=CONTROL/'preparation_checks'/role/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/name,dest)
            evidence.append({'path':str(dest.relative_to(ROOT)),'sha256':sha(dest),'original_result_path':str(source/name)})
    support=[p for d in [ROOT/'scripts/w37',ROOT/'docs/w37'] for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='preparation_seal.json']
    dump(CONTROL/'preparation_seal.json',{'status':'PASS_PREPARATION_LOCAL_CODE_AND_INPUTS',
        'original_repository_state_preserved':True,'original_state':after,
        'evidence':evidence,'support_files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(support)],
        'scope':'common baseline preparation only; A/B research not started',
        'git_status':'dedicated baseline commit pending; original index unchanged'})
    print(json.dumps({'status':'PASS','original_preserved':True,'evidence_files':len(evidence)}))

if __name__=='__main__':main()

"""Generate a portable derived-evidence package; never copy raw traces."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'releases/deepep32-r4-20261019';base=ROOT/'results/data-foundation'
    for slug in ['prediction-overview-r3','deepep-device-inventory-r1','deepep-fb-structure-r1','prediction-methods-r2']:
        shutil.copytree(base/slug,out/slug,dirs_exist_ok=True)
    model=out/'model';model.mkdir(exist_ok=True)
    for name in ['F.json','B.json','parameters.json','outer-prediction.json','expanded-dag.json.gz']:
        shutil.copy2(base/'deepep-stable-20261016-r1/model'/name,model/name)
    for name in ['model-seal.json','evaluation.json','inventory.json']:
        shutil.copy2(base/'deepep-stable-20261016-r1'/name,model/name)
    shutil.copy2(ROOT/'docs/data-foundation/DEEPEP_R4_FREEZE_20261019.md',out/'SOURCE_AND_REPRODUCTION.md')
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}
    assert not any('.pt.trace' in p for p in files)
    (out/'manifest.json').write_text(json.dumps({'scope':'portable frozen DAG replay and derived evidence; no raw trace','files':files},indent=2)+'\n')
    print('Packaged',len(files),'files')
if __name__=='__main__':main()

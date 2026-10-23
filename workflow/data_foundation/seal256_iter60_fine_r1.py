"""Archive code/adapters and existing raw-input hashes without scanning traces."""
import json,tarfile,subprocess
from datetime import datetime,timezone
from pathlib import Path
from calibrate256_fine_r1 import BASE,ROOT,load,save,sha
from assemble256_iter60_fine_r1 import select
def main():
    out=BASE/'iter60-fine-model-r1';selected,missing=select();assert not missing
    m=load(out/'manifest.json')
    for p,h in m['inputs'].items():assert sha(Path(p))==h,p
    for p,h in m['outputs'].items():assert sha(out/p)==h,p
    files=set((ROOT/'workflow/data_foundation').glob('*.py'));raw={};variants={}
    for (r,mb),report in selected.items():
        root=report.parent.parent;variants[f'{r}:{mb}']=str(report)
        files.update(root.glob('*.adapter.py'));files.update(root.glob('*.complete.json'))
        files.add(root/'pp32-f-split-r8.complete.json')
        for p,h in load(root/'pp32-f-split-r8.complete.json')['inputs'].items():
            if p.startswith('/var/local/'):raw[p]=h
    files.update((BASE/'calibrate256-zero-r2').glob('*.py'))
    with tarfile.open(out/'code-and-adapters.tar.gz','w:gz') as tar:
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(ROOT)))
    save(out/'provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),branch='32to256',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),uncommitted_candidate=True,code_files={str(p):sha(p) for p in sorted(files)},raw_input_hashes_from_completed_extractions=raw,selected_bindings=variants,archive_sha256=sha(out/'code-and-adapters.tar.gz'),notes=['Raw data not copied. Raw hashes inherited from completed extraction markers; not reread by this sealing step.','Frozen graph stdlib replay is self-contained. Re-extraction requires local raw data, original dependencies and recorded per-stage adapter snapshots.','Archived current workflow modules plus executed adapters and historical zero-work helpers; base Git SHA alone is insufficient.','Observed readiness and conditional edge repair remain part of calibration, not independent causal validation.']))
    print('PASS input/output hashes;',len(raw),'raw trace references;',len(files),'code/adapter/stage records')
if __name__=='__main__':main()

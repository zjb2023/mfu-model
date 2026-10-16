"""Freeze the three user-selected presentation artifacts; no experiments."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'
WEB = Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001')
OUT = BASE / 'current-1f1b-20261016'
PAGES = ['full155-device-inventory-r1', 'mccl-fb-structure-r1', 'prediction-methods-r1']

def main():
    OUT.mkdir(exist_ok=True)
    inventory = BASE / '2111-blocks-ui-r1/df-v001/full155-device-inventory-r1'
    inventory.mkdir(exist_ok=True)
    # The served inventory contains compact event payloads, not original traces.
    for p in sorted((WEB / PAGES[0]).iterdir()):
        if p.is_file():
            shutil.copyfile(p, inventory / p.name)
    paths = []
    for directory in [inventory, BASE / PAGES[1], BASE / PAGES[2]]:
        paths.extend(p for p in directory.iterdir() if p.is_file())
    for name in ['source80-fine-evaluation-r1', 'stable155-256-r1', 'full155-256-r1', 'hybrid-deepep-r1', 'hybrid-local-r3', 'ep-envelope-r1']:
        paths.append(BASE / name / 'report.json')
    paths.append(BASE / 'full155-device-inventory-r1/report.json')
    # Existing tracked legacy diagrams required by the version-switching page.
    for name in ['cp-ep8-structure-r1', 'b-four-layers-r10']:
        paths.append(BASE / '2111-blocks-ui-r1/df-v001' / name / 'index.html')
    records = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = dict(scope='Current 1F1B presentation/evidence checkpoint, not a complete prediction-pipeline freeze', files=records,
                    pages={n: f'http://192.168.8.16:43312/df-v001/{n}/' for n in PAGES},
                    raw_trace_included=False, new_experiments=False,
                    limits=['Other historical navigation targets are not copied into this checkpoint.', 'Offline fetch-based inventory needs a local HTTP server.', 'Prediction candidates retain their original limitations.'])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'Frozen {len(records)} files; manifest: {OUT / "manifest.json"}')

if __name__ == '__main__':
    main()

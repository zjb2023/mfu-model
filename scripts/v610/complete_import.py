"""Preserve optional analysis code and locally served historical URL aliases.

These supplement the 371-file audit; new inventory is never called an old seal.
"""
from pathlib import Path
import json
import shutil
from import_snapshot import ROOT, sha, put, git

source=Path('/home/zjb/Desktop/worktrees/fab-w37-1f1b')
assert git(source,'rev-parse','HEAD') == git(source,'rev-parse','b8330a0')
base=Path(json.loads((ROOT/'.local/v610.json').read_text())['artifact_root'])
m=json.loads((ROOT/'docs/v610/artifact_manifest.json').read_text())
code=json.loads((ROOT/'docs/v610/imported_code.json').read_text())
existing={r['path'] for r in code['files']}
for rel in git(source,'ls-files','case_256gpu_pp16_cp2_a2a/scripts').splitlines():
    dest=ROOT/'vendor/fab_w37'/rel
    if str(dest.relative_to(ROOT)) in existing: continue
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): assert sha(dest)==sha(source/rel), str(dest)
    else: shutil.copy2(source/rel,dest)
    code['files'].append(dict(path=str(dest.relative_to(ROOT)),source_path=str(source/rel),sha256=sha(dest)))
supplement=ROOT/'docs/v610/supplement_manifest.json'
extras=json.loads(supplement.read_text())['files'] if supplement.exists() else []
web=source/'results/w37/A/research-html-20260907'
paths=[f for f in web.iterdir() if f.is_file() and f.suffix in ('.html','.svg','.css','.js')]
paths.append(Path('/home/zjb/Desktop/fabric-data-analysis/case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_v684_blocking_pp_program_order_256_to_224/evaluator_only/dag_v684_blocking_pp_program_order.html'))
for src in sorted(paths):
    logical=str(src); relative='files/'+str(src.resolve()).lstrip('/')
    if logical in m['logical_paths']: continue
    dest=base/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): assert sha(dest)==sha(src)
    else: shutil.copy2(src,dest)
    r=dict(relative=relative,sha256=sha(dest),size_bytes=dest.stat().st_size)
    m['objects'].append(r);m['logical_paths'][logical]=relative
    extras.append(dict(source_path=logical,resolved_path=str(src.resolve()),**r,
                       role='history_served_alias',validation='new_inventory_not_prior_seal'))
put(ROOT/'docs/v610/artifact_manifest.json',m)
put(ROOT/'docs/v610/imported_code.json',code)
put(ROOT/'docs/v610/supplement_manifest.json',dict(files=extras,raw_reads=False))
print(json.dumps(dict(supplementary_artifacts=len(extras),code_files=len(code['files']))))

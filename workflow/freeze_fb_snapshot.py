"""Exact-scope import of the current untracked research snapshot; no raw traces."""
import ast,hashlib,json,shutil
from pathlib import Path
SRC=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610')
DST=Path(__file__).resolve().parents[1]
assert DST==Path('/home/zjb/Desktop/worktrees/mfu-16to256')
selected=set();excluded=[]
def add(p):
    assert p.is_file() and not p.is_symlink(),p
    assert '.pt.trace.json' not in p.name and '__pycache__' not in p.parts,p
    selected.add(p.relative_to(SRC))
for pattern in ['build_b_*.py','audit_b_*.py','build_four_layer_f_r1.py','evaluate_b_r10_60_to70.py','evaluate_four_layer_f_60_to70_r1.py','assemble_pp32_detailed_fb_r1.py','audit_pp2_module_compatibility_r1.py','extrapolate_detailed_fb_32to256_r1.py','pp_loss_slot_r1.py','ui_b*','ui_detailed_32to256_r1.html','ui_pp32_detailed_fb_r1.html','publish_b*','publish_detailed_32to256_r1.py','publish_pp32_detailed_fb_r1.py','test_b*','test_detailed_32to256_r1.cjs','test_pp32_detailed_fb_r1.cjs']:
    for p in (SRC/'workflow/data_foundation').glob(pattern):
        if p.is_file():add(p)
# Follow local Python imports so frozen template replay works without old worktree.
seen=set()
while True:
    pending=[p for p in selected-seen if p.suffix=='.py']
    if not pending:break
    for p in pending:
        seen.add(p)
        for n in ast.walk(ast.parse((SRC/p).read_text())):
            mods=([n.module] if isinstance(n,ast.ImportFrom) else [a.name for a in n.names] if isinstance(n,ast.Import) else [])
            for mod in mods:
                if mod:
                    local=SRC/p.parent/(mod.split('.')[0]+'.py')
                    if local.exists():add(local)
for pat in ['B_*.md','CP_EP_*.md','CP_EP_*.json','EP8_BACKBONE*.md','FORWARD_LAYER*.md','FOUR_LAYER*.md','PP32_*.md','DETAILED_FB_*.md']:
    for p in (SRC/'docs/data-foundation').glob(pat):add(p)
result_dirs=['four-layer-f-r1','b-four-layers-r10','pp32-detailed-fb-assembly-r1','detailed-fb-32to256-r1','pp2-module-compatibility-r1','b-r10-60to70-eval-r2','b-last-layer-variation-r1','four-layer-f-60to70-r1']
for name in result_dirs:
    for p in (SRC/'results/data-foundation'/name).glob('*.json'):
        if name in ['pp32-detailed-fb-assembly-r1','detailed-fb-32to256-r1'] and p.stem in ['graph','replay']:
            excluded.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason='large expanded artifact; regenerate from frozen templates'));continue
        add(p)
for name in ['pp32-minimal-r1/configs.json','pp32-to256-pipeline-r6/target-observation.json','pp32-gpu-boundary-r3b/parameters.json','pp32-gpu-boundary-r3b/source-observations.json','pp32-gpu-boundary-r3b/target-observations.json']:add(SRC/'results/data-foundation'/name)
ui_names=['cp-ep8-structure-r1','cp-ep8-structure-r2','four-layer-f-r1','four-layer-f-60to70-r1','b-cp-identity-r1','b-cost-blocks-r1','b-four-layers-r10','b-r10-60to70-r2','b-r10-60to70-r3','pp32-detailed-fb-r1','detailed-fb-32to256-r1']
for name in ui_names:
    for p in (SRC/'results/data-foundation/2111-blocks-ui-r1/df-v001'/name).rglob('*'):
        if p.is_file() and p.suffix in ['.html','.js','.css','.json']:add(p)
for name in ['b-four-layers-r10-ui3-check','b-r10-60to70-ui-check','b-last-layer-variation-ui-check','pp32-detailed-fb-ui-check','detailed-fb-32to256-ui-check']:
    add(SRC/'results/data-foundation'/name/'checks.json')
records=[]
for rel in sorted(selected):
    target=DST/rel;assert not target.exists(),target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SRC/rel,target)
    digest=hashlib.sha256(target.read_bytes()).hexdigest();assert digest==hashlib.sha256((SRC/rel).read_bytes()).hexdigest()
    records.append(dict(relative_path=str(rel),source_path=str(SRC/rel),sha256=digest,bytes=target.stat().st_size))
out=DST/'docs/data-foundation/FB_BRANCH_SNAPSHOT.json'
out.write_text(json.dumps(dict(version='fb-16to256-freeze-r1',source_branch='feat/w37-v610',source_commit='688740e',destination_branch='16to256',files=records,external_expanded_artifacts=excluded,raw_trace_policy='not copied; original read-only paths and hashes remain in provenance manifests',provenance_policy='historical absolute paths retained verbatim; snapshot relative_path is authoritative branch-local location'),ensure_ascii=False,indent=2)+'\n')
print('COPIED',len(records),'files',sum(x['bytes'] for x in records),'bytes')

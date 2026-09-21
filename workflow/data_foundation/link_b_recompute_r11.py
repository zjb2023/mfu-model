"""Add a candidate link to the existing page; preserve its graph and old HTML."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
folders=[ROOT/'results/data-foundation/2111-blocks-ui-r1/df-v001/b-four-layers-r10',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/b-four-layers-r10')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for folder in folders:
 p=folder/'index.html';s=p.read_text();assert 'id="r11-link"' not in s
 shutil.copyfile(p,folder/'index-before-r11-link.html');shutil.copyfile(folder/'manifest.json',folder/'manifest-before-r11-link.json')
 s=s.replace('<main>','<main><p id="r11-link" class="warning"><a href="../b-recompute-r11/">打开R11：重计算也已拆成可改计算/CP/EP成本的DAG</a>。本页保留R10，不改变旧模型。</p>',1);p.write_text(s)
 m=json.loads((folder/'manifest.json').read_text());m['version']='b-four-layers-r10-ui4-r11-link';m['link_patch_code']={str(Path(__file__).resolve()):sha(Path(__file__))};m['outputs']=[dict(path=str(p.resolve()),sha256=sha(p)) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'];(folder/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('PASS existing R10 now links to R11 candidate')

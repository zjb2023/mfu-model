"""Move overview before detail and explanations after plots; presentation only."""
import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FOLDERS=[ROOT/'results/data-foundation/2111-blocks-ui-r1/df-v001/b-recompute-r11',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/b-recompute-r11')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(FOLDERS[0]/'index.html')==sha(FOLDERS[1]/'index.html')
for folder in FOLDERS:
 p=folder/'index.html';old=p.read_text();html=old
 def section(s,identity):
  match=re.search(r'<section id="'+identity+r'">.*?</section>',s,re.S)
  assert match,identity
  return match.group()
 validation=section(html,'validation');explanation=section(html,'connection-costs');detail=section(html,'b-dag')
 cut=detail.index('<h2 id="dag-layer-caption">')
 overview=detail[:cut]+'</section>'
 backward='<section id="true-backward-dag">'+detail[cut:]
 html=html.replace(validation,'',1).replace(explanation,'',1).replace(detail,backward+explanation,1)
 html=html.replace('<section id="recompute-dag">',overview+'<section id="recompute-dag">',1)
 assert html.index('id="b-dag"')<html.index('id="recompute-dag"')<html.index('id="true-backward-dag"')<html.index('id="connection-costs"')
 assert '验证范围：新候选，不是新预测精度' not in html
 assert re.findall(r'<script\b[^>]*>.*?</script>',old,re.S)==re.findall(r'<script\b[^>]*>.*?</script>',html,re.S)
 for name,backup in [('index.html','index-before-ui3.html'),('manifest.json','manifest-before-ui3.json')]:
  assert not (folder/backup).exists();shutil.copyfile(folder/name,folder/backup)
 frozen={n:sha(folder/n) for n in ['graph.json','replay.json','cost-bindings.json','summary.json']}
 p.write_text(html)
 manifest=dict(version='b-recompute-r11-ui3',scope='layout only; validation panel removed, model validation artifacts retained',previous_manifest=str(folder/'manifest-before-ui3.json'),unchanged_model_hashes=frozen,code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PASS UI3: overview, recompute, true backward, connection explanations; JS unchanged')

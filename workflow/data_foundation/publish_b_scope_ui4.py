"""Label-only UI revision; keep graph and numerical payload byte-identical."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/data-foundation/2111-blocks-ui-r1/df-v001/b-four-layers-r10'
LIVE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/b-four-layers-r10')
PANEL='''<section id="b-scope"><h2>先看完整单层B：重计算＋真反向</h2><div class="flow"><div class="box reserved"><strong>① 前向重计算 · 当前保留为实测包络</strong><small>Attention＋CP → MoE前向：Dispatch → 专家FC1/FC2 → Combine</small><p>恢复求梯度需要的中间激活，功能流程与前向一致；当前未直接复用细粒度F DAG，也不假定耗时与原前向相同。</p></div><div class="box"><strong>② 真反向 · 下方两张八卡图展开的部分</strong><small>MoE反向 → Attention反向＋CP</small><p>计算并传递梯度。“MoE反向”和“Attention反向”都属于这里，前者不是重计算。</p></div></div><p>执行关系：①重计算 → ②真反向 → 下一层的重计算。箭头表示功能顺序，不是新增跨卡屏障。完整B还包含已建模的衔接与尾部；页面总B时间已经计入重计算，不能再额外叠加一次F成本。</p><small>UI4仅澄清展示范围，原模型参数、DAG及默认回放时间不变。</small></section>'''
OLD='八卡实际DAG投影（隐藏就绪/本地处理节点，点击箭头可追溯）'
NEW='仅真反向的八卡DAG投影（前置重计算以包络保留；隐藏就绪/本地处理节点，点击箭头可追溯）'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OUT/'index.html')==sha(LIVE/'index.html')
old=(OUT/'index.html').read_text();assert old.count(OLD)==1 and 'id="b-scope"' not in old
new=old.replace(OLD,NEW).replace('<section id="b-dag">',PANEL+'<section id="b-dag">',1)
new=new.replace('<h3>MoE反向 ·','<h3>真反向①：MoE反向 ·',1).replace('<h3>Attention反向 ·','<h3>真反向②：Attention反向 ·',1)
for folder in [OUT,LIVE]:
 assert not (folder/'index-before-ui4.html').exists()
 shutil.copyfile(folder/'index.html',folder/'index-before-ui4.html')
 shutil.copyfile(folder/'manifest.json',folder/'manifest-before-ui4.json')
 (folder/'index.html').write_text(new)
 manifest=dict(version='b-four-layers-r10-ui4',scope='labels and explanatory flow only; numerical payload and graph unchanged',
   inputs=[dict(path=str(folder/'index-before-ui4.html'),sha256=sha(folder/'index-before-ui4.html'))],
   code=[dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__)))],
   outputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'])
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert sha(OUT/'index.html')==sha(LIVE/'index.html')
print('PASS UI4 published; previous page and manifest archived; graph unchanged')

"""UI-only clarification of existing graph connections and cost accounting."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FOLDERS=[ROOT/'results/data-foundation/2111-blocks-ui-r1/df-v001/b-recompute-r11',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/b-recompute-r11')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
PANEL='''<section id="connection-costs"><h2>图与图如何连接，连接处如何计费？</h2>
<p><strong>完整单层B：</strong>重计算① Attention＋CP → 重计算② MoE → MoE真反向 → Attention真反向。重计算恢复前向激活，不计算梯度；“梯度处理出口”属于真反向。</p>
<div class="scroll"><table><thead><tr><th>连接位置</th><th>图中折叠的内容与成本</th></tr></thead><tbody>
<tr><td>重计算① → 重计算②</td><td>CP结束后还有本地处理、Router及分发准备，并非直接进入EP8。已建模的计算/本地处理节点各自计费，其余就绪间隔保留trace参数。</td></tr>
<tr><td>重计算完成 → MoE真反向</td><td>重计算出口完成后，经已有衔接/反向入口处理进入真反向。重计算已经计入B，不再额外叠加一次F成本。</td></tr>
<tr><td>MoE真反向 → Attention真反向</td><td>EP梯度合并及本地梯度处理完成后进入Attention反向。“梯度处理出口”与“Attention入口”是同一连接边界，已计费的处理不重复加入。</td></tr>
</tbody></table></div>
<p><strong>默认连接间隔：</strong>trace中的下一节点开始时间 − 最晚前驱完成时间。构图时校验该间隔非负（允许浮点舍入误差）。</p>
<p><strong>替换成本后的调度：</strong>下一节点开始 = 修改后最晚前驱完成时间 ＋ 保留的连接间隔。不是继续固定trace绝对开始时刻；上游变慢会沿依赖传播，重叠和已有时间余量可能吸收部分变化。</p>
<p>若一条投影箭头折叠了多个节点，按完整DAG逐节点计算，不能再给箭头统一加一次费用。CP2/EP8有效通信已在专门节点计费；连接处不重复收取通信成本。保留间隔是经验参数，不全部等于纯等待。</p>
<small>UI2：仅补充解释，节点、依赖、参数和默认回放均不改变。</small></section>'''
assert sha(FOLDERS[0]/'index.html')==sha(FOLDERS[1]/'index.html')
for folder in FOLDERS:
 p=folder/'index.html';html=p.read_text();assert 'id="connection-costs"' not in html
 frozen={n:sha(folder/n) for n in ['graph.json','replay.json','cost-bindings.json','summary.json']}
 for name in ['index.html','manifest.json']:
  backup=folder/('index-before-ui2.html' if name=='index.html' else 'manifest-before-ui2.json');assert not backup.exists();shutil.copyfile(folder/name,backup)
 html=html.replace('<section id="recompute-dag">',PANEL+'<section id="recompute-dag">',1)
 html=html.replace('<h3>重计算②：','<p><strong>连接①→②：</strong>上图CP及本地处理完成后，经过Router/分发准备进入下图EP8；折叠节点仍在DAG中计费。</p><h3>重计算②：',1)
 html=html.replace('<h3>MoE反向 ·','<p><strong>连接重计算→真反向：</strong>先完成上面的重计算子图，再经衔接节点进入下面的MoE梯度计算；不是另一次前向。</p><h3>MoE反向 ·',1)
 html=html.replace('<h3>Attention反向 ·','<p><strong>连接MoE反向→Attention反向：</strong>上图“梯度处理出口”接下图“Attention入口”；这里属于真反向，已计费的梯度处理不再重复相加。</p><h3>Attention反向 ·',1)
 p.write_text(html)
 assert frozen=={n:sha(folder/n) for n in frozen}
 manifest=dict(version='b-recompute-r11-ui2',scope='connection explanations only',previous_manifest=str(folder/'manifest-before-ui2.json'),code={str(Path(__file__).resolve()):sha(Path(__file__))},unchanged_model_hashes=frozen,outputs={str(p.resolve()):sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PASS UI2 connection explanations published; model hashes unchanged')

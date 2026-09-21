"""Publish new candidate without replacing r10 model or the frozen prediction."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';SRC=BASE/'b-recompute-r11'
OUT=BASE/'2111-blocks-ui-r1/df-v001/b-recompute-r11'
LIVE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/b-recompute-r11')
template=Path(__file__).with_name('ui_b_four_layers_r10.html');controller=Path(__file__).with_name('ui_b_dag_r2.js');recui=Path(__file__).with_name('ui_b_recompute_r11.js')
assert not OUT.exists() and not LIVE.exists()
payload={k:json.loads((SRC/(k+'.json')).read_text()) for k in ['graph','replay','summary']}
html=template.read_text()
html=html.replace('<h1>一个 B，四层逆序执行</h1>','<h1>R11：重计算＋真反向，均可接入成本</h1><p><a href="../b-four-layers-r10/">查看保留的R10版本</a> · <a href="cost-bindings.json">完整成本接口与口径</a></p>')
html=html.replace('重计算及反向首尾保留trace参数','重计算的计算/通信块可替换；本地重排、就绪残余及反向首尾仍保留trace参数')
start=html.index('<section id="validation">');end=html.index('<section id="b-dag">',start)
html=html[:start]+'''<section id="validation"><h2>验证范围：新候选，不是新预测精度</h2><p>默认回放与R10全部原节点完成点保持一致，B为868.833 ms；32个重计算包络已替换为零成本出口，原整段成本不再叠加。计算、CP2、EP8扰动检查通过。</p><p>当前只完成源32卡iter60 B0回放与敏感性检查。R11尚未重新评价iter70/256，旧R10验证数字不能作为新拓扑的独立验证。</p></section>
<section id="recompute-dag"><h2 id="recompute-caption"></h2><p>每层先执行此处重计算，再执行下面的MoE真反向与Attention真反向。成本来自重计算自身trace，不复制原F0时长。点击节点可修改计算或通信有效成本；只读的重排/就绪节点在图中折叠，箭头可追溯原路径。</p><h3>重计算①：Attention前向＋四次CP2</h3><div class="scroll"><svg id="recompute-attention" style="width:100%;min-width:1250px" aria-label="重计算Attention与CP2"></svg></div><h3>重计算②：MoE分发 → FC1/FC2 → EP8合并</h3><div class="scroll"><svg id="recompute-moe" style="width:100%;min-width:1300px" aria-label="重计算MoE八卡依赖"></svg></div><p class="warning">CP/EP端口替换的是包含同步/处理的有效成本，不可将后端纯service时间直接叠加。EP分发同步有效项与各卡本地处理分别计费；Combine使用八卡本地准备完成后的有效残余。这是条件耦合，不证明实际存在全组硬屏障。</p></section>'''+html[end:]
html=html.replace('前向重计算<small>整段trace保留，不额外叠加F成本','前向重计算<small>细粒度成本子图，不再保留整段包络，不额外叠加F成本')
html=html.replace('重计算时长（保留）','重计算子图时长').replace('重计算暂未成本化','重计算计算/通信成本已开放，经验就绪参数仍保留')
html=html.replace('rec.cost_ms.toFixed(2)', '(current[`r${rank}:u${u}:recompute`].end_ms-current[`r${rank}:u${u}:recompute:handoff`].end_ms).toFixed(2)')
html=html.replace('G[`r${rank}:u${u}:recompute`].cost_ms.toFixed(3)','(current[`r${rank}:u${u}:recompute`].end_ms-current[`r${rank}:u${u}:recompute:handoff`].end_ms).toFixed(3)')
html=html.replace('if(!k.startsWith(`U${unit}:`)||','if(!(k.startsWith(`U${unit}:`)||k.startsWith(`R${unit}:`))||')
html=html.replace("return Object.entries(o).filter(([k])=>", "return Object.entries(o).filter(([k,n])=>(!n.members||n.members.includes(rank))&&")
html=html.replace('renderBDag();}', 'renderBDag();renderRecompute();}')
html=html.replace('<a href="coverage.json">首尾覆盖范围</a>','<a href="coupling.json">重计算跨卡耦合</a>')
control=controller.read_text().replace("type:'reserved',unit:u}","type:'reserved',unit:u,boundary:true}")
control=control.replace('前置重计算以包络保留','重计算子图见上方')
html=html.replace('__DAG_CONTROLLER__',control+'\n'+recui.read_text()).replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False))
OUT.mkdir(parents=True)
for n in ['graph','replay','summary','tests','cost-bindings','coupling']:shutil.copyfile(SRC/(n+'.json'),OUT/(n+'.json'))
(OUT/'index.html').write_text(html)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(dict(version='b-recompute-r11-ui1',inputs={str(SRC/'manifest.json'):sha(SRC/'manifest.json')},code={str(p):sha(p) for p in [template,controller,recui,Path(__file__).resolve()]},outputs={str(p.resolve()):sha(p) for p in OUT.iterdir()}),indent=2)+'\n')
shutil.copytree(OUT,LIVE)
print('PASS published http://192.168.8.16:43312/df-v001/b-recompute-r11/')

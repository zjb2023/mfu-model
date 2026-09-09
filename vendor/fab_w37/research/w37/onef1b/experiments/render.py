"""Render the reviewed experiment proposal; no training or trace access."""
from pathlib import Path
import csv
import html
import json
import sys
from visuals16 import render as render_visuals16

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'results/w37/A/research-html-20260907/python-deps'))
import markdown

DOC = ROOT / 'docs/w37/1f1b'
rows = list(csv.DictReader((DOC / 'EXPERIMENTS_32PLUS.csv').open()))
rows.sort(key=lambda r: int(r['gpus']))
for r in rows:
    r['recommended'] = r['purpose'].startswith('核心') or r['experiment_id'] in {'V128', 'D224', 'D256', 'V224-N', 'V256-N'}
for r in rows:
    assert int(r['gpus']) >= 32
    assert int(r['gpus']) == int(r['pp']) * int(r['cp']) * int(r['tp']) * int(r['dp'])
    assert int(r['global_batch_size']) == int(r['micro_batch_size']) * int(r['microbatches']) * int(r['dp'])
    assert r['deepep_log'] == r['mtlink_capture'] == 'off'
assert len(rows) == 18 and sum(r['recommended'] for r in rows) == 11
md = markdown.Markdown(extensions=['extra', 'toc', 'sane_lists'])
body = md.convert((DOC / 'EXPERIMENTS_32PLUS.md').read_text())
body = body.replace('<table>', '<div class="scroll"><table>').replace('</table>', '</table></div>')
payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
sixteen_rows = list(csv.DictReader((DOC / 'EXPERIMENTS_16GPU.csv').open()))
assert len(sixteen_rows) == 14
for r in sixteen_rows:
    assert int(r['gpus']) == int(r['tp']) * int(r['pp']) * int(r['cp']) * int(r['dp']) == 16
    assert int(r['global_batch_size']) == int(r['micro_batch_size']) * int(r['dp']) * int(r['microbatches'])
source_info = markdown.markdown((DOC / 'EXPERIMENT_DATA_SOURCES.md').read_text(), extensions=['extra', 'sane_lists'])
source_info = source_info.replace('href="v610/config.json"', 'href="/docs/w37/1f1b/v610/config.json"').replace('href="binding/inputs.json"', 'href="/docs/w37/1f1b/binding/inputs.json"').replace('href="../coordination/source16_inventory.json"', 'href="/docs/w37/coordination/source16_inventory.json"')
source_info = source_info.replace('<h1>', '<h2>').replace('</h1>', '</h2>').replace('<table>', '<div class="scroll"><table>').replace('</table>', '</table></div>')
sixteen = markdown.markdown((DOC / 'EXPERIMENTS_16GPU.md').read_text(), extensions=['extra', 'sane_lists'])
sixteen = sixteen.replace('<h1>', '<h2>').replace('</h1>', '</h2>')
sixteen = sixteen.replace('<table>', '<div class="scroll"><table>').replace('</table>', '</table></div>')
page = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S5000 · MFU 外推实验方案</title><style>
:root{--ink:#17334b;--muted:#546a7a;--line:#dbe5ed;--teal:#006f77;--bg:#f1f5f8}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:80px}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.75 system-ui,-apple-system,"Microsoft YaHei",sans-serif}a{color:var(--teal);text-underline-offset:3px}header{background:#15374c;color:white;padding:48px max(24px,calc((100vw - 1220px)/2))}header p{max-width:880px;color:#d6e7ed}h1{font-size:36px;line-height:1.3;margin:14px 0}h2{font-size:25px;line-height:1.45}h3{font-size:19px}.eyebrow{color:#96dfd4;letter-spacing:2px;font-size:13px}.tags{display:flex;gap:10px;flex-wrap:wrap}.tags span{padding:4px 12px;border:1px solid #587383;border-radius:20px;font-size:13px}nav{position:sticky;top:0;background:#fffefb;border-bottom:1px solid var(--line);z-index:3;padding:12px 20px;display:flex;gap:22px;justify-content:center;flex-wrap:wrap}nav a{text-decoration:none;font-size:14px}main{max-width:1260px;margin:24px auto;padding:0 20px}section{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;margin:20px 0}.cards,.route{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.card,.route>div{padding:20px;background:#f1f7f8;border-radius:12px}.card strong{font-size:28px;display:block;color:var(--teal)}.route b{display:block;font-size:18px}.muted{color:var(--muted);font-size:14px}.note{border-left:4px solid #cf8b2e;background:#fff7e9;padding:12px 18px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf4f6;white-space:nowrap}#matrix table{min-width:1140px}#matrix td:last-child{min-width:230px}code{background:#edf3f7;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{white-space:pre-wrap}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:18px 0}button,select,input{font:inherit;padding:8px 12px;border:1px solid #9db6c2;border-radius:7px;background:white;color:var(--ink)}button{cursor:pointer}button[aria-pressed="true"]{background:var(--teal);color:white}input{max-width:100%;min-width:240px}.pill{display:inline-block;border-radius:4px;background:#e6f3ee;padding:2px 7px;font-size:12px;white-space:nowrap}.pill.target{background:#fff0dd}.pill.hold{background:#eaf0ff}details{margin:14px 0}summary{cursor:pointer;font-weight:650}#fulltext h1{font-size:25px}#fulltext h2{border-top:1px solid var(--line);padding-top:24px}#fulltext .toc{background:#f4f7fa;padding:16px;border-radius:10px}footer{padding:20px;text-align:center;color:var(--muted);font-size:13px}@media(max-width:760px){h1{font-size:28px}header{padding:30px 20px}.cards,.route{grid-template-columns:1fr}section{padding:18px}nav{position:static;justify-content:flex-start;gap:12px}main{padding:0 10px}input{min-width:0;width:100%}html{scroll-padding-top:12px}}@media print{nav,.controls{display:none}section{break-inside:auto;border:0}header{background:white;color:black}header p{color:black}.scroll{overflow:visible}#matrix table{min-width:0;font-size:10px}}
</style></head><body>
<header><div class="eyebrow">实验设计 · 2026-09-08</div><h1>S5000 大规模训练<br>MFU 外推实验方案</h1><p>先采集能复用的计算与通信规律，再验证它们能否组合成完整训练迭代。每一组实验都对应一个需要解释的变化，而不只是增加一个卡数点。</p><div class="tags"><span>S5000 · 80 GB/卡</span><span>8 卡/机</span><span>Spine–leaf 网络</span><span>方案待执行 · 显存尚未实测通过</span></div></header>
<nav><a href="#roadmap">实验路线</a><a href="#matrix">配置矩阵</a><a href="#telemetry">采集什么</a><a href="#memory">显存准入</a><a href="#fulltext">完整说明</a><a href="/research.html">已有模型报告</a></nav>
<main>
<section id="roadmap"><h2>01 · 从小规模采集，到大规模验证</h2><div class="cards"><div class="card"><strong>11 组</strong>32卡以上分阶段推荐配置</div><div class="card"><strong>18 组</strong>32卡以上全部候选；16卡另列14组</div><div class="card"><strong>≤70 GB</strong>每 rank 显存准入上限，需实测</div></div><p class="muted">上述组数不含独立重启重复和无profiler/短profiler计时配对；16卡的14组在上方单独列出。</p><div class="route"><div><b>① 1–16 卡：拆开测</b><p>算子 F/B、重计算；MoE 分发与合并；机内/机间通信、到达差和资源竞争。</p></div><div><b>② 16–64 卡：组合测</b><p>分别变化微批、CP、层数和 PP。保留原层内尺寸，验证局部成本如何组成完整 iter。</p></div><div><b>③ 128–256 卡：封存后测</b><p>先固定模型与预测，再评分。已有 224/256 案例继续标为开发回归。</p></div></div><p class="note">首轮主路线是 ≤64 卡多源校准 → 大规模预测。严格 ≤16 卡来源的对照单独报告；不能把加入 32/64 卡成本后的结果称为“16→256”。</p></section>
<section id="matrix"><h2>02 · 32卡以上实验组：承接16卡，按卡数递增</h2><p>保留序列8192和原模型层内尺寸；新增B32-P2、B32-EP4、B32-TP2桥接。表中显式列出全部并行维度，TP/EP不再假定全部相同。每个实际执行配置采全部rank的2–3个完整profiler迭代，先评估数据量；不采DeepEP日志或MTLINK，不新增NIC采集。</p><details><summary>先看名词：PP / CP / DP / MBS / MB / GBS 是什么？</summary><p><b>PP</b>：按层切成多少个流水阶段；<b>CP</b>：序列在多少卡间切分；<b>DP</b>：数据并行副本数；<b>TP</b>：层内张量切分；<b>EP</b>：专家并行组大小。</p><p><b>MBS</b>：每个微批的样本数；<b>MB</b>：每迭代、每数据并行副本执行的微批次数；<b>GBS</b>：全局批次样本数。当前约定：卡数 = TP × PP × CP × DP；GBS = MBS × DP × MB。EP 不再额外乘到卡数上。</p><p>B 表示桥接/校准候选，D 表示已知目标的开发回归，V 表示留出候选。数字表示卡数，后缀区分配置；V 只有经过历史使用核查与数据隔离后才具备留出资格。</p></details>
<div class="controls"><button id="recommended" aria-pressed="true">分阶段推荐 11 组</button><button id="all" aria-pressed="false">全部 18 组</button><label>数据用途 <select id="role"><option value="">全部用途</option><option value="calibration_candidate">源侧校准</option><option value="development_regression">开发回归</option><option value="heldout_candidate">留出候选</option></select></label><input id="search" aria-label="搜索配置" placeholder="搜索编号、卡数或实验目的"><button id="reset">重置</button></div><p id="count" class="muted" aria-live="polite"></p><div class="scroll"><table><thead><tr><th>实验</th><th>卡 / 机器</th><th>PP / CP / TP / DP / EP / ETP</th><th>层数</th><th>MBS / MB</th><th>GBS</th><th>数据用途</th><th>为什么做</th></tr></thead><tbody id="rows"></tbody></table></div><p class="note">所有行均为待执行候选，尚未实测显存和精度。224/256 卡的新 MB2 配置需核查历史未用于开发，再决定是否可以留出。增加层数与 PP 的主阶梯是联合外推，不是固定同一模型的强扩展。</p><p><a href="/docs/w37/1f1b/EXPERIMENTS_32PLUS.csv" download>下载32卡以上配置 CSV</a> · <a href="/docs/w37/1f1b/EXPERIMENTS_32PLUS.md">打开32卡以上方案说明</a></p></section>
<section id="telemetry"><h2>03 · 本轮采集要求：以 Profiler 为主，停止额外流量与日志采集</h2><div class="route"><div><b>必需</b><p>训练配置、训练时间日志、每配置2–3个完整迭代、该配置全部rank的profiler trace。显存复用已有框架峰值统计。</p></div><div><b>明确不采</b><p>MTLINK与DeepEP额外日志。保留训练原通信实现及profiler中的通信事件。</p></div><div><b>不新增</b><p>NIC采集、逐分配内存追踪、额外token日志。现成NIC数据可归档，但当前验收不依赖它。</p></div></div><p class="note">不再执行原方案的六种采集开关对照。先在02核验profiler扰动；没有token或链路证据时，不量化专家路由偏斜或物理链路流量贡献。</p></section>
<section id="memory"><h2>04 · 显存准入与运行边界</h2><p>默认每 rank 上限为 <code>min(70 GB, 实际总显存 − 10 GB)</code>，这里 GB 是十进制；70 GB ≈ 65.19 GiB。</p><div class="route"><div><b>先估算</b><p>权重、梯度、优化器、激活、通信 buffer、workspace 分账；专家参数按真实分片组计算。</p></div><div><b>再跑完整小负载</b><p>覆盖初始化、F/B 与首次 optimizer step。检查所有 rank，而不是仅 rank0。</p></div><div><b>逐级升到目标</b><p>确认路由偏斜与流水激活驻留后仍有余量。改变 MBS、序列或重计算须生成新配置。</p></div></div><p class="note">小卡数优先减少层数，保留原层内尺寸。top-k=6 不意味着只存 6 个专家；降低 MBS 也无法解决权重与优化器状态本身超限。</p><p>采集管线使用独立 Snakemake 入口：配置清单 → 显存准入 → 源侧采集 → 特征提取 → 拟合 → 预测封存 → 独立目标评分。普通报告生成不隐式申请 GPU。</p></section>
<section id="fulltext"><h2>05 · 完整方案与依据</h2><p>当前32卡以上的完整参数、桥接关系与Profiler采集要求；与上方表格和CSV一致。</p><details id="full-details"><summary>展开完整设计文档</summary>__TOC____BODY__</details></section>
</main><footer>设计方案，未执行训练或采集。生成入口：research/w37/onef1b/experiments/render.py</footer>
<script id="dataset" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('dataset').textContent);let recommended=true;
const names={calibration_candidate:'源侧校准候选',development_regression:'开发回归',heldout_candidate:'留出候选'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const role=document.getElementById('role').value,q=document.getElementById('search').value.trim().toLowerCase();const shown=data.filter(r=>(!recommended||r.recommended)&&(!role||r.data_role===role)&&(!q||Object.values(r).join(' ').toLowerCase().includes(q)));document.getElementById('rows').innerHTML=shown.map(r=>`<tr><td><b>${esc(r.experiment_id)}</b></td><td>${r.gpus} / ${r.gpus/8}</td><td>${r.pp} / ${r.cp} / ${r.tp} / ${r.dp} / ${r.ep} / ${r.expert_tp}</td><td>${r.layers}</td><td>${r.micro_batch_size} / ${r.microbatches}</td><td>${r.global_batch_size}</td><td><span class="pill ${r.data_role==='heldout_candidate'?'hold':r.data_role==='development_regression'?'target':''}">${names[r.data_role]}</span></td><td>${esc(r.purpose)}</td></tr>`).join('');document.getElementById('count').textContent=`当前显示 ${shown.length} 组 / 全部 18 组${shown.length?'':' · 没有匹配配置，请重置筛选'}`;document.getElementById('recommended').setAttribute('aria-pressed',recommended);document.getElementById('all').setAttribute('aria-pressed',!recommended)}
document.getElementById('recommended').onclick=()=>{recommended=true;render()};document.getElementById('all').onclick=()=>{recommended=false;render()};document.getElementById('role').onchange=render;document.getElementById('search').oninput=render;document.getElementById('reset').onclick=()=>{recommended=true;document.getElementById('role').value='';document.getElementById('search').value='';render()};render();
document.querySelectorAll('#full-details a[href^="#"]').forEach(a=>a.onclick=()=>document.getElementById('full-details').open=true);
</script></body></html>'''
page = page.replace('__TOC__', md.toc).replace('__BODY__', body).replace('__DATA__', payload)
page = page.replace('<nav>', '<nav><a href="#sixteen">优先：16 卡专项</a>', 1)
page = page.replace('<main>', '<main><section id="sixteen"><p class="note"><b>当前时间有限：优先 16 卡。</b>16卡原10组与4个并行候选合并为14组，优先6组；后续32卡以上另列18组，不再重复16卡配置。</p>' + '<div id="data-sources"><h2>先看数据来源：Profiler 是主证据</h2><p class="note">256 卡 trace 派生表提供当前主路线的计算成本；224 卡用于开发评分；16 卡是独立小源数据。新16卡实验每个配置都采短profiler窗口；本轮不采DeepEP日志、MTLINK，不新增NIC采集。</p><details><summary>展开三个 case 的数据来源与修正后的采集要求</summary>' + source_info + '</details></div>' + render_visuals16(sixteen_rows) + '<div id="sixteen-doc">' + sixteen + '</div><p><a href="/docs/w37/1f1b/EXPERIMENTS_16GPU.csv" download>下载 16 卡专项 CSV</a> · <a href="/docs/w37/1f1b/EXPERIMENTS_16GPU.md">打开 16 卡专项文档</a></p></section>', 1)
out = ROOT / 'results/w37/A/extrapolation-plan'
out.mkdir(parents=True, exist_ok=True)
(out / 'index.html').write_text(page, encoding='utf-8')
(DOC / 'EXTRAPOLATION_EXPERIMENT_PLAN.html').write_text(page, encoding='utf-8')
print('Rendered 14 small configurations and 18 large configurations (11 recommended); HTML in docs and results.')

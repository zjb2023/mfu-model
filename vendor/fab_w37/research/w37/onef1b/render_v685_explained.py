"""Extend reproduced v685 HTML using sealed results, without altering the baseline."""
from pathlib import Path
import csv, hashlib, html, json, re
from statistics import mean
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/w37/A/research-html-20260907'
BASE=ROOT/'results/w37/A/reproduce-v685-preparation/reproduced'
source=BASE/'evaluator_only/dag_v685_source_steady_calibration.html'
inputs={}
def read(p):
    b=p.read_bytes();inputs[str(p)]=hashlib.sha256(b).hexdigest();return b.decode()
def rows(p):return list(csv.DictReader(read(p).splitlines()))
def esc(s):return html.escape(str(s))
def table(head,data):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in data)+'</tbody></table></div>'
s=read(source)
payload=json.loads(re.search(r'window.DAG_V685=(.*?);</script>',s).group(1))
p=payload['prediction'];m=payload['metrics']
it=[r for r in rows(ROOT/'docs/w37/1f1b/post685/delivery/version_iteration_results.csv') if r['variant']=='v685_frozen' and r['split']=='development_primary']
ledger=[r for r in rows(ROOT/'results/w37/A/post685-candidate-readiness-r1/evaluator_only/phase_results.csv') if r['variant']=='v685_frozen' and r['split']=='development_primary']
assert len(it)==4 and len(ledger)==16
phase_names={'entry':'迭代准备','onef1b':'全 rank 的 1F1B 区间','tail':'更新与通信收尾','outer':'日志与 Profiler 计时差额'}
summary=[]
for phase,label in phase_names.items():
    g=[r for r in ledger if r['phase']==phase]
    pred=mean(float(r['predicted_ms']) for r in g);actual=mean(float(r['actual_ms']) for r in g)
    summary.append((phase,label,pred,actual,actual-pred,mean(float(r['ape_pct']) for r in g)))
assert abs(sum(r[2] for r in summary)-p['training_step_ms'])<1e-6
assert abs(sum(r[3] for r in summary)-mean(float(r['actual_training_ms']) for r in it))<1e-6
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 385" role="img" aria-label="v685预测与目标四轮实际均值的迭代时间分段"><rect width="1180" height="385" fill="#0b1b2b"/>']
def text(x,y,t,size=18):svg.append(f'<text x="{x}" y="{y}" fill="#e9f2fb" font-family="sans-serif" font-size="{size}">{esc(t)}</text>')
text(25,36,'v685 目标 224 卡：预测与实际平均迭代时间（85 / 90 / 95 / 100）',23)
colors=['#dfb65e','#36a6c9','#a58cde','#8a9ba9']
for j,(label,k) in enumerate([('模型预测',2),('实际四轮均值',3)]):
    y=100+j*100;text(25,y+25,label)
    start=210
    for r,color in zip(summary,colors):
        width=r[k]/1000*32
        svg.append(f'<rect x="{start}" y="{y}" width="{width}" height="42" fill="{color}"><title>{esc(r[1])}：{r[k]/1000:.3f} 秒</title></rect>');start+=width
    text(start+12,y+27,f'{sum(r[k] for r in summary)/1000:.3f} s')
for value in [0,5,10,15,20,25]:text(210+value*32,280,str(value),15)
text(600,307,'相对迭代起点（秒）',17)
for i,(r,color) in enumerate(zip(summary,colors)):
    x=25+i*285;svg.append(f'<rect x="{x}" y="335" width="15" height="15" fill="{color}"/>');text(x+23,348,r[1],16)
svg.append('</svg>')
(OUT/'v685_phase_overview.svg').write_text(''.join(svg),encoding='utf-8')
extra='<section class="panel" id="model"><h2>计算图怎样得到目标迭代时间</h2><p>主任务是从源侧校准成本，结合目标静态配置预测 224 卡。图覆盖全 224 rank，下面的分段对照是全局区间，不是 lane0，也不是各 GPU 耗时求和。</p><div class="flow"><div class="node">源侧 256 卡成本与调度</div><span class="arrow">→</span><div class="node">224 卡目标依赖图</div><span class="arrow">→</span><div class="node">按依赖求完成时刻</div><span class="arrow">→</span><div class="node">补偿项 → Training 时间 → MFU</div></div>'
extra+='<p>同一节点必须等前置依赖完成；并行分支取较晚者，再加本节点耗时。已有图中的计算、通信和等待按其成本口径计入，不能把互相覆盖的区间重复相加。</p></section>'
extra+='<section class="panel" id="topology"><h2>继承 v684：依赖结构与本地执行顺序</h2><div class="flow"><div class="node">本 rank 的 F/B 动作</div><span class="arrow">→</span><div class="node">阻塞式 PP 发送 / 接收完成</div><span class="arrow">→</span><div class="node">程序顺序门</div><span class="arrow">→</span><div class="node">本 rank 后续 F/B</div></div><p>rank 是训练进程编号，本配置每个 rank 对应一张 GPU。图包含流水线前向激活依赖、反向梯度依赖及本地程序顺序；没有把所有 rank 强制同步到同一全局步骤。</p>'
extra+=table(['结构项','v685 当前值'],[['目标节点数','327,746'],['目标边数','364,784'],['相对 v684 的拓扑变化','0；本版本更新成本，沿用正式依赖锁'],['拓扑 SHA256',payload['dependency_topology_sha256']]])
extra+='<p>v687 的“有效发送就绪约 93 ms”是后续独立研究候选的解释，不是本页 v685 新增的模型结构。</p></section>'
extra+='<section class="panel" id="timing"><h2>预测总时间与实际时间：差额在哪里</h2>'+''.join(svg)
extra+='<p>前三段合成 Profiler 区间；再加 outer 得到完整 Training iter。图按四轮均值展示；1F1B 包含内部计算、通信及等待，颜色不是 GPU 活跃率。outer 放在末尾仅表示加法分账，不代表它在真实时间线上集中发生于末尾。预测收尾段按 Profiler 总时间减准备与 1F1B 得到，包含残差补偿的账面影响，不是纯优化器耗时。</p>'
extra+=table(['时间段','预测（秒）','实际均值（秒）','实际 − 预测（ms）','该段 MAPE（%）'],[[r[1],f'{r[2]/1000:.3f}',f'{r[3]/1000:.3f}',f'{r[4]:+.3f}',f'{r[5]:.3f}'] for r in summary])
extra+='<p>差额为正表示低估，负表示高估。前三段差额相加得到 Profiler 低估，四段相加得到 Training 低估；各段 MAPE 的分母不同，不能相加。</p></section>'
extra+='<section class="panel" id="corrections"><h2>图外补偿来自哪里</h2>'
extra+=table(['项','数值','来源与使用'],[['目标原始图时间',f"{p['target_raw_graph_ms']:.6f} ms",'目标依赖图回放结果'],['源侧未分类残差',f"{p['source_reconciliation_ms']:.6f} ms",'源侧 Profiler 参考中位数 − 源图回放'],['目标残差补偿',f"{p['target_reconciliation_ms']:.6f} ms",'源残差 × microbatch 比例 3/4；加到目标图时间得到 Profiler 时间'],['outer',f"{p['outer_framework_ms']:.6f} ms",'源 256 卡 60、65、…、100 九轮 log − Profiler 差额的中位数；v5.4 计算、后续继承']])
extra+='<p>残差补偿约 604 ms 与 outer 约 1372 ms 是两个不同项。前者补在 Profiler 口径内；后者衔接 Profiler 与日志完整 iter。均未独立拆解为具体物理操作，outer 不能称为开启 profiler 的采集开销。<a href="/docs/w37/1f1b/OUTER_PROVENANCE.md">outer 完整来源</a></p><p>v685 并非所有成本都只由 85–100 重新拟合：旧非计算 floor、源图 gap、outer 等仍继承历史数据。源回放对新细粒度计算的绑定覆盖也存在后续审计指出的限制。</p></section>'
extra+='<section class="panel" id="iterations"><h2>目标 224 卡逐轮开发评估：时间与 MFU 分别报告</h2><p>Profiler 是 trace 对应区间；Training 是日志完整 iter。每轮 APE = |预测 − 实际| / 实际，MAPE 为各轮 APE 的平均。MFU 用同一 FLOPs / 峰值常数除以 Training 时间换算，参考 MFU 也由实际日志时间换算；它不是 GPU 忙碌率，分子未独立核验。</p>'
extra+=table(['迭代','预测 Profiler 秒','实际 Profiler 秒','Profiler APE %','预测 Training 秒','实际 Training 秒','Training APE %','预测 MFU %','参考 MFU %','MFU 相对 APE %'],[[r['iteration'],f"{float(r['predicted_profiler_ms'])/1000:.3f}",f"{float(r['actual_profiler_ms'])/1000:.3f}",f"{float(r['profiler_ape_pct']):.3f}",f"{float(r['predicted_training_ms'])/1000:.3f}",f"{float(r['actual_training_ms'])/1000:.3f}",f"{float(r['training_ape_pct']):.3f}",f"{float(r['predicted_mfu_pct']):.3f}",f"{float(r['actual_mfu_pct_derived']):.3f}",f"{float(r['mfu_relative_ape_pct']):.3f}"] for r in it])
extra+='<p>四轮 MAPE：Profiler 10.369528%，Training 10.718041%，MFU 相对误差 12.006689%。目标数据已用于开发，不能称独立盲测。</p></section>'
extra+='<section class="panel"><h2>相对 v684 改善了什么，仍缺什么</h2><p>在相同目标 85/90/95/100 四轮口径下，Profiler MAPE 从 v684 的 12.444830% 降为 v685 的 10.369528%，改善 2.075301 个百分点。目标 Profiler 仍平均低估约 2.474 秒，不能认定源到目标成本迁移已解决。</p><p>v682 页面中的历史 10.3% 与 75.8% 使用其他版本和 60–100 窗口，不挪作本版误差构成。本次借鉴其展示结构，所有新增表格使用 v685 同口径数据。</p><p><a href="/docs/w37/1f1b/WEB_INDEX.md">研究证据索引</a> · <a href="/docs/w37/BASELINE.md">基线口径</a> · <a href="/docs/w37/1f1b/REPORT.md">完整研究报告</a> · <a href="/results/w37/A/research-html-20260907/v685_iteration_results.csv">本页逐轮数据</a> · <a href="/v685-original.html">冻结原版 HTML</a></p></section>'
nav='<nav class="page-nav"><a href="/research.html">三种使用场景</a><a href="#model">计算方法</a><a href="#topology">依赖结构</a><a href="#timing">时间与误差分段</a><a href="#corrections">补偿来源</a><a href="#iterations">逐轮结果</a></nav><p class="note">本页是 v685 冻结结果的补充说明版，主目标为源侧建模、目标侧事前预测。图表仅展示既有结果，本次未重新拟合或修改正式图。</p>'
s=s.replace('<main>','<main>'+nav,1).replace('</main>',extra+'</main>',1)
s=s.replace('</style>','.panel{margin:20px 0}.scroll{overflow:auto}td{overflow-wrap:anywhere}.page-nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#71d4ee}.panel svg{width:100%;height:auto}th,td{min-width:100px}.scroll table{font-size:13px}.panel p{line-height:1.8}</style>')
(OUT/'v685-explained.html').write_text(s,encoding='utf-8')
for name,data in [('v685_iteration_results.csv',it),('v685_phase_results.csv',ledger)]:
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
(OUT/'v685-explained-provenance.json').write_text(json.dumps({'inputs_sha256':inputs,'new_prediction':False,'original_modified':False},indent=2))

#!/usr/bin/env python3
"""Render sealed research summaries only; no fitting or rescoring."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/w37/A/research-html-20260907'
RUNS = ROOT / 'docs/w37/1f1b/deadline_20260907/runs'
OLD = Path('/home/zjb/Desktop/fabric-data-analysis/case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_v685_source_steady_calibration_256_to_224/evaluator_only/dag_v685_source_steady_calibration.html')
inputs = {}
def read(p):
    data=p.read_bytes(); inputs[str(p)]=hashlib.sha256(data).hexdigest(); return data.decode()
def rows(p): return list(csv.DictReader(read(p).splitlines()))
def link(p, label): return '<a href="'+html.escape(os.path.relpath(p, OUT),quote=True)+'">'+html.escape(label)+'</a>'
def table(data, columns):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for key,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{float(row[key]):.3f}' if key.endswith(('_pct','_ms')) else str(row[key]))+'</td>' for key,label in columns)+'</tr>' for row in data)+'</tbody></table></div>'
def figure(wave, name, caption):
    p=RUNS/wave/'diagnose'/name
    svg=read(p); svg=svg[svg.index('<svg'):]
    return '<figure>'+svg+'<figcaption>'+caption+' · '+link(p,'原图')+'</figcaption></figure>'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics=rows(RUNS/'T38A/diagnose/target_metrics.csv')
    iterations=rows(RUNS/'T38A/diagnose/target_iteration_results.csv')
    read(OLD)
    body='<header><span>W37 · 224 卡 · 1F1B</span><h1>从冻结计算图到在线状态预测</h1><p>v6.8.5 基线与 T00–T42 研究过程说明 · 2026-09-07</p><p>阅读顺序：基线 → 命名与研究过程 → T35 前缀 → T38 状态 → 同口径结果与边界。</p></header>'
    body+='<nav><a href="#baseline">v685</a><a href="#route">研究过程</a><a href="#t35">T35</a><a href="#t38">T38</a><a href="#results">同口径结果</a><a href="#limits">验证边界</a>'+link(ROOT/'docs/w37/1f1b/WEB_INDEX.md','文档索引')+'</nav>'
    body+='<section id="baseline"><h2>01 · v685 模型与误差说明</h2><p>'+link(OLD,'打开 v6.8.5 补充说明 HTML')+' · '+(link(ROOT/'docs/w37/1f1b/post685/delivery/index.html','打开新版计算图册（误差矩阵＋时间线）').replace('index.html"','index.html?view=lane0-v2"'))+'</p><p>正式冻结图继承 v6.8.4 依赖锁：327,746 个节点、364,784 条边。目标 85/90/95/100 四轮 1F1B MAPE 为 11.340786%，Profiler Step 为 10.369528%，训练 Step 为 10.718041%，MFU 相对 MAPE 为 12.006689%。这些是已暴露目标数据上的开发评估。</p><p>v685 只部分重校准源成本，旧 floor、source 图成本和 outer 等仍继承历史参数。历史 75.8% 属于 v6.8.2、60–100 窗口的误差构成占比，不是当前阶段 MAPE。</p><p>'+link(ROOT/'docs/w37/BASELINE.md','冻结基线口径')+'</p></section>'
    route=[('T33A','全 rank 源 phase 图','source85/90 拟合，在已暴露95/100上全局 MAPE 0.332811%；建立独立语义图。'),('T34A','静态调度收缩','PP16/MB4→PP14/MB3预测收缩3.929秒，实际均值仅0.656秒；目标 MAPE 15.340746%，拒绝直接迁移。'),('T35A','当轮 F/B 方向前缀','用运行中首批 F/B 更新现有 phase wall；四轮在线 MAPE 1.774166%。'),('T36A','更早前缀','42.802%时可用，但剩余时间 MAPE 12.263149%；raw消融源门失败。'),('T37A','剩余时间口径','T35剩余时间 MAPE 4.201267%；更早的T36主候选被v685支配。'),('T38A','目标局部前轮状态','85初始化；90/95/100只用前一已完成采样轮base残差，MAPE 0.536422%。'),('T39A','状态新鲜度消融','lag-one 0.536422%、stale-2 1.082859%、扩展均值0.502136%；不按事后结果改选。'),('T40A','跨域反证','source MAPE 0.298320%→0.517489%，固定状态方法不具备跨域成立证据。'),('T41A','运行模式合同','正式冷启动、同轮开发、目标初始化开发三种口径分别使用。'),('T42A','交付审计','7条阶段链、783条封存文档链接、17类交付物与6条复现记录通过。')]
    body+='<section id="route" class="process"><h2>02 · 研究过程</h2>'+read(Path(__file__).with_name('research_process.html'))+'<div class="route">'
    for wave,title,desc in route: body+='<article><strong>'+wave+' · '+title+'</strong><p>'+desc+'</p>'+link(RUNS/wave/'acceptance.json','验收证据')+'</article>'
    body+='</div></section>'
    body+='<section id="t35"><h2>03 · T35：用运行中前缀识别方向慢化</h2><p>全stage的microbatch 0前向，以及后半PP stage的microbatch 0反向完成后，分别计算相对source85/90成本的比例，用20% winsor均值生成F/B倍率，再乘回现有phase wall。PP、本地runtime及图边沿用封存模型；不会把phase内部计算、通信和等待再次累加。</p><pre>phase_cost_online(F/B) = phase_cost_source(F/B) × prefix_factor(F/B)</pre><p>四轮平均在1F1B完成57.845%后可用，实际剩余约9.09秒。模型内F/B Shapley修正平均为+0.780/+2.146秒，B方向是主要修正来源；这解释图内修正来源，尚不能证明物理慢化的唯一根因。</p><p>phase MAE从99.427降到55.847ms，偏差从−96.428降到−4.612ms。所有目标倍率都超出源拟合范围，属于在线开发评估。</p>'
    body+=figure('T35A','online_prefix_nowcast.svg','模型预测与观测对照；数据来自封存T35结果')
    body+='<p>'+link(RUNS/'T35A/diagnose/target_online_nodes.csv.gz','模型节点表')+' · '+link(RUNS/'T35A/diagnose/target_online_edges.csv.gz','模型边表')+' · '+link(RUNS/'T35A/diagnose/target_online_phase_results.csv.gz','逐phase预测/观测')+'</p></section>'
    body+='<section id="t38"><h2>04 · T38：前轮残差是单独的目标状态</h2><pre>prediction(t) = T35_base(t) + [actual(previous) − T35_base(previous)]\niteration 85: correction = 0</pre><p>previous指当前采样序列的前驱：85→90→95→100，相隔5个训练iteration；现有结果不证明逐训练步t−1更新效果。当前与未来真值扰动不会改变当前预测。只使用前驱的base残差，不递归使用修正后的残差。</p><p>90–100平均预测构成：18,253.343ms静态phase基座 + 762.389ms前向修正 + 2,188.557ms反向修正 + 398.385ms前驱状态。最后一项没有被强行分配给计算、通信或等待，其他entry/tail/outer预测继承不变。</p>'
    body+=figure('T38A','causal_lagged_update.svg','T38因果更新和独立状态分账')
    body+='<p>'+link(RUNS/'T38A/diagnose/causal_prediction_ledger.csv','逐轮依赖链')+' · '+link(RUNS/'T38A/diagnose/causality_mutation_audit.csv','因果扰动核验')+' · '+link(RUNS/'T38A/diagnose/target_phase_ledger.csv','其他阶段回归')+'</p></section>'
    body+='<section id="results"><h2>05 · 同迭代范围比较</h2><p>所有相对误差单位均为%。1F1B与剩余时间采用各自实际时长作分母；Step与MFU分别计算。切换范围时表格直接展示封存值，不重新拟合。</p><label>评估范围 <select id="scope"><option value="all_including_initialization">85/90/95/100（含初始化，4轮）</option><option value="primary_causal_walk_forward">90/95/100（初始化后，3轮）</option></select></label>'
    cols=[('method','方法'),('iterations','轮数'),('onef1b_MAPE_pct','1F1B MAPE'),('remaining_MAPE_pct','剩余 MAPE'),('profiler_MAPE_pct','Profiler MAPE'),('training_MAPE_pct','训练Step MAPE'),('MFU_relative_MAPE_pct','MFU相对 MAPE')]
    for scope in ['all_including_initialization','primary_causal_walk_forward']:
        body+='<div data-scope="'+scope+'">'+table([r for r in metrics if r['split']==scope],cols)+'</div>'
    body+='<details><summary>展开逐迭代预测与误差（12行）</summary>'+table(iterations,[('method','方法'),('iteration','迭代'),('actual_onef1b_ms','观测ms'),('predicted_onef1b_ms','预测ms'),('onef1b_APE_pct','1F1B APE'),('training_APE_pct','Step APE'),('MFU_relative_APE_pct','MFU APE')])+'</details></section>'
    body+='<section id="limits"><h2>06 · 为什么仍保留正式 v685</h2><p>T38在目标三次转换均改善，但source只改善1/3；目标三次全改善的单侧sign-test p=0.125。有限历史区间仅覆盖1/2。扩展均值事后0.502136%不能用来改选；更早T36方案也没有通过最终可用性审查。</p><p>目标数据和source95/100已用于开发，均不称独立盲测。新状态方法验证需新增设计时未见的连续目标数据；冷启动仍缺动态router token矩阵和独立runtime readiness输入。隐藏graph、跨stream因果、跨lane物理成本迁移及MFU分子也未独立验证。</p>'
    body+=figure('T40A','cross_domain_state.svg','固定lag-one的source反证')
    body+='<p>'+link(RUNS/'T41A/diagnose/stop_resume_contract.json','恢复条件原文')+' · '+link(RUNS/'T41A/diagnose/reproduction_index.csv','Snakemake复现索引')+' · '+link(ROOT/'docs/w37/1f1b/REPORT.md','完整报告')+' · '+link(ROOT/'docs/w37/1f1b/ROUTE.md','完整研究路线')+'</p><p>本页为封存结果的展示层；图和指标内嵌，可离线阅读。原版HTML和证据链接依赖本机目录。旧Snakemake入口带2026-09-07 07:00截止保护，截止后不能照搬命令启动新run；需另登记复现计划，不覆盖旧输入或结果。</p></section>'
    css='body{margin:0;background:#f3f5f9;color:#192b3c;font:16px/1.75 system-ui,sans-serif}header,section,nav{max-width:1100px;margin:auto;padding:28px}header{background:#142b42;color:white;max-width:none;padding:55px max(28px,calc((100vw - 1100px)/2))}h1{font-size:36px;line-height:1.3}h2{color:#135b77}nav{display:flex;gap:24px;flex-wrap:wrap}a{color:#007b92}section{background:white;margin:20px auto;border-radius:12px;box-shadow:0 3px 18px #1231}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px;border-bottom:1px solid #dae3ea;text-align:left;white-space:nowrap}th{background:#eaf2f5}.scroll{overflow:auto}figure{margin:24px 0}figure svg{width:100%;height:auto}figcaption{color:#526579;font-size:14px}pre{background:#eef4f7;padding:18px;overflow:auto}.route{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px}article{border-left:4px solid #18849a;padding:15px;background:#f5f9fb}select{padding:10px;font-size:16px}details{margin-top:25px}summary{cursor:pointer;font-weight:bold}'
    css+='.process td{white-space:normal;vertical-align:top;min-width:110px}.process td:first-child{min-width:95px}.process h3{margin-top:30px;color:#135b77}.process code{overflow-wrap:anywhere}@media(max-width:600px){header,section,nav{padding:18px}h1{font-size:28px}.process table{min-width:620px}.route{grid-template-columns:1fr}}'
    script="const select=document.querySelector('#scope');function update(){document.querySelectorAll('[data-scope]').forEach(e=>e.hidden=e.dataset.scope!==select.value)}select.addEventListener('change',update);update();"
    def document(title, content, js=''):
        return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style>'+content+('<script>'+js+'</script>' if js else '')+'</html>'
    # Keep the complete explanation accessible from the index, outside the landing page.
    details=body.replace(os.path.relpath(OLD, OUT), '/v685.html')
    details='<nav><a href="/research.html">返回三种使用场景</a>'+link(ROOT/'docs/w37/1f1b/WEB_INDEX.md','研究与证据索引')+'</nav>'+details
    (OUT/'details.html').write_text(document('W37 · 研究过程与详细证据', details, script), encoding='utf-8')
    home=read(Path(__file__).with_name('research_home.html'))
    (OUT/'index.html').write_text(document('W37 · 从源侧优化目标侧预测', home), encoding='utf-8')
    (OUT/'render_manifest.json').write_text(json.dumps({'scope':'sealed-results presentation only','inputs_sha256':inputs,'output_sha256':hashlib.sha256((OUT/'index.html').read_bytes()).hexdigest(),'details_sha256':hashlib.sha256((OUT/'details.html').read_bytes()).hexdigest()},ensure_ascii=False,indent=2)+'\n')
    print(OUT/'index.html')
if __name__=='__main__':main()

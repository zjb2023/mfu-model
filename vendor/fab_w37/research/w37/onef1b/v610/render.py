"""One current-version report, with history accessible only through its index."""
import csv,html,json,re,sys
from pathlib import Path
import pandas as pd
from common import ROOT,RUN,dump,sha
from views import build as build_views
BASE='http://192.168.0.49:8037'

def build(spec,paths,out):
    release=json.loads((RUN/'model/release.json').read_text());p=release['prediction']
    metrics=json.loads((RUN/'evaluate/metrics.json').read_text());phase=pd.read_csv(RUN/'evaluate/phase_metrics.csv');iterations=pd.read_csv(RUN/'evaluate/iteration_results.csv')
    payload=build_views(paths,out,release)
    esc=lambda v:html.escape(str(v),quote=True)
    def table(headers,data):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in data)+'</tbody></table></div>'
    def note(s):return '<div class="note">'+s+'</div>'
    def details(title,s):return '<details><summary>'+esc(title)+'</summary>'+s+'</details>'
    def link(path,label):return '<a href="'+BASE+'/'+path+'">'+esc(label)+'</a>'
    data={}
    def explorer(id,title,headers,rows):
        data[id]=dict(columns=headers,rows=rows)
        return details(title,'<div id="'+id+'"><label>筛选文字 <input type="search" aria-label="'+esc(title)+'筛选"></label><div class="controls"><button data-prev>上一页</button><button data-next>下一页</button><span class="table-status"></span></div>'+table(headers,[])+'</div>')
    sys.path.insert(0,str(ROOT/'research/w37/onef1b/integration'))
    from organization import training,methods
    sections=[]
    def section(id,title,body):sections.append((id,title,body))
    section('training','训练任务与MFU建模',training(table,note)+methods(table,note))
    config=pd.read_csv(paths['static_parameters'],keep_default_na=False)
    names={'hidden_size':'隐藏维度','num_attention_heads':'注意力头数','ffn_hidden_size':'稠密FFN维度','moe_ffn_hidden_size':'路由专家维度','moe_shared_expert_intermediate_size':'共享专家维度','q_lora_rank':'Q低秩维度','kv_lora_rank':'KV低秩维度','qk_head_dim':'Q/K头维度','qk_pos_emb_head_dim':'Q/K位置部分维度','v_head_dim':'V头维度','num_experts':'专家数','moe_router_topk':'每token选择专家数','world_size':'GPU数','pp':'PP','cp':'CP','tp':'TP','dp':'DP','ep':'EP','num_layers':'层数','micro_batch_size':'每微批次样本数','global_batch_size':'全局batch','microbatches':'微批次数','sequence_length':'序列长度'}
    scenario='<p>使用256卡源侧成本与224卡静态训练配置，预测目标完整Training迭代时间，再换算MFU。rank是进程编号，本配置一rank对应一张GPU；每个PP stage有16个rank，14个stage共224卡。lane是stage内的位置，不能等同于全模型范围。</p>'
    scenario+=table(['配置','源256卡','目标224卡'],[['层 / PP / 微批次','60层 / PP16 / MB4','52层 / PP14 / MB3'],['MBS / GBS / 序列长度','2 / 64 / 8192','2 / 48 / 8192'],['TP / CP / DP / EP','1 / 2 / 8 / 8','1 / 2 / 8 / 8'],['层放置','[2,4×14,2]','[2,4×12,2]']])
    scenario+=details('查看完整模型维度与训练配置',table(['参数','源256卡','目标224卡'],[[names.get(r.field,r.field),r.source256,r.target224] for r in config.itertuples()]))
    scenario+='<div id="graph"><h3>执行图规模与依赖</h3>'+table(['项目','本版数值','含义'],[['图节点','327,746','计算、通信、运行时区间及控制点，不是kernel数'],['依赖边','364,784','沿用v6.8.4依赖锁，不按目标误差增加等待'],['内部计算重新赋值','70,192','覆盖224 rank、PP0–PP13、MB0–MB2'],['完整F/B动作','1,344','224 × 3 × 2；图中每stage聚成84条F/B包络']])+'</div>'
    scenario+='<pre>节点开始 = max(所有前驱的完成时刻)\n节点完成 = 节点开始 + 本节点时长\n1F1B区间 = 全rank最晚B结束 − 最早F开始</pre><p>前向依赖上游激活，反向依赖本地前向和下游梯度，其他先后顺序由训练调度及资源约束决定。等待由依赖计算，不能再统一插入固定等待；并行节点的时长不能直接求和当iter。</p>'
    section('task','当前训练配置与计算图',scenario)
    controls='''<div class="controls"><label>观测迭代 <select id="full-iter-choice"><option>85</option><option>90</option><option>95</option><option>100</option></select></label><label>显示范围 <select id="full-iter-window"><option value="all">完整Profiler迭代</option><option value="fb">首F到最后B</option><option value="tail">最后B附近与收尾</option></select></label><label><input id="full-iter-comms" type="checkbox" checked>显示通信与预测OPT</label></div>'''
    charts='<p>预测图来自本版完整节点的重新回放。F为前向、B为反向，数字为MB编号；同stage彩条是全部16 rank的最早开始到最晚结束，不是lane0，也不是GPU活跃时间之和。PP14/MB3统一展示完整1F1B，不用三种颜色暗示存在全流水线满载阶段。</p>'+controls
    charts+=note('完整Training包括准备、1F1B、最后B之后的收尾和outer。outer是Training与Profiler两种时钟的总时长差，尚未定位到具体事件，也没有均分到图中；604.440 ms图外补差另计。')
    charts+='<div class="scroll"><table id="full-iter-ledger"><thead><tr>'+''.join('<th>'+h+'</th>' for h in ['场景','准备秒','1F1B秒','最后B后收尾秒','Profiler秒','outer秒','Training秒'])+'</tr></thead><tbody></tbody></table></div><p id="full-iter-score"></p><div id="full-iter-plots"></div><div id="full-iter-selection" class="note"></div>'
    charts+='<p>DP是数据并行、EDP是专家数据并行，RS为梯度归约分片、AG为参数聚合，OPT为参数更新及相关运行时。观测DP RS覆盖组内16 rank，其余通信仅有代表rank区间；缓存未单列优化器计算明细，不能把图中空白解释为GPU空闲。源/目标图各自以Profiler起点计时，不表示两次运行的物理时钟同步。</p>'
    section('full-iter','v6.10完整迭代：预测与两侧观测',charts)
    a=release['example'];n=release['example_node'];key=a['source_key'].split('|');cost=a['new_compute_ns']/1e6;value=a['source_fit_compute_ns']/1e6;mult=a['inherited_shape_factor']
    params='<span id="binding-audit"></span><p>本版同时更新暴露计算和与通信并行的计算。成本参数是性能模型使用的耗时，与训练要更新的神经网络权重不同；一个源参数可以供多个rank的节点使用。</p>'
    params+=table(['参数或规则','本版采用值与来源','作用范围/边界'],[
        ['模型与训练结构','第02章静态配置','决定层数、张量尺寸、执行次数及通信组'],
        ['拓扑与程序顺序','v6.8.4冻结依赖；全局层号迁移','固定327,746节点/364,784边，不拟合依赖'],
        ['细粒度计算成本',f"源85/90中位数；{release['fit_parameters']:,}条源参数，{release['bound_fine_keys']:,}个键实际绑定",'方向 × MB × 全局层号 × 执行位置 × 语义槽'],
        ['计算覆盖','38,784暴露/退出节点 + 31,408重叠计算节点','全224 rank；重新赋值70,192节点，不表示每个值或时长都改变'],
        ['微批次运行时倍率','继承v6.7的1,344组rank/方向/MB有效倍率','约0.932365–1.095669；旧正成本重建误差<1 ns；不是按目标误差新拟合'],
        ['缺少新观测的内部节点','1,296个正成本节点保留继承值；8,576个原零成本且源记录缺失节点保留','未知不清零，缺记录不证明没有GPU工作'],
        ['边界计算成本','继承首尾3类边界键、144节点：13.681142 / 7.087467 / 0.378082 ms','保留边界语义；不是仅更新这144节点'],
        ['PP前向激活通信','624消息，固定4.704806 ms服务成本','沿用父版参数；不是整个F阶段只需4.7 ms'],
        ['PP反向梯度通信','源15接收stage × 16 lane × 4 MB的960记录，映射624目标消息','源85/90/95/100中位数；边界=B接收开始−B发送结束，含4.704806 ms服务及其余完成分量'],
        ['CP / EP通信','继承全局层对应的服务与本地完成成本','MoE dispatch→计算→combine分开；未按目标当轮token路由重建'],
        ['DP / EDP / 优化器','继承全部通信和1,120个优化器类节点成本','执行时刻随前置工作改变，不意味着成本重新拟合'],
        ['暴露计算与运行时','新节点时长=max(旧完整节点时长,新计算成本)','余量记为残差；旧完整时长下限未被独立证明是纯非计算成本'],
        ['图内入口','1265.388210 ms，继承源四轮估计','单独图节点，只计一次'],
        ['图到Profiler补差','604.439668 ms；源补差805.919557 ms × MB比例3/4','总量补差，没有定位为一段物理等待'],
        ['outer','1372.453390 ms；继承源60–100日志−Profiler差中位数','Training=Profiler+outer；不等于Profiler开关额外开销'],
        ['MFU换算','有效FLOPs 8.436548311982576e16；224 GPU × 每GPU 500 TFLOP/s','沿用冻结口径，分子未在本轮独立重算'],
        ['评价规则','目标85/90/95/100四轮开发评估','方法已在这些结果可见后由用户选择固化；不宣称盲测']])
    params+='<p>键中的forward/backward表示前向/反向；exposed_gap表示通信活动之外的计算区间，overlap_with_collective表示与通信活动重叠的计算区间。before_cp0是该层首个CP通信之前的位置；语义槽定位区间，不能把边界节点名当成独立kernel名。</p>'
    params+='<h3>参数如何对应到中间PP的具体计算节点</h3><figure id="parameter-node-map" class="parameter-node-map"><div class="scroll"><div class="node-map-canvas"><div class="node-map-transfer">'
    for title,body in [('① 源侧参数键',esc(a['source_key'])),('② 源值 × 继承倍率',f'{value:.6f} ms × {mult:.9f} = {cost:.6f} ms'),('③ 本版节点',esc(a['node_id']))]:params+='<article><b>'+title+'</b><p><code>'+body+'</code></p></article>'
    params+='</div>'+table(['方向','MB','全局层号','执行位置','语义槽'],[key])+table(['GPU/rank','PP stage','lane','MB','本版计算成本ms','本版节点时长ms'],[[int(n['rank']),int(n['pp_stage']),int(n['pp_lane']),int(n['microbatch']),f'{cost:.6f}',f"{n['duration_ns']/1e6:.6f}"]])+'</div></div><figcaption>计算键不含GPU编号；节点编号定位一次具体任务。r是rank，s是PP stage，l是lane（不是层号），fwd/bwd后数字是MB，q是本地F/B调度序号，其后描述层内节点位置。方框宽度不是耗时。</figcaption></figure>'
    coverage=pd.read_csv(RUN/'model/coverage.csv')
    params+=details('查看每个PP、前向/反向的节点绑定数量',table(['PP','方向','重新赋值节点','rank数','MB数'],[[int(r.pp_stage),'前向' if r.direction=='forward' else '反向',int(r.nodes),int(r.ranks),int(r.microbatches)] for r in coverage.itertuples()]))
    fits=pd.read_csv(RUN/'model/fine_parameters.csv',keep_default_na=False)
    params+=explorer('fine-parameters','查看本版细粒度源成本表',['方向','MB','全局层','执行位置','语义槽','源计算ms','样本数'],[[*r.fine_key.split('|'),f'{r.value_ns/1e6:.6f}',int(r.fit_samples)] for r in fits.itertuples()])
    params+='<p>'+link('results/w37/A/v610-release/model/node_parameter_bindings.csv.gz','下载70,192条节点绑定')+' · '+link('results/w37/A/v610-release/model/shape_factors.csv','下载1,344组继承倍率')+'</p>'
    section('calibration','本版全部参数与节点绑定',params)
    results='<span id="binding-results"></span><p>下面只展示v6.10。APE=单轮绝对误差/实际值，MAPE是四轮APE均值；MFU相对误差按实际MFU作分母，绝对误差另用百分点。阶段MAPE不能相加。</p>'
    results+=table(['指标','本版预测','四轮MAPE'],[['1F1B',f"{p['onef1b_ms']/1000:.6f} 秒",f"{metrics['onef1b_MAPE_pct']:.6f}%"],['Profiler',f"{p['profiler_ms']/1000:.6f} 秒",f"{metrics['profiler_MAPE_pct']:.6f}%"],['Training',f"{p['training_ms']/1000:.6f} 秒",f"{metrics['training_MAPE_pct']:.6f}%"],['MFU',f"{p['mfu_pct']:.6f}%",f"相对MAPE {metrics['mfu_relative_MAPE_pct']:.6f}%；绝对MAE {metrics['mfu_MAE_pp']:.6f}个百分点"]])
    results+=f"<pre>原始图 {p['raw_ms']:.6f} ms + 补差 604.439668 ms\n= Profiler {p['profiler_ms']:.6f} ms\n+ outer {p['outer_ms']:.6f} ms\n= Training {p['training_ms']:.6f} ms</pre>"
    phases={'entry':'准备','onef1b':'全rank 1F1B','tail':'最后B后收尾（含补差总账）','outer':'outer两时钟差'}
    results+=table(['阶段','预测ms','实际均值ms','MAPE%'],[[phases[r.phase],f'{r.predicted_ms:.6f}',f'{r.actual_ms:.6f}',f'{r.MAPE_pct:.6f}'] for r in phase.itertuples()])
    results+=table(['迭代','实际1F1B秒','1F1B APE%','实际Profiler秒','Profiler APE%','实际Training秒','Training APE%','实际MFU%','MFU相对APE%'],[[int(r.iteration),f'{r.onef1b_actual_ms/1000:.4f}',f'{r.onef1b_APE_pct:.4f}',f'{r.profiler_actual_ms/1000:.4f}',f'{r.profiler_APE_pct:.4f}',f'{r.training_actual_ms/1000:.4f}',f'{r.training_APE_pct:.4f}',f'{r.mfu_actual_pct:.4f}',f'{r.mfu_relative_APE_pct:.4f}'] for r in iterations.itertuples()])
    results+='<p>预测1F1B比父版长167.994406 ms，因原模型低估而减小误差；这是预测改善，不是实际训练加速。四轮平均仍低估1F1B约2.278秒。</p>'
    section('results','本版迭代结果与误差分账',results)
    limits='<p><b>版本身份：</b>v6.10是用户确认固化的当前开发版，包含完整可复现图、参数与预测。原v6.9编号用于另一项尚未发布的运行时设计，本版使用独立增量登记，不覆盖旧定义。</p>'
    limits+=table(['边界','当前事实'],[['全rank覆盖','指重新赋值覆盖全部224 rank，不表示所有成本都来自新拟合或所有参数已具备独立验证'],['数据划分','新计算拟合源85/90；源95/100为增量开发，历史成本使用60–100；目标四轮已用于开发'],['源整轮验证','旧源图缺少兼容的新计算键，尚未完成本方法的源侧完整迭代回归'],['时长下限','暴露节点保留旧完整节点时长下限，只能变长或不变；计算与运行时的物理归属仍待核验'],['其他阶段','收尾MAPE为35.788902%，比父版上升0.519705个百分点；入口、outer预测不变'],['收尾传播','优化器成本未变；最晚B推迟167.994406 ms，整图结束推迟171.853834 ms，相差3.859428 ms'],['适用范围','仅本次256→224配置与既有开发窗口；不将本版256拟合成本用于16源侧校准']])
    limits+=note('<b>唯一研究下一步：</b>建立兼容细粒度键的源侧完整回放，核对暴露区间旧时长下限和计算成本的归属，再对固定规则做源95/100整轮增量回归。')
    section('limits','版本边界与下一步',limits)
    links=[('docs/w37/1f1b/v610/REPORT.md','v6.10版本说明'),('docs/w37/1f1b/v610/HANDOFF.md','复现与交接'),('docs/w37/1f1b/v610/acceptance.json','版本验收'),('results/w37/A/v610-release/model/nodes.csv.gz','完整节点表'),('results/w37/A/v610-release/model/edges.csv.gz','冻结依赖边表'),('results/w37/A/v610-release/model/prediction_seal.json','预测封存'),('results/w37/A/v610-release/evaluate/iteration_results.csv','逐轮Step与MFU'),('results/w37/A/v610-release/evaluate/phase_iteration_results.csv','逐轮阶段结果'),('results/w37/A/v610-release/render/full_iter_payload.json','新版完整迭代图数据'),('w37-report-history.html','历史整合报告与旧版计算图'),('docs/w37/1f1b/binding/REPORT.md','方法研究与消融证据'),('docs/w37/1f1b/binding/ROUTE.md','研究推进路线'),('v685.html','v685历史版本'),('v684.html','v684历史版本')]
    index='<p>正文只呈现本版最终配置与结果。历史版本、选择过程、消融、在线辅助和外推案例通过索引单独查看。</p><ul>'+''.join('<li>'+link(a,b)+'</li>' for a,b in links)+'</ul><p>'+link('w37-report-history.html#scaleout','16→256已有外推案例')+' · '+link('w37-report-history.html#tool-todo','仿真工具调研')+'</p>'
    index+='<pre>/home/zjb/Desktop/fabric-data-analysis/.snakemake-venv/bin/python -m snakemake \\\n  --snakefile research/w37/onef1b/v610/Snakefile \\\n  --directory results/w37/A/v610-release --cores 1</pre><p>管线生成完整版本图并重放→封存→独立评分→HTML；只读冻结派生数据。重新生成请使用交接中的新run_root，避免覆盖已封存版本。</p>'
    section('sources','证据与历史索引',index)
    header='<header><div class="eyebrow">当前冻结开发版 · 256→224卡</div><h1>v6.10 · 计算图MFU预测模型</h1><p>源侧细粒度计算成本绑定到全224 rank；本页仅展示当前版本的模型、参数、完整迭代图及开发评估。</p><div class="cards">'+''.join('<article><b>'+t+'</b><div class="metric">'+v+'</div></article>' for t,v in [('1F1B MAPE','10.5616%'),('Training MAPE','10.0438%'),('MFU相对MAPE','11.1672%')])+'</div><div class="controls"><button id="download-html">下载完整HTML</button><button id="expand-content">展开所有详情</button><button id="collapse-content">收起所有详情</button></div></header>'
    nav='<nav class="toc"><div class="toc-inner">'+''.join('<a href="#'+id+'">'+f'{i:02d} '+title+'</a>' for i,(id,title,body) in enumerate(sections,1))+'</div></nav>'
    body=header+nav+'<main>'+''.join('<section id="'+id+'"><h2>'+f'{i:02d} · '+title+'</h2>'+body+'</section>' for i,(id,title,body) in enumerate(sections,1))+'</main><footer>v6.10 · 冻结开发版 · 历史与消融见证据索引</footer>'
    code=ROOT/'research/w37/onef1b/integration';css=(code/'style.css').read_text()+'\n.node-map-transfer{grid-template-columns:repeat(3,minmax(0,1fr));align-items:stretch}.node-map-transfer code{overflow-wrap:anywhere;white-space:normal} .toc-inner{gap:12px} .scroll{max-width:100%}'
    ui=(Path(__file__).with_name('ui.js')).read_text()
    timeline=(code/'full_iter.js').read_text().replace('这些误差仍包含其定义对应的完整区间；75.8% 属于后文 v682 的历史分账。','本版预测固定，观测与误差随所选轮次切换。')
    blob=lambda x:json.dumps(x,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v6.10｜计算图MFU预测模型</title><style>'+css+'</style></head><body>'+body+'<script id="report-data" type="application/json">'+blob(data)+'</script><script id="full-iter-data" type="application/json">'+blob(payload)+'</script><script>'+ui+'\n'+timeline+'</script></body></html>'
    (out/'index.html').write_text(doc)
    home='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>W37研究入口｜当前v6.10</title><style>'+css+'</style></head><body><header><div class="eyebrow">W37 · 当前模型</div><h1>v6.10 计算图MFU预测</h1><p>224卡 / PP14 / MB3 · 全rank计算成本绑定 · 冻结开发版</p><p><a class="button" href="/w37-report.html">打开当前v6.10完整说明</a></p></header><main><section><h2>当前结果</h2><p>1F1B MAPE 10.5616%；Training MAPE 10.0438%；MFU相对MAPE 11.1672%。</p><p>完整图、参数、逐轮结果及尚未验证边界均在当前版页面。历史研究不混入本版结果。</p><p><a href="/w37-report.html#calibration">本版参数</a> · <a href="/w37-report.html#full-iter">本版完整迭代图</a> · <a href="/w37-report.html#results">本版误差</a> · <a href="/w37-report-history.html">历史整合报告</a> · <a href="/research-history.html">原研究目录</a></p></section></main></body></html>'
    (out/'research-index.html').write_text(home)
    dump(out/'render_manifest.json',dict(version='v6.10',html_sha256=sha(out/'index.html'),source_model_sha256=sha(RUN/'model/release.json'),model_seal_sha256=sha(RUN/'model/prediction_seal.json'),prediction_used=p,sections=len(sections),fine_parameter_rows=len(fits),latest_version_only=True,history_linked=True,trace_preserved=True))
    print(json.dumps(dict(version='v6.10',html_bytes=len(doc.encode()),sections=len(sections))))

"""Build a self-contained Chinese report from the verified integration evidence."""
import base64, csv, html, json, re
from pathlib import Path
from statistics import mean
from common import *
from terminology import EDGE_NAMES
from full_iter import section as full_iter_section
from context_sections import background, scaleout, tool_todo
from parameter_sections import section as parameter_system_section, node_parameter_figure
from organization import organize
from binding_sections import audit_section as binding_audit_section, results_section as binding_results_section

verified=verify_inputs()
context_records=json.loads((DOC/'context_inputs.json').read_text())['inputs']
for r in context_records:
    cached=OUT/r['output']
    if cached.stat().st_size!=r['bytes'] or digest(cached)!=r['sha256']:
        raise ValueError('Committed evidence copy changed: '+r['key'])
audit=json.loads((OUT/'audit.json').read_text());p=audit['prediction']
esc=lambda s:html.escape(str(s),quote=True)
f=lambda x,n=3:f'{float(x):.{n}f}'
def table(headers,data,cls=''):
    return '<div class="scroll"><table class="'+cls+'"><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table></div>'
def details(title,content):return '<details><summary>'+esc(title)+'</summary>'+content+'</details>'
def note(content,warning=False):return '<div class="note'+(' warning' if warning else '')+'">'+content+'</div>'
def evidence(key,label=None):return '<span class="muted">证据：'+esc(label or key)+' · <a href="#sources">来源表 '+esc(key)+'</a></span>'
def figure(key,title,caption):
    src='data:image/svg+xml;base64,'+base64.b64encode(path(key).read_bytes()).decode()
    return '<figure><h3>'+esc(title)+'</h3><img loading="lazy" src="'+src+'" alt="'+esc(title)+'"><figcaption>'+esc(caption)+'</figcaption></figure>'
datasets={}
def explorer(id,title,headers,data):
    datasets[id]={'columns':headers,'rows':data}
    content='<div id="'+id+'"><label>筛选文字 <input type="search" aria-label="'+esc(title)+'筛选" placeholder="输入阶段、参数名或数值"></label><div class="controls"><button data-prev>上一页</button><button data-next>下一页</button><span class="table-status" aria-live="polite"></span></div>'+table(headers,[],'data-table')+'</div>'
    return details(title,content)

def bars_svg(title,labels,values,unit='秒',colors=None):
    maxv=max(values)*1.15 or 1
    a=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 '+str(105+len(values)*65)+'"><rect width="1100" height="100%" fill="#fff"/>']
    a.append('<text x="20" y="33" font-family="sans-serif" font-size="22" fill="#173249">'+esc(title)+'</text>')
    for i,(label,value) in enumerate(zip(labels,values)):
        y=62+i*65;w=value/maxv*570;color=(colors or ['#29868c']*len(values))[i]
        a.append(f'<text x="20" y="{y+25}" font-family="sans-serif" font-size="18" fill="#173249">{esc(label)}</text><rect x="370" y="{y}" width="{w}" height="35" rx="4" fill="{color}"/><text x="{380+w}" y="{y+25}" font-family="sans-serif" font-size="18" fill="#173249">{value:.3f} {esc(unit)}</text>')
    a.append('</svg>');return ''.join(a)

header='<header><div class="eyebrow">W37 · 大语言模型分布式训练性能预测</div><h1>从分布式训练到计算图 MFU 模型</h1><p>理解真实训练如何执行，比较三阶段与计算图建模，再追踪源侧校准、目标预测、版本增量和误差证据。</p><p><b>当前状态：</b>正式事前预测仍使用v685；新增计算绑定候选将目标1F1B MAPE降至10.5616%，但收尾退化且源整轮验证未完成，暂不升级正式版本。T35/T38作为在线辅助单列。</p><div class="controls no-print"><button id="download-html">下载完整 HTML</button><button id="expand-content">展开所有详情</button><button id="collapse-content">收起所有详情</button></div></header>'
parts=[]
parts.append(background(table,details,note))
parts.append(full_iter_section())

parts.append('<section id="summary"><h2>01 · 主结论与三种使用场景</h2><div class="cards">')
for title,badge,metric,desc in [
('源侧建模，目标运行前预测','主任务 · v685','11.341%','目标四轮 1F1B MAPE；不使用当轮前缀或前一目标轮残差修正。历史成本与残差的继承边界见校准表。'),
('当轮首批 F/B 完成后预测','辅助场景 · T35','1.774%','目标四轮 1F1B MAPE；平均到实际区间完成 57.845% 后可用，剩余时间 MAPE 为 4.201%。'),
('当轮前缀 + 前一采样轮残差','辅助场景 · T38','0.536%','85 轮初始化后，仅评价 90/95/100 三轮；需要当轮前缀及前一已完成目标采样轮，剩余时间 MAPE 为 1.263%。')]:
    parts.append('<article><span class="badge'+(' secondary' if '辅助' in badge else '')+'">'+badge+'</span><h3>'+title+'</h3><div class="metric">'+metric+'</div><p>'+desc+'</p></article>')
parts.append('</div>'+note('<b>输入条件和评价轮次不同，以上不是同一条件下从 11.341% 连续优化到 0.536%。</b> 所有目标数据已用于开发；本文不宣称独立盲测。')+'<p>文档管线只展示已封存结果；新增计算绑定实验由独立研究管线完成，正式图保持不变。关键图和数据已内嵌，下载后可离线阅读；原始证据文件的网页链接需要当前服务器。</p></section>')

terms=[
('计算图 / 节点 / 边','用节点表示计算、通信、运行时区间或控制点；有向边表示先后依赖。','节点数不是 kernel 数；零时长控制点不代表真实操作没有成本。'),
('rank / local','rank 是训练进程编号，本配置每个 rank 对应一张 GPU；local 是该 rank 内的 CPU/GPU 动作。','CPU 进入 API 不等于对应 GPU 已完成计算。'),
('PP stage / lane','PP stage 是流水线阶段；lane 是连接各 PP stage 的一条通道。目标有 14 个 stage × 16 条 lane = 224 rank。','lane0 是每个 stage 取一个 rank，共 14 个，不是一个 GPU，也不代表全 224 卡。'),
('MB / microbatch','一个训练迭代拆分出的微批次个数：源 4，目标 3。','MB3 是三个微批次；每微批次样本数 micro_batch_size 是另一参数。'),
('F / B；FWD / BWD','F/B 表示前向/反向计算；FWD 激活消息向下一 PP stage 传递，BWD 梯度消息向上一 stage 传递。','source 是源场景，sender 才是发送方。'),
('1F1B 区间','从全局最早的前向 F 开始到最晚的反向 B 结束的墙钟区间，含计算、通信、等待和重叠。','不是各 rank 的 F/B 耗时之和，也不是完整 Training iter。'),
('phase wall / 阶段墙钟时长','某个 rank、微批次、F/B 动作的 CPU 标注区间起止差。','内部已包含的计算与通信不能再加到这一总区间。'),
('API 入口 / 返回','CPU 进入通信调用 / 调用结束。单消息调用的首个返回可给出消息完成时刻的观测上界。','不把首个 API 返回叫作精确 GPU 数据完成。'),
('有效发送就绪延迟 d / 完成余项 c','用于解释发送/接收入口与完成上界关系的源侧拟合代理。','不是已独立测量的 CPU、GPU 或纯网络耗时。'),
('Profiler 时间 / Training 时间','前者是 trace 对应采样区间；后者是训练日志完整 iter 时间。当前 Training 预测 = Profiler 预测 + outer。','Profiler 误差不是开启采样的额外开销占比。'),
('outer / 图外残差补偿','outer 衔接两种 iter 时钟；图外残差补偿修正原始依赖图与 Profiler 的差额，二者不同。','都未完整拆成物理操作，不能叫纯框架或采集开销。'),
('MFU','模型有效 FLOPs ÷（全卡理论峰值算力 × Training 时间），百分数需乘 100；参考值由实际 Training 时间换算。','不是 GPU 忙碌率；224 卡沿用历史口径，16→256 的分子与用户确认峰值在对应章节单列。'),
('APE / MAPE / MAE','APE 为单轮 |预测−实际| / 实际；MAPE 为 APE 均值；MAE 为绝对毫秒差均值。','不把 MAPE 当耗时占比，也不将 100%−MAPE 称准确率。'),
('有符号差额 / 误差构成占比','本文分账中实际−预测为正表示低估；构成占比为一项差额占指定总差额的比例。','平均有符号差额会正负抵消；不等于 MAE 或 MAPE。'),
('校准 / 开发评估 / 事后诊断','校准用来确定参数；开发评估衡量已暴露数据上的效果；事后诊断可替换观测来解释差额。','预测封存能限制本次数据流，但不能让历史已见数据重新成为盲测。'),
('稳定采样窗口 / 调度 steady','v685 稳定窗口指源侧采样轮次时间波动较小；调度 steady 指代码中的局部 F/B 交替分支。','两者不同；PP14/MB3 图统一显示完整 1F1B，不能把颜色当全流水线满载阶段。'),
('Shapley 分账','计算多种组件替换组合，平均分配各组件通过依赖影响总时间的份额。','柱子不是单个节点耗时之和，也不是实际可节省的优化收益。'),
('CP / EP / DP / EDP','分别是上下文、专家、数据、专家数据并行；其组内通信可能含同步和等待。','通信 kernel 活动不等于纯有效字节传输。'),
('成本键 / 参数键','按方向、层类型、层内位置等字段组合成的查表索引；多个节点可以使用同一个成本值。','58 个可用键不等于 58 个键全部命中；144 个更新节点也不等于 144 个独立参数。'),
('暴露计算 / 重叠计算','分别指落在通信活动之外 / 与通信活动时间相交的计算活跃区间。','这是时间区间划分；不能直接当作对整图总时间的独立贡献。'),
('dense / MoE / token','dense 为稠密层，MoE 为混合专家层；数据 token 是路由和计算的序列单元。','节点类型中的 token 也可指控制或计时节点，不能一律理解成数据 token。'),
('上下文 / 语义槽','上下文描述处于哪类层前后；语义槽标识该层内的边界位置，例如首个 CP 通信之前或阶段退出处。','这是成本查表的归类标签，不能只按表名推断完整物理执行原因。'),
('P10 / P90 / 两端 20% 缩尾均值','P10/P90 是样本的第 10/90 百分位；缩尾均值先把两端各 20% 极端值压到边界值，再求平均。','分位数区间不是预测置信区间；缩尾不是删掉样本。'),
]
parts.append('<section id="terms"><h2>02 · 名词先说明，再读图和参数</h2><p>全文已按下表审核名称；图名写明场景、方向、范围和比较对象。详细代码字段保留在表中以便追溯。</p>'+details('展开名词与容易混淆的含义（'+str(len(terms))+' 项）',table(['名词','本文含义','避免误读'],terms))+note('最重要的范围区分：<b>正式模型覆盖 224 rank；lane0 诊断覆盖 14 rank。图中 3.420 秒是低估差额，不是 1F1B 总时长。</b>')+'</section>')

hist=audit['historical_v684'];histfb=next(r for r in audit['historical_v682_ledger']['components'] if r['component']=='forward_backward_envelope')
parts.append('<section id="baseline"><h2>00 · 历史版本结果与误差分账</h2><p>下表按模型版本与评价窗口核对结果。“冻结基线”指固定版本、输入、依赖与评价口径。当前v685的四轮评价口径见本章正文。</p>')
parts.append(table(['成果或版本','评价场景 / 轮次','指标','核验后的解释'],[
('三阶段 MFU v5.4 模型','224 卡，60–100 九个采样轮','Profiler MAPE 16.823672%','模型将 FWD、BWD（含重计算）、OPT 聚合建模；16.82% 是该模型整轮 Profiler 时间误差。'),
('v682 计算图 MFU 模型','224 卡，60–100 九轮',f"Profiler MAPE {audit['historical_v682_mape']:.6f}%",'与下面 75.8% 分账直接对应的约 10% 误差；没有去掉准备与收尾来算整轮 MAPE。'),
('v684 计算图','224 卡，60–100 九轮',f"Profiler MAPE {hist['metrics']['all_mape_pct']:.6f}%",'对应历史约 10.3% 的全窗口结果。'),
('v682 历史分账','224 卡，60–100 九轮',f"1F1B 差额占比 {histfb['error_share_pct']:.3f}%",f"约 {histfb['underprediction_ms']/1000:.3f} 秒 ÷ 总平均低估 {audit['historical_v682_ledger']['reconstructed_underprediction_ms']/1000:.3f} 秒；不是阶段 MAPE。"),
('v684 同尾部窗口','224 卡，85/90/95/100',f"Profiler MAPE {hist['metrics']['late_mape_pct']:.6f}%",'与 v685 比较必须改用这个四轮窗口。'),
('v685 正式冻结基线','224 卡，85/90/95/100','Profiler 10.369528%；1F1B 11.340786%','v684 拓扑保持不变，部分源侧成本重新校准。')]))
parts.append(note('<b>正确的阅读脉络：</b>三阶段 MFU 模型整轮误差约 16.82% → v682 计算图模型整轮误差约 10.46% → 对 v682 整轮低估分账，其中 75.8% 来自 1F1B。之后 v684 同九轮为 10.28%，v685 主评估改用四轮，不能混作同一组数字。<br><b>75.8% 的分母仍包含准备和收尾。</b>只在定义 1F1B 子区间时排除这两段；不是先去掉两段，再得到约 10% 的整轮误差。'))
parts.append(table(['v682 全 iter 分账（60–100 九轮均值）','预测 ms','实际 ms','低估 ms','占整轮低估 %'],[[r['label'],f(r['predicted_ms']),f(r['actual_mean_ms']),f(r['underprediction_ms']),f(r['error_share_pct'])] for r in audit['historical_v682_ledger']['components']]))
parts.append('<p>1F1B 占比 = 1859.632600 ÷（326.696062 + 1859.632600 + 265.730055）= 75.839644%。这是平均毫秒差额的构成，不是把 MAPE 的 10.46% 直接拆成百分比，也不是 1F1B 耗时占整轮时间的比例。</p>'+evidence('historical_three_stage')+' · '+evidence('historical_v682')+' · '+evidence('historical_v684')+'</section>')

kind_names={
'phase_boundary':'F/B 起止与边界成本节点','operator':'算子或局部运行区间节点','pp_p2p':'流水线点对点消息节点','collective_boundary':'组内通信边界节点','collective_service':'组内通信服务节点','optimizer':'参数更新步骤节点','iteration_boundary':'迭代起止控制点','compute_overlap_token':'与通信并行的计算分支节点','collective_release_token':'组内通信发布/完成控制或区间','scheduler_handoff':'本地调度交接节点','microbatch_release_gap':'微批次释放顺序节点',
'framework_entry':'迭代入口区间','initial_api_readiness':'rank 初始 API 入口准备','local_prelaunch':'本地动作前的运行时间','phase_compute_communication_runtime':'一个完整 F/B 阶段区间','api_publication':'通信 API 入口控制点','api_return':'通信 API 返回/收尾节点','pp_sender_effective_readiness':'有效发送就绪延迟','pp_postpublication_completion':'双端就绪后的消息完成余项'}
parts.append('<section id="graph"><h2>04 · 计算图有多少节点，节点代表什么</h2><p>正式 v685 是细粒度图，v687/T33/T34 是独立的阶段级研究图。它们用不同粒度表达工作，节点数不能直接比较成“优化了多少算子”。</p>')
parts.append(table(['图','全部节点','全部依赖边','rank / stage / lane','完整 F/B 阶段节点'],[[r['graph'],f"{r['node_count']:,}",f"{r['edge_count']:,}",f"{r['rank_count']} / {r['stage_count']} / {r['lane_count']}",'不采用这种聚合类型' if r['phase_node_count']==0 else f"{r['phase_node_count']:,}"] for r in audit['graph_summaries']]))
parts.append('<p>目标阶段图的 1,344 个 F/B 节点 = 224 rank × 3 个微批次 × 2 个方向；它只是 13,042 个总节点中的一类。T33 源图的对应数量为 256 × 4 × 2 = 2,048。v685 另有历史源回放图 '+f"{audit['v685_source_node_count']:,}"+' 个节点，结构与新目标细粒度图不同，不能当作完全对称的新计算验证。</p>')
parts.append(details('展开各类节点数量、正时长与零时长数量',table(['图','节点类型（中文）','原始字段','总数','正时长','零时长'],[[r['graph'],kind_names.get(r['kind'],'模型控制/运行时节点'),r['kind'],r['count'],r['positive_duration'],r['zero_duration']] for r in audit['node_types']])))
parts.append(details('展开全部依赖边类型计数',table(['图','依赖含义','代码字段','数量'],[[r['graph'],EDGE_NAMES[r['edge_type']],r['edge_type'],r['count']] for r in audit['edge_types']])))
parts.append(note('v685 共有 <b>197,427 个零时长节点</b>。部分是结构/控制点，部分成本在其他节点集中表达，不能据此判断真实操作没有耗时。正式 v684→v685 的节点和边集合不变；独立候选图没有替换正式锁。'))
parts.append('<pre>节点完成时刻 = max(所有前置节点的完成时刻) + 本节点耗时\n全局 1F1B 区间 = 最晚 B 结束 − 最早 F 开始\n各 rank 并行推进；不能把所有节点或所有 GPU 时长直接加成一个 iter。</pre>')
parts.append(details('查看 PP14 / MB3 的本地调度顺序',figure('schedule_svg','目标 PP14 / MB3：各阶段本地 F/B 执行顺序','顺序轴不是全局时间轴；统一显示完整 1F1B，不区分全流水线 warmup/steady/cooldown。')))
causal_evidence=figure('causal_svg','研究候选 v687：前向激活与反向梯度的完成条件','此图是独立研究候选的语义说明，不是 v685 已采用的新增等待结构。')
parts.append(evidence('v685_nodes')+' · '+evidence('v685_edges')+' · '+evidence('T33_nodes')+'</section>')

parts.append('<section id="calibration"><h2>05 · 详细配置与校准：究竟调了哪些参数</h2>')
parts.append(parameter_system_section(table,details,note))
parts.append('<h3>训练场景参数，与拟合成本分开</h3>')
static_names={'num_layers':'模型层数','hidden_size':'隐藏维度','num_attention_heads':'注意力头数','num_experts':'专家数','moe_router_topk':'路由选择专家数','micro_batch_size':'每微批次样本数','global_batch_size':'全局批大小','seq_length':'序列长度','world_size':'GPU/rank 数','tensor_model_parallel_size':'张量并行 TP','pipeline_model_parallel_size':'流水线并行 PP','context_parallel_size':'上下文并行 CP','expert_model_parallel_size':'专家并行 EP','data_parallel_size':'数据并行 DP'}
src={r['field']:r for r in rows('source_static')};tgt={r['field']:r for r in rows('target_static')}
static_table=[]
for key,r in src.items():
    t=tgt.get(key,{})
    static_table.append([static_names.get(key,key),r['declared_value'],t.get('declared_value','未提供'),'日志一致' if r['status']=='EXACT_LITERAL_MATCH' else '声明值；该日志片段未出现','日志一致' if t.get('status')=='EXACT_LITERAL_MATCH' else '声明值；该日志片段未出现'])
parts.append(details('展开源/目标静态配置及日志核验状态',table(['配置','源侧声明','目标声明','源核验','目标核验'],static_table)+'<p>源/目标微批次数分别为 4/3，由冻结场景合同给定。日志片段未出现不等于配置错误，也不能以默认值补证。源模型 60 层、目标 52 层，均在所读日志中确认；两者不只 GPU 数不同。</p>'))
parts.append('<h3>v685：选择稳定源窗口，部分重校准，其他成本继续继承</h3><p>在源侧 60、65、…、100 的候选尾部窗口中，只使用源 Profiler 时间的波动率选择至少四点、截至 100 的窗口；选出 85/90/95/100，波动系数 0.368147%。这里“稳定窗口”是统计选窗，不是流水线 steady 调度阶段。</p>')
parts.append('<figure id="v685-calibration-map"><div class="scroll">'+(OUT/'v685_calibration_map.svg').read_text()+'</div><figcaption>图：v685源输入、实际参数与目标作用。箭头表示参数数据流，不是训练依赖或时间轴。<a href="http://192.168.0.49:8037/results/w37/A/integration-20260907/v685_calibration_map.svg">打开 / 保存原尺寸SVG</a> · <a href="http://192.168.0.49:8037/results/w37/A/integration-20260907/v685_calibration_map.json">机读参数对应关系</a></figcaption></figure>')
calibration=[
('计算活跃区间成本','重新统计 + 有限绑定','源 256 卡 85/90/95/100；每个 PP stage 的 lane0 代表 rank','按方向、层类型、上下文、暴露/重叠位置、语义槽取活跃并集时长中位数','360 行统计 → 58 个物理键 → 实际 3 个键 → 144 个节点；仅首尾 PP stage 的 32 rank'),
('PP 反向梯度边界总时长','重新拟合并更新消息节点','源 256 卡四轮，15 对相邻 PP stage × 16 lane × 4 MB','接收 stage 的 B 开始 − 下游发送 stage 的 B 结束；按接收 stage/lane/MB 取四轮中位数','960 个源参数；624 个目标反向消息节点更新；依赖边无新增/删除'),
('PP 网络服务分量','继承','历史源侧通信模型','固定服务分量 4.704806 ms；新消息总时长减它得到其余完成分量','没有因 v685 重新拟合这个网络分量；剩余量不是纯软件工作实测'),
('迭代入口','重新拟合','源侧四轮 Profiler 起点至首个 F/B 的边界差','取中位数，写入唯一 framework_entry 节点','1265.388210 ms'),
('源图与 Profiler 差额','重新计算 + 比例迁移','源四轮 Profiler 中位数与源回放图','23831.161007 − 23025.241449 = 805.919557 ms；目标乘 MB 比例 3/4','目标补偿 604.439668 ms；不能当新增物理等待节点'),
('outer 两时钟差额','继承','源 256 卡 60–100 九轮；最初 v5.4','每轮 log−Profiler 差额取中位数；v685 读取 v684 父合同','1372.453390 ms；不是 profiler 开关对照得到的开销'),
('其余旧成本与拓扑','继承','v684 图及更早 v54/v67 等来源','保持旧非计算时长下限、未命中的计算/通信成本、CP/EP/DP/EDP等既有表示','未全量重校准；旧 gap 可含 95/100，不可冒称全新留出集'),
('MFU 计算量与峰值','继承配置常数','模型配置','每步 FLOPs 8.436548311982576e16；每 GPU 峰值 500 TFLOP/s；目标 224 卡','MFU = FLOPs / (224 × 峰值 × Training 秒数)；分子未独立核验'),
]
parts.append(table(['参数族','动作','数据来源','方法','实际作用 / 限制'],calibration))
parts.append(note('<b>两处旧说明已纠正：</b>“58 项参数更新所有计算”应改为“58 个可用物理键，实际 3 个键命中 144 节点”；“624 条目标边更新”应改为“624 个反向 PP 消息节点的成本更新”。节点/边数量和正式依赖锁没有变化。',True))
key_names={'backward|dense|dense_to_step_exit|exposed_gap|step_exit':'反向：稠密层之后到本阶段退出','backward|moe|step_entry_to_moe|exposed_gap|before_cp0':'反向：进入 MoE 阶段后、首个 CP 之前','forward|moe|moe_to_step_exit|exposed_gap|step_exit':'前向：MoE 层之后到本阶段退出'}
parts.append(binding_audit_section(table,details,note))
parts.append('<h3>正式v685实际命中的三个计算成本键</h3><p>与下方58个键使用相同的五列及原始取值：<b>方向 → 层类型 → 上下文 → 执行位置 → 语义槽</b>。五项按此顺序用 <code>|</code> 连接，就是节点表中的完整参数键；中文位置说明仅作解释，不另起名称。</p>'+table(['方向','层类型','上下文','执行位置','语义槽','中文位置说明','PP stage / rank数','更新节点','旧计算成本范围 ms','新计算成本 ms','节点总时长变化合计 ms'],[[*r['key'].split('|'),key_names[r['key']],r['stages']+' / '+str(r['rank_count']),r['nodes_updated'],f(r['old_compute_min_ms'],6)+'–'+f(r['old_compute_max_ms'],6),f(r['new_compute_ms'],6),f(r['summed_duration_delta_ms'],6)] for r in audit['compute_summary']]))
parts.append('<p>144 个命中节点均为暴露计算项：<code>新节点时长 = max(新计算成本, 继承的非计算时长下限)</code>。更新跨 32 rank、3 个微批次，每个键命中 48 个节点；不是 144 个不同算子参数。节点时长变化合计仅 1.643735 ms，这是跨节点求和，不能直接当 iter 改善量。源回放没有绑定这批新计算成本；T31/T32 已记录该验证缺口。</p>')
parts.append(node_parameter_figure())
parts.append('<p><b>对照下方两张明细表：</b>“58个参数”表列出计算键及对应成本；“144个节点更新”表列出实际使用成本的节点编号、位置与更新前后数值。先查计算键，再用节点编号定位具体任务。</p>')
physical=[r for r in rows('compute_parameters') if r['parameter_view']=='window_split' and r['cost_group']=='__physical_slot_total__' and r['autograd_phase']=='physical_mixed']
parts.append(explorer('physical-params','查看全部 58 个可绑定物理计算参数',['方向','层类型','上下文','执行位置','语义槽','中位数 ms','P10 ms','P90 ms','样本数','源rank数'],[[r['phase'],r['layer_type'],r['layer_context'],r['execution_scope'],r['semantic_slot'],f(int(r['median_active_union_ns'])/1e6,6),f(int(r['p10_active_union_ns'])/1e6,6),f(int(r['p90_active_union_ns'])/1e6,6),r['calibration_samples'],r['source_rank_count']] for r in physical]))
parts.append(explorer('compute-updates','查看 144 个目标计算节点的逐项更新',['节点ID','类型','rank','PP stage','参数键','旧节点 ms','新节点 ms','非计算下限 ms'],[[r['node_id'],r['kind'],r['rank'],r['pp_stage'],r['source_parameter_key'],f(int(r['old_duration_ns'])/1e6,6),f(int(r['new_duration_ns'])/1e6,6),f(int(r['noncompute_floor_ns'])/1e6,6)] for r in csv.DictReader((OUT/'compute_node_updates.csv').open())]))
parts.append(explorer('gradient-params','查看 960 个源侧反向梯度通信参数',['接收PP stage','lane','MB','四轮中位数 ms','P10 ms','P90 ms','样本数'],[[r['source_receiver_stage'],r['pp_lane'],r['microbatch'],f(int(r['gradient_wall_median_ns'])/1e6,6),f(int(r['gradient_wall_p10_ns'])/1e6,6),f(int(r['gradient_wall_p90_ns'])/1e6,6),r['sample_count']] for r in rows('gradient_parameters')]))
parts.append(details('目标反向消息参数如何映射源 PP stage',table(['目标接收 stage','源接收 stage'],list(enumerate([0,1,2,3,5,6,7,8,9,10,12,13,14,15])))+'<p>目标梯度消息实际接收 stage 为 0–12，使用映射的前 13 项；最后一项为配置保留。目标消息数 13 × 16 × 3 = 624。该映射属于 v685 的继承建模方案；不能与 v687 另一套阶段角色迁移映射混用。</p>'))
parts.append('<h3>独立候选与在线辅助：参数如何使用</h3>'+table(['研究','可变参数 / 状态','来源与拟合轮次','保持不变或限制'],[
('v687 / T33 阶段图','F/B 总区间、本地运行时区间、PP 有效就绪 d 和完成余项 c','source 85/90；phase/runtime按上下文中位数，d/c按方向和lane稳健拟合','独立候选图，不替换 v684/v685 正式锁；F/B 内部成本不重复加到总区间'),
('T34 静态迁移','PP16/MB4 改为 PP14/MB3 等配置','固定 T33 源成本，四个静态场景回放','不按目标误差拟合倍率或残差'),
('T35 当轮前缀','F、B 各一个方向倍率，乘回各阶段已有总区间','当轮全stage首F、后半stage首B；两端各20%极值缩尾后取平均','不改 PP d/c、本地 runtime 和图边；倍率来自目标运行中观测'),
('T36 更早前缀','可靠性缩放系数 0.611345','source85/90；只观察末端4个stage首B','主候选精度不足；不按目标更优的事后消融改选'),
('T38 前轮状态','上个已完成采样轮的实际−T35预测残差','目标85初始化，按85→90→95→100前驱使用','状态在整图预测外单列，不硬归为计算、通信或等待；不是源侧事前预测优化')]))
parts.append(table(['v687 消息方向','d 均值 ms','c 均值 ms','95/100 样本数','本版局部 MAE ms','v686 对照 MAE ms'],[['前向激活' if r['direction']=='F' else '反向梯度',f(r['ready_mean_ms'],6),f(r['completion_mean_ms'],6),r['validation_samples'],f(r['mae_ms'],6),f(r['v686_mae_ms'],6)] for r in audit['pp_summary']]))
pp_evidence=details('查看源侧前向/反向观测散点及完整关系式',figure('pp_svg','源 256 卡：前向激活与反向梯度通信的局部检验','输入是已观测 CPU API 入口，纵轴是首个 API 返回给出的完成上界；只用两端单消息调用。前向基本接近常量，误差没有优于 v686。'))
parts.append('<p>参数公式用于解释候选模型；其成立范围与源侧散点检验见<a href="#pp-diagnosis">第07章通信就绪诊断</a>。</p>')
parts.append(evidence('v685_code')+' · '+evidence('compute_updates')+' · '+evidence('gradient_code')+' · '+evidence('readiness_code')+'</section>')

parts.append('<section id="weekend"><h2>00 · 研究实验索引与增量记录</h2><p>研究按“核对依赖和重复计时 → 追踪 CPU/GPU 工作 → 检查全 rank 成本来源 → 静态迁移 → 在线状态”推进。详细 T 编号通过下方展开，不把任务数量当成模型升级数量。</p>')
parts.append(table(['阶段','主要问题','结果与后续方向'],[
('T00–T13','通信 API 包络是否重复计时；约93 ms能否由CP/EP尾部解释','获得部分运行时依据，但最新源侧候选目标 1F1B 14.994% 仍差于 v685 11.341%。'),
('T14–T25','设备活动、提交队列与隐藏graph是否解释空档','局部关联更完整；无可见活动不等于GPU空闲，不能无依据增加等待。'),
('T26–T32','局部条件预测能否接回完整图；计算校准覆盖是否可靠','发现末端与PP重叠、静态签名无增益、源回放计算绑定缺口，以及旧gap含验证轮。'),
('T33–T34','独立全rank阶段图在源侧和目标侧是否都成立','源95/100 1F1B MAPE 0.333%；直接迁移目标仍15.341%，只改静态PP/MB不足。'),
('T35–T38','已知当轮前缀和上一采样轮后能否更准','在线开发结果改善，但新增目标运行信息，不能宣称主任务已完成。'),
('T39–T42','状态规则可否推广；证据是否完整','source上同规则退化；停止在已见四轮上继续选法，形成恢复条件与证据审计。')]))
process=text('process_text');process=process[:process.index('<h3>T33–T42 验收入口</h3>')]
# Content is preserved but all external source links use absolute server URLs for downloaded copies.
parts.append(details('展开 T00–T42 详细过程与命名规则','<div class="process">'+process+'</div>'))
scenarios=rows('T34_scenarios')
parts.append(details('T34：固定源成本时，PP/MB 改变造成多少预测收缩',bars_svg('四种静态调度配置：预测 1F1B 时间',['源 PP16 / MB4','只减 MB：PP16 / MB3','只减 PP：PP14 / MB4','目标 PP14 / MB3'],[float(r['predicted_onef1b_ms'])/1000 for r in scenarios])+ '<p>源到目标预测收缩 3.929 秒，实际均值只收缩 0.656 秒。不能为了追平差额按目标结果补一个倍率。缺少可由源侧或独立输入预测的运行状态，是主任务仍未完成的重要原因。</p>'))
parts.append('</section>')

parts.append('<section id="results"><h2>07 · 结果：总时长、阶段差额、时间误差和 MFU 分开看</h2><h3>v685：预测整个 iter 的账如何闭合</h3>')
parts.append('<pre>目标原始图时间      20,781.256754 ms\n+ 目标图外残差补偿      604.439668 ms\n= 预测 Profiler 时间  21,385.696422 ms\n+ outer 两时钟差额    1,372.453390 ms\n= 预测 Training 时间  22,758.149812 ms\n→ 预测 MFU                   3.309862%</pre>')
parts.append(figure('v685_svg','v685 目标 224 卡：预测与实际四轮平均总时间','图中 outer 放在末尾只是加法分账，不表示真实开销集中在末尾。预测收尾段包含图外残差的账面影响，不是纯优化器耗时。'))
phase_names={'entry':'迭代准备','onef1b':'全 rank 1F1B 区间','tail':'更新与通信收尾（含补偿分账）','outer':'两时钟差额 outer'}
parts.append(table(['时间段','预测秒','实际均值秒','实际−预测 ms','该段 MAPE %'],[[phase_names[r['phase']],f(r['predicted_ms']/1000),f(r['actual_ms']/1000),f(r['underprediction_ms']),f(r['mape_pct'])] for r in audit['phase_summary']]))
parts.append('<p>前三段差额加总为 Profiler 平均低估约 2.474 秒；加 outer 差额后为 Training 平均低估约 2.733 秒。各段 MAPE 不能相加。T35/T38 保持入口、收尾与 outer 的预测值不变，主要改变 1F1B 及单列跨轮状态。</p>')
version_names={'v685_frozen':'v685 正式','v686_split_full':'v686 阶段候选','v687_split_full':'v687 就绪候选','v688_split_ensemble':'v688 两源轮场景','v688_full4_ensemble':'v688 四源轮场景'}
version_rows=[r for r in rows('version_iterations') if r['split']=='development_primary']
version_summary=[]
for key,label in version_names.items():
    g=[r for r in version_rows if r['variant']==key]
    if g:version_summary.append([label,'85/90/95/100',f(mean(float(r['profiler_ape_pct']) for r in g)),f(mean(float(r['training_ape_pct']) for r in g)),f(mean(float(r['mfu_relative_ape_pct']) for r in g))])
parts.append(details('v685–v688：相同目标四轮的 Step / MFU 回归',table(['模型','目标轮次','Profiler MAPE %','Training MAPE %','MFU 相对 MAPE %'],version_summary)+'<p>后续候选尚未在该事前预测条件下超过正式 v685。v688 的场景跨度不是统计置信区间。</p>'))
parts.append(explorer('version-iterations','查看 v685–v688 全部目标逐轮结果',['版本','轮次','Profiler预测秒','Profiler实际秒','Profiler APE %','Training预测秒','Training实际秒','Training APE %','MFU预测%','MFU参考%','MFU相对APE%'],[[version_names.get(r['variant'],r['variant']),r['iteration'],f(float(r['predicted_profiler_ms'])/1000),f(float(r['actual_profiler_ms'])/1000),f(r['profiler_ape_pct']),f(float(r['predicted_training_ms'])/1000),f(float(r['actual_training_ms'])/1000),f(r['training_ape_pct']),f(r['predicted_mfu_pct']),f(r['actual_mfu_pct_derived']),f(r['mfu_relative_ape_pct'])] for r in version_rows]))
parts.append(binding_results_section(table,details,note))
parts.append('<h3>三种使用场景：切换为同一组迭代再比较</h3><label>评价范围 <select id="result-scope"><option value="all_including_initialization">85/90/95/100，包含初始化的四轮</option><option value="primary_causal_walk_forward">90/95/100，初始化后的三轮</option></select></label>')
parts.append(details('T35 / T38 的计算关系：新增观测在什么位置使用','<p><b>T35：</b>等当轮规定的首批 F/B 完成，将每个已观测区间除以对应源侧区间，按前向/反向分别计算缩尾均值倍率；用倍率更新同方向 F/B 节点，然后重算整图。</p><pre>目标 F/B 节点时长 = 源侧 F/B 节点时长 × 当轮同方向倍率\nT35 预测 = 更新节点成本后的整图回放结果\n\nT38 当前轮预测 = T35 当前轮预测\n              +（前驱采样轮实际时长 − 前驱采样轮 T35 预测）\n85 轮：尚无前驱，历史修正为 0。</pre><p>T38 使用前驱的原始 T35 残差，不递归使用已经修正后的误差。当前轮真实结束时间只在结束后的评估中读取；本页复用已封存结果，没有重新寻找倍率或残差规则。</p>'))
method_names={'formal_v685_cold_start':'v685 事前预测','t35_midrun_winsor20':'T35 当轮前缀','t38_lagged_base_residual':'T38 前缀 + 历史残差'}
for scope in ['all_including_initialization','primary_causal_walk_forward']:
    g=[r for r in rows('T38_metrics') if r['split']==scope]
    parts.append('<div data-result-scope="'+scope+'">'+table(['方法','轮数','1F1B MAPE %','剩余时间 MAPE %','Profiler MAPE %','Training MAPE %','MFU相对MAPE %','平均已运行 %'],[[method_names[r['method']],r['iterations'],f(r['onef1b_MAPE_pct']),f(r['remaining_MAPE_pct']),f(r['profiler_MAPE_pct']),f(r['training_MAPE_pct']),f(r['MFU_relative_MAPE_pct']),f(r['mean_cutoff_fraction_pct'])] for r in g])+'</div>')
parts.append('<p>剩余时间从各方法可用时点开始算；已过去的时间越长，剩余误差的分母越小。MFU 与 Training 共用同一时钟但反比换算，误差率不同：例如时间低估10%，MFU相对高估约11.11%。</p>')
parts.append(explorer('online-iterations','查看三种场景的 12 行逐轮结果',['方法','轮次','实际1F1B秒','预测1F1B秒','1F1B APE %','剩余APE %','Training APE %','MFU相对APE %','已运行 %'],[[method_names[r['method']],r['iteration'],f(float(r['actual_onef1b_ms'])/1000),f(float(r['predicted_onef1b_ms'])/1000),f(r['onef1b_APE_pct']),f(r['remaining_APE_pct']),f(r['training_APE_pct']),f(r['MFU_relative_APE_pct']),f(r['cutoff_fraction_pct'])] for r in rows('T38_iterations')]))
factors=[r for r in rows('T35_factors') if r['method']=='online_midrun_last_half_winsor20']
assert len(factors)==4
parts.append(details('查看 T35 当轮倍率与 T38 前驱残差（不是源侧校准参数）',table(['方法','轮次','F倍率','B倍率','F样本','B样本','F在源范围内','B在源范围内'],[['T35 当轮前缀',r['iteration'],f(r['forward_factor'],4),f(r['backward_factor'],4),r['forward_samples'],r['backward_samples'],'是' if r['forward_within_source_fit_range']=='True' else '否','是' if r['backward_within_source_fit_range']=='True' else '否'] for r in factors])+table(['当前轮','前驱采样轮','当轮T35毫秒','前驱残差毫秒','最终预测毫秒'],[[r['iteration'],r['predecessor_iteration'] or '初始化',f(r['base_predicted_onef1b_ms']),f(r['lagged_base_residual_correction_ms']),f(r['predicted_onef1b_ms'])] for r in rows('T38_chain')])+ '<p>前驱是采样链85→90→95→100，间隔5个训练iter；并未验证连续每一步t−1更新效果。</p>'))
parts.append('</section>')
parts.append('<section id="diagnosis"><h2>00 · 误差诊断</h2>')
parts.append('<div id="pp-diagnosis"><h3>通信就绪：源侧局部关系与完整图预测分开</h3><p>先在已观测API入口条件下检查单条消息关系，再判断能否用模型生成的入口时刻接回完整图。v687的d/c是独立候选参数，不是v685已经采用的固定等待。</p>'+details('查看候选通信依赖语义图',causal_evidence)+pp_evidence+'</div>')
parts.append('<h3>lane0 诊断：解释的是低估差额，不能当新的预测精度</h3><p>局部 v687 原预测 <b>18.101 秒</b>，四轮实际平均 <b>21.522 秒</b>。用目标观测替换 F/B 内四类耗时后，重算为 <b>21.471 秒</b>；低估差额从 3.420 秒缩到有符号平均 51.1 ms。替换后平均绝对差额为 57.0 ms，且各轮正负不同。</p>')
parts.append(details('查看总时间、替换公式与误差分配图',figure('lane_svg','目标 lane0：无可见 GPU 活动与仅通信区间对低估影响最大','14 rank 的观测替换诊断；不是全部224卡误差，不是源侧预测升级，柱子不是可节省时间。')))
parts.append('<p>替换公式为 <code>新 F/B 时长 = 原预测时长 +（该类观测 − 该类预测）</code>，不是清零。无 GPU 活动记录可能包含提交间隔、等待或记录缺失；仅通信活动也可能含等待/轮询。两项影响大不能直接认定物理根因。</p></section>')

parts.append(scaleout(table,details,note,explorer))
parts.append('<section id="limits"><h2>08 · 已验证边界、未解决问题与下一步</h2>')
parameter_review=json.loads((OUT/'parameter_review.json').read_text())
assert parameter_review['status']=='PASS'
parts.append(note('<b>修正后的问题定位：</b>16、224、256 卡的主要训练参数和模型维度已经具备；当前问题是这些已知信息是否正确落实到节点、成本及执行顺序，不能笼统写成“缺少模型参数”或“形状变化导致当前误差”。下面列出的是待检验问题，不是已经证明的根因。'))
parts.append('<h3>先分清两条任务：静态形状相同与静态形状变化</h3>')
parts.append(table(['研究范围','已经掌握的配置','对后续工作的含义'],[
('256→224：当前 1F1B 主任务','两侧每微批次样本数均为2、序列8192、CP2、TP1；hidden 5120、MLA/FFN/MoE核心维度一致。层数60→52、PP16→PP14、每轮微批次数4→3。','对应层的核心静态算子形状基本一致；优先检查同形状成本绑定、层与微批次执行次数、调度及运行时迁移。配置相同不证明每次实际 kernel 耗时相同。'),
('16→256：独立外推案例','每微批次样本4→2、CP1→CP2，按序列均匀切分计算的局部token数32768→8192；层数4→60、PP2→PP16、微批次数1→4。','尺寸变化可由已有配置确定；待验证的是各类算子耗时如何随尺寸变化，不能把所有计算统一视为耗时缩至1/4。本节只列该任务后续方向，不启动其研究。')]))
parameter_names={'hidden_size':'隐藏维度','num_attention_heads':'注意力头数','ffn_hidden_size':'稠密FFN中间维度','moe_ffn_hidden_size':'路由专家中间维度','moe_shared_expert_intermediate_size':'共享专家中间维度','q_lora_rank':'Q低秩投影维度','kv_lora_rank':'KV低秩投影维度','qk_head_dim':'Q/K非位置部分头维度','qk_pos_emb_head_dim':'Q/K位置部分头维度','v_head_dim':'V头维度','num_experts':'路由专家总数','moe_router_topk':'每token选择专家数','world_size':'GPU/rank数','pp':'流水线并行PP','cp':'上下文并行CP','tp':'张量并行TP','dp':'数据并行DP','ep':'专家并行EP','num_layers':'模型层数','micro_batch_size':'每微批次样本数','global_batch_size':'全局batch','microbatches':'每轮微批次数','sequence_length':'序列长度'}
static_parameter_evidence='<div id="three-case-parameters">'+details('展开已有三组参数与证据范围',table(['参数','16卡','224卡','256卡','224卡本次证据'],[[parameter_names[r['field']],r['source16'],r['target224'],r['source256'],'启动日志字面值一致' if r['target224_evidence']=='LOGGED_LITERAL_MATCH' else '冻结运行合同' if r['target224_evidence']=='FROZEN_RUN_CONTRACT' else '架构配置已声明；有限日志片段未出现'] for r in parameter_review['rows']])+ '<p>16/256采用已提交实际参数记录，224采用既有运行合同与v6.0架构配置，细项与T24/T25已经提取的启动字段交叉核对。参数已知与“部署代码版本已独立证实”是两回事；有限日志中未出现也不表示参数缺失或使用默认值。本次未重扫原始日志或trace。</p><p><a href="/results/w37/A/integration-20260907/three_case_parameters.csv">三组参数机读表</a> · <a href="/results/w37/A/integration-20260907/parameter_review.json">固定输入与逐项核验</a></p>')+'</div>'
parts.append('<p>对应配置及其来源见<a href="#three-case-parameters">第04章三组参数对照</a>；这里聚焦已知配置下仍需核验的建模问题。</p>')
parts.append('<h3>256→224：按以下顺序处理主要问题</h3>')
parts.append(table(['优先事项','当前依据与未验证边界','先用现有材料完成什么'],[
('1 · 细粒度绑定已核验，整轮来源仍待闭合','55个聚合键未直接命中由格式/粒度差异解释；独立候选更新70,192个内部节点，目标1F1B改善但收尾退化。','为源侧建立兼容全局层号/MB键的整图回放，先核对计算与旧完整区间时长下限的归属；用源95/100做增量开发回归，区分局部成本改善与整轮有效性。'),
('2 · 执行次数、顺序与运行时是否迁移正确','PP/层数/微批次数变化已知；API入口、数据就绪、GPU完成仍不是同一时刻，部分延迟代理尚未物理拆清。','核对层放置、首尾stage、F/B与重计算次数、本地程序顺序和PP双端条件；先查现有代码及边表，确需改结构时另建候选，不改正式锁。'),
('3 · MoE实际路由是否改变成本','专家数与top-k已知，但224/256日志显示未强制负载均衡、未按容量固定填充；静态维度不能唯一决定每轮专家收到的token数。','先盘点既有DeepEP、路由及GPU活动记录，提取可用的每层/专家token分布和调用尺寸；区分已有但未绑定、记录未覆盖和确实需补测。目标观测仅用于开发诊断，不回填源校准。'),
('4 · 通信、等待和重叠是否重复或遗漏计时','阶段区间与通信活动可能相互包含；依赖图能表达并发，但这本身不证明已预测资源争用造成的减速。','逐项核对PP/CP/EP/DP/EDP的服务、就绪与等待边界，检查是否重复相加；区分源实测已包含的争用和目标需要重新预测的争用。先做源侧同口径消融，再检查目标与其他阶段回归。'),
('5 · 两项总量补差究竟承载什么','604.440 ms补到Profiler，1372.453 ms outer补到Training；只有总量，不能均分到节点或全部归到优化器。','用已有训练边界、API/GPU记录核对两种时间口径及未归属区间；记录覆盖与时钟限制。证据不足时保留明确残差，不人为插入物理等待。'),
('6 · 已知模型参数是否正确用于MFU计数','已有有效FLOPs常数及BF16 500 TFLOPS峰值输入；不等于已从当前模型与执行图独立复核，峰值是用户确认来源。','依据已有MLA/MoE配置核对有效前向/反向、全局batch和分片计数，排除额外重计算重复进入MFU分子；分别报告Training时间误差、MFU相对误差和百分点差。')]))
parts.append('<p><b>主任务唯一下一步：</b>在已完成键格式和绑定覆盖审计的基础上，为源侧建立兼容细粒度成本键的整图回放，核对暴露计算区间的旧时长下限是否重复承载计算，再验证固定规则的整轮误差；不在目标旧四轮上继续挑倍率或补等待。</p><p><b>16→256的后续方向：</b>在独立任务中建立算子类型与已知输入形状对应的耗时验证，重点检查1/4线性缩放、新增CP2通信和跨host运行条件；保留现有45.7773% Training MAPE与84.8177% MFU相对MAPE，不以目标已见数据改选主方法。</p>')
parts.append('<h3>验证边界继续保留</h3><p>当前主基线仍是v685、目标四轮1F1B MAPE为11.340786%。T35/T38使用当轮前缀或前驱目标残差，不能充作源侧事前预测升级；所有旧目标数据保持开发评价标签。T38在目标三次转换改善但源侧同规则退化，sign-test p=0.125，尚不足以推广。</p><p>隐藏graph设备覆盖、跨stream因果、部分跨host时钟和部署代码版本仍未完整证实。静态参数核对通过不会自动消除这些观测边界。在线辅助按T41合同，取得至少两轮设计时未见、更晚且连续的目标数据后再检验固定方法；不在旧四轮反复挑规则。</p>')
parts.append(note('<b>当前结论（两句）：</b><br>已解释58个聚合键仅3键直接命中的原因；保留层号/微批次及历史运行时倍率的独立候选，将224卡开发集1F1B MAPE从11.3408%降至10.5616%。<br>候选收尾MAPE上升0.5197个百分点，源侧整轮回放和旧时长下限尚未独立验证，因此保留正式v685；16→256、在线辅助与目标事后诊断仍按各自数据边界单列。'))
parts.append('</section>')

parts.append(tool_todo(table,note))
context_audit=json.loads((OUT/'context_audit.json').read_text())
parts.append('<section id="sources"><h2>09 · 证据、名词审核与复现</h2><p id="audit-summary">原有 '+str(audit['input_count'])+' 个固定输入，合计 '+f(audit['input_bytes']/1e6,2)+' MB；'+str(audit['check_count'])+' 项数值/哈希/范围断言通过。另有 14 个固定提交的历史与外推输入，共 289,988 字节，新增 '+str(context_audit['check_count'])+' 项来源和既有结果公式核验。节点/边来自派生表，未扫描原始 trace。本HTML生成步骤不运行模型或选择参数；新增绑定实验采用独立封存结果，输入、消融和验收见其研究入口。</p>')
parts.append('<pre>/home/zjb/Desktop/fabric-data-analysis/.snakemake-venv/bin/python -m snakemake \\\n  --snakefile research/w37/onef1b/integration/Snakefile \\\n  --directory results/w37/A/integration-20260907 --cores 1</pre><p>这是本次文档专用管线，执行输入校验 → 证据统计 → HTML 生成 → 文档检查；不调用历史模型 builder 或总 Snakemake 目标。浏览器检查另有固定脚本，验证网页与离线文件。</p>')
parts.append('<p class="no-print"><a href="http://192.168.0.49:8037/research.html">研究首页</a> · <a href="http://192.168.0.49:8037/v685.html">v685 说明</a> · <a href="http://192.168.0.49:8037/v684.html">v684 历史说明</a> · <a href="http://192.168.0.49:8037/docs/w37/1f1b/WEB_INDEX.md">文档索引</a> · <a href="http://192.168.0.49:8037/docs/w37/1f1b/integration/REPORT.md">本次整合审计报告</a> · <a href="http://192.168.0.49:8037/docs/w37/1f1b/integration/HANDOFF.md">复现与交接</a></p>')
baseurl='http://192.168.0.49:8037/results/w37/A/integration-20260907/'
artifacts=['audit.json','graph_inventory.csv','graph_node_types.csv','graph_edge_types.csv','compute_binding_summary.csv','compute_physical_parameters.csv','compute_node_updates.csv','gradient_parameters.csv','gradient_node_updates.csv','pp_direction_metrics.csv','v685_phase_summary.csv','version_iteration_results.csv','online_iteration_results.csv','online_metrics.csv','outer_source_clocks.csv','input_verification.json','audit_checks.json']
parts.append(details('机读结果与下载（在线证据）','<ul>'+''.join('<li><a href="'+baseurl+name+'">'+name+'</a></li>' for name in artifacts)+'</ul>'))
parts.append(details('展开全部输入来源与 SHA256',table(['证据键','路径','字节','SHA256'],[[r['key'],r['path'],r['bytes'],r['sha256']] for r in verified])))
parts.append(details('展开新增历史与外推证据的固定提交和 SHA256',table(['证据键','Git 提交','原始路径','字节','SHA256'],[[r['key'],r['commit'],r['git_path'],r['bytes'],r['sha256']] for r in context_records])))
parts.append('<p>所有名称审核遵循：先说明概念，再给字段或公式；源侧与发送方分开，API入口与完成上界分开，整体耗时与差额分开，全部rank与lane0分开，校准/继承/观测替换分开。历史科学产物保留，图中示例明确标为示意。</p></section>')
section_markup=''.join(parts)
# Place the already generated static evidence once, next to the model inputs.
section_markup=section_markup.replace('<h3>训练场景参数，与拟合成本分开</h3>', '<h3>训练场景参数，与拟合成本分开</h3>'+static_parameter_evidence)
body=header+organize(section_markup,table,details,note)+'<footer>W37 · 大语言模型分布式训练与 MFU 性能预测 · 关键图表和参数内嵌 · 文档更新 2026-09-08</footer>'
# Downloaded files retain usable online evidence links without depending on their local path.
body=re.sub(r'href="(/[^\"]*)"',lambda m:'href="http://192.168.0.49:8037'+m.group(1)+'"',body)
css=Path(__file__).with_name('style.css').read_text();js=Path(__file__).with_name('report.js').read_text()+'\n'+Path(__file__).with_name('full_iter.js').read_text()
blob=json.dumps(datasets,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
full_blob=(OUT/'full_iter_payload.json').read_text().replace('<','\\u003c').replace('&','\\u0026')
document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>W37｜分布式训练、MFU建模与预测验证</title><style>'+css+'</style></head><body>'+body+'<script id="report-data" type="application/json">'+blob+'</script><script id="full-iter-data" type="application/json">'+full_blob+'</script><script>'+js+'</script></body></html>'
(OUT/'integrated_report.html').write_text(document,encoding='utf-8')
dump('terminology_review.json',[{'term':r[0],'definition':r[1],'avoid':r[2],'reviewed':True} for r in terms])
dump('render_manifest.json',{'inputs':verified,'context_inputs':context_records,'context_audit_sha256':digest(OUT/'context_audit.json'),'html_sha256':digest(OUT/'integrated_report.html'),'datasets':{k:len(v['rows']) for k,v in datasets.items()},'calibration_map_sha256':digest(OUT/'v685_calibration_map.svg'),'model_cost_inventory_sha256':digest(OUT/'model_cost_inventory.json'),'full_iter_payload_sha256':digest(OUT/'full_iter_payload.json'),'full_iter_charts':3,'full_iter_iterations':[85,90,95,100],'self_contained_images':True,'organization_map_sha256':digest(OUT/'organization_map.json'),'main_sections':8,'new_prediction':False})
print(json.dumps({'stage':'render','html_bytes':len(document.encode()),'datasets':{k:len(v['rows']) for k,v in datasets.items()}},ensure_ascii=False))

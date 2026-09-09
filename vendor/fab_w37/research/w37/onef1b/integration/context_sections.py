"""Background, committed extrapolation results and explicitly unexecuted research TODOs."""
import html
import json
from common import OUT

COMMIT = 'e71ccb0b9157053e13a7e95e957659f2d8170764'
GITHUB = 'https://github.com/zjb2023/mfu-model/blob/' + COMMIT + '/'
BASE = 'http://192.168.0.49:8037/results/w37/A/integration-20260907/'
V4 = 'case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v4_oisa_s5000_slice256_buffer128/'


def link(url, label):
    return '<a href="' + html.escape(url, quote=True) + '">' + html.escape(label) + '</a>'


def background(table, details, note):
    guide = link(GITHUB + 'docs/MFU_THREE_STAGE_AND_DAG_MODEL_GUIDE.md', 'GitHub 原说明：三阶段与计算图（固定 e71ccb0）')
    minimal = link(GITHUB + 'case_256gpu_pp16_cp2_a2a/results/pp_dag_minimal/PP_DAG_MINIMAL.md', '最小 PP 计算图：前向、反向与等待')
    v1 = link(GITHUB + 'case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v1/ERROR_ANALYSIS.md', 'v1 误差分析')
    v2 = link(GITHUB + 'case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v2/PP_OPTIMIZER_DAG_V2.md', 'v2 软件处理校准')
    v4 = link(GITHUB + V4 + 'PP_OPTIMIZER_DAG_V4_OISA_S5000.md', '历史 v4：九类通信接入与校准回放')
    s = '<section id="why-dag"><h2>00 · 为什么改用计算图来预测 MFU，并走向跨规模外推</h2>'
    s += '<p><b>最终要回答的是：仅掌握源侧测量和目标静态配置，能否预测目标训练一整轮需要多久、MFU 是多少。</b>扩大 GPU 数量后，层放置、微批次数、通信组和流水线长度都会改变；旧配置的阶段平均耗时与平均等待，不能简单按卡数比例搬过去。</p>'
    s += '<h3>从“各阶段花多久”，推进到“每张卡何时能开始下一项工作”</h3><p>原三阶段 MFU 模型按前向 FWD、反向 BWD（含重计算）、优化器 OPT 汇总估算。这样的分类便于解释工作量，但仅有阶段汇总不足以确定不同卡之间的先后与并发。原 256 卡案例为 PP16、每轮 4 个微批次，各阶段层数不均衡，也没有很长的全流水线稳定运行区间。前端还在等待梯度时，末端可能已经进入收尾。</p><p>因此这里“用计算图替换三阶段模型”，具体指<b>用依赖图承担执行时间预测，保留前向、反向、优化器三阶段用于统计和说明</b>。不是取消这三个阶段，也不意味着所有三阶段近似方法都不可用。' + guide + '。</p>'
    s += table(['外推时要回答的问题', '只有阶段汇总时缺什么', '计算图怎样表达'], [
        ('PP2 变为 PP16，第一批激活和梯度何时到达各卡？', '流水线填充、排空及逐跳等待的具体顺序', '按目标 PP、微批次及训练调度代码重建前向/反向依赖'),
        ('某项通信增加 10 ms，会不会使整轮多 10 ms？', '该通信是否被其他工作覆盖、是否位于决定结束的依赖链上', '重算依赖完成时刻；非关键分支可能被覆盖，关键分支可能延长整轮'),
        ('一部分卡已做优化器，另一部分仍在反向，如何计时？', '跨卡并行与通信组成员就绪条件', '保留各 rank 与通信组的依赖，求全局结束；不把重叠区间再次相加'),
        ('卡数和并行方式改变后，哪个假设造成低估？', '平均值中计算、通信、运行时与等待混在一起', '分别登记节点成本来源与依赖来源，逐类消融后比较完整 iter')])
    s += '<h3>计算图如何连接源侧校准和目标预测</h3><div class="cards"><article><h3>① 源侧测量</h3><p>提取计算、通信服务及运行时成本，登记校准轮次和输入来源。</p></article><article><h3>② 目标静态配置</h3><p>用卡数、层放置、并行组、微批次数和调度代码建立目标节点与依赖。</p></article><article><h3>③ 重算目标执行</h3><p>将可迁移成本绑定到目标节点；依赖与资源顺序共同决定等待、重叠和结束时刻。</p></article><article><h3>④ 完整 iter → MFU</h3><p>保留准备、F/B 和优化器收尾，明确图外时间，再用相同 FLOPs 与峰值口径换算 MFU。</p></article></div>'
    s += '<pre>节点开始 = max(所有前置节点完成时刻)\n节点结束 = 节点开始 + 该节点的成本\nMFU(%) = 100 × 模型有效 FLOPs / (GPU 数 × 单卡峰值 FLOPs/s × Training 秒数)</pre><p>前置条件既包括数据依赖，也要表达训练代码要求的本地执行顺序与资源串行关系。OISA 提供网络传输成本；通信发起、同步、完成处理等运行时成本需要另有证据。某个阶段区间若已包含通信，就不能再加一遍。</p>'
    s += note('<b>计算图提供可外推的执行结构，不自动保证成本可外推。</b>模型内等待可由依赖推导，但未识别的运行时行为、错误的算子缩放和缺失的资源竞争仍会使预测失准。历史版本仍有实测残差补偿，不能宣称所有等待都已得到独立物理解释。第08章的16→256案例用于检验这一步。')
    s += '<div id="historical-development"><p>历史256卡模型先验证同配置回放，再检查跨配置迁移；以下结果保留各自的场景和轮次。</p>'
    s += table(['历史步骤', '观察范围', '结果', '推动的下一步'], [
        ('最小 PP 图', '256 卡 F/B 局部图：2,048 个 F/B 节点、1,920 条 PP 消息、7,312 条边', '表达逐卡、逐微批次前向与反向传递', '补入优化器和完整 iter；局部区间误差不能作整轮精度'),
        ('完整 iter v1', '256 卡同配置、第 55 轮', 'Profiler 19,970.490 ms，对照 21,638.095 ms；有符号时间误差 −7.7068%', '反向逐跳运行时完成延迟等成本需要补证'),
        ('运行时校准 v2', '同一 256 卡案例的校准回放', 'Profiler 21,616.236 ms；有符号误差 −0.1010%；MFU 相对误差 +0.0954%', '把网络与软件/同步代理分开；继续接入其他通信类别'),
        ('九类通信 v4', '同配置校准回放；九类非 PP 通信接入 OISA', '保持约 0.1% 回放误差，使用 OISA 基线加实测有符号残差', '需要换配置检验；低回放误差不等于跨规模能力'),
        ('W37 两条验证路径', '256→224 事前预测研究；16→256 源侧成本外推', '使用不同输入和指标窗口，后文分别展示', '核验调度结构、成本迁移和运行时缺口；不混作连续精度曲线')])
    s += '<p>v2 的反向完成代理为网络 4.712346 ms + 软件/同步等余项 93.463844 ms；后一项是实测校准代理，不是已单独测得的纯软件服务。历史 v4 保留这项 PP 校准，并把另九类通信接入 OISA；每类使用代表点及大小缩放，其中五类 F/B 通信仍包含在阶段成本内。它的校准残差有正有负，不能都当作可独立相加的物理开销。' + v1 + '；' + v2 + '；' + v4 + '。</p>'
    s += '<p><b>约 0.1% 到底指什么：</b>历史第 55 轮的 Profiler 时间误差绝对值约 0.1010%，不是多轮 MAPE。其 Training 实际/预测为 22,940.100 / 22,918.240 ms；MFU 实际/预测为 4.4065% / 4.4107%。这验证了已校准案例的还原能力，不能用作 224 卡或 16→256 的外推精度。第06章16.82%→10.46%是224卡另一组多轮评估。</p>'
    reading = [guide, minimal, v4, link(BASE + 'context/mfu/' + V4 + 'pp_optimizer_dag_v4_oisa_s5000.html', '历史 v4 完整流程可视化（固定提交副本，8037）'), link(BASE + 'context/mfu/' + V4 + 'slice_pipeline_principle.html', '历史通信流水原理：分片、Ring 与持久通信流（8037）'), v1 + ' → ' + v2]
    s += details('历史版本材料与交付来源', '<ol>' + ''.join('<li>' + x + '</li>' for x in reading) + '</ol><p>上述说明按 mfu-model/main 的固定提交 e71ccb0 阅读。用户当时的交付记录还列出 mfu-model/feat/parallel-strategy-224gpu：132d657，以及 oisa-simulator/feat/mfu-multiring-fct：4e29411；这是历史交付记录，不声明今天的分支最新状态。本次没有推送、合并或加入当时未跟踪的 oisa_s5000_256gpu_nine_class_channel_strategy_v2/。</p><p>原可视化曾部署在 192.168.0.49:8013。本页提供从同一提交提取的 8037 副本，不依赖旧服务是否仍在运行。历史文件保持原内容，作为历史证据查看。</p>')
    return s + '</div></section>'


def comparison_plot(target, key, title, unit, divisor=1):
    """Same 20 iterations; two separate axes/units for time and MFU."""
    e = html.escape
    maxv = max(float(r[p + '_' + key]) / divisor for r in target for p in ('predicted', 'actual')) * 1.15
    x = lambda i: 75 + i * 43
    y = lambda v: 280 - float(v) / divisor / maxv * 205
    s = f'<svg class="scaleout-plot" role="img" aria-label="{e(title)}" viewBox="0 0 1000 345" xmlns="http://www.w3.org/2000/svg"><rect width="1000" height="345" fill="white"/><text x="20" y="30" font-size="20" fill="#173249">{e(title)}</text>'
    for k in range(5):
        v = maxv * k / 4; yy = 280 - 205 * k / 4
        s += f'<path d="M75 {yy} H930" stroke="#dae2e8"/><text x="10" y="{yy+5}" font-size="14">{v:.1f}</text>'
    for prefix, color, label, dy in [('predicted', '#bd4f27', '源 16 卡成本模型预测', 0), ('actual', '#247e9a', '目标 256 卡观测换算' if key == 'mfu_pct' else '目标 256 卡训练日志', 20)]:
        pts = ' '.join(f'{x(i)},{y(r[prefix+"_"+key])}' for i, r in enumerate(target))
        s += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/><text x="610" y="{53+dy}" fill="{color}" font-size="15">{label}</text>'
        for i, r in enumerate(target):
            s += f'<circle cx="{x(i)}" cy="{y(r[prefix+"_"+key])}" r="3" fill="{color}"><title>第 {r["iteration"]} 轮：{float(r[prefix+"_"+key])/divisor:.4f} {unit}</title></circle>'
    for i, r in enumerate(target):
        s += f'<text x="{x(i)}" y="305" text-anchor="middle" font-size="13">{r["iteration"]}</text>'
    return '<figure>' + s + '<text x="20" y="54" font-size="14">' + e(unit) + '</text><text x="440" y="333" font-size="15">训练迭代编号</text></svg><figcaption>主方法固定预测与目标 20 轮逐点对照；横轴为采样轮次。两张图分别使用秒和 MFU 百分比。</figcaption></figure>'


def scaleout(table, details, note, explorer):
    audit = json.loads((OUT / 'context_audit.json').read_text())
    target = audit['target']; summary = audit['summary']
    f = lambda v: f'{float(v):.4f}'
    s = '<section id="scaleout"><h2>00 · 外推验证案例：只用 16 卡成本预测 256 卡</h2><p>这是独立的跨规模研究案例，整合自固定提交 <code>c3f39c2</code>。本页只复核已封存结果，不重新执行 Task B。它检验“源侧成本 + 目标静态结构 → 完整 Step → MFU”的端到端外推；不使用历史 source256 校准成本来冒充 source16 成本。</p>'
    s += note('<b>主方法的外推误差仍大：</b>256 卡 Training 时间 MAPE <b>45.7773%</b>，Profiler 时间 MAPE <b>44.0737%</b>，MFU 相对 MAPE <b>84.8177%</b>。源 16 卡留出集约 3% 只反映同配置留出效果，不是 16→256 精度。')
    s += table(['配置或数据角色', '源 16 卡', '目标 256 卡'], [
        ('TP / PP / CP / DP / EP', '1 / 2 / 1 / 8 / 8', '1 / 16 / 2 / 8 / 8'),
        ('层数 / stage 层放置', '4 / [2,2]', '60 / [2,4×14,2]'),
        ('每微批次样本 / 全局批量 / 每轮微批次数', '4 / 32 / 1', '2 / 64 / 4'),
        ('序列长度 / hidden / vocab', '8192 / 5120 / 128256', '相同'),
        ('硬件 / host', 'S5000，2 个八卡 host', 'S5000，32 个八卡 host'),
        ('校准', 'wo_mccllog_1116：5–60，每 5 轮，12 点', '不提供动态成本校准'),
        ('留出与评价', '65–100，每 5 轮，8 点计算隔离留出', '5–100，每 5 轮，20 点开发评价')])
    s += '<p>这不只是把卡数乘 16：模型由 4 层变为 60 层，PP2→PP16、CP1→CP2，批量与微批次数也改变。两侧采用同类 MLA/MoE、全 block 重计算、BF16 结构；另一份 16 卡采集 w_mccllog_1129 单独登记，不混入主成本。</p>'
    s += details('主方法如何把源成本迁移到目标', table(['项目', '已实施方法', '未验证边界'], [
        ('计算', '源每层/方向慢 rank GPU 服务中位数，按目标本地 token 比例 1/4 缩放', '源只有一种形状，线性效率迁移是假设'),
        ('PP 通信和流水等待', '源启动项与字节数代理；按目标 PP/MB 重建阻塞通信依赖，等待由双方释放与调度顺序产生', '不把源 SendRecv 含等待生命周期整体当作网络服务'),
        ('EP / CP / DP / EDP 通信', '源 EP 组服务；CP 使用源 host 内代理；DP/EDP 按目标静态组和 host 分层', '新增 CP2、跨 host 拥塞和目标集合通信行为缺独立微基准'),
        ('运行时与优化器', '源参数拥有量、GPU 服务及源残差/outer 迁移', '残差未物理拆清，目标双 gather 行为仍是假设')]) + '<p>训练 commit、NIC/交换机路径和跨 host 时钟偏差界尚未确认。这些是迁移风险，当前误差表本身不能确定各项物理贡献。</p>')
    names = {'M2_source_only_linear': 'M2 主方法：源成本按本地 token 线性迁移', 'B0_constant_work_efficiency': 'B0 对照：沿用源有效算力效率', 'B1_compute_bubble': 'B1 对照：计算与流水空泡估计', 'M2_no_shape_speedup_sensitivity': 'M2 敏感性：不假设小形状带来提速'}
    source = next(r for r in summary if r['case_id'] == 'source16_holdout' and r['method'] == 'M2_source_only_linear')
    s += '<h3>先分清源侧留出和真正的跨规模评价</h3>'
    s += table(['主方法评价', '点数', 'Training MAPE %', 'Profiler MAPE %', 'MFU 相对 MAPE %', 'MFU MAE（百分点）'], [[label, r['n'], f(r['training_mape_pct']), f(r['profiler_mape_pct']), f(r['mfu_relative_mape_pct']), f(r['mfu_mae_percentage_points'])] for label, r in [('源 16 卡留出', source), ('目标 256 卡外推', next(r for r in summary if r['case_id'] == 'target256' and r['method'] == 'M2_source_only_linear'))]])
    s += '<h3>同一目标 20 轮：主方法、两项对照和预定敏感性分支</h3>'
    order = ['M2_source_only_linear', 'B0_constant_work_efficiency', 'B1_compute_bubble', 'M2_no_shape_speedup_sensitivity']
    s += table(['方法', '预测 Training 秒', 'Training MAPE %', '预测 MFU %', 'MFU 相对 MAPE %', 'MFU MAE（百分点）'], [[names[key], f(r['predicted_training_step_ms']/1000), f(r['training_mape_pct']), f(r['predicted_mfu_pct']), f(r['mfu_relative_mape_pct']), f(r['mfu_mae_percentage_points'])] for key in order for r in summary if r['case_id'] == 'target256' and r['method'] == key])
    s += '<p>主方法预测 Training <b>12.7385 秒</b>，实际为 <b>22.1810–25.5773 秒</b>；预测 MFU <b>5.9905%</b>，逐轮实际 MFU 均值为 <b>3.2482%</b>。主方法时间误差小于 B0/B1，但没有达到低误差外推。敏感性分支的 MFU 相对误差更小、时间误差却更大，因此保留其预定诊断角色，不按看到的目标成绩重新选主方法。</p>'
    s += comparison_plot(target, 'training_step_ms', '16→256：完整 Training 时间预测与日志对照', '秒', 1000)
    s += comparison_plot(target, 'mfu_pct', '16→256：MFU 预测与实际 Training 时间换算值', 'MFU %')
    s += explorer('scaleout-iterations', '查看 16→256 主方法全部 20 轮结果', ['轮次', 'Training预测秒', 'Training实际秒', 'Training APE %', 'Profiler预测秒', 'Profiler实际秒', 'Profiler APE %', 'MFU预测%', 'MFU实际换算%', 'MFU相对APE%', 'MFU绝对差（百分点）'], [[r['iteration']] + [f(float(r[k])/d) for k, d in [('predicted_training_step_ms', 1000), ('actual_training_step_ms', 1000), ('training_ape_pct', 1), ('predicted_profiler_step_ms', 1000), ('actual_profiler_step_ms', 1000), ('profiler_ape_pct', 1), ('predicted_mfu_pct', 1), ('actual_mfu_pct', 1), ('mfu_relative_ape_pct', 1), ('absolute_mfu_error_percentage_points', 1)]] for r in target])
    s += details('MFU 的分子、分母和为何时间误差不等于 MFU 误差', '<p>本案例使用用户确认的单卡 BF16 dense 峰值 500 TFLOPS，未独立核验厂商规格。源/目标每轮有效计算量分别为 4.218697266757632×10¹⁵ / 9.767709113843712×10¹⁶ FLOPs，主口径采用因果注意力三角有效位置、乘加各计一次、F+B=3F，不含重计算、norm/激活/优化器算术。分母用完整 Training 时间；Profiler 使用全 rank 标记包络。</p><pre>MFU 相对有符号误差 = (实际 Training 时间 / 预测 Training 时间 − 1) × 100%\nTraining 相对有符号误差 = (预测 Training 时间 / 实际 Training 时间 − 1) × 100%</pre><p>2.7423 个百分点是两个 MFU 百分数相减的绝对差均值；84.8177% 则逐轮除以实际 MFU 后取绝对均值。此处 FLOPs 口径与前述历史 v4、224 卡研究并不统一，不能把各自绝对 MFU 拼成版本改进曲线。封存的全方阵注意力敏感性口径没有因目标日志而改选为主分子。</p>')
    s += note('<b>数据边界：</b>源留出是模型读取隔离的同配置留出，研究者并非盲测；256 卡数据已在历史工作中观察，时钟修订也发生在暴露之后。本次结果称条件外推的开发评估。目标分项事后诊断不能回填为源侧校准。', True)
    rows_path = 'context/scaleout/results/w37/B/final-validation-20260905T233625518711Z/'
    s += '<p>机读证据：' + link(BASE + rows_path + 'point_results.csv', '固定提交的 112 行原始结果') + ' · ' + link(BASE + 'scaleout_summary.csv', '8 行汇总') + ' · ' + link(BASE + 'context_audit.json', '本次逐点公式与来源核验') + '。原任务的封存及数据隔离检查引用原提交记录，本次没有重新执行其模型或隔离 worker。</p>'
    report_file = next(r['output'] for r in audit['inputs'] if r['key'] == 'scaleout_final')
    s += details('Task B 最终报告的固定原文快照（文本，保留当时措辞）', '<pre>' + html.escape((OUT/report_file).read_text()) + '</pre>')
    return s + '</section>'


def tool_todo(table, note):
    candidates = [
        ('Charon', 'https://proceedings.mlsys.org/paper_files/paper/2026/hash/dbc8ce0fdfcd55172d73fb05dbae07fc-Abstract-Conference.html', 'MLSys 2026 论文：细粒度训练与推理性能预测', '优先核实计算/通信/运行时建模、训练精度定义、集群规模及代码能否复现；论文误差不能直接当 MFU MAPE。'),
        ('ASTRA-sim', 'https://github.com/astra-sim/astra-sim', '官方仓库：分布式 AI 的系统与网络联合仿真', '检查能否仅由源成本和目标配置生成执行输入；核验 Chakra 输入、集合通信、资源争用及转 MFU 所需工作量。'),
        ('SimAI', 'https://github.com/aliyun/SimAI', '官方仓库：训练计算、集合通信与网络仿真', '核实工作负载生成、计算校准与网络模型，在同一冻结场景下比较；本工程历史三阶段近似不是官方 SimAI 成绩。'),
        ('vTrain', 'https://github.com/VIA-Research/vTrain', '官方仓库：基于 profiling 的训练迭代时间预测', '检查 profiling 是否能限制在源侧；核验新形状、PP/CP/EP/MoE 和重计算的支持范围。'),
        ('Proteus', 'https://arxiv.org/abs/2306.02267', '原论文：复杂并行策略的分布式训练仿真', '核实计算通信重叠与资源竞争假设；提取逐案例时间误差、校准样本及规模，而非只摘论文平均值。'),
        ('Phantora', 'https://github.com/QDelta/Phantora', '官方仓库：运行训练框架并模拟集群执行', '核实单卡执行与集群模拟边界、MFU 输出口径，以及 Megatron/MCCL 和当前 MoE/CP 场景适配成本。')]
    s = '<section id="tool-todo"><h2>00 · 后续调研：哪些工具能预测大规模训练集群的 MFU</h2><p>先确认候选的一手资料，再安排同口径实验。以下是候选清单，不是精度排名；本轮仅核查资料入口，<b>尚未安装、复现或实测这些工具</b>。工具即使主要输出迭代时间，也可在统一有效 FLOPs、卡数和峰值口径下换算 MFU；直接打印 MFU 不代表口径已经一致。</p>'
    s += '<div class="scroll"><table><thead><tr><th>候选与一手来源</th><th>资料说明的定位</th><th>待查证与复现问题</th><th>状态</th></tr></thead><tbody>' + ''.join('<tr><td>' + link(url, name) + '</td><td>' + html.escape(position) + '</td><td>' + html.escape(todo) + '</td><td>资料入口已核查；实验待做</td></tr>' for name, url, position, todo in candidates) + '</tbody></table></div>'
    s += '<h3>拟定推进路线与验收，不提前填写精度</h3>'
    s += table(['顺序', '后续 TODO', '应交付的证据'], [
        ('1 · 口径合同', '统一完整 Training / Profiler 时钟、有效 FLOPs、BF16 峰值；分别报告 Step MAPE、MFU 相对 MAPE 和百分点 MAE', '每个工具的输入/输出字段映射及公式；不可比项单列'),
        ('2 · 能力与成本盘点', '优先 Charon，再按现有训练框架适配性筛选；核实 PP/CP/EP/DP、MoE、重算、通信库及可用代码', '固定版本、支持/未支持列表、机器资源和预计模拟时间；大规模运行前评估'),
        ('3 · 源侧校准', '仅使用约定源数据与独立微基准确定成本；目标只提供静态配置', '校准/留出/目标开发/未来盲测分区，成本来源表，预测前封存'),
        ('4 · 同场景复现', '分别比较同配置回放、256→224 配置迁移、16→256 条件外推；保留各自窗口', '完整迭代和阶段逐点误差、节点/边与关键路径、时间/内存开销'),
        ('5 · 误差归因', '检查计算形状缩放、通信服务、运行时就绪、等待和重叠是否正确，是否重复计时', '固定输入消融；源侧可部署改进与目标观测诊断分开'),
        ('6 · 新数据检验', '固定方法后，在设计时未见的更大规模或新配置上验证', '真正未见数据及冻结证据；已看过的 16/224/256 数据继续称开发评估')])
    s += note('合理目标是形成“同输入条件、同 MFU 定义、同迭代范围”的比较。不同论文报告的平均误差，可能来自同配置回放、不同模型或不同硬件；在这些条件核实之前，不将其与本工程 45.78% / 84.82% 直接排名。')
    return s + '<p>资料核查日期：2026-09-07。后续执行应使用独立版本、专用 Snakemake 管线及资源评估；本次没有启动工具对比实验。</p></section>'

"""Human-readable parameter system, separated by provenance and model version."""
import csv
import html
import json
from common import OUT

def node_parameter_figure():
    node_id = 'r0:s0:l0:bwd0:q3:end'
    with (OUT/'compute_node_updates.csv').open() as f:
        row = next(r for r in csv.DictReader(f) if r['node_id'] == node_id)
    key = html.escape(row['source_parameter_key'])
    cost = f"{int(row['steady_compute_ns']) / 1e6:.6f}"
    duration = f"{int(row['new_duration_ns']) / 1e6:.6f}"
    floor = f"{int(row['noncompute_floor_ns']) / 1e6:g}"
    s = '<figure id="parameter-node-map" class="parameter-node-map"><div class="scroll"><div class="node-map-canvas">'
    s += '<h3>一份计算成本，怎样对应到具体GPU上的一次任务</h3><div class="node-map-transfer">'
    s += '<article><b>① 计算键：查哪类工作</b><p>反向 · 稠密层之后 · 退出前暴露计算</p><code>'+key+'</code></article>'
    s += '<span class="node-map-arrow" aria-label="查表">→</span><article><b>② 计算成本：使用多长时间</b><p class="node-map-value">'+cost+' ms</p><p>按已有规则绑定到PP0的16个rank、3个MB，共48个节点。</p></article>'
    s += '<span class="node-map-arrow" aria-label="绑定到节点">→</span><article><b>③ 选其中一个节点查看</b><code>'+node_id+'</code><p>GPU rank 0上，MB0反向任务的结束边界节点。</p></article></div>'
    s += '<h4>把这个节点编号拆开读：r0 : s0 : l0 : bwd0 : q3 : end</h4><div class="node-map-fields">'
    for token, meaning, explanation in [
        ('r0','rank 0','本配置对应GPU 0'),('s0','PP stage 0','流水线第一个阶段'),
        ('l0','lane 0','stage内第0个位置；不是layer'),('bwd0','反向，MB0','bwd=backward；0是微批次编号'),
        ('q3','本地调度序号3','从0计数，第4个F/B任务'),('end','本次反向结束边界','不是整轮iter结束')]:
        s += '<article><code>'+token+'</code><b>'+meaning+'</b><p>'+explanation+'</p></article>'
    s += '</div><h4>它在PP0本轮调度中的位置（PP14 / MB3）</h4><div class="node-map-schedule">'
    for q,(direction,mb) in enumerate([('F',0),('F',1),('F',2),('B',0),('B',1),('B',2)]):
        s += '<article'+(' class="node-map-selected"' if q == 3 else '')+'><span>q'+str(q)+'</span><b>'+direction+'(MB'+str(mb)+')</b><small>'+('选中的任务 ↓' if q == 3 else '本地顺序')+'</small></article>'
    s += '</div><div class="node-map-detail"><b>展开 B(MB0)：</b> start → 本stage内的反向任务节点 → <strong>end（当前选中节点）</strong><p>这个end节点承载退出前计算成本：节点时长 = max('+cost+', '+floor+') = '+duration+' ms。名字叫end，不代表零时长。</p></div>'
    s += '</div></div><figcaption>上排箭头表示查表与参数绑定；下排箭头表示PP0的本地任务顺序。方框宽度不表示耗时；13.681142 ms只属于该退出节点，不是整个B(MB0)的耗时。例子取自已有节点更新表；一个节点编号定位一次任务，一个计算键可供多个节点复用。</figcaption></figure>'
    return s

def section(table, details, note):
    inv = json.loads((OUT/'model_cost_inventory.json').read_text())
    s = '<div id="parameter-system"><h3>性能模型使用哪些参数，v685更新了哪些</h3>'
    s += '<div id="compute-key-definition"><h3>“计算键”是什么：查找一类计算耗时的标签</h3><p><b>计算键是计算成本表的查找标签。</b>它描述“这是哪一类、哪个位置的计算”，用于找到对应的耗时数值，再按映射规则赋给计算图中的节点。可以理解为：<b>键是标签，成本是查到的时间，节点是实际使用这份时间的一次任务。</b>这里的“键”不是神经网络权重，也不是一个张量或一个GPU编号。</p>'
    s += '<p>v685这里使用五个字段组合成一个标签。以实际命中的反向计算为例：</p>'
    s += table(['字段','例子的原始取值','直观含义'],[
        ('方向','backward','反向计算'),
        ('层类型','dense','稠密层'),
        ('上下文','dense_to_step_exit','从稠密层之后到本阶段退出这一段'),
        ('执行位置','exposed_gap','模型中单独计入的暴露计算项，与重叠计算项区分'),
        ('语义槽','step_exit','本阶段退出位置；用于进一步区分同类区间')])
    s += '<pre>计算键：backward|dense|dense_to_step_exit|exposed_gap|step_exit\n查到的计算成本：13.681142 ms\n按当前映射规则绑定：PP0，rank 0–15，MB0/1/2\n一个键 → 一份计算成本 → 16 × 3 = 48 个目标节点</pre>'
    s += '<p>这五个字段本身不含rank、PP或MB编号；具体作用位置还由节点语义与映射规则决定，所以同一个键可以被多个节点复用。“命中”指已有成本键与符合绑定条件的目标节点对应上。该例更新的是节点的计算成本，最终节点时长还要取它与继承的非计算时长下限的较大值。</p>'
    s += '<p><b>因此，“58个键、3个命中、144个节点”表示：</b>源侧统计得到58类可供查找的计算成本；当前绑定规则实际使用其中3类，分别用于48个节点，共144个。它不表示只有3个算子，也不表示其余节点没有成本——它们继续使用继承值。这里的“物理计算键”是现有成本表的分类名称，不能据此认为已经建立覆盖所有算子形状的通用耗时模型。</p></div>'
    s += note('<b>58 个物理计算键和 960 条反向消息成本记录，只是 v685 本轮校准的两个子集。</b> 144 是更新的计算节点数，624 是目标每个方向的 PP 消息节点数；它们不是全模型参数总数。模型还使用训练配置、继承成本、运行时、两项补差和 MFU 换算常数。')
    s += '<p>阅读顺序：配置决定“做多少工作、怎样切分” → 调度与依赖决定“什么时候可以做” → 成本表决定“做多久” → 回放求整轮时间 → 换算 MFU。下面按参数族列全；同一数值被多个节点复用，不按节点数计算独立参数数目。</p>'
    s += table(['类别','具体输入 / 参数与单位','来源与 v685 状态','作用和边界'], [
        ('1 · 模型结构与算子尺寸','层数、hidden、attention heads、FFN、专家数/top-k、LoRA维度、dtype；整数/枚举', '冻结模型配置；224目标52层、源256为60层；hidden5120、128头、160专家、top-k6等核心维度相同', '确定每层工作和张量尺寸；已有配置不等于已建立覆盖所有算子形状的耗时函数。三组配置的逐字段证据见本章参数对照。'),
        ('2 · 训练批量与并行切分','GPU数、TP/PP/CP/DP/EP、层放置、序列长度、MBS、GBS、微批次数、重计算配置', '目标224卡：TP1/PP14/CP2/DP8/EP8；序列8192，MBS2/GBS48/MB3，层放置[2,4×12,2]；源256为PP16/MB4', '决定rank/通信组、执行次数和局部尺寸。MB是次数，MBS是每微批次样本数；PP/MB减少不代表每个算子时间同比减少。'),
        ('3 · 调度、依赖与成本映射','F/B程序顺序、通信双端条件、资源串行边、全局层号映射、反向接收stage映射；结构规则', '继承v684正式锁：327746节点、364784边；全局层映射与PP消息映射分别保留', '决定先后、等待与重叠；不是可按误差自由拟合的参数。预测开始/结束和关键路径是输出，不是校准输入。'),
        ('4 · 计算活跃区间','暴露计算 / 重叠计算成本，按方向、层、微批次或物理槽绑定；ns', '旧源256成本为主体；v685在85/90/95/100重统计58个物理键，实际3键命中144节点', '前向48、反向96；未命中节点保留旧值。节点可代表聚合区间或计时控制点，不等于逐kernel都独立拟合。'),
        ('5 · PP前向激活通信','每条消息服务成本4.704806 ms', '624个目标前向消息均继承 sealed_v54_pp，本轮未重新拟合；已逐节点核验', '该消息节点的software_sync和framework_residual字段为0，不表示其他运行时/等待为0；不是前向整个F阶段只需4.7 ms。'),
        ('6 · PP反向梯度通信','按源接收stage×lane×MB拟合边界总时长；ns', '源15×16×4=960条参数记录 → 目标13×16×3=624消息节点；四轮中位数', '边界值=接收stage的B开始−下游发送stage的B结束。保留4.704806 ms服务分量，其余记为完成分量；并非纯网络或纯软件的独立测量。'),
        ('7 · CP / EP组内通信','CP all-to-all；MoE dispatch/combine；网络service与本地完成成本；ns，通信语义关联payload/组成员', '继承历史网络模型及源256局部完成模板；v65按全局层号等索引迁移，v685未全量重拟合', 'dispatch → 专家计算 → combine分开表达。源routing/负载效果已进入旧成本；当前不按目标新token路由重新生成所有专家负载。网络后端参数不是本轮新调参项。'),
        ('8 · 梯度同步、优化器与收尾','DP/EDP等通信组的释放/完成、参数更新计算、收尾依赖；ns与结构', '继承父图及更早源拟合/网络输入', '属于完整iter，必须参与总时间；不是去掉优化器后再称完整MFU预测。组成员就绪和service不能重复计等待。'),
        ('9 · 图内运行时与非计算下限','局部gap、完成/同步分量、framework_residual、旧非计算下限；ns', '大部分继承历史源成本；本轮命中计算节点保留旧非计算下限', '命中暴露节点时长=max(新计算成本,旧非计算下限)。残差可能混合提交、同步、无可见活动等原因，不能全部称CPU工作或GPU空闲。'),
        ('10 · 入口及两项整轮补差','入口1265.388210 ms；图外补差604.439668 ms；outer1372.453390 ms', '入口由源四轮重新拟合；源图残差805.919557 ms乘MB比3/4；outer继承源60–100日志−Profiler中位数', '入口有图内节点；两项补差是不同总账。outer既未证实连续位于末尾，也不是均分到iter或profiler开关实验的额外开销。'),
        ('11 · MFU换算常数','每步有效FLOPs=8.436548311982576e16；224 GPU；每GPU峰值500 TFLOP/s', '沿用目标224合同；本轮未独立重算分子', 'MFU(%)=100×FLOPs/(GPU数×单GPU峰值×Training秒数)。MFU不是GPU忙碌率；时间误差与MFU误差分别报告。'),
        ('12 · 校准与评价规则','源选窗60–100连续后缀、至少4点；最小Profiler波动率；中位数；目标评价85/90/95/100', 'v685选出源85/90/95/100；旧继承成本可来自60–100', '这是方法配置，不是硬件性能参数；目标实际时长用于评分，不能回填成源校准。P10/P90与样本数是统计摘要，不是额外运行成本。')])
    s += '<p><b>机读全图覆盖：</b>新增成本清单覆盖 '+str(inv['nodes'])+' 个节点，按节点类型、计时组件和实际时长来源分成 '+str(inv['cost_groups'])+' 组，列出六个成本字段的范围、正值节点数和示例节点。它是成本来源清单，不声称已查清所有历史拟合自由度；<code>source_parameter_key</code> 是索引，字段数量、非空键数量和独立拟合参数数量不能互换。</p>'
    s += '<p><a href="#pp-model-versions">直接看：前后向通信公式与版本区别</a> · <a href="#compute-binding-explained">直接看：144个计算节点的位置</a> · <a href="http://192.168.0.49:8037/results/w37/A/integration-20260907/model_cost_inventory.csv">全图成本来源清单 CSV</a> · <a href="http://192.168.0.49:8037/results/w37/A/integration-20260907/model_cost_inventory.json">核验结果 JSON</a></p>'
    s += '<h3 id="compute-binding-explained">58个键、144个节点：前向与反向分别更新哪里</h3>'
    s += table(['方向 / 位置','rank / PP / MB','计算成本','更新数量'], [
        ('前向：MoE层之后、本阶段退出前','PP13，rank208–223，MB0/1/2','0.378082 ms','1个键 → 48节点'),
        ('反向：稠密层之后、本阶段退出前','PP0，rank0–15，MB0/1/2','13.681142 ms','1个键 → 48节点'),
        ('反向：进入MoE阶段后、首个CP之前','PP13，rank208–223，MB0/1/2','7.087467 ms','1个键 → 48节点')])
    s += '<p>源计算校准取每个stage的lane0代表rank；目标绑定扩展到对应stage的全部16个rank，不是只修改目标lane0。PP1–PP12本轮没有计算成本更新，但反向PP消息成本和依赖传播仍可改变这些stage的预测时刻。尚不能说“58个键已覆盖所有内部算子”；新增绑定研究已解释55个聚合键为何未直接命中，见<a href="#binding-audit">下方独立审计</a>；源整轮回放与运行时归属仍待验证。</p>'
    s += '<h3 id="pp-model-versions">通信模型版本：v685继承/重拟合成本，v687另建就绪候选</h3>'
    s += '<p><b>先解释符号：</b>tS/tR为发送/接收方CPU进入通信API的时刻；不是GPU数据就绪的直接测量。d为拟合的有效发送就绪延迟，c为汇合后的完成余项；t表示时刻，d/c表示时长，单位ms。max取较晚时刻。Δ=tR−tS表示收发入口时间差。</p>'
    s += '<pre>v687候选：tDone ≈ max(tS + d, tR) + c\n双方都进入API后，剩余时间 ≈ max(d, Δ) − max(0, Δ) + c\n前向均值：tDone ≈ max(tS + 0.0506, tR) + 5.0311 ms\n反向均值：tDone ≈ max(tS + 93.0369, tR) + 5.2470 ms</pre>'
    s += '<p>v687按方向×16条lane拟合，共32条记录、64个d/c标量；以上是lane均值示意，局部评分使用各lane值。只用源85/90拟合，95/100做增量检验。首个单消息API返回给出完成上界，不是精确GPU完成；d/c还不能独立解释为CPU/GPU/传输硬件的实测分量。</p>'
    s += '<p><b>前向为什么看起来近似常量：</b>dF仅约0.05 ms，主要是双方较晚入口之后约5 ms；cF比v685固定服务值多约0.3263 ms。但v685的服务分量与v687的API完成余项口径不完全相同，不能认定是测得的额外物理开销，更不能把0.33 ms乘消息数当整轮变化。反向dB约93 ms可被接收方较晚进入的时间部分或全部覆盖，不应在两端工作之后再一律串行加93 ms。</p>'
    s += table(['版本/场景','参数与状态','输入范围与限制'], [
        ('v685正式模型','本节12族；前向固定4.704806 ms，反向960源记录映射624消息；依赖锁不变','源校准 + 历史继承；目标四轮为开发评估。未采用上述v687 d/c公式。'),
        ('v687 / T33独立候选','完整F/B区间、本地API前后运行时、双端消息d/c','F/B总区间已含内部工作，不能再叠加旧细粒度成本；API散点使用观测入口作局部检验，整图自由运行使用预测入口。候选不替换正式锁。'),
        ('T35在线辅助','当轮前向/反向各一倍率；修正已有F/B总区间','需要目标当轮首批F/B观测，d/c与依赖不随之重新拟合；不是仅凭源侧的事前预测。'),
        ('T38在线辅助','T35基础上，使用前一已完成采样轮的预测残差','85初始化，只评90/95/100；这是目标在线状态，不是新增物理参数或源侧校准精度。')])
    s += '<p><b>防止重复计时：</b>节点耗时与分量字段不能无条件再次相加；全rank节点耗时不能直接求和当iter；包络已含的通信不可再加；依赖max产生的等待不能再塞成固定等待；图内入口、图外残差、outer各计一次。详细前后向散点和逐参数旧表保留在下方。</p></div>'
    return s

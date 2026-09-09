"""Compose the evidence blocks as eight technical chapters, retaining old anchors."""
import re
from common import dump

CHAPTERS = [
    ('training', '大语言模型如何进行分布式训练', '分布式训练'),
    ('why-dag', '如何预测训练时间与 MFU', '两种建模方式'),
    ('task', '当前预测任务与模型结构', '任务与结构'),
    ('calibration', '性能模型参数、校准与源到目标迁移', '参数与校准'),
    ('development', '模型开发路线与版本增量', '版本增量'),
    ('results', '预测结果、评价口径与误差分账', '结果与分账'),
    ('diagnosis', '误差诊断：证据能解释到哪里', '误差诊断'),
    ('limits', '外推验证、能力边界与下一步', '外推与下一步'),
]

def training(table, note):
    s = '<p>本报告研究的是：给定大语言模型、训练配置和硬件，如何在目标运行之前预测一轮训练的时间与MFU。先理解真实训练怎样执行，再讨论用什么性能模型描述它。</p>'
    s += '<h3>神经网络、Transformer与参数张量</h3><p>这里的大语言模型是采用Transformer架构的神经网络。网络结构规定注意力、前馈网络等运算如何连接；参数张量保存各层使用的权重。MoE将部分前馈计算组织为专家网络，由路由决定每个token使用哪些专家。</p>'
    s += '<p>模型可写为 <code>输出 = f(输入；参数θ)</code>。训练通常保持结构固定，通过数据计算损失及其梯度，再由优化器更新参数θ。张量是多维数值数组；参数、激活和梯度都是张量，但承担不同角色。</p>'
    s += '<div class="training-flow" role="list" aria-label="一轮训练的主要动作">'
    for title,body in [('输入数据','将一批样本划分为微批次'),('前向 F','用当前权重计算激活与输出'),('损失','衡量输出与训练目标的差异'),('反向 B','求参数梯度，必要时重计算激活'),('优化器 OPT','利用梯度与优化器状态更新权重')]:
        s += '<article role="listitem"><b>'+title+'</b><p>'+body+'</p></article>'
    s += '</div><p>多个微批次可以累积梯度后共同完成一次参数更新。上图表示逻辑关系，真实多GPU执行会交错安排这些动作，不能把每张卡的区间直接相加。</p>'
    s += table(['张量','训练中的含义','分布式执行中的例子'],[
        ('参数 / 权重','训练持续更新的模型数值','在GPU间复制或切分；是否聚合取决于并行和优化器方案'),
        ('激活','前向产生的中间结果','PP前向将需要的激活传给下一个stage'),
        ('梯度','损失对参数或中间量的导数','PP反向传递激活梯度；DP同步参数梯度或归约分片')])
    s += '<h3>把同一轮训练分配到多张GPU</h3>'
    s += table(['并行方式','GPU分担什么','由此产生的协作'],[
        ('DP · 数据并行','不同训练样本','梯度同步；采用分片优化器时还涉及参数或状态分片'),
        ('PP · 流水线并行','不同网络层','前向传激活，反向传梯度；调度多个微批次'),
        ('TP · 张量并行','同一层内部的张量和运算','通过集合通信交换或归并部分结果'),
        ('CP · 上下文并行','序列的不同部分','交换注意力计算所需的数据；具体集合通信由实现决定'),
        ('EP · 专家并行','不同MoE专家','dispatch分发token → 专家计算 → combine汇总结果')])
    s += '<p>这些方式可以组合。rank是训练进程编号，本配置每rank对应一张GPU；PP stage是一组承载相应层的rank。微批次数MB和每微批次样本数MBS不同：前者决定迭代拆成几份，后者决定每份有多少样本。</p>'
    s += '<h3>1F1B调度如何影响时间</h3><p>1F1B规定各stage在可执行条件满足时交错安排一个前向和一个反向动作，并按stage位置与微批次数处理填充和排空。一个stage可能等待上游激活、下游梯度、本地计算完成或通信资源。PP14/MB3的具体执行顺序见第03章；不能把整个流水线假定为长时间满载。</p>'
    s += note('<b>两种“参数”先分开：</b>训练修改的是神经网络的权重θ；本报告校准的是预测执行时间所需的计算、通信和运行时成本。校准性能模型不等于训练大语言模型。')
    return s

def methods(table, note):
    s = '<h3>性能模型的目标：完整训练时间，再换算MFU</h3><p>有效计算量与硬件峰值口径确定后，当前性能建模的核心是预测完整Training迭代时间。通信、等待、运行时与重计算会影响这个时间，但不能因此全部计入MFU的模型有效计算量。</p>'
    s += '<pre>MFU(%) = 100 × 每轮模型有效FLOPs\n             / (GPU数 × 单GPU理论峰值FLOPs/s × Training秒数)</pre>'
    s += table(['建模层次','三阶段模型','计算图模型'],[
        ('如何组织工作','按前向FWD、反向BWD（含重计算）、优化器OPT组织阶段级成本','按rank、层、微批次拆成计算/通信/运行时节点和控制点'),
        ('如何确定执行时间','使用阶段级公式、汇总成本及实现规定的等待/重叠修正','将数据依赖、本地程序顺序与资源约束写入图，回放任务完成时刻'),
        ('如何描述等待和重叠','通过阶段级规则和近似表达；不是所有实现都简单三段相加','依赖决定可执行条件，独立分支允许重叠；争用减速仍需额外成本模型'),
        ('配置改变后要验证什么','阶段公式、平均值与修正关系是否仍适用','目标任务数和依赖是否正确；源成本是否能迁移到目标'),
        ('适合解释什么','总体计算/通信预算和主要缩放关系','逐任务的执行顺序、等待、关键路径与局部成本变化影响')])
    s += '<p><b>训练执行DAG = 神经网络训练运算 + 分布式切分与调度 + 任务依赖。</b>为DAG绑定耗时成本并求解，才成为时间预测模型。节点、边更多只说明表达更细，不自动意味着更准确。</p>'
    return s

def task_intro(table):
    return '<p>主任务是用源256卡的允许输入与目标静态配置，预测224卡训练的完整迭代，并重点解释1F1B误差。当前正式预测使用v685；目标运行中观测辅助与事后替换诊断分别展示。</p>' + table(
        ['配置','源256卡','目标224卡','对执行图的影响'],[
            ('模型层数 / PP','60层 / PP16','52层 / PP14','层放置和流水长度改变'),
            ('MBS / GBS / 微批次数','2 / 64 / 4','2 / 48 / 3','每轮任务次数改变；MBS没有改变'),
            ('TP / CP / DP / EP','1 / 2 / 8 / 8','1 / 2 / 8 / 8','同类并行语义；通信组和全局rank配置随PP变化'),
            ('序列长度 / hidden','8192 / 5120','8192 / 5120','对应层核心静态尺寸基本一致；不保证动态专家负载一致'),
            ('层放置','[2, 4×14, 2]','[2, 4×12, 2]','目标不是保留源PP编号后一一复制全部成本')]) + '<p>图规模、参数数量和校准覆盖是三个不同概念：本章说明任务和依赖，第04章说明成本及其校准，第06章说明结果。源/目标详细静态证据见<a href="#three-case-parameters">三组参数对照</a>。16→256为第08章的独立外推案例，不混入本任务校准。</p>'

def route(table, note):
    s = '<p>版本表示模型实现的变化；T编号表示研究实验。下表按“解决什么问题、改变什么、是否采用”组织，日期和完整实验过程留在证据记录。</p>'
    s += table(['路线 / 版本','要解决的问题','主要增量','结论与下一步'],[
        ('历史256卡：最小PP图 → v1/v2/v4','能否表达完整训练并还原已测场景','从F/B传递扩展到优化器、运行时和多类通信','建立同配置校准回放；再用跨配置任务检验可迁移性'),
        ('224卡：三阶段v5.4 → 计算图v682','阶段汇总如何表达逐卡等待与依赖','用执行图组织细分任务，加入按阶段匹配的反向PP成本','历史60–100九轮结果见第06章；不要与256卡v4单轮回放混用'),
        ('v684：固定正式依赖结构','PP通信及本地程序顺序是否符合调度','建立当前正式阻塞PP程序顺序依赖锁','后续成本更新保持该锁，不能用目标误差任意改边'),
        ('v685：当前正式预测','源测量波动与成本校准是否影响预测','选择源稳定窗口；有限计算绑定、反向PP和入口更新，其他成本继承','仍是当前正式版本；详细更新位置见第04章'),
        ('v686 → v687 → v688：独立候选','更明确的阶段/运行时/双端就绪能否改善事前预测','阶段图 → 有效发送就绪d/c → 多源轮场景组合','同一目标窗口未超过v685，保持候选身份，不替换正式锁'),
        ('T33 → T34：源检验与静态迁移','源侧阶段模型可否迁移到目标PP/MB','固定源参数先做源增量检验，再改变目标静态配置','源侧误差小不保证目标误差小；静态迁移存在缺口'),
        ('计算绑定候选：独立成本实验','58个聚合键为什么只直接命中3键','按内部层号/微批次细粒度键更新源成本，保留旧倍率和依赖锁','1F1B开发MAPE 11.3408%→10.5616%；收尾退化，源整轮缺口未闭合，暂不升级'),
        ('T35 → T38：在线辅助分支','已知当轮前缀和前轮状态后能否更准','当轮F/B倍率；再加入前一已完成采样轮的原始残差','新增目标运行信息，单列使用条件；不当作纯源侧预测的连续升级')])
    s += note('<b>三条分支分别判断：</b>正式模型用来给出基线预测；独立候选检验新的建模假设；在线辅助依赖额外目标信息。低误差必须连同输入条件和评价窗口一起阅读。')
    s += '<p><a href="#baseline">历史版本指标与75.8%分母</a> · <a href="#results">当前正式模型与候选回归</a> · <a href="#weekend">T00–T42详细研究记录</a> · <a href="#binding-audit">计算绑定研究</a> · <a href="#binding-results">新候选结果</a></p>'
    return s

def diagnosis_intro(table):
    return '<p>第06章回答“差了多少、差在哪个时间区间”；本章回答“哪些建模问题已有证据，哪些仍待检验”。分账、敏感性实验与物理因果不是同一层证据。</p>' + table(
        ['观察 / 检查','目前支持的判断','尚不能推出的结论'],[
            ('计算成本绑定审计','55个未直接命中的聚合键对应另一种内部细粒度键；新候选更新70,192节点后局部和目标误差改善','源整轮回放及旧区间时长下限仍未独立验证，不能将改善全部认定为计算物理根因'),
            ('PP双端API与完成边界','源侧局部数据支持方向不同的就绪/完成关系，反向出现约93ms代理','不是93ms纯CPU开销的独立测量，也不证明目标完整iter已改善'),
            ('lane0观测替换','部分F/B时间类别变化会通过依赖明显影响该lane预测','不是全224卡物理误差贡献，也不是可直接部署的源侧预测'),
            ('固定成本下改变PP/MB','T34表明静态调度收缩不能充分解释目标实测变化','不能按目标差额任意增加等待或拟合倍率'),
            ('整轮图外补差与outer','当前时间账可以闭合，两项残差的来源不同','总量没有定位为具体事件；不能均分到节点或全部归因于优化器')])

def organize(section_markup, table, details, note):
    # The render stage supplies sibling evidence sections. Keep their content and
    # IDs, and compose only the reading order and chapter wrappers here.
    found = re.findall(r'<section id="([^"]+)"><h2>[^<]*</h2>(.*?)</section>', section_markup, re.S)
    blocks = dict(found)
    expected = {'why-dag','full-iter','summary','terms','baseline','graph','calibration',
                'weekend','results','diagnosis','scaleout','limits','tool-todo','sources'}
    if set(blocks) != expected or len(found) != len(expected):
        raise ValueError('Unexpected evidence section layout: '+str(list(blocks)))
    def sub(id,title,content=None):
        return '<div class="subsection" id="'+id+'"><h3>'+title+'</h3>'+ (blocks[id] if content is None else content) +'</div>'
    histories = blocks['why-dag'].split('<div id="historical-development">')
    if len(histories) != 2 or not histories[1].endswith('</div>'):
        raise ValueError('Historical development boundary missing')
    method_body = histories[0]
    history_body = histories[1][:-6]
    limits_body = blocks['limits']
    contents = {
        'training': training(table,note),
        'why-dag': methods(table,note)+'<p><a href="#terms">术语索引</a></p>'+method_body+sub('terms','术语索引：按需展开查阅'),
        'task': task_intro(table)+'<p><a href="#full-iter">完整迭代时间图</a> · <a href="#graph">节点、依赖与图规模</a></p>'+sub('full-iter','完整迭代：224卡预测、224卡观测、256卡观测')+sub('graph','模型结构与图规模：节点表示什么'),
        'calibration': blocks['calibration'],
        'development': route(table,note)+details('历史256卡模型如何从局部图扩展到完整迭代',sub('historical-development','历史模型开发证据',history_body))+sub('weekend','研究实验索引与增量记录'),
        'results': '<p>先固定评价口径，再看当前正式模型，最后比较候选与不同信息条件下的辅助方法。各段MAPE不能相加，历史单轮误差不能当当前多轮MAPE。<a href="#summary">查看三种使用场景摘要</a>。</p>'
                   + '<div class="note" id="evaluation-contract"><b>当前比较口径：</b>正式v685、目标224卡、85/90/95/100四轮；拓扑继承v684并冻结。完整Training包含准备、F/B、收尾及outer；1F1B仅取最早F至最晚B。旧成本保留历史来源，目标数据为开发评估，不宣称独立盲测。</div>'
                   + blocks['results'] + details('三种使用场景的摘要：输入条件不同',sub('summary','事前预测与在线辅助的范围'))
                   + details('历史版本的同口径指标及75.8%分母',sub('baseline','历史模型结果与误差分账')),
        'diagnosis': diagnosis_intro(table)+blocks['diagnosis'],
        'limits': limits_body + '<div class="related-index"><p><b>相关内容索引：</b><a href="#scaleout">16→256完整外推案例与逐轮结果</a> · <a href="#tool-todo">仿真工具调研与对照计划</a> · <a href="#sources">全部证据与复现入口</a></p></div>'
                  + details('独立案例：16→256外推验证',sub('scaleout','16→256：既有外推结果与边界'))
                  + details('对照工具：资料与后续实验计划',sub('tool-todo','训练仿真工具与复现计划'))
                  + details('证据索引、机读产物与复现命令',sub('sources','证据与复现')),
    }
    nav = '<nav class="toc" aria-label="页内索引"><div class="toc-inner">'+''.join('<a href="#'+id+'">'+str(i).zfill(2)+' '+label+'</a>' for i,(id,title,label) in enumerate(CHAPTERS,1))+'</div></nav>'
    body = nav+'<main>'+''.join('<section id="'+id+'"><h2>'+str(i).zfill(2)+' · '+title+'</h2>'+contents[id]+'</section>' for i,(id,title,label) in enumerate(CHAPTERS,1))+'</main>'
    locations = {'full-iter':'task','graph':'task','terms':'why-dag','summary':'results',
                 'baseline':'results','weekend':'development','historical-development':'development',
                 'scaleout':'limits','tool-todo':'limits','sources':'limits'}
    dump('organization_map.json',{'chapters':[{'id':id,'title':title,'number':i} for i,(id,title,_) in enumerate(CHAPTERS,1)],
                                 'legacy_anchor_locations':locations,'all_previous_section_ids_preserved':True,
                                 'new_model_or_fit':False})
    return body

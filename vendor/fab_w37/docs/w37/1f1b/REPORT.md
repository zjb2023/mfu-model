# W37 1F1B：冻结基线与持续 autoresearch

更新：2026-09-06。正式模型仍保留 v6.8.5。第二轮得到了有源侧预测证据的 PP 就绪方法，但未得到可替换正式模型的 224 卡整体预测。目标误差的主要缺口集中在 F/B 内部通信独占与 GPU 空档，不能靠改 PP 边或放大计算成本弥补。

[图册与节点查看器](post685/delivery/index.html) · [逐迭代结果](post685/delivery/version_iteration_results.csv) · [阶段指标](post685/delivery/version_phase_metrics.csv) · [复现入口](post685/reproduce.py) · [持久目标](GOAL.md) · [交接](HANDOFF.md)

## 限时第三轮（进行中）

文件 Goal 持续推进，截止 2026-09-07 07:00 香港时间；原生 Goal 的正文和异常状态已在 [GOAL.md](GOAL.md) 留档。当前主目标是降低 **224 卡 1F1B MAPE 11.340786%**，全过程走 [专用 Snakemake 管线](deadline_20260907/PIPELINE.md)，路线追加到 [ROUTE.md](ROUTE.md)。T00 的 7 个步骤及旧 v687 节点/边/参数精确匹配均 PASS；T01 的 13 步和 12 份阶段 manifest 验收 PASS。

T01 只改变整段 F/B 成本的来源匹配，v687 图边不变。按 stage 角色、F/B、microbatch 角色、warmup/steady/cooldown、前一 API 和 lane 匹配最近源阶段，目标 1F1B MAPE 为 **15.241236%**（相对 v687 15.340746% 减少 0.099510 个百分点）；汇总同上下文的跨阶段成本则为 **16.546770%**。两者均未达到正式基线，不采纳。pool 的源侧增量 1F1B MAPE 升到 **2.587558%**，说明抹平阶段差异会损害关键路径预测，不能仅凭局部 F 耗时 MAE 下降判断整体改善。

T01 源侧留 stage 检验只用 85/90 拟合、95/100 评分，共 3,520 个同覆盖 F/B 观测，另 64 个观测无保留源上下文而显式排除；端点角色未伪装成可跨 stage 验证。目标逐迭代、分阶段、全部九轮和 MFU 结果见 `results/w37/A/autoresearch-t01-schedule-context/evaluations/`；nearest 的 Profiler/training/MFU 相对 MAPE 分别为 **13.894171% / 14.017247% / 16.304448%**，pool 为 **15.073916% / 15.121537% / 17.817597%**。entry/tail/outer 按隔离试验约定固定。正式 v685 拓扑和历史产物保留。

T02/T03 又核对了 21,006 条 wrapper、1,908 个 phase 的区间：CPU pre/post 与 wrapper wall 严格守恒，wrapper union 与既有 phase 分区一致；runtime sync 调用是这些区间的子集，不重复加时。现成派生表的切点是 **musaGraphLaunch 的 CPU API 入口**，没有 GPU 有效启动、payload-ready 或完成关联，因此旧提取说明中“service 启点”的解释需要收窄为“可观察 CPU 分区边界”。这次没有增加等待边或拟合目标成本。见 [边界图](deadline_20260907/runs/T03/diagnose/wrapper_boundary.svg)、[机读核验](deadline_20260907/runs/T03/diagnose/diagnostic.json)、[提取与训练代码证据](deadline_20260907/runs/T03/diagnose/)。

T04–T06 进一步形成了可解释的 CPU EP 内部候选图：源侧25,488次八卡调用、203,904条rank记录支持当前观察中的入口共同完成前提；将F/B整段替换为本地区间、group最早CPU完成和各rank返回尾部后，**92,690节点/119,552边**在九轮源数据中所有PP API、F/B与EP wrapper起止均为 **0 ns重建误差**，且边哈希不变。[节点/边与逐轮检查](deadline_20260907/runs/T06/diagnose/)已交付。该结果仅验证观察成本的可表示性，本身不证明新成本预测或224卡改善；随后T07已完成预测检验，结果如下。

T07 已完成 source85/90 新拟合、95/100 增量验证、封存后目标评分及去rank尾部消融。**v6.8.10 CPU EP 内部模型没有改善**：源侧1F1B MAPE **0.791940%**，目标 **15.906607%**，均差于v687；目标Profiler/training/MFU相对MAPE分别 **14.495433% / 14.580053% / 17.070742%**，MFU平均偏差+0.504398个百分点。去rank返回尾部后目标1F1B **16.424628%**，不采纳。entry/tail/outer预测固定且核验无变化。完整九轮、逐阶段与step/MFU在 [T07图册](deadline_20260907/runs/T07R/diagnose/index.html) 和 [机读表](deadline_20260907/runs/T07R/diagnose/)；14项测试、15份阶段manifest及正式60文件核验通过。

新图将父F/B成本置0，执行本地区间、EP共同CPU完成和rank尾部；等待由组内入口依赖产生，不再加入一个观测等待成本。目标关键路径上，共同CPU完成区间13,497.667ms、本地区间2,939.238ms、rank尾部128.855ms，其余为PP及API，共计1F1B18,131.338ms。**共同CPU区间混合了设备排队、计算、collective与runtime，不等于纯EP网络耗时**。静态主机只有15/28台目标主机出现在源中，stage位置没有相同主机匹配；尚无独立硬件状态泛化证据。下一步T08先验证完整源迭代场景，避免分项均值抹去关键路径选择和共同波动。

T08把每轮源EP成本单独回放后平均，得到 **v6.8.11 两轮场景**：源增量1F1B **0.332925%**、目标 **15.028381%**，比T07降低0.878226个百分点，仍未优于正式11.340786%。目标Profiler/training/MFU相对MAPE分别 **13.701824% / 13.837203% / 16.061420%**，MFU偏差+0.474572个百分点；固定PP的EP场景消融为 **15.037738% / 13.710279% / 13.845118% / 16.072082%**（1F1B/Profiler/training/MFU）。[T08图册与全部九轮表](deadline_20260907/runs/T08R/diagnose/index.html)已交付，entry/tail/outer固定且未变。

固定PP后，两轮场景平均节点成本与T07最多仅差0.5ns，边完全一致，但先逐轮回放使source/target包络增加145.993/187.336ms；这直接支持“分项均值会丢失影响汇合的共同波动”。完整场景的目标包络18,320.692ms，与旧v688整段phase场景仅差约0.853ms；内部图的细化本身尚未解决运行时条件迁移。两个场景18,095.776–18,545.608ms也不足以覆盖目标四轮均值21,561.771ms，不能当作预测区间。17项测试、15份阶段manifest、60份正式文件和T07控制图/参数精确一致均通过；继续T09源侧完成区间可辨识性审查，正式v685保留。

T09的源侧审查进一步发现：内部stage的B重算dispatch首层约62ms，后续层约155ms，fit与增量窗口的差额分别 **93.432ms / 93.090ms**，接近独立PP就绪模型的93.0369ms。训练源码中的逐层checkpoint和dispatch等待GPU信号支持调查“前一B层未完成工作在下一次dispatch或PP发送处暴露”的假设，尚不能证明唯一GPU原因。下一步T10先按逻辑层配对验证，再考虑用可解释的基项和未完成工作代理替换同一成本区间。

同stage两组差异和固定慢rank不够稳定，host与组/CP/数据影响又未分离，暂不按host标签拟合目标补偿。源runtime同步union缺少线程及CPU/GPU关联筛选，只是与wrapper时间窗相交的可见区间，不能当作额外成本。T09专用source_only流程4步PASS，未读取目标时长或运行新的目标评分；[图、同键误差与可辨识性证据](deadline_20260907/runs/T09/diagnose/)及失败依赖修复记录均保留。

T10完成了 **v6.8.12 跨层代理成本**：按同逻辑F层的dispatch基项，加上源PP就绪参数形成的前一B层代理，替换同一group CPU区间。后续B层局部验证MAE由2.568703降到2.488437ms，首层由1.736381升到2.170977ms；整图目标1F1B **15.028381%→15.008327%**，仅4.324ms的变化，源增量 **0.332925%→0.333982%**，因此不采纳。目标Profiler/training/MFU相对MAPE分别 **13.683702% / 13.820241% / 16.038576%**，MFU偏差+0.473897个百分点，其他阶段不变。[逐轮、分阶段与分账图](deadline_20260907/runs/T10R/diagnose/index.html)和 [21阶段/19测试/60正式文件验收](deadline_20260907/runs/T10/acceptance_v2.json)已交付。

去代理消融使目标/源增量1F1B误差升到32.831249%/21.064308%，说明当前模型依赖这项成本。目标模型内代理关键路径账为3,905.350ms，去代理的全图缩短量为3,842.792ms，二者因关键路径变化而不同；都不是独立实测GPU计算耗时。新的CPU成本关系解释了部分跨层排空行为，但未解决目标运行时差额。接下来审查现成source CP事件的时间域、缺失范围及其在EP/PP区间内的覆盖，避免把同一通信活动重复计入。

T11完成源CP时间域和覆盖审查：[字段合同](deadline_20260907/runs/T11/diagnose/field_contract.json)、[B尾部覆盖图](deadline_20260907/runs/T11/diagnose/source_cp_tail_coverage.svg)、[逐轮缺失表](deadline_20260907/runs/T11/diagnose/source_cp_coverage.csv)、[验收](deadline_20260907/runs/T11/acceptance.json)。449,273条派生记录与固定上游GPU kernel表逐项一致；7条事件缺失使7个CP pair-step整体无效，另377条观测因此排除，保留224,448组。源/目标静态CP2均在单host/同EP8组，不能根据卡数下降假设CP改成跨机。4F/9B每层的序号映射可复算，但还不是训练操作ID证明。

除stage0外，每个B标记结束后都有5个CP kernel，源拟合和增量窗口的union约43ms，最后结束距B标记约81–83ms；这比CPU标记更晚，却没有跨过下一个F/B开始。旧提取代码实际采用“下一个F/B开始”归属窗，不能按注释误读成“下一个PP操作开始”。service_fct只是GPU组尾部观测的别名，不是独立网络profile；13个rank-step的kernel有重叠，已用union避免求和重复。下一步对齐EP/PP包含关系，不能直接将43ms或81ms加到既有CPU图成本。此轮无新预测，目标与MFU指标保持T10结果。

T12进一步确认CP与EP/PP的时间包含关系：[图](deadline_20260907/runs/T12/diagnose/source_CP_EP_PP_alignment.svg)、[逐wrapper表](deadline_20260907/runs/T12/diagnose/source_EP_CPU_CP_disjoint_coverage.csv.gz)、[源PP对齐](deadline_20260907/runs/T12/diagnose/source_PP_B_CP_tail_alignment.csv.gz)、[验收](deadline_20260907/runs/T12/acceptance.json)。源95/100中，后续B重算dispatch相对同逻辑F层的**rank CPU wrapper增量93.432255ms=CP覆盖增量38.249752ms+未见CP区间增量55.182503ms**；后者包括计算、其他通信及空档，不能直接称GPU计算或CPU等待。该rank口径与T10 group最早返回减最后入口不同。

8,632条有效B消息中没有CP kernel越过首API返回；6,668条双端单消息且receiver已在末CP前发布的样本支持局部比较。源增量窗口末CP在sender入口后80.694154ms结束，距93.037983ms的拟合有效就绪仍12.343829ms，末CP到首API返回17.533629ms。下一步检验CP锚定、保留就绪余项的源侧预测；末CP比当前拟合就绪点约早12ms，仍不能命名为payload-ready。20测试、3阶段manifest/112产物/60正式文件通过，本轮没有新增目标预测。

T13完成 **v6.8.13 CP锚定的PP就绪预测**：以源侧stage角色/lane/MB角色预测CP尾部，再拟合其后非负就绪余项，替换原PP B就绪成本；EP跨层代理和其他节点成本固定。源95/100单消息局部MAE由0.585010变为0.593511ms；整图源增量MAPE **0.333982%→0.334014%**，目标 **15.008327%→14.994403%**，仅增加3.002325ms预测包络，仍差于正式11.340786%，不采纳。目标Profiler/training/MFU相对MAPE为 **13.671119% / 13.808462% / 16.022719%**，MFU偏差+0.473428个百分点；entry/tail/outer不变。[逐迭代、阶段及图](deadline_20260907/runs/T13R/diagnose/index.html)和[验收](deadline_20260907/runs/T13/acceptance.json)已交付。

去CP后潜在余项的消融，目标/源增量1F1B为15.843848%/0.784791%。目标关键路径账中API至CP末端预测区间1,216.456759ms、其后潜在就绪183.148529ms、其余16,928.413050ms，共18,328.018338ms；前者包含等待/计算空隙，不是CP kernel duration。固定图与费用原位替换排除了重复加时，但主要运行时缺口仍在内部CPU窗口。22测试、22完成阶段manifest、1051产物SHA和60正式文件通过；T13A标签修正与T13R CSV读取修复保留旧记录，不改模型分数。随后T14按资源方案读取源侧GPU/runtime小样本，核查未见CP区间的实际活动，方案见[T14](deadline_20260907/T14_RESOURCE_REVIEW.json)。

T14已完成：[四轮设备覆盖图](deadline_20260907/runs/T14B/diagnose/source_dispatch_GPU_coverage.svg)、[逐CPU窗口分区](deadline_20260907/runs/T14B/diagnose/source_CPU_window_GPU_runtime_coverage.csv.gz)、[字段合同](deadline_20260907/runs/T14B/diagnose/field_contract.json)、[资源实测](deadline_20260907/runs/T14B/diagnose/source_raw_resource_measurement.json)、[验收](deadline_20260907/runs/T14B/acceptance.json)。只读解析source rank16的85/90/95/100四文件，唯一输入156,349,592字节；40,684个kernel、1,080个设备拷贝/置零事件全部找到唯一runtime/driver关联，无GPU先于关联CPU启动。每轮8个F/B、16个PP、96个EP、208个CP边界与冻结表均0ns；384个同范围EP窗口的CP覆盖与T12逐项精确一致。

单文件首版遗漏driver关联及copy/set视图，r2补全并精简索引后峰值428,441,600字节、分析9.540s，原数据不变；r2作为显式SHA门槛通过后扩展四轮，峰值489,852,928字节、分析24.886s。9个完成阶段manifest/395产物SHA、23测试、60正式文件均通过；首样本重复抽取的五份表精确一致。资源方案保持冻结的事前状态，实际执行以验收和资源表为准。没有新拟合、预测或目标时长读取（只做SHA核验）。一个四层内部rank的覆盖不能代表全rank；其他设备活动包含计算、拷贝及未分类kernel，runtime同步是重叠视图，不能再加到CPU wall。

T15已完成：[GPU提交时点分区图](deadline_20260907/runs/T15/diagnose/source_pending_submission.svg)、[逐GPU/CPU窗口关联](deadline_20260907/runs/T15/diagnose/source_GPU_CPU_window_intersections.csv.gz)、[逐逻辑层配对](deadline_20260907/runs/T15/diagnose/source_logical_F_B_submission_pairs.csv)、[互斥family差额](deadline_20260907/runs/T15/diagnose/source_logical_F_B_family_summary.csv)、[验收](deadline_20260907/runs/T15/acceptance.json)。复用T14B的8.38MB压缩表及元数据，先验全阶段SHA与source-only读取审计通过后，只允许17个准确文件路径解析；没有重读raw或目标时长。384个CPU窗口、64个逻辑F/B配对、44,144互斥区间及22,082个GPU/窗口相交记录均守恒，41,764个GPU设备事件与runtime时间逐项匹配；26测试/3阶段manifest/129产物/60正式文件通过，峰值382.01MB、分析104.997s。

同rank16的source95/100后续B−F wrapper增量为 **94.260546ms**：CP覆盖增量 **38.496419ms**，无CP时已在wrapper进入前完成CPU提交的其他GPU活动增量 **54.074526ms**，EP独占增量0.889302ms、无非PP设备事件增量0.800495ms、进入后提交项−0.000195ms。85/90的进入前提交项为53.885010ms，较一致。无CP的互斥family差额中attention_backward为21.524836ms、matrix_kernel为9.324847ms、routing为7.269069ms，名称分类仍属启发式；混合family保持单独区间，不累加边际kernel wall。这个94.261ms来自单rank的24个增量配对，与T12全rank的93.432ms/5352配对不是同一分母。

这些GPU活动的关联runtime调用在wrapper进入前已结束，支持“CPU wrapper涵盖异步已提交工作继续执行”的时间证据；其中工作究竟属于前一逻辑B层还是当前重算预处理，还需CPU checkpoint父区间映射。T15没有拟合新成本或生成224预测，最新v6813仍14.994403%，正式11.340786%更好。

T16已完成：[CPU checkpoint与GPU时间线](deadline_20260907/runs/T16/diagnose/source_checkpoint_GPU_timeline.svg)、[逐GPU父层归属](deadline_20260907/runs/T16/diagnose/source_GPU_checkpoint_ownership.csv.gz)、[局部尾部预测](deadline_20260907/runs/T16/diagnose/source_GPU_tail_local_prediction.svg)、[PP GPU分账](deadline_20260907/runs/T16/diagnose/source_PP_GPU_start_end_alignment.csv)、[验收](deadline_20260907/runs/T16/acceptance.json)。64个同线程CheckpointFunctionBackward父区间各容纳4个已核验EP wrapper、4个重计算CP和5个autograd反向CP；22,204个GPU提交映射到父区间，另19,560个属于区间外（含F/PP/其他工作）并完整保留。source95/100后续dispatch中，前一逻辑层反向工作的非CP独占覆盖为46.215273ms，85/90为46.149705ms；T15的54.075ms是B−F进入前提交活动的差额，不能全部等同这个父层归属量。

源95/100末层GPU主stream完成在PP API入口后92.956858ms，末CP在80.621916ms，二者相差12.334942ms；这补充了此前CP后约12ms余项的设备时间证据。16次B发送均匹配一个同线程CPU API内提交的SendRecv kernel，主stream结束→PP kernel开始→kernel结束→首API返回逐项守恒；增量均值分别0.168136、4.866970、0.082702ms。`all_nonPP`仅指这个checkpoint中提交的非PP事件，未强行纳入父区间外的异步DP/EDP或其他未来工作；这些设备区间不是独立网络service。

source85/90拟合的每层主stream尾部，95/100局部MAE0.679468ms（32个parent）；以GPU尾部锚定的PP局部MAE **0.663167ms**，比原PP方法 **0.557697ms** 差（同8条rank16消息），CP锚点0.672333ms、父区间全非PP锚点0.680220ms，也未改善。局部预测使用观测API入口，参数/预测先SHA封存，再连接完成观测评分；不代表整图自由运行预测。未产生新的224预测或成本迁移，正式模型保留。

T16B首次PP名称过滤过窄导致失败，r2匹配实际`mcclKernel_SendRecv`名称及同线程API区间后通过；与T16A的15份科学表、参数和预测精确一致。28测试、7个完成阶段manifest/329产物/60正式文件核验通过；最终分析26.136s、峰值391.35MB，无raw解析。训练checkpoint/block/fused代码已固定SHA并交付文本，但部署revision仍未独立证实。

T17已完成：[source/target dispatch可见活动图](deadline_20260907/runs/T17B/diagnose/target_source_dispatch_visibility.svg)、[逐CPU窗口GPU/runtime分区](deadline_20260907/runs/T17B/diagnose/evaluator_only/CPU_window_GPU_runtime_coverage.csv.gz)、[逐轮F/B活动](deadline_20260907/runs/T17B/diagnose/evaluator_only/FB_activity_profile.csv)、[MB角色表](deadline_20260907/runs/T17B/diagnose/evaluator_only/dispatch_by_MB_role.csv)、[验收](deadline_20260907/runs/T17B/acceptance.json)。v6813 seal及全部预测逐项核验后，先解析target85/rank16单份30.74MB，峰值394.19MB/9.703s、关联和边界门槛通过，才扩展登记的90/95/100。目标四份唯一raw共123,843,399字节，33,816个设备事件（32,960 kernel和856 copy/set）均唯一runtime关联；24个F/B边界精确一致，288个EP wrapper对旧偏移的最大差341ns，符合旧浮点epoch表示的512ns检查范围。两轮都只在diagnose/evaluator解析目标raw，源/model阶段保持拒绝；原始文件未复制。

同rank16、四轮等权的目标−源平均B阶段差额为 **251.224406ms**：未见非PP GPU事件 **+172.127089ms**、EP独占 **+48.668490ms**、CP独占 **+40.153676ms**、其他GPU独占+7.986572ms、CP/其他重叠−17.710551ms、EP/其他重叠−0.000869ms。F阶段差额114.077564ms，其中未见非PP设备事件+75.106498ms。这里比较源4MB与目标3MB的每phase平均，不是全rank、整迭代误差归因，也不是CPU等待或纯网络耗时。各case自身时钟内保留整数ns，跨运行只比较时长；另列first/middle/last角色，避免把source中间MB2当成target末MB2。

目标checkpoint主stream尾部平均 **88.344548ms**，源为 **92.900068ms**；本样本不支持继续增大PP尾部来补偿目标整体低估。CPU同步与设备分区仍为重叠视图，未重复加时。840个CPU窗口分区守恒；单样本5表和覆盖与四轮复取精确一致，source尾部与T16精确一致。29测试、6阶段manifest/293产物/60正式文件及v6813 seal全部通过；四轮最终峰值556.29MB、分析28.317s。所有目标产物永久标注development/posthoc NEVER_MODEL_FIT；本轮没有新成本、模型或224预测，最新候选/正式1F1B MAPE仍14.994403%/11.340786%。

T18已完成：[源侧空档表](deadline_20260907/runs/T18A/diagnose/FB_gap_accounting.csv)、[源/目标提交与CPU位置图](deadline_20260907/runs/T18R/diagnose/evaluator_only/device_gap_context_review.svg)、[逐轮分账](deadline_20260907/runs/T18R/diagnose/evaluator_only/FB_gap_per_iteration.csv)、[逐空档及CPU/runtime分段](deadline_20260907/runs/T18R/diagnose/evaluator_only/gap_context_segments.csv.gz)、[验收](deadline_20260907/runs/T18R/acceptance.json)。先对T14B纯源表实施整数ns的设备union补集和提交边界分类，32个source F/B通过；再核验v6813 seal和分类器SHA后，用相同算法处理T17B缓存目标表，24个target F/B通过。源侧三张科学表精确一致，共49,553空档、94,096分段全部守恒；不重读raw。

rank16四轮等权、每F/B平均的目标−源空档差额拆分如下；“提交”指下一条可见非PP设备事件的关联runtime API，不证明它是唯一阻塞者。

| 阶段 | 空档差额 / ms | API开始前 / ms | API执行中 / ms | API结束后 / ms | 同时开始或阶段边界 / ms |
|---|---:|---:|---:|---:|---:|
| B | 172.127089 | 51.439087 | 17.437093 | 103.250495 | 0.000413 |
| F | 75.106498 | 17.975088 | 6.996102 | 50.103433 | 0.031876 |

B差额按另一组互斥CPU位置分区为：EP wrapper内127.786235ms、wrapper外但checkpoint内43.839727ms、checkpoint外0.501126ms。B的API结束后差额中，53.563938ms与同一提交线程的musaDeviceSynchronize相交，13.462256ms与musaStreamSynchronize相交，11.645442ms与musaStreamWaitEvent相交；这些是同一空档的上下文，不能加回阶段wall或等同纯网络/CPU开销。F/B首尾空档、同时开始的设备事件及跨阶段下一事件单列；源4MB和目标3MB按各轮phase均值比较，MB角色另表。

源侧耗时11.928s、峰值342.27MB；最终源/目标复核20.263s、峰值408.54MB，新增缓存清单17.47MB。33测试、9阶段manifest/430产物/60正式文件和v6813预测seal通过。首次测试暴露同名worker模块的导入顺序冲突，改为测试执行时导入后通过；原图标签重叠及纵轴范围问题保留，r2新增可读复核图，10张科学CSV解压后逐字相同。没有新成本或预测，正式/最新研究候选目标1F1B仍11.340786%/14.994403%；Profiler、training、MFU和其他阶段未变。下一项方案为[T19局部设备队列预测](deadline_20260907/T19_PLAN.json)，不能用目标差额设置等待成本。

T19已完成：[CPU提交/stream顺序审查](deadline_20260907/runs/T19A/diagnose/source_stream_order_audit.csv)、[局部预测图](deadline_20260907/runs/T19B/diagnose/source_device_queue_local_validation.svg)、[参数和预测seal](deadline_20260907/runs/T19B/diagnose/source_device_queue_prediction_seal.json)、[逐事件预测](deadline_20260907/runs/T19B/diagnose/source_device_queue_event_validation.csv.gz)、[CPU所提交工作的设备末端](deadline_20260907/runs/T19B/diagnose/source_device_queue_CPU_owned_device_endpoints.csv)、[验收](deadline_20260907/runs/T19B/acceptance.json)。两个source_only Snakemake流程各4步；41,764事件、44个stream/iteration、41,720条独立候选边均按CPU提交顺序核验，无同stream设备乱序、重叠、CPU提交歧义或多设备共用runtime关联。512个早于API返回的设备事件全是拷贝；按类别固定kernel API结束、copy/set API开始为局部释放代理，不能统一使用API结束。

source85/90拟合2,772行分层成本参数，层级为shape/context、operation/context、kernel/stream、family/stream；95/100共21,218个事件中16,519个使用第一层，4,698个回退operation/context，1个回退kernel/stream，没有缺失成本静默补0。四个版本共167,056条预测先封存，再连接GPU完成观测。下表为source95/100增量结果，事件列按事件数加权，末端列每F/B各8个窗口；末端指该CPU F/B所提交的非PP设备事件最后结束，不是CPU F/B边界或整图1F1B。

| 局部模型 | 设备事件结束MAE / ms | F设备末端MAE / ms | B设备末端MAE / ms |
|---|---:|---:|---:|
| 独立事件延迟 | 18.503569 | 12.553023 | 3.751851 |
| 同stream中位数成本 | 16.789743 | 0.031298 | 0.835315 |
| 同stream均值成本 | 162.998766 | 680.881481 | 0.604612 |
| 同stream零提交余项 | 27.389938 | 0.039477 | 41.345380 |

中位数队列对局部设备末端有改善，但它**使用观测CPU提交时点和事件结构作为条件**；所有GPU开始/结束/duration观测均禁止作为预测函数输入，95/100 GPU时长变更不改变参数或预测的测试通过。这不能视为224卡自由运行预测。零余项使B尾部大幅低估，说明余项含未恢复release约束；它不是独立CPU或GPU服务。均值版本的source95/MB0/F末端多估832.529ms，CPU条件关键链551事件中余项累计1,212.105ms；其中23个outside_FB同shape的aten::gt事件，均值余项比中位数累计多1,028.669ms，表明把偶发长release等待分摊到重复算子会产生虚假排队，均值版本拒绝。

384次源musaGraphLaunch全部位于EP wrapper，但与现有device表的直接关联为0。[graph→同线程/同External-id首个DeviceSynchronize返回的描述性括号](deadline_20260907/runs/T19B/diagnose/source_graph_launch_visibility.csv)中仍有无设备事件区间；这个括号不是纯graph耗时。T14未保留gpu_user_annotation和ac2g类别，尚不能把图执行时的可见性缺口当作设备空闲；下一步[T20单文件字段补查](deadline_20260907/T20_RESOURCE_REVIEW.json)先补证据，不回填目标成本。[MUSA流语义](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/history_version/rc4.3/programming_guide/Chapter03/)支持同stream顺序并允许跨stream依赖；本轮依据官方索引摘录，页面直接获取超时，且未验证部署revision/特殊执行标志，正式锁保持。

36测试、6阶段manifest/302产物SHA、60正式文件及既有v6813 seal通过；7张源审查CSV精确一致。最终峰值551.03MB、分析32.097s，不读取raw或目标时长。没有新的224整图结果，1F1B/Profiler/training/MFU及其他阶段指标均保持T13，正式模型仍优于研究候选。

T20已完成：[补取字段合同](deadline_20260907/runs/T20A/diagnose/field_contract.json)、[flow端点/CPU/设备/annotation图](deadline_20260907/runs/T20B/diagnose/source_graph_flow_coverage.svg)、[底层观测节点](deadline_20260907/runs/T20B/diagnose/source_flow_observation_nodes.csv.gz)、[CPU→设备关联边](deadline_20260907/runs/T20B/diagnose/source_CPU_GPU_correlation_edges.csv.gz)、[GraphLaunch缺失端点表](deadline_20260907/runs/T20B/diagnose/source_GraphLaunch_without_device_endpoint.csv)、[验收](deadline_20260907/runs/T20B/acceptance.json)。按冻结资源方案只重访source85/rank16的一份38,816,598字节trace，10,253设备、21,034 runtime/driver和49,955 CPU事件核心字段均与T14精确一致；补取280个GPU annotation、31,287个ac2g点和21,034组flow identity，峰值476.36MB、分析50.132s。没有扩展到其他source文件或target raw。

第二个source_only Snakemake阶段仅使用2.67MB缓存，全部flow按实际pid/tid/整数ns核验：10,253对s→f精确对应既有CPU API入口→设备事件开始；1,839个s单点、8,942个f单点均落在CPU API入口，f字母本身不能视为GPU完成。96次GraphLaunch全部只有CPU s单点，新增graph设备端点和跨stream依赖均为0；280个annotation均可按External id和名称找到CPU scope，但它们是范围标注，不计入设备busy。交付31,567个观测节点和10,253条提交关联边，后者不是完整因果依赖图，也不替换正式拓扑。

37测试、6阶段manifest/306产物SHA、60正式文件和v6813 seal核验通过；缓存复核峰值281.13MB、分析57.635s。遗漏类别没有修复这份trace中的graph设备执行覆盖，不能据此把无设备事件区间命名为空闲、CPU等待或纯网络成本；尚无依据为此增加成本。没有新拟合或224预测，正式/最新研究候选1F1B MAPE仍11.340786%/14.994403%，Profiler、training、MFU与其他阶段均不变。下一步改查独立源侧计数器的可用性，避免仅为重复覆盖缺口而继续扫trace。

T21已完成：[源计数器分辨率图](deadline_20260907/runs/T21/diagnose/source_counter_resolution_review.svg)、[四轮host/GPU/时间覆盖](deadline_20260907/runs/T21/diagnose/source_counter_runtime_scope_coverage.csv)、[既有归属方法审查](deadline_20260907/runs/T21/diagnose/source_existing_counter_method_review.csv)、[逐graph区间与采样元数据](deadline_20260907/runs/T21/diagnose/source_graph_counter_resolution.csv)、[验收](deadline_20260907/runs/T21/acceptance.json)。专用source_only Snakemake 4步通过；只读24.24MB已有派生表/元数据/提取代码，CP/EP旧表物理包含20轮，分析明确筛选85/90/95/100的rank16，不拟合任何成本。每轮208个CP事件的union与旧CP表精确一致，384个GraphLaunch描述性括号与T19逐字段相同，source fit与incremental标签分开。

对应worker33008/GPU0的四轮设备/runtime范围均落在同一74,694,601字节计数器文件的清单覆盖内。清单987,849行中包含718,581条PCIe记录，**文件全体行的5.000692ms中位数并不是MTLink专属或局部采样分辨率**；384个graph括号中192个短于这一文件指标，也不能据此断言每个短括号都无法观测。旧CP表按kernel窗口裁剪采样区间；旧EP表把相交样本全部字节按EP重叠比例分配，并假设非EP间隔不承载EP流量。二者都无法独立证明graph无设备事件区间的通信活动，不能直接作为graph服务成本。CP提取器隔离underflow及同link下一条恢复样本；EP提取器仅排除超int64的delta及非正时长，未执行同样的恢复样本隔离，历史产物保留。

37测试、4完成阶段manifest（含首版preflight）/208产物SHA、60正式文件与v6813 seal通过；最终峰值250.86MB、分析14.210s。首版把CP family误写为`CP`而断言失败，r2改为精确process-group/collective筛选并核验`CP_collective`，输入和统计时长未改。原始计数器仅stat、未解析；后续资源方案登记时只计算单文件SHA。采样end使用host_realtime的既有映射仍需检查实际字段；别处通用代码的host_mono校正不证明该采集具备字段或同一部署时钟。下一步[T22单文件资源方案](deadline_20260907/T22_RESOURCE_REVIEW.json)保留两者的可识别性问题，不调整offset追逐目标误差。

本轮没有新增224预测，正式/最新研究候选1F1B仍11.340786%/14.994403%；step、MFU、entry/tail/outer均保持此前版本。现有硬件表不能直接补齐graph可见性，仍需更细的独立数据或代码证据。

T22已完成：[MTLink采样与条件字节界限图](deadline_20260907/runs/T22/diagnose/source85_counter_interval_probe.svg)、[原始字段/读取范围合同](deadline_20260907/runs/T22/diagnose/source85_counter_field_and_physical_read_contract.json)、[选中采样表](deadline_20260907/runs/T22/diagnose/source85_MTLink_samples.csv.gz)、[graph及设备空档分段](deadline_20260907/runs/T22/diagnose/source85_graph_device_gap_segments.csv)、[逐窗口字节界限](deadline_20260907/runs/T22/diagnose/source85_graph_counter_bounds.csv)、[样本/窗口关联](deadline_20260907/runs/T22/diagnose/source85_graph_counter_sample_intersections.csv.gz)、[验收](deadline_20260907/runs/T22/acceptance.json)。专用Snakemake 4步通过，按预登记SHA仅分块解析74,694,601字节的一份源计数器文件；物理读987,849行、排除718,581条PCIe，269,268条MTLink用于顺序/reset检查，仅输出和分析source85的2,933条MTLink样本。没有重读profiler trace或读取目标计数器。

真实CSV仅有6个字段：host_realtime、link_id、tx/rx delta及mt_timestamp_begin/end；缺失host_mono、gpu_id、ret、link_state、采集iter和单列mt_dt。因此只能保留继承的`legacy_host_end`映射，host/GPU身份依据已冻结文件路径与manifest，未恢复独立设备状态或时钟校准。source85各MTLink链路采样时长中位数4.998830–5.007294ms；旧清单混入PCIe的5.000692ms不能替代本轮逐link结果。2,933条选中样本没有重复、underflow或非正时长，但按旧epoch映射有1,598条参与同link区间重叠，保守排除后1,335条用于字节界限；不把排除样本当作零流量。

96个graph括号各给出完整括号、无非PP设备事件、无任何设备事件三种重叠视图，共288行；5,729个样本/窗口关联可逐行重算。完全落在分析区间内、通过已有字段检查且不重叠的样本只给出**以时钟映射成立为前提的字节下界**；跨边界样本字节只形成已观测样本的cap，不按比例生成精确字节。全部288个窗口都有未覆盖的link×time，完整窗口上界均为null，未知覆盖不填0。source85的无任何设备事件视图中10/96个窗口有正TX下界，合计416,717,312字节；分为反向dispatch 5个、重算combine 3个、前向combine 2个。它支持进一步核验未显示的通信，但不是已证实的EP载荷、连续busy时间或graph服务成本；嵌套视图和跨窗口样本cap不能相加。

40测试、3阶段manifest/161产物SHA、60正式文件和v6813 seal通过；字节界限独立从关联表重算一致。峰值306.17MB、分析77.052s，通过1GiB/120s资源目标，未扩展其他源文件。新测试覆盖uint64 reset及同link恢复、重复样本、整数epoch映射、部分样本不可比例分摊、缺失link的未知上界、重叠排除；首个单测入口缺scripts/w37导入路径，修正测试bootstrap后全40项通过，科学管线首版即PASS。读取审计新增独立的raw_hardware_counter字段，既有raw_trace字段仍只统计profiler解析。

本轮没有拟合成本或新增224预测，正式/最新候选1F1B仍11.340786%/14.994403%，step/MFU及其他阶段保持原结果。下一项[T23缓存时钟敏感性核验](deadline_20260907/T23_PLAN.json)先判断这10个窗口的条件证据是否依赖采样映射，再讨论可独立预测的运行时成分；不能把时钟调整变成误差调参。

T23已完成：[时钟/样本消融图](deadline_20260907/runs/T23/diagnose/source85_counter_clock_sensitivity.svg)、[逐窗口敏感性](deadline_20260907/runs/T23/diagnose/source85_graph_counter_clock_sensitivity.csv)、[整数offset完整分段函数](deadline_20260907/runs/T23/diagnose/source85_graph_counter_offset_lower_curve.csv.gz)、[样本准入offset区间](deadline_20260907/runs/T23/diagnose/source85_counter_sample_containment_offset_ranges.csv.gz)、[字段与假设](deadline_20260907/runs/T23/diagnose/field_contract.json)、[验收](deadline_20260907/runs/T23/acceptance.json)。仅复用T22的572,048字节源缓存，两个版本均Snakemake 4步PASS，无raw解析或目标时长读取。2,933个样本在原文件逐link顺序的设备时间轴上没有倒序或重叠；旧host-end映射却有1,598个重叠参与样本，不能把这种映射重叠解释为设备重复计数。

host_realtime−mt_end的观测跨度为644,349ns，即0.644349ms；本轮对统一常数offset的所有整数ns情景精确求完全包含样本的字节下界最小值。这个观测范围**不是已校准的真实offset界限**，统一offset、实际查询延迟和漂移仍未验证。先固定T22的1,335个有效样本，原10/96个正TX下界变为9/96，逐窗口下界合计416,717,312→315,325,672字节；单独恢复设备时间轴不重叠的全部2,933样本后，仍为9/96，合计594,346,144字节。较大的字节数部分来自样本重新准入，不能归功于时钟调整或解释成预测改善。

失去全范围正下界的是source85/rank16、MB0、layer4的反向dispatch（graph event59749）：旧下界6,769,152字节，但某些offset情景下下界为0；不能因此断言无通信。其余9个为反向dispatch 4、重算combine 3、前向combine 2。这些仍只是所选缓存与时钟假设下的通信存在证据，缺失覆盖、真实时钟、EP载荷归属和独立graph服务时长均未解决；不新增服务成本或因果边。

r2补充同样本消融，r1全部原科学列精确不变。42测试、6阶段manifest/324产物SHA、60正式文件和v6813 seal通过；另独立在953个分段的1,906个首尾整数offset直接重算包含关系与TX/RX，全部一致。最终峰值210.62MB、分析23.976s，图例已核验。正式/最新研究1F1B仍为11.340786%/14.994403%，没有新的224预测，step/MFU和其他阶段维持已有结果。复现使用`wave23_source_counter_clock_review.json`，以新run-id运行专用pipeline；历史版本与事前方案保留。

下一项[T24源部署证据准入](deadline_20260907/T24_RESOURCE_REVIEW.json)转向已定位的训练启动日志与脚本：脚本声明DeepEP wheel 1.1.0+d7577a6、ACE等运行时开关，但尚未证实实际部署。先核验源侧实际配置，再决定是否有独立特征支撑成本迁移，不继续以不可辨识的计数器字节修补模型。

T24已完成：[源部署核验表](deadline_20260907/runs/T24/diagnose/SOURCE_DEPLOYMENT_REVIEW.md)、[逐字段/行号/SHA证据](deadline_20260907/runs/T24/diagnose/source_startup_static_observations.csv)、[启动脚本声明](deadline_20260907/runs/T24/diagnose/source_launch_script_claims.csv)、[字段缺口](deadline_20260907/runs/T24/diagnose/source_deployment_field_readiness.csv)、[读取合同](deadline_20260907/runs/T24/diagnose/field_contract.json)、[验收](deadline_20260907/runs/T24/acceptance.json)。专用Snakemake 4步PASS；单source256、worker33008/node2训练日志537,378字节先SHA核验，仅解析前262,144字节。原始profiler/硬件计数器均未读，目标时长仍仅SHA，日志没有复制。

得到384条白名单静态观测、48个字段，每个字段均出现8次且字面值一致；记录位于7行，其中第103行含两次同字段出现。它们不能独立证明8个rank的身份。实际配置确认PP16/CP2/EP8/TP1、60层、full/block重算4层、DeepEP启用、num_sms=20、flex dispatcher及关闭EP/shared-expert overlap；sequence_parallel=False、mtp_num_layers=None、FP8=None。脚本虽写了sequence-parallel开关、文件名含MTP1，均不能代替实际配置。14项scenario静态对照中8项一致、6项未观察，无可见冲突；未观察项保留未知，不从脚本或别名默认补值。

该受限前缀没有提取到ACE环境开关、DeepEP精确构建、GPU型号/时钟或部署代码revision；源码中的wheel 1.1.0+d7577a6安装请求不是安装成功证据。厂商[ACE图执行说明](https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/history_version/v5.1.0/libraries/mccl/mccl_over_ace/)描述了独立搬运引擎及含同步/拷贝/原子的MUSA Graph，只支持继续调查graph可见性的机制假设，不能证明本次自定义DeepEP走同一路径或提供服务时长。检索边界保存在[T24公共来源核验](deadline_20260907/T24_PUBLIC_SOURCE_REVIEW.json)。

45测试、3阶段manifest/162产物SHA、60正式文件和v6813 seal通过；峰值151.85MB、分析0.03695s，未扩日志范围。前缀内未命中训练指标停止行，不声称物理buffer从未包含源时序文本；只分析静态白名单、不拟合成本。验收首版错误地预期8个物理行，修正为7行/每字段8次出现，科学产物未修改。没有新增224预测，正式/最新研究1F1B仍11.340786%/14.994403%，step/MFU及其他阶段保持已有结果。

下一项[T25目标配置事后核验](deadline_20260907/T25_RESOURCE_REVIEW.json)固定T24解析器和源字段，只在已有全局seal后处理同口径目标日志前缀；目标产物永不回收为源拟合输入。先判断有无真实配置差异，再讨论可预测的成本迁移。

T25已完成：[源/目标配置对照](deadline_20260907/runs/T25/diagnose/evaluator_only/DEPLOYMENT_COMPARISON.md)、[全部白名单比较](deadline_20260907/runs/T25/diagnose/evaluator_only/source_target_static_field_comparison.csv)、[目标逐字段/行号/SHA证据](deadline_20260907/runs/T25/diagnose/evaluator_only/target_startup_static_observations.csv)、[目标scenario对照](deadline_20260907/runs/T25/diagnose/evaluator_only/target_declared_static_comparison.csv)、[读取合同](deadline_20260907/runs/T25/diagnose/field_contract.json)、[验收](deadline_20260907/runs/T25/acceptance.json)。先核验T24完整源阶段、解析器SHA和v6813全局seal，再只读取target224、worker33020/node2的一份497,890字节日志前262,144字节；专用Snakemake 4步PASS，没有raw trace/计数器扫描。

目标同样得到384条白名单观测、48个字段，各出现8次且值一致，分布在7个物理行。源/目标46个字段相同，两项差异仅为num_layers 60→52和PP16→14，已被现有场景建模；另43项因至少一侧缺少观测而无法比较。目标14项scenario检查中8项一致、6项未观察，无可见冲突。full/block重算、DeepEP启用/20个SM配置、EP overlap关闭等可见值一致，不能据此断言GPU/CPU状态、后端版本或全部环境相同；ACE开关和精确DeepEP构建仍未验证。

47测试、3阶段manifest/165产物SHA、60正式文件、原源产物与解析器及v6813 seal通过；91项字段比较独立重算一致。峰值154.43MB、分析0.05153s。受限前缀未命中训练指标停止行，读取合同明确buffer可能包含时序文本，但没有提取或拟合目标时长。目标及混合表始终位于evaluator_only，不能当作新的源成本输入。

本轮没有找到可支持成本修正的新配置差异，没有新224预测或正式图修改；正式/最新研究1F1B仍11.340786%/14.994403%，step/MFU及其他阶段沿用现有封存结果。下一项[T26A已封存设备模型重载](deadline_20260907/T26A_PLAN.json)转回源侧预测模型：先精确恢复T19 source85/90参数和全部旧预测，再单独检验目标在观测CPU提交条件下的设备完成误差，定位可见设备成本迁移与CPU/隐藏graph等待之间的剩余缺口。局部条件预测不等于全局1F1B预测。

T26A已完成：[逐轮/方法重载核验](deadline_20260907/runs/T26A/diagnose/source_device_queue_reload_per_iteration.csv)、[重载局部seal](deadline_20260907/runs/T26A/diagnose/source_device_queue_reload_seal.json)、[完整重载预测](deadline_20260907/runs/T26A/diagnose/source_device_queue_reloaded_predictions.csv.gz)、[原样源参数](deadline_20260907/runs/T26A/diagnose/source_device_queue_parameters.csv.gz)、[字段合同](deadline_20260907/runs/T26A/diagnose/field_contract.json)、[验收](deadline_20260907/runs/T26A/acceptance.json)。只准入T19的9个封存文件共4,982,712字节，专用Snakemake 4步PASS；新增显式source_model_stages准入，不放宽原先只接收未拟合观察数据的规则。

2,772行成本全部来自原source85/90参数；source95/100仍是历史已暴露的增量验证。重载四层cost查找表后，41,764个设备事件×四种固定方法的167,056条预测，时间、前驱、cost id和全部字段均与旧局部seal一致，解压CSV文本也完全相同。参数和旧端点指标逐字节不变；四个源模型/关联模块与T19代码快照SHA一致。没有重新拟合、目标语义读取或raw解析。

49测试、3阶段manifest/170产物SHA、60正式文件和v6813 seal通过；峰值486.60MB、分析5.718s。新测试验证重载等价、错误拟合范围/成本哈希拒绝、GPU真值字段拒绝及目标角色不能作为源参数。原四种方法保留，包括T19已知在F阶段严重退化的均值版本；没有根据目标选择版本。

这只是已有局部模型的精确恢复，没有新增全局224预测：正式/最新研究1F1B仍11.340786%/14.994403%，step/MFU和其他阶段未变。下一项[T26B目标条件迁移检验](deadline_20260907/T26B_PLAN.json)固定这些成本，在实际CPU提交和算子元数据条件下评估目标设备完成误差，检验可见设备成本迁移；不将局部条件端点冒充CPU F/B或完整1F1B预测。

## T26B：固定源设备成本的目标条件迁移

[T26B图](deadline_20260907/runs/T26B/diagnose/evaluator_only/conditional_device_queue_transfer.svg)、[逐事件](deadline_20260907/runs/T26B/diagnose/evaluator_only/target_device_queue_event_results.csv.gz)、[逐F/B末端](deadline_20260907/runs/T26B/diagnose/evaluator_only/target_CPU_owned_device_endpoints.csv)、[分账口径](deadline_20260907/runs/T26B/diagnose/evaluator_only/target_device_queue_metrics_by_scope.csv)、[端点事件选择](deadline_20260907/runs/T26B/diagnose/evaluator_only/target_CPU_owned_endpoint_event_selection.csv)、[释放代理反例](deadline_20260907/runs/T26B/diagnose/evaluator_only/target_release_proxy_violations.csv)、[局部seal](deadline_20260907/runs/T26B/diagnose/evaluator_only/target_conditional_device_prediction_seal.json)、[验收](deadline_20260907/runs/T26B/acceptance.json)。仅复用11.75MB已封存文件；专用Snakemake4步PASS，33,816个目标事件×4方法，源2,772行参数逐字节不变、无未覆盖成本。

| 固定源方法 | 源95/100 F末端MAE ms | 目标四轮 F末端MAE ms | 源95/100 B末端MAE ms | 目标四轮 B末端MAE ms |
|---|---:|---:|---:|---:|
| 独立延迟 | 12.553023 | 4.604809 | 3.751851 | 10.234830 |
| 同stream中位数队列 | 0.031298 | 0.846204 | 0.835315 | 11.420797 |
| 同stream均值队列 | 680.881481 | 582.187307 | 0.604612 | 11.367135 |
| 同stream零余项 | 0.039477 | 0.652732 | 41.345380 | 50.095379 |

目标每相位12个CPU归属的非PP设备末端，源增量每相位8个；实际目标CPU提交起止、算子/shape/phase是条件。中位数队列相对独立延迟改善目标F但B退化；均值F在两侧均失败。这些都不是CPU F/B、完整1F1B、Step或MFU误差。目标GPU真值在关联和顺序审计中已读取，投影到固定特征后才预测/seal/评分；不能称盲测。

所有事件中位数时长MAE2.233748ms被56个PP候选kernel显著放大：PP驻留MAE1,170.427964ms，含等待，不能当网络服务。排除PP后33,760事件时长MAE0.295985ms；进一步只看F/B归属22,692事件，时长MAE0.230005ms、启动MAE9.489640ms、完成MAE9.590851ms。逐事件严格满足完成误差=启动误差+时长误差，MAE及重叠事件时长不相加为phase成本。

312个目标kernel在API返回前启动，提前18ns–7.975261ms；同stream顺序无逆序/重叠，不能据此继续把API结束当普适释放时刻。12个B观察末端均落在同一成本键的elementwise事件，启动误差为毫秒量级、时长误差微秒量级，预测最大末端与观察最大末端事件亦不同；末端差不能全部解释为最后kernel计算成本漂移。下一轮[T27A](deadline_20260907/T27A_PLAN.json)只在源侧建立API开始代理候选，保留旧方法消融；隐藏graph和跨stream约束仍未识别。

r2只补充范围/反例/端点分账，r1全部16份科学CSV解压文本精确相同；50测试、6阶段manifest/379产物、60正式文件、源四模块及全局seal通过。峰值625.08MB、分析15.096s，无raw扫描。正式/最新全局研究1F1B仍11.340786%/14.994403%；其他阶段与Step/MFU没有新预测，维持已有封存结果。

周报可用两句：固定源设备模型在224卡观测CPU提交条件下，将前向设备末端MAE由4.605ms降到0.846ms，但反向由10.235ms退化到11.421ms，尚未形成全局1F1B改善。已核验API结束释放代理的312个kernel反例，并将PP等待混入驻留成本与设备启动偏差分别留证，下一步源侧检验API开始代理；独立CPU/隐藏graph、跨rank迁移和未暴露数据仍未验证。

## T27A：只改变源设备释放代理

[代理对照图](deadline_20260907/runs/T27A/diagnose/source_release_comparison.svg)、[末端误差对照](deadline_20260907/runs/T27A/diagnose/source_release_endpoint_comparison.csv)、[逐事件指标](deadline_20260907/runs/T27A/diagnose/source_release_event_comparison.csv)、[参数变化](deadline_20260907/runs/T27A/diagnose/source_release_parameter_changes.csv.gz)、[新候选逐轮结果](deadline_20260907/runs/T27A/diagnose/all_API_start/source_device_queue_per_iteration.csv)、[新候选完整预测](deadline_20260907/runs/T27A/diagnose/all_API_start/source_device_queue_predictions_sealed.csv.gz)、[比较seal](deadline_20260907/runs/T27A/diagnose/source_release_comparison_seal.json)、[验收](deadline_20260907/runs/T27A/acceptance.json)。复现入口为专用pipeline.py加`--spec docs/w37/1f1b/deadline_20260907/wave27a_source_API_start_release.json --run-id 新名称`。

T27A在独立模块将所有类别的释放代理改为实际CPU API开始时刻，原T19模块未改。仅复用8.38MB T14B源缓存，两版分别按原四层成本键、四种方法拟合85/90，再封存参数/特征/预测，检验95/100。API开始是观测条件的下边界，未证明等于有效设备释放；源拟合排队余项仍可能包含host提交、隐藏graph及跨stream约束。

| 源95/100中位数队列 | 旧API结束代理 | 新全部API开始代理 |
|---|---:|---:|
| F设备末端MAE ms，8个窗口 | 0.0312975 | 0.0369360 |
| B设备末端MAE ms，8个窗口 | 0.8353154 | 0.8349899 |
| 21,218事件完成MAE ms | 16.7897427 | 16.7981051 |

源侧没有实质改善；F略退化、B只改善约0.000326ms。均值方法的F大幅退化仍保留，680.881→681.452ms。2,772行成本键、分组、样本数、计算驻留参数及候选依赖边完全不变，变化只在API延迟和排队余项；未按目标选择版本或增删边。它修正已观察到的异步API边界反例，不能因此声称已解释目标较大运行时偏差。

专用Snakemake4步PASS；51测试、3阶段manifest/214产物SHA、60正式文件及全局seal通过。旧对照14份CSV解压文本与T19精确一致，含167,056条旧预测；两版共334,112条预测完成误差守恒，F/B设备末端独立重算一致，增量GPU真值扰动不影响拟合参数和条件特征。首次单测发现新字段误用Python保留字，运行管线前修复，失败原因保存在验收中。峰值658.97MB、分析51.696s，没有raw或目标时长解析。

独立候选保留，下一项[T27B](deadline_20260907/T27B_PLAN.json)在相同目标缓存上检验两版条件迁移。正式/最新全局研究1F1B仍11.340786%/14.994403%，其他阶段及Step/MFU均未产生新预测。

周报可用两句：已完成API开始释放代理的源侧同口径消融，确认驻留成本和依赖不变，局部预测误差没有实质改善。该修正处理了API返回前执行的边界反例，但独立CPU入口、隐藏graph、跨stream就绪与跨rank迁移仍未验证，暂不计入224卡全局1F1B改善。

## T27B：两种释放代理的目标条件消融

[对照图](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_comparison.svg)、[设备末端指标](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_endpoint_metrics.csv)、[F/B内逐事件分账](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_metrics_by_scope.csv)、[逐迭代结果](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_per_iteration.csv)、[末端事件选择](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_endpoint_event_selection.csv)、[释放代理反例](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_proxy_violations.csv)、[旧代理seal](deadline_20260907/runs/T27B/diagnose/evaluator_only/legacy_API_end/target_conditional_device_prediction_seal.json)、[新代理seal](deadline_20260907/runs/T27B/diagnose/evaluator_only/all_API_start/target_conditional_device_prediction_seal.json)、[验收](deadline_20260907/runs/T27B/acceptance.json)。复现入口为专用pipeline.py加`--spec docs/w37/1f1b/deadline_20260907/wave27b_target_API_start_release.json --run-id 新名称`。

T27B只在T17B已封存的target224/rank16缓存上迁移T27A两份source85/90参数，保留四种固定方法，并在连接GPU完成真值评分前分别封存条件预测。旧API结束代理的特征、270,528行中的旧候选预测及末端指标与T26B解压文本精确一致；两候选各2,772行成本参数，无源或目标参数更新、无未覆盖成本。旧/新目标局部seal分别为`96cf436f...`与`5ad2c080...`。

| 目标四轮CPU归属设备末端 MAE ms | 旧API结束代理 | 新全部API开始代理 |
|---|---:|---:|
| 独立延迟 F | 4.604809 | 4.604437 |
| 独立延迟 B | 10.234830 | 10.220776 |
| 同stream中位数 F | 0.846204 | 0.760596 |
| 同stream中位数 B | 11.420797 | 11.411286 |

API开始代理把释放边界反例从312个降为0，因它是CPU调用入口的下边界；这仍未证明设备在该时刻真正可执行。中位数方法的F/B末端分别小幅改善0.085609/0.009511ms，但F/B归属的22,692个非PP设备事件中，启动MAE **9.489640→9.526528ms**、完成MAE **9.590851→9.628238ms**，驻留时长MAE固定 **0.230005ms**。全可见事件完成MAE也由23.185155升至23.225993ms。因此只保留新代理作为语义更安全的局部候选，不提升正式或全局研究版本。

末端表还暴露了新的可解释性缺口：各窗口观察到的可见设备末端与“预测完成时间最大”的事件可以来自不同stream和不同算子；局部端点误差可能由跨stream最大值碰巧抵消，不能据一个phase的包络接近就说内部事件预测正确。下一轮只在源侧事前登记终端事件选择规则，再用95/100增量验证，不根据目标误差增删依赖或插等待。

专用Snakemake4步PASS；51测试、3阶段manifest/217产物SHA、60正式文件及全局seal通过，33,816个目标事件、270,528条预测。峰值902.05MB、分析26.480s，没有raw重扫。r1到r5依次暴露并保留：源seal文件准入、`complete.json`定位、旧目标seal参数准入、浮点比较和旧CSV附加候选列问题；r6通过，未删除失败目录。正式/最新全局研究1F1B仍 **11.340786% / 14.994403%**，本轮没有全局1F1B、其他阶段、Step或MFU新预测。

周报可用两句：两种源侧封存释放代理在224卡条件迁移中只带来F/B设备末端0.085609/0.009511ms的小幅改善，而F/B内逐事件完成MAE反而由9.590851ms升至9.628238ms，未形成全局1F1B改善。API开始代理消除了312个已观察边界矛盾，但有效释放时刻、跨stream终端选择、独立CPU入口、隐藏graph、跨rank迁移和未暴露数据仍未验证。

## T28A：纯源终端事件身份与端点选择

[语义对照图](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_semantics.svg)、[终端身份参数](deadline_20260907/runs/T28A/diagnose/source_terminal_identity_parameters.csv)、[源拟合选择参数](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_selection_parameters.csv)、[增量指标](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_metrics.csv)、[逐迭代结果](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_per_iteration.csv)、[逐窗口身份核验](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_event_selection_validation.csv.gz)、[预测seal](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_prediction_seal.json)、[验收](deadline_20260907/runs/T28A/acceptance.json)。复现入口为专用pipeline.py加`--spec docs/w37/1f1b/deadline_20260907/wave28a_source_terminal_endpoint.json --run-id 新名称`。

T28A重新从T14B纯源缓存构造两种释放代理和四种固定队列方法，只允许source85/90确定终端身份、阶段规则/方法和CPU标注后尾部；640行端点预测封存后才连接source95/100完成真值。改变95/100的GPU启动、结束、驻留与排队真值，5,544行设备成本、终端身份、选择参数、候选事件和全部预测保持精确不变。原“预测完成最大值”的全部端点指标与T27A精确一致。

source85/90的16个F/B拟合窗口都由stream0、`aten::_copy_from`、elementwise/kernel作为观察末端；source95/100另16个窗口保持同一身份，命中 **16/16**。但是按source85/90端点MAE选择规则后，反向仍选择`predicted_maximum/same_stream_mean`，其source95/100身份命中 **0/8**；前向选择稳定终端事件，身份命中 **8/8**。因此反向0.60ms量级的包络误差仍是stream15 `aten::add_`与真实stream0 copy端点的最大值抵消，不能解释为终端事件本身算准。

| source95/100 源拟合所选端点 | 旧API结束代理 | 新全部API开始代理 |
|---|---:|---:|
| F末端MAE ms / 身份命中 | 0.023677 / 8/8 | 0.013204 / 8/8 |
| B末端MAE ms / 身份命中 | 0.604612 / 0/8 | 0.605248 / 0/8 |
| 强制真实终端身份、独立延迟 B MAE ms | 0.676671 / 8/8 | 0.676665 / 8/8 |

强制终端身份的反向独立延迟仅比错误身份的最小包络多约0.072ms，却把归因从0/8提高到8/8；这是下一轮需保留的可解释性—精度权衡。源拟合组合相对同口径阶段最优控制只带来很小的平均端点改善，未生成CPU F/B或224卡全局预测。53测试、3阶段manifest/193产物、60正式文件及全局seal通过；峰值699.55MB、分析33.157s，无raw或目标时序读取。独立验收首次只因派生窗口计数的float/int dtype拒绝，改为数值精确比较后通过，科学产物未改。

周报可用两句：纯源终端核验确认32个F/B窗口的可见设备末端均为stream0 `aten::_copy_from`，前向源拟合规则在95/100达到8/8身份命中和0.013–0.024ms MAE。反向最低端点MAE约0.605ms仍由错误的stream15事件形成，强制真实终端身份为0.677ms且8/8命中；下一轮封存这项精度—可解释性权衡后做224卡开发诊断，独立CPU入口、隐藏graph、跨rank和全局1F1B仍未验证。

## T28B：源固定终端语义的目标条件迁移

[目标终端对照图](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_transfer.svg)、[总体指标](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_metrics.csv)、[逐迭代指标](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_per_iteration.csv)、[目标终端身份](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_observed_terminal_identity_counts.csv)、[候选事件选择](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_candidate_event_selection.csv.gz)、[预测seal](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_prediction_seal.json)、[验收](deadline_20260907/runs/T28B/acceptance.json)。复现入口为专用pipeline.py加`--spec docs/w37/1f1b/deadline_20260907/wave28b_target_terminal_endpoint.json --run-id 新名称`。

T28B在读取目标终点真值前固定四类候选：T27B四方法最大完成控制、T28A源拟合阶段选择、源固定真实终端身份的独立延迟，以及CPU F/B标注结束加源阶段中位尾部。输入只来自T28A源seal和T27B已封存的目标条件表；算子、stream、方法、边和等待均不能由目标结果选择。对目标终点时间字段做`-999999`突变后，336行预测仍精确一致；之后才连接终点时间和身份评分。

目标24个F/B窗口的可见设备末端全部保持stream0 `aten::_copy_from`、elementwise/kernel，源固定语义身份命中 **24/24**。在API开始释放代理下，语义独立延迟相对同方法“预测完成最大值”同时改善两个阶段：F由 **4.604437→0.624367ms**，B由 **10.220776→10.204995ms**；身份命中由F **1/12→12/12**、B **9/12→12/12**。同stream中位数控制为F/B **0.760596/11.411286ms**，所以语义候选也同时较低。源拟合“按时间选最优”规则的B仍为11.357917ms且身份0/12，再次证明只看包络误差会保留错误事件归因。

| API开始语义终端 MAE ms | 85 | 90 | 95 | 100 |
|---|---:|---:|---:|---:|
| F，均为3个窗口 | 0.014264 | 2.438860 | 0.015223 | 0.029121 |
| B，均为3个窗口 | 8.524507 | 17.674386 | 4.258244 | 10.362845 |

逐迭代结果暴露出target90前向2.439ms异常和反向4.258–17.674ms的大幅波动；语义终端解决了“预测的是哪个事件”，尚未预测产生这些CPU提交与设备排队差异的自由运行输入。观察CPU提交、事件元数据和CPU F/B标注结束仍是条件，不能把上述局部毫秒改善换算成全局1F1B MAPE。

专用Snakemake4步PASS；54测试、3阶段manifest/193产物、60正式文件通过，无source/target参数更新、raw重扫或正式拓扑变化。独立复核发现并修复gzip时间戳与SVG日期/随机ID后，r6/r7的12项科学表、seal、PNG和SVG均字节精确；r7为正式结果，seal为`353fd2c9142fe13a3e0208bce441e8ccc470ed0f3e081531e4d038b1e0581ad2`。r1浮点比较拒绝、r2缺真值突变门、r3压缩字节不稳定、r4/r5 SVG不稳定均保留。正式/最新全局研究1F1B仍 **11.340786% / 14.994403%**；其他阶段、Step和MFU没有新预测，因此既没有退化结论也没有改善声明。

周报可用两句：源固定stream0 `aten::_copy_from`终端在224卡24个F/B开发窗口全部命中，并将API开始候选的F/B设备终点MAE由4.604437/10.220776ms降至0.624367/10.204995ms，首次同时改善局部时间误差和事件归因。该结果仍以观测CPU提交和F/B标注结束为条件，尚未形成全局1F1B、Step或MFU改善；target90前向异常、反向迭代波动、隐藏graph、跨rank和未暴露数据仍未验证。

## T29A：CPU终端提交、设备完成与PP重复计时核验

[分解与重叠图](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition.svg)、[参数](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_parameters.csv)、[增量指标](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_metrics.csv)、[逐迭代结果](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_per_iteration.csv)、[区间摘要](deadline_20260907/runs/T29A/diagnose/source_terminal_PP_interval_summary.csv)、[逐窗口账本](deadline_20260907/runs/T29A/diagnose/source_terminal_PP_interval_ledger.csv.gz)、[正式图PP节点](deadline_20260907/runs/T29A/diagnose/sealed_v6813_source85_B_PP_readiness_nodes.csv)、[预测seal](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_prediction_seal.json)、[验收](deadline_20260907/runs/T29A/acceptance.json)。复现入口为专用pipeline.py加`--spec docs/w37/1f1b/deadline_20260907/wave29a_source_terminal_decomposition.json --run-id 新名称`。

T29A只准入T14B纯源缓存和T28A源终端seal。source85/90拟合“phase+microbatch role的CPU终端提交偏移”和“phase中位设备完成延迟”，160行四轮候选预测封存后才连接source95/100的CPU结束、API/设备终点及后续PP边界。预测仅使用F/B起点、phase和microbatch role；突变95/100的CPU结束、终端API/设备时刻及PP起止后，参数和预测精确不变。这里的F/B起点仍是source局部观测时间原点，进入全局图前必须换成预测调度边界。

| source95/100 | F | B |
|---|---:|---:|
| 分解端点MAE ms | 2.726681 | 6.659497 |
| CPU终端提交MAE ms | 2.728800 | 6.506944 |
| 仅CPU结束消融的端点MAE ms | 2.722242 | 99.673863 |

设备提交后的剩余误差很小，B的主要困难已收敛到CPU何时提交终端copy；但这个局部改善不能直接成为新成本。32/32个终端API均在F/B CPU标注内提交，B的16/16个设备终点均位于随后`send_backward` PP API内，且正终端尾部与PP区间重叠至少 **99.921487%**。逐窗口恒等式`F/B起点→PP返回 = CPU F/B wall + F/B尾至PP入口 + PP入口至终端设备结束 + 终端结束至PP返回`误差为0ns；终端没有跨过下一F/B起点。

现有v6813源图的B PP sender-readiness为93.079–93.287ms，随后另有5.157ms post-publication completion，与观察到的PP入口至终端结束约93.115ms、终端结束至PP返回约4.960ms语义相同。若再给F/B追加约93ms，必然重复计时；替换PP readiness是唯一可能的结构保持用法，但T16同口径source95/100局部证据已显示stream0 GPU端点 **0.663167ms** 差于现有PP方法 **0.557697ms**，因此也不登记T29B。正式拓扑、全局1F1B、其他阶段、Step和MFU保持不变；正式目标1F1B基线仍为 **11.340786%**。

专用Snakemake4步PASS；55测试、3阶段manifest/195产物及60正式文件通过，44个冻结输入、38,907,146字节，无raw或目标时序读取。r1/r2的12项科学表、seal、PNG和SVG字节精确；正式r2峰值342.54MB、分析5.268s。

周报可用两句：纯源分解表明source95/100语义终端的F/B端点MAE为2.726681/6.659497ms，其中主要误差来自CPU提交；32个终端API都在F/B内提交，反向16个设备终点全部落在随后PP API内。反向终端尾部与现有PP区间重叠至少99.921%，追加会重复计时，而既有替换实验又弱于原PP方法，因此不做目标迁移或全局误差改善声明；独立CPU提交、跨rank和未暴露数据仍未验证。

## T30A：CPU终端提交的静态可辨识性

[可辨识性图](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_identifiability.svg)、[源拟合静态签名](deadline_20260907/runs/T30A/diagnose/source_CPU_expected_static_signatures.csv)、[提交参数](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_parameters.csv)、[增量指标](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_metrics.csv)、[逐迭代结果](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_per_iteration.csv)、[32窗口签名验证](deadline_20260907/runs/T30A/diagnose/source_CPU_static_signature_validation.csv)、[算子计数底表](deadline_20260907/runs/T30A/diagnose/source_CPU_operation_signature_counts.csv.gz)、[预测seal](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_prediction_seal.json)、[验收](deadline_20260907/runs/T30A/acceptance.json)。复现入口为专用pipeline.py加`--spec docs/w37/1f1b/deadline_20260907/wave30a_source_CPU_submission_identifiability.json --run-id 新名称`。

T30A只从T14B纯源缓存提取F/B内跨CPU线程、完全包含的算子名称计数，不把时长或时间戳写入签名。source85/90确定期望签名、根操作和前序PP API，再分别封存“phase+microbatch role”和“再加静态签名”的64行提交预测；之后才接入source95/100的终端API时刻和观察签名。突变验证时序与签名时，拟合参数和预测精确不变。

F的每个窗口均为2,802个、79类CPU操作、根`forward_step`、前序`recv_forward`；B均为7,738个、179类、根`backward_step`、前序`recv_backward`。这些字段在source95/100的16/16窗口全部匹配source85/90。加入完整算子签名后，phase/role的6个等价类仍为6个，控制和扩展预测逐值相同；F/B提交MAE仍为 **2.728875/6.506913ms**。因此静态图可确定“提交哪些操作”，不能辨识迭代间CPU执行速度和运行时状态；本候选不向target迁移。

专用Snakemake4步PASS；56测试、3阶段manifest/199产物、60正式文件及全局seal通过，32个冻结输入38,627,214字节，无raw或目标读取。r1/r2的13项科学表、seal、PNG/SVG字节精确；正式r2峰值260.43MB、分析4.704s。正式图、全局1F1B、其他阶段、Step和MFU均未变化。

周报可用两句：源侧32个F/B窗口的CPU算子计数签名逐phase完全一致，source95/100的16个验证窗口全部匹配，但新增静态签名没有增加phase/role的6个预测等价类，F/B提交MAE仍为2.728875/6.506913ms。该结果证明当前缺口需要独立运行时状态或更完整的全rank成本估计，不能用算子身份重复加时；本轮未读目标、未改拓扑，也没有全局1F1B、Step或MFU改善。

## T31A：v6.8.5 成本切分与 source replay 绑定核验

[成本对照图](deadline_20260907/runs/T31A/diagnose/source_fullrank_cost_split.svg)、[分项指标](deadline_20260907/runs/T31A/diagnose/source_component_metrics.csv)、[逐迭代结果](deadline_20260907/runs/T31A/diagnose/source_component_per_iteration.csv)、[计算参数](deadline_20260907/runs/T31A/diagnose/source_compute_parameters.csv)、[PP参数](deadline_20260907/runs/T31A/diagnose/source_PP_parameters.csv)、[entry参数](deadline_20260907/runs/T31A/diagnose/source_entry_parameters.csv)、[绑定审计](deadline_20260907/runs/T31A/diagnose/v685_source_compute_binding_audit.json)、[分账表](deadline_20260907/runs/T31A/diagnose/component_accounting_ledger.csv)、[预测seal](deadline_20260907/runs/T31A/diagnose/source_fullrank_cost_prediction_seal.json)、[验收](deadline_20260907/runs/T31A/acceptance.json)。复现入口为专用pipeline.py加`--spec docs/w37/1f1b/deadline_20260907/wave31a_source_fullrank_cost_split.json --run-id 新名称`。

T31A事前固定`median85/90`和`recent90`两套估计，分别对物理计算槽、PP gradient wall与Profiler entry拟合；参数及四轮预测先seal，再连接source95/100真值。把验证集全部计算时长、PP边界和entry改成错误值后，参数与预测逐字节不变。正式v685的source85/90/95/100四轮中位只作为已暴露数据控制，不具有增量验证资格。

| source95/100 分项 MAE ms | median85/90 | recent90 | v685四轮控制（泄漏） |
|---|---:|---:|---:|
| 物理计算槽，每条观测 | 0.620091 | **0.597494** | 0.563250 |
| PP gradient wall，每条消息 | **4.537392** | 5.257209 | 2.684203 |
| Profiler entry，每轮 | **67.512658** | 108.400522 | 13.405091 |

`recent90`仅使计算槽MAE改善 **3.644%**，PP和entry分别退化 **15.864%**、**60.563%**，没有单一合格估计器同时改善三项。覆盖核验还纠正了“全rank计算成本”的表述：计算表覆盖16个PP stage各自lane0的代表rank，共16个rank、58个物理键；PP wall才覆盖256个rank。v6813在source95/100的全局1F1B MAPE为0.334014%，但它使用不同图，只保留为控制，不能替代v685计算绑定验证。

代码与节点表给出更根本的阻断：v685在目标图调用`apply_steady_compute`并更新144个节点；source replay读入旧图后只调用`apply_source_gradient_wall`，没有调用计算更新。其152,502个source节点缺少`source_parameter_key`、`compute_exposed_ns_model`和`compute_overlap_ns_model`，74,624个`local_gap`仍把593,661.079ms合计记为`unclassified_calibration_ns`，新计算参数绑定为 **0行**。因此source raw graph 23,025.241ms与由此得到的805.920ms reconciliation没有验证新计算成本；重新拟合残差会把缺失绑定吸收进去，不能作为目标预测依据。T31B不登记，正式拓扑与v685目标结果保持不变。

专用Snakemake4步PASS；58测试、3阶段manifest/208产物、60正式文件通过，20个冻结输入40,103,568字节，无raw或目标时序读取。r4/r5的19项科学表、seal、PNG与SVG字节精确；正式r5峰值1,094.66MB、分析13.186s。r1的Pandas列访问错误、r2暴露的计算覆盖误判、r3的plan字段读取错误均原样保留。

周报可用两句：v6.8.5分项复核显示，recent90只把source95/100物理计算槽MAE从0.620091ms降到0.597494ms，而PP wall和entry分别退化15.864%和60.563%，没有形成一致改善。进一步确认v6.8.5的source replay对新计算参数绑定为0行，805.920ms reconciliation不能证明新计算成本通过源侧全图验证，因此未进入224卡目标评分；计算跨lane、entry/reconciliation物理归属和兼容source图仍未验证。

## 冻结基线及数据口径

仅 Task A：source256（PP16、16 lanes、MB4、60 层）→target224（PP14、16 lanes、MB3、52 层）；不做 16→256。正式成本 v6.8.5、拓扑 v6.8.4，327,746 节点、364,784 边，锁 SHA256 `f1d560daaa344a5980697b67e71b97140c9bfd38fe12a2c676daf05c53bd8d04`。受保护 smoke 与完整 v685 重校准复现均 PASS，原始 trace 未扫描。

主评估固定为目标 85/90/95/100 四轮：Profiler MAPE **10.369528494%**、training MAPE **10.718040852%**。预测 Profiler 21,385.696422 ms、training 22,758.149812 ms。Profiler 分成 entry、第一 F/B 到最后 F/B 的 1F1B 包络、tail；training−Profiler 为 outer。

历史 16.823672% 是旧三阶段模型在 target224 的 60–100 九轮；v682 同九轮为 10.464727%，v685 当前数是四轮，不能直接连成同口径改善曲线。“10.3%”的确切历史版本仍有歧义。75.839644% 的分母是 v682 九轮平均总低估 2,452.058716 ms，分子是 1F1B 平均低估 1,859.632600 ms；它是带符号误差贡献，**不是阶段 MAPE**。详见 [历史核验](artifacts/evaluator_only/historical_metric_audit.json)。

主新函数只用 source85/90 拟合、source95/100 增量验证；source60–80 仅历史诊断。全四轮重拟合只作同窗口敏感性。历史模型与这些源迭代均有开发暴露，因此增量验证不等于独立盲测。target224 全部为开发评估或事后诊断。每轮先封存参数、节点/边、预测和源验证，再进入目标 evaluator；后续候选仍承认已知目标历史，不把访问隔离包装成盲测。

## 方法与判定

| 版本 | 可检验的改变 | 判定 |
|---|---|---|
| v685 冻结 | 现有细粒度计算/通信图 | 正式基线保留 |
| v686 research.1 | 完整 PP API 顺序；消息等双方发布，fused API 等全部消息；本地 prelaunch/return 单列 | 观测重建精确，但统一发布后成本掩盖 sender 就绪的重叠；目标回归失败 |
| v687 research.1 | `完成=max(sender_API_entry+d,receiver_API_entry)+c`，重新分配同一 PP wall | 保留为 PP 机制研究结果；源侧改善，目标整体仍失败 |
| v688 research.1 | 完整源迭代成本向量分别回放后等权平均，保留成本协方差和关键路径选择 | 相对 v687 目标略好，仍弱于正式基线；源场景跨度不足以覆盖目标 |

独立候选不覆盖正式锁，也不是现有 v6.9 的发布或绕过其输入门槛。细粒度正式图仍在 [第一轮图册](figures/index.html) 和 [结构核验](SEMANTICS.md)。新图采用 F/B 整段节点，内部物理依赖尚未全部恢复；每类依赖、成本来源和未确认开关见 [候选语义合同](post685/SEMANTICS.md)。

源侧 **52,992 个动作、17,280 对消息**匹配静态调度；用观测成本时，九轮全部 API/F-B 起止达到 **0 ns 重建差**。这只是可表示性检验。自由运行不用每轮真实发布时间。source85/90 的 B 单消息拟合出平均发送侧有效就绪 **93.0369 ms**、双方就绪后完成 **5.2470 ms**；95/100 局部 MAE **15.8588→0.5850 ms**、1F1B MAPE **1.064440→0.332811%**。93 ms 尚不能独立解释为 GPU 排空、CPU 阻塞或 transport 轮询。fused F 的单消息完成被另一 B 消息遮蔽，其 API 上界差不能当作独立传输误差。

## 同口径误差与消融

目标均为四轮开发 MAPE；源侧列仅评估新函数的 95/100 增量验证。MFU 误差与 step 误差分开。

| 模型 | 源侧 1F1B | 目标 1F1B | 目标 Profiler | 目标 training | MFU 相对误差 | MFU 平均偏差 / 百分点 |
|---|---:|---:|---:|---:|---:|---:|
| v685 | —，历史成本用过四轮 | 11.340786% | 10.369528% | 10.718041% | 12.006689% | +0.354752 |
| v686 split | 1.064440% | 14.503516% | 13.227529% | 13.393244% | 15.466473% | +0.456991 |
| v687 split | 0.332811% | 15.340746% | 13.984092% | 14.101418% | 16.418413% | +0.485121 |
| v688 两轮场景 | 0.332964% | 15.024427% | 13.698250% | 13.833858% | 16.056915% | +0.474439 |
| v688 四轮敏感性 | —，四轮已拟合 | 14.697425% | 13.402755% | 13.557263% | 15.685561% | +0.463465 |

隔离比较固定 baseline entry **1,265.388210 ms**、tail **1,004.535298 ms**、outer **1,372.453390 ms**，仅替换 1F1B 包络。它们的目标 MAPE 分别保持 18.456434%、35.269197%、15.579306%；这是固定实验约定，不是新模型已独立预测其他阶段。tail 含未完成物理归属的 reconciliation **604.439668 ms**；去掉它的敏感性全部单列，不隐藏改变总时钟。源侧当前只比较内部 1F1B 自由运行，没有新建 source optimizer/outer 预测。

v686 去本地 prelaunch/return 仅改变目标约 9.014 ms；删 receiver gate 虽让源侧汇总误差更小，却违背待检验的发布前提且目标更差，不采纳。v687 去掉 sender-ready 成本使源侧增量 1F1B MAPE 升至 **7.668593%**；ordinal MB 映射只把目标 Profiler MAPE 调至 13.846616%，不能解决主要缺口。v687 四轮 median-cost 同窗口 1F1B MAPE 1.227870%；v688 四轮场景均值在同窗口为 0.421548%，不称独立验证改善。

其他迭代也报告：目标 60–80 五轮 Profiler MAPE，v685 **6.375444%**、v686 **9.360802%**、v687 **10.151079%**、v688 两轮场景 **9.852499%**。源侧历史五轮 1F1B MAPE，v686/v687/v688 分别为 7.071896%、5.779798%、5.828326%，不保证运行时状态迁移。逐 PP/MB/F/B 与逐迭代结果在各版 `evaluator_only/`，源自由运行、局部成本、每 rank phase 起止在 `source_validation/`。

MFU 继承 FLOPs `8.436548311982576e16`、224 GPU×500 TF/s 峰值和 training 时钟；“实际 MFU”同样由实际 training 时间派生。未独立验证分子或得到另一个实测 MFU，不解释为测得硬件计算利用率。

## 1F1B 内部原因

源侧 lane0 的 GPU 四项互斥分区与 CPU phase wall 一致，时间戳浮点转换容忍 <0.001 ms；wrapper 的 prelaunch/postlaunch/outside 是另一组分区，不能与 GPU 分区再相加。其余 15 lanes 没有同等 GPU 分账。

目标 lane0 主四轮每个 B phase，源先验少估 GPU 空档 **86.012 ms**、仅通信 **45.017 ms**、仅非通信 **8.532 ms**；F 分别少估 **39.880 ms**、**22.155 ms**，仅非通信反而多估 **0.060 ms**。这不支持主要放大计算成本，但局部平均差也不能直接相加解释 step。

为处理关键路径变化，另做**目标辅助的事后诊断**：只在 lane0 子图中，对四类真实 phase 成本枚举 16 个替换子集，用精确 Shapley 分配交互影响；未修改封存预测。

| 对 lane0 平均低估的贡献 | ms |
|---|---:|
| 仅非通信 | +123.005 |
| 仅通信 | +1,157.363 |
| 非通信/通信重叠 | −188.648 |
| 无 traced GPU activity | +2,277.629 |
| 剩余 schedule / PP / 初始偏移 | +51.100 |
| 合计：实际 lane0 − 预测 lane0 | **3,420.448** |

四项 phase 反事实贡献合计 3,369.349 ms；换成真实 phase 成本后平均余差 51.100 ms，表明该局部合同可解释大部分 lane0 差额。它**不是新的预测准确率、不是全 rank 误差比例，也不证明唯一物理原因**。全 rank 原始 phase 包络另用新冻结 12,096 行表与旧 stage 包络交叉核验通过，不混淆覆盖范围。

[运行时可见性诊断](post685/runtime_review/runtime_visibility_summary.csv) 进一步显示：B 的 GPU 空档局部差额中，58.242 ms 位于 fused wrapper 内，19.116 ms 在 wrapper 外但没有可见 runtime wait，8.653 ms 与其他长 runtime call 相交；stream synchronize 仅约 0.001 ms。F 的主要差额同样在 wrapper 内（29.586 ms）。GPU 空档不能直接填入 CPU 计算成本或认定全部可优化。该源细分表原标记 `DIAGNOSTIC_VIEW_NOT_APPROVED_FOR_MODEL_FIT`、提取过程曾访问目标；本轮仅用于事后解释，没有进入预测参数。

v688 两轮源场景得到目标 1F1B **18,096.615–18,546.475 ms** 的经验跨度，远离实际四轮均值 **21,561.771 ms**。跨度不是置信区间，仅保留这些源样本的波动不足以覆盖目标运行时变化。

## 证据与复现

[最终验收](post685/verification/acceptance.json) PASS：259 份产物 SHA、60 份正式冻结文件、三版逐迭代结果精确复现、源事件重建、目标访问隔离、lane0 归因守恒与活动文档链接全部通过。9 项测试 PASS；[版本台账](post685/VERSIONS.json) 记录各版状态。

- [v686](post685/v686/)、[v687](post685/v687/)、[v688](post685/v688/)：节点/边、cost key、源验证、目标逐轮指标、prediction seal、输入审计。
- [图册和查看器](post685/delivery/index.html)：代码生成的 SVG/PNG 总览、局部依赖、调度分区、逐轮回归与归因图，附 API 等待表和局部节点/边。
- 第一轮基线与失败的 runtime-shape 优化保存在 [报告快照](history/20260905-round1/REPORT.md)；旧链接基准目录记录在 snapshot.json。v686/v687/v688 源码字节快照在 `research/w37/onef1b/post685/history/`；历史 seal 对应快照，不要求活动源码永远等于旧 SHA。
- 新输入固定于 [inputs.json](post685/inputs.json)。只用现成派生表；bubblewrap 代码/原工程只读、关闭网络、仅当次 results 可写。未启动大规模扫描或无关仿真。
- 两次 v686 入口失败保留：新脚本括号错误、迭代 55 不在历史 payload；修正后 r3 PASS，未因此调成本。图的首版标签布局问题保留原输出。最终复核还修正了场景模型的逐 stage 统计：先求每个场景的跨 rank 包络，再平均；不能对平均 rank 时间求包络。此修正不改变逐迭代 step、MFU 或主表结论。

复现（新 run-id，不覆盖旧证据）：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/1f1b/post685/reproduce.py --run-id fresh-post685
```

未验证：部署 revision/开关、有效就绪的 CPU/GPU 归属、fused-wrapper 内 GPU 空档的细依赖、独立 runtime/通信 profile、目标全部 lanes 的内部 GPU 分账、entry/reconciliation 物理归属、MFU 分子及跨规模泛化。正式升级需要相应成本预测证据，不能把目标反事实成本拿回去校准。

可用于周报的两句结论：**完成1F1B/PP/EP内部图、源成本验证和Snakemake封存评分，研究候选224卡1F1B开发MAPE为14.99%，仍未优于正式v6.8.5的11.34%，正式模型保留。源计数器时钟与样本消融保留9个无设备事件窗口的条件通信下界，但真实时钟、EP载荷/独立服务、跨rank迁移及MFU分子仍未验证，未将观测空档或字节回填成本。**

## T32A 补充：1F1B局部区间所有权和gap lineage

[分账图](deadline_20260907/runs/T32A/diagnose/source_gap_bridge.svg)、[机读区间账本](deadline_20260907/runs/T32A/diagnose/source_interval_accounting_ledger.csv.gz)、[局部phase验证](deadline_20260907/runs/T32A/diagnose/source_phase_envelope_metrics.csv)、[全rank gap lineage](deadline_20260907/runs/T32A/diagnose/v67_v685_gap_lineage.csv.gz)、[时钟审计](deadline_20260907/runs/T32A/diagnose/clock_origin_audit.json)、[验收](deadline_20260907/runs/T32A/acceptance.json)。T32A把82,695个lane0语义区间拆成互斥的计算独占、通信独占、计算/通信重叠和未解释时间，逐条与wall精确守恒；59,751条正计算区间与v6.1物理槽总量一一精确吻合。source85/90中位数对source95/100的区间MAE为0.690063ms，F/B局部链MAE为15.344871/13.357146ms。

该账本也给出重复计时边界：B局部链结束平均晚于PP annotation 75.988631ms，必须与PP send的所有权联合解释。v54的lane0 gap可以从九轮区间精确复算；经内部stage池化和v67九轮shape后，74,624个gap在v685仍逐值不变，计算记账为0、未分类合计593,661.079ms。旧shape使用source60–100，包含95/100，且内部物理分解只覆盖各stage的lane0代表rank，因此不能把现有source replay包装为全rank留出验证，也没有据此生成目标候选。

T32A专用Snakemake4步PASS，61测试、3阶段manifest/211产物通过，r3/r4十九项科学产物字节精确。正式v685、拓扑、目标1F1B **11.340786%**、Step和MFU均保持不变。未验证边界为其余15 lanes的内部物理分账、只用85/90的全rank自由运行图、独立runtime状态、隐藏graph和跨stream因果。

可用于周报的两句结论：**1F1B局部分账已逐条闭合：82,695个区间无重复计时，59,751条物理计算观测精确连接；source85/90对95/100的F/B局部链MAE为15.345/13.357ms。现有v6.7/v6.8.5全rank gap使用了含验证轮次的source60–100参数，不能证明留出全图误差或支持224卡更新，正式v6.8.5的11.34%仍保留；其他lanes分账、独立runtime/隐藏graph及MFU分子仍未验证。**

## T33A 补充：全rank phase级source验证

[总览](deadline_20260907/runs/T33A/diagnose/source_phase_replay.svg)、[逐轮全局误差](deadline_20260907/runs/T33A/diagnose/source_phase_global_iteration_results.csv)、[分阶段误差](deadline_20260907/runs/T33A/diagnose/source_phase_metrics.csv)、[底层节点绑定](deadline_20260907/runs/T33A/diagnose/source_phase_cost_bindings.csv.gz)、[边表](deadline_20260907/runs/T33A/diagnose/source_phase_edges.csv.gz)、[验收](deadline_20260907/runs/T33A/acceptance.json)。代码导出的非交错1F1B图有19,714节点、26,752边，覆盖256 rank和每轮2,048个F/B phase。source85/90中位数在source95/100的全局1F1B MAPE为 **0.332811%**、MAE74.057ms；recent90全量/仅phase/仅runtime以及phase上界均未改善。

中位图逐phase duration MAE为warmup 15.854377ms、steady 16.692988ms、cooldown 15.478955ms。seal后观察替换的两轮平均actual−prediction 34.719727ms由phase wall +21.204959ms、PP +13.427048ms和本地runtime +0.087719ms构成；完整替换0ns仅证明可表示性。该独立图不替换v684正式锁，也不支持新的target候选。64测试、3阶段manifest/214产物通过，r1/r2十九项科学输出字节精确；目标1F1B **11.340786%**、Step和MFU均保持正式v685结果。

可用于周报的两句结论：**已用source85/90建立覆盖全256 rank的代码语义phase图，在已暴露的source95/100开发验证上取得0.332811%全局1F1B MAPE，四个recency/上界消融均未改善。剩余源误差主要随phase wall在两轮间换号，本地runtime平均贡献不足0.1ms；该结果尚不能解释PP16/MB4到PP14/MB3后预测包络过度收缩，未更新224卡、Step或MFU，正式v6.8.5仍保留。**

## T34A 补充：静态调度收缩与目标缺口

[四场景图](deadline_20260907/runs/T34A/diagnose/schedule_contraction.svg)、[机读场景结果](deadline_20260907/runs/T34A/diagnose/schedule_scenario_summary.csv)、[两因素Shapley](deadline_20260907/runs/T34A/diagnose/schedule_contraction_shapley.csv)、[关键路径节点/边账本](deadline_20260907/runs/T34A/diagnose/schedule_critical_path_ledger.csv.gz)、[目标逐迭代结果](deadline_20260907/runs/T34A/diagnose/target_development_iteration_results.csv)、[目标阶段误差](deadline_20260907/runs/T34A/diagnose/target_development_phase_metrics.csv)、[验收](deadline_20260907/runs/T34A/acceptance.json)。固定T33A源中位参数后，PP16/MB4→PP14/MB3使预测1F1B从22,182.789ms降到18,253.343ms；3,929.446ms预测收缩中，PP深度贡献2,673.468ms，microbatch数贡献1,255.978ms。

同口径实际包络从source95/100均值22,217.509ms到目标四轮均值21,561.771ms只收缩655.738ms，静态图多收缩3,273.708ms，实际仅为预测的16.688%。这项差额没有源侧可准入的运行时状态代理，故没有转成倍率、残差或等待成本。目标phase-transfer 1F1B MAPE为 **15.340746%**，差于正式v685的 **11.340786%**；Profiler、训练Step和MFU相对MAPE为13.984092/14.101418/16.418413%，也差于正式10.369528/10.718041/12.006689%。其他阶段没有生成新预测，不能作无退化声明。

目标逐phase平均MAE/偏差为99.427/−96.428ms；warmup、steady、cooldown偏差分别为−58.612/−84.418/−138.247ms。区段依赖的慢化说明统一缩放即使能对齐总量，也会掩盖内部原因。66测试、3阶段manifest/222产物通过；r2/r3二十四项科学输出字节精确，正式拓扑和v685均保留。

可用于周报的两句结论：**固定源成本的调度图预测PP16/MB4到PP14/MB3会缩短3.929秒，但224卡开发观测只缩短0.656秒，静态调度解释了错误方向中的一小部分，不能据此拟合目标倍率。phase-transfer的224卡1F1B MAPE为15.34%，仍差于正式v6.8.5的11.34%；运行时就绪代理、区段相关慢化、其他阶段和MFU分子仍未独立验证。**

## T35A 补充：在线方向前缀解释运行中慢化

[总览图](deadline_20260907/runs/T35A/diagnose/online_prefix_nowcast.svg)、[冷启动能力](deadline_20260907/runs/T35A/diagnose/cold_start_runtime_capability_audit.json)、[source同口径结果](deadline_20260907/runs/T35A/diagnose/source_online_metrics.csv)、[target逐迭代结果](deadline_20260907/runs/T35A/diagnose/target_step_iteration_results.csv)、[Step/MFU汇总](deadline_20260907/runs/T35A/diagnose/target_step_metrics.csv)、[前缀特征](deadline_20260907/runs/T35A/diagnose/target_online_prefix_features.csv.gz)、[前缀倍率与域检查](deadline_20260907/runs/T35A/diagnose/target_online_factors.csv)、[可用时点](deadline_20260907/runs/T35A/diagnose/target_online_prefix_cutoffs.csv)、[F/B贡献](deadline_20260907/runs/T35A/diagnose/target_online_direction_shapley.csv)、[关键路径](deadline_20260907/runs/T35A/diagnose/target_online_critical_path_ledger.csv.gz)、[底层节点/边](deadline_20260907/runs/T35A/diagnose/target_online_nodes.csv.gz)、[验收](deadline_20260907/runs/T35A/acceptance.json)。现有冷启动合同明确缺失动态router token矩阵和独立runtime readiness profile；静态stage/workload只有14行、3种profile和0个迭代运行时字段，不能从执行前输入识别T34A的3.274秒缺口。

因此T35A另建在线口径：当全stage的microbatch0 F和后半PP stage的microbatch0 B完成后，计算观测phase wall相对source85/90成本的F/B方向比例，经20% winsor后分别乘回现有phase wall。该操作没有新增成本项，phase wall内部计算、通信、重叠和未解释runtime仍只计一次；本地runtime、PP readiness及全部边保持不变。source95/100已在开发中看过，主候选把1F1B MAPE从0.332811%降至0.316924%，phase MAE从15.827028ms降至14.373229ms，构成开发确认而非独立验证。

source算法和候选先seal，随后目标只投影1,344个显式prefix feature；非前缀phase全部扰动不改变目标预测。目标四轮在线1F1B为21,103.875/21,108.945/21,074.767/21,429.155ms，APE为1.352707/1.948060/2.255827/1.540070%，MAPE **1.774166%**，平均低估382.585ms。对应Profiler、训练Step、MFU相对MAPE为1.722853/2.625241/2.696833%，相对正式v685的10.369528/10.718041/12.006689%均降低；entry/tail/outer预测逐值未改，改善只来自1F1B在线更新。

模型内F/B Shapley平均贡献约+0.780/+2.146秒，表明224卡前缀观测到的B方向慢化是主要修正来源；目标phase MAE从冷启动99.427459ms降到55.846983ms，偏差从−96.427948ms降到−4.611534ms。限制同样明确：主候选平均在1F1B完成 **57.845%** 后才可用，F倍率1.133–1.168、B倍率1.173–1.212全部超出source85/90范围。它是已暴露目标数据上的在线开发nowcast，不是冷启动升级、盲测或跨规模独立验证，正式v685仍保留。

可用于周报的两句结论：**在不改1F1B拓扑、PP依赖或重复加时的前提下，方向化在线前缀把224卡四轮开发1F1B MAPE从正式冷启动的11.34%降到1.77%，并将phase偏差从−96.43ms压到−4.61ms，主要修正来自B方向。该结果平均到1F1B完成57.8%后才可用，且目标倍率全部超出source85/90范围；冷启动仍缺router token和独立runtime readiness输入，正式v6.8.5未替换。**

## T36A 补充：更早在线前缀的可靠性收缩

[图](deadline_20260907/runs/T36A/diagnose/early_prefix_reliability.svg)、[source可靠性拟合](deadline_20260907/runs/T36A/diagnose/source_reliability_fit.csv)、[source方法比较](deadline_20260907/runs/T36A/diagnose/source_early_metrics.csv)、[target逐迭代](deadline_20260907/runs/T36A/diagnose/target_step_iteration_results.csv)、[Step/MFU](deadline_20260907/runs/T36A/diagnose/target_step_metrics.csv)、[倍率与域检查](deadline_20260907/runs/T36A/diagnose/target_early_factors.csv)、[source-only区间覆盖](deadline_20260907/runs/T36A/diagnose/target_early_interval_results.csv)、[可用时点](deadline_20260907/runs/T36A/diagnose/target_early_prefix_cutoffs.csv)、[底层节点/边](deadline_20260907/runs/T36A/diagnose/target_early_nodes.csv.gz)、[验收](deadline_20260907/runs/T36A/acceptance.json)。候选用全stage首F和末端4个stage首B；source85/90的raw图响应通过原点最小二乘得到0.611345可靠性系数，两次单轮值0.561094/0.661596仅作为敏感性区间。

source95/100开发确认中，可靠性候选将1F1B MAPE从0.332811%降至0.312367%，phase MAE从15.827028ms降至14.698333ms。接入目标早期前缀后，四轮预测为20,143.592/20,165.717/19,526.963/20,357.892ms，逐轮APE 5.841425/6.329396/9.434500/6.462172%，MAPE **7.016873%**；Profiler、训练Step、MFU相对MAPE为6.459935/7.061302/7.612662%。前缀平均在42.802%时齐备，仍有57.198%的1F1B待运行。

提前的代价由区间核验直接暴露：source系数区间对目标实际1F1B覆盖0/4，最近距离仍为1.094–1.929秒；目标raw F/B倍率全部超出source85/90范围。未收缩消融在目标开发上达到1.725071%，但source95/100 MAPE为0.345547%，比控制差，违反事前source门，不能因目标结果改选。正式拓扑、v685、entry/tail/outer均未改。

可用于周报的两句结论：**source可靠性收缩让在线1F1B预测在完成42.8%时即可给出，224卡开发MAPE为7.02%，优于正式冷启动11.34%但弱于较晚T35前缀的1.77%。source-only系数区间对目标覆盖0/4；目标最优的未收缩消融因source退化被拒绝，说明跨规模运行时状态仍不可由source校准，正式v6.8.5保持。**

## T37A 补充：剩余时间才是在线预测的直接口径

[提前程度—误差图](deadline_20260907/runs/T37A/diagnose/remaining_time_pareto.svg)、[逐迭代结果](deadline_20260907/runs/T37A/diagnose/remaining_time_iteration_results.csv)、[方法汇总](deadline_20260907/runs/T37A/diagnose/remaining_time_metrics.csv)、[区间覆盖](deadline_20260907/runs/T37A/diagnose/remaining_time_interval_coverage.csv)、[验收](deadline_20260907/runs/T37A/acceptance.json)。在同一sealed总时长预测中减去前缀已耗时后，绝对误差不变，但分母变成真正需要预测的剩余时间。T35逐轮剩余时间APE为3.279252/4.524618/5.341961/3.659239%，平均4.201267%；它在平均完成57.844807%后可用，平均仍可预报9,089.677ms。

T36主候选虽然更早，在完成42.801638%、剩余12,333.264ms时即可输出，但剩余时间MAPE为12.263149%，高于0%时可用的v685 11.340786%，所以在可选集合中被严格支配。T36 raw为3.010076%，但source门已失败；不能用目标结果恢复资格。T36区间在source拟合轮覆盖2/2，在source95/100和target分别为0/2、0/4，证明其不确定性没有跨迭代或跨规模覆盖。此审计没有生成预测，也没有改变Step/MFU、图结构、等待或正式v685。

可用于周报的两句结论：**按真实剩余时间重新计分后，T35在线预测在尚余约9.09秒时MAPE为4.20%，与v685冷启动共同构成可选Pareto前沿。更早的T36主候选在尚余约12.33秒时MAPE为12.26%，被v685支配；目标上更好的raw消融因source门失败仍不可选，正式v6.8.5保持。**

## T38A 补充：一轮初始化后的目标同域状态

[总览与分解](deadline_20260907/runs/T38A/diagnose/causal_lagged_update.svg)、[因果预测链](deadline_20260907/runs/T38A/diagnose/causal_prediction_ledger.csv)、[扰动核验](deadline_20260907/runs/T38A/diagnose/causality_mutation_audit.csv)、[逐迭代结果](deadline_20260907/runs/T38A/diagnose/target_iteration_results.csv)、[Step/MFU汇总](deadline_20260907/runs/T38A/diagnose/target_metrics.csv)、[阶段回归](deadline_20260907/runs/T38A/diagnose/target_phase_ledger.csv)、[预测构成](deadline_20260907/runs/T38A/diagnose/prediction_decomposition.csv)、[有限历史区间](deadline_20260907/runs/T38A/diagnose/causal_interval_results.csv)、[验收](deadline_20260907/runs/T38A/acceptance.json)。规则在目标评分前独立登记：85轮只初始化；当前T35前缀齐备后，90/95/100轮分别使用85/90/95轮已完成时的T35 base残差，且不递归使用修正后残差。当前或未来真值扰动不改变当前预测，前驱扰动只改变对应后继，逐行依赖哈希链通过。

90–100三轮T38预测误差为−129.996/−66.997/+151.196ms，1F1B APE为0.603839/0.310732/0.694696%，MAPE **0.536422%**；同口径T35与v685为1.914652%和11.572438%。剩余时间MAPE为1.262981%，Profiler、训练Step、MFU相对MAPE为0.630637%、1.127495%、1.143873%。若把没有更新的85初始化也纳入，四轮1F1B/剩余时间MAPE为0.740494%/1.767049%。

解释账本将不同证据层分开：90–100平均由18,253.343ms phase-transfer基座、762.389ms F图修正、2,188.557ms B图修正和398.385ms跨轮残差状态组成，最终平均误差−15.266ms。B仍是图内主要修正；lag状态只表示目标局部时间连续性，不能据此归为计算、通信或等待。entry/tail/outer没有变化，未重复计时或修改拓扑。有限历史区间只在95/100可用且覆盖1/2；有效walk-forward只有三轮，结果不能代表冷启动、source迁移、独立盲测或长期稳定性。

可用于周报的两句结论：**在T35图内F/B前缀修正之上，使用紧邻已完成目标轮残差的因果状态更新，把224卡90–100三轮开发1F1B MAPE从1.91%降到0.54%，剩余时间MAPE降到1.26%，训练Step和MFU相对误差约1.13%和1.14%。该方法需要目标85轮初始化且只有三轮walk-forward，区间覆盖仅1/2；它是同域在线开发方法，正式冷启动v6.8.5和拓扑锁均未替换。**

## T39A 补充：三轮无法选出唯一状态滤波器

[稳健性图](deadline_20260907/runs/T39A/diagnose/state_robustness.svg)、[状态依赖](deadline_20260907/runs/T39A/diagnose/state_prediction_ledger.csv)、[逐轮评分](deadline_20260907/runs/T39A/diagnose/state_iteration_results.csv)、[同口径汇总](deadline_20260907/runs/T39A/diagnose/state_robustness_metrics.csv)、[500ms冲击响应](deadline_20260907/runs/T39A/diagnose/synthetic_shock_response.csv)、[验收](deadline_20260907/runs/T39A/acceptance.json)。固定T38以后再做的消融显示，零更新、lag-one、延迟两轮、扩展历史均值在90–100上的1F1B MAPE分别为1.914652%、0.536422%、1.082859%、0.502136%；对应剩余时间MAPE为4.508606%、1.262981%、2.535804%、1.180718%。lag-one三轮均降低绝对误差，说明时间状态信号存在；stale变差说明新鲜度有影响。

扩展历史均值事后仅比lag-one好0.034286pp，样本只有三次转换，不能据此完成方法选择，T38保持事前登记的lag-one。500ms合成冲击结果符合因果权重：lag-one只响应直接前驱，stale晚一轮响应，扩展均值按历史数稀释；这只是响应审计，没有给状态残差增加物理归因。r1暴露的3.64e−12ms差异来自CSV浮点往返，正式核验使用封存字段的1e−6ms精度。

可用于周报的两句结论：**固定T38后，lag-one在224卡三次walk-forward上逐轮优于零更新，1F1B MAPE为0.54%；延迟状态退化到1.08%，说明最近运行状态包含有效信息。扩展均值在同一三轮事后为0.50%，与lag-one只差0.034个百分点；样本不足以重选滤波器，T38仍保持开发候选而非正式冷启动模型。**

## T40A 补充：目标状态不能解释成跨域规律

[跨域比较](deadline_20260907/runs/T40A/diagnose/cross_domain_state.svg)、[逐转换结果](deadline_20260907/runs/T40A/diagnose/transition_state_results.csv)、[同口径汇总](deadline_20260907/runs/T40A/diagnose/domain_state_metrics.csv)、[状态误差容限](deadline_20260907/runs/T40A/diagnose/state_noise_tolerance.csv)、[验收](deadline_20260907/runs/T40A/acceptance.json)。T38固定规则不作任何调节地应用到T35 source在线序列后，90/95/100只在95改善，1F1B MAPE从0.298320%升到0.517489%，剩余时间MAPE从0.658998%升到1.147096%。同一规则在target三轮从1.914652%降到0.536422%，剩余时间从4.508606%降到1.262981%。

机制差异与误差一致：target三次转换的基础残差都同号持续，source仅95一次同号，lag-one理论改善条件也恰好只在这些同号且幅值合适的转换成立。target每轮的状态测量误差严格改善区间均包含零，最小对称break-even半径183.990ms；但n=3时即使全改善，精确单侧sign-test也只有p=0.125。source的1/3改善对应p=0.875。由此应把T38限定为目标同域、已初始化的在线状态，不作source迁移或普适运行时规律声称。

可用于周报的两句结论：**固定lag-one在224卡三轮把1F1B MAPE从1.91%降到0.54%，但在同构source序列仅改善1/3并把MAPE从0.30%恶化到0.52%，证据只支持目标局部状态。三轮全改善的精确单侧p值仍为0.125，最小状态误差容限约184ms；在新连续目标轮次到来前应冻结方法选择，正式v6.8.5保持。**

## T41A 补充：按输入可用性选择运行模式

[运行决策图](deadline_20260907/runs/T41A/diagnose/operational_decision_flow.svg)、[模式与误差](deadline_20260907/runs/T41A/diagnose/operational_mode_decision.csv)、[解释层](deadline_20260907/runs/T41A/diagnose/explanation_layers.csv)、[复现索引](deadline_20260907/runs/T41A/diagnose/reproduction_index.csv)、[停止/恢复合同](deadline_20260907/runs/T41A/diagnose/stop_resume_contract.json)、[验收](deadline_20260907/runs/T41A/acceptance.json)。T41没有重新拟合或评分，而是逐项核验T35–T40的acceptance和diagnose manifest后固定三层使用边界：运行前只有v685可作为正式冷启动；当轮全stage首F和后半stage首B完成后，可报告T35同轮开发nowcast；若紧邻前一目标轮也已完成，可另报告T38目标初始化后的顺序开发分析。T36和T39只保留拒绝或消融身份。

这三层解释不能混用。锁定调度图解释warmup/steady/cooldown、F/B、通信依赖和图内等待；T35用当轮方向前缀修正phase wall，其中B贡献更大；T38只用前一轮T35 base残差表达目标局部时间持续性，不能命名为物理计算、通信或等待。entry/tail/outer保持冻结，Step只随1F1B变化，MFU沿用原分子。T38三轮的1F1B、剩余时间、训练Step、MFU相对MAPE为0.536422%、1.262981%、1.127495%、1.143873%，但需要目标初始化、只有三次转换、source迁移退化，不能替代正式v685。

83项测试、3个Snakemake阶段、225个manifest产物、56个冻结输入和0个raw输入通过；r3/r4的6项科学输出逐字节一致。状态方法搜索在至少2个设计时未见的连续新目标轮次到来前冻结；若要恢复冷启动研究，必须同时具备动态router token矩阵和独立runtime readiness profile。T41未改正式拓扑、边、等待、成本或参数。

可用于周报的两句结论：**224卡1F1B现已形成按可用输入分层的预测口径：正式冷启动v6.8.5为11.34%，T35同轮开发nowcast为1.77%，T38在一轮目标初始化后的90–100顺序开发MAPE为0.54%，但三者不能互相替代。T38只由三次目标转换支持且source迁移退化，至少需要2个设计时未见的连续目标轮次才能恢复验证；动态router token、独立runtime readiness和MFU分子仍未验证，正式拓扑与v6.8.5保持。**

## T42A 补充：当前证据包可复核

[完整性合同](deadline_20260907/runs/T42A/diagnose/final_integrity_contract.json)、[T35–T41阶段链](deadline_20260907/runs/T42A/diagnose/evidence_chain_audit.csv)、[783条文档链接](deadline_20260907/runs/T42A/diagnose/documentation_link_audit.csv)、[复现索引重算](deadline_20260907/runs/T42A/diagnose/reproduction_recompute.csv)、[17类交付物](deadline_20260907/runs/T42A/diagnose/deliverable_inventory.csv)、[验收](deadline_20260907/runs/T42A/acceptance.json)。七个正式run链接及其manifest、acceptance、plan、spec和pipeline command哈希全部匹配；T41的六条Snakemake复现记录独立重算一致。总览图、局部图、节点/边表、在线与因果逐轮结果、分账、模式表和复现入口均在冻结清单内。

T42只是只读审计，没有产生新的误差结果。正式口径仍是v685冷启动11.340786%；T35四轮同轮开发nowcast为1.774166%，T38在目标85初始化后的90–100顺序开发为0.536422%。后两者使用运行中或既往目标状态，不能包装成冷启动或独立盲测。85项测试、3个Snakemake阶段和227个清单产物通过，r1/r2六项审计输出逐字节一致；没有参数、边、等待、拓扑或其他阶段预测变化。

可用于周报的两句结论：**W37 1F1B 已形成可复核证据包：T35–T41七条阶段链、783个既有文档链接、17类图表与交付物及六条Snakemake复现记录全部通过哈希审计，正式/同轮/目标初始化三种口径清楚分开。正式冷启动仍为v6.8.5的11.34%；0.54%的T38结果仅覆盖已暴露目标序列的三次转换，需至少2个新连续目标轮次才能恢复验证，router token、独立runtime readiness和MFU分子仍未验证。**

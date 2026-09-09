# DAG MFU 多版本依赖与 Git 管理

## 目的

这套流程把 v6.0→v6.8.5 预测主线和 v6.9 候选研究门槛声明成真实文件依赖。v6.8 的 ENTRY 固定为256卡Trace iteration 85的1252.168507ms；v6.8.1修正反向触发，v6.8.2细化PP梯度墙钟，v6.8.3去掉B→B重复观测耗时，v6.8.4依据静态1F1B和PP API顺序补齐阻塞发送依赖并去掉被替代的F→F/B→F观测耗时；v6.8.5保持v6.8.4依赖拓扑不变，只用256卡自身波动选择85–100稳态窗口并重估已有成本参数。224卡仅在预测封存后进入开发评估。配置、生成代码、封存文件、评估入口或结果锚点变化时，Snakemake 能发现需要重跑的审计；它不在审计过程中静默改写已封存版本。

版本定义唯一入口是 `workflow/config/dag_mfu_versions.toml`。已实现版本必须声明版本号、父版本、用途分支、冻结配置、构建脚本、结果目录和结果锚点；候选版本则单独登记设计、runtime contract、候选图和状态。生成的 `version_inventory.json` 记录这些文件的绝对路径和 SHA256；当前 Git 状态由独立发布门禁记录。

`version_lock.json` 使用“逻辑角色 + 文件 SHA256”生成稳定、可迁移的内容指纹：不把绝对路径、生成时间、分支、提交或 dirty 状态写入内容锁。`git_release_gate.json` 由单独规则判断 Git 发布就绪度。这样刷新 Git 审计只会更新门禁、迁移清单与总状态，不会让 224/256 科学分析级联重跑。当前版本文件虽完整，但大量源码/config尚未跟踪且部分版本缺少根目录小型 artifact manifest，因此尚不满足发布门禁，也没有执行提交或推送。

## 运行

只查看执行计划：

```bash
.snakemake-venv/bin/snakemake \
  --directory . \
  --snakefile workflow/Snakefile \
  -n -p dag_mfu_autoresearch --cores 2
```

执行增量流程：

```bash
.snakemake-venv/bin/snakemake \
  --directory . \
  --snakefile workflow/Snakefile \
  dag_mfu_autoresearch --cores 2
```

该目标是 opt-in，不属于原有 `rule all`，不会触发 194 GB 原始 Trace 的全量重扫，也不会运行 ns-3。

优化器尾部研究进一步把七个有序 collective 完成边界拆成“发起前空档、rank 尚未到齐、最后 rank 到齐后的完成尾部”。该规则同时读取两 case，但只做带 split 标签的事后诊断，不更新参数；网络 FCT 与 DAG release 的职责边界因此可以单独审计。

AG completion adapter 则采用物理隔离的两进程：拟合进程只读取256卡60–80和封存OISA lookup，85–100用于源侧回归；目标 evaluator 才读取224 Trace。它用于检验“孤立OISA service + source软件/仿真修正”能否迁移，不会把224结果写回参数。

## 当前版本线

```text
v6.0 operator IR
  -> v6.1 kernel calibration
  -> v6.2 cost binding
  -> v6.3 ordered DP/EDP tail
  -> v6.3.1 AG round barrier
  -> v6.4a F/B handoff
  -> v6.5 global layer transfer
  -> v6.6 optimizer stream
  -> v6.7 microbatch runtime shape
  -> v6.8 source256-frozen profiler ENTRY
  -> v6.8.1 causal backward trigger
  -> v6.8.2 stage/lane/microbatch PP gradient
  -> v6.8.3 frozen topology + causal B→B program order
  -> v6.8.4 code-derived blocking PP program order
  -> v6.8.5 source256 steady-window cost recalibration [development predictive head]
       \..> v6.9 runtime readiness + rank release [candidate only]

v6.9 已加入版本图的候选区，但 `release_eligible=false`，没有 tag，也没有精度声明。它只冻结候选接口和发布门槛：算子成本下沉、依赖等待与真实软件开销分离、rank release 与 FCT service 分离、AICB/workload token 输入以及 host/placement 参数。候选内容指纹只覆盖设计JSON、runtime contract与候选图；带时间戳的provenance不进入稳定内容指纹。
```

v6.8.5预测进程只使用冻结的v6.8.4图与256卡时序参数，不能用224卡时序回填；224卡评估仍需与预测封存物理分离。v6.8.4相对v6.8.3只新增624条`blocking_forward_send`和416条`blocking_backward_send`，这些边在256源图中已经存在，并由PP API调用顺序确认；父图边删除数为0。F→F/B→F完整观测间隔被零成本程序顺序取代。v6.8.5不再改变依赖边，只重新估计已有的计算槽、PP梯度、入口和源图残差；所有候选边仍必须匹配v6.8.4拓扑指纹。由于目标场景此前已经被分析，本次224结果只能称为development evaluation，不能误写成新的独立blind实验。

## Git 边界

当前仓库原本已有大量未跟踪的 v6 配置、脚本和结果，因此现阶段不能声称这些版本已经形成 clean Git lineage。权威快照暂时是：

1. `dag_mfu_versions.toml` 中的依赖关系；
2. `version_inventory.json/csv` 中逐文件 SHA256，以及独立 `git_release_gate.json` 中的 Git 状态；
3. 每版 `prediction_seal.json` 或声明的结果锚点；
4. 每项 autoresearch 的 provenance 和 manifest。

后续形成提交时只能使用精确 pathspec，不能使用 `git add -A`。建议把变更拆成三层提交：

1. workflow/manifest/tests/docs；
2. 逐版本 code + config + seal/evaluator contract；
3. 体积可控的报告、manifest、provenance，原始 Trace 和大型中间表不入 Git。

每次版本提交或 tag 描述必须写出：source/target case、iteration 数据边界、是否读取 target、结果锚点 SHA256，以及是否为预测主线。

## Autoresearch 数据边界

- v6.8.5是当前代码语义闭合后的开发预测头；224指标是已查看过的development evaluation，不是blind validation。
- 依赖边只能由静态1F1B schedule、PP消息方向、autograd和共享资源程序顺序生成。目标误差、Trace时间差和参数搜索均无权新增、删除或改向边；候选拓扑SHA256必须匹配v6.8.4锁。
- `dependency_topology_guard/`是fail-closed参数搜索门：它对候选edge表重新计算规范化SHA256；任何edge add/delete/direction/type/source变化都会返回非零退出码。若代码或schedule确实变化，必须新建语义版本并人工审阅新锁，不能把结构变更混入参数优化。
- 新算子参数实验固定用 source 256 iteration 60–80 拟合、85–100 做 source holdout。
- 224 trace 只在 evaluator/diagnostic 产物中读取；任何使用 224 payload、arrival、duration 或 host 差异的结果必须明确写 target-assisted。
- 不运行 ns-3：当前通信研究只审计 `rank arrival + service tail` 接口。OISA 后续只替换 service tail，不能重复加入 trace 等待。

当前研究得到的约束：v6.7 约 67.22% 算子节点自身零耗时，成本集中在物理 slot/scheduler；完整 F→B trace gap 在 source replay 中几乎被依赖遮住，不可作为可迁移 service；MoE token 条件化显著改善 source holdout；224 的 host/placement 异质性显著高于 256，且原始Trace表明新增差异主要是runtime就绪停顿，不是kernel活跃计算倍率。

`scenario_contract/` 把这些约束落实为机器可读接口：source256 校准场景和 target224 预测场景分别保存，后者只含静态模型、并行策略、层放置、rank 拓扑和 source 参数引用，明确禁止 target timing/payload。当前状态为 PARTIAL，因为正式预测还缺 target workload 的逐层/逐 EP 组 token 矩阵及独立 host/runtime 就绪停顿先验。

`reproducibility_audit/` 会逐项重算当前有效研究与版本管线中 manifest/provenance 的文件 SHA256；任何有效引用缺失、内容变更或 JSON 损坏都会使 Snakemake 目标失败。已被新模型明确替代的历史结果不删除，但必须在 `SUPERSEDED_RESEARCH_NAMESPACES` 中逐项声明替代原因和后继目录，审计会单列排除数量，不能静默跳过。该检查验证字节级证据链，不替代模型精度和数据防泄漏审计。

`structural_scaling/` 只计算静态任务量和流水依赖指标。它确认两个case的中间stage均为4层，224的中间stage因microbatch 4→3而不是承担更多静态工作；同时MB相对PP更少，使流水依赖等待更易暴露。该指标用于排除错误归因，不是完整1F1B耗时公式。

`lane_representativeness/` 审计 v6.1 只用每个PP stage的lane0做kernel映射是否足以代表16个rank。它在256 source上把60–80与85–100分开检查，并把224仅作为evaluator诊断；结果只能验证F/B整段时长的代表性，不能替代其他lane的kernel级映射。

`input_capability_audit/` 把下一版可用输入分成 predictor-safe、source calibration、evaluator-only 和 blocked 四类，并给出缺失的目标token矩阵与独立host/runtime就绪停顿先验的机器可读schema。MTLink/NIC只作为通信诊断，不能冒充运行时调度输入。

`target_aicb_static_workload/` 通过本地固定版本的 AICB adapter，从目标场景契约生成 `PP14/52层/MB3` 静态workload、stage profile和名义collective签名。该节点不读取224 Trace时钟，可作为图生成的predictor-safe静态输入；均匀路由假设不能替代逐层、逐microbatch、逐EP组的动态router token矩阵。

`aicb_static_shift/` 用相同adapter语义对齐source256与target224的8类MFU通信行为。它将AICB内部通用 `ALLGATHER` 名称按已验证的source contract归一化，只比较静态调用数和名义payload：EP每类调用数随 `MoE层数×MB` 变化，DP/Expert-DP总组数随PP stage数变化。它不含rank release、service FCT或时钟结论。

`target_fct_adapter/` 将PP14静态AICB基础调用按256 Trace校准的语义倍数展开，并按entry/interior/exit stage角色复用已封存的19项OISA service FCT，得到14×8的target适配表。该步骤不重跑ns-3；表中总service work不是墙钟时间，必须由DAG根据rank readiness和overlap决定暴露部分。

`v69_graph_input_bundle/` 把目标静态AICB、256源算子参数、source-only入口参数、PP/CP参数和14×8共享FCT网格装配为统一图输入契约。它采用fail-closed门禁：动态router token矩阵和独立host/runtime就绪停顿分布仍为空时，只允许审计结构与接口，不允许生成或发布v6.9预测；目标Trace只可在prediction seal之后进入evaluator。能力审计中的 `target_workload_token_matrix` 对应运行时字段 `dynamic_router_token_matrix`，二者是同一输入在能力层和执行层的名称。

`oisa_service_exposure/evaluator_only/` 在封存v6.7图上单独缩放DP/Expert-DP/EP service并重跑max-plus，量化service真正暴露到完成时间的响应曲线。用224平均误差求出的等效倍率只用于反证“FCT单独解释全部误差”，不得作为OISA或MFU校准系数。

`router_workload_progress/model/` 与 `evaluator_only/` 物理分离：前者只读256卡、用60–80拟合并在85–100留出验证；后者加载冻结参数后才读取224 payload做事后诊断。iteration编号不能作为可迁移物理参数，正式接口仍需数据集/sample游标、有效token数或router生成矩阵。

`source_proxy_stack/prediction/` 与 `evaluator_only/` 进一步物理分离组合实验：预测进程只读256卡Trace、冻结源侧入口参数、冻结workload-progress参数和不含目标真值的v6.7预测表，输出入口与payload趋势叠加的候选时钟；evaluator才读取224真值。组合精度只能说明两个缺失输入的联合解释能力，iteration仍是不可部署代理，因此不产生v6.9版本。

`source_proxy_residual_budget/evaluator_only/` 把组合后剩余误差严格分解到入口、F/B全局包络、F/B后最后一次DP all-reduce和退出尾部。该节点只定位下一版需要补齐的图边界与rank release输入；它读取224时钟，不允许回填模型参数。

`optimizer_collective_release_tail/`、`optimizer_ag_completion/` 和 `optimizer_ag_contention_placement/` 依次把优化器尾部拆为rank到达与到齐后完成，冻结256侧OISA/Trace完成适配，再用冻结service窗口检查224侧并发和stage/host放置。结果否定了“AG组越多就一定越慢”和“一个固定慢host即可解释”两种简单假设；目标残差与共享host重叠只有弱相关，因此这些节点只形成v6.9接口约束，不拟合224系数，也不发布新版本。

`optimizer_ag_rank_completion/` 再回到原始rank事件验证组内结束一致性。DP的早到rank会在collective内部驻留较久，但同组rank最终几乎同时结束；Expert-DP的两rank更是同时到达、同时结束，却在224整体出现更长完成尾部。因此DAG必须输出逐rank release，通信后端返回唯一group end；不能重复累加rank duration中的同步等待，也不能用单GPU慢点解释整组尾部。

`v67_optimizer_release_vector_audit/` 把上述接口要求反查到封存图：第一轮DP/Expert-DP AG仍有逐rank入边，第二轮在stage barrier或前一DP完成处被提前压成公共标量。标量max-plus仍能运行，但无法生成OISA的 `rank_release_offsets_ns`，也失去早到rank的部分网络推进。报告落盘下一候选版的机器可读release接口；状态保持PARTIAL，不把设计契约冒充已实现版本。

`dp_ag1_rank_release_adapter/` 进一步给出第二轮DP AG的只读反事实连边：现有逐rank `optimizer_post_ag0` 节点可以直接接入rank-release adapter，而且其最大结束时刻与原stage join逐组完全相同，所以接口修复不会改变v6.7的标量fallback。与此同时，图内隐藏release跨度远小于256 Trace，迁移到224后差距更大；因此该产物只证明“保留向量”的实现安全性，也明确把“逐rank成本不足”列为独立阻塞，不生成新模型版本。

`optimizer_ag2_release_chain/` 以rank恒等式继续拆分第二轮DP AG的到达：前一轮Expert-DP小组完成时刻加本rank的post-EDP本地工作。256卡中主要波动来自8个Expert-DP小组完成错位，v6.7却已把该错位同步为零；224卡又出现明显更大的post-EDP本地工作跨度。该节点把source结构缺失和target成本域偏移分开，明确两者都不能塞进同一个OISA FCT倍率。

`source_edp_group_elapsed_templates/` 完全不读取224文件，只用256卡60–80冻结全局、group slot、stage角色×slot和stage×slot四类静态elapsed模板，在85–100检查同一stage内8个Expert-DP组的完成跨度。静态模板即使单组elapsed相对误差较小，仍无法重建动态组间错位，因此在source holdout直接fail closed；这说明独立collective查表必须升级为联合并发/共享资源后端，不能等到224评估才发现问题。

`oisa_joint_collective_interface/` 只读固定OISA的rank-release与multiring/slice两个Git commit，核对请求类型、runner与CLI。两个版本都只能一次运行一个collective；multiring和slice改善单collective内部流水，但不能表达同stage八个Expert-DP组共享队列。产物给出联合batch的最小机器可读契约：一个时间原点、全局唯一flow ID、一次ns-3进程、按collective_id返回多个group end；本工程不修改OISA，也不把未实现接口标记为可用。

`source_edp_joint_batch_requests/` 在联合runner尚未实现时先准备完整source输入：从256卡60–100重建同stage八个Expert-DP AG1组，按共同时间原点落盘逐rank release、真实payload、1 channel和256 KiB slice；60–80校准与85–100留出标签固定。请求目录不含completion真值，evaluator-only目录单独保存评分答案，防止后端或参数选择读取留出结果。目录同时冻结了评估协议和可复用评估器：必须先只看60–80，再把OISA commit、二进制、拓扑、buffer、channel和slice写入冻结manifest，85–100评估缺少该manifest会直接拒绝运行；当前未生成任何ns-3结果或评分。

`source_post_edp_local_delay/` 把前序Expert-DP完成到DP AG2 rank到达之间的本地调度段独立出来。模型进程只读256卡，60–80通过留一iteration交叉验证冻结stage/lane模板，85–100做source留出；224卡只在 `evaluator_only/` 中回顾性检查，参数更新为0。结果表明source模板在source留出可用，但迁移后不能覆盖224卡新增的stage/host波动；两侧release-span评分都使用实测前驱Expert-DP completion作为component oracle，因此该节点是输入缺口诊断，不是新版本或端到端精度。

`post_edp_kernel_gap_attribution/` 进一步只读本地原始Profiler Trace，在iteration 60、80、85、100中抽取每个stage最早/最晚post-EDP rank。每段严格按GPU kernel区间并集与剩余空闲时间分解，并逐文件记录SHA256。结果显示224中间stage新增的rank错位几乎全部来自空闲/调度段，而不是optimizer kernel活跃时间；因此候选DAG必须把GPU计算成本和host/scheduler readiness分成不同节点，不能把差值塞进OISA FCT或计算倍率。该抽样用于跨度归属，不作为总体均值。

`fb_kernel_gap_attribution/` 将同一方法扩展到F/B：只比较256与224共有的microbatch 0–2，在四个稳定iteration内为每个stage×phase×microbatch选择最短/最长rank，并从原始Trace重建非通信GPU活跃区间、通信驻留kernel区间和无kernel可见时间。它直接回答F/B rank跨度是算子计算差还是等待差；通信驻留只作occupancy证据，不当作FCT。该节点读取224原始Trace，因此只属于evaluator诊断，不产生外推参数。

`source_runtime_readiness/` 把F/B原始Trace扫描扩大到256卡60–100全部rank：60–80只拟合无kernel可见时间的stage/phase/microbatch/rank位置先验，85–100做source holdout；输出中位数、p90、p99和长停顿分布。`evaluator_only/` 之后才读取224卡60–100，把目标kernel活跃并集作为component oracle评估冻结先验的迁移误差，参数更新为0。这一节点量化runtime readiness域偏移，不把target空洞回填模型，也不发布v6.9。

`runtime_readiness_covariates/` 只使用上述派生表与已存在的DeepEP组事件，比较runtime-gap与payload、arrival、service-tail、通信驻留、计算活跃的相关性，并检查最慢lane身份是否跨iteration和跨case稳定。该节点会读取224 evaluator数据，只用于判定未来输入应具有怎样的粒度；相关性和模态lane都不得回填成预测参数。

`runtime_gap_scale_decomposition/` 继续复用派生窗口，把无kernel可见时间严格拆为“每段小于10ms的短launch间隙总和”和“每段至少10ms的长runtime停顿总和”，并按stage角色与F/B分别比较256和224。该节点用于决定v6.9的runtime节点粒度：短间隙预算、长停顿随机过程及跨rank相关性必须分离；对应runtime输入schema已升级为v2。224结果仍是evaluator诊断，不产生目标参数。

`runtime_stall_topology/` 再把长停顿按 PP stage、stage内两台host及host内rank分层。它检查每个stage恰好由两台8卡host构成，并区分“两host共同”“单host分叉”“rank残差”三类空间结构。目标侧只用于诊断接口是否缺少层级相关性，绝不更新概率参数；因此v6.9不能再用独立rank采样或固定慢lane代替这一结构。

`runtime_stall_temporal/` 在相同派生状态上计算同host/stage的相邻microbatch、相邻iteration以及相邻PP stage转移概率，并同时报告基准发生率、条件差和phi相关。它用于判断runtime长停顿是否需要状态记忆；224条件概率始终留在evaluator诊断中，不写入预测参数。

`runtime_stall_threshold_sensitivity/` 扫描5/10/20/40/80 ms，避免空间/时间结论只成立于人为设定的10 ms。正式阈值用“256反向rank发生率最接近50%”的source-only规则选择；224只验证该阈值下是否仍存在域偏移。过低阈值饱和、过高阈值退化为零的限制会显式记录。

`source_hierarchical_stall_process/` 将接口落实为source-only概率分解：256卡60–80拟合stage共同事件、host条件/前态事件及rank比例残差，85–100留出；`evaluator_only/` 冻结同一参数后在224打分。条件分数使用观察到的父状态和前一状态，明确标作teacher-forced组件上限；端到端生成器仍必须自行滚动状态，且不能用224基础发生率回填。

同目录的 `free_running_source_holdout/` 与 `evaluator_only/free_running/` 使用8状态精确概率传播，自行滚动 `(stage共同, host0, host1)`，预测时不读观察父状态。它把“条件上有信息”与“端到端可生成”严格分开：若teacher-forced提升在自由滚动时消失，就说明还缺外部runtime状态，而不是需要继续增加事后条件。

`runtime_hardware_covariates/` 先用同一个窗口聚合器把256卡的MTLink/NIC原始计数转换为与224卡一致的iteration×host口径。当前iteration计数会被本次训练行为反向影响，所以只作相关性诊断；可运行的source-only探针严格使用同一host前一个已采样Profiler窗口，参数只在256卡60–80拟合、85–100留出，224卡60–100只评分。该采样间隔是Profiler标签相差5，并不等于相邻训练iteration；因此它最多是已有历史运行时的在线状态输入，不能冒充新场景首跑预测，也不能解除独立runtime telemetry门禁。

`runtime_sync_wait_attribution/` 读取两侧60–100全rank原始Profiler Trace，将F/B窗口内GPU无kernel区间与 `musaDeviceSynchronize`、`musaStreamSynchronize`、`musaEventSynchronize` 精确求交。它只做语义归因：同步API的实测持续时间是等待结果，不是可迁移service。候选DAG应把同步点表示为零/小API开销的max-plus join，其完成时间由计算、PP、CP、EP和优化器前驱自然产生；224卡仍只作evaluator-only诊断，参数更新为0。

`runtime_sync_structure/` 不再读取原始Trace，只消费上一步封存窗口。它用256卡60–80按归一化stage位置、F/B、microbatch位置和同步类型冻结调用存在率、调用数及首次/末次相对位置，85–100做source holdout，224卡60–100只做冻结模板的evaluator。该模块检验同步节点模板能否跨PP/MB迁移；它不复制同步等待时长，也不能单独判断每个join应连接哪条前驱。

`runtime_sync_delta_decomposition/` 在两侧相同的首/中/尾stage层数合同下，按60–100同标签iteration把单个F/B窗口严格拆成GPU活跃、device/stream/event同步空档、其他长runtime调用空档和无可见runtime等待。各项与窗口时长保持恒等加和；224只用于回顾性归因，任何分量差值都不得作为预测系数。

`runtime_sync_parent_mapping/` 扫描两侧60–100全rank Trace，并用External-id把三种runtime同步调用回连到父CPU算子。父算子节点模板只在256卡60–80冻结，85–100做source holdout，224卡60–100只作evaluator；实测同步duration仍是DAG等待结果，绝不能变成service常数。原始事件缺少External-id时显式记为`UNMAPPED`，不得误报为新算子。提取阶段有输入SHA约束的独立缓存检查点，报表聚合失败不会触发重复扫描。

`runtime_sync_semantic_order/` 在每个PP stage选定代表lane0，从256卡60–80冻结stage角色×F/B的有序同步join模板，以85–100做source holdout，再在224卡60–100只评分。全rank父算子计数由上游模块保证，本模块负责验证顺序：中间stage的FWD由9节点单元重复4次，BWD由13节点单元重复4次。模板能生成节点顺序，但不携带观察同步duration，也不证明所有lane等待时长相同。

`runtime_sync_predecessor_contract/` 把上述source-only语义节点与v6 operator IR交叉映射：六类Fused wrapper接到已有 `ep_*_service → ep_*_completion_sync`，host scalar/copy则标记为v6.9待补本地节点。统一规则为 `sync_end = max(全部必需前驱完成时钟) + API overhead`。256卡60–80中普通/重计算dispatch的stream同步并集中位数分别约59.5/152.1 ms，说明wrapper wall绝非纯网络FCT；这些时长只作诊断，不作为可迁移service常数。External-id仍不能证明device sync唯一等待的stream/kernel，所以该合同保持候选状态。

`training_code_sync_semantics/` 对236B目录中的Fused A2A和token dispatcher做只读代码语义核对并固定逐文件SHA：dispatch/combine显式携带event/current-stream join，token计数存在 `.sum().item()` host/device标量物化。代码证据只决定DAG边和节点类型，不提供等待时长；由于尚未找到并固定每份Trace的完整启动命令/参数，活跃代码路径仍标记为未完全确认。迁移到mfu-model时可携带审计产物，若要重新审计则需另外提供相同SHA的训练代码。

`runtime_sync_service_boundary/` 在每个stage的代表lane中用`musaGraphLaunch`把Fused wrapper拆为pre-service与post-launch两段。21,006个wrapper中，256留出和224 evaluator-only均为100%唯一launch与100%边界签名命中。256校准中普通FWD dispatch在launch前约61.4 ms（stream sync约59.5 ms），launch后device sync约1.15 ms；combine launch前约0.19 ms、launch后device sync约7.10 ms。由此确认dispatch大等待属于网络发起前的本地就绪，而非OISA FCT；实测时长仍不迁移。

`runtime_sync_fused_shift_budget/` 复用上述缓存观测，不再扫描Trace，把两侧稳定段60–100的每类Fused wrapper中位数按相同4层中间stage展开，并与完整F/B窗口的224−256差值对账。该回顾性预算显示Fused wrapper约覆盖FWD增长的36.8%，而BWD wrapper合计反而缩短约8.2 ms、与完整BWD增长141.3 ms方向相反；因此BWD外推缺口主要位于wrapper外计算包络、runtime readiness或PP调度，而不是DeepEP/OISA service。中位数求和不是严格加法恒等式，224仅作evaluator，参数更新为0。

`runtime_sync_fused_window_partition/` 使用每个wrapper相对F/B窗口的真实起止位置，在1,908个代表lane窗口上计算区间并集；每个窗口都严格满足 `phase = wrapper union + outside wrapper wall`，且wrapper之间没有可观测重叠。按稳定段均值，FWD增量约一半在wrapper内、一半在wrapper外；BWD只有约26.9%在wrapper内、73.1%在wrapper外。该严格窗口分区优先于上面的中位数求和预算，但仍只代表lane0且读取了224 evaluator时钟，不产生迁移参数。

`runtime_sync_postlaunch_completion/` 复用缓存观测，把launch后wrapper墙钟继续分成device/event/stream同步覆盖与其余墙钟。FWD增量约71.6%不在三类可见同步调用内；BWD增量约49.0%在device-sync completion join、47.9%在其余wrapper墙钟。device sync是等待前驱完成的join，不是固定API开销；OISA只能提供其中网络service的完成时钟，不能接管整个post-launch段。

`runtime_sync_postlaunch_wrapper_attribution/` 再把上述224−256差按Fused wrapper类型严格加回：FWD增量约70.9%来自FusedCombine，BWD最大单项同样是FusedCombine、约45.8%。它说明误差不是“所有通信统一慢一个系数”；网络完成、Combine/Dispatch完成join和wrapper软件路径必须分别拥有节点。该页是target evaluator-only归因，不产生目标参数。

`source_postlaunch_wrapper_shape/` 采用三进程物理隔离：`model/`只读取256卡60–80并用85–100选择策略，`prediction/`只读冻结参数与224不含时长的wrapper结构，`evaluator_only/`最后才读目标时钟。source留出选中的stage×wrapper形状WAPE约24.96%，冻结到224约33.11%，且相对统一phase×wrapper均值几乎无改善；这否定了仅靠PP/MB位置修补completion绝对尺度。

`runtime_sync_postlaunch_temporal_drift/` 证明60–100虽已避开启动warmup，却不是性能恒定区间。256卡BWD每5个采样iteration增长约9.79 ms，其中约86.6%的斜率落在device-sync完成join；224同方向但占比约57.4%。iteration编号不是物理变量，不能直接进入外推参数，候选接口需要真实前驱完成状态或独立runtime/hardware状态。

`source_postlaunch_hardware_probe/` 用前一个已采样窗口的MTLink/NIC状态做source-only在线探针：在256卡85–100留出上相对静态结构MAE改善约17.46%，但冻结到224后恶化约56.63%，且约60.3%的目标行至少一项选中特征越过256校准范围。它最多是同域、已有历史样本时的在线能力，不能冒充跨并行策略cold-start预测；v6.9合同据此明确区分 `cold_start_static` 与 `same_domain_online`。

`runtime_sync_outside_wrapper_partition/` 重新扫描代表lane的原始Trace，并用区间代数同时建立两条互斥墙钟恒等式：`phase = Fused wrapper + wrapper外GPU活跃 + wrapper外GPU空闲`，以及不受wrapper边界影响的 `phase = 通信独占 + 非通信独占 + 通信/非通信重叠 + GPU空闲`。1,908个窗口的恒等式误差为0。按完整F/B口径，224中间stage相对256的FWD/BWD增量主要是GPU空闲约43.17/97.34 ms与通信独占约22.01/46.40 ms；非通信独占仅约-1.78/+4.06 ms。该结果只作target evaluator边界审计，不产生目标参数；OISA只替换显式通信service，非通信算子和等待继续归DAG。

`source_phase_noncommunication_cost/` 使用独立的256-only进程扫描144份源Trace，按完整F/B窗口计算非通信GPU活跃并集，避免CPU Fused wrapper边界移动污染计算口径。60–80拟合、85–100选择后source留出WAPE约5.59%；冻结参数并只读取224静态AICB的PP14、3 microbatch和层放置时，target evaluator WAPE约3.64%（FWD/BWD约3.20%/3.81%）。这支持把非通信active成本作为候选冻结输入，但不代表完整MFU精度；通信、与计算重叠、runtime空闲、rank到达和outer clock仍单独建模。

`phase_kernel_family_shift/` 把完整F/B中的每类kernel暴露变化严格拆成“每窗调用次数”和“单次平均暴露时长”两项。中间stage最大正向变化是DeepEP与MCCL，主要来自单次暴露时间增长；FWD非通信族基本持平，BWD只有memory/gemm等小幅增加。算子事件会并发，因此族间暴露不可相加成墙钟；该页只用于判断下一版需要静态调用计数还是运行时成本输入。

`phase_idle_boundary_signatures/` 对270份代表lane Trace中的136万段GPU空闲分类其前后kernel类型，并逐类加回完整F/B空闲增量。新增空闲中“非通信→非通信”占FWD/BWD约51.1%/47.6%，“非通信→通信”占约30.2%/33.3%；“通信→非通信”不是主项。这些只是时间边界签名，不直接证明因果边，但排除了把全部空闲归入网络service的做法。

`precommunication_readiness/` 再把“非通信→通信”按后续通信族分开：中间stage的DeepEP发起前空闲从256到224，FWD约19.80→32.32 ms、BWD约39.26→71.37 ms，主要由device synchronize覆盖；MCCL发起前仅亚毫秒。候选DAG必须把DeepEP `pre_service_readiness_join` 与OISA `network_service_completion`分成两个节点，观察到的同步duration是join结果而非固定service。

`source_deepep_pre_service_readiness/` 把上述结论变成物理隔离的冻结迁移实验：拟合进程只读256卡GPU-idle gap，60–80拟合、85–100留出选策略；预测进程只读冻结参数与224静态PP14/MB3/52层scenario，最后由evaluator读取224真值。即使加入stage分箱、microbatch角色和层数，256留出WAPE仍为27.058%，224为51.905%，相对只按phase×层数的基线仅改善4.70%/4.57%，平均低估24.022 ms/窗。F/B的DeepEP发起次数倍率只有0.993/1.000，而单次等待倍率达到1.546/1.681；36个共同位置切片中27个增至1.25倍以上、5个反而降到0.8倍以下，不能用统一倍率表达。变化来自同步join等待结果而不是调用次数，静态位置模板不足以生成DeepEP发起前运行时状态，也不能用OISA service去补偿。

`runtime_sync_fused_partition_shape/` 将上述逐窗口分区映射到entry/early/middle/late/exit和first/middle/last语义位置，去除iteration整体漂移后比较PP与microbatch形状。两侧完整FWD/BWD的共同cell形状相关分别约0.962/0.975，说明source位置形状具有迁移价值；但224 BWD wrapper外最大增量位于early×first，绝对尺度明显变化。候选图应保留位置依赖，并从预测前workload/runtime输入生成尺度，不能用一个F/B均值或目标cell系数替代。

`source_fused_component_shape/` 将结论落实成物理隔离的source-only链：`model/`只读256卡60–80并在85–100选择phase/stage/microbatch形状策略；`prediction/`只读冻结参数和不含时长的224结构表；`evaluator_only/`最后才连接目标真实时长。source留出选中stage shape，WAPE由phase均值的14.54%降到5.25%；冻结到224后由24.31%降到17.19%。这证明位置结构有实际迁移收益，也证明尚缺独立的跨场景绝对尺度输入；该窗口分数不是完整iteration MFU精度。

`runtime_sync_fused_window_partition/` 还分别产出 `source_only_manifest.json`、`validation_safe/manifest.json` 和 `evaluator_only/ground_truth_manifest.json`。拟合、预测、评分三个进程只读取各自边界内的数据与manifest；上游总manifest不再进入source拟合，从文件依赖层面避免把目标真值路径带入校准证据。

同一evaluator的分组件结果显示，BWD pre-service WAPE约4.88%，而post-launch与wrapper外分别约33.82%和46.78%；FWD分别约8.19%、31.33%和41.69%。因此可复用的主要是发起前结构，下一研究优先级是service completion之后的等待与wrapper外计算/runtime成本尺度，而不是继续扩充pre-service位置参数。

`source_fused_component_factorial/` 把上述冻结的三项组件成本接入通用PP×microbatch 1F1B图，分别生成PP16/MB4、PP14/MB4、PP16/MB3和PP14/MB3，并对单组件+10%重跑max-plus。该实验只读256参数与静态结构：联合结构条件下路径缩短约16.88%，而source图内pre-launch成本最敏感。这个结果不能与224 evaluator的外推误差混为一谈：后者最大的尺度缺口在post-launch与wrapper外；前者回答“当前图里谁占关键路径”，后者回答“跨场景谁预测不准”。

`parameter_influence_ledger/` 把PP、microbatch、stage放置、F/B成本、Fused三段、rank到达、网络service/通信尾部和外层时钟放入同一证据表。每一行都分开记录source-only条件敏感性与224 evaluator-only诊断，并标注可识别性、预测输入准备度和模型动作；它只综合已有证据，不拟合目标参数。

`source_runtime_process/` 把该接口落成source-only随机过程基线：60–80通过留一iteration交叉验证选择stage/phase/microbatch语义粒度，分别拟合短间隙分布、长停顿概率与条件时长，并用exchangeable beta-binomial表达同cell跨rank相关性；85–100做source holdout。`evaluator_only/` 再冻结参数读取224派生窗口，只检查发生率、gap期望和rank-count区间覆盖，参数更新为0。

`runtime_process_features/` 进一步把可能的解释输入分级：静态stage/phase/MB语义、动态router payload、可由后端提供的service tail，以及内生rank-arrival。所有线性系数只在256卡60–80拟合、85–100留出；224 `evaluator_only/` 只比较冻结特征组的解释上限。payload只有来自预测前workload时才安全，observed arrival永远只是诊断上限，不会写回runtime参数。

`optimizer_tail_transfer/model/` 与 `evaluator_only/` 物理分离：前者只用256卡60–80冻结 `F/B结束 → global control两轮 → DP/Expert-DP AG1 → DP/Expert-DP AG2 → 最终DP AllReduce` 七段墙钟边界，并用85–100做源侧回归；后者才读取224卡。个别rank缺失一条事件时，按同PP stage启动时间簇恢复轮次；该节点用于判断尾部迁移失败发生在哪个完成边界，不把跨rank墙钟段误称为纯网络service。

`module_oracle_ceiling/evaluator_only/` 枚举v6.7四个可加Profiler时钟模块的16种oracle替换组合，用于评估“若某类输入完美，最多可消除多少开发集误差”。它读取224 evaluator时钟，不回填参数，不得当作预测精度或新版本。

`step_clock_decomposition/` 把Profiler时钟拆成入口、全局F/B包络和退出三项，验证256→224实测总时长接近是内部正负变化抵消；同时给出v6.7外推图时钟过度缩短的严格恒等分解。该产物是target-assisted evaluator诊断，不更新参数。

`critical_path_transfer_ledger/` 只比较封存的256 source replay和224 source-only预测DAG，严格重构raw graph缩短量，并把它分到账本类别。由于两张图使用不同成本schema，分类差异只用于定位转移问题，不作物理加速或因果结论。

`phase_cost_transfer/evaluator_only/` 对齐冻结预测与224 Trace的1344个rank/stage/F-B/microbatch单元，输出误差表和热图。它只定位成本迁移风险，所有目标时长保持evaluator-only，不回填模型。

`static_factorial_sensitivity/` 用2×2公式表分别改变PP/MB与层数/MB，只报告流水依赖压力和静态工作量方向。反事实没有执行DAG，数值不是时延、MFU或真实因果效应。

`parameter_sweep_readiness/` 审计封存v6.7的参数化边界：v6.7继承固定v6.6图并包含PP14/MB3、source256/MB4等假设，因此只改TOML不能形成可靠DAG反事实。报告冻结下一候选版需要的graph、placement、cost、service和factorial接口。

`schedule_factorial/` 是通用PP×MB的最小1F1B结构图工厂，以单位F/B节点生成四张无环DAG并重放关键路径。它只验证结构参数化，不含毫秒成本、通信、层级算子或MFU，也不作为新模型版本。

`source_envelope_factorial/` 在相同四张参数化图上冻结256卡60–100的16-rank F/B包络成本，形成不读取224时长的受控敏感性实验。它用于回答“只改变PP或microbatch时，抽象依赖图会把关键路径推向哪里”，但包络仍混合计算、通信和同步，不能当作因果估计或新版本精度。

`stage_mapping_sensitivity/` 固定保留source首尾stage，枚举从14个中间stage删除任意两个的91种PP14有序映射。它量化stage模板选择的不确定性，并将v6.7映射放入分布中；仍不读取224时长，也不替代逐层算子图生成。

`v67_local_sensitivity/` 在封存的327,745节点预测图上，逐类缩放compute、network service、microbatch scheduler、collective rank release和其他软件成本并重跑max-plus。基线必须严格复现seal；该实验只衡量当前图对输入误差的条件敏感性，不用224时长反推系数。

`phase_error_structure/evaluator_only/` 研究冻结F/B成本误差沿phase、PP stage、stage内两台host位置和lane的结构。它包含target-assisted MB0/1→MB2伪留出，只用于确定未来独立输入的粒度，不能提供部署系数。

`source_target_factor_transfer/evaluator_only/` 进一步检验上述结构的数值能否直接从256卡搬到224卡。结果显示，256卡残差系数在全局、phase、stage、host位置乃至逐cell粒度都没有改善224误差；目标端所需的host/放置因子必须来自预测前独立输入，不能复制源run中的位置残差。

## 迁移到 mfu-model

迁移不是直接复制 HTML。最小迁移单元包括：

- 版本 manifest 和父子关系；
- workflow 脚本与单元测试；
- 每版冻结 config、构建/封存/评估入口；
- 结果 manifest/provenance 和少量必要参数表；
- 对 fabric-data-analysis Trace 派生输入的显式 adapter contract。

迁移验收要求：新仓库解析后的版本拓扑顺序一致；所有被复制文件 SHA256 一致；未复制的大型输入由绝对来源、schema 和 hash 引用；source-only 与 target-assisted 分支仍物理分离。当前只准备此契约，不修改 `/home/zjb/Desktop/mfu-model`。

机器可读迁移候选清单为 `dag_mfu_version_pipeline/migration_bundle_manifest.json`，准备状态见 `MIGRATION_READINESS.md`。清单除代码/config/workflow 外，还固定阶段分区、source-only 非通信成本、kernel family、GPU idle 边界、DeepEP service 前就绪、参数影响账本以及 v6.9 候选契约的精简证据；原始 Trace 和大型逐事件表仅由 manifest/provenance 引用。清单生成不执行复制、提交或推送。清单内的可移植性门禁还会拒绝目标路径冲突、`..` 越界、仓库外复制源和硬编码当前 fabric-data-analysis 根目录；AICB、OISA、Trace 与论文等外部绝对路径单独列为迁移后重验依赖。

`migration_source_verification/` 会在当前源仓库逐项复核迁移清单里的代码、精简证据和源侧版本控制面历史快照，文件大小或 SHA256 任一不一致即失败。实际迁移后，在目标仓库执行 `python fabric_import/workflow/scripts/verify_dag_mfu_migration_snapshot.py --manifest <源migration_bundle_manifest.json> --mode target --target-root <mfu-model根目录> --output-dir <校验目录>`，可以用同一合同验证复制结果；校验器只读，不负责复制、提交或推送。之后应在目标仓库另行重生成inventory、version lock和Git gate：只比较版本拓扑、内容指纹、anchor/seal等稳定字段，路径、生成时间和目标Git状态预期不同，不能要求control plane整文件hash与源快照一致。

`extrapolation_bottleneck_map/` 把上述分散证据合成为一个严格可加和的墙钟预算：中间stage的224−256 FWD/BWD增量分别为62.905/136.623 ms/窗，其中GPU空闲为43.170/97.337 ms，通信独占为22.012/46.405 ms。空闲增量中非通信→非通信边界占51.1%/47.6%，发起通信前边界占30.2%/33.3%；后者的96.1%/99.1%发生在DeepEP前。该页面只决定候选节点优先级，不把224数值写入参数。

## 关键产物

- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_mfu_version_pipeline/version_inventory.json`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_mfu_version_pipeline/version_graph.mmd`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/AUTORESEARCH_STATUS.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/identifiability/IDENTIFIABILITY_AUDIT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/v69_candidate_design/V69_CANDIDATE_DESIGN.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/v69_candidate_design/v69_runtime_readiness_contract.json`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/cost_assignment_audit/COST_ASSIGNMENT_AUDIT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_token_conditioned_operator_cost/SOURCE_TOKEN_CONDITIONED_OPERATOR_COST.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/host_placement_heterogeneity/HOST_PLACEMENT_HETEROGENEITY.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/scenario_contract/SCENARIO_CONTRACT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/reproducibility_audit/REPRODUCIBILITY_AUDIT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/structural_scaling/STRUCTURAL_SCALING.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/lane_representativeness/LANE_REPRESENTATIVENESS.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/input_capability_audit/INPUT_CAPABILITY_AUDIT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/target_aicb_static_workload/TARGET_AICB_STATIC_WORKLOAD.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/aicb_static_shift/AICB_STATIC_SHIFT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/target_fct_adapter/TARGET224_SHARED_FCT_ADAPTER.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/v69_graph_input_bundle/V69_GRAPH_INPUT_BUNDLE.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/oisa_service_exposure/evaluator_only/oisa_service_exposure.html`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/router_workload_progress/evaluator_only/ROUTER_WORKLOAD_PROGRESS.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_proxy_stack/evaluator_only/SOURCE_PROXY_STACK_EVALUATION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_proxy_residual_budget/evaluator_only/SOURCE_PROXY_RESIDUAL_BUDGET.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/optimizer_ag_contention_placement/AG_CONTENTION_PLACEMENT_DIAGNOSTIC.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/optimizer_ag_rank_completion/AG_RANK_COMPLETION_DIAGNOSTIC.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/v67_optimizer_release_vector_audit/V67_OPTIMIZER_RELEASE_VECTOR_AUDIT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/dp_ag1_rank_release_adapter/DP_AG1_RANK_RELEASE_ADAPTER.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/optimizer_ag2_release_chain/OPTIMIZER_AG2_RELEASE_CHAIN.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_edp_group_elapsed_templates/SOURCE_EDP_GROUP_ELAPSED_TEMPLATES.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/oisa_joint_collective_interface/OISA_JOINT_COLLECTIVE_INTERFACE_AUDIT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_edp_joint_batch_requests/SOURCE_EDP_JOINT_BATCH_REQUESTS.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_post_edp_local_delay/SOURCE_POST_EDP_LOCAL_DELAY_MODEL.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_post_edp_local_delay/evaluator_only/TARGET_POST_EDP_LOCAL_DELAY_EVALUATION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/post_edp_kernel_gap_attribution/POST_EDP_KERNEL_GAP_ATTRIBUTION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/fb_kernel_gap_attribution/FB_KERNEL_GAP_ATTRIBUTION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/module_oracle_ceiling/evaluator_only/MODULE_ORACLE_CEILING.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/step_clock_decomposition/STEP_CLOCK_DECOMPOSITION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/critical_path_transfer_ledger/CRITICAL_PATH_TRANSFER_LEDGER.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/phase_cost_transfer/evaluator_only/PHASE_COST_TRANSFER_EVALUATION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/static_factorial_sensitivity/STATIC_FACTORIAL_SENSITIVITY.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/parameter_sweep_readiness/PARAMETER_SWEEP_READINESS.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/schedule_factorial/SCHEDULE_FACTORIAL.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_envelope_factorial/source_envelope_factorial.html`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/stage_mapping_sensitivity/stage_mapping_sensitivity.html`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/v67_local_sensitivity/v67_local_sensitivity.html`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/phase_error_structure/evaluator_only/PHASE_ERROR_STRUCTURE.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_target_factor_transfer/evaluator_only/SOURCE_TARGET_FACTOR_TRANSFER.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_hardware_covariates/RUNTIME_HARDWARE_COVARIATES.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_wait_attribution/RUNTIME_SYNC_WAIT_ATTRIBUTION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_structure/RUNTIME_SYNC_STRUCTURE.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_delta_decomposition/RUNTIME_SYNC_DELTA_DECOMPOSITION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_parent_mapping/RUNTIME_SYNC_PARENT_MAPPING.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_semantic_order/RUNTIME_SYNC_SEMANTIC_ORDER.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_predecessor_contract/RUNTIME_SYNC_PREDECESSOR_CONTRACT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/training_code_sync_semantics/TRAINING_CODE_SYNC_SEMANTICS.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_service_boundary/RUNTIME_SYNC_SERVICE_BOUNDARY.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_fused_shift_budget/RUNTIME_SYNC_FUSED_SHIFT_BUDGET.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_fused_window_partition/RUNTIME_SYNC_FUSED_WINDOW_PARTITION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_postlaunch_completion/RUNTIME_SYNC_POSTLAUNCH_COMPLETION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_postlaunch_wrapper_attribution/RUNTIME_SYNC_POSTLAUNCH_WRAPPER_ATTRIBUTION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_postlaunch_wrapper_shape/evaluator_only/TARGET_POSTLAUNCH_WRAPPER_SHAPE_EVALUATION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_postlaunch_temporal_drift/RUNTIME_SYNC_POSTLAUNCH_TEMPORAL_DRIFT.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_postlaunch_hardware_probe/evaluator_only/TARGET_POSTLAUNCH_HARDWARE_PROBE_EVALUATION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_outside_wrapper_partition/RUNTIME_SYNC_OUTSIDE_WRAPPER_PARTITION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_phase_noncommunication_cost/evaluator_only/TARGET_PHASE_NONCOMMUNICATION_COST_EVALUATION.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/runtime_sync_fused_partition_shape/RUNTIME_SYNC_FUSED_PARTITION_SHAPE.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_fused_component_shape/model/SOURCE_FUSED_COMPONENT_SHAPE_MODEL.md`
- `case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/autoresearch_224_vs_256_parameters/source_fused_component_shape/evaluator_only/TARGET_FUSED_COMPONENT_SHAPE_EVALUATION.md`

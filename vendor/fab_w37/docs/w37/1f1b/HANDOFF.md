# W37 1F1B 持续研究交接（2026-09-06）

正式 v6.8.5 保留；v686/v687/v688 为独立研究版本。用户已授权持续推进独立候选，不需要再次请求开展研究的许可；不得据此覆盖正式锁、推送或合并。只操作本 worktree，不做 16→256。

先读 [REPORT](REPORT.md)、[GOAL](GOAL.md)、[新候选语义](post685/SEMANTICS.md)。用户新增限时要求后，已验收旧里程碑关闭，新原生 Goal 成功建立为 active，截止 2026-09-07 07:00 香港时间。当前路线见 [ROUTE](ROUTE.md)，专用 Snakemake 管线与入口见 [PIPELINE](deadline_20260907/PIPELINE.md)。原第二轮文档保留在 history/20260906-round2。

## 可复核结果

| 证据 | 位置 |
|---|---|
| 正式 baseline smoke/重校准 | `results/w37/A/smoke-w37-resume/`、`reproduce-v685-w37-research/` |
| 静态 API 序列与互斥分账 | `results/w37/A/post685-audit-api-contract/` |
| v686 发布依赖候选 | [v686](post685/v686/) → `post685-candidate-causal-r3` |
| v687 发送侧有效就绪候选 | [v687](post685/v687/) → `post685-candidate-readiness-r1` |
| v688 成本场景和 lane0 反事实诊断 | [v688](post685/v688/) → `post685-candidate-scenarios-r2` |
| 图、节点查看器、逐轮结果 | [delivery](post685/delivery/index.html) → `post685-delivery-round2-final` |
| GPU 空档 runtime 可见性细分 | [runtime_review](post685/runtime_review/) → `post685-runtime-review-round2-final` |

v687 方程 `end=max(sender_entry+d,receiver_entry)+c`：B source85/90 参数均值 d=93.0369 ms、c=5.2470 ms；95/100 单消息 B MAE 15.8588→0.5850 ms，自由运行 1F1B MAPE 1.064440→0.332811%。目标四轮 Profiler MAPE：正式 10.369528%、v686 13.227529%、v687 13.984092%、v688 两轮场景 13.698250%。不可按目标误差采纳删 receiver gate 或回填真实 phase 成本。

lane0 反事实分账将 3,420.448 ms 平均缺口分为：GPU 空档 +2,277.629、仅通信 +1,157.363、仅非通信 +123.005、重叠 −188.648、剩余 +51.100 ms。它不是预测或全 rank 归因。B GPU 空档的局部源先验差额主要在 fused wrapper 内（58.242 ms/phase），不能直接命名为 CPU 等待。

[最终验收](post685/verification/acceptance.json) PASS：259 份产物 SHA、60 份冻结文件不变、三版逐轮结果精确复现；9 项测试通过。完整入口已实际执行，v688 stage 包络统计修复也已单独复现。

## 复现与开发边界

完整入口依次运行受保护 smoke、三版研究、渲染、runtime 诊断与测试：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/1f1b/post685/reproduce.py --run-id new-round2
```

只做单版：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B research/w37/onef1b/post685/run.py --mode candidate --study v687 --run-id new-v687
```

活动源码 `research/w37/onef1b/post685/`：pp_semantics.py 管顺序/消息/回放；pp_graph.py 管独立图；readiness.py 管两参数源侧拟合；scenarios.py 管成本场景；candidate.py 先封存后评分；diagnostics.py 只在 evaluator 做 16 子集归因。render_post685.py、post685_runtime_review.py 是单独入口。旧版本字节快照在各版本 history；原第一轮研究源码和冻结模型脚本未修改。

主新函数 source85/90 fit、95/100 增量检验；目标全为已见开发数据。source-only outside-wrapper 细表虽然无目标行，但提取过程读过目标且原标记诊断视图；本轮没有用于预测成本。勿把封存后读目标等同独立盲测。SHA 在 post685/inputs.json，输入变化时停止相关复现。

代码与测试本地提交：`9543e7f`；冻结模型代码未改。

大表在本 worktree results/w37/A，docs 通过本机符号链接交付；移机仍需冻结原工程派生输入与 Python/bubblewrap 环境。新 run-id 不覆盖旧目录，既有链接不自动移动。历史失败目录、seal 和源码快照保留。未推送、合并或修改 weekly-todo。

## 当前限时入口与结果

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B research/w37/onef1b/autoresearch/pipeline.py --spec docs/w37/1f1b/deadline_20260907/wave01_schedule_context.json --run-id new-t01
```

T00 专用 Snakemake 7 步 PASS，旧 v687 精确匹配；T01 13 步 PASS，nearest/pool 的目标 1F1B MAPE 15.241236%/16.546770%，均未替换正式 11.340786%。所有新流水线代码/spec 在 run 下 `control_snapshot/` 冻结，修改必须用新 run-id。T00–T13未扫raw，T14按事前资源方案执行source小样本解析；源拟合禁止访问已有目标诊断，源和目标时序仅按明确角色读取。

T07已完成：[模型与seal](deadline_20260907/runs/T07/)、[图与逐轮/阶段表](deadline_20260907/runs/T07R/diagnose/index.html)、[验收](deadline_20260907/runs/T07/acceptance.json)。v6810 CPU EP分项预测目标1F1B15.906607%、源增量0.791940%，去rank尾部消融16.424628%/1.420832%，不采纳；60份正式文件未变，14测试通过。使用wave07_ep_cpu_prediction.json和wave07_review.json复现，后者seal指向原封存模型。

T08也已完成：[完整场景/固定PP消融](deadline_20260907/runs/T08/)、[图和同口径表](deadline_20260907/runs/T08R/diagnose/index.html)、[验收](deadline_20260907/runs/T08/acceptance.json)。v6811完整两轮目标1F1B15.028381%、源增量0.332925%，仍差于正式基线。固定PP后平均节点成本与T07仅有0.5ns舍入差、边相同，目标包络增加187.336ms，支持共同波动机制。使用wave08_ep_joint_scenarios.json和wave08_review.json复现；原run-id保留快照，不能覆盖。17测试、15阶段manifest、60正式文件通过。

T09完成：[source_only证据](deadline_20260907/runs/T09/diagnose/)、[验收](deadline_20260907/runs/T09/acceptance.json)。B重算dispatch后续层相对首层多约93ms，fit/validation均稳定，与独立PP就绪参数接近；尚未做逻辑层配对或新预测。固定慢rank/纯host效应证据不足。inputs09补充已固定源提取器的只读代码语义，前两次缺key/旧evaluator角色失败保留，r3 PASS。

T10已完成：[源交叉验证](deadline_20260907/runs/T10A/diagnose/)、[三版预测与seal](deadline_20260907/runs/T10/)、[最终图/表](deadline_20260907/runs/T10R/diagnose/index.html)、[验收](deadline_20260907/runs/T10/acceptance_v2.json)。v6812目标1F1B15.008327%、源增量0.333982%，目标仅比T08改善0.020054pp，source略退化；不采纳。去代理消融32.831249%/21.064308%。19测试通过，21阶段manifest/60正式文件/控制T08图成本精确一致；成本替换而非额外叠加已核验。

T11已完成：[源GPU字段合同与覆盖](deadline_20260907/runs/T11/diagnose/field_contract.json)、[B尾部图](deadline_20260907/runs/T11/diagnose/source_cp_tail_coverage.svg)、[有效rank事件](deadline_20260907/runs/T11/diagnose/source_cp_rank_verified.csv.gz)、[验收](deadline_20260907/runs/T11/acceptance.json)。source_only 4步PASS；449,273条派生CP事件与固定上游Parquet逐项一致，7条缺失导致377条额外排除，224,448个有效配对调用。有效pair的最早end不早于最后start，最小余量6.489802ms；不把这个GPU窗口当纯网络服务。13个rank-step存在kernel重叠，采用union。除了stage0，每个B结束后5个CP kernel约43ms union，最后完成距B结束约81–83ms。

T11使用wave11_cp_intake.json、inputs11.json；新增约216MB派生输入经过资源评估，不扫raw、不读历史CP拟合参数。已核验源/目标CP2组都在同host/同EP8组。CP归属代码实际使用“下一个F/B开始”作为窗口上界，4F/9B每层序号是派生映射，部署操作ID依据仍缺失。19测试、3阶段manifest/111产物/60正式文件通过；未生成新预测。T10本地提交162d366。

T12已完成：[CP/EP/PP对齐图](deadline_20260907/runs/T12/diagnose/source_CP_EP_PP_alignment.svg)、[逐wrapper互斥覆盖](deadline_20260907/runs/T12/diagnose/source_EP_CPU_CP_disjoint_coverage.csv.gz)、[B发送尾部](deadline_20260907/runs/T12/diagnose/source_PP_B_CP_tail_alignment.csv.gz)、[验收](deadline_20260907/runs/T12/acceptance.json)。203,728个wrapper通过分区守恒，176个因CP整组缺失排除；8,632条B消息有效，8条排除，6,668条满足双端单消息及receiver在末CP前发布的局部比较范围。没有CP区间超过首个PP API返回。

源95/100逻辑层匹配后的rank wrapper B−F为93.432255ms=CP覆盖38.249752ms+未见CP区间55.182503ms；它不同于T10的group完成口径。后续B dispatch中约38.7–38.9ms是前一逻辑层CP；当前层CP约36ms。末CP完成在sender API后80.694154ms，拟合ready93.037983ms，仍余12.343829ms；末CP至首API返回17.533629ms混合后续工作与PP传输。20测试、3阶段manifest/112产物/60正式文件通过，源分析52.863s；wave12_cp_alignment.json复现，无新目标预测。T11本地提交b3d2a80。

T13已完成：[源局部验证](deadline_20260907/runs/T13A/diagnose/source_CP_readiness_local_metrics.csv)、[预测/seal](deadline_20260907/runs/T13/)、[图和全部逐轮/阶段/step/MFU](deadline_20260907/runs/T13R/diagnose/index.html)、[验收](deadline_20260907/runs/T13/acceptance.json)。v6813_CP_readiness目标1F1B14.994403%、源增量0.334014%，只比v6812增加3.002325ms预测包络，仍差于正式11.340786%，未采纳；去CP后余項15.843848%/0.784791%。新PP模型保持原EP pending代理不变。

T13A源B单消息MAE0.585010→0.593511ms，没有局部改善。CP尾部按source stage角色/lane/MB角色中位数预测，CP后潜在就绪余项约12.25–12.26ms，post-ready约5.24–5.26ms。自由运行不输入验证/目标真实CP结束；CP锚定区间包含非CP空隙，不是纯CP网络耗时。T13R所有节点/图边、源EP代理、首层0父成本、成本分解和关键路径账通过；22测试/22完成阶段/1051产物/60正式文件验收。

复现使用wave13_cp_readiness.json和wave13_review.json；后者指向本轮固定seal，新模型run须另复制review spec并更新run_root及seal SHA。T13A r2只纠正每成员实际拟合迭代标签，预测/参数精确一致；T13R r1遇到pandas分块混合类型usecols解析异常，r2显式string列及low_memory=False，模型与分数没有重跑。旧输出均保留。下一步资源方案在[T14_RESOURCE_REVIEW.json](deadline_20260907/T14_RESOURCE_REVIEW.json)，当时尚未解析raw，后续执行结果见T14；source rank16四文件stat合计156,349,592字节。旧source kernel-family汇总覆盖20轮，缺逐窗口事件，不能直接用于85/90新拟合。

T14已完成：[四轮设备覆盖图](deadline_20260907/runs/T14B/diagnose/source_dispatch_GPU_coverage.svg)、[逐CPU窗口分区](deadline_20260907/runs/T14B/diagnose/source_CPU_window_GPU_runtime_coverage.csv.gz)、[字段合同](deadline_20260907/runs/T14B/diagnose/field_contract.json)、[资源实测](deadline_20260907/runs/T14B/diagnose/source_raw_resource_measurement.json)、[验收](deadline_20260907/runs/T14B/acceptance.json)。只读解析source rank16的85/90/95/100四文件，唯一输入156,349,592字节；40,684个kernel、1,080个设备拷贝/置零事件全部找到唯一runtime/driver关联，无GPU先于关联CPU启动。每轮8个F/B、16个PP、96个EP、208个CP边界与冻结表均0ns；384个同范围EP窗口的CP覆盖与T12逐项精确一致。

单文件首版遗漏driver关联及copy/set视图，r2补全并精简索引后峰值428,441,600字节、分析9.540s，原数据不变；r2作为显式SHA门槛通过后扩展四轮，峰值489,852,928字节、分析24.886s。9个完成阶段manifest/395产物SHA、23测试、60正式文件均通过；首样本重复抽取的五份表精确一致。资源方案保持冻结的事前状态，实际执行以验收和资源表为准。没有新拟合、预测或目标时长读取（只做SHA核验）。一个四层内部rank的覆盖不能代表全rank；其他设备活动包含计算、拷贝及未分类kernel，runtime同步是重叠视图，不能再加到CPU wall。

T15已完成：[GPU提交时点分区图](deadline_20260907/runs/T15/diagnose/source_pending_submission.svg)、[逐GPU/CPU窗口关联](deadline_20260907/runs/T15/diagnose/source_GPU_CPU_window_intersections.csv.gz)、[逐逻辑层配对](deadline_20260907/runs/T15/diagnose/source_logical_F_B_submission_pairs.csv)、[互斥family差额](deadline_20260907/runs/T15/diagnose/source_logical_F_B_family_summary.csv)、[验收](deadline_20260907/runs/T15/acceptance.json)。复用T14B的8.38MB压缩表及元数据，先验全阶段SHA与source-only读取审计通过后，只允许17个准确文件路径解析；没有重读raw或目标时长。384个CPU窗口、64个逻辑F/B配对、44,144互斥区间及22,082个GPU/窗口相交记录均守恒，41,764个GPU设备事件与runtime时间逐项匹配；26测试/3阶段manifest/129产物/60正式文件通过，峰值382.01MB、分析104.997s。

同rank16的source95/100后续B−F wrapper增量为 **94.260546ms**：CP覆盖增量 **38.496419ms**，无CP时已在wrapper进入前完成CPU提交的其他GPU活动增量 **54.074526ms**，EP独占增量0.889302ms、无非PP设备事件增量0.800495ms、进入后提交项−0.000195ms。85/90的进入前提交项为53.885010ms，较一致。无CP的互斥family差额中attention_backward为21.524836ms、matrix_kernel为9.324847ms、routing为7.269069ms，名称分类仍属启发式；混合family保持单独区间，不累加边际kernel wall。这个94.261ms来自单rank的24个增量配对，与T12全rank的93.432ms/5352配对不是同一分母。

这些GPU活动的关联runtime调用在wrapper进入前已结束，支持“CPU wrapper涵盖异步已提交工作继续执行”的时间证据；其中工作究竟属于前一逻辑B层还是当前重算预处理，还需CPU checkpoint父区间映射。T15没有拟合新成本或生成224预测，最新v6813仍14.994403%，正式11.340786%更好。

T16已完成：[CPU checkpoint与GPU时间线](deadline_20260907/runs/T16/diagnose/source_checkpoint_GPU_timeline.svg)、[逐GPU父层归属](deadline_20260907/runs/T16/diagnose/source_GPU_checkpoint_ownership.csv.gz)、[局部尾部预测](deadline_20260907/runs/T16/diagnose/source_GPU_tail_local_prediction.svg)、[PP GPU分账](deadline_20260907/runs/T16/diagnose/source_PP_GPU_start_end_alignment.csv)、[验收](deadline_20260907/runs/T16/acceptance.json)。64个同线程CheckpointFunctionBackward父区间各容纳4个已核验EP wrapper、4个重计算CP和5个autograd反向CP；22,204个GPU提交映射到父区间，另19,560个属于区间外（含F/PP/其他工作）并完整保留。source95/100后续dispatch中，前一逻辑层反向工作的非CP独占覆盖为46.215273ms，85/90为46.149705ms；T15的54.075ms是B−F进入前提交活动的差额，不能全部等同这个父层归属量。

源95/100末层GPU主stream完成在PP API入口后92.956858ms，末CP在80.621916ms，二者相差12.334942ms；这补充了此前CP后约12ms余项的设备时间证据。16次B发送均匹配一个同线程CPU API内提交的SendRecv kernel，主stream结束→PP kernel开始→kernel结束→首API返回逐项守恒；增量均值分别0.168136、4.866970、0.082702ms。`all_nonPP`仅指这个checkpoint中提交的非PP事件，未强行纳入父区间外的异步DP/EDP或其他未来工作；这些设备区间不是独立网络service。

source85/90拟合的每层主stream尾部，95/100局部MAE0.679468ms（32个parent）；以GPU尾部锚定的PP局部MAE **0.663167ms**，比原PP方法 **0.557697ms** 差（同8条rank16消息），CP锚点0.672333ms、父区间全非PP锚点0.680220ms，也未改善。局部预测使用观测API入口，参数/预测先SHA封存，再连接完成观测评分；不代表整图自由运行预测。未产生新的224预测或成本迁移，正式模型保留。

T16B首次PP名称过滤过窄导致失败，r2匹配实际`mcclKernel_SendRecv`名称及同线程API区间后通过；与T16A的15份科学表、参数和预测精确一致。28测试、7个完成阶段manifest/329产物/60正式文件核验通过；最终分析26.136s、峰值391.35MB，无raw解析。训练checkpoint/block/fused代码已固定SHA并交付文本，但部署revision仍未独立证实。

T17资源计划已登记：[T17_RESOURCE_REVIEW.json](deadline_20260907/T17_RESOURCE_REVIEW.json)。该段为事前规划；后续实际解析与验收见T17。旧20轮全量汇总仍不作新源拟合。

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

T27B只在T17B已封存target224/rank16缓存上迁移两份T27A源参数；两候选各2,772行成本、四种方法固定，先封存条件预测再评分。旧API结束代理与T26B的特征、预测解压文本和末端指标精确一致，无源/目标参数更新、无未覆盖成本。API开始代理将释放反例312→0，但只是CPU入口下边界，不能证明设备有效释放。

| 目标四轮CPU归属设备末端 MAE ms | 旧API结束代理 | 新全部API开始代理 |
|---|---:|---:|
| 独立延迟 F / B | 4.604809 / 10.234830 | 4.604437 / 10.220776 |
| 同stream中位数 F / B | 0.846204 / 11.420797 | 0.760596 / 11.411286 |

F/B设备末端虽小幅改善，中位数方法在22,692个F/B归属非PP事件上的启动/完成MAE却由9.489640/9.590851升至9.526528/9.628238ms，驻留时长MAE固定0.230005ms；只保留新代理为语义更安全的局部候选，不提升全局版本。观察末端和预测最大末端可能来自不同stream/算子，当前phase包络精度含跨stream最大值抵消，下一轮先在纯源数据核验终端事件选择。

51测试、3阶段manifest/217产物及60正式文件通过；33,816个目标事件、270,528条预测，峰值902.05MB、分析26.480s，无raw重扫。r1–r5五类准入/比较失败均保留，r6通过。正式/最新全局研究1F1B仍11.340786%/14.994403%；其他阶段、Step/MFU无新预测。

周报可用两句：两种源侧封存释放代理在224卡条件迁移中只带来F/B设备末端0.085609/0.009511ms的小幅改善，而F/B内逐事件完成MAE反而由9.590851ms升至9.628238ms，未形成全局1F1B改善。API开始代理消除了312个已观察边界矛盾，但有效释放时刻、跨stream终端选择、独立CPU入口、隐藏graph、跨rank迁移和未暴露数据仍未验证。

## T28A：纯源终端事件身份与端点选择

[语义对照图](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_semantics.svg)、[终端身份参数](deadline_20260907/runs/T28A/diagnose/source_terminal_identity_parameters.csv)、[源拟合选择参数](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_selection_parameters.csv)、[增量指标](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_metrics.csv)、[逐窗口身份核验](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_event_selection_validation.csv.gz)、[预测seal](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_prediction_seal.json)、[验收](deadline_20260907/runs/T28A/acceptance.json)。只复用T14B纯源缓存，重新生成两释放代理、四方法和640行端点预测；source85/90拟合并封存身份/规则，之后才接入95/100真值。

32个源窗口观察末端全部是stream0 `aten::_copy_from`。源拟合选择在增量前向取得旧/新代理0.023677/0.013204ms且身份8/8；反向为0.604612/0.605248ms，但选择的预测最大事件不是实际末端，身份0/8。强制语义终端的独立延迟反向为0.676671/0.676665ms、身份8/8，只多约0.072ms。后续目标诊断必须同时保留两者，不能只报较小包络误差。

53测试、3阶段manifest/193产物、60正式文件及全局seal通过；验证GPU真值扰动不改变成本、身份、选择或预测。峰值699.55MB、分析33.157s，无raw或目标时序读取。正式/最新全局研究1F1B仍11.340786%/14.994403%；其他阶段、Step/MFU无新预测。

周报可用两句：纯源终端核验确认32个F/B窗口的可见设备末端均为stream0 `aten::_copy_from`，前向源拟合规则在95/100达到8/8身份命中和0.013–0.024ms MAE。反向最低端点MAE约0.605ms仍由错误的stream15事件形成，强制真实终端身份为0.677ms且8/8命中；下一轮封存这项精度—可解释性权衡后做224卡开发诊断，独立CPU入口、隐藏graph、跨rank和全局1F1B仍未验证。

## T28B：源固定终端语义的目标条件迁移

[对照图](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_transfer.svg)、[总体指标](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_metrics.csv)、[逐迭代结果](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_per_iteration.csv)、[终端身份](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_observed_terminal_identity_counts.csv)、[预测seal](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_prediction_seal.json)、[验收](deadline_20260907/runs/T28B/acceptance.json)。T28A源参数和T27B两份目标条件预测先按SHA核验，336行新候选在连接目标终点真值前封存；目标终点真值突变不改变预测。

目标24个F/B窗口的真实可见终端全部为stream0 `aten::_copy_from`。API开始语义独立延迟的F/B末端MAE为 **0.624367/10.204995ms**，均优于同API开始独立最大值控制 **4.604437/10.220776ms**，身份由1/12、9/12提升到12/12、12/12。target90的F误差2.438860ms，B四轮误差在4.258244–17.674386ms间波动；后续需预测CPU提交来源，不能把条件端点直接接成全局等待。

54测试、3阶段manifest/193产物、60正式文件通过。修复gzip和SVG非确定元数据后，r6/r7的12项科学产物字节精确，r7正式seal为`353fd2c9142fe13a3e0208bce441e8ccc470ed0f3e081531e4d038b1e0581ad2`。无参数更新、raw、拓扑、全局1F1B、Step或MFU预测；正式v685仍保留。

周报可用两句：源固定stream0 `aten::_copy_from`终端在224卡24个F/B开发窗口全部命中，并将API开始候选的F/B设备终点MAE由4.604437/10.220776ms降至0.624367/10.204995ms，首次同时改善局部时间误差和事件归因。该结果仍以观测CPU提交和F/B标注结束为条件，尚未形成全局1F1B、Step或MFU改善；target90前向异常、反向迭代波动、隐藏graph、跨rank和未暴露数据仍未验证。

## T29A：源CPU提交分解与PP重复计时门

[分解图](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition.svg)、[参数](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_parameters.csv)、[指标](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_metrics.csv)、[逐迭代](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_per_iteration.csv)、[区间摘要](deadline_20260907/runs/T29A/diagnose/source_terminal_PP_interval_summary.csv)、[逐窗口账本](deadline_20260907/runs/T29A/diagnose/source_terminal_PP_interval_ledger.csv.gz)、[预测seal](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_prediction_seal.json)、[验收](deadline_20260907/runs/T29A/acceptance.json)。source85/90拟合，160行四轮预测先seal，再连接95/100的CPU/设备/PP真值；验证真值突变不改参数和预测。

source95/100分解候选的F/B端点MAE为2.726681/6.659497ms，提交MAE为2.728800/6.506944ms。32/32个终端API在F/B内提交，B的16/16个设备终点在下一`send_backward`中；反向正尾部与PP重叠至少99.921487%，区间守恒最大误差0ns。现有图已用约93.08–93.29ms PP sender-readiness加5.157ms返回尾部表达同一边界，不能重复追加。

T16已封存的替换实验中，stream0设备端点局部MAE0.663167ms仍差于原PP方法0.557697ms。因此T29A同时拒绝“追加F/B尾部”和当前“替换PP readiness”，不登记T29B。55测试、3阶段manifest/195产物及60正式文件通过；r1/r2十二项科学输出字节精确，无raw、目标时序、正式图、全局1F1B、Step或MFU更新。

## T30A：CPU提交静态特征可辨识性

[对照图](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_identifiability.svg)、[静态签名](deadline_20260907/runs/T30A/diagnose/source_CPU_expected_static_signatures.csv)、[指标](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_metrics.csv)、[逐迭代](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_per_iteration.csv)、[签名验证](deadline_20260907/runs/T30A/diagnose/source_CPU_static_signature_validation.csv)、[预测seal](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_prediction_seal.json)、[验收](deadline_20260907/runs/T30A/acceptance.json)。source85/90静态标签与参数先seal，95/100的提交时刻和观察签名后接入；真值和签名突变不改预测。

F每窗口2,802个/79类操作，B为7,738个/179类；根操作、前序PP类型和完整名称计数在验证16/16匹配。静态扩展未增加phase/role已有的6个等价类，预测与控制精确相同，F/B提交MAE仍2.728875/6.506913ms。56测试、3阶段manifest/199产物和60正式文件通过，r1/r2十三项科学输出字节精确；拒绝目标迁移，无raw、图、全局1F1B、Step/MFU更新。

## T31A：v685 分项成本与 source replay 断点

[图](deadline_20260907/runs/T31A/diagnose/source_fullrank_cost_split.svg)、[指标](deadline_20260907/runs/T31A/diagnose/source_component_metrics.csv)、[逐迭代](deadline_20260907/runs/T31A/diagnose/source_component_per_iteration.csv)、[绑定审计](deadline_20260907/runs/T31A/diagnose/v685_source_compute_binding_audit.json)、[分账](deadline_20260907/runs/T31A/diagnose/component_accounting_ledger.csv)、[seal](deadline_20260907/runs/T31A/diagnose/source_fullrank_cost_prediction_seal.json)、[验收](deadline_20260907/runs/T31A/acceptance.json)。source85/90先拟合和seal，source95/100后评分；全量验证真值突变不改参数或预测。

recent90相对两轮中位只改善物理计算槽MAE 3.644%（0.620091→0.597494ms），PP wall和entry分别退化15.864%和60.563%，没有统一胜者。计算表实际为16个stage各一个lane0代表rank，并非256 rank全覆盖；PP wall为256 rank。v685目标图绑定144个新计算节点，但source replay绑定0个且缺少三类计算字段，只重放新PP；其805.920ms reconciliation不能校验新计算成本。T31B不登记，正式v685和拓扑不变。

58测试、3阶段manifest/208产物及60正式文件通过，20输入40,103,568字节，无raw/目标时序；r4/r5十九项科学输出字节精确，r1–r3失败保留。

## 唯一下一步

实施 T32A：只用冻结派生表审计v67/v685 source图的时钟原点、`local_gap`生成规则与v61物理计算槽的逐键可连接性；先列出每个gap包含的计算、通信、CPU/framework与等待口径，再用source85/90建立无重复计时的替换分账，在95/100验证局部守恒和全局1F1B。计算观测仅有各stage lane0，未经证据不得外推其余lane；`unclassified_calibration_ns`不得直接当作计算或等待。只有兼容source图闭合且增量全局/局部同时通过，才另登记目标候选；不读目标时序、不改正式锁，文件Goal持续至明早07:00。

## T32A：source gap所有权与全rank桥接结论

[图](deadline_20260907/runs/T32A/diagnose/source_gap_bridge.svg)、[区间指标](deadline_20260907/runs/T32A/diagnose/source_gap_partition_metrics.csv)、[phase指标](deadline_20260907/runs/T32A/diagnose/source_phase_envelope_metrics.csv)、[lineage](deadline_20260907/runs/T32A/diagnose/v67_v685_gap_lineage.csv.gz)、[所有权合同](deadline_20260907/runs/T32A/diagnose/component_ownership_contract.csv)、[验收](deadline_20260907/runs/T32A/acceptance.json)。82,695个lane0区间全部精确守恒，59,751条物理计算观测与区间计算union一一吻合。source85/90中位数在95/100的F/B局部链MAE为15.344871/13.357146ms；B链在PP标注后平均仍延伸75.988631ms，不能再作为独立等待追加。

v54→v67→v685的74,624个`local_gap`逐值lineage通过，但其shape拟合用source60–100，包含95/100；v685仍将593,661.079ms记为未分类且计算记账为0。现有全rank source replay没有留出验证，T32A拒绝目标迁移。61测试、3阶段manifest/211产物通过，r3/r4十九项科学产物字节精确；无raw、目标时序、正式拓扑、全局1F1B、Step或MFU更新。

唯一下一步是T33A：仅用全256 rank的source85/90 PP F/B phase拟合，在source95/100真值接入前seal；按训练调度与拓扑锁做自由运行全rank phase图，验证局部phase、warmup/steady/cooldown和全局1F1B。禁止使用含95/100的旧gap或把lane0物理计算外推到其他lane；只有source增量闭合后才登记目标迁移。

## T33A：全rank source phase回放

[图](deadline_20260907/runs/T33A/diagnose/source_phase_replay.svg)、[全局指标](deadline_20260907/runs/T33A/diagnose/source_phase_global_metrics.csv)、[阶段指标](deadline_20260907/runs/T33A/diagnose/source_phase_metrics.csv)、[节点绑定](deadline_20260907/runs/T33A/diagnose/source_phase_cost_bindings.csv.gz)、[边表](deadline_20260907/runs/T33A/diagnose/source_phase_edges.csv.gz)、[Shapley分账](deadline_20260907/runs/T33A/diagnose/source_phase_diagnostic_shapley.csv)、[验收](deadline_20260907/runs/T33A/acceptance.json)。source85/90中位图覆盖256 rank、每轮2,048个F/B phase，在95/100的全局1F1B MAPE为0.332811%；recent90全量/仅phase/仅runtime以及phase上界均未改善。

事后观察替换将两轮平均actual−prediction 34.719727ms分为phase wall +21.204959ms、PP +13.427048ms、本地runtime +0.087719ms；完整替换精确重建，但只作解释。64测试、3阶段manifest/214产物通过，r1/r2十九项科学输出字节精确；未登记T33B，无目标时序、正式拓扑、Step或MFU变化。

唯一下一步为T34A：固定T33A中位参数，在PP16/MB4、PP16/MB3、PP14/MB4、PP14/MB3四个静态图上拆解PP深度和microbatch数的包络收缩，预测先seal；之后目标四轮只做开发对比，不拟合跨规模倍率、残差或成本。

## T34A：静态调度收缩审计

[图](deadline_20260907/runs/T34A/diagnose/schedule_contraction.svg)、[场景](deadline_20260907/runs/T34A/diagnose/schedule_scenario_summary.csv)、[Shapley](deadline_20260907/runs/T34A/diagnose/schedule_contraction_shapley.csv)、[关键路径](deadline_20260907/runs/T34A/diagnose/schedule_critical_path_ledger.csv.gz)、[目标阶段误差](deadline_20260907/runs/T34A/diagnose/target_development_phase_metrics.csv)、[逐迭代](deadline_20260907/runs/T34A/diagnose/target_development_iteration_results.csv)、[验收](deadline_20260907/runs/T34A/acceptance.json)。T33A固定源成本在PP16/MB4、PP16/MB3、PP14/MB4、PP14/MB3上的预测为22,182.789/20,918.157/19,500.667/18,253.343ms；总收缩3,929.446ms，PP和MB的Shapley贡献2,673.468/1,255.978ms。

实际source95/100→target85/90/95/100均值只收缩655.738ms，静态模型过度收缩3,273.708ms。目标phase-transfer 1F1B MAPE 15.340746%，正式v685为11.340786%；Profiler/训练Step/MFU相对误差也均退化。目标phase平均低估96.428ms，cooldown低估138.247ms最强。未用该差额拟合倍率、残差或成本，正式锁未改。

66测试、3阶段manifest/222产物及60正式文件通过，r2/r3二十四项科学产物字节精确；首版关键路径账本缺`region`列的失败保留，修复只补语义字段。唯一下一步T35A：审计执行前runtime-readiness与静态工作负载输入的可辨识性；若仍缺代理，事前封存仅使用当轮首批F观测的在线前缀条件候选，在source95/100增量验证后才允许目标开发评分，并与冷启动结果明确分开。

## T35A：在线前缀条件预测

[图](deadline_20260907/runs/T35A/diagnose/online_prefix_nowcast.svg)、[能力审计](deadline_20260907/runs/T35A/diagnose/cold_start_runtime_capability_audit.json)、[source评分](deadline_20260907/runs/T35A/diagnose/source_online_metrics.csv)、[target逐轮](deadline_20260907/runs/T35A/diagnose/target_step_iteration_results.csv)、[Step/MFU汇总](deadline_20260907/runs/T35A/diagnose/target_step_metrics.csv)、[前缀倍率](deadline_20260907/runs/T35A/diagnose/target_online_factors.csv)、[cutoff](deadline_20260907/runs/T35A/diagnose/target_online_prefix_cutoffs.csv)、[方向分解](deadline_20260907/runs/T35A/diagnose/target_online_direction_shapley.csv)、[source seal](deadline_20260907/runs/T35A/diagnose/source_online_prediction_seal.json)、[target seal](deadline_20260907/runs/T35A/diagnose/target_online_prediction_seal.json)、[验收](deadline_20260907/runs/T35A/acceptance.json)。冷启动合同仍缺动态router token矩阵和独立runtime readiness，14行静态工作负载没有迭代状态字段，因此正式v685不变。

在线主候选只以当轮全stage首F和后半PP stage首B作为条件，用20% winsor均值分别缩放现有F/B phase wall一次；不改变本地runtime、PP readiness、边或等待。它在已暴露的source95/100开发确认中把全局MAPE 0.332811%降到0.316924%、phase MAE 15.827028降到14.373229ms。目标投影在source seal之后，目标非前缀扰动不改任何预测。

224卡四轮在线1F1B MAPE为 **1.774166%**，逐轮1.352707/1.948060/2.255827/1.540070%；Profiler/训练Step/MFU相对误差为1.722853/2.625241/2.696833%。F/B Shapley平均增加0.780/2.146秒，B方向是主要运行中修正。前缀平均在57.845%完成后才齐备，全部F/B倍率超出source85/90范围，故只能作为开发nowcast；entry/tail/outer保持正式预测逐值相同。

69测试、3阶段manifest/237产物、19个冻结输入30,431,207字节、0 raw通过；r2/r3三十六项科学产物字节精确，峰值1.216/1.219GB。r1错误使用一对一连接五个候选phase键而失败，已改为多对一真值连接，预测定义未变。唯一下一步T36A：source85/90拟合早期倍率的可靠性收缩，尝试全stage首F+末端4 stage首B，在source95/100双门通过后才做目标开发，继续报告cutoff与域外范围。

## T36A：source可靠性收缩的早期在线前缀

[图](deadline_20260907/runs/T36A/diagnose/early_prefix_reliability.svg)、[拟合账本](deadline_20260907/runs/T36A/diagnose/source_reliability_fit.csv)、[source评分](deadline_20260907/runs/T36A/diagnose/source_early_metrics.csv)、[target逐轮](deadline_20260907/runs/T36A/diagnose/target_step_iteration_results.csv)、[target汇总](deadline_20260907/runs/T36A/diagnose/target_step_metrics.csv)、[系数区间](deadline_20260907/runs/T36A/diagnose/target_early_interval_results.csv)、[cutoff](deadline_20260907/runs/T36A/diagnose/target_early_prefix_cutoffs.csv)、[source seal](deadline_20260907/runs/T36A/diagnose/source_early_prediction_seal.json)、[target seal](deadline_20260907/runs/T36A/diagnose/target_early_prediction_seal.json)、[验收](deadline_20260907/runs/T36A/acceptance.json)。全stage首F用20% winsor、末端4 stage首B用中位数；source85/90拟合将倍率向1收缩的系数为0.611345，单轮端点为0.561094/0.661596。

source95/100开发确认MAPE 0.312367%、phase MAE14.698333ms，均优于控制；目标在线MAPE7.016873%，在平均42.802%时可用，Step/MFU也优于正式冷启动但弱于T35。源系数区间覆盖目标0/4，raw倍率全域外；target最优的未收缩消融1.725071%因source退化为0.345547%而拒绝改选。

71测试、3阶段manifest/243产物、15输入30,248,941字节、0 raw通过；r1/r2三十九项科学输出字节精确，峰值1.098/1.088GB。唯一下一步T37A：复用T35/T36 seal做剩余时间与可用时点审计，不重拟合；后续同域更新必须明确目标校准轮次并保持在线标签。

## T37A：封存在线预测的剩余时间审计

[Pareto图](deadline_20260907/runs/T37A/diagnose/remaining_time_pareto.svg)、[逐轮剩余时间](deadline_20260907/runs/T37A/diagnose/remaining_time_iteration_results.csv)、[汇总指标](deadline_20260907/runs/T37A/diagnose/remaining_time_metrics.csv)、[区间逐轮](deadline_20260907/runs/T37A/diagnose/remaining_time_interval_results.csv)、[区间覆盖](deadline_20260907/runs/T37A/diagnose/remaining_time_interval_coverage.csv)、[验收](deadline_20260907/runs/T37A/acceptance.json)。T35在平均完成57.845%、实际剩余9,089.677ms时，剩余时间MAPE为4.201267%；T36主候选在完成42.802%、剩余12,333.264ms时为12.263149%，比0%时可用的v685 11.340786%更差，因而被支配。

T36 raw消融在同口径为3.010076%，若忽略资格会占据Pareto前沿，但它已经因source95/100退化而被事前门拒绝。T36可靠性区间对source85/90覆盖2/2，对source95/100和target分别只有0/2、0/4；减去同一cutoff不会改变覆盖事实。73测试、3阶段manifest/214产物、34输入30,323,159字节、0 raw通过；r2/r3七项科学输出字节精确，r1的hash-only能力缺失在preflight失败并保留。

唯一下一步T38A：在T35主候选之上登记target同域lag-one innovation。85轮初始化后，每轮只能读取紧邻的已完成轮残差；分开报告90–100 walk-forward和含初始化总体结果，不宣称冷启动或盲测。

## T38A：目标同域因果 lag-one 状态更新

[结果图](deadline_20260907/runs/T38A/diagnose/causal_lagged_update.svg)、[逐轮预测链](deadline_20260907/runs/T38A/diagnose/causal_prediction_ledger.csv)、[因果扰动门](deadline_20260907/runs/T38A/diagnose/causality_mutation_audit.csv)、[逐轮误差](deadline_20260907/runs/T38A/diagnose/target_iteration_results.csv)、[汇总](deadline_20260907/runs/T38A/diagnose/target_metrics.csv)、[phase账本](deadline_20260907/runs/T38A/diagnose/target_phase_ledger.csv)、[分解](deadline_20260907/runs/T38A/diagnose/prediction_decomposition.csv)、[区间](deadline_20260907/runs/T38A/diagnose/causal_interval_results.csv)、[验收](deadline_20260907/runs/T38A/acceptance.json)。85轮不修正；90/95/100只加上紧邻已完成轮的T35 base `actual-predicted`，修正289.388/419.385/486.382ms。当前及未来真值增加1,000,000ms仍不改变当前预测，逐轮依赖和哈希链通过。

主评估三轮的1F1B MAPE为0.536422%，对照T35为1.914652%、v685为11.572438%；剩余时间MAPE1.262981%，Profiler/训练Step/MFU相对误差0.630637/1.127495/1.143873%。含85初始化的四轮1F1B/剩余MAPE为0.740494/1.767049%。entry/tail/outer逐值不变；平均修正组成是phase-transfer 18,253.343ms、F 762.389ms、B 2,188.557ms和独立lag状态398.385ms，最终平均残差−15.266ms。lag状态没有物理组件归因。

有限历史区间在95/100才可用并覆盖1/2。76测试、3阶段manifest/221产物、31输入30,321,201字节、0 raw通过；r1/r2十一项科学输出字节精确，峰值199MB。唯一下一步T39A只做固定方法的stale/expanding-history稳健性消融，不按三轮目标结果改选。

## T39A：固定状态规则的稳健性消融

[图](deadline_20260907/runs/T39A/diagnose/state_robustness.svg)、[逐轮状态](deadline_20260907/runs/T39A/diagnose/state_prediction_ledger.csv)、[逐轮误差](deadline_20260907/runs/T39A/diagnose/state_iteration_results.csv)、[汇总](deadline_20260907/runs/T39A/diagnose/state_robustness_metrics.csv)、[冲击响应](deadline_20260907/runs/T39A/diagnose/synthetic_shock_response.csv)、[验收](deadline_20260907/runs/T39A/acceptance.json)。90–100的零更新/lag-one/stale-2/扩展均值1F1B MAPE为1.914652/0.536422/1.082859/0.502136%；lag-one三轮全改善，较扩展均值只差0.034286pp。扩展均值是事后排名，不改T38选择。

500ms合成冲击仅按已声明过去依赖传播；stale在90因无两轮历史回退零更新。79测试、3阶段manifest/220产物、29输入30,308,996字节、0 raw通过；r2/r3七项科学输出字节精确，峰值195/197MB。r1的唯一失败是iteration100 CSV往返3.64e−12ms与1e−12断言不符，随后按封存字段1e−6ms精度核验。唯一下一步T40A：在source同构序列原样检验固定lag-one，并算source/target break-even容错。

## T40A：跨域状态反证与容错边界

[跨域图](deadline_20260907/runs/T40A/diagnose/cross_domain_state.svg)、[转换账本](deadline_20260907/runs/T40A/diagnose/transition_state_results.csv)、[域汇总](deadline_20260907/runs/T40A/diagnose/domain_state_metrics.csv)、[break-even](deadline_20260907/runs/T40A/diagnose/state_noise_tolerance.csv)、[验收](deadline_20260907/runs/T40A/acceptance.json)。固定lag-one在target90/95/100改善3/3，MAPE 1.914652%→0.536422%；source只改善95一轮，90/100退化，三轮0.298320%→0.517489%。target基础残差同号3/3，source只有1/3，说明状态持续性是目标局部现象。

target三轮全改善的精确单侧sign-test p=0.125，最小对称break-even半径183.990ms；source改善1/3、p=0.875。81测试、3阶段manifest/222产物、28输入30,311,480字节、0 raw通过；r1/r2六项科学输出字节精确，峰值195/198MB。唯一下一步T41A冻结方法搜索，建立正式/在线模式决策表和复现索引；新连续target轮次是恢复方法研究的必要条件。

## T41A：运行模式、解释边界与复现索引

[决策流程图](deadline_20260907/runs/T41A/diagnose/operational_decision_flow.svg)、[模式表](deadline_20260907/runs/T41A/diagnose/operational_mode_decision.csv)、[解释分层](deadline_20260907/runs/T41A/diagnose/explanation_layers.csv)、[复现索引](deadline_20260907/runs/T41A/diagnose/reproduction_index.csv)、[停止/恢复合同](deadline_20260907/runs/T41A/diagnose/stop_resume_contract.json)、[验收](deadline_20260907/runs/T41A/acceptance.json)。唯一正式冷启动仍是v685；T35只在当轮F/B前缀齐备后作开发nowcast；T38还需要紧邻前一目标轮完成，且只作目标初始化后的顺序开发分析。T36主候选被v685支配，raw候选未过source门；T39扩展均值未被事前选择。

T38在target90/95/100的1F1B/剩余时间/训练Step/MFU相对MAPE为0.536422/1.262981/1.127495/1.143873%，但n=3的单侧sign-test最小p=0.125，且source迁移退化。状态方法研究只有在至少新增2个设计时未见、严格更晚、连续的目标轮次后恢复；冷启动方法研究要求动态router token矩阵和独立runtime readiness profile同时具备来源。83测试、3阶段manifest/225产物、56输入30,400,800字节、0 raw通过；r3/r4六项科学输出字节精确，六条复现索引哈希全部重算通过。T41没有新预测、拟合、目标评分、参数更新或图结构变化。

唯一下一步T42A：只读审计T35–T41的正式链接、acceptance/manifest/plan/spec/command哈希、文档相对链接和交付清单，生成机读完整性矩阵并重复复现。除证据链缺陷外不改方法；没有满足恢复合同的新证据时停止目标调参。

## T42A：最终证据链完整性

[完整性合同](deadline_20260907/runs/T42A/diagnose/final_integrity_contract.json)、[阶段链](deadline_20260907/runs/T42A/diagnose/evidence_chain_audit.csv)、[文档链接](deadline_20260907/runs/T42A/diagnose/documentation_link_audit.csv)、[复现重算](deadline_20260907/runs/T42A/diagnose/reproduction_recompute.csv)、[交付清单](deadline_20260907/runs/T42A/diagnose/deliverable_inventory.csv)、[验收](deadline_20260907/runs/T42A/acceptance.json)。T35–T41七条正式链接、manifest/acceptance/plan/spec/command哈希全部通过；四份文档在T42输入封存时的783个本地链接全部存在，17类总览/局部图、节点/边表、逐轮结果、分账和复现入口均通过冻结SHA，T41六条复现记录重新计算一致。

85测试、3阶段manifest/227产物、71输入37,411,250字节、0 raw通过；r1/r2六项审计输出字节精确，峰值151MB。T42未重新预测、拟合或评分，也未修改参数、边、等待和拓扑。当前有限证据包完整；唯一后续动作是等待T41三类恢复条件之一，并在恢复前先登记新计划。若截止时仍无新证据，只保存截止快照。

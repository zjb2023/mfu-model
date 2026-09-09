# W37 限时 autoresearch 推进路线

截止：2026-09-07 07:00（香港时间）；文件 Goal 持续推进，原生状态异常已留档。主目标：target224 四轮 1F1B MAPE < 正式 v685 的 11.340786%，以证据完整决定采纳。

| 时间（香港） | 路线/假设 | 行动与证据 | 结果/下一步 |
|---|---|---|---|
| 2026-09-06 09:17 | T00：建立限时、可恢复的实验数据管线 | 关闭已验收旧里程碑并建立新原生 Goal；固定 deadline、指标与边界 | 原生 Goal active |
| 2026-09-06 09:32 | T00：拆分管线不改变 v687 | Snakemake 7 步 PASS，source/target 节点、边和参数表逐项精确一致；模型只对目标字节核验 SHA | 目标 1F1B 15.340746%，源增量 0.332811%；管线验收 PASS，非新科学改善 |
| 2026-09-06 09:37 | T01：成本应按调度上下文而非仅 stage 编号迁移 | 预登记 nearest/pool；fit85/90、95/100 留 stage 增量验证；固定 v687 图边与 PP readiness | 13 步 PASS；nearest 15.241236%，pool 16.546770%，均未超过正式基线；不采纳 |

| 2026-09-06 09:43 | T02：wrapper 边界输入审查 | 0.9 MB 派生表与训练代码 SHA 固定；4 步 Snakemake，seal 先验核验 | 21,006 条记录、仅 lane0；没有 GPU-ready 字段 |
| 2026-09-06 09:48 | T03：CPU 边界与重复计时核验 | 4 步 PASS；1,908 个 phase 的 wrapper union、pre/post 和旧分区一致；附 SVG/PNG | CPU launch 不能直接视为网络 service 起点，runtime sync 是区间子集；未新增预测成本 |

| 2026-09-06 09:54 | T04：source EP 派生输入审查 | 4 步 PASS，25,488 group calls，提取记录未访问目标时序 | 历史九轮参数仅诊断，不作新拟合 |
| 2026-09-06 09:57 | T05：逐 rank 入口共同完成前提 | 203,904 anchors、18,432 phase、组成员和纳秒区间通过 | 0 次早于最后入口的返回，最小余量 0.692406 ms；支持进一步 source 图重建 |
| 2026-09-06 10:02 | T06：展开 F/B 内的 EP CPU 图 | 依据 full block 逐层重算及 wrapper 顺序，替换整段父成本 | 92,690节点/119,552边；九轮全部边界0ns，边哈希一致；非预测验收 |

每个后续实验记录 source 拟合与验证、target 开发与事后范围、输入 SHA、预测 seal、逐迭代指标、消融、资源开销和采纳/拒绝原因。不会用无效重复运行伪装持续研究。

T00 证据：`results/w37/A/autoresearch-t00-pipeline-control-r4/control_acceptance.json`、`summary/leaderboard.csv`、`pipeline.log`。r1/r2 的私有设备初始化失败、r3 的嵌套 uid-map 失败均保留，修复没有改变科学模型。

T01：nearest 仅比 v687 改善 0.099510 个百分点；pool 源侧增量 MAPE 0.332811→2.587558%。留 stage、跨迭代比较共支持 3,520 个 F/B 观测，另有 64 个 F 观测没有保留的同上下文源样本（不编造回退）；pool 局部 F MAE 11.612188→9.992420 ms，但整图变差，不能用局部均值改善代替全局验收。三候选的图边完全一致，两个内部时间视图各自守恒，12 个完成阶段 manifest 校验 PASS。证据：`autoresearch-t01-schedule-context/acceptance.json`、`source_audit/source_held_stage_metrics.csv` 与各模型 `prediction/*/target224_phase_cost_bindings.csv`。

T03：[图](deadline_20260907/runs/T03/diagnose/wrapper_boundary.svg)、[机读结论](deadline_20260907/runs/T03/diagnose/diagnostic.json)、[每类 wrapper 的差额](deadline_20260907/runs/T03/diagnose/interior_wrapper_source_target_delta.csv)。B 内 FusedDispatch 的 target−source 平均 prelaunch wall +1.334 ms，而可见 prelaunch stream-sync −6.811 ms，不能用单个 sync 增量解释缺口。CPU 阶段分账与 GPU 空档分账是两个视图，不能相加或把前者直接等同后者。

T06：[九轮校验](deadline_20260907/runs/T06/diagnose/source_observed_reassembly.csv)、[节点](deadline_20260907/runs/T06/diagnose/source85_nodes.csv.gz)、[边](deadline_20260907/runs/T06/diagnose/source85_edges.csv.gz)。独立 source EP 图 SHA 为 `8d3e85b013bfb268507641d17d0625d1db8bb381d37e9340bd8bddbc5b46b2cf`，正式 v684 锁未动。下一步为 T07 source-only 新拟合和回归；native Goal 保持 active。T00–T03 已本地提交 `a62646e`，未推送。

| 2026-09-06 10:19 | T07：source85/90 拟合 EP CPU 分项成本 | 13步 PASS；完整组共同完成、rank尾部和本地区间，保留PP readiness；95/100验证后封存 | v6810目标1F1B15.906607%、源增量0.791940%，均比v687差；去尾部16.424628%/1.420832%，不采纳 |
| 2026-09-06 10:27 | T07R：封存图与时间分账复核 | 4步 PASS；总览分列预测/trace、八rank局部等待、节点/边、逐轮和全阶段评分；14测试/15阶段manifest/60正式文件核验 | CPU共同完成含排队/计算/通信，不作纯EP耗时；T08检验完整迭代场景 |

T07：[图册](deadline_20260907/runs/T07R/diagnose/index.html)、[逐迭代](deadline_20260907/runs/T07R/diagnose/version_iteration_results.csv)、[分阶段](deadline_20260907/runs/T07R/diagnose/version_phase_results.csv)、[验收](deadline_20260907/runs/T07/acceptance.json)。新目标图61,630节点/79,120边，主模型与去尾部消融边相同；正式锁未改。目标1F1B预测18,131.338ms，关键路径中共同CPU完成区间13,497.667ms、本地区间2,939.238ms、rank尾部128.855ms，余项是PP及本地API；只能作为这一CPU候选图的互斥时间账，不能解释为物理EP网络占比。T04–T06已本地提交 `4cee02d`。

新拟合输入仅T05/T06已验收的source rank anchors，排除历史九轮拟合参数及group表中的service估计。85/90用于拟合、95/100仅增量验证；目标全部仍为开发评估。静态主机审查：源32/目标28台，共有15台、目标未见13台，按stage/EP位置相同的主机为0，EP8均在单主机内；主机差异与stage差异可能混杂，不能直接据此归因。

| 2026-09-06 10:37 | T08：成本共同波动与PP参数消融 | 17项测试通过；13步专用Snakemake已启动，最多2核/4096MB调度预算；源派生输入不变，无trace扫描 | 等待source验证、seal与target评分；未用目标结果选权重 |

| 2026-09-06 10:39 | T08：完整源EP成本场景回归 | 13步 PASS；joint目标1F1B15.028381%、源增量0.332925%；固定PP消融15.037738%/0.332879% | 相对T07改善0.878226个百分点，仍弱于正式11.340786%，不采纳 |
| 2026-09-06 10:43 | T08R：均值成本与均值回放核验 | 4步 PASS；固定PP后两场景平均节点成本与T07最大差0.5ns、边相同；17测试/15阶段manifest/60正式文件通过 | source/target包络分别增加145.993/187.336ms；下一步T09审查CPU共同完成的可迁移依据 |

T08：[图与解释](deadline_20260907/runs/T08R/diagnose/index.html)、[源场景跨度](deadline_20260907/runs/T08R/diagnose/scenario_spans.csv)、[逐迭代step/MFU](deadline_20260907/runs/T08R/diagnose/version_iteration_results.csv)、[逐阶段](deadline_20260907/runs/T08R/diagnose/version_phase_results.csv)、[均值节点成本检查](deadline_20260907/runs/T08R/diagnose/mean_cost_identity.json)、[验收](deadline_20260907/runs/T08/acceptance.json)。目标预测18,320.692ms，两个源场景范围18,095.776–18,545.608ms，实际四轮均值21,561.771ms；范围不是预测区间。Profiler/training/MFU相对MAPE为13.701824%/13.837203%/16.061420%，MFU偏差+0.474572pp；entry/tail/outer不变。T07代码/文档已本地提交 `9afe788`。

T08完整EP场景与此前v688整段phase场景的目标包络只差约0.853ms，说明当前内部展开主要恢复了统计聚合误差，还没有新增足以解释目标运行时变化的物理成本依据。内部CPU共同完成区间含设备排队/计算/通信，T09先审查其与入口差、wrapper位置及同stage双组差异，不按目标误差选host系数。

| 2026-09-06 10:50 | T09：源侧完成区间审查启动 | 新增source_only诊断规则；固定58MB anchors，不访问target时长 | 首次证据代码归档缺少输入key；r2继承旧evaluator角色导致源侧读代码被拒，失败目录保留 |

T09修复只补充源提取代码的只读SHA依赖：`inputs09.json`保留旧角色记录，审查固定源码的source-only提取循环后登记为source provenance text。代码不执行、目标读取权限不扩大、历史拟合参数不进入新成本。r3继续相同统计，没有按结果修改计算公式。

| 2026-09-06 10:53 | T09：源侧共同完成可辨识性 | source_only 4步 PASS；203,904 rank /25,488组，5.39s分析/32.1MB选择表；目标时长仅SHA | B重算dispatch首/后续层差93.432ms（85/90）、93.090ms（95/100）；T10检验与PP就绪代理的对应 |

T09：[位置图](deadline_20260907/runs/T09/diagnose/source_completion_positions.svg)、[源同键增量误差](deadline_20260907/runs/T09/diagnose/source_completion_local_metrics.csv)、[入口差变化](deadline_20260907/runs/T09/diagnose/source_delta_correlations.csv)、[同stage双组稳定性](deadline_20260907/runs/T09/diagnose/source_group_stability_metrics.csv)、[验收](deadline_20260907/runs/T09/acceptance.json)。内部stage的B recompute dispatch首层61.969/62.453ms、后续155.402/155.544ms（fit/validation），普通F dispatch约62–63ms。此前PP发送側有效就绪均值93.0369ms；二者接近只是新假设，仍需按逻辑层对齐并验证，不当作已证实GPU耗时。

最后入口rank跨两轮变动约81%–89%；同stage两组差异符号稳定率43.75%–75%，且与host/CP/数据条件混杂，不支持直接选固定慢rank或host乘数。runtime三种union每项都在wrapper范围内，和未越界也不证明它们因果独立或可相加。T08代码/证据已本地提交 `0476b21`。

| 2026-09-06 11:04 | T10A：按逻辑层配对并交叉检验PP就绪 | source_only 4步PASS；额外项验证均值92.363ms，独立PP代理93.030ms | 后续层纯预测MAE2.568703→2.488437ms，首层1.736381→2.170977ms；混合结果进入整图检验，不宣称已优化 |
| 2026-09-06 11:06 | T10：跨层代理成本与消融 | 19测试PASS，13步专用Snakemake启动；控制/代理替换/去代理3版，共用图边 | 拟合85/90逐轮场景，等待source回归、seal与224评分 |

| 2026-09-06 11:08 | T10：跨层未完成工作成本替换 | 13步PASS；目标1F1B15.028381→15.008327%，源增量0.332925→0.333982%；去代理目标32.831249% | 仅4.324ms/0.020054pp目标变化，正式基线更好，不采纳 |
| 2026-09-06 11:16 | T10R：分账、节点与图边复核 | 两成员/两场景的边和控制版相同，成本=F基项+代理、父phase为0；19测试、21阶段manifest/60正式文件通过 | 代理模型内关键路径3,905.350ms，去代理全图缩短3,842.792ms，路径变化需区分；T11审查CP可见覆盖 |

T10：[源交叉检验](deadline_20260907/runs/T10A/diagnose/source_pending_bridge_local_metrics.csv)、[图与完整逐迭代/阶段表](deadline_20260907/runs/T10R/diagnose/index.html)、[逐节点成本分解](deadline_20260907/runs/T10R/diagnose/pending_group_node_decomposition.csv.gz)、[验收](deadline_20260907/runs/T10/acceptance_v2.json)。源逻辑层配对后的后续B额外区间92.362908ms（95/100），独立PP代理93.030163ms；后续层纯预测MAE2.568703→2.488437ms，首层1.736381→2.170977ms，均已记录。目标Profiler/training/MFU相对MAPE为13.683702%/13.820241%/16.038576%，MFU偏差+0.473897pp；其他阶段固定不变。首版图例字段名含ms而轴使用秒，r2改为成本名称；所有科学表与首版精确一致，旧输出保留。

T11候选输入已定位为source CP派生表，仅做了目录、表头和元数据阅读：449,273 rank事件/224,448完整组，预期449,280/224,640，7个无效pair-step、43,200事件超出B annotation；service来自本集群trace尾部，不是独立OISA CP仿真。下一步必须SHA固定与完整范围核验，不能直接沿用3,120条历史拟合参数或把4/9个序号节点当作已证实的设备依赖。T09代码/文档已本地提交 `74634e9`。

| 2026-09-06 11:38 | T11：CP GPU时间域、覆盖与尾部核验 | 显式固定约216MB CSV/上游Parquet及只读提取代码；source_only Snakemake4步 | 449,273条源CP事件逐字段匹配上游kernel；19测试PASS，无raw扫描 |
| 2026-09-06 11:39 | T11：纳秒精度、整组排除和区间union | 7条缺失/377条额外排除/224,448有效组；3阶段manifest、111产物SHA、60正式文件通过 | 多数B后5个CP kernel约43ms union、81–83ms末端；进入T12 EP/PP包含关系核验，不新增目标成本 |

T11：[字段语义](deadline_20260907/runs/T11/diagnose/field_contract.json)、[B尾部图](deadline_20260907/runs/T11/diagnose/source_cp_tail_coverage.svg)、[逐轮覆盖](deadline_20260907/runs/T11/diagnose/source_cp_coverage.csv)、[验收](deadline_20260907/runs/T11/acceptance.json)。选择rank+upstream表内存176.7MB，分析42.285s；1024MB是调度预算，不伪称实测峰值。source85完整，90的B缺1条；95缺3条、100缺1条，全部通过整组规则排除，不能把缺失当0成本。有效组min(end)−max(start)最小6.489802ms，无负值；13个rank-step有kernel重叠，采用union。

提取链确认GPU kernel时间，不是CPU API或纯网络FCT。源/目标CP组都在单host/同EP8内；实际归属窗为[this F/B start,next F/B start)，末个F/B无后续起点上界。每层4F/9B序号是可复算假设，缺部署操作ID证明；历史3120条九轮CP参数仍未用于新拟合。目标时长仅SHA，没有本轮预测改善宣称。下一步T12按逻辑层将CP区间对齐EP wrapper与B发送API，检查93ms代理是否包含这些活动。T10已本地提交162d366，native Goal继续active。

| 2026-09-06 11:47 | T12：当前层/前一B层CP与EP/PP区间对齐 | 复用T11固定源输入；20测试PASS，含绝对纳秒精度和多层重叠互斥测试 | source_only Snakemake4步，CPU窗口与CP GPU视图不相加 |
| 2026-09-06 11:48 | T12：交叉区间核验完成 | 203,728 wrappers/8,632 B消息守恒；176 wrappers/8消息因CP配对缺失排除 | 源后续B−F增量93.432ms=CP38.250+其余55.183ms；末CP距拟合PP就绪仍12.344ms；进入T13预测检验 |

T12：[对齐图](deadline_20260907/runs/T12/diagnose/source_CP_EP_PP_alignment.svg)、[逐逻辑层配对](deadline_20260907/runs/T12/diagnose/source_logical_F_B_CP_coverage.csv.gz)、[PP尾部](deadline_20260907/runs/T12/diagnose/source_PP_B_CP_tail_alignment.csv.gz)、[验收](deadline_20260907/runs/T12/acceptance.json)。rank wrapper比较用5352个source95/100后续层配对；不是T10 group区间的同一分母。CP末端到sender API的偏移80.694ms，拟合ready93.038ms，末CP到首API返回17.534ms；这些局部观测不构成独立GPU就绪或纯网络耗时。所有8632消息CP区间都不越首API返回，不能再加到现有PP wall；下一步T13用源可预测CP尾部加潜在就绪余项替换同一ready成本，并做去余项消融。T11提交b3d2a80；本轮分析52.863s，3阶段manifest/112产物SHA/60正式文件通过，Goal继续active。

| 2026-09-06 11:58 | T13A：源CP锚定就绪局部检验 | CP尾部按stage角色/lane/MB角色源拟合，非负余项和post-ready按lane拟合 | 源B单消息MAE0.585010→0.593511ms，轻微退化；拟合余项约12.253ms，无局部改善宣称 |
| 2026-09-06 12:02 | T13A：每成员数据标签复核 | r2将单85/90模型的另一轮标为other-fit diagnostic；所有预测/参数与r1逐项一致 | 22测试PASS；旧输出保留，纠正口径不改公式 |
| 2026-09-06 12:03 | T13：PP成本替换与去余项消融 | 13步Snakemake启动；固定原EP pending代理，3版共用图边，CP观测经T11/T12同SHA角色准入 | 等待完整source回归、seal与224评分，不用目标结果选余项 |

| 2026-09-06 12:06 | T13：整图封存与目标评分完成 | 13步PASS；目标1F1B15.008327→14.994403%，源0.333982→0.334014% | 目标只多3.002325ms预测包络，不采纳；去余项15.843848%/0.784791% |
| 2026-09-06 12:09 | T13R：PP成本隔离和时间账 | 首次CSV分块混合类型读取失败；r2指定string方向列及low_memory=False后4步PASS | 图边与非B-PP成本（含EP代理）精确不变；22测试/22完成阶段/1051产物/60正式文件通过 |

T13：[逐轮/阶段/step/MFU与图](deadline_20260907/runs/T13R/diagnose/index.html)、[就绪节点成本表](deadline_20260907/runs/T13R/diagnose/CP_readiness_node_decomposition.csv.gz)、[验收](deadline_20260907/runs/T13/acceptance.json)。目标Profiler/training/MFU相对MAPE13.671119%/13.808462%/16.022719%，MFU偏差+0.473428pp。目标关键路径中的CP后潜在就绪183.148529ms，去项后整体缩短同量（舍入误差内）；源路径变化使其关键路径项220.206498ms不等于整体缩短220.071580ms。源GPU CP结束位置提高可解释性，但重新表达几乎相同的源成本未解决目标迁移。

T14资源边界已登记：[方案](deadline_20260907/T14_RESOURCE_REVIEW.json)。四份source rank16、85/90/95/100文件stat共156.35MB；只规划先测source85单文件，尚未解析raw。旧kernel-family/top-kernel汇总20轮且无逐窗口事件，不能直接新拟合；下一步精确固定SHA和字段，测RSS/耗时、比对CP/EP/PP边界后再扩展其余3份。单lane证据不外推全rank，不读取目标raw。T12本地提交751148d；Goal继续active。

| 2026-09-06 12:27 | T14A：source85单rank原始事件采集 | 专用Snakemake4步；38.8MB输入，8/16/96/208个F-B/PP/EP/CP边界0ns | 首版发现512个driver关联及copy/set未覆盖，保留输出并补齐视图 |
| 2026-09-06 12:32 | T14A r2：关联与资源门槛 | 9,983 kernel和270 copy/set唯一CPU关联；峰值428.44MB/9.540s | 默认阶段拒绝raw解析，只有登记的source_only诊断可读；通过后扩展 |
| 2026-09-06 12:35 | T14B：四轮source事件覆盖 | 总156.35MB，峰值489.85MB/24.886s；40,684 kernel与1,080 copy/set关联完整 | 384窗口与T12精确；23测试/9阶段manifest/395产物/60正式文件通过，无新预测 |

T14已完成：[四轮设备覆盖图](deadline_20260907/runs/T14B/diagnose/source_dispatch_GPU_coverage.svg)、[逐CPU窗口分区](deadline_20260907/runs/T14B/diagnose/source_CPU_window_GPU_runtime_coverage.csv.gz)、[字段合同](deadline_20260907/runs/T14B/diagnose/field_contract.json)、[资源实测](deadline_20260907/runs/T14B/diagnose/source_raw_resource_measurement.json)、[验收](deadline_20260907/runs/T14B/acceptance.json)。只读解析source rank16的85/90/95/100四文件，唯一输入156,349,592字节；40,684个kernel、1,080个设备拷贝/置零事件全部找到唯一runtime/driver关联，无GPU先于关联CPU启动。每轮8个F/B、16个PP、96个EP、208个CP边界与冻结表均0ns；384个同范围EP窗口的CP覆盖与T12逐项精确一致。

单文件首版遗漏driver关联及copy/set视图，r2补全并精简索引后峰值428,441,600字节、分析9.540s，原数据不变；r2作为显式SHA门槛通过后扩展四轮，峰值489,852,928字节、分析24.886s。9个完成阶段manifest/395产物SHA、23测试、60正式文件均通过；首样本重复抽取的五份表精确一致。资源方案保持冻结的事前状态，实际执行以验收和资源表为准。没有新拟合、预测或目标时长读取（只做SHA核验）。一个四层内部rank的覆盖不能代表全rank；其他设备活动包含计算、拷贝及未分类kernel，runtime同步是重叠视图，不能再加到CPU wall。

实施 T15：将T14B已封存的source GPU/runtime表按准确文件SHA和source-only阶段证据准入，按相同rank/迭代/逻辑层比较F与B dispatch，区分CP覆盖、其他设备活动、重叠及launch早于wrapper的未完成工作。先检查85/90与95/100的一致性和CPU-owner归属，再决定可迁移成本方法；不重读raw、不直接拟合目标、不把单rank证据扩作全rank。专用Snakemake继续运行，正式锁保持，Goal active至明早07:00。 T13已本地提交e87a1b5。

| 2026-09-06 12:50 | T15：源GPU提交时点及family分区 | 8.38MB已封存源表准入，25测试通过，专用source_only Snakemake4步启动 | 只读准确路径，其他旧产物只哈希；前置核验拒绝历史target时长解析 |
| 2026-09-06 12:52 | T15：源同rank异步工作覆盖 | 4步PASS，26测试/3阶段manifest/129产物/60正式文件通过 | 增量后续B−F94.261ms含CP38.496、进入前提交的其他GPU54.075；无新预测 |

T15已完成：[GPU提交时点分区图](deadline_20260907/runs/T15/diagnose/source_pending_submission.svg)、[逐GPU/CPU窗口关联](deadline_20260907/runs/T15/diagnose/source_GPU_CPU_window_intersections.csv.gz)、[逐逻辑层配对](deadline_20260907/runs/T15/diagnose/source_logical_F_B_submission_pairs.csv)、[互斥family差额](deadline_20260907/runs/T15/diagnose/source_logical_F_B_family_summary.csv)、[验收](deadline_20260907/runs/T15/acceptance.json)。复用T14B的8.38MB压缩表及元数据，先验全阶段SHA与source-only读取审计通过后，只允许17个准确文件路径解析；没有重读raw或目标时长。384个CPU窗口、64个逻辑F/B配对、44,144互斥区间及22,082个GPU/窗口相交记录均守恒，41,764个GPU设备事件与runtime时间逐项匹配；26测试/3阶段manifest/129产物/60正式文件通过，峰值382.01MB、分析104.997s。

同rank16的source95/100后续B−F wrapper增量为 **94.260546ms**：CP覆盖增量 **38.496419ms**，无CP时已在wrapper进入前完成CPU提交的其他GPU活动增量 **54.074526ms**，EP独占增量0.889302ms、无非PP设备事件增量0.800495ms、进入后提交项−0.000195ms。85/90的进入前提交项为53.885010ms，较一致。无CP的互斥family差额中attention_backward为21.524836ms、matrix_kernel为9.324847ms、routing为7.269069ms，名称分类仍属启发式；混合family保持单独区间，不累加边际kernel wall。这个94.261ms来自单rank的24个增量配对，与T12全rank的93.432ms/5352配对不是同一分母。

这些GPU活动的关联runtime调用在wrapper进入前已结束，支持“CPU wrapper涵盖异步已提交工作继续执行”的时间证据；其中工作究竟属于前一逻辑B层还是当前重算预处理，还需CPU checkpoint父区间映射。T15没有拟合新成本或生成224预测，最新v6813仍14.994403%，正式11.340786%更好。

实施 T16：复用T14B源表，按同线程CheckpointFunctionBackward父CPU区间及已核验EP逻辑层，为GPU提交标注当前重算/前一B层来源；检查延续至后续dispatch和末层PP API的GPU尾部。先用source85/90建立独立的局部尾部成本预测，在95/100增量验证并与CP锚点/PP就绪代理比较；分清观察重建和预测，不把单rank成本直接迁移全rank，不回填目标。保持Snakemake、正式锁与Goal active至明早07:00。 T14已本地提交593501c。

| 2026-09-06 13:05 | T16：checkpoint父层与局部尾部预测 | 28测试PASS，source-only4步；固定checkpoint/block/fused代码，局部预测SHA封存 | 不新增目标预测，不把CPU父层关联当调度边 |
| 2026-09-06 13:06 | T16A：源GPU尾部验证 | 64父区间各4重算/5反向CP，增量PP局部0.663167ms vs原0.557697ms | 未改善，不迁移单rank成本；继续PP GPU边界核验 |
| 2026-09-06 13:10 | T16B：PP GPU名称过滤失败 | 预期字符串MCCL_SendRecv与实际mcclKernel_SendRecv不匹配 | 保留失败，修正名称谓词，参数和预测不变 |
| 2026-09-06 13:11 | T16B r2：PP GPU顺序与分账 | 16条相同线程API关联、三个区间守恒；28测试/7阶段manifest/329产物/60正式文件通过 | 15科学表与T16A精确一致；源语义更清楚，但尚无目标改善 |

T16已完成：[CPU checkpoint与GPU时间线](deadline_20260907/runs/T16/diagnose/source_checkpoint_GPU_timeline.svg)、[逐GPU父层归属](deadline_20260907/runs/T16/diagnose/source_GPU_checkpoint_ownership.csv.gz)、[局部尾部预测](deadline_20260907/runs/T16/diagnose/source_GPU_tail_local_prediction.svg)、[PP GPU分账](deadline_20260907/runs/T16/diagnose/source_PP_GPU_start_end_alignment.csv)、[验收](deadline_20260907/runs/T16/acceptance.json)。64个同线程CheckpointFunctionBackward父区间各容纳4个已核验EP wrapper、4个重计算CP和5个autograd反向CP；22,204个GPU提交映射到父区间，另19,560个属于区间外（含F/PP/其他工作）并完整保留。source95/100后续dispatch中，前一逻辑层反向工作的非CP独占覆盖为46.215273ms，85/90为46.149705ms；T15的54.075ms是B−F进入前提交活动的差额，不能全部等同这个父层归属量。

源95/100末层GPU主stream完成在PP API入口后92.956858ms，末CP在80.621916ms，二者相差12.334942ms；这补充了此前CP后约12ms余项的设备时间证据。16次B发送均匹配一个同线程CPU API内提交的SendRecv kernel，主stream结束→PP kernel开始→kernel结束→首API返回逐项守恒；增量均值分别0.168136、4.866970、0.082702ms。`all_nonPP`仅指这个checkpoint中提交的非PP事件，未强行纳入父区间外的异步DP/EDP或其他未来工作；这些设备区间不是独立网络service。

source85/90拟合的每层主stream尾部，95/100局部MAE0.679468ms（32个parent）；以GPU尾部锚定的PP局部MAE **0.663167ms**，比原PP方法 **0.557697ms** 差（同8条rank16消息），CP锚点0.672333ms、父区间全非PP锚点0.680220ms，也未改善。局部预测使用观测API入口，参数/预测先SHA封存，再连接完成观测评分；不代表整图自由运行预测。未产生新的224预测或成本迁移，正式模型保留。

T16B首次PP名称过滤过窄导致失败，r2匹配实际`mcclKernel_SendRecv`名称及同线程API区间后通过；与T16A的15份科学表、参数和预测精确一致。28测试、7个完成阶段manifest/329产物/60正式文件核验通过；最终分析26.136s、峰值391.35MB，无raw解析。训练checkpoint/block/fused代码已固定SHA并交付文本，但部署revision仍未独立证实。

T17已完成元数据资源规划：[方案](deadline_20260907/T17_RESOURCE_REVIEW.json)。target rank16四文件共123843399字节，先85单份30,740,699字节；目前只读metadata/stat，raw SHA及事件内容均未读取。实施 T17：按已登记资源方案，先固定并只读解析target85/rank16的一份30.74MB trace，且只能在v6813 seal核验后的evaluator阶段进行；与冻结F/B及source rank16证据核对GPU/runtime可见性、checkpoint尾部和CPU空档。目标数据全为开发/事后诊断，不能回填成本；通过首文件资源和字段门槛后才考虑登记的90/95/100，最多123.84MB，不扩rank或扫描全网格。用结果选择下一项有source预测依据的运行时方法，正式锁保持，Goal active至明早07:00。 T15本地提交7c4d38a。

| 2026-09-06 13:27 | T17A：seal后目标85单样本 | 29测试PASS；30.74MB raw，8,236设备事件唯一关联，6 F/B精确 | 首文件峰值394.19MB/9.703s门槛通过，目标数据仅evaluator读 |
| 2026-09-06 13:30 | T17B：四轮目标可见性对照 | 总123.84MB，33,816设备事件唯一关联；峰值556.29MB/28.317s | 840窗口守恒；29测试/6阶段manifest/293产物/60正式文件与v6813 seal通过，无新预测 |

T17已完成：[source/target dispatch可见活动图](deadline_20260907/runs/T17B/diagnose/target_source_dispatch_visibility.svg)、[逐CPU窗口GPU/runtime分区](deadline_20260907/runs/T17B/diagnose/evaluator_only/CPU_window_GPU_runtime_coverage.csv.gz)、[逐轮F/B活动](deadline_20260907/runs/T17B/diagnose/evaluator_only/FB_activity_profile.csv)、[MB角色表](deadline_20260907/runs/T17B/diagnose/evaluator_only/dispatch_by_MB_role.csv)、[验收](deadline_20260907/runs/T17B/acceptance.json)。v6813 seal及全部预测逐项核验后，先解析target85/rank16单份30.74MB，峰值394.19MB/9.703s、关联和边界门槛通过，才扩展登记的90/95/100。目标四份唯一raw共123,843,399字节，33,816个设备事件（32,960 kernel和856 copy/set）均唯一runtime关联；24个F/B边界精确一致，288个EP wrapper对旧偏移的最大差341ns，符合旧浮点epoch表示的512ns检查范围。两轮都只在diagnose/evaluator解析目标raw，源/model阶段保持拒绝；原始文件未复制。

同rank16、四轮等权的目标−源平均B阶段差额为 **251.224406ms**：未见非PP GPU事件 **+172.127089ms**、EP独占 **+48.668490ms**、CP独占 **+40.153676ms**、其他GPU独占+7.986572ms、CP/其他重叠−17.710551ms、EP/其他重叠−0.000869ms。F阶段差额114.077564ms，其中未见非PP设备事件+75.106498ms。这里比较源4MB与目标3MB的每phase平均，不是全rank、整迭代误差归因，也不是CPU等待或纯网络耗时。各case自身时钟内保留整数ns，跨运行只比较时长；另列first/middle/last角色，避免把source中间MB2当成target末MB2。

目标checkpoint主stream尾部平均 **88.344548ms**，源为 **92.900068ms**；本样本不支持继续增大PP尾部来补偿目标整体低估。CPU同步与设备分区仍为重叠视图，未重复加时。840个CPU窗口分区守恒；单样本5表和覆盖与四轮复取精确一致，source尾部与T16精确一致。29测试、6阶段manifest/293产物/60正式文件及v6813 seal全部通过；四轮最终峰值556.29MB、分析28.317s。所有目标产物永久标注development/posthoc NEVER_MODEL_FIT；本轮没有新成本、模型或224预测，最新候选/正式1F1B MAPE仍14.994403%/11.340786%。

实施 T18：复用T14B源表先在source_only阶段定义并核验非PP设备可见性空档的互斥分区，再在v6813 seal后的evaluator用同一算法核对T17B目标表；按下一GPU事件的CPU提交是否已开始/结束及EP wrapper、checkpoint位置拆分，边界空档单列。不得把“无事件”直接当CPU计算/等待或把目标派生表用于源拟合，不再读raw。用该证据选择可由source85/90估计并在95/100核验的CPU提交或设备队列成本方法，避免继续无依据抬高PP尾部；正式锁和Goal active保持至明早07:00。 T16已本地提交565dde8。

| 2026-09-06 13:56 | T18A：纯源设备空档与CPU提交 | 32 F/B、28,426空档、48,568分段精确守恒；source-only，无raw | 冻结分类器后才准入目标缓存；测试导入顺序修复，算法不变 |
| 2026-09-06 14:01 | T18R：目标缓存同算法核验 | B额外空档172.127ms=提交前51.439+中17.437+后103.250+同时0.000413 | source三表精确相同，无新拟合或224预测 |
| 2026-09-06 14:03 | T18R r2：图复核 | 10科学CSV精确一致；33测试/9阶段manifest/430产物/60正式文件及seal通过 | 修正图标签与轴范围，保留旧图；转T19，Goal继续active |

T18已完成：[源侧空档表](deadline_20260907/runs/T18A/diagnose/FB_gap_accounting.csv)、[源/目标提交与CPU位置图](deadline_20260907/runs/T18R/diagnose/evaluator_only/device_gap_context_review.svg)、[逐轮分账](deadline_20260907/runs/T18R/diagnose/evaluator_only/FB_gap_per_iteration.csv)、[逐空档及CPU/runtime分段](deadline_20260907/runs/T18R/diagnose/evaluator_only/gap_context_segments.csv.gz)、[验收](deadline_20260907/runs/T18R/acceptance.json)。先对T14B纯源表实施整数ns的设备union补集和提交边界分类，32个source F/B通过；再核验v6813 seal和分类器SHA后，用相同算法处理T17B缓存目标表，24个target F/B通过。源侧三张科学表精确一致，共49,553空档、94,096分段全部守恒；不重读raw。

rank16四轮等权、每F/B平均的目标−源空档差额拆分如下；“提交”指下一条可见非PP设备事件的关联runtime API，不证明它是唯一阻塞者。

| 阶段 | 空档差额 / ms | API开始前 / ms | API执行中 / ms | API结束后 / ms | 同时开始或阶段边界 / ms |
|---|---:|---:|---:|---:|---:|
| B | 172.127089 | 51.439087 | 17.437093 | 103.250495 | 0.000413 |
| F | 75.106498 | 17.975088 | 6.996102 | 50.103433 | 0.031876 |

B差额按另一组互斥CPU位置分区为：EP wrapper内127.786235ms、wrapper外但checkpoint内43.839727ms、checkpoint外0.501126ms。B的API结束后差额中，53.563938ms与同一提交线程的musaDeviceSynchronize相交，13.462256ms与musaStreamSynchronize相交，11.645442ms与musaStreamWaitEvent相交；这些是同一空档的上下文，不能加回阶段wall或等同纯网络/CPU开销。F/B首尾空档、同时开始的设备事件及跨阶段下一事件单列；源4MB和目标3MB按各轮phase均值比较，MB角色另表。

源侧耗时11.928s、峰值342.27MB；最终源/目标复核20.263s、峰值408.54MB，新增缓存清单17.47MB。33测试、9阶段manifest/430产物/60正式文件和v6813预测seal通过。首次测试暴露同名worker模块的导入顺序冲突，改为测试执行时导入后通过；原图标签重叠及纵轴范围问题保留，r2新增可读复核图，10张科学CSV解压后逐字相同。没有新成本或预测，正式/最新研究候选目标1F1B仍11.340786%/14.994403%；Profiler、training、MFU和其他阶段未变。下一项方案为[T19局部设备队列预测](deadline_20260907/T19_PLAN.json)，不能用目标差额设置等待成本。

唯一下一步：实施 T19：只准入T14B纯源缓存，先核验同stream的CPU提交顺序与设备执行顺序、graph多事件关联和歧义覆盖；建立局部设备队列候选，以观测CPU API结束作为明确的条件输入，用source85/90拟合设备驻留及非负提交余项，按同stream前序预测完成递推预测GPU开始/结束，在95/100做增量评分并做无队列/零余项消融。跨stream/event依赖没有证据时单列缺失，不能按GPU目标时间添加边；不读取目标派生表或回填其成本。局部条件预测不等于整图自由运行，只有源侧可核验改善且CPU入口成本来源明确后再讨论整图接入和224封存回归；Goal active保持至明早07:00。

| 2026-09-06 14:18 | T19A：设备提交顺序与graph覆盖 | 41,764事件/44stream轮次顺序通过；512拷贝早于API返回；384graph无直接device关联 | 按类别固定释放代理，不把无事件区间算idle |
| 2026-09-06 14:23 | T19B：CPU条件式设备队列预测 | source95/100中位数队列末端F/B MAE0.031/0.835ms，均值F680.881ms | 中位数局部有改善，均值混合release等待失败；没有224新预测；36测试/6阶段manifest/302产物/60正式文件通过 |

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

唯一下一步：实施 T20：按新登记资源方案，只读重访已冻结source85/rank16的一份38.82MB trace，补取T14未保留的gpu_user_annotation、ac2g flow及Graph相关事件字段；先验证与T14核心设备/runtime表的对应关系，再检查384次源GraphLaunch无直接设备关联这一覆盖缺口能否由遗漏类别解释。annotation区间不能自动算设备busy，flow没有唯一匹配不能补因果边；不扩rank、不读目标raw、不拟合目标成本。若得到可复核的graph/stream活动范围，再判断可独立预测的CPU/设备release或graph成本；T19局部预测保留条件输入标签，正式锁与Goal active保持至明早07:00。

| 2026-09-06 14:39 | T20A：遗漏类别单源补查 | 38.82MB一文件，核心三表精确；280 annotation/31,287 flow点 | 未取得graph设备端点，不扩扫 |
| 2026-09-06 14:48 | T20B：缓存端点核验 | 10,253对CPU→设备；96 GraphLaunch只有CPU单点 | 37测试/6阶段manifest/306产物通过，目标误差未变 |

T20已完成：[补取字段合同](deadline_20260907/runs/T20A/diagnose/field_contract.json)、[flow端点/CPU/设备/annotation图](deadline_20260907/runs/T20B/diagnose/source_graph_flow_coverage.svg)、[底层观测节点](deadline_20260907/runs/T20B/diagnose/source_flow_observation_nodes.csv.gz)、[CPU→设备关联边](deadline_20260907/runs/T20B/diagnose/source_CPU_GPU_correlation_edges.csv.gz)、[GraphLaunch缺失端点表](deadline_20260907/runs/T20B/diagnose/source_GraphLaunch_without_device_endpoint.csv)、[验收](deadline_20260907/runs/T20B/acceptance.json)。按冻结资源方案只重访source85/rank16的一份38,816,598字节trace，10,253设备、21,034 runtime/driver和49,955 CPU事件核心字段均与T14精确一致；补取280个GPU annotation、31,287个ac2g点和21,034组flow identity，峰值476.36MB、分析50.132s。没有扩展到其他source文件或target raw。

第二个source_only Snakemake阶段仅使用2.67MB缓存，全部flow按实际pid/tid/整数ns核验：10,253对s→f精确对应既有CPU API入口→设备事件开始；1,839个s单点、8,942个f单点均落在CPU API入口，f字母本身不能视为GPU完成。96次GraphLaunch全部只有CPU s单点，新增graph设备端点和跨stream依赖均为0；280个annotation均可按External id和名称找到CPU scope，但它们是范围标注，不计入设备busy。交付31,567个观测节点和10,253条提交关联边，后者不是完整因果依赖图，也不替换正式拓扑。

37测试、6阶段manifest/306产物SHA、60正式文件和v6813 seal核验通过；缓存复核峰值281.13MB、分析57.635s。遗漏类别没有修复这份trace中的graph设备执行覆盖，不能据此把无设备事件区间命名为空闲、CPU等待或纯网络成本；尚无依据为此增加成本。没有新拟合或224预测，正式/最新研究候选1F1B MAPE仍11.340786%/14.994403%，Profiler、training、MFU与其他阶段均不变。下一步改查独立源侧计数器的可用性，避免仅为重复覆盖缺口而继续扫trace。

唯一下一步：实施 T21：只准入源侧既有硬件清单、CP/EP派生计数器表及其提取代码，核验source85/90/95/100、rank16的host/GPU映射、时间覆盖、采样精度和派生归属假设；判断MTLink是否有能力为GraphLaunch可见性缺口提供独立证据。先做专用Snakemake小表核验，不执行旧全rank脚本；若需74.69MB的单GPU原始计数器文件，另行固定SHA和资源门槛后才解析。计数器与阶段同时观测不能直接用作预测特征，比例分摊字节不能证明graph执行；不读取目标计数器或回填目标成本，正式锁与Goal active保持至明早07:00。

| 2026-09-06 15:04 | T21首版小表核验 | CP family名称选择错误触发断言 | 失败保留；不修改数据 |
| 2026-09-06 15:05 | T21-r2：硬件派生表与采样元数据审查 | 4轮CP union/384 graph括号精确；旧分位数混PCIe、EP比例归属有假设 | 37测试/4阶段manifest/208产物通过；登记T22单文件区间探查 |

T21已完成：[源计数器分辨率图](deadline_20260907/runs/T21/diagnose/source_counter_resolution_review.svg)、[四轮host/GPU/时间覆盖](deadline_20260907/runs/T21/diagnose/source_counter_runtime_scope_coverage.csv)、[既有归属方法审查](deadline_20260907/runs/T21/diagnose/source_existing_counter_method_review.csv)、[逐graph区间与采样元数据](deadline_20260907/runs/T21/diagnose/source_graph_counter_resolution.csv)、[验收](deadline_20260907/runs/T21/acceptance.json)。专用source_only Snakemake 4步通过；只读24.24MB已有派生表/元数据/提取代码，CP/EP旧表物理包含20轮，分析明确筛选85/90/95/100的rank16，不拟合任何成本。每轮208个CP事件的union与旧CP表精确一致，384个GraphLaunch描述性括号与T19逐字段相同，source fit与incremental标签分开。

对应worker33008/GPU0的四轮设备/runtime范围均落在同一74,694,601字节计数器文件的清单覆盖内。清单987,849行中包含718,581条PCIe记录，**文件全体行的5.000692ms中位数并不是MTLink专属或局部采样分辨率**；384个graph括号中192个短于这一文件指标，也不能据此断言每个短括号都无法观测。旧CP表按kernel窗口裁剪采样区间；旧EP表把相交样本全部字节按EP重叠比例分配，并假设非EP间隔不承载EP流量。二者都无法独立证明graph无设备事件区间的通信活动，不能直接作为graph服务成本。CP提取器隔离underflow及同link下一条恢复样本；EP提取器仅排除超int64的delta及非正时长，未执行同样的恢复样本隔离，历史产物保留。

37测试、4完成阶段manifest（含首版preflight）/208产物SHA、60正式文件与v6813 seal通过；最终峰值250.86MB、分析14.210s。首版把CP family误写为`CP`而断言失败，r2改为精确process-group/collective筛选并核验`CP_collective`，输入和统计时长未改。原始计数器仅stat、未解析；后续资源方案登记时只计算单文件SHA。采样end使用host_realtime的既有映射仍需检查实际字段；别处通用代码的host_mono校正不证明该采集具备字段或同一部署时钟。下一步[T22单文件资源方案](deadline_20260907/T22_RESOURCE_REVIEW.json)保留两者的可识别性问题，不调整offset追逐目标误差。

本轮没有新增224预测，正式/最新研究候选1F1B仍11.340786%/14.994403%；step、MFU、entry/tail/outer均保持此前版本。现有硬件表不能直接补齐graph可见性，仍需更细的独立数据或代码证据。

唯一下一步：实施 T22：按已冻结单文件资源方案，仅解析source85/rank16、worker33008/GPU0的一份74,694,601字节计数器CSV，分块保留MTLink区间和无效/reset/recovery标记，排除PCIe；核验真实字段、设备/host时钟映射、逐link采样精度及source85覆盖。对graph与无设备事件区间只给有观测支持的字节界限和未知覆盖，不能以按比例分摊或缺行推断连续busy/空闲。先完成一文件资源及语义门槛，不自动扩90/95/100或其他rank，不读目标计数器，不生成等待成本；若只有粗粒度或时钟未识别，如实保存不可辨识结论并另寻独立证据。正式锁与Goal active保持至明早07:00。

| 2026-09-06 15:20 | T22单源计数器区间探查 | 完成40测试后运行登记raw首文件；分别审计硬件/profiler读取 | 不扩迭代/rank |
| 2026-09-06 15:21 | T22：条件字节界限 | 2,933源样本；10个无设备事件窗口TX下界合计416.72MB，但clock未校准/完整上界均缺失 | 40测试/3阶段manifest/161产物通过，无新预测；转缓存时钟敏感性 |

T22已完成：[MTLink采样与条件字节界限图](deadline_20260907/runs/T22/diagnose/source85_counter_interval_probe.svg)、[原始字段/读取范围合同](deadline_20260907/runs/T22/diagnose/source85_counter_field_and_physical_read_contract.json)、[选中采样表](deadline_20260907/runs/T22/diagnose/source85_MTLink_samples.csv.gz)、[graph及设备空档分段](deadline_20260907/runs/T22/diagnose/source85_graph_device_gap_segments.csv)、[逐窗口字节界限](deadline_20260907/runs/T22/diagnose/source85_graph_counter_bounds.csv)、[样本/窗口关联](deadline_20260907/runs/T22/diagnose/source85_graph_counter_sample_intersections.csv.gz)、[验收](deadline_20260907/runs/T22/acceptance.json)。专用Snakemake 4步通过，按预登记SHA仅分块解析74,694,601字节的一份源计数器文件；物理读987,849行、排除718,581条PCIe，269,268条MTLink用于顺序/reset检查，仅输出和分析source85的2,933条MTLink样本。没有重读profiler trace或读取目标计数器。

真实CSV仅有6个字段：host_realtime、link_id、tx/rx delta及mt_timestamp_begin/end；缺失host_mono、gpu_id、ret、link_state、采集iter和单列mt_dt。因此只能保留继承的`legacy_host_end`映射，host/GPU身份依据已冻结文件路径与manifest，未恢复独立设备状态或时钟校准。source85各MTLink链路采样时长中位数4.998830–5.007294ms；旧清单混入PCIe的5.000692ms不能替代本轮逐link结果。2,933条选中样本没有重复、underflow或非正时长，但按旧epoch映射有1,598条参与同link区间重叠，保守排除后1,335条用于字节界限；不把排除样本当作零流量。

96个graph括号各给出完整括号、无非PP设备事件、无任何设备事件三种重叠视图，共288行；5,729个样本/窗口关联可逐行重算。完全落在分析区间内、通过已有字段检查且不重叠的样本只给出**以时钟映射成立为前提的字节下界**；跨边界样本字节只形成已观测样本的cap，不按比例生成精确字节。全部288个窗口都有未覆盖的link×time，完整窗口上界均为null，未知覆盖不填0。source85的无任何设备事件视图中10/96个窗口有正TX下界，合计416,717,312字节；分为反向dispatch 5个、重算combine 3个、前向combine 2个。它支持进一步核验未显示的通信，但不是已证实的EP载荷、连续busy时间或graph服务成本；嵌套视图和跨窗口样本cap不能相加。

40测试、3阶段manifest/161产物SHA、60正式文件和v6813 seal通过；字节界限独立从关联表重算一致。峰值306.17MB、分析77.052s，通过1GiB/120s资源目标，未扩展其他源文件。新测试覆盖uint64 reset及同link恢复、重复样本、整数epoch映射、部分样本不可比例分摊、缺失link的未知上界、重叠排除；首个单测入口缺scripts/w37导入路径，修正测试bootstrap后全40项通过，科学管线首版即PASS。读取审计新增独立的raw_hardware_counter字段，既有raw_trace字段仍只统计profiler解析。

本轮没有拟合成本或新增224预测，正式/最新候选1F1B仍11.340786%/14.994403%，step/MFU及其他阶段保持原结果。下一项[T23缓存时钟敏感性核验](deadline_20260907/T23_PLAN.json)先判断这10个窗口的条件证据是否依赖采样映射，再讨论可独立预测的运行时成分；不能把时钟调整变成误差调参。

唯一下一步：实施 T23：仅复用T22已封存的source85采样表和T21 graph区间，核验设备计数器原始时间轴上的顺序/重叠，并检查10个正字节下界对时钟映射不确定性的稳健性。host_realtime−mt_end的观测范围只能用作显式敏感性范围，不能冒充真实时钟offset界限；不按GPU/target边界拟合offset。保存保守不确定区间、能否确定通信存在及无法确定服务时长的证据，不重新读取raw、不扩其他迭代/rank，不将计数器字节或空档回填成本。完成后据可辨识性决定下一项独立运行时模型证据；正式锁与Goal active保持至明早07:00。

| 2026-09-06 15:37 | T23设备时间轴与offset敏感性 | 设备无重叠，9/96正下界；需拆开样本准入变化 | 保留首版，补同样本消融 |
| 2026-09-06 15:39 | T23-r2固定样本消融 | 固定1,335样本仍9/96；原科学列精确不变 | 42测试/6阶段manifest/324产物通过，无新预测；转T24部署证据 |

T23已完成：[时钟/样本消融图](deadline_20260907/runs/T23/diagnose/source85_counter_clock_sensitivity.svg)、[逐窗口敏感性](deadline_20260907/runs/T23/diagnose/source85_graph_counter_clock_sensitivity.csv)、[整数offset完整分段函数](deadline_20260907/runs/T23/diagnose/source85_graph_counter_offset_lower_curve.csv.gz)、[样本准入offset区间](deadline_20260907/runs/T23/diagnose/source85_counter_sample_containment_offset_ranges.csv.gz)、[字段与假设](deadline_20260907/runs/T23/diagnose/field_contract.json)、[验收](deadline_20260907/runs/T23/acceptance.json)。仅复用T22的572,048字节源缓存，两个版本均Snakemake 4步PASS，无raw解析或目标时长读取。2,933个样本在原文件逐link顺序的设备时间轴上没有倒序或重叠；旧host-end映射却有1,598个重叠参与样本，不能把这种映射重叠解释为设备重复计数。

host_realtime−mt_end的观测跨度为644,349ns，即0.644349ms；本轮对统一常数offset的所有整数ns情景精确求完全包含样本的字节下界最小值。这个观测范围**不是已校准的真实offset界限**，统一offset、实际查询延迟和漂移仍未验证。先固定T22的1,335个有效样本，原10/96个正TX下界变为9/96，逐窗口下界合计416,717,312→315,325,672字节；单独恢复设备时间轴不重叠的全部2,933样本后，仍为9/96，合计594,346,144字节。较大的字节数部分来自样本重新准入，不能归功于时钟调整或解释成预测改善。

失去全范围正下界的是source85/rank16、MB0、layer4的反向dispatch（graph event59749）：旧下界6,769,152字节，但某些offset情景下下界为0；不能因此断言无通信。其余9个为反向dispatch 4、重算combine 3、前向combine 2。这些仍只是所选缓存与时钟假设下的通信存在证据，缺失覆盖、真实时钟、EP载荷归属和独立graph服务时长均未解决；不新增服务成本或因果边。

r2补充同样本消融，r1全部原科学列精确不变。42测试、6阶段manifest/324产物SHA、60正式文件和v6813 seal通过；另独立在953个分段的1,906个首尾整数offset直接重算包含关系与TX/RX，全部一致。最终峰值210.62MB、分析23.976s，图例已核验。正式/最新研究1F1B仍为11.340786%/14.994403%，没有新的224预测，step/MFU和其他阶段维持已有结果。复现使用`wave23_source_counter_clock_review.json`，以新run-id运行专用pipeline；历史版本与事前方案保留。

下一项[T24源部署证据准入](deadline_20260907/T24_RESOURCE_REVIEW.json)转向已定位的训练启动日志与脚本：脚本声明DeepEP wheel 1.1.0+d7577a6、ACE等运行时开关，但尚未证实实际部署。先核验源侧实际配置，再决定是否有独立特征支撑成本迁移，不继续以不可辨识的计数器字节修补模型。

唯一下一步：实施 T24：先用专用source_only Snakemake准入source256的一份537KB训练启动日志前缀、已定位启动脚本及旧scenario静态字段，核验实际参数、DeepEP构建/运行时开关与采集身份；脚本默认值、日志实际值和无法证实的部署状态分列，不执行脚本，不继承旧拟合参数。先完成源侧证据门槛，再决定是否按seal后独立方案核查目标配置或构建可预测的运行时成本；不将目标时序、计数器字节或观测等待回填成本，正式锁和Goal active保持至明早07:00。

| 2026-09-06 16:11 | T24源训练日志静态准入 | 48字段各8次相同，8项scenario匹配、6项未见；精确后端构建未验证 | 45测试/3阶段manifest/162产物通过，无新预测；转T25目标配置 |

T24已完成：[源部署核验表](deadline_20260907/runs/T24/diagnose/SOURCE_DEPLOYMENT_REVIEW.md)、[逐字段/行号/SHA证据](deadline_20260907/runs/T24/diagnose/source_startup_static_observations.csv)、[启动脚本声明](deadline_20260907/runs/T24/diagnose/source_launch_script_claims.csv)、[字段缺口](deadline_20260907/runs/T24/diagnose/source_deployment_field_readiness.csv)、[读取合同](deadline_20260907/runs/T24/diagnose/field_contract.json)、[验收](deadline_20260907/runs/T24/acceptance.json)。专用Snakemake 4步PASS；单source256、worker33008/node2训练日志537,378字节先SHA核验，仅解析前262,144字节。原始profiler/硬件计数器均未读，目标时长仍仅SHA，日志没有复制。

得到384条白名单静态观测、48个字段，每个字段均出现8次且字面值一致；记录位于7行，其中第103行含两次同字段出现。它们不能独立证明8个rank的身份。实际配置确认PP16/CP2/EP8/TP1、60层、full/block重算4层、DeepEP启用、num_sms=20、flex dispatcher及关闭EP/shared-expert overlap；sequence_parallel=False、mtp_num_layers=None、FP8=None。脚本虽写了sequence-parallel开关、文件名含MTP1，均不能代替实际配置。14项scenario静态对照中8项一致、6项未观察，无可见冲突；未观察项保留未知，不从脚本或别名默认补值。

该受限前缀没有提取到ACE环境开关、DeepEP精确构建、GPU型号/时钟或部署代码revision；源码中的wheel 1.1.0+d7577a6安装请求不是安装成功证据。厂商[ACE图执行说明](https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/history_version/v5.1.0/libraries/mccl/mccl_over_ace/)描述了独立搬运引擎及含同步/拷贝/原子的MUSA Graph，只支持继续调查graph可见性的机制假设，不能证明本次自定义DeepEP走同一路径或提供服务时长。检索边界保存在[T24公共来源核验](deadline_20260907/T24_PUBLIC_SOURCE_REVIEW.json)。

45测试、3阶段manifest/162产物SHA、60正式文件和v6813 seal通过；峰值151.85MB、分析0.03695s，未扩日志范围。前缀内未命中训练指标停止行，不声称物理buffer从未包含源时序文本；只分析静态白名单、不拟合成本。验收首版错误地预期8个物理行，修正为7行/每字段8次出现，科学产物未修改。没有新增224预测，正式/最新研究1F1B仍11.340786%/14.994403%，step/MFU及其他阶段保持已有结果。

下一项[T25目标配置事后核验](deadline_20260907/T25_RESOURCE_REVIEW.json)固定T24解析器和源字段，只在已有全局seal后处理同口径目标日志前缀；目标产物永不回收为源拟合输入。先判断有无真实配置差异，再讨论可预测的成本迁移。

唯一下一步：实施 T25：按新冻结的单文件资源方案，在v6813 seal及T24源字段/解析器SHA核验后，只解析target224、worker33020的一份498KB训练日志前256KiB；沿用源侧白名单和未知/冲突规则，在evaluator_only比较实际配置。源表从T24纯源产物准确准入，禁止从目标诊断回收拟合输入；不重扫trace/计数器，不以配置缺失或目标时长生成补偿。先判断两次部署是否存在可证实的静态差异，再选择可预测成本假设；正式锁和Goal active保持至明早07:00。

| 2026-09-06 16:23 | T25目标训练日志静态对照 | 46字段相同、2为已建模差异，43项无法比较；未识别新修正依据 | 47测试/3阶段manifest/165产物通过，无新预测；转T26A源设备模型 |

T25已完成：[源/目标配置对照](deadline_20260907/runs/T25/diagnose/evaluator_only/DEPLOYMENT_COMPARISON.md)、[全部白名单比较](deadline_20260907/runs/T25/diagnose/evaluator_only/source_target_static_field_comparison.csv)、[目标逐字段/行号/SHA证据](deadline_20260907/runs/T25/diagnose/evaluator_only/target_startup_static_observations.csv)、[目标scenario对照](deadline_20260907/runs/T25/diagnose/evaluator_only/target_declared_static_comparison.csv)、[读取合同](deadline_20260907/runs/T25/diagnose/field_contract.json)、[验收](deadline_20260907/runs/T25/acceptance.json)。先核验T24完整源阶段、解析器SHA和v6813全局seal，再只读取target224、worker33020/node2的一份497,890字节日志前262,144字节；专用Snakemake 4步PASS，没有raw trace/计数器扫描。

目标同样得到384条白名单观测、48个字段，各出现8次且值一致，分布在7个物理行。源/目标46个字段相同，两项差异仅为num_layers 60→52和PP16→14，已被现有场景建模；另43项因至少一侧缺少观测而无法比较。目标14项scenario检查中8项一致、6项未观察，无可见冲突。full/block重算、DeepEP启用/20个SM配置、EP overlap关闭等可见值一致，不能据此断言GPU/CPU状态、后端版本或全部环境相同；ACE开关和精确DeepEP构建仍未验证。

47测试、3阶段manifest/165产物SHA、60正式文件、原源产物与解析器及v6813 seal通过；91项字段比较独立重算一致。峰值154.43MB、分析0.05153s。受限前缀未命中训练指标停止行，读取合同明确buffer可能包含时序文本，但没有提取或拟合目标时长。目标及混合表始终位于evaluator_only，不能当作新的源成本输入。

本轮没有找到可支持成本修正的新配置差异，没有新224预测或正式图修改；正式/最新研究1F1B仍11.340786%/14.994403%，step/MFU及其他阶段沿用现有封存结果。下一项[T26A已封存设备模型重载](deadline_20260907/T26A_PLAN.json)转回源侧预测模型：先精确恢复T19 source85/90参数和全部旧预测，再单独检验目标在观测CPU提交条件下的设备完成误差，定位可见设备成本迁移与CPU/隐藏graph等待之间的剩余缺口。局部条件预测不等于全局1F1B预测。

唯一下一步：实施 T26A：通过专用source_only Snakemake准入T19已封存且仅source85/90拟合的设备队列参数、条件特征与旧预测，核验局部seal、代码SHA及读取审计，按原四种方法重载回放并逐行匹配全部167,056条预测；不读取目标数据或重新拟合。通过后单独登记T26B，冻结相同成本在目标缓存上检验CPU提交条件下的设备完成误差，区分计算/队列成本迁移与CPU/隐藏graph等待缺口；条件预测不能冒充全局224卡1F1B改善，正式锁和Goal active保持至明早07:00。

| 2026-09-06 16:33 | T26A源设备模型重载 | 2,772参数原样、167,056预测CSV精确一致 | 49测试/3阶段manifest/170产物通过；转T26B条件迁移 |

T26A已完成：[逐轮/方法重载核验](deadline_20260907/runs/T26A/diagnose/source_device_queue_reload_per_iteration.csv)、[重载局部seal](deadline_20260907/runs/T26A/diagnose/source_device_queue_reload_seal.json)、[完整重载预测](deadline_20260907/runs/T26A/diagnose/source_device_queue_reloaded_predictions.csv.gz)、[原样源参数](deadline_20260907/runs/T26A/diagnose/source_device_queue_parameters.csv.gz)、[字段合同](deadline_20260907/runs/T26A/diagnose/field_contract.json)、[验收](deadline_20260907/runs/T26A/acceptance.json)。只准入T19的9个封存文件共4,982,712字节，专用Snakemake 4步PASS；新增显式source_model_stages准入，不放宽原先只接收未拟合观察数据的规则。

2,772行成本全部来自原source85/90参数；source95/100仍是历史已暴露的增量验证。重载四层cost查找表后，41,764个设备事件×四种固定方法的167,056条预测，时间、前驱、cost id和全部字段均与旧局部seal一致，解压CSV文本也完全相同。参数和旧端点指标逐字节不变；四个源模型/关联模块与T19代码快照SHA一致。没有重新拟合、目标语义读取或raw解析。

49测试、3阶段manifest/170产物SHA、60正式文件和v6813 seal通过；峰值486.60MB、分析5.718s。新测试验证重载等价、错误拟合范围/成本哈希拒绝、GPU真值字段拒绝及目标角色不能作为源参数。原四种方法保留，包括T19已知在F阶段严重退化的均值版本；没有根据目标选择版本。

这只是已有局部模型的精确恢复，没有新增全局224预测：正式/最新研究1F1B仍11.340786%/14.994403%，step/MFU和其他阶段未变。下一项[T26B目标条件迁移检验](deadline_20260907/T26B_PLAN.json)固定这些成本，在实际CPU提交和算子元数据条件下评估目标设备完成误差，检验可见设备成本迁移；不将局部条件端点冒充CPU F/B或完整1F1B预测。

唯一下一步：实施 T26B：先核验T26A精确重载门槛、T19局部seal和v6813全局seal，再仅用T17B已封存目标rank16四轮缓存，沿用固定源设备模型与CPU/设备关联代码。保留四种方法和既有cost层级，输出目标条件特征、局部预测seal、逐设备/逐F-B末端误差、成本覆盖与计算/队列残差分账；未覆盖成本明确失败，不按目标改stream映射或补零。实际CPU提交、算子/shape/phase为诊断条件，所有目标产物evaluator_only，不能称全局1F1B、Step或MFU改善；据结果选择下一个独立成本/运行时假设，正式锁与Goal active保持至明早07:00。

| 2026-09-06 16:46 | T26B目标条件队列迁移 | F改善/B退化，312个API返回前启动kernel；无全局改善 | 50测试/6阶段manifest/379产物通过；转T27A源释放代理候选 |

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

唯一下一步：实施 T27A：仅复用T14B源缓存，在独立模块中将全部设备的条件释放代理改为CPU API开始时刻；按原四层成本键和四种方法只拟合85/90，封存后检验95/100，并与旧API结束代理同口径消融。核验旧预测复现、时长参数不变、增量GPU真值不进入特征或拟合，分别报告启动/驻留/末端误差；通过源侧验收后再登记T27B目标条件迁移。实际CPU提交仍是条件，不能称全局224卡1F1B改善，正式锁与Goal active保持至明早07:00。

| 2026-09-06 17:00 | T27A源API开始代理消融 | 无实质改善；2,772行驻留成本/键/依赖不变，旧167,056预测精确 | 51测试/3阶段manifest/214产物通过；转T27B封存后目标条件消融 |

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

唯一下一步：实施 T27B：核验T27A源侧验收、两版局部seal和固定代码后，只在T17B目标缓存上比较旧API结束代理与新API开始代理；重载两份85/90参数，保留四种方法，先封存条件预测再评分。核验旧对照与T26B精确一致、目标成本覆盖、释放反例数及逐迭代/启动/驻留/F-B设备末端误差，不回填目标成本；据结果定位下一项源侧可检验的运行时缺口。局部CPU条件预测不称全局1F1B改善，正式锁与Goal active保持至明早07:00。

| 2026-09-06 18:10 | T27B目标API开始代理条件消融 | 末端小幅改善但F/B内事件退化；释放反例312→0，无成本更新 | 51测试/3阶段manifest/217产物通过；转T28A纯源终端选择 |

## T27B：两种释放代理的目标条件消融

[对照图](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_comparison.svg)、[设备末端指标](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_endpoint_metrics.csv)、[F/B内逐事件分账](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_metrics_by_scope.csv)、[逐迭代结果](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_per_iteration.csv)、[末端事件选择](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_endpoint_event_selection.csv)、[释放代理反例](deadline_20260907/runs/T27B/diagnose/evaluator_only/target_release_proxy_violations.csv)、[验收](deadline_20260907/runs/T27B/acceptance.json)。专用Snakemake4步PASS，只复用T27A源模型和T17B目标缓存，没有raw重扫。

两份源参数各2,772行、四种方法固定；33,816个目标事件产生270,528条条件预测，旧候选与T26B特征/预测解压文本和末端指标精确一致。API开始代理把释放边界反例312→0；目标中位数F/B设备末端MAE由0.846204/11.420797降至0.760596/11.411286ms，但22,692个F/B归属非PP事件的启动/完成MAE由9.489640/9.590851升至9.526528/9.628238ms，驻留时长MAE固定0.230005ms。语义边界更安全不等于预测整体改善，不提升全局版本。

观察设备末端与预测完成最大事件可来自不同stream和算子，phase包络接近可能是跨stream最大值抵消。下一轮返回纯源拟合/增量切分，事前登记终端事件选择规则；不得依据这次目标结果增删边或补等待。51测试、3阶段manifest/217产物、60正式文件及全局seal通过；r1–r5失败及r6通过均保留。正式/最新全局研究1F1B仍11.340786%/14.994403%，本轮无Step/MFU和其他阶段新预测。

周报可用两句：两种源侧封存释放代理在224卡条件迁移中只带来F/B设备末端0.085609/0.009511ms的小幅改善，而F/B内逐事件完成MAE反而由9.590851ms升至9.628238ms，未形成全局1F1B改善。API开始代理消除了312个已观察边界矛盾，但有效释放时刻、跨stream终端选择、独立CPU入口、隐藏graph、跨rank迁移和未暴露数据仍未验证。

唯一下一步：实施 T28A：仅准入T27A纯源两版seal、特征和预测，先用source85/90登记观察终端算子/stream身份以及固定的终端选择候选，再封存规则与参数；随后只在source95/100检验端点身份命中率、末端MAE/偏差和逐事件误差。控制项保留“预测完成最大值”，语义项只能选择拟合窗口稳定出现且运行时归属可复核的终端事件；缺失或验证退化必须如实拒绝。此轮不读目标缓存、不修改正式拓扑、不插等待；只有源增量证据通过后才单独登记T28B目标条件迁移，局部条件结果不称全局1F1B改善，文件Goal持续至明早07:00。

| 2026-09-06 18:28 | T28A纯源终端事件语义 | 32窗口真实末端均为stream0 copy；F可解释，B最低误差仍为错误事件抵消 | 53测试/3阶段manifest/193产物通过；转T28B封存后目标核验 |

## T28A：纯源终端事件身份与端点选择

[语义对照图](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_semantics.svg)、[终端身份参数](deadline_20260907/runs/T28A/diagnose/source_terminal_identity_parameters.csv)、[源拟合选择参数](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_selection_parameters.csv)、[增量指标](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_metrics.csv)、[逐窗口身份核验](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_event_selection_validation.csv.gz)、[预测seal](deadline_20260907/runs/T28A/diagnose/source_terminal_endpoint_prediction_seal.json)、[验收](deadline_20260907/runs/T28A/acceptance.json)。source85/90拟合终端身份和选择规则，640行端点预测封存后才接入95/100完成真值；目标时序未读。

源拟合与增量的32个F/B窗口观察末端均为stream0 `aten::_copy_from`。前向源拟合选择在95/100达到8/8身份命中和旧/新代理0.023677/0.013204ms MAE；反向最低包络0.604612/0.605248ms的身份命中0/8。若强制真实终端身份并用独立延迟，反向为0.676671/0.676665ms、身份8/8。约0.072ms的差价量化了当前反向“时间接近”和“事件归因正确”的冲突。

T27A全部预测最大值指标精确复现；改变95/100 GPU真值不改变5,544行成本、身份、选择或预测。53测试、3阶段manifest/193产物、60正式文件及全局seal通过；峰值699.55MB、分析33.157s，无raw扫描。无CPU F/B、全局1F1B、Step/MFU新预测。

周报可用两句：纯源终端核验确认32个F/B窗口的可见设备末端均为stream0 `aten::_copy_from`，前向源拟合规则在95/100达到8/8身份命中和0.013–0.024ms MAE。反向最低端点MAE约0.605ms仍由错误的stream15事件形成，强制真实终端身份为0.677ms且8/8命中；下一轮封存这项精度—可解释性权衡后做224卡开发诊断，独立CPU入口、隐藏graph、跨rank和全局1F1B仍未验证。

唯一下一步：实施 T28B：事前固定T28A的终端身份、源拟合组合、强制语义终端独立延迟和CPU标注尾部三类候选；核验T28A seal及T27B两份目标条件预测seal后，只复用已封存目标表计算四轮逐窗口端点时间与事件身份。预测文件再次先seal后评分，旧T27B端点控制必须精确；同时报告“最低时间误差”和“真实终端身份命中”两条轴，不按目标结果更换算子、stream、方法、边或等待。此轮仍是观测CPU提交条件下的224开发诊断；不称全局1F1B、Step或MFU改善，文件Goal持续至明早07:00。

| 2026-09-06 18:57 | T28B目标终端语义条件迁移 | API开始语义终端F/B MAE 0.624367/10.204995ms，真实事件24/24；仍依赖观测CPU提交 | 54测试/3阶段manifest/193产物通过；r6/r7科学产物字节精确，转T29A纯源CPU提交分解 |

## T28B：源固定终端语义的目标条件迁移

[对照图](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_transfer.svg)、[总体指标](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_metrics.csv)、[逐迭代](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_per_iteration.csv)、[终端身份](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_observed_terminal_identity_counts.csv)、[预测seal](deadline_20260907/runs/T28B/diagnose/evaluator_only/target_terminal_endpoint_prediction_seal.json)、[验收](deadline_20260907/runs/T28B/acceptance.json)。四类候选事前固定，T28A源身份/参数和T27B目标条件表按SHA准入；336行预测先seal，目标真值突变检查精确，再连接终点时间与身份。

24个目标F/B窗口均以stream0 `aten::_copy_from`为真实可见末端。API开始语义独立延迟相对同口径预测最大控制，把F/B终点MAE从4.604437/10.220776降到0.624367/10.204995ms，身份从1/12、9/12升至12/12、12/12。局部改进有事件语义支持，但输入仍含观测CPU提交和F/B标注结束；不提升全局版本。

r1浮点比较拒绝；r2通过但缺真值突变门；r3通过后独立复核发现gzip时间戳；r4/r5固定压缩后又发现SVG日期和随机ID；r6/r7最终12项科学表、seal和图字节精确。54测试、3阶段manifest/193产物及60正式文件通过，无raw、参数、图、Step或MFU更新。

周报可用两句：源固定stream0 `aten::_copy_from`终端在224卡24个F/B开发窗口全部命中，并将API开始候选的F/B设备终点MAE由4.604437/10.220776ms降至0.624367/10.204995ms，首次同时改善局部时间误差和事件归因。该结果仍以观测CPU提交和F/B标注结束为条件，尚未形成全局1F1B、Step或MFU改善；target90前向异常、反向迭代波动、隐藏graph、跨rank和未暴露数据仍未验证。

唯一下一步：实施 T29A：只准入T14B纯源缓存、T28A源终端seal及其精确代码，分解每个source F/B窗口的CPU标注起止、终端`aten::_copy_from` API起止和设备起止；先审计这些区间与现有F/B、wrapper、PP、计算、通信及等待成本是否重叠。只允许从调度时可获得的前序边界预测CPU提交及语义终端完成，不得把实际CPU标注结束或GPU终点作为自由运行输入；source85/90拟合并封存，source95/100增量验证。此轮不读目标时序、不改正式图；只有源侧端到端误差和重复计时门通过后，才登记T29B目标封存回归，文件Goal持续至明早07:00。

| 2026-09-06 19:21 | T29A源终端提交/完成分解 | 95/100 F/B端点MAE 2.726681/6.659497ms；B尾部与后续PP重叠至少99.921487% | 55测试/3阶段manifest/195产物通过；拒绝追加和既有替换，不运行T29B |

## T29A：源CPU终端提交、设备完成与PP重叠

[分解图](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition.svg)、[预测参数](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_parameters.csv)、[增量指标](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_metrics.csv)、[逐迭代结果](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_per_iteration.csv)、[区间摘要](deadline_20260907/runs/T29A/diagnose/source_terminal_PP_interval_summary.csv)、[逐窗口账本](deadline_20260907/runs/T29A/diagnose/source_terminal_PP_interval_ledger.csv.gz)、[预测seal](deadline_20260907/runs/T29A/diagnose/source_terminal_decomposition_prediction_seal.json)、[验收](deadline_20260907/runs/T29A/acceptance.json)。source85/90拟合的160行候选先seal，之后才接入95/100的CPU结束、终端API/GPU和PP边界；真值突变门精确。

分解候选在source95/100的F/B端点MAE为2.726681/6.659497ms，CPU提交MAE为2.728800/6.506944ms。全部32个终端API在F/B标注内提交；B的16个设备终点全在下一send_backward内，正尾部与PP重叠至少99.921487%，四段区间守恒0ns。现有v6813的约93ms sender-readiness加约5ms返回尾部已表达该边界，新增会重复计时。

结构保持的替换也未通过：T16的source95/100同口径局部MAE中，stream0设备端点0.663167ms差于现有PP 0.557697ms。T29A不读目标、不改正式图、不产生全局1F1B/Step/MFU预测；r1/r2十二项科学产物字节精确。

唯一下一步：实施 T30A：只用T14B纯源rank16缓存审计CPU终端提交偏移的可辨识性。事前限定离线调度可获得的phase、microbatch role、静态前序依赖类型和可由训练图确定的CPU操作签名，禁止使用当轮CPU结束、API/GPU时刻或目标数据；source85/90拟合并seal，source95/100验证。先证明新增特征在迭代间有信息增益且不与现有F/B成本重复；若无增益，形成不可辨识结论并转向其他成本项，不调目标误差、不改正式图，文件Goal持续至明早07:00。

| 2026-09-06 19:32 | T30A源CPU提交可辨识性 | F/B操作计数签名在32窗口固定；扩展等价类6→6、预测精确不变 | 56测试/3阶段manifest/199产物通过；拒绝静态签名迁移，转T31A正式图全rank成本切分 |

## T30A：CPU提交静态签名没有额外信息

[可辨识性图](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_identifiability.svg)、[期望签名](deadline_20260907/runs/T30A/diagnose/source_CPU_expected_static_signatures.csv)、[提交指标](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_metrics.csv)、[逐迭代](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_per_iteration.csv)、[签名验证](deadline_20260907/runs/T30A/diagnose/source_CPU_static_signature_validation.csv)、[预测seal](deadline_20260907/runs/T30A/diagnose/source_CPU_submission_prediction_seal.json)、[验收](deadline_20260907/runs/T30A/acceptance.json)。source85/90参数和期望标签先seal，验证时序与观察标签后接入；突变门精确。

F/B每窗口分别固定为2,802/7,738个、79/179类CPU操作，根操作和前序PP类型也逐phase不变，source95/100的16窗口全部匹配。加入这些静态字段没有增加phase/role的6个预测类，提交预测和F/B MAE 2.728875/6.506913ms均完全不变。该路线不能解释迭代运行时漂移，也不读目标或生成全局预测；r1/r2十三项科学产物字节精确。

| 2026-09-06 19:56 | T31A v685成本切分/绑定 | recent90计算槽MAE改善3.644%，PP/entry退化15.864%/60.563%；source新计算绑定0行 | 58测试/3阶段manifest/208产物通过；拒绝T31B，转T32A source gap桥接审计 |

## T31A：v685成本切分与source replay绑定

[成本图](deadline_20260907/runs/T31A/diagnose/source_fullrank_cost_split.svg)、[分项指标](deadline_20260907/runs/T31A/diagnose/source_component_metrics.csv)、[逐迭代](deadline_20260907/runs/T31A/diagnose/source_component_per_iteration.csv)、[绑定审计](deadline_20260907/runs/T31A/diagnose/v685_source_compute_binding_audit.json)、[分账表](deadline_20260907/runs/T31A/diagnose/component_accounting_ledger.csv)、[预测seal](deadline_20260907/runs/T31A/diagnose/source_fullrank_cost_prediction_seal.json)、[验收](deadline_20260907/runs/T31A/acceptance.json)。两套source85/90分项预测先seal，验证真值突变门精确；v685四轮参数仅作泄漏控制。

source95/100上，recent90只把物理计算槽MAE从0.620091降至0.597494ms，PP wall从4.537392升至5.257209ms，entry从67.512658升至108.400522ms。计算输入只覆盖16个stage各自lane0的代表rank；PP才覆盖256 rank。v685目标图更新144个计算节点，source replay没有计算键、计算分量或更新调用，新计算绑定0行；它基于旧local gap加新PP得到的805.920ms reconciliation不能闭合新计算验证。T31B不登记，无目标、图、全局1F1B、Step/MFU变化。r4/r5十九项科学输出字节精确，r1–r3失败保留。

唯一下一步：实施 T32A：只用冻结派生表审计v67/v685 source图的时钟原点、`local_gap`生成规则与v61物理计算槽的逐键可连接性；先列出每个gap包含的计算、通信、CPU/framework与等待口径，再用source85/90建立无重复计时的替换分账，在95/100验证局部守恒和全局1F1B。计算观测仅有各stage lane0，未经证据不得外推其余lane；`unclassified_calibration_ns`不得直接当作计算或等待。只有兼容source图闭合且增量全局/局部同时通过，才另登记目标候选；不读目标时序、不改正式锁，文件Goal持续至明早07:00。

| 2026-09-06 20:29 | T32A source gap桥接与重复计时审计 | lane0分账和物理计算逐条闭合；全rank gap含95/100，拒绝留出验证与目标提升 | 61测试/3阶段manifest/211产物通过；r3/r4科学产物精确，转T33A全rank phase回放 |

## T32A：局部无重复计时成立，全rank留出桥接不成立

[分账图](deadline_20260907/runs/T32A/diagnose/source_gap_bridge.svg)、[区间指标](deadline_20260907/runs/T32A/diagnose/source_gap_partition_metrics.csv)、[phase包络指标](deadline_20260907/runs/T32A/diagnose/source_phase_envelope_metrics.csv)、[物理计算连接](deadline_20260907/runs/T32A/diagnose/physical_compute_segment_join.csv.gz)、[v54→v67→v685 lineage](deadline_20260907/runs/T32A/diagnose/v67_v685_gap_lineage.csv.gz)、[所有权合同](deadline_20260907/runs/T32A/diagnose/component_ownership_contract.csv)、[预测seal](deadline_20260907/runs/T32A/diagnose/source_gap_partition_prediction_seal.json)、[验收](deadline_20260907/runs/T32A/acceptance.json)。source85/90的局部分区参数和四轮预测先seal，source95/100真值突变不改变参数或预测。

82,695个lane0语义区间逐条满足`compute_only + communication_only + overlap + non_gpu = wall`；59,751条有计算的区间与v6.1物理槽总量一一精确匹配，其余22,944条计算为0。source95/100中，两轮中位数的单区间MAE为0.690063ms；聚合到F/B局部链为15.344871/13.357146ms。B链在PP annotation之后平均仍有75.988631ms，说明该尾部需要与PP send所有权联合处理，不能直接加成新等待。

v54的4,664个lane0 gap参数均可由九轮区间中位数精确复算；内部stage池化再乘v67的九轮microbatch shape后，74,624个gap都在1ns内复现。v685逐值继承这些gap，`compute_work_ns`合计为0、`unclassified_calibration_ns`合计593,661.079ms。由于v54/v67参数使用source60–100，已经包含95/100，而且物理分解只有16个lane0代表rank，现有全rank source replay不能充当source95/100留出验证。T32A因此只采纳局部解释账本，不登记目标候选或新的全局1F1B。

专用Snakemake4步PASS；61测试、3阶段manifest/211产物及60正式文件通过，28个冻结输入79,490,013字节、无raw解析。r3/r4的19项科学表、seal及图字节精确；r1/r2的merge cardinality和遗漏输入失败保留。正式v685、拓扑、目标1F1B、Step和MFU均不变。

唯一下一步：实施 T33A：用全256 rank的source PP F/B观测，只在85/90拟合阶段包络成本并预先seal；按训练调度代码和正式拓扑锁建立自由运行的全rank 1F1B phase图，随后在95/100验证逐phase、warmup/steady/cooldown及全局包络。保留median85/90和recent90消融，不使用含95/100的v54/v67 gap、目标时序或lane0计算外推；只有source增量全局和局部同时闭合才登记T33B，文件Goal持续至明早07:00。

| 2026-09-06 20:48 | T33A全rank source phase回放 | median85/90在95/100全局MAPE 0.332811%；四个recency/上界消融均未改善 | 64测试/3阶段manifest/214产物通过；不登记T33B，转T34A调度收缩审计 |

## T33A：全256 rank source phase自由运行验证

[总览图](deadline_20260907/runs/T33A/diagnose/source_phase_replay.svg)、[全局逐轮结果](deadline_20260907/runs/T33A/diagnose/source_phase_global_iteration_results.csv)、[全局指标](deadline_20260907/runs/T33A/diagnose/source_phase_global_metrics.csv)、[warmup/steady/cooldown指标](deadline_20260907/runs/T33A/diagnose/source_phase_metrics.csv)、[节点绑定](deadline_20260907/runs/T33A/diagnose/source_phase_cost_bindings.csv.gz)、[边表](deadline_20260907/runs/T33A/diagnose/source_phase_edges.csv.gz)、[因果分账](deadline_20260907/runs/T33A/diagnose/source_phase_diagnostic_shapley.csv)、[seal](deadline_20260907/runs/T33A/diagnose/source_phase_prediction_seal.json)、[验收](deadline_20260907/runs/T33A/acceptance.json)。五套19,714节点/26,752边的图全部由固定`non-interleaved`调度生成，覆盖256 rank、16 stage、16 lane和每轮2,048个F/B phase；候选拓扑SHA一致，且独立图明确不替换正式v684锁。

source85/90中位图在95/100的全局1F1B MAPE为0.332811%、MAE74.057ms，精确复现T01历史开发参考。recent90全量为0.412295%，仅phase为0.426312%，仅runtime为0.332841%，phase取85/90上界为2.688238%，没有新方法通过全局提升门。中位图的验证phase duration MAE为warmup 15.854377ms、steady 16.692988ms、cooldown 15.478955ms；它们是逐phase误差，不可相加为step误差。

source95/100真值在预测seal后做三类替换。两轮平均actual−prediction 34.719727ms由Shapley分为phase wall +21.204959ms、PP readiness/completion +13.427048ms、本地runtime +0.087719ms；95与100的phase项分别为−49.819/+92.228ms，方向不稳定。三类全部替换时两轮全局包络误差均为0ns，证明图可表示，但不能把观察替换当预测。T33B不登记，也不使用v54/v67含95/100的gap。

专用Snakemake4步PASS；64测试、3阶段manifest/214产物、60正式文件通过，15个冻结输入30,248,941字节，无raw或目标时序。r1/r2十九项科学表、seal及图字节精确，峰值476.97MB。无正式图、目标1F1B、Step或MFU变化。

唯一下一步：实施T34A。固定T33A中位参数，构造PP16/MB4、PP16/MB3、PP14/MB4、PP14/MB3四个静态调度图，在目标时序接入前seal；用两因素Shapley拆解PP深度与microbatch数导致的预测收缩和关键路径换路。封存后才以目标开发数据比较实际收缩，不拟合倍率或残差；若差额不能由静态调度解释，记录为缺少源侧代理的跨规模运行时状态，文件Goal持续至明早07:00。

| 2026-09-06 21:00 | T34A静态调度收缩审计 | PP16/MB4→PP14/MB3预测收缩3,929.446ms，实际只收缩655.738ms，过度收缩3,273.708ms | 66测试/3阶段manifest/222产物通过；拒绝目标倍率修正，转T35A运行时代理与在线前缀可辨识性 |

## T34A：PP深度与microbatch静态收缩不足以解释224卡

[四场景图](deadline_20260907/runs/T34A/diagnose/schedule_contraction.svg)、[场景汇总](deadline_20260907/runs/T34A/diagnose/schedule_scenario_summary.csv)、[Shapley分解](deadline_20260907/runs/T34A/diagnose/schedule_contraction_shapley.csv)、[关键路径账本](deadline_20260907/runs/T34A/diagnose/schedule_critical_path_ledger.csv.gz)、[目标phase指标](deadline_20260907/runs/T34A/diagnose/target_development_phase_metrics.csv)、[逐迭代结果](deadline_20260907/runs/T34A/diagnose/target_development_iteration_results.csv)、[预测seal](deadline_20260907/runs/T34A/diagnose/schedule_contraction_prediction_seal.json)、[验收](deadline_20260907/runs/T34A/acceptance.json)。四个场景均使用T33A的source85/90中位成本，在目标时序接入前生成并seal；节点、边和关键路径由同一非交错调度代码产生，独立候选不替换正式锁。

PP16/MB4、PP16/MB3、PP14/MB4、PP14/MB3的预测1F1B分别为22,182.789、20,918.157、19,500.667、18,253.343ms。总预测收缩3,929.446ms，PP深度和microbatch两因素Shapley分别为2,673.468/1,255.978ms；但source95/100实际均值到目标四轮实际均值只收缩655.738ms，即实际为静态预测的16.688%，留下3,273.708ms过度收缩。

目标phase-transfer开发MAPE为15.340746%，正式v685为11.340786%；对应Profiler/训练Step/MFU相对误差为13.984092/14.101418/16.418413%，也均差于正式的10.369528/10.718041/12.006689%。逐phase平均偏差−96.428ms，warmup/steady/cooldown为−58.612/−84.418/−138.247ms，说明缺口随调度区段变化，不能用一个无源侧依据的目标倍率回填。66测试、3阶段manifest/222产物、60正式文件通过，r2/r3二十四项科学输出字节精确；无目标参数更新或拓扑变化。

唯一下一步：实施T35A。先审计冻结的runtime-readiness合同和静态工作负载字段，证明哪些字段可在执行前获得及其source增量信息量；若静态字段仍不可辨识，则只登记运行中首批F观测可用的在线前缀候选。算法、源参数和特征投影必须先seal，再在source95/100验证；通过后才可另行做目标开发评分，且必须与冷启动v685分栏，不把目标前缀或残差包装成独立输入。

| 2026-09-06 21:31 | T35A冷启动能力与在线方向前缀 | 冷启动仍缺2项输入；在线主候选source MAPE0.316924%，目标开发MAPE1.774166%，平均57.845%后可用 | 69测试/3阶段manifest/237产物通过；目标倍率全超source范围，不替换v685，转T36A更早的source拟合收缩前缀 |

## T35A：冷启动不可辨识与在线1F1B nowcast

[方法图](deadline_20260907/runs/T35A/diagnose/online_prefix_nowcast.svg)、[冷启动能力审计](deadline_20260907/runs/T35A/diagnose/cold_start_runtime_capability_audit.json)、[source指标](deadline_20260907/runs/T35A/diagnose/source_online_metrics.csv)、[target逐轮结果](deadline_20260907/runs/T35A/diagnose/target_step_iteration_results.csv)、[target汇总](deadline_20260907/runs/T35A/diagnose/target_step_metrics.csv)、[前缀因子](deadline_20260907/runs/T35A/diagnose/target_online_factors.csv)、[可用时点](deadline_20260907/runs/T35A/diagnose/target_online_prefix_cutoffs.csv)、[F/B Shapley](deadline_20260907/runs/T35A/diagnose/target_online_direction_shapley.csv)、[节点](deadline_20260907/runs/T35A/diagnose/target_online_nodes.csv.gz)、[边](deadline_20260907/runs/T35A/diagnose/target_online_edges.csv.gz)、[验收](deadline_20260907/runs/T35A/acceptance.json)。v69合同要求的动态router token矩阵和独立runtime readiness profile均缺失，静态stage表不能识别迭代状态，因此没有新的冷启动修正。

在线主候选在当轮全stage首F和后半PP stage首B完成后，分别以20% winsor均值更新既有F/B phase wall；PP readiness、本地runtime、节点和边不变。source95/100是已看过的开发确认：1F1B MAPE 0.332811%→0.316924%，phase MAE 15.827028→14.373229ms。源seal后才投影目标前缀，目标非前缀扰动不改变预测。

目标四轮在线1F1B MAPE为1.774166%，逐轮1.352707/1.948060/2.255827/1.540070%，平均偏差−382.585ms；正式v685冷启动为11.340786%。Profiler、训练Step、MFU相对MAPE为1.722853/2.625241/2.696833%。在线倍率给预测增加的2.926秒中，F/B Shapley均值约0.780/2.146秒，说明目标运行中可见慢化主要由B方向前缀传入模型。所有目标倍率都超出source85/90范围，且预测平均到57.845%才可用；这是开发集nowcast，不是冷启动升级。entry/tail/outer预测逐值不变，未引入其他阶段退化。

唯一下一步：实施T36A。只用source85/90拟合早期前缀可靠性收缩，候选输入限定为全stage首F与末端4个PP stage首B；在source95/100同时通过全局和phase门后才能接目标。输出截止比例与区间覆盖，不按T35目标结果选系数或方法，继续保持在线/冷启动分栏。

| 2026-09-06 21:47 | T36A早期前缀可靠性 | source系数0.611345使开发MAPE0.312367%；目标在线7.016873%，42.802%后可用；源区间覆盖目标0/4 | 71测试/3阶段manifest/243产物通过；拒绝目标最优但source退化的raw消融，转T37A剩余时间审计 |

## T36A：更早前缀与域偏移代价

[总览图](deadline_20260907/runs/T36A/diagnose/early_prefix_reliability.svg)、[可靠性拟合](deadline_20260907/runs/T36A/diagnose/source_reliability_fit.csv)、[source指标](deadline_20260907/runs/T36A/diagnose/source_early_metrics.csv)、[target逐轮](deadline_20260907/runs/T36A/diagnose/target_step_iteration_results.csv)、[target汇总](deadline_20260907/runs/T36A/diagnose/target_step_metrics.csv)、[前缀因子](deadline_20260907/runs/T36A/diagnose/target_early_factors.csv)、[区间覆盖](deadline_20260907/runs/T36A/diagnose/target_early_interval_results.csv)、[cutoff](deadline_20260907/runs/T36A/diagnose/target_early_prefix_cutoffs.csv)、[方向Shapley](deadline_20260907/runs/T36A/diagnose/target_early_direction_shapley.csv)、[节点/边](deadline_20260907/runs/T36A/diagnose/target_early_nodes.csv.gz)、[验收](deadline_20260907/runs/T36A/acceptance.json)。source85/90原始图响应与实际响应拟合出可靠性0.611345，两次单轮比例0.561094/0.661596构成非置信区间的source-only范围。

主候选在source95/100把1F1B MAPE 0.332811%降至0.312367%、phase MAE 15.827028降至14.698333ms，因而允许目标开发评分。目标四轮在线1F1B MAPE为7.016873%，逐轮5.841425/6.329396/9.434500/6.462172%，平均低估1,513.230ms；Profiler/训练Step/MFU相对误差6.459935/7.061302/7.612662%。它平均在42.802%时可用，较T35早15.043pp，但精度明显较低；F/B Shapley平均约0.477/1.318秒。

source系数区间的目标包络覆盖为0/4，目标原始倍率也全部超出source85/90范围。未收缩候选在目标MAPE为1.725071%，但其source95/100为0.345547%，差于0.332811%控制；因此不按目标误差改选。entry/tail/outer、正式拓扑和v685均未改变。

唯一下一步：实施T37A。只读取T35/T36已seal的逐轮预测、cutoff、区间和source门结果，统一转换成剩余时间误差与Pareto表；不重拟合。若再登记同域lagged更新，另建版本并把目标校准轮次与已暴露开发范围写清。

| 2026-09-06 21:59 | T37A剩余时间Pareto审计 | T35/T36主候选剩余MAPE 4.201267%/12.263149%，cutoff 57.844807%/42.801638%；T36区间source开发0/2、target 0/4 | 73测试/3阶段manifest/214产物通过；T36主候选被v685支配，转T38A因果同域更新 |

## T37A：总时长改善不等于剩余时间可用

[Pareto图](deadline_20260907/runs/T37A/diagnose/remaining_time_pareto.svg)、[逐轮表](deadline_20260907/runs/T37A/diagnose/remaining_time_iteration_results.csv)、[汇总表](deadline_20260907/runs/T37A/diagnose/remaining_time_metrics.csv)、[区间表](deadline_20260907/runs/T37A/diagnose/remaining_time_interval_results.csv)、[覆盖汇总](deadline_20260907/runs/T37A/diagnose/remaining_time_interval_coverage.csv)、[验收](deadline_20260907/runs/T37A/acceptance.json)。T37只读验证T35/T36 diagnose manifest和prediction seal，将预测与实际总时长同时减去逐轮cutoff。误差毫秒值保持不变，相对误差使用实际剩余时间作分母。

正式v685、T35主候选、T36主候选的剩余时间MAPE依次为11.340786%、4.201267%、12.263149%；平均cutoff依次为0%、57.844807%、42.801638%。可选Pareto前沿只有v685和T35，T36主候选同时晚于且差于v685。T36 raw消融为3.010076%，但其source门失败，只有“若可选”的诊断标记。T36区间在source拟合轮覆盖2/2，source95/100和target覆盖0/2、0/4。

r1因历史结果缺少preflight hash-only能力而在指标计算前fail closed；修复后的r2/r3三阶段通过，七项科学输出字节精确，峰值194/195MB。没有新预测、拟合、参数更新、加边或等待。

唯一下一步：实施T38A。以T35 sealed nowcast为基线，iteration85初始化；90/95/100只应用前一已完成轮的innovation，按因果walk-forward输出剩余时间、1F1B、Step、MFU和逐轮不确定性，保持target开发标签。

| 2026-09-06 22:07 | T38A目标同域lag-one更新 | 90–100因果1F1B MAPE 0.536422%，剩余1.262981%；Step/MFU 1.127495%/1.143873% | 76测试/3阶段manifest/221产物通过；保持图内F/B和跨轮状态分栏，转T39A稳健性消融 |

## T38A：封存图预测之上的因果目标状态

[图](deadline_20260907/runs/T38A/diagnose/causal_lagged_update.svg)、[预测链](deadline_20260907/runs/T38A/diagnose/causal_prediction_ledger.csv)、[因果门](deadline_20260907/runs/T38A/diagnose/causality_mutation_audit.csv)、[逐轮评分](deadline_20260907/runs/T38A/diagnose/target_iteration_results.csv)、[汇总](deadline_20260907/runs/T38A/diagnose/target_metrics.csv)、[阶段](deadline_20260907/runs/T38A/diagnose/target_phase_ledger.csv)、[分解](deadline_20260907/runs/T38A/diagnose/prediction_decomposition.csv)、[区间](deadline_20260907/runs/T38A/diagnose/causal_interval_results.csv)、[验收](deadline_20260907/runs/T38A/acceptance.json)。85轮初始化后，90/95/100只读取前一轮已完成的T35 base残差；当前/未来真值扰动不改当前预测，哈希链逐轮封存。

三轮1F1B MAPE从T35的1.914652%降为0.536422%，剩余时间MAPE从4.508606%降为1.262981%；Profiler/训练Step/MFU相对误差0.630637/1.127495/1.143873%。含初始化四轮为0.740494%/1.767049%。entry/tail/outer不变。

平均构成为phase-transfer 18,253.343ms、F 762.389ms、B 2,188.557ms、lag状态398.385ms，最终平均残差−15.266ms。前三项来自封存图；lag项是同域时间状态，未强行归入计算/通信/等待。有限历史区间覆盖1/2，只有三轮walk-forward。r1/r2十一项科学输出字节精确，峰值199MB。

唯一下一步：实施T39A。保持T38为已登记主方法，只读比较零更新、lag-one、延迟一轮和扩展历史均值的状态年龄与误差，判断改善对相邻状态的依赖，不在本批已暴露目标轮上重选规则。

| 2026-09-06 22:15 | T39A固定状态稳健性 | zero/lag-one/stale-2/expanding MAPE 1.914652/0.536422/1.082859/0.502136%；lag-one三轮全改善 | 79测试/3阶段manifest/220产物通过；不按0.034286pp事后优势改选，转source同构检验 |

## T39A：新鲜状态有效，但规则尚不可唯一识别

[图](deadline_20260907/runs/T39A/diagnose/state_robustness.svg)、[状态账本](deadline_20260907/runs/T39A/diagnose/state_prediction_ledger.csv)、[逐轮结果](deadline_20260907/runs/T39A/diagnose/state_iteration_results.csv)、[汇总](deadline_20260907/runs/T39A/diagnose/state_robustness_metrics.csv)、[冲击](deadline_20260907/runs/T39A/diagnose/synthetic_shock_response.csv)、[验收](deadline_20260907/runs/T39A/acceptance.json)。固定T38以后，90–100的zero/lag-one/stale-2/expanding历史均值1F1B MAPE为1.914652/0.536422/1.082859/0.502136%，剩余时间MAPE为4.508606/1.262981/2.535804/1.180718%。lag-one三轮全改善；stale较差，最近状态有增益。

扩展均值在同一已暴露三轮上只好0.034286pp，不能用于改选。500ms冲击严格按过去依赖传播，所有候选没有当前/未来真值依赖。r1因3.64e−12ms浮点往返超过1e−12断言失败；r2/r3改用封存字段1e−6ms精度，七项科学输出字节一致，峰值195/197MB。

唯一下一步：实施T40A。把固定lag-one原样用于source85/90/95/100在线序列，并对source/target每次转换计算状态测量误差break-even区间与三轮sign-test界限；不依据target消融重选规则。

| 2026-09-06 22:21 | T40A跨域状态与容错 | target改善3/3、1.914652%→0.536422%；source改善1/3、0.298320%→0.517489%；target sign p=0.125 | 81测试/3阶段manifest/222产物通过；限定target-local，冻结方法搜索并转决策表 |

## T40A：固定规则的source反证

[图](deadline_20260907/runs/T40A/diagnose/cross_domain_state.svg)、[逐转换](deadline_20260907/runs/T40A/diagnose/transition_state_results.csv)、[汇总](deadline_20260907/runs/T40A/diagnose/domain_state_metrics.csv)、[容限](deadline_20260907/runs/T40A/diagnose/state_noise_tolerance.csv)、[验收](deadline_20260907/runs/T40A/acceptance.json)。target基础误差连续三次同号，lag-one三轮都改善，MAPE 1.914652%→0.536422%；source误差符号为负、正、正、负，只在中间转换改善，三轮MAPE 0.298320%→0.517489%。固定规则没有跨域成立。

target三轮全改善的精确单侧sign-test下界为0.125，最小对称break-even半径183.990ms；source改善1/3、p=0.875。结果将T38限制为目标同域状态，不能推广为source迁移方法。r1/r2六项科学输出字节精确，峰值195/198MB。

唯一下一步：实施T41A。冻结目标方法搜索，汇总v685/T35/T38及拒绝方案的输入、cutoff、误差、解释范围、失效条件与复现入口；只有新增连续target轮次才恢复状态方法评估。

| 2026-09-06 22:31 | T41A运行模式与复现索引 | 固定v685冷启动、T35同轮开发、T38目标初始化开发三层；T36/T39保持拒绝或消融 | 83测试/3阶段manifest/225产物通过；6项输出字节精确，冻结方法搜索，转T42A证据链审计 |

## T41A：可操作的预测边界

[决策流程](deadline_20260907/runs/T41A/diagnose/operational_decision_flow.svg)、[6种模式](deadline_20260907/runs/T41A/diagnose/operational_mode_decision.csv)、[5层解释](deadline_20260907/runs/T41A/diagnose/explanation_layers.csv)、[6条复现索引](deadline_20260907/runs/T41A/diagnose/reproduction_index.csv)、[恢复合同](deadline_20260907/runs/T41A/diagnose/stop_resume_contract.json)、[验收](deadline_20260907/runs/T41A/acceptance.json)。T41只读校验T35–T40的manifest和acceptance，没有生成预测、重新评分或选择方法。运行前只允许v685正式冷启动；T35在同轮前缀齐备后可作开发nowcast；T38还需紧邻前一目标轮完成，只可作目标初始化的顺序开发分析。T36主候选被支配、raw未过source门，T39扩展均值是事后未选择消融。

T38三轮1F1B/剩余/Step/MFU相对MAPE为0.536422/1.262981/1.127495/1.143873%，但sign-test最小p=0.125且source迁移退化。至少新增2个设计时未见、严格更晚且连续的目标轮次才恢复状态研究；恢复冷启动研究则需动态router token矩阵和独立runtime readiness profile同时可用。r3/r4六项科学输出字节精确；56输入30,400,800字节、0 raw，峰值194/192MB。预注册验收文字的“five manifests”与同文件固定的T35–T40六项数组不一致，正式程序和验收均按六项执行，预注册文件保留不回改。

唯一下一步：实施T42A只读最终证据链审计，检查T35–T41正式链接、acceptance/manifest/plan/spec/command哈希、文档相对链接、复现入口和交付清单。只修正来源或链接缺陷；没有满足恢复合同的新证据时，不新增目标方法、预测或评分。

| 2026-09-06 22:41 | T42A最终证据链审计 | T35–T41七条链、783个既有文档链接、17类交付和6条复现记录全部通过，0 hash/link失败 | 85测试/3阶段manifest/227产物通过；当前证据包完整，等待T41恢复条件 |

## T42A：完整性关闭当前有限证据包

[合同](deadline_20260907/runs/T42A/diagnose/final_integrity_contract.json)、[阶段链](deadline_20260907/runs/T42A/diagnose/evidence_chain_audit.csv)、[文档链接](deadline_20260907/runs/T42A/diagnose/documentation_link_audit.csv)、[复现重算](deadline_20260907/runs/T42A/diagnose/reproduction_recompute.csv)、[交付清单](deadline_20260907/runs/T42A/diagnose/deliverable_inventory.csv)、[验收](deadline_20260907/runs/T42A/acceptance.json)。T42在输入封存后只读检查T35–T41；正式链接、manifest、acceptance、plan、wave spec、pipeline command和T41复现索引全部一致。783个既有本地文档链接无缺失，17类交付物的预检SHA和大小通过。

r1/r2六项审计产物字节精确；71冻结输入37,411,250字节、0 raw，峰值151MB。没有新预测、拟合、目标评分、参数、边、等待或拓扑变化。当前证据支持三层使用方式：v685正式冷启动，T35同轮开发nowcast，T38目标初始化后的顺序开发。

唯一下一步：监测T41停止/恢复合同。只有新增至少2个设计时未见的连续目标轮次、动态router token矩阵和独立runtime readiness profile同时到位、或出现独立拓扑证据时，才先登记新计划并恢复研究；否则到2026-09-07 07:00仅保存截止快照，不在旧四轮目标数据上继续选方法。

# MFU 校准盘点与 224-GPU 并行策略外推计划

工作目标截止：2026-08-28 23:00（Asia/Hong_Kong）。本 worktree 只处理
MFU 外推与并行策略；OISA 九类集合通信仿真/FCT 回填由另一条工作线负责。

## 当前 256-GPU 校准资产

| 层 | 已有证据/参数 | 当前状态 | 外推缺口 |
| --- | --- | --- | --- |
| MFU 口径 | 源 fixed iteration FLOPs `1.293891072e17`，500 TFLOP/s/GPU；training-log clock = ProfilerStep + outer residual | 目标 52 层/GBS48 FLOPs 已解析缩放为 `8.436548312e16`，与 224 日志反推中位只差 `-0.0696%` | outer residual 仍沿用源值，iteration 55 低估 `271.384 ms` |
| PP compute | 256 Rank × 4 FWD × 4 BWD，PP16、16 lane | 只有 iteration 55 的完整 compact Trace；目标 PP14 layer placement 已从 Fabric 确认 | 缺少目标 3-microbatch schedule 的配置匹配回放 |
| PP network | 80 MiB/op 的 reference FCT `4.712346 ms` | 来自 20 iteration、5,120 个 cell 的 per-cell minimum 中位 | PP14 物理放置/竞争改变后要用目标拓扑 FCT 验证 |
| PP 软件 completion | backward `93.463844 ms`；总 completion `98.176190 ms` | iteration 55、stage 1–14 的 848 个 `send_backward` 中位；所有 hop 共用 | forward 未校准；stage/API/microbatch 未分层；PP14 是否保持该软件路径未知 |
| Optimizer RS/AG | `dp_rs/edp_rs/dp_ag0/edp_ag0/dp_ag1/edp_ag1` 六类，共 20 iteration、30,717 rank calls、8,640 groups | Trace-reference FCT、arrival skew、rank completion skew 和依赖 lag 均已入 max-plus DAG | 统一 PP→optimizer 回放只物化 iteration 55；新 group size 和新 payload 需要目标 FCT |
| OISA 接口 | release offsets、elapsed/tail、逐 Rank done 已可传播 | Mock 覆盖接口；真实 OISA 仅固定 8-GPU 拓扑的单组 smoke | 不是 256/224 目标集群校准；本 worktree 不回填其结果 |
| 端到端精度 | 256 源 iteration 55：Step `-0.1010%`；224 目标 iteration 55：training Step `-2.117%`、目标配置 MFU relative `+2.162%` | 224 单点通过 3% 门槛；20 iteration Step 绝对误差 P90 `8.929%`，未通过 5% | 需要解释并建模后半程退化，不能只验一个点 |

代码重跑还发现冻结 v2 派生结果有轻微版本漂移：当前传播语义得到
optimizer service final-step marginal `224.741171 ms`，冻结 v2 产物仍为
`224.652265 ms`，且缺少新增的 `service_marginals_available` 字段。核心
Step/MFU 没有漂移。本分支不改写 256-GPU/OISA 冻结产物，以免与并行工作线
冲突；后续应由基线维护者统一重建。

## 224-GPU v1：可先识别的最小变化

采用 `TP1 / PP14 / CP2 / DP8 / EP8 = 224`。它保留源配置的 CP、DP、EP、
16 lane、dense16 和 EDP2 communicator，但 Fabric run contract 已确认目标为
52 层、GBS48、3 microbatch，而源任务是 60 层、GBS64、4 microbatch。因此
现有 v1 只能作为源工作量守恒的 PP 结构基线，不能再称为“只改变 PP”的完整
目标配置模型。

由于 v1 compact 输入没有带入 layer-to-stage map，它枚举
从 16 个源 stage 模板保留 14 个的全部 120 种组合，并对每条
`lane × phase × microbatch` 路径严格守恒 compute work；optimizer aggregate
payload 按 kind 守恒，service 暂按 payload 比例缩放。代表场景仅用于提供完整
DAG 工件，正式结论使用 P10/P50/P90 范围。

Fabric 工程已提供真实映射：stage0/13 各 2 层，stage1..12 各 4 层。当前结果
见 `case_224gpu_pp14_cp2_a2a/results/pp14_extrapolation_v1/`：结构状态为
`PASS_STRUCTURAL`，实测状态为 `PASS_POINT_FAIL_TEMPORAL_GENERALIZATION`。
主 MFU 分子已修正到目标配置；旧的 `4.9853%` 是源 FLOPs 误用口径，不能作为
目标 MFU。逐 iteration 校验见
`fabric_measurement_validation/FABRIC_MEASUREMENT_VALIDATION.md`。

## 其他合法 224-GPU 策略

在 TP1、EP8、目标 GBS48/MBS2、候选 PP `{7,14,28}`、CP `{1,2,4}` 下共有 6 个
整除且 microbatch 为整数的候选。优先级按 Trace 可识别性划分：

1. `PP14/CP2/DP8`：中高，拓扑只改变 PP，但 microbatch 从捕获 Trace 的 4 变为
   目标 3；v1 已运行源工作量守恒基线，v2 需换 schedule。
2. `PP14/CP1/DP16`、`PP14/CP4/DP4`：中，communicator 仍是 dense16/EDP2，
   但 CP compute、DP 和 microbatch 分别变为 2/8，不能直接复用 compute Trace。
3. `PP7/*` 与 `PP28/*`：低，lane/group size 变为 32/8，必须取得新 FCT、
   新 CP compute scaling 和新的 pipeline schedule 后再建模。

机器可读矩阵见 `legal_224gpu_strategies.csv`。

## 校验门槛与今晚工作顺序

- G0 结构门槛：Rank/stage/lane/group 网格完整，compute 纳秒级守恒，payload
  取整误差有界，所有 DAG 时间有限。v1 已通过。
- G1 源侧泛化：补齐多 iteration PP trace，固定参数做留出回放；不能再用同一
  iteration 同时拟合和验收。
- G2 网络门槛：目标 rank mapping 下六类 collective 和双向 PP FCT 请求覆盖
  100%，OISA/实测返回与请求签名一致。
- G3 目标校验：已有 20 个 224-GPU profiler iteration。iteration 55 已满足
  Step/MFU relative error ≤3%，但全时段绝对误差 P90 `8.929%` 未满足 ≤5%；
  还需检查逐 stage/逐 microbatch 到达误差，避免端到端抵消。
- G4 策略迁移：只有在 CP compute scaling、microbatch schedule、communicator
  FCT 三项齐全后，才比较 PP7/PP14/PP28 的优劣。

今晚 23:00 前的顺序：冻结已实测校验的 PP14 v1 与 MFU 分子修正；用真实
52-layer map 和 3-microbatch schedule 建立配置匹配 v2；分解 iteration 55 的
`229.699 ms` ProfilerStep 误差和 `271.384 ms` outer-residual 误差；再分析 20 个
iteration 后半程退化。OISA FCT 到位后只替换网络 service，不改写这套工作量与
时钟口径。

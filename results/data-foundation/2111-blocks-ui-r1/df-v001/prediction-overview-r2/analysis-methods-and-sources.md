# 32→256：1F1B外推误差分析总说明

更新：2026-09-22。工程工作区：`/home/zjb/Desktop/worktrees/mfu-16to256`，分支`32to256`。本说明汇总已有诊断，不新增拟合或修改冻结预测。

## 核心结论

当前1F1B外推偏差主要与MoE相关成本迁移有关。按PP规模重复源stage时，虽然保留了相同计算结构，却没有表达逐层专家路由可能带来的实际工作量与负载差异，因此固定成本模板出现系统性高估。

**结构可以重复，专家成本不能仅凭结构相同就视为常数。**

这一解释有明确工作量证据支持：相同PP1的源/目标专家输入行数和GEMM次数不同。但尚未证明全部差异由路由随layer变化造成，仍未完整排除容量处理、实现版本等因素；也未证明成本稳定呈“两头大、中间小”。当前模型采用32卡PP1模板，不是256卡两头插值。

从+6.16%到−0.51%的诊断同时涉及F/B专家计算，以及B中已识别的token重排/EP活动，不能表述为“只换FC就解释全部误差”。剩余正负抵消不等于全部机制已闭合。

## 配置与数据范围

| 项目 | 源32卡 | 目标256卡 |
|---|---|---|
| 流水线 |PP4|PP16|
| 层布局 |2＋4＋4＋2|2＋14×4＋2|
| CP / EP / TP / ETP |2 / 8 / 1 / 1|2 / 8 / 1 / 1|
| DP / 专家DP |4 / 1|8 / 2|
| 本次迭代 |iter60|iter60|
| 代表rank |PP1 rank8，F0/B0|PP1–14各stage×16，每stage四个F/B|

共同配置：hidden size5120、160专家、top-k6、专家中间维1536、MBS2、seq8192、GBS64。专家权重维度相同不保证收到的token数相同。

源原始数据位于`/home/zjb/Desktop/32/gpu32_gbs64_framework/2026-09-18-10:38/`；源rank8文件：

`/home/zjb/Desktop/32/gpu32_gbs64_framework/2026-09-18-10:38/worker34031/profiler/iteration_60/rank8.1789699916514.pt.trace.json`

目标入口`/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40/`，软链接解析数据根为`/var/local/zjb-data/gbs64/`。完整逐rank绝对路径及SHA256在各manifest的raw项，以及：

`/home/zjb/Desktop/worktrees/mfu-16to256/results/data-foundation/b-position-curves-r2/raw-provenance.json`

源B既有事件缓存引用：`/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/b-cost-blocks-r1/evidence.json`。

236B框架语义参考：`/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/`。它解释专家FC、激活、重排及checkpoint流程，不证明与采集实际安装版本完全一致。

## 分析方法

1. 冻结原DAG、源成本及目标实测边界，复现1F1B预测21653.357840ms、实测20396.013187ms，差1257.344653ms（6.164659%）。窗口是既有首F到末B口径，不包含启动、优化器更新尾段；不是纯稳态子区间。
2. 先将中间F/B整块换为目标观测，保持其他成本及外层依赖不变，重调度定位偏差集中区域。中间B解释1104.671894ms，即原高估的87.86%。
3. 按CPU框架调用→同线程runtime/driver→correlation关联设备事件，分开设备活动并集、包络、CPU时间与未覆盖区间。框架调用可能对应多个kernel。
4. B识别重计算和真反向GroupedLinear配对，以及专家激活/门控；F因checkpoint无梯度前向没有GroupedLinear标记，采用permute→silu→unpermute顺序边界，区间内只提取GEMM和明确激活调用，不替换整个包络。
5. token重排由明确permute/unpermute祖先识别；EP子集是Dispatch/Combine调用关联设备活动，不等于完整网络服务时间。无法确定的投影、拷贝、打包解包、同步等待继续单列。
6. 排他性时间分桶保留重叠项，防止把并发kernel累计时间重复加到墙钟。有效块替换采用`冻结块成本＋目标子项成本−源子项成本`，再按外层DAG的最大前驱完成时间调度；不直接将全部块差值相加。
7. 联合归因使用全部替换组合与顺序平均贡献（Shapley）；它与单次直接替换的降幅不同。注意分母：原1257ms、中间B1105ms、FC替换后427ms不可混用。

## 主要结果

| 条件 | 1F1B秒 | 相对实测误差 |
|---|---:|---:|
| 原冻结预测 |21.653|+6.16%|
| 仅替换中间B的FC（重计算＋真反向） |20.823|+2.09%|
| B完整专家计算（FC＋激活/门控） |20.667|+1.33%|
| 再加入B确认token重排/EP活动 |20.565|+0.83%|
| 在上一行基础上再替换F完整专家计算 |20.292|−0.51%|

B两类成本联合归因相对原高估：专家计算79.28%，确认token/EP活动7.87%；不是F+B联合的新占比。二者直接一起替换减少86.57%的原高估；不能将联合顺序平均与直接效果混为一谈。

F专家额外减少272.528ms，结果转为低估103.730ms，显示其他成本抵消。不得把−0.51%当作32卡独立外推精度。

## 证据及最小复现

下列路径均相对工作区；对应结果目录含report.json和manifest.json，源/目标事件记录及输入/代码/结果哈希。原始trace不复制入Git。

| 内容 | 文档 | 结果目录 | 脚本 |
|---|---|---|---|
| PP1算子/八卡专家工作量 |PP1_ALIGNMENT_R1.md|pp1-alignment-r1|compare_pp1_alignment_r1.py|
| B FC成本诊断 |FC_TRANSFER_R1.md|fc-transfer-r1-verified|diagnose_fc_transfer_r1.py|
| B剩余排他性归因 |B_REMAINING_R1.md|b-remaining-r1-verified|diagnose_b_remaining_r1.py|
| token/激活/Attention子集 |TOKEN_ATTENTION_R1.md|token-attention-r1|diagnose_token_attention_r1.py|
| 两大类别重新归因 |MOE_TWO_GROUPS_R1.md|moe-two-groups-r1|diagnose_moe_two_groups_r1.py|
| F专家子项诊断 |F_EXPERT_TRANSFER_R1.md|f-expert-transfer-r1-verified|diagnose_f_expert_r1.py|

文档前缀`docs/data-foundation/`，结果前缀`results/data-foundation/`，脚本前缀`workflow/data_foundation/`。复现示例：

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/diagnose_moe_two_groups_r1.py --out results/data-foundation/moe-two-groups-reproduction
python -B workflow/data_foundation/diagnose_f_expert_r1.py --out results/data-foundation/f-expert-reproduction
```

输出目录须不存在，第二条会只读源及14个目标rank trace。上游缓存和原路径需可访问；此为本机可追溯分析，不宣称仅clone仓库就可独立复现。重算前先核验manifest输入哈希。

## 验收与尚未闭合

已验证：原预测及全中间B oracle回归、56块F/B子项关联、分项时间守恒、组合归因总和、有效结果哈希和HTML桌面/手机交互。

尚未闭合：完整EP8内部关键路径反事实、第二EP8组、跨iter泛化、实际训练版本与路由/容量处理的精确成因。目标iter60已经曝光，仅限诊断；不能当盲测，也不能仅凭配置一致就宣称路由工作量一致。

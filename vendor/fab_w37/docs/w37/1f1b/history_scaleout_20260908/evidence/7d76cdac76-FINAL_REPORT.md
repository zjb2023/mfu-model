# W37 Task B 最终报告：16→256 Step/MFU方案与最小端到端验证

2026-09-06。**方案及实际最小端到端实验已完成。** 用户确认S5000单卡BF16峰值为500 TFLOPS后，已补齐绝对MFU预测、实际值及百分点误差；没有修改原有时间模型。主方法目标MFU预测5.9905%，实测20点均值3.2482%，平均绝对误差2.7423个百分点；MFU相对MAPE为84.8177%，training Step MAPE为45.7773%。完成实验不等于达到3%跨规模误差。

本任务是**源16卡校准、256卡静态构造、封存后目标评价的条件外推实验**。256数据早已被历史工作及v4评价观察；v5时钟修正和本次峰值归一化均披露此前暴露，不宣称新的独立盲测。本轮没有target-assisted成本修订。

## 实际配置、切分与成本迁移

| 项目 | 源16卡wo attempt | 目标256卡 |
|---|---|---|
| TP / PP / CP / DP / EP | 1 / 2 / 1 / 8 / 8 | 1 / 16 / 2 / 8 / 8 |
| 层数 / 每stage放置 | 4 / [2,2] | 60 / [2,4×14,2] |
| MBS / GBS / microbatches | 4 / 32 / 1 | 2 / 64 / 4 |
| 序列 / hidden / vocab | 8192 / 5120 / 128256 | 相同 |
| MLA/MoE、重算与精度 | 首层dense，其余MoE；全block重算；BF16 | 同类静态结构，59个MoE层；BF16 |
| 硬件 / 放置 | MTT S5000，2个8卡host | 同型号，32个8卡host；256 rank静态PG/host已核验 |

完整模型/并行、软件元数据、采集与未知项见[可比性盘点](COMPARABILITY.md)、[静态合同](scenarios_v3.json)及[v2/v3/v4报告](CONTINUATION_V4.md)。训练commit、NIC/交换机路径、跨host offset界仍未确认；这是明确记录的可迁移性限制。另一16卡attempt w_mccllog_1129单独登记、单独做源留出，不混入wo成本。

wo源iteration 5–60每5轮12点校准，65–100的8点留出；目标固定5–100每5轮20点。所有经验成本来自源校准提取物。源准备进程读取两分区用于提取，模型只读校准分区；该源留出是计算隔离留出，不是研究者盲测。输入契约将源动态、目标静态和目标评价动态严格分角色。

| 分项 | 实施方式 | 证据与限制 |
|---|---|---|
| 计算 | 源每层/相位慢rank GPU服务中位数；目标按本地token比例1/4迁移 | source GPU账本守恒及两份raw/parquet黄金对照；单一形状域，线性效率是外推假设 |
| 集合通信 | EP路由与服务尾部并集；CP静态payload+源host内服务代理；DP/EDP按静态组/host分层 | 源216个EP组、576个MCCL组；CP2与目标拥塞缺独立微基准 |
| PP通信 | 源startup+byte代理，目标静态payload/peer | 源192个PP配对；不继承包含等待的SendRecv全生命周期 |
| 流水等待 | 显式blocking 1F1B warmup/steady/cooldown；共享传输节点等待两端释放 | PP2→PP16重建依赖，不按rank倍率复制等待；调度语义检查通过 |
| 软件/optimizer | 源参数拥有量与GPU服务拟合；源未归属残差/outer常量迁移 | 来源明确，但残差不是独立识别出的纯软件服务；目标双gather行为是假设 |

实施细节、来源参数和成本图见[CONTINUATION_V4.md](CONTINUATION_V4.md)。历史256拟合节点成本、PP/CP/EP/DP/EDP时序、等待、entry/outer修正均未继承。通用结构代码独立使用；任务A后续接入必须引用明确commit及[共同接口](../INTERFACES.md)，本次不依赖兄弟未提交代码。

## MFU口径与500 TFLOPS来源

[用户确认原文](peak_evidence/user_confirmation_20260906.md)为“BF16 S5000是500TFLOPS”，对应此前询问的单卡BF16 dense语境，登记为500 TFLOPS/GPU。证据级别是**user-confirmed static input**，没有表述为独立厂商规格验证。用户随后明确恢复目标。新[峰值合同](peak_contract_user_20260906.json)固定原文、SHA和来源；旧空模板、scenarios_v3、v5时间预测及其输入均保留。

采用`MFU(%) = 100 × model_training_FLOPs / (training_step_seconds × GPU数量 × 500e12)`。主分子沿用封存的因果注意力三角有效位置口径，乘加各一次，F+B=3F，排除重算、norm/激活/optimizer算术。源/目标每步分别4.218697266757632e15 / 9.767709113843712e16 FLOPs。Profiler两侧统一为全rank ProfilerStep标记包络；MFU分母使用完整training Step。

封存时另保留full-square attention分子变体，其目标吞吐与训练日志名义TFLOPs相符；本次未因这一观察更换主分子。因此此处绝对MFU须连同FLOPs定义引用，不能直接与其他分子口径的MFU混比。

## 真实20点评价与基线

目标实际training Step为22.1810–25.5773s；主方法预测12.7385s、Profiler预测12.4099s。目标实际MFU范围2.9835%–3.4403%，算术均值3.2482%（逐点MFU均值，不是用平均Step先求倒数）。

| 预定方法 | 预测Step(s) | training MAPE | Profiler MAPE | 预测MFU | MFU MAE(百分点) | MFU相对MAPE |
|---|---:|---:|---:|---:|---:|---:|
| M2_source_only_linear，主方法 | 12.7385 | **45.7773%** | **44.0737%** | **5.9905%** | **2.7423** | **84.8177%** |
| B0_constant_work_efficiency | 6.1492 | 73.8254% | 74.4144% | 12.4098% | 9.1616 | 282.8646% |
| B1_compute_bubble | 10.8834 | 53.6736% | 52.4336% | 7.0116% | 3.7634 | 116.3196% |
| M2_no_shape_speedup_sensitivity | 39.8381 | 69.5748% | 78.0528% | 1.9155% | 1.3327 | 40.9033% |

主方法优于B0/B1，但误差明显；敏感性分支保持预设角色，不按MFU成绩更换主方法。旧256拟合模型训练输入不同，不是此source16-only任务的同条件对照。没有约定或实现3%目标误差。

主方法源8点留出预测MFU12.4024%，实际均值11.9962%，MAE0.4062个百分点；源training/Profiler/MFU相对MAPE分别3.2752%/3.1470%/3.4183%。这组源留出结果不能充当16→256成绩。

2.7423个百分点与84.8177%是不同指标。同目标F、N、P固定时，`MFU_pred/MFU_actual−1 = T_actual/T_pred−1`；时间误差用`T_pred/T_actual−1`，分母不同。500 TFLOPS只确定绝对MFU，时间误差和MFU相对误差均保持v5原值。

## 封存、逐点交付与复现

- v5时间seal：`d7766035f5cbb86d821df085ee5b0e19c2cf6b286f0f11af1508d463563775cb`，封存目录prediction-v5-20260905T115741115514Z。
- 本次绝对MFU[seal/checkpoint](../../../results/w37/B/mfu-addendum-v1-20260905T233332765810Z/seal_checkpoint.json)：`6fca5b4da8cb50da6739fd9de1dd3655ae0fd30d665c174b74ee314458016c05`。预测worker只读峰值静态证据和既有时间预测，目标读取两次拒绝探针通过；随后独立evaluator才读既有v5动态评价表。没有重扫原始trace。
- [统一112条逐点Step/MFU与误差](../../../results/w37/B/final-validation-20260905T233625518711Z/point_results.csv)：源8点×4方法和目标20点×4方法；[八行汇总](../../../results/w37/B/final-validation-20260905T233625518711Z/summary.json)。
- [最终独立核验](../../../results/w37/B/final-validation-20260905T233625518711Z/checks.json)：v4/v5/MFU三份seal、两阶段输出与代码SHA、完整网格、112点按F/T/N/P直接重算、全部原预测字段仅peak/MFU变化、时间和相对误差不变。原20项结构/时钟检查及6项MFU证据/执行检查保留。

在唯一worktree根目录执行本次数值复现：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/16to256/reproduce_mfu_addendum_v1.py --peak-contract docs/w37/16to256/peak_contract_user_20260906.json
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/16to256/reproduce_final_checks.py
```

第一条新建运行目录；第二条默认核验本文固定正式运行，也可用`--mfu-run <新目录>`核验第一条的新结果。源特征提取/时间预测/目标双时钟评价的完整命令分别见v3/v4/v5报告；本次复用其已核验提取物。共同基线smoke仅经scripts/w37/run_baseline.py，研究入口使用只读输入和输出目录隔离。移机需相同只读数据/环境；没有承诺脱离输入provider裸克隆可运行。

## 验收结论与后续接口

用户要求的六阶段、真实Step/MFU逐点预测与误差、基线比较、来源披露、复现、GOAL与HANDOFF均有证据，最低实验验收完成。大误差、单一源形状域、CP/网络代理和未绑定环境细节仍是结果适用范围，后续若优化应先补独立源/硬件证据；若读取目标分项做诊断，须单独标target-assisted，不能替换本次source-only成绩。

方案完成：已实现16→256分项成本迁移、PP2→PP16显式流水调度和来源隔离的预测封存/评价流程。

实验验证完成：基于用户确认的BF16 500 TFLOPS已完成20点Step/MFU评价，主方法预测MFU5.99%、实测均值3.25%，MFU MAE2.74个百分点、相对MAPE84.82%，training Step MAPE45.78%；约3%仅是源侧留出误差。

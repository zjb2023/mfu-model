# 三阶段模型 source16→target256：预测与评价

2026-09-08。**本轮三阶段预测、固定20点评价及网页交付已完成。** `T3_phase_pipeline_v1`预测256卡Training Step **12.96615365s**、Profiler **12.63759575s**、MFU **5.88534036%**。Training/Profiler MAPE分别 **44.80832419% / 43.04773894%**；MFU MAE **2.63712239个百分点**，MFU相对MAPE **81.57285217%**。

这是新实现的source16三阶段近似，与原M2使用同一源分项成本、静态配置和FLOPs口径；不是旧source256三阶段系数移植，也不是官方SimAI复现。已有256数据及先前M2/M3结果已被研究者观察，本轮仍是历史开发评价。没有用目标误差调节新公式、倍率或评价窗口。

## 三阶段如何预测

设P为PP阶段数、M为每轮microbatch数，f_s和b_s为stage s每microbatch前向/反向服务，反向含重计算；其计算、EP、CP和边界服务分别保留，发出的PP服务计入发送stage。前向最后stage、反向第一个stage无发出PP服务。

```
FWD = sum(f_s) + (M−1) × max(f_s)
BWD = sum(b_s) + (M−1) × max(b_s)
OPT = max(gradient_s) + before_optimizer_control
    + max(optimizer_s + parameter_gather_s) + after_optimizer_control
Profiler = entry_control + FWD + BWD + OPT + delta_source
Training = Profiler + outer_source
```

每个方向按重复microbatch在不均匀stage上的流水包络估计，再把三阶段相加。P=1时退化为M份服务；M=1时退化为逐stage服务之和；均匀stage时为(P+M−1)个服务槽。入口控制单列；OPT的gradient项取最慢stage，更新/聚合可跨stage并行，更新前全局控制保持独立。

该公式**不模拟真实blocking 1F1B前后向交错和跨阶段重叠**，不处理网络资源争用；不能因为名称叫三阶段就等同于所有历史三阶段实现。对历史v5.4代码的只读审阅表明它使用源phase wall减network service后按wave迁移等规则；16→256存在不同层数与形状，因此本次明确采用上述独立定义，不直接复用其针对256→224的系数或波次倍率。

## 校准、静态配置与限制

- 源：wo_mccllog_1116，16卡PP2/CP1/DP8/EP8、4层、MBS4、GBS32、M1。固定iterations5–60每5轮12点校准，65–100八点留出；另一attempt未混用。
- 目标：256卡PP16/CP2/DP8/EP8、60层[2,4×14,2]、MBS2、GBS64、M4。硬件S5000、BF16、序列8192、hidden5120。
- 计算：复用原M2的source16每层类型/F/R/B慢rank GPU服务中位数，目标本地token32768→8192按1/4迁移；这是未独立验证的形状效率假设。
- EP：源路由与通信尾部并集，按本地token比例迁移。CP2：静态payload和源机内通信代理。PP：源startup+bytes，不迁移包含等待的总生命周期。
- 梯度/参数通信：静态参数拥有量、DP_WITH_CP/EDP和源分层通信代理；optimizer使用源两stage参数拥有量拟合；双参数AG保持原目标假设。
- delta_source = 源Profiler中位数−源三阶段核心时间；负值直接拒绝，未静默截断。outer=median(源Training−源Profiler)。均为源侧常量迁移，不解释为已物理分解的软件时长。
- 预测器不调用DAG builder或replay，运行时相关函数被替换为拒绝调用探针。源成本fit和纯静态parameter/FLOPs函数复用已冻结代码。

## 目标预测分账（不是目标阶段实测）

| 分项 | 预测秒 |
|---|---:|
| FWD 前向 | 2.88153337 |
| BWD 反向含重计算 | 8.20552721 |
| OPT 梯度归约、控制、参数更新及聚合 | 0.37281858 |
| 三阶段合计 | 11.45987916 |
| 入口控制 | 0.01503086 |
| 源侧Profiler残差 | 1.16268572 |
| Profiler合计 | 12.63759575 |
| 源outer | 0.32855791 |
| Training合计 | 12.96615365 |

MFU使用原有效FLOPs `9.767709113843712e16`及用户确认单卡BF16 dense **500 TFLOPS**：`100×F/(Training秒×256×500e12)`。有效FLOPs排除重计算；其耗时仍包含在BWD中。

## 同窗口比较

| 方法 | 预测Step(s) | Training MAPE | 预测MFU | MFU MAE(百分点) | MFU相对MAPE |
|---|---:|---:|---:|---:|---:|
| 新T3三阶段 | 12.9662 | 44.8083% | 5.8853% | 2.6371 | 81.5729% |
| 原M2主模型 | 12.7385 | 45.7773% | 5.9905% | 2.7423 | 84.8177% |
| M3 rank/layer候选 | 12.2848 | 47.7087% | 6.2118% | 2.9636 | 91.6440% |
| B0等效率 | 6.1492 | 73.8254% | 12.4098% | 9.1616 | 282.8646% |
| B1计算与气泡 | 10.8834 | 53.6736% | 7.0116% | 3.7634 | 116.3196% |
| M2无计算形状加速敏感性 | 39.8381 | 69.5748% | 1.9155% | 1.3327 | 40.9033% |

目标实际Training22.1810–25.5773s。三阶段比原M2高估了0.22764828s的预测时间，时间MAPE降低约0.9690个百分点，仍显著低估目标耗时。**不能据此断言三阶段普遍优于DAG**；两种方法共同依赖未验证的形状/通信迁移，在本窗口三阶段较长的近似时间恰好更接近实际。原M2主结果不重写，M3完整细粒度对齐的PARTIAL状态不变。

源8点留出预测Training4.25189411s，Training/Profiler MAPE为3.275165%/3.147029%；该配置M1下三阶段退化，与原M2同源时钟口径吻合。源留出误差不是跨规模精度。

## 封存与实际执行证据

- [预测及阶段分账](../../../results/w37/B/prediction-three-stage-v1-20260908T033359749961Z/predict/)，[seal checkpoint](../../../results/w37/B/prediction-three-stage-v1-20260908T033359749961Z/seal_checkpoint.json)。seal SHA256：`25753532dda60bc5b2a9cb8cb07fd604e2d9e93901a7172e946b53899ff7d0f0`。
- [28条源/目标逐点结果](../../../results/w37/B/evaluation-three-stage-v1-20260908T033934645271Z/point_results.csv)、[结果汇总](../../../results/w37/B/evaluation-three-stage-v1-20260908T033934645271Z/summary.json)、[评分检查](../../../results/w37/B/evaluation-three-stage-v1-20260908T033934645271Z/checks.json)。
- 源预测worker精确白名单、bwrap全文件系统只读、仅输出可写、断网；目标结果表拒读探针通过。退出并保存worker外seal checkpoint后，再由独立evaluator读已核验时钟CSV；丢弃旧方法预测/误差列。
- 无新原始trace读取；本轮复用现有小型源成本产物和既有真实目标时钟。
- 六种方法合并为168条展示/下载行；这些方法分批研究封存，不宣称同时预注册或独立盲测。网页只汇总证据，不重算预测器。

## 复现

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/16to256/reproduce_prediction_three_stage_v1.py
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/16to256/reproduce_evaluation_three_stage_v1.py --prediction-run <上一条输出目录>
python -B docs/w37/16to256/build_methods_html.py
python -B docs/w37/16to256/publish_methods_html.py
```

数值命令每次新建目录；网页生成器默认引用本文封存运行。HTML发布到已有8038服务目录，旧单文件另行归档。访问：http://192.168.0.49:8038/w37-16to256.html#three-stage 。

方案完成：已建立仅用16卡校准的FWD/BWD/OPT三阶段外推公式、来源契约和封存评价流程。

实验验证完成：已完成256卡20点预测与误差评价，预测Step12.9662s、MFU5.8853%，Training MAPE44.8083%、MFU相对MAPE81.5729%，结果加入在线HTML。

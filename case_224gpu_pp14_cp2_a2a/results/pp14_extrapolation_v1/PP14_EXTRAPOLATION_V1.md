# 224-GPU PP14 MFU 外推 v1

> 结构校验通过；Fabric 224 卡实测已完成单点与 20-iteration 校验。iteration 55
> 单点接近，但时间漂移 P90 未通过，且 v1 时间 DAG 仍是源工作量守恒模型，不能
> 标记为 224 卡已完成校准。

## 策略与口径

- 源策略：`PP16 / CP2 / DP8 / EP8 / TP1 = 256 GPU`；
- 目标策略：`PP14 / CP2 / DP8 / EP8 / TP1 = 224 GPU`；
- 保持 16 条 pipeline lane、dense communicator 16 Rank 和 expert communicator 2 Rank；
- 时间 DAG 仍保持源 Trace 的 4 个 microbatch 和 60 层工作量；真实目标是 3 个 microbatch、52 层，因此该时间外推是诊断基线，不是配置完整的 v2；
- MFU 分子已从源 `1.293891072e+17` FLOPs 按 60 层/GBS64 → 52 层/GBS48 的解析工作比修正为目标 `8.436548312e+16` FLOPs；
- 对每条 `lane × phase × microbatch` 路径严格守恒 16-stage Trace 的计算总量；
- 对每类 DP/EDP RS/AG 严格守恒 aggregate payload，网络 service 按 payload 比例缩放；
- PP 网络 FCT `4.712346 ms` 和 backward 软件 completion `93.463844 ms` 暂沿用 256 卡校准。

## 120 场结构不确定性

PP16 压到 PP14 时，v1 紧凑输入没有带入 layer-to-stage 映射，因此枚举省略两个源 stage 模板的全部 `C(16,2)=120` 种组合，并将剩余模板按原顺序重映射到 PP14。Fabric 工程现已确认实际映射为 stage0/13 各 2 层、中间 stage 各 4 层；后续 v2 应直接使用该映射和 3-microbatch schedule 替换 ensemble。

| 指标 | min | P10 | P50 | P90 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| ProfilerStep (ms) | 21799.076 | 21841.496 | 21871.568 | 22271.536 | 23109.080 |
| MFU (%) | 3.0857 | 3.1954 | 3.2505 | 3.2548 | 3.2607 |

上尾主要来自 endpoint 模板不确定性：源 stage 0/15 的全 Rank compute 总量为
36528.4/36036.4 ms，内部 stage 中位为
65185.4 ms；省略 stage 0 的场景中位 Step 比保留它高
406.9 ms。实际 PP14 layer map 到位后应直接替换这项 ensemble，
区间会明显收窄。

代表场景 `omit_01_07` 是 MFU 最接近 120 场中位数的实际场景：

| 指标 | 256 卡回放 | 224 卡代表外推 |
| --- | ---: | ---: |
| ProfilerStep | 21616.236 ms | 21871.512 ms |
| training-log Step | 22918.240 ms | 23173.517 ms |
| 配置匹配固定 FLOPs MFU | 4.4107% | 3.2505% |
| DP/EDP service 最终暴露 | N/A（sweep 关闭反事实） | 259.389 ms |

## 校验边界

已校验：224 Rank 完整网格、PP14×lane16、每条 compute route 纳秒级精确守恒、六类 payload 取整误差受控、group size 保持 dense16/EDP2、120 场输出有限且可复现。Fabric iteration 55 上，training-log Step 误差 `-2.117%`，目标 FLOPs MFU 相对误差 `+2.162%`；详见 `fabric_measurement_validation/FABRIC_MEASUREMENT_VALIDATION.md`。

尚未建模：52 层/3-microbatch 的真实 PP schedule、PP14 的 P2P 软件 completion、目标拓扑上的六类 collective FCT，以及运行后半程的性能退化。20 个 iteration 的 Step 绝对相对误差 P90 为 `8.929%`，未通过 5% 门槛。

# Fabric 224-GPU 实测校验

## 结论

当前 v1 在对齐的 iteration 55 上，training-log Step 低估
`2.117%`；用目标配置 FLOPs 重算后，MFU
预测 `3.2505%`，实测
`3.1817%`，相对高估
`2.162%`（绝对
`0.0688` 个百分点）。

修正前/源 FLOPs 误用口径会输出
`4.9853%`。它沿用了源任务 60 层/GBS64 的
iteration FLOPs，和实测 52 层/GBS48 不是同一分子；若直接比较会相对高估
`56.684%`，只作为缺陷审计项保留。

## 对齐口径

| 项目 | 256 卡源任务 | 224 卡实测任务 |
| --- | ---: | ---: |
| 层数 | 60 | 52 |
| Global batch | 64 | 48 |
| Microbatch 数 | 4 | 3 |
| World size | 256 | 224 |

源日志校准 FLOPs 为 `1.293891072e+17`；按已审计
MLA/MoE 解析式缩放到目标配置后为
`8.436548312e+16`，缩放比
`0.652029255`。224 日志用“报告 TFLOP/s ×
Step × 224”反推的中位数为 `8.430678368e+16`，
两者只差 `-0.0696%`，说明目标 MFU 分子已
对齐。

## Iteration 55 误差分解

| 指标 | 当前预测 | Fabric 实测 | 误差 |
| --- | ---: | ---: | ---: |
| ProfilerStep cluster envelope | 21871.512 ms | 22101.211 ms | -1.039% |
| ProfilerStep 外残差 | 1302.005 ms | 1573.389 ms | -17.248% |
| training-log Step | 23173.517 ms | 23674.600 ms | -2.117% |
| 配置匹配 MFU | 3.2505% | 3.1817% | +2.162% relative |

ProfilerStep 实测取 224 Rank 的 `min(start_ns) → max(end_ns)` cluster envelope，
和 256 卡 `iteration_clocks.csv` 的构造口径一致。训练日志的额外误差主要来自
沿用源 iteration 55 的 outer residual：目标实测比沿用值多
`271.384 ms`。

## 20 个 profiler iteration 的稳定性

固定使用当前单点预测时，20 个 iteration 的 Step 绝对相对误差 P50 为
`2.640%`、P90 为
`8.929%`；配置匹配 MFU
相对误差绝对值 P90 为
`9.805%`。
iteration 55 单点通过 3% 门槛，但全时段 P90 未通过 5% 门槛，后半段明显变慢；
因此当前结果是“单点外推接近、时间漂移泛化失败”，不能据此宣布 224 卡模型已
完成校准。

逐 iteration 明细见 `fabric_validation_by_iteration.csv`，机器可读结论见
`fabric_validation_summary.json`。

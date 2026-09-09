# 两项研究的独立接口

共同基线只定义接口，未实现新的算子模型或16→256外推模型。两任务均可使用已提交的通用1F1B工厂，不等待对方未提交的改动。

## 结构与成本分别交付

| 接口 | 最小字段 | 限制 |
|---|---|---|
| scenario.json | case_id、world_size、pp/dp/cp/ep、microbatches、layer_placement、model/sequence/GBS、topology_id | 只含训练代码、启动配置或用户确定的静态值；未知填null并标PARTIAL |
| nodes.csv | node_id、rank、pp_stage、microbatch、layer_id、op_kind、stream/resource、cost_key | 通用结构不携带目标观测时间；无耗时来源不能伪装为已校准 |
| edges.csv | src、dst、edge_type、dependency_source | F沿PP递增、B沿PP递减；代码/调度语义决定依赖；不以误差添加等待边 |
| parameters.json | cost_key、value、unit、component、fit_case_id、fit_iterations、evidence_path、sha256 | B的所有经验时长只能来自16卡或独立硬件输入；256卡拟合参数禁止导入 |
| collective_request | kind/op、group_ranks、payload_bytes、rank_release_offsets_ns、topology_id、channels、slice_bytes | rank到达由预测图生成；service接口不重复加同步等待 |
| predictions.csv | iteration、predicted_profiler_step_ms、predicted_outer_framework_ms、predicted_training_step_ms、可选predicted_mfu_pct | training=profiler+outer；MFU依赖已确认FLOPs和硬件峰值，缺失则不可报告 |
| prediction_seal.json | 配置/图/参数/预测绝对路径与SHA、source/target边界、生成版本 | 评分前封存；evaluator-only目录持有目标时序和误差 |

网络后端可以返回 `first_arrival + collective_elapsed` 或 `last_arrival + tail_after_last_release`。两式只能择一，不能在last_arrival上再次加完整elapsed。PP暂用各自源case的实测性能；B不能复制256卡PP时序作为16→256源侧参数。

## 任务A：1F1B误差优化

允许入口：当前v6.8.5成本图与256中间表；224只作封存后开发集评分。参数变更优先保持v6.8.4拓扑SHA；若训练代码证明依赖有误，需要新语义版本、逐边证据和回归，不能为了精度恢复完整F→B观测等待。

建议新增目录：`research/w37/onef1b/`、`tests/w37/onef1b/`、`docs/w37/onef1b/`，结果到本worktree的`results/w37/A/`。保留旧版本文件可复现；通用接口代码修改先写明schema版本。交付同窗口逐轮时长/误差与准备、前后向、收尾的时间分账。

## 任务B：16→256方案

默认不读取任何256时长、残差、reported TFLOP/s、同步耗时或含拟合cost的DAG。现有v6.8.5是架构参考/隔离测试对象，不是B性能基线。目标256静态配置须逐字段登记来源，不能把Trace抽取的token分布或观察到达当作静态输入。

建议新增目录：`research/w37/scaleout/`、`tests/w37/scaleout/`、`docs/w37/scaleout/`，结果到本worktree的`results/w37/B/`。先独立确定16源case、参数/切分、层与MB可比性、CP1→CP2新增通信的输入来源及验证矩阵。目标256时序保留evaluator-only，预测封存后评分。此前256已被人观察的事实需在未来报告明确，不宣称全新盲测。

两任务按共同baseline SHA交付各自commit与参数清单；跨分支复用只能引用明确提交或经主线负责人评审的接口版本，不读取兄弟worktree里的临时结果。两个分支的本地提交不授权互相合并或推送。

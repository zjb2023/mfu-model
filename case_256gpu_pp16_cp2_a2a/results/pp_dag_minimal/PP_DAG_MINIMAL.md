# 最小版 PP-DAG（Trace 驱动）

## 已经建成什么

本版把 iteration 55 的每个 rank 拆成 4 个 FWD 与 4 个 BWD 计算节点，并用逐 microbatch 的 PP 消息节点连接。正向依赖为 `PP0 → PP15`，反向依赖为 `PP15 → PP0`。等待时间由 max-plus DAG 自动算出，没有输入任何 stage 的平均 bubble。

| 项目 | 结果 |
| --- | ---: |
| 计算节点 | 2,048 |
| 唯一 PP 消息节点 | 1,920 |
| 依赖边 | 7,312 |
| PP lane | 16 |
| 单条 80 MiB PP service proxy | 4.712346 ms |
| lane 前段 makespan P50 | 18330.071 ms |
| Trace 前段包络 P50 | 20024.309 ms |
| v0 前段回放误差 P50 | -8.464% |
| lane 前段 makespan范围 | 18248.205–18418.652 ms |
| 单计算节点等待 P50 / P90 | 179.232 / 4050.713 ms |
| 已连接 optimizer frontier 的 rank | 256 / 256 |
| BWD→首个 DP-RS 实测软件 lag P50 | 107.008 ms |

## DAG 的创建逻辑

1. 每个 rank 先按真实 Megatron 非交错 1F1B 顺序串起本地计算。
2. `F(s,m)` 完成后产生正向消息，消息完成后 `F(s+1,m)` 才能开始。
3. `B(s,m)` 完成后产生反向消息，消息完成后 `B(s-1,m)` 才能开始。
4. 当前训练关闭 P2P overlap，因此发送完成也约束发送 rank 的下一个计算节点。
5. 每个 rank 最后一个 BWD 节点输出到 `optimizer_frontier.csv`。表中保留实测 BWD→首个 DP-RS 的软件尾部 lag，供下一步接入现有 optimizer DAG。

关键 stage 的本地顺序：

- PP0：`F0 F1 F2 F3 B0 B1 B2 B3`
- PP13：`F0 F1 F2 B0 F3 B1 B2 B3`
- PP14：`F0 F1 B0 F2 B1 F3 B2 B3`
- PP15：`F0 B0 F1 B1 F2 B2 F3 B3`

## 当前边界

这是 v0 的**数据依赖 DAG**。FWD/BWD 节点时长直接取 Trace 注解，因此其内部的 CP/EP/重计算仍作为整体保留；九类集合通信尚未拆成子节点。PP 消息使用可替换的 FCT，尚未加入多链路竞争与 fused send/recv 的显式 post 节点。当前相对 Trace 前段包络仍有 -8.464% 的残差，这个数被明确保留为下一版的软件调度校准目标，不能塞进链路 service。optimizer 已有可连接的 frontier，但本次结果尚未把两张 DAG 合成一个最终 Step/MFU 数字。

查看 `pp_dag_minimal.html` 可切换 16 条 PP lane；灰色空档就是依赖产生的等待，而不是预先填写的 bubble。

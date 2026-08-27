# PP + RS/AG v1 回放误差分析

## 结论

iteration 55 的合并 DAG 将 ProfilerStep 预测为 19.970490 s，实测为 21.638095 s，少算 1.667605 s（-7.7068%）。固定模型 FLOPs 下，MFU 因分母偏小而从实测 4.4065% 被高估到 4.7519%：绝对偏差 +0.3454 个百分点，相对偏差 +7.8393%。

误差主体不在 RS/AG。optimizer RS/AG 使用实测 rank 到达时间独立回放时，ProfilerStep 最大误差只有 0.000191 ms；接入 PP frontier 后的 1.667605 s 误差，几乎全部继承自 PP v0 前段的 1.694788 s 残差。

## 误差定位

| 位置 | DAG 相对 Trace 提前量 P50 | 解释 |
| --- | ---: | --- |
| PP0 F3 开始 | 355.560 ms | PP0 的部分 `send_forward` 远高于 4.7 ms service，但较多被后续反向等待覆盖 |
| PP0 B0 开始 | 1,417.902 ms | 首个 backward 波从 PP15 返回 PP0 时逐 stage 累积的主要误差 |
| PP0 B1 开始 | 1,503.961 ms | B0 后继续少算约 86 ms |
| PP0 B2 开始 | 1,593.932 ms | 再少算约 90 ms |
| PP0 B3 开始 | 1,694.788 ms | 再少算约 101 ms，形成 PP 前段最终残差 |

B0 的提前量从 PP15 的 14.292 ms 增长到 PP0 的 1,417.902 ms。15 个 backward stage hop 的增量平均为 93.574 ms/hop，中位为 94.138 ms/hop。

这与 Trace P2P API 的观测一致：

| 量 | iteration 55 P50 |
| --- | ---: |
| v0 单条 PP service proxy | 4.712 ms |
| `send_backward`（全调用） | 98.141 ms |
| 两者差值 | 93.429 ms |
| B0 实际逐 hop 少算均值 | 93.574 ms |

因此，v0 把“消息完成”简化成固定 4.712 ms 后，遗漏了约 93–94 ms/hop 的 framework/P2P completion 开销。它可能包含 peer rendezvous、send/recv posting 时序、CPU launch、MCCL completion 语义和软件调度；它不是纯网络传输时间，不能通过放大 OISA/NS-3 FCT 来隐藏。

## 不是主要误差的部分

- FWD/BWD annotation duration 直接来自 Trace，计算节点本身没有使用 Simumax 均值重新估计。
- BWD→首个 DP-RS 的软件尾部 lag 按 rank 实测保存，P50 为 107.008 ms。
- DP/Expert-DP RS/AG 的 group rendezvous、service、两轮 AG 和 post-AG residual 已接入；实测到达锚点下可近似恒等回放。
- RS/AG 合并后，端到端误差由 1.694788 s 变为 1.667605 s，只吸收约 27.183 ms，不能解释主体残差。

## 下一步校准

1. 从 Trace 提取 `recv_forward`、`send_forward`、`recv_backward`、`send_backward`、`send_forward_recv_backward`、`send_backward_recv_forward`。
2. 在 compute 与 PP service 之间增加 `post / peer-ready / completion` 节点。
3. 网络 service 仍由 OISA/NS-3 或带宽系数提供；软件 completion 单独按硬件集群 Trace 标定。
4. 先验证 backward 每 hop 的约 94 ms 残差能否被解释，再重新检查完整 Step 和 MFU。

在该层完成之前，v1 的 4.7519% 只能视作少建模 PP 软件等待时的 MFU 上界，不是已经达到 1% 精度的最终 MFU 模型。

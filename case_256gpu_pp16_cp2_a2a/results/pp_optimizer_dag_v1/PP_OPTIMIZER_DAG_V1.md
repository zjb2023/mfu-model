# PP + Optimizer RS/AG 合并 DAG v1

> `results/pp_dag_minimal/` 的 v0 页面、CSV 和说明全部保留。本目录是独立的合并版本。

## 合并结果

iteration 55 已从 FWD/BWD、双向 PP message 一直连接到 DP/Expert-DP 的 RS 与两轮 AG。首个 DP-RS 不再使用固定实测开始时间，而由每个 rank 的 `预测B3完成 + 实测软件尾部lag` 产生；后续 collective 使用 `max(组内rank到达) + service` 重新计算。

| 项目 | 结果 |
| --- | ---: |
| PP 计算节点 | 2,048 |
| PP message 节点 | 1,920 |
| optimizer rank collective 调用 | 1,536 |
| optimizer collective group | 432 |
| 合并依赖边 | 9,616 |
| 单条 PP service | 4.712346 ms |
| PP service 对最终 Step 的边际 | 164.932 ms |
| 实测 ProfilerStep | 21638.095 ms |
| 合并 DAG ProfilerStep | 19970.490 ms |
| ProfilerStep 回放误差 | -1667.605 ms / -7.707% |
| 实测 training-log Step | 22940.100 ms |
| 合并 DAG training-log Step | 21272.495 ms |
| 固定 FLOPs 实测 MFU | 4.4065% |
| 合并 DAG MFU | 4.7519% |
| DP/EDP service 对最终 Step 的边际 | 224.652 ms |

## 完整依赖

```text
FWD(s,m) → PP forward message → FWD(s+1,m)
                                      ↓
BWD(s,m) ← PP backward message ← BWD(s+1,m)
    ↓
rank B3 + measured software-tail lag
    ↓
DP-RS → Expert-DP RS → world RS done
    ↓
DP-AG0 → Expert-DP AG0 → DP-AG1 → Expert-DP AG1
    ↓
post-AG residual → ProfilerStep → outer residual → training-log Step → MFU
```

## 当前解释

本次已经完成“传播关系”的合并：修改 PP FCT 会改变各 rank 的 B3、RS 到达波次、RS/AG 完成时间、最终 Step 和 MFU；修改 RS/AG service 也会沿同一条链传播。

当前 -7.707% 的端到端回放误差主要继承自 v0 PP 前段少建模的软件/P2P API 等待，而不是 RS/AG 未连接。下一步校准目标仍是把 `recv/send/send_recv` 的 post 与完成节点加入 PP 图，不能通过放大纯链路 FCT 隐藏该残差。

详细分解见 `ERROR_ANALYSIS.md`：B0 backward 波逐 stage 少算约 93.6 ms/hop，与 `send_backward` P50 98.1 ms 减去 4.7 ms service 基本一致。

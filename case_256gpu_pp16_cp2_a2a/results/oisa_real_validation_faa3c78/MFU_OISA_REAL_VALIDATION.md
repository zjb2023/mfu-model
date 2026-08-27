# 真实 OISA FCT → MFU DAG 单组联调

> 这是一项接口和传播验证，不是目标 256-GPU 集群的网络校准。OISA 使用本地提交 `faa3c789549d542484ca9aab132b7d4727afd14a` 和固定 8-GPU 验证拓扑。

## 做了什么

1. 从 iteration 55 的 MFU 基线 DAG 选择零 downstream slack 的关键通信组：
   `iter55:pp14:edp_ag0:round0:expert_dp_param_allgather:pg=1279`；
2. 保留实测 payload `1,887,436,800` bytes 和 Rank arrival offsets；
3. 因 OISA smoke 拓扑只有 8 Rank，把实测 Rank `[230, 238]` 按 `230→0, 238→1` 映射到拓扑本地 Rank；
4. 分别运行同步到达和实测错峰到达的 OISA/ns-3；
5. 校验 flow release/dependency gate、FCT 恒等式、commit 和 topology hash；
6. 将错峰 OISA 返回通过 `RecordedOisaFctProvider` 只替换这个通信组，其他 431 组仍使用 Trace-reference FCT；
7. 重新执行完整 PP→RS/AG max-plus DAG，并使用 OISA 的逐 Rank network-done 时间替换该组的 Trace completion skew。

## 通用错峰接口 smoke

在同一 OISA commit/topology 上额外运行了 4-Rank、8 MiB AllToAll：

| 指标 | 同步到达 | `[0,20,50,30] ms` 错峰到达 |
| --- | ---: | ---: |
| flow start times | `[0]` | `[20000000, 30000000, 50000000]` |
| collective elapsed | 0.342997 ms | 50.342512 ms |
| tail after last release | 0.342997 ms | 0.342512 ms |

12 条 flow 的 start time 全部由 0 变成 20/30/50 ms 三个 release 波次，且 12 条 flow 的 FCT 全部变化，说明错峰不是在最终结果上简单加 offset，而是真正改变了 ns-3 中的流量竞争。

## OISA 结果

| 指标 | 同步 Rank | 实测错峰 Rank |
| --- | ---: | ---: |
| arrival span | 0.000000 ms | 0.087570 ms |
| collective elapsed | 152.088036 ms | 152.175606 ms |
| tail after last release | 152.088036 ms | 152.088036 ms |
| dependency gate | True | True |

本例的实测 arrival span 是 87.570 μs。错峰结果满足：

```text
collective_elapsed
  = arrival_span + tail_after_last_release
  = 87570 + 152088036
  = 152175606 ns
```

## 放回 MFU DAG 后

| 指标 | Trace-reference | 只换 group FCT | 再换 Rank done |
| --- | ---: | ---: | ---: |
| 该组网络尾部 | 66.614241 ms | 152.088036 ms | 152.088036 ms |
| ProfilerStep | 21616.235827 ms | 21701.709622 ms | 21701.775659 ms |
| 相对基线 Step 增量 | — | +85.473795 ms | +85.539832 ms |
| 固定 FLOP MFU | 4.410689% | 4.394301% | 4.394288% |

目标组在基线关键路径上没有 downstream slack，因此该组网络尾部增加量 85.473795 ms 完整传播到 Step。进一步使用 OISA 的逐 Rank network-done 后，Step 又变化 +66.037 μs；这部分不是额外 group FCT，而是原 Trace Rank completion skew 被替换后的依赖传播。目标组 Rank completion model 为 `oisa_rank_network_done`。

## 不能据此下的结论

- 当前 OISA 拓扑是 8-GPU 验证拓扑，实测 DP communicator 是 16 Rank，不能直接仿真 DP16；
- 实测 Rank 到本地 Rank 的映射只是接口验证映射，不等价于 256-GPU 目标集群物理放置；
- `n_channels=1`、collective 算法和 payload 口径仍需与训练通信库对齐；
- 因此 152.088 ms 只能证明真实 OISA 返回可正确进入 MFU DAG，不能作为最终 MFU 校准参数。

拓扑 hash：`042913fea6744a475de66b225949a2c997ba6b637a21534bd61c0885318ec426`。

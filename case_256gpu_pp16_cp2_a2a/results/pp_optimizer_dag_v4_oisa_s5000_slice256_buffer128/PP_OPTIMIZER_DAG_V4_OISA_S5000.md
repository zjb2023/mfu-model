# MFU DAG v4：256-GPU S5000 九类集合通信回填

> v4 保留 v2 的 PP send/recv 网络与软件 completion 校准，只替换 PP 之外九类集合通信的网络 service。它是“每类一个代表点 + 按字节线性缩放”的首版接入，不等同于多尺寸、多并发条件下的最终 OISA 性能曲线。

## 回放结果

| 场景 | ProfilerStep | 相对实测误差 | 固定 FLOPs MFU | 相对 v2 Step 变化 |
| --- | ---: | ---: | ---: | ---: |
| Trace-reference（v2 语义） | 21616.236 ms | -0.1010% | 4.4107% | — |
| 诊断：仅 OISA 网络，FWD/BWD 五类 | 15729.037 ms | -27.3086% | 5.9354% | -5887.199 ms |
| 诊断：仅 OISA 网络，optimizer 四类 | 21533.217 ms | -0.4847% | 4.4267% | -83.018 ms |
| 诊断：仅 OISA 网络，九类合并 | 15646.019 ms | -27.6923% | 5.9644% | -5970.217 ms |
| 校准：OISA 网络 + Trace 有符号残差，FWD/BWD 五类 | 21616.236 ms | -0.1010% | 4.4107% | +0.000 ms |
| 校准：OISA 网络 + Trace 有符号残差，optimizer 四类 | 21616.236 ms | -0.1010% | 4.4107% | +0.000 ms |
| 最终：九类合并 | 21616.236 ms | -0.1010% | 4.4107% | +0.000 ms |

两个单独变化不能相加；九类合并后会重新执行完整 max-plus DAG，通信变慢先消耗 slack，只有越过 slack 的部分才拖累 iteration。

当前 baseline topology 与 target topology 相同，所以校准场景的网络 delta 为 0；432 个 optimizer group 中有 6 个 downstream slack 为 0，426 个有正 slack。未来 target OISA tail 变慢且逐 Rank completion 结构不变时，单组拖累按 `max(0, target_network_delta - downstream_slack)` 计算；多个组同时变化仍必须完整重放 DAG，不能把单组拖累直接相加。

## 接入逻辑

```text
五类 EP/CP A2A:
  保留 Trace 的 rank arrival wait 与 completion advance
  → network = OISA tail
  → calibration residual = Trace group service - baseline OISA network
  → target service = target OISA network + signed calibration residual
  → service 差额分配回对应 rank 的 FWD/BWD compute node

DP/Expert-DP RS/AG:
  DAG 实时产生各 rank arrival
  → arrival offsets 交给 RepresentativeOisaFctProvider
  → 按 payload/reference_payload 缩放 baseline/target OISA network tail
  → 保留每个 Trace request 的有符号基线校准残差
  → group_end = first_arrival + arrival_span + target_network + calibration_residual
```

五类 A2A/CP 暂时嵌入 FWD/BWD node，是因为当前 PP DAG 没有把这些框架通信拆成独立 node；这不会把实测同步等待再次相加，但尚不能表达同一 phase 内更细粒度的 compute/communication overlap。DP/EDP 六种 optimizer kind（RS、AG0、AG1）本来就是独立 DAG node，使用四类 OISA 代表值覆盖。有符号残差的正值可解释为软件/同步残差；负值表示当前 OISA 算法、payload 语义或重叠行为相对 Trace 的系统偏差，不能误称为负软件开销。

PP 通信不在这九类中，仍沿用 v2：网络 service 4.712346 ms，backward 软件 completion 93.463844 ms。

## 九类代表点

| behavior | group size | payload MiB | Trace service ms | OISA tail ms | OISA/Trace |
| --- | ---: | ---: | ---: | ---: | ---: |
| cp_all_to_all | 2 | 128.000 | 7.493 | 2.426 | 0.324× |
| dp_grad_reduce_scatter | 16 | 3009.688 | 15.881 | 16.865 | 1.062× |
| dp_param_allgather | 16 | 94.053 | 7.408 | 8.224 | 1.110× |
| ep_bwd_combine_backward_dispatch | 8 | 80.000 | 0.998 | 0.218 | 0.218× |
| ep_bwd_dispatch_backward_combine | 8 | 79.194 | 1.047 | 0.215 | 0.206× |
| ep_fwd_combine | 8 | 79.194 | 1.005 | 0.215 | 0.214× |
| ep_fwd_dispatch | 8 | 78.984 | 1.702 | 0.215 | 0.126× |
| expert_dp_grad_reduce_scatter | 2 | 7200.000 | 208.459 | 79.835 | 0.383× |
| expert_dp_param_allgather | 2 | 1800.000 | 53.615 | 39.919 | 0.745× |

## 可审计边界

- OISA simulator commit：`1fb9fa6cd750b63b3358f6b0ff5d8167de4143d1`；
- 仿真运行时 worktree dirty：`true`；对应完整源码快照已提交到 `feat/mfu-multiring-fct` / `4e29411cff7bb38a561289796723aee49fd6f21a`；
- simulator binary SHA256：`0a3fb5a76affee76476943b81add670c519b803ca49e626f95b2d2bccd927a11`；
- topology hash：`e9503e0eaf5bc80efd66ae95b5f09e68643725069d654082f374ec03f8b46a88`；
- network-only pipeline node 调整总量：-320113.992 ms；
- 保留有符号基线校准残差后的 pipeline node 调整总量：+0.000 ms；
- optimizer 动态请求数：432；
- pipeline 对齐覆盖：100.00%；
- 最终状态：`PASS`。

下一步若要承担并行策略外推，需对每类补充 payload、group size、跨机比例和并发度的多点 OISA 曲线；当前线性单点模型首先用于验证 service 回填、slack 隐藏和关键路径传播是否正确。

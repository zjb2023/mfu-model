# PP + Optimizer DAG v3：OISA FCT 接口演练

> 本目录保留 v0/v1/v2，v3 不声称预测了真实网络。它使用 `Mock OISA = Trace 网络尾部 × 1.200` 验证接口、依赖传播和 slack 计算，等真实 OISA 返回后只替换 Provider。

## 本次接入的准确语义

```text
MFU DAG 计算每个 rank 的 arrival
  → release_offset[r] = arrival[r] - min(arrival)
  → OISA(rank_release_offsets, op, payload, group)
  → collective_elapsed = arrival_span + tail_after_last_release
  → group_end = min(arrival) + collective_elapsed
```

模型不会使用 `max(arrival) + collective_elapsed`，因此 arrival span 不会被重复计算。网络 FCT 只替换 service；每个 rank 相对 group end 的 completion skew 仍来自 Trace，并与网络缩放解耦。

## Mock 场景结果

| 指标 | Trace-reference FCT | Mock OISA | 变化 |
| --- | ---: | ---: | ---: |
| ProfilerStep | 21616.236 ms | 21661.716 ms | +45.480 ms |
| 固定 FLOP MFU | 4.4107% | 4.4020% | -0.0087 pp |
| 相对实测 Step 误差 | -0.1010% | +0.1092% | — |

这里把六类已显式建模的 optimizer collective 网络尾部统一增加 20.0%。这是接口测试，不是 OISA 性能结论。

## Slack 审计

本次对 432 个 collective group 计算“只替换该组 FCT”的独立反事实敏感性。实现上只回放一次基线 DAG，再通过反向 max-plus 传播得到每组到 iteration 终点的 downstream slack；它与逐组重跑的正向 slowdown 结果等价：

- 完全隐藏：400 组；
- 至少拖累 iteration：32 组；
- 先消耗部分 slack、随后拖累：26 组。

单组字段定义：

```text
group_finish_shift = candidate_group_end - reference_group_end
iteration_drag     = candidate_step_end  - reference_step_end
hidden_by_slack    = group_finish_shift  - iteration_drag
exposure_ratio     = iteration_drag / group_finish_shift
```

各组 `iteration_drag` 是独立敏感性实验，不能直接相加；多个通信同时变慢时必须像本页 Mock 场景一样重新跑完整 DAG。

| kind | stage | group | 网络尾部增加 ms | Step 拖累 ms | slack 隐藏 ms | 暴露比例 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| edp_ag0 | PP14 | `pg=1279` | 13.323 | 13.323 | 0.000 | 100.0% |
| edp_ag1 | PP14 | `pg=1277` | 11.603 | 11.603 | 0.000 | 100.0% |
| edp_ag0 | PP14 | `pg=1281` | 12.602 | 11.557 | 1.045 | 91.7% |
| edp_rs | PP0 | `pg=1051` | 11.307 | 11.307 | 0.000 | 100.0% |
| edp_rs | PP0 | `pg=1049` | 11.318 | 11.306 | 0.012 | 99.9% |
| edp_rs | PP0 | `pg=1055` | 11.290 | 11.120 | 0.170 | 98.5% |
| edp_rs | PP0 | `pg=1045` | 11.176 | 10.473 | 0.704 | 93.7% |
| edp_ag1 | PP14 | `pg=1279` | 11.173 | 9.060 | 2.113 | 81.1% |

## 可替换边界

- `oisa_mock_requests.csv`：MFU 发给 OISA 的请求形状，包含各 rank release offsets；
- `oisa_mock_responses.csv`：符合约定的 OISA 返回形状；
- `collective_slack_audit.csv`：逐组独立 slowdown 的隐藏/暴露结果；
- `optimizer_groups.csv`：完整 Mock 场景传播后的 group 时间；
- `pp_optimizer_dag_v3_oisa_mock.html`：Mock 场景的完整关键 lane 时间线。

真实 OISA 上线后，Provider 必须返回 `collective_elapsed_ns`、`arrival_span_ns` 和 `tail_after_last_release_ns`，并满足：

```text
collective_elapsed_ns
  = arrival_span_ns + tail_after_last_release_ns
```

# Trace-driven MFU model

This repository contains the standalone 256-GPU MFU replay developed for a
PP16 / CP2 / EP8 / DP8 training capture.  It builds a max-plus dependency graph
from per-microbatch FWD/BWD trace annotations, pipeline P2P messages, and
DP/Expert-DP RS/AG process-group facts.

The current `v2` model keeps network and software costs separate:

```text
backward PP completion
  = 4.712346 ms network service
  + 93.463844 ms cluster software completion
```

The software term is the median of 848 `send_backward` annotations from PP
stages 1–14.  It is not folded into the OISA/NS-3 network FCT.

## Current replay result

For captured iteration 55:

| Metric | Measured | Predicted | Error |
| --- | ---: | ---: | ---: |
| ProfilerStep | 21,638.095 ms | 21,616.236 ms | -0.1010% |
| Fixed-FLOP MFU | 4.4065% | 4.4107% | +0.0954% relative |

This is an in-sample replay result for one fully materialized iteration.  The
next validation step is to hold out additional captured iterations before
treating the software-completion parameter as a general predictor.

## V4：真实 OISA 九类集合通信回填（当前版本）

v4 已在目标 256-GPU S5000 拓扑上完成 PP 之外九类代表请求：四类 EP
AllToAll、CP AllToAll、DP RS/AG、Expert-DP RS/AG。每条请求保留实测 Rank
release offsets，9/9 均由 runner 自然退出并生成标准结果。

```text
topology = 32 host × 8 GPU
in-host  = 448 Gbps fullmesh
scale-out = 每 GPU 400 Gbps NIC，32 leaf → 1 spine
```

- topology：`/tmp/oisa-rank-release-fct/topologies/s5000_256gpu_32host_1spine`；
- topology SHA256：`e9503e0eaf5bc80efd66ae95b5f09e68643725069d654082f374ec03f8b46a88`；
- Release simulator：`/tmp/oisa-rank-release-fct/bin/OISA_simulator_release`；
- 仿真运行时记录的 base commit：`faa3c789549d542484ca9aab132b7d4727afd14a`
  （结果中同时标记 `simulator_worktree_dirty=true`）；对应的完整源码快照随后已
  提交并推送为 `2bd0e53`，分支 `feat/rank-release-fct`。

绝对 OISA tail 不能直接覆盖整个 Trace collective tail。以 DP-RS 代表点为例，
OISA 为 `380.119 ms`，Trace 在最晚 Rank 到达后的尾部只有 `15.881 ms`，其中
包含算法、payload 语义和通信已发生重叠等系统性偏差。v4 因此使用基线差分
校准：

```text
signed_residual = Trace_tail - baseline_OISA_network_tail
target_tail = target_OISA_network_tail + signed_residual
            = Trace_tail + (target_OISA - baseline_OISA)
```

正残差可包含软件/同步开销；负残差记录 simulator/算法/重叠相对 Trace 的
系统偏差，不能解释为负软件开销。这样新 topology 只通过 OISA 网络 delta
改变 DAG，实测同步与软件栈行为不会被误删。

在 baseline 与 target 为同一硬件时，九类合并模型保持 v2 的回放结果：
ProfilerStep 误差 `-0.1010%`，MFU 相对误差 `+0.0954%`。作为诊断，如果
只保留绝对 OISA 网络值并删除校准残差，Step 会变成 `-24.2162%`；该场景
只用于证明“纯网络 FCT ≠ 端到端 collective 时间”，不是最终 MFU 预测。

optimizer 的 432 个 group 已完成 downstream slack 审计：6 个 slack 为 0，
426 个有正 slack。单组网络变慢的暴露量为
`max(0, network_delta - downstream_slack)`；多组同时变化仍需完整重放
max-plus DAG。

主要产物：

- `case_256gpu_pp16_cp2_a2a/results/oisa_s5000_256gpu_nine_class/oisa_results.csv`
- `case_256gpu_pp16_cp2_a2a/results/oisa_s5000_256gpu_nine_class/backfill/`
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v4_oisa_s5000/validation.json`
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v4_oisa_s5000/collective_slack_audit.csv`
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v4_oisa_s5000/PP_OPTIMIZER_DAG_V4_OISA_S5000.md`

## V3：OISA FCT 接入边界（历史 Mock）

`v2` 是 Trace-reference 校准基线。`v3` 在其上增加 OISA Provider
接口和 slack 审计；当时真实 OISA 的错峰 Rank 仿真仍在开发，因此 v3
结果明确使用 `Mock OISA = Trace 网络尾部 × 1.2`，只验证接入逻辑，不代表
网络性能预测。

每个 optimizer collective group 的调用方式为：

```text
MFU DAG 计算 rank arrivals
  → rank_release_offset[r] = arrival[r] - min(arrival)
  → OISA(op, payload, group_ranks, rank_release_offsets)
  → collective_elapsed = arrival_span + tail_after_last_release
  → group_end = min(arrival) + collective_elapsed
```

因此模型不会错误使用 `max(arrival) + collective_elapsed`。OISA 只替换
网络 FCT；Trace 中各 Rank 相对 group end 的 completion skew 独立保留，
不会再随网络 service 一起缩放。

当前 Mock 场景覆盖 DAG 已经显式建模的 432 个 DP/EDP RS、AG group。
六类网络尾部统一增加 20% 后，400 组完全被 downstream slack 隐藏，完整
ProfilerStep 增加 45.480 ms。这个结果说明多个 FCT 增量不能直接求和，
必须放回 max-plus DAG 重新传播。

真实 OISA 结果可以直接由 `RecordedOisaFctProvider` 读取。请求 ID 包含
Rank release offsets 的签名；如果上游时间变化导致 offsets 不同，旧结果会
被拒绝，并提示重新运行 OISA，而不会静默复用不匹配的 FCT：

```python
records = pd.read_csv("oisa_results.csv").to_dict(orient="records")
provider = RecordedOisaFctProvider(records)
result = build_unified_mfu_dag(optimizer_fct_provider=provider, ...)
```

有限的结果表只覆盖这一组 arrival signatures，因此模型会关闭额外的
`no_all_dp/no_rs/no_ag` 边际查询，并在结果中设置
`service_marginals_available=false`。如果 OISA 以在线 Provider 形式响应新
offsets，则可以继续计算这些动态反事实。

## 真实 OISA 单组联调

已使用 `/tmp/oisa-rank-release-fct` 的本地提交
`faa3c789549d542484ca9aab132b7d4727afd14a` 做过一次真实 ns-3 联调：

- 4-Rank AllToAll 从同步启动变为 `[0,20,50,30] ms` 错峰后，12 条 flow
  的实际 start time 和 FCT 全部变化，证明 release offset 真正进入了网络
  仿真，而不是只在最终 elapsed 上加常数；
- 从 iteration 55 选择零 downstream slack 的 EDP-AG0 group，保留实测
  1,887,436,800-byte payload 和 87.570 μs arrival span；
- OISA 返回 `collective_elapsed=152.175606 ms`，满足
  `87.570 μs + 152.088036 ms tail`；
- 只替换 group FCT 时 Step 增加 `85.473795 ms`；继续替换 OISA 逐 Rank
  network-done 后再增加 `66.037 μs`，验证 service 与 Rank completion
  必须分开传播。

该结果只验证接口和 DAG 传播。当前 smoke 入口固定 8-GPU 拓扑、
`n_channels=1`，而实测 DP communicator 是 16 Rank，因此不能把
`152.088036 ms` 当成目标 256-GPU 集群的校准 FCT。

## V1 到 V2：PP backward completion 校准

V1 已经把实测 FWD/BWD compute、PP16 的 1F1B 依赖以及 DP/Expert-DP
的 RS、AG0、AG1 合并进同一张 DAG，但它把每条 PP 消息都简化成
`4.712346 ms` 的纯网络 FCT。因此 iteration 55 的 ProfilerStep 只预测到
`19,970.490 ms`，比实测少 `1,667.605 ms`（`-7.7068%`）。

逐 stage 对齐 Trace 后可以看到，首个 backward 波从 PP15 向 PP0 传播时，
回放提前量平均每经过一个 stage 增加约 `93–94 ms`。这说明主要残差不在
RS/AG，而在 PP backward 的框架/通信库完成路径。

### PP API 的含义

这里的 PP API 是训练框架在 profiler Trace 中记录的高层流水线通信调用，
包括：

- `send_forward` / `recv_forward`
- `send_backward` / `recv_backward`
- `send_forward_recv_backward`
- `send_backward_recv_forward`

一个 PP API annotation 的 duration 是函数从进入到返回的时间，可能同时
包含 CPU 发起、peer rendezvous、网络传输、通信库完成、同步等待和框架
调度。因此它不是 OISA/NS-3 意义下的纯网络 FCT。

### backward 参数如何得到

V2 使用 iteration 55 中 PP stage 1–14 的 848 个、名称严格为
`send_backward` 的 annotation。stage 0 的 endpoint/no-op 和 stage 15 的
额外 drain tail 不参与校准。观测分布为：

| 分位数 | `send_backward` duration |
| --- | ---: |
| P10 | 97.087 ms |
| P50 | 98.176 ms |
| P90 | 99.691 ms |

V2 使用中位数，并显式拆开网络和当前集群的软件栈成本：

```text
modeled backward completion
  = 4.712346 ms network service
  + 93.463844 ms software completion
  = 98.176190 ms
```

`93.463844 ms` 只属于当前 Trace 所在硬件集群的软件完成参数，不会被
折叠进 OISA/NS-3 的网络 FCT。

### 参数应用到哪些 PP 边

该参数不是只应用于 `PP15→PP14`，而是当前统一应用于每条 backward PP
边：

```text
PP15→PP14, PP14→PP13, ..., PP1→PP0
```

PP16 有 15 条相邻 stage 边；乘以 4 个 microbatch 和 16 条 PP lane，
完整 iteration 中共生成 `15 × 4 × 16 = 960` 个 backward message 节点。
每个节点目前都保存：

```text
network_service_ns       = 4.712346 ms
software_completion_ns   = 93.463844 ms
duration_ns              = 98.176190 ms
```

每条反向路径的依赖保持为：

```text
上游 stage 的 BWD compute
  → backward message（network + software completion）
  → 下游 stage 对应 microbatch 的 BWD compute
```

同时，blocking-send 边要求发送 rank 等该 message 完成后才能执行下一项
本地操作。因此等待会沿 `PP15→...→PP0` 的真实 DAG 自然传播，而不是给
PP0 或某个代表 rank 人工附加一个总等待时间。

### 当前结论与边界

反向传播方向、相邻 stage 依赖和 blocking-send 结构已经覆盖全部 PP hop，
而且统一的 backward completion 参数把 Step 误差从 `-7.7068%` 降到
`-0.1010%`。不过，这仍不能证明每一条局部边都已经精确无误：

- 目前所有 stage 和 microbatch 共用同一个中位数；
- stage 15 没有进入参数校准，但 `PP15→PP14` 暂时也使用该统一参数；
- 组合 PP API 尚未按调用类型分别建模；
- forward 方向目前仍只使用网络 FCT，没有单独的软件 completion；
- iteration 55 是样本内回放，端到端误差小可能掩盖局部正负误差抵消。

下一步需要在其余 iteration 上做留出验证，并检查逐 stage、逐 microbatch
的局部到达误差，再决定采用单一分布、按方向分布，还是按 stage/API 类型
分层校准。

## Repository layout

- `src/x10000_analysis/pp_dag.py`: non-interleaved 1F1B schedule and PP DAG.
- `src/x10000_analysis/mfu_timeline.py`: DP/Expert-DP RS/AG max-plus model.
- `src/x10000_analysis/oisa_fct.py`: OISA request/result contract and mock provider.
- `src/x10000_analysis/unified_mfu_dag.py`: dynamic PP-frontier to RS/AG join.
- `case_256gpu_pp16_cp2_a2a/data/`: frozen, compact model inputs; no raw traces.
- `case_256gpu_pp16_cp2_a2a/results/pp_dag_minimal/`: preserved PP-only v0.
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v1/`: preserved merged v1.
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v2/`: calibrated current model.
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v3_oisa_mock/`: OISA
  interface exercise, request/response examples, and per-group slack audit.
- `case_256gpu_pp16_cp2_a2a/results/oisa_real_validation_faa3c78/`: real OISA
  smoke evidence and one-group MFU replay, with topology limitations recorded.
- `case_256gpu_pp16_cp2_a2a/results/oisa_topology_preflight_s5000_256_1spine/`:
  corrected global-rank OISA preflight on the 32-host S5000 topology.
- `case_256gpu_pp16_cp2_a2a/results/oisa_s5000_256gpu_nine_class/`: compact
  requests, nine real OISA results, and Trace/OISA baseline calibration tables.
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v4_oisa_s5000/`: current
  nine-class calibrated replay, network-only diagnostic, slack audit, and HTML.

## Rebuild

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'

.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_dag.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_optimizer_dag_v1.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_optimizer_dag_v2.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_optimizer_dag_v3.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/validate_real_oisa_fct.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_oisa_nine_class_requests.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/run_oisa_nine_class.py \
  --simulator /tmp/oisa-rank-release-fct/bin/OISA_simulator_release
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python \
  case_256gpu_pp16_cp2_a2a/scripts/build_oisa_nine_class_backfill.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_optimizer_dag_v4.py
.venv/bin/pytest -q
```

View the current self-contained dashboard:

```bash
python3 -m http.server 8013 --bind 0.0.0.0
```

Then open:

`http://127.0.0.1:8013/case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v4_oisa_s5000/pp_optimizer_dag_v4_oisa_s5000.html`

局域网访问时把 `127.0.0.1` 换成本机局域网 IP；8013 当前监听
`0.0.0.0`。

View the Mock-OISA integration exercise:

`http://127.0.0.1:8013/case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v3_oisa_mock/pp_optimizer_dag_v3_oisa_mock.html`

View the real-OISA single-group validation:

`http://127.0.0.1:8013/case_256gpu_pp16_cp2_a2a/results/oisa_real_validation_faa3c78/pp_optimizer_dag_oisa_real_single_group.html`

## Data boundary

The repository includes only derived timing facts needed to rebuild the model.
The original profiler JSON, training logs, topology dumps, and multi-gigabyte
kernel tables are intentionally excluded.  The optional `import_pp_*` scripts
require explicit `--source-case` and `--trace-root` paths when refreshing facts
on an authorized capture host.

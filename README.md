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
- `src/x10000_analysis/unified_mfu_dag.py`: dynamic PP-frontier to RS/AG join.
- `case_256gpu_pp16_cp2_a2a/data/`: frozen, compact model inputs; no raw traces.
- `case_256gpu_pp16_cp2_a2a/results/pp_dag_minimal/`: preserved PP-only v0.
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v1/`: preserved merged v1.
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v2/`: calibrated current model.

## Rebuild

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'

.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_dag.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_optimizer_dag_v1.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_optimizer_dag_v2.py
.venv/bin/pytest -q
```

View the current self-contained dashboard:

```bash
python3 -m http.server 8013 --bind 127.0.0.1
```

Then open:

`http://127.0.0.1:8013/case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v2/pp_optimizer_dag_v2.html`

## Data boundary

The repository includes only derived timing facts needed to rebuild the model.
The original profiler JSON, training logs, topology dumps, and multi-gigabyte
kernel tables are intentionally excluded.  The optional `import_pp_*` scripts
require explicit `--source-case` and `--trace-root` paths when refreshing facts
on an authorized capture host.

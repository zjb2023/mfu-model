# PP + Optimizer RS/AG 校准 DAG v2

> v0 `pp_dag_minimal` 与 v1 `pp_optimizer_dag_v1` 全部保留。v2 只新增集群软件 completion 校准，不修改网络 FCT 的语义。

## 精度结果

| 项目 | v2 |
| --- | ---: |
| 实测 ProfilerStep | 21638.095 ms |
| 预测 ProfilerStep | 21616.236 ms |
| Step 误差 | -21.860 ms / -0.1010% |
| 实测 training-log Step | 22940.100 ms |
| 预测 training-log Step | 22918.240 ms |
| 固定 FLOPs 实测 MFU | 4.4065% |
| v2 MFU | 4.4107% |
| MFU 相对误差 | +0.0954% |

## 校准拆分

backward message 不再等同于纯网络 FCT：

```text
modeled backward completion
  = network service  4.712346 ms
  + software completion 93.463844 ms
  = 98.176190 ms
```

软件项来自 iteration 55、PP stage 1–14 的 848 个 `send_backward` annotation，其中位为 98.176190 ms。stage 0 no-op 和 stage 15 drain tail 没有混入校准。该参数属于当前硬件集群的软件栈，不应作为 OISA/NS-3 网络 FCT。

| 对最终 ProfilerStep 的边际 | 时间 |
| --- | ---: |
| backward 软件 completion | 1645.746 ms |
| PP 网络 service | 164.932 ms |
| DP/EDP RS/AG service | 224.652 ms |

## 使用边界

v2 达到的是 iteration 55 的样本内回放精度。它比直接拟合 100 ms 更保守：采用独立 Trace API 的 98.176 ms 中位数，没有把最终 Step 误差调成零。下一步应在其余 19 个 iteration 上做留出验证，再决定软件 completion 使用单一分布、按方向分布还是按 stage/调用类型分层。

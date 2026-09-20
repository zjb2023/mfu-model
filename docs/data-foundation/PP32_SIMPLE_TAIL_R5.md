# 32 卡简化更新尾段 r5

2026-09-18。按用户批准先实现简单尾段，保留 r4 和原 v610，不读取新 target70/256 数据，不报告完整迭代精度。

## 模型

各 stage 本地 F/B＋PP 完成 → RS_PREP；四个 RS_PREP → GLOBAL_READY；GLOBAL_READY → 四个并行 UPDATE_AG；ITER_END 取最晚结束。

- 保留 r4 的全部 F/B、PP 依赖，RS 不等待其他 stage 的 B，因此可与其他 stage 的 B 重叠。
- 本地 B 内可能重叠的 RS 暂不拆出；RS_PREP 是本地完成到全局同步开始的聚合窗口，包括软件准备，不是纯 RS service time。
- GPU 证据为 OPT1 → AG1 → OPT2 → AG2，合并为 UPDATE_AG，不确定两次优化器调用的参数身份。
- UPDATE_AG 从同步结束到最后 AG 结束，包含梯度范数/裁剪等准备，不能当成纯 Adam 或网络耗时。
- 全局同步成本取最晚同步结束减最晚开始，依赖跨主机时钟可比假设；提前到达的等待由 DAG 产生，不另拟合。
- 采样 source60 rank 0/8/16/24；中间 stage 参数取均值并按角色共享。不是全 32 rank GPU 成本采样。
- 15 参数：6 F/B + 2 PP + 3 RS_PREP + 3 UPDATE_AG + 1 GLOBAL_READY。

## 结果与验证

实现检查 PASS；精度状态 PARTIAL。建模窗口 12261.291361749172 ms，参考零点至最后 UPDATE_AG，不含启动及更新后收尾同步。`predicted_iter_ms=null`；不可计算完整迭代误差或 MFU。

独立就绪节点消除调度器核对 1899 个节点的开始/结束时间通过。更新尾段取 max 而非 sum；PP1/2/3 的 RS 可以与其他 stage 的 B 重叠。HTTP 200、浏览器加载无页面异常，12 个图形色块、15 行参数表。

## 产物及复现

工作区：`/home/zjb/Desktop/worktrees/mfu-w37-v610`。

- `workflow/data_foundation/pp32_simple_tail_r5.py`：构建、读取四份冻结 source60 trace、基础断言。
- `workflow/data_foundation/publish_pp32_tail_r5.py`：独立校验及静态报告。
- `results/data-foundation/pp32-simple-tail-r5/`：协议、原始路径与 SHA256 清单、尾段证据、DAG、回放、检查。
- `results/data-foundation/2111-blocks-ui-r1/df-v001/pp32-simple-tail-r5/`：HTML、独立校验及发布哈希。

构建使用新目录，不覆盖已有结果：

```bash
cd /home/zjb/Desktop/worktrees/mfu-w37-v610
.venv/bin/python -B workflow/data_foundation/pp32_simple_tail_r5.py --out results/data-foundation/pp32-simple-tail-r5-reproduce
```

发布脚本入口固定为正式 r5 产物；发布目录已存在时拒绝覆盖。

网页：http://192.168.8.16:43312/df-v001/pp32-simple-tail-r5/

下一步：先补齐完整迭代窗口的启动/结束边界，再使用 source60 校准与已有开发验证70评价；不将当前窗口直接当完整迭代。暂不增加 OPT1/2、DP/EDP 的身份参数。

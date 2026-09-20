# 单元0 MoE真正反向连接 r6

2026-09-20，PARTIAL_MOE_CONDITIONAL_REPLAY_CONNECTED。

范围为32卡iter60 PP1 B0，rank8–15的checkpoint执行单元0。仅真正反向，不包含重计算，也不是四层完整B。

## 本轮完成

每卡核对两段`_GroupedLinearBackward`，即专家分组线性反向CPU调用；设备经runtime/driver correlation归属。每卡89～105个设备事件归并为8个同流功能块：可见反向Dispatch、两个GroupedLinear反向包络、三段本地autograd处理、两个流上的可见反向Combine尾部。CPU身份明确，不把包络中所有耗时叫纯GEMM。

将这64个MoE块连接到r5的本地梯度连接段和Attention/CP图。原r5的Combine实测完成锚点被改成可见Combine节点的派生完成点。各流仍保留源trace就绪残余，因此只是去掉一个显式时间戳输入，不意味着不再依赖实测。

初始同流版本不足以传递专家成本；最终候选增加两种本地跨流边：可见Dispatch→计算流首块，以及计算流Combine准备尾部→通信流Combine可见尾部。按默认起点重新扣算就绪残余，避免同时叠加原残余与新前驱耗时。这些边标为conditional，不声称恢复了event句柄关系。

同样本MoE及下游原边界回放PASS；16次独立GroupedLinear成本+1ms测试均使本卡Combine派生完成点+1ms。

## 尚未闭合的关键项

反向Dispatch可见kernel主要是permute；Combine可见kernel主要是unpermute及填充。它们不能代表完整EP8传输service。其他rank的专家成本尚未通过EP8传输边影响本卡Combine，不能把八卡覆盖说成EP8同步完备。

本版开放专家GroupedLinear有效包络替换；可见EP尾部与其他本地处理保留trace参数，不开放一个伪造的纯EP网络成本。CP pair沿用r4候选假设。

没有使用iter70或256时序，没有新增预测精度结论，没有扩大到四层、提交或推送。原v610及224版本不改。

## 当前版本与复现

工作区`/home/zjb/Desktop/worktrees/mfu-w37-v610`。

```bash
python workflow/data_foundation/build_b_moe_backward_r6.py --out /tmp/b-moe-backward-r6-new
```

输出目录必须不存在。当前结果为`results/data-foundation/b-moe-backward-r6-local-handoff/`：evidence、graph、replay、sensitivity、summary、manifest JSON，包含八个原始trace及代码/输入/输出哈希。`b-moe-backward-r6/`是本轮早期仅同流的中间结果，不作为当前候选。

唯一下一步：核对反向EP8完整组的专家完成与Combine可见入口/结束时序，形成显式组间耦合候选并做单卡成本扰动测试。若只能得到有效完成残余，就按有效残余接入，不把它命名为纯带宽service。

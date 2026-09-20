# B 梯度处理连接段 r5

2026-09-20。状态PARTIAL_TRACE_ANCHORED_EP8_TO_ATTENTION。

范围：32卡iter60 PP1 B0，rank8–15，仅checkpoint执行单元0。重新核对八个登记trace的SHA256，通过runtime与driver correlation定位FusedDispatchBackward（反向Combine）可见设备尾部及Attention入口，不使用CPU函数结束时刻代替GPU完成点。

## 已完成

反向Combine可见设备结束到Attention第一设备事件之间，各卡相隔10.881615～11.406371ms。每卡窗口涉及46个设备事件；属于该checkpoint的设备活动裁剪到窗口后求时间并集，为9.601504～10.031933ms左右；未覆盖部分1.2404～1.6310ms。精确值见bridges.json，不对重叠事件简单求和。

rank8单元0：连接段11.406371ms，活动并集9.7754ms，残余1.6310ms。可见GEMM、LayerNorm反向、Add、Cast与Copy，说明这段不能省略，也不能全部算通信或等待。数学上的具体投影身份尚未逐算子确认。

已将连接段加入四个CP2双卡图，统一源trace时间原点。八卡88个原Attention/CP功能块结束点回放全部通过，误差<1e-5ms。入口使用各卡实测反向Combine尾部完成点，没有加全EP8到齐屏障。

## 明确限制

- 这不是完整EP8反向模型：MoE通信/专家反向尚未自行预测，Combine完成仍是trace锚点。
- 连接段按活动并集及未覆盖区间保存为trace_reserved；裁剪片段不是独立纯计算成本，不开放其任意替换。
- 未覆盖区间不是已证明GPU空闲或通信等待。
- CP2双卡耦合及部分跨流边继承r4条件假设。
- 本轮没有读取iter70/256，没有改善或重新声明独立预测精度，没有更新旧HTML。

## 复现

工作区`/home/zjb/Desktop/worktrees/mfu-w37-v610`：

```bash
python workflow/data_foundation/build_b_gradient_bridge_r5.py --out /tmp/b-gradient-bridge-r5-new
```

输出必须不存在。已执行目录`results/data-foundation/b-gradient-bridge-r5/`包含bridges.json（逐事件与活动并集）、graph.json、replay.json、summary.json、manifest.json（数据/脚本/输入结果与输出SHA256）。历史r1–r4及原预测保留。

唯一下一步：对本执行单元的反向Dispatch→专家梯度计算→反向Combine建立成本与依赖子图，用它替换当前实测Combine锚点，再检查完整八卡的单元回放。不要提前扩展四层或256卡。

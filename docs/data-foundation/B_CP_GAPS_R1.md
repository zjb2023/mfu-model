# B内CP与设备覆盖间隔核对 r1

范围：224卡iter60 PP5/6，256卡iter60 PP1/14，各stage lane0代表rank，14个B。读取4份原trace并核验SHA。模型、冻结HTML和参数不变。

## CP识别

依据Attention框架调用所关联kernel的`Process Group Description=CONTEXT_PARALLEL_GROUP`及`Group size=2`，不是仅凭SendRecv名字猜测。每B的4层重计算16次CP2，真反向20次CP2，14个B全部通过数量检查。其余拆为FlashAttention计算、本地拷贝/重排、其他。各类并集可能重叠，不能求和作为壁钟贡献。

## 256卡B1：PP14减PP1

| 分项 | 时间变化ms |
|---|---:|
| 重计算CP2通信并集 | −0.240 |
| 真反向CP2通信并集 | +2.923 |
| 重计算FlashAttention并集 | +1.164 |
| 真反向FlashAttention并集 | +1.766 |
| B整体包络 | −27.710 |

此样本B缩短不能解释为CP通信或FlashAttention普遍加速。CP kernel仍可能包含等待，不是纯传输service。

## 间隔进一步核对

之前“未归属本B设备事件覆盖”PP1为155.442ms，PP14为123.492ms。扫描同device所有stream后，其中分别约74.71和75.09ms实际上有其他设备事件覆盖。因此此前这些间隔不能全叫GPU空闲。

排除其他设备事件后，无记录设备活动的间隔约80.73→48.40ms，减少约32.33ms。这与B缩短方向一致，是当前更强的局部线索，但无记录事件不等于硬件真正空闲（例如不可见传输）。

PP1 B1最大的三个归属间隔约14.02、11.13、8.24ms，均与CPU的musaDeviceSynchronize调用重叠；PP14最大的三个约7.67、7.62、7.60ms，与musaStreamSynchronize重叠。这是时间重叠证据，不证明同步API本身制造了全部间隔，也未核对具体跨rank就绪关系。两侧最大的间隔不一定属于相同模块，不能直接一一替换。

## 224卡对照

PP5→6三个B的CP2通信累计类别差分别约−13.159、−6.614、−10.070ms（前向重计算与反向类别差合计，非壁钟贡献）；对应FlashAttention变化很小。说明224的重复下降边确有CP相关耗时减少，但256端点对照不能套用同一结论。

## 下一步

优先对256 B1的长同步窗口关联CPU父算子、GPU前后事件及stream，确认与Dispatch/Combine或其他模块的关系；再决定是否需要同EP组其他rank。不要将局部32.33ms乘stage数作为6.16%误差归因。

结果：`results/data-foundation/b-cp-gaps-r1/`。逐gap保留绝对时间、其他设备覆盖时长和重叠runtime名称；逐CP类别保留事件索引。输入和代码SHA见manifest。

```bash
python -B workflow/data_foundation/audit_b_cp_gaps_r1.py --out /tmp/b-cp-gaps-repeat
```

输出目录须不存在。状态：局部统计验证通过，机制归因仍PARTIAL；未提交、推送。

# B Attention局部子图 r3

2026-09-20。32卡iter60 PP1 B0、rank8–15、每卡4个checkpoint执行单元。状态：PARTIAL_CONDITIONAL_LOCAL_DAG。

## 已执行与验证

八个登记trace重新核对SHA256，仅分析指定B0的Attention反向CPU父函数，通过runtime/driver correlation同时发现设备事件，不重扫全量case，不使用iter70。

32个单元各11个块：五个CP块、一个Flash Attention反向计算包络，以及五个本地处理/重排块。同流连续同类设备事件归并，保留不同stream间重叠；没有把重排与通信重叠部分串行重复计费。

CP1结束到Flash反向开始相隔5.891812～6.365769ms，关联的index_select活动覆盖至少96.7403%。rank8单元0间隔6.067007ms、index_select覆盖6.058087ms。不能把这段全叫等待。计算包络包括该Flash CPU调用关联的preprocess/fill/主反向kernel/cast，不是单一纯算术kernel。

每单元有12次StreamWaitEvent调用，证据保留事件编号和args；这些记录未给出可用的event句柄/源目标stream配对，因此不能恢复确定的Record→Wait连接。

## DAG与参数合同

- 同stream顺序边：same_stream。
- CP1→Flash、Flash→CP2、最后一个输入重排块→Flash：conditional_structure_not_event_handle_verified。为结构及设备顺序支持的局部候选，不是逐事件同步证明。
- 每块参数分为cost_ms与trace_ready_ms。后者为相对前驱最大结束时刻的非负就绪残余，显式保留，不叫纯等待。
- 五个CP和Flash块允许effective成本替换；其他本地块及就绪残余锁定为trace参数。CP设备包络可能含等待，不接受纯service直接替换的等价声明。
- 尚未添加CP2双rank到齐规则或EP8全局边。不能将当前独立局部子图拼起来称为同步完备的B模型。

求解start=max(dep.end)+trace_ready_ms，end=start+cost_ms。32个子图全部节点边界同样本回放通过（误差<1e-5ms）；每个Flash成本+1ms均传播到CP2，敏感性检查通过。这只是条件模型自洽验证，不是独立预测精度。rank8单元0选定Attention设备包络为78.339432ms，不是整个B时长。

## 可复现产物

工作区 `/home/zjb/Desktop/worktrees/mfu-w37-v610`。

```bash
python workflow/data_foundation/build_b_attention_replay_r3.py --out /tmp/b-attention-replay-r3-new
```

输出目录必须不存在。已执行产物目录：`results/data-foundation/b-attention-replay-r3/`，包括graphs.json（11块与边证据、默认与+1回放）、audit.json（重排覆盖与wait证据）、summary.json、manifest.json（输入与输出哈希）。原r1/r2、F冻结、v610/224及历史预测结果不变。本轮不更新旧HTML，以免把未完成跨卡同步的候选冒充原图已验证模型。

唯一下一步：核对四个CP2 pair内对应通信的本地准备与开始时刻，定义明确的双rank通信就绪候选，再把此Attention子图与MoE反向连接；仍先同样本回放，不直接外推256卡。

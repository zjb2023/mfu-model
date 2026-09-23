# 未进入计算的分配量：进一步证据

2026-09-23，仅读取现有记录；无重新训练、采集、模型修改或Git提交推送。

## 新确认

1. 236B参考代码的顺序是top6路由 → 容量筛选 → 被筛掉的概率置0 → DeepEP把零概率对应索引置−1、不发送。零概率屏蔽主要是在执行前面的淘汰结果，不是已证明存在的第二次独立损失来源。不能将两步丢弃数直接相加。
2. 256 iter60 MB0，rank16/L3与rank224/L58的实际trace都匹配该序列：top6 → 专家轴top615 → logical_and → mul → top6 → eq(0) → masked_fill(−1)。记录了事件索引和参数，但没有张量概率值。运行时源码提交同一性仍未证明。
3. 同一MB0，每个EP8组拥有160个专家：组A和组B在L3都只有59个专家实际收到输入，在L58都只有29个。这里是筛选后参与数量，不能直接当筛选前路由覆盖率。
4. 两组保留条目中，来自“每发送rank、每专家615条上限”的条目占比：L3为234315/304032=77.07%，L58为164820/179904=91.62%。这是容量上限在实际计数中的明显痕迹，不是容量丢弃占比。达到上限不单独证明筛选前严格超限。

这些证据支持：后层实际保留工作落在更少的专家上，而且更大比例的保留工作触及容量上限。容量筛选是有直接操作证据的解释路径，不应把“本来就零概率”说成另一个已经观察到的同等原因。

## 仍未确认

没有得到容量筛选前的逐专家计数/概率。因此仍不能精确量化差额中多少由容量超限引起，也不能排除选中的概率在筛选前已经数值为0。不能由后侧计数倒推出前侧分布，不能据此声称更深层必然路由更集中或解释全部1F1B误差。

检查了worker33008的full_moments、tf_logs、wandb、checkpoints目录，未找到可用张量/统计文件（仅此采集worker范围，不声称遍历了所有归档）。已有DeepEP日志记录筛选后计数。检查rank0运行日志发现moe_per_layer_logging=False；未找到可用于这一层/MB的筛选前概率记录。训练整体指标不能替代逐层路由证据。

## 可复现入口

工程根`/home/zjb/Desktop/worktrees/mfu-16to256`：

```bash
/usr/bin/python3 -S workflow/data_foundation/investigate_route_filter_origin_r1.py
```

输出目录`results/data-foundation/route-filter-origin-r1/`，存在时拒绝覆盖。report.json保存分组统计、两个trace精确事件序列和目录检查结果；manifest.json保存代码/数据SHA256。

参考源码位于`/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/moe/`，包括router.py、moe_utils.py、token_dispatcher.py。

下一步如要定量分离，需获得同一层/MB的容量前专家计数、容量前零概率条目数及容量筛选后的有效条目数。未授权新增训练采集前，保持该因果归因缺口，不把可能性写成确定比例。

# 小消息统计AG的迭代归属：r6修正

2026-09-22。新证据来自236B的MUSA补丁执行路径，而非仅看Megatron原training.py。

## 结论

逻辑工作归属：小消息计时AG汇总刚完成训练步的计时信息，属于该步的统计收尾。采集窗口归属：补丁先调用torch_profiler.step()切换窗口，再做training_log，因此该统计收尾落在后一个Profiler窗口的前缀；它又会阻塞下一轮训练进入F。这两种归属同时成立，并不矛盾。

它不是下一轮token路由或模型参数准备。旧称“Profiler启动段”仅为测量位置名称，不能视为全部属于下一轮训练逻辑。

## 代码证据

路径：/home/zjb/Desktop/fabric-data-analysis/0722/236B/megatron-lm-musa-patch/musa_patch/training.py

- 约803–815：执行train_step，得到该轮loss、梯度等结果。
- 849–850：调用torch_profiler.step()，切换采集窗口。
- 851：iteration加一，更新样本数等统计。
- 905–910：调用training_log，参数包括刚完成训练步的loss_dict、grad_norm等。
- training_log约526：interval-time.elapsed(barrier=True)，停止并重启计时，产生两次barrier/device synchronize。
- 597–605：report_memory，然后timers.log；报告开关DISABLE_MEMORY_REPORT未启用时可以每轮继续报告，不是原training.py默认只打印一次的逻辑。
- 同目录profiling.py约67：目录iteration编号取prof.step_num，不是直接取训练iteration变量。
- musa_patch/__init__.py约25导入training；training.py末尾替换megatron.training.training.train与training_log。

因此应使用补丁解释采集行为，而非主文件循环开头的另一条prof.step()路径。代码仍是参考快照，未证明采集容器逐文件完全同版本；但这条路径同时解释Profiler前缀、显存每轮报告和统计顺序，证据显著强于此前的“归属未知”。

## 与实测的对应

256卡目录iteration_60保存ProfilerStep#59。rank0的小AG在该窗口首F之前；前缀先出现按层辅助损失归约、两次barrier、主机区间、计时AG，再进入should_run_forward_backward、梯度清零、forward_step。日志相邻位置出现iteration59与after59显存报告，和“上一轮统计收尾→下一轮训练”一致。

不能只靠目录60与Profiler#59作归属判断，必须以补丁顺序和事件链为依据。16卡采集的同类AG位于Profiler尾部，不强制套用相同的窗口偏移或编号映射。

## 正确的阶段边界

逻辑顺序：训练步k的F/B与优化器更新完成 → 切Profiler窗口 → 训练步k的辅助损失/计时/日志统计 → 下一轮调度入口与梯度清零 → 训练步k+1的F。

统计AG结束不是整个收尾完成：后面仍有统计筛选、读取及可能日志处理。下一轮准备应从实际train_step/should_run_forward_backward、梯度清零等证据确定，不能直接把AG结束当新轮起点。完整train_step边界没有直接注释时，应保留代理边界名称。

建模时将本轮窗口前缀拆为“前一训练步统计收尾（当前Profiler窗口内）”与“本训练步准备”。不能删除统计成本，也不能从前缀移动到尾部后再保留原前缀重复计费。

若改成严格训练逻辑迭代口径，需要相邻连续训练步的起止事件；当前每5轮采样的20份trace不能机械拼成20份精确重划分Training Step。现有Profiler均值、整轮MFU与正式预测暂不改。

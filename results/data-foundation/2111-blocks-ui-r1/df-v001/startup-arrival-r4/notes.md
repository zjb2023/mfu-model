# 全局计时AG：是否全部在等rank到齐？r4

## 数值结论（256卡iter60）

完整提取256个rank的同一Profiler窗口内第一条全局24-FP32 AllGather，以及之前两条全局barrier。不是全量多iter扫描，没有实验、训练或预测参数变更。

- rank0 AG驻留：1068.895 ms。
- 依据前一次全局barrier建立的无跨机时钟偏移假设的到达时间差界：最后一个rank的AG kernel开始，比rank0晚 **1059.162～1073.029 ms**。
- 所有rank的AG kernel都开始后，rank0仍驻留 **0.000～9.733 ms**。
- 因此，等待参与者进入AG kernel可解释rank0驻留的约 **99.09%～100.00%**，剩余不能继续命名为“尚有rank未进入kernel”。

这不是将残余全部认定为纯传输：kernel已开始不代表内部协议/缓冲区/数据准备全部完成；残余仍可能包括协议等待、资源争用与传输。也不能把驻留100%归为到齐等待。

## 最晚的本地到达样本

表中每个时间差来自同一rank；不直接比较跨机原始ts。按barrier完成后到AG开始的本地间隔排序，不声称此排序就是严格全局到达顺序。

|rank|worker|barrier后到AG开始 ms|AG驻留 ms|CPU AG发起到GPU开始 ms|
|---:|---|---:|---:|---:|
|83|worker34009|1154.609|2.877|0.135|
|86|worker34009|1153.398|4.086|0.126|
|84|worker34009|1153.355|4.129|0.150|
|82|worker34009|1153.268|4.216|0.136|
|81|worker34009|1152.958|4.528|0.638|
|80|worker34009|1152.076|5.407|0.160|
|87|worker34009|1149.951|7.532|0.150|
|85|worker34009|1149.148|8.335|0.162|
|61|worker33090|1117.466|39.906|0.131|
|59|worker33090|1117.259|40.138|0.159|
|60|worker33090|1116.733|40.635|0.114|
|56|worker33090|1115.372|41.965|0.140|

## 为什么不需要把跨机绝对时间当作已校准？

对rank i，B_i/E_i为同一次前置barrier的GPU开始/结束，A_i为AG GPU开始，q_i=A_i−E_i。假设全局barrier满足“任何rank完成前，所有rank已进入”，则 B_i≤E_0 且 B_0≤E_i。

因此，最后AG开始相对rank0开始的时间差满足：

```
下界 = max(0, max_i(q_i) − q_0 − barrier_duration_0)
上界 = max(0, max_i(A_i−B_i) − q_0)
```

公式只使用各rank本地时间差，不依赖host之间的固定时钟偏移。前提：同一barrier实例、标准barrier语义、本地设备时间差可信。匹配依据为同一Profiler标签、default_pg=256、第一条24→6144全局AG及之前两次barrier；trace未提供独立后端collective序号，所以实例匹配仍有显式前提。

这给出的是“等待所有rank进入AG kernel”的时间界，不是通信算法内部每个字节何时就绪的精确分解。

## 为什么有的rank晚到：新增候选证据

实际训练日志在iteration59行后出现每rank `(after 59 iterations) memory ... max gmi memory usage`。236B参考 `training/utils.py:264` 的report_memory调用musa_patch.mem_utils.get_max_gpu0_to_7_mem_usage；补丁 `mem_utils.py:206` 同步启动外部mthreads-gmi，必要时再调用zjlab-gmi。training_log参考顺序是memory report后timers.log。

这使“显存查询/主机日志工作造成到达不齐”成为具体候选，比笼统参数量增大更贴近现有证据。没有子进程执行时间记录，且参考training.py的report_memory_flag逻辑与实际多轮日志不完全一致，不能把晚到全部归因于gmi，也不能宣称参考代码就是采集时精确版本。本任务没有执行gmi或修改日志配置。

## 建模含义与验证

保留“统计/主机准备→全局AG汇合”依赖，将rank到达差与到齐后有效成本分开。不能将rank0约1秒驻留当作96字节纯网络服务，也不能直接乘到其他通信节点。目标数据仅诊断，不更新32→256正式预测。

提取脚本audit_startup_arrival_r4.py；本文件由summarize_startup_arrival_r4.py生成。原始输入256条绝对路径、大小和SHA256见startup-arrival-r4/manifest.json；事件索引、CPU发起、GPU时间及进程组见report.json。检查256rank唯一覆盖、同一Profiler标签、消息量、两barrier和CPU/GPU关联通过。数值提取PASS；具体晚到原因及协议分离PARTIAL。

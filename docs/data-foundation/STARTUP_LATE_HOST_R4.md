# r4补充：晚到主机的停顿位置

基于startup-arrival-r4/report.json中rank83的已哈希原始trace，再检查前置第二次全局barrier结束到AG CPU发起之间的CPU事件。

- worker34009的rank80～87集中晚到，barrier结束到AG GPU开始1149.148～1154.609ms；自身AG仅2.877～8.335ms。
- rank83在barrier后0.138～0.698ms有标量读取/拷贝；接着直到1154.277ms才出现计时表aten::zeros。对应事件索引33266到33267之间约1153.579ms没有已记录的CPU算子。
- rank83的AG CPU调用到GPU kernel开始仅0.135ms，故主延迟在CPU发起AG之前，不是AG已经发起后在GPU队列里排了1秒。
- 同主机8个rank均有类似晚到，支持主机级准备/查询/调度因素；不能仅凭这一点断言物理主机故障。

实际日志路径：`/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40/worker34009/2026-07-31_1340/tp1_pp16_dp_mbs2_numbs_gbs64_gpus0_mtp1_forcelbfalse_pertensorfalse_NO_LOSS_REDUCE.RANK10.10.124.34.9.log`，包含8个rank的after59显存报告。236B补丁同步调用gmi是可核验的候选机制，尚缺子进程/OS调度耗时证据，不将这1153.579ms直接标成gmi执行时间。

全局barrier时序约束给出rank0最后参与者到达前至少1059.162ms，占1068.895ms驻留的99.089%；所有AG kernel开始后rank0剩余不超过9.733ms。这是“晚到限制了集体完成”的强证据，不是证明前1059ms里设备完全没有做通信。其他rank可以边等待边推进协议；剩余也不能全部叫纯传输。

结论仅针对256卡iter60这一次AG，不自动泛化到iter40/80、32卡或未来规模；正式模型不改，未提交推送。

# Profiler启动窗口：工作内容、依赖与成本口径

日期：2026-09-22。范围：当前32→256外推分析中的Profiler启动窗口，补充224卡与14个16卡case的比较。本文整理既有分析，不改变模型或成本参数。

更新：已定位236B MUSA补丁中train_step之后、training_log之前的torch_profiler.step()。它解释了为何上一训练步的统计收尾出现在下一Profiler窗口前缀；小计时AG的逻辑归属可据此澄清。详见 `STARTUP_ITERATION_OWNERSHIP_R6.md`。下文关于参考主文件Profiler位置/显存报告频率的缺口由该补丁得到解释，但采集容器精确版本仍未逐文件确认。

## 1. 这个阶段是什么

测量边界是 **rank0的ProfilerStep开始 → PP0首个forward_step所关联的GPU活动开始**。它是采集窗口的前缀，不是进程启动、加载权重或初始化通信组，也不能不加区分地称为训练算法必需的启动成本。

这个前缀包含两类工作：

- 训练状态的统计、计时、日志和跨rank协调；其中部分可能属于逻辑上一轮训练的统计收尾。
- 下一次前向执行前的梯度缓冲清零、训练调度与batch准备。

实际采集版本和参考236B代码在Profiler位置、显存报告频率上存在未闭合差异；目录迭代号、Profiler标签和日志迭代号均应分别保存，不强制视为同一编号。

## 2. 按顺序执行的工作

### A. 按层统计归约与本地整理

trace可见5次小消息AllReduce，组顺序为PP、TP-with-CP、DP-with-CP、PP、DP-with-CP，随后有缩放、求和、累积及统计向量清零。32/224/256卡的向量长度分别对应12/52/60层。

参考moe_utils.py的reduce_aux_losses_tracker_across_ranks、track_moe_metrics会按层汇总MoE辅助损失，和这条执行链强匹配。这是“汇总已经算出的统计量”，不是重新做router、token dispatch或专家FC。

CPU调用可以较快返回，GPU归约仍继续，后续barrier窗口可能覆盖这些前序工作；不能把CPU发起时间当完整归约成本。

### B. 停止当前计时：第一次全局barrier与设备同步

参考training_log调用interval-time.elapsed(barrier=True)。Timer.stop先调用全局barrier，再调用本地device synchronize，然后读取时间。作用是避免某个rank尚未推进到统计位置、或本地GPU仍未完成时，就结束计时。

barrier等待的是参与者到达，不搬运模型权重或token。设备同步用于收拢本地异步工作。它们的等待区间可能重叠，不能将CPU wait与GPU kernel驻留简单相加。

### C. 重新开始计时：第二次全局barrier与设备同步

如果计时器原来正在运行，Timer.elapsed在读取/重置后调用start(barrier=True)，再次全局barrier和设备同步，再记录新起点。因此两次barrier可以来自“停止计时”和“重启计时”，不是两轮不同的训练通信。

这是参考代码与trace模式的强匹配解释；没有采集时Python栈，不标记为逐调用确证。

### D. 各rank独立进行主机侧统计/日志处理

barrier完成后，各rank继续执行各自的主机工作，并不保持锁步。参考代码中有损失标量读取、日志格式化、输出以及显存报告。显存报告补丁会同步启动mthreads-gmi，匹配失败时再启动zjlab-gmi。

实际日志有相应显存报告，但trace未记录这些子进程和完整Python/OS执行，所以不能将记录空白直接认定为gmi耗时、CPU纯计算、睡眠或IO。当前可以确认的是该时间落在AG发起之前的主机侧区间。

224卡rank0的20轮该窗口均值1142.227ms，范围1077.569～1326.039ms；其中可见CPU算子活动并集平均仅0.569ms，设备活动并集约0.011ms。其余属于当前采集未覆盖的工作/等待，不能说“CPU没有工作”。

### E. 建表并全局汇总计时信息

各rank创建[world_size,24]的FP32计时表，以24个FP32（96字节/rank）作为输入执行全局AllGather，再逐列筛选有效计时结果。形状、24次筛选和timers.py路径强匹配，区别于token路由和参数AllGather。

各rank到达这个调用的时间可能再次分散：前面的barrier只对齐当时的位置，不能消除之后主机工作造成的差异。早到rank可能较早启动AG kernel并等待晚到者。

256卡iter60的全部256rank已核对：worker34009上rank80～87在前置barrier完成后约1.15秒才开始AG，自身AG只有2.877～8.335ms。根据同一barrier标准语义及各rank本地时间差，rank0的1068.895ms驻留期间至少1059.162ms仍有rank未开始AG；所有kernel开始后剩余不超过9.733ms。

这是到达不齐限制完成的证据，不证明设备前1059ms完全空等，也不证明残余全为纯网络服务。该比例仅适用于iter60，不自动推广到20轮平均。

### F. 读取统计结果、进入训练调度

可见select、gt、index、nonzero等操作和小量设备到主机拷贝，随后进入should_run_forward_backward。CPU读取结果可能阻塞到AG完成，所以长CPU nonzero和长GPU AG不是两个连续收费项。

### G. 清零梯度缓冲，并与首F的CPU准备交叠

新训练轮开始前，框架清零普通与专家梯度缓冲。当前PP0两个FP32缓冲分别为1,191,895,040和471,859,200元素，合计6,655,016,960字节。32/256卡PP0尺寸相同，iter60 GPU清零合计约4.80/4.85ms。这项与本rank缓冲大小有关，而不是直接随模型总层数或总rank数增加。

清零在GPU执行时，CPU可以进入forward_step、获取batch并发起准备操作。DataLoader在CPU首F包络内，不能把它与GPU清零机械串行相加。PP0的recv_forward注释也不能被当作从不存在的前置PP stage接收一次真实消息。

当首F关联GPU活动开始，本文的启动窗口结束，后续Attention、CP、MoE、EP进入F内部图。

## 3. 规模比较与不能混用的时间

rank0的20轮（目录iter5,10,…,100）均值：

|规模|整个启动 ms|其中统计AG ms|
|---|---:|---:|
|32|117.157|15.367|
|224|1549.707|85.616|
|256|1206.196|1073.545|

224卡20轮的不重叠窗口分解：统计归约发起1.350ms；两次barrier及前序工作排空308.409ms；主机处理区间1142.227ms；计时表/AG发起0.351ms；AG驻留85.616ms；统计处理/调度6.829ms；清零发起到首F GPU 4.924ms。合计1549.707ms。窗口是时间分段，不是各单一算子的纯服务时间。

单独取两个barrier的GPU kernel驻留，32卡均值1.669/0.815ms，224卡77.074/40.379ms，256卡7.330/7.124ms。224两个GPU驻留合计117.453ms不等于上述308.409ms同步窗口；后者还覆盖前序排队、工作完成及调用间隙。

16卡14case在当前可完整识别两次barrier的样本中，第一次各case均值约1.092～2.531ms，第二次0.125～0.321ms，每case有效样本10～16轮。这些统计通常处于迭代尾部，且无同名forward_step，因此不能声称已经得到了与32/224/256相同定义的启动均值。

结论：较大规模的协调开销更值得关注，但并非随rank数单调增长。224与256中rank0的主要耗时位置不同，不代表前者一定是全局最晚rank；224尚无相同的全rank到达诊断。不能仅用rank数、总参数量或60%带宽拟合整个窗口。

## 4. 接入DAG时的成本口径

保留顺序：按层统计/归约 → 计时停止barrier → 计时重启barrier → 每rank主机准备 → 全局计时AG → 结果处理 → 梯度清零与CPU首F准备 → 首F GPU。

- 统计归约成本依赖统计项、向量长度与通信组；不是专家计算成本。
- barrier/AG分别需要rank就绪依赖与通信服务项。已经包含等待的trace驻留不能再叠加同一份“到齐等待”。
- 主机准备先保留独立有效参数；来源未查清前不按总参数量或rank数缩放。
- 清零依赖本地缓冲字节量与显存有效带宽，并保留与CPU准备的重叠。
- 此依赖链是待接入的描述，不代表已经修改了预测模型。目标数据只作诊断。

## 5. 已闭合与尚未闭合

已闭合：窗口边界、主要设备/通信活动、统计代码的形状与顺序对应、224主要时间位置、256 iter60到达不齐的时间界、跨规模跨迭代均值。

未闭合：实际采集Python调用身份、主机长区间内gmi/IO/调度的比例、224全rank谁晚到、20轮逐轮等待因果分离。应明确交付“工作链已说明，主机慢因未完全查明”，不能将整段标记为因果分析全部完成。

## 6. 证据入口

- STARTUP_WORK_R3.md：工作清单、236B对应位置与CPU/GPU重叠。
- STARTUP_ARRIVAL_R4.md、STARTUP_LATE_HOST_R4.md：256 iter60全rank到达诊断与晚到主机。
- STARTUP_COHORT_R5.md：20轮均值、14case覆盖与缺失边界。
- results/data-foundation/startup-cohort-r5/report.json：20轮大规模rank0原始路径、SHA256、AG/barrier事件索引与时长。
- results/data-foundation/startup-cohort-r5-host-supplement/report.json：16卡其他主机的barrier初步清单；不替代16-verified的AG结果。
- results/data-foundation/startup-work-r3-verified/report.json、startup-arrival-r4/report.json：分段、事件、调用与rank级证据。

参考实现：/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/timers.py；training/training.py；training/utils.py；core/transformer/moe/moe_utils.py；core/distributed/distributed_data_parallel.py；core/distributed/param_and_grad_buffer.py。

显存查询参考：/home/zjb/Desktop/fabric-data-analysis/0722/236B/megatron-lm-musa-patch/musa_patch/mem_utils.py。

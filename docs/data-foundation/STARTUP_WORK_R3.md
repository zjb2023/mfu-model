# Profiler启动窗口工作清单与建模边界 · r3

## 结论

启动窗口不是一个纯准备算子：它混合了按层MoE统计归约、全局计时同步、主机侧未记录工作、计时表AllGather和结果读取，以及新训练轮的梯度清零与首F准备。当前证据更支持小AG是计时统计汇总，而不是token路由分发。不能将全部启动时间按模型总参数量、层数或rank数统一放大。

范围：32/256卡、iter40/60/80、rank0共六份冻结trace；只读解析，没有修改预测、拟合或训练。下表是iter60不重叠时间分段，不代表每段内部只有一种工作。

|窗口内工作/边界|32卡 ms|256卡 ms|证据与解释|
|---|---:|---:|---|
|按层统计归约发起、本地处理→首barrier CPU入口|1.500|1.437|5次小AllReduce，向量长12/60；GPU归约延伸至下一行|
|两次全局barrier及前序工作排空|7.219|24.648|barrier、wait与设备同步确实存在；含先前异步归约，不能当作纯barrier网络服务|
|末barrier wait结束→计时表创建|83.612|88.130|CPU已记录活动并集仅0.561/0.504ms，GPU约0.011ms；主要主机侧工作未被当前trace标注|
|计时表创建→AG设备开始|0.290|0.317|zeros[32,24]/[256,24]、view、AG调用|
|小全局AG设备驻留|2.210|1068.895|输入24个FP32，即96B/rank；输出world×24；不是专家token payload|
|AG结束→大缓冲清零CPU发起|7.473|8.437|24轮select/gt/index/nonzero、CPU结果读取、should_run_forward_backward|
|大缓冲清零CPU发起→首F关联GPU开始|4.873|4.959|普通/专家两块梯度缓冲清零；CPU已进入首F并取batch，CPU/GPU重叠|
|总计|107.177|1196.823|与r2冻结边界一致|

## 1. 按层MoE统计：和层数有关，但不是专家路由计算

六样本均为5次AllReduce：PP → TP-with-CP → DP-with-CP → PP → DP-with-CP。32卡各次输入12个FP32，256卡60个FP32，恰好对应模型总层数12/60；PP组4→16，DP-with-CP组8→16，TP-with-CP组保持2。之后两次mul/sum/div和两次同长度zero_。

236B `core/transformer/moe/moe_utils.py:727` 的reduce_aux_losses_tracker_across_ranks和track_moe_metrics按层保存aux loss，以PP及额外归约组汇总，缩放后清零tracker，结构与trace吻合。32卡实际rank0日志确认aux_loss与z_loss开启、num_layers=12。因此“MoE辅助损失统计”是强匹配推断，但没有实际Python栈，不能逐条宣称已确定统计名称。

这不是启动时重新做router、dispatch、专家GEMM；统计可以来源于已计算的MoE结果。工作落在当前Profiler窗口，不证明逻辑上属于当前还是上一轮训练。

## 2. 两次barrier与计时AG：更像统计路径

236B `core/timers.py:168` Timer.elapsed(barrier=True)在计时器运行时先stop再start，两者各执行barrier与device synchronize，能解释成对的barrier；`training.py:1509` training_log调用该路径。这是代码与事件序列匹配，不是实际调用栈证明。

更强证据链：zeros[world,24] → AllGather输入[24]、输出[world×24] → 对world长列做24次gt/index/nonzero。这对应 `timers.py:246–310` 的计时表汇总与逐timer过滤。32卡实际日志timing_log_option=minmax、timing_log_level=0、log_interval=1；log_timers_to_tensorboard=False不代表禁用控制台timers.log。

256卡iter60第一个aten::nonzero CPU持续1069.001ms，覆盖AG驻留；CPU长算子和GPU长AG不是两个串行成本。后续筛选、设备→主机小拷贝是读统计结果，不是token重排。尚未分离AG内的到齐等待、通信协议/软件、资源争用和纯传输。

前期已核对目标rank0/1/8/9该AG驻留均约1.07s（TAIL_PEERS_R2.md）。这排除仅rank0统计时长很长的解释，但不足以找到全部256rank的最后到达者；跨worker时钟未经校准，不能直接比较绝对时间。

## 3. 80～90ms主机区间仍不能强行命名

iter60末barrier完成后到计时表创建之间，32卡83.612ms、256卡88.130ms；其中可见CPU仅约0.5ms。236B对应位置有日志格式化、输出和状态处理，因此日志IO、Python处理、调度/采集开销是候选解释。没有Python栈、OS调度或IO证据，不能认定为打印耗时、checkpoint、数据加载或rank同步。

未记录设备活动不等于GPU物理空闲；未记录CPU不等于CPU没工作。当前保留为独立“主机统计/调度残余”有效参数，不能误算成网络成本。

## 4. 与参数量真正相关的准备：梯度缓冲清零

两种规模PP0都出现两个FP32缓冲清零：1,191,895,040元素和471,859,200元素，共6,655,016,960字节。通过External id关联GPU fill：

|清零块|32卡 GPU ms|256卡 GPU ms|
|---|---:|---:|
|普通梯度缓冲|3.438|3.481|
|专家梯度缓冲|1.361|1.371|

`training.py:1201`、`distributed_data_parallel.py:615`、`param_and_grad_buffer.py:862`给出训练前清零普通与专家grad_buffer的实现。匹配尺寸与先前RS全缓冲相同，身份依据强；它不是加载模型权重，也不是优化器更新。

这里应使用本rank实际缓冲字节量和有效显存写带宽，而不是全模型参数总数。本例总模型变大主要来自中间PP扩展，PP0两块缓冲未变，所以这项基本不变。CPU进入F并取batch与GPU清零重叠，不能把完整清零时间再加在F成本上。

## 5. 数据加载、PP接收和首F边界

CPU首F开始后，DataLoader.__next__耗时0.146/0.164ms，随后可见batch形状[2,8192]的broadcast调用。它们在首F CPU包络内部、首F GPU之前，不在前面的80～90ms空档。首F CPU之前的recv_forward注释仅0.033/0.063ms；PP0无前置PP stage，不能据此收取一次真实PP接收通信。

六样本整个启动窗口内未见需要归为EP dispatch/combine或专家GEMM的证据；这不代表首F内部没有路由，路由属于后续层内图。

## 6. 最简成本接口与外推限制

- 层统计：统计向量长度（12/60）、统计项数、PP/CP/DP组与归约路径；不是用总参数量预测。
- 全局同步/计时：barrier次数、world/topology、各rank到达分布、日志频率；AG消息量只描述传输，不能解释全部驻留。
- 主机残余：单独保留trace有效值，未得到调用证据前不拟合rank线性系数。
- 清零：本rank普通/专家grad_buffer字节数、有效显存带宽；首F CPU准备允许重叠。
- 首F交界：批次准备与发起成本，沿依赖取完成时间，避免与F内部重复计费。

六样本仅来自两个同时改变层数、PP、DP、world和采集时间的配置，不能独立识别这几个变量的系数。正式预测仍冻结；不回填目标约1.1秒使总误差变小。

唯一优先后续：围绕这条计时AG取有代表性的跨PP rank，核验同一collective序号及本地到达/结束相对关系；若要给出因果等待量，还需实际采集代码/同步时钟或可建立的时钟约束。并列缺口是主机80～90ms的Python/IO证据，不启动全trace扫描。

## 复现与验收

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/audit_startup_work_r3.py --out results/data-foundation/startup-work-r3-reproduction
```

已验证产物：`results/data-foundation/startup-work-r3-verified/{report,manifest}.json`，包含六原始路径/SHA256、事件索引、CPU/GPU事件、进程组元数据、七段守恒、清零关联和参考代码哈希。初次目录startup-work-r3为空的失败运行，保留未清理；修正计时AG wait不应计入前两次barrier wait后，六样本全部通过。无Python stack标记，因此来源归因为强匹配而非栈级确证。边界/清单PASS；因果等待分离PARTIAL。未提交推送。

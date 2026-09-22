# 尾段peer与AG调用核验 r2

2026-09-22；只读目标256卡iter60的rank0、1、8、9四份trace，无全量扫描、未修改模型、未提交推送。

## 已确认映射

以trace顶层`distributedInfo.pg_config`为准：DP-with-CP组成员0–15；rank0所在EDP组1043真实全局成员为[0,8]，rank1对应[1,9]。事件局部`Process Group Ranks`的[0,1]不能用于推断机内邻卡。

rank0/1文件位于worker33087，rank8/9位于worker33075。EDP为跨worker，DP16也跨两个worker；这是采集主机目录证据，仍非GPU→NIC→交换机路由证明。源400Gb/s网卡迁移到目标、每GPU独享NIC和DP分层算法仍为60%候选假设。

## AG不是重复记录或两个半量bucket

每个样本rank的普通DP AG和EDP AG分别各两次：独立GPU事件、runtime correlation、c10d调用、record_param_comms以及不同`step_with_ready_grads`父调用。

两次DP AG各输入74,493,440个BF16元素、输出1,191,895,040；两次EDP AG各输入235,929,600、输出471,859,200。CPU Input Dims均显示单tensor，不能当两个半量bucket；需要保留两轮AG逻辑，不能去重。尚未确认二者buffer身份和必要性，不能宣称是冗余通信。

236B语义参考支持这种嵌套结构：optimizer.py约1195行ChainedOptimizer逐个执行子优化器，distrib_optimizer.py约2325行每次step_with_ready_grads可调用model_chunk.start_param_sync。参考代码解释可能路径，不是采集安装版本的因果证明。

## 设备驻留与服务成本分离

| rank | DP RS ms | EDP RS ms | DP AG两次 ms | EDP AG两次 ms |
|---|---:|---:|---|---|
|0|23.797|55.194|12.943 / 12.875|14.103 / 14.830|
|1|23.909|54.588|13.054 / 13.074|14.120 / 13.577|
|8|129.934|55.149|12.801 / 12.001|14.078 / 14.830|
|9|129.814|54.681|12.618 / 11.855|14.183 / 13.589|

同一DP组两端RS驻留相差约106ms，不应直接把各rank kernel时长拟合为bytes/bandwidth。它提示到齐、前驱/通道等待或进度差异；未做跨机时钟校准，不用绝对时间断言谁先到齐。EDP AG约13.6–14.8ms，而400Gb/s×60%的单lane服务候选15.729ms略长，也说明60%只是预设有效效率，不能宣称已经验证吻合。

若只讨论60%分层假设中的跨机服务，普通DP AG为4.966ms/调用，EDP AG为15.729ms/调用，两轮资源工作量需要分别保留。两组在不同stream；是否并发、是否争用NIC及与优化器交叠不能用简单四项求和替代DAG。

## 启动小AllGather

四rank的96字节/rank全局AllGather驻留分别1068.895、1077.819、1071.333、1069.966ms。不是仅rank0异常；仍未确定256rank中最后到齐者或运行时原因。

236B `megatron/core/timers.py:246–287`的timer汇总会构造[world_size, len(names)] FP32矩阵并执行all_gather_into_tensor；形状与24个FP32计时值的可能用途吻合。但trace没有Python调用栈，故只记为“疑似全局计时统计同步”，不认定为模型输入通信，不把约1秒转成带宽惩罚。

## 复现与状态

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/audit_tail_peers_r2.py --out results/data-foundation/tail-peers-r2-reproduction
```

有效输出`results/data-foundation/tail-peers-r2/`，逐事件父调用、全局pg_config、路径/哈希及设备时间均落盘。目录iteration_60中的Profiler标记为ProfilerStep#59，保留原始编号，不擅自改成#60。

PASS：四rank哈希/身份、全局EDP成员、不同worker、两次AG独立父调用及单tensor形状。PARTIAL：目标NIC/机内带宽、调度争用、全rank就绪、timer来源及完整尾段预测仍未闭合。

下一步把尾段画成普通DP/EDP两个分支＋两轮AG节点，保留源优化器本地处理；待目标NIC共享与机内带宽确认后，按60%候选接入服务参数。启动统计同步作为独立预留项，不与训练通信带宽混算。

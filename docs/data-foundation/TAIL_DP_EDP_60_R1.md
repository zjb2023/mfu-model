# 启动与DP/EDP尾段：元数据核验及60%带宽候选

2026-09-22；只读源32卡、目标256卡iter60 rank0两份原trace，核验旧SHA256。模型、网页未改，未提交推送。

## 事实已确认

源RS属于DATA_PARALLEL_GROUP_WITH_CP，组8；目标两个RS分别为该组16和EXPERT_DATA_PARALLEL_GROUP组2。源AG有2个设备事件，目标DP AG和EDP AG各2个事件。保留逐事件，不因相同大小而自行合并，也不把coalesced元数据量盲目乘事件数。

| 项目 | 目标组规模 | RS输入字节 | AG完整输出字节 |
|---|---:|---:|---:|
| 普通参数DP with CP |16|4,767,580,160（FP32）|2,383,790,080（BF16）|
| 专家参数EDP |2|1,887,436,800（FP32）|943,718,400（BF16）|

普通参数源完整消息量与目标一致，变化为组8→16；新增EDP消息不属于“同一DP消息带宽变低”。EDP元数据Ranks记录[0,1]，不能直接认作全局rank0/1；物理放置和局部/全局编号尚待核验。

## 带宽事实与假设

源采集机独立NIC清单：
`/home/zjb/Desktop/32/gpu32_gbs64_link_nic/20260918_103850/worker34095/nic/metadata/tcc_nic_topology.json`

记录活动CX7端口400Gb/s。采用用户指定η=60%，有效带宽=400/8×0.6=30GB/s（十进制）。此为源链路事实＋利用率假设；目标同规格、每rank独享一个NIC lane及物理组映射未完成独立核验。

简化分层候选：DP16为两机各8卡，机内分片后跨机两端交换；跨机每rank字节S/8×1/2。EDP2假设跨机两rank，字节S/2。S为完整RS输入或AG输出。

| 跨机服务项（条件候选） | 字节/rank | 60%利用率耗时 |
|---|---:|---:|
| DP RS跨机部分 |297,973,760|9.932ms|
| DP AG跨机部分 |148,986,880|4.966ms|
| EDP RS |943,718,400|31.457ms|
| EDP AG |471,859,200|15.729ms|

这些不是完整集合通信/尾段时间，未包含机内阶段、固定启动、到齐等待及网卡争用；RS/AG多事件是否同一coalesced调用或不同bucket需核对。因此不能把四项直接相加解释源尾段296.626ms→目标376.997ms。

作为算法敏感性对照，若把DP16平面ring每rank的15/16×S全部按30GB/s处理，则RS148.987ms、AG74.493ms；这不是已确认实际算法。两种假设差异很大，必须确认分层/通道/网卡共享，不能挑更贴合实测的方案。

## 启动段：大差异不是大消息

ProfilerStep→PP0首F GPU：源107.177ms、目标1196.823ms。

首F之前的default_pg全局AllGather每rank输入24个FP32元素（96字节）：源组32、完整输出3072字节，GPU时长约2.21ms；目标组256、完整输出24576字节，GPU时长1068.895ms。目标相对ProfilerStep的开始约114.53ms，结束约1183.43ms；数据和事件索引已保存。

按完整ring接收量(6144−24)×4=24480字节和30GB/s，带宽项仅约0.000816ms（未计多跳启动延迟）。因此不能用60%利用率解释约1秒驻留；它提示到齐等待、全局同步或运行时问题，尚不能认定具体原因。启动必须与大消息RS/AG分别建模，不直接套带宽倍率。

## 复现、状态和下一步

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/audit_tail_dp_edp_60_r1.py --out results/data-foundation/tail-dp-edp-60-r1-reproduction
```

输出目录需不存在。有效结果`results/data-foundation/tail-dp-edp-60-r1/`含report.json与manifest.json；保存逐RS/AG完整metadata和启动区间设备事件、原始路径/哈希、NIC清单哈希。

PASS：DP/EDP身份、元素数×dtype字节、RS/AG组大小一致性、源NIC带宽来源和启动端点。PARTIAL：目标拓扑/网卡lane/实际算法、AG bucket语义、机内带宽、完整尾段调度及启动等待原因未闭合。

下一步：核对256卡rank到worker/NIC映射、EDP真实peer及AG CPU调用/bucket身份；以固定60%利用率将确认的跨机服务项接入依赖。启动全局AllGather另追跨rank就绪，不拟合成带宽惩罚。

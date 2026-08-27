# S5000 256 卡 OISA 拓扑前置条件验证

结论：修正 legacy topology header 后，32-host × 8-GPU 的 Rank 放置、跨机
路径、400 Gbps NIC 速率和 OISA 请求输入已经对齐，状态为 `PASS`。

## 拓扑

- 绝对路径：`/tmp/oisa-rank-release-fct/topologies/s5000_256gpu_32host_1spine`；
- SHA256：`e9503e0eaf5bc80efd66ae95b5f09e68643725069d654082f374ec03f8b46a88`；
- 32 hosts，每 host 8 GPU；机内每对 GPU 一条 448 Gbps fullmesh 链路；
- 每 GPU 一条 400 Gbps NIC 到本机 leaf；每 leaf 一条 3.2 Tbps 聚合链路到
  单 spine；共 289 nodes、1,184 links；
- legacy header 为 `289 8 0 33 1184 S5000`。第二字段在 ns-3 runtime 中是
  `gpus_per_server`，不能写成总 GPU 数 256，否则 runtime 会把全集群当成一台
  host。

实测 case 的 256 ranks 均满足 `local_rank == global_rank % 8`，32 台 host
各有连续且按 8 对齐的 Rank block。跨机样例 Rank 230/238 分别映射到 leaf
284/285，路径为：

```text
rank230 → leaf284 → spine288 → leaf285 → rank238
```

## 修正后的 ns-3 结果

验证请求为 iteration 55 的 Expert-DP AllGather，2 ranks，单方向 payload
1,887,436,800 bytes，release offsets 为 rank230=87,570 ns、rank238=0。

| 指标 | 结果 |
| --- | ---: |
| arrival span | 0.087570 ms |
| tail after last release | 75.588840 ms |
| collective elapsed | 75.676410 ms |
| 单方向 flow FCT | 37.794420 ms |
| 有效单方向速率 | 399.517 Gbps |
| runner lifecycle | `exited_normally` |

恒等式成立：

```text
75.676410 ms = 0.087570 ms + 75.588840 ms
collective_elapsed = arrival_span + tail_after_last_release
```

单方向 FCT 比 400 Gbps 线速下界 37.748736 ms 慢约 0.121%，与 topology
标称 NIC 速率一致。Debug runner 自然退出，无需从超时进程恢复结果；同一
请求的 Release 与 Debug 二进制输出逐字段一致。

## 可审计边界

- 运行时 base commit：`faa3c789549d542484ca9aab132b7d4727afd14a`（dirty
  provenance 已记录）；相同源码快照已提交并推送为 `2bd0e53`；
- Debug binary SHA256：`c79494ea98413870157d60a75e2e7e08b24753b85bb5b5b284b5c0c594faf3aa`；
- Release binary SHA256：`6f3689a7848d97a329894de1026ed52dfde14d45b1bc3d66b566846cc334e9f9`；
- 旧 header（第二字段 256）产生的结果已隔离为本地诊断文件，不再用于校准；
- 该拓扑已用于九类正式仿真和 MFU v4 回填，详细结果见
  `../oisa_s5000_256gpu_nine_class/` 与 `../pp_optimizer_dag_v4_oisa_s5000/`。

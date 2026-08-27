# 256-GPU S5000 九类 OISA 结果

9/9 请求均使用真实 iteration 55 Rank 列表、payload 和 release offsets，在
修正后的 32-host × 8-GPU 单-spine topology 上自然退出。

| 类别 | op | OISA tail after last release |
| --- | --- | ---: |
| EP forward dispatch | AllToAll | 0.214777 ms |
| EP forward combine | AllToAll | 0.215346 ms |
| EP backward combine→dispatch | AllToAll | 0.217527 ms |
| EP backward dispatch→combine | AllToAll | 0.215346 ms |
| CP | AllToAll | 2.426126 ms |
| DP gradient | ReduceScatter | 380.118965 ms |
| DP parameter | AllGather | 65.335286 ms |
| Expert-DP gradient | ReduceScatter | 151.169266 ms |
| Expert-DP parameter | AllGather | 75.588840 ms |

共同 provenance：

- 运行时 base commit：`faa3c789549d542484ca9aab132b7d4727afd14a`，并记录
  `simulator_worktree_dirty=true`；相同源码快照已提交并推送为 `2bd0e53`；
- topology SHA256：`e9503e0eaf5bc80efd66ae95b5f09e68643725069d654082f374ec03f8b46a88`；
- simulator SHA256：`6f3689a7848d97a329894de1026ed52dfde14d45b1bc3d66b566846cc334e9f9`；
- runner SHA256：`cbc4a93c835c9c9a3b48962a969d99876d27634e89352646b8081b40b4657439`。

`oisa_results.csv` 是紧凑标准结果，`inputs/` 保存可复现请求，`backfill/`
保存 Trace/OISA 分层校准。原始 ns-3 flow、monitor 和 simulator log 体积约
数百 MiB，只保存在本机 `runs/`，不进入 Git。

## MFU 接入结论

绝对 OISA tail 与 Trace collective tail 不能直接互换。例如 DP-RS 代表点
OISA tail 为 380.119 ms，而 Trace last-arrival 后 service 为 15.881 ms。
最终 v4 使用有符号基线残差：

```text
target_tail = Trace_tail + (target_OISA - baseline_OISA)
```

当前 target 与 baseline topology 相同，因此回放保持 `-0.1010%` Step 误差；
未来改变 topology、payload 或并行策略时，只有 OISA 预测出的网络 delta 会
穿过 DAG slack 并影响 iteration。

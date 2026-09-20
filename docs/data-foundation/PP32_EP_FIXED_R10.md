# r10：固定 EP 整体块，iter60 → iter70 验证

2026-09-18。实现检查 PASS；可外推性 PARTIAL，不能宣布固定块跨迭代精度通过。

## 口径

沿用 r9 的 PP1、F0、首个重复单元、rank8/9 DAG。先封存 iter60 的完整图、两档 CP 成本、本地块、源残差和预测，再读 iter70 的两个 rank trace。没有读取 256 卡，没有拟合 token 分布，没有改变历史 r9 结果。

EP 简化为每 rank 的 M 整体块，覆盖 LOCAL4、dispatch、LOCAL5（含专家相关计算）、combine 和其间间隔。不在 M 之外再收费 EP。M 是带 EP 的剩余工作包络，绝不是纯 EP 通信服务成本；其余六个 EP rank 的影响尚未显式建模。

## 实际结果

| 指标 | iter60 冻结/预测 | iter70 实测 |
| --- | ---: | ---: |
| 单元窗口 ms | 83.573831 | 104.214435 |
| rank8 M ms | 22.752180 | 43.144325 |
| rank9 M ms | 22.634089 | 43.212019 |

窗口低估 20.640604 ms，绝对相对误差 19.805897%。这是该单元的误差，不是整个 F、B 或 iter 的 MFU 误差。无事先精度阈值，不能把接口测试 PASS 当成精度 PASS。

进一步检查中间证据（各 rank 自己的 F 首设备活动为时间零点）：

| rank / iter | LOCAL5 起止 ms | combine 起止 ms |
| --- | --- | --- |
| 8 / 60 | 70.857–81.425 | 83.273–83.549 |
| 8 / 70 | 70.908–80.649 | 103.756–104.060 |
| 9 / 60 | 70.699–81.211 | 83.181–83.461 |
| 9 / 70 | 71.924–102.316 | 103.926–104.214 |

rank8 的 combine 前间隔由约 1.85 ms 变成 23.11 ms；rank9 的 LOCAL5 包络由约 10.51 ms 变成 30.39 ms。rank9 LOCAL5 设备活动时长之和仅由约 10.29 ms 变为 12.61 ms（不是包络，也不保证互斥），因此不能把约 20 ms 的包络增量都归为计算增量。两者 combine 单 kernel 均仍约 0.28–0.30 ms。

结论：固定 trace 整体块接口可用，但该样本把变化的到达/间隔行为也冻结在成本内，跨迭代不稳定。证据不足以断言是 token 不均衡、网络、CPU 延迟或 GPU 同步；不能把新增 20 ms 直接标记为纯通信等待。

## 复现与证据

工程根：`/home/zjb/Desktop/worktrees/mfu-w37-v610`；分支 `feat/w37-v610`。

```bash
cd /home/zjb/Desktop/worktrees/mfu-w37-v610
python workflow/data_foundation/pp32_ep_fixed_r10.py --out /tmp/pp32-ep-fixed-r10-fresh
```

输出目录必须不存在。仅解析 source32 iteration70 的 rank8、9 两个 trace；其余输入来自冻结中间产物。不全量扫描。

结果目录：`results/data-foundation/pp32-ep-fixed-r10/`。

- `frozen-model.json`、`sealed-prediction.json`、`seal.json`：验证 trace 读取前写出，读取后哈希检查通过。
- `validation-rank-8.json`、`validation-rank-9.json`：原始路径、SHA256、关联设备事件及边界。
- `comparison.json`：全部 15 个节点的预测/实测对照。
- `report.json`、`manifest.json`：指标、限制、代码与输入哈希。

检查：源/验证的 CP 和 EP marker 类型、序号、组、尺寸、dtype 一致；独立就绪节点消去算法验证全部 DAG 完成时刻。复用同一次运行 iter60 的进程组登记，不声称跨运行元数据通用。iter70 曾用于先前工作，因此这是输入隔离的冻结参数验证，不是研究历史意义的盲测。

`pp32_f_split_r8.extract` 新增可选 path、iteration，默认行为和 iter60 哈希校验保留；旧结果目录未覆盖。

## 唯一下一步

不急于 DeepEP token 模型，先检查 rank9 LOCAL5 增长的约 20 ms 在 GPU 流内和 CPU 提交侧的位置，以及其与 rank8 combine 前间隔的关系。仅在这些 trace 证据支持时将 M 拆为固定本地/通信工作与独立就绪依赖；不能从验证集回填该间隔。当前不推广至完整 F/B 和 256 卡。

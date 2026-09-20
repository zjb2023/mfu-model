# r9：单层双 rank CP 成本替换条件回放

日期：2026-09-18。状态：PARTIAL；接口和独立重放检查 PASS，预测精度未验证。

## 范围与结果

基于 r8 冻结证据，选择 source32 / iteration 60 / PP1 / F0 / 首个重复单元 / rank 8、9。窗口是首次关联 F 设备活动开始至首次 EP combine 关联设备活动结束，不是完整 iter，也不是已证实的 CPU layer 边界。没有修改 v610、原 224 卡模型或 256 卡预测封存结果。

网页：http://192.168.8.16:43312/df-v001/pp32-cp-replay-r9/

页面可修改 CP384 和 CP256 两档成本，点击节点查看前驱、时刻、未解释间隔；包含时间线与 DAG。

| 项目 | ms |
| --- | ---: |
| 实测窗口 | 83.561128 |
| 条件回放 | 83.573831 |
| 差值 | +0.012703 |
| CP384 基线有效成本 | 11.277492 |
| CP256 基线有效成本 | 7.074432 |
| 每次 CP 增加 1 ms 后的窗口 | 87.573831 |
| CP 成本置零后的窗口 | 48.332226 |

两档成本来自同一样本：每次通信取最晚 rank kernel 结束减最晚 rank kernel 开始，再按输入尺寸求均值。它是有效通信块代理，不是独立测得的纯传输 service。384/256 MiB 指每 rank 输入缓冲区尺寸，不等同于链路实际传输字节数。

## 图与边界

本地流 C0 → C1 → C2 → C3；CP 流 CP0 → CP1 → CP2 → CP3。准备块 C1/C2 可与前面的 CP 重叠，不能全部强行串联。CP 节点以相关 rank 前驱最大完成时刻为就绪条件，成本仅计一次；更新 CP 参数不修改本地节点成本。

跨流生产者/消费者边仍是建模假设。检查到的 runtime 同步事件缺少 event/stream handle，不能据此声称全部跨流依赖已证明。模型保留实测未解释间隔，尤其 C3 前 rank 8/9 约 7.505/7.529 ms；这些不是已识别的纯等待。EP 及其余六个 rank 的影响暂时折叠在 M 块和源样本残差中。

因此上述接近实测的结果仅证明同样本条件回放和参数注入可用，不是独立预测。不能将残差原样复制到 256 卡并声称外推成立。

## 入口与复现

工程根：`/home/zjb/Desktop/worktrees/mfu-w37-v610`。

```bash
cd /home/zjb/Desktop/worktrees/mfu-w37-v610
python workflow/data_foundation/pp32_cp_replay_r9.py --out /tmp/pp32-cp-r9-fresh
python workflow/data_foundation/pp32_cp_replay_r9.py --out /tmp/pp32-cp-r9-cost-demo-fresh --costs workflow/data_foundation/cp_costs_r9_demo.json
```

输出目录必须不存在。脚本读取 r8 中间证据，不重扫原始 trace。demo 成本为人工构造的接口测试数据，不是后端预测。

成本接口支持 `effective_cp_block_ms`，或 `transport_plus_software_ms`（必须显式提供 transport_ms、software_ms）。字段存在不代表二者已被可靠分离，尤其不得把含等待的实测 kernel 时长再次叠加到传输成本。

正式结果：`results/data-foundation/pp32-cp-replay-r9/`，包含 protocol、graph、baseline-costs、active-costs、replay、report、manifest JSON。manifest 登记输入和代码路径及哈希。

复跑：`results/data-foundation/pp32-cp-replay-r9-reproduce/`，graph、baseline-costs、active-costs、replay、report 五份文件字节一致。人工成本测试：`results/data-foundation/pp32-cp-replay-r9-override-demo/`。

发布与独立检查脚本：`workflow/data_foundation/publish_pp32_cp_replay_r9.py`。已独立验证全部 15 个节点；网页返回 200，参数 +1 ms、重置、节点详情、390px 布局检查通过，未发现浏览器脚本错误。截图在临时目录 `/tmp/pp32-cp-replay-r9.png`，不作为永久结果依赖。

## 唯一下一步

冻结本版双 rank 接口，核对 C3 前约 7.5 ms 间隔的来源和跨流消费边；用 source32 另一迭代检验同一拆分及冻结参数，不从验证迭代补入新的实测残差。若证据不足，保留显式未解释项并报告误差。通过后才扩展至四层、全部 CP 组和 256 卡；EP 服务成本独立接入，禁止与折叠 M 块重复计费。

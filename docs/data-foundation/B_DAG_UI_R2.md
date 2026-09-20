# B 依赖图 UI r2

2026-09-20。仅更新可视化，不修改 r10 图、成本参数或预测结论。

入口：http://192.168.8.16:43312/df-v001/b-four-layers-r10/#b-dag

## 展示

- 四层总览：按本 stage 第4→1层展示重计算与真正反向出口。
- 单层 MoE：rank8–15 的 Dispatch、两段专家反向、本地准备、EP8共享完成、Combine及梯度处理出口。
- 单层 Attention：八卡本地计算、四组 CP2 的五次共享通信成本节点及层出口；宽图可横向滚动。
- 点击层切换子图，点击节点查看实际 DAG 标识、成本及前驱；允许外部成本输入的节点可以修改。点击箭头查看折叠路径。
- 实线表示所选同流路径；虚线表示折叠或条件路径，不代表已经证实存在物理 barrier。就绪及处理成本仍保留在底层 DAG 中。图上位置不是时间比例。

## 版本与复现

工作区：`/home/zjb/Desktop/worktrees/mfu-w37-v610`。

模板：`workflow/data_foundation/ui_b_four_layers_r10.html`；控制器：`workflow/data_foundation/ui_b_dag_r2.js`；发布器：`workflow/data_foundation/publish_b_four_layers_r10.py`。

```bash
python workflow/data_foundation/publish_b_four_layers_r10.py --refresh
node workflow/data_foundation/test_b_four_layers_r10.cjs results/data-foundation/b-four-layers-r10-ui2-smoke
node workflow/data_foundation/test_b_dag_ui_r2.cjs results/data-foundation/b-four-layers-r10-ui2-dag-check
```

发布版本 `b-four-layers-r10-ui2`。旧 UI 保留于 `results/data-foundation/2111-blocks-ui-r1/df-v001/b-four-layers-r10-ui1-archive/`，当前发布目录内 manifest 登记生成代码及输入输出哈希。

## 验证

两套浏览器测试 PASS。逐层验证全部展示箭头的每一步均来自原 DAG 前驱关系；每层 MoE 49节点、Attention 52节点、EP8共享节点8个输入。通过成本+50传播、重置、只读节点、层与rank切换、负值拒绝、390px无整页溢出及无JS错误检查。重置后基线仍为868.833ms。

证据：`results/data-foundation/b-four-layers-r10-ui2-dag-check/checks.json` 及同目录三张截图；人工检查总览、MoE和Attention截图。采用 lark-apps 的可读性和浏览器验证指导，仅本地43312发布，无云端发布、Git提交或推送。

研究状态仍为 `PASS_HYBRID_REPLAY_PARTIAL_PREDICTOR`；本轮没有新增 iter70 或256卡预测验证。

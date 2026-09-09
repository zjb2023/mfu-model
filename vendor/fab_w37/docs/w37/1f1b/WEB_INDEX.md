# W37 1F1B 网页与文档索引

主要目标是从源侧优化模型、提高目标侧事前预测精度。T35/T38 使用额外目标运行信息，作为辅助研究单列。本索引汇总基线、研究过程与证据。Markdown 文档通过网页服务渲染为 HTML，保留原文件内容。

## 阅读入口

| 内容 | 入口 | 用途 |
| --- | --- | --- |
| 分布式训练与MFU性能模型 | [一页完整阅读](/w37-report.html) | 按分布式训练、建模方式、预测任务、参数校准、版本增量、结果分账、误差诊断、外推边界八章阅读；保留完整iter三图与离线下载 |
| 三种使用场景 | [研究首页](/research.html) | 主任务与两个辅助场景的简要介绍 |
| 研究过程 | [T00–T42 与命名规则](/results/w37/A/research-html-20260907/details.html#route) | 每项研究的方法、结论与转向 |
| 当轮前缀方法 | [T35 详细说明](/results/w37/A/research-html-20260907/details.html#t35) | 方法图、节点和观测证据 |
| 目标历史辅助方法 | [T38 详细说明](/results/w37/A/research-html-20260907/details.html#t38) | 前驱采样轮残差与因果核验 |
| 逐轮指标 | [结果表](/results/w37/A/research-html-20260907/details.html#results) | 评估范围切换与逐迭代结果 |
| 验证边界 | [消融与适用范围](/results/w37/A/research-html-20260907/details.html#limits) | 未验证条件与恢复条件 |
| v685 补充说明 HTML | [v6.8.5 说明](/v685.html) | 依赖结构、时间分段、补偿来源与逐轮误差 |
| v684 原版说明 | [v6.8.4 说明](/v684.html) | 阻塞式 PP 程序顺序、结构审计与历史评估 |
| v685 冻结原版 | [历史原页](/v685-original.html) | 保留原始页面内容 |
| 基线与边界 | [BASELINE.md](../BASELINE.md) | 指标口径、输入与复现边界 |
| 完整报告 | [REPORT.md](REPORT.md) | 各研究阶段结果、消融和未验证边界 |
| 推进日志 | [ROUTE.md](ROUTE.md) | 研究假设、结果和推进路线 |
| 交接文档 | [HANDOFF.md](HANDOFF.md) | 证据入口与交接信息 |
| 持久目标 | [GOAL.md](GOAL.md) | 目标、阶段验收和恢复条件 |
| 语义说明 | [SEMANTICS.md](SEMANTICS.md) | 计算图与调度依据 |

## lane0 交互查看器

[打开预测／观测边界查看器](/lane-viewer.html)：点击84格误差矩阵，查看v687研究候选在固定lane0的起止时间、phase耗时及四轮偏差。PP/MB/FWD/BWD是筛选条件；本页不是v685或T35/T38结果，也不是纯GPU计算分账。

## 指标阅读提醒

- v685 的 **11.34%** 是四轮开发数据上的正式冷启动 1F1B MAPE。
- T35 的 **1.77%** 使用当轮前缀，平均完成 57.8% 后可用。
- T38 的 **0.54%** 使用前驱采样轮残差，只覆盖初始化后的三轮开发评估。
- 三种口径有不同的输入条件；Step 与 MFU 误差分别报告。

## 网页服务复现

在本 worktree 根目录执行。依赖只安装到本次网页产物目录：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -m pip install --target results/w37/A/research-html-20260907/python-deps 'Markdown==3.7'
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python research/w37/onef1b/serve_research_story.py
```

服务地址为 `http://192.168.0.49:8037/research.html`；页面生成入口是 `research/w37/onef1b/render_research_story.py`。此服务只展示既有文档和封存结果，不运行模型或重新评分。

## 图表命名约定

图名优先写清“源侧 / 目标侧及卡数 → 计算或通信方向 → 观测、预测或诊断 → 比较对象”。FWD 首次写作“前向激活”，BWD 写作“反向梯度”；source 不用来指发送方。坐标必须注明时间起点、单位及观测边界；API 返回给出的完成上界不得简称为精确 GPU 完成时刻。局部条件预测、完整迭代预测和观测替换诊断分别命名。

[源侧前向 / 反向通信图](/docs/w37/1f1b/post685/delivery/pp_readiness_evidence.svg)同时展示局部放大与全部样本；[局部误差表](/results/w37/A/research-html-20260907/pp_direction_local_metrics.csv)来自封存 v687 参数的逐条复核，无重新拟合或目标评分。生成入口：`python research/w37/onef1b/render_pp_evidence.py`。

`/v685.html` 补充版生成入口：`python research/w37/onef1b/render_v685_explained.py`。读取本工作区已复现和封存的 v685 结果，不重跑模型。

整合文档复现与验收：[integration/HANDOFF.md](integration/HANDOFF.md)。访问地址为 `http://192.168.0.49:8037/w37-report.html`，专用 Snakemake 管线只做已封存证据核验与文档生成。

# v6.10复现与交接

当前版由用户确认的fine_both固化，完整说明见[REPORT.md](REPORT.md)。版本包在`results/w37/A/v610-release/`，输入合同为[config.json](config.json)，增量版本登记为`workflow/config/dag_mfu_w37_versions.toml`。冻结的v685输入、旧registry、v6.9设计均保留。

## 日常入口

- http://192.168.0.49:8037/research.html ：当前版本索引。
- http://192.168.0.49:8037/w37-report.html ：当前v6.10完整说明。
- http://192.168.0.49:8037/v610.html ：固定版本入口。
- http://192.168.0.49:8037/w37-report-history.html ：历史整合说明、旧预测图、消融及外推索引。
- http://192.168.0.49:8037/research-history.html ：历史研究目录。

## Snakemake复现

只在`/home/zjb/Desktop/worktrees/fab-w37-1f1b`执行。使用新run_root，不覆盖已封存非空目录：

```bash
/home/zjb/Desktop/fabric-data-analysis/.snakemake-venv/bin/python -m snakemake \
  --snakefile research/w37/onef1b/v610/Snakefile \
  --directory results/w37/A/v610-recheck --cores 1 \
  --config run_root=/home/zjb/Desktop/worktrees/fab-w37-1f1b/results/w37/A/v610-recheck
```

流程：model重建完整节点成本并验证逐节点回放→封存→evaluate目标四轮→render只展示本版。模型阶段拒读目标观测，哈希预检可读字节但不解释；输出仅当前stage可写，文件系统只读挂载，禁网络，单线程、900秒阶段超时，申报内存4GB。只读取冻结派生文件，不扫描原始trace或启动OISA。

版本封装读取之前已封存的源成本赋值，不重新选择或拟合方法。要复现这些源参数的拟合与研究消融，使用[binding/HANDOFF.md](../binding/HANDOFF.md)里的独立管线；v6.10配置记录其固定输入及SHA。更换研究输入意味着新版本研究，不应修改本版hash来绕过检查。

## 版本产物

- `model/nodes.csv.gz`：327,746完整节点，包含v6.10成本、来源、开始/结束与关键路径信息。
- `model/edges.csv.gz`：364,784依赖边，字节复用冻结边表。
- `model/node_parameter_bindings.csv.gz`：70,192条赋值，含完整node_id、rank、PP、MB、源键、倍率、旧新计算与时长。
- `model/fine_parameters.csv`、`shape_factors.csv`：源细粒度成本与继承倍率。
- `model/predictions.csv`、`prediction_seal.json`、`release.json`：本版预测、封存和声明。
- `evaluate/iteration_results.csv`、`phase_iteration_results.csv`、`metrics.json`：只有v6.10的逐轮与阶段评分。
- `render/full_iter_payload.json`、`full_iter_events.csv`、`full_iter_ledger.csv`：新版预测与两侧冻结观测的绘图证据。
- `render/index.html`、`research-index.html`：版本说明与当前索引；图数据、参数表、CSS/JS内嵌，下载后可离线阅读。
- 各阶段`complete.json`、`input_access.json`、`code_snapshot/`保留SHA、读取清单和执行代码。

当前HTML的科学断言及浏览器验收入口为`research/w37/onef1b/v610_check.py`和`v610_browser.cjs`。浏览器检查保存于`results/w37/A/v610-webcheck/`，避免改变已封存render目录。

原生旧限时Goal仍为blocked，本轮固化及唯一后续研究目标由[GOAL.md](../GOAL.md)管理。不自动启动agent，不写其他worktree或原项目，不推送/合并。

最终版本生成目录为`results/w37/A/v610-final-20260908/`；`v610-release`是指向它的稳定交付链接，网页及机读下载继续使用该路径。当前版的所有阶段均在新目录从固定输入重建并验证。此前试装与网页修正记录保留于`v610-first-package-20260908`及`v610-webcheck/history/`；不参与最终评分。

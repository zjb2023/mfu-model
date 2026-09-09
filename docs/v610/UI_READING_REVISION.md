# 完整阅读结构与独立参数图册

模型保持 v6.10；展示修订为 `ui-reading-r3`。仅使用封存派生数据，没有模型回放、拟合或原始 trace 扫描。
原 W37 报告、vendor 模板以及 `release-final` 的 HTML/payload/seal 均未覆盖。

## 阅读入口

- 当前报告：`http://192.168.0.49:43311/w37-report.html`。
- 两种建模方式：`/w37-report.html#why-dag`，复用历史报告的三阶段/DAG比较、依赖与等待解释和教学图。
- 参数与校准：`/w37-report.html#calibration`，先给简要说明及图册入口，再按需展开完整参数表。
- 独立参数图册：`/v610-parameter-atlas.html`，整轮→PP/MB/F/B→成本节点→参数值、源85/90样本、继承倍率和关键前驱。
- 教学图：`/v610-teaching-dag.html`，假设数值用于教学，不是224卡精度证据。
- 冻结原 v610 页面：`/v610-frozen.html`；历史版本：`/w37-report-history.html`。

按历史页恢复八章：分布式训练、两种建模方式、任务与结构、参数与校准、版本记录、结果与分账、误差诊断、边界与下一步。
主页面使用 v610 配置、参数、图和四轮结果；历史版本仅在记录与来源索引中保留，不复用旧图册的144条更新作为当前数据。
诊断表由既有四轮阶段结果汇总“实际−预测”平均差额，不是独立成本贡献或优化收益。

## 参数图册的精确范围

源图327746节点、364784边；6765条细粒度源参数、4387实际绑定键、70192更新节点、224 rank均保留。
图册显示每个PP的lane0代表rank：14张卡、84个F/B动作；这不是全224卡包络。通信汇总覆盖全部组，零时长节点保留计数。
图册数据保留代表切片对应的更新记录及全部6765源参数；全70192条绑定仍可从主报告下载。
入口/OPT及RS、AG可点击；区间使用真实冻结预测起止。图外补差和outer不伪装成GPU节点，节点时长不能直接求和当整轮。

## 文件与复现

源码：`scripts/v610/build_reading_ui.py`、`reading_ui_node.js`、`serve.py`、`check_reading_ui.cjs`。
输入：`results/v610/release-final/{model,evaluate,render}`，构建前逐文件核验旧 complete seal；历史阅读结构从已迁移的 SHA 固定快照提取。
输出：`results/v610/ui-reading-r3/` 中的 `index.html`、`PARAMETER_ITER_ATLAS.html`、`DAG_4GPU_DEMO.html`、`atlas-data.json`、`manifest.json`。
manifest 记录输入/代码/输出 SHA；HTML 内嵌数据，支持单文件离线交互。预览 `ui-reading-r1` / `ui-reading-r2` 另存，未覆盖；r3移除了沿用旧术语表中的3.420秒诊断示例，避免误认当前结果。

```bash
# 使用新目录生成另一份展示，不覆盖已有修订。
.venv/bin/python -B scripts/v610/build_reading_ui.py \
  --run-root results/v610/release-final --output results/v610/ui-reading-next
.venv/bin/python -B scripts/v610/serve.py \
  --run-root results/v610/release-final --ui-root results/v610/ui-reading-next \
  --host 127.0.0.1 --port 0
# 服务启动后在另一个终端执行
node scripts/v610/check_reading_ui.cjs
```

浏览器证据在 `results/v610/ui-reading-check/acceptance.json`，截图和下载后的离线 HTML 同目录。
最终43311服务上的33项浏览器检查通过，覆盖八章导航、6765条参数、独立图册的节点来源和往返、教学图返回、离线交互及移动端横向溢出。既有32项测试通过；模型与历史源文件未修改。
展示修订的清单与浏览器结果副本存于 `docs/v610/ui-reading-r3/`；它们记录本次工作区实现的内容 SHA，不将未提交代码冒充正式 Git 版本。
既有冻结模型验收仍见 `ACCEPTANCE.md`，其84项旧页面测试不用于声称新布局通过；新布局使用独立检查器。

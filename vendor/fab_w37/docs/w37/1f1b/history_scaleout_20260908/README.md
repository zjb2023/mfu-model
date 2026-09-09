# 历史报告的16→256章节替换

来源：`http://192.168.0.49:8038/w37-16to256.html`，抓取时刻与原文件SHA见 `source_manifest.json`。完整HTML及11份小型证据只读复制到本目录，没有修改8038所在工作区，也没有重新执行Task B。

更新入口：`http://192.168.0.49:8037/w37-report-history.html#scaleout`。
独立阅读：`http://192.168.0.49:8037/docs/w37/1f1b/history_scaleout_20260908/methods.html`。

新增三阶段T3与rank/layer M3，保留原M2、B0/B1与敏感性方法，合计6种方法、168行源留出/目标评价结果。36项汇总公式复核通过。来源页面的复现命令属于原Task B工作区，不是本目录的执行入口。

在本工作区重新生成与验证本次文档整合：

```bash
python research/w37/onef1b/history_scaleout/build.py
node research/w37/onef1b/history_scaleout/check.cjs
```

生成结果位于 `results/w37/A/history-scaleout-20260908/`，8037服务优先读取该版本。冻结的 `results/w37/A/integration-20260907/integrated_report.html` 保留原字节；v6.10当前报告路由保持不变。

整合清单见 `integration_manifest.json`。浏览器验收位于结果目录的 `browser_acceptance.json`，74项通过，包含阻断8038后本地交互/证据/下载仍可用、其他历史表格、手机宽度与当前报告入口。冻结输入与源派生结果的既有科学边界没有因文档整合改变。

第02章另增加4卡教学DAG入口及三阶段16.823672%历史时间误差摘要；示例页可返回该工作报告。该导航往返和指标口径文字已纳入浏览器验收，冻结模型结果不变。

2026-09-09：第06章新增224卡85/90/95/100四轮的实测时间换算MFU与有效算力表，区分Training整轮与Profiler区间。采用继承的每轮8.436548311982576e16 FLOPs、单卡500 TFLOP/s；分子未独立复核。生成逻辑为 `research/w37/onef1b/history_scaleout/mfu_results.py`，对原评价表SHA及MFU公式进行核验，CSV/JSON输出在本目录所述结果目录中；不执行模型或读取原始trace。

新增第08章 MLPerf DeepSeek-V3 配方接入TODO，用户纠正来源为 `~/SimAI/docs/mlperf_training_recipes/DEEPSEEK_V3_TRAINING_RECIPES_MFU.md` 后，已接入四条配方及三条MFU估算，保留R10缺日志状态；源文档/CSV/JSON/脚本只读复制，SHA与算术校验加入生成入口。来源清单、MFU账本和跨规模评价验收见 `../MLPERF_DEEPSEEK_V3_TODO.md`。

第04章以参数图册为主，默认正文只保留三项校准摘要；完整旧说明、参数表和审计折叠保存，旧锚点访问自动展开。第05章与模型数据不变。

v685四轮平均时间图追加MFU对比：预测3.309862%，按实际Training时间逐轮换算后的MFU均值2.955109%；时间与MFU分轴，均值差不是MAPE。生成器 `time_mfu_figure.py` 校验冻结原图、逐轮CSV和换算公式，输出扩展SVG及JSON均值，原图不改。

## 单HTML阅读入口（上一版，长图册例外见下）

主报告现在将教学DAG、参数图册、16→256完整方法报告、历史256卡DAG流程和通信分片原理共5份HTML以内嵌srcdoc保存。主报告目录与本地HTML阅读链接全部改为本页锚点；v685/v684/研究索引分别指向对应结果、历史基线、开发路线。原独立页面保留，但主报告阅读不再依赖跳转到它们。

`inline_reading.py` 在生成阶段完成内嵌与链接转换，处理子页面本地锚点，避免srcdoc继承base后导航出子页面；旧锚点继续自动展开详情。论文原始出处、CSV/JSON/PDF/Markdown证据下载仍保留来源链接。页面约16MB，内容增加来自内嵌冻结数据，无额外仿真或trace扫描。

`node research/w37/onef1b/history_scaleout/check_inline.cjs` 检查无本地HTML阅读跳转、全部主页面锚点存在、5份视图内嵌、章节与历史视图、阻断网络后离线参数绑定/EP交互/16→256方法选择。20项检查通过，原报告74项检查通过；证据见 `inline_acceptance.json`、`inline_link_audit.json` 和 `inline_reading_manifest.json`（结果目录）。

## 长页面改为独立图册

按用户最新阅读偏好，第04章参数图册和第02章教学图改为普通同标签页链接，不再内嵌。参数图册顶部返回第04章，教学图顶部返回第02章，浏览器后退也可返回。16→256完整报告及两份历史流程图仍在本页展开。主报告的本地HTML导航仅允许这两个长图册，其余目录和证据规则保持不变。

`check_inline.cjs` 已改为验证两个允许的独立页面、打开/返回往返、无长图册iframe，以及剩余三份内嵌视图和离线16→256交互。

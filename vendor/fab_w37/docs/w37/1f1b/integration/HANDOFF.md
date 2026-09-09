# 整合 HTML 交接与复现

唯一工作区：`/home/zjb/Desktop/worktrees/fab-w37-1f1b`。生成产物位于 `results/w37/A/integration-20260907/`，历史模型与研究证据保持不变。

## 当前版本：八章技术主线（2026-09-08）

八章依次为分布式训练、两种建模方式、预测任务与结构、性能模型参数与校准、版本增量、结果与误差分账、误差诊断、外推边界与下一步。章节内容归属见 [REPORT.md](REPORT.md)。不再按周五/周末组织主文。

`organization.py`负责章节结构和总体说明，`render.py`提供已有证据区块；`context_sections.py`提供历史依据与既有外推案例，`parameter_sections.py`维护参数清单。调整内容应修改这些生成器，不能直接改最终HTML。`organization_map.json`记录八章标题及旧入口归属。

保留所有旧ID与链接。`report.js`处理初次打开及点击页内锚点，自动展开其祖先详情；`#baseline`进入第06章历史结果，`#graph`进入第03章图规模，`#three-case-parameters`进入第04章参数证据。下载文件名为`W37_分布式训练与MFU性能预测.html`。宽表和大SVG在容器内横向滚动，正文长路径正常换行。

本轮161项文档检查、156项浏览器检查、39个本机地址通过；30项科学产物、两个交互JSON数据集、5张SVG图片和4张直接内嵌SVG保持不变。当前HTML哈希与文档/浏览器验收一致。历史外部链接与计算键命名专项报告保留原验收时点，不能当作本轮新访问或新拟合证据。后文早期修订的检查数保留作历史记录。

重组前快照在`history/before-technical-reorganization-20260908/`，对应源码`7f69383`。新增`reorganization_input.json`固定旧ID、链接、科学产物及数据/图形指纹；验证器直接比较这些值，旧HTML快照仅用于人工追溯，不是重建依赖。研究输入的80项、14项及5项清单均未更改。

## 阅读

- [整合 HTML](http://192.168.0.49:8037/w37-report.html)：本次主交付；使用页面“下载完整 HTML”可离线阅读内嵌图表与参数。
- [本次审计报告](REPORT.md)、[精简验收](acceptance.json)、[固定输入](inputs.json)。
- [当前文件 Goal](../GOAL.md)、[原目标完整快照](../history/20260907-before-integration-GOAL.md)。

## 文档专用 Snakemake

在唯一 worktree 根目录执行：

```bash
/home/zjb/Desktop/fabric-data-analysis/.snakemake-venv/bin/python -m snakemake \
  --snakefile research/w37/onef1b/integration/Snakefile \
  --directory results/w37/A/integration-20260907 --cores 1
```

规则包含 `audit → full_iteration`、`context`、`parameter_review` 和 `cost_inventory` 四条只读证据路径，汇合后执行 `render → validate`。`.snakemake` 元数据位于本次结果目录，不触发历史全局管线。首次验收通过 `--forceall` 运行了原有规则；完整 iter 修订通过增量规则生成。重复运行应显示结果最新。输入 SHA256 改变时应失败，不能为继续生成而自动重写哈希。

80 个输入约 25 MB，均为已有源码、文档或派生表，没有原始 trace。图节点/边来自已有压缩 CSV，重新统计不重扫原始采集。原工程只读；此环境依赖本机已有缓存及历史文件，不能宣称脱离数据的裸克隆也能重建。

## 网页服务与浏览器验收

现有服务监听 `0.0.0.0:8037`，新增 `/w37-report.html` 路由。研究首页和文档索引已经加入该入口。服务恢复命令：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B research/w37/onef1b/serve_research_story.py
```

沿用已经安装在结果目录中的 Markdown 3.7、Playwright 和本机 Chromium；不在原工程安装依赖。若服务已经占用端口，应复用现有进程，勿重复启动。

```bash
node research/w37/onef1b/integration/browser_check.cjs
```

浏览器脚本只检查展示层，产物为 `browser_validation.json` 和 `browser/` 截图、下载副本。离线测试直接打开实际下载的 HTML，阻断网络后检查图片、筛选和评价范围切换。没有重新拟合、评分或发送外部消息。

## 产物与维护

`audit.json` 汇总图规模、真实参数命中、时钟闭合和指标；`audit_checks.json` 保存逐项断言；`input_verification.json` 保存固定输入核验；各 CSV 保存图类型、参数与逐轮结果；`render_manifest.json`、`validation.json`、`browser_validation.json` 记录最终 HTML 哈希及检查结果。

`full_iter.py` 只从固定 v685 节点表提取既有预测时间，从固定 v682 展示 payload 读取 224/256 全 stage 观测包络，再与 v685 四轮总账及源侧日志时钟核对。没有调用历史 builder。`full_iter_payload.json` 包含三图全部数据，`full_iter_events.csv` 为 1,778 个绘图区间，`full_iter_ledger.csv` 为 12 行三场景分段总账，`full_iter_validation.json` 记录 1,080 项检查；展示代码为 `full_iter.js`。

修订前 HTML 与验收保存在结果目录的 `history/before-full-iter/`，上一版本源码及文档保存在提交 `279f963`。正式模型、旧研究结果及固定输入哈希保持不变。

所有核心数字由固定结果核验后整合。修改主内容用 `research/w37/onef1b/integration/render.py`，样式与交互分别为 `style.css`、`report.js`；边类型中文解释在 `terminology.py`。需要改历史结论时先补证据并创建新输入版本，勿覆盖历史科学产物。

本次只结束“文档整合”目标。主研究仍缺可迁移的源侧运行时/动态路由证据；在线辅助恢复需遵循 T41 新连续数据合同，不能拿旧四轮反复挑方法。

## 历史背景与外推案例修订

新增 `context.py` 从固定 Git 对象提取证据，按 `context_inputs.json` 中的 commit、blob、SHA256 和字节数核验。历史说明只读 `/home/zjb/Desktop/mfu-model` 的 e71ccb0 对象；Task B 只读当前仓库的 c3f39c2 对象，不读取兄弟工作区。14 份附加输入约 0.29 MB，导出在本结果目录 `context/`。移机复现需要这些固定 Git 对象；不自动 fetch 或改写提供方。

`context_audit.json` 记录 1,259 项检查，`scaleout_target_points.csv` 保存主方法 20 点，`scaleout_summary.csv` 保存 8 行汇总。原始 112 点在 `context/scaleout/results/w37/B/final-validation-20260905T233625518711Z/point_results.csv`。`context_sections.py` 生成建模动机、16→256 两张对照图和工具 TODO；此模块不重新拟合或执行外推预测。

新增历史可视化链接使用固定提交的 8037 副本，不依赖原 8013 服务。正文下载后仍可离线查看新增动机、20 点图表及工具待办，外部参考资料链接需要网络。来源快照原文按文本展示，以免复制来的 Markdown 相对链接指向不完整历史目录。

原 80 项输入仍由原清单锁定；新增输入使用独立清单，不为通过验收改写旧哈希。修订前页面和验收保存于 `history/before-context-scaleout/`。本次追加证据并不恢复旧限时原生 Goal，当前文档目标继续通过 `../GOAL.md` 管理。

outer 图示修订：`full_iter.js` 在图上方分别列出原始图→Profiler 补差、Profiler→Training outer，并明确没有事件定位；移除右端补差阴影，保留所有事件时刻与总量。修订前文件保存在 `history/before-outer-clarification/`。

## 三组静态参数与后续路线修订

`parameter_review.py`通过独立 `parameter_inputs.json` 核验5个已有文件，读取两份TOML、两份已提取启动字段CSV和固定16/256场景JSON，输出 `parameter_review.json` 与23行 `three_case_parameters.csv`。专用Snakemake增加 `parameter_review → render`，不修改原80项或14项冻结清单；第三清单含前次已有场景文件，不把清单项数简单相加当唯一来源数。

第08章根据已知配置区分256→224同形状成本绑定和16→256形状成本迁移；主任务先审计已有信息的落地，不能默认先要求新增数据。静态输入一致性不作为实际kernel耗时一致、部署版本独立核验或模型精度改善的证据。修订前页面与验收保存在 `history/before-next-steps-review/`。

## 完整参数体系维护

第04章直接展示参数体系，锚点`#parameter-system`；前后向版本公式为`#pp-model-versions`，计算更新位置为`#compute-binding-explained`。正文由`parameter_sections.py`生成；不要手改生成HTML。

`cost_inventory.py`只读已入旧80项清单的派生节点/参数表，生成`model_cost_inventory.json`和`model_cost_inventory.csv`，不调用模型builder。专用Snakemake已接入依赖；更改本模块后运行上述管线与浏览器验收。新统计是37个成本来源组，不能称37个独立模型参数。

当前验收：8项成本检查、70项文档检查、136项浏览器检查、37个本机地址；28项旧数值/证据产物不变。旧11个外部链接地址不变，复用其原访问记录，不声称本轮重新访问。最新文件SHA见acceptance.json。参数逐数值来源仍应查固定CSV和合同；本节未声称所有历史成本的物理拆分已验证。

“三个实际命中键”表与58键表统一使用五字段原值：方向、层类型、上下文、执行位置、语义槽；中文位置说明另列。展示修改位于`render.py`，不改源CSV或绑定逻辑。逐键验收见`key_naming_validation.json`。

第04章参数图锚点为`#v685-calibration-map`；独立文件在结果目录`v685_calibration_map.svg`，机读对应为`v685_calibration_map.json`。改图应编辑`calibration_diagram.py`后运行专用Snakemake与浏览器验收。图中计算成本和总时间读取原审计结果，固定位置说明来自已核验绑定；不要将参数流向箭头误作训练依赖。当前73项文档检查、140项浏览器检查、39个本机地址通过。

计算键定义入口：`/w37-report.html#compute-key-definition`，由`parameter_sections.py`在第04章参数总览前生成；五字段原值沿用已有成本键，中文释义与键/成本/节点的对应关系另列。本次只补充解释，无模型变更。

参数明细表前对应图由`parameter_sections.py::node_parameter_figure()`生成，在`render.py`的58参数/144节点明细表前调用；成本从已有节点更新CSV读取。样式在`style.css`的`.node-map-*`，浏览器脚本包含位置与编号检查、截图。入口`/w37-report.html#parameter-node-map`。

## 新增绑定研究的恢复入口

04章`#binding-audit`与06章`#binding-results`展示独立源成本迁移实验；正式v685的旧图和参数表仍保留。来源为`docs/w37/1f1b/binding/delivery_inputs.json`中固定SHA的小表，不覆盖原80份整合输入。详细科学复现及唯一研究下一步见[绑定研究交接](../binding/HANDOFF.md)。文档渲染入口和浏览器检查命令不变，当前179项文档/173项浏览器检查与47个HTTP入口通过。

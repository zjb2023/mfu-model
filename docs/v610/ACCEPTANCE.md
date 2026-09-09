# v6.10 接管验收（2026-09-09）

状态：PASS，范围是冻结复现与本地分支接入，不是新的泛化精度结论。

## 版本与位置

- 分支：`feat/w37-v610`；工作区：`/home/zjb/Desktop/worktrees/mfu-w37-v610`。
- 原 mfu 基线：`e71ccb0b9157053e13a7e95e957659f2d8170764`。
- 源交接：`b8330a0ffc1c4d73c0a2403291b89edd55999a58`；清单审计：`146cc97d5e6b717bd517b8c38040ff1081430f7f`。
- 源快照本地提交 `ba7c62d`；复现接入 `3610ac1`；归档适配修正 `9fe55b0`。最终证据提交在这三项之后；以 `git log -1` 获取当前 HEAD，不在文件中自引用本提交 SHA。
- 运行、输入、配置、代码快照、日志的绝对路径及 SHA：`docs/v610/run_manifest.json`。
- 数据/历史制品包：`/home/zjb/Desktop/mfu-model-artifacts/w37-v610-b8330a0`，379 对象 / 208061015 字节；371 原审计对象无缺失、SHA 无变化；8 个新增网页别名另列。
- 426 个 vendor 文件保持原字节；388 个核对源 Git blob，38 个此前仅 SHA 审计的历史派生证据在本分支明确纳管。最大文件是历史参数 HTML，不是原始 trace。

## 冻结数值与覆盖

| 指标 | 复现值 |
| --- | ---: |
| raw graph | 20953.110588 ms |
| ProfilerStep | 21557.550256125 ms |
| Training Step | 22930.003646124995 ms |
| MFU（派生） | 3.285055047379364% |
| 完整节点 / 边 | 327746 / 364784 |
| 细粒度参数 / 实际绑定键 | 6765 / 4387 |
| 唯一更新节点 | 70192 |
| 覆盖 | 224 rank / PP14 / 3 microbatch |

原图锁 `f1d560daaa344a5980697b67e71b97140c9bfd38fe12a2c676daf05c53bd8d04` 不变。
entry 1265.388210 ms 已在图内；图外补差 604.439668124998 ms 和 outer 1372.453390 ms 没有重复计时。
四轮已暴露目标评价保持 Profiler MAPE 9.649264825329237%、Training MAPE 10.043845141783976%、MFU 相对 MAPE 11.167230469707011%；这些是历史开发指标，不是迁移带来的改善。

## 已执行验收

- `results/v610/release-final`：独立 Python 环境下完整 model→evaluate→render，Snakemake 4/4 jobs 完成；后续 dry-run 无待运行任务。
- `results/v610/binding-final`：audit→model→review→evaluate，Snakemake 5/5 jobs 完成；四种原始变体重新生成，后续 dry-run 无待运行任务。
- `scripts/v610/verify.py` 共 40 项 PASS：26 张科学表解压后字节完全一致，包括每个节点的五类成本、时长/起止、零成本节点、拓扑边、绑定与全部四组时序；4 个科学 JSON 完全一致；另核验制品、代码、阶段 seal 及读取边界。见 `acceptance/verification.json`。
- binding 保持 58 个聚合键、原 3 键 / 144 节点直接绑定；optimizer 成本未改变，目标观测读取探针拒绝；源细粒度参数重新拟合结果完全一致。见 `acceptance/binding-audit.json`、`binding-review.json`。
- 32 项测试通过（原工程 25 项 + 7 项接入边界测试），含错误 SHA、断链、软链接 resolve、输出越界、未登记输入拒绝、冻结 FLOPs、无通信日志的合成 profiler 网格及 raw 显式授权。见 `acceptance/tests.xml`。
- Profiler 默认 DAG 2/2 jobs 完成，只 SHA 校验已有源派生表；无真实 raw 扫描。合成网格测试仅 256 个小型假文件名，无 trace 语义解析；真实硬件日志可选分支未运行。
- v610 页面 84 项浏览器断言、17 条 HTTP 链接通过；历史页 23 项通过，含章节锚点、两个图册往返、内嵌视图和离线 scaleout 交互。`browser-history-links.json` 中字段 `badLinks` 是原检查器对“非内联 HTML 候选链接”的命名，所列三个链接均属于允许的两个独立图册，实际往返已通过，不是 404。
- 主页面、历史页和两个图册 HTTP 200、UTF-8 解码通过。新服务 `http://192.168.0.49:43311/research.html`；原 8037 服务未停止、未修改。

## 运行隔离及保留情况

每次 worker 的原 W37、fabric-data-analysis、原 mfu 路径被隐藏；只使用新 worktree 的 vendor 与 `.venv`，不借旧工程代码或旧 Python 环境执行。全系统只读、当前 stage 独占可写、禁网络；原始 trace 不在输入白名单内。
旧 seal 原文和原 SHA 保留；新物理位置通过映射与新 seal 登记，未更新旧 SHA 绕过校验。原 mfu 仍是 main/e71ccb0，暂存区空，两个原未跟踪目录保留；W37 仍是 b8330a0 且工作区干净。
没有指定 raw 新位置，所以没有搬迁原始数据；未来搬迁必须留旧路径软链接。已测兼容机制不等于真实 raw 迁移完成。

## 中途问题和处理记录

- 最初 `frozen-release-r1` 在 Python 启动前因隔离环境缺少随机设备退出；补上私有 `/dev` 后 `frozen-release-r2` 与最终 `release-final` 均通过，失败目录保留。
- 首轮网页检查发现原 371 清单遗漏 `/v685.html` 对应的旧展示文件；只补充网页别名及 SHA，不改报告科学内容；初次失败记录保留于 `results/v610/webcheck-release/acceptance-before-alias-fix.json`。
- `binding-reproduction-r1` 运行成功，但补齐网页清单后 Snakemake 判断其输入清单变化；保留旧结果，在新目录 `binding-final` 重新生成，使最终清单与执行元数据一致。
- 归档器最初误把历史检查器的列表输出当对象，已修复，未影响任何模型结果。源 CSV 的 CRLF 按原 SHA 保留，vendor 不做空白格式化。

## 未解决的研究边界

256→224 是已暴露开发评价；完整源兼容细粒度图回放、125 条无新拟合覆盖的源开发观测、旧 floor 的独立物理归属仍有限制。raw→所有历史校准项的生成链仍不完整；没有启动原全局 Snakemake、legacy builder、OISA 或额外研究。16→256 不能把本版 256 校准时序作为隐含输入。

复现命令和后续数据契约见 [README.md](README.md)。本次在本地提交后结束，不推送、不合并、不修改 weekly-todo 或 W37 冻结材料。

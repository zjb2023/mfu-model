# W37 任务A：1F1B计算图优化交接

准备状态：PASS，研究尚未启动。只负责A，不执行16→256任务，不自动启动其他agent。

## 基线身份

- 工作区：`/home/zjb/Desktop/worktrees/fab-w37-1f1b`
- 分支：`feat/w37-1f1b`
- 共同基线提交：`818f10415b131dbc4f25c3dbfae458dc3cea4c9d`，在原HEAD `13f4178a7cc38cbef2c0cd18fc953d66b543d587` 上精确加入当前磁盘必要源码，不是单纯从旧HEAD开分支。
- 当前开发预测头v6.8.5，依赖锁来自v6.8.4；v6.9仍 `DESIGN_ONLY_BLOCKED_NOT_RELEASED`。
- 共同说明：[BASELINE.md](/home/zjb/Desktop/worktrees/fab-w37-1f1b/docs/w37/BASELINE.md)。版本登记：[dag_mfu_versions.toml](/home/zjb/Desktop/worktrees/fab-w37-1f1b/workflow/config/dag_mfu_versions.toml)。

## 配置和数据入口

- [v6.8.5配置](/home/zjb/Desktop/worktrees/fab-w37-1f1b/case_224gpu_pp14_cp2_a2a/config/dag_v685_source_steady_calibration_2026w36.toml)
- [代码快照清单](/home/zjb/Desktop/worktrees/fab-w37-1f1b/docs/w37/coordination/code_manifest.json)：60份代码/config的原路径与SHA。
- [冻结输入清单](/home/zjb/Desktop/worktrees/fab-w37-1f1b/docs/w37/coordination/external_inputs.json)：A允许只读原工程派生表和封存结果，所有读入按SHA核验；原始Trace不复制、不重扫。
- [版本核验记录](/home/zjb/Desktop/worktrees/fab-w37-1f1b/docs/w37/coordination/baseline_audit.json)：真实拓扑锁、指标和候选输入缺口。

## 已验证结果和最小命令

在本工作区执行：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B scripts/w37/run_baseline.py --task A --mode smoke
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B scripts/w37/run_baseline.py --task A --mode reproduce-v685
```

smoke已通过：327746节点、364784边回放；图内时间20781.256754ms，加继承的目标残差604.439668ms后Profiler为21385.696422ms，training为22758.149812ms。拓扑SHA为 `f1d560daaa344a5980697b67e71b97140c9bfd38fe12a2c676daf05c53bd8d04`。

准备区及本工作区完整重校准复现均已通过，Profiler MAPE为10.369528494%。所有结果生成于各自工作区，源码和原工程以只读挂载执行；物理挂载检查、原始Trace拒读和回写拒绝探针通过。精简验收证据另保存于 `docs/w37/acceptance/` 纳入本分支提交，运行代码与共同基线相同。

- [本工作区smoke结果](/home/zjb/Desktop/worktrees/fab-w37-1f1b/results/w37/A/smoke-preparation/smoke_result.json)
- [访问审计](/home/zjb/Desktop/worktrees/fab-w37-1f1b/results/w37/A/smoke-preparation/input_access_audit.json)
- [完整复现目录](/home/zjb/Desktop/worktrees/fab-w37-1f1b/results/w37/A/reproduce-v685-preparation)

## 首个研究动作与验收

从已封存v6.8.5建立同85/90/95/100窗口的前后向时间分账，定位计算活跃、等待和调度成本的剩余缺口；先做源侧守恒，再比较224开发集。验收是逐轮同口径变化、准备/前后向/收尾的可加时间账本和源侧回归，不只给一个平均误差。

v6.8.2的1.860秒/75.8%来自60–100窗口，不能直接当v6.8.5的误差构成。当前约2.474秒稳态低估仍待解释；不能把所有前后向包络缺口归为纯计算或OISA网络service。

## 修改范围与禁止事项

新增优先落在 `research/w37/onef1b/`、`tests/w37/onef1b/`、`docs/w37/onef1b/`，输出放 `results/w37/A/`。必要时修改本工作区模型脚本或新建版本；保留旧版输入与复现入口，避免旧seal失效。

参数改变保持v6.8.4依赖锁。若需修正依赖，必须提供训练代码/静态调度证据，登记新语义版本，不通过额外等待边追求更低误差。224时长只在封存后的evaluator读取，不能直接拟合参数；已看过的224结果只能称开发验证。

v6.8.5只部分重校准85–100，旧非计算floor与outer等仍来自先前256拟合。旧审计中的fit行范围不等于物理CSV读取范围。两套时钟分别报告，MAPE不能直接换算成MFU误差。

只允许本分支精确范围本地提交；不推送、不合并，不写原工程、兄弟工作区、weekly-todo或mfu-model。裸执行历史builder/总Snakemake目标未获本基线的输出隔离保证；从受保护入口扩展新的研究runner。

## 与B的接口

[INTERFACES.md](/home/zjb/Desktop/worktrees/fab-w37-1f1b/docs/w37/INTERFACES.md) 固定 scenario、图结构、成本表、collective FCT、双时钟和seal字段。交付通用结构与256拟合参数时分开命名，B不能隐式继承这些数值。任何共享变更以明确commit/schema交付，不引用兄弟未提交文件。

## 新会话第一条消息建议

“读取docs/w37/BASELINE_HANDOFF.md，担任任务A负责人。以共同基线818f104开始W37的1F1B误差分析与优化，先复核同窗口分账和源侧守恒；遵守正确依赖和输入边界，不执行任务B。”

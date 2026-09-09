# 计算键绑定研究交接

本轮完成键格式审计、源成本比较、固定拓扑候选与消融、目标同口径回归，以及报告/HTML。科学结论见[REPORT.md](REPORT.md)，路线见[ROUTE.md](ROUTE.md)。正式版本仍是v685，本候选不登记成已发布新版本。

## 恢复入口

唯一工作区 `/home/zjb/Desktop/worktrees/fab-w37-1f1b`，只负责Task A。不写原项目、其他worktree或weekly-todo，不推送合并，不自动启动agent。原生旧Goal被blocked未完成目标占用，新研究按[GOAL.md](../GOAL.md)管理。

唯一下一步：构建兼容全局层号/MB计算键的源侧完整回放，核对暴露区间旧完整时长下限与计算的归属；再对固定规则做源95/100增量开发回归。源旧图缺少兼容字段，不能直接把新成本套进旧source字段后宣称验证完成。源95/100与目标四轮都不是盲测。

## 复现

先检查冻结输入；遇SHA不一致停止相关复现，不修改hash来绕过。

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B scripts/w37/run_baseline.py \
  --task A --mode smoke --run-id binding-review

/home/zjb/Desktop/fabric-data-analysis/.snakemake-venv/bin/python -m snakemake \
  --snakefile research/w37/onef1b/binding/Snakefile \
  --directory results/w37/A/binding-review --cores 1 \
  --config run_root=/home/zjb/Desktop/worktrees/fab-w37-1f1b/results/w37/A/binding-review
```

每次使用新结果目录；runner拒绝覆盖任何非空阶段目录。若仅想检查本次已完成结果，使用相同命令但两处目录改为`binding-20260908-r2`；不加强制重跑。只读派生输入约23.15 MB，单线程，阶段超时900秒，申报内存4 GB，不扫描原始trace。

研究管线依次执行`audit → model → review → evaluate`。前三阶段内容读取受源侧白名单约束；封存后evaluator才读取目标实际值。所有输入先核验哈希，模型更新固定四组。预测和代码快照保存在各阶段，不能覆盖封存文件。

精简证据发布入口（固定读取本轮r2，验证全部阶段产物后复制小表）：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B research/w37/onef1b/binding_delivery.py
/home/zjb/Desktop/fabric-data-analysis/.snakemake-venv/bin/python -m snakemake \
  --snakefile research/w37/onef1b/integration/Snakefile \
  --directory results/w37/A/integration-20260907 --cores 1
node research/w37/onef1b/integration/browser_check.cjs
```

网页：http://192.168.0.49:8037/w37-report.html#binding-audit 以及 `#binding-results`；从research.html原入口仍可进入。

## 完整机读结果

结果根目录：`results/w37/A/binding-20260908-r2/`。小表复制到本目录的[evidence](evidence/)，提交的CSV副本统一为LF换行，delivery_inputs同时记录副本SHA、原结果SHA和转换方式；数值不变。其输入/输出哈希与验收见[research_acceptance.json](research_acceptance.json)。

- `audit/all_58_key_audit.csv`：逐聚合键的直接命中、细粒度关联、缺口原因。
- `audit/fine_node_bindings.csv.gz`：全部80,064细粒度节点，包括没有新拟合的记录。
- `audit/fine_parameters.csv`：6,765条源85/90参数。
- `model/node_updates.csv.gz`：三个非基线变体共140,384条赋值记录，含rank/PP/MB/完整node_id及旧新成本。
- `model/*_timings.csv.gz`、`*_critical.csv`：四组全节点预测时间和关键路径；依赖边复用冻结输入中的v685边表。
- `model/prediction_seal.json`：先预测后目标评分的文件封存与成本规则。
- `review/source_common_rows.csv.gz`、`source_method_comparison.csv`：相同源侧样本的方法比较。
- `review/tail_propagation.csv`、`update_effect_summary.csv`：收尾边界传播、成本变化与旧时长下限影响。
- `evaluate/iteration_results.csv`、`phase_iteration_results.csv`：四组×四轮Step/MFU及16组×4阶段；其他表汇总MAPE、偏差及退化。
- 各阶段`input_access.json`、`command.json`、`code_snapshot/`、`complete.json`记录读取、运行、代码和输出SHA。

r1完整保留在`results/w37/A/binding-20260908/`；r2只是执行向量化并补齐源比较/传播审查，四组预测及全部更新记录逐值一致。见[coverage_and_reproduction.json](coverage_and_reproduction.json)。不要把跨节点成本和直接当迭代时间改善，不把候选旧wall floor说成已独立测得的纯软件成本。

全量表在线入口：[逐节点赋值](http://192.168.0.49:8037/results/w37/A/binding-20260908-r2/model/node_updates.csv.gz) · [全部细粒度节点及缺口](http://192.168.0.49:8037/results/w37/A/binding-20260908-r2/audit/fine_node_bindings.csv.gz) · [源85/90细粒度参数](http://192.168.0.49:8037/results/w37/A/binding-20260908-r2/audit/fine_parameters.csv)。

> 2026-09-08三阶段任务已完成：[THREE_STAGE_V1.md](THREE_STAGE_V1.md)。T3源16校准→封存→目标20点评价，Step12.9662s、MFU5.8853%，Training MAPE44.8083%。在线8038报告新增三阶段分账、公式与六方法168行对照。预测/评分复现使用reproduce_prediction_three_stage_v1.py和reproduce_evaluation_three_stage_v1.py；HTML使用build_methods_html.py及publish_methods_html.py。M3细粒度对齐仍PARTIAL，旧结果不重写。

> 2026-09-08新增方法对齐续作：[METHOD_ALIGNMENT_V6.md](METHOD_ALIGNMENT_V6.md)。M3 rank/layer候选已跑通源校准→seal→固定20点目标评价，Step MAPE47.7087%，未替代旧M2。完整细粒度对齐仍PARTIAL；当前状态以GOAL.md顶部续作记录为准。以下原最终交接保持历史。

# W37 Task B 最终交接

2026-09-06。**方案与实际16→256 Step/MFU最小端到端验证完成。** 用户确认S5000 BF16为500 TFLOPS并恢复目标后，已补齐绝对MFU及百分点误差，保持v5时间预测、参数与FLOPs不变。报告见[FINAL_REPORT.md](FINAL_REPORT.md)，逐项验收见[COMPLETION_AUDIT.md](COMPLETION_AUDIT.md)。

主方法预测目标training Step 12.7385s，实际22.1810–25.5773s，training/Profiler MAPE45.7773%/44.0737%。MFU预测5.9905%，目标实际20点均值3.2482%（范围2.9835%–3.4403%），MAE2.7423个百分点、相对MAPE84.8177%。源16卡留出约3%不能报告成跨规模误差；敏感性分支未事后替换主方法。

## 交付与复现

- [统一112条逐点Step/MFU及误差](../../../results/w37/B/final-validation-20260905T233625518711Z/point_results.csv)、[八行汇总](../../../results/w37/B/final-validation-20260905T233625518711Z/summary.json)、[独立核验](../../../results/w37/B/final-validation-20260905T233625518711Z/checks.json)。源8点×4方法、目标20点×4方法，B0/B1均保留。
- 峰值500 TFLOPS/GPU采用[用户确认合同](peak_contract_user_20260906.json)，user-confirmed static input，不是独立厂商验证。MFU主分子为封存因果注意力三角有效FLOPs，排除重算等硬件工作；引用绝对MFU时同时说明此口径。
- v5时间seal：`d7766035f5cbb86d821df085ee5b0e19c2cf6b286f0f11af1508d463563775cb`；本次MFU seal：`6fca5b4da8cb50da6739fd9de1dd3655ae0fd30d665c174b74ee314458016c05`。
- 当前正式MFU目录mfu-addendum-v1-20260905T233332765810Z；最终统一结果目录final-validation-20260905T233625518711Z（UTC命名，对应本地2026-09-06）。

在本worktree根目录执行：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/16to256/reproduce_mfu_addendum_v1.py --peak-contract docs/w37/16to256/peak_contract_user_20260906.json
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/16to256/reproduce_final_checks.py
```

第二条默认核验正式运行；可加--mfu-run指定第一条新目录。源提取、时间预测、目标双时钟评价命令见v3/v4/v5报告。新建目录拒绝覆盖，输入/代码SHA改变必须停止；当前缺失峰值模板仍为空，用于保留旧版拒绝分支，不是当前已准入合同。

## 来源与版本边界

唯一worktree为/home/zjb/Desktop/worktrees/fab-w37-16to256，分支feat/w37-16to256-mfu，共同基线818f10415b131dbc4f25c3dbfae458dc3cea4c9d。只读原数据，生成物在results/w37/B；没有推送、合并或修改weekly-todo/原工程/兄弟worktree。

v5 seal固定experiment_v5.py、predict_v4.py、structure_v4.py、core.py及输入registry。新MFU seal另外固定mfu_addendum_v1.py、evaluator_v5.py、峰值合同和用户证据。不要修改任何这些封存文件；后续变更使用新版本。20项结构/时钟检查、6项MFU证据/执行检查及本次真实112点独立公式核验均有记录。

wo主attempt用于成本，w独立登记。旧256拟合成本、等待与entry/outer残差未用于源16校准；目标早有历史分析，本次封存评价不是新的独立盲测。CP2/跨host代理、单一源形状域、未绑定训练commit/NIC/时钟offset界仍是模型限制，不能把运行通过解释成这些假设已独立验证。

不等待任务A，不读其未提交目录；后续仅按明确commit与共同scenario→结构→来源参数→双时钟→seal→evaluator接口接入，继续审计成本来源。若做目标分项诊断，单独标target-assisted，不替换本次source-only成绩。

## 两句周报

方案完成：已实现16→256分项成本迁移、PP2→PP16显式流水调度和来源隔离的预测封存/评价流程。

实验验证完成：按用户确认的BF16 500 TFLOPS完成20点Step/MFU评价，主方法预测MFU5.99%、实测均值3.25%，MFU MAE2.74个百分点、相对MAPE84.82%，training Step MAPE45.78%；约3%仅是源侧留出误差。

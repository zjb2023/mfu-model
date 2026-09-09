# 16→256 输入与封存契约 v1

**最终补充（2026-09-06）：** [peak_contract_user_20260906.json](peak_contract_user_20260906.json)准入用户确认的单卡BF16 dense 500 TFLOPS，来源明确为user-confirmed而非独立厂商验证。峰值为独立静态归一化输入；沿用v5时间/FLOPs，先封存绝对MFU预测，再读取已封存的v5动态评价产物。112条绝对MFU/百分点误差已完成，见[最终报告](FINAL_REPORT.md)。以下v1/v3/v5段落保留各版本当时状态；旧scenarios与seal内null峰值不回填。

**v5补充：** [experiment_inputs_v5.json](experiment_inputs_v5.json)与[experiment_plan_v5.json](experiment_plan_v5.json)固定12轮源校准、目标静态物和四种方法；源8轮留出由模型拒读。两侧Profiler均为全rank ProfilerStep标记包络，training与实际日志一致。沿用v4非时钟成本并重新封存后评价，未用目标误差调整参数。20点双时钟及MFU相对误差已完成；绝对MFU仍缺BF16稠密峰值。v4目标评价暴露明确披露，见[CONTINUATION_V5.md](CONTINUATION_V5.md)。

v3研究准入追加：[source_cost_inputs_v3.json](source_cost_inputs_v3.json)固定192份wo源校准parquet；[scenarios_v3.json](scenarios_v3.json)绑定全部目标rank的静态PG/host。目标仅允许固定SHA和长度的参数块/trace元数据前缀，禁止动态事件及整个目标文件哈希；前缀准入与复现见[CONTINUATION_V3.md](CONTINUATION_V3.md)。本版本不改旧共同基线10份源输入或历史封存结果。

本契约对应共同基线 `818f10415b131dbc4f25c3dbfae458dc3cea4c9d` 和 [scenarios.json](scenarios.json)、[experiment_plan.json](experiment_plan.json)、[cost_provenance.json](cost_provenance.json)。当前允许执行源侧 Step 留出验证和单位结构检查；目标 Step/MFU 门禁为 BLOCKED。

## 三种数据角色

| 角色 | 允许内容 | 使用进程与限制 |
|---|---|---|
| source16 empirical | 每个 attempt 的冻结派生产物、后续独立登记的算子/通信服务特征 | prepare 核验并分割；fit 仅打开 calibration 分区。禁止跨 attempt 拼接、用目标误差选源轮次 |
| target256 static | 绑定实际运行的训练参数、代码版本、层分配、并行组、host/GPU/网络拓扑 | 可供建图；脚本默认值保留在 `candidate_defaults_only`，实际值未知为 null。禁止目标观测 token、时间、吞吐率、arrival、残差 |
| target256 evaluator-only | 同一 attempt 的 iteration、Profiler 边界、training wall、边界定义、时钟校准元数据 | 目标预测封存后由单独 evaluator 登记、校验、读取；本轮未登记、未读取，也没有生成目标预测 |

初始来源：[external_inputs.json](../coordination/external_inputs.json) 中 `allowed_tasks` 含 B 的 **10** 份文件；每 attempt 各有 `attempt_contract/input_manifest/per_iteration_metrics/readiness_report/validation_report`。外部路径保持原工程只读。SHA 不符立即失败，不更新哈希来绕过失败；原始归档 SHA 是历史声明，本轮未重验归档。

## 静态字段的证据等级

`scenarios.json` 的 source/target 实际字段与候选默认值完全分开。已确认源拓扑来自分 attempt 合同；目标 world_size=256、PP16、CP2 是任务指定场景，实际 launch binding 仍缺失。每条静态证据保存文件路径和 SHA。静态脚本只读、不执行 shell；脚本引用的远程训练目录不是本机已有版本证据。

所有场景最终必须有 `case_id/attempt_id/world_size/TP/PP/DP/EP/CP`、实际层清单、microbatch size、microbatch 数、GBS、sequence length、模型算子形状、dtype、重计算、VPP/MTP、训练代码 commit、运行软件版本、GPU SKU、host/group/PP peer 映射。MFU 另需 `flops_per_step/flops_definition/peak_flops_per_gpu`。未知不得填 0 或以 case 文件夹名补齐。

若配置已确认，检查 `world_size=TP×PP×CP×DP`；EP 是实现定义的子组/重排，不能再乘入 world_size。固定 batch、无 rampup 时 `microbatches=GBS/(DP×micro_batch_size)` 必须是整数，CP 不再作为 batch 副本计入。EDP 与 CP/EP 的关系必须核对具体 Megatron 版本的 group 生成代码，不能直接按 DP/EP 填值。

## 实测字段与参数

当前分区行：`attempt_id, iteration, profiler_window_ms, training_wall_ms`；必须正数、有限值、iteration 唯一且完整。输入原表虽含通信 exposure、bytes 和 spread，本轮 fit 不读取这些列。已有均值 exposure 不是关键路径服务成本，不能用 `Step−通信 exposure` 当计算耗时。

未来每个经验参数必须有 `cost_key/value/unit/component/origin_class/fit_case_id/fit_iterations/evidence_path/sha256`，另存 shape、资源、组与拓扑适用域、估计方法、样本数和源侧不确定性。`origin_class` 只允许 `source16_empirical` 或有独立文件证据的 `independent_hardware_measurement`；通用代码的结构常数另标 `code_semantics`，不能冒充时间参数。source256、target-assisted、旧 DAG 成本、entry/outer/residual/floor 全部拒绝进入 source-only 参数表。

当前参数是按 attempt 单独拟合的 Profiler/training 聚合常数，明确 `transfer_to_target_allowed=false`。预测中的 `outer_framework` 只是两种聚合时钟之差，保留符号；不声称它是软件服务耗时。

## 切分与已实现的可执行流程

主 attempt 预定为 `wo_mccllog_1116`（无 MCCL 日志），`w_mccllog_1129` 独立检验日志条件。两者都使用 5、10、…、60 的 12 点校准，65、70、…、100 的 8 点留出。主基线是分别取两种时钟中位数；同时报告均值基线，不按留出误差改主方法。非 Profiler 轮次不在这 10 份准入表内，因此本实验不用其 1–100 全量训练时间。

1. launcher 先冻结三份 JSON、研究代码 SHA、Python 身份与共同基线。prepare 校验 10 份输入 SHA，分别输出每个 attempt 的 calibration/holdout 与 partition manifest；此进程物理读取整份源表。此前源表也被人工式工具审阅，不能称源侧新盲测。
2. fit 在独立 bubblewrap 进程中只准读 calibration、静态快照、通用代码及元数据哈希，按已冻结切分拟合；运行时探针确认 holdout、target 动态和旧成本图拒读。两次 attempt 独立求值，函数拒绝混合行。
3. 写 `parameters.json/predictions.json/prediction_seal.json`。seal 包含配置、分区 manifest、校准行、参数、预测的相对路径与 SHA；run_inputs 绑定代码版本与全部来源。launcher 在 evaluator 启动前另存 seal SHA checkpoint。
4. 独立 source evaluator 先验证 checkpoint 与 seal 中全部哈希，再校验/打开 holdout、计算逐点误差；评分后重验封存。输出只到 `evaluate/`。本 evaluator 明确拒绝 target256 case。
5. 每进程文件系统 `/` 只读、仅自己输出目录可写、网络隔离；Python 拒绝未登记数据和子进程。保留每阶段读取审计、执行日志、命令、run manifest。该保护针对可信本地研究代码，不声称防恶意本地代码或人员篡改。

入口：`/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B docs/w37/16to256/reproduce.py`。每次创建新目录，已有 run-id 拒绝覆盖。共同基线 smoke 必须使用 `scripts/w37/run_baseline.py --task B --mode smoke`，不使用旧 builder/global Snakemake。

## 后续目标预测与评分的接口（待输入后实现）

实际 predictor 应仅挂载校准分区、目标静态合同和独立硬件证据；目标动态目录不挂载/不准读。输出语义 nodes/edges、显式 parameters、逐 iteration `predicted_profiler_step_ms/predicted_outer_framework_ms/predicted_training_step_ms/predicted_mfu_pct`，再封存全部输入、图、成本、预测、代码 SHA 与 source split。未知 MFU 时为 null，Step 所需成本缺失时不输出目标预测。

目标 evaluator 需要新增**独立入口和仅评价进程可读的注册表**：预定轮次 5–100 每 5 轮，共 20 点，先验证目标预测 scope、完整表格与 seal，再读取/哈希目标提取表。缺点、重复、非正值或时钟边界不可比应失败，不能自动缩为后四轮。此顺序包括目标数据的 SHA 读取：目标内容哈希也只能在 seal 后执行。

当前未实现目标时钟适配器或带成本目标预测器，因为服务特征和实际合同未准入。不能把当前 source seal 冒充 target seal。后续首次评分仍必须披露历史目标暴露；如果看到目标误差后改参数或边，应另开版本，成绩标开发集迭代。必要的 target-assisted 诊断单独放 `results/w37/B/<new-run>/target_assisted/`，不与 source-only 成绩合并。

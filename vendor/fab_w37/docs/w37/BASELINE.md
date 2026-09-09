# W37 共同基线：使用入口与证据边界

用户指定工作周 W37，准备日期 2026-09-05。本分支只准备两个独立研究入口。

## 权威版本与指标

- 开发预测头：v6.8.5；256卡部分参数按85/90/95/100重新拟合，224卡同窗口开发验证。
- 完整依赖拓扑继承v6.8.4：327746个节点，364784条边；规范化边哈希为 `f1d560daaa344a5980697b67e71b97140c9bfd38fe12a2c676daf05c53bd8d04`。
- 预测Profiler（性能采样覆盖的训练时间）21.385696422秒；完整training时钟22.758149812秒；Profiler MAPE 10.369528494%，训练时钟MAPE 10.718040852%。MAPE是平均绝对相对时间误差，不等于MFU百分比误差；不把100%-MAPE当作统计意义上的准确率。
- v6.8.4在同85–100窗口MAPE为12.444829677%。v6.8.2的75.8%指其60–100误差中前后向包络的占比；不得移作v6.8.5误差占比。
- v6.8.5并非所有参数只来自85–100：旧非计算成本下限、源图成本和outer framework（采样区间外训练开销）继续继承旧256拟合。新拟合读取的源表本身包含60–100，只对指定行求参数。旧审计的source_iterations_read字段不能当成物理文件只含四轮的证明。
- v6.9登记状态 `DESIGN_ONLY_BLOCKED_NOT_RELEASED`，没有可发布的端到端预测；实际输入就绪记录见 `coordination/baseline_audit.json`。

## 三类输入

1. 通用逻辑：`workflow/scripts/build_dag_mfu_schedule_factorial.py` 的 `schedule/build_graph/replay` 是无毫秒参数的最小1F1B调度工厂，只验证结构，不是完整细粒度MFU模型。
2. 256拟合状态：旧v6模型图、ENTRY、PP梯度、计算槽、残差、CP/EP/DP/EDP模板和已校准通信数据。A可以复现；B不得以这些数值开始16→256预测，即使文件名叫nodes/config、带source-only标记或存于224目录。
3. 16源侧数据：`case_16gpu_pp2`有两次独立采集；无日志与有通信日志两套attempt不能混用。冻结入口见 `coordination/source16_inventory.json`，只读数据路径和SHA见 `coordination/external_inputs.json`。

所有原始Trace留在原位置，本次不复制、不读取其内容、不入Git。16卡input_manifest里的原始归档SHA只是旧声明，本轮未重验原始归档。73个冻结外部引用都是小型报告或已计算中间产物；总计约72MB，原件未移动。输入变化时命令应失败，不自动用新哈希覆盖旧清单。

## 运行

在各自worktree根目录执行，Python使用本机现有只读环境：

```bash
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B scripts/w37/run_baseline.py --task A --mode smoke
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B scripts/w37/run_baseline.py --task A --mode reproduce-v685
/home/zjb/Desktop/fabric-data-analysis/.venv/bin/python -B scripts/w37/run_baseline.py --task B --mode smoke
```

A的smoke重新求解冻结327746节点的最长路径并与封存预测比较；reproduce-v685从现有256中间表重新估计已有参数并生成完整v6.8.5结果，封存后才读取224评分。B的smoke只检验单位成本图和两次16卡采集各20个采样点的数据入口，不产生16→256预测或校准结论。

运行使用bubblewrap：全文件系统只读，仅当前worktree本次`results/w37`输出和私有/tmp可写，网络隔离。Python读取审计进一步拒绝未登记输入和原始Trace，禁止子进程。不能以裸执行旧builder、旧Snakemake总目标代替该入口：部分历史脚本会重扫Trace或使用旧绝对输出路径。修改研究代码后建立自己的受保护运行入口，并继承这些边界测试。

命令默认创建新的时间戳结果目录；可指定`--run-id`，已有目录会拒绝覆盖。每次结果保存execution.log、command.json、smoke_result.json、input_access_audit.json及run_manifest.json。这里的校验是可信本地代码的运行保护，不是防恶意代码逃逸的安全沙箱。

## 可复现范围与后续问题

共同提交从原HEAD之上精确加入当前磁盘代码闭包、登记配置、输入清单和W37运行器；没有把所有dirty文件收进提交。代码清单记录60份当前源码/config的SHA；外部产物冻结73份引用。原HEAD自带的历史报告仍可见，但不是B模型的合格输入。

源Python3.11.9、pandas3.0.5、numpy2.4.6。依赖环境仍为同机只读引用；移机时按运行结果environment重建，不承诺裸克隆脱离外部数据即可运行。

16卡PP2/DP8/EP8/CP1已由合同声明；模型层数、microbatch、序列长度、GBS/FLOPs及目标CP2迁移需要B审阅训练代码/静态启动配置后明确。已有L6空间分析PASS不等于MFU校准准备完毕。B第一阶段可独立产出设计与缺口清单，不等A优化结果。

两任务共同接口见 [INTERFACES.md](INTERFACES.md)。研究输出不得写回原工程，不推送、不合并；专用分支允许精确路径本地提交。

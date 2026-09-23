# 224/256逐层专家工作量与耗时曲线 · r1

2026-09-23。只解释实测，不修改32→256模型，不声称消除了6.16%误差。16卡研究暂缓。

## 本轮结论

实际专家输入行数是成本变化的强解释变量；层号本身暂不能作为稳定的预测自变量。

- 224/256各iter40/60/80，代表rank跨层/MB的输入行数与B专家FC GPU成本相关系数0.965～0.987；原F为0.991～0.997。是描述性相关，不是因果贡献比例。
- 逐层高低位置不稳定：256同MB跨iter的工作量曲线相关系数约−0.045～0.115，B FC成本约−0.109～0.197；224工作量约−0.225～0.232。不能将单rank的一轮曲线固化成“深层必快”公式。也不等于否定更粗stage包络或组总量可能有稳定趋势。
- 256 iter60完整第一组EP8，56层×4MB总专家行数62865～164098，rank负载CV约0.173～0.862。总量确实变化，不只是8卡之间转移。这里行数是token-expert assignment处理量，不能叫唯一token数量。
- 例：iter60 MB0，逻辑层3的EP8总行数149900、最忙rank B FC成本16.74ms；层27为66293、8.48ms；层58为94043、11.71ms。仅为曲线上三个点，不据此证明U形规律。局部最大FC成本不是层DAG完成时间。

## 容量处理是重要机制，不只是路由分配

6份代表trace（224 PP1/6/12与256 PP1/7/14，iter60）共84个层×MB窗口，均找到输入[8192,160]的top6、专家轴top615以及logical_and。参考236B router在capacity_factor非空时执行容量筛选；moe_utils计算ceil(8192×6÷160×2)=615。8个发送rank汇总后单专家上限4920，与trace常见输入上限吻合。

256日志记录capacity_factor=2.0、pad_to_capacity=False、drop_policy=probs，同时有旧字段moe_token_dropping=False。不能仅凭后者断言没有容量筛选；trace实际算子序列更直接。

因此合理解释链是：层输入/路由分数变化 → top-k专家选择及容量筛选 → 实际保留的逐专家行数与非空专家数 → GEMM形状/次数/成本变化。尚未重建容量处理前的逐token完整路由概率，没有证明每个时间差都来自容量丢弃或层深度。236B为代码语义参照，未核实运行时源码提交完全相同。

## 范围与方法

- 256 PP1–14和224 PP1–12，iter40/60/80各取stage×16代表rank：78份trace。
- 256 iter60补其余7rank，每stage首组EP8完整：额外98份。共176份，2672条层×MB×rank记录。
- 首stage2层，中间stage4层，逻辑层（1-based）=3+4×(stage−1)+local_layer。CheckpointFunction与CheckpointFunctionBackward按Sequence number配对；B执行层序3→0，不当作前向顺序。
- F中split_with_sizes提供20个本地专家行数，两次FC一致；重计算_GroupedLinear行数列表逐项等于原F；真实反向按Sequence number匹配。
- CPU runtime/driver correlation归属GPU；FC成本仅统计GEMM设备活动并集。F使用split→silu→split→unpermute限定FC窗口；B使用精确GroupedLinear调用。两种方法的原始事件索引均保留。
- B专家FC=重计算FC1/FC2+真实反向FC1/FC2；inter_FC_local单独保留，不称纯门控FLOPs。未把EP、重排或未覆盖空隙算成FC，也未把gap当纯等待。
- 896个256 iter60 B FC调用与已冻结fc-transfer-r1-verified逐项一致（容差1e-8ms）。原始输入哈希与既有provenance一致；补充rank重新登记SHA256。
- 没有比较不同host绝对时钟；EP8 max/sum只描述局部工作量和成本，不替代DAG调度。

## 证据入口（根目录绝对路径）

工程：`/home/zjb/Desktop/worktrees/mfu-16to256`，分支`32to256`，起点提交`9f83971d36382c665dfd00c1ace86e28c97e9c10`。新增结果尚未提交。

- `results/data-foundation/layer-workload-224-r1/manifest.json`、`layer-workload-256-r1/manifest.json`：原始路径/哈希、代码哈希及逐trace衍生文件校验和；`layers.csv`为事实表。
- `results/data-foundation/layer-workload-analysis-r1/data.json`及manifest：相关系数、逐层代表rank、完整EP8汇总与896项回归。
- `results/data-foundation/layer-capacity-audit-r1/report.json`：84个trace窗口事件索引、路径/哈希、236B源码哈希。
- 参考源码：`/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/moe/router.py`（capacity入口）、同目录`moe_utils.py`（get_capacity/apply_router_token_dropping）。
- 日志：`/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40/worker33008/2026-07-31_1340/tp1_pp16_dp_mbs2_numbs_gbs64_gpus0_mtp1_forcelbfalse_pertensorfalse_NO_LOSS_REDUCE.RANK2.10.124.33.8.log`。

## 复现

在工程根执行。现有输出目录不可覆盖；提取器--out选择新目录。聚合脚本当前固定读r1路径且拒绝覆盖其输出，完整复现可在独立worktree保持路径布局执行。

```bash
/usr/bin/python3 -S workflow/data_foundation/extract_layer_workload_curves_r1.py --worlds 224 --out results/data-foundation/layer-workload-224-r1
/usr/bin/python3 -S workflow/data_foundation/extract_layer_workload_curves_r1.py --worlds 256 --ep8-iter60 --out results/data-foundation/layer-workload-256-r1
/usr/bin/python3 -S workflow/data_foundation/analyze_layer_workload_curves_r1.py
/usr/bin/python3 -S workflow/data_foundation/audit_layer_capacity_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer_workload_r1.py
```

网页：`http://192.168.8.16:43312/df-v001/layer-workload-r1/`。

## 唯一下一步

补256完整首组EP8的iter40/80，先验证组总工作量随层变化是否跨iter稳定，再决定是否值得建立位置特征。第二EDP副本与224完整EP8仍为缺口。不要将本页目标工作量作为source-only外推时提前已知的输入。

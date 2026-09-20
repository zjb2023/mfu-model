# 四单元B装配约束 r9

2026-09-20。PASS_PHASE_ASSEMBLY_GATE_NOT_PREDICTION。

236B `Megatron-LM/megatron/core/transformer/transformer_block.py:456`的block重计算分支对各个layer分别调用`checkpoint_handler(custom(layer_idx, layer_idx + 1))`。因此本配置recompute_num_layers=4表示四层分别checkpoint，不是uniform模式下一次checkpoint包四层。

对应32卡iter60 PP1 B0八卡设备关联证据，四单元均按如下顺序出现：

```text
u0重计算 → u0真正反向 → u1重计算 → u1真正反向
→ u2重计算 → u2真正反向 → u3重计算 → u3真正反向
```

理论上应对应本stage正向层的逆序；尚无trace module-id逐层核对，不把单元号强行改成全局layer号。

本轮生成64个阶段包络（8卡×4单元×2阶段）及逐rank交接参数。所有阶段结束边界回放通过，并保持旧B设备窗口868.833359ms。没有增加跨rank阶段barrier。包络之间的观察顺序仅适用于本trace，不据此认定所有配置必须全阶段串行。

八个单元外设备事件单独存入outside-units.json，不静默塞进某一层。阶段事件ID无重复；单元外尾部显式保留。

这是装配记账与回归门，不是新增粗参数预测模型。后续细粒度模块应替换相应包络，绝不能将包络耗时再加到r7节点成本上。r7仅覆盖单元0真正反向的一部分链路，尚未完整参数化四单元；重计算和反向剩余头尾都必须覆盖后，才能称为完整B。

复现（工作区`/home/zjb/Desktop/worktrees/mfu-w37-v610`）：

```bash
python workflow/data_foundation/build_b_four_unit_skeleton_r9.py --out /tmp/b-four-unit-skeleton-r9-new
```

输出必须不存在。已执行目录`results/data-foundation/b-four-unit-skeleton-r9/`包含graph、replay、phase-audit、outside-units、summary、manifest。manifest登记框架、源证据、脚本和输出哈希。无新原始trace扫描，无iter70/256数据，无提交推送。

唯一下一步：参数化r5–r7生成器的unit选择，逐单元替换对应真正反向子区间，并将未覆盖头尾显式留下；以本轮64边界回归门检查覆盖与重复计费。重计算暂留trace包络，不谎称其已成本化。

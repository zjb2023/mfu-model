# B r10：冻结60预测70，及实测跨迭代波动

2026-09-20。工作区 `/home/zjb/Desktop/worktrees/mfu-w37-v610`。

## 结论与范围

32卡 PP1 B0、rank8–15、四层含重计算；不是整轮所有B，也不是新全局MFU结果。
iter60 图及全部成本/就绪项在读取目标前封存，重新调度与原 replay 完全一致。
随后独立提取60/70同口径 CPU归属的设备包络，不进行目标拟合。

- 八卡 B0 窗口：60为868.833359ms，70为893.787567ms，慢24.954208ms（相对60 +2.872151%）。
- 冻结60预测70：868.833359ms，误差−24.954208ms，绝对相对误差2.791962%（分母70）。
- 64个重计算/反向阶段：时长MAE 3.118902ms，累计结束点MAE 11.232278ms。
- 同八卡各自 ProfilerStep 平均：12467.185640→12678.772175ms，增加211.586536ms，约1.697%。各卡相对增幅1.676%～1.711%。这是CPU各卡窗口均值，不是全32卡同步Training Step。
- 全部八个阶段的八卡平均时长都增加；最后执行的本stage第1层增量最大：重计算+4.105332ms、真正反向+7.339668ms。逐层累积，不是已证明的单一额外等待。

注意：由于图和输入全部固定，预测仍为60回放值。本次检验的是该冻结模型的跨迭代迁移误差，不能声称已建模或解释70的动态波动。源就绪残余、重计算包络和条件EP/CP耦合均保留。iter70已有历史曝光，不是盲测。没有预设验收门槛，因此仅评价执行/封存验证PASS，不能把精度自动标成PASS。

## 产物与复现

最终结果：`results/data-foundation/b-r10-60to70-eval-r2/`。

- `frozen-graph.json`、`sealed-prediction.json`、`seal.json`：读取目标前的封存及哈希。
- `facts-60.json`、`facts-70.json`：16个已有trace的归属提取；记录原路径、SHA256、时间原点、阶段和组件包络。
- `phase-comparison.json`：64个阶段的时长和累计边界对照，源侧边界与r10全部匹配。
- `component-variation.json`：320项（8卡×4层×10组件）两次实测差异，包含Dispatch、FC2、FC1、Combine、Attention及CP0–4。跨流包络可能重叠，不可相加当作误差因果分解；通信可见包络不等于纯service。
- `report.json`、`manifest.json`：报告及输入/输出/代码版本。

初版 `b-r10-60to70-eval-r1` 保留，只有MoE组件；r2补齐Attention/CP身份检查，完整B及阶段指标与r1一致。

```bash
python -B workflow/data_foundation/evaluate_b_r10_60_to70.py --out /tmp/b-r10-60to70-new
python -B workflow/data_foundation/publish_b_eval_r1.py
node workflow/data_foundation/test_b_eval_r1.cjs results/data-foundation/b-r10-60to70-ui-check-new
```

输出目录必须不存在；发布器面向当前固定r2目录，已发布时不覆盖。
本轮未修改原trace、旧B图、F图、224/v610冻结结果；未提交或推送。

## HTML

http://192.168.8.16:43312/df-v001/b-r10-60to70-r2/

逐卡预测/实测时间线、四层增量图、组件明细及整轮ProfilerStep明细。采用lark-apps技能的口径分离、可读性与浏览器验证指导，仅现有43312本地发布。

验证：`results/data-foundation/b-r10-60to70-ui-check/checks.json` 为PASS；HTTP 200、八卡切换、64阶段/320组件、时间线正宽度、390px无整页溢出、无JS错误；已检查同目录desktop.png。源图/输出/代码哈希核验PASS，r1/r2总体及阶段结果完全相同。

组件线索：最后执行层的八卡平均Combine可见跨流包络增加3.829ms，FC2/FC1分别增加0.630/1.257ms，Attention增加0.135ms，五次CP单项平均增量为+0.016/+0.007/−0.041/+0.018/+0.096ms。这里只是测量差异，不能把Combine包络增量全部解释为网络变慢或与其他项直接相加。

下一步：固定本次评价记录，优先审计最后执行层的Combine可见包络与本地处理/就绪变化，再决定哪些外部工作量特征值得进入成本模型；不得直接用70差值反填60参数。

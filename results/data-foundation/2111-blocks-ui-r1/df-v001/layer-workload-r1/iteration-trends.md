# 九轮完整EP8工作量趋势 · r1

2026-09-23。本轮只解释已有数据，未重新训练、未新增采集、未修改冻结预测模型。

## 已验证结果

- 36个iter×MB首末对比中，L58两组工作量均低于L3，但相邻层有升有降；不是逐层单调递减。
- 固定层×MB共224个位置，从iter40到80均增加；仅111个位置的八个采样间隔全部增加，不能称全部随iter单调上升。
- MB0两组总量：L3为83092（iter40）→304032（60）→411130（80）；L58为59380→179904→349712。
- MB0的L3→L58相对降幅跨九轮为28.54%、43.97%、53.12%、67.39%、40.83%、19.36%、32.72%、10.77%、14.94%。层间差异反复存在，但幅度不是常数，不能将iter60的40.83%固定用于其他轮次。
- MB0实际参与专家数：组A L3从18到92、L58从7到74；组B L3从18到88、L58从7到67（iter40→80）。这都是筛选后数量，不是参数本身的“集中程度”。
- 645120项逐专家计数守恒、896项iter60回归通过。未修改模型，未声称已分离权重更新与输入批次的因果贡献。

## 研究范围

区分“同一iter内，层位置不同”和“固定一层，iter变化”。样本为256卡iter40、45、50、55、60、65、70、75、80，即每5轮采样一次，不是40～80每个整数轮次。覆盖PP1～14中间56层、四个microbatch、两个EP8副本，首尾stage不纳入。

同一iter前向过程中，后层并不因前层刚完成而更新权重；其输入激活却来自前面的层。跨iter同时改变训练状态/权重及输入批次，所以曲线关联不能分离权重更新与输入数据的因果贡献。运行日志配置lr=min_lr=3.48e-05、mock_data=False，不据此假定每一步参数更新都已逐项验证。

## 证据路径

工程绝对根：`/home/zjb/Desktop/worktrees/mfu-16to256`。

- `results/data-foundation/layer-iteration-grid-r1/`：每stage/EP8组的逐层记录、日志行号、代表trace checkpoint事件索引、来源SHA256。
- `results/data-foundation/layer-iteration-analysis-r1/`：两组合计、固定层跨iter、各iter首末层及相邻层变化、iter60回归与manifest。
- `results/data-foundation/layer-iteration-grid-ui-r1/`：当前页面测试报告和截图。

## 身份与数量核对

1. 每stage每EP8组使用其lane0代表rank，共252份trace。CPU forward_step定位microbatch；四个CheckpointFunction按前向顺序对应逻辑层3+4×(stage−1)+local_layer。
2. 同rank的baseTimeNanoseconds与checkpoint时间窗口定位DeepEP dispatch_layout，after_dispatch也必须位于该窗口；其20维接收计数与trace中两个FC split计数逐项一致。
3. 以本次调用编号关联其余七卡。DeepEP日志iter是调用编号，不直接当训练轮次；不按“每轮固定若干调用”的公式猜测层号。
4. 对160个专家分别核对八发送rank之和等于八接收rank的本地专家计数拼接。两组独立完成后，才相加为16卡stage的工作量。
5. 4032个EP8/层/MB/iter实例、645120项逐专家守恒；非代表rank不重复读全部FC trace，不能把日志核对声称为每卡每轮的FC trace核对。迭代60另与此前两组FC证据896项计数回归。

## 指标

- 专家接收条目数：筛选后token→专家分配量，不是唯一token数。
- 实际参与专家数：收到至少1条输入的专家数。单组上限160；两组合计上限320个专家副本实例，不是320种不同专家。
- 每个点保留具体iter、层、MB；不跨层或MB平均。
- 首末变化比较L3与L58；相邻层下降次数、跨iter相邻变化和曲线相关性另列。端点下降不等于逐层单调，端点上升也不等于跨iter每步上升。
- 所有数据来自筛选后；没有筛选前逐专家直方图及概率，不把全部差额归为容量丢弃。

## 复现与页面重建

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
/usr/bin/python3 -S workflow/data_foundation/extract_layer_iteration_grid_r1.py
/usr/bin/python3 -S workflow/data_foundation/analyze_layer_iteration_grid_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer_workload_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_edp_pair_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer_iteration_grid_r1.py
```

提取器复用已完成stage/组文件，不覆盖原始数据；聚合输出目录存在时拒绝覆盖。页面重建必须最后运行九轮趋势publisher，否则只恢复历史页。浏览器测试需要现有Playwright环境及43312服务镜像更新。

页面④包含三个图：不同iter逐层折线、固定层跨iter折线、层×iter热力图。九轮按钮独立开关，不与①历史三轮开关联动；④的MB/EP8组/指标选择独立于前面章节。此前iter60容量证据保留为折叠内容。

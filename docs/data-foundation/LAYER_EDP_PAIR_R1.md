# 完整stage的两组EP8工作量核对

2026-09-23；工程`/home/zjb/Desktop/worktrees/mfu-16to256`，分支`32to256`。本轮新增第二EDP副本，不修改冻结32→256模型，不提交推送。

## 覆盖与统计口径

固定256卡iter60，中间PP1～14，每stage16卡，四个microbatch、每stage四个逻辑层。stage s的组A为rank16s～16s+7，组B为rank16s+8～16s+15。首尾stage不在本次范围。

每个点是一层、一个microbatch。把两个EP8组的专家实际接收token条目相加，得到整个stage在该层的总量；不是将stage四层相加，也不是去重原始token。组A/B是两个EDP副本中的路由专家组，拥有同一组160专家的不同副本，不应称成320种不同专家。

每组原始top6分配由trace形状推导为8192×6×8=393216；16卡合计786432。各组独立核对发送160维计数与8个接收rank各20维计数，再逐rank对齐trace中的FC输入。日志iter是调用计数，不能当训练轮号。时间戳仅在同rank匹配checkpoint窗口，不跨主机比较时间。

## 曲线如何读

- ②工作量：A、B、A+B三条线；可以检查两组是否都出现下降，不能只看其中一组代表整个stage。
- ②耗时：分别保留每组最忙rank的B专家FC时间，两组并行，禁止相加。组内最大FC时间也不是整层DAG完成时间。
- ②CV：分别描述各EP8组8个rank内部的工作量不均衡，不能把两个组混算后叫组内CV。
- ④：原始分配推导值与两组实际保留总量。差额是推导的未进入计算分配量，不是容量丢弃的直接测量。
- ①保持原始范围：224/256的iter40/60/80、组A lane0代表rank。本次没有把这些三轮代表rank曲线冒充完整stage的跨iter结果。
- PP交界处断线表示逻辑层所在物理卡组切换；不是同一EP8组执行所有层。

## 证据与验收

以下路径相对于工程绝对根，最终状态以报告为准：

- `results/data-foundation/layer-workload-edpB-r1/`：第二组112份trace的逐层事实、GPU成本与原始路径/哈希。
- `results/data-foundation/layer-filtering-edpB-r1/`：第二组发送—接收—FC核对、日志行号、来源SHA256。
- 第一组原始证据`layer-filtering-r1/`不覆盖。
- `results/data-foundation/layer-edp-pair-r1/report.json`：两组对应的224个层/MB点、分别成本、合计工作量及各MB的L3→L58端点对比。
- 两组共448个组/层/MB实例，3584个rank/层/MB实例；合计71680项逐专家发送—接收守恒检查。第二组FC计数还与日志独立逐项相等。

## 复现

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
/usr/bin/python3 -S workflow/data_foundation/extend_edp_pair_r1.py
/usr/bin/python3 -S workflow/data_foundation/aggregate_edp_pair_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer_workload_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_edp_pair_r1.py
```

扩展提取器支持复用已有单rank结果；聚合输出目录首次创建，已存在时拒绝覆盖。页面重建需按上述两个publisher顺序执行，否则基础publisher只恢复单组旧图。原始trace不改不复制，生成物均在本工程results；部署仅更新同项目43312服务的HTML镜像。

## 结论边界

发送—接收—FC全部核对通过（71680项逐专家守恒检查）。MB0端点：

|范围|L3|L58|
|---|---:|---:|
|组A|149900|94043|
|组B|154132|85861|
|16卡合计|304032|179904|

第二组没有补回第一组的减少，两个副本均下降。MB0整stage合计减少124128条（40.83%）；另三个MB的合计端点为319884→185496、312697→186133、305576→179978，降幅分别42.01%、40.47%、41.10%。原始top6合计分配量均为786432。这里仅对比首末点，不能声称相邻每层都下降，也不能据此将整条曲线拟合为固定层深系数。

整stage总量下降若在报告中成立，则排除“只因观察第一组EDP副本”的解释；不代表容量单独解释全部下降。参考代码存在容量筛选及零概率屏蔽，缺少筛选前逐专家分布和概率，不能拆分二者的独立贡献。两组的单轮观察也不能证明跨iter稳定、更不能将实测工作量作为外推时提前已知的输入。

# 逐层专家分配量核对：发送—接收—计算

范围：256卡iter60，PP1～14，每stage第一组EP8、四个microbatch。保留原模型与旧证据，不拟合、不提交推送。不是完整stage的两个EDP副本，也不是跨iter验证。

## 要回答的问题

图中后面的层实际专家输入变少，究竟是卡间转移，还是整组进入计算的分配变少？其中多少能确定为容量筛选？

## 数量口径

每个发送rank的路由输入为8192×160，先做token轴top6。原始分配条目数据此推导为8192×6=49152，EP8合计393216。这个数不是读取的筛选前逐专家直方图，也不是去重token数。各层原始分配总量固定，不代表每个专家分到同样的数量。

逐层、逐microbatch核对以下三项：

1. DeepEP `dispatch_layout.num_tokens_per_expert`：每个发送rank对160专家的筛选后分配计数；8个发送rank按专家索引相加。
2. DeepEP `after_dispatch.num_recv_tokens_per_expert_list`：每个接收rank拥有20专家，完整EP8拼成160个专家计数。
3. trace中专家FC输入的20个split计数；逐项匹配接收日志，而不是只匹配总量。

日志用同rank时间戳匹配原F的checkpoint CPU窗口，再核对计数向量。日志`iter`是dispatch调用计数，不当作训练iter60。没有跨host比较绝对时钟。

不要用`recv_tokens`或`total_dispatch_tokens`代替专家条目总数：传输时同一token给同卡多个专家可以共用输入；传输token与专家分配条目不是同一口径。

## 容量与概率屏蔽的边界

trace各窗口存在top6、专家轴top615、logical_and。参考236B router按capacity_factor执行容量限制；DeepEP metadata转换另以`token_probs == 0`将索引置为−1。

所以393216减去实际专家输入，可以称作**推导的未进入计算分配量**。不能直接称“实测容量丢弃数”：缺少容量筛选前的逐专家分配/概率和屏蔽前后mask，无法排除原先就为0的概率，也不能拆分“超容量”与“零概率屏蔽”的独立贡献。

两个层之间原始分配恒定，保留量减少与未进入计算量增加在数量上相等。这是会计恒等式，不是证明容量限制解释了100%下降。接收端少也不是通信丢包：发送—接收—计算计数守恒意味着减少在dispatch计数之前已经发生。

各层独立重新路由。本层一条专家分配被去掉，不代表原始token在后续层消失。不得把曲线解读成“token沿层被逐渐消耗”，也不得把首末两个点解释为单调下降。

## 结果与复现入口

已独立核对的端点例子（MB0，L3/PP1与L58/PP14，两个完整EP8组）：

|项目|L3|L58|
|---|---:|---:|
|原始top6分配（输入形状推导）|393216|393216|
|实际进入专家（发送/接收/FC三端一致）|149900|94043|
|推导未进入计算的分配量|243316|299173|

首末保留量减少55857条，约37.3%；差额相应增加55857条。确认不是只观察单rank造成，也不是发送到接收途中少了条目。容量与零概率过滤各自贡献仍无法分离；不称容量单独解释37.3%，也不把它对应成1F1B耗时减少37.3%。

另外三个microbatch的L3→L58保留量分别为164098→96614、161530→96494、156925→91215。四个MB都出现首末减少，但这仍不能证明每个相邻层单调减少，或跨iter保持同样规律。

工程绝对根：`/home/zjb/Desktop/worktrees/mfu-16to256`。

- `results/data-foundation/layer-filtering-r1/report.json`：最终核对状态、每层/MB的原始分配、实测保留量、推导差额。
- 同目录`w256-i60-s*-r*.json`：原trace路径/哈希、checkpoint和topk事件索引、日志路径/行号、发送与接收完整向量。
- 同目录`manifest.json`：输入与输出SHA256。
- `workflow/data_foundation/audit_layer_filtering_r1.py`：仅读取既有112份iter60 trace及其DeepEP日志；支持复用本次已落盘的单rank核对文件，聚合重新验证全部160维向量。四线程仅用于本地I/O提取，不启动新agent。

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
/usr/bin/python3 -S workflow/data_foundation/audit_layer_filtering_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer_workload_r1.py
```

最终数量以report.json状态PASS为准；网页④展示选定MB的首末层差额。原图①/②的逻辑层号保留，PP交界断线表示物理卡组发生切换。

## 后续真正缺的证据

若要准确给出“容量丢弃占下降多少”，需同一批层/MB额外记录筛选前每专家计数、容量mask移除数、容量前已为0的top-k概率数、最终有效分配数。现有日志只给后侧计数，不能从后侧直方图反推完整前侧分布。本轮不启动训练或新增采集，不将缺项伪装成已确认因果。

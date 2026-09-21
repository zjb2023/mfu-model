# 32→256预测全景 r1

## 交付范围

整合既有F/B图、256卡16-stage调度、PP成本、PP0启动前段、末B后的更新尾段。未进行新的误差拟合，未修改原预测或旧页面。本页采用lark-apps设计指导的信息层级和图表口径规范；发布严格限定到现有本地服务，没有妙搭资产、Git提交或推送。

入口：http://192.168.8.16:43312/df-v001/prediction-overview-r1/

开篇首先明确整轮误差N/A（首尾未外推），随后显示1F1B原预测+6.164659%。辅助拼接使用目标实测首尾，23.227178秒对21.969834秒，差+5.723050%；明确标为诊断，绝不称独立整轮预测。

数据目录iteration60原始标签为ProfilerStep#59。rank0实测：启动1196.822985ms，首F→末B20396.013187ms，末B→最后AG206.339052ms，最后AG→Step结束170.658364ms。最后AG只是本rank观测完成，不证明全世界优化完成。GPU通信事件和CPU包络分开展示，不能相加或将CPU调用结束当GPU完成。

原F/B页面通过直接链接及延迟加载嵌入保留；其中旧文字可能描述历史阶段，完整迭代覆盖以总览为准。编辑嵌入页面只影响局部演示，不会自动重算本页冻结预测。16-stage图直接来自封存outer-prediction；可筛选microbatch、隐藏PP、点选节点查看成本及前驱。

## 文件与复现

- 源码：`workflow/data_foundation/build_prediction_overview_r1.py`、`prediction_overview_r1.html`。
- 数据/HTML/验证/哈希：`/home/zjb/Desktop/worktrees/mfu-16to256/results/data-foundation/prediction-overview-r1/`。
- 服务副本：`/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r1/`。
- 浏览器验证脚本：`workflow/data_foundation/check_prediction_overview_r1.cjs`。
- 浏览器检查输出：`results/data-foundation/prediction-overview-r1-browser/`，以checks.json实际结果为准。

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/build_prediction_overview_r1.py --out /tmp/mfu-overview-repeat
node workflow/data_foundation/check_prediction_overview_r1.cjs /tmp/mfu-overview-browser-repeat
```

输出目录须不存在。生成仅重读目标rank0一份trace及既有结果；核验原trace SHA和封存输入不变。服务发布使用`--publish 新目录`且拒绝覆盖。本轮旧F/B页面不变，原始数据只读。网页测试依赖旧研究工作区的Playwright及本机Chromium，路径在测试脚本内。

## 后续

本轮只统一资料，不推进PP1/PP2均值或F/B规模拟合。下一轮才讨论1F1B内误差与校准/验证划分；整轮独立预测还缺启动、更新和收尾成本及全rank边界核验。

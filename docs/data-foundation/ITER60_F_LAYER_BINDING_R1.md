# 256卡iter60第一层F：同一DAG成本绑定

2026-10-23，状态：单层条件实测回放PASS；完整256卡细粒度模型仍为PARTIAL。

页面：http://192.168.8.16:43312/df-v001/iter60-f-layer-binding-r1/

## 本次结果

范围为PP1、MB0、rank16–23，第一执行层F。保留冻结32→256模型该层全部260个节点、330条前驱边，模板rank8–15显式映射到物理rank16–23。130个业务节点和130个保留就绪节点均有成本。

- 256实测及条件回放层出口：85.391501904 ms。
- 原32卡源模板同层出口：88.600424981 ms。
- 所有业务节点出口回放误差：0 ms（浮点结果）。
- 负成本/依赖时序冲突：0。

零误差来自同轮实测成本和观测就绪间隔的核账闭合，不是独立预测精度，也不是GPU因果依赖全部得到证明。源目标约3.209 ms差异只属于此层此组，不直接乘PP数作为整轮误差。

## 成本与证据

计算使用与源模型相同的局部设备包络；CP使用两个参与rank前驱就绪到本rank完成的有效区间；EP使用既有共享notify、dispatch、专家包络、共享combine及本地可见combine定义。有效通信成本可能包含等待。FusedCombine可见包络不能仅凭名称证明全为unpermute。

Attention输入重排尾段仍保留独立参数。最初仅通过runtime关联时找不到两个index_select；按原四层F构建器补充driver关联后，八个rank均匹配到两个。不是删除断言或把漏项改为等待。

没有修改原始trace或封存预测结果。提取器`pp32_f_split_r8.py`新增可选metadata、卡数及EP成员参数，默认行为不变。

## 文件和复现

以下相对路径均以 `/home/zjb/Desktop/worktrees/mfu-16to256` 为根：

- `workflow/data_foundation/bind256_f_layer_r1.py`：提取、绑定、调度及页面生成。
- `results/data-foundation/iter60-f-layer-binding-r1/rank-{16..23}.json`：CPU/runtime/driver/GPU及通信组证据。
- 同目录`bindings.json`：模板节点到物理rank、设备事件ID、目标成本和保留间隔。
- 同目录`graph.json`、`replay.json`、`dependency-conflicts.json`、`report.json`、`manifest.json`：可执行图、调度结果、冲突、结论及SHA256。
- 冻结源：`results/data-foundation/deepep-stable-20261016-r1/model/F.json`。
- 原始数据：`/var/local/zjb-data/gbs64/framework_256_gbs_64/2026-07-31-13_40/worker33008/profiler/iteration_60/`，只读取rank16–23。

生成命令：`python3 workflow/data_foundation/bind256_f_layer_r1.py`。完成目录受保护，不覆盖report；后续复现应配置独立输出和发布目录。未完成的同路径缓存只在原始文件SHA及metadata一致时恢复。

## 已执行验证

1. 输入/输出manifest哈希全部通过。
2. 精确前驱保持不变；所有成本非负；130个业务节点的实测出口全部闭合。
3. 共享Combine成本增加1 ms，层出口增加1 ms，确认成本可注入且未被额外包络重复覆盖。
4. 原32卡iter155 rank8使用默认提取入口，与冻结事实JSON完整相等。
5. 浏览器HTTP 200，无脚本错误；参数表130行、八卡图120个本地块。共享节点单列参数表。

## 下一步

先沿相同规则扩展PP1 MB0的其余三层F及另一组EP8（rank24–31），再推进B。必须逐项报告节点身份、跨组完成点、保留间隔及冲突；不按总F/B比例缩放细项。尾部仍需独立的框架/stream/event依赖核对，不能把实测总尾长回填当完整模型验证。

本次没有提交或推送Git。

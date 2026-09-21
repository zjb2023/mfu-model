# 原总览接入逐B位置证据

用户要求将iter40/60/80的逐B位置曲线纳入原1F1B误差分析页。已在原地址的overestimate区内新增证据入口和可展开iframe，不修改F/B模型或参数。

入口：http://192.168.8.16:43312/df-v001/prediction-overview-r2/#b-evidence

补充明确区分256 iter60两个诊断：

- 仅PP1–14中间B改用目标实测、F和PP及依赖不变：20548.685946ms，对20396.013187ms，剩余+152.672759ms / +0.748542%。
- PP0–15全部B改用目标实测、F和PP及依赖不变：20504.716388ms，剩余+108.703201ms / +0.532963%。

每次重跑冻结外层DAG，原版结果重现至1e−6ms，中间B与先前诊断一致。全部B是本次新增诊断，不同于之前“全部F/B”替换。不是删除B或将其成本归零，不是新预测精度，不可外推为iter40/80精度。

代码：`workflow/data_foundation/publish_b_evidence_index_r1.py`、`b_evidence_index_r1.js`。结果：`results/data-foundation/b-evidence-index-r1/`，包含HTML、b-only-diagnosis.json、manifest。发布脚本单次版本保护，不可重复覆盖；原服务页保留为index-before-b-evidence.html，源快照保持不变。

当前页面仍以ProfilerStep口径为主，模型误差6.16%未改。未提交或推送Git。

## 说明修订 UI r2

按用户要求移除该区的JSON入口，改为解释误差贡献：中间B消除1104.671894ms，占原1257.344653ms高估的87.857525%；全部B消除1148.641453ms，占91.354542%。首尾B额外43.969559ms，剩余108.703201ms。比例基于总偏差减少量，不是各B时长误差直接求和。

界面区分“B成本迁移偏差”与“位置波动”：当前替换同时校正整体偏移、stage和microbatch差异，未分离出位置波动的独立贡献。不能把87.9%全部说成波动导致。数据文件继续保留但不要求用户打开JSON。结果版本 `results/data-foundation/b-evidence-index-ui-r2/`；浏览器核验87.9%、91.4%、0.75%、0.53%正确，JSON入口已移除。

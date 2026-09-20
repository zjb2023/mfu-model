# B DAG UI r3：名称与跨迭代验证说明

本次只改展示，图结构与全部默认成本未改。旧ui2页面完整保留于
`results/data-foundation/2111-blocks-ui-r1/df-v001/b-four-layers-r10-ui2-archive/`。

- 专家反向1/2改为FC2输出投影反向、FC1输入投影反向；补充两者之间还有折叠的激活/门控等处理。
- EP8共享完成改为Combine跨卡耦合，明确条件模型，不是共享专家，也不是硬barrier的证明。
- 默认成本标明来自iter60，不是通用常数。
- 增加60→70 B0误差2.79%及逐层对比页入口；明确该指标仅对应未编辑的封存基线，不适用于用户改成本后的状态，也不是全迭代MFU结果。

发布版本 `b-four-layers-r10-ui3`，入口不变：
http://192.168.8.16:43312/df-v001/b-four-layers-r10/

源文件：`workflow/data_foundation/ui_b_four_layers_r10.html`、`ui_b_dag_r2.js`、`publish_b_four_layers_r10.py`。
已验证发布前后graph/replay/summary/tests/coverage逐字节一致；名称与验证入口检查通过。
浏览器DAG回归证据目录：`results/data-foundation/b-four-layers-r10-ui3-check/`。
未提交或推送Git。

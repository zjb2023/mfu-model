# R11：B重计算不再保留整段包络

## 本次实现

以R10四层B图为基线，为源32卡iter60 PP1 B0、ranks8–15的4层重计算建立成本子图。使用重计算自身的原始trace和框架算子归属，不移用F0时长。

- 32个重计算包络替换为零成本出口，原包络成本不再计入。
- 新增Attention计算、本地投影等计算、专家FC1/FC2、CP2有效通信、EP分发同步/本地处理、EP合并有效残余/本地可见处理。
- CP按四个双卡组、每层四次通信耦合；EP按八卡notify及Combine完成条件耦合。
- 本地重排和就绪差值暂作为分项trace参数保留，不再把计算与通信藏在整段B重计算成本中。
- R10真反向节点定义全部保持不变，仅其上游重计算出口换成新的子图。

网页：<http://192.168.8.16:43312/df-v001/b-recompute-r11/>。
旧R10页面保留，并增加新候选入口。全阶段/256预测未晋升新版本。

## 成本接口

图节点`policy`以`external`开头的项允许非负有限毫秒值替换。`cost-bindings.json`列出接口、原值、rank/unit及口径。重计算当前有616个可替换节点（含320个本地计算块），投影图仅显示主要结构，其他节点可从成本列表选择；这不是已完成最少参数优化的版本。

调用接口格式：

```json
{"unit":"ms","basis":"effective_replacement","nodes":{"R0:EP8:combine_effective":2.0}}
```

这表示把所指节点的成本替换成2ms，而不是额外增加2ms。CLI以`--overrides`输入该文件，输出独立的`overridden-replay.json`，默认图和默认回放仍保留。非法单位、负数、NaN和只读节点替换均拒绝。

CP端口是双卡共同设备重叠部分的有效成本，保留各卡完成偏移；旧CP整段成本已经移除，不重复收费。EP dispatch notify、各卡本地处理和Combine有效残余分开，通信端口不是直接等同后端纯service/FCT。外部纯通信成本接入前仍需转换到对应有效口径，不能在原成本上直接叠加。

## 依赖与局限

同流顺序来自trace；CP→重排/Attention、notify→专家、专家/本地准备→Combine为显式条件依赖。保留源trace就绪残余，未恢复所有event handle，不证明物理硬barrier。新图在专家成本变慢后会通过EP8条件耦合影响其他rank，而非固定绝对完成时间。

重计算功能流程与F一致，但成本、事件和每层边界来自当前B中的重计算。FC1/FC2采用框架配对调用顺序推定，未逐权重ID核验。设备分项是有效耗时，不全是纯计算或纯传输。

## 验证结果

- 8份source trace SHA与R10来源一致；覆盖4958个重计算设备事件。
- 旧图所有4375个节点的默认完成点保持一致；默认B868.833358765ms。
- 原32包络已零成本化，原真反向及其余旧节点定义保持不变；图共6687节点。
- 136项扰动检查通过：64个CP共享成本+1ms，4个notify及4个Combine共享成本+1ms，64个专家FC节点+100ms传播到本层八卡Combine和B结束。
- 浏览器四层切换、rank切换、计算/CP/EP成本编辑、重置及移动宽度检查通过，无JS异常。示例FC1+100ms使B增加98.876ms，差额是已有并行余量，不应强求每次B恰增100ms；EP合并有效成本+1ms使B增加1ms。
- **未重新评价iter70或256精度；同样本回放和扰动通过不是独立预测验证。**

## 文件与复现

- 实现：`workflow/data_foundation/build_b_recompute_r11.py`。
- 结果：`results/data-foundation/b-recompute-r11/`，包含graph/replay/evidence/coupling/cost-bindings/tests/summary/manifest。
- 网页生成：`workflow/data_foundation/publish_b_recompute_r11.py`；新增投影控制：`ui_b_recompute_r11.js`。
- 浏览器检查：`workflow/data_foundation/test_b_recompute_r11.cjs`。

```bash
python -B workflow/data_foundation/build_b_recompute_r11.py --out /tmp/b-recompute-r11-repeat
# 使用外部有效成本（输出目录必须不存在）：
python -B workflow/data_foundation/build_b_recompute_r11.py --out /tmp/b-recompute-r11-cost-test --overrides /absolute/path/costs.json
node workflow/data_foundation/test_b_recompute_r11.cjs
```

下一步：按新的重计算/真反向成本端口做源目标专家FC工作量核对与分项误差诊断，再评价跨迭代/256，不能自动晋升或宣称精度改善。本次未提交或推送。

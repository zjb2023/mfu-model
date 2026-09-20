# 单层前向成本模板 r1 / UI r3

日期：2026-09-20。范围：32 卡实例的中间 PP stage，rank 8–15，四个 CP2 组及一个 EP8 组。不是反向模板，不是全迭代模型，也不是已验收的 256 卡外推。

## 入口与版本

- 当前页面：http://192.168.8.16:43312/df-v001/cp-ep8-structure-r1/
- 版本页面：http://192.168.8.16:43312/df-v001/forward-layer-template-r1/
- 上一版归档：http://192.168.8.16:43312/df-v001/cp-ep8-structure-r1-ui-r2-archive/
- 工作区：`/home/zjb/Desktop/worktrees/mfu-w37-v610`，分支 `feat/w37-v610`。本轮没有提交或推送；上述名称是文件版本，不是正式 Git 发布。
- 实现：`workflow/data_foundation/forward_layer_template_r1.py`。
- 页面源及发布器：`workflow/data_foundation/ui_forward_layer_r1.html`、`publish_forward_layer_r1.py`。
- 冻结输出：`results/data-foundation/forward-layer-template-r1/`、`forward-layer-template-r1-four-layer-smoke/`，包含图、成本、调度结果、检查及 SHA256 manifest。

## 依赖与成本接口

单层：Attention/CP → 本地路由/准备 → Notify → Dispatch → 各 rank 专家计算 → Combine → 本地 unpermute → 公共完成边界 → 本地尾段。

四层通过各 rank 上一层尾段连接下一层 Attention。单层 93 节点，四层 369 节点；四层仅是中间 stage 的单个 microbatch 前向模板拼接检查，不等于 F+B。首尾 stage 层数另行配置，B 的反向/重计算规则尚未纳入。

统一规则：节点开始 = 最晚前驱完成；节点结束 = 开始 + 本节点成本。等待由依赖推导，不额外累加所有 rank 的等待。页面显示 rank、CP 组、EP 组泳道，可点击节点查看前驱、就绪等待和成本，切换一层/四层、改参数、导入成本 JSON 和导出结果。

14 个基础参数：A0、A1、A2、A3、CP384、CP256、router、dispatch_prepare、notify、dispatch、expert、combine、unpermute、tail。单位 ms；允许 rank_overrides 覆盖本地节点和 node_overrides 覆盖具体非边界节点，均要求有限非负数。

CP384/CP256 引用 r9 的有效耗时代理 11.277491569519043 / 7.074432373046875 ms，不等于纯网络 service。其余默认数值全部是演示值，未拟合 trace。CP/A 的重叠边沿用条件性结构假设；缺失系统间隙没有建模，不能解释为真实间隙为零。

Notify 是全部参与者就绪之后的开销，不能填入早到 rank 整段 notify kernel 耗时。Dispatch/Combine 是有效残余成本接口，放在共同就绪点之后属于粗粒度近似，并不证明真实 payload 只能在该时刻开始。Combine 公共出口是保守近似，不宣称已证实硬 barrier。

## 复现与验收

在工作区执行，输出目录必须尚不存在：

```bash
python workflow/data_foundation/forward_layer_template_r1.py --layers 1 --out /tmp/mfu-forward-r1-one
python workflow/data_foundation/forward_layer_template_r1.py --layers 4 --out /tmp/mfu-forward-r1-four
node workflow/data_foundation/test_forward_layer_ui_r1.cjs /tmp/mfu-forward-r1-browser.json
```

可以追加 `--costs /absolute/path/costs.json` 注入成本。独立调度检查针对实际输入；敏感性和重叠自测使用标准默认夹具，避免把合法零成本或覆盖配置误判成失败。

默认单层 75.46384788513184 ms、四层 301.85539154052736 ms，仅是演示计算。Python 独立调度、参数扰动、非法输入检查 PASS；浏览器与 Python 共 462 节点逐节点一致，Dispatch +1、rank15 专家 +20、节点查看、非法输入、移动端和 HTTP 检查 PASS。浏览器脚本可直接写入 JSON 测试记录。

本轮未修改 v610、原 224 卡版本及 r6/r9/r10 冻结预测，没有扫描原始 trace。

## 下一步

2026-09-20 展示补充：原 r2 逻辑框图改为当前页面上方默认展示，不再折叠在底部；增加逻辑/成本跳转及完整框图独立入口。沿用既有视觉和模型参数，未变更调度算法或精度结论。

用 32 卡同一层完整 EP8 的可追溯证据替换演示成本，分别登记本地计算、CP 有效成本、EP 就绪及残余成本，验证不重复计费。然后验证四层 F；独立构建 B 后再接 PP 调度。当前仅接口与调度功能 PASS，预测精度尚未验收。

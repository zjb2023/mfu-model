# 256卡iter60：与32→256模型对齐的DAG清单

日期：2026-10-23。状态：源结构清单与页面验证通过；256卡细粒度成本绑定未完成。

## 交付

- 页面：http://192.168.8.16:43312/df-v001/iter60-dag-alignment-r1/
- 生成器：`workflow/data_foundation/build_iter60_alignment_r1.py`
- 模板：`workflow/data_foundation/ui_iter60_alignment_r1.html`
- 结果目录：`results/data-foundation/iter60-dag-alignment-r1/`
- `alignment.json`：逐节点ID、类型、精确前驱、源成本、目标成本插槽与绑定要求。
- `manifest.json`：输入和输出的SHA256。

路径均相对于 `/home/zjb/Desktop/worktrees/mfu-16to256`。

## 共同结构与计费规则

直接读取冻结的 `deepep-stable-20261016-r1/model/F.json`、`B.json` 和 `outer-prediction.json`，不另造简化模型。

F有1106节点、1417依赖边；B有5514节点、6032依赖边。外层709节点的前驱与256同轮候选逐项一致。中间stage沿用四层F及包含重计算、真正MoE反向和Attention反向的B；首尾保留整块接口。插入内部子图时外层F/B壳成本必须置零。

概念图不等于逐节点串行执行，实际调度以清单精确前驱为准。F中的专家计算目前主要由expert_envelope承载，不能宣称已分别绑定FC1、激活、FC2。B的829个无kind节点保留UNSPECIFIED，不根据名称全部归为等待。

## 已有标定与缺口

`calibrate256-iter60-outer-r1`提供128个整块F/B、120条PP消息及尾部事件候选。本清单的6620个F/B内部目标成本全部为null：未绑定不等于零，也不以源值代替。该外层候选训练主体误差约−0.290%，不能当作详细DAG标定完成后的精度。

尾部候选包含570个设备活动及对应观测就绪节点。它依赖观测时序，尚非已证实的stream/event/框架因果图；不将精确重现尾长视为独立验证。

首个绑定单元为256卡iter60、PP1、MB0、rank16–23的一层F：核对框架身份、物理层、CPU到GPU关联、CP/EP组以及成本与间隔边界。然后扩至四层和B。rank24–31另一个EP8组需独立核对。只有完成绑定、检查重叠和重复计费并重新调度，才报告细粒度同轮回放误差。

## 验证与复现

2026-10-23已检查：manifest所有输入/输出哈希一致；所有细粒度目标槽保持空值；HTTP 200；浏览器无脚本异常；F/B、外层、尾部切换及筛选、分页、节点详情可用。检查时43312未监听，已按原静态根目录恢复服务。

生成命令：`python3 workflow/data_foundation/build_iter60_alignment_r1.py`。生成器要求结果目录和发布目录不存在，避免覆盖已冻结结果；当前结果已存在，勿直接重复执行。后续版本应使用新输出目录。

本次没有重新扫描原始trace、调参或重跑预测，没有提交或推送Git。既有32→256预测报告保持不变。

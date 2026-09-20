# 四层 F 同样本回放 r1

UI更新（2026-09-20）：默认第1层rank9，按模块分行显示名称与精确耗时，iter60冻结预测/iter70实测上下排列；可切换层、rank及八卡校准总览。CP/Dispatch只展示目标完成点，因为有效残余与原始通信包络不是同口径。重排尾段未建立目标同口径映射时明确留空，不伪造实测。旧页面保留在`four-layer-f-r1-ui-original/`。默认项、专家10.512/30.391ms展示、节点详情、总览切换、移动端及JS检查PASS，未修改预测参数或结果。

2026-09-20。范围为32卡 iter60、PP1、F0、rank8–15，四个MoE层。沿用单层r3主干的CP就绪、逐卡Dispatch出口、专家包络及Combine有效残余结构，但每层各自提取参数，不把第1层耗时乘4。

## 结果

八卡共同F设备窗口 **342.7138833999634 ms**。320个完成边界同样本回放误差为零；独立递归调度器核对PASS。每层Combine有效成本增加1ms，全F分别增加1ms，不按8卡重复加时。

|层|层尾第二次Add最晚结束 ms|Combine有效成本 ms|
|---|---:|---:|
|1|84.595075|0.969633|
|2|172.353920|1.008874|
|3|257.936022|1.300132|
|4|342.684564|1.077604|

时间均相对于八卡最早F设备事件。层尾结束是累计坐标，不可相加作为总时长。各rank的F起点、终点在summary中独立保存。

## 层间去重

旧LOCAL:6/12/18同时含上一层的两个Add与下一层Attention起始计算。新层A0从首个LayerNorm开始；两个Add留在上一层；仅添加本卡尾部到下一层LayerNorm开始之间的交接成本。没有增加全EP8层间屏障。第一层原r3将下一层入口交接纳入终点，而本报告每层尾端表以第二次Add结束为界，两种口径已区分。

第四层Add后还存在F收尾设备操作，单列为F_epilogue。全F终点是与CPU forward_step内runtime/driver调用关联的最后设备事件；不是CPU包络时长，不含其他PP stage、microbatch、B或完整迭代。

## 参数与证据边界

四层CP2后均检查了两个index_select kernel，保存32份逐rank逐层证据。原始八文件哈希重新核对，支持runtime和driver关联。仍使用有效CP/EP残余、trace交接与专家计算包络；未证明纯service分解或所有隐藏因果关系。所有成本与边界来自同样本，零误差是核算闭合，不是独立预测验证。

计算、CP/EP及尾部Add在cost-bindings中登记外部有效成本替换权限；其余默认trace保留。当前新页面用于四层只读核对，未把单层编辑器假装成四层编辑器。

## 产物

- `workflow/data_foundation/build_four_layer_f_r1.py`：四层构建、证据提取、校验。
- `workflow/data_foundation/publish_four_layer_f_r1.py`：网页生成。
- `results/data-foundation/four-layer-f-r1/`：图、回放、参数权限、320边界、重排及层尾证据、哈希manifest。
- 网页：http://192.168.8.16:43312/df-v001/four-layer-f-r1/

复现命令（输出目录需不存在）：

```bash
python workflow/data_foundation/build_four_layer_f_r1.py --out /tmp/mfu-four-layer-f-new
```

本次未修改r3冻结结果、v610或224卡成果，未提交/推送。下一步冻结这些参数后再评价其他迭代，不能在评价阶段使用目标迭代边界反求成本并报告为预测。

收尾验证：独立输出目录 `/tmp/mfu-four-layer-f-r1-repro-20260920` 的7份结果JSON与正式产物逐字节一致（manifest输出路径不同，不参与字节比较）。浏览器HTTP200、四行边界表、按层切换、节点真实前驱展示、移动端及无JS异常检查PASS；已查看截图。首次交互测试错误地期望专家节点直接依赖Dispatch，实际有显式交接节点；改为按模型真实deps核验后通过，未为迁就测试修改依赖。

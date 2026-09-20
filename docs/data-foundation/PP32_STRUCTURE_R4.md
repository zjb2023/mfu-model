# 32卡最简DAG结构审阅 r4

2026-09-18。用户优先级：先完善32卡DAG与最小参数接口，不继续拟合误差。
原r1/r2/r3b、v610和224卡成果均未修改；本轮没有重新校准、读取256时序、提交或推送Git。

## 页面

http://192.168.8.16:43312/df-v001/pp32-structure-r4/

沿用四卡原理页的色板与图优先布局。首屏按4个PP stage列出全部32rank，默认显示一条链、一个微批的14个节点。
支持点击rank切换8条PP链、选择8个微批、展开32卡及全部微批。点击节点显示共享参数和完整图中的直接前驱。
视图隐藏其他微批不代表删除依赖；横轴为逻辑布局，不是时间轴。

## 真正完成的范围

- 只读取32卡目录迭代60的32份trace，每rank各8个F和8个B；512个CPU F/B标记的顺序全部匹配非交错1F1B。
- 从各rank的distributedInfo读取全局通信组；得到8条PP链，每链4rank，不按文件名猜分组。
- 384条PP消息按方向、链及微批序号配对，显式连接收发就绪和完成；配对是调度身份推导，未声称逐条网络序列号验证。
- 图共1902节点、2860边。额外节点用于无耗时的就绪、完成及明确标记的尾段占位，不增加拟合参数。
- 独立检查：32rank、8链、节点唯一、边端点、无环性、微批覆盖、参数缺失保护PASS。

## 参数接口

17个候选共享槽位，而非证明17是最优解：

- F/B：首、中、尾各一对，共6个。
- PP：前向、反向各一个，共2个。
- 更新尾段：首、中、尾各DP_RS、OPT、DP_AG，共9个待测槽位。

前8项仅引用r3b四rank试点值，全部标记未在全32rank重新校准；后9项为null。
启动和尾段约101/298ms的经验包络已从这版正式参数中移除，作为尚未闭合的入口/末尾边界，不变相填零。
CP/EP内部通信包含在F/B大块中，不能再重复收费。内部多rank同步行为还需验证，不能以共享参数替代因果证明。

## 尾段证据与未完成项

已保存每rank的RS/AG调用、全局组成员、消息元素数量、dtype、优化器注释及相对最后CPU B的位置。
本样本DP-with-CP和专家DP相关RS/AG调用都位于最后CPU B结束之后；这只证明CPU观察顺序，不证明GPU完成或bucket就绪关系。
专家DP组为单rank时仍可能有框架集合通信调用，不能据调用存在认定发生跨卡传输。
图中的stage级DP_RS→OPT→DP_AG为聚合接口草案，44条相关边标记PROPOSED_UNVERIFIED并以虚线呈现；它们不是已经证实的真实组/优化器bucket图。
保留启动、跨stage同步、异步参数收集与跨轮依赖缺口。`predicted_iter_ms=null`，不输出整轮时间或MFU。

## 验证证据

结果：`/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/pp32-structure-r4/`。

- `protocol.json`：32rank、仅迭代60、只做结构、不拟合。
- `rank-*.json`：每rank原始路径/SHA、组成员、调度与尾段事实。
- `graph.json`：完整图、参数及状态。
- `independent-check.json`：独立图与身份检查PASS；不代表GPU全图或尾段PASS。
- `manifest.json`：输入、代码和结果校验和。

浏览器证据：`results/data-foundation/pp32-structure-r4-browser/check.json`。
64种链×微批组合通过；单链单微批14节点，全32卡单微批112节点，全32卡全微批896个F/B或PP节点；未知尾段显示待测；无JS异常、移动端无页面溢出。截图已检查。
页面发布目录带 `ui-manifest.json`，模型结果与页面结果分别记录。

## 复现

```bash
cd /home/zjb/Desktop/worktrees/mfu-w37-v610
.venv/bin/python -B workflow/data_foundation/pp32_structure_r4.py --out /tmp/pp32-structure-NEW
.venv/bin/python -B workflow/data_foundation/check_pp32_structure.py /tmp/pp32-structure-NEW
```

发布脚本 `workflow/data_foundation/publish_pp32_structure.py` 仅支持首次安装，已有目录拒绝覆盖，不能直接重跑。

## 下一步

优先核对9个尾段槽位及44条候选边：从框架与32卡GPU关联确认DP RS、OPT、AG的真实先后与重叠，删掉不适用的通信（例如单rank专家DP），明确哪些参数可共享。
通过后才评价整轮时间；不要用拟合补偿项掩盖缺失依赖。

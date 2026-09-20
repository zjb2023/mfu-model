# 32→256 流水线外推 r6

2026-09-18，工作区 `/home/zjb/Desktop/worktrees/mfu-w37-v610`，分支 `feat/w37-v610`。

## 口径与冻结

只评价 **PP0 首个 F 的关联 GPU 事件开始 → 最后一个 B 的关联 GPU 事件结束**，包含填充、1F1B 调度、排空；所有尾段 RS、优化器、AG 都排除。不是完整 iter/MFU。

用户要求的 PP0 RS 结束辅助窗口尚未评价：旧 RS_PREP 含软件准备，不是精确 RS 长度，不能替代该指标。当前简化依赖依然是本 stage 最后 B 完成才能做本地 RS；不宣称本地 B/RS 重叠。

冻结 32 卡 iteration60 的六个首/中/尾 F/B 成本和两个有效 PP 成本。没有移入 v610 的目标拟合参数，没有用256实测改参数。CPU 调度生成参考236B非交错流水线框架快照；生成的源32各stage F/B及组合P2P调用序列与源观察匹配。

源 PP4、8 microbatch；目标 PP16、4 microbatch。扩展中间stage并重建调度，不做时间线线性放大。目标多数stage没有稳定交替段，仍按同一非交错1F1B调度规则填充/排空。

`prediction-seal.json` 在本轮打开目标时序之前写入。目标固定选择256 iteration60 rank0，不按误差挑样本。历史v610已接触256数据，因此本实验不是全新盲测。

## 结果

| 窗口（rank0） | 预测 ms | 实测 ms | 绝对相对误差 |
|---|---:|---:|---:|
| 32 iter60 校准 | 11970.448837 | 12064.008904 | 0.775530% |
| 32 iter70 开发验证 | 11970.448837 | 12260.027324 | 2.361973% |
| 256 iter60 外推评价 | 21755.132041 | 20396.013187 | 6.663650% |

目标偏高1359.118854 ms。仅一条代表链模型对rank0窗口，不能宣称全256rank预测误差6.66%。归因尚未完成。

## 验证与局限

- 153组PP/microbatch组合的DAG构建与消息配对断言通过。
- 32卡实际观察的调度与生成结构匹配；256 rank0观察顺序及F/B设备包络不重叠检查通过。
- 独立就绪节点消除算法校验源图233、目标图705节点的开始/结束时间，通过；预测和参数封存哈希通过。
- 新目录重跑；parameters、predictions、target-observation、report、check、prediction-seal 六份文件逐字节一致。
- 本地HTTP200；浏览器2张图、144个F/B色块，无页面异常；390px移动端无页面横向溢出；截图已检查。
- **总体 PARTIAL**：GPU归属基于唯一runtime correlation与同进程CPU F/B时间包络，跨线程因果没有完整证明。
- F/B已含CP/EP通信，不重复计费。跨规模张量/框架/拓扑等价仍未完全证实；PP源成本不是已识别的纯网络service time。
- 没有逐stage实测对比，没有全rank统计，没有尾段评价；窗口内DP活动可能带来资源竞争，当前未单独建模。

## 产物与命令

相对本工作区：

- `workflow/data_foundation/pp32_to256_pipeline_r6.py`
- `workflow/data_foundation/publish_pp32_to256_r6.py`
- `results/data-foundation/pp32-to256-pipeline-r6/`：协议、8参数、配置、预测封存、目标边界事实、评价、清单。
- `results/data-foundation/pp32-to256-pipeline-r6-reproduce/`：重跑。
- `results/data-foundation/2111-blocks-ui-r1/df-v001/pp32-to256-pipeline-r6/`：HTML、图、数据副本和独立校验。

```bash
cd /home/zjb/Desktop/worktrees/mfu-w37-v610
.venv/bin/python -B workflow/data_foundation/pp32_to256_pipeline_r6.py --out /tmp/pp32-to256-r6-NEW
```

输出目录须不存在；不扫描全量trace，只读既有源中间事实与目标一份rank0原始trace。目标数据路径及SHA256在target-observation.json中。发布脚本固定正式r6入口，发布目录存在时拒绝覆盖。旧r4/r5、原v610及224卡材料未改，不提交/推送Git。

网页：http://192.168.8.16:43312/df-v001/pp32-to256-pipeline-r6/

网页沿用既有本地视觉规范，按lark-apps图表规范统一时间轴、分离模型图与实测证据，不发布云端。

## 唯一下一步

保持8个参数不变，抽查目标256同一轮的首/中/尾stage F/B耗时与关键边界，判断1.359秒偏差主要来自中间stage成本累计、PP传递还是块间间隔。先做归因，再决定是否增加参数。

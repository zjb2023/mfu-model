# 256卡iter60：当前外层DAG同轮标定候选 r1

2026-10-23，状态PARTIAL。本轮是新增同样本标定/回放，独立目录，不覆盖已封存的32→256预测或历史iter60诊断。不是跨iter预测，未提交/推送。

## 实际完成

1. 使用当前新DeepEP32 iter155–200模型的outer-prediction.json，先重算22119.790913ms基线并逐依赖保留当前外层结构。
2. 重新读取256卡iter60的16个代表rank（0,16,…240）原始trace，核对既有SHA；逐stage四个F、四个B共128个整块GPU包络来自该轮缓存。没有伪造细粒度256成本。
3. 根据本轮各stage的PP框架调用与消息序号配对，沿用当前模型的“两端前序GPU工作就绪后共同有效区间”定义，独立取F/B正值standalone样本中位数。标定F=5.031935ms、B=4.965194ms；不再沿用32卡PP成本。主机时间戳跨机可比性仍是假设，不是纯网络传输标定。
4. 将256 rank0尾段拆为570个实测设备活动节点及显式就绪间隔，并接在末B后；活动成本和间隔逐项在parameters.json中登记。尾段包含RS、梯度处理、优化器、AG、同步、拷贝等，不再只有一个总时长节点。
5. 重新进行外层及尾段节点调度，检查非负成本和所有依赖完成顺序；对照同轮rank0窗口并计算MFU。

## 结果

|范围|回放 ms|实测 ms|有符号时间误差|
|---|---:|---:|---:|
|1F1B|20335.742444|20396.013187|−0.295503%|
|首F GPU开始至ProfilerStep结束|20712.739860|20773.010603|−0.290140%|

主体MFU：回放4.880341%，实测耗时推导4.866182%，相对误差+0.290984%。FLOPs沿用1.293891072e17、256卡、500TFLOP/s/卡；不是新硬件峰值或独立架构校验。单轮结果不称跨轮平均误差。

## 不能省略的限制

- **当前只完成外层标定。** F/B用实测整块包络；内部371429节点的细粒度计算/通信并没有逐项从256卡重建。这是当前细图的外层结构，不是完整细图已标定。
- **尾段局部回放精确是构造保证。** 就绪节点保留同轮观测间隔，事件前驱选取已完成事件；这些不是已经从框架/事件句柄证明的因果边。0ms局部事件误差不能当作优化器模型验证通过。
- 残余约60.271ms仍可能来自PP有效成本汇总、框架间隔、包络、代表组与时钟；不能直接认定纯PP误差。不能把减小到0.29%解读为独立预测精度改善。
- 只处理rank0尾段，其他stage RS/EDP与B重叠、所有通信组同步未闭合。
- 下一步应先用框架与GPU事件/stream证据替换尾段“观测就绪”关系，并建立256 F/B内部参数映射；在此之前不发布“完整DAG已全部由256标定”的结论。

## 文件与复现

- 代码：`/home/zjb/Desktop/worktrees/mfu-16to256/workflow/data_foundation/calibrate256_iter60_outer_r1.py`
- 输出：`/home/zjb/Desktop/worktrees/mfu-16to256/results/data-foundation/calibrate256-iter60-outer-r1/`
- report.json：结果及限制；parameters.json：128块、PP、尾部节点参数；calibrated-graph.json/replay.json：完整候选外层与尾部节点/调度；pp-calibration.json：匹配及排除记录；manifest.json：来源/输出SHA。

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python3 workflow/data_foundation/calibrate256_iter60_outer_r1.py --out /tmp/calibrate256-iter60-outer-new
```

输出目录必须不存在；需要orjson和原始数据路径。扫描范围固定为iter60的16个代表rank，不是全256rank或多iter重扫。

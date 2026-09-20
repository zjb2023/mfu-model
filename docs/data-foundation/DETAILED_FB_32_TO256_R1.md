# 固定细化F/B成本：32→256外推 r1

2026-09-20。工作区 `/home/zjb/Desktop/worktrees/mfu-w37-v610`。

## 结果

- 来源：32卡iter60，PP1四层F模板342.713883ms、B r10模板868.833359ms；首尾角色标量及PP有效成本沿用源32卡。
- 目标配置：PP16、microbatch4；中间stage由2扩为14，每个4层；首尾各2层保留。核对hidden5120、seq8192、MBS2、专家160、TP1、CP2、EP8及重计算配置等对应项一致。
- 256卡iter60、rank0首F关联设备开始→末B关联设备完成：预测21653.357840ms，实测20396.013187ms，偏大1257.344653ms，ARE **6.164659%**。
- PP0各F结束点误差依次+6.22/+7.21/+16.39/+25.86ms；B0–B3结束点误差+1027.21/+1131.22/+1209.37/+1257.34ms。较大偏差在首次反向返回PP0时已出现，不能仅由PP0局部F误差解释；尚未定位其具体stage/成本原因。

## 实际实施及验证

中间14stage×4microbatch×F/B共112个模板实例，292525展开节点。入口接外层依赖，原中间F/B标量节点改为零成本出口，避免重复计费。每个外层节点与等价标量收缩图一致。32卡回归11917.227759ms不变。单个内部专家节点+1000ms的接线压力测试使最终窗口+312.305448ms。

图/参数/策略/预测先落盘封存，再读取目标缓存并重新解析一个原始rank0 trace；SHA与旧缓存一致，八个F/B块边界及窗口复现一致。评价后全部封存哈希一致。目标只进评价器，未反填成本。

## 条件与限制

- 256每stage16卡，EDP2；32每stage8卡，EDP1。本版假设两个EP8副本成本相同，仅展开代表副本，没有描述副本间路由负载不均衡。图中保留的是模板rank/local_ep_rank，不伪造完整256全局rank映射。
- 外层仍为stage级完成与简化PP rendezvous，不是全世界逐rank真实PP重叠。PP有效成本迁移沿用源32，未验证跨拓扑效率变化。
- 全部中间stage/microbatch复用源F0/B0及源就绪保留项；首尾为源角色标量。Loss预留为0。
- 只评价一个目标迭代的rank0设备区间，不等于完整256世界makespan、ProfilerStep或MFU；不包含RS/OPT/AG尾段。
- 目标数据有历史曝光，不是盲测；没有预先设精度阈值，不宣称精度PASS。

状态：装配/评价执行检查PASS，预测适用性 `PARTIAL_CONDITIONAL_256_EXTRAPOLATION`。

## 证据与复现

结果：`results/data-foundation/detailed-fb-32to256-r1/`；graph/replay/bindings、parameters/policy/config-snapshot、sealed-prediction/outer-prediction/seal、target-observation/comparison/checks/report/manifest。manifest登记绝对路径和SHA256，复用旧输入不覆盖。

```bash
python -B workflow/data_foundation/extrapolate_detailed_fb_32to256_r1.py --out /tmp/detailed-fb-32to256-new
python -B workflow/data_foundation/publish_detailed_32to256_r1.py
node workflow/data_foundation/test_detailed_32to256_r1.cjs results/data-foundation/detailed-fb-32to256-ui-check-new
```

输出目录必须不存在；发布器使用已保存结果，已发布时拒绝覆盖。
网页：http://192.168.8.16:43312/df-v001/detailed-fb-32to256-r1/
浏览器检查PASS（HTTP200、128预测块、16预测/实测条块、点击详情、正宽度、移动端无整页溢出、无JS错误）；截图已检查。证据 `results/data-foundation/detailed-fb-32to256-ui-check/`。登记的模型输入/代码/输出和UI哈希复核通过。

唯一下一步：保持本版封存，选少量目标stage的首F/首B边界作评价侧定位，查清PP0首B返回已偏大约1.03秒主要积累于前向传播、反向计算还是PP连接；不通过改源参数消除差值、不拟合token。

未修改原始数据、旧冻结预测或旧v610/224；没有Git提交或推送。

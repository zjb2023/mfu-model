# B CP 调用身份核对 r1

2026-09-20，阶段1状态 PARTIAL。保留 v610、224版本、F冻结参数与现有1F1B预测。

## 已完成的证据核对

范围为32卡 iter60 PP1 B0，rank8–15，每卡四个CheckpointFunctionBackward执行单元。读取已登记的八个trace并核对SHA256，不发现/扫描全量目录、不使用iter70。

- 64组阶段签名一致；共288个CP设备事件，其中128个重计算、160个真正反向。
- 重计算每单元输入张量MiB：384、384、256、256。
- 真正反向每单元输入张量MiB：256、256、384、384、256。
- 每个设备事件均通过已有runtime/driver关联回到CPU launch，再以同pid/tid区间包含验证唯一Attention父函数。
- 重计算父函数为`AttnFuncWithCPAndQKVOA2A`；反向为`AttnFuncWithCPAndQKVOA2ABackward`。均为CP2 all_to_all，BF16；不是EP误分类。

消息量为输入元素数×2，不能直接当链路字节数。设备时间是可含等待的实测包络，不是后端纯service成本。trace组成员字段原样保留，不把其中的局部编号擅自当全局rank。

## 尚未确认及模型处理

236B的`Megatron-LM/megatron/core/extensions/transformer_engine.py:789`继承TE DotProductAttention，872附近把CP group、stream及comm_type交给TE。已检查的236B源码树未找到上述实际Attention autograd函数的定义。因此缺的是采集环境安装版Transformer Engine的实现及版本，而不是五次通信是否存在。

不能仅根据尺寸把五次命名为O/dO/dQ/dK/dV，也不能肯定用户所说的“CP O两次”具体是哪两次。本轮用RC0–RC3、BC0–BC4作阶段内顺序身份，消息量相同仍保留独立节点。它们不是F与B一一对应的算子名称。

现有框架`fused_a2a.py`支持MoE反向语义：FusedCombineBackward执行dispatch；FusedDispatchBackward执行combine。主页面新增B功能顺序示意，不把该顺序解释成完整GPU同步图。

未完成：真实layer编号、Attention反向各张量身份、跨流event依赖和跨卡CP/EP就绪边。未将新B块接入1F1B，也未重算或宣称改善2.37%混合模型误差。

## 文件、复现与下一步

工作区：`/home/zjb/Desktop/worktrees/mfu-w37-v610`。

```bash
cd /home/zjb/Desktop/worktrees/mfu-w37-v610
python workflow/data_foundation/audit_b_cp_identity_r1.py --out /tmp/b-cp-identity-new
```

输出必须不存在；包含events.json、signatures.json、summary.json、manifest.json、静态证据HTML。manifest记录原始数据、脚本、框架入口、输入缓存及输出哈希。

主图：http://192.168.8.16:43312/df-v001/cp-ep8-structure-r1/#b-semantics

事件证据：http://192.168.8.16:43312/df-v001/b-cp-identity-r1/

唯一下一步：定位采集环境安装版`AttnFuncWithCPAndQKVOA2A`实现（Python文件、包版本/commit）；对照上述事件给BC0–BC4确认张量语义，再检查通信→计算的跨流等待边。未取得源码时可以继续event依赖审计，但不能把推测标为已确认。

## UI验收

`node workflow/data_foundation/test_b_cp_identity_r1.cjs results/data-foundation/b-cp-identity-r1-ui-check` 已执行PASS：9个CP槽位、288条证据行、8个rank、原F逻辑iframe保留、Combine +1 ms及重置回归、390px移动端无整页横向溢出、无JS错误。检查与截图分别见该目录`checks.json`和`b-logic.png`。页面沿用lark-apps设计技能的既有样式/可读性检查；按本工程约束仅本地静态发布，未进行云发布或Git提交。

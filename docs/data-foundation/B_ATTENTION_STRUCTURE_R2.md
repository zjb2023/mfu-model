# B Attention结构 r2：使用236B代码和32卡已有采集继续推进

2026-09-20。修正r1的推进方式：不再把取得安装版TE源码作为继续建模的前置条件。用户指定以236B作为框架依据；trace负责实际调用、张量形状、分组和设备时间证据。包级版本缺口仅限制精确实现身份声明。

## 已验证事实

范围：iter60、PP1 B0、rank8–15、每卡四个checkpoint执行单元，共32个单元。不读取iter70、不改预测参数。

32个单元全部通过CPU提交顺序、消息尺寸、Attention父函数与GPU先后边界检查：

```text
BC0 256MiB → BC1 256MiB → Flash Attention反向计算
                         → BC2 384MiB → BC3 384MiB → BC4 256MiB
```

这里只省略本地重排和其他算子，并不声称CPU/GPU无间隙或所有rank存在统一barrier。两次前置CP在关联的Flash反向kernel之前完成；后三次CP在该Flash调用关联的kernel全部完成后开始，32/32均成立（检查容差1μs）。时间顺序兼容依赖，但不替代显式跨流event证据。

trace记录了`record_shapes=1`、通信输入形状、CPU autograd父函数、runtime/driver关联、设备stream以及distributedInfo全局通信组。用`pg_name`与顶层组配置关联，解析真实CP组，不再使用设备事件中容易混淆的局部[0,1]作为全局rank。

## 框架、配置与张量形状相互印证

worker34031训练日志包含：128 attention heads、qk_head_dim=128、qk_pos_emb_head_dim=64、v_head_dim=128、CP2、cp_comm_type=a2a、recompute_num_layers=4。

236B的`megatron-lm-musa-patch/musa_patch/recomupte_variance/multi_latent_attention.py`中，Q/K每头128+64=192维，V为128维；147–170及198–216行对应投影、拆分与旋转位置分量处理。此为框架解释依据，不冒充本次训练确定执行了这个补丁分支。

trace Flash反向输入形状包含[2,64,8192,192]和[2,64,8192,128]；BF16分别是384MiB与256MiB。前两次CP形状[2,2,4096,64,128]，后三次分别[2,2,2,2048,64,192]、同形状、[2,2,2,2048,64,128]；逐项元素乘积均匹配通信输入字节数。

## 结构解释与成本入口

结合Attention反向结构，前两次可解释为O/dO输入重分布候选，后两项192维通信为dQ/dK候选，最后128维为dV候选。这是代码+形状+调用位置支持的结构推断，不是张量地址追踪或精确源码逐语句确认。O与dO谁先、dQ与dK谁先尚不明确；不强行给BC序号命名。

现在已能保留五个独立CP有效成本槽位，并把Attention反向计算置于前二/后三组之间。相同尺寸不合并事件，也不代表必须使用不同的带宽模型；未来可共享同一成本函数，但分别评估每次就绪时刻。

下一步直接用已有runtime/driver、stream wait及设备顺序核对两组CP与Attention计算之间的本地重排/等待块，构建保留trace残余的可执行B子图；无需先向用户索取新采集。跨卡就绪边尚未闭合前，不将时间邻接当硬屏障、不替换整段1F1B旧B模型。

## 产物与复现

工作区 `/home/zjb/Desktop/worktrees/mfu-w37-v610`。

```bash
python workflow/data_foundation/audit_b_attention_structure_r2.py --out /tmp/b-attention-structure-r2-new
```

输出目录必须不存在。已运行结果：`results/data-foundation/b-attention-structure-r2/{evidence,summary,manifest}.json`。manifest登记八个原始trace、训练日志、236B参考代码、输入证据及脚本SHA256。状态为trace结构检查PASS、完整DAG仍PARTIAL。r1及所有预测冻结结果保留；本轮无新预测精度结论。

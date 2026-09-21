# 阶段冻结：32→256的1F1B误差分析 r1

本阶段资料闭合，物理原因未闭合。256卡iter60：中间B成本差异解释原1.257345秒高估的87.8575%；全部B解释91.3545%。仅中间B/全部B替换后残余为+0.748542%/+0.532963%。这是使用目标实测的诊断，不是新模型独立预测精度。原1F1B预测仍+6.164659%，未拟合k。

逐B位置证据冻结为224/256、iter40/60/80。iter100按用户要求排除当前展示但历史结果不删；不将总体成本差异的解释量等同于位置波动的独立贡献。

## 本地网页完整入口

`results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2/index.html`

包括总览、逐B曲线、F/B成本图、源32及目标256评价和页面引用数据。已有11个页面目录与线上内容逐文件一致；新增总览和曲线目录。可用以下命令独立浏览，无需运行旧工程：

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -m http.server 43313 --bind 0.0.0.0 --directory results/data-foundation/2111-blocks-ui-r1
```

访问 `/df-v001/prediction-overview-r2/#b-evidence`。43312现有服务不切换。页面可离线本地查看；历史provenance内绝对路径保留原来源，不冒充已迁移原始trace。归档清单 `ERROR_ANALYSIS_RELEASE_R1.json` 提供仓库相对路径、SHA和已检查的本地索引链接。

模型与实测的边界：以ProfilerStep为主指标、Training Step仅附录；整轮0.40%误差有启动低估和1F1B高估抵消，不代表内部成本准确。MFU仍继承历史有效FLOPs及峰值口径。

原始trace、训练原始日志不入Git。保留分析脚本、结构化派生结果、说明和网页。只本地提交，不推送/合并。

## 后续主线：优化器尾段

将32卡源末B后296.626ms、256卡目标376.997ms的尾段，拆成可追溯的梯度RS、梯度处理/优化器计算、参数AG和末尾同步。先对源/目标rank0做相同GPU归属口径，明确CPU提交与GPU执行的区别，不把各stage已重叠的RS全部追加到末尾。当前尾段倍率1为经验假设，DP4→8和EDP1→2的影响待核对。

# B位置曲线 r2 · iter40/60/80

按用户要求，从当前曲线及重复位置统计中移除iter100，替换为iter40。100的排除是用户指定的尾部样本选择，不声称本轮已证明尾部效应的原因。

224、256两种规模分别保留逐B数据，不取microbatch均值。读取26份iter40代表rank原始trace；复用60/80共52份记录，合计276个B点。源模型与预测参数不改。

重新统计后，224的PP5→6在三轮全部B中仍下降，PP11→12仍回升；256依然没有三轮全部B共同满足的逐stage递减规律。以网页单个B完整曲线为判断依据，不用整体均值替代。

原地址已更新：http://192.168.8.16:43312/df-v001/b-position-curves-r1/

历史源结果 `results/data-foundation/b-position-curves-r1/` 不变；新版本 `results/data-foundation/b-position-curves-r2/`。服务目录的旧HTML、数据和manifest分别留存为 `index-60-80-100.html`、`data-60-80-100.json`、`raw-provenance-60-80-100.json`、`manifest-60-80-100.json`；旧HTML中的数据内嵌，历史曲线仍可查看。

复现：

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/build_b_position_curves_r2.py --out /tmp/b-position-r2-repeat
```

输出目录须不存在；发布参数为一次性版本保护，不可重复覆盖。原始路径、resolved_path及SHA保存于raw-provenance.json。未提交/推送或删除历史数据。

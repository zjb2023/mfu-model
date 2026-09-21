# 逐microbatch B的PP位置曲线 r1

2026-09-21。按用户要求，不用stage内B均值判断位置趋势。224、256分别检查iter60/80/100，所有中间stage的lane0代表rank。每点是单独的B GPU包络，不混CPU，不合并microbatch。

页面：http://192.168.8.16:43312/df-v001/b-position-curves-r1/

可选规模、B编号、原始ms或相对该轮PP1的差值；每张图叠加三轮，另有全部B小图与逐点表。相对视图只减PP1，不拟合或缩放。PP编号递增与反向传播方向相反，位置趋势不能直接解释为执行先后导致的加速。

## 结果

224：PP5→PP6在3轮×3个B中全部下降，每个单独差值介于−175.552至−68.725ms；PP11→PP12全部回升，每个差值+94.567至+214.656ms。支持重复的分段位置差异，不支持简单单调变快。没有判定该分段的代码/层/硬件原因。

256：没有三轮、全部B共同满足的相邻下降位置。各B首尾差值方向也会变化。所有21条（224九条、256十二条）曲线都不是严格单调递减；这不排除局部下降或局部趋势，只是不支持统一的逐stage递减规律。

| 规模 | iter | B0末−首 ms | B1末−首 ms | B2末−首 ms | B3末−首 ms |
|---|---|---:|---:|---:|---:|
| 224 | 60 | −48.4 | −100.4 | +5.7 | 不适用 |
| 224 | 80 | −40.4 | −35.9 | −42.2 | 不适用 |
| 224 | 100 | +65.3 | +49.0 | −31.0 | 不适用 |
| 256 | 60 | +2.9 | −27.7 | −7.1 | −3.2 |
| 256 | 80 | −7.4 | +26.9 | +29.2 | −9.3 |
| 256 | 100 | +0.1 | +18.6 | −4.4 | −14.8 |

首尾仅辅助查数，不能取代完整曲线。尚未检查全部rank或所有迭代，不能称普适规律。

## 证据与检查

- 78份rank/iter记录，276个独立B点；复用23份既有GPU记录，新读取55份原始trace。
- GPU归属方法与现模型一致：唯一runtime correlation、同进程CPU B包络包含，首末设备事件形成包络。该归属仍是条件方法，不证明所有跨线程因果。
- 新抽取CPU持续时间与历史逐B缓存一致，容差0.001ms；每stage的B数检查通过。
- 原始路径、resolved_path与SHA记录在raw-provenance.json；既有缓存不伪称本轮全量重验raw。
- 模型图、预测参数及原始数据不改，不拟合k，不提交或推送。

目录：`/home/zjb/Desktop/worktrees/mfu-16to256/results/data-foundation/b-position-curves-r1/`。

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/build_b_position_curves_r1.py --out /tmp/b-position-repeat
node workflow/data_foundation/check_b_position_curves_r1.cjs
```

输出目录须不存在；不带publish只生成本地结果。浏览器检查记录 `results/data-foundation/b-position-curves-r1-browser.json`：规模/B切换、相对视图、点选、手机端及JS错误检查。

后续若要建模位置因素，优先核对224的PP5/6、PP11/12分界处配置/层身份和负载差异，不能直接套用统一线性递减系数。

# 16to256 分支：当前 F/B 模型冻结

分支名按用户要求为 `16to256`；本次实际研究基线是 **32卡→256卡**，不是16卡数据外推。
源工作区 `mfu-w37-v610` 的未跟踪研究成果已按明确清单复制；旧目录、暂存区、分支、服务保持不变。

## 冻结内容

- CP2/EP8四层F模板，四层B r10（包含重计算、FC2/FC1反向、Combine条件耦合、Attention/CP及保留项）。
- 最后PP的Loss成本入口，默认0；token工作量驱动的FC1/FC2候选仅记录、暂缓。
- 32卡PP4/m8细化模块装配及256卡PP16/m4、中间2→14stage的固定成本外推。
- 模板全部参数、已封存预测/报告、相关生成代码、可离线打开的HTML及既有检查结果。

结果：32卡iter70区间ARE2.796075%；256卡iter60 rank0区间预测21653.357840ms、实测20396.013187ms，偏大6.164659%。不是完整MFU或全世界逐rank完成时间。EDP2两个EP8副本同成本、stage级PP连接、源就绪残余迁移仍是条件假设。

## 推荐复现（不依赖原工作区或原始trace）

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/verify_fb_snapshot.py --out /tmp/mfu-fb-freeze-check-new.json
```

输出路径必须不存在。检查已复制文件哈希、重建32/256展开图和回放，逐字节SHA256与原封存展开产物比较。大型展开图/回放不入Git，可从模板完全重建。

本次实际离线检查PASS：230份快照文件哈希一致；32卡83617节点、256卡292525节点的图和回放SHA256全部匹配原封存。证据：`results/data-foundation/fb-branch-verification-r1/checks.json`。没有读取旧工作区或原始trace。

从头重新提取trace不属于离线模板复现：相关历史生成器可能需要未纳入本分支的上游中间产物和原始数据。重新执行256评价器会只读原始rank0 trace；原路径和SHA留在manifest中。这里没有移动、复制或删除原始trace。

## 本地网页

```bash
python -m http.server 43313 --bind 0.0.0.0 --directory results/data-foundation/2111-blocks-ui-r1
```

上面是可选启动命令，本次不切换现有43312服务。入口：

- `/df-v001/cp-ep8-structure-r1/`：F逻辑图和成本入口。
- `/df-v001/b-four-layers-r10/`：B依赖图和成本入口。
- `/df-v001/b-r10-60to70-r3/`：B跨iter验证与最后层诊断。
- `/df-v001/pp32-detailed-fb-r1/`：32卡装配。
- `/df-v001/detailed-fb-32to256-r1/`：256卡预测与实测对照。

历史文档及manifest中的绝对路径保留原样，属于原始provenance，不是分支本地查找入口；本地文件位置以 `FB_BRANCH_SNAPSHOT.json` 的relative_path为准。旧HTML中通往未归档研究页的链接可能需要原43312服务；上述主要页面已保存。

## 范围与下一步

本次只做本地冻结提交，不推送、不合并，不改weekly-todo或其他工程。基于源提交688740e，分支冻结提交以 `git log -1` 为准。
下一研究步骤仍为评价侧定位256卡PP0首次B返回约1.03秒偏差的累积位置，不用目标时间调参，不启动token拟合。

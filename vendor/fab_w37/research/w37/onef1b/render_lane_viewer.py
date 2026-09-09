"""Presentation-only upgrade of the frozen v687 lane0 explorer."""
import json,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'docs/w37/1f1b/post685/delivery/index.html'
OUT=ROOT/'results/w37/A/research-html-20260907'
s=OLD.read_text(); data=json.loads(re.search(r'const rows=(.*?);const stage=',s,re.S).group(1))
assert len(data)==84 and all(len(r['observed'])==4 for r in data)
template=(Path(__file__).parent/'lane_viewer.html').read_text()
(OUT/'lane-viewer.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False)))
(OUT/'lane-viewer-provenance.json').write_text(json.dumps({'source':str(OLD),'sha256':hashlib.sha256(OLD.read_bytes()).hexdigest(),'rows':84,'observations':336,'model':'v687_split_full','new_prediction':False},indent=2))

# Upgrade the original gallery index in a new output; preserve sealed historical bytes.
viewer=template.replace('__DATA__',json.dumps(data,ensure_ascii=False))
style=re.search(r'<style>.*?</style>',viewer,re.S).group(0)
sections=viewer[viewer.index('<section>'):viewer.index('<script>')]
script=re.search(r'<script>.*?</script>',viewer,re.S).group(0)
combined=re.sub(r'<script>.*?</script>','',s,re.S)
a=combined.index('<h2>同 rank');b=combined.index('<figure>',a)
combined=combined[:a]+style+'<p><b>交互查看器：v687_split_full · lane0 · 四轮开发观测；选项只筛选封存结果。</b></p>'+sections+'<h2>研究图册与证据</h2>'+combined[b:]+script
combined=combined.replace('<meta charset="utf-8">','<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',1)
combined=combined.replace('训练调度推导的 warmup / steady / cooldown。局部顺序轴不是全局时间。', 'PP14 / MB3 完整 1F1B 区间：颜色仅区分 F / B，横轴为本地动作顺序，非全局时间。')
combined=combined.replace('目标四轮同口径开发回归。MFU 使用继承 FLOPs/peak 和 training 时钟。', '三图依次比较 Profiler 区间时间、完整 iter 时间和由完整 iter 时间换算的 MFU。Training 预测 = Profiler 预测 + 1.372 秒 outer；outer 来自 256 卡九轮 log 与 Profiler 时间差的中位数，v685 沿用旧值。误差率不是开销占比。<a href="/docs/w37/1f1b/OUTER_PROVENANCE.md">outer 来源</a>')
combined=combined.replace('lane0 事后反事实分账，不是可部署预测、全 rank 误差贡献或盲测。', '四轮平均实际 lane0 1F1B 为 21.522 秒，原预测为 18.101 秒，少算约 3.420 秒；观测替换后为 21.471 秒。柱子分解的是少算的差额，不是总耗时。剩余 51.1 ms 为有符号平均差额，不代表每轮误差。这张图检查哪类时间少算或多算：将 F/B 内某类预测耗时换为观测耗时，再重算 lane0 的整体 1F1B 时间。替换不是清零，例如该类预测 50 ms、观测 150 ms，就给原 F/B 节点增加差额 100 ms。无可见 GPU 活动与仅通信两项影响最大，但尚不能确定物理原因；柱子不是可节省的时间，也不是全部 224 卡的误差贡献。')
combined=combined.replace('源侧 PP 就绪机制：观测发布时刻只用于局部检验，自由运行使用模型预测发布。', '源侧 256 卡前向 / 反向通信：分别用 85/90 参数预测 95/100 在双方 API 进入后的剩余时间。前向就绪延迟约 0.051 ms，反向约 93.037 ms。这里使用实际入口时刻作条件；纵轴观测为首个 API 返回给出的完成上界，不是独立 GPU 完成测量。<a href="/results/w37/A/research-html-20260907/pp_direction_local_metrics.csv">前向 / 反向局部误差表</a>')
combined=combined.replace('独立候选依赖合同；潜在就绪尚未细分为 CPU/GPU/transport 内部动作。', '前向近似为较晚的 API 入口 + 5.03 ms；反向为 max(发送入口 + 93.04 ms, 接收入口) + 5.25 ms。两者共用消息完成与 API 返回条件，参数为源侧拟合代理。')
titles={
    'pp_readiness_evidence':'源侧 256 卡：前向激活与反向梯度通信的观测 / 局部预测',
    'causal_contract':'v687 前向 / 反向通信模型：两端入口、发送就绪与调用返回条件',
    'schedule_regions':'目标 PP14 / MB3：每个流水线阶段的前向 / 反向执行顺序',
    'timeline_overview':'目标 224 卡 lane0：预测与第 95 轮观测时间线',
    'iteration_regression':'目标四轮开发误差：Profiler 时间、完整 iter 时间与 MFU',
    'internal_attribution':'目标 lane0：总时长低估及内部耗时替换诊断',
}
for name,title in titles.items():
    combined=combined.replace('<figure><img src="'+name+'.svg">','<figure><h3>'+title+'</h3><img alt="'+title+'" src="'+name+'.svg">')
(OUT/'post685-index.html').write_text(combined)

"""Add readable definitions to frozen charts without changing plotted values."""
from pathlib import Path
import html
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/w37/A/research-html-20260907'
OLD=ROOT/'docs/w37/1f1b/post685/delivery'

def render(name,title,above,below,plot_height):
    y=100+len(above)*32
    h=y+plot_height+50+len(below)*32
    a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="{h}" viewBox="0 0 1560 {h}"><rect width="1560" height="{h}" fill="#f4f7fb"/>']
    def text(y,s,size=22):a.append(f'<text x="40" y="{y}" font-family="sans-serif" font-size="{size}" fill="#203448">{html.escape(s)}</text>')
    text(45,title,28)
    for i,s in enumerate(above):text(92+i*32,s)
    svg=ET.fromstring((OLD/(name+'.svg')).read_text())
    svg.set('x','40');svg.set('y',str(y));svg.set('width','1480');svg.set('height',str(plot_height))
    a.append(ET.tostring(svg,encoding='unicode'))
    for i,s in enumerate(below):text(y+plot_height+30+i*32,s,21)
    a.append('</svg>');(OUT/(name+'.svg')).write_text(''.join(a),encoding='utf-8')

render('iteration_regression','v685–v688：同一模型的三种误差口径（目标 224 卡，四轮开发评估）',[
'Profiler：预测的 Profiler 区间时间，与 trace 中对应 iter 区间时间比较。',
'Training：预测 Profiler 时间 + outer，与训练 log 中完整 iter 时间比较。',
'MFU：分别用预测、实际 Training 时间换算 MFU，再比较二者；不是独立测得的 GPU 忙碌率。'],[
'纵轴 APE = |预测值 − 参考值| / 参考值 × 100%；表示预测误差，不是额外开销占比。',
'outer = 1372.453390 ms：源侧 256 卡九轮（60、65、…、100）的 log − Profiler 时间差的中位数。',
'该值最初由 v5.4 计算，v6.8.5 从 v6.8.4 继承，当前直接用于 224 卡；本轮没有重新拟合。',
'outer 未拆解为具体操作，不能称为 profiler 采集开销；MFU 使用继承的 FLOPs / 峰值常数。'],410)
# Reuse sealed numerical results; simplify the explanation and labels.
import csv
from statistics import mean
run = ROOT / 'results/w37/A/post685-candidate-scenarios-r1/evaluator_only'
with (run/'diagnostic_shapley_components.csv').open() as f:
    components = list(csv.DictReader(f))
with (run/'diagnostic_lane0_error_ledger.csv').open() as f:
    ledger = list(csv.DictReader(f))
keys = ['phase_gpu_idle_ms', 'phase_communication_only_ms',
        'phase_noncommunication_only_ms', 'phase_comm_noncommunication_overlap_ms']
values = [mean(float(r['attributed_delta_ms']) for r in components if r['component']==k) for k in keys]
values.append(mean(float(r['remaining_schedule_pp_initial_skew_error_ms']) for r in ledger))
labels = ['没有记录到 GPU 活动的区间', '只有通信活动的区间',
          '只有非通信活动的区间', '通信与非通信活动重叠的区间', '替换后仍未解释的差额']
colors = ['#bd7e2b','#397cae','#188977','#729d81','#8b8692']
a = ['<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="1620" viewBox="0 0 1560 1620" role="img" aria-labelledby="title desc"><title id="title">lane0 误差诊断：哪类时间的预测偏差影响最大</title><desc id="desc">用观测分类耗时替换预测分类耗时，重算局部图。替换不是清零，柱子不是可节省时间。</desc><rect width="1560" height="1620" fill="#f4f7fb"/>']
def text(x,y,s,size=21,bold=False):
    weight=' font-weight="bold"' if bold else ''
    a.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="#203448"{weight}>{html.escape(s)}</text>')
def panel(y,h):
    a.append(f'<rect x="40" y="{y}" width="1480" height="{h}" rx="12" fill="white" stroke="#d6e1e9"/>')
text(40,46,'lane0 误差诊断：哪类时间的预测偏差，对整体低估影响最大？',29,True)
text(40,85,'范围：v687 局部图；lane0 每个 PP stage 一个 rank，共 14 个 rank；目标 85 / 90 / 95 / 100 四轮。')
actual = mean(float(r['actual_lane0_onef1b_ms']) for r in ledger)
predicted = mean(float(r['predicted_lane0_onef1b_ms']) for r in ledger)
replaced = mean(float(r['all_observed_phase_components_counterfactual_ms']) for r in ledger)
residuals = [float(r['remaining_schedule_pp_initial_skew_error_ms']) for r in ledger]
assert abs(actual-predicted-sum(values)) < 1e-6
text(40,128,'先看总时长：下面是四轮平均的 lane0 1F1B 区间，不是完整 Training iter，也不是各 GPU 耗时之和。',21,True)
for x,label,value,color in [(40,'实际观测总时长',actual,'#e0f1ec'),(540,'原模型预测总时长',predicted,'#e2eefb'),(1040,'观测替换后的模型总时长',replaced,'#fff0d6')]:
    a.append(f'<rect x="{x}" y="150" width="480" height="120" rx="12" fill="{color}" stroke="#c2d2de"/>')
    text(x+22,188,label,23,True)
    text(x+22,238,f'{value/1000:.3f} 秒',34,True)
text(40,309,f'原模型少算：{actual/1000:.3f} − {predicted/1000:.3f} ≈ {(actual-predicted)/1000:.3f} 秒。下方柱子分解的是这个差额，不是总耗时。',23,True)
text(40,347,f'四类观测替换使模型增加 {(replaced-predicted)/1000:.3f} 秒；替换后平均仍差 {actual-replaced:.1f} ms。原来的 21 / 17 秒仅为示意。',21)
text(40,382,'51.1 ms 是有符号平均差额，存在正负抵消；各轮差额（实际 − 替换后）如下：',21)
text(40,418,'；'.join(f"{r['iteration']} 轮 {v:+.1f} ms" for r,v in zip(ledger,residuals))+f'。平均绝对差额为 {mean(abs(v) for v in residuals):.1f} ms。',21)
a.append('<g transform="translate(0 350)">')
panel(110,144)
text(60,145,'怎么做：保留图的依赖，把 F/B 内某类预测耗时换成该类观测耗时，然后重算整体 1F1B 时间。',22,True)
text(60,182,'替换后的 F/B 节点耗时 = 原预测耗时 +（该类观测耗时 − 该类预测耗时）',23)
text(60,220,'“替换”不是把预测值清零，也不是在原预测上再加一遍观测值；PP 消息节点不随这一项一起替换。')
text(40,296,'柱子怎么读：向右 → 整体预测变长，解释部分低估；向左 → 整体预测变短，抵消部分低估。',22,True)
text(40,330,'前四项按全部 16 种替换组合分配影响（Shapley）；单位 ms，为四轮平均，不是单独替换一次的结果。',20)
zero=660
scale=.31
for tick in [-500,0,500,1000,1500,2000,2500]:
    x=zero+tick*scale
    a.append(f'<path d="M{x} 355 V755" stroke="#d0dbe4"/>')
    text(x-22,784,str(tick),17)
for i,(label,v,color) in enumerate(zip(labels,values,colors)):
    y=370+i*75
    text(60,y+30,label,21)
    left=zero+min(v,0)*scale
    a.append(f'<rect x="{left}" y="{y}" width="{abs(v)*scale}" height="44" rx="4" fill="{color}"/>')
    text(zero+max(v,0)*scale+12,y+30,f'{v:+.1f} ms',21,True)
text(500,823,'对 lane0 整体 1F1B 时间低估的分配影响（ms）',20)
panel(852,174)
text(60,889,'“没有记录到 GPU 活动”怎样得到观测值？',23,True)
text(60,925,'该类观测时长 = CPU 标记的 F/B 区间时长 − 区间内已记录 GPU 活动的并集时长。')
text(60,961,'例子（示意）：F/B 实际 500 ms，GPU 活动并集 350 ms，则该类观测值为 150 ms。')
text(60,997,'若原预测该类为 50 ms，就替换为 150 ms：F/B 节点增加 100 ms；整体增加多少还要重算依赖。')
text(40,1069,'主要发现：无可见 GPU 活动区间与仅通信区间的替换，分别解释约 +2.278 秒、+1.157 秒的低估。',22,True)
text(40,1107,'含义边界：无可见 GPU 活动不等于确实空闲，可能含提交间隔、等待或观测缺失；物理原因尚未确定。')
text(40,1142,'仅通信区间可能包含通信 kernel 内的等待和轮询，不是纯传输；替换的是分类总时长，不是逐个 kernel。')
text(40,1177,'灰色项 = 实际 lane0 时间 − 四类全部替换后的模型时间。柱子不是可节省的优化收益，也不代表全部 224 卡。')
text(40,1220,'用途：用已知目标观测解释已有误差，帮助寻找源侧可建模的原因；不能把替换后的结果当作事前预测。',20)
a.append('</g></svg>')
(OUT/'internal_attribution.svg').write_text(''.join(a),encoding='utf-8')

render('timeline_overview','目标 224 卡 lane0：模型预测与第 95 轮实际 F/B 时间线',[
'上图：v687 源侧参数生成的预测；下图：第 95 轮 trace 观测。F = 前向，B = 反向，PP stage = 流水线阶段。',
'横轴：相对本轮 1F1B 起点的秒数；纵轴：PP stage。lane0 为每阶段选同一通道的一个 rank，共 14 个。'],[
'灰色 API 区间表示 CPU 进入通信调用到返回的完整时间，可能包含等待，也可能与其他 rank 的 F/B 重叠。',
'它与 PP 模型节点描述同一通信过程的不同视图，不能把灰色区间与 PP 节点耗时再次相加。',
'彩色 F/B 区间是 CPU 标注的阶段墙钟时间，含内部计算、通信和等待；不是纯 GPU 计算时间。'],850)

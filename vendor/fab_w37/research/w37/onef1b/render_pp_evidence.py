"""Render both PP directions from sealed v687 predictions; no refitting."""
from pathlib import Path
import csv, gzip, hashlib, html, json
from statistics import mean
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/w37/A/research-html-20260907'
RUN=ROOT/'results/w37/A/post685-candidate-readiness-r1'
parameters=RUN/'prediction/v687_split_full/source256_readiness_parameters.csv'
local=RUN/'source_validation/v687_split_full_pp_local_results.csv.gz'
# Check the parameter file against its existing prediction seal.
seal=json.loads((RUN/'prediction_seal.json').read_text())
entry=next(r for r in seal['files'] if (RUN/r['path']).resolve()==parameters.resolve())
assert hashlib.sha256(parameters.read_bytes()).hexdigest()==entry['sha256']
params=list(csv.DictReader(parameters.open()))
with gzip.open(local,'rt') as f:
    rows=[r for r in csv.DictReader(f) if r['both_endpoints_single_message']=='True' and r['iteration'] in ['85','90','95','100']]
lookup={(r['direction'],r['pp_lane']):r for r in params}
for r in rows:
    p=lookup[r['direction'],r['pp_lane']]
    d,c=float(p['sender_effective_ready_ms']),float(p['post_ready_completion_ms'])
    x=float(r['receiver_minus_sender_ms'])
    assert abs(max(d,x)-max(0,x)+c-float(r['predicted_postpublication_ms']))<1e-8
summary=[]
for direction in ['F','B']:
    group=[r for r in params if r['direction']==direction]
    for split in ['source_fit','source_incremental_validation']:
        g=[r for r in rows if r['direction']==direction and r['split']==split]
        summary.append(dict(direction=direction,split=split,samples=len(g),ready_mean_ms=mean(float(r['sender_effective_ready_ms']) for r in group),completion_mean_ms=mean(float(r['post_ready_completion_ms']) for r in group),mae_ms=mean(abs(float(r['error_ms'])) for r in g),old_mae_ms=mean(abs(float(r['v686_error_ms'])) for r in g)))
a=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1650" viewBox="0 0 1600 1650" role="img" aria-labelledby="title desc"><title id="title">源侧256卡：前向激活与反向梯度通信的观测和局部预测</title><desc id="desc">前向和反向分别展示散点、模型曲线、局部放大和完整横轴范围；预测以已观测API入口为条件。</desc><rect width="1600" height="1650" fill="#f4f7fb"/>']
def text(x,y,s,size=20,bold=False):
    a.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="#203448"'+(' font-weight="bold"' if bold else '')+'>'+html.escape(s)+'</text>')
text(40,45,'源侧 256 卡：前向激活 / 反向梯度通信的观测与局部预测',29,True)
text(40,85,'场景：PP16 / MB4；FWD = 前向激活消息，BWD = 反向梯度消息；source 指源场景，不是发送方。')
text(40,125,'符号：tS / tR = 发送方 / 接收方的 CPU API 入口时刻；x = tR − tS，正值表示发送方先进入。')
text(40,161,'纵轴 y = 首个 API 返回时刻 − max(tS, tR)：双方进入后仍需多久；这是消息完成的观测上界。')
text(40,197,'模型：tDone = max(tS + d, tR) + c；对应图中曲线 ŷ = max(d, x) − max(0, x) + c。')
text(40,233,'d = 有效发送就绪延迟，c = 汇合后的完成余项；均为模型代理，不代表独立测得的 CPU / GPU / 网络耗时。')
text(40,273,'蓝点：85/90 拟合轮；橙点：95/100 已暴露开发检查轮；细灰线：16 条 lane 模型；黑线：参数均值示意。',19)

def plot(direction,x0,y0,full=False):
    g=[r for r in rows if r['direction']==direction]
    p=[r for r in params if r['direction']==direction]
    xmin=min(float(r['receiver_minus_sender_ms']) for r in g)*1.03 if full else -250
    xmax=400
    ymax=max(float(r['actual_postpublication_ms']) for r in g)*1.08
    ymax=max(ymax,6 if direction=='F' else 110)
    w,h=635,225
    title=('前向激活 FWD' if direction=='F' else '反向梯度 BWD')+(' · 全部样本范围' if full else ' · 入口相近时放大')
    text(x0,y0,title,23,True)
    top=y0+30
    px=lambda x:x0+(x-xmin)/(xmax-xmin)*w
    py=lambda y:top+h-y/ymax*h
    a.append(f'<rect x="{x0}" y="{top}" width="{w}" height="{h}" fill="white" stroke="#b5c5d3"/>')
    for frac in [0,.5,1]:
        value=ymax*frac;yy=py(value)
        a.append(f'<path d="M{x0} {yy} H{x0+w}" stroke="#e1e7ed"/>')
        text(x0-45,yy+5,f'{value:.1f}',14)
    for value in ([xmin,0,400] if full else [-250,0,100,400]):
        xx=px(value);text(xx-20,top+h+25,f'{value:.0f}',14)
    inside=0
    for r in g:
        x,y=float(r['receiver_minus_sender_ms']),float(r['actual_postpublication_ms'])
        if xmin<=x<=xmax:
            inside+=1
            color='#5476a3' if r['split']=='source_fit' else '#c76327'
            a.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="2" fill="{color}" opacity=".35"/>')
    def curve(d,c,color,width):
        xs=sorted(set([xmin,xmax,0,max(xmin,min(xmax,d))]))
        points=' '.join(f'{px(x):.2f},{py(max(d,x)-max(0,x)+c):.2f}' for x in xs if xmin<=x<=xmax)
        a.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"/>')
    for r in p:curve(float(r['sender_effective_ready_ms']),float(r['post_ready_completion_ms']),'#a6aeb9',1)
    d=mean(float(r['sender_effective_ready_ms']) for r in p);c=mean(float(r['post_ready_completion_ms']) for r in p)
    curve(d,c,'#172c40',2.5)
    text(x0,top+h+56,'横轴：接收入口 − 发送入口（ms）',18)
    text(x0,top+h+86,f'显示 {inside}/{len(g)} 点；纵轴：双方进入后至首个返回（ms）',17)
plot('F',100,320);plot('B',895,320)
plot('F',100,740,True);plot('B',895,740,True)
text(40,1160,'关系式与误差：参数分别按方向、lane 在源侧 85/90 拟合；下列 d、c 是 16 lane 的均值。',22,True)
for i,direction in enumerate(['F','B']):
    s=next(r for r in summary if r['direction']==direction and r['split']=='source_incremental_validation')
    d,c=s['ready_mean_ms'],s['completion_mean_ms']
    text(40,1203+i*77,f"{'前向 FWD' if direction=='F' else '反向 BWD'}：tDone ≈ max(tS + {d:.4f}, tR) + {c:.4f} ms；95/100 局部 MAE = {s['mae_ms']:.4f} ms。",22,True)
    text(40,1237+i*77,('前向就绪延迟接近 0，双方进入后剩余时间约 5 ms，未呈现反向约 93 ms 的明显下降区间。' if direction=='F' else '反向发送方领先 0 / 50 / 100 ms 时，均值曲线剩余时间约 98.28 / 48.28 / 5.25 ms。'),20)
text(40,1400,'这里预测了什么：给定实际 tS、tR，用既有源拟合参数预测后续完成；本次逐条复核全部 6912 条封存预测。',21)
text(40,1440,'没有重新拟合，也没有新增目标侧评分。MAE 使用每条消息对应 lane 的参数，黑色均值曲线不用于评分。',21)
text(40,1480,'只选两端均为单消息的 API，排除融合调用的另一条消息干扰。API 返回仍可能含收尾，因此观测量是上界。',21)
text(40,1520,'结论边界：支持局部通信时序关系；不等于事前自由运行结果，也不证明 224 卡完整 iter / MFU 已改善。',21)
text(40,1555,'lane = 流水线通道；MAE = 平均绝对误差。前向 v686 对照为 0.1205 ms，本版 0.1232 ms，未改善。',18)
text(40,1600,'数据：v687_split_full 源侧局部结果与就绪参数；代码：post685/readiness.py、pp_semantics.py。',18)
a.append('</svg>');OUT.mkdir(parents=True,exist_ok=True)
(OUT/'pp_readiness_evidence.svg').write_text(''.join(a),encoding='utf-8')
with (OUT/'pp_direction_local_metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
(OUT/'pp_evidence_provenance.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [parameters,local]},'prediction_rows_verified':len(rows),'refit':False,'target_scored':False},indent=2))
print(json.dumps(summary,indent=2))

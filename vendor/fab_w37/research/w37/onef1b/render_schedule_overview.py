"""Presentation of the unchanged PP14/MB3 action order, without region colors."""
from pathlib import Path
from post685.pp_semantics import action_sequence
import html

out = Path(__file__).resolve().parents[3] / 'results/w37/A/research-html-20260907'
a = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1190" viewBox="0 0 1400 1190"><rect width="1400" height="1190" fill="#f4f7fb"/>']
def text(x, y, s, size=20):
    a.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="#203448">{html.escape(s)}</text>')
text(40, 45, 'PP14 / MB3：完整 1F1B 区间内，各 stage 的本地 F/B 顺序', 28)
text(40, 85, 'PP：流水线阶段，编号 0–13；MB：微批次，编号 0–2。F0 表示微批次 0 的前向，B2 表示微批次 2 的反向。', 19)
text(40, 120, '展示口径：F0 至 B2 统一归入 1F1B 区间；若统计中称为 steady，不代表流水线处于持续满载稳态。', 19)
text(40, 157, '颜色仅区分：蓝色 F = 前向计算；紫色 B = 反向计算。箭头仅表示同一行的先后顺序。', 19)
text(40, 194, '横轴是本地计算动作序号，不是全局时间；列对齐不表示同时执行。通信及其依赖未在此概览展开。', 19)
for j in range(6): text(235 + j*185, 238, f'动作 {j+1}', 19)
for stage in range(14):
    y = 260 + stage*53
    phases = [r for r in action_sequence(stage,14,3) if r['kind']=='phase']
    assert len(phases)==6
    text(45, y+29, f'PP{stage}', 21)
    for j,r in enumerate(phases):
        x=195+j*185
        forward=r['name']=='forward'
        color='#d9ebf7' if forward else '#ede2f7'
        label=('F' if forward else 'B')+str(r['microbatch'])
        a.append(f'<rect x="{x}" y="{y}" width="155" height="40" rx="7" fill="{color}" stroke="#b7c6d5"/>')
        text(x+61,y+28,label,22)
        if j<5:text(x+163,y+28,'→',20)
text(40, 1045, '为何不再按 warmup / steady / cooldown 着色？', 24)
text(40, 1080, '原标签来自每个 stage 的局部代码分支，不能当作全流水线统一的三个时间段。MB3 远小于 PP14。', 19)
text(40, 1114, '局部 steady 循环次数：PP0–10 为 0，PP11 为 1，PP12 为 2，PP13 为 3；底层顺序与依赖保持不变。', 19)
text(40, 1153, '顺序依据：post685/pp_semantics.py::action_sequence；这里只调整展示分类，不修改预测或阶段误差口径。', 18)
a.append('</svg>')
out.mkdir(parents=True,exist_ok=True)
(out/'schedule_regions.svg').write_text(''.join(a),encoding='utf-8')

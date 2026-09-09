"""Render the v687 model contract with a glossary before its symbols are used."""
from pathlib import Path
import html

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/w37/A/research-html-20260907'
WIDTH, HEIGHT = 1600, 1740
parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">前向激活与反向梯度消息：两端条件与完成时刻</title>
<desc id="desc">先解释名词、时刻和时长，再展示发送端与接收端的依赖汇合、消息完成和通信调用返回。93毫秒是源侧拟合代理，可以与接收方前置工作重叠。</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#47647b"/></marker></defs>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f4f7fb"/>''']

def text(x, y, value, size=19, bold=False, color='#203448'):
    weight = ' font-weight="bold"' if bold else ''
    parts.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{color}"{weight}>{html.escape(value)}</text>')

def box(x, y, w, h, lines, fill='#fff', size=19):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#c2d2de"/>')
    for i, line in enumerate(lines):
        text(x + 20, y + 32 + i * 29, line, size)

def arrow(points):
    path = 'M' + ' L'.join(f'{x} {y}' for x, y in points)
    parts.append(f'<path d="{path}" fill="none" stroke="#47647b" stroke-width="2.5" marker-end="url(#arrow)"/>')

text(40, 46, '前向激活与反向梯度消息，分别何时完成？', 30, True)
text(40, 80, 'v687_split_full 研究候选 · 阅读顺序：名词 → 符号 → 依赖流程 → 数值例子', 20)
text(40, 126, '01  名词：图中每个对象指什么', 23, True)
box(40, 145, 745, 158, [
    'rank：训练进程编号；本配置每个 rank 对应一张 GPU。',
    'local：该 rank 内的 CPU / GPU 动作；S：发送方 rank。',
    'R：接收方 rank；FWD：前向激活，BWD：反向梯度消息。',
    'F / B：前向 / 反向计算；两种消息的参数分别拟合。'])
box(810, 145, 750, 158, [
    'PP：流水线并行；PP 消息：流水线阶段之间的数据传递。',
    'API：通信函数调用；入口：CPU 执行到该调用的时刻。',
    'API 返回：调用结束，本 rank 可继续后续程序动作。',
    '融合 API：一次调用关联多条消息，返回需等它们全部完成。'])
text(40, 343, '02  符号：t 表示时刻，d / c 表示时长；单位均为 ms（毫秒）', 23, True)
box(40, 362, 745, 184, [
    'tS：发送方 S 的 API 入口时刻。',
    'tR：接收方 R 的 API 入口时刻。',
    'd：有效发送就绪延迟；前向记 dF，反向记 dB。',
    'tReady = tS + d：模型中的有效发送就绪时刻。',
    '“有效就绪”是潜在代理，尚未区分 CPU、GPU 或传输原因。'])
box(810, 362, 750, 184, [
    'max(a, b)：取两者较大值；对时刻而言，就是取较晚者。',
    'tJoin = max(tReady, tR)：两项模型条件的汇合时刻。',
    'c：汇合后的完成余项；前向记 cF，反向记 cB。',
    'tDone = tJoin + c：本条消息的模型完成时刻。',
    'd、c 均为拟合时长，不能直接解释成纯网络耗时。'])
text(40, 587, '03  两种方向并列：结构相同，就绪延迟不同；箭头表示依赖，框宽不表示耗时', 23, True)
text(40, 622, '参数取源侧 256 卡 85/90 拟合的 16 条流水线通道均值，仅作图示；实际模型按方向、通道分别取参数。', 19)
for x,forward in [(40,True),(810,False)]:
    name='前向激活 FWD' if forward else '反向梯度 BWD'
    d='0.0506' if forward else '93.0369'
    c='5.0311' if forward else '5.2470'
    suffix='F' if forward else 'B'
    box(x,647,745,92,[name+'：模型关系式',f'tDone = max(tS + {d}, tR) + {c}  （单位 ms）'], '#e4edf8',21)
    box(x,760,345,94,['发送方 S 的 API 入口','tS：CPU 进入发送调用'], '#e0f1ec')
    box(x+400,760,345,94,['接收方 R 的 API 入口','tR：CPU 进入接收调用'], '#e0f1ec')
    arrow([(x+172,854),(x+172,886)])
    box(x,886,345,100,['有效发送就绪 tReady',f'tReady = tS + d{suffix}',f'd{suffix} ≈ {d} ms'], '#ede5f5')
    arrow([(x+172,986),(x+172,1010),(x+372,1010),(x+372,1032)])
    arrow([(x+572,854),(x+572,1010),(x+372,1010)])
    box(x,1032,745,94,['两项条件取较晚者：tJoin = max(tReady, tR)',f'再加完成余项 c{suffix} ≈ {c} ms，得到消息完成时刻 tDone。'], '#fff0d6')
    if forward:
        box(x,1150,745,116,['前向的直观近似：', 'tDone ≈ max(tS, tR) + 5.03 ms', '即“较晚的 API 入口 + 约 5 ms”；忽略了很小的 dF。'], '#e0f1ec',21)
    else:
        box(x,1150,745,116,['反向的关键差异：', '发送方需考虑约 93 ms 有效就绪代理，再与接收入口取 max。', '这段时间可与接收方前置工作重叠，不能汇合后再加一次。'], '#ede5f5',19)
text(40,1310,'04  两种方向共用的返回条件',23,True)
box(40,1330,1520,91,['本 rank 的通信 API 等关联消息完成后返回，随后按程序顺序继续 F/B 或其他通信。', '融合 API 关联多条消息：须等所有关联消息完成；各 rank 分别推进，不是全流水线一起进入下一步。'], '#fff0d6',21)
text(40,1465,'05  同一到达条件下的例子：tS = 0，tR = 20 ms（示意，非 trace 样本）',23,True)
box(40,1485,745,86,['前向：max(0.0506, 20) + 5.0311 ≈ 25.031 ms', '接收方较晚进入，决定汇合时刻。'], '#fff',20)
box(810,1485,750,86,['反向：max(93.0369, 20) + 5.2470 ≈ 98.284 ms', '发送方有效就绪较晚，决定汇合时刻。'], '#fff',20)
text(40,1610,'边界：“入口”指 CPU 进入 API，不是 GPU 数据已就绪；tDone 是模型完成时刻，观测图用首个 API 返回作上界。',19)
text(40,1646,'前向约 5 ms、反向约 93 ms / 5.25 ms 都是当前源场景拟合代理；未证实可跨消息大小和配置通用。',19)
text(40,1682,'本次只补充图示，不修改预测参数、正式 v684 / v685 拓扑锁；来源：post685/readiness.py、pp_graph.py。',18)
parts.append('</svg>')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'causal_contract.svg').write_text(''.join(parts), encoding='utf-8')

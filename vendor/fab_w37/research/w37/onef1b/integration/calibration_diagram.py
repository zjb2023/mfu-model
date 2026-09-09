"""Generate a standalone evidence-to-parameter map from the audited v685 values."""
import html
import json
from common import OUT, dump, digest

def build():
    audit = json.loads((OUT/'audit.json').read_text())
    inv = json.loads((OUT/'model_cost_inventory.json').read_text())
    p = audit['prediction']
    c = {r['key']: r for r in audit['compute_summary']}
    keys = ['backward|dense|dense_to_step_exit|exposed_gap|step_exit',
            'backward|moe|step_entry_to_moe|exposed_gap|before_cp0',
            'forward|moe|moe_to_step_exit|exposed_gap|step_exit']
    positions = [('PP0 · rank0–15 · MB0/1/2', '反向：稠密层后 → 本阶段退出'),
                 ('PP13 · rank208–223 · MB0/1/2', '反向：MoE入口 → 首个CP之前'),
                 ('PP13 · rank208–223 · MB0/1/2', '前向：MoE层后 → 本阶段退出')]
    rows = []
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 1660" role="img" aria-labelledby="cal-map-title cal-map-desc" class="calibration-map">',
           '<title id="cal-map-title">v685：源侧输入如何成为224卡预测参数</title>',
           '<desc id="cal-map-desc">从左到右依次为源侧输入、参数处理和目标作用。三条计算成本、反向消息和入口重新校准，旧成本与outer继承，图外补差单列。箭头是参数数据流，不是新增训练依赖边。</desc>',
           '<defs><marker id="cal-map-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#6b8192"/></marker></defs>',
           '<rect width="1440" height="1660" rx="12" fill="#f4f7fb"/>']
    def text(x,y,s,size=19,fill='#203448',bold=False):
        svg.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{fill}"'+(' font-weight="700"' if bold else '')+'>'+html.escape(s)+'</text>')
    def box(x,y,w,h,title,lines,color):
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="white" stroke="{color}" stroke-width="2"/>')
        text(x+16,y+30,title,21,color,True)
        for i,line in enumerate(lines):text(x+16,y+62+i*29,line,18)
    def arrow(y):
        for x1,x2 in [(450,487),(968,1005)]:
            svg.append(f'<path d="M{x1} {y} H{x2}" stroke="#6b8192" stroke-width="2" marker-end="url(#cal-map-arrow)"/>')
    text(28,43,'v685：哪些源输入 → 哪些参数 → 改到目标图哪里',29,bold=True)
    text(28,78,'源256卡 → 目标224卡；以下是参数数据流，不是训练时间轴，也不是新增依赖边。',19)
    text(28,112,'蓝色：本轮重新拟合 / 更新    灰色：继承旧值    橙色：整轮时间补差',19)
    for x,label in [(28,'① 具体输入'),(495,'② 参数与处理'),(1013,'③ 224卡预测中的作用')]:text(x,152,label,23,bold=True)
    blue='#157b9c';gray='#61758b';orange='#ad6517'
    box(28,174,422,354,'计算活跃区间 · 源85/90/95/100',[
        '已有 operator_cost_group_observations',
        '每个PP stage取lane0代表rank',
        '统计暴露 / 重叠的计算活跃并集',
        '360行统计 → 58个物理成本键',
        '按五字段匹配目标节点的成本键',
        '实际命中下列3键；其余保留旧值',
        '五字段命名与下方58键表一致',
        '不是58个键全部更新目标计算'],blue)
    # Each key is rendered verbatim across two lines, preserving the table's five-field order.
    box(495,174,473,354,'三个实际命中的键（五字段原值）',[],blue)
    box(1013,174,399,354,'更新计算节点：48 + 48 + 48',[],blue)
    for i,key in enumerate(keys):
        r=c[key];parts=key.split('|');y=240+i*95
        text(510,y,' | '.join(parts[:3]),16)
        text(510,y+25,' | '.join(parts[3:]),16)
        text(510,y+50,f"新计算成本 {r['new_compute_ms']:.6f} ms",18,blue,True)
        text(1028,y,positions[i][0],18)
        text(1028,y+27,positions[i][1],17)
        text(1028,y+54,f"{r['nodes_updated']}节点；每rank的3个微批次",17)
        rows.append({'source':'source256 85/90/95/100 lane0 compute active union',
                     'key':key,'new_compute_ms':r['new_compute_ms'],'target_position':positions[i][0],
                     'target_nodes':r['nodes_updated'],'action':'refit and bind'})
    arrow(340)
    box(28,549,422,167,'反向PP边界 · 源四轮全部lane',[
        '已有 source_pp_trace_events',
        '接收stage的B开始 − 发送stage的B结束',
        '15条PP边界 × 16 lane × 4 MB'],blue)
    box(495,549,473,167,'重新拟合960条反向消息成本',[
        '每个接收stage / lane / MB取四轮中位数',
        '总成本 = 4.704806 ms + 其余完成分量',
        '4.704806 ms只是继承的服务分量'],blue)
    box(1013,549,399,167,'映射后更新624个反向消息节点',[
        '13条PP边界 × 16 lane × 3 MB',
        'lane0 / MB0 / PP13→PP12示例：',
        '100.629 ms = 4.705 + 95.924 ms'],blue)
    arrow(638)
    box(28,736,422,118,'迭代入口 · 源四轮',[
        'Profiler起点 → 首个F/B的边界差',
        '已有 source256_profiler_entry 表'],blue)
    box(495,736,473,118,'重新拟合入口时间',[
        '取四轮中位数：1265.388210 ms',
        '图内 framework_entry 成本'],blue)
    box(1013,736,399,118,'更新1个全局入口节点',[
        '位于首个F/B之前的依赖链',
        '已计入下方原始图20.781257秒'],blue)
    arrow(800)
    box(28,874,422,170,'v684父图 / 历史源256成本',[
        '已有图节点、依赖、计算/通信模板',
        '旧成本可来自60–100窗口',
        '目标PP14 / MB3配置已经存在'],gray)
    box(495,874,473,170,'继承：并非本轮全部重新拟合',[
        '前向PP服务：4.704806 ms / 消息',
        '未命中计算、旧非计算下限、CP/EP等',
        'DP/EDP、优化器、收尾及正式依赖锁'],gray)
    box(1013,874,399,170,'继续供全图使用',[
        '624个前向PP消息保持原成本',
        '命中计算节点仍保留非计算下限',
        '327746节点 / 364784边不变'],gray)
    arrow(960)
    box(28,1064,422,145,'源Profiler与源图的总量差',[
        '源Profiler中位数：23831.161007 ms',
        '源回放图：23025.241449 ms',
        '差额：805.919557 ms'],orange)
    box(495,1064,473,145,'重新计算图外补差',[
        '源差额 × 目标/源微批次数',
        '805.919557 × 3/4',
        '这是一项比例迁移假设'],orange)
    box(1013,1064,399,145,'加到目标原始图总时间',[
        '604.439668 ms',
        '得到Profiler口径；不是新增等待边',
        '源回放未绑定这批新计算成本'],orange)
    arrow(1144)
    box(28,1229,422,116,'历史源256两种时钟 · 60–100',[
        '训练日志整轮时间 − Profiler时间',
        '九轮中位数；最初来自v5.4'],gray)
    box(495,1229,473,116,'继承outer两时钟差额',[
        '1372.453390 ms',
        '不重新拟合，不均分到各节点'],gray)
    box(1013,1229,399,116,'加到预测Profiler总时间',[
        '得到Training整轮时间',
        '未定位为某个连续事件区间'],gray)
    arrow(1295)
    svg.append('<rect x="28" y="1367" width="1384" height="204" rx="10" fill="#eaf2f7" stroke="#a9bfce"/>')
    text(48,1402,'汇总：沿原依赖图求完成时刻，再衔接两种整轮时钟',23,bold=True)
    text(48,1438,f"原始图 {p['target_raw_graph_ms']:.6f} ms + 图外补差 {p['target_reconciliation_ms']:.6f} ms = Profiler {p['profiler_step_ms']:.6f} ms",21)
    text(48,1475,f"Profiler {p['profiler_step_ms']:.6f} ms + outer {p['outer_framework_ms']:.6f} ms = Training {p['training_step_ms']:.6f} ms",21)
    text(48,1512,f"MFU = 有效FLOPs / (224 × 单GPU峰值 × Training秒数) = {p['mfu_pct']:.6f}%",21)
    text(48,1548,'继承换算输入：每步8.436548311982576e16 FLOPs；单GPU峰值500 TFLOP/s；分子未独立重验。',18)
    text(28,1609,'重要边界：图内成本不能跨rank直接求和；消息成本的分量不能再额外重复加入；v687约93 ms代理未用于此图。',18)
    text(28,1640,'来源：已封存v685节点/参数/合同与本页审计；本图没有新拟合、改成本或新增目标评分。',18)
    svg.append('</svg>')
    (OUT/'v685_calibration_map.svg').write_text(''.join(svg),encoding='utf-8')
    dump('v685_calibration_map.json',{'version':'v685','compute_mapping':rows,
         'forward_pp':inv['forward_pp'],'backward_pp':inv['backward_pp'],
         'prediction':p,'audit_sha256':digest(OUT/'audit.json'),
         'svg_sha256':digest(OUT/'v685_calibration_map.svg'),
         'arrows':'parameter data flow, not training dependencies','new_fit':False})

if __name__ == '__main__': build()

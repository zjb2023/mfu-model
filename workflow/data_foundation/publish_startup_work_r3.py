"""Publish an additive startup inventory page without changing model costs."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = Path('results/data-foundation/2111-blocks-ui-r1/df-v001')
MIRROR = Path('/home/zjb/Desktop/worktrees/mfu-w37-v610')
AUDIT = ROOT/'results/data-foundation/startup-work-r3-verified'
DOC = ROOT/'docs/data-foundation/STARTUP_WORK_R3.md'
SRC = Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron')
LOG = Path('/home/zjb/Desktop/32/gpu32_gbs64_framework/2026-09-18-10:38/worker34095/2026-09-18_1038/tp1_pp4_dp_mbs2_numbs_gbs64_gpus0_mtp1_forcelbfalse_pertensorfalse_NO_LOSS_REDUCE.RANK0.10.124.34.95.log')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows = json.loads((AUDIT/'report.json').read_text())['rows']
    pair = [r for r in rows if r['iteration']==60]
    assert [r['world'] for r in pair] == [32,256]
    table = ''.join(f'<tr><td>{html.escape(a["name"])}</td><td>{a["elapsed_ms"]:.3f}</td><td>{b["elapsed_ms"]:.3f}</td></tr>' for a,b in zip(pair[0]['parts'],pair[1]['parts']))
    colors=['#687789','#b45309','#9ca3af','#268086','#c04a37','#397ab8','#8057a0']
    bars=''
    for r in pair:
        blocks=''.join(f'<div title="{html.escape(p["name"])}：{p["elapsed_ms"]:.3f} ms" style="width:{p["elapsed_ms"]/1250*100:.6f}%;background:{colors[i]}"></div>' for i,p in enumerate(r['parts']))
        bars+=f'<p>{r["world"]}卡 · {r["startup_ms"]:.3f} ms</p><div class="bar">{blocks}</div>'
    notes=html.escape(DOC.read_text())
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Profiler启动工作清单 · r3</title><style>
body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f3f5f8;color:#182434}main{max-width:1080px;margin:auto;padding:24px}section{background:white;padding:20px;border-radius:12px;margin:18px 0}h1{font-size:28px}h2{font-size:21px}.bar{display:flex;height:32px;background:#edf0f4;border:1px solid #dae0e8}.bar div{height:100%;flex-shrink:0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #ddd;padding:10px}td:not(:first-child){white-space:nowrap}.flow{display:flex;flex-wrap:wrap;gap:8px}.flow span{background:#edf2f8;padding:8px 12px;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui}a{color:#1761a0}.muted{color:#526072}.scroll{overflow-x:auto}*{box-sizing:border-box}</style><main>
<a href="../prediction-overview-r2/#tail-startup-candidate">← 返回整轮预测</a><h1>启动窗口里究竟在做什么？</h1><p>ProfilerStep开始 → PP0首F关联GPU开始；32/256卡各iter40/60/80。此页是实测诊断，不修改预测成本。</p>
<section><h2>不是单一准备工作，也不是token路由分发</h2><div class="flow"><span>按层MoE统计归约</span><span>→ 两次全局barrier</span><span>→ 主机统计/调度残余</span><span>→ 全局计时AG与读回</span><span>→ 梯度清零 / 首F准备</span></div><p>“计时统计”由张量形状、调用顺序及236B代码强匹配支持；缺少采集时Python调用栈，尚非栈级确证。</p></section>
<section><h2>iter60：同一时间比例尺（0～1250 ms）</h2>'''+bars+'''<p class="muted">红色长段是小消息AG设备驻留，不等于纯传输。灰色是barrier后主机侧记录稀疏区间。CPU读取统计结果与AG重叠，不能各加一次。</p><div class="scroll"><table><thead><tr><th>不重叠时间窗口</th><th>32卡 ms</th><th>256卡 ms</th></tr></thead><tbody>'''+table+'''</tbody></table></div><p>分段按时间边界划分，异步工作可跨段；例如前面发起的AllReduce在barrier窗口内继续执行。</p></section>
<section><h2>哪些规模因素真正有证据？</h2><ul><li>层数：小AllReduce长度12→60，吻合按层辅助损失统计，不是专家token分发。</li><li>rank数：计时表32×24→256×24，全局同步参与者变多；但秒级驻留仍未分离到齐等待与通信服务。</li><li>本地参数/缓冲区：PP0两块梯度缓冲尺寸不变，清零约4.80→4.85ms。不能按全模型参数量同比放大。</li><li>主机残余：约83.6/88.1ms，当前trace不足以确认Python、日志IO或调度占比。</li></ul><p>最简建模应分别保留统计/同步、主机残余、清零与首F发起项，并保留CPU/GPU重叠；不拟合一个统一rank倍数。</p></section>
<section><h2>下一步与边界</h2><p>先核对代表性跨PP rank的同一计时AG与本地就绪顺序；没有可验证跨机时钟时，不宣称找到了最后到达者。另需实际代码或主机调用证据才能解释80～90ms残余。</p><p>清单/边界检查PASS；等待机制归因PARTIAL。原1F1B与整轮预测未改。</p><a href="notes.md">方法与复现文档</a> · <a href="manifest.json">来源与哈希</a><details><summary>展开完整分析说明</summary><pre>'''+notes+'''</pre></details></section></main></html>'''
    orig=ROOT/REL/'prediction-overview-r2'
    mirror=MIRROR/REL/'prediction-overview-r2'
    assert (orig/'index.html').read_bytes()==(mirror/'index.html').read_bytes()
    injection='''<script>(()=>{const p=document.createElement('p');p.id='startup-work-r3-link';p.innerHTML='<a href="../startup-work-r3/">查看启动窗口完整工作链：按层统计、barrier、主机残余、计时AG、梯度清零</a>';document.querySelector('#tail-startup-candidate').appendChild(p);})();</script>'''
    for base in [ROOT,MIRROR]:
        dest=base/REL/'startup-work-r3'
        overview=base/REL/'prediction-overview-r2'
        assert not dest.exists()
        assert not (overview/'index-before-startup-work-r3.html').exists()
    for base in [ROOT,MIRROR]:
        dest=base/REL/'startup-work-r3';dest.mkdir()
        (dest/'index.html').write_text(page)
        (dest/'notes.md').write_bytes(DOC.read_bytes())
        (dest/'report.json').write_bytes((AUDIT/'report.json').read_bytes())
        (dest/'audit-manifest.json').write_bytes((AUDIT/'manifest.json').read_bytes())
        inputs={str(p):sha(p) for p in [DOC,AUDIT/'report.json',AUDIT/'manifest.json',LOG,SRC/'core/distributed/distributed_data_parallel.py',SRC/'core/distributed/param_and_grad_buffer.py']}
        manifest=dict(inputs=inputs,code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p):sha(p) for p in dest.iterdir()})
        (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        overview=base/REL/'prediction-overview-r2'
        index=overview/'index.html';mf=overview/'manifest.json'
        (overview/'index-before-startup-work-r3.html').write_bytes(index.read_bytes())
        (overview/'manifest-before-startup-work-r3.json').write_bytes(mf.read_bytes())
        old=index.read_text();assert old.count('</body>')==1
        index.write_text(old.replace('</body>',injection+'\n</body>'))
        m=json.loads(mf.read_text());m['version']='prediction-overview-r2-startup-work-r3-link';m['code'][str(Path(__file__).resolve())]=sha(Path(__file__))
        for p in [index,overview/'index-before-startup-work-r3.html',overview/'manifest-before-startup-work-r3.json']:
            m['outputs'][str(p)]=sha(p)
        mf.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    print('PASS published bounded startup inventory and additive overview link')

if __name__=='__main__':main()

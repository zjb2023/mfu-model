import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';data=BASE/'optimizer-tail-audit-r1'
dest=BASE/'2111-blocks-ui-r1/df-v001/optimizer-tail-audit-r1'
live=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/optimizer-tail-audit-r1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
dest.mkdir(parents=True,exist_ok=False)
for name in ['report.json','source.json','target.json']:shutil.copyfile(data/name,dest/name)
r=json.loads((data/'report.json').read_text());a=json.loads((data/'source.json').read_text());b=json.loads((data/'target.json').read_text())
table=''.join('<tr><td>'+x['name']+'</td>'+''.join('<td>'+f'{x[k]:.3f}'+'</td>' for k in ['source32_ms','target256_ms','target_minus_source_ms'])+'</tr>' for x in r['windows'])
svg='<svg viewBox="0 0 1100 175" role="img" aria-label="尾段窗口时序"><text x="130" y="20">末B完成 = 0；以下为观测窗口，不代表窗口内工作严格串行</text>'
colors=['#a66a12','#6a7988','#315d98','#8c4873']
for j,x in enumerate([a,b]):
    y=40+j*60;svg+=f'<text x="0" y="{y+23}">{x["world"]}卡</text>'
    for i,w in enumerate(x['windows']):
        svg+=f'<rect x="{130+w["start_ms"]*2.3}" y="{y}" width="{w["duration_ms"]*2.3}" height="32" fill="{colors[i]}"><title>{w["name"]} {w["duration_ms"]:.3f} ms</title></rect>'
    svg+=f'<text x="{140+x["tail_ms"]*2.3}" y="{y+24}">{x["tail_ms"]:.1f} ms</text>'
svg+='</svg>'
html='''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>优化器尾段 · 首次事实核对</title><style>*{box-sizing:border-box}body{font:16px/1.65 sans-serif;color:#24344a;background:#f3f6fa;margin:0}main{max-width:1200px;margin:auto;padding:24px}section{background:white;border:1px solid #cbd5e0;padding:20px;margin:20px 0}.scroll{overflow:auto}svg{width:100%;min-width:850px}svg text{font:14px sans-serif;fill:#24344a}table{width:100%;border-collapse:collapse}th,td{padding:12px;text-align:left;border-bottom:1px solid #cbd5e0}.warning{background:#fff1d9;padding:14px}a{color:#315d98}.legend{display:flex;gap:14px;flex-wrap:wrap}.small{font-size:14px;color:#586c82}</style></head><body><main><a href="../prediction-overview-r2/">返回已冻结的F/B误差分析</a><h1>优化器尾段：先定位80.4 ms差在哪里</h1><p>32卡 → 256卡，iter60、rank0。首F至末B模型不改；此页是后续尾段工作的独立记录。</p><section><h2>同一口径：PP0末B GPU完成 → ProfilerStep结束</h2><div class="scroll">'''+svg+'''</div><div class="legend"><span>棕：到最后RS</span><span>灰：RS后到更新GPU开始</span><span>蓝：更新开始到最后AG</span><span>紫：AG后收尾</span></div><div class="scroll"><table><thead><tr><th>观测窗口</th><th>源32 ms</th><th>目标256 ms</th><th>目标−源 ms</th></tr></thead><tbody>'''+table+'''</tbody></table></div><p class="warning">尾段总差80.372ms，其中62.117ms落在“末B→最后RS完成”，16.014ms落在“最后AG→Step结束”。这是端点窗口差异，不是纯网络service差异。</p></section><section><h2>当前怎么建模，下一步改哪里</h2><p>冻结版本仍使用源32卡尾段296.626ms，目标实测376.997ms仅用于评价。先把原尾段分为上述四个显式预留参数，四项保持源值即可重现原预测；不将目标95.360ms直接回填为新RS预测。</p><p>首要核对：RS的DP/EDP组别、输入量、调用就绪与GPU完成。32→256的DP4→8、EDP1→2可能改变工作和依赖，但目前尚未逐kernel证实，不能先将62.117ms归因为带宽或EDP。</p><p>其他stage的RS可能与前面B重叠；不能将所有stage的RS再加到PP0尾部。这里没有建立全世界同步屏障，也没有证明最后AG代表全256rank完成。</p></section><section><h2>数据边界</h2><p>CPU调用只用于关联设备事件，不将CPU提交结束当GPU完成。每类GPU活动并集可能相互重叠，不能求和代替尾段；无设备覆盖间隔也不直接叫纯CPU开销。</p><p class="small">当前仅一个迭代、源/目标各一个rank，状态PARTIAL。原始trace只读，F/B冻结提交8e55d28保持不变。本页不宣称新尾段预测精度。</p><a href="report.json">诊断报告</a> · <a href="manifest.json">来源与SHA</a></section></main></body></html>'''
(dest/'index.html').write_text(html)
m=dict(inputs={str(p):sha(p) for p in data.glob('*.json')},code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in dest.iterdir()})
(dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');shutil.copytree(dest,live);print('PASS published tail audit')

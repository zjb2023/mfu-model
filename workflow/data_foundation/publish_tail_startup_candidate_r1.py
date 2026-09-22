"""Append tail candidate and startup scale evidence without changing frozen prediction."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
folders=[BASE/'2111-blocks-ui-r1/df-v001/prediction-overview-r2',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(folders[0]/'index.html')==sha(folders[1]/'index.html')
report=BASE/'profiler-startup-r2/report.json';r=json.loads(report.read_text());rows=''.join(f'<tr><td>{it}</td><td>'+ '</td><td>'.join(f'{next(x for x in r["rows"] if x["world"]==w and x["iteration"]==it)["startup_GPU_ms"]:.3f}' for w in [32,256])+'</td></tr>' for it in [40,60,80])
panel=f'''<section id="tail-startup-candidate" class="panel" style="margin-top:24px">
<h2>下一阶段：启动与优化器尾段（候选，未改正式预测）</h2>
<h3>尾段预计增加约51.5 ms，再加独立同步开销</h3>
<p>简化公式：32卡尾段＋PP0普通DP RS延长量＋PP0 EDP RS＋EDP AG暴露增量＋额外同步开销。AG跨stage并发，不按PP个数相加。</p>
<div class="scroll"><table><thead><tr><th>增量项</th><th>候选耗时</th></tr></thead><tbody><tr><td>普通DP RS新增跨机部分</td><td>9.93 ms</td></tr><tr><td>PP0新增EDP RS</td><td>31.46 ms</td></tr><tr><td>两轮EDP AG相对源普通AG的暴露增量</td><td>10.12 ms</td></tr><tr><td>合计</td><td>51.51 ms</td></tr></tbody></table></div>
<p><strong>296.6＋51.5＝348.1 ms，再加同步开销。</strong>目标尾段约377.0 ms，差28.9 ms不能直接认定为同步成本。</p>
<p class="small">源NIC400Gb/s×60%=30GB/s；目标同规格、每rank独享NIC lane、DP分层跨机、原机内及普通AG成本不变、PP0 EDP RS全部暴露均为假设。EDP AG每轮15.73ms，与源普通AG11.02/10.32ms分别取max，只加4.71/5.41ms。两轮为trace中独立调用，非重复记录；源AG有效时长可能包含等待。</p>
<h3>Profiler口径启动段：ProfilerStep开始 → PP0首F GPU开始</h3>
<div class="scroll"><table><thead><tr><th>迭代</th><th>32卡整个启动段 ms</th><th>256卡整个启动段 ms</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>这张表统计整个启动窗口，不是单个AllGather。iter60中：32卡107.18ms，256卡1196.82ms。首F CPU到首F GPU仅约4.61/4.33ms；主要差异发生在框架进入首F之前。</p>
<p>窗口内以小全局AG为诊断分界，扣除该事件驻留后其余时间约104.97/127.93ms；其驻留差约1066.69ms，对整个启动差1089.65ms的时间分段占比约97.9%。这是窗口定位，不证明全部为纯网络或等待。参考代码可能是计时统计汇总，但未证明实际调用来源，也不能直接认作上一轮工作。</p>
<p>256卡三轮整个启动窗口约1.20–1.26秒，32卡约0.103–0.129秒。有规模关联，但两个规模来自不同采集运行，rank数并非唯一解释变量；不按卡数倍率放大，不将启动整体套用60%带宽模型。</p>
<p class="small">尾段大消息按60%服务候选；启动统计／同步单独保留。未把目标耗时回填正式预测。</p>
<p><a href="startup-scale-notes.md">方法、假设与复现</a> · <a href="startup-scale-report.json">跨迭代证据</a></p></section>'''
script="(()=>{const main=document.querySelector('main');const footer=main.querySelector('footer');if(footer)footer.insertAdjacentHTML('beforebegin',PANEL);else main.insertAdjacentHTML('beforeend',PANEL);})();".replace('PANEL',json.dumps(panel,ensure_ascii=False).replace('</','<\\/'))
for folder in folders:
 p=folder/'index.html';h=p.read_text();assert 'tail-startup-candidate' not in h
 for name in ['index.html','manifest.json']:
  dest=folder/name.replace('.', '-before-tail-startup.',1);assert not dest.exists();shutil.copyfile(folder/name,dest)
 p.write_text(h.replace('</body>','<script>'+script+'</script></body>'))
 for origin,name in [(report,'startup-scale-report.json'),(report.with_name('manifest.json'),'startup-scale-manifest.json'),(ROOT/'docs/data-foundation/PROFILER_STARTUP_R2.md','startup-scale-notes.md')]:shutil.copyfile(origin,folder/name)
 manifest=dict(version='prediction-overview-r2-tail-startup-candidate-r1',code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(q.resolve()):sha(q) for q in folder.rglob('*') if q.is_file() and q!=folder/'manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PASS appended tail/startup candidates; frozen metrics retained')

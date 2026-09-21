"""Refresh existing overview with recomputed r11 results; preserve prior UI."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
SRC=BASE/'prediction-r11-recompute-r1-verified'
FOLDERS=[BASE/'2111-blocks-ui-r1/df-v001/prediction-overview-r2',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((SRC/'report.json').read_text());assert sha(FOLDERS[0]/'index.html')==sha(FOLDERS[1]/'index.html')
script='''
document.querySelector('header .small').textContent='32to256 / R11重计算成本子图已重新装配与评价 / 源成本未调参';
const refreshPanel=document.createElement('section');refreshPanel.id='r11-recalculation';refreshPanel.className='panel';
refreshPanel.innerHTML='<h2>已按R11重算：结构细化，默认成本未校准</h2><p>256卡展开421,997个节点、112个F/B实例；32卡回归与外层节点收缩检查通过。重计算不再整段保留，计算、CP2、EP8成本可分别替换。</p><p><strong>1F1B预测21.653秒 / 实测20.396秒，误差+6.16%。</strong>整轮ProfilerStep预测22.057秒 / 实测21.970秒，误差+0.40%；派生MFU预测4.583% / 实测推导4.601%。</p><p>默认结果与R10一致，不代表新校准改善。整轮低误差仍包含启动低估与1F1B高估抵消；启动、优化器尾段及PP成本没有更新。</p><p><a href="../b-recompute-r11/#b-dag">打开R11重计算＋真反向成本图</a> · <a href="r11-recalculation-report.json">本次复算报告</a> · <a href="r11-recalculation-manifest.json">复现来源与哈希</a></p>';
document.querySelector('#overall').before(refreshPanel);
const evidence=document.querySelector('#b-evidence');if(evidence){const p=document.createElement('p');p.className='small';p.id='r11-evidence-note';p.textContent='R11已重新装配，默认B及1F1B完成时间与此前一致。因此下方B位置曲线和oracle诊断仍作为既有误差定位证据保留，不冒充R11成本校准或独立验证。';evidence.insertBefore(p,evidence.children[1]||null);}
for(const p of document.querySelectorAll('p')){if(p.textContent.startsWith('下一步：核对32卡PP1与PP2的B成本'))p.textContent='下一步：通过R11计算/通信成本接口，核对源32与目标256专家FC1/FC2的工作量和有效成本，再单项替换重调度；不直接拟合整体系数。';}
document.querySelector('footer').textContent='R11重新装配与缓存评价完成；默认1F1B仍+6.16%。目标iter60已有曝光，非盲测；MFU沿用历史有效FLOPs口径。';
'''
for folder in FOLDERS:
 p=folder/'index.html';html=p.read_text();assert 'id=\'r11-recalculation\'' not in html and 'refreshPanel.id=' not in html
 for name,backup in [('index.html','index-before-r11-recalculation.html'),('manifest.json','manifest-before-r11-recalculation.json'),('report.json','report-before-r11-recalculation.json')]:
  if (folder/name).exists():assert not (folder/backup).exists();shutil.copyfile(folder/name,folder/backup)
 # Replace the full-step payload used by the existing Profiler-first renderer.
 marker='const FULL=';start=html.index(marker)+len(marker);_,length=json.JSONDecoder().raw_decode(html[start:]);html=html[:start]+json.dumps(report,ensure_ascii=False).replace('</','<\\/')+html[start+length:]
 html=html.replace('../b-four-layers-r10/#b-dag','../b-recompute-r11/#b-dag')
 html=html.replace('前向重计算<br>trace保留包络','前向重计算<br>R11计算/CP/EP成本子图')
 html=html.replace('重计算、间隔与后处理保留项','重计算内的就绪、重排与后处理保留项')
 html=html.replace('打开原四层B依赖与成本编辑图','打开R11四层B依赖与成本编辑图')
 html=html.replace('在本页展开原B交互图（四层依赖与成本编辑）','在本页展开R11 B交互图（重计算＋真反向成本）').replace('B反向原交互图','R11重计算与真反向交互图')
 html=html.replace('</body>','<script>'+script+'</script></body>');p.write_text(html)
 for n in ['report','manifest','expanded-checks','sealed-prediction','outer-prediction']:
  shutil.copyfile(SRC/(n+'.json'),folder/('r11-recalculation-'+n+'.json'))
 shutil.copyfile(SRC/'report.json',folder/'report.json')
 manifest=dict(version='prediction-overview-r2-r11-refresh',inputs={str(SRC/'manifest.json'):sha(SRC/'manifest.json')},code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PASS overview refreshed with R11 result; old page archived')

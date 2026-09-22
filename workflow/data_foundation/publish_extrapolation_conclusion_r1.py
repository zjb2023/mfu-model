"""Front-page conclusion plus self-contained documentation/evidence bundle."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
folders=[BASE/'2111-blocks-ui-r1/df-v001/prediction-overview-r2',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(folders[0]/'index.html')==sha(folders[1]/'index.html')
doc=ROOT/'docs/data-foundation/MOE_EXTRAPOLATION_ANALYSIS_20260922.md'
panel='''<section id="extrapolation-conclusion" class="panel" style="margin:22px 0;border:2px solid #2563eb;border-left:8px solid #2563eb;background:#eef5ff">
<div class="small">核心结论 · 2026-09-22 · 成本迁移诊断</div>
<h2>结构可以重复，专家成本不能仅凭结构相同就视为常数。</h2>
<p style="font-size:18px"><strong>当前1F1B外推偏差主要与MoE相关成本迁移有关。</strong>按PP规模重复源stage时，虽然保留了相同计算结构，却没有表达逐层专家路由可能带来的实际工作量与负载差异，因此固定成本模板出现系统性高估。</p>
<p>相同hidden size只固定每个token的特征宽度；不同层、不同专家实际收到的token数和负载分布可以不同，进而影响FC、激活／门控、重排及通信就绪。源／目标PP1已观察到输入行数与GEMM次数不同。</p>
<p><strong>原预测 +6.16% → 成本替换诊断 −0.51%。</strong>替换包含F/B专家计算，以及B中已识别的token重排／EP活动，不是“只换FC”。模型参数未改，正式预测仍为+6.16%。</p>
<p class="small">边界：尚未证明全部差异由路由随layer变化造成，也未证明“两头大、中间小”；当前使用32卡PP1模板，不是256卡两头插值。代表rank目标实测替换不等于完整EP8因果验证，−0.51%不是独立外推精度。</p>
<p><a href="analysis-methods-and-sources.md">分析方法、数据来源与复现文档</a> · <a href="analysis-evidence-index.json">证据清单与SHA256</a> · <a href="#overestimate">查看整体1F1B归因</a> · <a href="#moe-attribution-r1">查看B/F专家子项</a></p>
</section>'''
script="(()=>{document.querySelector('header').insertAdjacentHTML('afterend',"+json.dumps(panel,ensure_ascii=False).replace('</','<\\/')+");})();"
packages=[('PP1_ALIGNMENT_R1','pp1-alignment-r1'),('FC_TRANSFER_R1','fc-transfer-r1-verified'),('B_REMAINING_R1','b-remaining-r1-verified'),('TOKEN_ATTENTION_R1','token-attention-r1'),('MOE_TWO_GROUPS_R1','moe-two-groups-r1'),('F_EXPERT_TRANSFER_R1','f-expert-transfer-r1-verified')]
for folder in folders:
 p=folder/'index.html';html=p.read_text();assert 'id=\\"extrapolation-conclusion' not in html
 for name in ['index.html','manifest.json']:
  backup=folder/name.replace('.', '-before-front-conclusion.',1);assert not backup.exists();shutil.copyfile(folder/name,backup)
 p.write_text(html.replace('</body>','<script>'+script+'</script></body>'));shutil.copyfile(doc,folder/'analysis-methods-and-sources.md')
 evidence=folder/'analysis-evidence';evidence.mkdir(exist_ok=True);entries=[]
 for name,directory in packages:
  entry={'name':name,'files':[]}
  for source in [ROOT/'docs/data-foundation'/(name+'.md'),BASE/directory/'report.json',BASE/directory/'manifest.json']:
   dest=evidence/(name+'-'+source.name);shutil.copyfile(source,dest)
   entry['files'].append(dict(path=str(source),url=str(dest.relative_to(folder)),sha256=sha(source)))
  entries.append(entry)
 (folder/'analysis-evidence-index.json').write_text(json.dumps(dict(summary_doc=dict(path=str(doc),sha256=sha(doc)),entries=entries),ensure_ascii=False,indent=2)+'\n')
 manifest=dict(version='prediction-overview-r2-front-conclusion-r1',inputs={str(doc):sha(doc)},code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(q.resolve()):sha(q) for q in folder.rglob('*') if q.is_file() and q!=folder/'manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PASS conclusion and documentation published; previous page preserved')

"""Append F expert-only conditions to the existing overview diagnosis table."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
folders=[BASE/'2111-blocks-ui-r1/df-v001/prediction-overview-r2',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')]
src=BASE/'f-expert-transfer-r1-verified/report.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads(src.read_text());assert sha(folders[0]/'index.html')==sha(folders[1]/'index.html')
labels=['B两类子项替换（下列F诊断的共同基线）','在上述基础上，仅换F的FC1＋FC2','在上述基础上，仅换F的激活／门控','在上述基础上，换F的FC1＋激活／门控＋FC2']
rows=[[label,f'{s["window_ms"]/1000:.3f}',f'{s["error_pct"]:+.2f}%'] for label,s in zip(labels,r['scenarios'])]
script='''(()=>{
const t=[...document.querySelectorAll('#overestimate table')].find(t=>t.querySelector('th')?.textContent==='诊断条件');if(!t)throw Error('diagnostic table not found');
t.id='diagnostic-conditions';const body=t.tBodies[0]||t;
const sep=document.createElement('tr');sep.innerHTML='<td colspan="3"><strong>新增：保留其他成本，只替换专家计算子项（iter60诊断）</strong></td>';body.append(sep);
for(const values of ROWS){const tr=document.createElement('tr');tr.dataset.fExpert='r1';for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}body.append(tr);}
const note=document.createElement('p');note.id='f-expert-diagnosis-note';note.className='small';note.textContent='新增四行均已固定替换中间B的两类子项：专家计算（含重计算与真反向FC及激活／门控）＋已识别token重排／EP活动。各F条件均相对第一行共同基线，不是依次累加。F的Attention、CP、EP、token重排、等待及外层依赖保持不变；不是整块F替换。F完整专家计算额外减少272.5ms，由高估168.8ms变为低估103.7ms（−0.51%）。代表rank目标实测替换诊断，非独立预测精度。';t.after(note);
const link=document.createElement('a');link.href='f-expert-notes.md';link.textContent='F专家子项诊断：证据与复现';note.after(link);
})();'''.replace('ROWS',json.dumps(rows,ensure_ascii=False))
for folder in folders:
 p=folder/'index.html';html=p.read_text();assert 'f-expert-diagnosis-note' not in html
 for name in ['index.html','manifest.json']:
  backup=folder/name.replace('.', '-before-f-expert-table.',1);assert not backup.exists();shutil.copyfile(folder/name,backup)
 p.write_text(html.replace('</body>','<script>'+script+'</script></body>'))
 for source,name in [(src,'f-expert-report.json'),(src.with_name('manifest.json'),'f-expert-manifest.json'),(ROOT/'docs/data-foundation/F_EXPERT_TRANSFER_R1.md','f-expert-notes.md')]:shutil.copyfile(source,folder/name)
 manifest=dict(version='prediction-overview-r2-f-expert-table-r1',inputs={str(src):sha(src)},code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PASS both copies updated; old table rows preserved')

"""Keep one requested conclusion after the whole-1F1B diagnosis."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
folders=[ROOT/'results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(folders[0]/'index.html')==sha(folders[1]/'index.html')
sentence='当前1F1B外推偏差主要与MoE相关成本迁移有关。按PP规模重复源stage时，虽然保留了相同计算结构，却没有表达逐层专家路由带来的实际工作量与负载差异，因此固定成本模板出现系统性高估。'
script='''(()=>{
const old=document.getElementById('extrapolation-conclusion'),table=document.getElementById('diagnostic-conditions');if(!old||!table)throw Error('missing conclusion or 1F1B table');
const p=document.createElement('p');p.id='extrapolation-conclusion';p.style.cssText='padding:14px 18px;border-left:4px solid #2563eb;background:#eef5ff;margin:18px 0';p.textContent=TEXT;
old.remove();(table.closest('.scroll')||table).after(p);
})();'''.replace('TEXT',json.dumps(sentence,ensure_ascii=False))
for folder in folders:
 p=folder/'index.html';html=p.read_text();assert "old.remove();(table.closest" not in html
 for name in ['index.html','manifest.json']:
  dest=folder/name.replace('.', '-before-short-conclusion.',1);assert not dest.exists();shutil.copyfile(folder/name,dest)
 p.write_text(html.replace('</body>','<script>'+script+'</script></body>'))
 manifest=dict(version='prediction-overview-r2-short-conclusion-r1',code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(q.resolve()):sha(q) for q in folder.rglob('*') if q.is_file() and q!=folder/'manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PASS short conclusion placed after whole-1F1B table; detailed docs retained')

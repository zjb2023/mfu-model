"""Present whole-block diagnosis before the finer B/F expert substitution."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
folders=[BASE/'2111-blocks-ui-r1/df-v001/prediction-overview-r2',Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(folders[0]/'index.html')==sha(folders[1]/'index.html')
script='''(()=>{
const original=document.getElementById('diagnostic-conditions'),box=document.getElementById('b-evidence'),panel=document.getElementById('moe-attribution-r1');
if(!original||!box||!panel)throw Error('missing diagnosis sections');
const wrapper=original.closest('.scroll')||original;wrapper.after(box);
const heading=document.createElement('h3');heading.textContent='F专家计算子项：固定B两类替换后进一步诊断';heading.id='f-expert-conditions';
const wrap=document.createElement('div');wrap.style.overflowX='auto';const table=document.createElement('table');table.id='f-expert-conditions-table';table.append(original.tHead.cloneNode(true));const body=document.createElement('tbody');table.append(body);wrap.append(table);
for(const row of [...original.querySelectorAll('tr[data-f-expert]')])body.append(row);
for(const row of [...original.querySelectorAll('tr')])if(row.querySelector('td[colspan="3"]'))row.remove();
const note=document.getElementById('f-expert-diagnosis-note'),link=note.nextElementSibling;
const summary=panel.querySelector('details');summary.before(heading,wrap,note);if(link&&link.tagName==='A')note.after(link);
panel.querySelector('h2').textContent='进一步拆解B与F：专家计算子项替换后的误差';
const over=document.getElementById('overestimate');over.querySelector('h2').textContent='1F1B整体误差 → B/F专家计算子项诊断';
})();'''
for folder in folders:
 p=folder/'index.html';html=p.read_text();assert 'f-expert-conditions-table' not in html
 for name in ['index.html','manifest.json']:
  backup=folder/name.replace('.', '-before-diagnosis-reorder.',1);assert not backup.exists();shutil.copyfile(folder/name,backup)
 p.write_text(html.replace('</body>','<script>'+script+'</script></body>'))
 manifest=dict(version='prediction-overview-r2-whole-before-expert-r1',code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PASS reordered both copies with backups; numerical payload unchanged')

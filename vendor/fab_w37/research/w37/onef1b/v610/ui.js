const datasets=JSON.parse(document.getElementById('report-data').textContent);
const state={};
function textCell(row,value){const cell=document.createElement('td');cell.textContent=value===null?'未提供':String(value);row.append(cell)}
function showTable(id){const spec=datasets[id],box=document.getElementById(id);const q=(box.querySelector('input')?.value||'').toLowerCase();const filtered=spec.rows.filter(row=>!q||row.some(v=>String(v).toLowerCase().includes(q)));const pages=Math.max(1,Math.ceil(filtered.length/20));state[id]=Math.min(state[id]||0,pages-1);const body=box.querySelector('tbody');body.replaceChildren();for(const row of filtered.slice(state[id]*20,(state[id]+1)*20)){const tr=document.createElement('tr');for(const value of row)textCell(tr,value);body.append(tr)}box.querySelector('.table-status').textContent=`共 ${filtered.length} 行，第 ${state[id]+1} / ${pages} 页（每页最多 20 行）`;box.querySelector('[data-prev]').disabled=state[id]===0;box.querySelector('[data-next]').disabled=state[id]>=pages-1;}
for(const id of Object.keys(datasets)){const box=document.getElementById(id);box.querySelector('input').addEventListener('input',()=>{state[id]=0;showTable(id)});box.querySelector('[data-prev]').addEventListener('click',()=>{state[id]--;showTable(id)});box.querySelector('[data-next]').addEventListener('click',()=>{state[id]++;showTable(id)});showTable(id)}
document.getElementById('expand-content').addEventListener('click',()=>document.querySelectorAll('details').forEach(el=>el.open=true));document.getElementById('collapse-content').addEventListener('click',()=>document.querySelectorAll('details').forEach(el=>el.open=false));
document.getElementById('download-html').addEventListener('click',()=>{const clone=document.documentElement.cloneNode(true);const blob=new Blob(['<!doctype html>\n'+clone.outerHTML],{type:'text/html;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v6.10_计算图MFU预测模型.html';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000)});

// Old and new deep links reveal any enclosing evidence panels, online or offline.
function revealAnchor(){
  let id;try{id=decodeURIComponent(location.hash.slice(1))}catch{return}
  const target=document.getElementById(id);if(!target)return;
  if(id==='three-case-parameters')target.querySelector('details').open=true;
  for(let p=target.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;
  requestAnimationFrame(()=>target.scrollIntoView({block:'start',behavior:'instant'}));
}
addEventListener('hashchange',routeAnchor);
document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(a&&a.hash===location.hash)revealAnchor()});
const oldAnchors=['baseline','weekend','scaleout','tool-todo','development','diagnosis','summary','terms','three-case-parameters'];
function routeAnchor(){
  if(oldAnchors.includes(location.hash.slice(1)))location.replace('http://192.168.0.49:8037/w37-report-history.html'+location.hash);
  else if(location.hash)revealAnchor();
}
routeAnchor();

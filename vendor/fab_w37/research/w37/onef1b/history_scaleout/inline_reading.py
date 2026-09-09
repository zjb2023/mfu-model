"""Keep short views inline; open long teaching and parameter atlases as returnable pages."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin,urlsplit
import html,json
ROOT=Path(__file__).resolve().parents[4]
ORIGIN='http://192.168.0.49:8037'
DEMO='/docs/w37/1f1b/DAG_4GPU_DEMO.html'
ATLAS='/docs/w37/1f1b/PARAMETER_ITER_ATLAS.html'
METHODS='/docs/w37/1f1b/history_scaleout_20260908/methods.html'
LEGACY='/results/w37/A/integration-20260907/context/mfu/case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v4_oisa_s5000_slice256_buffer128/'
MAP={METHODS:'scaleout',LEGACY+'pp_optimizer_dag_v4_oisa_s5000.html':'legacy-256-inline',LEGACY+'slice_pipeline_principle.html':'legacy-ring-inline','/research.html':'development','/v685.html':'results','/v684.html':'baseline','/w37-report.html':'results','/w37-16to256.html':'scaleout'}
class Rewrite(HTMLParser):
 def __init__(self,text,base,child=False,frames=None):
  super().__init__(convert_charrefs=False);self.text=text;self.base=base;self.child=child;self.frames=frames or {};self.changes=[];self.offsets=[0]
  for line in text.splitlines(keepends=True):self.offsets.append(self.offsets[-1]+len(line))
  self.feed(text)
 def handle_starttag(self,tag,attrs):
  a=dict(attrs);changed=False
  if tag=='a' and 'href' in a:
   href=a['href'];u=urlsplit(urljoin(self.base,href))
   if not href.startswith(('#','data:','blob:','javascript:')):
    local=u.hostname in ['192.168.0.49','127.0.0.1','localhost']
    target=None
    if local and u.path=='/w37-report-history.html':target=u.fragment or 'training'
    elif local and u.path in MAP:target=MAP[u.path]
    elif local and u.path in [DEMO,ATLAS]:
     a['href']=u.path+('#'+u.fragment if u.fragment else '');a.pop('target',None);changed=True
    elif local and u.path.endswith('.html'):raise ValueError('Unmapped reading link '+href+' in '+self.base)
    if target:
     a['href']='#'+target
     if self.child:a['target']='_parent'
     else:a.pop('target',None)
     changed=True
    elif self.child and not u.scheme.startswith('data'):
     a['href']=urljoin(self.base,href);changed=True
  if tag=='iframe' and a.get('src') in self.frames:
   a['srcdoc']=self.frames[a.pop('src')];changed=True
  if self.child and tag in ['img','script','link']:
   key='href' if tag=='link' else 'src'
   if a.get(key) and not a[key].startswith(('data:','http:','https:','#')):
    a[key]=urljoin(self.base,a[key]);changed=True
  if changed:
   start=self.offsets[self.getpos()[0]-1]+self.getpos()[1];raw=self.get_starttag_text()
   self.changes.append((start,start+len(raw),'<'+tag+''.join(' '+k+('="'+html.escape(v,quote=True)+'"' if v is not None else '') for k,v in a.items())+(' />' if raw.endswith('/>') else '>')))
 def result(self):
  out=self.text
  for a,b,value in reversed(self.changes):out=out[:a]+value+out[b:]
  return out

def integrate(document,out):
 paths=[METHODS,LEGACY+'pp_optimizer_dag_v4_oisa_s5000.html',LEGACY+'slice_pipeline_principle.html']
 frames={p:Rewrite((ROOT/p.lstrip('/')).read_text(),ORIGIN+p,True).result() for p in paths}
 # srcdoc inherits the parent base URL: handle child-local anchors without navigating the frame to the parent HTML.
 local_anchor_script = """<script>document.addEventListener('click',function(e){const a=e.target.closest('a[href^="#"]');if(!a||a.target==='_parent'||a.target==='_top')return;let id;try{id=decodeURIComponent(a.getAttribute('href').slice(1))}catch{return}const n=document.getElementById(id);if(!n)return;e.preventDefault();if(n.tagName==='DETAILS')n.open=true;for(let p=n.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;window.scrollTo({top:Math.max(0,n.getBoundingClientRect().top+window.scrollY-90),behavior:'instant'});});</script>"""
 frames={p:(v.replace('</body>',local_anchor_script+'</body>') if '</body>' in v else v+local_anchor_script) for p,v in frames.items()}
 frames[METHODS]=frames[METHODS].replace('打开来源页面','返回本页16→256章节')
 document=Rewrite(document,ORIGIN+'/w37-report-history.html',False,frames).result()
 def embed(id,title,path,height):
  return '<details id="'+id+'" class="inline-viewer"><summary>'+title+'</summary><p>内容已内嵌当前HTML，无需另开页面。框内可独立滚动和交互。</p><iframe title="'+title+'" loading="lazy" style="width:100%;height:'+str(height)+'px;border:1px solid #d4e1ea;border-radius:12px" srcdoc="'+html.escape(frames[path],quote=True)+'"></iframe></details>'
 marker='<section id="why-dag"><h2>02 · 如何预测训练时间与 MFU</h2>'
 assert document.count(marker)==1
 document=document.replace(marker,marker+'<div id="teaching-dag-inline" class="subsection"><p><a href="'+DEMO+'">打开独立教学图：PP4、PP2×TP2、EP2×TP1</a> · 图中可返回本章。</p></div>')
 marker='<section id="development"><h2>05 · 模型开发路线与版本增量</h2>'
 assert document.count(marker)==1
 document=document.replace(marker,marker+embed('legacy-256-inline','本页查阅：历史256卡完整DAG流程',paths[1],1000)+embed('legacy-ring-inline','本页查阅：通信分片与流水原理',paths[2],1000))
 document=document.replace("const target=document.getElementById(id);if(!target)return;", "const target=document.getElementById(id);if(!target)return;if(target.tagName==='DETAILS')target.open=true;")
 document=document.replace('</head>','<style>section[id],details[id],.subsection[id]{scroll-margin-top:90px}iframe{scroll-margin-top:110px}</style></head>')
 document=document.replace('单独打开完整16→256图文报告','在本节查看完整16→256图文报告')
 document=document.replace('单独打开可获得更大阅读区域。','完整内容已内嵌本页。')
 (out/'inline_reading_manifest.json').write_text(json.dumps({'embedded_html_paths':paths,'reading_link_targets':MAP,'standalone_reading_paths':[DEMO,ATLAS],'policy':'Teaching and parameter atlases open standalone with return links; three other views remain inline; evidence links retained.'},ensure_ascii=False,indent=2)+'\n')
 return document

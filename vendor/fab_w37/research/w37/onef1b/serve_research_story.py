from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlsplit, urljoin
import os
import re
import html
import sys
ROOT=Path('/home/zjb/Desktop/worktrees/fab-w37-1f1b')
OUT=ROOT/'results/w37/A/research-html-20260907'
sys.path.insert(0,str(OUT/'python-deps'))
import markdown
OLD=Path('/home/zjb/Desktop/fabric-data-analysis/case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_v685_source_steady_calibration_256_to_224/evaluator_only/dag_v685_source_steady_calibration.html')
class Handler(SimpleHTTPRequestHandler):
 def guess_type(self, path):
  kind=super().guess_type(path)
  if str(path).lower().endswith('.md'):
   return 'text/plain; charset=utf-8'
  if kind.startswith('text/') or kind in ('application/json','application/javascript','image/svg+xml'):
   return kind+'; charset=utf-8'
  return kind
 def end_headers(self):
  self.send_header('Cache-Control','no-store, max-age=0')
  super().end_headers()

 def do_GET(self):
  path=unquote(urlsplit(self.path).path)
  if path in ('/w37-report.html','/v610.html'):
   b=(ROOT/'results/w37/A/v610-release/render/index.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if path=='/w37-report-history.html':
   updated=ROOT/'results/w37/A/history-scaleout-20260908/integrated_report.html'
   b=(updated if updated.is_file() else ROOT/'results/w37/A/integration-20260907/integrated_report.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if path in ('/','/index.html','/research.html'):
   b=(ROOT/'results/w37/A/v610-release/render/research-index.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if path in tuple(prefix + name for prefix in ('/docs/w37/1f1b/post685/delivery/', '/results/w37/A/post685-delivery-round2-final/') for name in ('causal_contract.svg', 'schedule_regions.svg', 'iteration_regression.svg', 'internal_attribution.svg', 'pp_readiness_evidence.svg', 'timeline_overview.svg')):
   b=(OUT/Path(path).name).read_bytes();self.send_response(200);self.send_header('Content-Type','image/svg+xml; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if path=='/lane-viewer.html' or (path=='/docs/w37/1f1b/post685/delivery/index.html' and 'legacy=1' not in self.path):
   b=(OUT/('lane-viewer.html' if path=='/lane-viewer.html' else 'post685-index.html')).read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if path in ('/research-history.html','/results/w37/A/research-html-20260907/','/results/w37/A/research-html-20260907/index.html','/docs/w37/1f1b/research_story.html'):
   data=(OUT/'index.html').read_text().replace(os.path.relpath(OLD,OUT),'/v685.html')
   # Preserve fragment navigation on this page; root only file links.
   data=re.sub(r'href="([^"]+)"', lambda m: 'href="'+(m.group(1) if m.group(1).startswith('#') else urljoin('/results/w37/A/research-html-20260907/',m.group(1)))+'"', data)
   b=data.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if path=='/v684.html':
   p=OLD.parents[2]/'dag_v684_blocking_pp_program_order_256_to_224/evaluator_only/dag_v684_blocking_pp_program_order.html'
   b=p.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if path in ('/v685.html','/v685-original.html'):
   b=((OUT/'v685-explained.html').read_bytes() if path=='/v685.html' else OLD.read_bytes());self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if path=='/baseline.html': path='/docs/w37/BASELINE.md'
  p=(ROOT/path.lstrip('/')).resolve()
  if not any(p.is_relative_to(ROOT/x) for x in ('docs/w37','results/w37','research/w37')) or not p.is_file():
   self.send_error(404);return
  if p.suffix.lower()=='.md':
   text=p.read_text(encoding='utf-8')
   # Use the requested document URL, not its resolved symlink target, for relative links.
   parent=path.rsplit('/',1)[0]+'/'
   content=markdown.markdown(text,extensions=['extra','toc','sane_lists'],extension_configs={'toc':{'permalink':True}})
   def rewrite(m):
    target=html.unescape(m.group(2))
    if target.startswith(str(ROOT)+'/'): target='/'+target[len(str(ROOT))+1:]
    elif not target.startswith(('#','http://','https://','/','mailto:')): target=urljoin(parent,target)
    return m.group(1)+'="'+html.escape(target,quote=True)+'"'
   content=re.sub(r'(href|src)="([^"]*)"',rewrite,content)
   content=re.sub(r'(<table>.*?</table>)',r'<div class="table-scroll">\1</div>',content,flags=re.S)
   css='body{max-width:1080px;margin:32px auto;padding:0 24px;font:16px/1.85 system-ui,sans-serif;color:#192b3c;background:#f7f9fc}main{background:white;padding:32px;border-radius:12px;overflow-wrap:anywhere}h1{font-size:30px;line-height:1.4}h2{border-bottom:1px solid #ddd;padding-bottom:8px;color:#135b77}h3{color:#135b77}a{color:#007b92}nav{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}pre{overflow:auto;background:#eef3f7;padding:18px;border-radius:8px;font:14px/1.65 monospace;white-space:pre}code{background:#eef3f7;padding:2px 4px}pre code{padding:0}table{border-collapse:collapse;min-width:100%;font-size:14px}td,th{padding:10px 14px;border:1px solid #dce4ec;text-align:left;min-width:100px}th{background:#eaf2f5}.table-scroll{overflow:auto}blockquote{border-left:4px solid #18849a;margin:20px 0;padding:8px 20px;background:#f0f6f9}img,svg{max-width:100%}.headerlink{font-size:14px;padding-left:8px}li{margin:6px 0}@media(max-width:600px){body{padding:0 10px}main{padding:18px}h1{font-size:24px}}'
   document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(p.name)+'</title><style>'+css+'</style></head><body><nav><a href="/research.html">研究路线</a><a href="/docs/w37/1f1b/WEB_INDEX.md">文档索引</a><a href="/v685.html">v685 原版</a></nav><main>'+content+'</main></body></html>'
   # ASCII transport with numeric character references prevents legacy charset misdecoding.
   b=document.encode('ascii',errors='xmlcharrefreplace')
   self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  super().do_GET()
 def translate_path(self,path):return str(ROOT/unquote(urlsplit(path).path).lstrip('/'))
 def list_directory(self,path):self.send_error(404);return None
if __name__=='__main__':
 ThreadingHTTPServer(('0.0.0.0',8037),Handler).serve_forever()

"""Serve relocated v610 + frozen historical reports without touching port 8037.

Stored HTML is untouched. Only delivery links are rewritten to this origin.
"""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit, urljoin
import markdown
from x10000_analysis.v610 import ROOT, VENDOR, artifact_root, manifest, resolve_artifact, dump

OLD = '/home/zjb/Desktop/worktrees/fab-w37-1f1b'
FAB = '/home/zjb/Desktop/fabric-data-analysis'
HISTORY = '/results/w37/A/history-scaleout-20260908/integrated_report.html'


def serve(run, host, port):
    m, base = manifest(), artifact_root()
    aliases = {
        '/w37-report-history.html': OLD+HISTORY,
        '/research-history.html': OLD+'/results/w37/A/research-html-20260907/index.html',
        '/v685.html': OLD+'/results/w37/A/research-html-20260907/v685-explained.html',
        '/v685-original.html': FAB+'/case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_v685_source_steady_calibration_256_to_224/evaluator_only/dag_v685_source_steady_calibration.html',
        '/v684.html': FAB+'/case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_v684_blocking_pp_program_order_256_to_224/evaluator_only/dag_v684_blocking_pp_program_order.html',
    }
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = unquote(urlsplit(self.path).path)
            if '..' in Path(url).parts:
                self.send_error(400); return
            if url in ('/','/research.html','/index.html'):
                path = run/'render/research-index.html'
            elif url in ('/w37-report.html','/v610.html'):
                path = run/'render/index.html'
            elif url.startswith('/results/w37/A/v610-release/'):
                path = run/url.split('/results/w37/A/v610-release/',1)[1]
            else:
                logical = aliases.get(url, OLD+url)
                if logical in m['logical_paths']:
                    path = resolve_artifact(logical,m,base)
                elif url.startswith('/docs/w37/') and (VENDOR/url.lstrip('/')).is_file():
                    path = VENDOR/url.lstrip('/')
                else:
                    self.send_error(404); return
            if not path.is_file():
                self.send_error(404); return
            data = path.read_bytes()
            kind = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            if path.suffix in ('.html','.md'):
                doc = data.decode('utf-8')
                doc = doc.replace('http://192.168.0.49:8037','')
                doc = doc.replace(OLD+'/', '/')
                if path.suffix == '.md':
                    body = markdown.markdown(doc,extensions=['extra','toc','sane_lists'])
                    body = re.sub(r'(href|src)="([^"]*)"',
                        lambda x:x[1]+'="'+html.escape(urljoin(url,x[2]),quote=True)+'"',body)
                    doc = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>body{max-width:1100px;margin:30px auto;font:16px/1.7 system-ui;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}pre{overflow:auto}</style><nav><a href="/research.html">mfu-model v6.10</a></nav>'+body+'</html>'
                elif url in ('/w37-report.html','/v610.html'):
                    doc = doc.replace('/home/zjb/Desktop/fabric-data-analysis/.snakemake-venv/bin/python', '.venv/bin/python')
                    doc = doc.replace('research/w37/onef1b/v610/Snakefile', 'workflow/v610/Snakefile')
                    doc = doc.replace('--directory results/w37/A/v610-release', '--config pipeline=release run_root='+str(ROOT/'results/v610/new-release'))
                data = doc.encode('utf-8'); kind = 'text/html; charset=utf-8'
            elif kind.startswith('text/') or kind in ('application/json','application/javascript','image/svg+xml'):
                kind += '; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.end_headers(); self.wfile.write(data)
    server = ThreadingHTTPServer((host,port),Handler)
    actual = server.server_port
    dump(ROOT/'results/v610/service.json',dict(host=host,port=actual,pid=__import__('os').getpid(),
        local_url=f'http://127.0.0.1:{actual}/research.html',run_root=str(run),source_service_modified=False))
    print(f'http://127.0.0.1:{actual}/research.html',flush=True)
    server.serve_forever()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=0)
    a=p.parse_args();serve(a.run_root.resolve(),a.host,a.port)

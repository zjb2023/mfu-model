"""Generate the presentation script from reviewed Markdown; no model execution."""
from pathlib import Path
import html
import re
import sys
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'results/w37/A/research-html-20260907/python-deps'))
import markdown

DOC = ROOT / 'docs/w37/1f1b'
md = markdown.Markdown(extensions=['extra', 'toc', 'sane_lists'])
body = md.convert((DOC / 'TALK_MFU_EXTRAPOLATION.md').read_text())
def link(m):
    target = html.unescape(m[1])
    if not target.startswith(('#', '/', 'http:', 'https:', 'mailto:')):
        target = urljoin('/docs/w37/1f1b/', target)
    return 'href="'+html.escape(target, quote=True)+'"'
body = re.sub(r'href="([^"]+)"', link, body)
body = body.replace('<table>', '<div class="scroll"><table>').replace('</table>', '</table></div>')
page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MFU外推模型 · 大白话讲稿与演示路线</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f6f8;color:#193447;font:17px/1.95 system-ui,"Microsoft YaHei",sans-serif}header,main{max-width:1080px;margin:auto;padding:26px}header{display:flex;gap:18px;flex-wrap:wrap;align-items:center}a{color:#00767d;text-underline-offset:4px}main{background:white;border-radius:14px;margin-bottom:32px}h1{font-size:32px;line-height:1.4}h2{font-size:24px;margin-top:40px;padding-top:22px;border-top:1px solid #dce6eb;line-height:1.5}p{margin:14px 0}strong{color:#095f67}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;min-width:760px;width:100%}th,td{padding:12px;border:1px solid #dce6eb;vertical-align:top;text-align:left}th{background:#edf4f6}code{background:#f0f4f7;overflow-wrap:anywhere}button{font:inherit;border:1px solid #91aebd;background:white;color:#174d5f;border-radius:7px;padding:6px 14px;cursor:pointer}details{background:#f0f7f8;padding:14px 22px;border-radius:10px;margin:20px 0}summary{cursor:pointer;font-weight:650}.toc ul{padding-left:24px}li{margin:7px 0}blockquote{padding:10px 18px;background:#edf7f5;border-left:4px solid #168c83}@media(max-width:650px){header,main{padding:18px}h1{font-size:26px}h2{font-size:21px}body{font-size:16px}}@media print{header,details{display:none}body{background:white;font-size:12pt}main{padding:0;max-width:none}h2{break-after:avoid}p{orphans:3;widows:3}a{color:inherit}table{min-width:0;font-size:10pt}.scroll{overflow:visible}}
</style></head><body><header><a href="/docs/w37/1f1b/DAG_4GPU_DEMO.html">4卡教学DAG</a><a href="/w37-report.html">当前模型报告</a><a href="/w37-report-history.html#scaleout">16→256案例</a><a href="/docs/w37/1f1b/EXTRAPOLATION_EXPERIMENT_PLAN.html#sixteen">实验方案</a><a href="/docs/w37/1f1b/TALK_MFU_EXTRAPOLATION.md">Markdown讲稿</a><button id="print" onclick="window.print()">打印 / 保存PDF</button></header><main><details><summary>讲稿目录：五站演示、完整口述、3分钟版与答疑</summary>'''+md.toc+'''</details>'''+body+'''</main></body></html>'''
(DOC / 'TALK_MFU_EXTRAPOLATION.html').write_text(page)
print('Rendered reviewed talk script HTML')

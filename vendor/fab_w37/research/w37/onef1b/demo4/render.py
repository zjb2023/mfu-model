"""Build a standalone four-GPU teaching page from the existing F/B order factory."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
source_path = ROOT / 'workflow/scripts/build_dag_mfu_schedule_factorial.py'
source = source_path.read_text()
function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), namespace)
orders = [namespace['schedule'](s, 4, 2) for s in range(4)]
payload = {'orders': orders, 'schedule_source': str(source_path.relative_to(ROOT)), 'schedule_sha256': hashlib.sha256(source.encode()).hexdigest()}
page = (HERE / 'page.html').read_text().replace('__MODEL_DATA__', json.dumps(payload, ensure_ascii=False))
page = page.replace('<section id="historical-result">', (HERE/'detail.html').read_text() + (HERE/'ep.html').read_text() + '<section id="historical-result">')
page = page.replace('</body>', '<script>window.tpOrders='+json.dumps([namespace['schedule'](s,2,2) for s in range(2)])+';'+(HERE/'detail.js').read_text()+(HERE/'ep.js').read_text()+'</script></body>')
dest = ROOT / 'docs/w37/1f1b/DAG_4GPU_DEMO.html'
dest.write_text(page)
print('Rendered 4-GPU teaching demo; F/B order read from existing schedule() only.')

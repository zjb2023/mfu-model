"""Render a read-only, hash-pinned SimAI documentation snapshot and verify its arithmetic."""
import csv, hashlib, html, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/w37/1f1b/mlperf_deepseek_20260909'
def section():
    manifest=json.loads((DOC/'source_manifest.json').read_text())
    for f in manifest['files']:
        raw=(DOC/f['name']).read_bytes()
        assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'], 'MLPerf snapshot changed: '+f['name']
    recipes={r['id']:r for r in csv.DictReader((DOC/'recipes.csv').open(encoding='utf-8-sig'))}
    results={r['recipe']:r for r in json.loads((DOC/'deepseek_v3_mfu.json').read_text())['results']}
    rows=[]
    for id in ['R10','R12','R11','R13']:
        r=recipes[id];num=lambda k:int(r[k])
        assert num('world_size')==num('tp')*num('pp')*num('cp')*num('dp')
        assert num('gbs')==num('mbs')*num('dp')*num('gradient_accumulation')
        values=[id,r['accelerator'],r['world_size'],r['tp'],r['pp'],r['ep'],r['dp'],r['gbs'],r['gradient_accumulation']]
        if id in results:
            a=results[id];assert a['gpus']==num('world_size') and a['gbs']==num('gbs')
            rate=a['model_tflop_per_sample']*a['gbs']/a['gpus']/a['mean_step_seconds']
            assert abs(rate-a['achieved_model_tflops_per_gpu'])<1e-8
            assert abs(rate/a['peak_dense_fp8_tflops_per_gpu']*100-a['estimated_mfu_percent'])<1e-10
            values += [str(a['recorded_steps']),f"{a['mean_step_seconds']:.6f}",f"{rate:.2f}",f"{a['estimated_mfu_percent']:.2f}%"]
        else:values+=['未匹配','—','—','—']
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    template=(Path(__file__).parent/'mlperf_todo.html').read_text()
    assert template.count('__RECIPE_ROWS__')==1
    return template.replace('__RECIPE_ROWS__',''.join(rows))

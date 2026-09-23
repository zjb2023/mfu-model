"""Read-only 32GPU comparison using the exact 256GPU anchor/conservation method."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import extract_layer_iteration_grid_r1 as shared

RAW=Path('/home/zjb/Desktop/32/gpu32_gbs64_framework/2026-09-18-10:38')
OUT=shared.BASE/'layer32-iteration-grid-r1'

def main():
    OUT.mkdir(exist_ok=True)
    paths={}
    for it in shared.ITERATIONS:
        paths[str(it)]={}
        for p in RAW.glob(f'*/profiler/iteration_{it}/rank*.pt.trace.json'):
            rank=p.name.split('.')[0][4:]
            assert rank not in paths[str(it)]
            paths[str(it)][rank]=str(p)
        assert all(str(r) in paths[str(it)] for r in range(8,24))
    tasks=[(s,0,paths,dict(stage_width=8,microbatches=8,out=str(OUT))) for s in (1,2)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(shared.process,t) for t in tasks]):print(*f.result(),flush=True)
    parts=[json.loads((OUT/f'pp{s}-group0.json').read_text()) for s in (1,2)]
    rows=[r for p in parts for r in p['rows']];assert len(rows)==576
    report=dict(status='PASS',iterations=shared.ITERATIONS,rows=rows,representative_trace_files=18,expert_conservation_checks=len(rows)*160,scope='32GPU PP1 L3-L6 and PP2 L7-L10, single EP8 per stage, 8 microbatches; representative trace anchors plus all-eight-rank DeepEP logs; forward only')
    (OUT/'report.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    inputs={k:v for p in parts for k,v in p['inputs'].items()}
    for p in (Path(__file__),Path(shared.__file__)):inputs[str(p.resolve())]=shared.sha(p)
    (OUT/'manifest.json').write_text(json.dumps(dict(inputs=inputs,outputs={p.name:shared.sha(p) for p in OUT.glob('*.json') if p.name!='manifest.json'}),indent=2)+'\n')
    print('PASS',len(rows),'rows')
if __name__=='__main__':main()

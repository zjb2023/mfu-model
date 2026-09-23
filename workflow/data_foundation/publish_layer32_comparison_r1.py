"""Append an idempotent source32 section, preserving all earlier sections."""
import json,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    src=BASE/'layer32-iteration-grid-r1';d=json.loads((src/'report.json').read_text());assert d['status']=='PASS'
    target=BASE/'layer-iteration-analysis-r1/report.json';t=json.loads(target.read_text())
    dest=BASE/'2111-blocks-ui-r1/df-v001/layer-workload-r1'
    indexed={(r['iteration'],r['mb'],r['layer']):r for r in d['rows']}
    down=sum(indexed[it,mb,10]['retained']<indexed[it,mb,3]['retained'] for it in d['iterations'] for mb in range(4))
    up=sum(indexed[80,mb,l]['retained']>indexed[40,mb,l]['retained'] for mb in range(4) for l in range(3,11))
    summary=f'<p><strong>32卡实测：</strong>在与256卡共有的MB0–3范围，L10条目少于L3的有{down}/36组；固定层与MB，iter80条目多于iter40的有{up}/32个位置。这里只报告端点，不将其称为逐层或逐轮单调趋势，也不能据此推断L11–L58。</p>'
    sample=[indexed[60,0,l] for l in (3,6,7,10)]
    summary+='<p>32卡iter60、MB0示例：'+'；'.join(f"L{r['layer']}：{r['retained']:,}条 / {r['active_experts']}个专家" for r in sample)+'。</p>'
    snippet=(ROOT/'workflow/data_foundation/ui_layer32_comparison_r1.html').read_text().replace('__SUMMARY32__',summary).replace('__DATA32__',json.dumps(d,separators=(',',':'))).replace('__DATA256__',json.dumps(t['rows'],separators=(',',':')))
    section,script=snippet.split('<script id="source32-script">',1)
    html=(dest/'index.html').read_text()
    html=re.sub(r'<section id="source32-comparison">.*?</section>','',html,flags=re.S)
    html=re.sub(r'<script id="source32-script">.*?</script>','',html,flags=re.S)
    pos=html.rfind('</section>')+len('</section>');assert pos>len('</section>')
    html=html[:pos]+section+html[pos:]
    html=html.replace('</html>','<script id="source32-script">'+script+'</html>')
    (dest/'index.html').write_text(html)
    doc=f'''# 32GPU layer workload comparison r1

Scope: source32 PP1 L3–L6 / PP2 L7–L10, iterations {d['iterations']}, all 8 microbatches. The public comparison uses MB0–3 only, matching the 256GPU study. Forward work only; not a new timing prediction.

One EP8 group per source stage versus two independent EP8 groups per target stage. Compare source with each target group, never with their sum. Same iteration IDs are not matched batches/checkpoints across runs. Same layer numbering is not proof of identical weights or activations.

18 representative traces (rank8 and rank16 across 9 iterations) locate four CPU CheckpointFunction layers per forward step. Same-rank timestamps anchor DeepEP calls. All eight ranks' sender counts sum exactly to the 160 receiver-expert counts, and representative FC split counts match the receiver log. {d['expert_conservation_checks']} per-expert conservation checks PASS. Nonrepresentative FC traces were not re-parsed. Raw inputs remain read-only; absolute paths, hashes, event indices and log line numbers are in the extraction files and manifest.

Shared-MB endpoint comparisons: L10<L3 in {down}/36 cases; iter80>iter40 at {up}/32 layer×MB positions. These are endpoint observations, not monotonicity or causal tests. Source depth cannot establish the long target depth trend.

Reproduce from the mfu-16to256 worktree:

```
/usr/bin/python3 -S workflow/data_foundation/extract_layer32_iteration_grid_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer32_comparison_r1.py
```

The extractor reuses verified cached per-stage JSON when present. Publish after existing layer-workload/pair/nine-iteration publishers if rebuilding the entire page. No prediction costs or frozen results were changed.
'''
    (ROOT/'docs/data-foundation/LAYER32_WORKLOAD_COMPARISON_R1.md').write_text(doc)
    for p,name in [(src/'report.json','source32-report.json'),(src/'manifest.json','source32-manifest.json'),(ROOT/'docs/data-foundation/LAYER32_WORKLOAD_COMPARISON_R1.md','source32-method.md')]: (dest/name).write_bytes(p.read_bytes())
    m=json.loads((dest/'manifest.json').read_text())
    for p in [dest/'index.html',target,Path(__file__).resolve(),ROOT/'workflow/data_foundation/ui_layer32_comparison_r1.html',*[dest/n for n in ['source32-report.json','source32-manifest.json','source32-method.md']]]:m['evidence'][str(p)]=sha(p)
    m['ui_revision']='r12: appended source32 per-EP8 workload comparison; existing sections preserved; no model changes'
    (dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(summary)
if __name__=='__main__':main()

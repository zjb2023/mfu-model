"""Add nine-iteration results to section4 after both established publishers."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def main():
    report_path=BASE/'layer-iteration-analysis-r1/report.json';d=json.loads(report_path.read_text());assert d['status']=='PASS'
    depth=d['depth_summary'];temporal=d['temporal'];down=sum(r['endpoint_decrease_pct']>0 for r in depth);up=sum(r['last']>r['first'] for r in temporal)
    mb0=[r for r in depth if r['mb']==0];first,last=mb0[0],mb0[-1]
    intro=f'''<p><strong>已验证：层位置变化与跨iter变化不能混为一谈。</strong>9个iter×4个microbatch共36组首末层对比中，L58工作量低于L3的有{down}组。固定层和microbatch，从iter40到80工作量增加的有{up}/224个位置。</p><p>例如MB0：L3两组接收量由iter40的{first['first']['retained_total']:,}条变为iter80的{last['first']['retained_total']:,}条；L58由{first['last']['retained_total']:,}条变为{last['last']['retained_total']:,}条。这是两个端点的实测变化，完整九轮曲线如下；不将端点变化当作单调趋势。</p><p><strong>证据范围：</strong>252份代表rank trace定位前向层/MB，两个EP8组的八卡日志完成645,120项逐专家发送—接收核对；iter60与已冻结结果896项回归一致。没有修改预测模型。</p>'''
    monotonic=sum(r['increase_steps']==8 for r in temporal)
    intro+=f'<p><strong>不是“训练越久，专家工作越少”。</strong>该样本中固定位置的工作量从40到80增加；但只有{monotonic}/224个位置在全部八个采样间隔中都增加，其余存在回落或持平。当前证据支持层位置差异与跨iter变化同时存在，不能将它们拟合成一个固定的层深衰减系数。</p>'
    dest=BASE/'2111-blocks-ui-r1/df-v001/layer-workload-r1';html=(dest/'index.html').read_text();assert 'const ITERDATA=' not in html,'Run base and pair publishers before this one'
    start=html.index('<section id="capacity-evidence">');end=html.index('</section>',start)+len('</section>');old=html[start:end]
    inner=old[old.index('</h2>')+5:-len('</section>')]
    historical='<details id="capacity-evidence"><summary>iter60已核对结果与容量筛选边界</summary>'+inner+'</details>'
    snippet=(ROOT/'workflow/data_foundation/ui_layer_iteration_trends_r1.html').read_text().replace('__CONCLUSIONS__',intro).replace('__HISTORICAL_EVIDENCE__',historical).replace('__ITERATION_DATA__',json.dumps(d,separators=(',',':')))
    # Defer the new script until after the existing page scripts; original validation still exists.
    section,script=snippet.split('<script>',1)
    html=html[:start]+section+html[end:];html=html.replace('</html>','<script>'+script+'</html>');(dest/'index.html').write_text(html)
    for p,name in [(ROOT/'docs/data-foundation/LAYER_ITERATION_TRENDS_R1.md','iteration-trends.md'),(report_path,'iteration-trends-report.json'),(BASE/'layer-iteration-analysis-r1/manifest.json','iteration-trends-manifest.json')]:
        (dest/name).write_bytes(p.read_bytes())
    manifest=json.loads((dest/'manifest.json').read_text())
    for p in [dest/'index.html',dest/'iteration-trends.md',dest/'iteration-trends-report.json',dest/'iteration-trends-manifest.json',Path(__file__).resolve(),ROOT/'workflow/data_foundation/ui_layer_iteration_trends_r1.html']:manifest['evidence'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest['ui_revision']='r11: section4 nine-iteration full-EP8 workload trends, layer/iteration curves and heatmap; old iter60 evidence preserved; model unchanged.'
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(dest)
if __name__=='__main__':main()

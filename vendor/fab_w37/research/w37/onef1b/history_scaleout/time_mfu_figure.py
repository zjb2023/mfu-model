"""Extend the frozen time breakdown with same-window Training-clock MFU bars."""
import base64,csv,hashlib,io,json,re
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[4]
def replace_figure(document,out):
    specs=json.loads((ROOT/'docs/w37/1f1b/integration/inputs.json').read_text())['inputs']
    def read(key):
        spec=specs[key];raw=(ROOT/spec['path']).read_bytes()
        assert len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'], 'Frozen input changed: '+key
        return raw.decode()
    svg=read('v685_svg')
    rows=[r for r in csv.DictReader(io.StringIO(read('version_iterations'))) if r['variant']=='v685_frozen' and r['split']=='development_primary']
    assert [int(r['iteration']) for r in rows]==[85,90,95,100]
    f=8.436548311982576e16;peak=224*500e12
    for r in rows:
        for clock,mfu in [('predicted_training_ms','predicted_mfu_pct'),('actual_training_ms','actual_mfu_pct_derived')]:
            assert abs(f/(float(r[clock])/1000)/peak*100-float(r[mfu]))<1e-10
    pred=mean(float(r['predicted_mfu_pct']) for r in rows);actual=mean(float(r['actual_mfu_pct_derived']) for r in rows)
    summary=dict(iterations=[85,90,95,100],predicted_training_seconds_mean=mean(float(r['predicted_training_ms'])/1000 for r in rows),actual_training_seconds_mean=mean(float(r['actual_training_ms'])/1000 for r in rows),predicted_mfu_pct_mean=pred,actual_mfu_pct_mean=actual,mean_mfu_difference_percentage_points=pred-actual,aggregation='Arithmetic mean of per-iteration MFU; not MFU computed from mean time',inputs={k:specs[k] for k in ['v685_svg','version_iterations']})
    svg=svg.replace('viewBox="0 0 1180 385"','viewBox="0 0 1180 735"').replace('height="385" fill="#0b1b2b"','height="735" fill="#0b1b2b"')
    svg=svg.replace('aria-label="v685预测与目标四轮实际均值的迭代时间分段"','aria-label="v685预测与目标四轮实际均值：整轮时间分段及MFU"')
    extra='<line x1="25" x2="1155" y1="393" y2="393" stroke="#516679"/>'
    def text(x,y,s,size=17):return f'<text x="{x}" y="{y}" fill="#e9f2fb" font-family="sans-serif" font-size="{size}">{s}</text>'
    extra+=text(25,430,'同一四轮的 MFU：以 Training 整轮时间换算，逐轮求值后取平均',22)
    for y,label,value,color in [(460,'模型预测 MFU',pred,'#36a6c9'),(525,'实测耗时换算 MFU',actual,'#dfb65e')]:
        extra+=text(25,y+28,label,17)+f'<rect x="230" y="{y}" width="{value*205}" height="38" fill="{color}"/>'+text(240+value*205,y+27,f'{value:.6f}%',20)
    for v in range(5):extra+=text(230+205*v,590,f'{v}%',14)
    extra+=text(25,628,f'MFU 均值差：预测高 {pred-actual:.6f} 个百分点；这是均值差，不是 MAPE。')
    extra+=text(25,662,'采用每轮有效 FLOPs 8.436548311982576×10¹⁶，224 卡 × 单卡 500 TFLOP/s。',16)
    extra+=text(25,695,'“实测”指用训练日志时间换算；不是 trace 直接读出的利用率。有效 FLOPs 分子未独立复核。',16)
    svg=svg.replace('</svg>',extra+'</svg>')
    out.mkdir(parents=True,exist_ok=True);(out/'v685_time_and_mfu.svg').write_text(svg);(out/'v685_time_and_mfu.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    pattern=r'<figure><h3>v685 目标 224 卡：预测与实际四轮平均总时间</h3><img[^>]*><figcaption>(.*?)</figcaption></figure>'
    match=re.search(pattern,document,re.S);assert match
    figure='<figure id="v685-time-mfu"><h3>v685 目标 224 卡：预测与实际四轮平均总时间及 MFU</h3><img loading="lazy" src="data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()+'" alt="v685四轮平均Training时间和MFU对比"><figcaption>'+match[1]+' MFU单独使用百分比轴，按85/90/95/100逐轮换算后取平均，不是用平均时间再计算MFU。</figcaption><p><a href="/results/w37/A/history-scaleout-20260908/v685_time_and_mfu.svg" download>下载时间与MFU图 SVG</a> · <a href="/results/w37/A/history-scaleout-20260908/v685_time_and_mfu.json">查看图中均值与计算口径</a></p></figure>'
    return document[:match.start()]+figure+document[match.end():]

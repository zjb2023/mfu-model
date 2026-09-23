"""Append static, evidence-derived cross-iteration diagnostic plots."""
import json,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    src=BASE/'token-window-iterations-r1';d=json.loads((src/'report.json').read_text());assert d['status']=='PASS_DIAGNOSTIC_ONLY'
    rows=d['rows'];fit=d['linear_gap_to_error'];dest=BASE/'2111-blocks-ui-r1/df-v001/layer-workload-r1'
    X=lambda x:85+x/250000*830
    Y=lambda y:335-y/2200*275
    svg=['<svg id="token-window-scatter" viewBox="0 0 1000 405" style="width:100%;display:block" role="img" aria-label="九轮token差距与1F1B耗时偏差散点图">']
    for value in range(0,2001,500):
        y=Y(value);svg.append(f'<path d="M85 {y}H915" stroke="#e2e8f0"/><text x="75" y="{y+4}" text-anchor="end">{value}</text>')
    for value in range(0,250001,50000):svg.append(f'<text x="{X(value)}" y="360" text-anchor="middle">{value/10000:g}万</text>')
    svg+=['<text x="85" y="24">纵轴：预测−实测1F1B耗时（ms）</text>','<text x="500" y="394" text-anchor="middle">横轴：32卡源模板−256卡目标平均专家token条目（单层 / MB / EP8组）</text>']
    lo=min(r['token_gap'] for r in rows);hi=max(r['token_gap'] for r in rows)
    svg.append(f'<path d="M{X(lo)} {Y(fit["intercept"]+fit["slope"]*lo)}L{X(hi)} {Y(fit["intercept"]+fit["slope"]*hi)}" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="6 5"/>')
    offsets={40:(-45,-13),45:(-45,20),50:(-52,16)}
    for r in rows:
        x=X(r['token_gap']);y=Y(r['error_ms']);dx,dy=offsets.get(r['iteration'],(8,-9));color='#d97706' if r['iteration'] in (45,50) else '#2563eb'
        svg.append(f'<circle data-iter="{r["iteration"]}" cx="{x}" cy="{y}" r="6" fill="{color}"><title>iter{r["iteration"]}：token差距{r["token_gap"]:,.1f}条；耗时偏差{r["error_ms"]:.1f}ms；相对误差{r["error_pct"]:.2f}%</title></circle><text x="{x+dx}" y="{y+dy}">iter{r["iteration"]}</text>')
    svg.append('</svg>')
    table='<div style="overflow:auto"><table id="token-window-table"><thead><tr><th>iter</th><th>目标平均条目</th><th>源−目标条目</th><th>实测1F1B（s）</th><th>预测高估（ms）</th><th>相对误差</th></tr></thead><tbody>'
    for r in rows:table+=f'<tr><td>{r["iteration"]}</td><td>{r["target_mean_tokens"]:,.0f}</td><td>{r["token_gap"]:,.0f}</td><td>{r["truth_ms"]/1000:.3f}</td><td>{r["error_ms"]:.1f}</td><td>+{r["error_pct"]:.2f}%</td></tr>'
    table+='</tbody></table></div>'
    section=f'''<section id="token-window-study">
<h2>⑥ 跨iter验证：专家工作量差距能否描述1F1B高估？</h2>
<p style="padding:14px;background:#f0f8fa"><strong>总体趋势一致：目标专家工作量越接近固定32卡模板，1F1B高估通常越小；但token量不能独自解释每轮波动。</strong>iter40→80，高估由10.07%降至0.65%，原模型和预测参数未修改。</p>
<p>源固定为32卡iter60、PP1、MB0四层平均<strong>268,536条</strong>，预测始终<strong>21.653秒</strong>。目标每轮汇总PP1–14的56层、4个MB、两组EP8，统一为<strong>单层、单MB、单EP8组均值</strong>。图中条目是筛选后前向专家输入，不是去重token数，也不是B操作量。</p>
<h3>一个点就是一个iter · 越靠右，源模板比目标多出的工作量越大</h3>
{''.join(svg)}
<p>灰色虚线为九点线性描述，橙色标出iter45、50。总体相关系数<strong>r={fit['pearson']:.3f}</strong>，R²={fit['r2']:.3f}。这描述九轮偏差的共同变化，<strong>不是“6.16%误差中93.3%已由token因果解释”</strong>。</p>
<h3>反例与检验：总体趋势强，逐轮变化仍有其他因素</h3>
<p><strong>iter45→50：</strong>token差距由228,724缩小到217,885条，高估却由7.83%增大到9.70%。八个相邻采样间隔的变化相关系数只有{d['first_difference_correlation']['pearson']:.3f}。仅用iter编号描述偏差，R²也有{d['iteration_only_control']['r2']:.3f}，所以训练进展、输入变化等共同趋势不能忽略。</p>
<p>用iter40–60五点拟合，65–80四点检验，剩余偏差平均绝对值约<strong>{d['later_mae_ms']:.0f}ms</strong>；九点留一检验约{d['loo_mae_ms']:.0f}ms，但相邻训练样本不独立。两种检验都使用<strong>目标实测token量</strong>，属于离线条件诊断，不是纯32卡外推的新精度。</p>
{table}
<p>实测窗口沿用冻结口径：rank0第一个F关联GPU开始→最后一个B关联GPU结束，含流水线填充/排空，不含启动与RS/OPT/AG；不是全256rank makespan。iter60边界与原6.16%诊断完全一致。</p>
<p><a href="token-window-method.md">分析方法与复现</a> · <a href="token-window-report.json">九轮数据与检验结果</a> · <a href="token-window-manifest.json">证据路径与SHA256</a></p>
</section>'''
    html=(dest/'index.html').read_text();html=re.sub(r'<section id="token-window-study">.*?</section>','',html,flags=re.S)
    pos=html.rfind('</section>')+len('</section>');assert pos>len('</section>');html=html[:pos]+section+html[pos:];(dest/'index.html').write_text(html)
    for p,name in [(ROOT/'docs/data-foundation/TOKEN_WINDOW_ITERATIONS_R1.md','token-window-method.md'),(src/'report.json','token-window-report.json'),(src/'manifest.json','token-window-manifest.json')]: (dest/name).write_bytes(p.read_bytes())
    m=json.loads((dest/'manifest.json').read_text());m['ui_revision']='r16: append nine-iteration token-gap versus fixed 1F1B error diagnostic, preserving chapters1–5'
    for p in [dest/'index.html',Path(__file__).resolve(),*[dest/n for n in ['token-window-method.md','token-window-report.json','token-window-manifest.json']]]:m['evidence'][str(p)]=sha(p)
    (dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(dest)
if __name__=='__main__':main()

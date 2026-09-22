"""Publish regrouped attribution into the existing overview, preserving prior UI."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
LOCAL=BASE/'2111-blocks-ui-r1/df-v001/prediction-overview-r2'
LIVE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-overview-r2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
src=BASE/'moe-two-groups-r1/report.json';r=json.loads(src.read_text())
assert sha(LOCAL/'index.html')==sha(LIVE/'index.html'),'local/live divergence'
labels=['专家计算（重计算＋真反向）','token重排／还原＋已识别EP活动','其他中间B净贡献','中间B之外残余']
colors=['#2563eb','#0d9488','#d97706','#64748b']
rows=''.join(f'<tr><td>{label}</td><td>{x["contribution_ms"]:.1f} ms</td><td>{x["percent_of_total_error"]:.2f}%</td><td>{x["error_percentage_points"]:.3f}</td></tr>' for label,x in zip(labels,r['attribution']))
bars=''.join(f'<span style="display:block;width:{x["percent_of_total_error"]}%;background:{color}" title="{label}：{x["percent_of_total_error"]:.2f}%"></span>' for label,color,x in zip(labels,colors,r['attribution']))
legend=' · '.join(f'<span style="color:{c}">{l} {x["percent_of_total_error"]:.1f}%</span>' for l,c,x in zip(labels,colors,r['attribution']))
panel=f'''<section id="moe-attribution-r1" style="padding:16px;border:2px solid #2563eb;border-radius:12px;margin-bottom:24px">
<h2>最新结论：原6.16%高估，主要来自专家计算成本复用偏高</h2>
<p><strong>原预测21.653秒，实测20.396秒，高估1.257秒。</strong>以下占比均以这1.257秒为分母，不是通信占整轮训练时间的比例。</p>
<div role="img" aria-label="原高估贡献：专家计算79.3%，数据组织和已识别EP活动7.9%，其他中间B0.7%，中间B外12.1%" style="display:flex;height:32px;border-radius:6px;overflow:hidden">{bars}</div>
<p class="small">{legend}</p>
<div style="overflow-x:auto"><table><thead><tr><th>归因类别</th><th>贡献</th><th>占原高估</th><th>误差百分点</th></tr></thead><tbody>{rows}</tbody></table></div>
<h3>① 专家计算：不再只统计FC1、FC2</h3>
<p><strong>前向重计算：</strong>FC1 → 激活／门控 → FC2。<br><strong>真正反向：</strong>FC2梯度计算 → 激活／门控反向 → FC1梯度计算。</p>
<p>反向FC计算权重梯度和输入梯度；激活／门控反向把梯度穿过SiLU、乘法等操作。重计算重新生成激活，不是在计算梯度；这些也不等于优化器更新。内部保留分项成本与依赖。</p>
<h3>② 数据组织／EP活动：保留子项，不当作纯通信</h3>
<p>包括已确认token重排／还原及其反向，以及Dispatch／Combine调用关联的独占GPU活动。<strong>7.9%不是全部EP通信的误差占比：</strong>未完整覆盖DeepEP打包解包、传输及等待其他rank；未确认部分仍在残余项中。同步／等待、纯传输和本地处理不能混算。</p>
<h3>替换诊断如何逐步缩小误差</h3>
<div style="overflow-x:auto"><table><thead><tr><th>使用目标实测成本的诊断</th><th>1F1B时长</th><th>对实测误差</th></tr></thead><tbody>
<tr><td>原冻结预测，不替换</td><td>21.653秒</td><td>+6.16%</td></tr>
<tr><td>只替换FC1／FC2（含重计算与真反向）</td><td>20.823秒</td><td>+2.09%</td></tr>
<tr><td>完整专家计算：FC＋激活／门控</td><td>20.667秒</td><td>+1.33%</td></tr>
<tr><td>再加入已识别数据组织／EP活动</td><td>20.565秒</td><td>+0.83%</td></tr>
<tr><td>中间B全部替换，作为归因回归门</td><td>20.549秒</td><td>+0.75%</td></tr>
</tbody></table></div>
<p class="small">上图是三组联合替换、六种顺序平均归因：前两项合计87.15%。直接同时替换前两项减少1.089秒、解释86.57%；两种统计因关键路径交互略有差别，不混用。其他中间B净贡献小，是正负项抵消，不代表各算子没有误差。</p>
<details><summary>FC替换后剩余2.09%：token与Attention的证据</summary><p>以剩余426.6ms为分母，确认token重排约101.5ms（23.8%），专家激活／门控约156.5ms（36.7%），合计60.5%；确认Attention非CP＋旋转编码约30.0ms（7.0%）。该Attention子集不含CP和身份未确认的投影，不能等同整个Attention。以上是FC固定后细分条件归因，不直接与上图百分比相加。</p></details>
<p class="warning"><strong>诊断边界：</strong>iter60、PP1–PP14共56个B，代表rank有效B成本替换后重调度外层DAG；不是完整EP8内图验证。目标实测已参与，+0.83%不是独立外推的新预测精度。正式预测仍为+6.16%，没有修改参数。</p>
<p><strong>原因表述：</strong>支持“32卡源样本专家成本复用偏高是主要已定位因素”。同PP1的源／目标专家输入量、GEMM次数也不同；尚未证明全部差异由路由造成，更未证明专家成本稳定呈“两头大、中间小”。当前使用32卡PP1模板，不是用256卡两头插值。</p>
<p class="small">证据更新：2026-09-22。<a href="moe-two-groups-report.json">机读归因结果</a> · <a href="moe-two-groups-manifest.json">来源与哈希</a> · <a href="moe-two-groups-notes.md">完整说明与复现</a></p>
</section>'''
script='''(()=>{const box=document.getElementById('b-evidence');if(!box)throw Error('missing b-evidence');box.insertAdjacentHTML('afterbegin',PANEL);
for(const p of document.querySelectorAll('p'))if(p.textContent.startsWith('下一步：通过R11计算/通信成本接口'))p.textContent='下一步可验证完整EP8内部成本替换与跨迭代稳定性；当前已完成代表rank的专家计算／数据组织归因，不直接拟合整体系数。';
document.querySelector('footer').textContent='2026-09-22归因更新：专家计算79.3%，已识别数据组织／EP活动7.9%。正式1F1B预测仍+6.16%；替换诊断非独立预测，目标iter60已有曝光。';})();'''.replace('PANEL',json.dumps(panel,ensure_ascii=False).replace('</','<\\/'))
files={src:'moe-two-groups-report.json',BASE/'moe-two-groups-r1/manifest.json':'moe-two-groups-manifest.json',ROOT/'docs/data-foundation/MOE_TWO_GROUPS_R1.md':'moe-two-groups-notes.md',BASE/'token-attention-r1/report.json':'token-attention-report.json'}
for folder in [LOCAL,LIVE]:
 p=folder/'index.html';html=p.read_text();assert 'id=\\"moe-attribution-r1' not in html and '最新结论：原6.16%' not in html
 for name in ['index.html','manifest.json']:
  backup=folder/(name.replace('.', '-before-moe-attribution.',1));assert not backup.exists();shutil.copyfile(folder/name,backup)
 p.write_text(html.replace('</body>','<script>'+script+'</script></body>'))
 for origin,name in files.items():shutil.copyfile(origin,folder/name)
 manifest=dict(version='prediction-overview-r2-moe-attribution-r1',inputs={str(p):sha(p) for p in files},code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'})
 (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
assert sha(LOCAL/'index.html')==sha(LIVE/'index.html')
print('PASS local/live published with prior page backups')

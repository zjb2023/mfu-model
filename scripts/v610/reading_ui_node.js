function nodeInfo(n){
 selected=n.node_id;
 const u=data.updates.find(r=>r.node_id===selected), fit=u?data.fits[u.source_key]:null;
 const fields=[['成本查找键',u?.source_key||n.source_parameter_key||'无独立键'],
 ['节点类别 / 成本类别',n.kind+' / '+n.timing_component],['成本来源',n.timing_source],
 ['暴露计算',ms(n.compute_exposed_ns_model)+' ms'],['重叠计算',ms(n.compute_overlap_ns_model)+' ms'],
 ['网络服务',ms(n.network_service_ns_model)+' ms'],['软件 / 同步',ms(n.software_sync_ns_model)+' ms'],
 ['框架余项',ms(n.framework_residual_ns_model)+' ms'],['冻结关键前驱',n.critical_predecessor||'未记录']];
 if(u)fields.unshift(['源校准窗口','256卡 iteration 85 / 90'],['源细粒度计算',ms(u.source_fit_compute_ns)+' ms'],
   ['源样本数',fit?.fit_samples??'见完整源参数表'],['继承形状倍率',u.inherited_shape_factor],
   ['绑定后计算',ms(u.new_compute_ns)+' ms'],['旧完整节点时长',ms(u.old_duration_ns)+' ms'],
   ['新完整节点时长',ms(u.new_duration_ns)+' ms']);
 $('node-info').innerHTML=`<h3>${esc(name(n))}</h3><p><b>PP${n.pp_stage} / rank${n.rank} / lane${n.pp_lane} / MB${n.microbatch} / ${esc(n.phase)} / 层${esc(n.layer_id)}</b><br><code>${esc(n.node_id)}</code></p><p>预测开始 ${ms(n.predicted_start_ns)} ms → 结束 ${ms(n.predicted_end_ns)} ms；自身成本 <b>${ms(n.duration_ns)} ms</b>。</p><p class="note">${u?'本版细粒度绑定：源计算 × 继承形状倍率。曝露节点时长=max(旧完整时长,新计算)，重叠计算节点替换独立计算时长；旧下限未独立证明为纯非计算成本。':'不在本版70192条细粒度更新记录中，保留其冻结成本与来源；不代表没有成本或从未校准。'}</p><table><tbody>${fields.map(r=>`<tr><th>${esc(r[0])}</th><td><code>${esc(r[1])}</code></td></tr>`).join('')}</tbody></table><p>节点成本分量已经包含在duration里，不能额外相加；跨节点时长之和不等于整轮时间或可实现的优化收益。</p>`;
}

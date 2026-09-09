"""Full-iteration display data from sealed nodes and cached stage envelopes.

No builder imports, raw trace reads, replay, fitting, or change to any model.
F/B grouping follows the historical v681 phase_bars definition: per-rank
min/max over its FWD/BWD nodes, then min/max over all 16 ranks of a stage.
"""
from collections import defaultdict
import json
import math
import re
from common import OUT, obj, rows, iterrows, dump, write_csv, verify_inputs

def build():
    verify_inputs()
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append({'check':name,'passed':True})
    def close(name,a,b):check(name,abs(a-b)<1e-6)
    rank_phases={};opts=defaultdict(list);services=[];release=defaultdict(list)
    for r in iterrows('v685_nodes'):
        rank=int(r['rank']);stage=int(r['pp_stage']);start=int(r['predicted_start_ns']);end=int(r['predicted_end_ns'])
        if rank>=0 and r['phase'] in ('FWD','BWD'):
            key=(rank,stage,r['phase'],int(r['microbatch']))
            old=rank_phases.get(key,(start,end))
            rank_phases[key]=(min(old[0],start),max(old[1],end))
        if r['kind']=='optimizer' and r['op_name'] in ('optimizer_update','optimizer_pre_ag0','optimizer_post_ag0'):
            opts[stage,r['op_name']].append((rank,start,end))
        if re.fullmatch(r'tail:dp_with_cp_stage\d+:rs_arrival_join::release:r\d+',r['node_id']):
            release[stage].append((rank,end))
        if re.fullmatch(r'tail:dp_with_cp_stage\d+:(rs|ag0|ag1)_service',r['node_id']) or re.fullmatch(r'tail:expert_dp_stage\d+_lane2:(rs|ag0|ag1)_service',r['node_id']):
            services.append(r)
    phases=defaultdict(list)
    for (rank,stage,phase,mb),(start,end) in rank_phases.items():phases[stage,phase,mb].append((rank,start,end))
    bars=[]
    for (stage,phase,mb),items in sorted(phases.items()):
        check(f'predicted_stage_rank_coverage:{stage}:{phase}:{mb}',len(items)==16 and len({r[0] for r in items})==16)
        bars.append({'id':f's{stage}:{"F" if phase=="FWD" else "B"}{mb}','stage':stage,'phase':'forward' if phase=='FWD' else 'backward','microbatch':mb,'start_ms':min(r[1] for r in items)/1e6,'end_ms':max(r[2] for r in items)/1e6,'rank_count':16,'scope':'同stage全部16rank的时间包络'})
    collectives=[]
    for r in services:
        stage=int(r['pp_stage']);group='expert_dp' if ':expert_dp_' in r['node_id'] else 'dp';coll=r['node_id'].split(':')[-1].removesuffix('_service')
        start=int(r['predicted_start_ns'])/1e6;last=None
        if group=='dp' and coll=='rs':
            rel=release[stage];check('DP_release_16ranks:'+str(stage),len(rel)==16)
            close('DP_release_matches_service:'+str(stage),max(x[1] for x in rel)/1e6,start)
            last=start;start=min(x[1] for x in rel)/1e6
        collectives.append({'id':r['node_id'],'stage':stage,'group_type':group,'collective':coll,'start_ms':start,'end_ms':int(r['predicted_end_ns'])/1e6,'last_release_ms':last,'scope':'DP组全部16rank到达至完成' if last is not None else 'DP组服务节点' if group=='dp' else 'EDP中lane2所在组的服务节点'})
    for (stage,op),items in sorted(opts.items()):
        check(f'optimizer_16ranks:{stage}:{op}',len(items)==16)
        collectives.append({'id':f'optimizer:s{stage}:{op}','stage':stage,'group_type':'optimizer','collective':op,'start_ms':min(x[1] for x in items)/1e6,'end_ms':max(x[2] for x in items)/1e6,'scope':'同stage全部16rank的模型参数更新区间'})
    p=obj('v685_contract')['prediction'];phase_rows=[r for r in rows('v685_phase_rows') if r['variant']=='v685_frozen' and r['split']=='development_primary']
    start=min(b['start_ms'] for b in bars);end=max(b['end_ms'] for b in bars)
    for field,value in [('entry',start),('onef1b',end-start),('tail',p['profiler_step_ms']-end)]:
        for r in phase_rows:
            if r['phase']==field:close('predicted_ledger:'+field+r['iteration'],value,float(r['predicted_ms']))
    check('predicted_FB_84',len(bars)==84);check('predicted_communications_and_optimizer',len(services)==84 and len(collectives)==126)
    prediction={'id':'prediction','label':'A · v685 预测｜224 卡，PP14 / MB3','stages':14,'microbatches':3,'bars':bars,'collectives':collectives,'entry_ms':start,'fb_end_ms':end,'profiler_ms':p['profiler_step_ms'],'training_ms':p['training_step_ms'],'outer_ms':p['outer_framework_ms'],'raw_graph_ms':p['target_raw_graph_ms'],'reconciliation_ms':p['target_reconciliation_ms'],'optimizer_observed':False}
    h=obj('historical_v682');ev={int(r['iteration']):r for r in h['evaluation']};source_clocks={int(r['iteration']):r for r in rows('outer_clocks')};versions={int(r['iteration']):r for r in rows('version_iterations') if r['variant']=='v685_frozen' and r['split']=='development_primary'}
    traces={};ledger=[];events=[]
    for i in [85,90,95,100]:
        traces[str(i)]={}
        for kind,stages,mb in [('target',14,3),('source',16,4)]:
            cache=h[kind+'_trace'][str(i)]
            if kind=='target':
                en=ev[i]['actual_entry_ms'];prof=ev[i]['actual_profiler_ms'];training=float(versions[i]['actual_training_ms']);outer=training-prof
                close('target_Profiler_clock:'+str(i),prof,float(versions[i]['actual_profiler_ms']))
            else:
                en=h['source_boundaries'][str(i)]['actual_entry_ms'];prof=h['source_boundaries'][str(i)]['actual_profiler_ms'];training=int(source_clocks[i]['training_log_ns'])/1e6;outer=int(source_clocks[i]['outer_residual_ns'])/1e6
                close('source_Profiler_clock:'+str(i),prof,int(source_clocks[i]['profiler_step_ns'])/1e6)
            t={'id':kind,'label':f'{"B · Trace 观测｜224 卡，PP14 / MB3" if kind=="target" else "C · Trace 观测｜256 卡，PP16 / MB4"}','stages':stages,'microbatches':mb,'entry_ms':en,'profiler_ms':prof,'training_ms':training,'outer_ms':outer,'bars':[],'collectives':[],'optimizer_observed':False}
            for b in cache['bars']:
                check(f'{kind}_coverage:{i}:{b["id"]}',b['rank_count']==16)
                t['bars'].append({**b,'start_ms':b['start_ms']+en,'end_ms':b['end_ms']+en,'scope':'同stage全部16rank的CPU标注区间包络'})
            for b in cache['collectives']:
                full=b['group_type']=='dp' and b['collective']=='rs' and b.get('group_observed_rank_count')==16
                q={**b,'start_ms':b['start_ms']+en,'end_ms':b['end_ms']+en,'scope':'DP组全部16rank的观测包络' if full else f'代表rank{b["rank"]}（lane{b["rank"]%16}）的GPU通信区间'}
                if 'last_release_ms' in b:q['last_release_ms']=b['last_release_ms']+en
                t['collectives'].append(q)
            t['fb_end_ms']=max(b['end_ms'] for b in t['bars'])
            check(f'{kind}_complete_FB_grid:{i}',len(t['bars'])==stages*mb*2 and len({(b['stage'],b['phase'],b['microbatch']) for b in t['bars']})==stages*mb*2)
            check(f'{kind}_complete_communication_grid:{i}',len(t['collectives'])==stages*6)
            close(f'{kind}_firstF_is_entry:{i}',min(b['start_ms'] for b in t['bars']),en)
            close(f'{kind}_training_clock:{i}',prof+outer,training)
            check(f'{kind}_events_within_Profiler:{i}',all(0<=b['start_ms']<=b['end_ms']<=prof+1e-6 for b in t['bars']+t['collectives']))
            if kind=='target':
                for r in phase_rows:
                    if int(r['iteration'])==i:
                        value={'entry':en,'onef1b':t['fb_end_ms']-en,'tail':prof-t['fb_end_ms'],'outer':outer}[r['phase']]
                        close('target_v685_phase_ledger:'+r['phase']+str(i),value,float(r['actual_ms']))
            traces[str(i)][kind]=t
        for t in [prediction,traces[str(i)]['target'],traces[str(i)]['source']]:
            ledger.append({'view':t['id'],'iteration':i,'entry_ms':t['entry_ms'],'onef1b_ms':t['fb_end_ms']-t['entry_ms'],'post_backward_ms':t['profiler_ms']-t['fb_end_ms'],'profiler_ms':t['profiler_ms'],'outer_ms':t['outer_ms'],'training_ms':t['training_ms']})
    for t,iteration in [(prediction,'fixed')]+[(t,i) for i,pair in traces.items() for t in pair.values()]:
        for group in ['bars','collectives']:
            for b in t[group]:events.append({'view':t['id'],'iteration':iteration,'stage':b['stage'],'kind':b.get('phase',b.get('group_type')),'operation':b.get('collective',b.get('microbatch')),'start_ms':b['start_ms'],'end_ms':b['end_ms'],'scope':b['scope']})
    out={'schema':'w37-complete-iteration-view-v1','default_iteration':85,'iterations':[85,90,95,100],'prediction':prediction,'trace':traces,'ledger':ledger,'axis_origin':'each respective Profiler iteration start; not synchronized physical clocks','trace_FB_scope':'16-rank envelope per stage, not lane0','trace_optimizer_detail':'not independently present in this frozen visualization cache; post-B wall and known communications retained','target_trace_role':'previously exposed development; visualization only','historical_context':{'three_stage_model':'三阶段 MFU v5.4（FWD / BWD 含重计算 / OPT）','three_stage_profiler_MAPE_pct':16.823672,'v682_profiler_MAPE_pct':h['metrics']['all_mape_pct'],'v682_error_ledger':h['error_analysis']['additive_wall_clock_ledger']}}
    dump('full_iter_payload.json',out);write_csv('full_iter_ledger.csv',ledger);write_csv('full_iter_events.csv',events)
    dump('full_iter_validation.json',{'status':'PASS','check_count':len(checks),'checks':checks,'prediction_FB_bars':84,'prediction_communication_bars':84,'prediction_optimizer_bars':42,'target_FB_bars_per_iteration':84,'source_FB_bars_per_iteration':128,'events':len(events),'new_model':False,'raw_trace_read':False})
    print(json.dumps({'stage':'full_iter','status':'PASS','checks':len(checks),'events':len(events)}))

def section():
    return '''<section id="full-iter"><h2>00 · 完整迭代：224卡预测、224卡观测、256卡观测</h2>
<p><b>阅读顺序：</b>先看整轮时间账，再看各 PP stage 怎样执行，最后进入 1F1B 内部诊断。这里使用正式 <b>v685 预测</b>；两侧 trace 来自 v682 页面已有的封存观测，已核对与 v685 当前四轮的时间边界一致。</p>
<p><b>图中名词：</b>F 是前向，B 是反向，数字 0/1/2/3 是微批次编号；PP 是流水线阶段。DP/EDP 为数据/专家数据并行；RS 为梯度归约分片，AG 为参数聚合；OPT 表示模型中的参数更新及相关运行时区间。彩色 F/B 条覆盖该阶段全部 16 张 GPU 对应 rank 的最早开始至最晚完成，<b>不是 lane0，也不是纯 GPU 活跃时间</b>。</p>
<div class="controls"><label>观测迭代 <select id="full-iter-choice"><option>85</option><option>90</option><option>95</option><option>100</option></select></label><label>显示范围 <select id="full-iter-window"><option value="all">完整 Profiler iter（默认）</option><option value="fb">从首个 F 到最后一个 B</option><option value="tail">最后一个 B 附近与收尾</option></select></label><label><input id="full-iter-comms" type="checkbox" checked> 显示通信与预测 OPT</label></div>
<div class="note"><b>整轮时间边界：</b>Profiler 起点 → 首个 F 前准备 → 完整 F/B → 最后一个 B 后收尾 → Profiler 终点。“最后一个 B”按全体 rank 的最晚结束时刻确定，不按 PP 编号最大的一行确定。Training = Profiler + outer；outer 只在下表作两时钟差额，不虚构它在真实 trace 中的位置。1F1B 外的两个时间段始终保留在总账中。</div>
<div class="scroll"><table id="full-iter-ledger"><thead><tr><th>场景</th><th>准备 秒</th><th>1F1B 秒</th><th>最后 B 后收尾 秒</th><th>Profiler 秒</th><th>outer 秒</th><th>Training 整轮秒</th></tr></thead><tbody></tbody></table></div>
<p id="full-iter-score" aria-live="polite"></p>
<div class="timeline-legend"><span><i style="background:#c08b27"></i>首 F 前准备</span><span><i style="background:#1488b3"></i>F 前向</span><span><i style="background:#ce6045"></i>B 反向</span><span><i style="background:#237650"></i>DP</span><span><i style="background:#7350a8"></i>EDP</span><span><i style="background:#8e6613"></i>预测 OPT</span><span><i style="background:#d7cbe6"></i>最后 B 后收尾</span></div>
<p class="muted">三图横轴统一为各自 Profiler 起点后的秒数，保持相同比例；相同迭代编号来自两次不同配置的运行，并非物理时钟同步。可点击彩条查看时间、阶段、微批次和观测范围；手机上图表可横向滚动。</p>
<div id="full-iter-plots"></div><div id="full-iter-selection" class="note" aria-live="polite">点击图中彩条查看详情。</div>
<p><b>完整时间不等于完整设备明细：</b>F/B 覆盖全部 rank；观测 DP RS 使用组内 16 rank 包络，其余通信来自代表 lane2 的 GPU 区间。预测通信为对应组节点，二者不能直接逐条相减。缓存没有独立的优化器计算明细，因此 trace 图保留真实收尾区间和已知通信，OPT 行标为“未单列”；不把未知空白改成虚构的优化器耗时。</p>
<p><b>重叠与补偿：</b>DP/EDP 可能在其他 stage 仍执行 F/B 时启动，按真实位置绘制，不统一搬到尾部。预测图的 604.440 ms 是原始图到 Profiler 的总量补差，1,372.453 ms outer 是预测 Profiler 到 Training 的总时长差。两者均未定位到事件，也未均分到节点。补差仅在各图上方的时间总账列出，不再画成右端阴影。上方“收尾总账（含补差）”是总时长闭合项，不能据此把 604.440 ms 归因到优化器或断定它发生在最后一个 B 之后。</p>
<p><a href="#baseline">16.82%、约 10% 和 75.8% 的历史关系</a> · <a href="http://192.168.0.49:8037/results/w37/A/integration-20260907/full_iter_ledger.csv">整轮分段 CSV</a> · <a href="http://192.168.0.49:8037/results/w37/A/integration-20260907/full_iter_events.csv">全部绘图区间 CSV</a> · <a href="http://192.168.0.49:8037/results/w37/A/integration-20260907/full_iter_payload.json">图数据 JSON</a></p></section>'''

if __name__=='__main__':build()

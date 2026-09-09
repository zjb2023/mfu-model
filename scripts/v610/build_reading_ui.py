"""Presentation revision only: historical reading structure with sealed v610 data.

Never calls a model, fits parameters, scans raw traces or overwrites sealed HTML.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import html
import json
import re
from pathlib import Path
from x10000_analysis.v610 import ROOT, VENDOR, checked_run, resolve_artifact, sha, dump

HISTORY='/home/zjb/Desktop/worktrees/fab-w37-1f1b/results/w37/A/history-scaleout-20260908/integrated_report.html'


def section(text, key):
    # These frozen renderers do not nest section tags. Fail closed on a change.
    match=re.search(r'<section id="'+re.escape(key)+r'">(.*?)</section>',text,re.S)
    assert match and '<section' not in match[1], key
    return match[1]


def rows(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        yield from csv.DictReader(stream)


def build_atlas(run, out):
    updates=list(rows(run/'model/node_parameter_bindings.csv.gz'))
    assert len(updates)==70192 and len({r['source_key'] for r in updates})==4387
    update_ids={r['node_id'] for r in updates}
    keep=['node_id','kind','rank','pp_stage','pp_lane','phase','microbatch','layer_id','op_name',
          'op_family','parallelism','duration_ns','predicted_start_ns','predicted_end_ns',
          'compute_exposed_ns_model','compute_overlap_ns_model','network_service_ns_model',
          'software_sync_ns_model','framework_residual_ns_model','timing_component','timing_source',
          'source_parameter_key','critical_predecessor']
    phases={};entry=[];tail=[];total=0;visible_ids=set()
    for r in rows(run/'model/nodes.csv.gz'):
        total+=1;n={k:r[k] for k in keep};n['updated']=r['node_id'] in update_ids
        if r['kind']=='framework_entry':entry.append(n)
        if r['phase']=='OPT' and int(r['duration_ns'])>0:tail.append(n)
        if r['pp_lane']!='0' or r['phase'] not in ('FWD','BWD') or r['microbatch'] not in ('0','1','2'):continue
        key=f"{r['pp_stage']}:{r['phase']}:{r['microbatch']}"
        p=phases.setdefault(key,dict(stage=int(r['pp_stage']),rank=int(r['rank']),phase=r['phase'],
                        mb=int(r['microbatch']),start=10**30,end=0,count=0,zero=0,nodes=[]))
        p['count']+=1;p['start']=min(p['start'],int(r['predicted_start_ns']));p['end']=max(p['end'],int(r['predicted_end_ns']))
        if int(r['duration_ns'])==0:p['zero']+=1
        else:p['nodes'].append(n);visible_ids.add(n['node_id'])
    assert total==327746 and len(phases)==84 and len(entry)==1
    fits={r['fine_key']:r for r in rows(run/'model/fine_parameters.csv')}
    assert len(fits)==6765
    payload=dict(version='v6.10',phases=phases,entry=entry[0],tail=tail,
                 updates=[r for r in updates if r['node_id'] in visible_ids],
                 coverage=dict(full_nodes=total,full_bindings=len(updates),bound_keys=4387,fit_parameters=6765,
                               display='lane0 representative ranks; all groups in communication summary'),
                 source=dict(path=str(run/'model/nodes.csv.gz'),sha256=sha(run/'model/nodes.csv.gz')),
                 fits=fits)
    here=VENDOR/'research/w37/onef1b/parameter_atlas'
    page=(here/'page.html').read_text()
    page=page.replace('/w37-report-history.html#','/w37-report.html#')
    page=page.replace('v685：参数在预测迭代图中的位置','v6.10：参数在预测迭代图中的位置').replace('v685 冻结预测','v6.10 冻结预测')
    page=re.sub(r'<p class="note">第一版显示.*?</p>', '<p class="note">显示每个PP的lane0代表rank，共14张卡、84个F/B动作的冻结预测切片，不是全224卡包络或trace观测。完整绑定覆盖224 rank、4387键和70192节点；代表卡只是图册显示范围。零成本节点保留计数，通信汇总覆盖全部组；模型未重跑。</p>',page,count=1,flags=re.S)
    page=page.replace('仅v685新绑定','仅本版细粒度绑定')
    page=page.replace('13.681142 ms用在哪里？','查看PP6反向绑定').replace('0.378082 ms用在哪里？','查看PP6前向绑定').replace('7.087467 ms用在哪里？','查看PP13反向绑定')
    page=page.replace("['example-b0',0,'BWD']","['example-b0',6,'BWD']").replace("['example-f13',13,'FWD']","['example-f13',6,'FWD']")
    page=page.replace('本轮144个节点更新来自3个键；其他节点仍有继承成本。第04章的58键表并不等于全部算子的通用性能库。','本版6765条源细粒度参数中4387键实际绑定70192节点；其余节点保留继承成本。源参数按方向、微批、全局层号及语义槽查找，同一参数可绑定多个rank。')
    page=page.replace('12类参数及来源','全部参数类别及来源').replace('3键 / 144节点覆盖范围','70192节点绑定与参数来源')
    page=page.replace('#parameter-system','#calibration').replace('#compute-binding-explained','#binding-audit')
    page=page.replace('/results/w37/A/integration-20260907/model_cost_inventory.csv','/results/w37/A/v610-release/model/nodes.csv.gz')
    page=page.replace('/results/w37/A/parameter-atlas/data.json','/v610-atlas-data.json')
    families=[['配置 / 层与并行','224卡 / PP14 / CP2 / DP8 / EP8 / TP1，MB3，52层。配置决定工作量与结构，不是毫秒成本。'],
      ['入口成本','图内入口1265.388210 ms，只计一次。'],
      ['F/B计算','源85/90拟合6765条细粒度参数；4387键绑定70192节点，覆盖224 rank。橙色表示出现在本版绑定表中，不代表所有值或时长都变了。'],
      ['PP / CP / EP通信','前向PP：624消息，服务4.704806 ms；反向源960记录映射624目标消息。CP/EP保留冻结成本，不用网络服务代表整个通信调用。'],
      ['运行时 / 收尾','软件、框架余项、优化器及DP/EDP成本保留。节点成本分量已经计入duration，不再额外累加；等待由依赖传播。'],
      ['图外两项补差','raw 20953.110588 ms + 补差604.439668 ms = Profiler21557.550256 ms；加outer1372.453390 ms = Training22930.003646 ms。图外两项未映射为具体GPU节点。'],
      ['MFU / 评价','有效FLOPs 8.436548311982576e16，224卡×500 TFLOP/s，按Training时间换算MFU 3.285055%。目标四轮是已暴露开发评价。']]
    page=re.sub(r'const families=\[.*?\];',lambda _: 'const families='+json.dumps(families,ensure_ascii=False)+';',page,count=1,flags=re.S)
    page=page.replace("if(n.updated)return n.source_parameter_key.includes('step_exit')?'退出前计算（本轮更新）':'首个CP前计算（本轮更新）';", "if(n.updated)return '细粒度绑定 · '+n.op_name;")
    node_js=(ROOT/'scripts/v610/reading_ui_node.js').read_text()
    page=re.sub(r'function nodeInfo\(n\)\{.*?\nfunction draw\(',lambda _:node_js+'\nfunction draw(',page,count=1,flags=re.S)
    page=page.replace('__TAIL_JS__',(here/'tail.js').read_text())
    page=page.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    (out/'PARAMETER_ITER_ATLAS.html').write_text(page)
    dump(out/'atlas-data.json',payload)
    return payload['coverage']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=checked_run(args.run_root);out=checked_run(args.output)
    assert not out.exists(), 'Use a new UI revision directory'
    # Verify sealed model/evaluator/render contents before using any data.
    inputs={}
    for stage in ('model','evaluate','render'):
        complete=json.loads((run/stage/'complete.json').read_text())
        for rel,expected in complete['files'].items():assert sha(run/stage/rel)==expected,rel
        inputs[str(run/stage/'complete.json')]=sha(run/stage/'complete.json')
    history_path=resolve_artifact(HISTORY);history=history_path.read_text();current=(run/'render/index.html').read_text()
    from x10000_analysis.v610 import manifest,artifact_root
    m=manifest();r=next(r for r in m['objects'] if r['relative']==m['logical_paths'][HISTORY])
    assert sha(history_path)==r['sha256']
    inputs[str(history_path)]=sha(history_path)
    out.mkdir(parents=True)
    coverage=build_atlas(run,out)
    demo=(VENDOR/'docs/w37/1f1b/DAG_4GPU_DEMO.html').read_text().replace('/w37-report-history.html#why-dag','/w37-report.html#why-dag')
    (out/'DAG_4GPU_DEMO.html').write_text(demo)
    why=section(history,'why-dag').replace('/docs/w37/1f1b/DAG_4GPU_DEMO.html','/v610-teaching-dag.html')
    why=re.sub(r'<div class="note" id="four-gpu-demo-index">.*?</div>', '<div class="note" id="four-gpu-demo-index">下方对比三阶段与计算图的方法差异。独立教学图使用可调的假设耗时解释依赖和等待，不是224卡实测或模型精度证据。</div>',why,count=1,flags=re.S)
    why=why.replace('v685 稳定窗口','历史稳定采样窗口').replace('58 个可用键不等于 58 个键全部命中；144 个更新节点也不等于 144 个独立参数。','6765条源参数不等于6765键全部命中；70192个绑定节点不等于独立参数个数。')
    why=why.replace('第08章的16→256案例用于检验这一步。','跨规模历史案例可在第05章的历史版本记录中查阅。')
    why=why.replace('图中 3.420 秒是低估差额，不是 1F1B 总时长。','当前四轮阶段差额在第07章列出，不能与1F1B总时长混淆。')
    why=re.sub(r'<div class="note"><b>最重要的范围区分：</b>.*?</div>','',why,flags=re.S)
    atlas='<div id="parameter-atlas-entry" class="subsection"><p>在图中选择PP、微批和F/B，点击模块查看参数值与来源；入口、OPT、RS和AG也可放大。图册使用本版冻结节点，显示各PP的lane0代表卡；通信汇总覆盖全部组，不是trace观测。</p><p><a href="/v610-parameter-atlas.html">打开独立参数图册：整轮 → F/B模块 → 参数与来源</a> · 图册顶部可返回本章。</p></div>'
    brief='<div id="calibration-brief"><h3>本版校准，只记住这几项</h3><ul><li><b>计算：</b>源85/90拟合6765条细粒度参数；4387键绑定70192节点，覆盖全部224 rank。</li><li><b>通信与入口：</b>保留PP、CP/EP、DP/EDP和优化器成本；入口1265.388210 ms在图内。</li><li><b>图外时间：</b>补差604.439668 ms、outer1372.453390 ms单列，不画成未经证实的GPU工作。</li></ul><p>参数值不等于整个F/B时长；节点耗时不能直接相加成整轮，预测结果和差额分别见第06/07章。</p></div>'
    params=re.sub(r'^<h2>.*?</h2>','',section(current,'calibration'),count=1,flags=re.S)
    result=re.sub(r'^<h2>.*?</h2>','<h3>本版逐轮结果与误差分账</h3>',section(current,'results'),count=1,flags=re.S)
    full=re.sub(r'^<h2>.*?</h2>','<h3>完整迭代：预测与两侧观测</h3>',section(current,'full-iter'),count=1,flags=re.S)
    phase_rows=list(rows(run/'evaluate/phase_iteration_results.csv'))
    phase_summary=[]
    for name in dict.fromkeys(r['phase'] for r in phase_rows):
        group=[r for r in phase_rows if r['phase']==name]
        actual=sum(float(r['actual_ms']) for r in group)/len(group);pred=sum(float(r['predicted_ms']) for r in group)/len(group)
        phase_summary.append(f'<tr><td>{html.escape(name)}</td><td>{pred:.6f}</td><td>{actual:.6f}</td><td>{actual-pred:+.6f}</td></tr>')
    diagnosis='<h2>07 · 误差诊断：时间差额，不是参数贡献</h2><p>以下为85/90/95/100四轮平均，差额=实际−预测，正值表示低估。阶段平均差额可分账，但不能解释成某项独立优化收益；阶段MAPE不能相加。</p><div class="scroll"><table><thead><tr><th>阶段</th><th>预测ms</th><th>实际均值ms</th><th>差额ms</th></tr></thead><tbody>'+''.join(phase_summary)+'</tbody></table></div><p>图册可追溯节点成本、来源和关键前驱；未建模等待或继承下限的物理归属仍需独立验证。当前不补拟合残差、不调整参数。</p>'
    development='<h2>05 · 模型版本记录</h2><p>当前为v6.10冻结开发版：保留依赖图和非计算成本，增加细粒度计算绑定及四组消融，迁移复现没有改变数值。</p><details><summary>历史版本与校准过程</summary><p>v685及此前版本的聚合校准、历史三阶段对照、旧参数图册和16→256研究均保留于历史报告，不替代本版结果。</p><p><a href="/w37-report-history.html#calibration">完整历史校准记录</a> · <a href="/w37-report-history.html#baseline">历史三阶段与DAG评价</a> · <a href="/w37-report-history.html#development">历史版本增量</a></p></details>'
    sections=[('training',section(history,'training')),('why-dag',why),('task',section(current,'task')),
              ('calibration','<h2>04 · 性能模型参数、校准与源到目标迁移</h2>'+atlas+brief+'<details id="calibration-evidence"><summary>按需展开：完整参数表、校准方法与节点绑定</summary>'+params+'</details>'),
              ('development',development),('results','<h2>06 · 预测结果、完整迭代与误差分账</h2><div id="full-iter">'+full+'</div>'+result),
              ('diagnosis',diagnosis),('limits',section(current,'limits')+'<div id="sources">'+re.sub(r'^<h2>.*?</h2>','<h3>证据与历史索引</h3>',section(current,'sources'),count=1,flags=re.S)+'</div>')]
    sections=[(id,re.sub(r'(<h2>)(?:02|06) ·',lambda m:m[1]+('03' if id=='task' else '08')+' ·',body,count=1) if id in ('task','limits') else body) for id,body in sections]
    # Retain current data and JS (same interactions), not historical result payloads.
    scripts=''.join('<script'+a+'>'+b+'</script>' for a,b in re.findall(r'<script([^>]*)>(.*?)</script>',current,re.S))
    scripts=re.sub(r'const oldAnchors=\[.*?\];','const oldAnchors=[];',scripts,count=1)
    header=re.search(r'<header[^>]*>.*?</header>',current,re.S)[0]
    header=header.replace('本页仅展示当前版本的模型、参数、完整迭代图及开发评估。','先理解分布式训练与三阶段/DAG差异，再沿整轮、F/B模块查看参数来源，最后核对当前预测与开发评价。')
    nav=re.search(r'<nav[^>]*>.*?</nav>',history,re.S)[0]
    css=''.join(re.findall(r'<style[^>]*>.*?</style>',history,re.S)+re.findall(r'<style[^>]*>.*?</style>',current,re.S))
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v6.10｜分布式训练、MFU建模与参数图册</title>'+css+'</head><body>'+header+nav+'<main>'+''.join('<section id="'+id+'">'+body+'</section>' for id,body in sections)+'</main>'+scripts+'</body></html>'
    page=page.replace('http://192.168.0.49:8037','')
    # Route removed historical anchors to history explicitly, never silently.
    ids=set(re.findall(r'\bid="([^"]+)"',page))
    page=re.sub(r'href="#([^"]+)"',lambda m:m[0] if m[1] in ids else 'href="/w37-report-history.html#'+m[1]+'"',page)
    (out/'index.html').write_text(page)
    code=[Path(__file__),ROOT/'scripts/v610/reading_ui_node.js',VENDOR/'research/w37/onef1b/parameter_atlas/page.html',VENDOR/'research/w37/onef1b/parameter_atlas/tail.js']
    dump(out/'manifest.json',dict(model='v6.10',ui_revision=out.name,model_execution=False,raw_reads=0,
        run_root=str(run),inputs=inputs,code={str(p):sha(p) for p in code},coverage=coverage,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('PASS: v610 reading UI + 84 F/B groups + standalone parameter atlas',out)


if __name__=='__main__':main()

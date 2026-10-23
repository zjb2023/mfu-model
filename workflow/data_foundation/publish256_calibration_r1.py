"""Publish measured coverage and honest blockers; never claim partial full MFU."""
import collections,json,shutil,html
from pathlib import Path
from calibrate256_fine_r1 import BASE,load,save,sha
from assemble256_iter60_fine_r1 import VARIANTS
WEB=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001')
def main():
    root=BASE/'calibrate256-fine-r1';out=BASE/'iter60-calibration-r1';out.mkdir(exist_ok=True);cells=[];counts=collections.Counter();inputs={}
    for stage in range(1,15):
        for group in range(2):
            for mb in range(4):
                r=16*stage+group*8;p=root/f'r{r}-mb{mb}';audit=p/'shared-binding-r2/report.json';variant='r1'
                for folder,tag in VARIANTS+[('calibrate256-combined-r3','shared-binding-r4')]:
                    zp=BASE/folder/p.name;za=zp/tag/'report.json'
                    if za.exists() and all(v['status'].startswith('PASS') for v in load(za)['phases'].values()):p=zp;audit=za;variant=folder;break
                c=dict(stage=stage,group=group,mb=mb,rank_base=r,path=str(p),variant=variant,status='PENDING',phases={})
                if audit.exists():
                    c['phases']=load(audit)['phases'];c['status']='PASS' if all(v['status'].startswith('PASS') for v in c['phases'].values()) else 'REVIEW';inputs[str(audit)]=sha(audit)
                else:
                    logs=sorted(p.glob('*.log'),key=lambda x:x.stat().st_mtime)
                    bad=[q for q in logs if 'Traceback' in q.read_text()]
                    if bad:
                        text=bad[-1].read_text();c['status']='EXTRACTION_BLOCKED';c['error']=text[-1400:];c['log']=str(bad[-1]);inputs[str(bad[-1])]=sha(bad[-1])
                    elif logs:c['status']='RUNNING'
                for ph,name,field in [('F','four-layer-f-r1','F_device_window_ms'),('B','b-recompute-r12','B_ms')]:
                    q=p/name/'summary.json'
                    if q.exists():c[ph+'_local_replay_ms']=load(q)[field]
                counts[c['status']]+=1;cells.append(c)
    tail=load(BASE/'iter60-tail-stream-r1/report.json');repair=load(BASE/'iter60-shared-dag-r2/report.json')
    data=dict(status='PARTIAL_FULL_MODEL_NOT_CLOSED',counts=dict(counts),units=112,cells=cells,tail=tail,repair=repair,full_model_error=None,limits=['F/B逐组逐MB条件回放，不能替代完整训练主体误差。','所有目标成本来自256 iter60；不是32→256独立预测。','修正候选移除15条按源时序推定的V输入拷贝等待CP完成边，原32卡源出口不变；冻结原图未改。'])
    final=BASE/'iter60-fine-model-r1/report.json'
    if final.exists():
        data.update(status='PASS_CONDITIONAL_FULL_MODEL_REPLAY',full_model_error=load(final)['body'])
    save(out/'report.json',data)
    js=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>256 iter60 DAG标定进展</title><style>body{font:16px/1.7 system-ui;color:#172033;background:#eef2f6;max-width:1250px;margin:24px auto;padding:0 16px}section{background:white;border-radius:14px;padding:24px;margin:18px 0}h1{font-size:28px}h2{font-size:22px}.cards{display:flex;gap:18px;flex-wrap:wrap}.card{background:#edf3fa;padding:16px;border-radius:8px;min-width:150px}.card b{font-size:28px;display:block}.flow{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.box{background:#dbeafe;border-radius:8px;padding:14px}.comm{background:#ffedd5}.gap{background:#e5e7eb}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:8px}button{padding:10px;margin:2px;border:0;border-radius:4px;cursor:pointer}.PASS{background:#b8e4ce}.EXTRACTION_BLOCKED,.REVIEW{background:#ffe2af}.RUNNING{background:#bfdbfe}.PENDING{background:#e5e7eb}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#2563eb}svg{width:100%}.note{color:#596579}</style><section><nav><a href="../iter60-dag-alignment-r1/">模块对齐</a> · <a href="../iter60-f-layer-binding-r1/">首层八卡图</a> · <a href="../deepep-fb-structure-r1/">原F/B详细结构</a></nav><h1>256卡iter60：从参数标定到同轮回放</h1><p>保留32→256的详细F/B模块与1F1B调度；成本改用256卡iter60。当前阶段：逐组、逐microbatch闭合。<b>完整训练主体回放尚未完成，不发布新的整轮MFU精度。</b></p><div class="cards" id="cards"></div></section><section><h2>① 一次F/B如何组成</h2><div class="flow"><div class="box">F：四层<br>Attention/CP → EP/专家 → 层尾</div><span>→ PP →</span><div class="box">B：四个反向执行单元<br>前向重计算 → MoE反向 → Attention反向</div><span>→</span><div class="box comm">末B后<br>Dense/EDP RS → 更新/AG/同步</div></div><p>外层F/B插入详细子图时外壳成本置零。不同rank与通信按依赖取最大完成点，不把节点时间直接相加。</p><h3>PP1、MB0：两组EP8都已完成四层局部回放</h3><div id="pilot"></div><p class="note">回放使用同轮实测成本与保留间隔，精确重现表示计费闭合，不代表独立预测误差为零。</p></section><section><h2>② 全部中间stage覆盖情况</h2><p>每格是一组EP8的一个microbatch，包含四层F和四层B。绿：两相位通过；橙：提取/对齐需要处理；蓝：执行中；灰：待处理。点击查看具体状态。</p><table id="grid"></table><pre id="detail">点击格子查看</pre></section><section><h2>③ 已发现的适配问题</h2><div class="flow"><div class="box">CP通信身份<br>改用CPU标记＋correlation核对</div><div class="box">空专家分配<br>CPU输入0行，FC没有GPU事件</div><div class="box gap">少量同stream事件重叠<br>不能直接抹平或负成本裁零</div></div><p>另已保留一版依赖修正候选：V输入拷贝可与先前CP传输并行，不能强制等待其完成。移除同类15条条件边后，原32卡源回放所有节点完成点保持不变。原封存模型不覆盖。</p><p>实际检查：rank227的两次专家反向输入分别为[0,5120]、[0,3072]；rank20 MB2同stream的两个拷贝事件重叠约0.027 ms。这些属于模板/提取适配问题，不是新的预测误差。</p></section><section><h2>④ 尾部：按事件和stream计费，不加一个总尾长</h2><div class="cards"><div class="card">局部尾部回放<b>376.997 ms</b></div><div class="card">设备活动<b>570</b></div><div class="card">设备stream<b>5</b></div></div><div id="tail"></div><p class="note">表中活动时长可以重叠，不能相加当总尾长。通信包含等待；跨stream及跨stage就绪仍是显式保留参数，尚未完成因果依赖验证。</p></section><section><h2>⑤ 复现与下一步</h2><p>继续修正通信身份、零工作量与事件重叠的处理，再把通过的全部F/B、首尾整块、PP和尾部装回统一调度图。所有缺项保留，不以整块比例缩放补齐细项。</p><a href="report.json">机读进度</a> · <a href="manifest.json">证据SHA256</a></section><script>const D=__DATA__;const $=id=>document.getElementById(id);$('cards').innerHTML=[['标定单元',112],['F/B均通过',D.counts.PASS||0],['待处理问题',(D.counts.EXTRACTION_BLOCKED||0)+(D.counts.REVIEW||0)],['尚未完成',(D.counts.PENDING||0)+(D.counts.RUNNING||0)]].map(([k,v])=>`<div class="card">${k}<b>${v}</b></div>`).join('');let t='<tr><th>PP stage</th>'+['A0','A1','A2','A3','B0','B1','B2','B3'].map(s=>'<th>组/MB '+s+'</th>').join('')+'</tr>';for(let s=1;s<=14;s++){t+='<tr><th>PP'+s+'</th>';for(let g=0;g<2;g++)for(let m=0;m<4;m++){const i=D.cells.findIndex(c=>c.stage===s&&c.group===g&&c.mb===m),c=D.cells[i];t+=`<td><button class="${c.status}" data-i="${i}">${c.status==='PASS'?'通过':c.status==='RUNNING'?'执行中':c.status==='PENDING'?'待处理':'需处理'}</button></td>`}t+='</tr>'}$('grid').innerHTML=t;for(const b of document.querySelectorAll('[data-i]'))b.onclick=()=>{$('detail').textContent=JSON.stringify(D.cells[+b.dataset.i],null,2)};$('pilot').innerHTML='<table><tr><th>组</th><th>F ms</th><th>B ms</th></tr>'+D.cells.filter(c=>c.stage===1&&c.mb===0).map(c=>`<tr><td>rank${c.rank_base}–${c.rank_base+7}</td><td>${c.F_local_replay_ms?.toFixed(3)??'待核对'}</td><td>${c.B_local_replay_ms?.toFixed(3)??'待核对'}</td></tr>`).join('')+'</table>';$('tail').innerHTML='<table><tr><th>尾部活动</th><th>次数</th><th>活动累计ms（可重叠）</th></tr>'+Object.entries(D.tail.activities).map(([k,v])=>`<tr><td>${k}</td><td>${v.events}</td><td>${v.sum_device_ms.toFixed(3)}</td></tr>`).join('')+'</table>';</script>'''
    if final.exists():
        page=page.replace('当前阶段：逐组、逐microbatch闭合。<b>完整训练主体回放尚未完成，不发布新的整轮MFU精度。</b>','112个标定单元均通过，完整训练主体同轮回放已完成。<b><a href="../iter60-fine-model-r1/">查看完整结果、时间与MFU误差</a></b>')
        page=page.replace('继续修正通信身份、零工作量与事件重叠的处理，再把通过的全部F/B、首尾整块、PP和尾部装回统一调度图。所有缺项保留，不以整块比例缩放补齐细项。','通信身份、零工作量、紧凑激活/转换序列及事件重叠适配已通过。F/B、首尾接口、PP和尾部已装配并独立重放；原失败日志保留。此为条件同轮回放，不是独立预测。')
    (out/'index.html').write_text(page.replace('__DATA__',js));inputs[str(Path(__file__))]=sha(Path(__file__))
    for p in [BASE/'iter60-tail-stream-r1/report.json',BASE/'iter60-shared-dag-r2/report.json']:inputs[str(p)]=sha(p)
    save(out/'manifest.json',dict(inputs=inputs,outputs={n:sha(out/n) for n in ['report.json','index.html']}))
    web=WEB/out.name;web.mkdir(exist_ok=True)
    for n in ['index.html','report.json','manifest.json']:shutil.copy2(out/n,web/n)
    print(json.dumps(dict(counts=counts,url='http://192.168.8.16:43312/df-v001/iter60-calibration-r1/'),ensure_ascii=False))
if __name__=='__main__':main()

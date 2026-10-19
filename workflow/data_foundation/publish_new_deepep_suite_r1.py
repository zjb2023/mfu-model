"""Publish frozen new DeepEP candidate: overview, source inventory and F/B DAG."""
import collections, hashlib, json, re, shutil, zipfile, statistics
from pathlib import Path
from build_b_cp_pair_r4 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
SOURCE=BASE/'deepep-stable-20261016-r1'
WEB=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001')
INPUTS={}
def read(p):
    blob=p.read_bytes();INPUTS[str(p)]=hashlib.sha256(blob).hexdigest();return json.loads(blob)
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':')))
def main():
    ev=read(SOURCE/'evaluation.json');seal=read(SOURCE/'model-seal.json')
    assert INPUTS[str(SOURCE/'model-seal.json')]==ev['source_seal_sha256']
    for p,h in seal['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    common=dict(candidate='新采集DeepEP32 · iter155–200 · 对齐DAG r4',evaluation=ev,parameters=read(SOURCE/'model/parameters.json'),outer=read(SOURCE/'model/outer-prediction.json'),alignment=read(SOURCE/'aligned-r4/audit.json'),source_root='/var/local/zjb-data/32-deepep-200iter-20261016/extracted/framework/2026-10-15-17:29')
    completion=SOURCE/'profiler-completion-r1/report.json'
    if completion.exists():
        common['completion']=read(completion)
        c=common['completion'];baseline=c['source_summary']['tail_ms']['mean'];target=c['target_summary']['tail_ms']['mean'];estimate=c['prediction']['tail_cross_network_increment_ms']
        common['tail_diagnosis']=dict(status='ERROR_DIAGNOSIS_NOT_MODEL_CORRECTION',source_baseline_ms=baseline,target_mean_ms=target,observed_gap_ms=target-baseline,bandwidth_hypothesis_ms=estimate,unassigned_remainder_ms=target-baseline-estimate,
            hypothesis_share_percent=100*estimate/(target-baseline),rows=[dict(iteration=r['iteration'],source_baseline_ms=baseline,target_tail_ms=r['tail_ms'],underestimate_ms=r['tail_ms']-baseline) for r in c['rows']],
            limits=['Source10-iteration mean vs target20-iteration mean, rank0 lastB to Profiler end.', 'Bandwidth estimate is a hypothesis, not verified causal attribution.', 'Unassigned remainder is not automatically synchronization.', 'No cost correction applied; prior full-Profiler adjusted candidate is superseded as the active presentation.'])
        p=c['prediction'];last4=c['rows'][-4:];assert [r['iteration'] for r in last4]==[85,90,95,100]
        predictions=dict(startup_ms=p['startup_ms'],onef1b_ms=p['onef1b_ms'],tail_ms=baseline)
        predictions['profiler_ms']=sum(predictions.values())
        stages={}
        for key,value in predictions.items():
            truth=[r[key] for r in last4];errors=[100*(value/t-1) for t in truth]
            stages[key]=dict(predicted_ms=value,observed_mean_ms=statistics.mean(truth),mean_delta_ms=value-statistics.mean(truth),signed_mean_percent=statistics.mean(errors),mape_percent=statistics.mean(map(abs,errors)),signed_std_pp=statistics.pstdev(errors))
        assert abs(sum(stages[k]['mean_delta_ms'] for k in ['startup_ms','onef1b_ms','tail_ms'])-stages['profiler_ms']['mean_delta_ms'])<1e-6
        common['last4_overview']=dict(stages=stages,rows=[dict(iteration=r['iteration'],**{k:r[k] for k in predictions},predicted_profiler_ms=predictions['profiler_ms'],error_percent=100*(predictions['profiler_ms']/r['profiler_ms']-1)) for r in last4],policy='Uncorrected source-copy startup/tail plus sealed DAG1F1B; no51.777ms bandwidth addition; rank0 Profiler, not TrainingStep.')
        scopes={}
        for name,keys in [('profiler',['startup_ms','onef1b_ms','tail_ms']),('body',['onef1b_ms','tail_ms']),('onef1b',['onef1b_ms'])]:
            predicted=sum(predictions[k] for k in keys);cohorts={}
            for cohort,rows in [('last4',last4),('all20',c['rows'])]:
                values=[sum(r[k] for k in keys) for r in rows]
                errors=[100*(predicted/v-1) for v in values]
                cohorts[cohort]=dict(observed_mean_ms=statistics.mean(values),signed_mean_percent=statistics.mean(errors),mape_percent=statistics.mean(map(abs,errors)),signed_std_pp=statistics.pstdev(errors),absolute_std_pp=statistics.pstdev(map(abs,errors)),rows=[dict(iteration=r['iteration'],observed_ms=v,error_percent=e) for r,v,e in zip(rows,values,errors)])
            scopes[name]=dict(predicted_ms=predicted,**cohorts)
        common['scope_comparison']=dict(scopes=scopes,policy='Time errors, not MFU errors. Same frozen prediction; scope excludes matching prefix on both sides. No optimizer correction. Prefix is not established profiler-induced overhead.')
        physics=read(BASE/'full-step-32to256-r1/sealed-prediction.json')['physics']
        mfu_factor=100*physics['model_flops_per_iteration']/(physics['world_size']*physics['peak_flops_per_gpu_s'])*1000
        common['scope_comparison']['mfu_physics']=physics
        for scope in scopes.values():
            scope['predicted_mfu_percent']=mfu_factor/scope['predicted_ms']
            for cohort in ['last4','all20']:
                rows=scope[cohort]['rows']
                for r in rows:
                    r['observed_mfu_percent']=mfu_factor/r['observed_ms']
                    r['mfu_relative_error_percent']=100*(scope['predicted_mfu_percent']/r['observed_mfu_percent']-1)
                scope[cohort]['mfu']=dict(observed_mean_percent=statistics.mean(r['observed_mfu_percent'] for r in rows),signed_mean_percent=statistics.mean(r['mfu_relative_error_percent'] for r in rows),mape_percent=statistics.mean(abs(r['mfu_relative_error_percent']) for r in rows),mean_delta_pp=scope['predicted_mfu_percent']-statistics.mean(r['observed_mfu_percent'] for r in rows))
    tail_review=BASE/'tail-source155-r1/report.json'
    if tail_review.exists():
        review=read(tail_review)
        for row in review['rows']:
            target=read(tail_review.parent/f'target-{row["iteration"]}.json')
            rs=sorted((e for e in target['device_events'] if e['kind']=='RS'),key=lambda e:e['start_ms'])
            assert len(rs)==2
            row['dense_rs_ms']=rs[0]['end_ms']-rs[0]['start_ms'];row['edp_rs_ms']=rs[1]['end_ms']-rs[1]['start_ms']
        common['tail_review']=review
        common['tail_increment_shares']=read(BASE/'tail-source155-r1/increment-shares.json')
        source155=read(SOURCE/'profiler-completion-r1/source-155.json')
        target_prefix=read(BASE/'startup-tail-r7-20260929/report.json')
        targets=[r for r in target_prefix['rows'] if r['case']=='256' and r['iteration'] in [85,90,95,100]]
        tp=statistics.mean(r['startup_GPU_ms'] for r in targets);ta=statistics.mean(r['selected_AG_ms'] for r in targets)
        common['prefix_review']=[]
        for name,sp,sa in [('正式源155–200均值',common['completion']['source_summary']['startup_ms']['mean'],common['completion']['source_summary']['startup_small_ag_union_ms']['mean']),('单源iter155',source155['startup_ms'],source155['startup_small_ag_union_ms'])]:
            common['prefix_review'].append(dict(name=name,source_prefix_ms=sp,source_ag_ms=sa,target_prefix_ms=tp,target_ag_ms=ta,prefix_delta_ms=tp-sp,ag_delta_ms=ta-sa,share_percent=100*(ta-sa)/(tp-sp)))
    common['source40_trend']=read(SOURCE/'stability-review.json')
    assert len(common['source40_trend']['summary'])==40
    source_schedules=[]
    for it in range(155,201,5):
        ranks=[read(SOURCE/f'outer/gpu-{it}-{r}.json') for r in [0,8,16,24]]
        origin=min(o['absolute_start_ms'] for o in ranks[0]['ops'] if o['kind']=='F')
        nodes=[dict(stage=d['stage'],rank=d['rank'],kind=o['kind'],mb=o['mb'],start_ms=o['absolute_start_ms']-origin,end_ms=o['absolute_end_ms']-origin) for d in ranks for o in d['ops'] if o['kind'] in ['F','B']]
        assert len(nodes)==64
        source_schedules.append(dict(iteration=it,nodes=nodes,end_ms=max(n['end_ms'] for n in nodes),clock='Recorded trace timestamps relative to rank0 first F; no new cross-host clock calibration.'))
    common['source_schedules']=source_schedules
    graphs={}
    for ph in ['F','B']:
        g=read(SOURCE/f'model/{ph}.json');t=run(g);rows=[]
        for k,n in g.items():
            m=re.search(r'(?:^|::)(?:L|R|U)(\d+)',k) or re.search(r':u(\d+)',k);r=n.get('rank')
            if r is None:
                mr=re.search(r'(?:^|:)r(\d+)',k)
                if mr:r=int(mr.group(1))
            rows.append(dict(id=k,rank=r,layer=int(m.group(1)) if m else n.get('unit',n.get('layer')),kind=n.get('kind') or 'trace_reserved',cost=n['cost_ms'],start=t[k]['start_ms'],end=t[k]['end_ms'],deps=n['deps'],policy=n.get('policy','effective_source_cost'),basis=n.get('basis','source semantic cost'),names=n.get('names',[])))
        graphs[ph]=rows
    template=Path(__file__).with_name('ui_new_deepep_suite_r1.html').read_text()
    pages={'prediction-overview-r3':'overview','deepep-device-inventory-r1':'inventory','deepep-fb-structure-r1':'structure'}
    for slug,mode in pages.items():
        out=BASE/slug;out.mkdir(exist_ok=True);web=WEB/slug;web.mkdir(exist_ok=True)
        data=dict(common,mode=mode)
        if mode=='structure':data['graphs']=graphs
        if mode=='inventory':
            for it in range(155,201,5):
                source=SOURCE/f'costs-r2/i{it}';bs=read(source/'b-cost-blocks-r1/evidence.json')
                for b in bs:
                    rank=b['rank'];f=read(source/f'pp32-f-split-r8/rank-{rank}.json');origin=min(d['start_ms'] for d in b['devices'])
                    markers={m['id']:m for m in f['markers']}
                    fs=[]
                    for d in f['devices']:
                        m=markers.get(d['semantic'],{});fs.append(dict(start=d['start_ms'],end=d['end_ms'],stream=str(d['stream']),kind=m.get('kind',d['tag']),name=d['name'],event=d['event'],launch=d['runtime_event'],layer=None))
                    rows=[dict(start=d['start_ms']-origin,end=d['end_ms']-origin,stream=d['stream'],kind=d['kind'],name=d['name'],event=d['event'],launch=d['launch_event'],layer=d['unit'],phase=d['phase']) for d in b['devices']]
                    payload=dict(iteration=it,rank=rank,path=f['path'],sha256=f['sha256'],F=fs,B=rows)
                    path=out/f'i{it}-r{rank}.json';save(path,payload);shutil.copyfile(path,web/path.name)
        page_template=Path(__file__).with_name('ui_deepep_fb_structure_r2.html') if mode=='structure' else Path(__file__).with_name('ui_new_deepep_suite_r1.html')
        html=page_template.read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False))
        (out/'index.html').write_text(html);shutil.copyfile(out/'index.html',web/'index.html')
        if mode=='overview':
            if 'tail_diagnosis' in common:
                save(out/'tail-error-diagnosis.json',common['tail_diagnosis']);shutil.copyfile(out/'tail-error-diagnosis.json',web/'tail-error-diagnosis.json')
                save(out/'last4-overview.json',common['last4_overview']);shutil.copyfile(out/'last4-overview.json',web/'last4-overview.json')
                save(out/'scope-comparison.json',common['scope_comparison']);shutil.copyfile(out/'scope-comparison.json',web/'scope-comparison.json')
                if 'tail_increment_shares' in common:
                    save(out/'tail-increment-shares.json',common['tail_increment_shares']);shutil.copyfile(out/'tail-increment-shares.json',web/'tail-increment-shares.json')
            package=out/'new-deepep32-stable155-200-dag-r4.zip'
            with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
                for name in ['model/F.json','model/B.json','model/parameters.json','model/outer-prediction.json','model/expanded-dag.json.gz','model-seal.json','evaluation.json','aligned-r4/audit.json']:
                    z.write(SOURCE/name,name)
                for name in ['model_deepep_stable_r4.py','align_deepep_stable_r4.py']:
                    z.write(Path(__file__).with_name(name),'code/'+name)
                if 'tail_diagnosis' in common:
                    z.write(out/'tail-error-diagnosis.json','diagnostics/tail-error-diagnosis.json')
                    z.write(out/'last4-overview.json','diagnostics/last4-overview.json')
                z.writestr('README.txt','Frozen candidate model data, not standalone runtime. Use the repository Snakefile-r4 to reproduce. Effective CP/EP replacement costs, not pure FCT. 1F1B only; startup/optimizer tail not recalibrated.\n')
            shutil.copyfile(package,web/package.name)
        save(out/'manifest.json',dict(candidate=common['candidate'],inputs=INPUTS,code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),page_template]},outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},new_prediction_experiment=False))
        shutil.copyfile(out/'manifest.json',web/'manifest.json')
        print('http://192.168.8.16:43312/df-v001/'+slug+'/')
if __name__=='__main__':main()

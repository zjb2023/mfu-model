"""Present existing sealed evaluations only; no trace scan or prediction rerun."""
import hashlib
import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/data-foundation'
OUT = BASE / 'prediction-methods-r1'
SERVE = Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001/prediction-methods-r1')

def main():
    OUT.mkdir(exist_ok=True)
    sources = {}
    def read(name):
        p = BASE / name / 'report.json'
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    old = read('source80-fine-evaluation-r1')
    truth = [{k: r[k] for k in ('iteration', 'truth_ms')} for r in old['rows']]
    methods = []
    def add(id, name, color, report, prediction, scope, limit, link, group):
        rows = report.get('rows', report.get('target256'))
        assert [(r['iteration'], r['truth_ms']) for r in rows] == [(r['iteration'], r['truth_ms']) for r in truth]
        errors = [(prediction / r['truth_ms'] - 1) * 100 for r in truth]
        def stats(values):
            absolute = list(map(abs, values))
            return dict(signed_mean=statistics.mean(values), signed_std=statistics.pstdev(values), absolute_mean=statistics.mean(absolute), absolute_std=statistics.pstdev(absolute))
        methods.append(dict(id=id, name=name, color=color, prediction_ms=prediction, scope=scope, limit=limit, link=link, group=group, errors=errors, all20=stats(errors), last4=stats(errors[-4:])))
    add('old60','A · DeepEP iter60 原成本（蓝）','#2563eb',old,old['details']['60']['prediction_ms'],'旧32卡 DeepEP iter60：中间细粒度 F/B、首尾整块 F/B、PP 成本。','固定源成本；不是目标 token 修正模型。','prediction-overview-r2','原细粒度 DAG')
    add('old80','B · DeepEP iter80 全量1F1B源成本','#0891b2',old,old['details']['80']['prediction_ms'],'从 A 的原语义依赖出发：用旧32卡 DeepEP iter80 更新中间全部成本、首尾 F/B 与 PP。','B 源回放仍长 13.9 ms（1.56%），迁移尚未完全闭合。','source80-fine-r1','原细粒度 DAG')
    r=read('stable155-256-r1')
    add('expert','C · DeepEP iter60基线＋MCCL AlltoAll iter155–200专家计算（绿）','#16a34a',r,r['prediction']['one_f_one_b_ms'],'从 A 出发，不经过 B：只把中间 F/B 的专家 FC1、激活/门控、FC2，以及重计算和真正反向对应项替换为新32卡 iter155–200 均值。','Attention、CP/EP、就绪/残余、首尾和 PP 仍为旧32卡 iter60；不是旧80基线。','stable155-256-r1','原细粒度 DAG')
    r=read('hybrid-deepep-r1')
    add('hybrid','D · C再换MCCL AlltoAll iter155–200的Flash／CP／首尾／PP','#9333ea',r,r['prediction']['predicted_1f1b_ms'],'从 C 出发：再将 Flash Attention、CP、首尾整块 F/B、PP 成本更新为新32卡 iter155–200 均值；中间 DeepEP 项保留旧32卡 iter60。','未匹配本地项/残余仍为旧60；新首尾整块自身可能包含 MCCL 行为。','hybrid-deepep-r1','原细粒度 DAG')
    r=read('hybrid-local-r3')
    add('local','E · D再换MCCL AlltoAll iter155–200的F Attention／B重计算周边计算与重排','#d97706',r,r['prediction']['predicted_ms'],'从 D 出发：使用新版 MCCL AlltoAll iter155–200 均值，替换 F 前向 Attention 周边本地项96个、B内重计算 Attention 周边计算64个、B内重计算 Attention 重排/拷贝124个节点的 GPU 活动成本；节点内原残余和其他成本沿用 D。','本地项指单GPU操作，不是跨卡传输；不含已单列的 Flash Attention 主体、专家 FC1/FC2 或 CP/EP 通信。有条件 kernel 序列匹配；真正反向的本地处理与梯度桥接等未全量迁移。','hybrid-local-r3','原细粒度 DAG')
    r=read('full155-256-r1')
    add('full','F · MCCL AlltoAll iter155–200全量1F1B成本＋重建条件图（红）','#dc2626',r,r['prediction']['predicted_1f1b_ms'],'独立重建候选，不从 E 继续替换：使用新32卡 iter155–200 普通 MCCL AlltoAll 全部1F1B源成本。','固定 gap 吸收了大量耗时，通信替换敏感性不足；不能用其误差直接归因后端。','full155-256-r1','不同图的诊断候选')
    r=read('ep-envelope-r1')
    add('envelope','G · E的EP区域改用MCCL AlltoAll iter155–200分发—计算—合并包络','#be185d',r,r['prediction']['predicted_256_ms'],'从 E 出发：包络外成本不变；EP 区域改用新32卡 iter155–200 的八卡分发—专家计算—合并完整实测包络均值，移除被包络覆盖的内部专家成本，避免重复计费。','改为组级共同就绪/完成；不是原细粒度图、不是纯通信替换，未用 cali_final 公式。','ep-envelope-r1','不同图的诊断候选')
    data=dict(truth=truth, methods=methods, inputs=sources, evaluation='Existing predictions; statistics recomputed from identical 20 target windows, population std.')
    (OUT/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    template=(Path(__file__).with_name('ui_prediction_methods_r1.html')).read_text()
    (OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False)))
    manifest=dict(inputs=sources, outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'index.html',OUT/'comparison.json']}, code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('ui_prediction_methods_r1.html')]}, checks={'same_target_windows':'PASS','new_experiments':False})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    SERVE.mkdir(exist_ok=True)
    for name in ['index.html','comparison.json','manifest.json']:
        shutil.copyfile(OUT/name,SERVE/name)
    print('http://192.168.8.16:43312/df-v001/prediction-methods-r1/')

if __name__ == '__main__':
    main()

"""Regroup existing exclusive evidence; no trace reads and no prediction fitting."""
import argparse,graphlib,hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 paths=[BASE/'token-attention-r1/report.json',BASE/'token-attention-r1/source.json',BASE/'b-remaining-r1-verified/report.json',BASE/'prediction-r11-recompute-r1-verified/outer-prediction.json',BASE/'1f1b-overestimate-r1/report.json',BASE/'1f1b-overestimate-r1/blocks.json']
 fine,source,coarse,outer,diag,blocks=map(load,paths);nodes=outer['nodes'];cost=nodes['s1:B0']['duration_ms'];obs={b['node']:b['observed_duration_ms'] for b in blocks}
 def parts(total,fc,c,ep):
  expert=fc+c['expert_activation'];data=c['token_reorder']+ep
  return dict(expert_compute=expert,token_reorder_EP_visible=data,other_middle_B=total-expert-data)
 src=parts(cost,source['blocks'][0]['FC'],fine['source'],coarse['source']['EP']);cm={r['node']:r for r in coarse['rows']};rows=[]
 for r in fine['rows']:
  rows.append(dict(node=r['node'],stage=r['stage'],mb=r['mb'],components=parts(obs[r['node']],r['FC'],r['components'],cm[r['node']]['components']['EP'])))
 assert len(rows)==56
 order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order());groups=list(src)
 def run(chosen):
  overrides={r['node']:cost+sum(r['components'][g]-src[g] for g in chosen) for r in rows};t={};assert min(overrides.values())>0
  for k in order:
   n=nodes[k];t[k]=max((t[d] for d in n['dependencies']),default=0)+overrides.get(k,n['duration_ms'])
  return t['s0:B3']-(t['s0:F0']-nodes['s0:F0']['duration_ms'])
 values={}
 for bits in itertools.product([0,1],repeat=3):
  k=tuple(g for g,b in zip(groups,bits) if b);values[k]=run(k)
 base=values[()];allb=values[tuple(groups)];truth=diag['truth_ms'];assert abs(base-diag['baseline_ms'])<1e-5;assert abs(allb-coarse['oracle_ms'])<1e-5
 contributions={g:0. for g in groups}
 for perm in itertools.permutations(groups):
  chosen=set();previous=base
  for g in perm:
   chosen.add(g);cur=values[tuple(x for x in groups if x in chosen)];contributions[g]+=(previous-cur)/6;previous=cur
 contributions['outside_middle_B_residual']=allb-truth
 assert abs(sum(contributions.values())-(base-truth))<1e-5
 report=dict(status='PARTIAL_REGROUPED_ORACLE_NOT_PREDICTION',baseline_ms=base,truth_ms=truth,total_error_ms=base-truth,total_error_pct=100*(base/truth-1),source_components_ms=src,rows=rows,
  attribution=[dict(group=g,contribution_ms=v,percent_of_total_error=100*v/(base-truth),error_percentage_points=100*v/truth) for g,v in contributions.items()],
  scenarios=[dict(replaced=list(k),window_ms=v,reduction_ms=base-v,percent_of_total_error=100*(base-v)/(base-truth),error_pct=100*(v/truth-1)) for k,v in values.items()],
  limits=['Expert compute includes recompute and true-backward FC GEMM plus identified expert activation/gating; unidentified work remains residual',
  'Second group includes confirmed token permutation and exclusive EP-parent-associated GPU activity only, not complete EP transport/packing/wait',
  'Router, unassigned copies/linears, inter-category overlap and gaps remain other_middle_B',
  'Representative-rank effective B substitution and outer DAG scheduling, not full EP8 inner-node causal validation',
  'Outside-middle-B residual is unexplained remainder, not pure PP cost',
  'Target iter60 evidence evaluator-only; Shapley varies with grouping, do not add previous percentages'])
 save(a.out/'report.json',report)
 save(a.out/'manifest.json',dict(inputs={str(p):sha(p) for p in paths},code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p.resolve()):sha(p) for p in a.out.glob('*.json')}))
 print(json.dumps(dict(attribution=report['attribution'],scenarios=report['scenarios']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

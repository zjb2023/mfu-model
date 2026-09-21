"""Target-only stage sweep and conditional DAG diagnostic, never a prediction fit."""
import argparse,collections,graphlib,hashlib,json,statistics,math
from pathlib import Path
from audit_b_cp_gaps_r1 import merged
from audit_b_operator_position_r1 import union
from audit_b_sync_owners_r1 import intersections
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
def load(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 refs=BASE/'b-position-curves-r2/raw-provenance.json';rawrefs=[r for r in load(refs) if r['world']==256 and r['iteration']==60];assert len(rawrefs)==14
 rows=[]
 for ref in sorted(rawrefs,key=lambda r:r['stage']):
  raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];es=json.loads(raw)['traceEvents'];del raw
  bs=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='backward_step'],key=lambda e:e['ts']);assert len(bs)==4
  runtime=collections.defaultdict(list)
  for e in es:
   if e.get('cat')=='privateuse1_runtime':runtime[e.get('args',{}).get('correlation')].append(e)
  gpu=[e for e in es if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']];assert len({e['args'].get('device') for e in gpu})==1
  occupied=merged([(e['ts'],e['ts']+e['dur']) for e in gpu]);idle=[(l[1],r[0]) for l,r in zip(occupied,occupied[1:])]
  owned=collections.defaultdict(list)
  for e in gpu:
   calls=runtime.get(e.get('args',{}).get('correlation'),[])
   if len(calls)!=1:continue
   c=calls[0]
   for mb,b in enumerate(bs):
    if c['pid']==b['pid'] and b['ts']<=c['ts'] and c['ts']+c.get('dur',0)<=b['ts']+b['dur']+.01:owned[mb].append(e)
  sync=[e for calls in runtime.values() for e in calls if 'Synchronize' in e.get('name','')]
  for mb,b in enumerate(bs):
   ds=owned[mb];lo=min(e['ts'] for e in ds);hi=max(e['ts']+e['dur'] for e in ds);gaps=intersections(idle,lo,hi)
   ops=[e for e in es if e.get('cat')=='cpu_op' and e.get('name') in ['FusedCombine','FusedDispatchBackward'] and e['pid']==b['pid'] and b['ts']<=e['ts'] and e['ts']+e['dur']<=b['ts']+b['dur']+.01];assert len(ops)==8
   hits=[]
   for c in sync:
    if any(c['pid']==o['pid'] and c['tid']==o['tid'] and o['ts']<=c['ts'] and c['ts']+c.get('dur',0)<=o['ts']+o['dur']+.01 for o in ops):hits+=intersections(gaps,c['ts'],c['ts']+c['dur'])
   rows.append(dict(stage=ref['stage'],mb=mb,B_ms=(hi-lo)/1000,combine_sync_gap_ms=union(hits),all_no_device_ms=union(gaps)))
  print('PP',ref['stage'],'PASS',flush=True)
 outerp=BASE/'detailed-fb-32to256-r1/outer-prediction.json';diagp=BASE/'1f1b-overestimate-r1/report.json';blockp=BASE/'1f1b-overestimate-r1/blocks.json'
 nodes=load(outerp)['nodes'];diag=load(diagp);order=list(graphlib.TopologicalSorter({k:n['dependencies'] for k,n in nodes.items()}).static_order())
 def run(overrides):
  t={}
  for k in order:
   n=nodes[k];start=max((t[d][1] for d in n['dependencies']),default=0);t[k]=(start,start+overrides.get(k,n['duration_ms']))
  return t['s0:B3'][1]-t['s0:F0'][0]
 obs={f's{r["stage"]}:B{r["mb"]}':r['B_ms'] for r in rows};cost={f's{r["stage"]}:B{r["mb"]}':r['combine_sync_gap_ms'] for r in rows}
 for b in load(blockp):
  if b['node'] in obs:assert abs(obs[b['node']]-b['observed_duration_ms'])<1e-4
 # All other costs and graph unchanged. Target PP1 baseline per microbatch;
 # this measures target spatial variation, not source32->target cost difference.
 corrected={k:nodes[k]['duration_ms']+cost[k]-cost[f's1:B{int(k.split("B")[1])}'] for k in obs}
 equalized={k:obs[k]-cost[k]+cost[f's1:B{int(k.split("B")[1])}'] for k in obs}
 assert min(corrected.values())>=0 and min(equalized.values())>=0
 baseline=run({});oracle=run(obs);trial=run(corrected);normalized_oracle=run(equalized)
 assert abs(baseline-diag['baseline_ms'])<1e-6 and abs(oracle-diag['scenarios']['middle_B']['window_ms'])<1e-4
 stats=[]
 for mb in range(4):
  rs=[r for r in rows if r['mb']==mb];x=[r['combine_sync_gap_ms'] for r in rs];y=[r['B_ms'] for r in rs];z=[b-a for a,b in zip(x,x[1:])]
  xx=[v-statistics.mean(x) for v in x];yy=[v-statistics.mean(y) for v in y];den=math.sqrt(sum(v*v for v in xx)*sum(v*v for v in yy))
  stats.append(dict(mb=mb,pearson_gap_vs_B=sum(a*b for a,b in zip(xx,yy))/den if den else None,gap_min_ms=min(x),gap_max_ms=max(x),decreasing_edges=sum(d<0 for d in z),edge_count=13))
 result=dict(status='TARGET_ONLY_CONDITIONAL_DIAGNOSTIC',rows=rows,spatial_stats=stats,baseline_ms=baseline,truth_ms=diag['truth_ms'],middle_B_oracle_ms=oracle,
   baseline_with_target_PP1_relative_gap_deltas_ms=trial,conditional_reduction_ms=baseline-trial,
   fraction_of_original_error=(baseline-trial)/(baseline-diag['truth_ms']),fraction_of_middle_B_oracle_reduction=(baseline-trial)/(baseline-oracle),
   oracle_with_gaps_equalized_to_PP1_ms=normalized_oracle,oracle_background_spatial_effect_ms=normalized_oracle-oracle,
   limits=['Target PP1 is a chosen spatial reference, NOT measured source32 Combine cost','Intervals are sync-overlapping no-recorded-device time, not proven pure wait or removable overhead','Two DAG backgrounds test conditional spatial sensitivity; not causal attribution or independent prediction','Only iter60 lane0 representative rank per stage, not EP8 readiness or cross-iteration validation','All F, PP and boundary B costs unchanged; original files not modified'])
 (a.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 (a.out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in [refs,outerp,diagp,blockp]},raw=rawrefs,code={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('audit_b_cp_gaps_r1.py'),Path(__file__).with_name('audit_b_operator_position_r1.py'),Path(__file__).with_name('audit_b_sync_owners_r1.py')]},outputs={str((a.out/'report.json').resolve()):sha(a.out/'report.json')}),indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['rows','limits']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

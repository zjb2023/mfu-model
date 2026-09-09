"""Read one pinned derived node table; no raw trace scans or model execution."""
from pathlib import Path
import csv,gzip,hashlib,json,collections
ROOT=Path(__file__).resolve().parents[4]; HERE=Path(__file__).parent
OUT=ROOT/'results/w37/A/parameter-atlas';OUT.mkdir(parents=True,exist_ok=True)
spec=json.loads((ROOT/'docs/w37/1f1b/integration/inputs.json').read_text())['inputs']
def checked(key):
 r=spec[key];p=ROOT/r['path'];b=p.read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];return p
updates=list(csv.DictReader(checked('compute_updates').open())); update_ids={r['node_id'] for r in updates}
inv=json.loads((ROOT/'results/w37/A/integration-20260907/model_cost_inventory.json').read_text());rec=inv['input_nodes'];src=ROOT/rec['path'];raw=src.read_bytes();assert len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256']
phases={}; extras=[];tail=[];total=0
keep=['node_id','kind','rank','pp_stage','pp_lane','phase','microbatch','layer_id','op_name','op_family','parallelism','duration_ns','predicted_start_ns','predicted_end_ns','compute_exposed_ns_model','compute_overlap_ns_model','network_service_ns_model','software_sync_ns_model','framework_residual_ns_model','timing_component','timing_source','source_parameter_key','critical_predecessor']
with gzip.open(src,'rt') as f:
 for r in csv.DictReader(f):
  total+=1
  if r['kind']=='framework_entry':extras.append({k:r[k] for k in keep})
  if r['phase']=='OPT' and int(r['duration_ns'])>0:tail.append({k:r[k] for k in keep})
  if r['pp_lane']!='0':continue
  if r['phase'] not in ['FWD','BWD'] or r['microbatch'] not in ['0','1','2']:continue
  key=f"{r['pp_stage']}:{r['phase']}:{r['microbatch']}"
  if key not in phases:phases[key]=dict(stage=int(r['pp_stage']),rank=int(r['rank']),phase=r['phase'],mb=int(r['microbatch']),start=1e99,end=0,count=0,zero=0,nodes=[])
  p=phases[key];p['count']+=1;p['start']=min(p['start'],int(r['predicted_start_ns']));p['end']=max(p['end'],int(r['predicted_end_ns']));duration=int(r['duration_ns'])
  if duration==0:p['zero']+=1;continue
  n={k:r[k] for k in keep};n['updated']=r['node_id'] in update_ids;p['nodes'].append(n)
assert len(extras)==1 and len([n for n in tail if n['timing_component']=='tail_ag1_completion'])==126
assert total==327746 and len(phases)==84 and len(updates)==144
payload=dict(version='v685 frozen target224',source=rec,update_source=spec['compute_updates'],phases=phases,entry=extras[0],tail=tail,updates=updates,notes='Exact lane0 node intervals from frozen prediction. Positive-cost nodes only in expanded view; zero-cost counts retained. No inferred edges or changed costs.')
(OUT/'data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
page=(HERE/'page.html').read_text().replace('__TAIL_JS__',(HERE/'tail.js').read_text()).replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
(ROOT/'docs/w37/1f1b/PARAMETER_ITER_ATLAS.html').write_text(page)
print('Built parameter atlas',len(phases),'F/B groups;',sum(len(p['nodes']) for p in phases.values()),'positive-cost nodes; 144 verified bindings')

"""Check and publish compact release acceptance, without editing sealed outputs."""
import csv,hashlib,json,re,tomllib
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'results/w37/A/v610-release';DOC=ROOT/'docs/w37/1f1b/v610'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,value):
    assert value,name
    checks.append(dict(name=name,passed=True))
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[];self.remote=[]
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if 'id'in a:self.ids.append(a['id'])
        if t=='a' and 'href'in a:self.links.append(a['href'])
        if t in ['script','img','link'] and a.get('src',a.get('href','')).startswith(('http','/')):self.remote.append(a)
records={}
for stage in ['model','evaluate','render']:
    m=json.loads((RUN/stage/'complete.json').read_text());check('stage:'+stage,m['status']=='PASS')
    for name,h in m['files'].items():check('sealed:'+stage+'/'+name,sha(RUN/stage/name)==h)
    records[stage]=dict(complete_sha256=sha(RUN/stage/'complete.json'),elapsed_s=m['elapsed_s'],files=m['files'])
spec=json.loads((DOC/'config.json').read_text())
for key,r in spec['inputs'].items():check('input:'+key,sha(Path(r['path']))==r['sha256'])
base=tomllib.loads((ROOT/'workflow/config/dag_mfu_versions.toml').read_text());overlay=tomllib.loads((ROOT/'workflow/config/dag_mfu_w37_versions.toml').read_text())
check('unique_release_version',not any(r['id']=='v6.10' for k in ['version','candidate'] for r in base.get(k,[])))
check('registered_current_release',overlay['meta']['current_development_release']=='v6.10' and len(overlay['version'])==1)
check('old_v69_design_preserved',next(r for r in base['candidate'] if r['id']=='v6.9')['status']=='DESIGN_ONLY_BLOCKED_NOT_RELEASED')
for r in json.loads((ROOT/'docs/w37/coordination/code_manifest.json').read_text())['files']:check('baseline_code_unchanged:'+r['path'],sha(ROOT/r['path'])==r['sha256'])
release=json.loads((RUN/'model/release.json').read_text());seal=json.loads((RUN/'model/prediction_seal.json').read_text())
check('full_rank_release',release['updated_nodes']==70192 and release['updated_ranks']==224 and release['nodes']==327746 and release['edges']==364784)
check('replay_exact',release['exact_prior_timings'])
check('seal_scope',seal['target_observation_read'] is False and seal['guard_probe'] and seal['prior_research_selection_known'])
doc=(RUN/'render/index.html').read_text();parsed=Parser();parsed.feed(doc)
check('unique_ids',len(parsed.ids)==len(set(parsed.ids)))
for link in parsed.links:
    if link.startswith('#'):check('anchor:'+link,link[1:] in parsed.ids)
check('offline_assets',not parsed.remote)
check('latest_only',all(x not in doc for x in ['独立候选：两类都更新','消融：只更新','v685 正式','暂不升级正式版本']))
for t in ['v6.10','70,192','4,387','6,765','10.561626','11.167230','35.788902','旧完整节点时长','源95/100','不宣称盲测']:check('visible_fact:'+t,t in doc)
payload=json.loads(re.search(r'<script id="full-iter-data" type="application/json">(.*?)</script>',doc,re.S).group(1))
check('new_graph_label',payload['prediction']['label'].startswith('A · v6.10'))
check('new_graph_times',abs(payload['prediction']['training_ms']-22930.003646125)<1e-6)
old=json.loads(Path(spec['inputs']['historical_view']['path']).read_text())
check('old_trace_semantics',payload['trace']==old['trace'])
check('new_FB_timings',payload['prediction']['bars']!=old['prediction']['bars'])
check('node_example_interior',release['example']['pp_stage']==6 and release['example']['rank']==96)
check('fine_parameters',len(json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',doc,re.S).group(1))['fine-parameters']['rows'])==6765)
check('version_not_new_blind',not release['source_full_iteration_validated'] and not release['wall_floor_independently_validated'])
evidence=DOC/'evidence';evidence.mkdir(exist_ok=True);copied={}
for stage,names in {'model':['release.json','prediction_seal.json','predictions.csv','coverage.csv'],'evaluate':['metrics.json','phase_metrics.csv','iteration_results.csv','phase_iteration_results.csv'],'render':['render_manifest.json','view_checks.json','full_iter_ledger.csv']}.items():
    for name in names:
        src=RUN/stage/name;dest=evidence/(stage+'_'+name);content=src.read_bytes().replace(b'\r\n',b'\n') if src.suffix=='.csv' else src.read_bytes();dest.write_bytes(content)
        copied[str(dest.relative_to(DOC))]=dict(sha256=sha(dest),source=str(src.relative_to(ROOT)),source_sha256=sha(src))
result=dict(status='SCIENCE_AND_DOCUMENT_PASS_BROWSER_PENDING',version='v6.10',check_count=len(checks),checks=checks,stages=records,config_sha256=sha(DOC/'config.json'),registry_sha256=sha(ROOT/'workflow/config/dag_mfu_w37_versions.toml'),html_sha256=sha(RUN/'render/index.html'),compact_evidence=copied,physical_release_directory=str(RUN.resolve().relative_to(ROOT)),render_helpers={name:sha(ROOT/'research/w37/onef1b/integration'/name) for name in ['style.css','full_iter.js','organization.py']},version_selection_authorized_after_development_scores=True,source_full_iteration_validated=False,wall_floor_independently_validated=False)
web=ROOT/'results/w37/A/v610-webcheck/acceptance.json'
if web.exists():
    w=json.loads(web.read_text())
    if w['status']=='PASS' and w['html_sha256']==result['html_sha256']:result.update(status='PASS',browser=w,browser_artifact_sha256=sha(web))
(DOC/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],checks=len(checks))))

"""Read only pinned derived inputs; no raw trace reads or model execution."""
import csv, gzip, hashlib, json, re
from collections import Counter
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, unquote
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/w37/migration-audit-20260909'
OUT.mkdir(parents=True, exist_ok=True)
files, checks, cache = [], [], {}
def check(name, ok, detail=None):
    checks.append(dict(name=name, passed=bool(ok), detail=detail))
def record(path, role, expected=None, origin=None):
    p=Path(path); p=p if p.is_absolute() else ROOT/p
    try: resolved=p.resolve(strict=True)
    except (OSError,RuntimeError): resolved=None
    row=dict(path=str(p),resolved_path=str(resolved) if resolved else None,role=role,origin=origin,expected_sha256=expected,exists=p.is_file(),symlinks=[])
    for ancestor in [p,*p.parents]:
        if ancestor.is_symlink(): row['symlinks'].append(dict(path=str(ancestor),target=str(ancestor.readlink())))
    if p.is_file():
        key=str(resolved)
        if key not in cache:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            cache[key]=(p.stat().st_size,h.hexdigest())
        row['bytes'],row['sha256']=cache[key]
    row['status']='MISSING' if not row['exists'] else ('SHA_MISMATCH' if expected and expected!=row['sha256'] else 'PASS')
    files.append(row)
    return row
specs={}
for label,rel in [('binding','docs/w37/1f1b/binding/inputs.json'),('v610','docs/w37/1f1b/v610/config.json')]:
    record(rel,'configuration'); spec=json.loads((ROOT/rel).read_text());specs[label]=spec
    for key,item in spec['inputs'].items(): record(item['path'],item.get('role','input'),item.get('sha256'),label+':'+key)
    for key,h in spec['code_inputs'].items():record(key,'pinned_code',h,label)
release=ROOT/'results/w37/A/v610-release'
for p in sorted((release/'model').iterdir()):
    if p.is_file():record(p,'release_model')
seal=json.loads((release/'model/prediction_seal.json').read_text())
for name,h in seal['files'].items():record(release/'model'/name,'sealed_model',h,'v610_seal')
for stage in ['evaluate','render']:
    for p in sorted((release/stage).iterdir()):
        if p.is_file():record(p,'release_'+stage)
# Preserve source-side ablation seal and its compact cost/prediction products.
parent=Path(specs['v610']['inputs']['parent_seal']['path'])
for name,h in json.loads(parent.read_text())['files'].items():record(parent.parent/name,'binding_sealed_evidence',h,'binding_seal')
for directory in ['research/w37/onef1b/v610','research/w37/onef1b/binding','research/w37/onef1b/history_scaleout','research/w37/onef1b/parameter_atlas','research/w37/onef1b/demo4','research/w37/onef1b/integration']:
    for p in sorted((ROOT/directory).iterdir()):
        if p.is_file():record(p,'workflow_or_report_source')
for rel in ['workflow/Snakefile','workflow/config/case256.yaml','workflow/requirements-snakemake.txt','research/w37/onef1b/serve_research_story.py','case_224gpu_pp14_cp2_a2a/config/dag_v61_kernel_calibration_2026w36.toml','docs/w37/coordination/external_inputs.json','docs/w37/coordination/source16_inventory.json','docs/w37/1f1b/history_scaleout_20260908/source_manifest.json','results/w37/A/integration-20260907/integrated_report.html']:
    record(rel,'upstream_or_history_provenance')
# Preserve inherited calibration and Task A frozen provenance, never raw trace inputs.
base=ROOT/'results/w37/A/reproduce-v685-preparation/reproduced'
for directory in ['calibration','source_replay']:
    for p in sorted((base/directory).iterdir()):
        if p.is_file():record(p,'inherited_calibration_or_source_replay')
record(base/'provenance.json','v685_provenance')
for item in json.loads((base/'provenance.json').read_text())['inputs']:
    record(item['path'],'v685_pinned_input',item['sha256'],'v685_provenance')
for item in json.loads((ROOT/'docs/w37/coordination/external_inputs.json').read_text())['files']:
    if 'A' in item.get('allowed_tasks',[]) and item.get('raw_trace') is False:
        record(item['path'],'task_A_frozen_derived_input',item['sha256'],'external_inputs')
# Additional static configuration and recorded source-attempt identities.
for name in ['inputs','parameter_inputs']:
    rel='docs/w37/1f1b/integration/'+name+'.json'
    record(rel,'report_input_manifest')
    for key,item in json.loads((ROOT/rel).read_text())['inputs'].items():
        record(item['path'], 'report_or_static_input', item.get('sha256'),rel+':'+key)
for item in json.loads((ROOT/'docs/w37/coordination/source16_inventory.json').read_text())['attempts']:
    record(item['path'],'source16_attempt_contract',item['sha256'],'source16_inventory')
# Record only upstream derived/static files; do not open paths inside raw inventories.
config_text=(ROOT/'case_224gpu_pp14_cp2_a2a/config/dag_v61_kernel_calibration_2026w36.toml').read_text()
input_section=config_text.split('[inputs]')[1].split('[mapping]')[0]
for key,value in re.findall(r'^(\w+) = "([^"]+)"',input_section,re.M):
    local=ROOT/value
    chosen=local if local.is_file() else Path('/home/zjb/Desktop/fabric-data-analysis')/value
    record(chosen,'upstream_derived_or_static_no_expected_hash',origin='v61:'+key)
check('all_pinned_inputs_and_code_match',all(r['status']=='PASS' for r in files))
# Stream full node costs rather than loading graph in memory.
COMP=['compute_exposed_ns_model','compute_overlap_ns_model','network_service_ns_model','software_sync_ns_model','framework_residual_ns_model']
def rows(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8',newline='') as f:yield from csv.DictReader(f)
counts=Counter(); totals=Counter(); invalid=0; nodes=0
for r in rows(release/'model/nodes.csv.gz'):
    nodes+=1;counts[r['kind']]+=1
    costs=[int(r[k]) for k in COMP];invalid+=int(min(costs)<0 or sum(costs)!=int(r['duration_ns']))
    for k,v in zip(COMP,costs):totals[k]+=v
check('node_count',nodes==327746,nodes);check('node_cost_components_conserved',invalid==0,invalid)
edges=sum(1 for _ in rows(release/'model/edges.csv.gz'));check('edge_count',edges==364784,edges)
bindings=list(rows(release/'model/node_parameter_bindings.csv.gz'))
check('binding_count',len(bindings)==70192,len(bindings))
check('binding_unique_nodes',len({r['node_id'] for r in bindings})==70192)
check('all_224_ranks',len({r['rank'] for r in bindings})==224)
check('all_14_stages',len({r['pp_stage'] for r in bindings})==14)
check('all_3_microbatches',len({r['microbatch'] for r in bindings})==3)
check('bound_keys',len({r['source_key'] for r in bindings})==4387)
fine_count=sum(1 for _ in rows(release/'model/fine_parameters.csv'));check('fine_parameters',fine_count==6765,fine_count)
p=next(rows(release/'model/predictions.csv'));pnum={k:float(p[k]) for k in ['raw_ms','entry_ms','onef1b_ms','tail_ms','outer_ms','profiler_ms','training_ms','mfu_pct']}
contract=json.loads(Path(specs['v610']['inputs']['v685_contract']['path']).read_text())['prediction']
check('profiler_ledger',abs(pnum['raw_ms']+contract['target_reconciliation_ms']-pnum['profiler_ms'])<1e-6)
check('training_ledger',abs(pnum['profiler_ms']+pnum['outer_ms']-pnum['training_ms'])<1e-6)
check('phase_ledger',abs(sum(pnum[k] for k in ['entry_ms','onef1b_ms','tail_ms','outer_ms'])-pnum['training_ms'])<1e-6)
mfu=100*8.436548311982576e16/(224*500e12*(pnum['training_ms']/1000))
check('mfu_formula_matches_binding_worker',abs(mfu-pnum['mfu_pct'])<1e-12)
# Bounded static dependency audit: root report and the two requested standalone atlases.
ORIGIN='http://192.168.0.49:8037'
ROUTES={'/w37-report-history.html':ROOT/'results/w37/A/history-scaleout-20260908/integrated_report.html','/w37-report.html':release/'render/index.html','/research.html':release/'render/research-index.html','/v610.html':release/'render/index.html'}
class Page(HTMLParser):
    def __init__(self,text):
        super().__init__();self.ids=set();self.links=[];self.embedded=[];self.feed(text)
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if a.get('id'):self.ids.add(a['id'])
        if a.get('name'):self.ids.add(a['name'])
        for k in ['href','src']:
            if a.get(k):self.links.append((a[k],a.get('target','')))
        if a.get('srcdoc'):self.embedded.append(a['srcdoc'])
links=[];external=[]
def audit_page(url,text,label,parent_ids=None):
    page=Page(text)
    for href,target_frame in page.links:
        if href.startswith(('data:','blob:','javascript:','mailto:')):continue
        full=urljoin(url,href);u=urlsplit(full)
        if u.netloc!=urlsplit(ORIGIN).netloc:
            external.append(dict(page=label,url=full,status='EXTERNAL_NOT_FETCHED'));continue
        target=ROUTES.get(u.path,ROOT/unquote(u.path).lstrip('/'))
        status='PASS' if target.is_file() else ('DIRECTORY' if target.is_dir() else 'MISSING_OR_SERVER_ROUTE')
        if u.fragment and u.path==urlsplit(url).path:
            ids=parent_ids if parent_ids is not None and target_frame in ('_parent','_top') else page.ids
            status='PASS' if unquote(u.fragment) in ids else 'FRAGMENT_UNRESOLVED_STATIC'
        elif u.fragment and target.is_file() and target.suffix=='.html':
            status='PASS' if unquote(u.fragment) in Page(target.read_text()).ids else 'FRAGMENT_UNRESOLVED_STATIC'
        links.append(dict(page=label,url=full,path=str(target),status=status))
        if target.is_file():record(target,'report_link_dependency',origin=label)
    for i,child in enumerate(page.embedded):audit_page(url,child,label+':srcdoc'+str(i),page.ids)
    return page
for url in ['/w37-report-history.html','/docs/w37/1f1b/PARAMETER_ITER_ATLAS.html','/docs/w37/1f1b/DAG_4GPU_DEMO.html']:
    path=ROUTES.get(url,ROOT/url.lstrip('/'));record(path,'required_report');page=audit_page(ORIGIN+url,path.read_text(),url)
    if url=='/w37-report-history.html':check('calibration_anchor_exists','calibration' in page.ids)
for directory in ['docs/w37/1f1b/history_scaleout_20260908','docs/w37/1f1b/mlperf_deepseek_20260909']:
    for p in sorted((ROOT/directory).rglob('*')):
        if p.is_file():record(p,'history_source_snapshot')
unique={r['resolved_path']:r for r in files if r['exists']}
summary=dict(scope='Pinned derived inputs, release parameters and direct HTML dependencies including srcdoc; no raw trace reads, simulation, or complete upstream rebuild',checks=checks,files_records=len(files),unique_files=len(unique),unique_bytes=sum(r['bytes'] for r in unique.values()),file_failures=[r for r in files if r['status']!='PASS'],static_link_issues=[r for r in links if r['status']!='PASS'],external_links_not_fetched=len(external),nodes=nodes,edges=edges,node_kinds=dict(counts),cost_component_sum_ns=dict(totals),predictions=pnum,mfu_constants=dict(effective_flops_per_iteration=8.436548311982576e16,gpus=224,peak_flops_per_gpu_s=500e12,source='research/w37/onef1b/binding/worker.py:123',physical_basis_independently_validated=False),symlink_policy='Keep lexical path, record resolved destination and all symlink ancestors; content SHA must match after relocation; never rewrite old seals',limitations=['Static links do not cover JS-created URLs or nested dependency closure of arbitrary linked pages.','Upstream raw data and regenerated calibration equivalence not verified.','Existing raw provenance is preserved as metadata, no raw file hashes refreshed.'])
for name,data in [('files.json',files),('report_links.json',links),('external_links.json',external),('summary.json',summary)]:
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['unique_files','unique_bytes','files_records','file_failures','static_link_issues']},ensure_ascii=False))
print('CHECKS',sum(c['passed'] for c in checks),'/',len(checks))
if summary['file_failures'] or summary['static_link_issues'] or not all(c['passed'] for c in checks):
    raise SystemExit('Migration audit requires review: inspect summary.json')

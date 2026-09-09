"""Check the rendered document and its linkage to the frozen evidence."""
import base64
import hashlib
import re
import json
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from common import *

class Document(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]; self.images=[]; self.remote_assets=[]; self.text=[]; self.tables=[]; self.table=None; self.script=False
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='img':self.images.append(a)
        if tag in ('script','img','iframe','link'):
            u=a.get('src',a.get('href',''))
            if u and not u.startswith('data:'):self.remote_assets.append(u)
        if tag=='table':self.table={'rows':0,'text':[],'dynamic':a.get('class')=='data-table' or a.get('id')=='full-iter-ledger'}
        if tag=='tr' and self.table is not None:self.table['rows']+=1
        if tag=='script':self.script=True
    def handle_endtag(self, tag):
        if tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None
        if tag=='script':self.script=False
    def handle_data(self, data):
        if not self.script:self.text.append(data)
        if self.table is not None:self.table['text'].append(data)

verified=verify_inputs()
doc=(OUT/'integrated_report.html').read_text(encoding='utf-8')
parsed=Document();parsed.feed(doc)
manifest=json.loads((OUT/'render_manifest.json').read_text())
audit=json.loads((OUT/'audit.json').read_text())
checks=[]
def check(name, ok):
    assert ok,name
    checks.append({'check':name,'passed':True})

check('render_sha_matches',digest(OUT/'integrated_report.html')==manifest['html_sha256'])
check('all_frozen_inputs_unchanged',len(verified)==80)
check('no_remote_assets_for_offline',not parsed.remote_assets)
check('unique_element_ids',len(parsed.ids)==len(set(parsed.ids)))
for href in set(parsed.links):
    if href.startswith('#'):check('anchor:'+href,href[1:] in parsed.ids)
for i,img in enumerate(parsed.images):
    check('image_label:'+str(i),bool(img.get('alt')))
    check('image_is_embedded_svg:'+str(i),img['src'].startswith('data:image/svg+xml;base64,'))
    svg=ET.fromstring(base64.b64decode(img['src'].split(',',1)[1]))
    check('image_valid_svg:'+str(i),svg.tag.endswith('svg') and bool(svg.attrib.get('viewBox')))
check('five_key_figures',len(parsed.images)==5)
visible=' '.join(parsed.text)
check('no_replacement_or_reported_mojibake',all(s not in visible for s in ['\ufffd','A鐨剆moke','閲嶆柊','姹傝']))
check('no_empty_static_table',all(t['dynamic'] or t['rows']>1 for t in parsed.tables))
check('T35_four_factor_rows',any('F倍率' in ''.join(t['text']) and t['rows']==5 for t in parsed.tables))
check('parameter_table_coverage',manifest['datasets']=={'physical-params':58,'compute-updates':144,'gradient-params':960,'version-iterations':20,'online-iterations':12,'scaleout-iterations':20})
for token in ['327,746','364,784','13,042','17,680','1,344','2,048','144','624','75.8','51.1','21.522','18.101','21.471','1,372.453390','604.439668']:
    check('reported_fact:'+token,token in visible)
check('three_scenarios_distinguished',all(t in visible for t in ['主任务 · v685','辅助场景 · T35','辅助场景 · T38']))
check('uncertainty_visible',all(t in visible for t in ['不宣称独立盲测','分子未独立核验','未验证边界']))
check('audited_terms',len(json.loads((OUT/'terminology_review.json').read_text()))==23)
full=json.loads((OUT/'full_iter_validation.json').read_text())
check('full_iter_numeric_validation',full['status']=='PASS')
check('complete_iter_first',doc.index('id="full-iter"')<doc.index('id="summary"'))
check('background_first',doc.index('<section id="training">')<doc.index('<section id="why-dag">')<doc.index('id="full-iter"'))
check('context_numeric_validation',json.loads((OUT/'context_audit.json').read_text())['status']=='PASS')
check('scaleout_scope_and_metrics', all(t in visible for t in ['45.7773%', '84.8177%', '3.2752', '条件外推', '不宣称新的独立盲测']))
check('tools_are_todo',all(t in visible for t in ['Charon','ASTRA-sim','SimAI','vTrain','Proteus','Phantora','尚未安装、复现或实测']))
check('three_stage_model_named', '三阶段 MFU v5.4 模型' in visible)
check('historical_denominator_retains_all_parts','分母仍包含准备和收尾' in visible)
inventory=json.loads((OUT/'model_cost_inventory.json').read_text())
check('cost_inventory_verified',inventory['status']=='PASS' and inventory['nodes']==327746)
check('cost_inventory_hash',digest(OUT/'model_cost_inventory.json')==manifest['model_cost_inventory_sha256'])
check('parameter_system_in_document',all(x in parsed.ids for x in ['parameter-system','compute-binding-explained','pp-model-versions']))
check('forward_and_backward_versions_explicit',all(x in visible for x in ['4.704806','0.3263','32条记录、64个d/c标量','未采用上述v687 d/c公式']))
check('all_parameter_families',all(x in visible for x in ['1 · 模型结构与算子尺寸','12 · 校准与评价规则','8 · 梯度同步、优化器与收尾']))
calibration_map=ET.fromstring((OUT/'v685_calibration_map.svg').read_text())
check('calibration_diagram_embedded', 'id="v685-calibration-map"' in doc and 'cal-map-title' in parsed.ids)
check('calibration_diagram_hash',digest(OUT/'v685_calibration_map.svg')==manifest['calibration_map_sha256'])
check('calibration_diagram_valid_svg',calibration_map.tag.endswith('svg') and calibration_map.attrib['viewBox']=='0 0 1440 1660')
# Reorganization must preserve the scientific content, even when it moves.
prior_spec=json.loads((DOC/'reorganization_input.json').read_text())
for id in prior_spec['ids']:check('legacy_id_preserved:'+id,id in parsed.ids)
def hash_text(s):return hashlib.sha256(s.encode()).hexdigest()
def payload_hash(s,id):
    data=json.loads(re.search(r'<script id="'+id+r'" type="application/json">(.*?)</script>',s,re.S).group(1))
    return hash_text(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':')))
for id,expected_hash in prior_spec['payload_sha256'].items():
    check('scientific_payload_preserved:'+id,payload_hash(doc,id)==expected_hash)
check('original_embedded_figures_preserved',sorted(hash_text(x['src']) for x in parsed.images)==prior_spec['embedded_image_sha256'])
check('original_inline_figures_preserved',sorted(hash_text(x) for x in re.findall(r'<svg\b.*?</svg>',doc,re.S))==prior_spec['inline_svg_sha256'])
check('all_old_evidence_links_preserved',set(prior_spec['links'])<=set(parsed.links))
for file,expected_hash in prior_spec['scientific_artifact_sha256'].items():
    check('scientific_artifact_byte_unchanged:'+file,digest(OUT/file)==expected_hash)
chapter_ids=re.findall(r'<section id="([^"]+)">',doc)
check('eight_technical_chapters',chapter_ids==['training','why-dag','task','calibration','development','results','diagnosis','limits'])
check('chapter_numbering',re.findall(r'<section id="[^"]+"><h2>([0-9]+) · ',doc)==[f'{i:02d}' for i in range(1,9)])
check('chronological_titles_removed',not any(t in ''.join(re.findall(r'<h[123][^>]*>(.*?)</h[123]>',doc,re.S)) for t in ['周五','周末']))
check('training_vs_performance_parameters',all(t in visible for t in ['神经网络','Transformer','参数θ','校准性能模型不等于训练大语言模型']))
check('diagnostic_scope_separate',doc.index('<section id="diagnosis">')<doc.index('id="pp-diagnosis"')<doc.index('lane0 诊断：'))
check('parameter_evidence_with_calibration',doc.index('<section id="calibration">')<doc.index('id="three-case-parameters"')<doc.index('<section id="development">'))
check('organization_manifest',digest(OUT/'organization_map.json')==manifest['organization_map_sha256'])
from binding_sections import load as load_binding
binding_manifest,binding_rows=load_binding()
check('binding_published_inputs_verified',binding_manifest['status']=='PASS')
check('binding_audit_under_calibration',doc.index('<section id="calibration">')<doc.index('id="binding-audit"')<doc.index('<section id="development">'))
check('binding_results_under_results',doc.index('<section id="results">')<doc.index('id="binding-results"')<doc.index('<section id="diagnosis">'))
for row in binding_rows('evaluate_metrics.csv'):
    for metric in ['onef1b_MAPE_pct','training_MAPE_pct','mfu_relative_MAPE_pct']:
        check('binding_score:'+row['variant']+':'+metric,f'{float(row[metric]):.4f}' in visible)
check('binding_floor_and_source_gap_visible',all(x in visible for x in ['暴露区间只能变长或不变','源侧整轮回放','35.7889','3.8594','暂不升级正式版本','13,116']))
dump('validation.json',{'status':'PASS','html_sha256':digest(OUT/'integrated_report.html'),'checks':checks,'check_count':len(checks),'evidence_checks':audit['check_count'],'frozen_input_count':len(verified),'embedded_figures':len(parsed.images),'unique_links':len(set(parsed.links)),'interactive_datasets':manifest['datasets'],'browser_validation_separate':True})
print(json.dumps({'stage':'validate','status':'PASS','checks':len(checks),'unique_links':len(set(parsed.links))}))

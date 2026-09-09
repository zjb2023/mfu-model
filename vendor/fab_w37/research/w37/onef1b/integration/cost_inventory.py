"""Inventory sealed model costs; no trace reading, replay or fitting."""
from collections import Counter, defaultdict
from common import *

verified = verify_inputs()
fields = ['duration_ns', 'compute_exposed_ns_model', 'compute_overlap_ns_model',
          'network_service_ns_model', 'software_sync_ns_model', 'framework_residual_ns_model']
groups = {}
pp = defaultdict(list)
updates = {r['node_id']: r for r in rows('compute_updates')}
positions = Counter()
examples = {}
total = 0
for r in iterrows('v685_nodes'):
    total += 1
    key = (r['kind'], r['timing_component'], r['timing_source'])
    if key not in groups:
        groups[key] = {'count': 0, 'keys': set(), 'example': r['node_id'],
                       'fields': {f: [] for f in fields}}
    g = groups[key]; g['count'] += 1
    if r['source_parameter_key']: g['keys'].add(r['source_parameter_key'])
    for f in fields: g['fields'][f].append(int(r[f]))
    if r['kind'] == 'pp_p2p': pp[r['phase']].append(r)
    if r['node_id'] in updates:
        positions[(r['phase'], r['pp_stage'])] += 1
    if r['source_parameter_key'].startswith('compute_gap|') and 'internal_key' not in examples:
        examples['internal_key'] = {k: r[k] for k in ['node_id', 'source_parameter_key', 'pp_stage', 'microbatch']}

checks = []
def check(name, ok):
    if not ok: raise AssertionError(name)
    checks.append({'check': name, 'passed': True})

check('all_nodes_covered', total == 327746)
check('forward_messages', len(pp['FWD']) == 624)
check('backward_messages', len(pp['BWD']) == 624)
check('forward_inherited_service', all(int(r['duration_ns']) == 4704806 and
      int(r['network_service_ns_model']) == 4704806 and
      int(r['software_sync_ns_model']) == 0 and int(r['framework_residual_ns_model']) == 0 and
      r['timing_source'] == 'sealed_v54_pp' and not r['source_parameter_key'] for r in pp['FWD']))
check('backward_account', all(int(r['duration_ns']) == int(r['network_service_ns_model']) +
      int(r['software_sync_ns_model']) and int(r['network_service_ns_model']) == 4704806
      for r in pp['BWD']))
check('compute_update_positions', positions == Counter({('FWD', '13'): 48, ('BWD', '0'): 48, ('BWD', '13'): 48}))
check('source_backward_grid', len(rows('gradient_parameters')) == 960)
ready = rows('readiness_parameters')
check('candidate_direction_lane_grid', len(ready) == 32 and
      len({(r['direction'], r['pp_lane']) for r in ready}) == 32)

inventory = []
for (kind, component, source), g in sorted(groups.items()):
    row = {'kind': kind, 'timing_component': component, 'timing_source': source,
           'nodes': g['count'], 'distinct_nonempty_key_strings': len(g['keys']), 'example_node': g['example']}
    for f, values in g['fields'].items():
        row[f+'_positive_nodes'] = sum(v > 0 for v in values)
        row[f+'_min'] = min(values); row[f+'_max'] = max(values)
    inventory.append(row)
write_csv('model_cost_inventory.csv', inventory)
dump('model_cost_inventory.json', {
    'status': 'PASS', 'check_count': len(checks), 'checks': checks,
    'input_nodes': load_spec()['inputs']['v685_nodes'],
    'nodes': total, 'cost_groups': len(inventory),
    'forward_pp': {'nodes': 624, 'service_ms': 4.704806, 'source': 'sealed_v54_pp'},
    'backward_pp': {'source_parameter_records': 960, 'target_nodes': 624},
    'compute_updates': [{'phase': p, 'pp_stage': int(s), 'nodes': n} for (p,s),n in sorted(positions.items())],
    'candidate_v687': {'direction_lane_records': len(ready), 'd_c_scalar_values': 2*len(ready)},
    'examples': examples,
    'limits': ['Key strings are lookup identifiers, not a count of independent fitted scalars.',
               'Component columns are accounting metadata; do not sum all fields or all nodes into an iteration.',
               'Inherited cost groups retain historical provenance; this is not full historical refitting or physical remeasurement.']})
print(json.dumps({'stage': 'cost_inventory', 'checks': len(checks), 'nodes': total, 'groups': len(inventory)}))

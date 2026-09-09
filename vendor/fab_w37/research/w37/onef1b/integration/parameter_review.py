"""Review existing static inputs without extracting traces or fitting costs."""
import ast
import csv
import json
import tomllib
from common import ROOT, OUT, DOC, digest, dump, write_csv


def build():
    records = json.loads((DOC/'parameter_inputs.json').read_text())['inputs']
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'passed': bool(ok)})
        if not ok:
            raise ValueError(name)
    for key, r in records.items():
        p = ROOT/r['path']
        check('frozen:' + key, p.stat().st_size == r['bytes'] and digest(p) == r['sha256'])
    read = lambda key: (ROOT/records[key]['path']).read_text()
    architecture = tomllib.loads(read('architecture'))['architecture']
    target = tomllib.loads(read('target_contract'))['topology']
    cases = json.loads(read('source16_target256'))
    common = {
        'hidden_size': 'hidden_size', 'num_attention_heads': 'num_attention_heads',
        'ffn_hidden_size': 'dense_ffn_hidden_size', 'moe_ffn_hidden_size': 'moe_ffn_hidden_size',
        'moe_shared_expert_intermediate_size': 'shared_expert_intermediate_size',
        'q_lora_rank': 'q_lora_rank', 'kv_lora_rank': 'kv_lora_rank',
        'qk_head_dim': 'qk_head_dim', 'qk_pos_emb_head_dim': 'qk_pos_emb_head_dim',
        'v_head_dim': 'v_head_dim', 'num_experts': 'num_experts', 'moe_router_topk': 'moe_router_topk',
    }
    observed = {}
    for key in ['source_startup', 'target_startup']:
        fields = {}
        for r in csv.DictReader(read(key).splitlines()):
            fields.setdefault(r['field'], set()).add(r['value_literal'])
        observed[key] = fields
    rows = []
    for field, archfield in common.items():
        expected = architecture[archfield]
        for case in ['source', 'target']:
            check(case + ':' + field, cases[case]['actual_arguments'][field] == expected)
        statuses = {}
        for key, values in observed.items():
            literals = values.get(field, set())
            if literals:
                check(key + ':' + field, {ast.literal_eval(v) for v in literals} == {expected})
            statuses[key] = 'LOGGED_LITERAL_MATCH' if literals else 'DECLARED_NOT_OBSERVED_IN_BOUNDED_PREFIX'
        rows.append({'field': field, 'source16': expected, 'target224': expected, 'source256': expected,
                     'target224_evidence': statuses['target_startup'], 'source256_log_crosscheck': statuses['source_startup']})
    topology_fields = [('world_size', 'world_size'), ('pp', 'pp'), ('cp', 'cp'), ('tp', 'tp'),
                       ('dp', 'dp'), ('ep', 'ep'), ('num_layers', 'num_layers'),
                       ('micro_batch_size', 'micro_batch_size'), ('global_batch_size', 'global_batch_size'),
                       ('microbatches', 'num_micro_batches'), ('sequence_length', 'sequence_length')]
    for field, targetfield in topology_fields:
        rows.append({'field': field, 'source16': cases['source'][field], 'target224': target[targetfield],
                     'source256': cases['target'][field], 'target224_evidence': 'FROZEN_RUN_CONTRACT',
                     'source256_log_crosscheck': 'COMMITTED_SCENARIO'})
    for label, mbs, seq, cp, gbs, dp, mb in [
        ('source16', cases['source']['micro_batch_size'], cases['source']['sequence_length'], cases['source']['cp'], cases['source']['global_batch_size'], cases['source']['dp'], cases['source']['microbatches']),
        ('target224', target['micro_batch_size'], target['sequence_length'], target['cp'], target['global_batch_size'], target['dp'], target['num_micro_batches']),
        ('source256', cases['target']['micro_batch_size'], cases['target']['sequence_length'], cases['target']['cp'], cases['target']['global_batch_size'], cases['target']['dp'], cases['target']['microbatches'])]:
        check(label + ':batch closure', gbs == mbs * dp * mb)
        check(label + ':sequence divisibility', seq % cp == 0)
    for field in ['micro_batch_size', 'sequence_length', 'cp', 'tp']:
        check('256→224 same static shape:' + field, target[field] == cases['target'][field])
    for key, fields in observed.items():
        for field in ['moe_pad_expert_input_to_capacity', 'moe_router_force_load_balancing']:
            check(key + ':' + field, {ast.literal_eval(v) for v in fields[field]} == {False})
    write_csv('three_case_parameters.csv', rows)
    dump('parameter_review.json', {'status': 'PASS', 'check_count': len(checks), 'checks': checks,
        'inputs': records, 'rows': rows, 'input_bytes': sum(r['bytes'] for r in records.values()),
        'scope': 'Static input compatibility, not measured latency equivalence or deployed code revision proof',
        'target_dynamic_timing_reads': 0, 'raw_trace_reads': 0, 'new_cost_fit': False})
    print(json.dumps({'stage': 'parameter_review', 'status': 'PASS', 'checks': len(checks)}))


if __name__ == '__main__':
    build()

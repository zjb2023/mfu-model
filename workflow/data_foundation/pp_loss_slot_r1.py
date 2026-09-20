"""Optional last-stage loss slot; preserve frozen pipeline implementations."""
import graphlib
import math

from pp32_to256_pipeline_r6 import build as build_pipeline


def build(p, m, params):
    """loss_tail_ms is additional, uncovered cost per microbatch, not per iter.

    Keep zero when loss is already included in the measured last-stage F.
    To split an included loss, subtract its cost from last F before enabling it.
    This reserves a scalar slot, not a measured loss implementation model.
    """
    cost = float(params.get('loss_tail_ms', 0.0))
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('loss_tail_ms must be finite and nonnegative')
    model = build_pipeline(p, m, params)
    nodes = model['nodes']
    for mb in range(m):
        forward = f's{p-1}:F{mb}'
        loss = f's{p-1}:LOSS{mb}'
        for node in nodes.values():
            node['dependencies'] = [loss if d == forward else d
                                    for d in node['dependencies']]
        nodes[loss] = dict(duration_ms=cost, dependencies=[forward],
                           kind='LOSS', stage=p-1, parameter='loss_tail_ms',
                           status='RESERVED_NOT_CALIBRATED')
    for key in graphlib.TopologicalSorter(
            {k: n['dependencies'] for k, n in nodes.items()}).static_order():
        node = nodes[key]
        node['start_ms'] = max((nodes[d]['end_ms'] for d in node['dependencies']), default=0)
        node['end_ms'] = node['start_ms'] + node['duration_ms']
    model['window_ms'] = nodes[f's0:B{m-1}']['end_ms'] - nodes['s0:F0']['start_ms']
    model['loss_slot'] = dict(parameter='loss_tail_ms', value_ms=cost,
                              scope='last PP stage, per microbatch',
                              status='RESERVED_NOT_CALIBRATED',
                              accounting='additional cost only; avoid double counting last F')
    return model

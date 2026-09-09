"""Static MLA/MoE work ledger and blocking non-interleaved 1F1B graph.

No training clocks or fitted model imports. Units are supplied explicitly by
the caller. Matrix FLOPs count multiply and add separately; recompute is a
runtime cost and is excluded from useful-model MFU numerator.
"""
from collections import defaultdict, deque
import math


def parameter_ledger(scenario, vocabulary):
    a=scenario['actual_arguments'];h=a['hidden_size'];heads=a['num_attention_heads']
    qr,kr=a['q_lora_rank'],a['kv_lora_rank']
    qk,rope,v=a['qk_head_dim'],a['qk_pos_emb_head_dim'],a['v_head_dim']
    if not (a['multi_latent_attention'] and a['swiglu'] and not a['add_bias_linear']
            and a['untie_embeddings_and_output_weights'] and a['qk_layernorm']):
        raise ValueError('unsupported static model schema')
    projections={'q_down':h*qr,'q_up':qr*heads*(qk+rope),'kv_down':h*(kr+rope),
                 'kv_up':kr*heads*(qk+v),'attention_output':heads*v*h}
    norms=2*h+qr+kr
    dense_mlp=3*h*a['ffn_hidden_size']
    shared_mlp=3*h*a['moe_shared_expert_intermediate_size']
    router=h*a['num_experts']
    one_expert=3*h*a['moe_ffn_hidden_size']
    layers=[];stage_rows=[];offset=0
    placement=scenario['layer_counts']
    if sum(placement)!=a['num_layers'] or len(placement)!=scenario['pp']:
        raise ValueError('static layer placement inconsistent')
    ep=scenario['ep']
    if a['num_experts']%ep:raise ValueError('uneven expert ownership unsupported')
    for stage,count in enumerate(placement):
        shared=0;expert=0
        for layer in range(offset,offset+count):
            moe=bool(a['moe_layer_freq'][layer])
            nonexpert=sum(projections.values())+norms+(shared_mlp+router if moe else dense_mlp)
            expert_global=one_expert*a['num_experts'] if moe else 0
            activated_weights=sum(projections.values())+(shared_mlp+router+one_expert*a['moe_router_topk'] if moe else dense_mlp)
            layers.append({'layer':layer,'pp_stage':stage,'kind':'moe' if moe else 'dense',
                           'nonexpert_parameters':nonexpert,'global_expert_parameters':expert_global,
                           'active_matrix_weights_per_token':activated_weights})
            shared+=nonexpert;expert+=expert_global//ep
        embedding=h*vocabulary if stage==0 else 0
        output=h*vocabulary if stage==scenario['pp']-1 else 0
        final_norm=h if stage==scenario['pp']-1 else 0
        stage_rows.append({'pp_stage':stage,'layers':count,'nonexpert_parameters':shared+embedding+output+final_norm,
                           'local_expert_parameters':expert,'local_total_parameters':shared+expert+embedding+output+final_norm,
                           'embedding_parameters':embedding,'output_parameters':output,'final_norm_parameters':final_norm})
        offset+=count
    total=sum(x['nonexpert_parameters']+x['local_expert_parameters']*ep for x in stage_rows)
    return {'projections_per_layer':projections,'norm_parameters_per_layer':norms,
            'one_expert_parameters':one_expert,'layers':layers,'stages':stage_rows,'global_parameters':total}


def flop_ledger(scenario, vocabulary):
    p=parameter_ledger(scenario,vocabulary);a=scenario['actual_arguments']
    seq=scenario['sequence_length'];batch=scenario['global_batch_size'];tokens=seq*batch
    matrix_forward=2*tokens*(sum(x['active_matrix_weights_per_token'] for x in p['layers'])
                              +a['hidden_size']*vocabulary)
    pair_weights=2*a['num_attention_heads']*(a['qk_head_dim']+a['qk_pos_emb_head_dim']+a['v_head_dim'])
    causal_forward=batch*a['num_layers']*pair_weights*(seq*(seq+1)//2)
    square_forward=batch*a['num_layers']*pair_weights*seq*seq
    return {'definition':'useful matrix FLOPs: F+B=3*F, causal triangular attention, no checkpoint recompute; no embedding lookup, normalization, activation or optimizer arithmetic',
            'matrix_forward_flops':matrix_forward,'causal_attention_forward_flops':causal_forward,
            'model_training_flops':3*(matrix_forward+causal_forward),
            'full_square_attention_variant_flops':3*(matrix_forward+square_forward),
            'dtype':'BF16','peak_bf16_dense_flops_per_gpu':None,
            'source':'actual static arguments and startup vocabulary; algebra independently checked against source parameter buffers',
            'parameter_ledger':p}


def blocking_graph(pp, microbatches):
    """Generate synchronous warmup/steady/cooldown PP program blocks.

    A send/receive tensor is one shared transfer node. A batched send+receive
    block releases both transfers together and waits for both; serializing the
    pair would create false deadlock. First/last-stage absent transfers vanish.
    """
    if pp<1 or microbatches<1:raise ValueError('positive PP and microbatches required')
    nodes={};edges=set();programs={}
    def compute(stage,phase,mb):
        key=f's{stage}:{phase}{mb}'
        nodes[key]={'node_id':key,'kind':'compute','pp_stage':stage,'phase':phase,'microbatch':mb}
        return key
    def transfer(boundary,phase,mb):
        if not 0<=boundary<pp-1:return None
        key=f'p{boundary}:{phase}{mb}'
        nodes[key]={'node_id':key,'kind':'pp_transfer','pp_stage':boundary,'phase':phase,'microbatch':mb}
        return key
    for stage in range(pp):
        blocks=[]
        def block(*keys):
            items=[x for x in keys if x is not None]
            if items:blocks.append(items)
        warmup=min(pp-stage-1,microbatches);remaining=microbatches-warmup
        for mb in range(warmup):
            block(transfer(stage-1,'F',mb));block(compute(stage,'F',mb));block(transfer(stage,'F',mb))
        if remaining:block(transfer(stage-1,'F',warmup))
        for i in range(remaining):
            block(compute(stage,'F',warmup+i))
            block(transfer(stage,'F',warmup+i),transfer(stage,'B',i))
            block(compute(stage,'B',i))
            block(transfer(stage-1,'B',i),transfer(stage-1,'F',warmup+i+1) if i<remaining-1 else None)
        for mb in range(remaining,microbatches):
            block(transfer(stage,'B',mb));block(compute(stage,'B',mb));block(transfer(stage-1,'B',mb))
        programs[stage]=blocks
        for left,right in zip(blocks,blocks[1:]):
            for src in left:
                for dst in right:
                    if src==dst:raise ValueError('a transfer repeated in adjacent blocks')
                    edges.add((src,dst))
    expected_compute=2*pp*microbatches;expected_comm=2*(pp-1)*microbatches
    if sum(x['kind']=='compute' for x in nodes.values())!=expected_compute or len(nodes)!=expected_compute+expected_comm:
        raise ValueError('missing graph nodes')
    return list(nodes.values()),sorted(edges),programs


def replay(nodes,edges,durations):
    ids={x['node_id'] for x in nodes}
    if len(ids)!=len(nodes) or set(durations)!=ids:raise ValueError('missing or duplicate cost keys')
    if any(isinstance(v,bool) or not math.isfinite(v) or v<0 for v in durations.values()):
        raise ValueError('finite nonnegative explicit costs required')
    parents=defaultdict(list);children=defaultdict(list);degree=dict.fromkeys(ids,0)
    for src,dst in edges:
        if src not in ids or dst not in ids:raise ValueError('edge endpoint absent')
        parents[dst].append(src);children[src].append(dst);degree[dst]+=1
    ready=deque(sorted(k for k,v in degree.items() if v==0));starts={};ends={};critical_parent={}
    while ready:
        key=ready.popleft();parent=max(parents[key],key=lambda k:(ends[k],k)) if parents[key] else None
        starts[key]=ends[parent] if parent else 0.;ends[key]=starts[key]+durations[key];critical_parent[key]=parent
        for child in sorted(children[key]):
            degree[child]-=1
            if degree[child]==0:ready.append(child)
    if len(ends)!=len(ids):raise ValueError('blocking PP graph deadlocked/cyclic')
    tail=max(ends,key=lambda k:(ends[k],k));critical=[];cursor=tail
    while cursor is not None:critical.append(cursor);cursor=critical_parent[cursor]
    critical.reverse()
    rows=[dict(x,duration=durations[x['node_id']],start=starts[x['node_id']],end=ends[x['node_id']],
               critical=x['node_id'] in critical) for x in nodes]
    return {'makespan':ends[tail],'nodes':rows,'critical_path':critical}

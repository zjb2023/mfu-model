"""Configuration-aware model-FLOP scaling for MFU comparisons."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelArchitecture:
    hidden_size: int
    sequence_length: int
    global_batch_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_pos_emb_head_dim: int
    v_head_dim: int
    dense_ffn_hidden_size: int
    moe_ffn_hidden_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    moe_router_topk: int
    vocab_size: int
    first_dense_layers: int

    def validate(self, num_layers: int) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        for name, value in vars(self).items():
            if name == "first_dense_layers":
                continue
            if value <= 0:
                raise ValueError(f"architecture {name} must be positive")
        if not 0 <= self.first_dense_layers <= num_layers:
            raise ValueError("first_dense_layers must be within the model")

    def with_global_batch_size(self, global_batch_size: int) -> "ModelArchitecture":
        return replace(self, global_batch_size=global_batch_size)


def analytical_iteration_flops(
    architecture: ModelArchitecture, num_layers: int
) -> float:
    """Return forward + backward model FLOPs using the audited MLA/MoE formula.

    The training model counts backward as twice the forward FLOPs.  The result
    is used as a relative configuration weight; a measured source FLOP total
    remains the absolute calibration anchor.
    """
    architecture.validate(num_layers)
    a = architecture
    tokens = a.global_batch_size * a.sequence_length
    query_width = a.num_attention_heads * (
        a.qk_head_dim + a.qk_pos_emb_head_dim
    )
    kv_down_width = a.kv_lora_rank + a.qk_pos_emb_head_dim
    kv_up_width = a.num_attention_heads * (a.qk_head_dim + a.v_head_dim)
    value_width = a.num_attention_heads * a.v_head_dim
    mla_linear_per_token = 2 * (
        a.hidden_size * a.q_lora_rank
        + a.q_lora_rank * query_width
        + a.hidden_size * kv_down_width
        + a.kv_lora_rank * kv_up_width
        + value_width * a.hidden_size
    )
    attention_per_token = (
        2
        * a.num_attention_heads
        * a.sequence_length
        * (a.qk_head_dim + a.qk_pos_emb_head_dim + a.v_head_dim)
    )
    dense_mlp_per_token = 6 * a.hidden_size * a.dense_ffn_hidden_size
    moe_mlp_per_token = (
        2 * a.hidden_size * a.num_experts
        + 6 * a.moe_router_topk * a.hidden_size * a.moe_ffn_hidden_size
        + 6 * a.hidden_size * a.shared_expert_intermediate_size
    )
    dense_layer = tokens * (
        mla_linear_per_token + attention_per_token + dense_mlp_per_token
    )
    moe_layer = tokens * (
        mla_linear_per_token + attention_per_token + moe_mlp_per_token
    )
    output_head = tokens * 2 * a.hidden_size * a.vocab_size
    forward_flops = (
        a.first_dense_layers * dense_layer
        + (num_layers - a.first_dense_layers) * moe_layer
        + output_head
    )
    return float(3 * forward_flops)


def scale_calibrated_model_flops(
    source_calibrated_flops: float,
    source_architecture: ModelArchitecture,
    source_num_layers: int,
    target_architecture: ModelArchitecture,
    target_num_layers: int,
) -> float:
    """Move a log-calibrated FLOP total between workload configurations."""
    if source_calibrated_flops <= 0:
        raise ValueError("source_calibrated_flops must be positive")
    source_weight = analytical_iteration_flops(
        source_architecture, source_num_layers
    )
    target_weight = analytical_iteration_flops(
        target_architecture, target_num_layers
    )
    return float(source_calibrated_flops * target_weight / source_weight)

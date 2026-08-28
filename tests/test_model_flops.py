from __future__ import annotations

import pytest

from x10000_analysis.model_flops import (
    ModelArchitecture,
    analytical_iteration_flops,
    scale_calibrated_model_flops,
)


def _architecture(global_batch_size: int = 64) -> ModelArchitecture:
    return ModelArchitecture(
        hidden_size=5120,
        sequence_length=8192,
        global_batch_size=global_batch_size,
        num_attention_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        dense_ffn_hidden_size=12288,
        moe_ffn_hidden_size=1536,
        shared_expert_intermediate_size=3072,
        num_experts=160,
        moe_router_topk=6,
        vocab_size=163840,
        first_dense_layers=1,
    )


def test_audited_source_and_target_analytical_flops() -> None:
    source = analytical_iteration_flops(_architecture(), num_layers=60)
    target = analytical_iteration_flops(
        _architecture(global_batch_size=48), num_layers=52
    )
    assert source == pytest.approx(1.2991228098379776e17)
    assert target == pytest.approx(8.470660773209702e16)
    assert target / source == pytest.approx(0.6520292545911142)


def test_target_flops_preserve_source_log_calibration() -> None:
    target = scale_calibrated_model_flops(
        1.293891072e17,
        _architecture(),
        60,
        _architecture(global_batch_size=48),
        52,
    )
    assert target == pytest.approx(8.436548311982576e16)

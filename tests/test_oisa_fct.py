from __future__ import annotations

import pytest

from x10000_analysis.oisa_fct import (
    CollectiveFctRequest,
    RecordedOisaFctProvider,
    RepresentativeOisaFctProvider,
    validate_fct_result,
)


def _request() -> CollectiveFctRequest:
    return CollectiveFctRequest(
        iteration=55,
        pp_stage=3,
        behavior="dp_grad_reduce_scatter",
        kind="dp_rs",
        round=0,
        group_key="pg=7",
        group_size=2,
        observed_ranks=2,
        payload_bytes=1024,
        rank_release_offsets_ns=((6, 0), (7, 10)),
        reference_tail_after_last_release_ns=20,
    )


def test_recorded_oisa_result_can_replace_mock_provider() -> None:
    request = _request()
    provider = RecordedOisaFctProvider(
        [
            {
                "request_id": request.request_id,
                "first_release_ns": 0,
                "last_release_ns": 10,
                "last_flow_end_ns": 40,
                "collective_elapsed_ns": 40,
                "arrival_span_ns": 10,
                "tail_after_last_release_ns": 30,
                "rank_network_done_offsets_ns": {"6": 40, "7": 35},
                "source": "oisa_ns3",
                "simulator_commit": "abc123",
                "topology_hash": "topo456",
            }
        ]
    )
    result = validate_fct_result(request, provider(request))
    assert request.op == "reduce_scatter"
    assert result.collective_elapsed_ns == 40
    assert result.tail_after_last_release_ns == 30
    assert result.rank_network_done_offsets_ns == {6: 40, 7: 35}


def test_recorded_oisa_result_must_match_release_signature() -> None:
    request = _request()
    provider = RecordedOisaFctProvider(
        [
            {
                "request_id": request.request_id,
                "first_release_ns": 0,
                "last_release_ns": 11,
                "last_flow_end_ns": 40,
                "collective_elapsed_ns": 40,
                "arrival_span_ns": 11,
                "tail_after_last_release_ns": 29,
            }
        ]
    )
    with pytest.raises(ValueError, match="arrival span"):
        validate_fct_result(request, provider(request))


def test_representative_oisa_scales_only_tail_by_payload() -> None:
    request = _request()
    provider = RepresentativeOisaFctProvider(
        [
            {
                "kind": "dp_rs",
                "request_id": "representative_dp_rs",
                "reference_payload_bytes": 512,
                "tail_after_last_release_ns": 30,
                "simulator_commit": "abc123",
                "topology_hash": "topo456",
            }
        ]
    )
    result = validate_fct_result(request, provider(request))
    assert result.arrival_span_ns == 10
    assert result.tail_after_last_release_ns == 60
    assert result.collective_elapsed_ns == 70
    assert result.source == "oisa_representative_linear:representative_dp_rs"


def test_representative_oisa_can_preserve_signed_trace_calibration_residual() -> None:
    request = _request()
    provider = RepresentativeOisaFctProvider(
        [
            {
                "kind": "dp_rs",
                "request_id": "representative_dp_rs",
                "reference_payload_bytes": 512,
                "tail_after_last_release_ns": 45,
                "baseline_tail_after_last_release_ns": 5,
            }
        ],
        preserve_trace_calibration_residual=True,
    )
    result = validate_fct_result(request, provider(request))
    assert result.network_tail_after_last_release_ns == 90
    assert result.trace_calibration_residual_ns == 10
    assert result.software_residual_ns == 10
    assert result.simulator_bias_correction_ns == 0
    assert result.tail_after_last_release_ns == 100
    assert result.collective_elapsed_ns == 110
    assert result.source == (
        "oisa_representative_linear+trace_calibration_residual:representative_dp_rs"
    )


def test_representative_oisa_preserves_negative_simulator_bias() -> None:
    request = _request()
    provider = RepresentativeOisaFctProvider(
        [
            {
                "kind": "dp_rs",
                "reference_payload_bytes": 512,
                "tail_after_last_release_ns": 30,
            }
        ],
        preserve_trace_calibration_residual=True,
    )
    result = validate_fct_result(request, provider(request))
    assert result.network_tail_after_last_release_ns == 60
    assert result.trace_calibration_residual_ns == -40
    assert result.software_residual_ns == 0
    assert result.simulator_bias_correction_ns == -40
    assert result.tail_after_last_release_ns == 20

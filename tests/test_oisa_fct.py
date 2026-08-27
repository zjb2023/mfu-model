from __future__ import annotations

import pytest

from x10000_analysis.oisa_fct import (
    CollectiveFctRequest,
    RecordedOisaFctProvider,
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

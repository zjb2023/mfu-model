"""OISA collective-FCT boundary for the trace-driven MFU DAG.

The MFU model owns rank arrival times and dependency propagation.  An FCT
provider owns only the network interval after those ranks are released.  The
request/response objects in this module deliberately keep that boundary small
so the current deterministic mock can later be replaced by an OISA process or
RPC client without changing the DAG executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from typing import Iterable, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class CollectiveFctRequest:
    """One collective query with release offsets relative to its first rank."""

    iteration: int
    pp_stage: int
    behavior: str
    kind: str
    round: int
    group_key: str
    group_size: int
    observed_ranks: int
    payload_bytes: int | None
    rank_release_offsets_ns: tuple[tuple[int, int], ...]
    reference_tail_after_last_release_ns: int
    topology_id: str = "target_cluster"
    traffic_matrix_id: str = ""

    @property
    def group_id(self) -> str:
        safe_group = self.group_key.replace(" ", "_")
        return (
            f"iter{self.iteration}:pp{self.pp_stage}:{self.kind}:"
            f"round{self.round}:{safe_group}"
        )

    @property
    def release_signature(self) -> str:
        encoded = ",".join(
            f"{rank}={offset}" for rank, offset in self.rank_release_offsets_ns
        )
        return sha1(encoded.encode("utf-8")).hexdigest()[:12]

    @property
    def request_id(self) -> str:
        # The same group may be queried more than once with different arrivals
        # while an upstream counterfactual propagates through the DAG.
        return f"{self.group_id}:rel-{self.release_signature}"

    @property
    def op(self) -> str:
        if self.kind in {"dp_rs", "edp_rs"}:
            return "reduce_scatter"
        if self.kind in {"dp_ag0", "edp_ag0", "dp_ag1", "edp_ag1"}:
            return "all_gather"
        if "all_to_all" in self.kind or "alltoall" in self.kind:
            return "all_to_all"
        return self.kind

    @property
    def arrival_span_ns(self) -> int:
        return max(offset for _rank, offset in self.rank_release_offsets_ns)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(rank for rank, _offset in self.rank_release_offsets_ns)

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "group_id": self.group_id,
            "iteration": self.iteration,
            "pp_stage": self.pp_stage,
            "behavior": self.behavior,
            "kind": self.kind,
            "op": self.op,
            "round": self.round,
            "group_key": self.group_key,
            "group_size": self.group_size,
            "observed_ranks": self.observed_ranks,
            "payload_bytes": self.payload_bytes,
            "group_ranks": ",".join(str(rank) for rank in self.ranks),
            "rank_release_offsets_ns": ";".join(
                f"{rank}:{offset}"
                for rank, offset in self.rank_release_offsets_ns
            ),
            "arrival_span_ns": self.arrival_span_ns,
            "reference_tail_after_last_release_ns": (
                self.reference_tail_after_last_release_ns
            ),
            "topology_id": self.topology_id,
            "traffic_matrix_id": self.traffic_matrix_id,
        }


@dataclass(frozen=True)
class CollectiveFctResult:
    """Collective completion returned by OISA or a compatible provider.

    ``tail_after_last_release_ns`` is the duration consumed by the MFU DAG.
    A raw OISA result contains only network time.  A calibrated provider may
    add a signed Trace-derived calibration residual while retaining its
    positive software-like and negative simulator-bias components for audit.
    """

    request_id: str
    first_release_ns: int
    last_release_ns: int
    last_flow_end_ns: int
    collective_elapsed_ns: int
    arrival_span_ns: int
    tail_after_last_release_ns: int
    source: str
    rank_network_done_offsets_ns: Mapping[int, int] | None = None
    simulator_commit: str = ""
    topology_hash: str = ""
    network_tail_after_last_release_ns: int | None = None
    trace_calibration_residual_ns: int = 0
    software_residual_ns: int = 0
    simulator_bias_correction_ns: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "first_release_ns": self.first_release_ns,
            "last_release_ns": self.last_release_ns,
            "last_flow_end_ns": self.last_flow_end_ns,
            "collective_elapsed_ns": self.collective_elapsed_ns,
            "arrival_span_ns": self.arrival_span_ns,
            "tail_after_last_release_ns": self.tail_after_last_release_ns,
            "network_tail_after_last_release_ns": (
                self.network_tail_after_last_release_ns
                if self.network_tail_after_last_release_ns is not None
                else self.tail_after_last_release_ns
            ),
            "trace_calibration_residual_ns": self.trace_calibration_residual_ns,
            "software_residual_ns": self.software_residual_ns,
            "simulator_bias_correction_ns": self.simulator_bias_correction_ns,
            "rank_network_done_offsets_ns": (
                json.dumps(
                    {
                        str(rank): int(offset)
                        for rank, offset in sorted(
                            self.rank_network_done_offsets_ns.items()
                        )
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if self.rank_network_done_offsets_ns is not None
                else ""
            ),
            "source": self.source,
            "simulator_commit": self.simulator_commit,
            "topology_hash": self.topology_hash,
        }


class CollectiveFctProvider(Protocol):
    """Callable boundary implemented by mocks, local OISA, or an RPC client."""

    def __call__(self, request: CollectiveFctRequest) -> CollectiveFctResult: ...


def make_collective_request(group: object) -> CollectiveFctRequest:
    """Build and validate a normalized request from one per-rank group frame."""
    starts = {
        int(row.rank): int(row.predicted_start_ns)
        for row in group.itertuples(index=False)
    }
    if not starts:
        raise ValueError("collective FCT request cannot be empty")
    first = min(starts.values())
    offsets = tuple(sorted((rank, start - first) for rank, start in starts.items()))
    if offsets[0][1] < 0 or min(offset for _rank, offset in offsets) != 0:
        raise ValueError("collective release offsets must be normalized to zero")
    first_row = group.iloc[0]
    if len(offsets) != int(first_row["observed_ranks"]):
        raise ValueError("collective release ranks do not match observed_ranks")
    if int(first_row["observed_ranks"]) > int(first_row["group_size"]):
        raise ValueError("collective observed_ranks exceeds configured group_size")
    reference_tails = group["service_ns"].astype("int64").unique()
    if len(reference_tails) != 1:
        raise ValueError("collective group has inconsistent reference service")
    payload = None
    if "payload_bytes" in group:
        payloads = group["payload_bytes"].astype("int64").unique()
        if len(payloads) != 1:
            raise ValueError("collective group has inconsistent payload")
        payload = int(payloads[0])
    return CollectiveFctRequest(
        iteration=int(first_row["iteration"]),
        pp_stage=int(first_row["pp_stage"]),
        behavior=str(first_row["behavior"]),
        kind=str(first_row["kind"]),
        round=int(first_row["round"]),
        group_key=str(first_row["group_key"]),
        group_size=int(first_row["group_size"]),
        observed_ranks=int(first_row["observed_ranks"]),
        payload_bytes=payload,
        rank_release_offsets_ns=offsets,
        reference_tail_after_last_release_ns=int(reference_tails[0]),
    )


def validate_fct_result(
    request: CollectiveFctRequest, result: CollectiveFctResult
) -> CollectiveFctResult:
    """Reject responses that would double-count arrivals in the MFU DAG."""
    numeric = {
        "first_release_ns": result.first_release_ns,
        "last_release_ns": result.last_release_ns,
        "last_flow_end_ns": result.last_flow_end_ns,
        "collective_elapsed_ns": result.collective_elapsed_ns,
        "arrival_span_ns": result.arrival_span_ns,
        "tail_after_last_release_ns": result.tail_after_last_release_ns,
    }
    if any(not np.isfinite(value) or value < 0 for value in numeric.values()):
        raise ValueError(f"OISA FCT result contains invalid timing: {numeric}")
    if result.request_id != request.request_id:
        raise ValueError(
            f"OISA request id mismatch: {result.request_id!r} != {request.request_id!r}"
        )
    if result.first_release_ns != 0:
        raise ValueError("OISA release offsets must be relative to first_release_ns=0")
    if result.arrival_span_ns != request.arrival_span_ns:
        raise ValueError("OISA response arrival span does not match the MFU request")
    if result.last_release_ns != result.arrival_span_ns:
        raise ValueError("OISA last_release_ns must equal normalized arrival span")
    if result.last_flow_end_ns != result.collective_elapsed_ns:
        raise ValueError("normalized OISA last_flow_end must equal collective elapsed")
    if (
        result.collective_elapsed_ns
        != result.arrival_span_ns + result.tail_after_last_release_ns
    ):
        raise ValueError(
            "OISA result must conserve elapsed = arrival_span + tail_after_last_release"
        )
    if result.network_tail_after_last_release_ns is not None:
        if result.network_tail_after_last_release_ns < 0:
            raise ValueError("OISA network tail must be non-negative")
        if result.software_residual_ns < 0:
            raise ValueError("Trace software residual must be non-negative")
        if result.simulator_bias_correction_ns > 0:
            raise ValueError("simulator bias correction must be non-positive")
        if (
            result.tail_after_last_release_ns
            != result.network_tail_after_last_release_ns
            + result.trace_calibration_residual_ns
        ):
            raise ValueError(
                "collective tail must equal network tail plus Trace calibration residual"
            )
        if result.software_residual_ns != max(
            result.trace_calibration_residual_ns, 0
        ):
            raise ValueError("software residual is not the positive calibration component")
        if result.simulator_bias_correction_ns != min(
            result.trace_calibration_residual_ns, 0
        ):
            raise ValueError("simulator bias is not the negative calibration component")
    if result.rank_network_done_offsets_ns is not None:
        done = {
            int(rank): int(offset)
            for rank, offset in result.rank_network_done_offsets_ns.items()
        }
        if set(done) != set(request.ranks):
            raise ValueError("OISA rank done offsets do not cover the request ranks")
        releases = dict(request.rank_release_offsets_ns)
        if any(done[rank] < releases[rank] for rank in request.ranks):
            raise ValueError("OISA rank completed before its release offset")
        if any(offset > result.last_flow_end_ns for offset in done.values()):
            raise ValueError("OISA rank completion falls after last_flow_end_ns")
        if max(done.values()) != result.last_flow_end_ns:
            raise ValueError("OISA rank completions do not conserve last_flow_end_ns")
    return result


class TraceReferenceFctProvider:
    """Reference provider reproducing the measured tail after the last arrival."""

    source = "trace_reference"
    supports_dynamic_arrivals = True

    def __call__(self, request: CollectiveFctRequest) -> CollectiveFctResult:
        span = request.arrival_span_ns
        tail = request.reference_tail_after_last_release_ns
        elapsed = span + tail
        return CollectiveFctResult(
            request_id=request.request_id,
            first_release_ns=0,
            last_release_ns=span,
            last_flow_end_ns=elapsed,
            collective_elapsed_ns=elapsed,
            arrival_span_ns=span,
            tail_after_last_release_ns=tail,
            source=self.source,
        )


class ScaledMockOisaFctProvider:
    """Deterministic stand-in for OISA while its staggered-rank path is built.

    Only the network tail after the last release is scaled.  The arrival span
    remains owned by the MFU DAG and is therefore never multiplied.
    """

    def __init__(
        self,
        kind_scales: Mapping[str, float] | None = None,
        *,
        default_scale: float = 1.0,
        simulator_commit: str = "mock",
        topology_hash: str = "mock-target-cluster",
    ) -> None:
        scales = {str(key): float(value) for key, value in (kind_scales or {}).items()}
        values = [float(default_scale), *scales.values()]
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("mock OISA scales must be finite and non-negative")
        self.kind_scales = scales
        self.default_scale = float(default_scale)
        self.simulator_commit = simulator_commit
        self.topology_hash = topology_hash
        self.supports_dynamic_arrivals = True

    def __call__(self, request: CollectiveFctRequest) -> CollectiveFctResult:
        scale = self.kind_scales.get(request.kind, self.default_scale)
        span = request.arrival_span_ns
        tail = int(round(request.reference_tail_after_last_release_ns * scale))
        elapsed = span + tail
        return CollectiveFctResult(
            request_id=request.request_id,
            first_release_ns=0,
            last_release_ns=span,
            last_flow_end_ns=elapsed,
            collective_elapsed_ns=elapsed,
            arrival_span_ns=span,
            tail_after_last_release_ns=tail,
            source=f"mock_oisa_scale_{scale:g}",
            simulator_commit=self.simulator_commit,
            topology_hash=self.topology_hash,
        )


class RecordingFctProvider:
    """Record requests and responses without changing provider semantics."""

    def __init__(self, provider: CollectiveFctProvider) -> None:
        self.provider = provider
        self.supports_dynamic_arrivals = bool(
            getattr(provider, "supports_dynamic_arrivals", True)
        )
        self.requests: list[dict[str, object]] = []
        self.responses: list[dict[str, object]] = []

    def __call__(self, request: CollectiveFctRequest) -> CollectiveFctResult:
        result = validate_fct_result(request, self.provider(request))
        self.requests.append(request.as_dict())
        self.responses.append(result.as_dict())
        return result


class RecordedOisaFctProvider:
    """Replay real OISA JSON/CSV records by request id.

    A caller can pass ``pandas.read_csv(path).to_dict(orient="records")`` or
    decoded JSON objects.  The MFU executor still validates the response
    identity and arrival-span conservation at the call site.
    """

    required_fields = {
        "request_id",
        "first_release_ns",
        "last_release_ns",
        "last_flow_end_ns",
        "collective_elapsed_ns",
        "arrival_span_ns",
        "tail_after_last_release_ns",
    }

    def __init__(self, records: Iterable[Mapping[str, object]]) -> None:
        self.supports_dynamic_arrivals = False
        self.results: dict[str, CollectiveFctResult] = {}
        for record in records:
            missing = sorted(self.required_fields - set(record))
            if missing:
                raise ValueError(f"recorded OISA result missing fields: {missing}")
            request_id = str(record["request_id"])
            if request_id in self.results:
                raise ValueError(f"duplicate recorded OISA request id: {request_id}")
            raw_rank_done = record.get("rank_network_done_offsets_ns")
            if raw_rank_done is None or (
                isinstance(raw_rank_done, float) and np.isnan(raw_rank_done)
            ) or raw_rank_done == "":
                rank_done = None
            else:
                if isinstance(raw_rank_done, str):
                    raw_rank_done = json.loads(raw_rank_done)
                if not isinstance(raw_rank_done, Mapping):
                    raise ValueError(
                        "recorded OISA rank_network_done_offsets_ns must be a mapping"
                    )
                rank_done = {
                    int(rank): int(offset)
                    for rank, offset in raw_rank_done.items()
                }
            self.results[request_id] = CollectiveFctResult(
                request_id=request_id,
                first_release_ns=int(record["first_release_ns"]),
                last_release_ns=int(record["last_release_ns"]),
                last_flow_end_ns=int(record["last_flow_end_ns"]),
                collective_elapsed_ns=int(record["collective_elapsed_ns"]),
                arrival_span_ns=int(record["arrival_span_ns"]),
                tail_after_last_release_ns=int(record["tail_after_last_release_ns"]),
                source=str(record.get("source", "oisa_recorded")),
                rank_network_done_offsets_ns=rank_done,
                simulator_commit=str(record.get("simulator_commit", "")),
                topology_hash=str(record.get("topology_hash", "")),
            )
        if not self.results:
            raise ValueError("recorded OISA provider requires at least one result")

    def __call__(self, request: CollectiveFctRequest) -> CollectiveFctResult:
        try:
            return self.results[request.request_id]
        except KeyError as exc:
            raise ValueError(
                "recorded OISA results do not cover request "
                f"{request.request_id}; rerun OISA for these rank release offsets"
            ) from exc


class RepresentativeOisaFctProvider:
    """Reuse one measured OISA point per kind with byte-proportional scaling.

    This provider is intentionally explicit about being a single-point model,
    not an exact replay.  It preserves the request's dynamic rank-arrival span
    and scales only the network tail after the last release.
    """

    required_fields = {
        "kind",
        "reference_payload_bytes",
        "tail_after_last_release_ns",
    }

    def __init__(
        self,
        records: Iterable[Mapping[str, object]],
        *,
        preserve_trace_calibration_residual: bool = False,
    ) -> None:
        self.supports_dynamic_arrivals = True
        self.preserve_trace_calibration_residual = bool(
            preserve_trace_calibration_residual
        )
        self.records: dict[str, dict[str, object]] = {}
        for raw in records:
            missing = sorted(self.required_fields - set(raw))
            if missing:
                raise ValueError(
                    f"representative OISA calibration missing fields: {missing}"
                )
            kind = str(raw["kind"])
            if kind in self.records:
                raise ValueError(f"duplicate representative OISA kind: {kind}")
            payload = int(raw["reference_payload_bytes"])
            tail = int(raw["tail_after_last_release_ns"])
            if payload <= 0 or tail < 0:
                raise ValueError("representative OISA payload/tail is invalid")
            self.records[kind] = dict(raw)
        if not self.records:
            raise ValueError("representative OISA provider requires calibration rows")

    def __call__(self, request: CollectiveFctRequest) -> CollectiveFctResult:
        try:
            record = self.records[request.kind]
        except KeyError as exc:
            raise ValueError(
                f"representative OISA calibration does not cover kind {request.kind}"
            ) from exc
        if request.payload_bytes is None or request.payload_bytes <= 0:
            raise ValueError("representative OISA scaling requires request payload_bytes")
        reference_payload = int(record["reference_payload_bytes"])
        reference_tail = int(record["tail_after_last_release_ns"])
        network_tail = int(
            round(reference_tail * request.payload_bytes / reference_payload)
        )
        baseline_reference_tail = int(
            record.get("baseline_tail_after_last_release_ns", reference_tail)
        )
        baseline_network_tail = int(
            round(
                baseline_reference_tail
                * request.payload_bytes
                / reference_payload
            )
        )
        calibration_residual = 0
        if self.preserve_trace_calibration_residual:
            calibration_residual = (
                request.reference_tail_after_last_release_ns
                - baseline_network_tail
            )
        software_residual = max(calibration_residual, 0)
        simulator_bias = min(calibration_residual, 0)
        tail = network_tail + calibration_residual
        if tail < 0:
            raise ValueError(
                "target OISA network delta exceeds the calibrated Trace tail"
            )
        span = request.arrival_span_ns
        elapsed = span + tail
        representative_id = str(record.get("request_id", request.kind))
        source_suffix = (
            "+trace_calibration_residual"
            if self.preserve_trace_calibration_residual
            else ""
        )
        return CollectiveFctResult(
            request_id=request.request_id,
            first_release_ns=0,
            last_release_ns=span,
            last_flow_end_ns=elapsed,
            collective_elapsed_ns=elapsed,
            arrival_span_ns=span,
            tail_after_last_release_ns=tail,
            source=(
                f"oisa_representative_linear{source_suffix}:"
                f"{representative_id}"
            ),
            simulator_commit=str(record.get("simulator_commit", "")),
            topology_hash=str(record.get("topology_hash", "")),
            network_tail_after_last_release_ns=network_tail,
            trace_calibration_residual_ns=calibration_residual,
            software_residual_ns=software_residual,
            simulator_bias_correction_ns=simulator_bias,
        )


class SelectiveFctProvider:
    """Use a candidate provider for one group and a reference for all others."""

    def __init__(
        self,
        target_group_id: str,
        reference: CollectiveFctProvider,
        candidate: CollectiveFctProvider,
    ) -> None:
        self.target_group_id = target_group_id
        self.reference = reference
        self.candidate = candidate
        self.supports_dynamic_arrivals = bool(
            getattr(reference, "supports_dynamic_arrivals", True)
            and getattr(candidate, "supports_dynamic_arrivals", True)
        )

    def __call__(self, request: CollectiveFctRequest) -> CollectiveFctResult:
        provider = (
            self.candidate if request.group_id == self.target_group_id else self.reference
        )
        return provider(request)

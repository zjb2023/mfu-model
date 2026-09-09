from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ConfigFact:
    """A typed configuration value tied to the exact source line that states it."""

    key: str
    value: Any
    source_path: str
    source_line: int
    source_excerpt: str
    raw_value: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_TRAINING_FACT_TYPES: dict[str, type] = {
    "attention_softmax_in_fp32": bool,
    "bf16": bool,
    "context_parallel_size": int,
    "data_parallel_size": int,
    "encoder_num_layers": int,
    "encoder_seq_length": int,
    "exp_avg_dtype": str,
    "exp_avg_sq_dtype": str,
    "expert_model_parallel_size": int,
    "expert_tensor_parallel_size": int,
    "global_batch_size": int,
    "hidden_size": int,
    "main_grads_dtype": str,
    "main_params_dtype": str,
    "micro_batch_size": int,
    "moe_aux_loss_coeff": float,
    "moe_enable_deepep": bool,
    "moe_ffn_hidden_size": int,
    "moe_grouped_gemm": bool,
    "moe_router_dtype": str,
    "moe_router_load_balancing_type": str,
    "moe_router_pre_softmax": bool,
    "moe_router_score_function": str,
    "moe_router_topk": int,
    "moe_shared_expert_intermediate_size": int,
    "moe_shared_expert_overlap": bool,
    "moe_token_dispatcher_type": str,
    "mtp_num_layers": int,
    "multi_latent_attention": bool,
    "normalization": str,
    "num_attention_heads": int,
    "num_experts": int,
    "num_layers": int,
    "optimizer": str,
    "overlap_grad_reduce": bool,
    "overlap_moe_expert_parallel_comm": bool,
    "overlap_param_gather": bool,
    "params_dtype": str,
    "pipeline_model_parallel_size": int,
    "position_embedding_type": str,
    "recompute_granularity": str,
    "recompute_method": str,
    "recompute_num_layers": int,
    "seq_length": int,
    "swiglu": bool,
    "tensor_model_parallel_size": int,
    "use_distributed_optimizer": bool,
}

_TOML_FACT_TYPES: dict[str, type] = {
    "world_size": int,
    "host_split_rank": int,
    "profiler_iter_start": int,
    "profiler_iter_end": int,
    "profiler_iter_step": int,
}
_ARGUMENT_RE = re.compile(r"^  (?P<key>[a-zA-Z0-9_]+)\s+\.{2,}\s(?P<value>.*)$")


def _typed_value(raw_value: str, expected_type: type, *, key: str) -> Any:
    text = raw_value.strip()
    if expected_type is str:
        if not text:
            raise ValueError(f"Required configuration key {key!r} has an empty value")
        value: Any = text
    else:
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Required configuration key {key!r} has unparseable value {text!r}"
            ) from exc
    if type(value) is not expected_type:
        raise TypeError(
            f"Required configuration key {key!r} must be {expected_type.__name__}; "
            f"got {value!r} ({type(value).__name__})"
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError(f"Required configuration key {key!r} must be finite; got {value!r}")
    return value


def _toml_line_facts(config_path: Path, workspace: Path) -> dict[str, ConfigFact]:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    in_run = False
    found: dict[str, ConfigFact] = {}
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("["):
            in_run = stripped == "[run]"
            continue
        if not in_run or "=" not in line:
            continue
        key, raw = (part.strip() for part in line.split("=", 1))
        if key not in _TOML_FACT_TYPES:
            continue
        if key in found:
            raise ValueError(f"Duplicate [run] key {key!r} in {config_path}")
        value = _typed_value(raw, _TOML_FACT_TYPES[key], key=f"run.{key}")
        found[key] = ConfigFact(
            key=f"run.{key}",
            value=value,
            source_path=str(config_path.relative_to(workspace)),
            source_line=line_number,
            source_excerpt=line,
            raw_value=raw,
        )
    missing = sorted(set(_TOML_FACT_TYPES) - set(found))
    if missing:
        raise ValueError(f"Missing required [run] keys in {config_path}: {missing}")
    return {fact.key: fact for fact in found.values()}


def _rank0_line_facts(log_path: Path) -> dict[str, ConfigFact]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    in_arguments = False
    occurrences: dict[str, list[ConfigFact]] = {
        key: [] for key in REQUIRED_TRAINING_FACT_TYPES
    }
    for line_number, line in enumerate(lines, start=1):
        if line.startswith("------------------------ arguments"):
            if in_arguments:
                raise ValueError(f"Nested argument table in {log_path}")
            in_arguments = True
            continue
        if line.startswith("-------------------- end of arguments"):
            in_arguments = False
            continue
        if not in_arguments:
            continue
        match = _ARGUMENT_RE.match(line)
        if match is None:
            continue
        key = match.group("key")
        if key not in occurrences:
            continue
        raw_value = match.group("value")
        value = _typed_value(raw_value, REQUIRED_TRAINING_FACT_TYPES[key], key=key)
        occurrences[key].append(
            ConfigFact(
                key=key,
                value=value,
                source_path=str(log_path),
                source_line=line_number,
                source_excerpt=line,
                raw_value=raw_value,
            )
        )
    missing = sorted(key for key, values in occurrences.items() if not values)
    duplicates = sorted(key for key, values in occurrences.items() if len(values) > 1)
    if missing or duplicates:
        raise ValueError(
            f"RANK0 argument table must contain every required key exactly once; "
            f"missing={missing}, duplicates={duplicates}, path={log_path}"
        )
    return {key: values[0] for key, values in occurrences.items()}


def extract_training_config_facts(workspace: str | Path) -> dict[str, ConfigFact]:
    """Read typed facts from one RANK0 argument table and the run TOML, failing closed."""

    root = Path(workspace).resolve()
    config_path = root / "configs/run1555_v5.toml"
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    training_log_dir = Path(config["paths"]["training_log_dir"])
    rank0_logs = sorted(training_log_dir.glob("*.RANK0.*.log"))
    if len(rank0_logs) != 1:
        raise ValueError(
            f"Expected exactly one RANK0 training log in {training_log_dir}; "
            f"found {len(rank0_logs)}: {[path.name for path in rank0_logs]}"
        )
    facts = _rank0_line_facts(rank0_logs[0])
    facts.update(_toml_line_facts(config_path, root))
    return facts


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Required payload field {field} must be numeric; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Required payload field {field} must be finite; got {value!r}")
    return result


def _median_field(rows: list[dict[str, Any]], field: str, *, allow_null: bool = False) -> float | None:
    values = [row.get(field) for row in rows]
    if allow_null:
        values = [value for value in values if value is not None]
        if not values:
            return None
    if not values or any(value is None for value in values):
        raise ValueError(f"Cannot compute median for required field {field!r} from {len(rows)} rows")
    numeric = sorted(_finite_number(value, field=field) for value in values)
    middle = len(numeric) // 2
    if len(numeric) % 2:
        return numeric[middle]
    return (numeric[middle - 1] + numeric[middle]) / 2.0


def _require_direct_domain(
    rows: list[dict[str, Any]], iterations: list[int], *, label: str
) -> None:
    actual = [int(row["iter"]) for row in rows]
    if len(actual) != len(iterations) or set(actual) != set(iterations):
        raise ValueError(
            f"{label} must cover the explicit direct iteration domain exactly once; "
            f"expected={iterations}, actual={actual}"
        )


def _summarize_phase_timeline(
    phase_groups: list[dict[str, Any]], typical_iter: int
) -> dict[str, Any]:
    matches = [group for group in phase_groups if int(group["iter"]) == typical_iter]
    if len(matches) != 1:
        raise ValueError(f"Expected one phase_intervals group for iter {typical_iter}; got {len(matches)}")
    intervals = matches[0].get("intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(f"phase_intervals for iter {typical_iter} must be a non-empty list")

    summaries: dict[str, dict[str, Any]] = {}
    for interval in intervals:
        start = _finite_number(interval.get("start_ms"), field="phase.start_ms")
        end = _finite_number(interval.get("end_ms"), field="phase.end_ms")
        if end < start:
            raise ValueError(f"Negative phase interval at iter {typical_iter}: {interval}")
        exclusive = str(interval.get("exclusive", "idle_or_unattributed"))
        labels = [str(value) for value in interval.get("labels", [])]
        joined = "|".join([exclusive, *labels])
        if "deepep_forward_scaleup" in joined:
            category = "deepep_forward_scaleup"
        elif "deepep_backward_scaleup" in joined:
            category = "deepep_backward_scaleup"
        elif "edp_rs_scaleout" in joined:
            category = "edp_rs_scaleout"
        elif "edp_ag_scaleout" in joined:
            category = "edp_ag_scaleout"
        elif "dp_rs_hybrid" in joined:
            category = "dp_rs_hybrid"
        elif "dp_ag_hybrid" in joined:
            category = "dp_ag_hybrid"
        elif "gradient_finalize_sync" in joined:
            category = "gradient_finalize_sync"
        elif "tiny_sync_collective" in joined:
            category = "tiny_sync_collective"
        elif "other_communication" in joined:
            category = "other_communication"
        elif "compute" in joined:
            category = "compute_active"
        else:
            category = "idle_or_unattributed"
        summary = summaries.setdefault(
            category,
            {
                "stage": category,
                "first_start_ms": start,
                "last_end_ms": end,
                "interval_count": 0,
                "union_duration_ms": 0.0,
            },
        )
        summary["first_start_ms"] = min(summary["first_start_ms"], start)
        summary["last_end_ms"] = max(summary["last_end_ms"], end)
        summary["interval_count"] += 1
        summary["union_duration_ms"] += end - start
    ordered = sorted(summaries.values(), key=lambda row: (row["first_start_ms"], row["stage"]))
    return {
        "iter": typical_iter,
        "source": f"/profiler_view/data/phase_intervals (iter={typical_iter}; atomic intervals summarized by label)",
        "semantics": (
            "trace label first-appearance summary; it is not a reconstructed compute-graph dependency order"
        ),
        "stages": ordered,
    }


def _reference_process_group_domains(
    workspace: Path, *, reference_iter: int, world_size: int
) -> dict[str, Any]:
    """Read one direct iteration's per-rank PG configs and return verified rank domains."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError("pandas with parquet support is required for process-group evidence") from exc

    directory = workspace / "derived_v5/pg_config_profiler" / f"iter={reference_iter}"
    files = sorted(directory.glob("rank=*.parquet"))
    if len(files) != world_size:
        raise ValueError(
            "Reference PG evidence must contain one parquet per rank; "
            f"iter={reference_iter}, expected={world_size}, found={len(files)}"
        )
    domains: dict[str, set[tuple[int, ...]]] = {}
    reporting_ranks: set[int] = set()
    for path in files:
        frame = pd.read_parquet(
            path,
            columns=["iter", "reporting_rank", "pg_description", "global_ranks"],
        )
        if frame.empty or set(int(value) for value in frame["iter"].unique()) != {reference_iter}:
            raise ValueError(f"Invalid reference PG iteration domain in {path}")
        file_ranks = {int(value) for value in frame["reporting_rank"].unique()}
        if len(file_ranks) != 1:
            raise ValueError(f"Reference PG file must contain one reporting_rank: {path}")
        reporting_ranks.update(file_ranks)
        for row in frame.itertuples(index=False):
            raw_ranks = row.global_ranks
            if isinstance(raw_ranks, str):
                raw_ranks = ast.literal_eval(raw_ranks)
            rank_domain = tuple(int(value) for value in list(raw_ranks))
            if len(rank_domain) != len(set(rank_domain)):
                raise ValueError(f"Duplicate rank in PG domain {row.pg_description}: {rank_domain}")
            domains.setdefault(str(row.pg_description), set()).add(rank_domain)
    if reporting_ranks != set(range(world_size)):
        raise ValueError(
            f"Reference PG evidence reporting-rank domain mismatch: {sorted(reporting_ranks)}"
        )

    def required(description: str) -> list[list[int]]:
        values = domains.get(description)
        if not values:
            raise ValueError(f"Missing required process-group identity {description!r}")
        return [list(domain) for domain in sorted(values)]

    dp_identities = sorted(
        description for description in domains if description.startswith("DATA_PARALLEL_GROUP")
    )
    dp_domains = sorted(
        {domain for description in dp_identities for domain in domains[description]}
    )
    return {
        "reference_iter": reference_iter,
        "source": (
            f"derived_v5/pg_config_profiler/iter={reference_iter}/rank=*.parquet "
            "(all reporting ranks; direct pg_description/global_ranks domains)"
        ),
        "ep_groups": required("EXPERT_MODEL_PARALLEL_GROUP"),
        "edp_pairs": required("EXPERT_DATA_PARALLEL_GROUP"),
        "dp_process_group_identities": dp_identities,
        "dp_rank_domains": [list(domain) for domain in dp_domains],
        "tp_rank_domains": required("TENSOR_MODEL_PARALLEL_GROUP"),
        "pp_rank_domains": required("PIPELINE_MODEL_PARALLEL_GROUP"),
        "cp_rank_domains": required("CONTEXT_PARALLEL_GROUP"),
    }


def build_iteration_knowledge_model(
    workspace: str | Path, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the reusable Task-5 knowledge model from config facts and direct payload atoms."""

    root = Path(workspace).resolve()
    if payload is None:
        payload_path = root / "derived_v5/run1555_html_payload_v5.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "profiler_view" not in payload:
        raise ValueError("Payload must contain profiler_view; root KPI/data fallbacks are forbidden")

    facts = extract_training_config_facts(root)
    view = payload["profiler_view"]
    scope = view["scope"]
    iterations = [int(value) for value in scope["iterations"]]
    if scope.get("mode") != "direct_profiler_only":
        raise ValueError(f"Expected direct_profiler_only view; got {scope.get('mode')!r}")
    if len(iterations) != 24 or len(set(iterations)) != 24:
        raise ValueError(f"Expected 24 explicit unique direct iterations; got {iterations}")
    if int(scope.get("iter_count", -1)) != len(iterations):
        raise ValueError("profiler_view.scope.iter_count disagrees with explicit iterations")
    data = view["data"]
    iter_rows = list(data["iter_summary"])
    _require_direct_domain(iter_rows, iterations, label="profiler_view.data.iter_summary")
    if any(row.get("analysis_source") != "direct_profiler" for row in iter_rows):
        raise ValueError("Every profiler_view iter_summary row must be direct_profiler")

    wall_median = _median_field(iter_rows, "wall_ms")
    assert wall_median is not None
    eligible = [
        row
        for row in iter_rows
        if row.get("wall_outlier") is False and row.get("network_bytes_outlier") is False
    ]
    if not eligible:
        raise ValueError("Typical-case selection has no non-wall/non-network-outlier candidate")
    typical_row = min(
        eligible,
        key=lambda row: (
            abs(_finite_number(row["wall_ms"], field="wall_ms") - wall_median),
            int(row["iter"]),
        ),
    )
    rows_by_iter = {int(row["iter"]): row for row in iter_rows}
    if len(rows_by_iter) != 24 or 24 not in rows_by_iter or 88 not in rows_by_iter:
        raise ValueError("Direct iter_summary must contain unique rows including iter24 and iter88")

    stage_share = list(data["stage_share"])
    all_communication = [row for row in stage_share if row.get("stage") == "all_communication"]
    _require_direct_domain(all_communication, iterations, label="all_communication stage_share")
    tiny_share = [row for row in stage_share if row.get("stage") == "tiny_sync_collective"]
    _require_direct_domain(tiny_share, iterations, label="tiny_sync_collective stage_share")

    communication_stages = [
        "deepep_backward_scaleup",
        "deepep_forward_scaleup",
        "dp_ag_hybrid",
        "dp_rs_hybrid",
        "edp_ag_scaleout",
        "edp_rs_scaleout",
        "gradient_finalize_sync",
        "other_communication",
        "tiny_sync_collective",
    ]
    headroom_rows = list(data["stage_headroom"])
    if len(headroom_rows) != len(iterations) * len(communication_stages):
        raise ValueError(
            "stage_headroom must contain the direct iteration × communication-stage grid; "
            f"got {len(headroom_rows)} rows"
        )
    stage_medians: list[dict[str, Any]] = []
    for stage in communication_stages:
        share_rows = [row for row in stage_share if row.get("stage") == stage]
        stage_headroom = [row for row in headroom_rows if row.get("stage") == stage]
        _require_direct_domain(share_rows, iterations, label=f"{stage} stage_share")
        _require_direct_domain(stage_headroom, iterations, label=f"{stage} stage_headroom")
        stage_medians.append(
            {
                "stage": stage,
                "sample_count": len(share_rows),
                "critical_median_ms": _median_field(share_rows, "critical_exposure_ms"),
                "critical_median_pct": _median_field(share_rows, "critical_exposure_pct"),
                "service_median_ms": _median_field(share_rows, "service_ms", allow_null=True),
                "service_median_pct": _median_field(share_rows, "service_pct", allow_null=True),
                "exclusive_median_ms": _median_field(share_rows, "exclusive_ms"),
                "exclusive_median_pct": _median_field(share_rows, "exclusive_wall_pct"),
                "observed_headroom_median_ms": _median_field(
                    stage_headroom, "wall_exposed_observed_ms"
                ),
                "observed_headroom_median_pct": _median_field(
                    stage_headroom, "wall_exposed_observed_pct"
                ),
                "source": (
                    "/profiler_view/data/stage_share + /profiler_view/data/stage_headroom "
                    f"(stage={stage}; median over explicit 24 direct rows)"
                ),
            }
        )
    tiny_medians = next(item for item in stage_medians if item["stage"] == "tiny_sync_collective")

    tiny_calls = [
        row
        for row in data["collective_calls"]
        if row.get("collective") == "allreduce"
        and row.get("size_bytes") == 4
        and row.get("occurrence_index") == 0
        and row.get("pg_description") == "default_pg"
    ]
    _require_direct_domain(tiny_calls, iterations, label="default_pg 4-byte allreduce occurrence0")

    direct_summary = data["spatial_temporal"]["edp"]["direct_summary"]
    try:
        edp_pairs = sorted({tuple(ast.literal_eval(str(row["pair"]))) for row in direct_summary})
    except (SyntaxError, ValueError) as exc:
        raise ValueError("EDP direct pair domain contains an invalid pair literal") from exc
    if any(len(pair) != 2 or not all(isinstance(rank, int) for rank in pair) for pair in edp_pairs):
        raise ValueError(f"Invalid EDP pair domain: {edp_pairs}")
    edp_pair_summary = sorted(
        [
            {
                "pair": str(row["pair"]),
                "collective": str(row["collective"]),
                "n": int(row["n"]),
                "service_p50_ms": _finite_number(
                    row["service_p50_ms"], field="edp.service_p50_ms"
                ),
                "critical_p50_ms": _finite_number(
                    row["critical_p50_ms"], field="edp.critical_p50_ms"
                ),
                "arrival_p50_ms": _finite_number(
                    row["arrival_p50_ms"], field="edp.arrival_p50_ms"
                ),
            }
            for row in direct_summary
        ],
        key=lambda row: (tuple(ast.literal_eval(row["pair"])), row["collective"]),
    )
    if len(edp_pair_summary) != 16 or any(row["n"] not in (24, 48) for row in edp_pair_summary):
        raise ValueError("EDP direct summary must contain 16 pair/collective rows over the direct scope")

    nic = data["spatial_temporal"]["nic"]
    nic_indexes: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for name in ("bytes_entity_summary", "rate_entity_summary", "active_rate_entity_summary"):
        rows = list(nic[name])
        index = {(str(row["host"]), str(row["device"])): row for row in rows}
        if len(rows) != 8 or len(index) != 8:
            raise ValueError(f"NIC {name} must contain eight unique host/device direct summaries")
        if any(int(row["n_iter"]) != len(iterations) for row in rows):
            raise ValueError(f"NIC {name} must be based on all explicit direct iterations")
        nic_indexes[name] = index
    nic_keys = set(nic_indexes["bytes_entity_summary"])
    if any(set(index) != nic_keys for index in nic_indexes.values()):
        raise ValueError("NIC bytes/full-window/active-rate summary device domains disagree")
    nic_device_summary = []
    for host, device in sorted(nic_keys):
        nic_device_summary.append(
            {
                "host": host,
                "device": device,
                "sample_count": len(iterations),
                "bytes_median": _finite_number(
                    nic_indexes["bytes_entity_summary"][(host, device)]["value_median"],
                    field="nic.bytes.value_median",
                ),
                "full_window_rate_median_gbps": _finite_number(
                    nic_indexes["rate_entity_summary"][(host, device)]["value_median"],
                    field="nic.rate.value_median",
                ),
                "active_rate_median_gbps": _finite_number(
                    nic_indexes["active_rate_entity_summary"][(host, device)]["value_median"],
                    field="nic.active_rate.value_median",
                ),
            }
        )

    collective_calls = list(data["collective_calls"])
    algorithms = sorted({str(row["algo"]) for row in collective_calls})
    protocols = sorted({str(row["protocol"]) for row in collective_calls})

    world_size = facts["run.world_size"].value
    host_split = facts["run.host_split_rank"].value
    if world_size != facts["data_parallel_size"].value:
        raise ValueError("TOML world_size and RANK0 data_parallel_size disagree")
    if host_split <= 0 or world_size != host_split * 2:
        raise ValueError("Task-5 topology requires exactly two equal-size hosts")
    ep_size = facts["expert_model_parallel_size"].value
    expected_edp_pairs = [(rank, rank + host_split) for rank in range(host_split)]
    if edp_pairs != expected_edp_pairs:
        raise ValueError(
            f"EDP direct pair domain disagrees with host split: expected={expected_edp_pairs}, "
            f"actual={edp_pairs}"
        )
    if ep_size != host_split:
        raise ValueError("EP size must match the host-local rank count for this run")
    pg_domains = _reference_process_group_domains(
        root, reference_iter=min(iterations), world_size=world_size
    )
    expected_ep_groups = [
        list(range(start, start + ep_size)) for start in range(0, world_size, ep_size)
    ]
    if pg_domains["ep_groups"] != expected_ep_groups:
        raise ValueError(
            "Direct EXPERT_MODEL_PARALLEL_GROUP domains disagree with config/host split: "
            f"expected={expected_ep_groups}, actual={pg_domains['ep_groups']}"
        )
    if pg_domains["edp_pairs"] != [list(pair) for pair in expected_edp_pairs]:
        raise ValueError(
            "Direct EXPERT_DATA_PARALLEL_GROUP domains disagree with collective pair domain"
        )
    if pg_domains["dp_rank_domains"] != [list(range(world_size))]:
        raise ValueError("Direct DATA_PARALLEL_GROUP identities must share the full world rank domain")

    mbs = facts["micro_batch_size"].value
    gbs = facts["global_batch_size"].value
    sequence_length = facts["seq_length"].value
    denominator = mbs * facts["data_parallel_size"].value
    if gbs % denominator:
        raise ValueError(f"GBS must be divisible by MBS×DP; got {gbs} / {denominator}")

    case24 = rows_by_iter[24]
    case88 = rows_by_iter[88]
    nic_difference_pct = (
        _finite_number(case88["nic_bidir_bytes"], field="iter88.nic_bidir_bytes")
        / _finite_number(case24["nic_bidir_bytes"], field="iter24.nic_bidir_bytes")
        - 1.0
    ) * 100.0

    def case_from_row(row: dict[str, Any]) -> dict[str, Any]:
        iter_id = int(row["iter"])
        return {
            "iter": iter_id,
            "wall_ms": _finite_number(row["wall_ms"], field=f"iter{iter_id}.wall_ms"),
            "profiled_global_wall_ms": _finite_number(
                row["profiled_global_wall_ms"], field=f"iter{iter_id}.profiled_global_wall_ms"
            ),
            "profiled_gap_ms": _finite_number(
                row["cycle_minus_profiled_wall_ms"],
                field=f"iter{iter_id}.cycle_minus_profiled_wall_ms",
            ),
            "nic_bidir_bytes": int(row["nic_bidir_bytes"]),
            "nic_cluster_full_window_gbps": _finite_number(
                row["nic_cluster_full_window_gbps"],
                field=f"iter{iter_id}.nic_cluster_full_window_gbps",
            ),
            "wall_outlier": bool(row["wall_outlier"]),
            "network_bytes_outlier": bool(row["network_bytes_outlier"]),
            "source": f"/profiler_view/data/iter_summary (iter={iter_id})",
        }

    comm_speedup = _finite_number(
        view["kpis"]["communication_only_speedup_median_pct"],
        field="profiler_view.kpis.communication_only_speedup_median_pct",
    )
    topology = {
        "host_count": 2,
        "gpus_per_host": host_split,
        "world_size": world_size,
        "hosts": [
            {"host": "A", "ranks": list(range(0, host_split))},
            {"host": "B", "ranks": list(range(host_split, world_size))},
        ],
    }
    parallelism = {
        "DP": {
            "size": facts["data_parallel_size"].value,
            "cross_rank_communication": facts["data_parallel_size"].value > 1,
            "role": "dense gradient/parameter synchronization domain",
        },
        "EP": {
            "size": ep_size,
            "cross_rank_communication": ep_size > 1,
            "role": "host-local expert token routing domain",
        },
        "EDP": {
            "size": world_size // ep_size,
            "cross_rank_communication": world_size // ep_size > 1,
            "role": "same local expert position across hosts",
        },
        "TP": {
            "size": facts["tensor_model_parallel_size"].value,
            "cross_rank_communication": facts["tensor_model_parallel_size"].value > 1,
            "role": "tensor-parallel operator shards",
        },
        "PP": {
            "size": facts["pipeline_model_parallel_size"].value,
            "cross_rank_communication": facts["pipeline_model_parallel_size"].value > 1,
            "role": "pipeline stages",
        },
        "CP": {
            "size": facts["context_parallel_size"].value,
            "cross_rank_communication": facts["context_parallel_size"].value > 1,
            "role": "sequence/context shards",
        },
    }

    return {
        "schema_version": "run1555-iteration-knowledge-v1",
        "scope": {
            "direct_iterations": iterations,
            "direct_count": len(iterations),
            "inferred_excluded_count": int(scope["excluded_inference_count"]),
            "numerical_source": "/profiler_view only; 24 explicit direct atoms reaggregated",
            "evidence_boundary": (
                "75 validated-inference iterations are mentioned only as excluded evidence scope; "
                "they do not contribute to numerical results"
            ),
            "model_scope": (
                f"actual num_layers={facts['num_layers'].value}; 1188B is a directory/task label, "
                "not permission to extrapolate to an unobserved full-layer model"
            ),
        },
        "config_facts": {key: fact.to_dict() for key, fact in sorted(facts.items())},
        "topology": topology,
        "groups": {
            "dp_group": list(range(world_size)),
            "dp_process_group_identities": pg_domains["dp_process_group_identities"],
            "dp_rank_domains": pg_domains["dp_rank_domains"],
            "ep_groups": pg_domains["ep_groups"],
            "ep_group_source": pg_domains["source"],
            "edp_pairs": [list(pair) for pair in edp_pairs],
            "edp_pair_source": (
                "/profiler_view/data/spatial_temporal/edp/direct_summary + "
                f"{pg_domains['source']} "
                "(unique direct pair domain, cross-checked against PG config and TOML host split)"
            ),
        },
        "parallelism": parallelism,
        "batch": {
            "micro_batch_size": mbs,
            "global_batch_size": gbs,
            "sequence_length": sequence_length,
            "tokens_per_rank_microbatch": mbs * sequence_length,
            "global_tokens_per_iteration": gbs * sequence_length,
            "num_microbatches": gbs // denominator,
            "num_microbatches_formula": "GBS / (MBS × DP)",
        },
        "data_objects": [
            {
                "name": "Token",
                "zh": "Token（逻辑路由单元；不等同线上原始 token ID）",
                "moves_at": "batch输入与MoE路由身份；不声称原始token ID跨fabric传输",
                "owner_change": "逻辑归属保持；通信对象是与token关联的activation/payload",
            },
            {
                "name": "Activation",
                "zh": "Activation（激活）",
                "moves_at": "forward保存/重算、DeepEP dispatch/combine与backward消费",
                "owner_change": f"token-associated activation/payload与routing metadata路由到EP{ep_size} expert rank",
            },
            {
                "name": "Gradient",
                "zh": "Gradient（梯度）",
                "moves_at": "backward/finalize/ReduceScatter",
                "owner_change": "optimizer owner shard",
            },
            {
                "name": "Optimizer shard",
                "zh": "Optimizer shard（优化器分片）",
                "moves_at": "Adam本地更新",
                "owner_change": "分片owner保有FP32 main params/moments",
            },
            {
                "name": "Parameter",
                "zh": "Parameter（参数）",
                "moves_at": "更新后的AllGather",
                "owner_change": "恢复下一轮所需副本/视图",
            },
        ],
        "iteration_steps": [
            {"order": 1, "name": "batch_prepare", "object": "Token", "placement": "iteration入口", "evidence": "教学机制；当前stage未逐项标注"},
            {"order": 2, "name": "forward_dense_attention", "object": "Activation", "placement": "Forward内部", "evidence": "配置确认 + aggregate compute_active"},
            {"order": 3, "name": "forward_moe_with_deepep", "object": "Activation/routing metadata", "placement": "Forward内部嵌套DeepEP dispatch/expert/ combine", "evidence": "direct DeepEP聚合stage；内部算子非逐项stage"},
            {"order": 4, "name": "mtp_and_loss", "object": "Activation/loss", "placement": "Forward输出端", "evidence": "配置确认；当前stage未逐项标注"},
            {"order": 5, "name": "backward_recompute", "object": "Activation/Gradient", "placement": "Backward内部", "evidence": "recompute配置 + aggregate compute_active"},
            {"order": 6, "name": "backward_moe_with_deepep", "object": "Activation Gradient/routing metadata", "placement": "Backward内部嵌套DeepEP backward", "evidence": "direct DeepEP聚合stage；张量语义来自通用机制"},
            {"order": 7, "name": "distributed_optimizer_region", "object": "Gradient/Optimizer shard/Parameter", "placement": "每个参数域内RS/Adam/AG，可跨bucket交错", "evidence": "配置 + direct collective/Adam事件；kernel payload未直接标类型"},
            {"order": 8, "name": "tiny_control_occurrences", "object": "control scalar/small tensor", "placement": "可在iteration多个位置出现，不是统一尾声", "evidence": "direct collective calls；多个occurrence", "may_repeat": True},
            {"order": 9, "name": "step_external_gap", "object": "not attributed", "placement": "profiled step边界之外", "evidence": "cycle minus profiled wall only"},
        ],
        "primitives": [
            {
                "name": "DeepEP dispatch",
                "input": "token-associated activation/payload + routing metadata",
                "purpose": f"route token-associated payload to selected experts in EP{ep_size}; not raw token IDs",
                "fabric": "host-local scale-up (MTLink is observed, but not all MTLink bytes are DeepEP)",
            },
            {
                "name": "DeepEP combine",
                "input": "expert output activation；backward时为对应activation gradients",
                "purpose": "return expert outputs/gradients toward the original token computation path",
                "fabric": "host-local scale-up",
            },
            {
                "name": "ReduceScatter",
                "input": "Gradient（框架机制语义；collective kernel未直接标payload类型）",
                "purpose": "within each parameter domain, reduce contributions and leave one optimizer shard per owner",
                "fabric": (
                    f"DP{facts['data_parallel_size'].value} hybrid or EDP pair scale-out "
                    "according to parameter domain"
                ),
            },
            {
                "name": "AllGather",
                "input": "Parameter（框架机制语义；collective kernel未直接标payload类型）",
                "purpose": "within each parameter domain, collect updated shards needed by later compute",
                "fabric": (
                    f"DP{facts['data_parallel_size'].value} hybrid or EDP pair scale-out "
                    "according to parameter domain"
                ),
            },
            {
                "name": "AllReduce",
                "input": "control scalar/small tensor",
                "purpose": "produce the same reduction result on every participating rank",
                "fabric": "collective process group",
            },
            {
                "name": "barrier",
                "input": "control token",
                "purpose": "wait until all group participants arrive",
                "fabric": "control synchronization",
            },
        ],
        "metrics": {
            "wall_median_ms": {
                "value": wall_median,
                "source": (
                    "/profiler_view/data/iter_summary[*].wall_ms "
                    "(median over the explicit 24 direct iterations)"
                ),
            },
            "all_communication": {
                "critical_median_ms": _median_field(all_communication, "critical_exposure_ms"),
                "critical_median_pct": _median_field(all_communication, "critical_exposure_pct"),
                "source": (
                    "/profiler_view/data/stage_share (stage=all_communication; "
                    "median over 24 direct rows)"
                ),
            },
            "comm_only_speedup_median_pct": {
                "value": comm_speedup,
                "source": "/profiler_view/kpis/communication_only_speedup_median_pct",
                "semantics": (
                    f"direct-{len(iterations)} interval-bound counterfactual; "
                    "not a dependency-DAG prediction"
                ),
            },
            "tiny_sync": {**tiny_medians},
            "default_4b_allreduce_occ0": {
                "size_bytes": int(tiny_calls[0]["size_bytes"]),
                "occurrence_index": int(tiny_calls[0]["occurrence_index"]),
                "pg_description": str(tiny_calls[0]["pg_description"]),
                "call_count": len(tiny_calls),
                "arrival_skew_median_ms": _median_field(tiny_calls, "arrival_skew_ms"),
                "service_median_ms": _median_field(tiny_calls, "service_ms"),
                "finish_spread_median_ms": _median_field(tiny_calls, "finish_spread_ms"),
                "critical_median_ms": _median_field(tiny_calls, "critical_exposure_ms"),
                "source": (
                    "/profiler_view/data/collective_calls filtered by collective=allreduce, "
                    "size_bytes=4, occurrence_index=0, pg_description=default_pg; "
                    "median over 24 direct calls"
                ),
            },
            "stage_medians": stage_medians,
            "edp_pair_summary": edp_pair_summary,
            "nic_device_summary": nic_device_summary,
            "collective_algorithms": {
                "algo": algorithms,
                "protocol": protocols,
                "source": (
                    "/profiler_view/data/collective_calls "
                    f"(unique labels over {len(collective_calls)} direct calls)"
                ),
            },
            "wall_outlier_iters": [
                int(row["iter"]) for row in iter_rows if row.get("wall_outlier") is True
            ],
            "network_outlier_iters": [
                int(row["iter"])
                for row in iter_rows
                if row.get("network_bytes_outlier") is True
            ],
        },
        "cases": {
            "typical": {
                **case_from_row(typical_row),
                "selection_rule": (
                    "among direct rows with wall_outlier=false and network_bytes_outlier=false, "
                    f"minimize absolute distance to the direct-{len(iterations)} wall median; "
                    "tie-break by smaller iter"
                ),
                "selection_is_population_claim": False,
                "phase_timeline": _summarize_phase_timeline(
                    list(data["phase_intervals"]), int(typical_row["iter"])
                ),
            },
            "iter24": case_from_row(case24),
            "iter88": case_from_row(case88),
            "iter24_vs_iter88": {
                "nic_difference_pct": nic_difference_pct,
                "source": (
                    "/profiler_view/data/iter_summary iter24/iter88 nic_bidir_bytes; "
                    "(iter88 / iter24 - 1) × 100"
                ),
                "fact": (
                    "both cases have large cycle-minus-profiled gaps: "
                    f"{case24['cycle_minus_profiled_wall_ms']:.6f} ms and "
                    f"{case88['cycle_minus_profiled_wall_ms']:.6f} ms"
                ),
                "interpretation": (
                    "wall inflation lies outside the profiled step and NIC bytes do not change proportionally"
                ),
                "hypothesis": (
                    "scheduling, input pipeline, or host synchronization are candidates; none is observed"
                ),
            },
        },
        "claim_audit": {
            "allowed": [
                "【通用知识】state mechanism definitions without claiming run-specific observation",
                "【本次观察】state only values/configuration supported by cited direct atoms/source lines",
            ],
            "narrow": [
                "【解释】separate endpoint-sum bytes, full-window rate, active rate, arrival skew and service",
                "【解释】describe EDP odd/even pair bimodality as stable observation, not route causation",
            ],
            "discussion": [
                "【待验证假设】test scheduling/input/host-sync candidates with new markers",
                "【待验证假设】test rail/ring/channel mapping or local-position remapping",
            ],
            "prohibited": [
                "do not turn a counterfactual interval bound into an achievable speedup guarantee",
                (
                    "do not infer the unobserved full 1188B layer stack from "
                    f"num_layers={facts['num_layers'].value}"
                ),
                "do not claim RING/SIMPLE reveals ring order, rail, or channel",
                (
                    "do not call the measured cycle-minus-profiled gaps communication time: "
                    f"iter{case24['iter']}={case24['cycle_minus_profiled_wall_ms']:.6f} ms, "
                    f"iter{case88['iter']}={case88['cycle_minus_profiled_wall_ms']:.6f} ms"
                ),
            ],
        },
        "quiz": [
            {
                "question": (
                    f"TP={facts['tensor_model_parallel_size'].value}时为什么没有跨rank TP AllReduce？"
                ),
                "answer": (
                    f"只有{facts['tensor_model_parallel_size'].value}个TP成员，"
                    "不存在跨rank张量分片需要归并。"
                ),
            },
            {"question": "bond5 full-window rate较低能否证明端口慢？", "answer": "不能；先比较bytes、active-rate与覆盖窗口，低full-window rate可能只是分子较小或分母较长。"},
            {
                "question": f"iter{case24['iter']}平均NIC Gbps为何下降？",
                "answer": (
                    "endpoint-sum NIC bytes近似不变，而cycle wall含 "
                    f"{case24['cycle_minus_profiled_wall_ms']:.3f} ms 未归因gap，"
                    "full-window rate的分母变大。"
                ),
            },
            {"question": "EDP奇偶pair为何比全局p50更有诊断价值？", "answer": "全局p50会把稳定的pair双峰折叠；pair域保留local-position结构，但不证明rail/ring/channel原因。"},
            {
                "question": (
                    f"tiny {tiny_calls[0]['size_bytes']}B kernel长是否等于"
                    f"{tiny_calls[0]['size_bytes']}B传输慢？"
                ),
                "answer": "不等于；应拆arrival skew与post-last-arrival service，前者主要是早到rank等待。",
            },
            {"question": "overlap关闭能证明开启后无效吗？", "answer": "不能；本run只确认关闭状态，开启效果需要对照实验。"},
            {"question": "为何tiny critical median不能直接当作wall headroom？", "answer": "阶段区间会与计算/其他通信重叠；只有不重叠且满足依赖的部分才可能缩短wall，当前反事实仍只是interval bound。"},
        ],
    }

_GUIDE_ROOT_KEYS = {
    "schema_version",
    "scope",
    "topology",
    "groups",
    "parallelism",
    "batch",
    "config_facts",
    "objects",
    "steps",
    "primitives",
    "concepts",
    "cases",
    "metrics",
    "boundaries",
    "claim_audit",
    "quiz",
}
_GUIDE_OBJECT_NAMES = (
    "Token",
    "Activation",
    "Gradient",
    "Optimizer shard",
    "Parameter",
)
_GUIDE_STEP_KEYS = {
    "key",
    "order",
    "label",
    "object_names",
    "compute_object",
    "communication_object",
    "ranks",
    "fabric",
    "primitive",
    "evidence",
    "evidence_level",
    "placement",
    "description",
    "is_communication",
    "may_repeat",
}
_GUIDE_STEP_ORDER = (
    "batch_prepare",
    "forward_dense_attention",
    "forward_moe_with_deepep",
    "mtp_and_loss",
    "backward_recompute",
    "backward_moe_with_deepep",
    "distributed_optimizer_region",
    "tiny_control_occurrences",
    "step_external_gap",
)
_GUIDE_PRIMITIVE_NAMES = (
    "DeepEP dispatch",
    "DeepEP combine",
    "ReduceScatter",
    "AllGather",
    "AllReduce",
)
_GUIDE_CONCEPT_NAMES = (
    "ReduceScatter",
    "AllGather",
    "AllReduce",
    "DeepEP dispatch/combine",
    "arrival skew",
    "critical path",
)
_GUIDE_CONFIG_KEYS = (
    "num_layers",
    "data_parallel_size",
    "expert_model_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "num_experts",
    "moe_enable_deepep",
    "moe_token_dispatcher_type",
    "moe_grouped_gemm",
    "moe_router_dtype",
    "moe_router_pre_softmax",
    "moe_router_score_function",
    "moe_router_topk",
    "moe_router_load_balancing_type",
    "use_distributed_optimizer",
    "optimizer",
    "main_params_dtype",
    "main_grads_dtype",
    "exp_avg_dtype",
    "exp_avg_sq_dtype",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "overlap_grad_reduce",
    "overlap_param_gather",
    "overlap_moe_expert_parallel_comm",
    "moe_shared_expert_overlap",
)
_GUIDE_METRIC_KEYS = (
    "wall_median_ms",
    "all_communication",
    "comm_only_speedup_median_pct",
    "tiny_sync",
    "default_4b_allreduce_occ0",
    "stage_medians",
    "wall_outlier_iters",
    "network_outlier_iters",
    "edp_pair_summary",
    "nic_device_summary",
    "collective_algorithms",
)
_GUIDE_STAGE_NAMES = {
    "deepep_backward_scaleup",
    "deepep_forward_scaleup",
    "dp_ag_hybrid",
    "dp_rs_hybrid",
    "edp_ag_scaleout",
    "edp_rs_scaleout",
    "gradient_finalize_sync",
    "other_communication",
    "tiny_sync_collective",
}


def _guide_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping; got {type(value).__name__}")
    return value


def _guide_exact_keys(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{path} keys must match exactly; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _guide_int(value: Any, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer; got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}; got {value}")
    return value


def _validate_finite_tree(value: Any, *, path: str = "guide") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must be finite; got {value!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported non-JSON value {value!r}")


def _validate_guide_source_model(model: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "scope",
        "config_facts",
        "topology",
        "groups",
        "parallelism",
        "batch",
        "data_objects",
        "iteration_steps",
        "primitives",
        "metrics",
        "cases",
        "claim_audit",
        "quiz",
    }
    missing = required - set(model)
    if missing:
        raise ValueError(f"knowledge model is missing Guide inputs: {sorted(missing)}")
    if model.get("schema_version") != "run1555-iteration-knowledge-v1":
        raise ValueError("knowledge model schema_version is not Task-5 v1")

    scope = _guide_mapping(model["scope"], path="model.scope")
    _guide_exact_keys(
        scope,
        {
            "direct_iterations",
            "direct_count",
            "inferred_excluded_count",
            "numerical_source",
            "evidence_boundary",
            "model_scope",
        },
        path="model.scope",
    )
    direct = scope["direct_iterations"]
    if not isinstance(direct, list):
        raise TypeError(
            "model.scope.direct_iterations must be an explicit list; range/modulo domains are forbidden"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in direct):
        raise TypeError("model.scope.direct_iterations must contain integer IDs")
    if (
        _guide_int(scope["direct_count"], path="model.scope.direct_count") != 24
        or len(direct) != 24
        or len(set(direct)) != 24
    ):
        raise ValueError("model.scope must contain exactly 24 explicit unique direct IDs")
    if _guide_int(
        scope["inferred_excluded_count"], path="model.scope.inferred_excluded_count"
    ) != 75:
        raise ValueError("model.scope.inferred_excluded_count must be exactly 75")
    if "1188B" not in str(scope["model_scope"]) or "num_layers=" not in str(
        scope["model_scope"]
    ):
        raise ValueError("model.scope must preserve the num_layers/1188B boundary")

    topology = _guide_mapping(model["topology"], path="model.topology")
    _guide_exact_keys(
        topology,
        {"host_count", "gpus_per_host", "world_size", "hosts"},
        path="model.topology",
    )
    host_count = _guide_int(topology["host_count"], path="model.topology.host_count", minimum=1)
    gpus_per_host = _guide_int(
        topology["gpus_per_host"], path="model.topology.gpus_per_host", minimum=1
    )
    world_size = _guide_int(topology["world_size"], path="model.topology.world_size", minimum=1)
    hosts = topology["hosts"]
    if not isinstance(hosts, list) or len(hosts) != host_count or host_count != 2:
        raise ValueError("model.topology must contain exactly two hosts")
    host_ranks: list[int] = []
    for index, raw_host in enumerate(hosts):
        host = _guide_mapping(raw_host, path=f"model.topology.hosts[{index}]")
        _guide_exact_keys(host, {"host", "ranks"}, path=f"model.topology.hosts[{index}]")
        ranks = host["ranks"]
        if not isinstance(ranks, list) or len(ranks) != gpus_per_host:
            raise ValueError(f"model.topology.hosts[{index}].ranks has the wrong size")
        if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
            raise TypeError(f"model.topology.hosts[{index}].ranks must be integers")
        host_ranks.extend(ranks)
    if world_size != host_count * gpus_per_host or host_ranks != list(range(world_size)):
        raise ValueError("model.topology rank domain is missing, duplicate, reordered, or wrong")

    groups = _guide_mapping(model["groups"], path="model.groups")
    _guide_exact_keys(
        groups,
        {
            "dp_group",
            "dp_process_group_identities",
            "dp_rank_domains",
            "ep_groups",
            "ep_group_source",
            "edp_pairs",
            "edp_pair_source",
        },
        path="model.groups",
    )
    if groups["dp_group"] != host_ranks or groups["dp_rank_domains"] != [host_ranks]:
        raise ValueError("model.groups DP domain must equal the topology rank domain")
    identities = groups["dp_process_group_identities"]
    if not isinstance(identities, list) or not identities or len(set(identities)) != len(identities):
        raise ValueError("model.groups DP identities must be unique and non-empty")
    if groups["ep_groups"] != [host["ranks"] for host in hosts]:
        raise ValueError("model.groups EP domains must equal the PG-verified host domains")
    ep_source = str(groups["ep_group_source"])
    if "pg_config_profiler" not in ep_source or ".parquet" not in ep_source:
        raise ValueError("model.groups EP source must identify PG parquet evidence")
    edp_pairs = groups["edp_pairs"]
    if not isinstance(edp_pairs, list) or len(edp_pairs) != 8:
        raise ValueError("model.groups.edp_pairs must contain exactly eight pairs")
    flattened_edp: list[int] = []
    for index, pair in enumerate(edp_pairs):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in pair)
            or len(set(pair)) != 2
        ):
            raise ValueError(f"model.groups.edp_pairs[{index}] is not a unique two-rank domain")
        flattened_edp.extend(pair)
    if len(flattened_edp) != world_size or set(flattened_edp) != set(host_ranks):
        raise ValueError("model.groups.edp_pairs must cover every topology rank exactly once")

    parallelism = _guide_mapping(model["parallelism"], path="model.parallelism")
    _guide_exact_keys(
        parallelism, {"DP", "EP", "EDP", "TP", "PP", "CP"}, path="model.parallelism"
    )
    for name, raw_item in parallelism.items():
        item = _guide_mapping(raw_item, path=f"model.parallelism.{name}")
        _guide_exact_keys(
            item,
            {"size", "cross_rank_communication", "role"},
            path=f"model.parallelism.{name}",
        )
        size = _guide_int(item["size"], path=f"model.parallelism.{name}.size", minimum=1)
        if item["cross_rank_communication"] is not (size > 1):
            raise ValueError(f"model.parallelism.{name} communication flag disagrees with size")
    if parallelism["DP"]["size"] != world_size:
        raise ValueError("model.parallelism.DP.size disagrees with topology")
    if parallelism["EP"]["size"] != len(groups["ep_groups"][0]):
        raise ValueError("model.parallelism.EP.size disagrees with EP domain")
    if parallelism["EDP"]["size"] != len(edp_pairs[0]):
        raise ValueError("model.parallelism.EDP.size disagrees with EDP domain")

    batch = _guide_mapping(model["batch"], path="model.batch")
    _guide_exact_keys(
        batch,
        {
            "micro_batch_size",
            "global_batch_size",
            "sequence_length",
            "tokens_per_rank_microbatch",
            "global_tokens_per_iteration",
            "num_microbatches",
            "num_microbatches_formula",
        },
        path="model.batch",
    )
    for key in (
        "micro_batch_size",
        "global_batch_size",
        "sequence_length",
        "tokens_per_rank_microbatch",
        "global_tokens_per_iteration",
        "num_microbatches",
    ):
        _guide_int(batch[key], path=f"model.batch.{key}", minimum=1)
    if batch["tokens_per_rank_microbatch"] != batch["micro_batch_size"] * batch["sequence_length"]:
        raise ValueError("model.batch per-rank token arithmetic is inconsistent")
    if batch["global_tokens_per_iteration"] != batch["global_batch_size"] * batch["sequence_length"]:
        raise ValueError("model.batch global token arithmetic is inconsistent")

    facts = _guide_mapping(model["config_facts"], path="model.config_facts")
    missing_facts = set(_GUIDE_CONFIG_KEYS) - set(facts)
    if missing_facts:
        raise ValueError(f"model.config_facts lacks Guide facts: {sorted(missing_facts)}")
    for key in _GUIDE_CONFIG_KEYS:
        fact = _guide_mapping(facts[key], path=f"model.config_facts.{key}")
        if not {"value", "source_path", "source_line"} <= set(fact):
            raise ValueError(f"model.config_facts.{key} lacks value/source file:line")
        if not str(fact["source_path"]):
            raise ValueError(f"model.config_facts.{key}.source_path must be non-empty")
        _guide_int(fact["source_line"], path=f"model.config_facts.{key}.source_line", minimum=1)

    objects = model["data_objects"]
    if not isinstance(objects, list) or len(objects) != 5:
        raise ValueError("model.data_objects must contain exactly five rows")
    if [item.get("name") if isinstance(item, Mapping) else None for item in objects] != list(
        _GUIDE_OBJECT_NAMES
    ):
        raise ValueError("model.data_objects has a missing, duplicate, or wrong object domain")
    for index, raw_object in enumerate(objects):
        item = _guide_mapping(raw_object, path=f"model.data_objects[{index}]")
        _guide_exact_keys(
            item,
            {"name", "zh", "moves_at", "owner_change"},
            path=f"model.data_objects[{index}]",
        )

    steps = model["iteration_steps"]
    if not isinstance(steps, list) or len(steps) != 9:
        raise ValueError("model.iteration_steps must contain exactly nine rows")
    if [item.get("name") if isinstance(item, Mapping) else None for item in steps] != list(
        _GUIDE_STEP_ORDER
    ):
        raise ValueError("model.iteration_steps has a missing, duplicate, reordered, or wrong domain")
    if [item.get("order") for item in steps if isinstance(item, Mapping)] != list(range(1, 10)):
        raise ValueError("model.iteration_steps orders must be exactly 1..9")
    if steps[7].get("may_repeat") is not True:
        raise ValueError("model tiny/control step must explicitly allow repeats")

    primitives = model["primitives"]
    if not isinstance(primitives, list) or [
        item.get("name") if isinstance(item, Mapping) else None for item in primitives
    ] != [*_GUIDE_PRIMITIVE_NAMES, "barrier"]:
        raise ValueError("model.primitives has a missing, duplicate, reordered, or wrong domain")

    metrics = _guide_mapping(model["metrics"], path="model.metrics")
    missing_metrics = set(_GUIDE_METRIC_KEYS) - set(metrics)
    if missing_metrics:
        raise ValueError(f"model.metrics lacks Guide metrics: {sorted(missing_metrics)}")
    stages = metrics["stage_medians"]
    if not isinstance(stages, list) or len(stages) != 9:
        raise ValueError("model.metrics.stage_medians must contain exactly nine rows")
    stage_names = [row.get("stage") if isinstance(row, Mapping) else None for row in stages]
    if len(set(stage_names)) != 9 or set(stage_names) != _GUIDE_STAGE_NAMES:
        raise ValueError("model.metrics.stage_medians has a missing, duplicate, or wrong domain")
    if any(row.get("sample_count") != 24 for row in stages if isinstance(row, Mapping)):
        raise ValueError("model.metrics.stage_medians must use the direct-24 domain")
    if not isinstance(metrics["edp_pair_summary"], list) or len(metrics["edp_pair_summary"]) != 16:
        raise ValueError("model.metrics.edp_pair_summary must contain 8 pairs x 2 collectives")
    edp_metric_domain = {
        (str(row.get("pair")), str(row.get("collective")))
        for row in metrics["edp_pair_summary"]
        if isinstance(row, Mapping)
    }
    expected_edp_metric_domain = {
        (str(pair), collective)
        for pair in edp_pairs
        for collective in (
            "allgather_into_tensor_coalesced",
            "reduce_scatter_tensor_coalesced",
        )
    }
    if edp_metric_domain != expected_edp_metric_domain:
        raise ValueError("model.metrics.edp_pair_summary has the wrong pair/collective domain")
    nic_rows = metrics["nic_device_summary"]
    if not isinstance(nic_rows, list) or len(nic_rows) != 8:
        raise ValueError("model.metrics.nic_device_summary must contain eight rows")
    if len({(row.get("host"), row.get("device")) for row in nic_rows}) != 8:
        raise ValueError("model.metrics.nic_device_summary device domain must be unique")
    for key in ("wall_outlier_iters", "network_outlier_iters"):
        values = metrics[key]
        if (
            not isinstance(values, list)
            or len(values) != len(set(values))
            or not set(values) <= set(direct)
        ):
            raise ValueError(f"model.metrics.{key} must be a unique direct-domain subset")

    cases = _guide_mapping(model["cases"], path="model.cases")
    if set(cases) != {"typical", "iter24", "iter88", "iter24_vs_iter88"}:
        raise ValueError("model.cases has a missing, duplicate, or wrong domain")
    for key in ("typical", "iter24", "iter88"):
        case = _guide_mapping(cases[key], path=f"model.cases.{key}")
        if case.get("iter") not in direct:
            raise ValueError(f"model.cases.{key}.iter is outside the direct domain")
    if cases["iter24"].get("iter") != 24 or cases["iter88"].get("iter") != 88:
        raise ValueError("model.cases iter24/iter88 keys disagree with IDs")
    if not isinstance(model["quiz"], list) or len(model["quiz"]) != 7:
        raise ValueError("model.quiz must contain exactly seven rows")
    _validate_finite_tree(model, path="model")


def _guide_case(key: str, case: Mapping[str, Any]) -> dict[str, Any]:
    iteration = int(case["iter"])
    if key == "typical":
        label = f"iter{iteration} 典型展示基准"
        selection_rule = (
            "在 direct 行中排除 wall/network bytes outlier，选择最接近 direct-24 wall "
            "中位数的 iteration；并列时取较小 iter"
        )
        fact = (
            f"iter{iteration} 是按非异常 wall 中位规则确定的 direct-scope 展示案例。"
        )
        interpretation = "它只是展示基准，不代表总体、稳态或未采样 iteration。"
        hypothesis = "典型展示案例不承载异常因果假设。"
    else:
        label = f"iter{iteration} step 外 gap 诊断案例"
        selection_rule = "从 explicit direct iteration 域固定选取的诊断案例"
        fact = (
            f"iter{iteration} 的 cycle wall、profiled wall、两者 gap 与 endpoint-sum NIC bytes "
            "均为 direct 观察。"
        )
        interpretation = (
            "wall 膨胀位于 profiled step 外；gap 未归因，且不是通信流程阶段。"
        )
        hypothesis = (
            "调度、输入 pipeline 或 host synchronization 是候选，需新增 marker 验证。"
        )
    return {
        "key": key,
        "label": label,
        "iter": iteration,
        "wall_ms": copy.deepcopy(case["wall_ms"]),
        "profiled_global_wall_ms": copy.deepcopy(case["profiled_global_wall_ms"]),
        "profiled_gap_ms": copy.deepcopy(case["profiled_gap_ms"]),
        "nic_bidir_bytes": copy.deepcopy(case["nic_bidir_bytes"]),
        "nic_cluster_full_window_gbps": copy.deepcopy(case["nic_cluster_full_window_gbps"]),
        "wall_outlier": copy.deepcopy(case["wall_outlier"]),
        "network_bytes_outlier": copy.deepcopy(case["network_bytes_outlier"]),
        "source": str(case["source"]),
        "selection_rule": selection_rule,
        "selection_is_population_claim": False,
        "gap_attribution": "unattributed",
        "gap_is_flow_stage": False,
        "fact": fact,
        "interpretation": interpretation,
        "hypothesis": hypothesis,
    }


def _validate_iteration_guide_payload(payload: Mapping[str, Any]) -> None:
    _guide_exact_keys(payload, _GUIDE_ROOT_KEYS, path="guide")
    if payload.get("schema_version") != "run1555-iteration-guide-v1":
        raise ValueError("guide.schema_version must be run1555-iteration-guide-v1")

    scope = _guide_mapping(payload["scope"], path="guide.scope")
    _guide_exact_keys(
        scope,
        {
            "direct_iterations",
            "direct_count",
            "inferred_excluded_count",
            "numerical_source",
            "evidence_boundary",
            "model_scope",
        },
        path="guide.scope",
    )
    direct = scope["direct_iterations"]
    if (
        not isinstance(direct, list)
        or scope["direct_count"] != 24
        or len(direct) != 24
        or len(set(direct)) != 24
        or any(isinstance(value, bool) or not isinstance(value, int) for value in direct)
    ):
        raise ValueError("guide.scope must preserve 24 explicit unique direct iteration IDs")
    if scope["inferred_excluded_count"] != 75:
        raise ValueError("guide.scope must own the excluded-inference count of 75")

    objects = payload["objects"]
    if not isinstance(objects, list) or [item.get("name") for item in objects] != list(
        _GUIDE_OBJECT_NAMES
    ):
        raise ValueError("guide.objects must contain the exact five-object domain")

    steps = payload["steps"]
    if not isinstance(steps, list) or len(steps) != 9:
        raise ValueError("guide.steps must contain exactly nine rows")
    if [step.get("order") for step in steps] != list(range(1, 10)):
        raise ValueError("guide.steps order domain must be unique and exactly 1..9")
    if [step.get("key") for step in steps] != list(_GUIDE_STEP_ORDER):
        raise ValueError("guide.steps key domain is missing, duplicate, reordered, or wrong")
    for index, raw_step in enumerate(steps):
        step = _guide_mapping(raw_step, path=f"guide.steps[{index}]")
        _guide_exact_keys(step, _GUIDE_STEP_KEYS, path=f"guide.steps[{index}]")
        object_names = step["object_names"]
        if (
            not isinstance(object_names, list)
            or len(object_names) != len(set(object_names))
            or not set(object_names) <= set(_GUIDE_OBJECT_NAMES)
        ):
            raise ValueError(f"guide.steps[{index}].object_names has a wrong object domain")
        if not isinstance(step["is_communication"], bool) or not isinstance(
            step["may_repeat"], bool
        ):
            raise TypeError(f"guide.steps[{index}] flags must be booleans")
    by_key = {step["key"]: step for step in steps}
    gap = by_key["step_external_gap"]
    if (
        gap["is_communication"]
        or gap["communication_object"] is not None
        or gap["primitive"] is not None
        or "未归因" not in gap["description"]
    ):
        raise ValueError("guide step 9 gap must be unattributed and non-communication")
    for key, parent in (
        ("forward_moe_with_deepep", "forward"),
        ("backward_moe_with_deepep", "backward"),
    ):
        placement = str(by_key[key]["placement"])
        if "嵌套" not in placement or parent.title() not in placement:
            raise ValueError(f"guide.steps.{key} must remain nested in {parent.title()}")
    if by_key["tiny_control_occurrences"]["may_repeat"] is not True:
        raise ValueError("guide tiny/control step must allow multiple placements")
    optimizer = by_key["distributed_optimizer_region"]
    if (
        "每个分片" not in optimizer["placement"]
        or "kernel payload" not in optimizer["evidence"]
        or "不能直接" not in optimizer["evidence"]
    ):
        raise ValueError("guide optimizer step must preserve mechanism/evidence limits")
    tiny = by_key["tiny_control_occurrences"]
    tiny_ranks = _guide_mapping(tiny["ranks"], path="guide.steps.tiny_control_occurrences.ranks")
    if (
        tiny_ranks.get("domain") != "PG-dependent"
        or tiny_ranks.get("all_occurrences_same_domain") is not False
        or tiny_ranks.get("example")
        != {
            "collective": "default_pg 4B AllReduce occurrence0",
            "ranks": list(range(payload["topology"]["world_size"])),
        }
    ):
        raise ValueError("guide tiny/control ranks must remain PG-dependent with one scoped example")

    primitives = payload["primitives"]
    if not isinstance(primitives, list) or [item.get("name") for item in primitives] != list(
        _GUIDE_PRIMITIVE_NAMES
    ):
        raise ValueError("guide.primitives must contain the exact five-primitives domain")
    concepts = payload["concepts"]
    if not isinstance(concepts, list) or [item.get("name") for item in concepts] != list(
        _GUIDE_CONCEPT_NAMES
    ):
        raise ValueError("guide.concepts must contain the exact six-concept domain")
    concept_keys = {
        "name",
        "definition",
        "object",
        "layout_or_timing",
        "run_evidence",
        "evidence_level",
    }
    for index, concept in enumerate(concepts):
        _guide_exact_keys(concept, concept_keys, path=f"guide.concepts[{index}]")

    cases = _guide_mapping(payload["cases"], path="guide.cases")
    if set(cases) != {"typical", "iter24", "iter88"}:
        raise ValueError("guide.cases must contain exactly typical/iter24/iter88")
    case_keys = {
        "key",
        "label",
        "iter",
        "wall_ms",
        "profiled_global_wall_ms",
        "profiled_gap_ms",
        "nic_bidir_bytes",
        "nic_cluster_full_window_gbps",
        "wall_outlier",
        "network_bytes_outlier",
        "source",
        "selection_rule",
        "selection_is_population_claim",
        "gap_attribution",
        "gap_is_flow_stage",
        "fact",
        "interpretation",
        "hypothesis",
    }
    for key, case in cases.items():
        _guide_exact_keys(case, case_keys, path=f"guide.cases.{key}")
        if case["key"] != key or case["iter"] not in direct:
            raise ValueError(f"guide.cases.{key} has a wrong key/iteration domain")
        if case["gap_attribution"] != "unattributed" or case["gap_is_flow_stage"] is not False:
            raise ValueError(f"guide.cases.{key} gap must remain unattributed and outside flow stages")
    if cases["iter24"]["iter"] != 24 or cases["iter88"]["iter"] != 88:
        raise ValueError("guide iter24/iter88 IDs disagree with their stable keys")

    metrics = _guide_mapping(payload["metrics"], path="guide.metrics")
    _guide_exact_keys(metrics, set(_GUIDE_METRIC_KEYS), path="guide.metrics")
    stages = metrics["stage_medians"]
    if (
        not isinstance(stages, list)
        or len(stages) != 9
        or {row.get("stage") for row in stages} != _GUIDE_STAGE_NAMES
        or len({row.get("stage") for row in stages}) != 9
    ):
        raise ValueError("guide.metrics.stage_medians must preserve the exact nine-stage domain")
    if not isinstance(metrics["edp_pair_summary"], list) or len(metrics["edp_pair_summary"]) != 16:
        raise ValueError("guide.metrics.edp_pair_summary must contain 8 pairs x 2 collectives")
    if not isinstance(metrics["nic_device_summary"], list) or len(metrics["nic_device_summary"]) != 8:
        raise ValueError("guide.metrics.nic_device_summary must contain eight devices")

    config = _guide_mapping(payload["config_facts"], path="guide.config_facts")
    _guide_exact_keys(config, set(_GUIDE_CONFIG_KEYS), path="guide.config_facts")
    for key, fact in config.items():
        _guide_exact_keys(fact, {"value", "source"}, path=f"guide.config_facts.{key}")
        if not str(fact["source"]) or ":" not in str(fact["source"]):
            raise ValueError(f"guide.config_facts.{key}.source must be file:line")
    if not isinstance(payload["quiz"], list) or len(payload["quiz"]) != 7:
        raise ValueError("guide.quiz must contain exactly seven questions")
    if set(payload["boundaries"]) != {
        "direct_evidence",
        "model_scale",
        "trace_resolution",
        "counterfactual",
        "gap",
        "fabric_attribution",
    }:
        raise ValueError("guide.boundaries has a missing, duplicate, or wrong domain")
    _validate_finite_tree(payload)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


def build_iteration_guide_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strict compact Task-6 Guide projection of a Task-5 knowledge model.

    The projection is pure: it accepts no workspace/path and performs no file or root-payload reads.
    All runtime values and rank domains come from the already-validated authoritative model.
    """

    source = _guide_mapping(model, path="model")
    _validate_guide_source_model(source)
    scope = source["scope"]
    groups = source["groups"]
    metrics = source["metrics"]
    facts = source["config_facts"]
    source_steps = {step["name"]: step for step in source["iteration_steps"]}
    selected_facts = {
        key: {
            "value": copy.deepcopy(facts[key]["value"]),
            "source": f"{facts[key]['source_path']}:{facts[key]['source_line']}",
        }
        for key in _GUIDE_CONFIG_KEYS
    }

    steps = [
        {
            "key": "batch_prepare",
            "order": 1,
            "label": "Batch准备",
            "object_names": ["Token"],
            "compute_object": "Token",
            "communication_object": None,
            "ranks": copy.deepcopy(source["topology"]["hosts"]),
            "fabric": "无跨 rank fabric",
            "primitive": None,
            "evidence": str(source_steps["batch_prepare"]["evidence"]),
            "evidence_level": "general_knowledge",
            "placement": "iteration 入口",
            "description": "各 rank 准备 microbatch 与逻辑 token 路由单元；此处不声称原始 token ID 跨 fabric 传输。",
            "is_communication": False,
            "may_repeat": False,
        },
        {
            "key": "forward_dense_attention",
            "order": 2,
            "label": "Forward·Dense/Attention",
            "object_names": ["Token", "Activation"],
            "compute_object": "Activation",
            "communication_object": None,
            "ranks": copy.deepcopy(groups["dp_group"]),
            "fabric": "本地 GPU 计算",
            "primitive": None,
            "evidence": str(source_steps["forward_dense_attention"]["evidence"]),
            "evidence_level": "configuration_plus_aggregate_observation",
            "placement": "Forward 内部",
            "description": "Embedding、attention、residual 与 dense 运算生成 Activation；当前 trace 只给聚合 compute_active。",
            "is_communication": False,
            "may_repeat": True,
        },
        {
            "key": "forward_moe_with_deepep",
            "order": 3,
            "label": "Forward·MoE（内嵌DeepEP）",
            "object_names": ["Activation"],
            "compute_object": "Activation",
            "communication_object": "token 关联的 activation/payload 与 routing metadata",
            "ranks": copy.deepcopy(groups["ep_groups"]),
            "fabric": "host 内 scale-up",
            "primitive": "DeepEP dispatch/combine",
            "evidence": str(source_steps["forward_moe_with_deepep"]["evidence"]),
            "evidence_level": "direct_aggregate_plus_mechanism",
            "placement": "嵌套于 Forward",
            "description": "在 PG 核验的 host 内 EP8 域 dispatch 至 expert，计算后 combine 回原 token 计算路径；不传原始 token ID。",
            "is_communication": True,
            "may_repeat": True,
        },
        {
            "key": "mtp_and_loss",
            "order": 4,
            "label": "Forward·MTP与Loss",
            "object_names": ["Activation"],
            "compute_object": "Activation",
            "communication_object": None,
            "ranks": copy.deepcopy(groups["dp_group"]),
            "fabric": "本地 GPU 计算",
            "primitive": None,
            "evidence": str(source_steps["mtp_and_loss"]["evidence"]),
            "evidence_level": "run_configuration",
            "placement": "Forward 输出端",
            "description": "MTP 与 loss 形成供 autograd 消费的训练目标；当前 stage trace 未逐项计时。",
            "is_communication": False,
            "may_repeat": False,
        },
        {
            "key": "backward_recompute",
            "order": 5,
            "label": "Backward·重算与梯度",
            "object_names": ["Activation", "Gradient"],
            "compute_object": "Gradient",
            "communication_object": None,
            "ranks": copy.deepcopy(groups["dp_group"]),
            "fabric": "本地 GPU 计算",
            "primitive": None,
            "evidence": str(source_steps["backward_recompute"]["evidence"]),
            "evidence_level": "configuration_plus_aggregate_observation",
            "placement": "Backward 内部",
            "description": "按 full/uniform 配置重建所需 Activation，并沿 autograd 依赖传播 Gradient。",
            "is_communication": False,
            "may_repeat": True,
        },
        {
            "key": "backward_moe_with_deepep",
            "order": 6,
            "label": "Backward·MoE（内嵌DeepEP）",
            "object_names": ["Activation", "Gradient"],
            "compute_object": "Gradient",
            "communication_object": "activation gradient 与 routing metadata",
            "ranks": copy.deepcopy(groups["ep_groups"]),
            "fabric": "host 内 scale-up",
            "primitive": "DeepEP dispatch/combine",
            "evidence": str(source_steps["backward_moe_with_deepep"]["evidence"]),
            "evidence_level": "direct_aggregate_plus_mechanism",
            "placement": "嵌套于 Backward",
            "description": "路由 expert activation gradient，执行 expert backward，再 combine input gradient；不是 optimizer moment。",
            "is_communication": True,
            "may_repeat": True,
        },
        {
            "key": "distributed_optimizer_region",
            "order": 7,
            "label": "Distributed optimizer区域",
            "object_names": ["Gradient", "Optimizer shard", "Parameter"],
            "compute_object": "Optimizer shard",
            "communication_object": "先 Gradient，后 Parameter",
            "ranks": {
                "DP": copy.deepcopy(groups["dp_rank_domains"]),
                "EDP": copy.deepcopy(groups["edp_pairs"]),
            },
            "fabric": "DP hybrid 与 EDP pair scale-out",
            "primitive": "ReduceScatter -> local Adam -> AllGather",
            "evidence": (
                "运行配置加 direct collective/Adam 事件；collective kernel payload 字段不能直接标识 "
                "Gradient 或 Parameter 语义。"
            ),
            "evidence_level": "direct_events_plus_mechanism",
            "placement": "每个分片的机制依赖；不同参数域、bucket 与 stream 可交错",
            "description": "对单个 shard 是 ReduceScatter→owner Adam→AllGather；这不是所有 bucket 的全局严格顺序。",
            "is_communication": True,
            "may_repeat": True,
        },
        {
            "key": "tiny_control_occurrences",
            "order": 8,
            "label": "Tiny/control（多处occurrence）",
            "object_names": [],
            "compute_object": None,
            "communication_object": "控制标量或小张量",
            "ranks": {
                "domain": "PG-dependent",
                "all_occurrences_same_domain": False,
                "example": {
                    "collective": "default_pg 4B AllReduce occurrence0",
                    "ranks": copy.deepcopy(groups["dp_group"]),
                },
            },
            "fabric": "由各 occurrence 的 collective process group 决定",
            "primitive": "AllReduce or barrier semantics",
            "evidence": str(source_steps["tiny_control_occurrences"]["evidence"]),
            "evidence_level": "direct_observation",
            "placement": "可在 iteration 多个位置出现；不是统一尾声",
            "description": "不同 occurrence 的 rank 域由各自 PG 决定，不能合并成统一 DP16 域；4B default_pg occurrence0 只是覆盖16 rank的示例。",
            "is_communication": True,
            "may_repeat": True,
        },
        {
            "key": "step_external_gap",
            "order": 9,
            "label": "Step外未归因gap",
            "object_names": [],
            "compute_object": None,
            "communication_object": None,
            "ranks": [],
            "fabric": "未观测",
            "primitive": None,
            "evidence": str(source_steps["step_external_gap"]["evidence"]),
            "evidence_level": "unattributed",
            "placement": "profiled step 边界之外",
            "description": "cycle-minus-profiled-wall 时间未归因、不是通信阶段，也不套用任何通信 stage 颜色。",
            "is_communication": False,
            "may_repeat": False,
        },
    ]

    primitive_index = {item["name"]: item for item in source["primitives"]}
    tiny = metrics["default_4b_allreduce_occ0"]
    concepts = [
        {
            "name": "ReduceScatter",
            "definition": "归并各参与 rank 的贡献，并在每个 owner rank 只留下结果分片。",
            "object": "Gradient（机制语义；kernel payload 未直接命名）",
            "layout_or_timing": "结果是 owner shard，不是每个 rank 都保留完整副本",
            "run_evidence": "metrics 中 DP/EDP ReduceScatter 的 direct stage 中位数与 collective 标签",
            "evidence_level": "direct_events_plus_mechanism",
        },
        {
            "name": "AllGather",
            "definition": "收集成员分片，形成后续计算所需的 Parameter 视图。",
            "object": "Parameter（机制语义；kernel payload 未直接命名）",
            "layout_or_timing": "更新后的 owner shards 汇集为所需视图",
            "run_evidence": "metrics 中 DP/EDP AllGather 的 direct stage 中位数与 collective 标签",
            "evidence_level": "direct_events_plus_mechanism",
        },
        {
            "name": "AllReduce",
            "definition": "归并一个值，并把完整归并结果返回给每个参与 rank。",
            "object": "控制标量或小张量",
            "layout_or_timing": "所有成员到达后，每个参与者都得到完整结果；可见时长含等待",
            "run_evidence": {
                "size_bytes": copy.deepcopy(tiny["size_bytes"]),
                "call_count": copy.deepcopy(tiny["call_count"]),
                "source": str(tiny["source"]),
            },
            "evidence_level": "direct_observation_plus_mechanism",
        },
        {
            "name": "DeepEP dispatch/combine",
            "definition": "把 token 关联的 activation/payload 路由到选中 expert，再把 expert 结果送回原计算路径。",
            "object": "Forward 是 Activation；Backward 是对应 activation gradient；不是原始 token ID",
            "layout_or_timing": "嵌套在 Forward/Backward 内，具体 EP 域由 PG 证据核验为 host-local",
            "run_evidence": {
                "ep_groups": copy.deepcopy(groups["ep_groups"]),
                "source": str(groups["ep_group_source"]),
            },
            "evidence_level": "direct_aggregate_plus_mechanism",
        },
        {
            "name": "arrival skew",
            "definition": "同一 collective 中最早 rank 到达至最晚 rank 到达的时间差。",
            "object": "collective 参与者的到达时序",
            "layout_or_timing": "所有参与者到齐前的等待；必须与 post-last-arrival service 分开",
            "run_evidence": {
                "arrival_skew_median_ms": copy.deepcopy(tiny["arrival_skew_median_ms"]),
                "service_median_ms": copy.deepcopy(tiny["service_median_ms"]),
                "source": str(tiny["source"]),
            },
            "evidence_level": "direct_observation",
        },
        {
            "name": "critical path",
            "definition": "遵守依赖并决定完成时刻的 wall-time 路径。",
            "object": "iteration 区间与依赖",
            "layout_or_timing": "重叠暴露不能相加；当前 counterfactual interval bound 不是完整 DAG",
            "run_evidence": {
                "all_communication_critical_median_ms": copy.deepcopy(
                    metrics["all_communication"]["critical_median_ms"]
                ),
                "communication_only_speedup_median_pct": copy.deepcopy(
                    metrics["comm_only_speedup_median_pct"]["value"]
                ),
                "source": str(metrics["comm_only_speedup_median_pct"]["source"]),
            },
            "evidence_level": "direct_interval_bound",
        },
    ]

    payload: dict[str, Any] = {
        "schema_version": "run1555-iteration-guide-v1",
        "scope": copy.deepcopy(scope),
        "topology": copy.deepcopy(source["topology"]),
        "groups": copy.deepcopy(groups),
        "parallelism": copy.deepcopy(source["parallelism"]),
        "batch": copy.deepcopy(source["batch"]),
        "config_facts": selected_facts,
        "objects": copy.deepcopy(source["data_objects"]),
        "steps": steps,
        "primitives": [copy.deepcopy(primitive_index[name]) for name in _GUIDE_PRIMITIVE_NAMES],
        "concepts": concepts,
        "cases": {
            key: _guide_case(key, source["cases"][key])
            for key in ("typical", "iter24", "iter88")
        },
        "metrics": {key: copy.deepcopy(metrics[key]) for key in _GUIDE_METRIC_KEYS},
        "boundaries": {
            "direct_evidence": "Guide 运行数值只使用 scope.direct_iterations 中显式列出的24个 direct ID；75个 inferred 全部排除。",
            "model_scale": "实际 num_layers=2；不得由目录名中的1188B外推未观测的完整层栈。",
            "trace_resolution": (
                "compute_active 只是聚合证据；当前 stage trace 未逐项计时 Forward/Backward 内部算子。"
            ),
            "counterfactual": "communication-only speedup 是 interval bound，不是依赖 DAG 预测或收益承诺。",
            "gap": "cycle-minus-profiled-wall gap 未归因、不是通信，也位于流程 stage 之外。",
            "fabric_attribution": (
                "MTLink/NIC 观测流量不能全部归因于 DeepEP、rail 或 channel；endpoint-sum 也不等于单份 wire payload。"
            ),
        },
        "claim_audit": copy.deepcopy(source["claim_audit"]),
        "quiz": copy.deepcopy(source["quiz"]),
    }
    _validate_iteration_guide_payload(payload)
    return payload


def _format_number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA（当前证据未观测）"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_iteration_knowledge_markdown(model: dict[str, Any]) -> str:
    """Render a deterministic, self-contained Chinese guide exclusively from the model."""

    facts = model["config_facts"]
    scope = model["scope"]
    topology = model["topology"]
    groups = model["groups"]
    parallel = model["parallelism"]
    batch = model["batch"]
    metrics = model["metrics"]
    cases = model["cases"]

    def f(key: str) -> Any:
        return facts[key]["value"]

    def source(key: str) -> str:
        fact = facts[key]
        return f"`{fact['source_path']}:{fact['source_line']}`"

    host_lines = []
    for host in topology["hosts"]:
        ranks = ", ".join(f"rank{rank}" for rank in host["ranks"])
        host_lines.append(f"Host {host['host']}: [{ranks}]")
    ep_text = "; ".join(str(group) for group in groups["ep_groups"])
    edp_text = ", ".join(str(pair) for pair in groups["edp_pairs"])

    parallel_rows = []
    for name in ("DP", "EP", "EDP", "TP", "PP", "CP"):
        item = parallel[name]
        parallel_rows.append(
            [
                name,
                item["size"],
                "是" if item["cross_rank_communication"] else "否",
                item["role"],
            ]
        )

    object_rows = [
        [item["name"], item["zh"], item["moves_at"], item["owner_change"]]
        for item in model["data_objects"]
    ]
    primitive_rows = [
        [item["name"], item["input"], item["purpose"], item["fabric"]]
        for item in model["primitives"]
    ]
    step_rows = [
        [item["order"], item["name"], item["object"], item["placement"], item["evidence"]]
        for item in model["iteration_steps"]
    ]

    stage_name_zh = {
        "deepep_backward_scaleup": "DeepEP backward scale-up",
        "deepep_forward_scaleup": "DeepEP forward scale-up",
        "dp_ag_hybrid": "DP AllGather hybrid",
        "dp_rs_hybrid": "DP ReduceScatter hybrid",
        "edp_ag_scaleout": "EDP AllGather scale-out",
        "edp_rs_scaleout": "EDP ReduceScatter scale-out",
        "gradient_finalize_sync": "Gradient finalize sync",
        "other_communication": "Other communication",
        "tiny_sync_collective": "Tiny/control collective",
    }
    stage_rows = []
    for row in metrics["stage_medians"]:
        stage_rows.append(
            [
                stage_name_zh[row["stage"]],
                row["sample_count"],
                _format_number(row["critical_median_ms"]),
                _format_number(row["service_median_ms"]),
                _format_number(row["exclusive_median_ms"]),
                _format_number(row["observed_headroom_median_ms"]),
                _format_number(row["observed_headroom_median_pct"]),
            ]
        )
    edp_rows = [
        [
            row["pair"],
            row["collective"],
            row["n"],
            _format_number(row["arrival_p50_ms"]),
            _format_number(row["service_p50_ms"]),
            _format_number(row["critical_p50_ms"]),
        ]
        for row in metrics["edp_pair_summary"]
    ]
    nic_rows = [
        [
            row["host"],
            row["device"],
            row["sample_count"],
            f"{row['bytes_median']:.0f}",
            _format_number(row["full_window_rate_median_gbps"]),
            _format_number(row["active_rate_median_gbps"]),
        ]
        for row in metrics["nic_device_summary"]
    ]
    algorithm_labels = "/".join(
        [
            "+".join(metrics["collective_algorithms"]["algo"]),
            "+".join(metrics["collective_algorithms"]["protocol"]),
        ]
    )

    typical = cases["typical"]
    timeline_rows = [
        [
            row["stage"],
            _format_number(row["first_start_ms"]),
            _format_number(row["last_end_ms"]),
            row["interval_count"],
            _format_number(row["union_duration_ms"]),
        ]
        for row in typical["phase_timeline"]["stages"]
    ]
    case_rows = []
    for key, label in (("typical", "typical展示案例"), ("iter24", "iter24极端gap案例"), ("iter88", "iter88极端gap案例")):
        case = cases[key]
        case_rows.append(
            [
                label,
                case["iter"],
                _format_number(case["wall_ms"]),
                _format_number(case["profiled_global_wall_ms"]),
                _format_number(case["profiled_gap_ms"]),
                f"{case['nic_bidir_bytes']:,}",
                _format_number(case["nic_cluster_full_window_gbps"]),
            ]
        )

    selected_config_keys = [
        "bf16",
        "params_dtype",
        "main_params_dtype",
        "main_grads_dtype",
        "exp_avg_dtype",
        "exp_avg_sq_dtype",
        "num_layers",
        "seq_length",
        "hidden_size",
        "num_attention_heads",
        "normalization",
        "multi_latent_attention",
        "position_embedding_type",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "num_experts",
        "moe_enable_deepep",
        "moe_grouped_gemm",
        "moe_router_dtype",
        "moe_router_pre_softmax",
        "moe_router_score_function",
        "moe_router_topk",
        "moe_aux_loss_coeff",
        "moe_ffn_hidden_size",
        "moe_shared_expert_intermediate_size",
        "moe_shared_expert_overlap",
        "moe_token_dispatcher_type",
        "mtp_num_layers",
        "micro_batch_size",
        "global_batch_size",
        "data_parallel_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
        "use_distributed_optimizer",
        "optimizer",
        "overlap_grad_reduce",
        "overlap_param_gather",
        "overlap_moe_expert_parallel_comm",
    ]
    config_rows = [
        [key, f"`{facts[key]['value']}`", source(key)] for key in selected_config_keys
    ]
    source_rows = [
        [
            key,
            f"`{fact['value']}`",
            f"`{fact['source_path']}:{fact['source_line']}`",
            f"`{fact['source_excerpt'].strip()}`",
        ]
        for key, fact in sorted(facts.items())
    ]

    tiny = metrics["tiny_sync"]
    tiny_call = metrics["default_4b_allreduce_occ0"]
    all_comm = metrics["all_communication"]
    speedup = metrics["comm_only_speedup_median_pct"]

    sections: list[str] = []
    sections.append(
        f"""# Run 1555：一个 Iteration 的计算与通信

> 这是一份可脱离 Dashboard/Guide HTML 独立阅读的中文知识文档。它把训练配置、通用机制与本次 direct profiler 证据分开陈述，不把解释写成事实，也不把假设写成结论。

## 阅读契约与证据边界

- 【本次观察】本文运行数值只由 payload 的 `profiler_view` 中显式列出的 **{scope['direct_count']} 个 direct** iteration 原子重算；成员为 `{scope['direct_iterations']}`。数值入口：`{scope['numerical_source']}`。
- 【本次观察】另外 **{scope['inferred_excluded_count']} 个 inferred** iteration 只用于说明 Evidence 边界，**不参与本文运行数值**，也不替代 direct trace。
- 【本次观察】实际 `num_layers={f('num_layers')}`；“1188B”只是目录/任务标识，**不可外推**到未观测的全层训练行为。配置来源：{source('num_layers')}。
- 【本次观察】当前 stage trace 对计算只给出聚合标签 `compute_active`；embedding、RMSNorm、MLA、router、expert GEMM 等内部算子没有被当前 stage trace 逐项直接标注。
- 【解释】文中的 counterfactual 是区间压缩的 **interval bound**，不是依赖 DAG，也不是承诺可实现的端到端收益。

### 四层标签

- **【通用知识】**：与具体运行无关的机制。
- **【本次观察】**：能被原始配置行或 direct payload 原子直接支持的陈述。
- **【解释】**：对观察的受限解释，必须保留适用条件。
- **【待验证假设】**：需要新 marker、route 映射、对照实验或微基准才能判断。
"""
    )

    sections.append(
        f"""## 第 1 章：先看这次任务的配置

### 1.1 物理拓扑与 rank

【本次观察】TOML 给出的 world size 为 `{f('run.world_size')}`，host split rank 为 `{f('run.host_split_rank')}`；二者分别来自 {source('run.world_size')} 与 {source('run.host_split_rank')}。由此得到 `{topology['host_count']} host × {topology['gpus_per_host']} GPU`：

```text
{host_lines[0]}
      EP host-local group；MTLink scale-up
        |                                      |
        +---------- NIC scale-out ------------+
        |                                      |
{host_lines[1]}
EDP cross-host pairs: {edp_text}
```

- Host A 是 rank `{topology['hosts'][0]['ranks'][0]}`–`{topology['hosts'][0]['ranks'][-1]}`；Host B 是 rank `{topology['hosts'][1]['ranks'][0]}`–`{topology['hosts'][1]['ranks'][-1]}`。
- EP groups 是 `{ep_text}`，每组留在单 host 内；该具体rank域由 `{groups['ep_group_source']}` 直接核验，不是只凭 `EP=8` 猜连续分组。
- EDP pair 域由 direct collective 数据核验为 `{edp_text}`，来源：`{groups['edp_pair_source']}`。
- DP rank域覆盖 `{groups['dp_group']}`；日志中的 `data_parallel_size={f('data_parallel_size')}` 来自 {source('data_parallel_size')}。trace 中存在 `{groups['dp_process_group_identities']}` 等多个PG identity，它们在本run共享同一16-rank域；这些identity不能与EDP pair当成同一个组。

### 1.2 六种并行维度在本 run 中的状态

{_markdown_table(['维度', 'size', '发生跨rank通信', '本任务角色'], parallel_rows)}

【解释】TP、PP、CP 的 size 都为 `{parallel['TP']['size']}`，所以本 run 没有跨 rank 的 TP tensor collective、PP activation send/recv 或 CP sequence collective。这不等于框架没有相关代码路径，只表示当前配置没有形成多成员通信域。EP 与 EDP 不能混写：EP{parallel['EP']['size']} 是 host 内 token 路由域，EDP pair 是同一 local-position 跨 host 的 expert 参数同步域。

### 1.3 关键训练配置

{_markdown_table(['key', 'value', '原始来源'], config_rows)}

### 1.4 batch 与 token 算术

【本次观察】MBS=`{batch['micro_batch_size']}`，GBS=`{batch['global_batch_size']}`，sequence length=`{batch['sequence_length']}`，DP=`{parallel['DP']['size']}`。模型构建时校验 `GBS % (MBS × DP) == 0`，再得到：

- 每 rank、每 microbatch token：`MBS × seq = {batch['micro_batch_size']} × {batch['sequence_length']} = {batch['tokens_per_rank_microbatch']}`。
- 每个 global iteration token：`GBS × seq = {batch['global_batch_size']} × {batch['sequence_length']} = {batch['global_tokens_per_iteration']}`。
- microbatch 数：`{batch['num_microbatches_formula']} = {batch['global_batch_size']} / ({batch['micro_batch_size']} × {parallel['DP']['size']}) = {batch['num_microbatches']}`。

【通用知识】iteration 不是“一个样本做一次 forward/backward”的同义词。它还包含 batch 准备、可能的多个 microbatch、MoE token 搬运、反向重算、梯度收尾、distributed optimizer、控制同步，以及 step 结束至下一 step 开始之间未必被 profiler step 覆盖的空档。
"""
    )

    sections.append(
        f"""## 第 2 章：一个 iteration 的总览

### 2.1 先沿数据对象看

{_markdown_table(['对象英文名', '中文含义', '何时移动/变化', '所有权或位置'], object_rows)}

五个对象不能混为一谈：

1. **Token** 是 MoE 路由的逻辑单元；这里不把它等同于在线上传输的原始 token ID。DeepEP dispatch 的安全表述是移动 **token-associated activation/payload 与 routing metadata**。
2. **Activation** 是 forward 中间结果。DeepEP dispatch/combine 路由的是与 token 关联的 activation/payload，combine 把 expert 输出 activation 放回原计算路径；Backward 要读取它，full recompute 会用计算换存储，再生部分 activation。
3. **Gradient** 从 loss 沿 autograd 逆序产生。ReduceScatter 同时完成跨成员归并与 owner 分片。
4. **Optimizer shard** 是 owner rank 本地持有的更新状态，包含 main parameter 视图与 Adam moments 等。
5. **Parameter** 在更新后经 AllGather 形成下一轮计算需要的参数副本或视图。

### 2.2 依赖教学顺序

{_markdown_table(['教学序号', '阶段', '主要对象', '嵌套/出现位置', '证据类型'], step_rows)}

【通用知识】可读的顶层依赖主线是：batch prepare → Forward（内部包含MoE/DeepEP）→ loss → Backward/recompute（内部包含MoE/DeepEP backward）→ distributed optimizer region → cycle boundary。对单个参数shard，ReduceScatter → owner Adam update → AllGather 是机制依赖；多个参数域、bucket与stream可以交错，**不是全局严格时序**。

【本次观察】tiny/control collective 有多个 occurrence，可在 iteration 多个位置出现，不能统一塞到 optimizer 之后。`gradient_finalize_sync` 是分析器根据 process group 与 AllReduce 赋予的**语义分类标签**，不是 trace 原生算子名，也不能单独证明一个严格位于 Backward 之后的全局步骤。

【解释】这张表是教学依赖顺序，不是把 trace 文件打印顺序强行当成计算图。phase interval 的时间标签可说明“何时有某类活动”，但单靠 first appearance 不能恢复所有 stream、event 和 autograd edge。

### 2.3 四种核心通信原语与一个控制原语

{_markdown_table(['原语', '输入对象', '解决的问题', '通信域/fabric'], primitive_rows)}

【通用知识】ReduceScatter 与 AllReduce 的结果布局不同：前者每个成员只留下归并结果的一片，适合分片 optimizer；后者让每个成员都得到完整归并结果，适合控制标量或需要全副本的张量。AllGather 则把各成员的 shard 汇集起来。barrier 没有“大 payload”也会阻塞，因为它表达的是到达条件。
"""
    )

    sections.append(
        f"""## 第 3 章：Forward 计算

### 3.1 从输入到 loss 的概念路径

【通用知识】一个 Transformer/MoE forward 可按下列概念路径理解：

`Token ids → embedding → RoPE/position encoding → RMSNorm → MLA attention → residual → RMSNorm → MoE router → DeepEP dispatch → grouped GEMM expert MLP/SwiGLU → DeepEP combine → shared expert/residual → MTP layer → loss`

这里“概念路径”只解释算子职责，不声称每一项在 stage trace 中都有独立 span。

### 3.2 通用概念、配置确认、trace 标注三分表

| 计算环节 | 【通用知识】做什么 | 【本次观察】配置确认 | 当前 trace 是否逐算子直接标注 |
| --- | --- | --- | --- |
| Embedding | 把 token id 映射到 hidden vector | position type=`{f('position_embedding_type')}`，hidden size=`{f('hidden_size')}` | 否；只进入 aggregate `compute_active` |
| RoPE | 把相对位置信息注入 attention 表示 | position embedding type=`{f('position_embedding_type')}` | 否 |
| RMSNorm | 规范化残差流尺度 | normalization=`{f('normalization')}` | 否 |
| MLA Q/K/V | 产生 query/key/value 表示并组织 latent attention | multi_latent_attention=`{f('multi_latent_attention')}`，heads=`{f('num_attention_heads')}` | 否 |
| attention score | Q 与 K 形成注意力分数 | attention softmax FP32=`{f('attention_softmax_in_fp32')}` | 否 |
| softmax | 把 score 归一成权重 | attention softmax FP32=`{f('attention_softmax_in_fp32')}` | 否 |
| value aggregation | 权重加权 V，随后 output projection | BF16 parameters=`{f('params_dtype')}` | 否 |
| residual | 合并主分支与子层输出 | hidden size=`{f('hidden_size')}` | 否 |
| MoE router | 计算 expert 分数、softmax、top-k 与 aux balancing signal | router dtype=`{f('moe_router_dtype')}`；score=`{f('moe_router_score_function')}`；top-k=`{f('moe_router_topk')}`；aux loss=`{f('moe_aux_loss_coeff')}` | router 内部否；DeepEP stage 有聚合标签 |
| Expert MLP | grouped GEMM 执行 expert FFN，SwiGLU 提供门控非线性 | grouped GEMM=`{f('moe_grouped_gemm')}`；SwiGLU=`{f('swiglu')}`；expert intermediate=`{f('moe_ffn_hidden_size')}` | 内部 GEMM 否 |
| Shared expert | 为 token 提供共享 expert 路径 | shared intermediate=`{f('moe_shared_expert_intermediate_size')}`；overlap=`{f('moe_shared_expert_overlap')}` | 否；日志没有 shared-expert 数量字段，不能写成“1个shared expert” |
| MTP/loss | MTP 辅助预测并汇入训练目标 | MTP layers=`{f('mtp_num_layers')}` | 没有细粒度 stage 标签 |

### 3.3 dtype 不是一个全局开关

【本次观察】模型参数 dtype=`{f('params_dtype')}`，`bf16={f('bf16')}`；optimizer main params=`{f('main_params_dtype')}`，main grads=`{f('main_grads_dtype')}`，Adam first/second moments 分别为 `{f('exp_avg_dtype')}` / `{f('exp_avg_sq_dtype')}`。Router 是 `{f('moe_router_dtype')}`，attention softmax FP32=`{f('attention_softmax_in_fp32')}`。

【解释】这些 dtype 处于不同数值角色：BF16 参数/梯度节约存储与带宽；FP32 router/softmax 改善归一化稳定性；FP32 main params/moments 支持 optimizer 更新精度。不能只用“本 run 是 BF16”概括全部对象。

### 3.4 trace 能说什么、不能说什么

【本次观察】阶段聚合能看到 `compute_active` 与各通信 stage 的区间并集、重叠和 exclusive wall；它没有逐项给出 embedding、QKV projection、attention softmax、expert grouped GEMM 的独立时长。因此本文只把这些算子作为由配置确认的 forward 结构，不输出虚构的逐算子 latency，也不把 trace 时间顺序等同完整计算依赖。
"""
    )

    sections.append(
        f"""## 第 4 章：MoE/DeepEP 通信

### 4.1 dispatch → expert compute → combine

【通用知识】Router 为每个 token 选择 top-k expert。若 expert 不在当前rank，DeepEP dispatch 移动的是 **token-associated activation/payload 与路由 metadata**，不是原始token ID；目标 rank 对到达表示做 expert grouped GEMM/SwiGLU；DeepEP combine 再把 expert 输出 activation 按原计算路径汇回。Backward 中方向与对象相应变化：expert output 对应的 activation gradient 路由到 expert，expert input gradient 再合并回原路径。

【本次观察】本 run 的 EP size=`{parallel['EP']['size']}`，EP groups=`{ep_text}`，dispatcher=`{f('moe_token_dispatcher_type')}`，DeepEP enabled=`{f('moe_enable_deepep')}`，top-k=`{f('moe_router_topk')}`，experts=`{f('num_experts')}`。EP group 与 host split 对齐，所以这里的 token dispatch/combine 是 host 内 EP{parallel['EP']['size']} scale-up 角色。

### 4.2 MTLink 与 DeepEP 的证据边界

- 【通用知识】MTLink 是本任务的 scale-up fabric；NIC 是跨 host 的 scale-out fabric。
- 【本次观察】payload 同时记录 MTLink、NIC、DeepEP instrumented stage，但它们是不同观测口径。
- 【解释】DeepEP forward/backward stage 与 host 内 token 路由机制一致；然而**不能把全部 MTLink bytes 归因 DeepEP**。MTLink 还可能承载其他 host 内 collective、数据复制或框架通信。
- 【解释】同理，DeepEP instrumented elapsed 不是 collective 的 post-last-arrival service；二者起止语义不同。

### 4.3 overlap 配置只说明本 run 的状态

【本次观察】`overlap_moe_expert_parallel_comm={f('overlap_moe_expert_parallel_comm')}`，来源 {source('overlap_moe_expert_parallel_comm')}。

【解释】本 run 关闭 MoE expert-parallel overlap，只能说明当前样本的配置状态；**不能据此推断开启后也无效**。要评价 overlap，需要配置外的对照运行，并同时检查 token 分布、stream 依赖、compute 可覆盖窗口和 wall 变化。

【待验证假设】如果后续打开 overlap，可以比较同一 direct scope 下 DeepEP critical/exclusive、compute overlap 与 wall，而不是只比较某个 kernel duration。也应确认路由负载与 batch 相当，否则配置差异会和 workload 差异混在一起。
"""
    )

    sections.append(
        f"""## 第 5 章：Backward 与梯度

### 5.1 autograd 逆序和 activation

【通用知识】Backward 从 loss 沿 autograd graph 逆序传播。Forward 保存的 Activation 提供局部导数所需输入；若保存不足，就在 backward 前重算对应 forward。Weight gradient、input gradient 与 router/expert 路径都遵守依赖，不应仅按日志打印顺序理解。

【本次观察】recompute granularity=`{f('recompute_granularity')}`，method=`{f('recompute_method')}`，recompute layers=`{f('recompute_num_layers')}`，实际 layers=`{f('num_layers')}`。来源分别是 {source('recompute_granularity')}、{source('recompute_method')}、{source('recompute_num_layers')}、{source('num_layers')}。

【解释】`full` + `recompute_num_layers={f('recompute_num_layers')}` 表示配置请求对相应层执行 full recompute；当前 canonical stage 没有明确的 recompute marker。它不意味着所有 activation 都不保存，也不能从 aggregate `compute_active` 精确拆出重算开销。对 `{f('num_layers')}`-layer 实测配置的观察不能外推到目录名暗示的其他规模。

### 5.2 dense 与 expert backward

1. Attention/MLP backward 产生 activation gradient 与 weight gradient。
2. MoE backward 需要把 expert output gradient dispatch 到对应 expert，执行 expert backward，再 combine expert input gradient。
3. Shared expert 与 routed expert 的梯度在其各自参数域形成。
4. 框架通常还需完成梯度收尾与参数桶依赖；但当前 `gradient_finalize_sync` 只是按 PG+AllReduce 形成的分析分类，不是 trace 原生算子标签，也不授权一个固定的全局时序位置。

【本次观察】trace 对 DeepEP backward 有 `deepep_backward_scaleup` 聚合stage；分析器还把特定 PG+AllReduce 分类为 `gradient_finalize_sync`。后者是分类器语义标签，不是trace原生“gradient finalize”算子。计算内部仍是 `compute_active` 聚合，因此不能声称 trace 已逐项标出 attention dgrad、wgrad、每个 expert GEMM或精确的梯度收尾顺序。

### 5.3 数据对象变化

- Forward 输出 **Activation**；recompute 再生 Activation。
- Backward 消费 Activation，产生 **Gradient**。
- DeepEP backward 搬运的是与 expert 路由相关的 gradient/metadata，不是 optimizer moment。
- ReduceScatter 后 Gradient 变为 owner shard；Adam 本地更新 **Optimizer shard**；AllGather 搬运更新后的 **Parameter** shard。

【待验证假设】若要定位到达偏斜上游，应沿最早/最晚 rank 的 event 依赖回溯到具体 compute/dispatch 结束点；当前聚合 stage 不能独自识别某一个内部算子是 straggler。
"""
    )

    sections.append(
        f"""## 第 6 章：Distributed optimizer 通信

### 6.1 每个参数shard的 ReduceScatter → Adam → AllGather 机制

【通用知识】Distributed optimizer 把传统“所有 rank 都保存完整 optimizer state”的方案改成 owner 分片。下列顺序描述**一个参数shard的机制依赖**，不是所有bucket/参数域的全局严格trace顺序：

1. **ReduceScatter(Gradient)**：归并各 rank 对同一参数的梯度贡献，并把不同 shard 留在不同 owner。
2. **Adam local update(Optimizer shard)**：owner 使用 FP32 main params 与 FP32 moments 更新自己负责的 shard。
3. **AllGather(Parameter)**：聚合更新后的 parameter shards，让下一次 forward/backward 获得需要的参数视图。

不同dense/expert参数域、bucket和stream可交错执行；direct collective atom直接给出collective、PG、消息量与时间，但没有把payload字段命名为“Gradient”或“Parameter”。这里的数据对象映射属于通用distributed-optimizer机制加本run配置/事件支持。

【本次观察】`use_distributed_optimizer={f('use_distributed_optimizer')}`，optimizer=`{f('optimizer')}`，main params=`{f('main_params_dtype')}`，moments=`{f('exp_avg_dtype')}`/`{f('exp_avg_sq_dtype')}`。这些值来自 RANK0 原始参数表，不由报告标题推断。

### 6.2 DP{parallel['DP']['size']} 与 EDP pair 的不同角色

- Dense 参数的同步域可覆盖 DP=`{parallel['DP']['size']}` rank，payload 中对应 DP ReduceScatter/AllGather hybrid stage。
- Expert 参数不能把不同 expert 混在同一 dense 归并中；本 run 的 EDP direct pair 是 `{edp_text}`，即相同 host-local position 跨 host 的两 rank 同步。
- EP{parallel['EP']['size']} 负责 token 去哪个 expert rank；EDP pair 负责对应 expert 参数/梯度跨副本域同步。一个是 token 路由，一个是 expert data-parallel owner 域。

### 6.3 overlap 与 exposed wall

【本次观察】`overlap_grad_reduce={f('overlap_grad_reduce')}`，`overlap_param_gather={f('overlap_param_gather')}`，来源 {source('overlap_grad_reduce')} 与 {source('overlap_param_gather')}。

【解释】两者关闭时，相关通信**可能**更直接暴露在 wall；“可能”不是因果量化。要证明多少 wall 来自 overlap 配置，必须有控制变量一致的开关实验。当前报告只提供 direct interval 的 critical/exclusive/headroom 描述。

### 6.4 {algorithm_labels} 的窄解释

【本次观察】direct collective call 的唯一 algo/protocol 标签是 `{algorithm_labels}`，来源：`{metrics['collective_algorithms']['source']}`。**{algorithm_labels} 不提供 ring order、rail 或 channel**，也不揭示某条物理路径。它只标明算法/协议类别。若要解释 EDP local-position 双峰，仍需要 communicator topology、route/channel mapping 或 rank remapping 对照。

### 6.5 EDP direct pair 摘要

{_markdown_table(['pair', 'collective', 'calls', 'arrival p50 ms', 'service p50 ms', 'critical p50 ms'], edp_rows)}

【本次观察】表中 AllGather 行汇总两个 occurrence，因此每个 pair 的调用数与 ReduceScatter 行不同；所有值来自 `/profiler_view/data/spatial_temporal/edp/direct_summary`。pair 之间的分层跨 direct scope 重复出现，可称稳定空间观察；不能据此指定物理 rail/ring/channel 原因。

【待验证假设】做 local rank 奇偶置换：若慢组跟随 local-position 移动，支持位置相关解释；若跟随物理设备/路径不动，则支持 fabric/device 候选。无论哪种结果，都要结合 route mapping，而不是从 algo 字符串猜环顺序。
"""
    )

    sections.append(
        f"""## 第 7 章：控制同步与 straggler

### 7.1 AllReduce 和 barrier 的同步语义

【通用知识】AllReduce 同时有数据归并与所有成员得到结果的语义；barrier 主要表达“所有参与者都到达”。即使 payload 很小，早到 rank 也可能长时间等待迟到 rank，因此 collective kernel 的可见跨度不能直接解释为 wire transfer service。

### 7.2 三段时间必须拆开

- **arrival skew**：最早 rank 到达至最后 rank 到达的差。
- **post-last-arrival service**：最后一个 rank 到达后，collective 完成数据服务所需时间。
- **finish spread**：各 rank 完成时刻的离散。
- critical exposure 常包含等待与 service；exclusive wall 还要扣除与 compute/其他 stage 的重叠。

【本次观察】`{tiny_call['pg_description']}`、size_bytes=`{tiny_call['size_bytes']}`、occurrence_index=`{tiny_call['occurrence_index']}` 的 AllReduce 在 direct scope 有 `{tiny_call['call_count']}` 次调用。动态摘要：

- arrival skew median: **{tiny_call['arrival_skew_median_ms']:.3f} ms**
- post-last-arrival service median: **{tiny_call['service_median_ms']:.3f} ms**
- finish spread median: **{tiny_call['finish_spread_median_ms']:.3f} ms**
- critical exposure median: **{tiny_call['critical_median_ms']:.3f} ms**

聚合来源：`{tiny_call['source']}`。

【解释】该案例支持“长可见跨度主要由 rank 到达等待构成”，不支持“{tiny_call['size_bytes']}-byte payload 在网络上传输了整个 critical 时长”。因此排查顺序应先追迟到 rank 的上游依赖，再讨论网络 service。

【待验证假设】候选上游包括 compute straggler、stream/event 依赖、host-side launch 差异或框架控制路径。当前 call atom 只观测到 arrival/service/finish，不能识别候选中哪一个成立。
"""
    )

    sections.append(
        f"""## 第 8 章：计算通信重叠与 critical path

### 8.1 interval union：为什么占比不能相加

【通用知识】若通信 A 与通信 B 在同一 wall 区间重叠，`duration(A)+duration(B)` 会重复计算；若通信又与 compute 重叠，把阶段 critical 百分比直接相加更会超过真实 wall。正确基础是 atomic interval union，再按 label 集合区分 compute-only、communication-only、compute-communication overlap 与 idle/unattributed。

【本次观察】direct wall median: **{metrics['wall_median_ms']['value']:.3f} ms**。所有通信 interval union 的 critical 中位数为 **{all_comm['critical_median_ms']:.3f} ms**，占 wall 中位百分比 **{all_comm['critical_median_pct']:.3f}%**。来源：`{metrics['wall_median_ms']['source']}` 与 `{all_comm['source']}`。

### 8.2 direct {scope['direct_count']} stage 中位数

{_markdown_table(['stage', 'N', 'critical ms', 'service ms', 'exclusive ms', 'observed headroom ms', 'observed headroom %'], stage_rows)}

每一行来源都是 `/profiler_view/data/stage_share` 与 `/profiler_view/data/stage_headroom` 的同 stage direct `{scope['direct_count']}` 行中位数；headroom grid 总计 `{len(metrics['stage_medians']) * scope['direct_count']}` 行。DeepEP 的 service 是 NA，因为当前 instrumented active union 不是 post-last-arrival collective service，不能伪造该值。

### 8.3 tiny critical 与 wall headroom 的差别

【本次观察】tiny/control stage 的 critical median 是 **{tiny['critical_median_ms']:.3f} ms**（**{tiny['critical_median_pct']:.3f}%**），service median 是 **{tiny['service_median_ms']:.3f} ms**（**{tiny['service_median_pct']:.3f}%**），exclusive median 是 **{tiny['exclusive_median_ms']:.3f} ms**（**{tiny['exclusive_median_pct']:.3f}%**），observed wall headroom median 只有 **{tiny['observed_headroom_median_ms']:.3f} ms**（**{tiny['observed_headroom_median_pct']:.3f}%**）。来源：`{tiny['source']}`。

【解释】这正是“阶段 critical 暴露不能直接相加、也不能直接当收益”的例子。大量 tiny waiting 与其他 compute/communication interval 重叠；只有压缩后能真正缩短 wall 且不被依赖链替代的部分才是 wall headroom。

### 8.4 counterfactual 的适用范围

【本次观察】communication-only median speedup: **{speedup['value']:.4f}%**。来源：`{speedup['source']}`。

【解释】这个数字由 direct scope 的 interval-bound counterfactual 得到；它不是依赖 DAG，也没有模拟资源竞争、上游 arrival 改变、下游 critical path 切换或算法实现成本。因此它是排序候选的边界，不是性能承诺。旧报告中的另一口径不得覆盖当前 profiler_view 动态 KPI。
"""
    )

    sections.append(
        f"""## 第 9 章：完整 iteration 案例

### 9.1 typical 展示案例如何选

【本次观察】typical 选择规则是：在 `wall_outlier=false` 且 `network_bytes_outlier=false` 的 direct rows 中，选择 wall 最接近 direct `{scope['direct_count']}` wall median 的行；距离相同取较小 iter。得到 iter `{typical['iter']}`。

**典型案例不代表普遍稳态**：它只是可重复、确定性的展示基准，不能替代分布统计，也不能代表未采样 iteration。选择规则来自模型字段，而非手工指定 iter。

### 9.2 三个案例的同口径数字

{_markdown_table(['案例', 'iter', 'cycle wall ms', 'profiled wall ms', 'gap ms', 'endpoint-sum NIC bytes', 'NIC full-window Gbps'], case_rows)}

- iter24 gap: **{cases['iter24']['profiled_gap_ms']:.3f} ms**
- iter88 gap: **{cases['iter88']['profiled_gap_ms']:.3f} ms**
- iter88 相对 iter24 的 endpoint-sum NIC bytes 差异：**{cases['iter24_vs_iter88']['nic_difference_pct']:.6f}%**，公式与来源：`{cases['iter24_vs_iter88']['source']}`。

### 9.3 iter `{typical['iter']}` 的 phase interval 概要

{_markdown_table(['trace label类别', 'first start ms', 'last end ms', 'atomic interval数', 'union duration ms'], timeline_rows)}

来源：`{typical['phase_timeline']['source']}`。语义：`{typical['phase_timeline']['semantics']}`。

【解释】上表按 direct `phase_intervals` 的 label 汇总大量 atomic interval 中的当前案例，不转储每个区间。它可展示标签首次出现、最后结束与 union duration；因为同一类别可多次穿插，`first→last` 不是连续占用，也不能把 trace 顺序强行等同计算图依赖。教学依赖仍按 forward → backward → optimizer 理解。

### 9.4 iter24/88 的 FACT / INTERPRETATION / HYPOTHESIS

- **FACT /【本次观察】**：两个案例的 cycle wall、profiled step wall、gap 与 endpoint-sum NIC bytes 如表；二者 wall outlier=true，而 network bytes outlier=false。iter28 也在 wall outlier 列表 `{metrics['wall_outlier_iters']}` 中，但 iter24/88 被选为极端 gap 对照。网络 bytes outlier 列表是 `{metrics['network_outlier_iters']}`。
- **INTERPRETATION /【解释】**：wall inflation 位于 profiled step 外，NIC bytes 没有同比增长。因为 full-window Gbps 用更长 cycle wall 作分母，平均 Gbps 会下降；这不是端口 active service 变慢的充分证据。
- **HYPOTHESIS /【待验证假设】**：调度、输入 pipeline、host synchronization 是候选；当前没有对应 marker。**gap 未归因，不能填成通信**，也不能在阶段图中伪造一个“通信阶段”。
"""
    )

    sections.append(
        f"""## 第 10 章：读图和诊断方法

### 10.1 先定义口径，再比较大小

1. **先看 bytes，再看 Gbps。** `endpoint-sum NIC` 是端点收发计数求和，不等于 logical wire payload；复制、方向与端点统计都会影响口径。
2. **区分 full-window rate 与 active rate。** full-window rate 的分母是整个 cycle/window，含 idle 或未归因 gap；active rate 只在设备活跃区间估算，两者回答不同问题。
3. **先拆 arrival/service，再说网络。** collective critical 很长可能是到达等待，不等于 post-last-arrival service 慢。
4. **先看 pair/rank/local-position 分布，再看全局 median。** 全局 p50 会折叠稳定双峰与设备层差异。
5. **direct / inferred / not-observed 不混用。** 本文运行数值仅来自 explicit direct list；inferred 只保留在证据边界说明中。

### 10.2 fabric 角色

- **MTLink scale-up**：host 内 GPU 间通路，与 EP{parallel['EP']['size']}/DeepEP token 路由的物理方向一致；但不能把全部 MTLink 流量都命名为 DeepEP。
- **NIC scale-out**：host 间通路，承载 EDP pair 等跨机通信；endpoint-sum bytes 不等于线上的单份 payload。
- **DP hybrid**：DP collective 可跨 host 与 host 内路径组合，stage 名的 hybrid 不给出具体 rail/channel。

### 10.3 EDP pair 双峰与 GPU local-position

【本次观察】EDP direct pair 域是 `{edp_text}`；已有 direct 空间摘要表现为奇偶 pair 的稳定双峰。这里可把 pair 的首个 rank 看作 host 内 GPU **local-position**，所以 pair 级分组比全局 p50 保留更多结构。

【解释】“local-position 信号”只是一种定位方式，不是物理原因。现有字段没有 route/channel 完整映射；不能把双峰窄化成 rail、ring order 或 channel 故障。`RING/SIMPLE` 也不补足这些证据。

【待验证假设】用奇偶 local rank remap、固定 payload EDP 微基准、communicator route dump 做三角验证，观察慢组跟随逻辑位置、设备还是物理路径移动。

### 10.4 NIC device direct 摘要与 bond5 措辞

{_markdown_table(['host', 'device', 'N', 'bytes median', 'full-window Gbps median', 'active Gbps median'], nic_rows)}

来源：`/profiler_view/data/spatial_temporal/nic/{{bytes_entity_summary,rate_entity_summary,active_rate_entity_summary}}`；每行都由 explicit direct scope 汇总。

【本次观察】表中 bond5 的 full-window rate 较低，而 active rate 并不相应低；这支持“流量分配/窗口差异”的观察，不足以给端口下性能退化结论。

【解释】诊断 bond5 时应同表比较 bytes、coverage、full-window rate 与 active rate。若 bytes 较少，full-window rate 自然较低；若 cycle 含长 gap，同样 bytes 的 full-window rate 也会下降。只有固定 payload、固定 active window 的端口微基准或错误计数/重传/链路状态证据，才接近设备性能判断。

### 10.5 一套可复用排查顺序

- Scope：确认 explicit direct member list，拒绝 range/modulo 猜测。
- Window：区分 cycle wall、profiled step wall 与两者 gap。
- Volume：核对 NIC/MTLink bytes 与不确定字节。
- Rate：分别看 full-window 与 active rate。
- Collective：拆 arrival skew、service、finish spread。
- Spatial：按 EDP pair、rank、GPU local-position、host/device 分组。
- Overlap：用 interval union、exclusive 与 wall headroom，不把 stage pct 相加。
- Causality：事实、解释、待验证假设分栏；缺 marker 就明确“未观测”。
"""
    )

    quiz_parts = [
        f"### {index}. {item['question']}\n\n**答案：** {item['answer']}"
        for index, item in enumerate(model["quiz"], start=1)
    ]
    sections.append(
        "## 第 11 章：自测题\n\n"
        "每题答案直接可见，使这份 Markdown 在无 JavaScript、无 HTML 时也能完成学习闭环。\n\n"
        + "\n\n".join(quiz_parts)
        + "\n\n### 学习闭环\n\n"
        "答题时应始终指出对象、通信域、fabric、证据类型与不能推出的内容。"
        "例如解释 tiny waiting 时，答案必须同时包含 arrival 与 service；解释 headroom 时，"
        "必须说明 overlap 与 interval-bound 限制。"
    )

    audit = model["claim_audit"]
    audit_parts = []
    for key, title in (
        ("allowed", "Allowed"),
        ("narrow", "Narrow"),
        ("discussion", "Discussion"),
        ("prohibited", "Prohibited"),
    ):
        audit_parts.append(f"### {title}\n\n" + "\n".join(f"- {item}" for item in audit[key]))

    audit_text = "\n\n".join(audit_parts)
    sections.append(
        f"""## 附录 A：Claim audit

本文不写论文贡献，只审计陈述的证据层级。Allowed 可直接使用；Narrow 必须保留限定；Discussion 只能作为验证议题；Prohibited 不进入结论。

{audit_text}

### 中文核对

- 【通用知识】机制定义可跨运行复用，但不能伪装成本 run 的 trace 观察。
- 【本次观察】每个配置值应落到原始 RANK0/TOML 行，每个运行数值应落到 profiler_view JSON pointer/聚合说明。
- 【解释】MTLink/NIC、arrival/service、critical/exclusive/headroom 的口径必须分开。
- 【待验证假设】step 外 gap、EDP 双峰物理原因与 overlap 效果都需要新增证据。

## 附录 B：配置事实与原始来源行

下表由 parser 动态生成。每个训练参数在唯一 RANK0 argument table 中必须恰好出现一次、类型正确且数值有限；TOML 事实保留真实行号。计划文档不进入事实源。

{_markdown_table(['key', 'typed value', 'source file:line', 'source excerpt'], source_rows)}

## 附录 C：动态数值来源索引

| 内容 | JSON pointer / 聚合规则 |
| --- | --- |
| direct wall median | `{metrics['wall_median_ms']['source']}` |
| all communication critical | `{all_comm['source']}` |
| stage medians/headroom | `/profiler_view/data/stage_share` + `/profiler_view/data/stage_headroom`，按 stage 对 explicit direct rows 取 median |
| default {tiny_call['size_bytes']}B AllReduce | `{tiny_call['source']}` |
| communication-only speedup | `{speedup['source']}` |
| typical case | `{typical['source']}` + `{typical['selection_rule']}` |
| typical phase timeline | `{typical['phase_timeline']['source']}` |
| iter24/88 | `{cases['iter24']['source']}`；`{cases['iter88']['source']}` |
| EDP pair domain | `{groups['edp_pair_source']}` |

本文所有运行结果文字均由 model 格式化注入；修改传入 payload 的 direct atom/KPI 后重新 build/render，正文中的对应 wall、gap、speedup 与 call metric 会随之变化。生成过程不读取 root KPI 作为回退，也不使用 inferred rows 补数。
"""
    )

    text = "\n\n".join(section.strip() for section in sections) + "\n"
    return text


def write_iteration_knowledge_report(
    workspace: str | Path,
    output_path: str | Path = "reports/HWN_GUIDE_20260714_Run1555_一个Iteration的计算与通信.md",
) -> Path:
    """Build, render, and atomically write the deterministic Task-5 Markdown report."""

    root = Path(workspace).resolve()
    target = Path(output_path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_iteration_knowledge_markdown(build_iteration_knowledge_model(root))
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Run 1555 iteration knowledge guide")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/HWN_GUIDE_20260714_Run1555_一个Iteration的计算与通信.md"),
    )
    args = parser.parse_args(argv)
    path = write_iteration_knowledge_report(args.workspace, args.output)
    print(json.dumps({"output": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

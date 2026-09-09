#!/usr/bin/env python3
"""Transfer source-only rank/stage/microbatch runtime shape into DAG v6.7."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from build_dag_v63_ordered_optimizer_tail import (
    atomic_json,
    atomic_text,
    configured_path,
    git_output,
    max_plus,
    sha256,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "case_224gpu_pp14_cp2_a2a/config/dag_v67_microbatch_runtime_shape_2026w36.toml"
SOURCE_ITERATIONS = tuple(range(60, 101, 5))
TARGET_PHASE_COMPONENTS = (
    "compute_exposed_ns_model",
    "compute_overlap_ns_model",
    "framework_residual_ns_model",
)
TARGET_ALL_COMPONENTS = TARGET_PHASE_COMPONENTS + (
    "network_service_ns_model",
    "software_sync_ns_model",
)
SOURCE_PHASE_COMPONENTS = ("compute_work_ns", "unclassified_calibration_ns")
SOURCE_ALL_COMPONENTS = SOURCE_PHASE_COMPONENTS + (
    "network_service_ns_model",
    "software_sync_ns_model",
)


def checked(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def atomic_csv(path: Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def verify_v66_seal(path: Path) -> dict[str, Any]:
    seal = json.loads(checked(path).read_text())
    if seal.get("status") != "SEALED_BEFORE_V66_EVALUATOR_ACCESS":
        raise ValueError("v6.6 prediction is not sealed")
    for artifact in seal["artifacts"]:
        source = checked(Path(artifact["path"]))
        if source.stat().st_size != int(artifact["size_bytes"]) or sha256(source) != artifact["sha256"]:
            raise ValueError(f"sealed v6.6 artifact changed: {source}")
    return seal


def schedule(stage: int, pp: int, microbatches: int) -> list[tuple[str, int]]:
    warmup = min(pp - stage - 1, microbatches)
    remaining = microbatches - warmup
    operations: list[tuple[str, int]] = [("forward", microbatch) for microbatch in range(warmup)]
    for index in range(remaining):
        operations.extend((("forward", warmup + index), ("backward", index)))
    operations.extend(("backward", microbatch) for microbatch in range(remaining, microbatches))
    return operations


def transition_category(left: tuple[str, int], right: tuple[str, int]) -> str:
    return f"{left[0][0].upper()}2{right[0][0].upper()}"


def source_release_gap_parameters(
    events: pd.DataFrame, *, pp: int, microbatches: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_lookup = {
        (int(row.iteration), int(row.rank), str(row.phase), int(row.microbatch)):
        (int(row.observed_start_ns), int(row.observed_end_ns))
        for row in events.itertuples(index=False)
    }
    samples: list[dict[str, Any]] = []
    ranks = events[["rank", "pp_stage", "pp_lane"]].drop_duplicates()
    for rank_row in ranks.itertuples(index=False):
        rank = int(rank_row.rank)
        stage = int(rank_row.pp_stage)
        sequence = schedule(stage, pp, microbatches)
        category_ordinals: dict[str, int] = {}
        for iteration in SOURCE_ITERATIONS:
            category_ordinals.clear()
            for sequence_index, (left, right) in enumerate(zip(sequence, sequence[1:])):
                category = transition_category(left, right)
                ordinal = category_ordinals.get(category, 0)
                category_ordinals[category] = ordinal + 1
                left_time = phase_lookup[(iteration, rank, left[0], left[1])]
                right_time = phase_lookup[(iteration, rank, right[0], right[1])]
                samples.append({
                    "iteration": iteration,
                    "rank": rank,
                    "pp_stage": stage,
                    "pp_lane": int(rank_row.pp_lane),
                    "sequence_index": sequence_index,
                    "transition_category": category,
                    "category_ordinal": ordinal,
                    "from_phase": left[0],
                    "from_microbatch": left[1],
                    "to_phase": right[0],
                    "to_microbatch": right[1],
                    "observed_release_gap_ns": right_time[0] - left_time[1],
                })
    sample_frame = pd.DataFrame(samples)
    keys = [
        "rank", "pp_stage", "pp_lane", "sequence_index", "transition_category",
        "category_ordinal", "from_phase", "from_microbatch", "to_phase", "to_microbatch",
    ]
    parameters = sample_frame.groupby(keys, as_index=False).agg(
        release_gap_p10_ns=("observed_release_gap_ns", lambda values: values.quantile(0.10)),
        release_gap_median_ns=("observed_release_gap_ns", "median"),
        release_gap_p90_ns=("observed_release_gap_ns", lambda values: values.quantile(0.90)),
        release_gap_min_ns=("observed_release_gap_ns", "min"),
        release_gap_max_ns=("observed_release_gap_ns", "max"),
        release_gap_samples=("observed_release_gap_ns", "size"),
    )
    for column in ("release_gap_p10_ns", "release_gap_median_ns", "release_gap_p90_ns"):
        parameters[column] = parameters[column].clip(lower=0).round().astype("int64")
    parameters["calibration_source"] = "source256_rank_trace_iterations_60_100"
    return parameters, sample_frame


def mapped_release_parameter(
    parameters: pd.DataFrame,
    *,
    source_stage: int,
    lane: int,
    category: str,
    target_ordinal: int,
    target_category_count: int,
) -> Any:
    candidates = parameters[
        parameters["pp_stage"].eq(source_stage)
        & parameters["pp_lane"].eq(lane)
        & parameters["transition_category"].eq(category)
    ].sort_values("category_ordinal")
    if candidates.empty:
        raise ValueError(f"no source release-gap category for stage={source_stage}, lane={lane}, {category}")
    if target_category_count <= 1 or len(candidates) == 1:
        selected = 0
    else:
        selected = int(round(target_ordinal * (len(candidates) - 1) / (target_category_count - 1)))
    return candidates.iloc[selected]


def edge_record(columns: list[str], source: str, target: str, edge_type: str, case_id: str) -> dict[str, Any]:
    row = {column: "" for column in columns}
    row.update({
        "case_id": case_id,
        "src": source,
        "dst": target,
        "edge_type": edge_type,
        "tensor_key": "microbatch_release",
        "dependency_source": "dag_v67_source_rank_release_gap",
    })
    return row


def release_node_from_template(
    template: pd.Series,
    *,
    node_id: str,
    duration_ns: int,
    rank: int,
    stage: int,
    lane: int,
    microbatch: int,
    category: str,
    target: bool,
) -> pd.Series:
    row = template.copy()
    row["node_id"] = node_id
    row["kind"] = "microbatch_release_gap"
    row["rank"] = rank
    row["pp_stage"] = stage
    row["pp_lane"] = lane
    row["phase"] = "RELEASE" if target else "release"
    row["microbatch"] = microbatch
    row["duration_ns"] = duration_ns
    row["predicted_start_ns"] = 0
    row["predicted_end_ns"] = 0
    row["critical_predecessor"] = ""
    row["on_critical_path"] = False
    if target:
        for column in TARGET_ALL_COMPONENTS:
            row[column] = 0
        row["software_sync_ns_model"] = duration_ns
        row["op_name"] = f"microbatch_release_{category.lower()}"
        row["op_family"] = "software_sync"
        row["stream"] = "runtime"
        row["resource"] = "host_runtime"
        row["timing_component"] = "microbatch_release_software"
        row["timing_source"] = "source256_rank_transition_gap_median"
        row["source_parameter_key"] = category
        row["source_v54_node_id"] = ""
    else:
        for column in SOURCE_ALL_COMPONENTS:
            row[column] = 0
        row["software_sync_ns_model"] = duration_ns
        row["model_component"] = "microbatch_release_software"
        row["semantic_region"] = category
        row["layer_id"] = -1
        row["parent_node_id"] = ""
    return row


def source_runtime_parameters(events: pd.DataFrame, source_nodes: pd.DataFrame) -> pd.DataFrame:
    required = {"iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch", "duration_ns"}
    if not required.issubset(events.columns):
        raise ValueError(f"source trace event fields missing: {sorted(required - set(events.columns))}")
    if tuple(sorted(events["iteration"].astype(int).unique())) != SOURCE_ITERATIONS:
        raise ValueError("source runtime-shape window must be exact iterations 60--100")
    if events["rank"].nunique() != 256 or set(events["phase"]) != {"forward", "backward"}:
        raise ValueError("source trace runtime grid is incomplete")
    observed = events.groupby(
        ["rank", "pp_stage", "pp_lane", "phase", "microbatch"], as_index=False
    ).agg(
        trace_median_ns=("duration_ns", "median"),
        trace_p10_ns=("duration_ns", lambda values: values.quantile(0.10)),
        trace_p90_ns=("duration_ns", lambda values: values.quantile(0.90)),
        trace_mean_ns=("duration_ns", "mean"),
        trace_std_ns=("duration_ns", "std"),
        trace_samples=("duration_ns", "size"),
    )
    selected = source_nodes[
        source_nodes["rank"].ge(0)
        & source_nodes["phase"].isin(["forward", "backward"])
        & source_nodes["microbatch"].ge(0)
    ].copy()
    selected["model_adjustable_ns"] = selected[list(SOURCE_PHASE_COMPONENTS)].fillna(0).sum(axis=1)
    selected["model_fixed_ns"] = selected[["network_service_ns_model", "software_sync_ns_model"]].fillna(0).sum(axis=1)
    modeled = selected.groupby(
        ["rank", "pp_stage", "pp_lane", "phase", "microbatch"], as_index=False
    ).agg(
        model_adjustable_ns=("model_adjustable_ns", "sum"),
        model_fixed_ns=("model_fixed_ns", "sum"),
        model_node_count=("node_id", "size"),
    )
    modeled["model_phase_ns"] = modeled["model_adjustable_ns"] + modeled["model_fixed_ns"]
    parameters = observed.merge(
        modeled,
        on=["rank", "pp_stage", "pp_lane", "phase", "microbatch"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not parameters["_merge"].eq("both").all() or len(parameters) != 256 * 2 * 4:
        raise ValueError("source trace/model rank-stage-microbatch grid mismatch")
    parameters = parameters.drop(columns="_merge")
    group = ["rank", "pp_stage", "pp_lane", "phase"]
    parameters["phase_trace_total_ns"] = parameters.groupby(group)["trace_median_ns"].transform("sum")
    parameters["phase_model_total_ns"] = parameters.groupby(group)["model_phase_ns"].transform("sum")
    parameters["shape_target_phase_ns"] = (
        parameters["trace_median_ns"]
        * parameters["phase_model_total_ns"]
        / parameters["phase_trace_total_ns"]
    )
    parameters["raw_adjustable_shape_factor"] = (
        (parameters["shape_target_phase_ns"] - parameters["model_fixed_ns"])
        / parameters["model_adjustable_ns"]
    )
    if not parameters["raw_adjustable_shape_factor"].map(math.isfinite).all():
        raise ValueError("non-finite source runtime-shape factor")
    if parameters["raw_adjustable_shape_factor"].le(0).any():
        bad = parameters.loc[
            parameters["raw_adjustable_shape_factor"].le(0),
            ["rank", "pp_stage", "phase", "microbatch"],
        ].head().to_dict("records")
        raise ValueError(f"runtime shape would make adjustable work non-positive: {bad}")
    parameters["runtime_p10_over_median"] = parameters["trace_p10_ns"] / parameters["trace_median_ns"]
    parameters["runtime_p90_over_median"] = parameters["trace_p90_ns"] / parameters["trace_median_ns"]
    parameters["calibration_source"] = "source256_rank_trace_iterations_60_100"
    return parameters.sort_values(group + ["microbatch"]).reset_index(drop=True)


def redistribute_group(
    nodes: pd.DataFrame,
    indices: list[int],
    components: tuple[str, ...],
    factors: dict[int, float],
    all_components: tuple[str, ...],
) -> tuple[dict[int, int], dict[int, int]]:
    cells: list[list[Any]] = []
    old_by_mb: dict[int, int] = {}
    for index in indices:
        microbatch = int(nodes.at[index, "microbatch"])
        for component in components:
            value = int(nodes.at[index, component])
            old_by_mb[microbatch] = old_by_mb.get(microbatch, 0) + value
            if value > 0:
                cells.append([index, component, microbatch, value, value * factors[microbatch]])
    old_total = sum(old_by_mb.values())
    weighted_total = sum(float(cell[4]) for cell in cells)
    if old_total <= 0 or weighted_total <= 0:
        raise ValueError("runtime-shape group has no adjustable duration")
    normalization = old_total / weighted_total
    allocations: list[list[Any]] = []
    for index, component, microbatch, old, weighted in cells:
        exact = float(weighted) * normalization
        floor_value = int(math.floor(exact))
        allocations.append([index, component, microbatch, old, floor_value, exact - floor_value])
    remainder = old_total - sum(int(item[4]) for item in allocations)
    for item in sorted(allocations, key=lambda value: float(value[5]), reverse=True)[:remainder]:
        item[4] += 1
    new_by_mb = {microbatch: 0 for microbatch in old_by_mb}
    for index, component, microbatch, _old, allocated, _fraction in allocations:
        nodes.at[index, component] = int(allocated)
        new_by_mb[int(microbatch)] += int(allocated)
    for index in indices:
        nodes.at[index, "duration_ns"] = int(sum(int(nodes.at[index, column]) for column in all_components))
    if sum(new_by_mb.values()) != old_total:
        raise ValueError("runtime-shape redistribution did not conserve adjustable duration")
    return old_by_mb, new_by_mb


def apply_runtime_shape(
    nodes: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    target: bool,
    source_stage_for_target: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = nodes.copy()
    if target:
        phase_names = {"FWD": "forward", "BWD": "backward"}
        components = TARGET_PHASE_COMPONENTS
        all_components = TARGET_ALL_COMPONENTS
        expected_microbatches = 3
    else:
        phase_names = {"forward": "forward", "backward": "backward"}
        components = SOURCE_PHASE_COMPONENTS
        all_components = SOURCE_ALL_COMPONENTS
        expected_microbatches = 4
    lookup = {
        (int(row.pp_stage), int(row.pp_lane), str(row.phase), int(row.microbatch)): row
        for row in parameters.itertuples(index=False)
    }
    selected = output[
        output["rank"].ge(0)
        & output["phase"].isin(phase_names)
        & output["microbatch"].ge(0)
    ]
    transfers: list[dict[str, Any]] = []
    for (rank, stage, lane, graph_phase), group_indices in selected.groupby(
        ["rank", "pp_stage", "pp_lane", "phase"], sort=True
    ).groups.items():
        source_stage = source_stage_for_target[int(stage)] if target else int(stage)
        source_phase = phase_names[str(graph_phase)]
        factors: dict[int, float] = {}
        parameter_rows: dict[int, Any] = {}
        for microbatch in range(expected_microbatches):
            key = (source_stage, int(lane), source_phase, microbatch)
            if key not in lookup:
                raise ValueError(f"missing source runtime shape: {key}")
            parameter_rows[microbatch] = lookup[key]
            factors[microbatch] = float(lookup[key].raw_adjustable_shape_factor)
        indices = [int(value) for value in group_indices]
        old_by_mb, new_by_mb = redistribute_group(output, indices, components, factors, all_components)
        for microbatch in range(expected_microbatches):
            source_row = parameter_rows[microbatch]
            transfers.append({
                "graph_case": "target224" if target else "source256",
                "rank": int(rank),
                "pp_stage": int(stage),
                "pp_lane": int(lane),
                "phase": str(graph_phase),
                "microbatch": microbatch,
                "source_pp_stage": source_stage,
                "source_rank": source_stage * 16 + int(lane),
                "source_raw_shape_factor": factors[microbatch],
                "effective_adjustable_shape_factor": new_by_mb[microbatch] / old_by_mb[microbatch],
                "adjustable_ns_before": old_by_mb[microbatch],
                "adjustable_ns_after": new_by_mb[microbatch],
                "runtime_p10_over_median": float(source_row.runtime_p10_over_median),
                "runtime_p90_over_median": float(source_row.runtime_p90_over_median),
            })
    transfer = pd.DataFrame(transfers)
    conservation = transfer.groupby(["graph_case", "rank", "pp_stage", "phase"])[
        ["adjustable_ns_before", "adjustable_ns_after"]
    ].sum()
    if not conservation["adjustable_ns_before"].eq(conservation["adjustable_ns_after"]).all():
        raise ValueError("per-rank phase adjustable duration changed")
    return output, transfer


def apply_release_gaps(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    target: bool,
    pp: int,
    microbatches: int,
    source_stage_for_target: list[int],
    case_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_nodes = nodes.copy()
    output_edges = edges.copy()
    phase_name = {"forward": "FWD", "backward": "BWD"} if target else {
        "forward": "forward", "backward": "backward"
    }
    selected = output_nodes[
        output_nodes["rank"].ge(0)
        & output_nodes["phase"].isin(phase_name.values())
        & output_nodes["microbatch"].ge(0)
    ]
    node_to_phase = {
        str(row.node_id): (int(row.rank), str(row.phase), int(row.microbatch))
        for row in selected.itertuples(index=False)
    }
    direct_type = "rank_program_order" if target else "local_compute_order"
    direct_edges: dict[tuple[tuple[int, str, int], tuple[int, str, int]], int] = {}
    for edge in output_edges[output_edges["edge_type"].eq(direct_type)].itertuples(index=True):
        left_key = node_to_phase.get(str(edge.src))
        right_key = node_to_phase.get(str(edge.dst))
        if left_key is not None and right_key is not None and left_key != right_key:
            direct_edges[(left_key, right_key)] = int(edge.Index)
    if target:
        boundaries = output_nodes[
            output_nodes["op_name"].isin(["fwd_start", "fwd_end", "bwd_start", "bwd_end"])
        ]
        start_ids = {
            (int(row.rank), str(row.phase), int(row.microbatch)): str(row.node_id)
            for row in boundaries[boundaries["op_name"].str.endswith("start")].itertuples(index=False)
        }
        end_ids = {
            (int(row.rank), str(row.phase), int(row.microbatch)): str(row.node_id)
            for row in boundaries[boundaries["op_name"].str.endswith("end")].itertuples(index=False)
        }
        node_index = {str(node_id): int(index) for index, node_id in enumerate(output_nodes["node_id"])}
    additions_nodes: list[pd.Series] = []
    additions_edges: list[dict[str, Any]] = []
    remove_edges: list[int] = []
    transfer_rows: list[dict[str, Any]] = []
    for rank in range((pp * 16)):
        stage = rank // 16
        lane = rank % 16
        source_stage = source_stage_for_target[stage] if target else stage
        sequence = schedule(stage, pp, microbatches)
        category_counts: dict[str, int] = {}
        for left, right in zip(sequence, sequence[1:]):
            category = transition_category(left, right)
            category_counts[category] = category_counts.get(category, 0) + 1
        category_ordinals: dict[str, int] = {}
        for sequence_index, (left, right) in enumerate(zip(sequence, sequence[1:])):
            category = transition_category(left, right)
            ordinal = category_ordinals.get(category, 0)
            category_ordinals[category] = ordinal + 1
            parameter = mapped_release_parameter(
                parameters,
                source_stage=source_stage,
                lane=lane,
                category=category,
                target_ordinal=ordinal,
                target_category_count=category_counts[category],
            )
            duration = int(parameter.release_gap_median_ns)
            left_key = (rank, phase_name[left[0]], left[1])
            right_key = (rank, phase_name[right[0]], right[1])
            existing_handoff = ""
            if target:
                left_end = end_ids[left_key]
                right_start = start_ids[right_key]
                existing_handoff = f"handoff:{left_end}__to__{right_start}"
            if target and existing_handoff in node_index:
                position = node_index[existing_handoff]
                for component in TARGET_ALL_COMPONENTS:
                    output_nodes.at[position, component] = 0
                output_nodes.at[position, "duration_ns"] = duration
                output_nodes.at[position, "software_sync_ns_model"] = duration
                output_nodes.at[position, "timing_component"] = "microbatch_release_software"
                output_nodes.at[position, "timing_source"] = "source256_rank_transition_gap_median"
                output_nodes.at[position, "source_parameter_key"] = (
                    f"release:{source_stage}:lane{lane}:{category}:{int(parameter.category_ordinal)}"
                )
                node_id = existing_handoff
                placement = "updated_existing_phase_handoff_branch"
            else:
                edge_key = (left_key, right_key)
                if edge_key not in direct_edges:
                    raise ValueError(f"missing {direct_type} edge for {edge_key}")
                edge_index = direct_edges[edge_key]
                source_node = str(output_edges.at[edge_index, "src"])
                target_node = str(output_edges.at[edge_index, "dst"])
                node_id = (
                    f"release:r{rank}:s{stage}:q{sequence_index}:"
                    f"{left[0][0]}{left[1]}_to_{right[0][0]}{right[1]}"
                )
                template = output_nodes.loc[output_nodes["node_id"].eq(target_node)].iloc[0]
                additions_nodes.append(release_node_from_template(
                    template,
                    node_id=node_id,
                    duration_ns=duration,
                    rank=rank,
                    stage=stage,
                    lane=lane,
                    microbatch=right[1],
                    category=category,
                    target=target,
                ))
                additions_edges.extend([
                    edge_record(list(output_edges.columns), source_node, node_id, "microbatch_release_start", case_id),
                    edge_record(list(output_edges.columns), node_id, target_node, "microbatch_release_complete", case_id),
                ])
                remove_edges.append(edge_index)
                placement = "inserted_on_rank_program_order_branch"
            transfer_rows.append({
                "graph_case": "target224" if target else "source256",
                "rank": rank,
                "pp_stage": stage,
                "pp_lane": lane,
                "sequence_index": sequence_index,
                "transition_category": category,
                "category_ordinal": ordinal,
                "from_phase": left[0],
                "from_microbatch": left[1],
                "to_phase": right[0],
                "to_microbatch": right[1],
                "source_pp_stage": source_stage,
                "source_rank": source_stage * 16 + lane,
                "source_category_ordinal": int(parameter.category_ordinal),
                "release_gap_p10_ns": int(parameter.release_gap_p10_ns),
                "release_gap_median_ns": duration,
                "release_gap_p90_ns": int(parameter.release_gap_p90_ns),
                "placement": placement,
                "node_id": node_id,
            })
    if additions_nodes:
        output_nodes = pd.concat([output_nodes, pd.DataFrame(additions_nodes)], ignore_index=True)
    if additions_edges:
        if len(remove_edges) != len(set(remove_edges)):
            raise ValueError("a rank-program edge was selected more than once")
        output_edges = pd.concat(
            [output_edges.drop(index=remove_edges), pd.DataFrame(additions_edges)], ignore_index=True
        )
    return output_nodes, output_edges, pd.DataFrame(transfer_rows)


def replay(nodes: pd.DataFrame, edges: pd.DataFrame, completion_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end, predecessor, critical = max_plus(nodes, edges, completion_id)
    output = nodes.copy()
    ids = output["node_id"].astype(str).tolist()
    output["predicted_start_ns"] = start
    output["predicted_end_ns"] = end
    output["critical_predecessor"] = [ids[value] if value >= 0 else "" for value in predecessor]
    output["on_critical_path"] = [position in critical for position in range(len(output))]
    return output, output[output["on_critical_path"]].sort_values("predicted_start_ns")


def phase_runtime_envelope(nodes: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    selected = nodes[
        nodes["rank"].ge(0) & nodes["phase"].isin(["FWD", "BWD"]) & nodes["microbatch"].ge(0)
    ]
    phase = selected.groupby(
        ["rank", "pp_stage", "pp_lane", "phase", "microbatch"], as_index=False
    ).agg(
        predicted_start_ns=("predicted_start_ns", "min"),
        predicted_end_ns=("predicted_end_ns", "max"),
        node_count=("node_id", "size"),
    )
    phase["predicted_duration_ns"] = phase["predicted_end_ns"] - phase["predicted_start_ns"]
    factors = transfer[[
        "rank", "pp_stage", "pp_lane", "phase", "microbatch", "source_pp_stage", "source_rank",
        "runtime_p10_over_median", "runtime_p90_over_median",
    ]]
    phase = phase.merge(
        factors,
        on=["rank", "pp_stage", "pp_lane", "phase", "microbatch"],
        validate="one_to_one",
    )
    phase["runtime_p10_duration_ns"] = (
        phase["predicted_duration_ns"] * phase["runtime_p10_over_median"]
    ).round().astype("int64")
    phase["runtime_p90_duration_ns"] = (
        phase["predicted_duration_ns"] * phase["runtime_p90_over_median"]
    ).round().astype("int64")
    phase["band_semantics"] = "source256 marginal p10-p90 duration; not propagated DAG completion quantiles"
    return phase


def component_conserved(nodes: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    conserved = nodes[list(columns)].fillna(0).sum(axis=1).round().astype("int64")
    return bool(conserved.eq(nodes["duration_ns"].astype("int64")).all())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = checked(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if tuple(config["split"]["source_calibration"]) != SOURCE_ITERATIONS:
        raise ValueError("v6.7 source split must be stable iterations 60--100")
    v66 = configured_path(config["inputs"]["v66_run_dir"])
    v66_seal_path = checked(v66 / "predictions/prediction_seal.json")
    verify_v66_seal(v66_seal_path)
    target_node_input = checked(v66 / "predictions/dag_v66_nodes.csv.gz")
    target_edge_input = checked(v66 / "predictions/dag_v66_edges.csv.gz")
    target_contract_input = checked(v66 / "prediction_contract.json")
    source_dir = v66 / "source_replay"
    source_node_input = checked(source_dir / "dag_v66_source_nodes.csv.gz")
    source_edge_input = checked(source_dir / "dag_v66_source_edges.csv.gz")
    source_contract_input = checked(source_dir / "source_replay_contract.json")
    source_access_input = checked(source_dir / "input_access_audit.json")
    trace_input = checked(configured_path(config["inputs"]["source_trace_events"]))
    source_access = json.loads(source_access_input.read_text())
    if source_access.get("status") != "PASS_SOURCE_ONLY" or int(source_access["target_timing_files_read"]) != 0:
        raise ValueError("v6.6 source replay was not source-isolated")

    source_nodes_v66 = pd.read_csv(source_node_input, low_memory=False)
    source_edges = pd.read_csv(source_edge_input, low_memory=False)
    target_nodes_v66 = pd.read_csv(target_node_input, low_memory=False)
    target_edges = pd.read_csv(target_edge_input, low_memory=False)
    events = pd.read_csv(trace_input)
    parameters = source_runtime_parameters(events, source_nodes_v66)
    release_parameters, release_samples = source_release_gap_parameters(
        events,
        pp=int(config["source"]["pp"]),
        microbatches=int(config["source"]["microbatches"]),
    )
    stage_map = [int(value) for value in config["model"]["source_stage_for_target"]]
    release_stage_map = [int(value) for value in config["model"]["release_gap_source_stage_for_target"]]
    if len(stage_map) != int(config["target"]["pp"]):
        raise ValueError("target-to-source stage map length mismatch")
    if len(release_stage_map) != int(config["target"]["pp"]):
        raise ValueError("target-to-source release-gap stage map length mismatch")

    source_nodes, source_transfer = apply_runtime_shape(
        source_nodes_v66, parameters, target=False, source_stage_for_target=stage_map
    )
    source_nodes, source_edges, source_release_transfer = apply_release_gaps(
        source_nodes,
        source_edges,
        release_parameters,
        target=False,
        pp=int(config["source"]["pp"]),
        microbatches=int(config["source"]["microbatches"]),
        source_stage_for_target=release_stage_map,
        case_id=config["source"]["case_id"],
    )
    source_nodes, source_critical = replay(source_nodes, source_edges, "iteration:collectives_complete")
    source_raw_ns = int(source_nodes.loc[
        source_nodes["node_id"].eq("iteration:collectives_complete"), "predicted_end_ns"
    ].iloc[0])
    source_contract_v66 = json.loads(source_contract_input.read_text())
    source_profiler_ns = int(round(float(source_contract_v66["source_profiler_median_ms"]) * 1e6))
    source_reconciliation_ns = source_profiler_ns - source_raw_ns
    if source_reconciliation_ns < 0:
        raise ValueError(f"v6.7 shaped source graph exceeds profiler by {-source_reconciliation_ns / 1e6:.6f} ms")

    target_nodes, target_transfer = apply_runtime_shape(
        target_nodes_v66, parameters, target=True, source_stage_for_target=stage_map
    )
    target_nodes, target_edges, target_release_transfer = apply_release_gaps(
        target_nodes,
        target_edges,
        release_parameters,
        target=True,
        pp=int(config["target"]["pp"]),
        microbatches=int(config["target"]["microbatches"]),
        source_stage_for_target=release_stage_map,
        case_id=config["target"]["case_id"],
    )
    target_nodes, target_critical = replay(target_nodes, target_edges, "iteration:completion_join")
    target_raw_ns = int(target_nodes.loc[
        target_nodes["node_id"].eq("iteration:completion_join"), "predicted_end_ns"
    ].iloc[0])
    target_reconciliation_ms = source_reconciliation_ns / 1e6 * (
        int(config["target"]["microbatches"]) / int(config["source"]["microbatches"])
    )
    v66_contract = json.loads(target_contract_input.read_text())
    outer_ms = float(v66_contract["prediction"]["outer_framework_ms"])
    raw_ms = target_raw_ns / 1e6
    profiler_ms = raw_ms + target_reconciliation_ms
    training_ms = profiler_ms + outer_ms
    mfu = 100.0 * float(config["target"]["model_flops_per_iteration"]) / (
        training_ms / 1000.0
        * int(config["target"]["world_size"])
        * float(config["target"]["peak_tflops_per_gpu"])
        * 1e12
    )
    if not component_conserved(source_nodes, SOURCE_ALL_COMPONENTS):
        raise ValueError("source component conservation failed")
    if not component_conserved(target_nodes, TARGET_ALL_COMPONENTS):
        raise ValueError("target component conservation failed")

    output = configured_path(config["outputs"]["output_dir"])
    prediction = output / "predictions"
    calibration = output / "calibration"
    source_output = output / "source_replay"
    source_node_path = source_output / "dag_v67_source_nodes.csv.gz"
    source_edge_path = source_output / "dag_v67_source_edges.csv.gz"
    source_critical_path = source_output / "dag_v67_source_critical_path.csv"
    source_transfer_path = source_output / "source_runtime_shape_transfer.csv"
    source_release_transfer_path = source_output / "source_release_gap_transfer.csv"
    parameter_path = calibration / "source_rank_microbatch_runtime_shape_parameters.csv"
    release_parameter_path = calibration / "source_rank_program_release_gap_parameters.csv"
    release_sample_path = calibration / "source_rank_program_release_gap_samples.csv"
    target_transfer_path = calibration / "target_runtime_shape_transfer.csv"
    target_release_transfer_path = calibration / "target_release_gap_transfer.csv"
    node_path = prediction / "dag_v67_nodes.csv.gz"
    edge_path = prediction / "dag_v67_edges.csv.gz"
    critical_path = prediction / "dag_v67_critical_path.csv"
    envelope_path = prediction / "phase_runtime_envelope.csv"
    atomic_csv(source_node_path, source_nodes, compression="gzip")
    atomic_csv(source_edge_path, source_edges, compression="gzip")
    atomic_csv(source_critical_path, source_critical)
    atomic_csv(source_transfer_path, source_transfer)
    atomic_csv(source_release_transfer_path, source_release_transfer)
    atomic_csv(parameter_path, parameters)
    atomic_csv(release_parameter_path, release_parameters)
    atomic_csv(release_sample_path, release_samples)
    atomic_csv(target_transfer_path, target_transfer)
    atomic_csv(target_release_transfer_path, target_release_transfer)
    atomic_csv(node_path, target_nodes, compression="gzip")
    atomic_csv(edge_path, target_edges, compression="gzip")
    atomic_csv(critical_path, target_critical)
    envelope = phase_runtime_envelope(target_nodes, target_transfer)
    atomic_csv(envelope_path, envelope)

    source_contract = {
        "schema": "dag-v6.7-source256-runtime-shape-replay-v1",
        "status": "PASS_SOURCE_REPLAY",
        "source_iterations": list(SOURCE_ITERATIONS),
        "source_parameter_rows": len(parameters),
        "source_release_parameter_rows": len(release_parameters),
        "source_rank_count": int(events["rank"].nunique()),
        "v66_raw_graph_ms": float(source_contract_v66["v66_raw_graph_ms"]),
        "v67_raw_graph_ms": source_raw_ns / 1e6,
        "source_profiler_median_ms": source_profiler_ns / 1e6,
        "source_reconciliation_ms": source_reconciliation_ns / 1e6,
        "target_reconciliation_scale": 0.75,
        "target_reconciliation_ms": target_reconciliation_ms,
        "adjustable_time_conserved_per_rank_stage_phase": True,
        "rank_program_release_gaps": "source per-rank median max-plus floors",
        "target_timing_files_read": 0,
        "component_conservation": True,
        "acyclic": True,
    }
    source_contract_path = source_output / "source_replay_contract.json"
    atomic_json(source_contract_path, source_contract)
    contract = {
        "schema": "dag-v6.7-microbatch-runtime-shape-prediction-v1",
        "status": "PREDICTIVE_UNEVALUATED",
        "prediction": {
            "raw_graph_ms": raw_ms,
            "source_reconciliation_origin_ms": source_reconciliation_ns / 1e6,
            "target_reconciliation_ms": target_reconciliation_ms,
            "source_reconciliation_ms": target_reconciliation_ms,
            "profiler_step_ms": profiler_ms,
            "outer_framework_ms": outer_ms,
            "training_step_ms": training_ms,
            "mfu_pct": mfu,
        },
        "v66_raw_graph_ms": float(v66_contract["prediction"]["raw_graph_ms"]),
        "v67_minus_v66_raw_graph_ms": raw_ms - float(v66_contract["prediction"]["raw_graph_ms"]),
        "source_replay_status": source_contract["status"],
        "runtime_shape_semantics": "source rank/stage/lane/phase/microbatch median shape with marginal p10-p90 band",
        "runtime_band_limitation": "marginal phase duration only; not a propagated end-to-end completion quantile",
        "release_gap_semantics": "source rank/stage program-order median gap as local max-plus release floor",
        "target_stage_to_source_stage": stage_map,
        "target_release_gap_stage_to_source_stage": release_stage_map,
        "target_timing_read": False,
        "accuracy_claim_allowed": False,
    }
    contract_path = output / "prediction_contract.json"
    atomic_json(contract_path, contract)
    grid = [*config["split"]["target_regression_only"], *config["split"]["target_validation"]]
    methods = pd.DataFrame([{
        "protocol_id": config["protocol"]["id"],
        "source_case": config["source"]["case_id"],
        "target_case": config["target"]["case_id"],
        "method": "DAG v6.7 microbatch runtime shape",
        "method_version": "dag-v6.7-source60-100-v1",
        "status": "PREDICTIVE_UNEVALUATED",
        "split": "regression_only" if int(iteration) == 55 else "validation",
        "iteration": int(iteration),
        "predicted_raw_graph_ms": raw_ms,
        "predicted_source_reconciliation_ms": target_reconciliation_ms,
        "predicted_source_reconciliation_origin_ms": source_reconciliation_ns / 1e6,
        "predicted_target_reconciliation_ms": target_reconciliation_ms,
        "predicted_profiler_step_ms": profiler_ms,
        "predicted_outer_framework_ms": outer_ms,
        "predicted_training_step_ms": training_ms,
        "predicted_mfu_pct": mfu,
        "target_timing_read": False,
    } for iteration in grid])
    method_path = prediction / "method_predictions.csv"
    atomic_csv(method_path, methods)

    source_band_width_ms = (parameters["trace_p90_ns"] - parameters["trace_p10_ns"]) / 1e6
    audit = {
        "schema": "dag-v6.7-runtime-shape-audit-v1",
        "status": "PASS_SOURCE_ONLY_SHAPE_TRANSFER",
        "source_iterations_read": list(SOURCE_ITERATIONS),
        "source_rank_count": int(events["rank"].nunique()),
        "source_parameter_rows": len(parameters),
        "source_release_parameter_rows": len(release_parameters),
        "target_timing_files_read": 0,
        "parameter_updates_from_target": 0,
        "target_adjustable_time_conserved_per_rank_stage_phase": True,
        "source_adjustable_time_conserved_per_rank_stage_phase": True,
        "source_runtime_band_width_ms": {
            "median": float(source_band_width_ms.median()),
            "p90": float(source_band_width_ms.quantile(0.90)),
            "maximum": float(source_band_width_ms.max()),
        },
        "source_shape_factor": {
            "minimum": float(parameters["raw_adjustable_shape_factor"].min()),
            "median": float(parameters["raw_adjustable_shape_factor"].median()),
            "maximum": float(parameters["raw_adjustable_shape_factor"].max()),
        },
        "target_release_gap_nodes": int(len(target_release_transfer)),
        "source_release_gap_nodes": int(len(source_release_transfer)),
        "component_conservation": True,
        "acyclic": True,
    }
    audit_path = output / "runtime_shape_audit.json"
    atomic_json(audit_path, audit)
    input_paths = [
        config_path, v66_seal_path, target_node_input, target_edge_input, target_contract_input,
        source_node_input, source_edge_input, source_contract_input, source_access_input, trace_input,
    ]
    access_path = output / "input_access_audit.json"
    atomic_json(access_path, {
        "schema": "dag-v6.7-input-access-v1",
        "status": "PASS_NO_TARGET_TIMING",
        "source_iterations": list(SOURCE_ITERATIONS),
        "target_timing_files_read": 0,
        "target_fields_read": ["sealed v6.6 target-static predicted graph only"],
        "source_trace_fields_read": [
            "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch",
            "observed_start_ns", "observed_end_ns", "duration_ns"
        ],
        "forbidden_target_fields_read": [],
        "inputs": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in input_paths
        ],
    })
    report_path = output / "DAG_V67_MICROBATCH_RUNTIME_SHAPE.md"
    atomic_text(report_path, f"""# DAG v6.7 microbatch runtime shape

- Status: `PREDICTIVE_UNEVALUATED`
- Source calibration: all 256 ranks, iterations `{list(SOURCE_ITERATIONS)}` only
- Source duration-shape grid: `{len(parameters)}` rank/stage/lane/phase/microbatch cells
- Source program-release grid: `{len(release_parameters)}` rank/transition cells
- Target timing read by model process: `0`
- v6.6 raw graph: `{float(v66_contract['prediction']['raw_graph_ms']):.6f} ms`
- v6.7 raw graph: `{raw_ms:.6f} ms`
- v6.7 profiler prediction: `{profiler_ms:.6f} ms`
- v6.7 training prediction: `{training_ms:.6f} ms`

v6.6 already had separate microbatch nodes, but each node used a deterministic
source median.  v6.7 additionally learns the *relative shape* among F0/F1/F2/F3
for every source rank, PP stage and F/B phase.  It redistributes only adjustable
compute/framework time; the total adjustable time of every rank/stage/phase is
kept exactly unchanged.  Network service and software synchronization are not
scaled by this step.

v6.7 also models the previously missing gaps between consecutive microbatches.
For every source rank it measures F→F, B→B, F→B and B→F program-order gaps,
then inserts the source median as a local max-plus release floor.  This is not
blindly added to PP waiting: the next phase still starts at the maximum of the
local release floor and its PP/autograd predecessor, so a later external
dependency hides the local floor.

The deterministic graph uses the source median shape.  The separate
`phase_runtime_envelope.csv` exposes source p10--p90 marginal duration bands.
Those bands explain that a concrete run can be visibly less smooth, but they are
not end-to-end completion quantiles and they do not guess which microbatch will
be slow in an unseen target iteration.

Target PP13 is mapped to source PP15 for the last-stage duration role; target
PP0 maps to source PP0 and internal PP1--PP12 retain their source stage index.
Release-gap stages use normalized PP position `{release_stage_map}` because the
schedule/bubble role depends on distance through the pipeline, not layer ID.
No 224-GPU timestamp, duration, completion, Step or reported throughput was read
while fitting or applying these parameters.
""")
    reproduction_path = output / "reproduction_command.txt"
    atomic_text(
        reproduction_path,
        f"cd {REPO}\n{REPO / '.venv/bin/python'} {Path(__file__).resolve()} --config {config_path}\n",
    )
    log_path = output / "logs/build.log"
    atomic_text(log_path, "\n".join([
        audit["status"],
        f"source_parameter_rows={len(parameters)}",
        f"source_release_parameter_rows={len(release_parameters)}",
        f"source_raw_graph_ms={source_raw_ns / 1e6:.6f}",
        f"source_reconciliation_ms={source_reconciliation_ns / 1e6:.6f}",
        f"target_raw_graph_ms={raw_ms:.6f}",
        f"predicted_profiler_step_ms={profiler_ms:.6f}",
        f"predicted_training_step_ms={training_ms:.6f}",
        "target_timing_files_read=0",
        "parameter_updates_from_target=0",
    ]) + "\n")
    provenance_path = output / "provenance.json"
    artifacts = [
        source_node_path, source_edge_path, source_critical_path, source_transfer_path,
        source_release_transfer_path, source_contract_path, parameter_path,
        release_parameter_path, release_sample_path, target_transfer_path,
        target_release_transfer_path, node_path, edge_path,
        critical_path, envelope_path, method_path, contract_path, audit_path, access_path,
        report_path, reproduction_path, log_path,
    ]
    atomic_json(provenance_path, {
        "schema": "dag-v6.7-provenance-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status_short": git_output("status", "--short"),
        "artifacts": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in artifacts
        ],
    })
    seal_artifacts = [
        method_path, node_path, edge_path, critical_path, envelope_path, parameter_path,
        release_parameter_path, target_transfer_path, target_release_transfer_path,
        source_contract_path, contract_path, audit_path, access_path,
        report_path, reproduction_path, log_path, config_path, Path(__file__).resolve(),
    ]
    atomic_json(prediction / "prediction_seal.json", {
        "schema": "dag-v6.7-prediction-seal-v1",
        "status": "SEALED_BEFORE_V67_EVALUATOR_ACCESS",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_timing_opened_by_sealer": False,
        "prediction_status": contract["status"],
        "parameter_mutation_after_seal": "FORBIDDEN",
        "artifacts": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in seal_artifacts
        ],
    })
    print(json.dumps({
        "status": audit["status"],
        "source_parameters": len(parameters),
        "source_raw_graph_ms": source_raw_ns / 1e6,
        "target_raw_graph_ms": raw_ms,
        "predicted_profiler_step_ms": profiler_ms,
        "predicted_training_step_ms": training_ms,
        "output_dir": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

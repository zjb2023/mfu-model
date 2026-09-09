"""T32A: audit source local-gap ownership and physical compute joinability."""
from __future__ import annotations

import json
import math
import resource
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from guards import ROOT
from smoke_worker import dump, sha
from worker import csv


FIT = [85, 90]
VALIDATION = [95, 100]
ITERATIONS = FIT + VALIDATION
METHODS = {"median_85_90": FIT, "recent_90": [90]}
PARTITIONS = ["compute_only_ns", "communication_only_ns", "overlap_ns", "non_gpu_ns"]
PARAMETER_KEYS = [
    "rank", "pp_stage", "pp_lane", "phase", "microbatch", "event_index",
    "v54_stage_local_layer", "physical_layer_id", "segment_kind", "semantic_slot",
    "communication_kind",
]


def _rounded_median(values: pd.Series) -> int:
    return int(round(float(values.astype(float).median())))


def _semantic_slot(kind: str, region: str) -> str:
    if kind == "gap":
        if region == "FINAL":
            return "step_exit"
        semantic = region.removeprefix("before:").split(":")[-1].removesuffix("_wall")
        return "before_" + semantic
    return region.split(":")[-1].removesuffix("_wall")


def prepare_segments(
    segments: pd.DataFrame, layer_map: pd.DataFrame, *, strict: bool = True,
) -> pd.DataFrame:
    required = {
        "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch",
        "event_index", "layer_id", "communication_kind", "segment_kind",
        "semantic_region", "observed_start_ns", "observed_end_ns",
        "observed_duration_ns", "compute_active_union_ns",
        "communication_active_union_ns", "compute_communication_overlap_ns",
        "gpu_any_active_union_ns", "runtime_active_union_ns", "exposed_non_gpu_ns",
    }
    assert required.issubset(segments.columns)
    assert {"layer_id", "pp_stage", "stage_local_layer"}.issubset(layer_map.columns)
    frame = segments.copy()
    observed_iterations = sorted(frame.iteration.astype(int).unique().tolist())
    if strict:
        assert observed_iterations == list(range(60, 101, 5))
        assert frame["rank"].nunique() == 16
        assert frame["pp_stage"].nunique() == 16
        assert set(frame.pp_lane.astype(int)) == {0}
        assert set(frame.segment_kind) == {"gap", "communication"}

    layer_to_local = layer_map.set_index("layer_id")["stage_local_layer"].astype(int).to_dict()
    terminal: dict[tuple[int, str], int] = {}
    for stage, group in layer_map.groupby("pp_stage"):
        layers = group.sort_values("stage_local_layer").layer_id.astype(int).tolist()
        terminal[(int(stage), "forward")] = layers[-1]
        terminal[(int(stage), "backward")] = layers[0]

    frame["v54_stage_local_layer"] = frame.layer_id.map(layer_to_local).fillna(-1).astype("int64")
    frame["physical_layer_id"] = frame.layer_id.astype("int64")
    final = frame.semantic_region.eq("FINAL")
    if final.any():
        frame.loc[final, "physical_layer_id"] = [
            terminal[(int(stage), str(phase))]
            for stage, phase in zip(frame.loc[final, "pp_stage"], frame.loc[final, "phase"])
        ]
    frame["execution_scope"] = np.where(
        frame.segment_kind.eq("gap"), "exposed_gap", "overlap_with_collective"
    )
    frame["semantic_slot"] = [
        _semantic_slot(str(kind), str(region))
        for kind, region in zip(frame.segment_kind, frame.semantic_region)
    ]
    frame["overlap_ns"] = frame.compute_communication_overlap_ns.astype("int64")
    frame["compute_only_ns"] = frame.compute_active_union_ns.astype("int64") - frame.overlap_ns
    frame["communication_only_ns"] = (
        frame.communication_active_union_ns.astype("int64") - frame.overlap_ns
    )
    frame["non_gpu_ns"] = (
        frame.observed_duration_ns.astype("int64")
        - frame.compute_only_ns - frame.communication_only_ns - frame.overlap_ns
    )
    if strict:
        assert frame[PARTITIONS].ge(0).all().all()
        assert frame[PARTITIONS].sum(axis=1).eq(frame.observed_duration_ns.astype("int64")).all()
        assert (frame.observed_end_ns - frame.observed_start_ns).eq(frame.observed_duration_ns).all()
        assert not frame.duplicated(["iteration"] + PARAMETER_KEYS).any()
    frame["split"] = np.select(
        [frame.iteration.isin(FIT), frame.iteration.isin(VALIDATION)],
        ["source_fit", "source_incremental_validation"],
        default="historical_exposed_not_used_in_T32_fit_or_score",
    )
    return frame.sort_values(["iteration"] + PARAMETER_KEYS, kind="stable").reset_index(drop=True)


def fit_partition_candidates(
    prepared: pd.DataFrame, *, strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return source-fit parameters and a truth-free fixed prediction grid."""
    parameters: list[pd.DataFrame] = []
    prediction_grid: list[pd.DataFrame] = []
    for method, iterations in METHODS.items():
        selected = prepared[prepared.iteration.isin(iterations)]
        aggregations = {
            f"predicted_{column}": (column, _rounded_median)
            for column in [*PARTITIONS, "observed_duration_ns"]
        }
        fitted = selected.groupby(PARAMETER_KEYS, dropna=False, sort=True).agg(
            **aggregations,
            calibration_samples=("iteration", "size"),
            calibration_iterations=("iteration", "nunique"),
        ).reset_index()
        fitted.insert(0, "method", method)
        fitted["fit_iterations"] = "|".join(map(str, iterations))
        fitted["predicted_partition_total_ns"] = fitted[
            [f"predicted_{column}" for column in PARTITIONS]
        ].sum(axis=1)
        fitted["partition_vs_wall_rounding_ns"] = (
            fitted.predicted_partition_total_ns - fitted.predicted_observed_duration_ns
        )
        fitted.insert(1, "parameter_id", [f"{method}:p{index:05d}" for index in range(len(fitted))])
        parameters.append(fitted)
        for iteration in ITERATIONS:
            predicted = fitted.drop(columns=["calibration_samples", "calibration_iterations"]).copy()
            predicted.insert(2, "iteration", iteration)
            predicted.insert(
                0, "segment_prediction_id",
                [f"{method}:i{iteration}:s{index:05d}" for index in range(len(predicted))],
            )
            prediction_grid.append(predicted)
    parameter_frame = pd.concat(parameters, ignore_index=True)
    predictions = pd.concat(prediction_grid, ignore_index=True)
    if strict:
        assert parameter_frame.groupby("method").size().eq(9200).all()
        assert len(predictions) == len(METHODS) * len(ITERATIONS) * 9200
    assert predictions["segment_prediction_id"].is_unique
    assert not set(PARTITIONS).intersection(predictions.columns)
    assert {f"predicted_{column}" for column in PARTITIONS}.issubset(predictions.columns)
    return parameter_frame, predictions


def _physical_join(prepared: pd.DataFrame, observations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "iteration", "rank", "phase", "microbatch", "layer_id", "autograd_phase",
        "parameter_view", "execution_scope", "semantic_slot", "cost_group", "active_union_ns",
    }
    assert required.issubset(observations.columns)
    physical = observations[
        observations.parameter_view.eq("window_split")
        & observations.cost_group.eq("__physical_slot_total__")
        & observations.autograd_phase.eq("physical_mixed")
    ].copy()
    keys = ["iteration", "rank", "phase", "microbatch", "execution_scope", "semantic_slot"]
    left = prepared.drop(columns=["layer_id"]).rename(columns={"physical_layer_id": "layer_id"})
    joined = left.merge(
        physical[[*keys, "layer_id", "active_union_ns"]],
        on=[*keys, "layer_id"], how="outer", indicator=True, validate="one_to_one",
    )
    matched = joined._merge.eq("both")
    unmatched_segments = joined._merge.eq("left_only")
    assert not joined._merge.eq("right_only").any()
    assert joined.loc[matched, "active_union_ns"].astype("int64").eq(
        joined.loc[matched, "compute_active_union_ns"].astype("int64")
    ).all()
    assert joined.loc[unmatched_segments, "compute_active_union_ns"].eq(0).all()
    joined["physical_compute_observation_present"] = matched
    summary = {
        "segment_rows": int(len(prepared)),
        "physical_compute_rows": int(len(physical)),
        "one_to_one_exact_active_union_rows": int(matched.sum()),
        "zero_compute_segments_without_physical_row": int(unmatched_segments.sum()),
        "positive_compute_segment_mapping_coverage_pct": 100.0,
        "right_only_physical_rows": 0,
        "source_rank_count": int(physical["rank"].nunique()),
        "source_PP_lane_count": int(physical["pp_lane"].nunique()) if "pp_lane" in physical else 1,
    }
    keep = [
        "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch", "event_index",
        "segment_kind", "semantic_region", "execution_scope", "semantic_slot", "layer_id",
        "observed_duration_ns", "compute_active_union_ns", "active_union_ns",
        "physical_compute_observation_present", "split",
    ]
    return joined[keep].sort_values(keep[:7], kind="stable").reset_index(drop=True), summary


def _v54_parameter_audit(
    prepared: pd.DataFrame, gap_parameters: pd.DataFrame, communication_parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    keys = [
        "pp_stage", "pp_lane", "phase", "microbatch", "event_index",
        "v54_stage_local_layer", "semantic", "communication_kind",
    ]
    local = prepared.copy()
    local["semantic"] = np.where(
        local.semantic_region.eq("FINAL"), "final",
        local.semantic_region.str.split(":").str[-1],
    )
    gaps = local[local.segment_kind.eq("gap")]
    recomputed_gap = gaps.groupby(keys, as_index=False).agg(
        recomputed_wall_median_ns=("observed_duration_ns", _rounded_median),
        recomputed_iterations=("iteration", "nunique"),
    )
    formal_gap = gap_parameters.rename(columns={"stage_local_layer": "v54_stage_local_layer"})
    gap_join = formal_gap.merge(recomputed_gap, on=keys, how="inner", validate="one_to_one")
    gap_join["formal_minus_recomputed_ns"] = (
        gap_join.gap_before_ns_median - gap_join.recomputed_wall_median_ns
    )
    assert len(gap_join) == 4664
    assert gap_join.formal_minus_recomputed_ns.eq(0).all()
    assert gap_join.calibration_iterations.eq(gap_join.recomputed_iterations).all()

    communications = local[local.segment_kind.eq("communication")]
    recomputed_communication = communications.groupby(keys, as_index=False).agg(
        recomputed_observed_wall_median_ns=("observed_duration_ns", _rounded_median),
        recomputed_iterations=("iteration", "nunique"),
    )
    formal_communication = communication_parameters.rename(
        columns={"stage_local_layer": "v54_stage_local_layer"}
    )
    communication_join = formal_communication.merge(
        recomputed_communication, on=keys, how="inner", validate="one_to_one"
    )
    communication_join["formal_minus_observed_median_ns"] = (
        communication_join.predicted_duration_ns
        - communication_join.recomputed_observed_wall_median_ns
    )
    assert len(communication_join) == 4536
    assert communication_join.predicted_duration_ns.eq(
        communication_join.network_service_ns + communication_join.software_completion_ns
    ).all()
    assert communication_join.calibration_iterations.eq(
        communication_join.recomputed_iterations
    ).all()
    by_kind = {}
    for kind, group in communication_join.groupby("communication_kind"):
        delta = group.formal_minus_observed_median_ns
        by_kind[str(kind)] = {
            "rows": int(len(group)), "exact_observed_median_rows": int(delta.eq(0).sum()),
            "minimum_delta_ns": int(delta.min()), "maximum_delta_ns": int(delta.max()),
            "mean_delta_ns": float(delta.mean()),
        }
    summary = {
        "v54_gap_parameter_rows_total": int(len(gap_parameters)),
        "lane0_gap_parameters_recomputed": int(len(gap_join)),
        "lane0_gap_parameters_exact": int(gap_join.formal_minus_recomputed_ns.eq(0).sum()),
        "v54_gap_fit_iterations": sorted(gap_parameters.calibration_iterations.astype(int).unique().tolist()),
        "v54_communication_parameter_rows_total": int(len(communication_parameters)),
        "lane0_communication_parameters_recomputed": int(len(communication_join)),
        "communication_duration_is_network_plus_software": True,
        "formal_communication_minus_observed_median_by_kind": by_kind,
    }
    return gap_join, communication_join, summary


def _pooled_v54_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    group = [
        "pp_lane", "phase", "microbatch", "event_index", "stage_local_layer", "semantic",
    ]
    frames = []
    for stage in range(16):
        pool = gaps[gaps.pp_stage.eq(stage)] if stage in {0, 15} else gaps[gaps.pp_stage.between(1, 14)]
        fitted = pool.groupby(group, as_index=False).gap_before_ns_median.agg(_rounded_median)
        fitted["pp_stage"] = stage
        fitted["source_stage_policy"] = (
            "exact_boundary_stage" if stage in {0, 15} else "pooled_internal_stage_median"
        )
        frames.append(fitted)
    return pd.concat(frames, ignore_index=True)


def _graph_gap_lineage(
    gaps: pd.DataFrame, v67_nodes: pd.DataFrame, transfer: pd.DataFrame,
    shape_parameters: pd.DataFrame, v685_nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    v67 = v67_nodes[v67_nodes.kind.eq("local_gap")].copy()
    v67["event_index"] = v67.node_id.str.extract(r":r([0-9]+):")[0].astype(int) // 2
    v67["stage_local_layer"] = (
        v67.semantic_region.str.extract(r"before:L(-?[0-9]+):")[0].fillna(-1).astype(int)
    )
    v67["semantic"] = np.where(
        v67.semantic_region.eq("FINAL"), "final", v67.semantic_region.str.split(":").str[-1]
    )
    pooled = _pooled_v54_gaps(gaps)
    pool_keys = [
        "pp_stage", "pp_lane", "phase", "microbatch", "event_index",
        "stage_local_layer", "semantic",
    ]
    lineage = v67.merge(pooled, on=pool_keys, validate="many_to_one")
    transfer_keys = ["rank", "pp_stage", "pp_lane", "phase", "microbatch"]
    lineage = lineage.merge(
        transfer[[*transfer_keys, "effective_adjustable_shape_factor"]],
        on=transfer_keys, validate="many_to_one",
    )
    lineage["expected_v67_duration_ns"] = (
        lineage.gap_before_ns_median * lineage.effective_adjustable_shape_factor
    ).round().astype("int64")
    lineage["v67_rounding_delta_ns"] = (
        lineage.duration_ns - lineage.expected_v67_duration_ns
    )
    assert len(lineage) == 74624
    assert lineage.v67_rounding_delta_ns.abs().le(1).all()
    assert lineage.compute_work_ns.eq(0).all()
    assert lineage.unclassified_calibration_ns.eq(lineage.duration_ns).all()

    newer = v685_nodes[v685_nodes.kind.eq("local_gap")][
        ["node_id", "duration_ns", "unclassified_calibration_ns", "compute_work_ns"]
    ].rename(columns={
        "duration_ns": "v685_duration_ns",
        "unclassified_calibration_ns": "v685_unclassified_calibration_ns",
        "compute_work_ns": "v685_compute_work_ns",
    })
    lineage = lineage.merge(newer, on="node_id", validate="one_to_one")
    assert lineage.v685_duration_ns.eq(lineage.duration_ns).all()
    assert lineage.v685_unclassified_calibration_ns.eq(lineage.unclassified_calibration_ns).all()
    assert lineage.v685_compute_work_ns.eq(0).all()
    assert len(shape_parameters) == 2048
    assert shape_parameters.trace_samples.eq(9).all()
    assert set(shape_parameters.calibration_source) == {"source256_rank_trace_iterations_60_100"}
    summary = {
        "v67_local_gap_rows": int(len(lineage)),
        "v54_boundary_exact_then_internal_stage_pooling_rows": int(len(lineage)),
        "v67_shape_rounding_within_one_ns_rows": int(lineage.v67_rounding_delta_ns.abs().le(1).sum()),
        "v67_shape_parameter_rows": int(len(shape_parameters)),
        "v67_shape_samples_per_cell": 9,
        "v67_shape_fit_iterations": list(range(60, 101, 5)),
        "v67_local_gap_compute_work_ns": int(lineage.compute_work_ns.sum()),
        "v67_local_gap_unclassified_ns": int(lineage.unclassified_calibration_ns.sum()),
        "v685_gap_rows_exactly_equal_v67": int(lineage.v685_duration_ns.eq(lineage.duration_ns).sum()),
        "v685_local_gap_compute_work_ns": int(lineage.v685_compute_work_ns.sum()),
        "heldout_global_source_graph_available": False,
        "reason": "All 256-rank v67/v685 local gaps use v54 and v67 parameters fitted on source60-100; only lane0 has per-iteration physical decomposition.",
    }
    keep = [
        "node_id", "rank", "pp_stage", "pp_lane", "phase", "microbatch", "event_index",
        "stage_local_layer", "semantic", "source_stage_policy", "gap_before_ns_median",
        "effective_adjustable_shape_factor", "expected_v67_duration_ns", "duration_ns",
        "v67_rounding_delta_ns", "compute_work_ns", "unclassified_calibration_ns",
        "v685_duration_ns", "v685_compute_work_ns", "v685_unclassified_calibration_ns",
    ]
    return lineage[keep].sort_values(keep[:7], kind="stable").reset_index(drop=True), summary


def _score_segments(predictions: pd.DataFrame, prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_columns = ["iteration", *PARAMETER_KEYS, "observed_duration_ns", *PARTITIONS]
    truth = prepared[prepared.iteration.isin(ITERATIONS)][truth_columns]
    scored = predictions.merge(
        truth, on=["iteration", *PARAMETER_KEYS], how="left", validate="many_to_one", indicator=True,
        suffixes=("_predicted", "_observed"),
    )
    scored["truth_available"] = scored._merge.eq("both")
    scored["error_ns"] = scored.predicted_partition_total_ns - scored.observed_duration_ns
    scored["absolute_error_ns"] = scored.error_ns.abs()
    scored["split"] = np.where(
        scored.iteration.isin(FIT), "source_fit", "source_incremental_validation"
    )
    rows = []
    available = scored[scored.truth_available]
    for (method, split, phase, kind), group in available.groupby(
        ["method", "split", "phase", "segment_kind"], sort=True
    ):
        rows.append({
            "method": method, "split": split, "phase": phase, "segment_kind": kind,
            "observations": len(group), "MAE_ms": group.absolute_error_ns.mean() / 1e6,
            "bias_ms": group.error_ns.mean() / 1e6,
            "WAPE_pct": 100.0 * group.absolute_error_ns.sum() / group.observed_duration_ns.sum(),
        })
    for (method, split), group in available.groupby(["method", "split"], sort=True):
        rows.append({
            "method": method, "split": split, "phase": "all", "segment_kind": "all",
            "observations": len(group), "MAE_ms": group.absolute_error_ns.mean() / 1e6,
            "bias_ms": group.error_ns.mean() / 1e6,
            "WAPE_pct": 100.0 * group.absolute_error_ns.sum() / group.observed_duration_ns.sum(),
        })
    metrics = pd.DataFrame(rows)
    return scored.drop(columns="_merge"), metrics


def _phase_validation(
    scored: pd.DataFrame, pp_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    operation = ["iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch", "method"]
    expected = scored.groupby(operation, as_index=False).size().rename(columns={"size": "predicted_segments"})
    available = scored[scored.truth_available].groupby(operation, as_index=False).agg(
        truth_segments=("segment_prediction_id", "size"),
        predicted_chain_envelope_ns=("predicted_partition_total_ns", "sum"),
        observed_chain_envelope_ns=("observed_duration_ns", "sum"),
    )
    phase = expected.merge(available, on=operation, how="left", validate="one_to_one")
    phase["complete_truth"] = phase.predicted_segments.eq(phase.truth_segments)
    source_pp = pp_events[
        pp_events.iteration.isin(ITERATIONS) & pp_events.pp_lane.eq(0)
    ][["iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch", "duration_ns"]]
    phase = phase.merge(
        source_pp, on=operation[:-1], how="left", validate="many_to_one"
    ).rename(columns={"duration_ns": "PP_annotation_wall_ns"})
    phase["post_annotation_rank_local_completion_ns"] = (
        phase.observed_chain_envelope_ns - phase.PP_annotation_wall_ns
    )
    phase["chain_error_ns"] = (
        phase.predicted_chain_envelope_ns - phase.observed_chain_envelope_ns
    )
    phase["annotation_error_ns"] = (
        phase.predicted_chain_envelope_ns - phase.PP_annotation_wall_ns
    )
    phase["split"] = np.where(phase.iteration.isin(FIT), "source_fit", "source_incremental_validation")
    complete = phase[phase.complete_truth]
    assert complete.post_annotation_rank_local_completion_ns.ge(0).all()
    rows = []
    for (method, split, phase_name), group in complete.groupby(["method", "split", "phase"], sort=True):
        rows.append({
            "method": method, "split": split, "phase": phase_name,
            "complete_operations": len(group),
            "chain_envelope_MAE_ms": group.chain_error_ns.abs().mean() / 1e6,
            "chain_envelope_bias_ms": group.chain_error_ns.mean() / 1e6,
            "PP_annotation_MAE_ms": group.annotation_error_ns.abs().mean() / 1e6,
            "PP_annotation_bias_ms": group.annotation_error_ns.mean() / 1e6,
            "post_annotation_completion_mean_ms": group.post_annotation_rank_local_completion_ns.mean() / 1e6,
            "post_annotation_completion_max_ms": group.post_annotation_rank_local_completion_ns.max() / 1e6,
            "post_annotation_completion_positive": int(group.post_annotation_rank_local_completion_ns.gt(0).sum()),
        })
    metrics = pd.DataFrame(rows)
    spill = complete.drop_duplicates(operation[:-1])
    summary = {
        "complete_lane0_phase_operations": int(len(spill)),
        "missing_phase_operations": int(16 * 2 * 4 * len(ITERATIONS) - len(spill)),
        "forward_post_annotation_completion_max_ms": float(
            spill.loc[spill.phase.eq("forward"), "post_annotation_rank_local_completion_ns"].max() / 1e6
        ),
        "backward_post_annotation_completion_mean_ms": float(
            spill.loc[spill.phase.eq("backward"), "post_annotation_rank_local_completion_ns"].mean() / 1e6
        ),
        "backward_post_annotation_completion_max_ms": float(
            spill.loc[spill.phase.eq("backward"), "post_annotation_rank_local_completion_ns"].max() / 1e6
        ),
        "backward_positive_spill_operations": int(
            spill.loc[spill.phase.eq("backward"), "post_annotation_rank_local_completion_ns"].gt(0).sum()
        ),
        "clock_rule": "rank-local chain ends at max(PP annotation end, last communication end)",
    }
    return phase, metrics, summary


def _draw(out: Path, prepared: pd.DataFrame, phase_metrics: pd.DataFrame) -> None:
    import os
    os.environ["MPLCONFIGDIR"] = str(out / "mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "w37-t32a-source-gap-bridge"
    import matplotlib.pyplot as plt

    validation = prepared[prepared.iteration.isin(VALIDATION)]
    totals = validation.groupby("segment_kind")[PARTITIONS].sum() / 1e9
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    bottom = np.zeros(len(totals))
    colors = ["#2a9d8f", "#457b9d", "#f4a261", "#8d99ae"]
    labels = ["compute only", "communication only", "compute/comm overlap", "non-GPU / unresolved"]
    for column, label, color in zip(PARTITIONS, labels, colors):
        values = totals[column].to_numpy()
        axes[0].bar(totals.index, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0].set_ylabel("source95/100 accumulated wall (s), lane0 ranks")
    axes[0].set_title("Disjoint occupancy accounting")
    axes[0].legend(fontsize=8)

    local = phase_metrics[
        phase_metrics.split.eq("source_incremental_validation")
    ]
    positions = np.arange(2)
    for index, method in enumerate(METHODS):
        values = local[local.method.eq(method)].set_index("phase").reindex(["forward", "backward"])
        axes[1].bar(positions + (index - .5) * .35, values.chain_envelope_MAE_ms, .34, label=method)
    axes[1].set_xticks(positions, ["forward", "backward"])
    axes[1].set_ylabel("source95/100 local chain MAE (ms)")
    axes[1].set_title("Fit on source85/90; 16 representative ranks")
    axes[1].legend(fontsize=8)
    fig.suptitle("T32A source local-gap ownership and compute bridge")
    fig.text(
        .5, .015,
        "Partitions conserve each observed interval. v67/v685 all-rank gaps use exposed 60-100 fits, so no held-out global 1F1B claim.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out / "source_gap_bridge.svg", metadata={"Date": None})
    fig.savefig(out / "source_gap_bridge.png", dpi=150)
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    begin = perf_counter()
    assert plan["diagnostic"] == "source_gap_bridge_audit"
    assert plan["diagnostic_access"] == "source_only" and not plan["variants"]
    review = json.loads((ROOT / plan["resource_review_file"]).read_text())
    assert review["fit_iterations"] == FIT and review["incremental_validation"] == VALIDATION
    for item in review["builder_evidence"]:
        path = Path(item["path"]) if str(item["path"]).startswith("/") else ROOT / item["path"]
        assert sha(path) == item["sha256"]

    raw_segments = pd.read_csv(paths["source_trace_segment_components.csv"])
    layer_map = pd.read_csv(paths["source_layer_stage_map.csv"])
    prepared = prepare_segments(raw_segments, layer_map)
    parameters, predictions = fit_partition_candidates(prepared)

    mutated = prepared.copy()
    validation_mask = mutated.iteration.isin(VALIDATION)
    mutated.loc[validation_mask, ["observed_duration_ns", *PARTITIONS]] = -999999999
    mutated_parameters, mutated_predictions = fit_partition_candidates(mutated)
    pd.testing.assert_frame_equal(parameters, mutated_parameters, check_exact=True)
    pd.testing.assert_frame_equal(predictions, mutated_predictions, check_exact=True)

    csv(out, "source_gap_partition_parameters.csv", parameters)
    csv(out, "source_gap_partition_predictions_sealed.csv.gz", predictions)
    seal_files = ["source_gap_partition_parameters.csv", "source_gap_partition_predictions_sealed.csv.gz"]
    dump(out / "source_gap_partition_prediction_seal.json", {
        "status": "SEALED_SOURCE85_90_DISJOINT_INTERVAL_PARTITION_PREDICTION",
        "fit_iterations": FIT, "incremental_validation": VALIDATION,
        "eligible_methods": list(METHODS), "target_parameter_updates": 0,
        "validation_truth_attached_before_seal": False,
        "files": [{"path": name, "sha256": sha(out / name)} for name in seal_files],
        "mutation_gate": "Changing every source95/100 wall and partition truth leaves parameters and the complete prediction grid exact.",
    })

    scored, segment_metrics = _score_segments(predictions, prepared)
    pp_events = pd.read_csv(paths["source_pp_trace_events_60_100.csv"])
    phase, phase_metrics, phase_summary = _phase_validation(scored, pp_events)
    physical_join, physical_summary = _physical_join(
        prepared, pd.read_csv(paths["operator_cost_group_observations.csv.gz"])
    )
    gap_audit, communication_audit, v54_summary = _v54_parameter_audit(
        prepared,
        pd.read_csv(paths["source_rank_local_gap_parameters.csv"]),
        pd.read_csv(paths["source_rank_local_communication_parameters.csv"]),
    )
    v67_nodes = pd.read_csv(paths["dag_v67_source_nodes.csv.gz"], low_memory=False)
    v685_nodes = pd.read_csv(paths["formal_v685_source_nodes.csv.gz"], low_memory=False)
    lineage, graph_summary = _graph_gap_lineage(
        pd.read_csv(paths["source_rank_local_gap_parameters.csv"]),
        v67_nodes,
        pd.read_csv(paths["source_runtime_shape_transfer.csv"]),
        pd.read_csv(paths["source_rank_microbatch_runtime_shape_parameters.csv"]),
        v685_nodes,
    )

    result_columns = [
        "segment_prediction_id", "method", "iteration", *PARAMETER_KEYS,
        *[f"predicted_{column}" for column in PARTITIONS], "predicted_partition_total_ns",
        "truth_available", "observed_duration_ns", *PARTITIONS, "error_ns", "absolute_error_ns", "split",
    ]
    csv(out, "source_gap_partition_validation.csv.gz", scored[result_columns])
    csv(out, "source_gap_partition_metrics.csv", segment_metrics)
    csv(out, "source_phase_envelope_validation.csv", phase)
    csv(out, "source_phase_envelope_metrics.csv", phase_metrics)
    csv(out, "physical_compute_segment_join.csv.gz", physical_join)
    csv(out, "v54_lane0_gap_parameter_audit.csv", gap_audit)
    csv(out, "v54_lane0_communication_parameter_audit.csv", communication_audit)
    csv(out, "v67_v685_gap_lineage.csv.gz", lineage)

    selected = prepared[prepared.iteration.isin(ITERATIONS)].copy()
    selected_columns = [
        "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch", "event_index",
        "v54_stage_local_layer", "physical_layer_id", "segment_kind", "semantic_region",
        "execution_scope", "semantic_slot", "communication_kind", "observed_duration_ns",
        "compute_active_union_ns", "communication_active_union_ns",
        "compute_communication_overlap_ns", *PARTITIONS, "split",
    ]
    csv(out, "source_interval_accounting_ledger.csv.gz", selected[selected_columns])

    component_totals = selected.groupby(["split", "segment_kind"])[
        ["observed_duration_ns", *PARTITIONS]
    ].sum().reset_index()
    csv(out, "source_interval_component_totals.csv", component_totals)
    accounting = pd.DataFrame([
        {
            "scope": "gap interval", "start": "operation start or prior rank-local communication end",
            "end": "next rank-local communication arrival",
            "owned_wall": "one observed interval",
            "disjoint_accounting": "compute_only + communication_only + compute_communication_overlap + non_GPU_unresolved",
            "double_count_rule": "serialize the wall once; partitions explain occupancy and are not extra nodes",
        },
        {
            "scope": "communication interval", "start": "rank-local communication arrival",
            "end": "rank-local communication completion",
            "owned_wall": "one observed interval",
            "disjoint_accounting": "compute_only + communication_only + compute_communication_overlap + non_GPU_unresolved",
            "double_count_rule": "compute overlap lies inside this wall and cannot be added after communication",
        },
        {
            "scope": "v54 communication model", "start": "rank-local arrival",
            "end": "modeled completion",
            "owned_wall": "network_service + software_completion",
            "disjoint_accounting": "model components conserve predicted_duration",
            "double_count_rule": "arrival span is generated by dependencies; do not add it to service or completion",
        },
        {
            "scope": "PP annotation", "start": "F/B annotation start", "end": "F/B annotation end",
            "owned_wall": "annotation wall",
            "disjoint_accounting": "backward local chain may finish after annotation end",
            "double_count_rule": "post-annotation local completion must be reconciled with PP send ownership before graph insertion",
        },
        {
            "scope": "v67/v685 local_gap", "start": "relative max-plus graph clock",
            "end": "relative start plus pooled and shaped duration",
            "owned_wall": "unclassified_calibration_ns",
            "disjoint_accounting": "not resolved in the formal all-rank source graph",
            "double_count_rule": "do not relabel the whole duration as compute, framework, or wait",
        },
    ])
    csv(out, "component_ownership_contract.csv", accounting)

    clocks = pd.read_csv(paths["source_stable_clocks.csv"])
    v67_contract = json.loads(paths["v67_source_replay_contract.json"].read_text())
    v685_contract = json.loads(paths["prediction_contract.json"].read_text())
    all9_median_ms = float(clocks.profiler_step_ns.median() / 1e6)
    steady_median_ms = float(clocks[clocks.iteration.isin(ITERATIONS)].profiler_step_ns.median() / 1e6)
    assert math.isclose(all9_median_ms, float(v67_contract["source_profiler_median_ms"]), abs_tol=1e-9)
    assert math.isclose(
        steady_median_ms,
        float(v685_contract["prediction"]["source_profiler_reference_median_ms"]), abs_tol=1e-9,
    )
    clock_audit = {
        "absolute_trace_clock": "v55 segment start/end use profiler baseTimeNanoseconds absolute timestamps",
        "operation_clock": phase_summary,
        "graph_clock": "v67/v685 max-plus nodes start at relative zero and close at a named completion node",
        "v67": {
            "profiler_reference": "median source60-100",
            "source_profiler_median_ms": all9_median_ms,
            "source_raw_graph_ms": float(v67_contract["v67_raw_graph_ms"]),
            "source_reconciliation_ms": float(v67_contract["source_reconciliation_ms"]),
        },
        "v685": {
            "profiler_reference": "median source85/90/95/100",
            "source_profiler_median_ms": steady_median_ms,
            "source_raw_graph_ms": float(v685_contract["prediction"]["source_raw_graph_ms"]),
            "source_reconciliation_ms": float(v685_contract["prediction"]["source_reconciliation_ms"]),
        },
        "same_local_gap_values_v67_to_v685": True,
        "reconciliation_comparability": "The reference window changes and v685 keeps v67 local gaps; neither reconciliation is held-out compute validation.",
    }
    dump(out / "clock_origin_audit.json", clock_audit)
    dump(out / "gap_bridge_summary.json", {
        "status": "LOCAL_ACCOUNTING_VALIDATED_GLOBAL_SOURCE_BRIDGE_REJECTED",
        "physical_compute_join": physical_summary,
        "v54_parameter_origin": v54_summary,
        "v67_v685_graph_lineage": graph_summary,
        "phase_envelope": phase_summary,
    })
    _draw(out, prepared, phase_metrics)

    for item in json.loads((out / "source_gap_partition_prediction_seal.json").read_text())["files"]:
        assert sha(out / item["path"]) == item["sha256"]
    validation_metrics = phase_metrics[phase_metrics.split.eq("source_incremental_validation")]
    metric_lookup = {
        (str(row.method), str(row.phase)): row
        for row in validation_metrics.itertuples(index=False)
    }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= review["resource"]["maximum_peak_RSS_bytes"]
    assert elapsed <= review["resource"]["target_analysis_seconds"]
    dump(out / "field_contract.json", {
        "local_interval_partition": "compute-only, communication-only, their overlap, and non-GPU/unresolved are mutually exclusive and exactly conserve wall time.",
        "physical_compute": "v6.1 physical slot totals match every positive-compute v5.5 segment exactly; coverage is lane0 at 16 stages.",
        "v54_gap": "Exact per-rank median of the rank-local interval, then internal stages are pooled in the v5.4 source replay.",
        "v67_gap": "Pooled v5.4 gap redistributed across microbatches with parameters fitted on all source60-100.",
        "v685_gap": "Byte-value equivalent to v6.7 local gaps; compute_work remains zero and the wall remains unclassified.",
        "validation": "Source95/100 was historically inspected and is development validation, not blind validation.",
        "global_boundary": "Only 16 lane0 ranks have per-iteration internal decomposition; all-rank graph parameters include 95/100, so no held-out global 1F1B prediction is emitted.",
        "target": "No target timing, target fit, topology update, Step result, or MFU result in T32A.",
    })
    dump(out / "diagnostic.json", {
        "status": "SOURCE_GAP_BRIDGE_ACCOUNTING_PASS_GLOBAL_PROMOTION_REJECTED",
        "new_prediction": True, "new_global_prediction": False,
        "used_to_fit_model": True, "new_target_timing_read": False,
        "raw_trace_scanned": False, "source_fit_iterations": FIT,
        "source_incremental_validation": VALIDATION,
        "representative_rank_count": 16, "represented_PP_stages": 16,
        "represented_PP_lanes": 1,
        "positive_compute_segment_mapping_coverage_pct": 100.0,
        "v67_v685_local_gap_rows": 74624,
        "v67_v685_local_gap_values_exact": True,
        "v67_v685_local_gap_compute_work_ns": 0,
        "v67_v685_local_gap_unclassified_ns": graph_summary["v67_local_gap_unclassified_ns"],
        "median85_90_forward_phase_MAE_ms": float(metric_lookup[("median_85_90", "forward")].chain_envelope_MAE_ms),
        "median85_90_backward_phase_MAE_ms": float(metric_lookup[("median_85_90", "backward")].chain_envelope_MAE_ms),
        "recent90_forward_phase_MAE_ms": float(metric_lookup[("recent_90", "forward")].chain_envelope_MAE_ms),
        "recent90_backward_phase_MAE_ms": float(metric_lookup[("recent_90", "backward")].chain_envelope_MAE_ms),
        "source_global_1F1B_candidate_available": False,
        "target_candidate_registered": False,
        "promotion": "REJECT_TARGET_TRANSFER_NO_HELDOUT_ALL_RANK_SOURCE_GRAPH",
        "formal_topology_changed": False, "target_parameter_updates": 0,
        "prediction_seal_sha256": sha(out / "source_gap_partition_prediction_seal.json"),
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed,
        "next": "Build an all-rank source85/90 phase-envelope replay on the locked schedule and validate source95/100 globally before considering any target transfer; retain the lane0 disjoint ledger only as explanatory evidence.",
    })

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .stage_hardware import counter_stream_overlap_pct


class ValidationError(RuntimeError):
    pass


def _require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise ValidationError(f"missing or empty required output: {path}")
    return path


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_m1_outputs(
    root: Path,
    expected_profiled_iters: tuple[int, ...],
    world_size: int,
) -> dict[str, int | str | float]:
    """Fail-closed semantic validation for the v5 M1 direct-observation outputs."""
    expected_iters = list(expected_profiled_iters)
    expected_trace_count = len(expected_iters) * world_size

    manifest = pd.read_csv(_require(root / "profiler_trace_manifest_v5.csv"))
    _assert(len(manifest) == expected_trace_count, "profiler trace count mismatch")
    _assert(sorted(manifest["iter"].unique().tolist()) == expected_iters, "profiled iter domain mismatch")
    for iter_id, group in manifest.groupby("iter"):
        _assert(
            sorted(group["rank"].tolist()) == list(range(world_size)),
            f"profiler rank coverage mismatch for iter {iter_id}",
        )
    _assert(100 not in manifest["iter"].values, "iter100 leaked into profiler manifest")

    event_parts = list(root.glob("events_profiler/iter=*/rank=*.parquet"))
    pg_parts = list(root.glob("pg_config_profiler/iter=*/rank=*.parquet"))
    _assert(len(event_parts) == expected_trace_count, "profiler event partition count mismatch")
    _assert(len(pg_parts) == expected_trace_count, "pg_config partition count mismatch")
    _assert(not any("iter=100" in str(path) for path in event_parts + pg_parts), "iter100 partition exists")

    calls = pd.read_parquet(_require(root / "collective_calls_v5.parquet"))
    per_rank = pd.read_parquet(_require(root / "collective_call_per_rank_v5.parquet"))
    _assert(100 not in calls["iter"].values, "iter100 leaked into collective calls")
    _assert(calls.groupby("iter").size().nunique() == 1, "collective call count differs by iter")
    _assert(int(calls.groupby("iter").size().iloc[0]) == 35, "expected 35 collective calls per profiled iter")
    _assert(len(calls) == 35 * len(expected_iters), "collective call total mismatch")
    _assert(len(per_rank) == 5_376, "collective per-rank event count mismatch")
    _assert(set(calls["algo"].dropna()) == {"RING"}, "unexpected MCCL algorithm")
    _assert(set(calls["protocol"].dropna()) == {"SIMPLE"}, "unexpected MCCL protocol")

    failure_path = root / "call_alignment_failures_v5.csv"
    if failure_path.exists() and failure_path.stat().st_size > 1:
        try:
            failures = pd.read_csv(failure_path)
        except pd.errors.EmptyDataError:
            failures = pd.DataFrame()
        _assert(failures.empty, "collective call alignment failures are non-empty")

    global_windows = pd.read_csv(_require(root / "iter_windows_profiled_v5.csv"))
    global_atomic = pd.read_parquet(_require(root / "phase_intervals_profiled_v5.parquet"))
    global_share = pd.read_csv(_require(root / "iter_phase_share_profiled_v5.csv"))
    _assert(sorted(global_windows["iter"].tolist()) == expected_iters, "global phase iter coverage mismatch")
    global_closure = global_atomic.groupby("iter")["duration_ns"].sum().rename("atomic_ns").to_frame().join(
        global_windows.set_index("iter")["wall_ns"]
    )
    _assert(
        int((global_closure["atomic_ns"] - global_closure["wall_ns"]).abs().max()) == 0,
        "global atomic phase intervals do not close to wall",
    )
    _assert(global_share.groupby("iter").size().eq(11).all(), "global stage row count mismatch")
    _assert(global_share["critical_exposure_pct"].between(0, 100).all(), "invalid global exposure pct")

    rank_windows = pd.read_csv(_require(root / "iter_windows_profiled_per_rank_v5.csv"))
    rank_atomic = pd.read_parquet(_require(root / "phase_intervals_profiled_per_rank_v5.parquet"))
    rank_share = pd.read_csv(_require(root / "iter_phase_share_profiled_per_rank_v5.csv"))
    _assert(len(rank_windows) == expected_trace_count, "rank phase window count mismatch")
    rank_closure = (
        rank_atomic.groupby(["iter", "rank"])["duration_ns"]
        .sum()
        .rename("atomic_ns")
        .to_frame()
        .join(rank_windows.set_index(["iter", "rank"])["wall_ns"])
    )
    _assert(
        int((rank_closure["atomic_ns"] - rank_closure["wall_ns"]).abs().max()) == 0,
        "rank atomic phase intervals do not close to wall",
    )
    _assert(
        rank_share.groupby(["iter", "rank"]).size().eq(11).all(),
        "rank stage row count mismatch",
    )

    deep = pd.read_parquet(_require(root / "events_deepep_v5.parquet"))
    deep_windows = pd.read_csv(_require(root / "iter_windows_deepep_v5.csv"))
    _assert(deep_windows["iter"].tolist() == list(range(1, 100)), "DeepEP iter domain mismatch")
    _assert(sorted(deep["rank"].unique().tolist()) == list(range(world_size)), "DeepEP rank domain mismatch")
    _assert(100 not in deep["iter"].values, "iter100 leaked into DeepEP events")

    nic_manifest = pd.read_csv(_require(root / "nic_corpus_manifest_v5.csv"))
    mt_manifest = pd.read_csv(_require(root / "mtlink_corpus_manifest_v5.csv"))
    _assert(len(nic_manifest) == 8, "NIC device count mismatch")
    _assert(len(mt_manifest) == 16, "MTLink device count mismatch")
    _assert((nic_manifest["row_count"] > 0).all(), "empty NIC device output")
    _assert((mt_manifest["row_count"] > 0).all(), "empty MTLink device output")

    tiny = calls[np.logical_and(calls["collective"] == "allreduce", calls["size_bytes"] <= 8)]
    first_default = tiny[
        np.logical_and(tiny["pg_description"] == "default_pg", tiny["occurrence_index"] == 0)
    ]
    _assert(len(first_default) == len(expected_iters), "default tiny AllReduce call coverage mismatch")
    tiny_arrival_p50_ms = float(first_default["arrival_skew_ns"].median() / 1e6)
    tiny_service_p50_ms = float(first_default["service_after_last_arrival_ns"].median() / 1e6)
    _assert(tiny_arrival_p50_ms > 100, "tiny AllReduce arrival skew anchor not reproduced")
    _assert(tiny_service_p50_ms < 1, "tiny AllReduce service anchor not reproduced")

    edp = calls[calls["pg_description"] == "EXPERT_DATA_PARALLEL_GROUP"]
    _assert(edp["expected_ranks"].nunique() == 8, "EDP pair count mismatch")

    return {
        "status": "PASS",
        "profiler_trace_count": len(manifest),
        "collective_call_count": len(calls),
        "collective_per_rank_event_count": len(per_rank),
        "global_phase_iter_count": len(global_windows),
        "rank_phase_window_count": len(rank_windows),
        "deepep_iter_count": len(deep_windows),
        "nic_device_count": len(nic_manifest),
        "mtlink_device_count": len(mt_manifest),
        "tiny_default_arrival_p50_ms": tiny_arrival_p50_ms,
        "tiny_default_service_p50_ms": tiny_service_p50_ms,
    }


def validate_stage_hardware_fusion(root: Path) -> dict[str, object]:
    """Fail-closed validation for the 24-iteration profiler/hardware fusion."""
    expected_iters = list(range(4, 100, 4))
    intervals = pd.read_parquet(_require(root / "stage_hardware_rank_intervals_profiled_v5.parquet"))
    allocated = pd.read_parquet(_require(root / "stage_hardware_allocated_segments_profiled_v5.parquet"))
    per_entity = pd.read_parquet(_require(root / "stage_hardware_per_iter_entity_profiled_v5.parquet"))
    summary = pd.read_csv(_require(root / "stage_hardware_summary_profiled_v5.csv"))
    scenario = pd.read_csv(_require(root / "stage_hardware_top_level_scenario_profiled_v5.csv"))
    sensitivity = pd.read_csv(_require(root / "stage_hardware_sensitivity_profiled_v5.csv"))
    robust_sensitivity = pd.read_csv(_require(root / "stage_hardware_sensitivity_no_deepep_timing_v6.csv"))
    overlap_sensitivity = pd.read_csv(_require(root / "stage_hardware_overlap_sensitivity_profiled_v5.csv"))

    _assert(sorted(intervals["iter"].unique().tolist()) == expected_iters, "fusion interval iter domain mismatch")
    _assert(len(intervals) == 19_200, "fusion exact rank-stage interval count mismatch")
    _assert(100 not in per_entity["iter"].values, "iter100 leaked into stage/hardware fusion")
    _assert(per_entity["iter"].nunique() == 24, "fusion per-entity iter coverage mismatch")
    _assert(per_entity["coverage_pct"].median() > 99.9, "fusion counter coverage median is too low")
    _assert(len(sensitivity) == 72, "fusion sensitivity must contain 3 scenarios x 24 iters")
    _assert(
        set(sensitivity["scenario_label"])
        == {"lower_target", "central_target", "aggressive_conditional"},
        "fusion sensitivity scenario domain mismatch",
    )
    _assert(len(robust_sensitivity) == 72, "robust fusion sensitivity must contain 3 scenarios x 24 iters")
    _assert(
        set(robust_sensitivity["scenario_label"])
        == {"lower_target", "central_target", "aggressive_conditional"},
        "robust fusion sensitivity scenario domain mismatch",
    )
    _assert(robust_sensitivity["excluded_stage"].eq("deepep_comm").all(), "robust sensitivity excluded stage mismatch")
    _assert(len(overlap_sensitivity) == 72, "fusion overlap sensitivity must contain 3 assignments x 24 iters")
    _assert(
        set(overlap_sensitivity["overlap_assignment"])
        == {"excluded", "all_dp_edp_ag_overlap_to_dp_ag", "all_dp_edp_ag_overlap_to_edp_ag"},
        "fusion overlap sensitivity assignment domain mismatch",
    )

    for hardware in ["nic", "mtlink"]:
        segments = pd.read_parquet(_require(root / f"stage_hardware_{hardware}_atomic_segments_profiled_v5.parquet"))
        for (iter_id, host, device), group in segments.groupby(["iter", "host", "device"], sort=False):
            ordered = group.sort_values(["start_ns", "end_ns"])
            starts = ordered["start_ns"].to_numpy(dtype=np.int64)
            ends = ordered["end_ns"].to_numpy(dtype=np.int64)
            _assert(
                bool(np.all(starts[1:] >= ends[:-1])),
                f"atomic hardware segments overlap for {hardware}/{iter_id}/{host}/{device}",
            )

    nic_stream_overlap: list[float] = []
    for path in root.glob("counter_nic/host=*/device=*/part-*.parquet"):
        nic_stream_overlap.append(counter_stream_overlap_pct(pd.read_parquet(path)))
    mtlink_stream_overlap: list[float] = []
    for path in root.glob("counter_mtlink/host=*/device=*/part-*.parquet"):
        mtlink_stream_overlap.append(
            counter_stream_overlap_pct(pd.read_parquet(path), stream_columns=["link_id"])
        )
    _assert(nic_stream_overlap and max(nic_stream_overlap) < .01, "NIC within-stream overlap exceeds 0.01%")
    _assert(mtlink_stream_overlap and max(mtlink_stream_overlap) < 1.0, "MTLink within-link overlap exceeds 1%")

    nic_manifest = pd.read_csv(_require(root / "nic_corpus_manifest_v5.csv"))
    mt_manifest = pd.read_csv(_require(root / "mtlink_corpus_manifest_v5.csv"))
    for hardware, manifest in [("nic", nic_manifest), ("mtlink", mt_manifest)]:
        rows = allocated[allocated["hardware"] == hardware]
        allocated_tx = float((rows["allocated_tx_bytes"] + rows["uncertain_tx_bytes"]).sum())
        allocated_rx = float((rows["allocated_rx_bytes"] + rows["uncertain_rx_bytes"]).sum())
        _assert(allocated_tx <= float(manifest["tx_bytes"].sum()), f"{hardware} TX allocation exceeds corpus bytes")
        _assert(allocated_rx <= float(manifest["rx_bytes"].sum()), f"{hardware} RX allocation exceeds corpus bytes")

    def metric(stage: str, count: int) -> pd.Series:
        rows = summary[
            (summary["hardware"] == "nic")
            & (summary["stage"] == stage)
            & (summary["active_rank_count"] == count)
        ]
        _assert(len(rows) == 1, f"missing unique NIC summary for {stage}/count={count}")
        return rows.iloc[0]

    edp_rs = metric("edp_rs_scaleout", 2)
    edp_ag = metric("edp_ag_scaleout", 2)
    _assert(210 < edp_rs["tx_rate_median"] < 225, "EDP RS two-rank NIC anchor not reproduced")
    _assert(370 < edp_ag["tx_rate_median"] < 380, "EDP AG two-rank NIC anchor not reproduced")
    _assert(scenario["iter"].tolist() == expected_iters, "fusion scenario iter domain mismatch")
    uplift = float(scenario["throughput_and_relative_mfu_uplift_pct"].median())
    sensitivity_medians = (
        sensitivity.groupby("scenario_label")["throughput_and_relative_mfu_uplift_pct"]
        .median()
        .to_dict()
    )
    robust_sensitivity_medians = (
        robust_sensitivity.groupby("scenario_label")["throughput_and_relative_mfu_uplift_pct"]
        .median()
        .to_dict()
    )
    _assert(10 < uplift < 15, "fusion top-level uplift outside audited range")
    _assert(
        sensitivity_medians["lower_target"]
        <= sensitivity_medians["central_target"]
        <= sensitivity_medians["aggressive_conditional"],
        "fusion sensitivity is not monotonic",
    )
    _assert(
        robust_sensitivity_medians["lower_target"]
        <= robust_sensitivity_medians["central_target"]
        <= robust_sensitivity_medians["aggressive_conditional"],
        "robust fusion sensitivity is not monotonic",
    )
    _assert(
        5 < robust_sensitivity_medians["lower_target"]
        and robust_sensitivity_medians["aggressive_conditional"] < 12,
        "robust fusion sensitivity outside audited range",
    )
    _assert(
        all(robust_sensitivity_medians[key] < sensitivity_medians[key] for key in robust_sensitivity_medians),
        "excluding DeepEP timing did not reduce every scenario",
    )
    overlap_sensitivity_medians = (
        overlap_sensitivity.groupby("overlap_assignment")["throughput_and_relative_mfu_uplift_pct"]
        .median()
        .to_dict()
    )
    overlap_sensitivity_span = max(overlap_sensitivity_medians.values()) - min(overlap_sensitivity_medians.values())
    _assert(overlap_sensitivity_span < 1.0, "fusion overlap assignment changes median uplift by >=1 point")

    nic = allocated[allocated["hardware"] == "nic"]
    total_nic_bytes = (
        nic["allocated_tx_bytes"] + nic["allocated_rx_bytes"]
        + nic["uncertain_tx_bytes"] + nic["uncertain_rx_bytes"]
    ).sum()
    uncertain_pct = 100.0 * (nic["uncertain_tx_bytes"] + nic["uncertain_rx_bytes"]).sum() / total_nic_bytes
    ambiguous = nic["stage"] == "multi_stage_overlap"
    ambiguous_pct = 100.0 * (
        nic.loc[ambiguous, "allocated_tx_bytes"] + nic.loc[ambiguous, "allocated_rx_bytes"]
    ).sum() / total_nic_bytes
    _assert(uncertain_pct < .1, "fusion NIC uncertain byte share too high")
    _assert(ambiguous_pct < 2.5, "fusion NIC unresolved-overlap byte share too high")

    return {
        "status": "PASS",
        "profiled_iter_count": len(expected_iters),
        "exact_rank_stage_interval_count": len(intervals),
        "per_iter_entity_rows": len(per_entity),
        "edp_rs_two_rank_tx_gbps_median": float(edp_rs["tx_rate_median"]),
        "edp_ag_two_rank_tx_gbps_median": float(edp_ag["tx_rate_median"]),
        "legacy_full_scenario_uplift_pct_median": uplift,
        "legacy_full_sensitivity_uplift_pct_medians": {
            key: float(value) for key, value in sensitivity_medians.items()
        },
        "robust_no_deepep_timing_sensitivity_uplift_pct_medians": {
            key: float(value) for key, value in robust_sensitivity_medians.items()
        },
        "overlap_assignment_uplift_pct_medians": {
            key: float(value) for key, value in overlap_sensitivity_medians.items()
        },
        "overlap_assignment_uplift_pct_span": float(overlap_sensitivity_span),
        "nic_counter_stream_overlap_pct_max": float(max(nic_stream_overlap)),
        "mtlink_counter_stream_overlap_pct_max": float(max(mtlink_stream_overlap)),
        "nic_uncertain_bytes_pct": float(uncertain_pct),
        "nic_unresolved_overlap_bytes_pct": float(ambiguous_pct),
    }


def validate_v5_outputs(root: Path) -> dict[str, object]:
    """Fail-closed validation for the complete 99-iter v5 analysis."""
    expected_iters = list(range(1, 100))
    cycle = pd.read_csv(_require(root / "cycle_windows_all99_v5.csv"))
    share = pd.read_csv(_require(root / "iter_phase_share_all99_v5.csv"))
    atomic = pd.read_parquet(_require(root / "phase_intervals_all99_v5.parquet"))
    summary = pd.read_csv(_require(root / "iter_summary_all99_v5.csv"))
    headroom = pd.read_csv(_require(root / "stage_headroom_all99_v5.csv"))
    joint = pd.read_csv(_require(root / "joint_headroom_profiled_v5.csv"))
    outliers = pd.read_csv(_require(root / "iter_outlier_analysis_v5.csv"))

    _assert(cycle["iter"].tolist() == expected_iters, "cycle iter domain mismatch")
    _assert(summary["iter"].tolist() == expected_iters, "all-iter summary domain mismatch")
    _assert(100 not in share["iter"].values, "iter100 leaked into phase share")
    _assert(len(cycle) == 99, "expected 99 cycle windows")
    _assert(len(share) == 1_188, "expected 12 phase-share rows per iter")
    _assert(share.groupby("iter").size().eq(12).all(), "phase-share row count differs by iter")
    required_stages = {
        "deepep_forward_scaleup",
        "deepep_backward_scaleup",
        "dp_ag_hybrid",
        "dp_rs_hybrid",
        "edp_ag_scaleout",
        "edp_rs_scaleout",
        "gradient_finalize_sync",
        "other_communication",
        "tiny_sync_collective",
        "all_communication",
        "compute_active",
        "idle_or_unattributed",
    }
    _assert(set(share["stage"]) == required_stages, "all-iter stage domain mismatch")
    closure = atomic.groupby("iter")["duration_ns"].sum().rename("atomic_ns").to_frame().join(
        cycle.set_index("iter")["wall_ns"]
    )
    _assert(
        int((closure["atomic_ns"] - closure["wall_ns"]).abs().max()) == 0,
        "all-iter atomic intervals do not close to cycle wall",
    )
    _assert(share["critical_exposure_pct"].between(0, 100).all(), "invalid all-iter phase pct")

    inferred_rows = share[share["provenance"] == "inferred_validated_template"]
    direct_rows = share[share["provenance"] == "direct_profiled_collective"]
    _assert(inferred_rows["iter"].nunique() == 75, "inferred iter count mismatch")
    _assert(direct_rows["iter"].nunique() == 24, "profiled direct iter count mismatch")
    _assert(inferred_rows["exclusive_ns"].isna().all(), "inferred exclusive wall must remain unknown")
    _assert(inferred_rows["overlap_with_compute_ns"].isna().all(), "inferred compute overlap must remain unknown")

    gate = json.loads(_require(root / "inference_gate_v5.json").read_text())
    _assert(bool(gate.get("passed")), "inference gate did not pass")
    _assert(gate["metrics"]["boundary_p95_ms"] <= 50, "inference boundary p95 exceeds gate")
    _assert(gate["metrics"]["nic_bytes_error_median_pct"] <= 5, "inference NIC-byte error exceeds gate")

    _assert(len(headroom) == 891, "expected 9 headroom rows per iter")
    _assert(headroom.groupby("iter").size().eq(9).all(), "headroom stage count differs by iter")
    nonprofiled = headroom[~headroom["iter"].isin(range(4, 100, 4))]
    _assert(
        nonprofiled["wall_exposed_headroom_conservative_ns"].isna().all(),
        "nonprofiled wall-exposed headroom must remain unknown",
    )
    _assert(len(joint) == 24, "joint counterfactual must cover 24 profiled iters")
    _assert((joint["joint_counterfactual_wall_conservative_ns"] > 0).all(), "invalid counterfactual wall")

    confidence = pd.read_csv(_require(root / "inference_stage_confidence_v5.csv"))
    major = confidence[confidence["stage"].isin(["dp_ag_hybrid", "dp_rs_hybrid", "edp_ag_scaleout", "edp_rs_scaleout"])]
    _assert(set(major["inference_confidence"]) == {"high"}, "major inferred stages lack high confidence")

    nic = pd.read_parquet(_require(root / "nic_per_iter_device_v5.parquet"))
    mtlink = pd.read_parquet(_require(root / "mtlink_per_iter_link_v5.parquet"))
    _assert(nic["iter"].nunique() == 99 and len(nic) == 792, "NIC all-iter coverage mismatch")
    _assert(mtlink["iter"].nunique() == 99 and len(mtlink) == 22_176, "MTLink all-iter coverage mismatch")

    required_spatial = [
        "rank_collective_duration_entity_summary_v5.csv",
        "edp_pair_service_entity_summary_v5.csv",
        "nic_device_full_window_rate_entity_summary_v5.csv",
        "mtlink_link_full_window_rate_entity_summary_v5.csv",
        "deepep_rank_stage_duration_entity_summary_v5.csv",
    ]
    for filename in required_spatial:
        _require(root / filename)

    outlier_iters = sorted(outliers.loc[outliers["wall_outlier"].astype(bool), "iter"].astype(int).tolist())
    _assert(24 in outlier_iters and 88 in outlier_iters, "iter24/88 wall outliers not reproduced")

    return {
        "status": "PASS",
        "iter_count": len(cycle),
        "phase_share_rows": len(share),
        "stage_headroom_rows": len(headroom),
        "profiled_iter_count": direct_rows["iter"].nunique(),
        "inferred_iter_count": inferred_rows["iter"].nunique(),
        "inference_gate_passed": bool(gate["passed"]),
        "inference_boundary_p95_ms": float(gate["metrics"]["boundary_p95_ms"]),
        "inference_nic_bytes_error_median_pct": float(gate["metrics"]["nic_bytes_error_median_pct"]),
        "outlier_iters": outlier_iters,
    }

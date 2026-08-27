#!/usr/bin/env python3
"""Convert nine representative OISA FCTs into MFU DAG backfill inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_SHARES = {
    "ep_fwd_dispatch": {"forward": 0.5, "backward": 0.5},
    "ep_fwd_combine": {"forward": 0.5, "backward": 0.5},
    "ep_bwd_combine_backward_dispatch": {"backward": 1.0},
    "ep_bwd_dispatch_backward_combine": {"backward": 1.0},
    "cp_all_to_all": {"forward": 1.0 / 3.0, "backward": 2.0 / 3.0},
}
GROUP_COLUMN = {
    "ep_fwd_dispatch": "ep_group",
    "ep_fwd_combine": "ep_group",
    "ep_bwd_combine_backward_dispatch": "ep_group",
    "ep_bwd_dispatch_backward_combine": "ep_group",
    "cp_all_to_all": "cp_group",
}
PAYLOAD_COLUMN = {behavior: "expected_tx_bytes" for behavior in PIPELINE_SHARES}
OPTIMIZER_KIND_BEHAVIOR = {
    "dp_rs": "dp_grad_reduce_scatter",
    "dp_ag0": "dp_param_allgather",
    "dp_ag1": "dp_param_allgather",
    "edp_rs": "expert_dp_grad_reduce_scatter",
    "edp_ag0": "expert_dp_param_allgather",
    "edp_ag1": "expert_dp_param_allgather",
}


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    result_root = case_root / "results/oisa_s5000_256gpu_nine_class"
    fabric_case = Path("/home/zjb/Desktop/fabric-data-analysis/case_256gpu_pp16_cp2_a2a")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=55)
    parser.add_argument(
        "--events",
        type=Path,
        default=fabric_case
        / "results/collective_bw_no_pp/collective_events_no_pp_pre_mtlink.parquet",
    )
    parser.add_argument(
        "--rank-topology",
        type=Path,
        default=fabric_case / "results/topology/rank_topology.csv",
    )
    parser.add_argument(
        "--trace-events", type=Path, default=case_root / "data/pp_dag_trace_events.csv"
    )
    parser.add_argument(
        "--selection", type=Path, default=result_root / "inputs/representative_calls.csv"
    )
    parser.add_argument(
        "--oisa-results", type=Path, default=result_root / "oisa_results.csv"
    )
    parser.add_argument("--output", type=Path, default=result_root / "backfill")
    return parser.parse_args()


def _allocate_target(original: np.ndarray, target_total: int) -> np.ndarray:
    if target_total < len(original):
        raise ValueError("target phase duration cannot keep every compute node positive")
    raw = original.astype(float) * target_total / int(original.sum())
    allocated = np.floor(raw).astype("int64")
    allocated = np.maximum(allocated, 1)
    difference = int(target_total - allocated.sum())
    if difference > 0:
        order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        for index in order[:difference]:
            allocated[index] += 1
    elif difference < 0:
        order = np.argsort(raw - np.floor(raw), kind="stable")
        remaining = -difference
        for index in order:
            removable = min(int(allocated[index] - 1), remaining)
            allocated[index] -= removable
            remaining -= removable
            if remaining == 0:
                break
        if remaining:
            raise ValueError("failed to allocate positive compute durations")
    if int(allocated.sum()) != target_total or (allocated <= 0).any():
        raise ValueError("compute-duration allocation is not conservative")
    return allocated


def _build_node_adjustments(
    cells: pd.DataFrame,
    trace: pd.DataFrame,
    *,
    iteration: int,
    delta_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase_rows: list[dict[str, object]] = []
    for row in cells.itertuples(index=False):
        for phase, share in PIPELINE_SHARES[str(row.behavior)].items():
            phase_rows.append({
                "rank": int(row.rank),
                "pp_stage": int(row.pp_stage),
                "phase": phase,
                "behavior": str(row.behavior),
                "behavior_delta_ns": int(
                    round(int(getattr(row, delta_column)) * share)
                ),
            })
    phase_behavior = pd.DataFrame(phase_rows)
    phase_delta = (
        phase_behavior.groupby(["rank", "pp_stage", "phase"], as_index=False)[
            "behavior_delta_ns"
        ]
        .sum()
        .rename(columns={"behavior_delta_ns": "phase_adjustment_ns"})
    )

    node_rows: list[dict[str, object]] = []
    for (rank, phase), frame in trace.groupby(["rank", "phase"], sort=True):
        frame = frame.sort_values("microbatch")
        match = phase_delta[
            phase_delta["rank"].eq(rank) & phase_delta["phase"].eq(phase)
        ]
        adjustment = int(match["phase_adjustment_ns"].iloc[0]) if len(match) else 0
        original = frame["duration_ns"].astype("int64").to_numpy()
        target = int(original.sum()) + adjustment
        allocated = _allocate_target(original, target)
        for source, duration in zip(frame.itertuples(index=False), allocated):
            node_rows.append({
                "iteration": iteration,
                "rank": int(rank),
                "pp_stage": int(source.pp_stage),
                "pp_lane": int(source.pp_lane),
                "phase": str(phase),
                "microbatch": int(source.microbatch),
                "trace_duration_ns": int(source.duration_ns),
                "adjusted_duration_ns": int(duration),
                "adjustment_ns": int(duration) - int(source.duration_ns),
            })
    node_adjustments = pd.DataFrame(node_rows)
    conserved = node_adjustments.groupby(
        ["rank", "pp_stage", "phase"], as_index=False
    )["adjustment_ns"].sum().merge(
        phase_delta, on=["rank", "pp_stage", "phase"], how="left"
    )
    conserved["phase_adjustment_ns"] = conserved["phase_adjustment_ns"].fillna(0)
    if not np.array_equal(
        conserved["adjustment_ns"].to_numpy(),
        conserved["phase_adjustment_ns"].astype("int64").to_numpy(),
    ):
        raise ValueError("pipeline node adjustment does not conserve phase delta")
    return phase_behavior, node_adjustments, conserved


def main() -> int:
    args = parse_args()
    selection = pd.read_csv(args.selection)
    oisa = pd.read_csv(args.oisa_results)
    if set(selection["behavior"]) != set(oisa["kind"]) or len(oisa) != 9:
        raise ValueError("nine-class selection/OISA coverage is incomplete")
    calibration = selection.merge(
        oisa,
        left_on=["behavior", "request_id"],
        right_on=["kind", "request_id"],
        validate="one_to_one",
        suffixes=("_trace", "_oisa"),
    )
    calibration["reference_payload_bytes"] = calibration["payload_bytes"]
    calibration["service_ratio_oisa_to_trace"] = (
        calibration["tail_after_last_release_ns"]
        / calibration["observed_service_ns"]
    )
    calibration["trace_calibration_residual_ns"] = (
        calibration["observed_service_ns"]
        - calibration["tail_after_last_release_ns"]
    ).astype("int64")
    calibration["software_residual_ns"] = calibration[
        "trace_calibration_residual_ns"
    ].clip(lower=0).astype("int64")
    calibration["simulator_bias_correction_ns"] = calibration[
        "trace_calibration_residual_ns"
    ].clip(upper=0).astype("int64")
    calibration["calibrated_service_ns"] = (
        calibration["tail_after_last_release_ns"]
        + calibration["trace_calibration_residual_ns"]
    )

    columns = [
        "behavior", "iteration", "rank", "group_size", "start_ns", "end_ns",
        "duration_ns", "logical_input_bytes", "expected_tx_bytes",
    ]
    events = pd.read_parquet(args.events, columns=columns)
    events = events[
        events["iteration"].eq(args.iteration)
        & events["behavior"].isin(PIPELINE_SHARES)
    ].copy()
    topology = pd.read_csv(
        args.rank_topology,
        usecols=["rank", "pp_stage", "cp_group", "ep_group"],
    )
    events = events.merge(topology, on="rank", validate="many_to_one")
    events = events.sort_values(["behavior", "rank", "start_ns", "end_ns"])
    events["slot_index"] = events.groupby(["behavior", "rank"]).cumcount()
    reference_for = calibration.set_index("behavior").to_dict(orient="index")

    call_frames: list[pd.DataFrame] = []
    for behavior in PIPELINE_SHARES:
        selected = events[events["behavior"].eq(behavior)].copy()
        selected["sync_group"] = selected[GROUP_COLUMN[behavior]].astype("int64")
        keys = ["behavior", "sync_group", "slot_index"]
        calls = (
            selected.groupby(keys, as_index=False)
            .agg(
                participant_count=("rank", "nunique"),
                group_size=("group_size", "max"),
                ready_ns=("start_ns", "max"),
                group_end_ns=("end_ns", "max"),
                call_payload_bytes=(PAYLOAD_COLUMN[behavior], "median"),
            )
        )
        calls["aligned"] = calls["participant_count"].eq(calls["group_size"])
        reference = reference_for[behavior]
        calls["predicted_network_service_ns"] = np.rint(
            int(reference["tail_after_last_release_ns"])
            * calls["call_payload_bytes"]
            / int(reference["reference_payload_bytes"])
        ).astype("int64")
        calls.loc[~calls["aligned"], "predicted_network_service_ns"] = -1
        call_frames.append(selected.merge(calls, on=keys, validate="many_to_one"))
    detailed = pd.concat(call_frames, ignore_index=True)
    detailed["arrival_wait_ns"] = detailed["ready_ns"] - detailed["start_ns"]
    detailed["completion_advance_ns"] = detailed["group_end_ns"] - detailed["end_ns"]
    detailed["trace_group_service_ns"] = (
        detailed["group_end_ns"] - detailed["ready_ns"]
    )
    detailed["trace_calibration_residual_ns"] = (
        detailed["trace_group_service_ns"]
        - detailed["predicted_network_service_ns"]
    )
    detailed["software_residual_ns"] = detailed[
        "trace_calibration_residual_ns"
    ].clip(lower=0)
    detailed["simulator_bias_correction_ns"] = detailed[
        "trace_calibration_residual_ns"
    ].clip(upper=0)
    detailed["predicted_calibrated_service_ns"] = (
        detailed["predicted_network_service_ns"]
        + detailed["trace_calibration_residual_ns"]
    )
    predicted_network_only = (
        detailed["arrival_wait_ns"]
        + detailed["predicted_network_service_ns"]
        - detailed["completion_advance_ns"]
    ).clip(lower=1)
    predicted_calibrated = (
        detailed["arrival_wait_ns"]
        + detailed["predicted_calibrated_service_ns"]
        - detailed["completion_advance_ns"]
    ).clip(lower=1)
    detailed["network_only_predicted_duration_ns"] = np.where(
        detailed["aligned"], predicted_network_only, detailed["duration_ns"]
    ).astype("int64")
    detailed["calibrated_predicted_duration_ns"] = np.where(
        detailed["aligned"], predicted_calibrated, detailed["duration_ns"]
    ).astype("int64")
    detailed["network_only_duration_delta_ns"] = (
        detailed["network_only_predicted_duration_ns"] - detailed["duration_ns"]
    )
    detailed["calibrated_duration_delta_ns"] = (
        detailed["calibrated_predicted_duration_ns"] - detailed["duration_ns"]
    )

    cells = (
        detailed.groupby(["behavior", "rank", "pp_stage"], as_index=False)
        .agg(
            observed_calls=("slot_index", "size"),
            aligned_calls=("aligned", "sum"),
            trace_duration_ns=("duration_ns", "sum"),
            network_only_predicted_duration_ns=(
                "network_only_predicted_duration_ns", "sum"
            ),
            calibrated_predicted_duration_ns=(
                "calibrated_predicted_duration_ns", "sum"
            ),
            network_only_duration_delta_ns=(
                "network_only_duration_delta_ns", "sum"
            ),
            calibrated_duration_delta_ns=(
                "calibrated_duration_delta_ns", "sum"
            ),
        )
    )
    trace = pd.read_csv(args.trace_events)
    trace = trace[trace["iteration"].eq(args.iteration)].copy()
    phase_behavior_network_only, node_adjustments_network_only, conserved_network = (
        _build_node_adjustments(
            cells,
            trace,
            iteration=args.iteration,
            delta_column="network_only_duration_delta_ns",
        )
    )
    phase_behavior, node_adjustments, conserved = _build_node_adjustments(
        cells,
        trace,
        iteration=args.iteration,
        delta_column="calibrated_duration_delta_ns",
    )

    optimizer_rows = []
    for kind, behavior in OPTIMIZER_KIND_BEHAVIOR.items():
        record = reference_for[behavior]
        optimizer_rows.append({
            "kind": kind,
            "behavior": behavior,
            "request_id": record["request_id"],
            "reference_payload_bytes": int(record["reference_payload_bytes"]),
            "tail_after_last_release_ns": int(record["tail_after_last_release_ns"]),
            "baseline_tail_after_last_release_ns": int(
                record["tail_after_last_release_ns"]
            ),
            "representative_trace_service_ns": int(record["observed_service_ns"]),
            "representative_software_residual_ns": int(
                record["software_residual_ns"]
            ),
            "representative_simulator_bias_correction_ns": int(
                record["simulator_bias_correction_ns"]
            ),
            "representative_trace_calibration_residual_ns": int(
                record["trace_calibration_residual_ns"]
            ),
            "simulator_commit": record["simulator_commit"],
            "topology_hash": record["topology_hash"],
            "scaling_rule": "tail_ns * request_payload / reference_payload",
            "residual_rule": (
                "trace request tail - baseline OISA network tail (signed)"
            ),
        })
    optimizer_calibration = pd.DataFrame(optimizer_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(args.output / "collective_service_calibration.csv", index=False)
    cells.to_csv(args.output / "pipeline_framework_backfill_cells.csv", index=False)
    phase_behavior.to_csv(args.output / "pipeline_phase_behavior_delta.csv", index=False)
    phase_behavior_network_only.to_csv(
        args.output / "pipeline_phase_behavior_delta_network_only.csv", index=False
    )
    node_adjustments.to_csv(args.output / "pipeline_node_adjustments.csv", index=False)
    node_adjustments_network_only.to_csv(
        args.output / "pipeline_node_adjustments_network_only.csv", index=False
    )
    optimizer_calibration.to_csv(
        args.output / "optimizer_kind_calibration.csv", index=False
    )
    validation = {
        "schema": "mfu-oisa-nine-class-backfill-v1",
        "status": "PASS",
        "iteration": args.iteration,
        "collective_classes": len(calibration),
        "pipeline_classes": len(PIPELINE_SHARES),
        "optimizer_kinds": len(optimizer_calibration),
        "pipeline_event_alignment_fraction": float(detailed["aligned"].mean()),
        "pipeline_adjustment_conservation_error_ns_max": int(
            (conserved["adjustment_ns"] - conserved["phase_adjustment_ns"]).abs().max()
        ),
        "pipeline_adjustment_ns_total": int(node_adjustments["adjustment_ns"].sum()),
        "pipeline_network_only_adjustment_ns_total": int(
            node_adjustments_network_only["adjustment_ns"].sum()
        ),
        "pipeline_network_only_adjustment_conservation_error_ns_max": int(
            (
                conserved_network["adjustment_ns"]
                - conserved_network["phase_adjustment_ns"]
            ).abs().max()
        ),
        "representative_software_residual_ns_total": int(
            calibration["software_residual_ns"].sum()
        ),
        "representative_simulator_bias_correction_ns_total": int(
            calibration["simulator_bias_correction_ns"].sum()
        ),
        "service_rule": (
            "preserve measured rank arrival wait/completion advance and the signed "
            "Trace-minus-baseline-OISA calibration residual; only the target-minus-"
            "baseline OISA network delta changes the calibrated service"
        ),
        "fallback": "unaligned Trace calls retain their measured duration",
    }
    (args.output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

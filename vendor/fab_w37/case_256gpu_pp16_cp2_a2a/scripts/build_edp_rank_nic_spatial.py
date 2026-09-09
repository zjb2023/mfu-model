#!/usr/bin/env python3
"""Build rank-specific Expert-DP NIC spatial data for the 256-GPU case.

The 256-GPU corpus has no full per-rank dashboard set.  This script uses the
explicit carry-over GPU-slot mapping documented by build_rank_nic_mapping.py
to integrate raw NIC delta-byte samples over each rank's own Expert-DP event
intervals.

The result is window-aligned correlation.  It is not proof that every byte in
an interval was caused exclusively by the labeled collective.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_edp_nic_spatial import (
    EDP_BEHAVIORS,
    ITERATIONS,
    attribute_device,
    intersection_duration,
    merge_intervals,
)


RANKS = tuple(range(256))


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    case_root = Path(__file__).resolve().parents[1]
    collective_dir = case_root / "results" / "collective_bw_no_pp"
    dashboard_dir = case_root / "results" / "dashboard_reference"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events-parquet",
        type=Path,
        default=collective_dir / "collective_events_no_pp_pre_mtlink.parquet",
    )
    parser.add_argument(
        "--hardware-root",
        type=Path,
        default=case_root / ".raw_view",
    )
    parser.add_argument(
        "--nic-manifest",
        type=Path,
        default=case_root / "results" / "readiness" / "nic_files.csv",
    )
    parser.add_argument(
        "--event-windows",
        type=Path,
        default=case_root / "results" / "readiness" / "event_windows.csv",
    )
    parser.add_argument(
        "--rank-topology",
        type=Path,
        default=case_root / "results" / "topology" / "rank_topology.csv",
    )
    parser.add_argument(
        "--dashboard-mapping",
        type=Path,
        default=dashboard_dir / "dashboard_rank_nic_mapping.csv",
    )
    parser.add_argument(
        "--dashboard-stage-summary",
        type=Path,
        default=dashboard_dir / "dashboard_iteration5_nic_stage_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=collective_dir / "edp_nic_iteration_rank_device.csv",
    )
    parser.add_argument(
        "--per-nic-summary-output",
        type=Path,
        default=collective_dir / "edp_nic_per_nic_summary.csv",
    )
    parser.add_argument(
        "--device-slot-summary-output",
        type=Path,
        default=collective_dir / "edp_nic_device_slot_summary.csv",
    )
    parser.add_argument(
        "--hotspot-output",
        type=Path,
        default=collective_dir / "edp_nic_hotspot_cells.csv",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=collective_dir / "edp_nic_rank_device_validation.json",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--nic-long-gap-ms", type=float, default=10.0)
    parser.add_argument("--hotspots-per-behavior", type=int, default=25)
    return parser.parse_args()


def portable_path(path: Path) -> str:
    repository_root = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root))
    except ValueError:
        return str(resolved)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_mapping(
    mapping_path: Path,
    topology_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(mapping_path)
    topology = pd.read_csv(topology_path)
    manifest = pd.read_csv(manifest_path)
    expected_mapping_columns = {
        "rank",
        "host",
        "local_rank",
        "gpu_id",
        "nic_device",
        "expected_nic_device",
        "mapping_status",
        "dashboard_profiler_step",
        "nic_metric",
        "raw_nic_point_match_count",
        "nic_point_count",
        "raw_nic_max_abs_value_difference",
    }
    missing = expected_mapping_columns - set(mapping.columns)
    if missing:
        raise ValueError(f"dashboard mapping missing columns: {sorted(missing)}")
    topology_columns = ["rank", "host", "local_rank", "gpu_id"]
    identity = mapping.merge(
        topology[topology_columns],
        on="rank",
        suffixes=("_dashboard", "_topology"),
        validate="one_to_one",
    )
    identity_mismatch = (
        identity["host_dashboard"].ne(identity["host_topology"])
        | identity["local_rank_dashboard"].ne(
            identity["local_rank_topology"]
        )
        | identity["gpu_id_dashboard"].ne(identity["gpu_id_topology"])
    )
    manifest_keys = set(
        zip(manifest["host"].astype(str), manifest["device"].astype(str))
    )
    mapping_keys = set(
        zip(mapping["host"].astype(str), mapping["nic_device"].astype(str))
    )
    checks = [
        len(mapping) == len(RANKS),
        set(mapping["rank"]) == set(RANKS),
        mapping["rank"].nunique() == len(RANKS),
        mapping["host"].nunique() == 32,
        mapping.groupby("host")["rank"].nunique().eq(8).all(),
        mapping.groupby("host")["nic_device"].nunique().eq(8).all(),
        mapping["mapping_status"].eq("CARRYOVER_VALIDATED_SLOT_MAP").all(),
        mapping["nic_device"].eq(mapping["expected_nic_device"]).all(),
        not identity_mismatch.any(),
        mapping_keys == manifest_keys,
    ]
    if not all(checks):
        raise ValueError("dashboard rank/NIC mapping validation failed")
    return mapping.sort_values("rank").reset_index(drop=True), manifest


def build_cells(
    events_path: Path,
    event_windows_path: Path,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, list[tuple[int, int, int]]], dict]:
    columns = [
        "behavior",
        "iteration",
        "rank",
        "host",
        "start_ns",
        "end_ns",
    ]
    events = pd.read_parquet(events_path, columns=columns)
    behavior_names = {str(item["behavior"]) for item in EDP_BEHAVIORS}
    edp = events[events["behavior"].isin(behavior_names)].copy()
    other = events[~events["behavior"].isin(behavior_names)].copy()
    windows = pd.read_csv(event_windows_path)
    window_map = windows.set_index(["iteration", "rank"]).to_dict("index")
    if (
        len(windows) != len(ITERATIONS) * len(RANKS)
        or set(windows["iteration"]) != set(ITERATIONS)
        or set(windows["rank"]) != set(RANKS)
        or not windows["nic_devices_covering"].eq(8).all()
    ):
        raise ValueError("rank ProfilerStep/NIC coverage grid is incomplete")

    edp_groups = {
        (str(behavior), int(iteration), int(rank)): group
        for (behavior, iteration, rank), group in edp.groupby(
            ["behavior", "iteration", "rank"]
        )
    }
    other_groups = {
        (int(iteration), int(rank)): merge_intervals(
            zip(group["start_ns"].astype(int), group["end_ns"].astype(int))
        )
        for (iteration, rank), group in other.groupby(["iteration", "rank"])
    }
    mapping_by_rank = mapping.set_index("rank").to_dict("index")
    targets_by_rank: dict[int, list[tuple[int, int, int]]] = {
        rank: [] for rank in RANKS
    }
    rows: list[dict[str, Any]] = []
    for behavior_item in EDP_BEHAVIORS:
        behavior = str(behavior_item["behavior"])
        expected = int(behavior_item["expected_events_per_rank"])
        for iteration in ITERATIONS:
            for rank in RANKS:
                cell_id = len(rows)
                identity = mapping_by_rank[rank]
                group = edp_groups.get((behavior, iteration, rank))
                event_count = 0 if group is None else int(len(group))
                if event_count == 0:
                    intervals: list[tuple[int, int]] = []
                    status = "missing_source_event"
                else:
                    intervals = merge_intervals(
                        zip(
                            group["start_ns"].astype(int),
                            group["end_ns"].astype(int),
                        )
                    )
                    status = (
                        "observed"
                        if event_count == expected
                        else "partial_source_event"
                    )
                window = window_map[(iteration, rank)]
                union_duration_ns = sum(end - start for start, end in intervals)
                if intervals and any(
                    start < int(window["start_ns"])
                    or end > int(window["end_ns"])
                    for start, end in intervals
                ):
                    raise ValueError("Expert-DP event outside rank ProfilerStep")
                other_overlap_ns = intersection_duration(
                    intervals, other_groups.get((iteration, rank), [])
                )
                rows.append(
                    {
                        "cell_id": cell_id,
                        "behavior_order": int(
                            behavior_item["behavior_order"]
                        ),
                        "behavior_family": behavior_item["behavior_family"],
                        "behavior": behavior,
                        "behavior_label": behavior_item["behavior_label"],
                        "behavior_description": behavior_item[
                            "behavior_description"
                        ],
                        "iteration": iteration,
                        "rank": rank,
                        "node_index": rank // 8,
                        "host": str(identity["host"]),
                        "local_rank": int(identity["local_rank"]),
                        "gpu_id": int(identity["gpu_id"]),
                        "device": str(identity["nic_device"]),
                        "nic_entity_id": (
                            f"{identity['host']} / r{rank} / "
                            f"GPU{identity['gpu_id']} / {identity['nic_device']}"
                        ),
                        "dashboard_mapping_status": str(
                            identity["mapping_status"]
                        ),
                        "dashboard_mapping_source": str(
                            identity["relative_path"]
                        ),
                        "event_count": event_count,
                        "expected_event_count": expected,
                        "event_count_delta": event_count - expected,
                        "data_status": status,
                        "edp_union_segment_count": len(intervals),
                        "edp_window_start_ns": (
                            min(start for start, _ in intervals)
                            if intervals
                            else pd.NA
                        ),
                        "edp_window_end_ns": (
                            max(end for _, end in intervals)
                            if intervals
                            else pd.NA
                        ),
                        "edp_union_duration_ns": (
                            union_duration_ns if intervals else pd.NA
                        ),
                        "profiler_step_start_ns": int(window["start_ns"]),
                        "profiler_step_end_ns": int(window["end_ns"]),
                        "profiler_step_duration_ns": int(
                            window["duration_ns"]
                        ),
                        "other_collective_overlap_ns": other_overlap_ns,
                        "other_collective_overlap_pct": (
                            other_overlap_ns / union_duration_ns * 100.0
                            if union_duration_ns
                            else pd.NA
                        ),
                    }
                )
                for start, end in intervals:
                    targets_by_rank[rank].append((start, end, cell_id))

    cross_cell_overlap_ns = 0
    for rank, targets in targets_by_rank.items():
        targets.sort()
        for left, right in zip(targets, targets[1:]):
            cross_cell_overlap_ns += max(0, left[1] - right[0])
        if any(end <= start for start, end, _ in targets):
            raise ValueError(f"invalid target interval for rank {rank}")
    if cross_cell_overlap_ns:
        raise ValueError(
            "rank Expert-DP target windows overlap; attribution would "
            "duplicate NIC bytes"
        )
    return (
        pd.DataFrame(rows),
        targets_by_rank,
        {
            "edp_source_event_count": int(len(edp)),
            "other_collective_source_event_count": int(len(other)),
            "target_interval_count": int(
                sum(len(value) for value in targets_by_rank.values())
            ),
            "cross_cell_overlap_ns": int(cross_cell_overlap_ns),
        },
    )


def add_hotspot_fields(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    metric = "nic_active_window_bidir_bandwidth_GBps"
    groups = frame.groupby(["behavior", "iteration"])[metric]
    frame["nic_bidir_iteration_median_GBps"] = groups.transform("median")
    frame["nic_bidir_vs_iteration_median_ratio"] = (
        frame[metric]
        / frame["nic_bidir_iteration_median_GBps"].replace(0, np.nan)
    )
    frame["nic_bidir_percentile_in_iteration"] = groups.rank(
        method="average", pct=True
    )
    frame["nic_bidir_hotspot_p95"] = (
        frame["nic_bidir_percentile_in_iteration"].ge(0.95)
        & frame[metric].notna()
    )
    frame["nic_tx_share_pct"] = (
        frame["nic_attributed_tx_bytes"]
        / frame["nic_attributed_bidir_bytes"].replace(0, np.nan)
        * 100.0
    )
    frame["nic_tx_rx_imbalance_pct"] = (
        (
            frame["nic_attributed_tx_bytes"]
            - frame["nic_attributed_rx_bytes"]
        ).abs()
        / frame["nic_attributed_bidir_bytes"].replace(0, np.nan)
        * 100.0
    )
    return frame


def build_summaries(
    frame: pd.DataFrame, hotspots_per_behavior: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric = "nic_active_window_bidir_bandwidth_Gbps"
    keys = [
        "behavior_order",
        "behavior",
        "behavior_label",
        "rank",
        "node_index",
        "host",
        "local_rank",
        "gpu_id",
        "device",
        "nic_entity_id",
    ]
    summary = (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(
            observed_iteration_count=(metric, "count"),
            missing_iteration_count=(
                "data_status",
                lambda values: int((values == "missing_source_event").sum()),
            ),
            partial_iteration_count=(
                "data_status",
                lambda values: int((values == "partial_source_event").sum()),
            ),
            nic_bidir_bandwidth_median_Gbps=(metric, "median"),
            nic_bidir_bandwidth_p95_Gbps=(
                metric,
                lambda values: values.quantile(0.95),
            ),
            nic_bidir_bandwidth_max_Gbps=(metric, "max"),
            nic_hotspot_p95_iteration_count=("nic_bidir_hotspot_p95", "sum"),
            nic_min_window_coverage_pct=("nic_window_coverage_pct", "min"),
            nic_max_long_gap_bidir_pct=("nic_long_gap_bidir_pct", "max"),
            nic_max_other_collective_overlap_pct=(
                "other_collective_overlap_pct",
                "max",
            ),
            nic_tx_share_median_pct=("nic_tx_share_pct", "median"),
            nic_tx_rx_imbalance_median_pct=(
                "nic_tx_rx_imbalance_pct",
                "median",
            ),
        )
    )
    max_rows = (
        frame.dropna(subset=[metric])
        .sort_values(
            keys + [metric, "iteration"],
            ascending=[True] * len(keys) + [False, True],
        )
        .drop_duplicates(keys)
        [keys + ["iteration"]]
        .rename(columns={"iteration": "nic_bidir_bandwidth_max_iteration"})
    )
    summary = summary.merge(max_rows, on=keys, validate="one_to_one")

    slot_keys = ["behavior_order", "behavior", "behavior_label", "device"]
    slot = (
        frame.groupby(slot_keys, as_index=False)
        .agg(
            physical_nic_entity_count=("nic_entity_id", "nunique"),
            valid_cell_count=(metric, "count"),
            nic_bidir_bandwidth_median_Gbps=(metric, "median"),
            nic_bidir_bandwidth_p95_Gbps=(
                metric,
                lambda values: values.quantile(0.95),
            ),
            nic_bidir_bandwidth_max_Gbps=(metric, "max"),
            nic_hotspot_p95_cell_count=("nic_bidir_hotspot_p95", "sum"),
            nic_window_coverage_min_pct=("nic_window_coverage_pct", "min"),
            nic_long_gap_bidir_p95_pct=(
                "nic_long_gap_bidir_pct",
                lambda values: values.quantile(0.95),
            ),
            nic_tx_share_median_pct=("nic_tx_share_pct", "median"),
        )
    )
    slot_mean = frame.groupby(slot_keys)[metric].transform("mean")
    slot_std = frame.groupby(slot_keys)[metric].transform("std")
    cv = (
        frame.assign(_mean=slot_mean, _std=slot_std)
        .groupby(slot_keys, as_index=False)
        .agg(_mean=("_mean", "first"), _std=("_std", "first"))
    )
    cv["nic_bidir_bandwidth_cv"] = cv["_std"] / cv["_mean"].replace(
        0, np.nan
    )
    slot = slot.merge(
        cv[slot_keys + ["nic_bidir_bandwidth_cv"]],
        on=slot_keys,
        validate="one_to_one",
    )

    hotspot_columns = [
        "behavior_order",
        "behavior",
        "behavior_label",
        "iteration",
        "rank",
        "host",
        "local_rank",
        "gpu_id",
        "device",
        "nic_entity_id",
        "data_status",
        metric,
        "nic_active_window_tx_bandwidth_Gbps",
        "nic_active_window_rx_bandwidth_Gbps",
        "nic_bidir_vs_iteration_median_ratio",
        "nic_bidir_percentile_in_iteration",
        "nic_window_coverage_pct",
        "nic_long_gap_bidir_pct",
        "other_collective_overlap_pct",
    ]
    hotspot = (
        frame.dropna(subset=[metric])
        .sort_values(
            ["behavior_order", metric],
            ascending=[True, False],
        )
        .groupby("behavior", group_keys=False)
        .head(hotspots_per_behavior)[hotspot_columns]
        .reset_index(drop=True)
    )
    return (
        summary.sort_values(["behavior_order", "rank"]),
        slot.sort_values(["behavior_order", "device"]),
        hotspot,
    )


def main() -> int:
    args = parse_args()
    mapping, manifest = validate_mapping(
        args.dashboard_mapping.resolve(),
        args.rank_topology.resolve(),
        args.nic_manifest.resolve(),
    )
    cells, targets_by_rank, build_validation = build_cells(
        args.events_parquet.resolve(),
        args.event_windows.resolve(),
        mapping,
    )
    manifest_map = {
        (str(host), str(device)): [str(value) for value in group["path"]]
        for (host, device), group in manifest.sort_values(
            ["host", "device", "sequence"]
        ).groupby(["host", "device"])
    }
    long_gap_ns = round(args.nic_long_gap_ms * 1_000_000)
    attributed_rows: list[dict[str, Any]] = []
    for item in mapping.itertuples(index=False):
        key = (str(item.host), str(item.nic_device))
        relative_paths = manifest_map.get(key)
        if relative_paths is None:
            raise ValueError(f"mapped NIC not in manifest: {key}")
        raw_paths = [
            args.hardware_root.resolve() / relative_path
            for relative_path in relative_paths
        ]
        if any(not raw_path.is_file() for raw_path in raw_paths):
            raise FileNotFoundError(raw_paths)
        attributed_rows.extend(
            attribute_device(
                path=raw_paths,
                host=str(item.host),
                device=str(item.nic_device),
                targets=targets_by_rank[int(item.rank)],
                cell_count=len(cells),
                chunk_size=args.chunk_size,
                long_gap_ns=long_gap_ns,
            )
        )

    attributed = pd.DataFrame(attributed_rows).drop(columns=["host", "device"])
    result = cells.merge(
        attributed, on="cell_id", how="left", validate="one_to_one"
    )
    duration = pd.to_numeric(result["edp_union_duration_ns"], errors="coerce")
    for direction in ("tx", "rx", "bidir"):
        result[f"nic_active_window_{direction}_bandwidth_GBps"] = (
            result[f"nic_attributed_{direction}_bytes"] / duration
        )
        result[f"nic_active_window_{direction}_bandwidth_Gbps"] = (
            result[f"nic_active_window_{direction}_bandwidth_GBps"] * 8.0
        )
    result["nic_window_coverage_pct"] = (
        result["nic_sampled_overlap_ns"] / duration * 100.0
    )
    result["nic_long_gap_bidir_pct"] = (
        (
            result["nic_uncertain_long_gap_tx_bytes"]
            + result["nic_uncertain_long_gap_rx_bytes"]
        )
        / result["nic_attributed_bidir_bytes"].replace(0, np.nan)
        * 100.0
    )
    result["nic_long_gap_warning"] = (
        result["nic_long_gap_bidir_pct"].fillna(0).gt(5.0)
    )

    dashboard_stage = pd.read_csv(args.dashboard_stage_summary.resolve())
    dashboard_stage = dashboard_stage[
        [
            "rank",
            "nic_device",
            "nic_point_count_in_edp_windows",
            "nic_xmit_gbps_median_in_edp_windows",
            "nic_xmit_gbps_p95_in_edp_windows",
            "nic_xmit_gbps_max_in_edp_windows",
        ]
    ].rename(columns={"nic_device": "dashboard_nic_device"})
    result = result.merge(
        dashboard_stage, on="rank", how="left", validate="many_to_one"
    )
    if result["device"].ne(result["dashboard_nic_device"]).any():
        raise ValueError("stage summary NIC differs from dashboard mapping")
    dashboard_columns = [
        column
        for column in result.columns
        if column.startswith("nic_point_count_in_edp")
        or column.startswith("nic_xmit_gbps_")
    ]
    result.loc[result["iteration"].ne(5), dashboard_columns] = np.nan
    result = result.drop(columns="dashboard_nic_device")
    result = add_hotspot_fields(result)
    result = result.sort_values(
        ["behavior_order", "iteration", "rank"]
    ).drop(columns="cell_id")

    per_nic, slot, hotspots = build_summaries(
        result, args.hotspots_per_behavior
    )
    status_counts = {
        str(key): int(value)
        for key, value in result["data_status"].value_counts().items()
    }
    observed = result[result["data_status"].ne("missing_source_event")]
    attributed_bidir = float(observed["nic_attributed_bidir_bytes"].sum())
    uncertain_bidir = float(
        observed["nic_uncertain_long_gap_tx_bytes"].sum()
        + observed["nic_uncertain_long_gap_rx_bytes"].sum()
    )
    validation: dict[str, Any] = {
        "status": "PASS",
        "schema_version": "256gpu-edp-nic-rank-device-v1",
        "attribution_granularity": (
            "behavior x iteration x rank x carryover-mapped NIC"
        ),
        "dashboard_mapping_validated_rank_count": int(len(mapping)),
        "dashboard_mapping_status_counts": {
            str(key): int(value)
            for key, value in mapping["mapping_status"].value_counts().items()
        },
        "dashboard_raw_nic_point_count": int(
            mapping["nic_point_count"].sum()
        ),
        "dashboard_raw_nic_point_match_count": int(
            mapping["raw_nic_point_match_count"].sum()
        ),
        "dashboard_reference_scope": (
            "GPU-slot/NIC mapping carried over from the validated 224-GPU "
            "hardware family; no fresh 256-rank dashboard set is available"
        ),
        "raw_telemetry_scope": "training iterations 5..100",
        "nic_byte_attribution": (
            "sample_delta_bytes * overlap_ns / sample_interval_ns"
        ),
        "active_window_bandwidth_formula": (
            "mapped NIC attributed bytes / rank EDP union duration_ns"
        ),
        "display_bandwidth_unit": "decimal Gbit/s (Gb/s)",
        "integrated_byte_rate_columns_retained": "decimal GB/s",
        "dashboard_point_rate_unit": "not available in this 256-GPU corpus",
        "events_parquet": portable_path(args.events_parquet),
        "dashboard_mapping": portable_path(args.dashboard_mapping),
        "output_csv": portable_path(args.output_csv),
        "per_nic_summary_output": portable_path(args.per_nic_summary_output),
        "device_slot_summary_output": portable_path(
            args.device_slot_summary_output
        ),
        "hotspot_output": portable_path(args.hotspot_output),
        "row_count": int(len(result)),
        "expected_row_count": (
            len(EDP_BEHAVIORS) * len(ITERATIONS) * len(RANKS)
        ),
        "per_nic_summary_row_count": int(len(per_nic)),
        "device_slot_summary_row_count": int(len(slot)),
        "hotspot_row_count": int(len(hotspots)),
        "behavior_count": int(result["behavior"].nunique()),
        "iteration_count": int(result["iteration"].nunique()),
        "rank_count": int(result["rank"].nunique()),
        "host_count": int(result["host"].nunique()),
        "physical_nic_entity_count": int(result["nic_entity_id"].nunique()),
        "data_status_counts": status_counts,
        "target_interval_count": build_validation["target_interval_count"],
        "cross_cell_overlap_ns": build_validation["cross_cell_overlap_ns"],
        "edp_source_event_count": build_validation[
            "edp_source_event_count"
        ],
        "minimum_observed_window_coverage_pct": float(
            observed["nic_window_coverage_pct"].min()
        ),
        "maximum_observed_window_coverage_pct": float(
            observed["nic_window_coverage_pct"].max()
        ),
        "invalid_overlap_piece_count": int(
            observed["nic_invalid_overlap_piece_count"].sum()
        ),
        "attributed_nic_bidir_bytes": attributed_bidir,
        "long_gap_attributed_bidir_fraction": (
            uncertain_bidir / attributed_bidir
            if attributed_bidir
            else None
        ),
        "interpretation_limits": [
            (
                "The GPU-slot/NIC pairing is a documented carry-over mapping "
                "from the validated 224-GPU hardware family, not a fresh "
                "256-page dashboard audit; it does not prove exclusive "
                "collective byte ownership."
            ),
            (
                "This 256-GPU case has no dashboard point-rate baseline; "
                "derived values are raw delta-byte overlap integrals."
            ),
            (
                "A NIC sampling interval can contain unrelated traffic; "
                "other-collective overlap and long-gap fractions are retained."
            ),
            (
                "NIC, MTLink and Trace are distinct byte domains and must not "
                "be summed."
            ),
        ],
    }
    checks = [
        len(result) == validation["expected_row_count"],
        validation["behavior_count"] == 2,
        validation["iteration_count"] == len(ITERATIONS),
        validation["rank_count"] == len(RANKS),
        validation["host_count"] == 32,
        validation["physical_nic_entity_count"] == len(RANKS),
        len(per_nic) == len(EDP_BEHAVIORS) * len(RANKS),
        len(slot) == len(EDP_BEHAVIORS) * 8,
        len(hotspots)
        == len(EDP_BEHAVIORS) * args.hotspots_per_behavior,
        validation["cross_cell_overlap_ns"] == 0,
        validation["invalid_overlap_piece_count"] == 0,
        validation["minimum_observed_window_coverage_pct"] >= 99.0,
    ]
    if not all(checks):
        validation["status"] = "FAIL"
        raise RuntimeError(
            json.dumps(clean_json(validation), ensure_ascii=False, indent=2)
        )

    atomic_csv(result, args.output_csv)
    atomic_csv(per_nic, args.per_nic_summary_output)
    atomic_csv(slot, args.device_slot_summary_output)
    atomic_csv(hotspots, args.hotspot_output)
    atomic_json(validation, args.validation_output)
    print(json.dumps(clean_json(validation), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

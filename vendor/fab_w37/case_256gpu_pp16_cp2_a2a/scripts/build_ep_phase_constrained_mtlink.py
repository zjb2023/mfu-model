#!/usr/bin/env python3
"""Build EP MTLink cells with strict serial-phase attribution.

The legacy heatmap allocates each MTLink sample to an event by

    sample_delta_bytes * event_sample_overlap / sample_interval.

DeepEP phases are strictly serial within one rank.  This variant treats the
non-communication gaps between those phases as carrying no EP traffic.  For
every MTLink sample that intersects one or more EP phases, all sample bytes
are distributed only among those intersecting phases:

    phase_weight = phase_sample_overlap / sum(EP phase overlaps in sample)

The output is a complete copy of the existing per-iteration/rank collective
table with only the four EP MTLink fields replaced.  Trace/DeepEP expected
bytes and every non-EP behavior remain byte-for-byte numerically unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EP_BEHAVIORS = (
    "ep_fwd_dispatch",
    "ep_fwd_combine",
    "ep_bwd_combine_backward_dispatch",
    "ep_bwd_dispatch_backward_combine",
)
MAX_I64 = np.uint64(2**63 - 1)
PCIE_LINK_ID = 2**32 - 1
MTLINK_COLUMNS = (
    "mtlink_attributed_tx_bytes_total",
    "mtlink_attributed_rx_bytes_total",
    "mtlink_attributed_bidir_bytes_total",
    "mtlink_uncertain_tx_bytes_total",
    "mtlink_uncertain_rx_bytes_total",
    "mtlink_sample_count",
    "mtlink_observed_link_count_min",
    "mtlink_normalized_overlap_count",
    "mtlink_14link_tx_bandwidth_GBps",
    "mtlink_14link_rx_bandwidth_GBps",
    "mtlink_14link_bidir_bandwidth_GBps",
    "mtlink_long_gap_bidir_pct",
)


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    result_dir = case_root / "results" / "collective_bw_no_pp"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-parquet",
        type=Path,
        default=result_dir / "collective_events_no_pp_pre_mtlink.parquet",
    )
    parser.add_argument(
        "--source-cell-csv",
        type=Path,
        default=result_dir / "collective_iter_rank_bandwidth_core_no_pp.csv",
    )
    parser.add_argument(
        "--mtlink-manifest",
        type=Path,
        default=case_root / "results" / "readiness" / "mtlink_files.csv",
    )
    parser.add_argument(
        "--hardware-root",
        type=Path,
        default=case_root / ".raw_view",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=(
            result_dir
            / "collective_iter_rank_bandwidth_ep_phase_constrained_no_pp.csv"
        ),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=result_dir / "ep_phase_constrained_mtlink_validation.json",
    )
    parser.add_argument("--mtlink-long-gap-ms", type=float, default=10.0)
    parser.add_argument("--world-size", type=int, default=256)
    parser.add_argument(
        "--expected-deepep-events", type=int, default=453_120
    )
    return parser.parse_args()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def phase_overlap_count(events: pd.DataFrame) -> int:
    overlap_count = 0
    for _, group in events.groupby("rank", sort=False):
        ordered = group.sort_values(["start_ns", "end_ns"])
        starts = ordered["start_ns"].to_numpy(dtype=np.int64)
        ends = ordered["end_ns"].to_numpy(dtype=np.int64)
        if len(starts) > 1:
            overlap_count += int(np.count_nonzero(starts[1:] < ends[:-1]))
    return overlap_count


def attribute_rank(
    events: pd.DataFrame,
    mtlink_paths: list[Path],
    long_gap_ns: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    usecols = [
        "host_realtime_ns",
        "link_id",
        "tx_delta_bytes",
        "rx_delta_bytes",
        "mt_timestamp_begin_ns",
        "mt_timestamp_end_ns",
    ]
    frames = [
        pd.read_csv(
            path,
            usecols=usecols,
            dtype={
                "host_realtime_ns": "int64",
                "link_id": "int64",
                "tx_delta_bytes": "uint64",
                "rx_delta_bytes": "uint64",
                "mt_timestamp_begin_ns": "int64",
                "mt_timestamp_end_ns": "int64",
            },
        )
        for path in mtlink_paths
    ]
    samples = pd.concat(frames, ignore_index=True)
    samples = samples[
        samples["link_id"].between(0, 13)
        & samples["link_id"].ne(PCIE_LINK_ID)
    ].drop_duplicates().sort_values("host_realtime_ns", kind="stable")
    sample_end = samples["host_realtime_ns"].to_numpy(dtype=np.int64)
    sample_duration = (
        samples["mt_timestamp_end_ns"].to_numpy(dtype=np.int64)
        - samples["mt_timestamp_begin_ns"].to_numpy(dtype=np.int64)
    )
    sample_start = sample_end - sample_duration
    tx = samples["tx_delta_bytes"].to_numpy(dtype=np.uint64)
    rx = samples["rx_delta_bytes"].to_numpy(dtype=np.uint64)
    valid = (sample_duration > 0) & (tx <= MAX_I64) & (rx <= MAX_I64)
    maximum_duration = int(sample_duration[valid].max())

    duration_bins: list[tuple[np.ndarray, int]] = []
    lower = 0
    for upper in (10_000_000, 25_000_000, 50_000_000, 100_000_000, maximum_duration):
        if upper < lower:
            continue
        indices = np.flatnonzero(
            valid & (sample_duration > lower) & (sample_duration <= upper)
        )
        if len(indices):
            duration_bins.append((indices, int(upper)))
        lower = int(upper)

    pair_samples: list[np.ndarray] = []
    pair_events: list[np.ndarray] = []
    pair_overlaps: list[np.ndarray] = []
    starts = events["start_ns"].to_numpy(dtype=np.int64)
    ends = events["end_ns"].to_numpy(dtype=np.int64)
    for event_index, (event_start, event_end) in enumerate(zip(starts, ends)):
        for bin_indices, upper in duration_bins:
            bin_ends = sample_end[bin_indices]
            left = int(np.searchsorted(bin_ends, event_start, side="right"))
            right = int(
                np.searchsorted(bin_ends, event_end + upper, side="right")
            )
            if right <= left:
                continue
            indices = bin_indices[left:right]
            overlap = np.minimum(sample_end[indices], event_end) - np.maximum(
                sample_start[indices], event_start
            )
            keep = overlap > 0
            if not np.any(keep):
                continue
            indices = indices[keep]
            pair_samples.append(indices)
            pair_events.append(
                np.full(len(indices), event_index, dtype=np.int64)
            )
            pair_overlaps.append(overlap[keep].astype(np.float64))

    if not pair_samples:
        raise RuntimeError(f"no EP/MTLink intersections for {mtlink_paths}")

    sample_index = np.concatenate(pair_samples)
    event_index = np.concatenate(pair_events)
    overlap_ns = np.concatenate(pair_overlaps)
    overlap_sum = np.bincount(
        sample_index, weights=overlap_ns, minlength=len(samples)
    )
    pair_count = np.bincount(sample_index, minlength=len(samples))
    weight = overlap_ns / overlap_sum[sample_index]
    event_count = len(events)
    tx_float = tx[sample_index].astype(np.float64)
    rx_float = rx[sample_index].astype(np.float64)
    long_gap = sample_duration[sample_index] > long_gap_ns
    shared_phase = pair_count[sample_index] > 1

    result = events.copy()
    result["phase_mtlink_attributed_tx_bytes"] = np.bincount(
        event_index, weights=tx_float * weight, minlength=event_count
    )
    result["phase_mtlink_attributed_rx_bytes"] = np.bincount(
        event_index, weights=rx_float * weight, minlength=event_count
    )
    result["phase_mtlink_uncertain_tx_bytes"] = np.bincount(
        event_index,
        weights=tx_float * weight * long_gap,
        minlength=event_count,
    )
    result["phase_mtlink_uncertain_rx_bytes"] = np.bincount(
        event_index,
        weights=rx_float * weight * long_gap,
        minlength=event_count,
    )
    result["phase_mtlink_sample_count"] = np.bincount(
        event_index, minlength=event_count
    )
    result["phase_mtlink_shared_sample_count"] = np.bincount(
        event_index, weights=shared_phase, minlength=event_count
    )
    event_link = np.unique(
        event_index * 14
        + samples["link_id"].to_numpy(dtype=np.int64)[sample_index]
    )
    result["phase_mtlink_observed_link_count"] = np.bincount(
        event_link // 14, minlength=event_count
    )

    per_sample_weight = np.bincount(
        sample_index, weights=weight, minlength=len(samples)
    )
    used = pair_count > 0
    shared_bytes = float(
        (tx.astype(np.float64)[used & (pair_count > 1)]).sum()
    )
    intersecting_bytes = float(tx.astype(np.float64)[used].sum())
    return result, {
        "pair_count": int(len(weight)),
        "intersecting_sample_count": int(np.count_nonzero(used)),
        "shared_phase_sample_count": int(np.count_nonzero(pair_count > 1)),
        "shared_phase_tx_byte_fraction": (
            shared_bytes / intersecting_bytes if intersecting_bytes else 0.0
        ),
        "max_sample_weight_sum": float(per_sample_weight[used].max()),
        "min_sample_weight_sum": float(per_sample_weight[used].min()),
    }


def aggregate_phase_cells(events: pd.DataFrame) -> pd.DataFrame:
    keys = ["behavior", "iteration", "rank"]
    cells = (
        events.groupby(keys, as_index=False)
        .agg(
            event_count_check=("duration_ns", "size"),
            duration_ns_sum_check=("duration_ns", "sum"),
            mtlink_attributed_tx_bytes_total=(
                "phase_mtlink_attributed_tx_bytes",
                "sum",
            ),
            mtlink_attributed_rx_bytes_total=(
                "phase_mtlink_attributed_rx_bytes",
                "sum",
            ),
            mtlink_uncertain_tx_bytes_total=(
                "phase_mtlink_uncertain_tx_bytes",
                "sum",
            ),
            mtlink_uncertain_rx_bytes_total=(
                "phase_mtlink_uncertain_rx_bytes",
                "sum",
            ),
            mtlink_sample_count=("phase_mtlink_sample_count", "sum"),
            mtlink_observed_link_count_min=(
                "phase_mtlink_observed_link_count",
                "min",
            ),
            mtlink_normalized_overlap_count=(
                "phase_mtlink_shared_sample_count",
                "sum",
            ),
        )
        .sort_values(keys)
    )
    cells["mtlink_attributed_bidir_bytes_total"] = (
        cells["mtlink_attributed_tx_bytes_total"]
        + cells["mtlink_attributed_rx_bytes_total"]
    )
    duration = cells["duration_ns_sum_check"].replace(0, np.nan)
    cells["mtlink_14link_tx_bandwidth_GBps"] = (
        cells["mtlink_attributed_tx_bytes_total"] / duration
    )
    cells["mtlink_14link_rx_bandwidth_GBps"] = (
        cells["mtlink_attributed_rx_bytes_total"] / duration
    )
    cells["mtlink_14link_bidir_bandwidth_GBps"] = (
        cells["mtlink_attributed_bidir_bytes_total"] / duration
    )
    cells["mtlink_long_gap_bidir_pct"] = (
        (
            cells["mtlink_uncertain_tx_bytes_total"]
            + cells["mtlink_uncertain_rx_bytes_total"]
        )
        / cells["mtlink_attributed_bidir_bytes_total"].replace(0, np.nan)
        * 100.0
    ).fillna(0.0)
    return cells


def main() -> int:
    args = parse_args()
    event_columns = [
        "event_id",
        "source",
        "behavior",
        "iteration",
        "rank",
        "host",
        "gpu_id",
        "start_ns",
        "end_ns",
        "duration_ns",
        "expected_tx_bytes",
    ]
    events = pd.read_parquet(args.event_parquet, columns=event_columns)
    events = events[
        events["source"].eq("deepep")
        & events["behavior"].isin(EP_BEHAVIORS)
    ].sort_values(["rank", "start_ns", "end_ns", "behavior"])
    source = pd.read_csv(args.source_cell_csv)
    source_ep = source[source["behavior"].isin(EP_BEHAVIORS)].copy()
    non_ep_before = source[~source["behavior"].isin(EP_BEHAVIORS)].copy()

    manifest = read_manifest(args.mtlink_manifest)
    paths: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for row in manifest:
        if not row.get("error"):
            paths[(str(row["host"]), int(row["gpu_id"]))].append(
                args.hardware_root / row["path"]
            )
    for entity_paths in paths.values():
        entity_paths.sort()
    attributed: list[pd.DataFrame] = []
    rank_checks: list[dict[str, float | int]] = []
    long_gap_ns = round(args.mtlink_long_gap_ms * 1_000_000)
    for rank, group in events.groupby("rank", sort=True):
        host = str(group["host"].iloc[0])
        gpu_id = int(group["gpu_id"].iloc[0])
        result, check = attribute_rank(
            group.reset_index(drop=True),
            paths[(host, gpu_id)],
            long_gap_ns,
        )
        attributed.append(result)
        rank_checks.append({"rank": int(rank), **check})
    attributed_events = pd.concat(attributed, ignore_index=True)
    phase_cells = aggregate_phase_cells(attributed_events)

    keys = ["behavior", "iteration", "rank"]
    checks = source_ep[keys + ["event_count", "duration_ns_sum"]].merge(
        phase_cells[
            keys + ["event_count_check", "duration_ns_sum_check"]
        ],
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    if not (
        checks["event_count"].eq(checks["event_count_check"]).all()
        and checks["duration_ns_sum"].eq(checks["duration_ns_sum_check"]).all()
    ):
        raise RuntimeError("phase cells differ from canonical EP event counts/durations")

    replacement = phase_cells[keys + list(MTLINK_COLUMNS)]
    output = source.drop(columns=list(MTLINK_COLUMNS)).merge(
        replacement,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    ep_mask = output["behavior"].isin(EP_BEHAVIORS)
    for column in MTLINK_COLUMNS:
        original = source.set_index(keys)[column]
        output_index = output.set_index(keys).index
        original_values = original.reindex(output_index).to_numpy()
        output.loc[~ep_mask, column] = original_values[~ep_mask.to_numpy()]

    output = output[source.columns]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)

    new_ep = output[output["behavior"].isin(EP_BEHAVIORS)]
    behavior_summary = (
        new_ep.groupby("behavior", as_index=False)
        .agg(
            expected_tx_bytes=("expected_tx_bytes_total", "sum"),
            phase_mtlink_tx_bytes=("mtlink_attributed_tx_bytes_total", "sum"),
            duration_ns=("duration_ns_sum", "sum"),
        )
        .sort_values("behavior")
    )
    behavior_summary["tx_coverage_pct"] = (
        behavior_summary["phase_mtlink_tx_bytes"]
        / behavior_summary["expected_tx_bytes"]
        * 100.0
    )
    behavior_summary["expected_tx_bandwidth_GBps"] = (
        behavior_summary["expected_tx_bytes"]
        / behavior_summary["duration_ns"]
    )
    behavior_summary["phase_mtlink_tx_bandwidth_GBps"] = (
        behavior_summary["phase_mtlink_tx_bytes"]
        / behavior_summary["duration_ns"]
    )
    expected_total = float(behavior_summary["expected_tx_bytes"].sum())
    attributed_total = float(behavior_summary["phase_mtlink_tx_bytes"].sum())

    non_ep_after = output[~output["behavior"].isin(EP_BEHAVIORS)]
    numeric_columns = source.select_dtypes(include=[np.number]).columns
    non_ep_numeric_equal = bool(
        np.allclose(
            non_ep_before[numeric_columns],
            non_ep_after[numeric_columns],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    )
    expected_columns = [
        "logical_input_bytes_total",
        "expected_tx_bytes_total",
        "expected_rx_bytes_total",
        "trace_logical_input_bandwidth_GBps",
        "trace_expected_tx_bandwidth_GBps",
        "trace_expected_rx_bandwidth_GBps",
    ]
    expected_unchanged = bool(
        np.allclose(
            source[expected_columns],
            output[expected_columns],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    )
    validation = {
        "status": "PASS",
        "schema_version": "256gpu-ep-mtlink-strict-serial-phase-v1",
        "formula": (
            "sample_delta_bytes * phase_overlap_ns / "
            "sum(all EP phase overlaps in sample)"
        ),
        "expected_tx_formula": (
            "sum(DeepEP send_bytes) / sum(DeepEP elapsed_ns); "
            "no additional algorithm factor"
        ),
        "event_count": int(len(events)),
        "cell_count": int(len(new_ep)),
        "rank_count": int(events["rank"].nunique()),
        "iteration_count": int(events["iteration"].nunique()),
        "adjacent_ep_event_overlap_count": phase_overlap_count(events),
        "maximum_sample_weight_sum": max(
            float(item["max_sample_weight_sum"]) for item in rank_checks
        ),
        "minimum_sample_weight_sum": min(
            float(item["min_sample_weight_sum"]) for item in rank_checks
        ),
        "shared_phase_tx_byte_fraction_rank_mean": float(
            np.mean(
                [
                    float(item["shared_phase_tx_byte_fraction"])
                    for item in rank_checks
                ]
            )
        ),
        "expected_tx_bytes_total": expected_total,
        "phase_mtlink_tx_bytes_total": attributed_total,
        "total_tx_coverage_pct": attributed_total / expected_total * 100.0,
        "behavior_summary": behavior_summary.to_dict("records"),
        "expected_fields_unchanged": expected_unchanged,
        "non_ep_numeric_fields_unchanged": non_ep_numeric_equal,
        "output_csv": str(args.output_csv),
        "notes": [
            "This is an attribution estimate, not an exact counter reconstruction.",
            "All MTLink bytes in a sample intersecting EP are assigned only to intersecting EP phases.",
            "Samples spanning multiple EP phases are split by phase overlap duration.",
        ],
    }
    required = [
        len(events) == args.expected_deepep_events,
        len(new_ep) == 4 * 20 * args.world_size,
        validation["rank_count"] == args.world_size,
        validation["iteration_count"] == 20,
        validation["adjacent_ep_event_overlap_count"] == 0,
        validation["maximum_sample_weight_sum"] <= 1.0 + 1e-12,
        validation["minimum_sample_weight_sum"] >= 1.0 - 1e-12,
        np.isfinite(validation["total_tx_coverage_pct"]),
        validation["total_tx_coverage_pct"] > 0,
        expected_unchanged,
        non_ep_numeric_equal,
    ]
    if not all(required):
        validation["status"] = "FAIL"
    atomic_text(
        args.validation_output,
        json.dumps(clean_json(validation), ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(clean_json(validation), ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

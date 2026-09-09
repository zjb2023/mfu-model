#!/usr/bin/env python3
"""Build the tagged CSV and standalone no-PP collective heatmap for 256 GPUs.

The page displays one collective behavior at a time.  Its embedded data is
generated from the tagged CSV written by this script, so the CSV remains the
canonical input for the visualization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.offline.offline import get_plotlyjs


ITERATIONS = tuple(range(5, 101, 5))
RANKS = tuple(range(256))
NIC_DEVICES = (
    "mlx5_0",
    "mlx5_1",
    "mlx5_2",
    "mlx5_3",
    "mlx5_6",
    "mlx5_7",
    "mlx5_8",
    "mlx5_9",
)

BEHAVIORS = (
    {
        "behavior": "ep_fwd_dispatch",
        "family": "EP",
        "label": "EP-fwd-dispatch",
        "description": "Forward expert dispatch",
        "layer_view": "all_moe",
    },
    {
        "behavior": "ep_fwd_combine",
        "family": "EP",
        "label": "EP-fwd-combine",
        "description": "Forward expert combine",
        "layer_view": "all_moe",
    },
    {
        "behavior": "ep_bwd_combine_backward_dispatch",
        "family": "EP",
        "label": "EP-bwd-dispatch",
        "description": "Backward dispatch (combine backward)",
        "layer_view": "all_moe",
    },
    {
        "behavior": "ep_bwd_dispatch_backward_combine",
        "family": "EP",
        "label": "EP-bwd-combine",
        "description": "Backward combine (dispatch backward)",
        "layer_view": "all_moe",
    },
    {
        "behavior": "cp_all_to_all",
        "family": "CP",
        "label": "CP-all-to-all",
        "description": "Context-parallel all-to-all",
        "layer_view": "not_applicable",
    },
    {
        "behavior": "dp_grad_reduce_scatter",
        "family": "DP",
        "label": "DP-reduce-scatter",
        "description": "Gradient reduce-scatter",
        "layer_view": "not_applicable",
    },
    {
        "behavior": "dp_param_allgather",
        "family": "DP",
        "label": "DP-all-gather",
        "description": "Parameter all-gather",
        "layer_view": "not_applicable",
    },
    {
        "behavior": "expert_dp_grad_reduce_scatter",
        "family": "Expert-DP",
        "label": "Expert-DP-reduce-scatter",
        "description": "Expert gradient reduce-scatter",
        "layer_view": "not_applicable",
    },
    {
        "behavior": "expert_dp_param_allgather",
        "family": "Expert-DP",
        "label": "Expert-DP-all-gather",
        "description": "Expert parameter all-gather",
        "layer_view": "not_applicable",
    },
)

METRICS = (
    {
        "column": "mtlink_14link_tx_bandwidth_GBps",
        "label": "MTLink TX",
        "description": "14-link transmitted bytes / collective event time",
        "dimension": "rank",
    },
    {
        "column": "mtlink_14link_rx_bandwidth_GBps",
        "label": "MTLink RX",
        "description": "14-link received bytes / collective event time",
        "dimension": "rank",
    },
    {
        "column": "mtlink_14link_bidir_bandwidth_GBps",
        "label": "MTLink TX+RX",
        "description": "Full-duplex activity; not a single-direction rate",
        "dimension": "rank",
    },
    {
        "column": "trace_logical_input_bandwidth_GBps",
        "label": "Trace logical input",
        "description": "Logical collective payload / collective event time",
        "dimension": "rank",
    },
    {
        "column": "trace_expected_tx_bandwidth_GBps",
        "label": "Trace expected TX",
        "description": "Algorithm-expected transmitted bytes / event time",
        "dimension": "rank",
    },
    {
        "column": "trace_expected_rx_bandwidth_GBps",
        "label": "Trace expected RX",
        "description": "Algorithm-expected received bytes / event time",
        "dimension": "rank",
    },
)

EDP_NIC_METRICS = (
    {
        "column": "nic_active_window_tx_bandwidth_Gbps",
        "label": "EDP-NIC mapped TX",
        "description": "Mapped NIC TX bit rate / rank EDP window",
        "dimension": "rank_nic",
    },
    {
        "column": "nic_active_window_rx_bandwidth_Gbps",
        "label": "EDP-NIC mapped RX",
        "description": "Mapped NIC RX bit rate / rank EDP window",
        "dimension": "rank_nic",
    },
    {
        "column": "nic_active_window_bidir_bandwidth_Gbps",
        "label": "EDP-NIC mapped TX+RX",
        "description": "Mapped NIC full-duplex bit activity / rank EDP window",
        "dimension": "rank_nic",
    },
)

DP_ACTIVE_METRICS = (
    {
        "column": "mtlink_active_tx_bandwidth_GBps",
        "label": "DP-MTLink active TX",
        "description": "Active MTLink TX bytes / MTLink TX active union time",
        "dimension": "rank",
        "unit": "GB/s",
        "source": "dp_active",
        "domain": "MTLink",
        "bytes_column": "mtlink_active_tx_bytes",
        "duration_column": "mtlink_active_tx_duration_ns",
        "segment_column": "mtlink_active_tx_segment_count",
    },
    {
        "column": "mtlink_active_rx_bandwidth_GBps",
        "label": "DP-MTLink active RX",
        "description": "Active MTLink RX bytes / MTLink RX active union time",
        "dimension": "rank",
        "unit": "GB/s",
        "source": "dp_active",
        "domain": "MTLink",
        "bytes_column": "mtlink_active_rx_bytes",
        "duration_column": "mtlink_active_rx_duration_ns",
        "segment_column": "mtlink_active_rx_segment_count",
    },
    {
        "column": "nic_active_tx_bandwidth_Gbps",
        "label": "DP-NIC mapped active TX",
        "description": "Active mapped-NIC TX bytes / NIC TX active union time",
        "dimension": "rank",
        "unit": "Gb/s",
        "source": "dp_active",
        "domain": "mapped NIC",
        "bytes_column": "nic_active_tx_bytes",
        "duration_column": "nic_active_tx_duration_ns",
        "segment_column": "nic_active_tx_segment_count",
        "long_gap_column": "nic_long_gap_bidir_pct",
        "touched_column": "nic_touched_sample_count",
    },
    {
        "column": "nic_active_rx_bandwidth_Gbps",
        "label": "DP-NIC mapped active RX",
        "description": "Active mapped-NIC RX bytes / NIC RX active union time",
        "dimension": "rank",
        "unit": "Gb/s",
        "source": "dp_active",
        "domain": "mapped NIC",
        "bytes_column": "nic_active_rx_bytes",
        "duration_column": "nic_active_rx_duration_ns",
        "segment_column": "nic_active_rx_segment_count",
        "long_gap_column": "nic_long_gap_bidir_pct",
        "touched_column": "nic_touched_sample_count",
    },
)

CP_ACTIVE_METRICS = (
    {
        "column": "cp_mtlink_active_tx_bandwidth_GBps",
        "source_column": "mtlink_active_tx_bandwidth_GBps",
        "label": "CP-MTLink active TX",
        "description": "Active MTLink TX bytes / MTLink TX active union time",
        "dimension": "rank",
        "unit": "GB/s",
        "source": "cp_active",
        "domain": "MTLink",
        "bytes_column": "mtlink_active_tx_bytes",
        "duration_column": "mtlink_active_tx_duration_ns",
        "segment_column": "mtlink_active_tx_segment_count",
        "long_gap_column": "mtlink_long_gap_bidir_pct",
        "touched_column": "mtlink_touched_sample_count",
    },
    {
        "column": "cp_mtlink_active_rx_bandwidth_GBps",
        "source_column": "mtlink_active_rx_bandwidth_GBps",
        "label": "CP-MTLink active RX",
        "description": "Active MTLink RX bytes / MTLink RX active union time",
        "dimension": "rank",
        "unit": "GB/s",
        "source": "cp_active",
        "domain": "MTLink",
        "bytes_column": "mtlink_active_rx_bytes",
        "duration_column": "mtlink_active_rx_duration_ns",
        "segment_column": "mtlink_active_rx_segment_count",
        "long_gap_column": "mtlink_long_gap_bidir_pct",
        "touched_column": "mtlink_touched_sample_count",
    },
)

TRACE_METRICS = tuple(
    metric
    for metric in METRICS
    if str(metric["column"]).startswith("trace_")
)

EXPECTED_EVENTS_PER_CELL = {
    "dp_grad_reduce_scatter": 1,
    "dp_param_allgather": 2,
    "expert_dp_grad_reduce_scatter": 1,
    "expert_dp_param_allgather": 2,
}

PP_DESCRIPTIONS = {
    "PIPELINE_MODEL_PARALLEL_GROUP",
    "MODEL_PARALLEL_GROUP",
    "EMBEDDING_GROUP",
    "POSITION_EMBEDDING_GROUP",
}
EXCLUDED_COLLECTIVES = {"send", "recv", "wait"}


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    output_dir = case_root / "results" / "collective_bw_no_pp"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=output_dir / "collective_iter_rank_bandwidth_core_no_pp.csv",
    )
    parser.add_argument(
        "--rank-topology",
        type=Path,
        default=case_root / "results" / "topology" / "rank_topology.csv",
    )
    parser.add_argument(
        "--edp-nic-device-source-csv",
        type=Path,
        default=output_dir / "edp_nic_iteration_rank_device.csv",
    )
    parser.add_argument(
        "--dp-active-source-csv",
        type=Path,
        default=output_dir / "dp_active_physical_iteration_rank.csv",
    )
    parser.add_argument(
        "--cp-active-source-csv",
        type=Path,
        default=output_dir / "cp_active_mtlink_iteration_rank.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=output_dir / "communication_spatial_heatmap_data_no_pp.csv",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=output_dir
        / "communication_spatial_heatmap_validation_no_pp.json",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=case_root
        / "figures"
        / "communication_spatial_heatmaps.html",
    )
    parser.add_argument(
        "--ep-mtlink-attribution",
        choices=("proportional-overlap", "strict-serial-phase"),
        default="proportional-overlap",
        help=(
            "Describes the EP MTLink byte attribution already present in "
            "--source-csv; it changes page labels/notes, not the data."
        ),
    )
    return parser.parse_args()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


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


def behavior_frame() -> pd.DataFrame:
    rows = []
    for order, item in enumerate(BEHAVIORS):
        rows.append(
            {
                "behavior": item["behavior"],
                "behavior_order": order,
                "behavior_family": item["family"],
                "behavior_label": item["label"],
                "behavior_description": item["description"],
                "layer_view": item["layer_view"],
                "expected_events_per_iter_rank": EXPECTED_EVENTS_PER_CELL.get(
                    str(item["behavior"]), pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def build_tagged_csv(
    source_path: Path, topology_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(source_path)
    topology = pd.read_csv(topology_path)
    behavior_tags = behavior_frame()
    behavior_names = set(behavior_tags["behavior"])
    if set(source["behavior"]) != behavior_names:
        raise ValueError(
            "source behavior mismatch: "
            f"missing={sorted(behavior_names - set(source['behavior']))}, "
            f"extra={sorted(set(source['behavior']) - behavior_names)}"
        )
    if source.duplicated(["behavior", "iteration", "rank"]).any():
        raise ValueError("duplicate behavior × iteration × rank source cells")

    rank_map = topology[
        ["rank", "host", "local_rank", "gpu_id"]
    ].sort_values("rank")
    if (
        len(rank_map) != len(RANKS)
        or set(rank_map["rank"]) != set(RANKS)
        or rank_map["host"].nunique() != 32
    ):
        raise ValueError("rank topology is not a complete 256-rank/32-host map")
    identity = source.merge(
        rank_map,
        on="rank",
        suffixes=("_source", "_topology"),
        validate="many_to_one",
    )
    identity_mismatch = (
        identity["host_source"].ne(identity["host_topology"])
        | identity["gpu_id_source"].ne(identity["gpu_id_topology"])
    )
    if identity_mismatch.any():
        raise ValueError("source host/GPU identity differs from rank topology")

    source = source.drop(columns=["host", "gpu_id"])
    grid = pd.MultiIndex.from_product(
        [behavior_tags["behavior"], ITERATIONS, RANKS],
        names=["behavior", "iteration", "rank"],
    ).to_frame(index=False)
    result = (
        grid.merge(
            source,
            on=["behavior", "iteration", "rank"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            behavior_tags,
            on="behavior",
            how="left",
            validate="many_to_one",
        )
        .merge(rank_map, on="rank", how="left", validate="many_to_one")
    )

    static_columns = (
        "source",
        "collective",
        "pg_description",
        "group_size",
    )
    for column in static_columns:
        values = (
            source[["behavior", column]]
            .dropna()
            .drop_duplicates()
            .groupby("behavior")[column]
            .agg(list)
        )
        if any(len(item) != 1 for item in values):
            raise ValueError(f"{column} is not static within behavior")
        mapping = {behavior: items[0] for behavior, items in values.items()}
        result[column] = result[column].fillna(result["behavior"].map(mapping))

    host_order = (
        rank_map.sort_values("rank")
        .drop_duplicates("host")
        .reset_index(drop=True)["host"]
    )
    result["node_index"] = result["host"].map(
        {host: index for index, host in enumerate(host_order)}
    )
    result["cell_observed"] = result["event_count"].notna()
    result["data_status"] = "observed"
    result.loc[~result["cell_observed"], "data_status"] = (
        "missing_source_event"
    )
    expected = pd.to_numeric(
        result["expected_events_per_iter_rank"], errors="coerce"
    )
    observed_events = pd.to_numeric(result["event_count"], errors="coerce")
    partial = (
        result["cell_observed"]
        & expected.notna()
        & observed_events.lt(expected)
    )
    result.loc[partial, "data_status"] = "partial_source_event"
    result["event_count_delta"] = observed_events - expected
    result.loc[~result["cell_observed"], "event_count"] = 0
    result["event_count"] = result["event_count"].astype("Int64")
    result["mtlink_long_gap_warning"] = (
        result["mtlink_long_gap_bidir_pct"].fillna(0).gt(5.0)
    )
    result["mtlink_overlap_normalized"] = (
        result["mtlink_normalized_overlap_count"].fillna(0).gt(0)
    )

    pp_rows = result[
        result["pg_description"].isin(PP_DESCRIPTIONS)
        | result["pg_description"].str.contains(
            "PIPELINE", case=False, na=False
        )
        | result["collective"].str.lower().isin(EXCLUDED_COLLECTIVES)
    ]
    first_columns = [
        "behavior_order",
        "behavior_family",
        "behavior",
        "behavior_label",
        "behavior_description",
        "layer_view",
        "iteration",
        "rank",
        "node_index",
        "host",
        "local_rank",
        "gpu_id",
        "data_status",
        "cell_observed",
        "expected_events_per_iter_rank",
        "event_count_delta",
        "mtlink_long_gap_warning",
        "mtlink_overlap_normalized",
    ]
    result = result[
        first_columns
        + [column for column in result.columns if column not in first_columns]
    ].sort_values(["behavior_order", "iteration", "rank"])

    status_counts = {
        str(key): int(value)
        for key, value in result["data_status"].value_counts().items()
    }
    validation = {
        "status": "PASS",
        "schema_version": "256gpu-collective-spatial-heatmap-v6",
        "pp_included": False,
        "source_csv": portable_path(source_path),
        "source_row_count": int(len(source)),
        "output_row_count": int(len(result)),
        "expected_output_row_count": (
            len(BEHAVIORS) * len(ITERATIONS) * len(RANKS)
        ),
        "behavior_count": int(result["behavior"].nunique()),
        "behaviors": [str(item["behavior"]) for item in BEHAVIORS],
        "iteration_count": int(result["iteration"].nunique()),
        "iterations": list(ITERATIONS),
        "rank_count": int(result["rank"].nunique()),
        "host_count": int(result["host"].nunique()),
        "data_status_counts": status_counts,
        "duplicate_cell_count": int(
            result.duplicated(["behavior", "iteration", "rank"]).sum()
        ),
        "pp_row_count": int(len(pp_rows)),
        "bandwidth_unit": "decimal GB/s",
        "bandwidth_formula": "sum(bytes)/sum(event_duration_ns)",
        "missing_cells_are_null_not_imputed": True,
    }
    checks = [
        len(result) == validation["expected_output_row_count"],
        validation["behavior_count"] == len(BEHAVIORS),
        validation["iteration_count"] == len(ITERATIONS),
        validation["rank_count"] == len(RANKS),
        validation["host_count"] == 32,
        validation["duplicate_cell_count"] == 0,
        validation["pp_row_count"] == 0,
    ]
    if not all(checks):
        validation["status"] = "FAIL"
        raise RuntimeError(json.dumps(clean_json(validation), ensure_ascii=False))
    return result, validation


def matrix(
    frame: pd.DataFrame, behavior: str, column: str
) -> list[list[Any]]:
    values = (
        frame[frame["behavior"].eq(behavior)]
        .pivot(index="rank", columns="iteration", values=column)
        .reindex(index=RANKS, columns=ITERATIONS)
    )
    return clean_json(values.to_numpy().tolist())


def build_edp_nic_frame(source_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(source_path)
    expected_behaviors = {
        "expert_dp_grad_reduce_scatter",
        "expert_dp_param_allgather",
    }
    required_columns = {
        "behavior",
        "iteration",
        "node_index",
        "host",
        "first_rank",
        "last_rank",
        "event_count",
        "expected_event_count",
        "data_status",
        "edp_union_segment_count",
        "edp_union_duration_ns",
        "nic_device_count",
        "nic_min_device_window_coverage_pct",
        "nic_long_gap_bidir_pct",
        "other_collective_overlap_pct",
        *(str(item["column"]) for item in EDP_NIC_METRICS),
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"EDP-NIC source missing columns: {missing_columns}")
    duplicate_count = int(
        frame.duplicated(["behavior", "iteration", "host"]).sum()
    )
    host_meta = (
        frame[
            ["node_index", "host", "first_rank", "last_rank"]
        ]
        .drop_duplicates()
        .sort_values("node_index")
    )
    checks = [
        len(frame) == len(expected_behaviors) * len(ITERATIONS) * 32,
        set(frame["behavior"]) == expected_behaviors,
        set(frame["iteration"]) == set(ITERATIONS),
        frame["host"].nunique() == 32,
        len(host_meta) == 32,
        set(host_meta["node_index"]) == set(range(32)),
        host_meta["last_rank"].sub(host_meta["first_rank"]).eq(7).all(),
        frame["nic_device_count"].eq(8).all(),
        duplicate_count == 0,
        frame[list(str(item["column"]) for item in EDP_NIC_METRICS)]
        .notna()
        .all()
        .all(),
    ]
    validation = {
        "schema_version": "256gpu-edp-nic-device-spatial-v1",
        "source_csv": portable_path(source_path),
        "row_count": int(len(frame)),
        "expected_row_count": len(expected_behaviors) * len(ITERATIONS) * 32,
        "behavior_count": int(frame["behavior"].nunique()),
        "iteration_count": int(frame["iteration"].nunique()),
        "host_count": int(frame["host"].nunique()),
        "duplicate_cell_count": duplicate_count,
        "data_status_counts": {
            str(key): int(value)
            for key, value in frame["data_status"].value_counts().items()
        },
        "rank_to_nic_attribution_performed": False,
        "display_dimension": "host",
    }
    if not all(checks):
        raise RuntimeError(json.dumps(clean_json(validation), ensure_ascii=False))
    return frame, validation


def build_edp_nic_device_frame(
    source_path: Path,
    host_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(source_path)
    expected_behaviors = {
        "expert_dp_grad_reduce_scatter",
        "expert_dp_param_allgather",
    }
    required_columns = {
        "behavior",
        "iteration",
        "host",
        "device",
        "nic_attributed_tx_bytes",
        "nic_attributed_rx_bytes",
        "nic_attributed_bidir_bytes",
        "nic_uncertain_long_gap_tx_bytes",
        "nic_uncertain_long_gap_rx_bytes",
        "nic_sampled_overlap_ns",
        "nic_overlap_piece_count",
        "nic_invalid_overlap_piece_count",
        "edp_union_duration_ns",
        "nic_window_coverage_pct",
        *(str(item["column"]) for item in EDP_NIC_METRICS),
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"EDP-NIC device source missing columns: {missing_columns}"
        )

    host_details = host_frame[
        [
            "behavior",
            "iteration",
            "host",
            "node_index",
            "first_rank",
            "last_rank",
            "event_count",
            "expected_event_count",
            "data_status",
            "edp_union_segment_count",
            "edp_union_duration_ns",
            "other_collective_overlap_pct",
        ]
    ].rename(columns={"edp_union_duration_ns": "host_edp_union_duration_ns"})
    frame = frame.merge(
        host_details,
        on=["behavior", "iteration", "host"],
        how="left",
        validate="many_to_one",
    )
    duration_mismatch = ~np.isclose(
        frame["edp_union_duration_ns"],
        frame["host_edp_union_duration_ns"],
        rtol=0.0,
        atol=0.0,
    )
    if duration_mismatch.any():
        raise ValueError("device and host EDP union durations differ")
    frame = frame.drop(columns="host_edp_union_duration_ns")
    device_order = {device: index for index, device in enumerate(NIC_DEVICES)}
    frame["device_order"] = frame["device"].map(device_order)
    if frame["device_order"].isna().any():
        raise ValueError("unexpected NIC device")
    frame["device_order"] = frame["device_order"].astype(int)
    frame["nic_entity_index"] = (
        frame["node_index"].astype(int) * len(NIC_DEVICES)
        + frame["device_order"]
    )
    frame["nic_entity_id"] = (
        frame["host"].astype(str) + " / " + frame["device"].astype(str)
    )
    frame["nic_long_gap_bidir_pct"] = (
        (
            frame["nic_uncertain_long_gap_tx_bytes"]
            + frame["nic_uncertain_long_gap_rx_bytes"]
        )
        / frame["nic_attributed_bidir_bytes"].replace(0, np.nan)
        * 100.0
    )
    frame["nic_long_gap_warning"] = (
        frame["nic_long_gap_bidir_pct"].fillna(0).gt(5.0)
    )

    duplicate_count = int(
        frame.duplicated(["behavior", "iteration", "host", "device"]).sum()
    )
    entity_meta = (
        frame[
            [
                "nic_entity_index",
                "nic_entity_id",
                "node_index",
                "host",
                "device",
                "device_order",
                "first_rank",
                "last_rank",
            ]
        ]
        .drop_duplicates()
        .sort_values("nic_entity_index")
    )

    keys = ["behavior", "iteration", "host"]
    device_sum = (
        frame.groupby(keys, as_index=False)
        .agg(
            nic_attributed_tx_bytes_device_sum=(
                "nic_attributed_tx_bytes",
                "sum",
            ),
            nic_attributed_rx_bytes_device_sum=(
                "nic_attributed_rx_bytes",
                "sum",
            ),
            nic_attributed_bidir_bytes_device_sum=(
                "nic_attributed_bidir_bytes",
                "sum",
            ),
            nic_device_count=("device", "nunique"),
        )
        .merge(
            host_frame[
                keys
                + [
                    "nic_attributed_tx_bytes",
                    "nic_attributed_rx_bytes",
                    "nic_attributed_bidir_bytes",
                ]
            ],
            on=keys,
            how="left",
            validate="one_to_one",
        )
    )
    conservation_checks = {}
    for direction in ("tx", "rx", "bidir"):
        device_column = (
            f"nic_attributed_{direction}_bytes_device_sum"
        )
        host_column = f"nic_attributed_{direction}_bytes"
        difference = (
            device_sum[device_column] - device_sum[host_column]
        ).abs()
        conservation_checks[direction] = {
            "maximum_absolute_byte_difference": float(difference.max()),
            "allclose": bool(
                np.allclose(
                    device_sum[device_column],
                    device_sum[host_column],
                    rtol=1e-12,
                    atol=1e-3,
                )
            ),
        }

    checks = [
        len(frame)
        == len(expected_behaviors)
        * len(ITERATIONS)
        * 32
        * len(NIC_DEVICES),
        set(frame["behavior"]) == expected_behaviors,
        set(frame["iteration"]) == set(ITERATIONS),
        frame["host"].nunique() == 32,
        set(frame["device"]) == set(NIC_DEVICES),
        len(entity_meta) == 256,
        set(entity_meta["nic_entity_index"]) == set(range(256)),
        duplicate_count == 0,
        frame["nic_window_coverage_pct"].ge(99.0).all(),
        frame["nic_invalid_overlap_piece_count"].eq(0).all(),
        device_sum["nic_device_count"].eq(8).all(),
        all(item["allclose"] for item in conservation_checks.values()),
    ]
    validation = {
        "source_csv": portable_path(source_path),
        "row_count": int(len(frame)),
        "expected_row_count": (
            len(expected_behaviors)
            * len(ITERATIONS)
            * 32
            * len(NIC_DEVICES)
        ),
        "behavior_count": int(frame["behavior"].nunique()),
        "iteration_count": int(frame["iteration"].nunique()),
        "host_count": int(frame["host"].nunique()),
        "nic_device_name_count": int(frame["device"].nunique()),
        "nic_entity_count": int(len(entity_meta)),
        "duplicate_cell_count": duplicate_count,
        "minimum_window_coverage_pct": float(
            frame["nic_window_coverage_pct"].min()
        ),
        "invalid_overlap_piece_count": int(
            frame["nic_invalid_overlap_piece_count"].sum()
        ),
        "host_sum_conservation": conservation_checks,
        "rank_to_nic_attribution_performed": False,
        "display_dimension": "host / NIC device",
    }
    if not all(checks):
        raise RuntimeError(json.dumps(clean_json(validation), ensure_ascii=False))
    return (
        frame.sort_values(
            ["behavior", "iteration", "nic_entity_index"]
        ),
        validation,
    )


def build_edp_rank_nic_frame(
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the dashboard-mapped rank/NIC EDP grid used by the page."""
    frame = pd.read_csv(source_path)
    expected_behaviors = {
        "expert_dp_grad_reduce_scatter",
        "expert_dp_param_allgather",
    }
    required_columns = {
        "behavior",
        "iteration",
        "rank",
        "node_index",
        "host",
        "local_rank",
        "gpu_id",
        "device",
        "nic_entity_id",
        "dashboard_mapping_status",
        "event_count",
        "expected_event_count",
        "data_status",
        "edp_union_segment_count",
        "edp_union_duration_ns",
        "nic_window_coverage_pct",
        "nic_long_gap_bidir_pct",
        "other_collective_overlap_pct",
        "nic_overlap_piece_count",
        "nic_invalid_overlap_piece_count",
        "nic_bidir_vs_iteration_median_ratio",
        "nic_bidir_percentile_in_iteration",
        "nic_tx_share_pct",
        "nic_tx_rx_imbalance_pct",
        *(str(item["column"]) for item in EDP_NIC_METRICS),
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"EDP rank/NIC source missing columns: {missing_columns}"
        )
    duplicate_count = int(
        frame.duplicated(["behavior", "iteration", "rank"]).sum()
    )
    frame["nic_entity_index"] = frame["rank"].astype(int)
    entity_meta = (
        frame[
            [
                "nic_entity_index",
                "nic_entity_id",
                "node_index",
                "rank",
                "host",
                "local_rank",
                "gpu_id",
                "device",
                "dashboard_mapping_status",
            ]
        ]
        .drop_duplicates()
        .sort_values("nic_entity_index")
    )
    status_counts = {
        str(key): int(value)
        for key, value in frame["data_status"].value_counts().items()
    }
    observed = frame[
        frame["data_status"].ne("missing_source_event")
    ]
    missing = frame[frame["data_status"].eq("missing_source_event")]
    metric_columns = [
        str(item["column"]) for item in EDP_NIC_METRICS
    ]
    checks = [
        len(frame)
        == len(expected_behaviors) * len(ITERATIONS) * len(RANKS),
        set(frame["behavior"]) == expected_behaviors,
        set(frame["iteration"]) == set(ITERATIONS),
        set(frame["rank"]) == set(RANKS),
        frame["host"].nunique() == 32,
        len(entity_meta) == len(RANKS),
        set(entity_meta["nic_entity_index"]) == set(RANKS),
        entity_meta["nic_entity_id"].nunique() == len(RANKS),
        entity_meta["dashboard_mapping_status"]
        .eq("CARRYOVER_VALIDATED_SLOT_MAP").all(),
        entity_meta.groupby("host")["device"].nunique().eq(8).all(),
        duplicate_count == 0,
        observed[metric_columns].notna().all().all(),
        missing[metric_columns].isna().all().all(),
        observed["nic_window_coverage_pct"].ge(99.0).all(),
        observed["nic_invalid_overlap_piece_count"].eq(0).all(),
    ]
    validation = {
        "schema_version": "256gpu-edp-nic-rank-device-spatial-v1",
        "source_csv": portable_path(source_path),
        "row_count": int(len(frame)),
        "expected_row_count": (
            len(expected_behaviors) * len(ITERATIONS) * len(RANKS)
        ),
        "behavior_count": int(frame["behavior"].nunique()),
        "iteration_count": int(frame["iteration"].nunique()),
        "rank_count": int(frame["rank"].nunique()),
        "host_count": int(frame["host"].nunique()),
        "physical_nic_entity_count": int(len(entity_meta)),
        "duplicate_cell_count": duplicate_count,
        "data_status_counts": status_counts,
        "minimum_observed_window_coverage_pct": float(
            observed["nic_window_coverage_pct"].min()
        ),
        "invalid_overlap_piece_count": int(
            observed["nic_invalid_overlap_piece_count"].sum()
        ),
        "rank_to_nic_attribution_performed": True,
        "rank_to_nic_mapping_source": (
            "GPU-slot mapping carried from the raw-point-validated 224-GPU "
            "hardware-family case; no fresh 256-page dashboard audit"
        ),
        "display_bandwidth_unit": "decimal Gbit/s (Gb/s)",
        "display_dimension": "global rank / GPU / mapped NIC device",
    }
    if not all(checks):
        raise RuntimeError(json.dumps(clean_json(validation), ensure_ascii=False))
    return (
        frame.sort_values(
            ["behavior", "iteration", "nic_entity_index"]
        ),
        validation,
    )


def build_dp_active_frame(
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the DP MTLink/NIC Trace-window and active-time grid."""
    frame = pd.read_csv(source_path)
    expected_behaviors = {
        "dp_grad_reduce_scatter",
        "dp_param_allgather",
    }
    metric_columns = {str(item["column"]) for item in DP_ACTIVE_METRICS}
    detail_columns = {
        "behavior",
        "iteration",
        "rank",
        "host",
        "local_rank",
        "gpu_id",
        "nic_device",
        "nic_entity_id",
        "nic_mapping_status",
        "event_count",
        "expected_event_count",
        "data_status",
        "trace_union_segment_count",
        "trace_event_union_duration_ns",
        "mtlink_long_gap_bidir_pct",
        "nic_long_gap_bidir_pct",
        "mtlink_active_bidir_fraction_of_trace_pct",
        "nic_active_bidir_fraction_of_trace_pct",
        "mtlink_touched_sample_count",
        "nic_touched_sample_count",
        "nic_mtlink_active_window_tx_coverage_pct",
        "nic_mtlink_active_window_rx_coverage_pct",
        "nic_mtlink_active_window_bidir_coverage_pct",
    }
    for metric in DP_ACTIVE_METRICS:
        detail_columns.add(str(metric["bytes_column"]))
        detail_columns.add(str(metric["duration_column"]))
        detail_columns.add(str(metric["segment_column"]))
        for optional in (
            "fraction_column",
            "long_gap_column",
            "touched_column",
        ):
            if optional in metric:
                detail_columns.add(str(metric[optional]))
    missing_columns = sorted(
        (metric_columns | detail_columns) - set(frame.columns)
    )
    if missing_columns:
        raise ValueError(f"DP active source missing columns: {missing_columns}")
    duplicate_count = int(
        frame.duplicated(["behavior", "iteration", "rank"]).sum()
    )
    observed = frame[frame["data_status"].ne("missing_source_event")]
    missing = frame[frame["data_status"].eq("missing_source_event")]
    checks = [
        len(frame) == len(expected_behaviors) * len(ITERATIONS) * len(RANKS),
        set(frame["behavior"]) == expected_behaviors,
        set(frame["iteration"]) == set(ITERATIONS),
        set(frame["rank"]) == set(RANKS),
        frame["host"].nunique() == 32,
        frame["nic_entity_id"].nunique() == len(RANKS),
        frame.groupby("host")["nic_device"].nunique().eq(8).all(),
        frame["nic_mapping_status"]
        .eq("CARRYOVER_VALIDATED_SLOT_MAP")
        .all(),
        duplicate_count == 0,
        observed[list(metric_columns)].notna().all().all(),
        missing[list(metric_columns)].isna().all().all(),
        observed["mtlink_active_bidir_duration_ns"]
        .le(observed["trace_event_union_duration_ns"])
        .all(),
        observed["nic_active_bidir_duration_ns"]
        .le(observed["trace_event_union_duration_ns"])
        .all(),
        observed["nic_mtlink_active_window_tx_duration_ns"]
        .eq(observed["mtlink_active_tx_duration_ns"])
        .all(),
        observed["nic_mtlink_active_window_rx_duration_ns"]
        .eq(observed["mtlink_active_rx_duration_ns"])
        .all(),
        observed["nic_mtlink_active_window_bidir_duration_ns"]
        .eq(observed["mtlink_active_bidir_duration_ns"])
        .all(),
        observed[
            [
                "nic_mtlink_active_window_tx_coverage_pct",
                "nic_mtlink_active_window_rx_coverage_pct",
                "nic_mtlink_active_window_bidir_coverage_pct",
            ]
        ]
        .ge(99.0)
        .all()
        .all(),
    ]
    validation = {
        "schema_version": "256gpu-dp-active-physical-heatmap-v4",
        "source_csv": portable_path(source_path),
        "row_count": int(len(frame)),
        "expected_row_count": (
            len(expected_behaviors) * len(ITERATIONS) * len(RANKS)
        ),
        "behavior_count": int(frame["behavior"].nunique()),
        "iteration_count": int(frame["iteration"].nunique()),
        "rank_count": int(frame["rank"].nunique()),
        "host_count": int(frame["host"].nunique()),
        "physical_nic_entity_count": int(frame["nic_entity_id"].nunique()),
        "duplicate_cell_count": duplicate_count,
        "data_status_counts": {
            str(key): int(value)
            for key, value in frame["data_status"].value_counts().items()
        },
        "metric_count": len(DP_ACTIVE_METRICS),
        "display_denominators_are_domain_native": True,
        "mtlink_gated_common_window_metrics_displayed": False,
        "nic_own_active_tx_rx_metrics_displayed": True,
        "source_audit_minimum_common_window_nic_coverage_pct": float(
            observed[
                [
                    "nic_mtlink_active_window_tx_coverage_pct",
                    "nic_mtlink_active_window_rx_coverage_pct",
                    "nic_mtlink_active_window_bidir_coverage_pct",
                ]
            ].min().min()
        ),
        "rank_to_nic_mapping_source": (
            "GPU-slot mapping carried from the raw-point-validated 224-GPU "
            "hardware-family case"
        ),
    }
    if not all(checks):
        raise RuntimeError(json.dumps(clean_json(validation), ensure_ascii=False))
    return (
        frame.sort_values(["behavior", "iteration", "rank"]),
        validation,
    )


def build_cp_active_frame(
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the CP MTLink native-active grid (CP has no NIC view)."""
    frame = pd.read_csv(source_path)
    expected_behavior = "cp_all_to_all"
    source_metric_columns = {
        str(item.get("source_column", item["column"]))
        for item in CP_ACTIVE_METRICS
    }
    detail_columns = {
        "behavior",
        "iteration",
        "rank",
        "host",
        "local_rank",
        "gpu_id",
        "pp_stage",
        "cp_group",
        "cp_rank",
        "kernel_name",
        "event_count",
        "expected_event_count",
        "data_status",
        "kernel_union_duration_ns",
        "kernel_span_ns",
        "mtlink_long_gap_bidir_pct",
        "mtlink_touched_sample_count",
    }
    for metric in CP_ACTIVE_METRICS:
        detail_columns.add(str(metric["bytes_column"]))
        detail_columns.add(str(metric["duration_column"]))
        detail_columns.add(str(metric["segment_column"]))
        for optional in ("long_gap_column", "touched_column"):
            if optional in metric:
                detail_columns.add(str(metric[optional]))
    missing_columns = sorted(
        (source_metric_columns | detail_columns) - set(frame.columns)
    )
    if missing_columns:
        raise ValueError(f"CP active source missing columns: {missing_columns}")

    duplicate_count = int(
        frame.duplicated(["behavior", "iteration", "rank"]).sum()
    )
    observed = frame[frame["data_status"].ne("missing_source_event")]
    missing = frame[frame["data_status"].eq("missing_source_event")]
    expected_kernel = (
        "mcclKernel_SendRecv_RING_SIMPLE_Sum_int8_t(mcclWorkElem)"
    )
    cp_rank_meta = frame[
        ["rank", "host", "cp_group", "cp_rank"]
    ].drop_duplicates()
    cp_group_shape = cp_rank_meta.groupby("cp_group").agg(
        rank_count=("rank", "nunique"),
        host_count=("host", "nunique"),
        cp_rank_count=("cp_rank", "nunique"),
    )
    cp_groups_two_ranks_one_host = bool(
        cp_group_shape["rank_count"].eq(2).all()
        and cp_group_shape["host_count"].eq(1).all()
        and cp_group_shape["cp_rank_count"].eq(2).all()
    )
    checks = [
        len(frame) == len(ITERATIONS) * len(RANKS),
        set(frame["behavior"]) == {expected_behavior},
        set(frame["iteration"]) == set(ITERATIONS),
        set(frame["rank"]) == set(RANKS),
        frame["host"].nunique() == 32,
        len(cp_group_shape) == 128,
        cp_groups_two_ranks_one_host,
        duplicate_count == 0,
        observed[list(source_metric_columns)].notna().all().all(),
        missing[list(source_metric_columns)].isna().all().all(),
        observed["kernel_name"].eq(expected_kernel).all(),
        observed["mtlink_active_tx_duration_ns"]
        .le(observed["kernel_union_duration_ns"])
        .all(),
        observed["mtlink_active_rx_duration_ns"]
        .le(observed["kernel_union_duration_ns"])
        .all(),
    ]
    validation = {
        "schema_version": "256gpu-cp-active-mtlink-heatmap-v1",
        "source_csv": portable_path(source_path),
        "row_count": int(len(frame)),
        "expected_row_count": len(ITERATIONS) * len(RANKS),
        "behavior_count": int(frame["behavior"].nunique()),
        "iteration_count": int(frame["iteration"].nunique()),
        "rank_count": int(frame["rank"].nunique()),
        "host_count": int(frame["host"].nunique()),
        "cp_group_count": int(len(cp_group_shape)),
        "cp_groups_two_ranks_one_host": cp_groups_two_ranks_one_host,
        "duplicate_cell_count": duplicate_count,
        "data_status_counts": {
            str(key): int(value)
            for key, value in frame["data_status"].value_counts().items()
        },
        "metric_count": len(CP_ACTIVE_METRICS),
        "kernel_name": expected_kernel,
        "physical_denominator": "MTLink direction-native active union",
        "nic_source_read": False,
        "nic_metrics_displayed": False,
        "legacy_kernel_window_mtlink_metrics_displayed": False,
        "tx_plus_rx_metric_displayed": False,
    }
    if not all(checks):
        raise RuntimeError(json.dumps(clean_json(validation), ensure_ascii=False))
    return (
        frame.sort_values(["behavior", "iteration", "rank"]),
        validation,
    )


def nic_device_matrix(
    frame: pd.DataFrame,
    behavior: str,
    column: str,
    entity_indices: list[int],
) -> list[list[Any]]:
    values = (
        frame[frame["behavior"].eq(behavior)]
        .pivot(index="nic_entity_index", columns="iteration", values=column)
        .reindex(index=entity_indices, columns=ITERATIONS)
    )
    return clean_json(values.to_numpy().tolist())


def build_payload(
    frame: pd.DataFrame,
    validation: dict[str, Any],
    edp_nic_device: pd.DataFrame,
    dp_active: pd.DataFrame,
    cp_active: pd.DataFrame,
) -> dict:
    rank_meta_frame = (
        frame[
            ["rank", "host", "local_rank", "gpu_id", "node_index"]
        ]
        .drop_duplicates()
        .sort_values("rank")
    )
    rank_nic_meta = (
        edp_nic_device[
            [
                "rank",
                "device",
                "nic_entity_id",
                "dashboard_mapping_status",
            ]
        ]
        .drop_duplicates("rank")
        .rename(columns={"device": "nic_device"})
    )
    rank_meta = rank_meta_frame.merge(
        rank_nic_meta, on="rank", how="left", validate="one_to_one"
    ).to_dict("records")
    nic_entity_meta = (
        edp_nic_device[
            [
                "nic_entity_index",
                "nic_entity_id",
                "node_index",
                "rank",
                "host",
                "local_rank",
                "gpu_id",
                "device",
                "dashboard_mapping_status",
            ]
        ]
        .drop_duplicates()
        .sort_values("nic_entity_index")
        .to_dict("records")
    )
    nic_entity_indices = [
        int(item["nic_entity_index"]) for item in nic_entity_meta
    ]
    payload: dict[str, Any] = {
        "schema_version": validation["schema_version"],
        "iterations": list(ITERATIONS),
        "ranks": list(RANKS),
        "rank_meta": clean_json(rank_meta),
        "nic_entity_indices": nic_entity_indices,
        "nic_entity_meta": clean_json(nic_entity_meta),
        # Put each specialized physical view first: DP and CP open on their
        # domain-native active metrics and Expert-DP opens on mapped NIC.
        "metrics": list(DP_ACTIVE_METRICS)
        + list(CP_ACTIVE_METRICS)
        + list(EDP_NIC_METRICS)
        + list(METRICS),
        "behaviors": [],
        "data_status_counts": validation["data_status_counts"],
        "ep_mtlink_attribution": validation["ep_mtlink_attribution"],
    }
    for behavior in BEHAVIORS:
        name = str(behavior["behavior"])
        item = dict(behavior)
        item["has_mtlink_quality"] = str(behavior["family"]) not in {
            "CP",
            "DP",
            "Expert-DP",
        }
        base_metrics = (
            TRACE_METRICS
            if str(behavior["family"]) in {"CP", "DP", "Expert-DP"}
            else METRICS
        )
        item["metrics"] = {
            metric["column"]: matrix(frame, name, str(metric["column"]))
            for metric in base_metrics
        }
        detail_columns = [
            "event_count",
            "duration_ns_sum",
            "data_status",
        ]
        if item["has_mtlink_quality"]:
            detail_columns.extend(
                [
                    "mtlink_long_gap_bidir_pct",
                    "mtlink_observed_link_count_min",
                    "mtlink_normalized_overlap_count",
                ]
            )
        item["details"] = {
            column: matrix(frame, name, column)
            for column in detail_columns
        }
        nic_behavior = edp_nic_device[
            edp_nic_device["behavior"].eq(name)
        ]
        if not nic_behavior.empty:
            item["metrics"].update(
                {
                    metric["column"]: nic_device_matrix(
                        edp_nic_device,
                        name,
                        str(metric["column"]),
                        nic_entity_indices,
                    )
                    for metric in EDP_NIC_METRICS
                }
            )
            nic_detail_columns = (
                "event_count",
                "expected_event_count",
                "edp_union_segment_count",
                "edp_union_duration_ns",
                "nic_window_coverage_pct",
                "nic_long_gap_bidir_pct",
                "other_collective_overlap_pct",
                "nic_overlap_piece_count",
                "nic_bidir_vs_iteration_median_ratio",
                "nic_bidir_percentile_in_iteration",
                "nic_tx_share_pct",
                "nic_tx_rx_imbalance_pct",
                "data_status",
            )
            item["nic_details"] = {
                column: nic_device_matrix(
                    edp_nic_device,
                    name,
                    column,
                    nic_entity_indices,
                )
                for column in nic_detail_columns
            }
        dp_behavior = dp_active[dp_active["behavior"].eq(name)]
        if not dp_behavior.empty:
            item["metrics"].update(
                {
                    metric["column"]: matrix(
                        dp_active, name, str(metric["column"])
                    )
                    for metric in DP_ACTIVE_METRICS
                }
            )
            dp_detail_columns = {
                "event_count",
                "expected_event_count",
                "trace_event_union_duration_ns",
                "mtlink_long_gap_bidir_pct",
                "mtlink_touched_sample_count",
                "data_status",
            }
            for metric in DP_ACTIVE_METRICS:
                dp_detail_columns.add(str(metric["bytes_column"]))
                dp_detail_columns.add(str(metric["duration_column"]))
                dp_detail_columns.add(str(metric["segment_column"]))
                for optional in (
                    "fraction_column",
                    "long_gap_column",
                    "touched_column",
                ):
                    if optional in metric:
                        dp_detail_columns.add(str(metric[optional]))
            item["dp_active_details"] = {
                column: matrix(dp_active, name, column)
                for column in sorted(dp_detail_columns)
            }
        cp_behavior = cp_active[cp_active["behavior"].eq(name)]
        if not cp_behavior.empty:
            item["metrics"].update(
                {
                    metric["column"]: matrix(
                        cp_active,
                        name,
                        str(metric.get("source_column", metric["column"])),
                    )
                    for metric in CP_ACTIVE_METRICS
                }
            )
            cp_detail_columns = {
                "event_count",
                "expected_event_count",
                "kernel_name",
                "kernel_union_duration_ns",
                "kernel_span_ns",
                "mtlink_long_gap_bidir_pct",
                "mtlink_touched_sample_count",
                "data_status",
            }
            for metric in CP_ACTIVE_METRICS:
                cp_detail_columns.add(str(metric["bytes_column"]))
                cp_detail_columns.add(str(metric["duration_column"]))
                cp_detail_columns.add(str(metric["segment_column"]))
                for optional in ("long_gap_column", "touched_column"):
                    if optional in metric:
                        cp_detail_columns.add(str(metric[optional]))
            item["cp_active_details"] = {
                column: matrix(cp_active, name, column)
                for column in sorted(cp_detail_columns)
            }
        payload["behaviors"].append(item)
    return clean_json(payload)


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>256-GPU Collective Communication Spatial Heatmap (No PP)</title>
<style>
:root {
  --ink: #162331;
  --muted: #607080;
  --panel: #ffffff;
  --line: #dbe3ea;
  --accent: #0d6863;
  --bg: #f3f6f8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font-family: "IBM Plex Sans", "Noto Sans CJK SC", system-ui, sans-serif;
}
main { max-width: 1760px; margin: 0 auto; padding: 22px 28px 42px; }
h1 { margin: 0; font-size: 26px; letter-spacing: -0.02em; }
.subtitle { color: var(--muted); margin: 6px 0 18px; }
.reference-link {
  display: inline-block;
  margin: -6px 0 16px;
  color: var(--accent);
  font-weight: 650;
  text-decoration: none;
}
.reference-link:hover { text-decoration: underline; }
.controls {
  display: grid;
  grid-template-columns: minmax(260px, 1.5fr) minmax(220px, 1fr) 180px;
  gap: 12px;
  padding: 16px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
}
label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
select {
  width: 100%;
  height: 38px;
  padding: 0 10px;
  border: 1px solid #bfcbd5;
  border-radius: 7px;
  background: #fff;
  color: var(--ink);
  font-size: 14px;
}
.cards {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 10px;
  margin: 12px 0;
}
.card {
  min-height: 70px;
  padding: 10px 12px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
}
.card .name { color: var(--muted); font-size: 11px; }
.card .value { margin-top: 7px; font: 600 18px "IBM Plex Mono", monospace; }
.plot-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}
#heatmap { width: 100%; height: 1540px; }
.notes {
  margin-top: 12px;
  padding: 14px 16px;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  line-height: 1.6;
  font-size: 13px;
}
.status-observed { color: #18794e; }
.status-warning { color: #b45309; }
@media (max-width: 900px) {
  main { padding: 14px; }
  .controls { grid-template-columns: 1fr; }
  .cards { grid-template-columns: repeat(2, 1fr); }
}
</style>
<script>__PLOTLY_JS__</script>
</head>
<body>
<main>
  <h1>256-GPU 集合通信带宽空间热力图__TITLE_SUFFIX__</h1>
  <div class="subtitle">
    PP=16 / CP=2 / EP=8 · No PP view · 256 rank 级 Trace/MTLink · DP/Expert-DP mapped-NIC__SUBTITLE_SUFFIX__
  </div>
  <section class="controls">
    <div>
      <label for="behavior-select">集合通信</label>
      <select id="behavior-select"></select>
    </div>
    <div>
      <label for="metric-select">带宽口径</label>
      <select id="metric-select"></select>
    </div>
    <div>
      <label for="scale-select">颜色上限</label>
      <select id="scale-select">
        <option value="p99">P99（突出空间热点）</option>
        <option value="max">实际最大值</option>
      </select>
    </div>
  </section>

  <section class="cards">
    <div class="card"><div class="name">有效单元</div><div class="value" id="valid-cells">—</div></div>
    <div class="card"><div class="name">缺失 / 部分</div><div class="value" id="quality-cells">—</div></div>
    <div class="card"><div class="name">中位数</div><div class="value" id="median-value">—</div></div>
    <div class="card"><div class="name">P95</div><div class="value" id="p95-value">—</div></div>
    <div class="card"><div class="name">最大值</div><div class="value" id="max-value">—</div></div>
    <div class="card"><div class="name">最大值位置</div><div class="value" id="max-location">—</div></div>
  </section>

  <section class="plot-panel"><div id="heatmap"></div></section>
  <section class="notes">
    Trace logical/expected 保留为算法对照。EP 的 MTLink 单元格为
    <code>Σbytes / Σevent_duration_ns</code>，单位 decimal GB/s；Expert-DP
    不展示 MTLink，只展示 Trace 对照和 mapped-NIC TX/RX/TX+RX。
    NIC 字节积分结果乘 8，以 Gb/s 展示，便于和链路标称能力比较。
    CP group size=2 且每组两张 GPU 均在同一 host，因此 CP 不展示 NIC。
    CP 物理页只展示 MTLink TX/RX：先限制在 Trace CP kernel union 内，按每链路
    0.05 GB/s 阈值选取 active sample interval，再对 14 条链路的 active 时间取并集，
    TX/RX 分别用自己的 active union 作分母。CP 实际 GPU kernel 为
    <code>mcclKernel_SendRecv_RING_SIMPLE_Sum_int8_t(mcclWorkElem)</code>。
    counter reset/underflow 行及同链路紧随其后的 recovery 行不参与积分。
    DP RS/AG 不再展示旧 MTLink/NIC Trace-window、共同窗口或任何 TX+RX 页面，
    只保留两个物理域各自的 active TX/RX。MTLink 使用 MTLink CSV：在 Trace DP
    event union 内选出每链路速率不低于 0.05 GB/s 的 active sample interval，
    对 14 条链路的时间做并集，并以该 MTLink union 为分母。mapped-NIC 使用
    RDMA NIC CSV：以 1 Gb/s 阈值选择 NIC 自己的 active sample interval，并以
    NIC active union 为分母。两者的分母互不替代。active sample union 受采样
    粒度约束，是 interval-censored 的活动时间估计，不等于逐 kernel 精确起止。
    Trace/MTLink 的纵轴为 256 个 global rank；
    EDP-NIC 的纵轴也为 256 个 global rank，每行同时标注对应 GPU 和物理 NIC。
    EP 为 All MoE layers 聚合。__EP_ATTRIBUTION_NOTE__
    灰色空单元表示原始Trace缺少事件，未做带宽补值。
    rank→NIC 使用上一 224 卡同硬件族已全量验证的 GPU-slot 映射
    GPU0–7 → mlx5_0/1/2/3/6/7/8/9；本 256 卡原始数据未提供 256 份 dashboard，
    因此这是明确标记的 carry-over mapping，不是本次全量页面复验。
    EDP-NIC 使用每个 rank 自己的 EDP 事件有效窗口，
    每行数值只来自标注的 <code>host / GPU / mlx5_x</code>。
    该映射和时间对齐可证明采样来源与窗口一致，但不能证明窗口内所有 NIC
    字节均由该集合通信独占产生；悬浮信息保留采样长间隔和其他通信重叠指标。
    NIC、MTLink、Trace logical 和 Trace expected 属于不同字节域，不应相加。
  </section>
</main>
<script>
const DATA = __PAYLOAD_JSON__;
const COLORSCALE = [
  [0.00, "#00204C"],
  [0.20, "#2D3F73"],
  [0.45, "#596B78"],
  [0.65, "#8B8E68"],
  [0.82, "#C7B84B"],
  [1.00, "#F9E547"]
];
const behaviorSelect = document.getElementById("behavior-select");
const metricSelect = document.getElementById("metric-select");
const scaleSelect = document.getElementById("scale-select");

const familyGroups = new Map();
for (const behavior of DATA.behaviors) {
  if (!familyGroups.has(behavior.family)) {
    const group = document.createElement("optgroup");
    group.label = behavior.family;
    familyGroups.set(behavior.family, group);
    behaviorSelect.appendChild(group);
  }
  const option = document.createElement("option");
  option.value = behavior.behavior;
  option.textContent = `${behavior.label} — ${behavior.description}`;
  familyGroups.get(behavior.family).appendChild(option);
}
function percentile(values, probability) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function fmt(value, digits = 2) {
  return value === null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function currentBehavior() {
  return DATA.behaviors.find(item => item.behavior === behaviorSelect.value);
}

function currentMetric() {
  return DATA.metrics.find(item => item.column === metricSelect.value);
}

function displayMetricLabel(behavior, metric) {
  if (behavior.family === "EP" && metric.column === "trace_expected_tx_bandwidth_GBps") {
    return "DeepEP Algo-expected TX";
  }
  if (behavior.family === "EP" && metric.column === "trace_expected_rx_bandwidth_GBps") {
    return "DeepEP Algo-expected RX";
  }
  const strictEpMtlink =
    behavior.family === "EP"
    && metric.column.startsWith("mtlink_")
    && DATA.ep_mtlink_attribution === "strict-serial-phase";
  return strictEpMtlink
    ? `${metric.label}（EP严格串行阶段约束）`
    : metric.label;
}

function metricUnit(metric) {
  return metric.unit || (metric.dimension === "rank_nic" ? "Gb/s" : "GB/s");
}

function populateMetrics() {
  const behavior = currentBehavior();
  const previous = metricSelect.value;
  metricSelect.replaceChildren();
  for (const metric of DATA.metrics) {
    if (!(metric.column in behavior.metrics)) continue;
    const option = document.createElement("option");
    option.value = metric.column;
    option.textContent = `${displayMetricLabel(behavior, metric)} — ${metric.description}`;
    metricSelect.appendChild(option);
  }
  if ([...metricSelect.options].some(option => option.value === previous)) {
    metricSelect.value = previous;
  } else {
    metricSelect.selectedIndex = 0;
  }
}

function buildRankCustomData(behavior) {
  const details = behavior.details;
  return DATA.ranks.map((rank, rankIndex) =>
    DATA.iterations.map((iteration, iterationIndex) => {
      const meta = DATA.rank_meta[rankIndex];
      const duration = details.duration_ns_sum[rankIndex][iterationIndex];
      return [
        meta.host,
        meta.gpu_id,
        meta.local_rank,
        details.event_count[rankIndex][iterationIndex],
        duration === null ? null : duration / 1e6,
        details.mtlink_long_gap_bidir_pct
          ? details.mtlink_long_gap_bidir_pct[rankIndex][iterationIndex]
          : null,
        details.mtlink_observed_link_count_min
          ? details.mtlink_observed_link_count_min[rankIndex][iterationIndex]
          : null,
        details.mtlink_normalized_overlap_count
          ? details.mtlink_normalized_overlap_count[rankIndex][iterationIndex]
          : null,
        details.data_status[rankIndex][iterationIndex]
      ];
    })
  );
}

function buildNicDeviceCustomData(behavior) {
  const details = behavior.nic_details;
  return DATA.nic_entity_indices.map((entityIndex, nicIndex) =>
    DATA.iterations.map((iteration, iterationIndex) => {
      const meta = DATA.nic_entity_meta[nicIndex];
      const duration = details.edp_union_duration_ns[nicIndex][iterationIndex];
      return [
        meta.host,
        meta.rank,
        meta.gpu_id,
        meta.local_rank,
        meta.device,
        details.event_count[nicIndex][iterationIndex],
        details.expected_event_count[nicIndex][iterationIndex],
        details.edp_union_segment_count[nicIndex][iterationIndex],
        duration === null ? null : duration / 1e6,
        details.nic_window_coverage_pct[nicIndex][iterationIndex],
        details.nic_long_gap_bidir_pct[nicIndex][iterationIndex],
        details.other_collective_overlap_pct[nicIndex][iterationIndex],
        details.nic_overlap_piece_count[nicIndex][iterationIndex],
        details.nic_bidir_vs_iteration_median_ratio[nicIndex][iterationIndex],
        details.nic_bidir_percentile_in_iteration[nicIndex][iterationIndex],
        details.nic_tx_share_pct[nicIndex][iterationIndex],
        details.nic_tx_rx_imbalance_pct[nicIndex][iterationIndex],
        details.data_status[nicIndex][iterationIndex]
      ];
    })
  );
}

function buildDpActiveCustomData(behavior, metric) {
  const details = behavior.dp_active_details;
  const domainPrefix = metric.domain === "MTLink" ? "mtlink" : "nic";
  const longGapColumn = metric.long_gap_column || `${domainPrefix}_long_gap_bidir_pct`;
  const touchedColumn = metric.touched_column || `${domainPrefix}_touched_sample_count`;
  return DATA.ranks.map((rank, rankIndex) =>
    DATA.iterations.map((iteration, iterationIndex) => {
      const meta = DATA.rank_meta[rankIndex];
      const traceDuration = details.trace_event_union_duration_ns[rankIndex][iterationIndex];
      const denominator = details[metric.duration_column][rankIndex][iterationIndex];
      const bytes = details[metric.bytes_column][rankIndex][iterationIndex];
      const segments = details[metric.segment_column][rankIndex][iterationIndex];
      return [
        meta.host,
        meta.gpu_id,
        meta.local_rank,
        meta.nic_device,
        details.event_count[rankIndex][iterationIndex],
        details.expected_event_count[rankIndex][iterationIndex],
        traceDuration === null ? null : traceDuration / 1e6,
        denominator === null ? null : denominator / 1e6,
        bytes === null ? null : bytes / 1e9,
        segments,
        traceDuration === null || denominator === null || traceDuration === 0
          ? null
          : denominator / traceDuration * 100,
        details[longGapColumn][rankIndex][iterationIndex],
        details[touchedColumn][rankIndex][iterationIndex],
        details.data_status[rankIndex][iterationIndex]
      ];
    })
  );
}

function buildCpActiveCustomData(behavior, metric) {
  const details = behavior.cp_active_details;
  const longGapColumn = metric.long_gap_column || "mtlink_long_gap_bidir_pct";
  const touchedColumn = metric.touched_column || "mtlink_touched_sample_count";
  return DATA.ranks.map((rank, rankIndex) =>
    DATA.iterations.map((iteration, iterationIndex) => {
      const meta = DATA.rank_meta[rankIndex];
      const kernelUnion = details.kernel_union_duration_ns[rankIndex][iterationIndex];
      const kernelSpan = details.kernel_span_ns[rankIndex][iterationIndex];
      const denominator = details[metric.duration_column][rankIndex][iterationIndex];
      const bytes = details[metric.bytes_column][rankIndex][iterationIndex];
      const segments = details[metric.segment_column][rankIndex][iterationIndex];
      return [
        meta.host,
        meta.gpu_id,
        meta.local_rank,
        details.kernel_name[rankIndex][iterationIndex],
        details.event_count[rankIndex][iterationIndex],
        details.expected_event_count[rankIndex][iterationIndex],
        kernelUnion === null ? null : kernelUnion / 1e6,
        kernelSpan === null ? null : kernelSpan / 1e6,
        denominator === null ? null : denominator / 1e6,
        bytes === null ? null : bytes / 1e9,
        segments,
        kernelUnion === null || denominator === null || kernelUnion === 0
          ? null
          : denominator / kernelUnion * 100,
        details[longGapColumn][rankIndex][iterationIndex],
        details[touchedColumn][rankIndex][iterationIndex],
        details.data_status[rankIndex][iterationIndex]
      ];
    })
  );
}

function render() {
  const behavior = currentBehavior();
  const metric = currentMetric();
  const metricLabel = displayMetricLabel(behavior, metric);
  const unit = metricUnit(metric);
  const nicDeviceDimension = metric.dimension === "rank_nic";
  const dpActiveMetric = metric.source === "dp_active";
  const cpActiveMetric = metric.source === "cp_active";
  const axisValues = nicDeviceDimension
    ? DATA.nic_entity_indices
    : DATA.ranks;
  const details = nicDeviceDimension
    ? behavior.nic_details
    : (dpActiveMetric
        ? behavior.dp_active_details
        : (cpActiveMetric ? behavior.cp_active_details : behavior.details));
  const z = behavior.metrics[metric.column];
  const finite = [];
  let missing = 0;
  let partial = 0;
  let maxValue = -Infinity;
  let maxAxisValue = null;
  let maxIteration = null;
  for (let axisIndex = 0; axisIndex < z.length; axisIndex++) {
    for (let iterationIndex = 0; iterationIndex < z[axisIndex].length; iterationIndex++) {
      const value = z[axisIndex][iterationIndex];
      const status = details.data_status[axisIndex][iterationIndex];
      if (status === "missing_source_event") missing++;
      if (status === "partial_source_event") partial++;
      if (value !== null && Number.isFinite(value)) {
        finite.push(value);
        if (value > maxValue) {
          maxValue = value;
          maxAxisValue = nicDeviceDimension
            ? DATA.nic_entity_meta[axisIndex].nic_entity_id
            : axisValues[axisIndex];
          maxIteration = DATA.iterations[iterationIndex];
        }
      }
    }
  }
  const p99 = percentile(finite, 0.99);
  const actualMax = finite.length ? Math.max(...finite) : 1;
  const zMax = scaleSelect.value === "p99" ? p99 : actualMax;
  const customdata = nicDeviceDimension
    ? buildNicDeviceCustomData(behavior)
    : (dpActiveMetric
        ? buildDpActiveCustomData(behavior, metric)
        : (cpActiveMetric
            ? buildCpActiveCustomData(behavior, metric)
            : buildRankCustomData(behavior)));
  const hover = nicDeviceDimension
    ? [
        `<b>${behavior.label} · rank-mapped NIC window</b>`,
        `${behavior.description}`,
        `Iteration %{x}`,
        `Host: %{customdata[0]}`,
        `Rank / GPU / local rank: r%{customdata[1]} / GPU%{customdata[2]} / %{customdata[3]}`,
        `Mapped NIC: %{customdata[4]}`,
        `${metricLabel}: %{z:.3f} ${unit}`,
        `Events: %{customdata[5]} / %{customdata[6]}`,
        `Rank EDP segments: %{customdata[7]}`,
        `Rank EDP union time: %{customdata[8]:.3f} ms`,
        `NIC window coverage: %{customdata[9]:.2f}%`,
        `Long-gap bytes: %{customdata[10]:.2f}%`,
        `Other collective overlap: %{customdata[11]:.2f}%`,
        `Sample overlap pieces: %{customdata[12]}`,
        `vs iteration median: %{customdata[13]:.3f}×`,
        `Iteration percentile: %{customdata[14]:.1%}`,
        `TX share: %{customdata[15]:.2f}%`,
        `TX/RX imbalance: %{customdata[16]:.2f}%`,
        `Data status: %{customdata[17]}`,
        "<extra></extra>"
      ].join("<br>")
    : dpActiveMetric
    ? [
        `<b>${behavior.label} · ${metric.domain}</b>`,
        `${behavior.description}`,
        `Iteration %{x}`,
        `Rank %{y}`,
        `Host: %{customdata[0]}`,
        `GPU/local rank: %{customdata[1]} / %{customdata[2]}`,
        `Mapped NIC: %{customdata[3]}`,
        `${metricLabel}: %{z:.3f} ${unit}`,
        `Events: %{customdata[4]} / %{customdata[5]}`,
        `Trace union time: %{customdata[6]:.3f} ms`,
        `Metric denominator: %{customdata[7]:.3f} ms`,
        `Attributed bytes: %{customdata[8]:.3f} GB`,
        `Denominator/trace ratio: %{customdata[10]:.2f}%`,
        `Active segments: %{customdata[9]}`,
        `Long-gap bytes: %{customdata[11]:.2f}%`,
        `Touched samples: %{customdata[12]}`,
        `Data status: %{customdata[13]}`,
        "<extra></extra>"
      ].join("<br>")
    : cpActiveMetric
    ? [
        `<b>${behavior.label} · MTLink native-active</b>`,
        `${behavior.description}`,
        `Iteration %{x}`,
        `Rank %{y}`,
        `Host: %{customdata[0]}`,
        `GPU/local rank: %{customdata[1]} / %{customdata[2]}`,
        `Kernel: %{customdata[3]}`,
        `${metricLabel}: %{z:.3f} ${unit}`,
        `Events: %{customdata[4]} / %{customdata[5]}`,
        `Kernel union time: %{customdata[6]:.3f} ms`,
        `First-to-last kernel span: %{customdata[7]:.3f} ms`,
        `MTLink active denominator: %{customdata[8]:.3f} ms`,
        `Attributed active bytes: %{customdata[9]:.3f} GB`,
        `Active/kernel union ratio: %{customdata[11]:.2f}%`,
        `Active segments: %{customdata[10]}`,
        `Long-gap bytes: %{customdata[12]:.2f}%`,
        `Touched samples: %{customdata[13]}`,
        `Data status: %{customdata[14]}`,
        "<extra></extra>"
      ].join("<br>")
    : behavior.has_mtlink_quality
    ? [
        `<b>${behavior.label}</b>`,
        `${behavior.description}`,
        `Iteration %{x}`,
        `Rank %{y}`,
        `Host: %{customdata[0]}`,
        `GPU/local rank: %{customdata[1]} / %{customdata[2]}`,
        `${metricLabel}: %{z:.3f} ${unit}`,
        `Events: %{customdata[3]}`,
        `Event time: %{customdata[4]:.3f} ms`,
        `Long-gap bytes: %{customdata[5]:.2f}%`,
        `Observed links min: %{customdata[6]}/14`,
        `Normalized overlaps: %{customdata[7]}`,
        `Data status: %{customdata[8]}`,
        "<extra></extra>"
      ].join("<br>")
    : [
        `<b>${behavior.label}</b>`,
        `${behavior.description}`,
        `Iteration %{x}`,
        `Rank %{y}`,
        `Host: %{customdata[0]}`,
        `GPU/local rank: %{customdata[1]} / %{customdata[2]}`,
        `${metricLabel}: %{z:.3f} ${unit}`,
        `Events: %{customdata[3]}`,
        `Event time: %{customdata[4]:.3f} ms`,
        `Data status: %{customdata[8]}`,
        "<extra></extra>"
      ].join("<br>");

  const nodeShapes = [];
  for (let entity = 7.5; entity < 223; entity += 8) {
    nodeShapes.push({
      type: "line", xref: "paper", x0: 0, x1: 1,
      yref: "y", y0: entity, y1: entity,
      line: {color: "rgba(255,255,255,0.72)", width: 1}
    });
  }
  const tickvals = [];
  const ticktext = [];
  if (nicDeviceDimension) {
    for (let index = 0; index < DATA.nic_entity_meta.length; index += 8) {
      const meta = DATA.nic_entity_meta[index];
      tickvals.push(index + 3.5);
      ticktext.push(`${meta.host}<br>r${meta.rank}–${meta.rank + 7}`);
    }
  } else {
    for (let index = 0; index < DATA.rank_meta.length; index += 8) {
      tickvals.push(index + 3.5);
      ticktext.push(`${DATA.rank_meta[index].host}<br>r${index}–${index + 7}`);
    }
  }
  document.getElementById("heatmap").style.height = "1540px";
  const trace = {
    type: "heatmap",
    x: DATA.iterations,
    y: axisValues,
    z,
    customdata,
    zmin: 0,
    zmax: zMax > 0 ? zMax : 1,
    colorscale: COLORSCALE,
    hoverongaps: false,
    hovertemplate: hover,
    colorbar: {
      title: {text: unit},
      thickness: 18,
      len: 0.82
    }
  };
  const layout = {
    title: {
      text: `${behavior.label}<br><sup>${behavior.family} · ${behavior.description} · ${metricLabel} · ${nicDeviceDimension ? "global rank / GPU / mapped NIC" : "global rank / GPU"}</sup>`,
      x: 0.01,
      xanchor: "left",
      font: {size: 22, color: "#173B3A"}
    },
    margin: {l: 170, r: 95, t: 95, b: 75},
    paper_bgcolor: "#FFFFFF",
    plot_bgcolor: "#D9DEE3",
    font: {family: "IBM Plex Sans, Noto Sans CJK SC, sans-serif", color: "#162331"},
    xaxis: {
      title: "Profiled iteration",
      tickmode: "array",
      tickvals: DATA.iterations,
      ticktext: DATA.iterations.map(String),
      side: "top",
      showgrid: false,
      ticks: "outside"
    },
    yaxis: {
      title: nicDeviceDimension
        ? "Global rank / GPU / mapped NIC"
        : "Global rank / host",
      tickmode: "array",
      tickvals,
      ticktext,
      autorange: "reversed",
      showgrid: false,
      ticks: "outside",
      fixedrange: false
    },
    shapes: nodeShapes,
    hoverlabel: {
      bgcolor: "#FFFFFF",
      bordercolor: "#27364A",
      font: {size: 13, color: "#172033"}
    }
  };
  Plotly.react("heatmap", [trace], layout, {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  });

  document.getElementById("valid-cells").textContent = finite.length.toLocaleString();
  document.getElementById("quality-cells").textContent = `${missing} / ${partial}`;
  document.getElementById("median-value").textContent =
    `${fmt(percentile(finite, 0.5))} ${unit}`;
  document.getElementById("p95-value").textContent =
    `${fmt(percentile(finite, 0.95))} ${unit}`;
  document.getElementById("max-value").textContent =
    `${fmt(maxValue)} ${unit}`;
  document.getElementById("max-location").textContent =
    maxAxisValue === null
      ? "—"
      : `${nicDeviceDimension ? maxAxisValue : `r${maxAxisValue}`} / i${maxIteration}`;
}

behaviorSelect.value = DATA.behaviors[0].behavior;
populateMetrics();
behaviorSelect.addEventListener("change", () => {
  populateMetrics();
  render();
});
metricSelect.addEventListener("change", render);
scaleSelect.addEventListener("change", render);
render();
</script>
</body>
</html>
"""


def render_html(
    frame: pd.DataFrame,
    validation: dict[str, Any],
    edp_nic_device: pd.DataFrame,
    dp_active: pd.DataFrame,
    cp_active: pd.DataFrame,
) -> str:
    payload = json.dumps(
        build_payload(
            frame, validation, edp_nic_device, dp_active, cp_active
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    strict_ep = (
        validation["ep_mtlink_attribution"] == "strict-serial-phase"
    )
    title_suffix = "（EP严格串行阶段约束版）" if strict_ep else ""
    subtitle_suffix = " · EP MTLink 严格串行阶段约束" if strict_ep else ""
    ep_note = (
        "EP Algo-expected TX 为 Σ DeepEP send_bytes / Σ DeepEP elapsed_ns，"
        "不乘额外算法因子。EP MTLink 使用严格串行阶段约束：每个相交采样桶的 TX/RX bytes "
        "只在该桶实际相交的 EP 通信阶段之间按 overlap 时长分配；它是归因估计，"
        "不是精确逐 kernel counter。"
        if strict_ep
        else (
            "EP MTLink 使用 sample/event 时间重叠占 sample interval 的比例归因。"
        )
    )
    return (
        HTML_TEMPLATE.replace("__PLOTLY_JS__", get_plotlyjs())
        .replace("__PAYLOAD_JSON__", payload)
        .replace("__TITLE_SUFFIX__", title_suffix)
        .replace("__SUBTITLE_SUFFIX__", subtitle_suffix)
        .replace("__EP_ATTRIBUTION_NOTE__", ep_note)
    )


def main() -> int:
    args = parse_args()
    frame, validation = build_tagged_csv(
        args.source_csv.resolve(), args.rank_topology.resolve()
    )
    validation["ep_mtlink_attribution"] = args.ep_mtlink_attribution
    edp_nic_device, edp_nic_device_validation = (
        build_edp_rank_nic_frame(
            args.edp_nic_device_source_csv.resolve(),
        )
    )
    dp_active, dp_active_validation = build_dp_active_frame(
        args.dp_active_source_csv.resolve()
    )
    cp_active, cp_active_validation = build_cp_active_frame(
        args.cp_active_source_csv.resolve()
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    validation["output_csv"] = portable_path(args.output_csv)
    # Render from the persisted CSV, not the in-memory frame, so the delivered
    # table is exactly the canonical source embedded in the standalone page.
    persisted_frame = pd.read_csv(args.output_csv)
    # EDP-NIC sources are also re-read from their persisted canonical CSVs.
    persisted_edp_nic_device, _ = build_edp_rank_nic_frame(
        args.edp_nic_device_source_csv.resolve(),
    )
    persisted_dp_active, _ = build_dp_active_frame(
        args.dp_active_source_csv.resolve()
    )
    persisted_cp_active, _ = build_cp_active_frame(
        args.cp_active_source_csv.resolve()
    )
    html = render_html(
        persisted_frame,
        validation,
        persisted_edp_nic_device,
        persisted_dp_active,
        persisted_cp_active,
    )
    atomic_text(args.html_output, html)
    validation["html_output"] = portable_path(args.html_output)
    validation["html_data_sources"] = [
        portable_path(args.output_csv),
        portable_path(args.edp_nic_device_source_csv),
        portable_path(args.dp_active_source_csv),
        portable_path(args.cp_active_source_csv),
    ]
    validation["edp_nic_rank_device"] = edp_nic_device_validation
    validation["dp_active_physical"] = dp_active_validation
    validation["cp_active_mtlink"] = cp_active_validation
    validation["html_size_bytes"] = args.html_output.stat().st_size
    validation["html_behavior_options"] = len(BEHAVIORS)
    validation["html_metric_options_total"] = (
        len(METRICS)
        + len(EDP_NIC_METRICS)
        + len(DP_ACTIVE_METRICS)
        + len(CP_ACTIVE_METRICS)
    )
    validation["html_metric_options_by_family"] = {
        "EP": len(METRICS),
        "CP": len(TRACE_METRICS) + len(CP_ACTIVE_METRICS),
        "DP": len(TRACE_METRICS) + len(DP_ACTIVE_METRICS),
        "Expert-DP": len(TRACE_METRICS) + len(EDP_NIC_METRICS),
    }
    validation["html_rank_metric_options"] = len(METRICS)
    validation["html_edp_nic_metric_options"] = len(EDP_NIC_METRICS)
    validation["html_dp_active_metric_options"] = len(DP_ACTIVE_METRICS)
    validation["html_cp_active_metric_options"] = len(CP_ACTIVE_METRICS)
    validation["html_cp_nic_metric_options"] = 0
    validation["html_cp_tx_plus_rx_metric_options"] = 0
    validation["html_cp_legacy_mtlink_metric_options"] = 0
    validation["html_dp_tx_plus_rx_metric_options"] = 0
    validation["html_dp_legacy_physical_metric_options"] = 0
    validation["html_dp_common_window_metric_options"] = 0
    validation["html_dp_nic_native_active_metric_options"] = 2
    validation["html_edp_mtlink_metric_options"] = 0
    validation["html_edp_nic_display_dimension"] = (
        "256 global-rank/GPU/mapped-NIC rows"
    )
    validation["html_self_contained_plotly"] = True
    atomic_text(
        args.validation_output,
        json.dumps(clean_json(validation), ensure_ascii=False, indent=2)
        + "\n",
    )
    print(json.dumps(clean_json(validation), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

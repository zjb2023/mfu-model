from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import load_run_config
from .headroom import (
    COMMUNICATION_STAGES,
    compute_joint_counterfactual,
    compute_stage_headroom,
)
from .spatiotemporal import robust_two_way_decomposition


SCHEMA_VERSION = "run1555-html-payload-v5.2"
GENERATOR_VERSION = "1.3.0"
_INPUT_PATHS = (
    "configs/run1555_v5.toml",
    "derived_v5/iter_summary_all99_v5.csv",
    "derived_v5/iter_outlier_analysis_v5.csv",
    "derived_v5/iter_phase_share_all99_v5.csv",
    "derived_v5/phase_intervals_all99_v5.parquet",
    "derived_v5/stage_headroom_all99_v5.csv",
    "derived_v5/joint_headroom_profiled_v5.csv",
    "derived_v5/edp_pair_service_entity_summary_v5.csv",
    "derived_v5/edp_pair_direct_24iter_v5.csv",
    "derived_v5/nic_per_iter_device_v5.parquet",
    "derived_v5/nic_device_bytes_entity_summary_v5.csv",
    "derived_v5/nic_device_full_window_rate_entity_summary_v5.csv",
    "derived_v5/mtlink_per_iter_gpu_v5.parquet",
    "derived_v5/mtlink_gpu_full_window_rate_entity_summary_v5.csv",
    "derived_v5/deepep_rank_stage_duration_iter_summary_v5.csv",
    "derived_v5/deepep_rank_stage_duration_entity_summary_v5.csv",
    "derived_v5/deepep_per_iter_rank_stage_v5.parquet",
    "derived_v5/collective_calls_v5.parquet",
    "derived_v5/inference_gate_v5.json",
    "derived_v5/inference_stage_confidence_v5.csv",
    "derived_v5/checkpoints/v5_validation_report.json",
)

_STAGE_TAXONOMY = (
    ("compute_active", "计算活动", "#334E68", False),
    ("deepep_forward_scaleup", "DeepEP 前向 Scale-Up", "#0F9D8A", True),
    ("deepep_backward_scaleup", "DeepEP 反向 Scale-Up", "#0F9D8A", True),
    ("dp_rs_hybrid", "DP ReduceScatter 混合通信", "#4F46E5", True),
    ("dp_ag_hybrid", "DP AllGather 混合通信", "#4F46E5", True),
    ("edp_rs_scaleout", "EDP ReduceScatter Scale-Out", "#E4572E", True),
    ("edp_ag_scaleout", "EDP AllGather Scale-Out", "#E4572E", True),
    ("tiny_sync_collective", "小消息同步 Collective", "#D99B16", True),
    ("gradient_finalize_sync", "梯度收尾同步", "#7C3AED", True),
    ("other_communication", "其他通信", "#7A6652", True),
    ("idle_or_unattributed", "空闲或未归因", "#CBD5E1", False),
    ("all_communication", "全部通信（区间并集）", "#1F2937", False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    if path.suffix == ".csv":
        return int(len(pd.read_csv(path)))
    if path.suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    return 1


def _input_metadata(workspace: Path) -> list[dict[str, Any]]:
    records = []
    for relative_path in _INPUT_PATHS:
        path = workspace / relative_path
        records.append(
            {
                "path": relative_path,
                "sha256": _sha256(path),
                "row_count": _row_count(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _json_safe(value: Any) -> Any:
    """Recursively convert Python/pandas/numpy values into strict JSON values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, np.complexfloating):
        raise ValueError(f"complex value cannot be serialized as strict JSON: {value}")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError(f"non-finite value cannot be serialized as strict JSON: {value}")
        return value
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def serialize_strict_json(payload: dict[str, Any]) -> str:
    """Serialize compact RFC-compliant JSON, preserving missing values as null."""
    return json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _nullable_float(value: Any, *, divisor: float = 1.0) -> float | None:
    if pd.isna(value):
        return None
    return float(value) / divisor


def _required_bool(value: Any, *, field_name: str) -> bool:
    """Return a required boolean scalar without truthiness coercion."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(
        f"Required boolean field {field_name!r} must be bool or numpy.bool_, "
        f"got {value!r} ({type(value).__name__})"
    )


def _parse_active_labels(value: Any) -> list[str]:
    """Normalize a pipe-delimited active-label scalar."""
    if value is None or value is pd.NA or value is pd.NaT:
        return []
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return []
    return [segment for part in str(value).split("|") if (segment := part.strip())]


def _profiler_cadence(profiled_iters: list[int]) -> int:
    """Return the uniform profiler iteration step."""
    if len(profiled_iters) < 2:
        raise ValueError(
            "Profiler cadence requires at least two profiled iterations; "
            f"got {profiled_iters}"
        )
    steps = [current - previous for previous, current in zip(profiled_iters, profiled_iters[1:])]
    if any(step != steps[0] for step in steps[1:]):
        raise ValueError(
            f"Profiler iterations must have uniform cadence; got steps {steps}"
        )
    return steps[0]


def _recompute_profiled_outliers(
    outliers: pd.DataFrame,
    profiled_iters: list[int],
) -> pd.DataFrame:
    """Recompute modified robust z-scores strictly within the direct 24-iteration scope."""
    if len(profiled_iters) != 24 or len(set(profiled_iters)) != 24:
        raise ValueError(
            "Profiled outlier recomputation requires 24 unique explicit iterations; "
            f"got {profiled_iters}"
        )

    atom_to_z = {
        "wall_ns": "robust_z__wall_ns",
        "nic_bidir_bytes": "robust_z__nic_bidir_bytes",
        "mtlink_bidir_bytes": "robust_z__mtlink_bidir_bytes",
        "deepep_active_union_sum_ns": "robust_z__deepep_active_union_sum_ns",
    }
    required_columns = ["iter", *atom_to_z]
    missing_columns = [column for column in required_columns if column not in outliers.columns]
    if missing_columns:
        raise ValueError(
            "Profiled outlier recomputation missing required columns: "
            f"{missing_columns}"
        )

    direct = outliers.loc[outliers["iter"].isin(profiled_iters)].copy()
    if direct["iter"].duplicated().any():
        duplicate_iters = sorted(
            direct.loc[direct["iter"].duplicated(keep=False), "iter"].astype(int).unique()
        )
        raise ValueError(
            "Profiled outlier recomputation requires one row per iteration; "
            f"duplicates={duplicate_iters}"
        )
    actual_iters = direct["iter"].astype(int).tolist()
    if set(actual_iters) != set(profiled_iters) or len(actual_iters) != 24:
        raise ValueError(
            "Profiled outlier recomputation iteration domain mismatch: "
            f"expected {profiled_iters}, got {actual_iters}"
        )
    direct = direct.set_index("iter").loc[profiled_iters].reset_index()

    for atom, z_column in atom_to_z.items():
        values = pd.to_numeric(direct[atom], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Profiled outlier atom {atom!r} must contain 24 finite values"
            )
        median = float(np.median(values))
        absolute_deviations = np.abs(values - median)
        mad = float(np.median(absolute_deviations))
        if mad == 0.0:
            if np.all(values == values[0]):
                direct[z_column] = np.zeros(len(values), dtype=float)
                continue
            raise ValueError(
                f"Profiled outlier atom {atom!r} has zero MAD but non-constant values"
            )
        direct[z_column] = 0.6744897501960817 * (values - median) / mad

    direct["wall_outlier"] = direct["robust_z__wall_ns"] > 3.0
    direct["network_bytes_outlier"] = (
        direct["robust_z__nic_bidir_bytes"].abs() > 3.0
    ) | (direct["robust_z__mtlink_bidir_bytes"].abs() > 3.0)
    direct["diagnosis"] = np.select(
        [
            direct["wall_outlier"] & ~direct["network_bytes_outlier"],
            direct["wall_outlier"] & direct["network_bytes_outlier"],
        ],
        [
            "wall inflation without proportional network-volume increase",
            "wall inflation with network-volume anomaly",
        ],
        default="no strong wall-only inflation",
    )
    return direct


def _build_iter_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        result.append(
            {
                "iter": int(row.iter),
                "analysis_source": (
                    "direct_profiler"
                    if _required_bool(row.is_profiled, field_name="is_profiled")
                    else "validated_inference"
                ),
                "wall_ms": _nullable_float(row.wall_ns, divisor=1_000_000),
                "window_provenance": str(row.window_provenance),
                "nic_bidir_bytes": int(row.nic_bidir_bytes),
                "nic_uncertain_bytes": int(row.nic_uncertain_bytes),
                "mtlink_bidir_bytes": int(row.mtlink_bidir_bytes),
                "mtlink_uncertain_bytes": int(row.mtlink_uncertain_bytes),
                "deepep_active_union_ms": _nullable_float(
                    row.deepep_active_union_sum_ns, divisor=1_000_000
                ),
                "nic_cluster_full_window_gbps": _nullable_float(
                    row.nic_cluster_full_window_gbps
                ),
                "mtlink_cluster_full_window_gbps": _nullable_float(
                    row.mtlink_cluster_full_window_gbps
                ),
                "profiled_global_wall_ms": _nullable_float(
                    row.profiled_global_wall_ns, divisor=1_000_000
                ),
                "cycle_minus_profiled_wall_ms": _nullable_float(
                    row.cycle_minus_profiled_wall_ns, divisor=1_000_000
                ),
                "robust_z_wall": _nullable_float(row.robust_z__wall_ns),
                "robust_z_nic_bidir_bytes": _nullable_float(
                    row.robust_z__nic_bidir_bytes
                ),
                "robust_z_mtlink_bidir_bytes": _nullable_float(
                    row.robust_z__mtlink_bidir_bytes
                ),
                "robust_z_deepep_active_union": _nullable_float(
                    row.robust_z__deepep_active_union_sum_ns
                ),
                "wall_outlier": _required_bool(
                    row.wall_outlier, field_name="wall_outlier"
                ),
                "network_bytes_outlier": _required_bool(
                    row.network_bytes_outlier, field_name="network_bytes_outlier"
                ),
                "diagnosis": str(row.diagnosis),
            }
        )
    return result


def _build_stage_share(rows: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        result.append(
            {
                "iter": int(row.iter),
                "stage": str(row.stage),
                "critical_exposure_ms": _nullable_float(
                    row.critical_exposure_ns, divisor=1_000_000
                ),
                "service_ms": _nullable_float(row.service_ns, divisor=1_000_000),
                "exclusive_ms": _nullable_float(row.exclusive_ns, divisor=1_000_000),
                "overlap_with_compute_ms": _nullable_float(
                    row.overlap_with_compute_ns, divisor=1_000_000
                ),
                "critical_exposure_pct": _nullable_float(row.critical_exposure_pct),
                "service_pct": _nullable_float(row.service_pct),
                "exclusive_wall_pct": _nullable_float(row.exclusive_wall_pct),
                "overlap_with_compute_pct": _nullable_float(row.overlap_with_compute_pct),
                "service_provenance": str(row.service_provenance),
                "provenance": str(row.provenance),
                "confidence": str(row.inference_confidence),
            }
        )
    return result


def _build_phase_intervals(
    intervals: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    starts = summary.set_index("iter")["start_ns"].astype("int64")
    profiled = summary.set_index("iter")["is_profiled"].map(
        lambda value: _required_bool(value, field_name="is_profiled")
    )
    result: list[dict[str, Any]] = []
    for iter_id, group in intervals.groupby("iter", sort=True):
        cycle_start_ns = int(starts.loc[iter_id])
        compact = []
        for row in group.itertuples(index=False):
            compact.append(
                {
                    "start_ms": (int(row.start_ns) - cycle_start_ns) / 1_000_000,
                    "end_ms": (int(row.end_ns) - cycle_start_ns) / 1_000_000,
                    "labels": _parse_active_labels(row.active_labels),
                    "exclusive": str(row.exclusive_category),
                    "provenance": str(row.provenance),
                }
            )
        result.append(
            {
                "iter": int(iter_id),
                "analysis_source": (
                    "direct_profiler" if profiled.loc[iter_id] else "validated_inference"
                ),
                "intervals": compact,
            }
        )
    return result


def _build_stage_headroom(rows: pd.DataFrame) -> list[dict[str, Any]]:
    special_names = {
        "wall_exposed_headroom_conservative_ns": "wall_exposed_conservative_ms",
        "wall_exposed_headroom_observed_ns": "wall_exposed_observed_ms",
        "wall_exposed_headroom_hard_ns": "wall_exposed_hard_ms",
        "wall_exposed_headroom_conservative_pct": "wall_exposed_conservative_pct",
        "wall_exposed_headroom_observed_pct": "wall_exposed_observed_pct",
        "wall_exposed_headroom_hard_pct": "wall_exposed_hard_pct",
        "inference_confidence": "confidence",
    }
    result: list[dict[str, Any]] = []
    for source in rows.to_dict(orient="records"):
        target: dict[str, Any] = {}
        for name, value in source.items():
            output_name = special_names.get(name, name)
            if name.endswith("_ns"):
                output_name = special_names.get(name, name.removesuffix("_ns") + "_ms")
                target[output_name] = _nullable_float(value, divisor=1_000_000)
            else:
                target[output_name] = _json_safe(value)
        result.append(target)
    return result


def _build_joint_counterfactual(rows: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows.to_dict(orient="records"):
        target: dict[str, Any] = {}
        for name, value in source.items():
            if name.endswith("_ns"):
                target[name.removesuffix("_ns") + "_ms"] = _nullable_float(
                    value, divisor=1_000_000
                )
            else:
                target[name] = _json_safe(value)
        result.append(target)
    return result


def _selected_records(rows: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return _json_safe(rows.loc[:, columns].to_dict(orient="records"))


def _build_collective_calls(rows: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        result.append(
            {
                "call_id": str(row.call_id),
                "iter": int(row.iter),
                "pg_name": str(row.pg_name),
                "pg_description": str(row.pg_description),
                "collective": str(row.collective),
                "size_bytes": int(row.size_bytes),
                "occurrence_index": int(row.occurrence_index),
                "pair": str(row.expected_ranks),
                "rank_count": int(row.rank_count),
                "algo": str(row.algo),
                "protocol": str(row.protocol),
                "arrival_skew_ms": _nullable_float(row.arrival_skew_ns, divisor=1_000_000),
                "service_ms": _nullable_float(
                    row.service_after_last_arrival_ns, divisor=1_000_000
                ),
                "finish_spread_ms": _nullable_float(
                    row.finish_spread_ns, divisor=1_000_000
                ),
                "critical_exposure_ms": _nullable_float(
                    row.critical_exposure_ns, divisor=1_000_000
                ),
                "provenance": str(row.provenance),
            }
        )
    return result


def _build_spatial_temporal(
    edp_entity: pd.DataFrame,
    edp_direct: pd.DataFrame,
    nic_per_iter: pd.DataFrame,
    nic_bytes_entity: pd.DataFrame,
    nic_rate_entity: pd.DataFrame,
    mtlink_per_iter: pd.DataFrame,
    mtlink_entity: pd.DataFrame,
    deepep_iter: pd.DataFrame,
    deepep_entity: pd.DataFrame,
    nic_active_rate_entity: pd.DataFrame | None = None,
) -> dict[str, Any]:
    nic = {
        "per_iter_device": _selected_records(
            nic_per_iter,
            [
                "iter",
                "source",
                "host",
                "device",
                "bidir_bytes",
                "uncertain_bidir_bytes",
                "uncertainty_pct",
                "full_window_gbps",
                "coverage_pct",
                "active_rate_gbps",
            ],
        ),
        "bytes_entity_summary": _selected_records(
            nic_bytes_entity, list(nic_bytes_entity.columns)
        ),
        "rate_entity_summary": _selected_records(
            nic_rate_entity, list(nic_rate_entity.columns)
        ),
    }
    if nic_active_rate_entity is not None:
        nic["active_rate_entity_summary"] = _selected_records(
            nic_active_rate_entity, list(nic_active_rate_entity.columns)
        )

    return {
        "edp": {
            "entity_summary": _selected_records(edp_entity, list(edp_entity.columns)),
            "direct_summary": _selected_records(edp_direct, list(edp_direct.columns)),
        },
        "nic": nic,
        "mtlink": {
            "per_iter_gpu": _selected_records(
                mtlink_per_iter,
                [
                    "iter",
                    "host",
                    "device",
                    "bidir_bytes",
                    "uncertain_bidir_bytes",
                    "full_window_gbps",
                ],
            ),
            "entity_summary": _selected_records(mtlink_entity, list(mtlink_entity.columns)),
        },
        "deepep": {
            "iter_summary": _selected_records(deepep_iter, list(deepep_iter.columns)),
            "entity_summary": _selected_records(deepep_entity, list(deepep_entity.columns)),
        },
    }


def _build_conclusion_anchors(
    calls: pd.DataFrame,
    outliers: pd.DataFrame,
) -> dict[str, Any]:
    tiny = calls[
        calls["pg_description"].eq("default_pg")
        & calls["collective"].eq("allreduce")
        & calls["size_bytes"].eq(4)
        & calls["occurrence_index"].eq(0)
    ]
    if tiny.empty:
        raise ValueError("canonical collective data lacks tiny default 4-byte AllReduce calls")

    wall_outliers = outliers["wall_outlier"].map(
        lambda value: _required_bool(value, field_name="wall_outlier")
    )
    outlier_rows = outliers.loc[wall_outliers].copy()
    profiled_gap_rows = outlier_rows.loc[outlier_rows["profiled_global_wall_ns"].notna()]
    return {
        "tiny_default_allreduce": {
            "call_count": int(len(tiny)),
            "arrival_skew_median_ms": float(tiny["arrival_skew_ns"].median() / 1_000_000),
            "service_median_ms": float(
                tiny["service_after_last_arrival_ns"].median() / 1_000_000
            ),
            "finish_spread_median_ms": float(
                tiny["finish_spread_ns"].median() / 1_000_000
            ),
            "critical_exposure_median_ms": float(
                tiny["critical_exposure_ns"].median() / 1_000_000
            ),
            "provenance": sorted(str(value) for value in tiny["provenance"].unique()),
        },
        "outlier_iters": sorted(outlier_rows["iter"].astype(int).tolist()),
        "profiled_gap_outliers": [
            {
                "iter": int(row.iter),
                "cycle_wall_ms": float(row.wall_ns) / 1_000_000,
                "profiled_global_wall_ms": float(row.profiled_global_wall_ns) / 1_000_000,
                "profiled_gap_ms": float(row.cycle_minus_profiled_wall_ns) / 1_000_000,
                "nic_bidir_bytes": int(row.nic_bidir_bytes),
                "diagnosis": str(row.diagnosis),
            }
            for row in profiled_gap_rows.itertuples(index=False)
        ],
    }


def _validate_profiler_iteration_domain(
    rows: pd.DataFrame,
    expected_iters: list[int],
    *,
    source_name: str,
) -> None:
    if "iter" not in rows.columns:
        raise ValueError(f"{source_name} profiler iteration domain mismatch: missing iter column")
    actual_iters = sorted(rows["iter"].astype(int).unique().tolist())
    if actual_iters != expected_iters:
        raise ValueError(
            f"{source_name} profiler iteration domain mismatch: "
            f"expected {expected_iters}, got {actual_iters}"
        )


def _validate_profiler_grid(
    rows: pd.DataFrame,
    expected_iters: list[int],
    *,
    source_name: str,
    rows_per_iter: int,
    entity_columns: list[str],
) -> None:
    _validate_profiler_iteration_domain(rows, expected_iters, source_name=source_name)
    missing_columns = sorted(set(entity_columns) - set(rows.columns))
    if missing_columns:
        raise ValueError(
            f"{source_name} profiler grid mismatch: missing columns {missing_columns}"
        )

    counts = rows.groupby("iter", sort=True).size()
    bad_counts = {
        int(iter_id): int(count)
        for iter_id, count in counts.items()
        if int(count) != rows_per_iter
    }
    if bad_counts:
        raise ValueError(
            f"{source_name} profiler grid mismatch: expected {rows_per_iter} rows "
            f"per iteration, got {bad_counts}"
        )

    key_columns = ["iter", *entity_columns]
    if rows.duplicated(key_columns).any():
        raise ValueError(
            f"{source_name} profiler grid mismatch: duplicate keys for {key_columns}"
        )
    expected_entities: set[tuple[str, ...]] | None = None
    for iter_id, group in rows.groupby("iter", sort=True):
        entities = {
            tuple(str(value) for value in values)
            for values in group[entity_columns].itertuples(index=False, name=None)
        }
        if expected_entities is None:
            expected_entities = entities
        elif entities != expected_entities:
            raise ValueError(
                f"{source_name} profiler grid mismatch: entity domain differs at iter {int(iter_id)}"
            )


def _validate_entity_summary(
    rows: pd.DataFrame,
    *,
    source_name: str,
    expected_rows: int,
    expected_n_iter: int,
) -> None:
    if len(rows) != expected_rows or "n_iter" not in rows.columns:
        raise ValueError(
            f"{source_name} profiler grid mismatch: expected {expected_rows} entity rows "
            "with n_iter"
        )
    actual_n_iter = sorted(rows["n_iter"].astype(int).unique().tolist())
    if actual_n_iter != [expected_n_iter]:
        raise ValueError(
            f"{source_name} profiler grid mismatch: expected n_iter={expected_n_iter}, "
            f"got {actual_n_iter}"
        )


def _entity_summary(
    rows: pd.DataFrame,
    *,
    entity_columns: list[str],
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    return robust_two_way_decomposition(
        rows,
        entity_columns=entity_columns,
        group_columns=group_columns,
        iter_column="iter",
        value_column=value_column,
    ).entity_summary


def _build_profiler_view(
    *,
    profiled_iters: list[int],
    inferred_count: int,
    summary: pd.DataFrame,
    outliers: pd.DataFrame,
    stage_share: pd.DataFrame,
    intervals: pd.DataFrame,
    edp_entity: pd.DataFrame,
    edp_direct: pd.DataFrame,
    nic_per_iter: pd.DataFrame,
    mtlink_per_iter: pd.DataFrame,
    deepep_per_iter: pd.DataFrame,
    calls: pd.DataFrame,
) -> dict[str, Any]:
    membership = set(profiled_iters)
    direct_summary = summary[summary["iter"].isin(membership)].copy()
    direct_outliers = _recompute_profiled_outliers(outliers, profiled_iters)
    direct_stage_share = stage_share[stage_share["iter"].isin(membership)].copy()
    direct_intervals = intervals[intervals["iter"].isin(membership)].copy()
    direct_headroom_source = direct_stage_share[
        direct_stage_share["stage"].isin(COMMUNICATION_STAGES)
    ].copy()
    direct_nic = nic_per_iter[nic_per_iter["iter"].isin(membership)].copy()
    direct_mtlink = mtlink_per_iter[mtlink_per_iter["iter"].isin(membership)].copy()
    direct_deepep = deepep_per_iter[deepep_per_iter["iter"].isin(membership)].copy()
    direct_calls = calls[calls["iter"].isin(membership)].copy()

    _validate_profiler_grid(
        direct_summary,
        profiled_iters,
        source_name="summary",
        rows_per_iter=1,
        entity_columns=[],
    )
    _validate_profiler_grid(
        direct_outliers,
        profiled_iters,
        source_name="outlier analysis",
        rows_per_iter=1,
        entity_columns=[],
    )
    outlier_profiled_mask = outliers["is_profiled"].map(
        lambda value: _required_bool(value, field_name="is_profiled")
    )
    outlier_profiled_iters = outliers.loc[outlier_profiled_mask, "iter"].astype(int).tolist()
    if outlier_profiled_iters != profiled_iters:
        raise ValueError(
            "Canonical outlier direct-profiler domain mismatch: "
            f"expected {profiled_iters}, got {outlier_profiled_iters}"
        )
    _validate_profiler_grid(
        direct_stage_share,
        profiled_iters,
        source_name="stage share",
        rows_per_iter=12,
        entity_columns=["stage"],
    )
    _validate_profiler_grid(
        direct_headroom_source,
        profiled_iters,
        source_name="stage headroom source",
        rows_per_iter=len(COMMUNICATION_STAGES),
        entity_columns=["stage"],
    )
    direct_headroom = compute_stage_headroom(direct_headroom_source)
    _validate_profiler_grid(
        direct_headroom,
        profiled_iters,
        source_name="stage headroom",
        rows_per_iter=len(COMMUNICATION_STAGES),
        entity_columns=["stage"],
    )
    if not direct_headroom["reference_count"].astype(int).eq(len(profiled_iters)).all():
        counts = sorted(direct_headroom["reference_count"].astype(int).unique().tolist())
        raise ValueError(
            "stage headroom direct reference count mismatch: "
            f"expected {len(profiled_iters)}, got {counts}"
        )
    direct_joint = compute_joint_counterfactual(direct_headroom, direct_stage_share)
    _validate_profiler_iteration_domain(
        direct_intervals, profiled_iters, source_name="phase intervals"
    )
    _validate_profiler_grid(
        direct_joint,
        profiled_iters,
        source_name="joint counterfactual",
        rows_per_iter=1,
        entity_columns=[],
    )
    _validate_profiler_grid(
        direct_nic,
        profiled_iters,
        source_name="NIC",
        rows_per_iter=8,
        entity_columns=["source", "host", "device"],
    )
    _validate_profiler_grid(
        direct_mtlink,
        profiled_iters,
        source_name="MTLink",
        rows_per_iter=16,
        entity_columns=["host", "device"],
    )
    _validate_profiler_grid(
        direct_deepep,
        profiled_iters,
        source_name="DeepEP",
        rows_per_iter=32,
        entity_columns=["stage_hint", "rank"],
    )
    _validate_profiler_grid(
        direct_calls,
        profiled_iters,
        source_name="collective calls",
        rows_per_iter=35,
        entity_columns=[
            "pg_name",
            "pg_description",
            "collective",
            "size_bytes",
            "occurrence_index",
            "expected_ranks",
            "rank_count",
            "algo",
            "protocol",
        ],
    )
    _validate_entity_summary(
        edp_entity,
        source_name="EDP",
        expected_rows=24,
        expected_n_iter=len(profiled_iters),
    )
    if (
        len(edp_direct) != 16
        or "n" not in edp_direct.columns
        or not edp_direct["n"].astype(int).isin(
            [len(profiled_iters), 2 * len(profiled_iters)]
        ).all()
    ):
        raise ValueError(
            "EDP direct profiler grid mismatch: expected 16 pair rows with 24/48 observations"
        )

    nic_bytes_entity = _entity_summary(
        direct_nic,
        entity_columns=["host", "device"],
        group_columns=["source"],
        value_column="bidir_bytes",
    )
    nic_rate_entity = _entity_summary(
        direct_nic,
        entity_columns=["host", "device"],
        group_columns=["source"],
        value_column="full_window_gbps",
    )
    nic_active_rate_entity = _entity_summary(
        direct_nic,
        entity_columns=["host", "device"],
        group_columns=["source"],
        value_column="active_rate_gbps",
    )
    mtlink_entity = _entity_summary(
        direct_mtlink,
        entity_columns=["host", "device"],
        group_columns=[],
        value_column="full_window_gbps",
    )
    deepep_decomposition = robust_two_way_decomposition(
        direct_deepep,
        entity_columns=["rank"],
        group_columns=["stage_hint"],
        iter_column="iter",
        value_column="duration_ms",
    )
    for source_name, entity_rows, expected_rows in (
        ("NIC bytes", nic_bytes_entity, 8),
        ("NIC full-window rate", nic_rate_entity, 8),
        ("NIC active rate", nic_active_rate_entity, 8),
        ("MTLink full-window rate", mtlink_entity, 16),
        ("DeepEP duration", deepep_decomposition.entity_summary, 32),
    ):
        _validate_entity_summary(
            entity_rows,
            source_name=source_name,
            expected_rows=expected_rows,
            expected_n_iter=len(profiled_iters),
        )
    _validate_profiler_grid(
        deepep_decomposition.iter_summary,
        profiled_iters,
        source_name="DeepEP derived iteration summary",
        rows_per_iter=2,
        entity_columns=["stage_hint"],
    )

    direct_all_comm = direct_stage_share[
        direct_stage_share["stage"].eq("all_communication")
    ]
    wall_outlier_mask = direct_outliers["wall_outlier"].map(
        lambda value: _required_bool(value, field_name="wall_outlier")
    )
    return {
        "scope": {
            "mode": "direct_profiler_only",
            "iterations": profiled_iters,
            "iter_count": len(profiled_iters),
            "excluded_inference_count": inferred_count,
            "membership_source": "run.coverage.direct_profiler_iters",
        },
        "kpis": {
            "wall_median_ms": float(direct_summary["wall_ns"].median() / 1_000_000),
            "direct_all_communication_exposure_median_pct": float(
                direct_all_comm["critical_exposure_pct"].median()
            ),
            "communication_only_speedup_median_pct": float(
                direct_joint["comm_only_speedup_pct"].median()
            ),
            "main_outlier_count": int(wall_outlier_mask.sum()),
        },
        "conclusion_anchors": _build_conclusion_anchors(direct_calls, direct_outliers),
        "data": {
            "iter_summary": _build_iter_summary(direct_outliers),
            "stage_share": _build_stage_share(direct_stage_share),
            "phase_intervals": _build_phase_intervals(direct_intervals, direct_summary),
            "stage_headroom": _build_stage_headroom(direct_headroom),
            "joint_counterfactual": _build_joint_counterfactual(direct_joint),
            "spatial_temporal": _build_spatial_temporal(
                edp_entity,
                edp_direct,
                direct_nic,
                nic_bytes_entity,
                nic_rate_entity,
                direct_mtlink,
                mtlink_entity,
                deepep_decomposition.iter_summary,
                deepep_decomposition.entity_summary,
                nic_active_rate_entity,
            ),
            "collective_calls": _build_collective_calls(direct_calls),
        },
    }


def _metric_semantics() -> dict[str, dict[str, str]]:
    return {
        "critical_exposure": {
            "zh_label": "关键暴露",
            "unit": "ms or % of cycle wall",
            "definition": "From the earliest participating rank entering a stage to the latest rank leaving it; includes arrival skew.",
        },
        "service": {
            "zh_label": "最后到达后的服务时间",
            "unit": "ms or % of cycle wall",
            "definition": "From the last participating rank arrival to the latest rank finish; distinct from arrival skew.",
        },
        "exclusive_wall": {
            "zh_label": "独占墙钟时间",
            "unit": "ms or % of cycle wall",
            "definition": "Stage time not overlapped by compute or another communication stage; unavailable for inferred iterations.",
        },
        "communication_time_potential": {
            "zh_label": "通信时间潜力",
            "unit": "ms or % of cycle wall",
            "definition": "Stage-duration reduction under conservative, observed-attainable, or hard targets; not automatically wall-clock gain.",
        },
        "wall_exposed_headroom": {
            "zh_label": "墙钟暴露提升空间",
            "unit": "ms or % of cycle wall",
            "definition": "Headroom after overlap clipping; reported only for direct profiler iterations.",
        },
        "endpoint_sum_bytes": {
            "zh_label": "端点双向字节和",
            "unit": "bytes",
            "definition": "TX plus RX summed over measured endpoints; it is not logical wire payload.",
        },
        "full_window_rate": {
            "zh_label": "全窗口速率",
            "unit": "Gb/s",
            "definition": "Endpoint-sum bytes divided by the complete cycle wall; long gaps lower this rate.",
        },
        "active_interval_rate": {
            "zh_label": "活跃区间速率",
            "unit": "Gb/s",
            "definition": "Endpoint-sum bytes divided by measured active counter time, separate from full-window rate.",
        },
        "profiled_gap": {
            "zh_label": "Profiler step 外空档",
            "unit": "ms",
            "definition": "Canonical cycle wall minus direct profiled global wall; unavailable without profiler coverage.",
        },
    }


def build_html_payload(workspace: Path) -> dict[str, Any]:
    """Build the browser-facing Run 1555 payload from v5 canonical outputs."""
    workspace = Path(workspace)
    cfg = load_run_config(workspace / "configs" / "run1555_v5.toml")
    derived = workspace / "derived_v5"

    summary = pd.read_csv(derived / "iter_summary_all99_v5.csv")
    outliers = pd.read_csv(derived / "iter_outlier_analysis_v5.csv")
    stage_share = pd.read_csv(derived / "iter_phase_share_all99_v5.csv")
    intervals = pd.read_parquet(derived / "phase_intervals_all99_v5.parquet")
    headroom = pd.read_csv(derived / "stage_headroom_all99_v5.csv")
    joint = pd.read_csv(derived / "joint_headroom_profiled_v5.csv")
    edp_entity = pd.read_csv(derived / "edp_pair_service_entity_summary_v5.csv")
    edp_direct = pd.read_csv(derived / "edp_pair_direct_24iter_v5.csv")
    nic_per_iter = pd.read_parquet(derived / "nic_per_iter_device_v5.parquet")
    nic_bytes_entity = pd.read_csv(derived / "nic_device_bytes_entity_summary_v5.csv")
    nic_rate_entity = pd.read_csv(
        derived / "nic_device_full_window_rate_entity_summary_v5.csv"
    )
    mtlink_per_iter = pd.read_parquet(derived / "mtlink_per_iter_gpu_v5.parquet")
    mtlink_entity = pd.read_csv(
        derived / "mtlink_gpu_full_window_rate_entity_summary_v5.csv"
    )
    deepep_iter = pd.read_csv(derived / "deepep_rank_stage_duration_iter_summary_v5.csv")
    deepep_entity = pd.read_csv(
        derived / "deepep_rank_stage_duration_entity_summary_v5.csv"
    )
    deepep_per_iter = pd.read_parquet(derived / "deepep_per_iter_rank_stage_v5.parquet")
    calls = pd.read_parquet(derived / "collective_calls_v5.parquet")
    gate = json.loads((derived / "inference_gate_v5.json").read_text(encoding="utf-8"))
    confidence = pd.read_csv(derived / "inference_stage_confidence_v5.csv")
    validator = json.loads(
        (derived / "checkpoints" / "v5_validation_report.json").read_text(encoding="utf-8")
    )

    expected_iters = list(cfg.analysis_iters)
    actual_iters = summary["iter"].astype(int).tolist()
    if actual_iters != expected_iters:
        raise ValueError(
            f"Canonical iteration domain mismatch: expected {expected_iters}, got {actual_iters}"
        )
    outlier_iters = outliers["iter"].astype(int).tolist()
    if outlier_iters != expected_iters:
        raise ValueError(
            f"Canonical outlier iteration domain mismatch: expected {expected_iters}, got {outlier_iters}"
        )

    profiled_mask = summary["is_profiled"].map(
        lambda value: _required_bool(value, field_name="is_profiled")
    )
    profiled_iters = summary.loc[profiled_mask, "iter"].astype(int).tolist()
    profiler_step = _profiler_cadence(profiled_iters)
    expected_profiled = list(cfg.profiler_iters)
    if profiled_iters != expected_profiled:
        raise ValueError(
            "Canonical direct-profiler domain mismatch: "
            f"expected {expected_profiled}, got {profiled_iters}"
        )

    inferred_count = len(actual_iters) - len(profiled_iters)
    direct_all_comm = stage_share[
        stage_share["iter"].isin(profiled_iters)
        & stage_share["stage"].eq("all_communication")
    ]
    kpis = {
        "wall_median_ms": float(summary["wall_ns"].median() / 1_000_000),
        "direct_all_communication_exposure_median_pct": float(
            direct_all_comm["critical_exposure_pct"].median()
        ),
        "communication_only_speedup_median_pct": float(
            joint["comm_only_speedup_pct"].median()
        ),
        "inference_boundary_p95_ms": float(gate["metrics"]["boundary_p95_ms"]),
        "main_outlier_count": int(
            outliers["wall_outlier"]
            .map(lambda value: _required_bool(value, field_name="wall_outlier"))
            .sum()
        ),
    }

    host_span = cfg.host_split_rank
    if cfg.world_size % host_span != 0:
        raise ValueError("world_size must be divisible by host_split_rank for fixed host topology")
    host_count = cfg.world_size // host_span
    if host_count != 2:
        raise ValueError(f"Run 1555 HTML topology requires exactly two hosts, got {host_count}")

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "generator": {
                "name": "scripts/60_build_run1555_html_payload.py",
                "version": GENERATOR_VERSION,
            },
            "inputs": _input_metadata(workspace),
        },
        "validator": validator,
        "run": {
            "run_id": cfg.run_id,
            "experiment": cfg.experiment,
            "topology": {
                "host_count": host_count,
                "accelerators_per_host": host_span,
                "world_size": cfg.world_size,
                "host_split_rank": cfg.host_split_rank,
                "rank_ranges": {"A": [0, host_span - 1], "B": [host_span, cfg.world_size - 1]},
            },
            "config": {
                "deepep_group": cfg.deepep_group,
                "analysis_iter_range": [min(actual_iters), max(actual_iters)],
                "profiler_iter_range": [min(profiled_iters), max(profiled_iters)],
                "profiler_iter_step": profiler_step,
                "thresholds": dict(cfg.thresholds),
            },
            "coverage": {
                "iter_start": min(actual_iters),
                "iter_end": max(actual_iters),
                "iter_count": len(actual_iters),
                "excluded_iters": list(cfg.excluded_iters),
                "direct_profiler_iters": profiled_iters,
                "direct_profiler_count": len(profiled_iters),
                "validated_inference_count": inferred_count,
            },
        },
        "kpis": kpis,
        "stage_taxonomy": [
            {
                "id": stage_id,
                "zh_label": zh_label,
                "color": color,
                "is_communication_stage": is_communication,
            }
            for stage_id, zh_label, color, is_communication in _STAGE_TAXONOMY
        ],
        "semantics": {
            "provenance": {
                "direct": {
                    "zh_label": "直接观测",
                    "definition": "Value measured directly by profiler, DeepEP, NIC, or MTLink instrumentation.",
                },
                "inferred": {
                    "zh_label": "验证后推断",
                    "definition": "Value inferred by the template that passed the v5 inference gate.",
                },
                "not_observed": {
                    "zh_label": "未观测",
                    "definition": "Value is unavailable from source evidence and remains null, never zero-filled.",
                },
            },
            "metrics": _metric_semantics(),
        },
        "conclusion_anchors": _build_conclusion_anchors(calls, outliers),
        "profiler_view": _build_profiler_view(
            profiled_iters=profiled_iters,
            inferred_count=inferred_count,
            summary=summary,
            outliers=outliers,
            stage_share=stage_share,
            intervals=intervals,
            edp_entity=edp_entity,
            edp_direct=edp_direct,
            nic_per_iter=nic_per_iter,
            mtlink_per_iter=mtlink_per_iter,
            deepep_per_iter=deepep_per_iter,
            calls=calls,
        ),
        "data": {
            "iter_summary": _build_iter_summary(outliers),
            "stage_share": _build_stage_share(stage_share),
            "phase_intervals": _build_phase_intervals(intervals, summary),
            "stage_headroom": _build_stage_headroom(headroom),
            "joint_counterfactual": _build_joint_counterfactual(joint),
            "spatial_temporal": _build_spatial_temporal(
                edp_entity,
                edp_direct,
                nic_per_iter,
                nic_bytes_entity,
                nic_rate_entity,
                mtlink_per_iter,
                mtlink_entity,
                deepep_iter,
                deepep_entity,
            ),
            "collective_calls": _build_collective_calls(calls),
            "inference": {
                "gate": gate,
                "stage_confidence": _selected_records(
                    confidence, list(confidence.columns)
                ),
            },
        },
    }

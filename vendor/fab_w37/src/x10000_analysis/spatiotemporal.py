from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpatiotemporalResult:
    observations: pd.DataFrame
    entity_summary: pd.DataFrame
    iter_summary: pd.DataFrame


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    slopes: list[float] = []
    for left, right in combinations(range(len(x)), 2):
        delta_x = float(x[right] - x[left])
        if delta_x != 0:
            slopes.append(float(y[right] - y[left]) / delta_x)
    return float(np.median(slopes)) if slopes else float("nan")


def _mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def robust_two_way_decomposition(
    data: pd.DataFrame,
    entity_columns: list[str],
    group_columns: list[str],
    iter_column: str,
    value_column: str,
) -> SpatiotemporalResult:
    """Separate per-iter common effect, persistent entity effect, and residual."""
    required = {iter_column, value_column, *entity_columns, *group_columns}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"spatiotemporal input missing columns: {missing}")
    if data.empty:
        raise ValueError("spatiotemporal input must not be empty")

    observations_parts: list[pd.DataFrame] = []
    entity_rows: list[dict[str, object]] = []
    iter_rows: list[dict[str, object]] = []
    group_iterator = data.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), data)]

    for group_key, group in group_iterator:
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_context = dict(zip(group_columns, group_values))
        work = group.copy()
        iter_effect_map = work.groupby(iter_column)[value_column].median()
        work["iter_effect"] = work[iter_column].map(iter_effect_map).astype(float)
        work["iter_centered"] = work[value_column].astype(float) - work["iter_effect"]
        entity_effect_map = work.groupby(entity_columns, dropna=False)["iter_centered"].median()

        if len(entity_columns) == 1:
            work["entity_effect"] = work[entity_columns[0]].map(entity_effect_map).astype(float)
        else:
            entity_keys = pd.MultiIndex.from_frame(work[entity_columns])
            work["entity_effect"] = entity_effect_map.reindex(entity_keys).to_numpy(dtype=float)
        work["temporal_residual"] = work["iter_centered"] - work["entity_effect"]
        observations_parts.append(work)

        for entity_key, entity_group in work.groupby(entity_columns, dropna=False, sort=True):
            entity_values = entity_key if isinstance(entity_key, tuple) else (entity_key,)
            context = {**group_context, **dict(zip(entity_columns, entity_values))}
            ordered = entity_group.sort_values(iter_column)
            x = ordered[iter_column].to_numpy(dtype=float)
            raw = ordered[value_column].to_numpy(dtype=float)
            residual = ordered["temporal_residual"].to_numpy(dtype=float)
            residual_median = float(np.median(residual))
            residual_mad = _mad(residual)
            threshold = 3.0 * 1.4826 * residual_mad
            if threshold == 0:
                spike_count = int(np.sum(np.abs(residual - residual_median) > 1e-12))
            else:
                spike_count = int(np.sum(np.abs(residual - residual_median) > threshold))
            q25, q75 = np.percentile(raw, [25, 75])
            entity_rows.append(
                {
                    **context,
                    "n_iter": len(ordered),
                    "value_median": float(np.median(raw)),
                    "value_iqr": float(q75 - q25),
                    "value_p95": float(np.percentile(raw, 95)),
                    "persistent_entity_effect": float(np.median(ordered["entity_effect"])),
                    "residual_mad": residual_mad,
                    "temporal_instability_score": residual_mad
                    / max(abs(float(np.median(raw))), 1e-12),
                    "raw_trend_per_iter": _theil_sen_slope(x, raw),
                    "residual_trend_per_iter": _theil_sen_slope(x, residual),
                    "spike_count": spike_count,
                }
            )

        for iter_id, iter_group in work.groupby(iter_column, sort=True):
            values = iter_group[value_column].to_numpy(dtype=float)
            residuals = iter_group["temporal_residual"].to_numpy(dtype=float)
            iter_rows.append(
                {
                    **group_context,
                    iter_column: iter_id,
                    "entity_count": len(iter_group),
                    "iter_effect": float(iter_effect_map.loc[iter_id]),
                    "value_min": float(np.min(values)),
                    "value_max": float(np.max(values)),
                    "value_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                    "residual_mad": _mad(residuals),
                }
            )

    return SpatiotemporalResult(
        observations=pd.concat(observations_parts, ignore_index=True),
        entity_summary=pd.DataFrame(entity_rows),
        iter_summary=pd.DataFrame(iter_rows),
    )

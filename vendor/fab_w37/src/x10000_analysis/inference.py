from __future__ import annotations

import numpy as np
import pandas as pd

from .intervals import Interval, merge_intervals
from .phases import classify_collective_stage


def _landmarks(windows: pd.DataFrame, deepep_events: pd.DataFrame | None) -> pd.DataFrame:
    landmarks = windows[["iter", "start_ns", "end_ns"]].copy().rename(
        columns={"start_ns": "window_start", "end_ns": "window_end"}
    )
    if deepep_events is not None and not deepep_events.empty:
        deep = (
            deepep_events.groupby("iter", as_index=False)
            .agg(deepep_first_start=("start_ns", "min"), deepep_last_end=("end_ns", "max"))
        )
        landmarks = landmarks.merge(deep, on="iter", how="left", validate="one_to_one")
    return landmarks


def _direct_intervals(
    calls: pd.DataFrame,
    windows: pd.DataFrame,
    deepep_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    landmarks = _landmarks(windows, deepep_events).set_index("iter")
    rows: list[dict[str, object]] = []
    work = calls.copy()
    work["stage"] = [
        classify_collective_stage(pg, op, int(size))
        for pg, op, size in zip(work["pg_description"], work["collective"], work["size_bytes"])
    ]
    for (iter_id, stage), group in work.groupby(["iter", "stage"], sort=True):
        if iter_id not in landmarks.index:
            raise ValueError(f"missing window for profiled iter {iter_id}")
        merged = merge_intervals(
            [
                Interval(int(row.earliest_start_ns), int(row.latest_end_ns))
                for row in group.itertuples(index=False)
            ]
        )
        landmark_row = landmarks.loc[iter_id]
        context = {
            column: int(landmark_row[column])
            for column in landmarks.columns
            if pd.notna(landmark_row[column])
        }
        for occurrence, interval in enumerate(merged):
            rows.append(
                {
                    "iter": int(iter_id),
                    "stage": str(stage),
                    "occurrence": occurrence,
                    "start_ns": interval.start_ns,
                    "end_ns": interval.end_ns,
                    "duration_ns": interval.duration_ns,
                    **context,
                }
            )
    return pd.DataFrame(rows)


def _spread_score(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _select_template(direct: pd.DataFrame) -> pd.DataFrame:
    candidate_anchors = [
        column
        for column in ("window_start", "deepep_first_start", "deepep_last_end")
        if column in direct.columns and direct[column].notna().all()
    ]
    if not candidate_anchors:
        raise ValueError("no complete observable anchor available for collective template")
    rows: list[dict[str, object]] = []
    for (stage, occurrence), group in direct.groupby(["stage", "occurrence"], sort=True):
        candidates: list[tuple[float, int, str, np.ndarray, np.ndarray]] = []
        for order, anchor in enumerate(candidate_anchors):
            start_offsets = group["start_ns"].to_numpy(dtype=np.int64) - group[anchor].to_numpy(dtype=np.int64)
            end_offsets = group["end_ns"].to_numpy(dtype=np.int64) - group[anchor].to_numpy(dtype=np.int64)
            score = _spread_score(start_offsets) + _spread_score(end_offsets)
            candidates.append((score, order, anchor, start_offsets, end_offsets))
        # The short catch-all communication interval is tied to the cycle
        # boundary by contract.  DeepEP landmarks happen to reduce its sample
        # IQR on Run1555, but do not describe its semantics and caused the
        # original handoff to vary across dependency versions.
        if stage == "other_communication" and occurrence == 0:
            candidates = [item for item in candidates if item[2] == "window_start"]
        _, _, anchor, start_offsets, end_offsets = min(candidates, key=lambda item: (item[0], item[1]))
        rows.append(
            {
                "stage": stage,
                "occurrence": int(occurrence),
                "anchor": anchor,
                "start_offset_ns": int(round(float(np.median(start_offsets)))),
                "end_offset_ns": int(round(float(np.median(end_offsets)))),
                "training_iter_count": int(group["iter"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def cross_validate_collective_template(
    calls: pd.DataFrame,
    windows: pd.DataFrame,
    deepep_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Leave one profiled iter out and validate observable-landmark templates."""
    direct = _direct_intervals(calls, windows, deepep_events)
    rows: list[dict[str, object]] = []
    for held_iter in sorted(direct["iter"].unique()):
        train = direct[direct["iter"] != held_iter]
        held = direct[direct["iter"] == held_iter]
        template = _select_template(train)
        merged = held.merge(
            template,
            on=["stage", "occurrence"],
            how="left",
            validate="one_to_one",
        )
        if merged[["anchor", "start_offset_ns", "end_offset_ns"]].isna().any().any():
            raise ValueError(f"template occurrence missing while holding out iter {held_iter}")
        for row in merged.itertuples(index=False):
            anchor_value = int(getattr(row, row.anchor))
            predicted_start = anchor_value + int(row.start_offset_ns)
            predicted_end = anchor_value + int(row.end_offset_ns)
            start_error = abs(int(row.start_ns) - predicted_start)
            end_error = abs(int(row.end_ns) - predicted_end)
            direct_duration = int(row.duration_ns)
            predicted_duration = predicted_end - predicted_start
            rows.append(
                {
                    "held_out_iter": int(held_iter),
                    "stage": row.stage,
                    "occurrence": int(row.occurrence),
                    "anchor": row.anchor,
                    "direct_start_ns": int(row.start_ns),
                    "direct_end_ns": int(row.end_ns),
                    "predicted_start_ns": predicted_start,
                    "predicted_end_ns": predicted_end,
                    "start_error_ns": start_error,
                    "end_error_ns": end_error,
                    "boundary_error_ns": max(start_error, end_error),
                    "duration_error_pct": abs(predicted_duration - direct_duration)
                    / max(direct_duration, 1)
                    * 100,
                    "direct_duration_ns": direct_duration,
                    "predicted_duration_ns": predicted_duration,
                }
            )
    return pd.DataFrame(rows)


def evaluate_inference_gate(
    validation: pd.DataFrame,
    boundary_median_ms_max: float,
    boundary_p95_ms_max: float,
    duration_error_median_pct_max: float,
) -> dict[str, object]:
    if validation.empty:
        raise ValueError("inference validation must not be empty")
    boundary_ms = validation["boundary_error_ns"].to_numpy(dtype=float) / 1e6
    duration_error = validation["duration_error_pct"].to_numpy(dtype=float)
    metrics = {
        "boundary_median_ms": float(np.median(boundary_ms)),
        "boundary_p95_ms": float(np.percentile(boundary_ms, 95)),
        "duration_error_median_pct": float(np.median(duration_error)),
    }
    checks = {
        "boundary_median": metrics["boundary_median_ms"] <= boundary_median_ms_max,
        "boundary_p95": metrics["boundary_p95_ms"] <= boundary_p95_ms_max,
        "duration_error_median": metrics["duration_error_median_pct"]
        <= duration_error_median_pct_max,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def infer_collective_intervals(
    calls: pd.DataFrame,
    training_windows: pd.DataFrame,
    target_windows: pd.DataFrame,
    deepep_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply full-training observable-landmark templates to target iter windows."""
    training_iter_set = set(int(value) for value in training_windows["iter"])
    if deepep_events is None:
        training_deep = target_deep = None
    else:
        training_deep = deepep_events[deepep_events["iter"].isin(training_iter_set)]
        target_iter_set = set(int(value) for value in target_windows["iter"])
        target_deep = deepep_events[deepep_events["iter"].isin(target_iter_set)]
    direct = _direct_intervals(calls, training_windows, training_deep)
    template = _select_template(direct)
    targets = _landmarks(target_windows, target_deep).set_index("iter")
    rows: list[dict[str, object]] = []
    for iter_id, landmark in targets.iterrows():
        for row in template.itertuples(index=False):
            if row.anchor not in targets.columns or pd.isna(landmark[row.anchor]):
                raise ValueError(f"target iter {iter_id} lacks required anchor {row.anchor}")
            anchor_value = int(landmark[row.anchor])
            start_ns = anchor_value + int(row.start_offset_ns)
            end_ns = anchor_value + int(row.end_offset_ns)
            rows.append(
                {
                    "iter": int(iter_id),
                    "stage": row.stage,
                    "occurrence": int(row.occurrence),
                    "anchor": row.anchor,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": end_ns - start_ns,
                    "training_iter_count": int(row.training_iter_count),
                    "provenance": "inferred_validated_template",
                }
            )
    return pd.DataFrame(rows)

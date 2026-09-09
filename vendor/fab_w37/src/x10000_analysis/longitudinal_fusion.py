from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DEFAULT_HOST_IDS: Mapping[str, str] = {
    "A": "225a01104",
    "B": "225c10502",
}
TRANSFER_ELEMENTS_CALIBRATION_ITER = 4
TRANSFER_ELEMENTS_CALIBRATION_HOST = "B"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _rounded_int64_mean(values: pd.Series) -> np.int64:
    """Return an exact nearest-even integer mean without float epoch loss."""
    integers = [int(value) for value in values.to_numpy(dtype=np.int64)]
    if not integers:
        raise ValueError("cannot average an empty MTLink collection cycle")
    base = integers[0]
    offset_sum = sum(value - base for value in integers)
    quotient, remainder = divmod(offset_sum, len(integers))
    rounded = base + quotient
    if 2 * remainder > len(integers) or (
        2 * remainder == len(integers) and rounded % 2 != 0
    ):
        rounded += 1
    return np.int64(rounded)


def aggregate_mtlink_collection_cycles(
    raw: pd.DataFrame,
    expected_links: int = 14,
) -> pd.DataFrame:
    """Rebuild one GPU's plotted MTLink curve from the raw per-link rows.

    The collector emits one row per link and labels all links sampled in one
    collection cycle with the same ``iter``.  Each row's calibrated epoch is
    ``mt_timestamp_end_ns + (host_realtime_ns - host_mono_ns)``; a collection
    cycle is the exact rounded mean of its 14 calibrated row timestamps and
    the sum of its per-link decimal GB/s rates.
    """
    required = {
        "iter", "host_realtime_ns", "host_mono_ns", "mt_timestamp_end_ns",
        "link_id", "tx_GBps", "rx_GBps",
    }
    _require_columns(raw, required, "raw MTLink frame")
    if int(expected_links) <= 0:
        raise ValueError("expected_links must be positive")

    grouped = raw.groupby("iter", sort=True, observed=True)
    size = grouped.size()
    unique_links = grouped["link_id"].nunique()
    valid = size.eq(int(expected_links)) & unique_links.eq(int(expected_links))
    if not bool(valid.all()):
        bad = valid.index[~valid].tolist()[:10]
        raise ValueError(
            f"each MTLink collection cycle must contain exactly {expected_links} links; bad cycles={bad}"
        )

    calibrated = (
        pd.to_numeric(raw["mt_timestamp_end_ns"], errors="raise").astype("int64")
        + pd.to_numeric(raw["host_realtime_ns"], errors="raise").astype("int64")
        - pd.to_numeric(raw["host_mono_ns"], errors="raise").astype("int64")
    )
    enriched = raw.assign(_calibrated_epoch_ns=calibrated)
    result = enriched.groupby("iter", sort=True, observed=True).agg(
        timestamp_ns=("_calibrated_epoch_ns", _rounded_int64_mean),
        tx_GBps=("tx_GBps", "sum"),
        rx_GBps=("rx_GBps", "sum"),
        link_count=("link_id", "nunique"),
    ).reset_index(names="collection_iter")
    result["timestamp_ns"] = pd.to_numeric(result["timestamp_ns"], errors="raise").astype("int64")
    result["link_count"] = result["link_count"].astype("int16")
    return result


def validate_rank_host_mapping(
    frame: pd.DataFrame,
    host_ids: Mapping[str, str] = DEFAULT_HOST_IDS,
    host_split_rank: int = 8,
) -> None:
    required = {"rank", "host", "host_id", "local_rank"}
    _require_columns(frame, required, "rank identity frame")
    ranks = pd.to_numeric(frame["rank"], errors="raise").astype("int64")
    expected_host = np.where(ranks < int(host_split_rank), "A", "B")
    expected_local_rank = ranks % int(host_split_rank)
    expected_host_id = pd.Series(expected_host, index=frame.index).map(dict(host_ids))
    invalid = (
        frame["host"].astype(str).ne(expected_host)
        | frame["host_id"].astype(str).ne(expected_host_id.astype(str))
        | pd.to_numeric(frame["local_rank"], errors="raise").astype("int64").ne(expected_local_rank)
    )
    if bool(invalid.any()):
        sample = frame.loc[invalid, ["rank", "host", "host_id", "local_rank"]].head(10).to_dict("records")
        raise ValueError(f"rank/host mapping violation: {sample}")


def fuse_event_windows(
    events: pd.DataFrame,
    mtlink_samples: pd.DataFrame,
    nominal_tx_reference_GBps: float = 112.0,
) -> pd.DataFrame:
    """Fuse DeepEP event windows with the correctly mapped per-GPU TX curve."""
    event_required = {
        "iter",
        "rank",
        "host",
        "host_id",
        "local_rank",
        "event",
        "event_sequence",
        "start_ns",
        "end_ns",
    }
    sample_required = {"host", "device", "timestamp_ns", "tx_GBps", "rx_GBps"}
    _require_columns(events, event_required, "event frame")
    _require_columns(mtlink_samples, sample_required, "MTLink sample frame")
    if float(nominal_tx_reference_GBps) <= 0:
        raise ValueError("nominal_tx_reference_GBps must be positive")
    validate_rank_host_mapping(events)

    sample_arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (host, device), group in mtlink_samples.groupby(["host", "device"], sort=False):
        ordered = group.sort_values("timestamp_ns")
        timestamps = ordered["timestamp_ns"].to_numpy(dtype=np.int64)
        if len(timestamps) > 1 and bool(np.any(timestamps[1:] <= timestamps[:-1])):
            raise ValueError(f"non-increasing MTLink timestamps for {host}/{device}")
        sample_arrays[(str(host), str(device))] = (
            timestamps,
            ordered["tx_GBps"].to_numpy(dtype=np.float64),
            ordered["rx_GBps"].to_numpy(dtype=np.float64),
        )

    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start_ns = int(event.start_ns)
        end_ns = int(event.end_ns)
        if end_ns < start_ns:
            raise ValueError(f"event end precedes start for iter={event.iter}, rank={event.rank}")
        device = f"gpu{int(event.local_rank)}"
        key = (str(event.host), device)
        if key not in sample_arrays:
            raise ValueError(f"missing MTLink sample stream for {key}")
        timestamps, tx_values, rx_values = sample_arrays[key]
        left = int(np.searchsorted(timestamps, start_ns, side="left"))
        right = int(np.searchsorted(timestamps, end_ns, side="right"))
        tx = tx_values[left:right]
        rx = rx_values[left:right]
        count = int(right - left)
        base = event._asdict()
        base.update(
            {
                "device": device,
                "sample_count": count,
                "event_peak_tx_GBps": float(np.max(tx)) if count else np.nan,
                "event_sample_mean_tx_GBps": float(np.mean(tx)) if count else np.nan,
                "event_peak_rx_GBps": float(np.max(rx)) if count else np.nan,
                "event_sample_mean_rx_GBps": float(np.mean(rx)) if count else np.nan,
                "event_peak_reference_util_pct": (
                    float(np.max(tx)) / float(nominal_tx_reference_GBps) * 100.0 if count else np.nan
                ),
                "nominal_tx_reference_GBps": float(nominal_tx_reference_GBps),
                "time_reliability": "approximate_locator_only",
                "provenance": "direct_profiled_event_plus_raw_mtlink_collection_curve",
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def add_transfer_element_weights(
    event_metrics: pd.DataFrame,
    transfer_weights: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["iter", "rank", "event", "event_sequence"]
    _require_columns(event_metrics, set(keys), "event metrics")
    _require_columns(transfer_weights, {*keys, "transfer_elements"}, "transfer weights")
    weight_iters = pd.to_numeric(transfer_weights["iter"], errors="raise").astype("int64")
    if bool(weight_iters.ne(TRANSFER_ELEMENTS_CALIBRATION_ITER).any()):
        raise ValueError(
            "transfer_elements are admitted only for iter4 calibration; other iterations must remain NA"
        )
    if transfer_weights.duplicated(keys).any():
        raise ValueError("transfer weights are not unique by event key")
    result = event_metrics.merge(
        transfer_weights[keys + ["transfer_elements"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    values = pd.to_numeric(result["transfer_elements"], errors="coerce")
    if bool(values.dropna().le(0).any()):
        raise ValueError("transfer_elements must be positive when present")
    result["transfer_elements"] = values
    if "host" in result.columns:
        admitted = (
            pd.to_numeric(result["iter"], errors="raise").astype("int64").eq(TRANSFER_ELEMENTS_CALIBRATION_ITER)
            & result["host"].astype(str).eq(TRANSFER_ELEMENTS_CALIBRATION_HOST)
        )
        result.loc[~admitted, "transfer_elements"] = np.nan
    result["transfer_weight_status"] = np.where(
        result["transfer_elements"].notna(),
        "exact_colleague_html",
        "missing_not_imputed",
    )
    return result


def weighted_peak_reference_utilization(frame: pd.DataFrame) -> float:
    required = {"host", "transfer_elements", "event_peak_reference_util_pct"}
    _require_columns(frame, required, "weighted utilization frame")
    hosts = frame["host"].dropna().astype(str).unique()
    if len(hosts) > 1:
        raise ValueError("cross-host weighted utilization is forbidden")
    if frame.empty or frame["transfer_elements"].isna().any() or frame["event_peak_reference_util_pct"].isna().any():
        return float("nan")
    weights = pd.to_numeric(frame["transfer_elements"], errors="raise").to_numpy(dtype=np.float64)
    values = pd.to_numeric(frame["event_peak_reference_util_pct"], errors="raise").to_numpy(dtype=np.float64)
    if bool(np.any(weights <= 0)) or not float(weights.sum()) > 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def deepep_main_comm_event_label(name: object) -> str | None:
    """Label DeepEP main communication kernels; notify/control kernels return None."""
    text = str(name)
    if "deep_ep::intranode::dispatch<" in text:
        return "intranode::dispatch"
    if "deep_ep::intranode::combine<" in text:
        return "intranode::combine"
    return None


def assign_event_sequence(
    events: pd.DataFrame,
    key_cols: list[str] | tuple[str, ...] = ("iter", "rank", "event"),
    order_col: str = "start_ns",
    output_col: str = "event_sequence",
) -> pd.DataFrame:
    """Number events 0..n-1 in time order within each (iter, rank, event) group.

    This matches the colleague HTML's ``event_index_within_rank_and_type``
    convention so transfer-element weights can be joined fail-closed by key.
    """
    keys = list(key_cols)
    _require_columns(events, {*keys, order_col}, "event sequence frame")
    if events.duplicated(keys + [order_col]).any():
        raise ValueError("duplicate event start timestamps within a sequence group")
    ordered = events.sort_values(keys + [order_col], kind="mergesort")
    ordered[output_col] = ordered.groupby(keys, sort=False, observed=True).cumcount().astype("int32")
    return ordered.reset_index(drop=True)


def entity_volatility_summary(
    frame: pd.DataFrame,
    value_col: str,
    entity_cols: list[str] | tuple[str, ...],
    iter_col: str = "iter",
) -> pd.DataFrame:
    """Persistent per-entity behaviour across iterations.

    Entities are ranked by their 24-iteration median, never by a single
    extremum, so a one-off spike cannot define a permanent hotspot.
    """
    entities = list(entity_cols)
    _require_columns(frame, {iter_col, value_col, *entities}, "entity volatility frame")
    clean = frame[[iter_col, *entities, value_col]].dropna(subset=[value_col]).copy()
    rows: list[dict[str, object]] = []
    for entity_key, group in clean.groupby(entities, sort=True, observed=True):
        key_values = entity_key if isinstance(entity_key, tuple) else (entity_key,)
        ordered = group.sort_values(iter_col)
        values = pd.to_numeric(ordered[value_col], errors="raise").to_numpy(dtype=np.float64)
        iters = ordered[iter_col].astype("int64").to_numpy()
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        minimum_index = int(np.argmin(values))
        maximum_index = int(np.argmax(values))
        rows.append(
            dict(zip(entities, key_values))
            | {
                "observation_count": int(len(values)),
                "median": float(np.median(values)),
                "mean": mean,
                "std": std,
                "cv": std / abs(mean) if mean != 0 else np.nan,
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
                "min": float(values[minimum_index]),
                "min_iter": int(iters[minimum_index]),
                "max": float(values[maximum_index]),
                "max_iter": int(iters[maximum_index]),
            }
        )
    result = pd.DataFrame(rows).sort_values("median", ascending=False).reset_index(drop=True)
    result["median_rank"] = np.arange(1, len(result) + 1, dtype=np.int64)
    return result


def metric_correlations(
    features: pd.DataFrame,
    target_col: str,
    metric_cols: list[str] | tuple[str, ...] | None = None,
    iter_col: str = "iter",
) -> pd.DataFrame:
    """Spearman and Pearson correlation of each metric with a target across iterations."""
    _require_columns(features, {iter_col, target_col}, "correlation frame")
    metrics = list(metric_cols) if metric_cols is not None else [
        column
        for column in features.columns
        if column not in {iter_col, target_col} and pd.api.types.is_numeric_dtype(features[column])
    ]
    rows: list[dict[str, object]] = []
    target = pd.to_numeric(features[target_col], errors="coerce")
    for metric in metrics:
        values = pd.to_numeric(features[metric], errors="coerce")
        valid = target.notna() & values.notna()
        count = int(valid.sum())
        if count < 2:
            rows.append({"metric": metric, "target": target_col, "observation_count": count,
                         "spearman_rho": np.nan, "pearson_r": np.nan})
            continue
        x = values[valid]
        y = target[valid]
        rows.append(
            {
                "metric": metric,
                "target": target_col,
                "observation_count": count,
                "spearman_rho": float(x.rank().corr(y.rank())),
                "pearson_r": float(x.corr(y)),
            }
        )
    return pd.DataFrame(rows)


def _median_absolute_deviation(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def spatial_iteration_summary(
    frame: pd.DataFrame,
    value_col: str,
    entity_col: str,
    iter_col: str = "iter",
    expected_entities: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    _require_columns(frame, {iter_col, entity_col, value_col}, "spatial frame")
    source = frame[[iter_col, entity_col, value_col]].copy()
    duplicate_mask = source.duplicated([iter_col, entity_col], keep=False)
    if duplicate_mask.any():
        examples = source.loc[duplicate_mask, [iter_col, entity_col]].head(5).to_dict("records")
        raise ValueError(f"spatial frame has duplicate (iteration, entity) rows: {examples}")
    expected = None if expected_entities is None else {str(item) for item in expected_entities}
    rows: list[dict[str, object]] = []
    clean = source.dropna(subset=[value_col]).copy()
    if expected is not None:
        for iter_id, group in source.groupby(iter_col, sort=True):
            observed = set(group.loc[group[value_col].notna(), entity_col].astype(str))
            if observed != expected:
                raise ValueError(
                    f"iteration {iter_id} does not match expected entity set: "
                    f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
                )
    for iter_id, group in clean.groupby(iter_col, sort=True):
        values = pd.to_numeric(group[value_col], errors="raise").to_numpy(dtype=np.float64)
        entities = group[entity_col].astype(str).to_numpy()
        minimum_index = int(np.argmin(values))
        maximum_index = int(np.argmax(values))
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        rows.append(
            {
                "iter": int(iter_id),
                "entity_count": int(len(values)),
                "mean": mean,
                "median": float(np.median(values)),
                "p25": float(np.quantile(values, 0.25)),
                "p75": float(np.quantile(values, 0.75)),
                "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "mad": _median_absolute_deviation(values),
                "std": std,
                "cv": std / abs(mean) if mean != 0 else np.nan,
                "min": float(values[minimum_index]),
                "max": float(values[maximum_index]),
                "spread": float(values[maximum_index] - values[minimum_index]),
                "min_entity": entities[minimum_index],
                "max_entity": entities[maximum_index],
            }
        )
    return pd.DataFrame(rows)


def temporal_trend_summary(
    frame: pd.DataFrame,
    value_col: str,
    iter_col: str = "iter",
    edge_count: int = 8,
) -> dict[str, float | int]:
    _require_columns(frame, {iter_col, value_col}, "temporal frame")
    clean = frame[[iter_col, value_col]].dropna().sort_values(iter_col)
    if clean[iter_col].duplicated().any():
        raise ValueError("temporal trend requires one value per iteration")
    count = len(clean)
    if count < 2:
        raise ValueError("temporal trend requires at least two observations")
    edge = min(int(edge_count), count // 2)
    if edge <= 0:
        raise ValueError("edge_count must be positive")
    x = pd.to_numeric(clean[iter_col], errors="raise").to_numpy(dtype=np.float64)
    y = pd.to_numeric(clean[value_col], errors="raise").to_numpy(dtype=np.float64)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(count - 1)
        for j in range(i + 1, count)
        if x[j] != x[i]
    ]
    first = float(np.median(y[:edge]))
    last = float(np.median(y[-edge:]))
    rho = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    return {
        "observation_count": int(count),
        "first_edge_count": int(edge),
        "first_edge_median": first,
        "last_edge_median": last,
        "last_minus_first": last - first,
        "last_vs_first_pct": (last / first - 1.0) * 100.0 if first != 0 else np.nan,
        "spearman_rho": rho,
        "theil_sen_per_10_iter": float(np.median(slopes) * 10.0),
        "overall_median": float(np.median(y)),
        "overall_mad": _median_absolute_deviation(y),
        "minimum": float(np.min(y)),
        "maximum": float(np.max(y)),
    }


def add_robust_outlier_flags(
    frame: pd.DataFrame,
    value_col: str,
    threshold_mad: float = 3.0,
) -> pd.DataFrame:
    _require_columns(frame, {value_col}, "outlier frame")
    result = frame.copy()
    values = pd.to_numeric(result[value_col], errors="coerce")
    median = float(values.median())
    mad = float((values - median).abs().median())
    result[f"{value_col}_robust_center"] = median
    result[f"{value_col}_robust_mad"] = mad
    if mad == 0 or np.isnan(mad):
        result[f"{value_col}_is_robust_outlier"] = False
    else:
        result[f"{value_col}_is_robust_outlier"] = (values - median).abs() > float(threshold_mad) * mad
    return result

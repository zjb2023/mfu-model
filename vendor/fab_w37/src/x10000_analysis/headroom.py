from __future__ import annotations

import numpy as np
import pandas as pd


def _best_quartile_median(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan")
    q25 = float(clean.quantile(0.25))
    return float(clean[clean <= q25].median())


def compute_stage_headroom(stage_share: pd.DataFrame) -> pd.DataFrame:
    """Quantify communication-time potential and separately bounded wall exposure."""
    required = {
        "iter",
        "stage",
        "wall_ns",
        "critical_exposure_ns",
        "service_ns",
        "exclusive_ns",
        "service_provenance",
        "provenance",
    }
    missing = sorted(required - set(stage_share.columns))
    if missing:
        raise ValueError(f"stage-share input missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for stage, group in stage_share.groupby("stage", sort=True):
        direct_service_mask = group["service_provenance"].astype(str).str.startswith("direct_profiled")
        direct_service = group.loc[direct_service_mask, "service_ns"].dropna().astype(float)
        use_service = not direct_service.empty
        if use_service:
            target_conservative = float(direct_service.quantile(0.25))
            target_observed = _best_quartile_median(direct_service)
            reference_count = len(direct_service)
            basis = "post_last_arrival_service"
        else:
            direct_critical_mask = group["provenance"].astype(str).str.startswith("direct")
            direct_critical = group.loc[direct_critical_mask, "critical_exposure_ns"].dropna().astype(float)
            if direct_critical.empty:
                direct_critical = group["critical_exposure_ns"].dropna().astype(float)
            target_conservative = float(direct_critical.quantile(0.25))
            target_observed = _best_quartile_median(direct_critical)
            reference_count = len(direct_critical)
            basis = "instrumented_active_union"

        for row in group.itertuples(index=False):
            critical = float(row.critical_exposure_ns)
            service = float(row.service_ns) if pd.notna(row.service_ns) else float("nan")
            exclusive = float(row.exclusive_ns) if pd.notna(row.exclusive_ns) else float("nan")
            if use_service and np.isfinite(service):
                arrival = max(critical - service, 0.0)
                service_conservative = max(service - target_conservative, 0.0)
                service_observed = max(service - target_observed, 0.0)
                potential_conservative = arrival + service_conservative
                potential_observed = arrival + service_observed
            else:
                arrival = float("nan")
                service_conservative = float("nan")
                service_observed = float("nan")
                potential_conservative = max(critical - target_conservative, 0.0)
                potential_observed = max(critical - target_observed, 0.0)

            if np.isfinite(exclusive):
                wall_conservative = min(exclusive, potential_conservative)
                wall_observed = min(exclusive, potential_observed)
                wall_hard = exclusive
            else:
                wall_conservative = wall_observed = wall_hard = float("nan")
            wall_ns = float(row.wall_ns)
            rows.append(
                {
                    "iter": int(row.iter),
                    "stage": stage,
                    "wall_ns": int(row.wall_ns),
                    "critical_exposure_ns": critical,
                    "service_ns": service,
                    "exclusive_ns": exclusive,
                    "headroom_basis": basis,
                    "reference_count": reference_count,
                    "service_target_conservative_ns": target_conservative if use_service else float("nan"),
                    "service_target_observed_ns": target_observed if use_service else float("nan"),
                    "active_target_conservative_ns": target_conservative if not use_service else float("nan"),
                    "active_target_observed_ns": target_observed if not use_service else float("nan"),
                    "arrival_skew_headroom_ns": arrival,
                    "service_headroom_conservative_ns": service_conservative,
                    "service_headroom_observed_ns": service_observed,
                    "potential_reduction_conservative_ns": potential_conservative,
                    "potential_reduction_observed_ns": potential_observed,
                    "potential_reduction_hard_ns": critical,
                    "wall_exposed_headroom_conservative_ns": wall_conservative,
                    "wall_exposed_headroom_observed_ns": wall_observed,
                    "wall_exposed_headroom_hard_ns": wall_hard,
                    "potential_reduction_conservative_pct": potential_conservative / wall_ns * 100,
                    "potential_reduction_observed_pct": potential_observed / wall_ns * 100,
                    "potential_reduction_hard_pct": critical / wall_ns * 100,
                    "wall_exposed_headroom_conservative_pct": wall_conservative / wall_ns * 100
                    if np.isfinite(wall_conservative)
                    else float("nan"),
                    "wall_exposed_headroom_observed_pct": wall_observed / wall_ns * 100
                    if np.isfinite(wall_observed)
                    else float("nan"),
                    "wall_exposed_headroom_hard_pct": wall_hard / wall_ns * 100
                    if np.isfinite(wall_hard)
                    else float("nan"),
                    "provenance": row.provenance,
                    "service_provenance": row.service_provenance,
                    "inference_confidence": getattr(row, "inference_confidence", None),
                }
            )
    return pd.DataFrame(rows)


COMMUNICATION_STAGES = (
    "deepep_backward_scaleup",
    "deepep_forward_scaleup",
    "dp_ag_hybrid",
    "dp_rs_hybrid",
    "edp_ag_scaleout",
    "edp_rs_scaleout",
    "gradient_finalize_sync",
    "other_communication",
    "tiny_sync_collective",
)

_JOINT_COLUMNS = (
    "iter",
    "comm_wall_headroom_conservative_ns",
    "comm_wall_headroom_observed_ns",
    "comm_wall_headroom_hard_ns",
    "wall_ns",
    "critical_exposure_ns",
    "gap_headroom_conservative_ns",
    "gap_headroom_observed_ns",
    "joint_wall_headroom_conservative_ns",
    "joint_wall_headroom_observed_ns",
    "joint_wall_headroom_conservative_pct",
    "joint_wall_headroom_observed_pct",
    "comm_only_counterfactual_wall_ns",
    "joint_counterfactual_wall_conservative_ns",
    "joint_counterfactual_wall_observed_ns",
    "comm_only_speedup_pct",
    "joint_speedup_conservative_pct",
    "joint_speedup_observed_pct",
    "counterfactual_assumption",
)


def _require_columns(rows: pd.DataFrame, required: set[str], *, source_name: str) -> None:
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{source_name} missing columns: {missing}")


def compute_joint_counterfactual(
    stage_headroom: pd.DataFrame,
    stage_share: pd.DataFrame,
) -> pd.DataFrame:
    """Combine direct communication headroom with direct idle critical exposure.

    Both inputs must describe the same complete iteration domain. Communication
    headroom is an exact nine-stage grid; the idle baseline is computed solely
    from the matching ``idle_or_unattributed`` rows in ``stage_share``.
    """
    headroom_value_columns = [
        "wall_exposed_headroom_conservative_ns",
        "wall_exposed_headroom_observed_ns",
        "wall_exposed_headroom_hard_ns",
    ]
    _require_columns(
        stage_headroom,
        {"iter", "stage", "wall_ns", *headroom_value_columns},
        source_name="stage headroom",
    )
    _require_columns(
        stage_share,
        {"iter", "stage", "wall_ns", "critical_exposure_ns"},
        source_name="stage share",
    )
    if stage_headroom.empty:
        raise ValueError("stage headroom grid is empty")
    if stage_headroom.duplicated(["iter", "stage"]).any():
        raise ValueError("stage headroom grid contains duplicate iteration/stage keys")
    if stage_share.duplicated(["iter", "stage"]).any():
        raise ValueError("stage share contains duplicate iteration/stage keys")

    expected_stages = set(COMMUNICATION_STAGES)
    actual_stages = set(stage_headroom["stage"].astype(str))
    if actual_stages != expected_stages:
        raise ValueError(
            "stage headroom grid stage domain mismatch: "
            f"expected {sorted(expected_stages)}, got {sorted(actual_stages)}"
        )
    expected_iters = sorted(stage_headroom["iter"].astype(int).unique().tolist())
    for iter_id, group in stage_headroom.groupby("iter", sort=True):
        stages = set(group["stage"].astype(str))
        if len(group) != len(COMMUNICATION_STAGES) or stages != expected_stages:
            raise ValueError(
                "stage headroom grid is incomplete at iteration "
                f"{int(iter_id)}: expected {list(COMMUNICATION_STAGES)}, got {sorted(stages)}"
            )

    idle = stage_share[stage_share["stage"].eq("idle_or_unattributed")].copy()
    idle_iters = sorted(idle["iter"].astype(int).unique().tolist())
    if idle_iters != expected_iters or len(idle) != len(expected_iters):
        raise ValueError(
            "stage share idle grid iteration domain mismatch: "
            f"expected {expected_iters}, got {idle_iters}"
        )
    idle = idle.sort_values("iter").set_index("iter")

    numeric_headroom = stage_headroom[["wall_ns", *headroom_value_columns]].astype(float)
    numeric_idle = idle[["wall_ns", "critical_exposure_ns"]].astype(float)
    if not np.isfinite(numeric_headroom.to_numpy()).all() or not np.isfinite(
        numeric_idle.to_numpy()
    ).all():
        raise ValueError("joint counterfactual inputs must be finite")
    if (numeric_headroom[headroom_value_columns] < 0).any().any() or (
        numeric_idle["critical_exposure_ns"] < 0
    ).any():
        raise ValueError("joint counterfactual inputs must be nonnegative")
    if (numeric_headroom["wall_ns"] <= 0).any() or (numeric_idle["wall_ns"] <= 0).any():
        raise ValueError("joint counterfactual wall_ns must be positive")

    wall_counts = stage_headroom.groupby("iter", sort=True)["wall_ns"].nunique()
    if (wall_counts != 1).any():
        raise ValueError("stage headroom grid has inconsistent wall_ns within an iteration")
    communication_walls = stage_headroom.groupby("iter", sort=True)["wall_ns"].first()
    if not np.array_equal(
        communication_walls.astype(float).to_numpy(),
        idle["wall_ns"].astype(float).to_numpy(),
    ):
        raise ValueError("stage headroom and idle stage share wall_ns mismatch")

    comm = stage_headroom.groupby("iter", sort=True)[headroom_value_columns].sum()
    comm = comm.rename(
        columns={
            "wall_exposed_headroom_conservative_ns": "comm_wall_headroom_conservative_ns",
            "wall_exposed_headroom_observed_ns": "comm_wall_headroom_observed_ns",
            "wall_exposed_headroom_hard_ns": "comm_wall_headroom_hard_ns",
        }
    )
    result = idle[["wall_ns", "critical_exposure_ns"]].join(comm)

    gap_target_conservative = float(result["critical_exposure_ns"].quantile(0.25))
    gap_target_observed = _best_quartile_median(result["critical_exposure_ns"])
    if not np.isfinite(gap_target_conservative) or not np.isfinite(gap_target_observed):
        raise ValueError("idle critical exposure cannot produce finite gap targets")
    result["gap_headroom_conservative_ns"] = (
        result["critical_exposure_ns"] - gap_target_conservative
    ).clip(lower=0.0)
    result["gap_headroom_observed_ns"] = (
        result["critical_exposure_ns"] - gap_target_observed
    ).clip(lower=0.0)
    result["joint_wall_headroom_conservative_ns"] = (
        result["comm_wall_headroom_conservative_ns"]
        + result["gap_headroom_conservative_ns"]
    )
    result["joint_wall_headroom_observed_ns"] = (
        result["comm_wall_headroom_observed_ns"]
        + result["gap_headroom_observed_ns"]
    )
    result["joint_wall_headroom_conservative_pct"] = (
        result["joint_wall_headroom_conservative_ns"] / result["wall_ns"] * 100.0
    )
    result["joint_wall_headroom_observed_pct"] = (
        result["joint_wall_headroom_observed_ns"] / result["wall_ns"] * 100.0
    )
    result["comm_only_counterfactual_wall_ns"] = (
        result["wall_ns"] - result["comm_wall_headroom_conservative_ns"]
    )
    result["joint_counterfactual_wall_conservative_ns"] = (
        result["wall_ns"] - result["joint_wall_headroom_conservative_ns"]
    )
    result["joint_counterfactual_wall_observed_ns"] = (
        result["wall_ns"] - result["joint_wall_headroom_observed_ns"]
    )
    counterfactual_columns = [
        "comm_only_counterfactual_wall_ns",
        "joint_counterfactual_wall_conservative_ns",
        "joint_counterfactual_wall_observed_ns",
    ]
    if (result[counterfactual_columns] <= 0).any().any():
        raise ValueError("joint counterfactual wall must remain positive")

    result["comm_only_speedup_pct"] = (
        result["wall_ns"] / result["comm_only_counterfactual_wall_ns"] - 1.0
    ) * 100.0
    result["joint_speedup_conservative_pct"] = (
        result["wall_ns"] / result["joint_counterfactual_wall_conservative_ns"] - 1.0
    ) * 100.0
    result["joint_speedup_observed_pct"] = (
        result["wall_ns"] / result["joint_counterfactual_wall_observed_ns"] - 1.0
    ) * 100.0
    result["counterfactual_assumption"] = (
        "serial_compaction_of_nonoverlapped_intervals_plus_stable_gap_target"
    )
    return result.reset_index().loc[:, list(_JOINT_COLUMNS)]

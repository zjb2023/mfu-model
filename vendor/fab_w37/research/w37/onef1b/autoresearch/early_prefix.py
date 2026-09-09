"""T36A: source-fitted reliability shrinkage for an earlier online prefix."""
from __future__ import annotations

import json
import math
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from candidate import baseline, critical_ledger
from guards import ROOT
from online_prefix import (
    DirectionalScaleCosts, direction_shapley, score_iterations, score_phase,
    winsorized_mean,
)
from pp_graph import build, envelope
from pp_semantics import align_observations
from schedule_contraction import RoleMappedReadinessCosts
from smoke_worker import dump, sha
from worker import csv


FIT = [85, 90]
SOURCE_DEVELOPMENT = [95, 100]
TARGET_DEVELOPMENT = [85, 90, 95, 100]
MFU_NUMERATOR = 100 * 8.436548311982576e16 / (224 * 500e12) * 1000
PHASE_KEY = ["rank", "pp_stage", "pp_lane", "phase", "microbatch"]
PRIMARY = "early_source_reliability"
CANDIDATE_ORDER = [
    "early_cold_control", "early_raw_unshrunk", "early_coverage_quarter", PRIMARY,
]


def reliability_coefficient(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denominator = float(np.dot(x, x))
    assert x.size == y.size and x.size and denominator > 0
    return float(np.clip(np.dot(x, y) / denominator, 0.0, 1.0))


def _phase_column(frame: pd.DataFrame) -> pd.Series:
    return frame["phase"] if "phase" in frame else frame["name"]


def early_prefix_mask(frame: pd.DataFrame, pp: int) -> pd.Series:
    phase = _phase_column(frame)
    forward = phase.eq("forward") & frame.microbatch.eq(0)
    backward = (
        phase.eq("backward") & frame.microbatch.eq(0)
        & frame.pp_stage.ge(pp - 4)
    )
    return forward | backward


def raw_prefix_projection(
    phases: pd.DataFrame, iteration: int, pp: int, base,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = phases[phases.iteration.eq(iteration)].copy()
    frame["phase"] = _phase_column(frame)
    prefix = frame[early_prefix_mask(frame, pp)].copy()
    assert not prefix.empty
    prefix["source_cost_ns"] = [
        base.phase({
            "pp_stage": int(row["pp_stage"]), "pp_lane": int(row["pp_lane"]),
            "name": row["phase"], "microbatch": int(row["microbatch"]),
        })
        for row in prefix.to_dict("records")
    ]
    prefix["observed_duration_ns"] = prefix.duration_ns.astype("int64")
    prefix["observed_to_source_ratio"] = (
        prefix.observed_duration_ns / prefix.source_cost_ns
    )
    prefix["feature_role"] = np.where(
        prefix.phase.eq("forward"), "all_stage_F_microbatch0",
        "last_four_stage_B_microbatch0",
    )
    forward = prefix[prefix.phase.eq("forward")].observed_to_source_ratio
    backward = prefix[prefix.phase.eq("backward")].observed_to_source_ratio
    factors = {
        "iteration": iteration, "PP": pp,
        "raw_forward_factor": winsorized_mean(forward, 0.2),
        "raw_backward_factor": float(backward.median()),
        "forward_samples": int(len(forward)), "backward_samples": int(len(backward)),
    }
    prefix.insert(0, "iteration_key", iteration)
    return prefix[[
        "iteration_key", *PHASE_KEY, "feature_role", "observed_duration_ns",
        "source_cost_ns", "observed_to_source_ratio",
    ]], factors


def derive_reliability(
    phases: pd.DataFrame, base, pp: int, mb: int, actual: dict[int, float],
) -> tuple[dict[str, object], pd.DataFrame]:
    base_nodes, _, _ = build(pp, mb, base)
    base_value = float(envelope(base_nodes)["onef1b_ms"])
    rows, x, y, single = [], [], [], []
    for iteration in FIT:
        _, raw = raw_prefix_projection(phases, iteration, pp, base)
        raw_nodes, _, _ = build(
            pp, mb, DirectionalScaleCosts(
                base, raw["raw_forward_factor"], raw["raw_backward_factor"]
            )
        )
        raw_value = float(envelope(raw_nodes)["onef1b_ms"])
        response = raw_value - base_value
        observed = float(actual[iteration]) - base_value
        assert response != 0
        ratio = float(np.clip(observed / response, 0.0, 1.0))
        x.append(response)
        y.append(observed)
        single.append(ratio)
        rows.append({
            "iteration": iteration, "cold_prediction_ms": base_value,
            "raw_prediction_ms": raw_value, "actual_onef1b_ms": float(actual[iteration]),
            "raw_response_x_ms": response, "observed_response_y_ms": observed,
            "single_iteration_reliability": ratio,
            "raw_forward_factor": raw["raw_forward_factor"],
            "raw_backward_factor": raw["raw_backward_factor"],
        })
    coefficient = reliability_coefficient(x, y)
    result = {
        "fit_iterations": FIT,
        "formula": "clip(sum(x*y)/sum(x*x),0,1)",
        "reliability_coefficient": coefficient,
        "coefficient_interval_low": min(single),
        "coefficient_interval_high": max(single),
        "target_timing_used": False,
    }
    return result, pd.DataFrame(rows)


def adjusted_factors(raw: dict[str, object], coefficient: float) -> tuple[float, float]:
    forward = 1 + coefficient * (float(raw["raw_forward_factor"]) - 1)
    backward = 1 + coefficient * (float(raw["raw_backward_factor"]) - 1)
    return forward, backward


def build_prediction_set(
    phases: pd.DataFrame, base, pp: int, mb: int, iterations: list[int],
    reliability: dict[str, object],
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    coefficients = {
        "early_cold_control": 0.0,
        "early_raw_unshrunk": 1.0,
        "early_coverage_quarter": 0.25,
        PRIMARY: float(reliability["reliability_coefficient"]),
    }
    feature_rows, factor_rows, prediction_rows, nodes_all = [], [], [], []
    shared_edges = None
    shared_topology = None
    for iteration in iterations:
        features, raw = raw_prefix_projection(phases, iteration, pp, base)
        feature_rows.append(features)
        for method in CANDIDATE_ORDER:
            coefficient = coefficients[method]
            forward, backward = adjusted_factors(raw, coefficient)
            costs = DirectionalScaleCosts(base, forward, backward)
            nodes, edges, topology = build(pp, mb, costs)
            if shared_edges is None:
                shared_edges, shared_topology = edges, topology
            else:
                pd.testing.assert_frame_equal(shared_edges, edges, check_exact=True)
                assert topology == shared_topology
            result = envelope(nodes)
            factor_rows.append({
                "method": method, **raw, "reliability_coefficient": coefficient,
                "adjusted_forward_factor": forward,
                "adjusted_backward_factor": backward,
            })
            prediction_rows.append({
                "method": method, "iteration": iteration,
                "predicted_onef1b_ms": result["onef1b_ms"],
                "predicted_program_ms": result["program_ms"],
                "node_count": len(nodes), "edge_count": len(edges),
                "candidate_topology_sha256": topology,
            })
            nodes.insert(0, "iteration", iteration)
            nodes.insert(0, "method", method)
            nodes_all.append(nodes)
    assert shared_edges is not None
    return (
        pd.concat(feature_rows, ignore_index=True), pd.DataFrame(factor_rows),
        pd.DataFrame(prediction_rows), pd.concat(nodes_all, ignore_index=True),
    ), shared_edges


def interval_predictions(
    phases: pd.DataFrame, base, pp: int, mb: int, iterations: list[int],
    reliability: dict[str, object],
) -> pd.DataFrame:
    rows = []
    low = float(reliability["coefficient_interval_low"])
    high = float(reliability["coefficient_interval_high"])
    for iteration in iterations:
        _, raw = raw_prefix_projection(phases, iteration, pp, base)
        values = []
        for coefficient in [low, high]:
            forward, backward = adjusted_factors(raw, coefficient)
            nodes, _, _ = build(pp, mb, DirectionalScaleCosts(base, forward, backward))
            values.append(float(envelope(nodes)["onef1b_ms"]))
        rows.append({
            "iteration": iteration, "coefficient_low": low, "coefficient_high": high,
            "predicted_interval_low_ms": min(values),
            "predicted_interval_high_ms": max(values),
            "interval_source": "source85/90 single-iteration reliability endpoints",
        })
    return pd.DataFrame(rows)


def cutoff_scores(
    phases: pd.DataFrame, predictions: pd.DataFrame, pp: int,
    actual: dict[int, float], domain: str,
) -> pd.DataFrame:
    rows = []
    for iteration in sorted(predictions.iteration.unique()):
        frame = phases[phases.iteration.eq(iteration)].copy()
        frame["phase"] = _phase_column(frame)
        selected = frame[early_prefix_mask(frame, pp)]
        elapsed = (
            float(selected.observed_end_ns.max()) - float(frame.observed_start_ns.min())
        ) / 1e6
        total = float(actual[int(iteration)])
        rows.append({
            "domain": domain, "iteration": int(iteration),
            "prefix_available_elapsed_ms": elapsed, "actual_onef1b_ms": total,
            "prefix_available_fraction_pct": 100 * elapsed / total,
            "remaining_fraction_pct": 100 * (total - elapsed) / total,
        })
    return pd.DataFrame(rows)


def score_interval(intervals: pd.DataFrame, actual: dict[int, float]) -> pd.DataFrame:
    result = intervals.copy()
    result["actual_onef1b_ms"] = result.iteration.map(actual)
    result["covered"] = result.actual_onef1b_ms.between(
        result.predicted_interval_low_ms, result.predicted_interval_high_ms
    )
    result["distance_to_interval_ms"] = np.maximum(
        result.predicted_interval_low_ms - result.actual_onef1b_ms,
        np.maximum(result.actual_onef1b_ms - result.predicted_interval_high_ms, 0),
    )
    return result


def target_step_score(
    predictions: pd.DataFrame, actual_onef1b: dict[int, float],
    paths: dict[str, Path], baseline_record: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = pd.read_csv(paths["iteration_ground_truth.csv"])
    truth = truth[truth.iteration.isin(TARGET_DEVELOPMENT)].copy()
    payload = json.loads(paths["dag_v682_stage_aware_pp_gradient_payload.json"].read_text())
    evaluation = {int(row["iteration"]): row for row in payload["evaluation"]}
    online = predictions.set_index(["method", "iteration"]).predicted_onef1b_ms
    rows, ledger = [], []
    for observed in truth.itertuples(index=False):
        iteration = int(observed.iteration)
        actual = {
            "entry": float(evaluation[iteration]["actual_entry_ms"]),
            "onef1b": float(actual_onef1b[iteration]),
            "tail": float(observed.actual_profiler_step_ms)
                    - float(evaluation[iteration]["actual_entry_ms"])
                    - float(actual_onef1b[iteration]),
            "outer": float(observed.actual_training_step_ms)
                     - float(observed.actual_profiler_step_ms),
        }
        for variant in ["v685_frozen", *CANDIDATE_ORDER]:
            onef1b = (
                float(baseline_record["onef1b_ms"])
                if variant == "v685_frozen" else float(online.loc[(variant, iteration)])
            )
            predicted = {
                "entry": float(baseline_record["entry_ms"]), "onef1b": onef1b,
                "tail": float(baseline_record["tail_ms"]),
                "outer": float(baseline_record["outer_ms"]),
            }
            profiler = predicted["entry"] + onef1b + predicted["tail"]
            training = profiler + predicted["outer"]
            predicted_mfu = MFU_NUMERATOR / training
            actual_mfu = MFU_NUMERATOR / float(observed.actual_training_step_ms)
            rows.append({
                "variant": variant, "iteration": iteration,
                "prediction_mode": "cold_start" if variant in ["v685_frozen", "early_cold_control"] else "online_prefix_conditioned",
                "actual_onef1b_ms": actual["onef1b"], "predicted_onef1b_ms": onef1b,
                "onef1b_error_ms": onef1b - actual["onef1b"],
                "onef1b_APE_pct": 100 * abs(onef1b - actual["onef1b"]) / actual["onef1b"],
                "actual_profiler_ms": float(observed.actual_profiler_step_ms),
                "predicted_profiler_ms": profiler,
                "profiler_APE_pct": 100 * abs(profiler - float(observed.actual_profiler_step_ms)) / float(observed.actual_profiler_step_ms),
                "actual_training_ms": float(observed.actual_training_step_ms),
                "predicted_training_ms": training,
                "training_APE_pct": 100 * abs(training - float(observed.actual_training_step_ms)) / float(observed.actual_training_step_ms),
                "actual_MFU_pct_derived": actual_mfu, "predicted_MFU_pct": predicted_mfu,
                "MFU_error_pp": predicted_mfu - actual_mfu,
                "MFU_relative_APE_pct": 100 * abs(predicted_mfu - actual_mfu) / actual_mfu,
                "MFU_basis": "inherited FLOPs/peak and training clock; numerator not independently verified",
            })
            for phase in ["entry", "onef1b", "tail", "outer"]:
                ledger.append({
                    "variant": variant, "iteration": iteration, "phase": phase,
                    "actual_ms": actual[phase], "predicted_ms": predicted[phase],
                    "error_ms": predicted[phase] - actual[phase],
                    "prediction_changed_by_early_prefix": phase == "onef1b" and variant not in ["v685_frozen", "early_cold_control"],
                })
    result = pd.DataFrame(rows)
    metrics = result.groupby(["variant", "prediction_mode"], sort=True).agg(
        iterations=("iteration", "size"), onef1b_MAPE_pct=("onef1b_APE_pct", "mean"),
        onef1b_bias_ms=("onef1b_error_ms", "mean"),
        profiler_MAPE_pct=("profiler_APE_pct", "mean"),
        training_MAPE_pct=("training_APE_pct", "mean"),
        MFU_relative_MAPE_pct=("MFU_relative_APE_pct", "mean"),
        MFU_bias_pp=("MFU_error_pp", "mean"),
    ).reset_index()
    return result, pd.DataFrame(ledger), metrics


def target_direction_shapley(base, factors: pd.DataFrame) -> pd.DataFrame:
    base_nodes, _, _ = build(14, 3, base)
    base_value = float(envelope(base_nodes)["onef1b_ms"])
    rows = []
    for item in factors[factors.method.eq(PRIMARY)].itertuples(index=False):
        values = {}
        for label, forward, backward in [
            ("forward_only", item.adjusted_forward_factor, 1.0),
            ("backward_only", 1.0, item.adjusted_backward_factor),
            ("both", item.adjusted_forward_factor, item.adjusted_backward_factor),
        ]:
            nodes, _, _ = build(14, 3, DirectionalScaleCosts(base, forward, backward))
            values[label] = float(envelope(nodes)["onef1b_ms"])
        f_value, b_value = direction_shapley(
            base_value, values["forward_only"], values["backward_only"], values["both"]
        )
        for component, value in [("forward_phase_factor", f_value), ("backward_phase_factor", b_value)]:
            rows.append({
                "iteration": int(item.iteration), "component": component,
                "predicted_onef1b_delta_ms": value,
                "cold_start_onef1b_ms": base_value,
                "online_onef1b_ms": values["both"],
                "scope": "sealed model allocation; not observed-error attribution",
            })
    return pd.DataFrame(rows)


def _actual_envelope(phases: pd.DataFrame, iterations: list[int]) -> dict[int, float]:
    frame = phases[phases.iteration.isin(iterations)].groupby("iteration").agg(
        first=("observed_start_ns", "min"), last=("observed_end_ns", "max")
    )
    return ((frame["last"] - frame["first"]) / 1e6).to_dict()


def _draw(out: Path, source: pd.DataFrame, target: pd.DataFrame,
          cutoffs: pd.DataFrame, intervals: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "w37-t36a-early-prefix"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].barh(source.method, source.onef1b_MAPE_pct, color="#348ABD")
    axes[0].set_xlabel("source95/100 1F1B MAPE (%)")
    axes[0].set_title("Source-fitted reliability")
    local = target[target.variant.isin(["v685_frozen", *CANDIDATE_ORDER])]
    axes[1].barh(local.variant, local.onef1b_MAPE_pct, color="#E24A33")
    axes[1].set_xlabel("target development MAPE (%)")
    axes[1].set_title("Cold start vs early online")
    centers = (intervals.predicted_interval_low_ms + intervals.predicted_interval_high_ms) / 2
    half_width = (intervals.predicted_interval_high_ms - intervals.predicted_interval_low_ms) / 2
    axes[2].errorbar(
        intervals.iteration, centers, yerr=half_width, fmt="o", capsize=3,
        label="source-only interval",
    )
    axes[2].plot(intervals.iteration, intervals.actual_onef1b_ms, "kx", label="actual")
    cutoff = cutoffs.prefix_available_fraction_pct.mean()
    axes[2].set_title(f"Source-only coefficient interval; cutoff {cutoff:.1f}%")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("1F1B ms")
    fig.suptitle("T36A earlier online prefix with source reliability shrinkage")
    fig.text(.5, .012, "Intervals use only source85/90 coefficient endpoints. Target results remain exposed development nowcasts.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out / "early_prefix_reliability.svg", metadata={"Date": None})
    fig.savefig(out / "early_prefix_reliability.png", dpi=150, metadata={"Software": "w37-t36a"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    begin = perf_counter()
    assert plan["diagnostic"] == "early_prefix_reliability_audit"
    assert plan["diagnostic_access"] == "evaluator" and not plan["variants"]
    review = json.loads((ROOT / plan["resource_review_file"]).read_text())
    assert review["fit_iterations"] == FIT
    for item in review["training_semantics"]:
        assert sha(Path(item["path"])) == item["sha256"]

    source_phases = pd.read_csv(paths["source_pp_trace_events_60_100.csv"])
    source_apis = pd.read_csv(paths["source_pp_api_events_60_100.csv"])
    source_boundaries = pd.read_csv(paths["source256_profiler_entry_60_100.csv"])
    aligned, pairs = align_observations(source_phases, source_apis)
    source_base = RoleMappedReadinessCosts(aligned, pairs, FIT, pp=16, mb=4)
    source_actual = source_boundaries[
        source_boundaries.iteration.isin(FIT + SOURCE_DEVELOPMENT)
    ].set_index("iteration").phase_envelope_ms.to_dict()
    reliability, reliability_rows = derive_reliability(
        source_phases, source_base, 16, 4, source_actual
    )
    source_bundle, source_edges = build_prediction_set(
        source_phases[source_phases.iteration.isin(FIT + SOURCE_DEVELOPMENT)],
        source_base, 16, 4, FIT + SOURCE_DEVELOPMENT, reliability,
    )
    source_features, source_factors, source_predictions, source_nodes = source_bundle
    mutated_source = source_phases[source_phases.iteration.isin(FIT + SOURCE_DEVELOPMENT)].copy()
    mutation = mutated_source.iteration.isin(SOURCE_DEVELOPMENT) & ~early_prefix_mask(mutated_source, 16)
    mutated_source.loc[mutation, "duration_ns"] = -999999999
    changed_bundle, changed_edges = build_prediction_set(
        mutated_source, source_base, 16, 4, FIT + SOURCE_DEVELOPMENT, reliability,
    )
    for original, changed in zip(source_bundle, changed_bundle):
        pd.testing.assert_frame_equal(original, changed, check_exact=True)
    pd.testing.assert_frame_equal(source_edges, changed_edges, check_exact=True)

    source_intervals = interval_predictions(
        source_phases, source_base, 16, 4, FIT + SOURCE_DEVELOPMENT, reliability
    )
    csv(out, "source_reliability_fit.csv", reliability_rows)
    dump(out / "source_reliability_contract.json", reliability)
    csv(out, "source_early_prefix_features.csv.gz", source_features)
    csv(out, "source_early_factors.csv", source_factors)
    csv(out, "source_early_iteration_predictions.csv", source_predictions)
    csv(out, "source_early_interval_predictions.csv", source_intervals)
    csv(out, "source_early_nodes.csv.gz", source_nodes)
    csv(out, "source_early_edges.csv.gz", source_edges)
    csv(out, "source_phase_readiness_parameters.csv", pd.DataFrame(source_base.parameter_rows))
    dump(out / "early_estimator_contract.json", {
        "status": "SEALED_SOURCE85_90_RELIABILITY_AND_EARLY_PREFIX_RULE",
        "feature": review["pre_registered_feature"],
        "candidates": review["pre_registered_candidates"],
        "primary_selected_before_target_prefix": PRIMARY,
        "target_timing_used": False,
        "formal_topology_replaced": False,
    })
    source_files = [
        "source_reliability_fit.csv", "source_reliability_contract.json",
        "source_early_prefix_features.csv.gz", "source_early_factors.csv",
        "source_early_iteration_predictions.csv", "source_early_interval_predictions.csv",
        "source_early_nodes.csv.gz", "source_early_edges.csv.gz",
        "source_phase_readiness_parameters.csv", "early_estimator_contract.json",
    ]
    dump(out / "source_early_prediction_seal.json", {
        "status": "SEALED_BEFORE_TARGET_EARLY_PREFIX_ACCESS",
        "source_development_nonprefix_mutation_invariant": True,
        "files": [{"path": name, "sha256": sha(out / name)} for name in source_files],
    })

    source_results, _ = score_iterations(source_predictions, source_actual, "source")
    source_phase_results, source_phase_metrics = score_phase(source_nodes, source_phases)
    source_results["split"] = np.where(
        source_results.iteration.isin(FIT), "source_fit", "source_development_confirmation"
    )
    source_metrics = source_results[source_results.split.eq("source_development_confirmation")].groupby("method", sort=True).agg(
        iterations=("iteration", "size"), onef1b_MAPE_pct=("APE_pct", "mean"),
        onef1b_MAE_ms=("error_ms", lambda values: values.abs().mean()),
        onef1b_bias_ms=("error_ms", "mean"),
    ).reset_index()
    phase_dev = source_phase_results[source_phase_results.iteration.isin(SOURCE_DEVELOPMENT)]
    phase_dev = phase_dev.groupby("method", sort=True).agg(
        phase_duration_MAE_ms=("duration_error_ms", lambda values: values.abs().mean()),
        phase_duration_bias_ms=("duration_error_ms", "mean"),
    ).reset_index()
    source_metrics = source_metrics.merge(phase_dev, on="method", validate="one_to_one")
    source_interval_results = score_interval(source_intervals, source_actual)
    source_cutoffs = cutoff_scores(source_phases, source_predictions, 16, source_actual, "source256")
    csv(out, "source_early_iteration_results.csv", source_results)
    csv(out, "source_early_metrics.csv", source_metrics)
    csv(out, "source_early_phase_results.csv.gz", source_phase_results)
    csv(out, "source_early_phase_metrics.csv", source_phase_metrics)
    csv(out, "source_early_interval_results.csv", source_interval_results)
    csv(out, "source_early_prefix_cutoffs.csv", source_cutoffs)
    index = source_metrics.set_index("method")
    cold = index.loc["early_cold_control"]
    primary = index.loc[PRIMARY]
    source_gate = bool(
        primary.onef1b_MAPE_pct < cold.onef1b_MAPE_pct
        and primary.phase_duration_MAE_ms < cold.phase_duration_MAE_ms
    )
    assert math.isclose(
        float(cold.onef1b_MAPE_pct),
        review["reproduction_reference"]["source95_100_cold_start_MAPE_pct"],
        abs_tol=1e-12,
    )
    assert source_gate
    for item in json.loads((out / "source_early_prediction_seal.json").read_text())["files"]:
        assert sha(out / item["path"]) == item["sha256"]

    target_phases = pd.read_csv(paths["target_phase_rank_events_60_100.csv"])
    target_base = RoleMappedReadinessCosts(aligned, pairs, FIT, pp=14, mb=3, role_transfer=True)
    target_bundle, target_edges = build_prediction_set(
        target_phases[target_phases.iteration.isin(TARGET_DEVELOPMENT)],
        target_base, 14, 3, TARGET_DEVELOPMENT, reliability,
    )
    target_features, target_factors, target_predictions, target_nodes = target_bundle
    mutated_target = target_phases[target_phases.iteration.isin(TARGET_DEVELOPMENT)].copy()
    mutated_target.loc[~early_prefix_mask(mutated_target, 14), "duration_ns"] = -999999999
    changed_target_bundle, changed_target_edges = build_prediction_set(
        mutated_target, target_base, 14, 3, TARGET_DEVELOPMENT, reliability,
    )
    for original, changed in zip(target_bundle, changed_target_bundle):
        pd.testing.assert_frame_equal(original, changed, check_exact=True)
    pd.testing.assert_frame_equal(target_edges, changed_target_edges, check_exact=True)
    target_intervals = interval_predictions(
        target_phases, target_base, 14, 3, TARGET_DEVELOPMENT, reliability
    )
    ranges = source_factors[
        source_factors.iteration.isin(FIT) & source_factors.method.eq(PRIMARY)
    ].agg({
        "raw_forward_factor": ["min", "max"],
        "raw_backward_factor": ["min", "max"],
    })
    target_factors["raw_forward_within_source_fit_range"] = target_factors.raw_forward_factor.between(
        ranges.loc["min", "raw_forward_factor"], ranges.loc["max", "raw_forward_factor"]
    )
    target_factors["raw_backward_within_source_fit_range"] = target_factors.raw_backward_factor.between(
        ranges.loc["min", "raw_backward_factor"], ranges.loc["max", "raw_backward_factor"]
    )
    csv(out, "target_early_prefix_features.csv.gz", target_features)
    csv(out, "target_early_factors.csv", target_factors)
    csv(out, "target_early_iteration_predictions.csv", target_predictions)
    csv(out, "target_early_interval_predictions.csv", target_intervals)
    csv(out, "target_early_nodes.csv.gz", target_nodes)
    csv(out, "target_early_edges.csv.gz", target_edges)
    target_files = [
        "target_early_prefix_features.csv.gz", "target_early_factors.csv",
        "target_early_iteration_predictions.csv", "target_early_interval_predictions.csv",
        "target_early_nodes.csv.gz", "target_early_edges.csv.gz",
    ]
    dump(out / "target_early_prediction_seal.json", {
        "status": "SEALED_EARLY_PREFIX_PREDICTIONS_BEFORE_NONPREFIX_TRUTH_SCORING",
        "source_estimator_seal_sha256": sha(out / "source_early_prediction_seal.json"),
        "primary_selected_from_source_gate": PRIMARY,
        "target_nonprefix_mutation_invariant": True,
        "target_parameter_updates": 0, "formal_topology_replaced": False,
        "files": [{"path": name, "sha256": sha(out / name)} for name in target_files],
    })
    target_sealed = {
        item["path"]: item["sha256"]
        for item in json.loads((out / "target_early_prediction_seal.json").read_text())["files"]
    }

    target_actual = _actual_envelope(target_phases, TARGET_DEVELOPMENT)
    target_results, target_metrics = score_iterations(target_predictions, target_actual, "target_development")
    target_phase_results, target_phase_metrics = score_phase(target_nodes, target_phases)
    target_interval_results = score_interval(target_intervals, target_actual)
    target_cutoffs = cutoff_scores(target_phases, target_predictions, 14, target_actual, "target224")
    baseline_record, _, formal_contract = baseline(paths)
    step_results, step_ledger, step_metrics = target_step_score(
        target_predictions, target_actual, paths, baseline_record
    )
    shapley = target_direction_shapley(target_base, target_factors)
    primary_nodes = target_nodes[target_nodes.method.eq(PRIMARY)]
    critical_rows = []
    for iteration in TARGET_DEVELOPMENT:
        frame = primary_nodes[primary_nodes.iteration.eq(iteration)].copy()
        regions = frame.set_index("node_id").region.to_dict()
        for row in critical_ledger(frame, PRIMARY, f"target224_iteration{iteration}"):
            row["region"] = regions[row["node_id"]]
            row["iteration"] = iteration
            critical_rows.append(row)
    csv(out, "target_early_iteration_results.csv", target_results)
    csv(out, "target_early_metrics.csv", target_metrics)
    csv(out, "target_early_phase_results.csv.gz", target_phase_results)
    csv(out, "target_early_phase_metrics.csv", target_phase_metrics)
    csv(out, "target_early_interval_results.csv", target_interval_results)
    csv(out, "target_early_prefix_cutoffs.csv", target_cutoffs)
    csv(out, "target_early_direction_shapley.csv", shapley)
    csv(out, "target_early_critical_path_ledger.csv.gz", pd.DataFrame(critical_rows))
    csv(out, "target_step_iteration_results.csv", step_results)
    csv(out, "target_step_phase_ledger.csv", step_ledger)
    csv(out, "target_step_metrics.csv", step_metrics)
    dump(out / "formal_v685_prediction_contract.json", formal_contract)
    for name, expected in target_sealed.items():
        assert sha(out / name) == expected
    step_index = step_metrics.set_index("variant")
    assert math.isclose(
        float(step_index.loc["early_cold_control", "onef1b_MAPE_pct"]),
        review["reproduction_reference"]["target_phase_transfer_cold_start_MAPE_pct"],
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(step_index.loc["v685_frozen", "onef1b_MAPE_pct"]),
        review["reproduction_reference"]["formal_v685_target_onef1b_MAPE_pct"],
        abs_tol=1e-12,
    )
    target_primary = step_index.loc[PRIMARY]
    target_phase_primary = target_phase_metrics[
        target_phase_metrics.method.eq(PRIMARY)
        & target_phase_metrics.region.eq("all")
        & target_phase_metrics.phase.eq("all")
    ].iloc[0]
    _draw(out, source_metrics, step_metrics, target_cutoffs, target_interval_results)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= review["resource"]["maximum_peak_RSS_bytes"]
    assert elapsed <= review["resource"]["target_analysis_seconds"]
    dump(out / "field_contract.json", {
        "online_prefix": "All-stage F0 plus last-four-stage B0 is an explicit same-iteration condition.",
        "reliability": "One scalar is fitted from source85/90 max-plus responses and applied to both direction-factor deviations from one.",
        "phase_ownership": "The existing aggregate phase wall is scaled once; local runtime and PP costs are unchanged.",
        "uncertainty": "Coefficient endpoints are the two source-fit single-iteration response ratios; coverage is diagnostic, not a calibrated confidence interval.",
        "evaluation": "Source95/100 and target85/90/95/100 are exposed development sets; mutation isolation does not make them blind.",
    })
    dump(out / "diagnostic.json", {
        "status": "EARLIER_ONLINE_PREFIX_EVALUATED_NO_COLD_START_PROMOTION",
        "new_prediction": True, "online_prediction": True,
        "source_gate_passed": source_gate, "primary_candidate": PRIMARY,
        "reliability_coefficient": reliability["reliability_coefficient"],
        "reliability_interval": [reliability["coefficient_interval_low"], reliability["coefficient_interval_high"]],
        "source95_100_control_MAPE_pct": float(cold.onef1b_MAPE_pct),
        "source95_100_primary_MAPE_pct": float(primary.onef1b_MAPE_pct),
        "source95_100_control_phase_MAE_ms": float(cold.phase_duration_MAE_ms),
        "source95_100_primary_phase_MAE_ms": float(primary.phase_duration_MAE_ms),
        "target_primary_online_onef1b_MAPE_pct": float(target_primary.onef1b_MAPE_pct),
        "target_primary_onef1b_bias_ms": float(target_primary.onef1b_bias_ms),
        "target_formal_v685_cold_start_MAPE_pct": float(step_index.loc["v685_frozen", "onef1b_MAPE_pct"]),
        "target_primary_profiler_MAPE_pct": float(target_primary.profiler_MAPE_pct),
        "target_primary_training_MAPE_pct": float(target_primary.training_MAPE_pct),
        "target_primary_MFU_relative_MAPE_pct": float(target_primary.MFU_relative_MAPE_pct),
        "target_primary_phase_MAE_ms": float(target_phase_primary.duration_MAE_ms),
        "target_primary_phase_bias_ms": float(target_phase_primary.duration_bias_ms),
        "target_prefix_mean_fraction_pct": float(target_cutoffs.prefix_available_fraction_pct.mean()),
        "target_interval_coverage_iterations": int(target_interval_results.covered.sum()),
        "target_interval_total_iterations": len(target_interval_results),
        "target_raw_factors_all_within_source_fit_range": bool(target_factors[["raw_forward_within_source_fit_range", "raw_backward_within_source_fit_range"]].all(axis=None)),
        "target_parameter_updates": 0, "formal_topology_replaced": False,
        "source_prediction_seal_sha256": sha(out / "source_early_prediction_seal.json"),
        "target_prediction_seal_sha256": sha(out / "target_early_prediction_seal.json"),
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed,
        "decision": "Keep the candidate as an earlier online development nowcast. Source-fitted shrinkage and explicit interval do not supply the missing cold-start runtime state.",
        "next": "Compare T35 and T36 by forecast remaining-time error and uncertainty calibration; improve the operational output without selecting on target total error.",
    })

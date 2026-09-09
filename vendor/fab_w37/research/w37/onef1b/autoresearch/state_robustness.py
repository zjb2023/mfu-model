"""T39A: fixed-method robustness audit for the T38 target-local state update."""
from __future__ import annotations

import json
import math
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ITERATIONS = [85, 90, 95, 100]
PRIMARY = [90, 95, 100]
COMMON_SUPPORT = [95, 100]
METHODS = [
    "zero_update", "lag_one_base_residual", "stale_two_step_residual",
    "expanding_past_mean_residual",
]


def state_correction(
    method: str, iteration: int, base: dict[int, float], actual: dict[int, float],
    ordered_iterations: list[int] = ITERATIONS,
) -> tuple[float, list[int], float | None]:
    """Return correction, truth dependencies, and mean state age in sequence steps."""
    position = ordered_iterations.index(iteration)
    earlier = ordered_iterations[:position]
    residual = lambda item: float(actual[item]) - float(base[item])
    if method == "zero_update" or not earlier:
        dependencies: list[int] = []
    elif method == "lag_one_base_residual":
        dependencies = [earlier[-1]]
    elif method == "stale_two_step_residual":
        dependencies = [] if position < 2 else [ordered_iterations[position - 2]]
    elif method == "expanding_past_mean_residual":
        dependencies = earlier
    else:
        raise ValueError(method)
    correction = float(np.mean([residual(item) for item in dependencies])) if dependencies else 0.0
    ages = [position - ordered_iterations.index(item) for item in dependencies]
    return correction, dependencies, float(np.mean(ages)) if ages else None


def prediction_rows(base_frame: pd.DataFrame, actual: dict[int, float]) -> pd.DataFrame:
    base = base_frame.set_index("iteration").predicted_onef1b_ms.to_dict()
    rows = []
    for iteration in PRIMARY:
        for method in METHODS:
            correction, dependencies, mean_age = state_correction(method, iteration, base, actual)
            rows.append({
                "method": method, "iteration": iteration,
                "base_predicted_onef1b_ms": float(base[iteration]),
                "correction_ms": correction,
                "predicted_onef1b_ms": float(base[iteration]) + correction,
                "truth_dependency_iterations": json.dumps(dependencies, separators=(",", ":")),
                "history_count": len(dependencies), "mean_state_age_iterations": mean_age,
                "maximum_truth_dependency_iteration": max(dependencies) if dependencies else None,
                "strictly_past_only": all(item < iteration for item in dependencies),
                "selection_role": "fixed_T38_method" if method == "lag_one_base_residual" else "robustness_ablation_only",
            })
    result = pd.DataFrame(rows)
    assert result.strictly_past_only.all()
    return result


def shock_response(base_frame: pd.DataFrame, actual: dict[int, float], shock_ms: float = 500.0) -> pd.DataFrame:
    base = base_frame.set_index("iteration").predicted_onef1b_ms.to_dict()
    rows = []
    for iteration in PRIMARY:
        for source in [item for item in ITERATIONS if item < iteration]:
            mutated = dict(actual)
            mutated[source] += shock_ms
            for method in METHODS:
                original, dependencies, _ = state_correction(method, iteration, base, actual)
                changed, _, _ = state_correction(method, iteration, base, mutated)
                response = changed - original
                expected = shock_ms / len(dependencies) if source in dependencies else 0.0
                assert math.isclose(response, expected, abs_tol=1e-12)
                rows.append({
                    "method": method, "prediction_iteration": iteration,
                    "shocked_truth_iteration": source, "shock_ms": shock_ms,
                    "prediction_response_ms": response,
                    "dependency_active": source in dependencies,
                })
    return pd.DataFrame(rows)


def _verify_prior(ref: dict, status: str, tests: int) -> dict:
    from guards import checked_stage
    from smoke_worker import sha

    stage = Path(ref["stage_root"])
    acceptance_path = Path(ref["acceptance_path"])
    assert sha(stage / "run_manifest.json") == ref["manifest_sha256"]
    checked_stage(stage)
    assert sha(acceptance_path) == ref["acceptance_sha256"]
    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance["status"] == status and acceptance["tests_passed"] == tests
    assert acceptance["diagnose_manifest_sha256"] == ref["manifest_sha256"]
    return acceptance


def _score(predictions: pd.DataFrame, t38_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_rows = t38_results[t38_results.method.eq("t35_midrun_winsor20")].set_index("iteration")
    rows = []
    for prediction in predictions.itertuples(index=False):
        truth = base_rows.loc[prediction.iteration]
        correction = float(prediction.correction_ms)
        onef1b = float(prediction.predicted_onef1b_ms)
        profiler = float(truth.predicted_profiler_ms) + correction
        training = float(truth.predicted_training_ms) + correction
        predicted_mfu = float(truth.predicted_MFU_pct) * float(truth.predicted_training_ms) / training
        actual_remaining = float(truth.actual_remaining_ms)
        predicted_remaining = onef1b - float(truth.prefix_available_elapsed_ms)
        base_abs = abs(float(truth.onef1b_error_ms))
        error = onef1b - float(truth.actual_onef1b_ms)
        rows.append({
            **prediction._asdict(),
            "actual_onef1b_ms": float(truth.actual_onef1b_ms),
            "onef1b_error_ms": error,
            "onef1b_absolute_error_ms": abs(error),
            "onef1b_APE_pct": 100 * abs(error) / float(truth.actual_onef1b_ms),
            "absolute_error_reduction_vs_zero_ms": base_abs - abs(error),
            "prefix_available_elapsed_ms": float(truth.prefix_available_elapsed_ms),
            "actual_remaining_ms": actual_remaining, "predicted_remaining_ms": predicted_remaining,
            "remaining_APE_pct": 100 * abs(predicted_remaining - actual_remaining) / actual_remaining,
            "predicted_profiler_ms": profiler,
            "profiler_APE_pct": 100 * abs(profiler - float(truth.actual_profiler_ms)) / float(truth.actual_profiler_ms),
            "predicted_training_ms": training,
            "training_APE_pct": 100 * abs(training - float(truth.actual_training_ms)) / float(truth.actual_training_ms),
            "predicted_MFU_pct": predicted_mfu,
            "MFU_relative_APE_pct": 100 * abs(predicted_mfu - float(truth.actual_MFU_pct_derived)) / float(truth.actual_MFU_pct_derived),
        })
    results = pd.DataFrame(rows)
    summaries = []
    for split, iterations in [("primary_90_100", PRIMARY), ("common_support_95_100", COMMON_SUPPORT)]:
        frame = results[results.iteration.isin(iterations)]
        for method, group in frame.groupby("method", sort=False):
            summaries.append({
                "split": split, "method": method, "iterations": len(group),
                "onef1b_MAPE_pct": group.onef1b_APE_pct.mean(),
                "onef1b_MAE_ms": group.onef1b_absolute_error_ms.mean(),
                "onef1b_bias_ms": group.onef1b_error_ms.mean(),
                "onef1b_worst_APE_pct": group.onef1b_APE_pct.max(),
                "iterations_improved_vs_zero": int(group.absolute_error_reduction_vs_zero_ms.gt(0).sum()),
                "mean_absolute_error_reduction_vs_zero_ms": group.absolute_error_reduction_vs_zero_ms.mean(),
                "remaining_MAPE_pct": group.remaining_APE_pct.mean(),
                "profiler_MAPE_pct": group.profiler_APE_pct.mean(),
                "training_MAPE_pct": group.training_APE_pct.mean(),
                "MFU_relative_MAPE_pct": group.MFU_relative_APE_pct.mean(),
                "mean_history_count": group.history_count.mean(),
                "mean_state_age_iterations": group.mean_state_age_iterations.mean(),
                "selection_status": "fixed_T38_method" if method == "lag_one_base_residual" else "posthoc_ablation_not_selectable",
            })
    metrics = pd.DataFrame(summaries)
    metrics["posthoc_error_rank_within_split"] = metrics.groupby("split").onef1b_MAPE_pct.rank(method="min").astype(int)
    return results, metrics


def _draw(out: Path, results: pd.DataFrame, metrics: pd.DataFrame) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams["svg.hashsalt"] = "w37-t39a"
    import matplotlib.pyplot as plt

    labels = {
        "zero_update": "zero/T35", "lag_one_base_residual": "lag-one/T38",
        "stale_two_step_residual": "stale-2", "expanding_past_mean_residual": "expanding mean",
    }
    colors = {
        "zero_update": "#777777", "lag_one_base_residual": "#009E73",
        "stale_two_step_residual": "#E69F00", "expanding_past_mean_residual": "#0072B2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for method, group in results.groupby("method", sort=False):
        axes[0].plot(group.iteration, group.onef1b_APE_pct, marker="o", linewidth=1.7,
                     color=colors[method], label=labels[method])
    axes[0].set_title("State-age robustness by iteration")
    axes[0].set_xlabel("Target development iteration")
    axes[0].set_ylabel("1F1B APE (%)")
    axes[0].set_xticks(PRIMARY)
    axes[0].grid(True, alpha=.25)
    axes[0].legend(fontsize=8)
    primary = metrics[metrics.split.eq("primary_90_100")]
    axes[1].bar([labels[m] for m in primary.method], primary.onef1b_MAPE_pct,
                color=[colors[m] for m in primary.method])
    axes[1].set_title("Three-transition posthoc aggregate")
    axes[1].set_ylabel("1F1B MAPE (%)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "state_robustness.svg", metadata={"Date": None})
    fig.savefig(out / "state_robustness.png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    from smoke_worker import dump, sha
    from worker import csv

    begin = perf_counter()
    refs = plan["sealed_diagnostic_stages"]
    assert [ref["name"] for ref in refs] == ["t35", "t38"]
    accept35 = _verify_prior(refs[0], "PASS_ONLINE_PREFIX_NOWCAST_IMPROVES_DEVELOPMENT_COLD_START_STILL_BLOCKED", 69)
    accept38 = _verify_prior(refs[1], "PASS_CAUSAL_TARGET_LAGGED_ONLINE_UPDATE_DEVELOPMENT", 76)
    assert sha(paths["t35_diagnose_target_online_prediction_seal_json"]) == accept35["target_prediction_seal_sha256"]
    assert sha(paths["t38_diagnose_causal_prediction_seal_json"]) == accept38["causal_prediction_seal_sha256"]

    t35_all = pd.read_csv(paths["t35_diagnose_target_online_iteration_predictions_csv"])
    base = t35_all[t35_all.method.eq("online_midrun_last_half_winsor20")].sort_values("iteration")
    t38_results = pd.read_csv(paths["t38_diagnose_target_iteration_results_csv"])
    base_truth = t38_results[t38_results.method.eq("t35_midrun_winsor20")].sort_values("iteration")
    actual = base_truth.set_index("iteration").actual_onef1b_ms.to_dict()
    assert np.allclose(base.predicted_onef1b_ms, base_truth.predicted_onef1b_ms, rtol=0, atol=1e-9)

    predictions = prediction_rows(base, actual)
    sealed_t38 = pd.read_csv(paths["t38_diagnose_causal_prediction_ledger_csv"]).set_index("iteration")
    recomputed = predictions[predictions.method.eq("lag_one_base_residual")].set_index("iteration")
    sealed_values = sealed_t38.loc[PRIMARY].predicted_onef1b_ms
    lag_recompute_max_abs_difference_ms = float(
        np.max(np.abs(recomputed.predicted_onef1b_ms.to_numpy() - sealed_values.to_numpy()))
    )
    assert lag_recompute_max_abs_difference_ms <= 1e-6
    shocks = shock_response(base, actual)
    results, metrics = _score(predictions, t38_results)
    csv(out, "state_prediction_ledger.csv", predictions)
    csv(out, "state_iteration_results.csv", results)
    csv(out, "state_robustness_metrics.csv", metrics)
    csv(out, "synthetic_shock_response.csv", shocks)
    _draw(out, results, metrics)

    idx = metrics[metrics.split.eq("primary_90_100")].set_index("method")
    selected = idx.loc["lag_one_base_residual"]
    smooth = idx.loc["expanding_past_mean_residual"]
    stale = idx.loc["stale_two_step_residual"]
    zero = idx.loc["zero_update"]
    assert selected.onef1b_MAPE_pct < zero.onef1b_MAPE_pct
    assert selected.iterations_improved_vs_zero == len(PRIMARY)
    assert smooth.onef1b_MAPE_pct < selected.onef1b_MAPE_pct
    assert stale.onef1b_MAPE_pct < zero.onef1b_MAPE_pct
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= plan["resource"]["maximum_peak_RSS_bytes"] and elapsed <= plan["resource"]["target_analysis_seconds"]
    dump(out / "ablation_contract.json", {
        "selected_method_before_audit": "lag_one_base_residual",
        "selection_changed": False, "ablations": plan["ablations"],
        "physical_attribution": "None. All alternatives act only on the separately labelled target-local temporal residual state.",
    })
    dump(out / "diagnostic.json", {
        "status": "FIXED_TARGET_STATE_ROBUSTNESS_AUDIT_PASS",
        "new_prediction": False, "diagnostic_counterfactual_predictions": True,
        "selected_method_unchanged": "lag_one_base_residual",
        "target_data_scope": "three already exposed causal development transitions",
        "primary_onef1b_MAPE_pct": {method: float(idx.loc[method, "onef1b_MAPE_pct"]) for method in METHODS},
        "primary_remaining_MAPE_pct": {method: float(idx.loc[method, "remaining_MAPE_pct"]) for method in METHODS},
        "lag_one_improved_iterations_vs_zero": int(selected.iterations_improved_vs_zero),
        "lag_one_recompute_max_abs_difference_ms": lag_recompute_max_abs_difference_ms,
        "lag_one_recompute_tolerance_ms": 1e-6,
        "posthoc_aggregate_best_method_not_selected": "expanding_past_mean_residual",
        "lag_one_minus_expanding_mean_MAPE_pp": float(selected.onef1b_MAPE_pct - smooth.onef1b_MAPE_pct),
        "stale_iteration90_fallback": "zero_update_due_to_no_two-step_history",
        "synthetic_shock_ms": 500.0,
        "formal_topology_replaced": False, "added_edges_or_waits": 0,
        "target_model_parameter_updates": 0, "selected_method_updates": 0,
        "prior_manifest_hashes": {ref["name"]: ref["manifest_sha256"] for ref in refs},
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed,
        "decision": "Lag-one improves all three transitions over zero update, but expanding-history mean is slightly better in the same exposed aggregate. Three transitions do not identify a unique state rule, so T38 remains fixed and no new method is selected.",
        "next": "Quantify error bars and break-even tolerances for T38 versus T35, then stop target-method search unless new sequential target iterations become available.",
    })

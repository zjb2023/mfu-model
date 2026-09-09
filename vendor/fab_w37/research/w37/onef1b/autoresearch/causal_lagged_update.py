"""T38A: target-local causal lag-one correction above the sealed T35 nowcast."""
from __future__ import annotations

import hashlib
import json
import math
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ITERATIONS = [85, 90, 95, 100]
WALK_FORWARD = [90, 95, 100]
BASE = "t35_midrun_winsor20"
UPDATED = "t38_lagged_base_residual"
FORMAL = "formal_v685_cold_start"
T35_VARIANT = "online_midrun_last_half_winsor20"


def causal_prediction(
    iteration: int, base_predictions: dict[int, float], actual_history: dict[int, float],
    ordered_iterations: list[int] = ITERATIONS,
) -> tuple[float, int | None, float]:
    """Predict t from sealed base(t) and the base residual at predecessor(t) only."""
    position = ordered_iterations.index(iteration)
    if position == 0:
        return float(base_predictions[iteration]), None, 0.0
    predecessor = ordered_iterations[position - 1]
    correction = float(actual_history[predecessor]) - float(base_predictions[predecessor])
    return float(base_predictions[iteration]) + correction, predecessor, correction


def sequential_predictions(base: pd.DataFrame, actual: dict[int, float]) -> pd.DataFrame:
    """Materialize the causal sequence and a deterministic row hash chain."""
    base_map = base.set_index("iteration").predicted_onef1b_ms.to_dict()
    rows = []
    previous_hash = "GENESIS"
    for iteration in ITERATIONS:
        prediction, predecessor, correction = causal_prediction(iteration, base_map, actual)
        row = {
            "method": UPDATED, "iteration": iteration,
            "base_method": BASE, "base_predicted_onef1b_ms": float(base_map[iteration]),
            "predecessor_iteration": predecessor,
            "predecessor_actual_onef1b_ms": None if predecessor is None else float(actual[predecessor]),
            "predecessor_base_predicted_onef1b_ms": None if predecessor is None else float(base_map[predecessor]),
            "lagged_base_residual_correction_ms": correction,
            "predicted_onef1b_ms": prediction,
            "truth_dependency_max_iteration": predecessor,
            "previous_prediction_hash": previous_hash,
        }
        payload = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        row_hash = hashlib.sha256(payload.encode()).hexdigest()
        row["prediction_hash"] = row_hash
        rows.append(row)
        previous_hash = row_hash
    result = pd.DataFrame(rows)
    assert result[result.iteration.eq(85)].predecessor_iteration.isna().all()
    assert (result[result.iteration.gt(85)].truth_dependency_max_iteration < result[result.iteration.gt(85)].iteration).all()
    return result


def causal_interval_rows(predictions: pd.DataFrame, actual: dict[int, float]) -> pd.DataFrame:
    """Use only already scored corrected errors for a running max-absolute band."""
    pred = predictions.set_index("iteration").predicted_onef1b_ms.to_dict()
    realized_errors: list[float] = []
    rows = []
    for iteration in WALK_FORWARD:
        if realized_errors:
            radius = max(abs(value) for value in realized_errors)
            low, high = pred[iteration] - radius, pred[iteration] + radius
            covered = low <= actual[iteration] <= high
        else:
            radius = low = high = None
            covered = None
        rows.append({
            "iteration": iteration, "history_corrected_errors": len(realized_errors),
            "available": bool(realized_errors), "half_width_ms": radius,
            "predicted_interval_low_ms": low, "predicted_interval_high_ms": high,
            "actual_onef1b_ms": actual[iteration], "covered": covered,
            "scope": "causal finite-history diagnostic band; not calibrated confidence",
        })
        realized_errors.append(pred[iteration] - actual[iteration])
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


def _causality_audit(base: pd.DataFrame, actual: dict[int, float]) -> pd.DataFrame:
    base_map = base.set_index("iteration").predicted_onef1b_ms.to_dict()
    rows = []
    for iteration in ITERATIONS:
        original, predecessor, _ = causal_prediction(iteration, base_map, actual)
        mutated = {key: value + (1_000_000.0 if key >= iteration else 0.0) for key, value in actual.items()}
        current_or_future_mutated, _, _ = causal_prediction(iteration, base_map, mutated)
        invariant = original == current_or_future_mutated
        predecessor_sensitivity = None
        if predecessor is not None:
            predecessor_mutated = dict(actual)
            predecessor_mutated[predecessor] += 123.0
            changed, _, _ = causal_prediction(iteration, base_map, predecessor_mutated)
            predecessor_sensitivity = changed - original
            assert math.isclose(predecessor_sensitivity, 123.0, abs_tol=1e-12)
        assert invariant
        rows.append({
            "iteration": iteration, "predecessor_iteration": predecessor,
            "current_and_future_truth_mutation_ms": 1_000_000.0,
            "prediction_invariant_to_current_and_future_truth": invariant,
            "predecessor_truth_mutation_ms": None if predecessor is None else 123.0,
            "prediction_change_from_predecessor_mutation_ms": predecessor_sensitivity,
        })
    return pd.DataFrame(rows)


def _score(
    predictions: pd.DataFrame, step: pd.DataFrame, cutoffs: pd.DataFrame,
    phase_input: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = step[step.variant.eq(T35_VARIANT)].set_index("iteration")
    formal = step[step.variant.eq("v685_frozen")].set_index("iteration")
    base_pred = predictions.set_index("iteration")
    cutoff = cutoffs[cutoffs.method.eq(T35_VARIANT)].set_index("iteration")
    assert list(truth.index) == ITERATIONS and list(formal.index) == ITERATIONS
    rows = []
    for iteration in ITERATIONS:
        actual = truth.loc[iteration]
        for method in [FORMAL, BASE, UPDATED]:
            if method == FORMAL:
                original = formal.loc[iteration]
                predicted_onef1b = float(original.predicted_onef1b_ms)
                predicted_profiler = float(original.predicted_profiler_ms)
                predicted_training = float(original.predicted_training_ms)
                predicted_mfu = float(original.predicted_MFU_pct)
                elapsed = 0.0
                correction = 0.0
            elif method == BASE:
                original = actual
                predicted_onef1b = float(actual.predicted_onef1b_ms)
                predicted_profiler = float(actual.predicted_profiler_ms)
                predicted_training = float(actual.predicted_training_ms)
                predicted_mfu = float(actual.predicted_MFU_pct)
                elapsed = float(cutoff.loc[iteration, "prefix_available_elapsed_ms"])
                correction = 0.0
            else:
                original = actual
                predicted_onef1b = float(base_pred.loc[iteration, "predicted_onef1b_ms"])
                correction = float(base_pred.loc[iteration, "lagged_base_residual_correction_ms"])
                predicted_profiler = float(actual.predicted_profiler_ms) + correction
                predicted_training = float(actual.predicted_training_ms) + correction
                predicted_mfu = float(actual.predicted_MFU_pct) * float(actual.predicted_training_ms) / predicted_training
                elapsed = float(cutoff.loc[iteration, "prefix_available_elapsed_ms"])
            actual_remaining = float(actual.actual_onef1b_ms) - elapsed
            predicted_remaining = predicted_onef1b - elapsed
            row = {
                "method": method, "iteration": iteration,
                "evaluation_role": "initialization" if iteration == 85 else "causal_walk_forward",
                "prefix_available_elapsed_ms": elapsed,
                "cutoff_fraction_pct": 100 * elapsed / float(actual.actual_onef1b_ms),
                "actual_remaining_ms": actual_remaining, "predicted_remaining_ms": predicted_remaining,
                "remaining_error_ms": predicted_remaining - actual_remaining,
                "remaining_APE_pct": 100 * abs(predicted_remaining - actual_remaining) / actual_remaining,
                "actual_onef1b_ms": float(actual.actual_onef1b_ms), "predicted_onef1b_ms": predicted_onef1b,
                "onef1b_error_ms": predicted_onef1b - float(actual.actual_onef1b_ms),
                "onef1b_APE_pct": 100 * abs(predicted_onef1b - float(actual.actual_onef1b_ms)) / float(actual.actual_onef1b_ms),
                "actual_profiler_ms": float(actual.actual_profiler_ms), "predicted_profiler_ms": predicted_profiler,
                "profiler_APE_pct": 100 * abs(predicted_profiler - float(actual.actual_profiler_ms)) / float(actual.actual_profiler_ms),
                "actual_training_ms": float(actual.actual_training_ms), "predicted_training_ms": predicted_training,
                "training_APE_pct": 100 * abs(predicted_training - float(actual.actual_training_ms)) / float(actual.actual_training_ms),
                "actual_MFU_pct_derived": float(actual.actual_MFU_pct_derived), "predicted_MFU_pct": predicted_mfu,
                "MFU_error_pp": predicted_mfu - float(actual.actual_MFU_pct_derived),
                "MFU_relative_APE_pct": 100 * abs(predicted_mfu - float(actual.actual_MFU_pct_derived)) / float(actual.actual_MFU_pct_derived),
                "lagged_base_residual_correction_ms": correction,
            }
            assert math.isclose(row["remaining_error_ms"], row["onef1b_error_ms"], abs_tol=1e-9)
            rows.append(row)
    scored = pd.DataFrame(rows)
    summaries = []
    for name, iterations in [
        ("primary_causal_walk_forward", WALK_FORWARD),
        ("all_including_initialization", ITERATIONS),
    ]:
        selected = scored[scored.iteration.isin(iterations)]
        for method, group in selected.groupby("method", sort=False):
            summaries.append({
                "split": name, "method": method, "iterations": len(group),
                "remaining_MAPE_pct": group.remaining_APE_pct.mean(),
                "onef1b_MAPE_pct": group.onef1b_APE_pct.mean(),
                "onef1b_MAE_ms": group.onef1b_error_ms.abs().mean(),
                "onef1b_bias_ms": group.onef1b_error_ms.mean(),
                "profiler_MAPE_pct": group.profiler_APE_pct.mean(),
                "training_MAPE_pct": group.training_APE_pct.mean(),
                "MFU_relative_MAPE_pct": group.MFU_relative_APE_pct.mean(),
                "MFU_bias_pp": group.MFU_error_pp.mean(),
                "mean_cutoff_fraction_pct": group.cutoff_fraction_pct.mean(),
                "mean_actual_remaining_ms": group.actual_remaining_ms.mean(),
            })
    original_phases = phase_input[phase_input.variant.eq(T35_VARIANT)].copy()
    assert len(original_phases) == len(ITERATIONS) * 4
    original_phases["method"] = UPDATED
    original_phases = original_phases.rename(columns={"predicted_ms": "t35_predicted_ms"})
    correction = predictions.set_index("iteration").lagged_base_residual_correction_ms
    original_phases["prediction_change_ms"] = np.where(
        original_phases.phase.eq("onef1b"), original_phases.iteration.map(correction), 0.0
    )
    original_phases["t38_predicted_ms"] = original_phases.t35_predicted_ms + original_phases.prediction_change_ms
    original_phases["changed_by_lagged_update"] = original_phases.prediction_change_ms.ne(0)
    phases = original_phases[[
        "method", "iteration", "phase", "actual_ms", "t35_predicted_ms", "t38_predicted_ms",
        "prediction_change_ms", "changed_by_lagged_update",
    ]]
    assert phases[phases.phase.ne("onef1b")].prediction_change_ms.eq(0).all()
    return scored, pd.DataFrame(summaries), phases


def _decomposition(predictions: pd.DataFrame, shapley: pd.DataFrame, actual: dict[int, float]) -> pd.DataFrame:
    directions = shapley.pivot(index="iteration", columns="component", values="predicted_onef1b_delta_ms")
    meta = shapley.groupby("iteration").first()
    pred = predictions.set_index("iteration")
    rows = []
    for iteration in ITERATIONS:
        cold = float(meta.loc[iteration, "cold_start_onef1b_ms"])
        forward = float(directions.loc[iteration, "forward_phase_factor"])
        backward = float(directions.loc[iteration, "backward_phase_factor"])
        correction = float(pred.loc[iteration, "lagged_base_residual_correction_ms"])
        final = cold + forward + backward + correction
        assert math.isclose(final, float(pred.loc[iteration, "predicted_onef1b_ms"]), abs_tol=2e-6)
        rows.extend([
            {"iteration": iteration, "component": "cold_start_phase_transfer", "contribution_ms": cold, "attribution_scope": "sealed graph base"},
            {"iteration": iteration, "component": "forward_phase_factor", "contribution_ms": forward, "attribution_scope": "sealed T35 within-iteration graph Shapley"},
            {"iteration": iteration, "component": "backward_phase_factor", "contribution_ms": backward, "attribution_scope": "sealed T35 within-iteration graph Shapley"},
            {"iteration": iteration, "component": "lagged_base_residual_state", "contribution_ms": correction, "attribution_scope": "target-local temporal state; no physical compute/communication/wait attribution"},
            {"iteration": iteration, "component": "unexplained_actual_residual", "contribution_ms": actual[iteration] - final, "attribution_scope": "observed development residual after prediction"},
        ])
    return pd.DataFrame(rows)


def _draw(out: Path, scores: pd.DataFrame, decomposition: pd.DataFrame) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams["svg.hashsalt"] = "w37-t38a"
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    current = scores[scores.iteration.isin(WALK_FORWARD)]
    colors = {FORMAL: "#777777", BASE: "#0072B2", UPDATED: "#009E73"}
    labels = {FORMAL: "v685", BASE: "T35", UPDATED: "T38 lag-one"}
    for method, group in current.groupby("method", sort=False):
        axes[0].plot(group.iteration, group.onef1b_APE_pct, marker="o", linewidth=1.8,
                     color=colors[method], label=labels[method])
    axes[0].set_title("Causal walk-forward 1F1B error")
    axes[0].set_xlabel("Target development iteration")
    axes[0].set_ylabel("APE (%)")
    axes[0].set_xticks(WALK_FORWARD)
    axes[0].grid(True, alpha=.25)
    axes[0].legend()
    components = decomposition[
        decomposition.iteration.isin(WALK_FORWARD)
        & decomposition.component.isin(["forward_phase_factor", "backward_phase_factor", "lagged_base_residual_state"])
    ].pivot(index="iteration", columns="component", values="contribution_ms")
    components[["forward_phase_factor", "backward_phase_factor", "lagged_base_residual_state"]].plot(
        kind="bar", stacked=True, ax=axes[1], color=["#56B4E9", "#E69F00", "#009E73"]
    )
    axes[1].set_title("Correction above phase-transfer base")
    axes[1].set_xlabel("Target development iteration")
    axes[1].set_ylabel("Contribution (ms)")
    axes[1].legend(["F graph", "B graph", "lagged residual"], fontsize=8)
    axes[1].grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "causal_lagged_update.svg", metadata={"Date": None})
    fig.savefig(out / "causal_lagged_update.png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    from smoke_worker import dump, sha
    from worker import csv

    begin = perf_counter()
    refs = plan["sealed_diagnostic_stages"]
    assert [r["name"] for r in refs] == ["t35", "t37"]
    accept35 = _verify_prior(refs[0], "PASS_ONLINE_PREFIX_NOWCAST_IMPROVES_DEVELOPMENT_COLD_START_STILL_BLOCKED", 69)
    accept37 = _verify_prior(refs[1], "PASS_SEALED_ONLINE_REMAINING_TIME_PARETO_AUDIT", 73)
    target_seal_path = paths["t35_diagnose_target_online_prediction_seal_json"]
    assert sha(target_seal_path) == accept35["target_prediction_seal_sha256"]
    target_seal = json.loads(target_seal_path.read_text())
    sealed_files = {item["path"]: item["sha256"] for item in target_seal["files"]}
    assert sha(paths["t35_diagnose_target_online_iteration_predictions_csv"]) == sealed_files["target_online_iteration_predictions.csv"]

    all_predictions = pd.read_csv(paths["t35_diagnose_target_online_iteration_predictions_csv"])
    base = all_predictions[all_predictions.method.eq(T35_VARIANT)].sort_values("iteration")
    assert base.iteration.tolist() == ITERATIONS
    step = pd.read_csv(paths["t35_diagnose_target_step_iteration_results_csv"])
    truth = step[step.variant.eq(T35_VARIANT)].sort_values("iteration")
    actual = truth.set_index("iteration").actual_onef1b_ms.to_dict()
    assert truth.iteration.tolist() == ITERATIONS
    assert np.allclose(base.predicted_onef1b_ms, truth.predicted_onef1b_ms, rtol=0, atol=1e-9)
    predictions = sequential_predictions(base, actual)
    causality = _causality_audit(base, actual)
    csv(out, "causal_prediction_ledger.csv", predictions)
    csv(out, "causality_mutation_audit.csv", causality)
    dump(out / "causal_prediction_seal.json", {
        "status": "SEALED_CAUSAL_SEQUENCE_DEPENDENCY_LEDGER",
        "base_target_prediction_seal_sha256": sha(target_seal_path),
        "algorithm": plan["update_rule"],
        "target_iterations": ITERATIONS, "initialization_iteration": 85,
        "final_prediction_hash": predictions.iloc[-1].prediction_hash,
        "files": [
            {"path": "causal_prediction_ledger.csv", "sha256": sha(out / "causal_prediction_ledger.csv")},
            {"path": "causality_mutation_audit.csv", "sha256": sha(out / "causality_mutation_audit.csv")},
        ],
        "current_or_future_truth_mutation_invariant": bool(causality.prediction_invariant_to_current_and_future_truth.all()),
        "online_state_updates": 3, "target_model_parameter_updates": 0,
    })
    sealed = json.loads((out / "causal_prediction_seal.json").read_text())

    cutoffs = pd.read_csv(paths["t35_diagnose_target_online_prefix_cutoffs_csv"])
    phase_input = pd.read_csv(paths["t35_diagnose_target_step_phase_ledger_csv"])
    scores, metrics, phases = _score(predictions, step, cutoffs, phase_input)
    intervals = causal_interval_rows(predictions, actual)
    shapley = pd.read_csv(paths["t35_diagnose_target_online_direction_shapley_csv"])
    decomposition = _decomposition(predictions, shapley, actual)
    csv(out, "target_iteration_results.csv", scores)
    csv(out, "target_metrics.csv", metrics)
    csv(out, "target_phase_ledger.csv", phases)
    csv(out, "causal_interval_results.csv", intervals)
    csv(out, "prediction_decomposition.csv", decomposition)
    for item in sealed["files"]:
        assert sha(out / item["path"]) == item["sha256"]

    metric_index = metrics.set_index(["split", "method"])
    primary = metric_index.loc[("primary_causal_walk_forward", UPDATED)]
    base_metric = metric_index.loc[("primary_causal_walk_forward", BASE)]
    formal_metric = metric_index.loc[("primary_causal_walk_forward", FORMAL)]
    all_updated = metric_index.loc[("all_including_initialization", UPDATED)]
    assert primary.onef1b_MAPE_pct < base_metric.onef1b_MAPE_pct < formal_metric.onef1b_MAPE_pct
    available_intervals = intervals[intervals.available]
    _draw(out, scores, decomposition)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= plan["resource"]["maximum_peak_RSS_bytes"] and elapsed <= plan["resource"]["target_analysis_seconds"]
    dump(out / "accounting_contract.json", {
        "graph": "Exact sealed T35 prediction; no node, edge, F/B factor, PP or local-runtime change.",
        "temporal_state": "One prior base residual added once at the final 1F1B envelope.",
        "phase_scope": "Only onef1b changes; entry, tail and outer remain inherited. Tail is verified from the original phase ledger input.",
        "attribution": "F/B Shapley explains the sealed graph correction; lagged residual is separately labelled and has no physical component attribution.",
    })
    dump(out / "diagnostic.json", {
        "status": "CAUSAL_TARGET_LAGGED_ONLINE_UPDATE_DEVELOPMENT_PASS",
        "new_prediction": True, "prediction_mode": "target-domain causal online state update",
        "target_data_scope": "already exposed sequential development evaluation; iteration85 initialization",
        "primary_iterations": WALK_FORWARD,
        "primary_onef1b_MAPE_pct": float(primary.onef1b_MAPE_pct),
        "base_t35_onef1b_MAPE_pct": float(base_metric.onef1b_MAPE_pct),
        "formal_v685_onef1b_MAPE_pct": float(formal_metric.onef1b_MAPE_pct),
        "primary_remaining_MAPE_pct": float(primary.remaining_MAPE_pct),
        "primary_profiler_MAPE_pct": float(primary.profiler_MAPE_pct),
        "primary_training_MAPE_pct": float(primary.training_MAPE_pct),
        "primary_MFU_relative_MAPE_pct": float(primary.MFU_relative_MAPE_pct),
        "all_including_initialization_onef1b_MAPE_pct": float(all_updated.onef1b_MAPE_pct),
        "all_including_initialization_remaining_MAPE_pct": float(all_updated.remaining_MAPE_pct),
        "causal_interval_coverage": [int(available_intervals.covered.sum()), len(available_intervals)],
        "mean_walk_forward_lagged_correction_ms": float(predictions[predictions.iteration.isin(WALK_FORWARD)].lagged_base_residual_correction_ms.mean()),
        "mean_walk_forward_residual_ms": float(scores[(scores.method.eq(UPDATED)) & scores.iteration.isin(WALK_FORWARD)].onef1b_error_ms.mean()),
        "current_or_future_truth_mutation_invariant": True,
        "online_state_updates": 3, "target_model_parameter_updates": 0,
        "formal_topology_replaced": False, "added_edges_or_waits": 0,
        "entry_tail_outer_updates": 0,
        "causal_prediction_seal_sha256": sha(out / "causal_prediction_seal.json"),
        "prior_manifest_hashes": {r["name"]: r["manifest_sha256"] for r in refs},
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed,
        "decision": "The lagged base residual materially improves the three causal walk-forward development iterations while preserving T35's graph explanation. The result requires one completed target iteration and is not a cold-start or source-transfer promotion.",
        "next": "Stress-test the causal update against stale-state, zero-update and expanding-history ablations without selecting a new rule from these four exposed target iterations.",
    })

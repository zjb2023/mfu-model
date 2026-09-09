"""T40A: homologous source audit and break-even bounds for fixed T38 lag-one state."""
from __future__ import annotations

import json
import math
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ITERATIONS = [85, 90, 95, 100]
WALK = [90, 95, 100]
T35 = "online_midrun_last_half_winsor20"


def exact_one_sided_sign_p(improved: int, total: int) -> float:
    assert 0 <= improved <= total
    return sum(math.comb(total, k) for k in range(improved, total + 1)) / (2 ** total)


def transition_audit(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Apply fixed lag-one and derive its exact additive state-noise improvement interval."""
    data = frame.sort_values("iteration").set_index("iteration")
    assert list(data.index) == ITERATIONS
    rows = []
    for position, iteration in enumerate(ITERATIONS[1:], start=1):
        predecessor = ITERATIONS[position - 1]
        previous_error = float(data.loc[predecessor, "predicted_onef1b_ms"] - data.loc[predecessor, "actual_onef1b_ms"])
        base_error = float(data.loc[iteration, "predicted_onef1b_ms"] - data.loc[iteration, "actual_onef1b_ms"])
        correction = -previous_error
        updated_error = base_error + correction
        algebraic = previous_error * (2 * base_error - previous_error) > 0
        improved = abs(updated_error) < abs(base_error)
        assert algebraic == improved
        lower = -abs(base_error) - updated_error
        upper = abs(base_error) - updated_error
        contains_zero = lower < 0 < upper
        symmetric_radius = min(-lower, upper) if contains_zero else 0.0
        distance_to_entry = 0.0 if contains_zero else min(abs(lower), abs(upper))
        actual_total = float(data.loc[iteration, "actual_onef1b_ms"])
        elapsed = float(data.loc[iteration, "prefix_available_elapsed_ms"])
        remaining = actual_total - elapsed
        assert remaining > 0
        rows.append({
            "domain": domain, "iteration": iteration, "predecessor_iteration": predecessor,
            "predecessor_base_error_ms": previous_error, "current_base_error_ms": base_error,
            "residual_sign_persistent": np.sign(previous_error) == np.sign(base_error),
            "lagged_correction_ms": correction, "updated_error_ms": updated_error,
            "theoretical_improvement_condition": algebraic, "absolute_error_improved": improved,
            "base_onef1b_APE_pct": 100 * abs(base_error) / actual_total,
            "updated_onef1b_APE_pct": 100 * abs(updated_error) / actual_total,
            "base_remaining_APE_pct": 100 * abs(base_error) / remaining,
            "updated_remaining_APE_pct": 100 * abs(updated_error) / remaining,
            "state_noise_strict_lower_ms": lower, "state_noise_strict_upper_ms": upper,
            "zero_noise_inside_improvement_interval": contains_zero,
            "symmetric_break_even_radius_ms": symmetric_radius,
            "noise_distance_to_improvement_interval_ms": distance_to_entry,
            "prefix_available_elapsed_ms": elapsed, "actual_remaining_ms": remaining,
            "split": str(data.loc[iteration, "split"]),
        })
    return pd.DataFrame(rows)


def _summaries(transitions: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("source256", "mixed_source_fit_development_90_100", WALK),
        ("source256", "source_current_development_95_100", [95, 100]),
        ("target224", "target_development_90_100", WALK),
    ]
    rows = []
    for domain, split, iterations in groups:
        frame = transitions[transitions.domain.eq(domain) & transitions.iteration.isin(iterations)]
        assert len(frame) == len(iterations)
        improved = int(frame.absolute_error_improved.sum())
        rows.append({
            "domain": domain, "split": split, "iterations": len(frame),
            "improved_iterations": improved,
            "persistent_sign_transitions": int(frame.residual_sign_persistent.sum()),
            "base_onef1b_MAPE_pct": frame.base_onef1b_APE_pct.mean(),
            "updated_onef1b_MAPE_pct": frame.updated_onef1b_APE_pct.mean(),
            "delta_onef1b_MAPE_pp": frame.updated_onef1b_APE_pct.mean() - frame.base_onef1b_APE_pct.mean(),
            "base_remaining_MAPE_pct": frame.base_remaining_APE_pct.mean(),
            "updated_remaining_MAPE_pct": frame.updated_remaining_APE_pct.mean(),
            "minimum_positive_symmetric_break_even_radius_ms": (
                frame.loc[frame.symmetric_break_even_radius_ms.gt(0), "symmetric_break_even_radius_ms"].min()
                if frame.symmetric_break_even_radius_ms.gt(0).any() else 0.0
            ),
            "exact_one_sided_sign_test_p": exact_one_sided_sign_p(improved, len(frame)),
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


def _draw(out: Path, transitions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams["svg.hashsalt"] = "w37-t40a"
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    colors = {"source256": "#E69F00", "target224": "#0072B2"}
    for domain, frame in transitions.groupby("domain"):
        axes[0].plot(frame.iteration, frame.current_base_error_ms, marker="o", linestyle="--",
                     color=colors[domain], alpha=.55, label=f"{domain} base")
        axes[0].plot(frame.iteration, frame.updated_error_ms, marker="o", linestyle="-",
                     color=colors[domain], label=f"{domain} lag-one")
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_title("Fixed lag-one residual by domain")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Signed 1F1B error (ms)")
    axes[0].set_xticks(WALK)
    axes[0].grid(True, alpha=.25)
    axes[0].legend(fontsize=8)
    selected = metrics[metrics.split.isin(["mixed_source_fit_development_90_100", "target_development_90_100"])]
    x = np.arange(len(selected))
    axes[1].bar(x - .18, selected.base_onef1b_MAPE_pct, width=.36, color="#999999", label="base")
    axes[1].bar(x + .18, selected.updated_onef1b_MAPE_pct, width=.36,
                color=[colors[d] for d in selected.domain], label="lag-one")
    axes[1].set_xticks(x, selected.domain)
    axes[1].set_ylabel("1F1B MAPE (%)")
    axes[1].set_title("Homologous three-transition audit")
    axes[1].grid(True, axis="y", alpha=.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "cross_domain_state.svg", metadata={"Date": None})
    fig.savefig(out / "cross_domain_state.png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    from smoke_worker import dump, sha
    from worker import csv

    begin = perf_counter()
    refs = plan["sealed_diagnostic_stages"]
    assert [ref["name"] for ref in refs] == ["t35", "t38"]
    accept35 = _verify_prior(refs[0], "PASS_ONLINE_PREFIX_NOWCAST_IMPROVES_DEVELOPMENT_COLD_START_STILL_BLOCKED", 69)
    accept38 = _verify_prior(refs[1], "PASS_CAUSAL_TARGET_LAGGED_ONLINE_UPDATE_DEVELOPMENT", 76)
    source_seal_path = paths["t35_diagnose_source_online_prediction_seal_json"]
    assert sha(source_seal_path) == accept35["source_prediction_seal_sha256"]
    source_seal = json.loads(source_seal_path.read_text())
    source_files = {item["path"]: item["sha256"] for item in source_seal["files"]}
    assert sha(paths["t35_diagnose_source_online_iteration_predictions_csv"]) == source_files["source_online_iteration_predictions.csv"]
    assert sha(paths["t38_diagnose_causal_prediction_seal_json"]) == accept38["causal_prediction_seal_sha256"]

    source_results = pd.read_csv(paths["t35_diagnose_source_online_iteration_results_csv"])
    source = source_results[source_results.method.eq(T35)].sort_values("iteration").copy()
    source_cutoff = pd.read_csv(paths["t35_diagnose_source_online_prefix_cutoffs_csv"])
    source_cutoff = source_cutoff[source_cutoff.method.eq(T35)][["iteration", "prefix_available_elapsed_ms"]]
    source = source.merge(source_cutoff, on="iteration", validate="one_to_one")
    source = source.rename(columns={"error_ms": "sealed_error_ms"})
    assert np.allclose(
        source.predicted_onef1b_ms - source.actual_onef1b_ms,
        source.sealed_error_ms, rtol=0, atol=1e-9,
    )

    target_results = pd.read_csv(paths["t38_diagnose_target_iteration_results_csv"])
    target = target_results[target_results.method.eq("t35_midrun_winsor20")].sort_values("iteration").copy()
    target["split"] = "target_development"
    source_transitions = transition_audit(source, "source256")
    target_transitions = transition_audit(target, "target224")
    transitions = pd.concat([source_transitions, target_transitions], ignore_index=True)
    metrics = _summaries(transitions)
    tolerances = transitions[[
        "domain", "iteration", "state_noise_strict_lower_ms", "state_noise_strict_upper_ms",
        "zero_noise_inside_improvement_interval", "symmetric_break_even_radius_ms",
        "noise_distance_to_improvement_interval_ms",
    ]]
    csv(out, "transition_state_results.csv", transitions)
    csv(out, "domain_state_metrics.csv", metrics)
    csv(out, "state_noise_tolerance.csv", tolerances)
    _draw(out, transitions, metrics)

    target_idx = metrics.set_index("split").loc["target_development_90_100"]
    source_idx = metrics.set_index("split").loc["mixed_source_fit_development_90_100"]
    assert target_idx.improved_iterations == 3 and math.isclose(target_idx.exact_one_sided_sign_test_p, .125)
    assert source_idx.improved_iterations == 1 and source_idx.updated_onef1b_MAPE_pct > source_idx.base_onef1b_MAPE_pct
    sealed_target = pd.read_csv(paths["t38_diagnose_causal_prediction_ledger_csv"]).set_index("iteration")
    target_recomputed = target_transitions.set_index("iteration").updated_error_ms + target.set_index("iteration").actual_onef1b_ms
    target_diff = float(np.max(np.abs(target_recomputed.loc[WALK] - sealed_target.loc[WALK].predicted_onef1b_ms)))
    assert target_diff <= 1e-6
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= plan["resource"]["maximum_peak_RSS_bytes"] and elapsed <= plan["resource"]["target_analysis_seconds"]
    dump(out / "method_contract.json", {
        "fixed_rule": plan["fixed_rule"], "coefficient_updates": 0,
        "transition_contract": plan["transition_contract"], "statistical_contract": plan["statistical_contract"],
        "physical_attribution": "Residual persistence audit only; no compute, communication or wait allocation.",
    })
    dump(out / "diagnostic.json", {
        "status": "CROSS_DOMAIN_FIXED_STATE_AND_TOLERANCE_AUDIT_PASS",
        "new_prediction": False, "diagnostic_fixed_rule_projection": True,
        "target_data_scope": "already exposed development", "source_data_scope": "mixed fit and development; not blind",
        "target_base_onef1b_MAPE_pct": float(target_idx.base_onef1b_MAPE_pct),
        "target_updated_onef1b_MAPE_pct": float(target_idx.updated_onef1b_MAPE_pct),
        "target_improved_transitions": int(target_idx.improved_iterations),
        "target_exact_one_sided_sign_test_p": float(target_idx.exact_one_sided_sign_test_p),
        "target_minimum_break_even_radius_ms": float(target_idx.minimum_positive_symmetric_break_even_radius_ms),
        "source_base_onef1b_MAPE_pct": float(source_idx.base_onef1b_MAPE_pct),
        "source_updated_onef1b_MAPE_pct": float(source_idx.updated_onef1b_MAPE_pct),
        "source_improved_transitions": int(source_idx.improved_iterations),
        "source_exact_one_sided_sign_test_p": float(source_idx.exact_one_sided_sign_test_p),
        "source_persistent_sign_transitions": int(source_idx.persistent_sign_transitions),
        "target_persistent_sign_transitions": int(target_idx.persistent_sign_transitions),
        "target_recompute_max_abs_difference_ms": target_diff,
        "formal_topology_replaced": False, "added_edges_or_waits": 0,
        "source_parameter_updates": 0, "target_parameter_updates": 0,
        "prior_manifest_hashes": {ref["name"]: ref["manifest_sha256"] for ref in refs},
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed,
        "decision": "Fixed lag-one succeeds on all three target transitions but only one of three source transitions and worsens source aggregate error. Its evidence is target-local residual persistence, not a cross-domain runtime law; n=3 gives a best possible one-sided sign p-value of 0.125.",
        "next": "Freeze target-method search pending new sequential target iterations; consolidate the operational contract, verified boundaries and deployment decision table.",
    })

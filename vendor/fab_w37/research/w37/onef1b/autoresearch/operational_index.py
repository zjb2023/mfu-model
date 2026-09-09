"""T41A: consolidate sealed online modes, evidence boundaries and reproduction entrypoints."""
from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter

import pandas as pd


EXPECTED = {
    "t35": ("PASS_ONLINE_PREFIX_NOWCAST_IMPROVES_DEVELOPMENT_COLD_START_STILL_BLOCKED", 69),
    "t36": ("PASS_EARLIER_ONLINE_NOWCAST_SOURCE_GATE_TARGET_UNCERTAINTY_FAILS", 71),
    "t37": ("PASS_SEALED_ONLINE_REMAINING_TIME_PARETO_AUDIT", 73),
    "t38": ("PASS_CAUSAL_TARGET_LAGGED_ONLINE_UPDATE_DEVELOPMENT", 76),
    "t39": ("PASS_FIXED_TARGET_STATE_ROBUSTNESS_AUDIT", 79),
    "t40": ("PASS_CROSS_DOMAIN_FIXED_STATE_TOLERANCE_AUDIT", 81),
}


def operational_mode(t35_prefix_available: bool, prior_target_complete: bool, formal_only: bool = False) -> str:
    """Return the mode allowed by the evidence contract, without promoting research modes."""
    if formal_only or not t35_prefix_available:
        return "formal_v685_cold_start"
    if prior_target_complete:
        return "t38_target_initialized_lag_one_development"
    return "t35_same_iteration_prefix_development"


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


def _mode_table(paths: dict[str, Path], accepted: dict[str, dict]) -> pd.DataFrame:
    t38_metrics = pd.read_csv(paths["t38_diagnose_target_metrics_csv"]).set_index(["split", "method"])
    t39_metrics = pd.read_csv(paths["t39_diagnose_state_robustness_metrics_csv"]).set_index(["split", "method"])
    t37_metrics = pd.read_csv(paths["t37_diagnose_remaining_time_metrics_csv"]).set_index("method")
    formal = t38_metrics.loc[("all_including_initialization", "formal_v685_cold_start")]
    t35 = t38_metrics.loc[("all_including_initialization", "t35_midrun_winsor20")]
    t38 = t38_metrics.loc[("primary_causal_walk_forward", "t38_lagged_base_residual")]
    expanding = t39_metrics.loc[("primary_90_100", "expanding_past_mean_residual")]
    a36 = accepted["t36"]
    d36 = a36["target_development"]
    rem36 = t37_metrics.loc["t36_early_source_reliability"]
    raw36 = t37_metrics.loc["t36_early_raw_unshrunk"]
    rows = [
        {
            "mode": "formal_v685_cold_start", "availability": "before_iteration_without_runtime_observation",
            "required_runtime_inputs": "none", "required_prior_target_iterations": 0,
            "evaluation_scope": "target85/90/95/100 exposed development", "iterations": 4,
            "mean_cutoff_fraction_pct": 0.0, "mean_actual_remaining_ms": formal.mean_actual_remaining_ms,
            "onef1b_MAPE_pct": formal.onef1b_MAPE_pct, "remaining_MAPE_pct": formal.remaining_MAPE_pct,
            "profiler_MAPE_pct": formal.profiler_MAPE_pct, "training_MAPE_pct": formal.training_MAPE_pct,
            "MFU_relative_MAPE_pct": formal.MFU_relative_MAPE_pct,
            "eligibility": "only_formal_cold_start", "allowed_claim": "frozen formal development baseline",
            "failure_or_limit": "missing dynamic router-token matrix and independent runtime-readiness profile",
            "graph_or_state_change": "none; frozen v685",
        },
        {
            "mode": "t35_same_iteration_prefix", "availability": "after_all_stage_F0_and_latter_half_stage_B0",
            "required_runtime_inputs": "current iteration 224 F and 112 B phase prefix samples",
            "required_prior_target_iterations": 0, "evaluation_scope": "target85/90/95/100 exposed online development", "iterations": 4,
            "mean_cutoff_fraction_pct": t35.mean_cutoff_fraction_pct, "mean_actual_remaining_ms": t35.mean_actual_remaining_ms,
            "onef1b_MAPE_pct": t35.onef1b_MAPE_pct, "remaining_MAPE_pct": t35.remaining_MAPE_pct,
            "profiler_MAPE_pct": t35.profiler_MAPE_pct, "training_MAPE_pct": t35.training_MAPE_pct,
            "MFU_relative_MAPE_pct": t35.MFU_relative_MAPE_pct,
            "eligibility": "source_gate_passed_online_development", "allowed_claim": "same-iteration development nowcast",
            "failure_or_limit": "available after 57.8%; all target F/B factors outside source85/90 range",
            "graph_or_state_change": "sealed F/B phase factors only; topology unchanged",
        },
        {
            "mode": "t38_target_initialized_lag_one", "availability": "T35 prefix plus immediately preceding completed target iteration",
            "required_runtime_inputs": "T35 prefix and predecessor actual-minus-T35-base residual",
            "required_prior_target_iterations": 1, "evaluation_scope": "target90/95/100 causal exposed development", "iterations": 3,
            "mean_cutoff_fraction_pct": t38.mean_cutoff_fraction_pct, "mean_actual_remaining_ms": t38.mean_actual_remaining_ms,
            "onef1b_MAPE_pct": t38.onef1b_MAPE_pct, "remaining_MAPE_pct": t38.remaining_MAPE_pct,
            "profiler_MAPE_pct": t38.profiler_MAPE_pct, "training_MAPE_pct": t38.training_MAPE_pct,
            "MFU_relative_MAPE_pct": t38.MFU_relative_MAPE_pct,
            "eligibility": "fixed_target_local_research_development", "allowed_claim": "target-initialized sequential development analysis",
            "failure_or_limit": "n=3; sign p=0.125; source transfer worsens; finite-history interval 1/2",
            "graph_or_state_change": "T35 graph exact plus one separately labelled lag residual; topology unchanged",
        },
        {
            "mode": "t36_early_source_reliability", "availability": "after_all_stage_F0_and_last_four_stage_B0",
            "required_runtime_inputs": "current iteration 224 F and 64 B phase prefix samples",
            "required_prior_target_iterations": 0, "evaluation_scope": "target85/90/95/100 exposed online development", "iterations": 4,
            "mean_cutoff_fraction_pct": rem36.mean_cutoff_fraction_pct, "mean_actual_remaining_ms": rem36.mean_actual_remaining_ms,
            "onef1b_MAPE_pct": d36["primary_online_onef1b_MAPE_pct"], "remaining_MAPE_pct": rem36.remaining_MAPE_pct,
            "profiler_MAPE_pct": d36["primary_profiler_MAPE_pct"], "training_MAPE_pct": d36["primary_training_MAPE_pct"],
            "MFU_relative_MAPE_pct": d36["primary_MFU_relative_MAPE_pct"],
            "eligibility": "rejected_dominated_online_candidate", "allowed_claim": "diagnostic ablation only",
            "failure_or_limit": "dominated by v685 on cutoff/remaining error; source-dev and target interval coverage 0",
            "graph_or_state_change": "source reliability scales F/B phase wall; topology unchanged",
        },
        {
            "mode": "t36_early_raw_unshrunk", "availability": "same T36 early prefix",
            "required_runtime_inputs": "current iteration raw early F/B prefix",
            "required_prior_target_iterations": 0, "evaluation_scope": "target85/90/95/100 exposed post-gate diagnostic", "iterations": 4,
            "mean_cutoff_fraction_pct": raw36.mean_cutoff_fraction_pct, "mean_actual_remaining_ms": raw36.mean_actual_remaining_ms,
            "onef1b_MAPE_pct": raw36.total_onef1b_MAPE_pct, "remaining_MAPE_pct": raw36.remaining_MAPE_pct,
            "profiler_MAPE_pct": None, "training_MAPE_pct": None, "MFU_relative_MAPE_pct": None,
            "eligibility": "rejected_source_gate", "allowed_claim": "target diagnostic only",
            "failure_or_limit": "source95/100 MAPE 0.345547% worse than 0.332811% control",
            "graph_or_state_change": "raw F/B phase factors; topology unchanged",
        },
        {
            "mode": "t39_expanding_history_mean", "availability": "T35 prefix plus all preceding target residuals",
            "required_runtime_inputs": "T35 prefix and every completed target base residual",
            "required_prior_target_iterations": 1, "evaluation_scope": "target90/95/100 posthoc robustness", "iterations": 3,
            "mean_cutoff_fraction_pct": float(t38.mean_cutoff_fraction_pct), "mean_actual_remaining_ms": float(t38.mean_actual_remaining_ms),
            "onef1b_MAPE_pct": expanding.onef1b_MAPE_pct, "remaining_MAPE_pct": expanding.remaining_MAPE_pct,
            "profiler_MAPE_pct": expanding.profiler_MAPE_pct, "training_MAPE_pct": expanding.training_MAPE_pct,
            "MFU_relative_MAPE_pct": expanding.MFU_relative_MAPE_pct,
            "eligibility": "posthoc_unselected", "allowed_claim": "robustness sensitivity only",
            "failure_or_limit": "ranked after target exposure; only 0.034286pp below fixed T38 on three transitions",
            "graph_or_state_change": "posthoc temporal smoother; no graph change",
        },
    ]
    return pd.DataFrame(rows)


def _explanation_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"layer": "locked_schedule_graph", "evidence": "training schedule semantics plus locked node/edge tables", "explains": "warmup/steady/cooldown, F/B, communication, dependencies and graph waits", "does_not_explain": "unobserved target runtime state", "accounting": "max-plus edges; no cost inserted by T35-T41"},
        {"layer": "phase_transfer_base", "evidence": "source85/90 aggregate phase wall and runtime/PP costs", "explains": "static cold-start 1F1B envelope", "does_not_explain": "224-card F/B slowdown and readiness state", "accounting": "each compute/communication/runtime component counted once"},
        {"layer": "T35_directional_prefix", "evidence": "same-iteration all-stage F0 and latter-half B0", "explains": "within-iteration F/B scale correction; B contribution dominates", "does_not_explain": "why a remaining target-local residual persists between iterations", "accounting": "aggregate F/B phase wall multiplied once; no new edge/wait"},
        {"layer": "T38_temporal_state", "evidence": "immediately preceding completed target base residual", "explains": "target-local one-step residual persistence", "does_not_explain": "physical compute, communication or wait ownership; source transfer", "accounting": "one envelope correction, separately labelled; entry/tail/outer unchanged"},
        {"layer": "step_and_MFU", "evidence": "frozen entry/tail/outer plus predicted 1F1B and inherited FLOPs numerator", "explains": "Profiler/training clock propagation and derived MFU error", "does_not_explain": "independent FLOPs numerator accuracy", "accounting": "Step changes only through 1F1B; MFU numerator unchanged"},
    ])


def _reproduction_index(root: Path, accepted: dict[str, dict], refs: list[dict]) -> pd.DataFrame:
    specs = {
        "t35": ("T35A_PLAN.json", "wave35a_online_prefix.json", "T35A"),
        "t36": ("T36A_PLAN.json", "wave36a_early_prefix.json", "T36A"),
        "t37": ("T37A_PLAN.json", "wave37a_remaining_time.json", "T37A"),
        "t38": ("T38A_PLAN.json", "wave38a_causal_lagged.json", "T38A"),
        "t39": ("T39A_PLAN.json", "wave39a_state_robustness.json", "T39A"),
        "t40": ("T40A_PLAN.json", "wave40a_cross_domain_state.json", "T40A"),
    }
    from smoke_worker import sha
    by_name = {ref["name"]: ref for ref in refs}
    rows = []
    directory = root / "docs/w37/1f1b/deadline_20260907"
    for name, (plan_name, wave_name, link_name) in specs.items():
        plan_path, wave_path = directory / plan_name, directory / wave_name
        run = Path(by_name[name]["stage_root"]).parent
        command = run / "pipeline_command.json"
        rows.append({
            "wave": name.upper(), "status": accepted[name]["status"],
            "plan_path": str(plan_path.relative_to(root)), "plan_sha256": sha(plan_path),
            "wave_spec_path": str(wave_path.relative_to(root)), "wave_spec_sha256": sha(wave_path),
            "official_run_link": str((directory / "runs" / link_name).relative_to(root)),
            "run_id": accepted[name]["run_id"], "diagnose_manifest_sha256": by_name[name]["manifest_sha256"],
            "pipeline_command_path": str(command.relative_to(root)), "pipeline_command_sha256": sha(command),
            "reproduce_argv": f"python research/w37/onef1b/autoresearch/pipeline.py --spec {wave_path.relative_to(root)} --run-id NEW_UNIQUE_RUN_ID",
        })
    return pd.DataFrame(rows)


def _draw(out: Path) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams["svg.hashsalt"] = "w37-t41a"
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [
        (0.3, 3.35, 2.5, 1.1, "No current prefix\nv685 formal cold start", "#DDDDDD"),
        (3.25, 3.35, 2.5, 1.1, "T36 early prefix\ndo not switch", "#F4C7C3"),
        (6.2, 3.35, 2.5, 1.1, "T35 midrun prefix\n1.77% / rem 4.20%", "#CFE8F3"),
        (9.15, 3.35, 2.5, 1.1, "Prior target complete\nT38 dev 0.54%", "#CDECCF"),
        (6.2, 1.15, 2.5, 1.1, "No prior target\nreport T35 dev only", "#E8F2F8"),
        (9.15, 1.15, 2.5, 1.1, "Need new later runs\nbefore validation", "#FFF0C2"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#333333"))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10)
    for x1, y1, x2, y2 in [(2.8,3.9,3.25,3.9),(5.75,3.9,6.2,3.9),(8.7,3.9,9.15,3.9),(7.45,3.35,7.45,2.25),(10.4,3.35,10.4,2.25)]:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops={"arrowstyle":"->","color":"#444444","lw":1.5})
    ax.text(6.0, 4.75, "W37 1F1B operational evidence boundary", ha="center", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "operational_decision_flow.svg", metadata={"Date": None})
    fig.savefig(out / "operational_decision_flow.png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    from smoke_worker import dump
    from worker import csv

    begin = perf_counter()
    refs = plan["sealed_diagnostic_stages"]
    assert [ref["name"] for ref in refs] == ["t35", "t36", "t37", "t38", "t39", "t40"]
    accepted = {name: _verify_prior(ref, *EXPECTED[name]) for name, ref in zip(EXPECTED, refs)}
    modes = _mode_table(paths, accepted)
    explanations = _explanation_table()
    root = Path(__file__).resolve().parents[4]
    reproduction = _reproduction_index(root, accepted, refs)
    csv(out, "operational_mode_decision.csv", modes)
    csv(out, "explanation_layers.csv", explanations)
    csv(out, "reproduction_index.csv", reproduction)
    dump(out / "stop_resume_contract.json", {
        "status": "TARGET_METHOD_SEARCH_FROZEN_PENDING_NEW_EVIDENCE",
        "resume_conditions": plan["resume_conditions"],
        "minimum_additional_consecutive_target_iterations": 2,
        "existing_transition_sign_test_p": 0.125,
        "best_possible_five_all_improved_sign_test_p": 0.03125,
        "new_iterations_must_be_unseen_at_method_design": True,
        "formal_v685_remains": True,
    })
    _draw(out)
    assert operational_mode(False, False) == "formal_v685_cold_start"
    assert operational_mode(True, False) == "t35_same_iteration_prefix_development"
    assert operational_mode(True, True) == "t38_target_initialized_lag_one_development"
    assert operational_mode(True, True, formal_only=True) == "formal_v685_cold_start"
    assert modes.loc[modes["mode"].eq("t38_target_initialized_lag_one"), "onef1b_MAPE_pct"].iloc[0] < 1
    assert modes.loc[modes["mode"].eq("t36_early_source_reliability"), "eligibility"].iloc[0].startswith("rejected")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= plan["resource"]["maximum_peak_RSS_bytes"] and elapsed <= plan["resource"]["target_analysis_seconds"]
    dump(out / "diagnostic.json", {
        "status": "OPERATIONAL_DECISION_AND_REPRODUCTION_INDEX_PASS",
        "new_prediction": False, "target_method_search_frozen": True,
        "modes": len(modes), "evidence_layers": len(explanations), "reproduction_entries": len(reproduction),
        "formal_cold_start_mode": "formal_v685_cold_start",
        "best_supported_same_iteration_mode": "t35_same_iteration_prefix",
        "best_target_initialized_development_mode": "t38_target_initialized_lag_one",
        "rejected_or_unselected_modes": modes.loc[
            ~modes.eligibility.isin([
                "only_formal_cold_start", "source_gate_passed_online_development",
                "fixed_target_local_research_development",
            ]), "mode"
        ].tolist(),
        "required_new_target_iterations": 2,
        "formal_topology_replaced": False, "added_edges_or_waits": 0,
        "source_parameter_updates": 0, "target_parameter_updates": 0,
        "prior_manifest_hashes": {ref["name"]: ref["manifest_sha256"] for ref in refs},
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed,
        "decision": "Keep v685 as the only formal cold-start mode, T35 as a same-iteration development nowcast, and T38 as target-initialized sequential development analysis. Freeze target state method search until at least two new prospective consecutive target iterations arrive.",
        "next": "Audit documentation links, acceptance hashes, Snakemake reproduction entries and repository cleanliness; correct provenance defects only.",
    })

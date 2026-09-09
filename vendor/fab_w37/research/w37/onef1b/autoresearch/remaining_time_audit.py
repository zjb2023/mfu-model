"""T37A: read-only lead-time and remaining-error audit of sealed T35/T36 nowcasts."""
from __future__ import annotations

import json
import math
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ITERATIONS = [85, 90, 95, 100]
METHODS = {
    "formal_v685_cold_start": {
        "run": "t35", "variant": "v685_frozen", "cutoff": "zero",
        "selection_status": "formal_benchmark", "eligible": True,
    },
    "t35_midrun_winsor20": {
        "run": "t35", "variant": "online_midrun_last_half_winsor20", "cutoff": "t35",
        "selection_status": "source_gate_passed_online_primary", "eligible": True,
    },
    "t36_early_source_reliability": {
        "run": "t36", "variant": "early_source_reliability", "cutoff": "t36",
        "selection_status": "source_gate_passed_online_primary", "eligible": True,
    },
    "t36_early_raw_unshrunk": {
        "run": "t36", "variant": "early_raw_unshrunk", "cutoff": "t36",
        "selection_status": "rejected_source_gate_target_diagnostic_only", "eligible": False,
    },
}


def remaining_forecasts(rows: pd.DataFrame) -> pd.DataFrame:
    """Convert sealed total nowcasts to remaining-time nowcasts at a known cutoff."""
    result = rows.copy()
    result["actual_remaining_ms"] = result.actual_onef1b_ms - result.prefix_available_elapsed_ms
    result["predicted_remaining_ms"] = result.predicted_onef1b_ms - result.prefix_available_elapsed_ms
    result["remaining_error_ms"] = result.predicted_remaining_ms - result.actual_remaining_ms
    result["remaining_absolute_error_ms"] = result.remaining_error_ms.abs()
    result["remaining_APE_pct"] = 100 * result.remaining_absolute_error_ms / result.actual_remaining_ms
    result["cutoff_fraction_pct"] = 100 * result.prefix_available_elapsed_ms / result.actual_onef1b_ms
    result["remaining_fraction_pct"] = 100 - result.cutoff_fraction_pct
    result["lead_time_ms"] = result.actual_remaining_ms
    assert (result.actual_remaining_ms > 0).all()
    assert np.allclose(result.remaining_error_ms, result.onef1b_error_ms, rtol=0, atol=1e-9)
    return result


def mark_pareto(metrics: pd.DataFrame, eligibility: str | None = "eligible") -> pd.Series:
    """Mark rows not weakly dominated on cutoff and remaining MAPE."""
    mask = pd.Series(True, index=metrics.index)
    candidates = metrics if eligibility is None else metrics[metrics[eligibility]]
    for index, row in metrics.iterrows():
        if eligibility is not None and not bool(row[eligibility]):
            mask.loc[index] = False
            continue
        dominated = (
            (candidates.mean_cutoff_fraction_pct <= row.mean_cutoff_fraction_pct)
            & (candidates.remaining_MAPE_pct <= row.remaining_MAPE_pct)
            & (
                (candidates.mean_cutoff_fraction_pct < row.mean_cutoff_fraction_pct)
                | (candidates.remaining_MAPE_pct < row.remaining_MAPE_pct)
            )
        ).any()
        mask.loc[index] = not bool(dominated)
    return mask


def _verify_prior(ref: dict, expected_status: str, expected_tests: int) -> dict:
    from guards import checked_stage
    from smoke_worker import sha

    stage = Path(ref["stage_root"])
    acceptance_path = Path(ref["acceptance_path"])
    assert sha(stage / "run_manifest.json") == ref["manifest_sha256"]
    checked_stage(stage)
    assert sha(acceptance_path) == ref["acceptance_sha256"]
    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance["status"] == expected_status
    assert acceptance["tests_passed"] == expected_tests
    assert acceptance["diagnose_manifest_sha256"] == ref["manifest_sha256"]
    assert acceptance["target_parameter_updates"] == 0
    assert not acceptance["formal_topology_replaced"]
    return acceptance


def _assemble(paths: dict[str, Path]) -> pd.DataFrame:
    t35 = pd.read_csv(paths["t35_diagnose_target_step_iteration_results_csv"])
    t36 = pd.read_csv(paths["t36_diagnose_target_step_iteration_results_csv"])
    actual35 = t35[t35.variant.eq("v685_frozen")].sort_values("iteration")
    actual36 = t36[t36.variant.eq("v685_frozen")].sort_values("iteration")
    assert actual35.iteration.tolist() == actual36.iteration.tolist() == ITERATIONS
    for column in ["actual_onef1b_ms", "predicted_onef1b_ms", "onef1b_error_ms", "onef1b_APE_pct"]:
        assert np.allclose(actual35[column], actual36[column], rtol=0, atol=1e-12)

    cutoff35 = pd.read_csv(paths["t35_diagnose_target_online_prefix_cutoffs_csv"])
    cutoff35 = cutoff35[cutoff35.method.eq("online_midrun_last_half_winsor20")].set_index("iteration")
    cutoff36 = pd.read_csv(paths["t36_diagnose_target_early_prefix_cutoffs_csv"]).set_index("iteration")
    rows = []
    frames = {"t35": t35, "t36": t36}
    for method, spec in METHODS.items():
        frame = frames[spec["run"]]
        selected = frame[frame.variant.eq(spec["variant"])].sort_values("iteration").copy()
        assert selected.iteration.tolist() == ITERATIONS
        selected.insert(0, "method", method)
        selected["prior_variant"] = spec["variant"]
        selected["selection_status"] = spec["selection_status"]
        selected["eligible"] = spec["eligible"]
        if spec["cutoff"] == "zero":
            selected["prefix_available_elapsed_ms"] = 0.0
        elif spec["cutoff"] == "t35":
            selected["prefix_available_elapsed_ms"] = selected.iteration.map(cutoff35.prefix_available_elapsed_ms)
        else:
            selected["prefix_available_elapsed_ms"] = selected.iteration.map(cutoff36.prefix_available_elapsed_ms)
        rows.append(selected[[
            "method", "prior_variant", "selection_status", "eligible", "iteration",
            "actual_onef1b_ms", "predicted_onef1b_ms", "onef1b_error_ms", "onef1b_APE_pct",
            "prefix_available_elapsed_ms",
        ]])
    return remaining_forecasts(pd.concat(rows, ignore_index=True))


def _summarize(forecasts: pd.DataFrame) -> pd.DataFrame:
    metrics = forecasts.groupby(
        ["method", "prior_variant", "selection_status", "eligible"], as_index=False
    ).agg(
        iterations=("iteration", "size"),
        mean_cutoff_ms=("prefix_available_elapsed_ms", "mean"),
        mean_cutoff_fraction_pct=("cutoff_fraction_pct", "mean"),
        mean_actual_remaining_ms=("actual_remaining_ms", "mean"),
        mean_remaining_fraction_pct=("remaining_fraction_pct", "mean"),
        remaining_MAPE_pct=("remaining_APE_pct", "mean"),
        remaining_MAE_ms=("remaining_absolute_error_ms", "mean"),
        remaining_bias_ms=("remaining_error_ms", "mean"),
        total_onef1b_MAPE_pct=("onef1b_APE_pct", "mean"),
    )
    metrics["eligible_pareto_frontier"] = mark_pareto(metrics, "eligible")
    metrics["would_be_pareto_if_eligible"] = mark_pareto(metrics, None)
    formal = float(metrics.loc[metrics.method.eq("formal_v685_cold_start"), "remaining_MAPE_pct"].iloc[0])
    metrics["remaining_MAPE_delta_vs_formal_pp"] = metrics.remaining_MAPE_pct - formal
    return metrics.sort_values(["mean_cutoff_fraction_pct", "remaining_MAPE_pct"]).reset_index(drop=True)


def _interval_rows(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for domain, interval_key, cutoff_key in [
        ("source256", "t36_diagnose_source_early_interval_results_csv", "t36_diagnose_source_early_prefix_cutoffs_csv"),
        ("target224", "t36_diagnose_target_early_interval_results_csv", "t36_diagnose_target_early_prefix_cutoffs_csv"),
    ]:
        interval = pd.read_csv(paths[interval_key])
        cutoff = pd.read_csv(paths[cutoff_key])[["iteration", "prefix_available_elapsed_ms"]]
        frame = interval.merge(cutoff, on="iteration", validate="one_to_one")
        frame.insert(0, "domain", domain)
        frame["split"] = np.where(
            frame.iteration.isin([85, 90]),
            "source_fit" if domain == "source256" else "target_development",
            "source_development_confirmation" if domain == "source256" else "target_development",
        )
        frame["actual_remaining_ms"] = frame.actual_onef1b_ms - frame.prefix_available_elapsed_ms
        frame["predicted_remaining_low_ms"] = frame.predicted_interval_low_ms - frame.prefix_available_elapsed_ms
        frame["predicted_remaining_high_ms"] = frame.predicted_interval_high_ms - frame.prefix_available_elapsed_ms
        frame["remaining_interval_width_ms"] = frame.predicted_remaining_high_ms - frame.predicted_remaining_low_ms
        frame["remaining_interval_width_pct"] = 100 * frame.remaining_interval_width_ms / frame.actual_remaining_ms
        recomputed = frame.actual_remaining_ms.between(
            frame.predicted_remaining_low_ms, frame.predicted_remaining_high_ms
        )
        assert (recomputed == frame.covered).all()
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    summary = rows.groupby(["domain", "split"], as_index=False).agg(
        iterations=("iteration", "size"), covered_iterations=("covered", "sum"),
        coverage_pct=("covered", lambda x: 100 * x.mean()),
        mean_remaining_interval_width_ms=("remaining_interval_width_ms", "mean"),
        mean_remaining_interval_width_pct=("remaining_interval_width_pct", "mean"),
        mean_distance_to_interval_ms=("distance_to_interval_ms", "mean"),
    )
    return rows, summary


def _draw(out: Path, metrics: pd.DataFrame) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams["svg.hashsalt"] = "w37-t37a"
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    colors = {
        "formal_v685_cold_start": "#555555", "t35_midrun_winsor20": "#0072B2",
        "t36_early_source_reliability": "#E69F00", "t36_early_raw_unshrunk": "#CC79A7",
    }
    labels = {
        "formal_v685_cold_start": "v685 cold start", "t35_midrun_winsor20": "T35 midrun",
        "t36_early_source_reliability": "T36 early", "t36_early_raw_unshrunk": "T36 raw (rejected)",
    }
    for row in metrics.itertuples(index=False):
        marker = "o" if row.eligible else "X"
        ax.scatter(row.mean_cutoff_fraction_pct, row.remaining_MAPE_pct, s=95, marker=marker,
                   color=colors[row.method], edgecolor="black", linewidth=.6, zorder=3)
        ax.annotate(labels[row.method], (row.mean_cutoff_fraction_pct, row.remaining_MAPE_pct),
                    xytext=(7, 6), textcoords="offset points", fontsize=9)
    frontier = metrics[metrics.eligible_pareto_frontier].sort_values("mean_cutoff_fraction_pct")
    ax.plot(frontier.mean_cutoff_fraction_pct, frontier.remaining_MAPE_pct, "--", color="#009E73",
            linewidth=1.4, label="eligible Pareto frontier")
    ax.set_xlabel("Mean 1F1B completed when prediction becomes available (%)")
    ax.set_ylabel("Remaining-time MAPE (%)")
    ax.set_title("T37A sealed online forecast: lead time versus remaining-time error")
    ax.grid(True, alpha=.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "remaining_time_pareto.svg", metadata={"Date": None})
    fig.savefig(out / "remaining_time_pareto.png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    from smoke_worker import dump, sha
    from worker import csv

    begin = perf_counter()
    reviews = plan["sealed_diagnostic_stages"]
    assert [r["name"] for r in reviews] == ["t35", "t36"]
    accept35 = _verify_prior(
        reviews[0], "PASS_ONLINE_PREFIX_NOWCAST_IMPROVES_DEVELOPMENT_COLD_START_STILL_BLOCKED", 69
    )
    accept36 = _verify_prior(
        reviews[1], "PASS_EARLIER_ONLINE_NOWCAST_SOURCE_GATE_TARGET_UNCERTAINTY_FAILS", 71
    )
    assert sha(paths["t35_diagnose_target_online_prediction_seal_json"]) == accept35["target_prediction_seal_sha256"]
    assert sha(paths["t36_diagnose_target_early_prediction_seal_json"]) == accept36["target_prediction_seal_sha256"]

    source35 = pd.read_csv(paths["t35_diagnose_source_online_metrics_csv"]).set_index("method")
    source36 = pd.read_csv(paths["t36_diagnose_source_early_metrics_csv"]).set_index("method")
    assert source35.loc["online_midrun_last_half_winsor20", "onef1b_MAPE_pct"] < source35.loc["cold_start_phase_transfer", "onef1b_MAPE_pct"]
    assert source36.loc["early_source_reliability", "onef1b_MAPE_pct"] < source36.loc["early_cold_control", "onef1b_MAPE_pct"]
    assert source36.loc["early_raw_unshrunk", "onef1b_MAPE_pct"] > source36.loc["early_cold_control", "onef1b_MAPE_pct"]

    forecasts = _assemble(paths)
    metrics = _summarize(forecasts)
    intervals, coverage = _interval_rows(paths)
    csv(out, "remaining_time_iteration_results.csv", forecasts)
    csv(out, "remaining_time_metrics.csv", metrics)
    csv(out, "remaining_time_interval_results.csv", intervals)
    csv(out, "remaining_time_interval_coverage.csv", coverage)
    _draw(out, metrics)

    idx = metrics.set_index("method")
    assert bool(idx.loc["formal_v685_cold_start", "eligible_pareto_frontier"])
    assert bool(idx.loc["t35_midrun_winsor20", "eligible_pareto_frontier"])
    assert not bool(idx.loc["t36_early_source_reliability", "eligible_pareto_frontier"])
    assert not bool(idx.loc["t36_early_raw_unshrunk", "eligible_pareto_frontier"])
    assert bool(idx.loc["t36_early_raw_unshrunk", "would_be_pareto_if_eligible"])
    target_coverage = coverage[coverage.domain.eq("target224")].iloc[0]
    source_dev_coverage = coverage[
        coverage.domain.eq("source256") & coverage.split.eq("source_development_confirmation")
    ].iloc[0]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= plan["resource"]["maximum_peak_RSS_bytes"]
    assert elapsed <= plan["resource"]["target_analysis_seconds"]
    dump(out / "metric_contract.json", plan["metric_contract"])
    dump(out / "diagnostic.json", {
        "status": "SEALED_ONLINE_REMAINING_TIME_PARETO_AUDIT_PASS",
        "new_prediction": False, "online_prediction_audit": True,
        "target_data_scope": "already exposed development evaluation",
        "formal_v685_remaining_MAPE_pct": float(idx.loc["formal_v685_cold_start", "remaining_MAPE_pct"]),
        "t35_remaining_MAPE_pct": float(idx.loc["t35_midrun_winsor20", "remaining_MAPE_pct"]),
        "t35_mean_cutoff_fraction_pct": float(idx.loc["t35_midrun_winsor20", "mean_cutoff_fraction_pct"]),
        "t36_primary_remaining_MAPE_pct": float(idx.loc["t36_early_source_reliability", "remaining_MAPE_pct"]),
        "t36_primary_mean_cutoff_fraction_pct": float(idx.loc["t36_early_source_reliability", "mean_cutoff_fraction_pct"]),
        "t36_raw_rejected_remaining_MAPE_pct": float(idx.loc["t36_early_raw_unshrunk", "remaining_MAPE_pct"]),
        "t36_raw_selection_status": METHODS["t36_early_raw_unshrunk"]["selection_status"],
        "eligible_pareto_methods": metrics.loc[metrics.eligible_pareto_frontier, "method"].tolist(),
        "t36_source_development_interval_coverage": [int(source_dev_coverage.covered_iterations), int(source_dev_coverage.iterations)],
        "t36_target_interval_coverage": [int(target_coverage.covered_iterations), int(target_coverage.iterations)],
        "target_parameter_updates": 0, "source_parameter_updates": 0,
        "formal_topology_replaced": False, "added_edges_or_waits": 0,
        "prior_manifest_hashes": {r["name"]: r["manifest_sha256"] for r in reviews},
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed,
        "decision": "T35 is the eligible online accuracy point; formal v685 is the earliest eligible point. T36 source-reliability is dominated on mean cutoff and remaining-time MAPE. The T36 raw point would be Pareto-efficient but remains ineligible because its preregistered source gate failed.",
        "next": "If pursued, register a separate causal target-domain online update using only completed earlier target iterations; retain explicit development-only labeling.",
    })

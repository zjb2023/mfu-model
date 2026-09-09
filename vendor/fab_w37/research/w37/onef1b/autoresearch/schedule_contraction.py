"""T34A: explain PP16/MB4 to PP14/MB3 schedule-envelope contraction."""
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
from pp_graph import build, envelope
from pp_semantics import align_observations
from readiness import ReadinessCosts
from smoke_worker import dump, sha
from worker import csv


FIT = [85, 90]
SOURCE_VALIDATION = [95, 100]
TARGET_DEVELOPMENT = [85, 90, 95, 100]
PHASE_KEY = ["rank", "pp_stage", "pp_lane", "phase", "microbatch"]
SCENARIOS = {
    "source_PP16_MB4": (16, 4),
    "MB_only_PP16_MB3": (16, 3),
    "PP_only_PP14_MB4": (14, 4),
    "target_PP14_MB3": (14, 3),
}
MFU_NUMERATOR = 100 * 8.436548311982576e16 / (224 * 500e12) * 1000


class RoleMappedReadinessCosts(ReadinessCosts):
    """Map the last reduced-microbatch role to source MB3 for every PP depth."""

    def phase(self, action):
        stage = self.mapped_stage(action)
        microbatch = action["microbatch"]
        if self.mb < 4 and microbatch == self.mb - 1:
            microbatch = 3
        key = (stage, action["pp_lane"], action["name"], microbatch)
        self.last_key = f"phase:{key}"
        return self.source_phase[key]


def build_scenarios(
    aligned: pd.DataFrame, pairs: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    nodes_by_scenario = {}
    edges_by_scenario = {}
    summaries = []
    ledgers = []
    for scenario, (pp, microbatches) in SCENARIOS.items():
        costs = RoleMappedReadinessCosts(
            aligned, pairs, FIT, pp=pp, mb=microbatches, role_transfer=True
        )
        nodes, edges, topology = build(pp, microbatches, costs)
        nodes_by_scenario[scenario] = nodes
        edges_by_scenario[scenario] = edges
        result = envelope(nodes)
        summaries.append({
            "scenario": scenario,
            "PP": pp,
            "microbatches": microbatches,
            "predicted_onef1b_ms": result["onef1b_ms"],
            "predicted_program_ms": result["program_ms"],
            "node_count": len(nodes),
            "edge_count": len(edges),
            "topology_sha256": topology,
            "fit_iterations": "85|90",
            "target_timing_used": False,
        })
        region_by_node = nodes.set_index("node_id").region.to_dict()
        scenario_ledger = critical_ledger(nodes, scenario, scenario)
        for row in scenario_ledger:
            row["region"] = region_by_node[row["node_id"]]
        ledgers.extend(scenario_ledger)
    return (
        nodes_by_scenario,
        edges_by_scenario,
        pd.DataFrame(summaries),
        pd.DataFrame(ledgers),
    )


def contraction_shapley(summaries: pd.DataFrame) -> pd.DataFrame:
    values = summaries.set_index("scenario").predicted_onef1b_ms.to_dict()
    base = values["source_PP16_MB4"]
    pp_only = values["PP_only_PP14_MB4"]
    mb_only = values["MB_only_PP16_MB3"]
    both = values["target_PP14_MB3"]
    pp_effect = 0.5 * ((pp_only - base) + (both - mb_only))
    mb_effect = 0.5 * ((mb_only - base) + (both - pp_only))
    assert math.isclose(pp_effect + mb_effect, both - base, abs_tol=1e-10)
    return pd.DataFrame([
        {
            "factor": "PP_depth_16_to_14",
            "shapley_envelope_delta_ms": pp_effect,
            "absolute_contraction_ms": -pp_effect,
            "interpretation": "code-derived schedule and stage-role transfer",
        },
        {
            "factor": "microbatches_4_to_3",
            "shapley_envelope_delta_ms": mb_effect,
            "absolute_contraction_ms": -mb_effect,
            "interpretation": "code-derived schedule with last-role mapping to source MB3",
        },
    ])


def _target_phase_score(
    predicted_nodes: pd.DataFrame, target_phases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predicted = predicted_nodes[
        predicted_nodes.kind.eq("phase_compute_communication_runtime")
    ][[*PHASE_KEY, "region", "predicted_start_ns", "predicted_end_ns", "duration_ns"]].copy()
    origin = int(predicted.predicted_start_ns.min())
    predicted["predicted_start_relative_ns"] = predicted.predicted_start_ns - origin
    predicted["predicted_end_relative_ns"] = predicted.predicted_end_ns - origin
    truth = target_phases[target_phases.iteration.isin(TARGET_DEVELOPMENT)].copy()
    truth_origin = truth.groupby("iteration").observed_start_ns.transform("min")
    truth["observed_start_relative_ns"] = truth.observed_start_ns - truth_origin
    truth["observed_end_relative_ns"] = truth.observed_end_ns - truth_origin
    scored = truth.merge(
        predicted, on=PHASE_KEY, how="left", validate="many_to_one",
        suffixes=("_observed", "_predicted"), indicator=True,
    )
    assert scored._merge.eq("both").all()
    assert len(scored) == len(TARGET_DEVELOPMENT) * 14 * 16 * 2 * 3
    scored["duration_error_ms"] = (
        scored.duration_ns_predicted - scored.duration_ns_observed
    ) / 1e6
    scored["start_error_ms"] = (
        scored.predicted_start_relative_ns - scored.observed_start_relative_ns
    ) / 1e6
    scored["end_error_ms"] = (
        scored.predicted_end_relative_ns - scored.observed_end_relative_ns
    ) / 1e6
    metric_rows = []
    for keys in [
        ["region", "phase"], ["region"], ["pp_stage", "phase"], [],
    ]:
        iterator = [((), scored)] if not keys else scored.groupby(keys, sort=True)
        for values, group in iterator:
            values = values if isinstance(values, tuple) else (values,)
            record = dict(zip(keys, values))
            record.setdefault("region", "all")
            record.setdefault("phase", "all")
            record.setdefault("pp_stage", -1)
            metric_rows.append({
                **record,
                "observations": len(group),
                "duration_MAE_ms": group.duration_error_ms.abs().mean(),
                "duration_bias_ms": group.duration_error_ms.mean(),
                "start_MAE_ms": group.start_error_ms.abs().mean(),
                "start_bias_ms": group.start_error_ms.mean(),
                "end_MAE_ms": group.end_error_ms.abs().mean(),
                "end_bias_ms": group.end_error_ms.mean(),
            })
    metrics = pd.DataFrame(metric_rows)
    global_truth = truth.groupby("iteration").agg(
        first=("observed_start_ns", "min"), last=("observed_end_ns", "max")
    ).reset_index()
    global_truth["actual_onef1b_ms"] = (global_truth["last"] - global_truth["first"]) / 1e6
    return scored.drop(columns="_merge"), metrics, global_truth[["iteration", "actual_onef1b_ms"]]


def _target_iteration_score(
    records: list[dict], global_truth: pd.DataFrame, paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = pd.read_csv(paths["iteration_ground_truth.csv"])
    truth = truth[truth.iteration.isin(TARGET_DEVELOPMENT)].copy()
    payload = json.loads(paths["dag_v682_stage_aware_pp_gradient_payload.json"].read_text())
    evaluation = {int(row["iteration"]): row for row in payload["evaluation"]}
    actual_onef1b = global_truth.set_index("iteration").actual_onef1b_ms.to_dict()
    results = []
    ledger = []
    for observed in truth.itertuples(index=False):
        iteration = int(observed.iteration)
        bars = payload["target_trace"][str(iteration)]["bars"]
        bar_envelope = max(row["end_ms"] for row in bars) - min(row["start_ms"] for row in bars)
        assert math.isclose(bar_envelope, actual_onef1b[iteration], abs_tol=1e-9)
        actual = {
            "entry": float(evaluation[iteration]["actual_entry_ms"]),
            "onef1b": float(actual_onef1b[iteration]),
            "tail": float(observed.actual_profiler_step_ms)
                    - float(evaluation[iteration]["actual_entry_ms"])
                    - float(actual_onef1b[iteration]),
            "outer": float(observed.actual_training_step_ms)
                     - float(observed.actual_profiler_step_ms),
        }
        for record in records:
            predicted_mfu = MFU_NUMERATOR / record["training_ms"]
            actual_mfu = MFU_NUMERATOR / float(observed.actual_training_step_ms)
            row = {
                "variant": record["variant"],
                "iteration": iteration,
                "actual_onef1b_ms": actual["onef1b"],
                "predicted_onef1b_ms": record["onef1b_ms"],
                "onef1b_error_ms": record["onef1b_ms"] - actual["onef1b"],
                "onef1b_APE_pct": 100 * abs(record["onef1b_ms"] - actual["onef1b"]) / actual["onef1b"],
                "actual_profiler_ms": float(observed.actual_profiler_step_ms),
                "predicted_profiler_ms": record["profiler_ms"],
                "profiler_error_ms": record["profiler_ms"] - float(observed.actual_profiler_step_ms),
                "profiler_APE_pct": 100 * abs(record["profiler_ms"] - float(observed.actual_profiler_step_ms)) / float(observed.actual_profiler_step_ms),
                "actual_training_ms": float(observed.actual_training_step_ms),
                "predicted_training_ms": record["training_ms"],
                "training_error_ms": record["training_ms"] - float(observed.actual_training_step_ms),
                "training_APE_pct": 100 * abs(record["training_ms"] - float(observed.actual_training_step_ms)) / float(observed.actual_training_step_ms),
                "actual_MFU_pct_derived": actual_mfu,
                "predicted_MFU_pct": predicted_mfu,
                "MFU_error_pp": predicted_mfu - actual_mfu,
                "MFU_relative_APE_pct": 100 * abs(predicted_mfu - actual_mfu) / actual_mfu,
                "MFU_basis": "inherited FLOPs/peak and training clock; numerator not independently verified",
            }
            results.append(row)
            for phase in ["entry", "onef1b", "tail", "outer"]:
                ledger.append({
                    "variant": record["variant"], "iteration": iteration,
                    "phase": phase, "actual_ms": actual[phase],
                    "predicted_ms": record[f"{phase}_ms"],
                    "error_ms": record[f"{phase}_ms"] - actual[phase],
                    "APE_pct": 100 * abs(record[f"{phase}_ms"] - actual[phase]) / actual[phase],
                })
    frame = pd.DataFrame(results)
    metrics = frame.groupby("variant", sort=True).agg(
        iterations=("iteration", "size"),
        onef1b_MAPE_pct=("onef1b_APE_pct", "mean"),
        onef1b_bias_ms=("onef1b_error_ms", "mean"),
        profiler_MAPE_pct=("profiler_APE_pct", "mean"),
        profiler_bias_ms=("profiler_error_ms", "mean"),
        training_MAPE_pct=("training_APE_pct", "mean"),
        training_bias_ms=("training_error_ms", "mean"),
        MFU_relative_MAPE_pct=("MFU_relative_APE_pct", "mean"),
        MFU_bias_pp=("MFU_error_pp", "mean"),
    ).reset_index()
    return frame, pd.DataFrame(ledger), metrics


def _draw(
    out: Path, summaries: pd.DataFrame, effects: pd.DataFrame,
    target_metrics: pd.DataFrame, region_metrics: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "w37-t34a-schedule-contraction"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    axes[0].bar(summaries.scenario, summaries.predicted_onef1b_ms, color="#348ABD")
    axes[0].set_ylabel("predicted 1F1B (ms)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Fixed source85/90 costs")
    axes[1].bar(effects.factor, effects.absolute_contraction_ms, color="#7A68A6")
    axes[1].set_ylabel("Shapley contraction (ms)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_title("Static schedule factors")
    local = region_metrics[
        region_metrics.pp_stage.eq(-1) & region_metrics.phase.eq("all")
        & region_metrics.region.ne("all")
    ]
    axes[2].bar(local.region, local.duration_bias_ms, color="#E24A33")
    axes[2].axhline(0, color="black", linewidth=.7)
    axes[2].set_ylabel("target phase duration bias (ms)")
    axes[2].set_title(
        "Target development; 1F1B MAPE "
        f"{float(target_metrics.loc[target_metrics.variant.eq('phase_transfer_control'), 'onef1b_MAPE_pct'].item()):.3f}%"
    )
    fig.suptitle("T34A schedule contraction and target mismatch")
    fig.text(
        .5, .012,
        "Prediction sealed before target timing. Target panels are development diagnostics and do not fit a correction.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out / "schedule_contraction.svg", metadata={"Date": None})
    fig.savefig(out / "schedule_contraction.png", dpi=150, metadata={"Software": "w37-t34a"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    begin = perf_counter()
    assert plan["diagnostic"] == "schedule_contraction_audit"
    assert plan["diagnostic_access"] == "evaluator" and not plan["variants"]
    review = json.loads((ROOT / plan["resource_review_file"]).read_text())
    assert review["fit_iterations"] == FIT
    for item in review["training_semantics"]:
        assert sha(Path(item["path"])) == item["sha256"]

    source_phases = pd.read_csv(paths["source_pp_trace_events_60_100.csv"])
    source_apis = pd.read_csv(paths["source_pp_api_events_60_100.csv"])
    aligned, pairs = align_observations(source_phases, source_apis)
    nodes, edges, summaries, critical = build_scenarios(aligned, pairs)
    changed_aligned = aligned.copy()
    changed_pairs = pairs.copy()
    mask = changed_aligned.iteration.isin(SOURCE_VALIDATION)
    changed_aligned.loc[mask, [
        "start_ns", "end_ns", "duration_ns", "local_prelaunch_gap_ns",
        "api_postjoin_return_ns",
    ]] = -999999999
    pair_mask = changed_pairs.iteration.isin(SOURCE_VALIDATION)
    changed_pairs.loc[pair_mask, [
        "sender_post_ns", "receiver_post_ns", "both_published_ns",
        "first_api_return_ns", "last_api_return_ns", "post_publication_upper_bound_ns",
        "sender_wait_for_receiver_ns", "receiver_wait_for_sender_ns",
    ]] = -999999999
    changed_nodes, changed_edges, changed_summaries, changed_critical = build_scenarios(
        changed_aligned, changed_pairs
    )
    pd.testing.assert_frame_equal(summaries, changed_summaries, check_exact=True)
    pd.testing.assert_frame_equal(critical, changed_critical, check_exact=True)
    for scenario in SCENARIOS:
        pd.testing.assert_frame_equal(nodes[scenario], changed_nodes[scenario], check_exact=True)
        pd.testing.assert_frame_equal(edges[scenario], changed_edges[scenario], check_exact=True)

    effects = contraction_shapley(summaries)
    critical_summary = critical.groupby(
        ["variant", "case", "kind", "region"], sort=True
    ).critical_contribution_ms.sum().reset_index()
    baseline_record, _, contract = baseline(paths)
    target_onef1b = float(
        summaries.loc[summaries.scenario.eq("target_PP14_MB3"), "predicted_onef1b_ms"].item()
    )
    phase_record = {**baseline_record, "variant": "phase_transfer_control"}
    phase_record["onef1b_ms"] = target_onef1b
    phase_record["profiler_ms"] = (
        phase_record["entry_ms"] + phase_record["onef1b_ms"] + phase_record["tail_ms"]
    )
    phase_record["training_ms"] = phase_record["profiler_ms"] + phase_record["outer_ms"]
    records = [baseline_record, phase_record]

    csv(out, "schedule_scenario_summary.csv", summaries)
    csv(out, "schedule_contraction_shapley.csv", effects)
    csv(out, "schedule_critical_path_ledger.csv.gz", critical)
    csv(out, "schedule_critical_path_summary.csv", critical_summary)
    dump(out / "step_prediction_controls.json", records)
    dump(out / "formal_v685_prediction_contract.json", contract)
    prediction_files = [
        "schedule_scenario_summary.csv", "schedule_contraction_shapley.csv",
        "schedule_critical_path_ledger.csv.gz", "schedule_critical_path_summary.csv",
        "step_prediction_controls.json", "formal_v685_prediction_contract.json",
    ]
    for scenario in SCENARIOS:
        node_name = f"{scenario}_nodes.csv.gz"
        edge_name = f"{scenario}_edges.csv.gz"
        csv(out, node_name, nodes[scenario])
        csv(out, edge_name, edges[scenario])
        prediction_files.extend([node_name, edge_name])
    dump(out / "schedule_contraction_prediction_seal.json", {
        "status": "SEALED_SOURCE85_90_STATIC_SCHEDULE_CONTRACTION",
        "fit_iterations": FIT,
        "source_incremental_validation": SOURCE_VALIDATION,
        "target_development_iterations": TARGET_DEVELOPMENT,
        "target_timing_attached_before_seal": False,
        "target_parameter_updates": 0,
        "formal_topology_replaced": False,
        "mutation_gate": "Changing all source95/100 phase/API timing leaves the four scenario graphs, critical paths and summaries exact.",
        "files": [{"path": name, "sha256": sha(out / name)} for name in prediction_files],
    })
    sealed_hashes = {
        item["path"]: item["sha256"]
        for item in json.loads((out / "schedule_contraction_prediction_seal.json").read_text())["files"]
    }

    target_phases = pd.read_csv(paths["target_phase_rank_events_60_100.csv"])
    target_phase_results, target_phase_metrics, global_truth = _target_phase_score(
        nodes["target_PP14_MB3"], target_phases
    )
    iteration_results, phase_ledger, target_metrics = _target_iteration_score(
        records, global_truth, paths
    )
    csv(out, "target_development_phase_results.csv.gz", target_phase_results)
    csv(out, "target_development_phase_metrics.csv", target_phase_metrics)
    csv(out, "target_development_iteration_results.csv", iteration_results)
    csv(out, "target_development_step_phase_ledger.csv", phase_ledger)
    csv(out, "target_development_metrics.csv", target_metrics)
    source_actual = source_phases[
        source_phases.iteration.isin(SOURCE_VALIDATION)
    ].groupby("iteration").agg(
        first=("observed_start_ns", "min"), last=("observed_end_ns", "max")
    )
    source_actual_mean_ms = float(((source_actual["last"] - source_actual["first"]) / 1e6).mean())
    target_actual_mean_ms = float(global_truth.actual_onef1b_ms.mean())
    source_predicted_ms = float(
        summaries.loc[summaries.scenario.eq("source_PP16_MB4"), "predicted_onef1b_ms"].item()
    )
    predicted_contraction_ms = source_predicted_ms - target_onef1b
    observed_contraction_ms = source_actual_mean_ms - target_actual_mean_ms
    contraction_audit = {
        "source95_100_actual_mean_onef1b_ms": source_actual_mean_ms,
        "target85_100_actual_mean_onef1b_ms": target_actual_mean_ms,
        "observed_source_to_target_contraction_ms": observed_contraction_ms,
        "source85_90_fit_predicted_PP16_MB4_ms": source_predicted_ms,
        "source85_90_fit_predicted_PP14_MB3_ms": target_onef1b,
        "predicted_schedule_contraction_ms": predicted_contraction_ms,
        "predicted_minus_observed_contraction_ms": predicted_contraction_ms - observed_contraction_ms,
        "observed_contraction_fraction_of_predicted": observed_contraction_ms / predicted_contraction_ms,
        "interpretation": "Target timing is post-seal development evidence. The mismatch is not converted into a fitted scale, residual, node cost, or wait.",
    }
    dump(out / "source_target_contraction_audit.json", contraction_audit)
    for name, expected in sealed_hashes.items():
        assert sha(out / name) == expected
    phase_control = target_metrics.set_index("variant").loc["phase_transfer_control"]
    formal_control = target_metrics.set_index("variant").loc["v685_frozen"]
    assert math.isclose(
        float(phase_control.onef1b_MAPE_pct),
        review["reproduction_reference"]["phase_transfer_target_onef1b_MAPE_pct"],
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(formal_control.onef1b_MAPE_pct),
        review["reproduction_reference"]["formal_v685_target_onef1b_MAPE_pct"],
        abs_tol=1e-12,
    )
    _draw(out, summaries, effects, target_metrics, target_phase_metrics)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= review["resource"]["maximum_peak_RSS_bytes"]
    assert elapsed <= review["resource"]["target_analysis_seconds"]
    dump(out / "field_contract.json", {
        "static_factors": "PP depth and microbatch count change only the code-derived schedule and source85/90 semantic cost bindings.",
        "stage_mapping": "Target stages 0-12 map to source 0-12; target terminal stage13 maps to source terminal stage15, matching 2-layer boundaries and 4-layer interiors.",
        "last_microbatch_mapping": "Reduced MB3 last role maps to source MB3; ordinal MB2 is not substituted for the last role.",
        "contraction_shapley": "Two-factor exact allocation of predicted envelope change; it is not target error attribution.",
        "target_development": "Target85/90/95/100 was historically exposed and is used only after seal to quantify mismatch.",
        "unresolved": "The excess predicted contraction has no admitted source-side runtime proxy and remains unmodeled.",
    })
    dump(out / "diagnostic.json", {
        "status": "STATIC_SCHEDULE_OVERCONTRACTION_CONFIRMED_NO_CORRECTION_FIT",
        "new_prediction": False,
        "historical_phase_transfer_reproduced": True,
        "target_timing_read_after_seal": True,
        "source_fit_iterations": FIT,
        "source_incremental_validation": SOURCE_VALIDATION,
        "target_development_iterations": TARGET_DEVELOPMENT,
        "predicted_schedule_contraction_ms": predicted_contraction_ms,
        "observed_schedule_contraction_ms": observed_contraction_ms,
        "overcontraction_ms": predicted_contraction_ms - observed_contraction_ms,
        "PP_depth_shapley_contraction_ms": float(effects.loc[effects.factor.eq("PP_depth_16_to_14"), "absolute_contraction_ms"].item()),
        "microbatch_shapley_contraction_ms": float(effects.loc[effects.factor.eq("microbatches_4_to_3"), "absolute_contraction_ms"].item()),
        "phase_transfer_target_onef1b_MAPE_pct": float(phase_control.onef1b_MAPE_pct),
        "formal_v685_target_onef1b_MAPE_pct": float(formal_control.onef1b_MAPE_pct),
        "target_parameter_updates": 0,
        "formal_topology_replaced": False,
        "prediction_seal_sha256": sha(out / "schedule_contraction_prediction_seal.json"),
        "peak_RSS_bytes": peak,
        "analysis_seconds": elapsed,
        "next": "Search frozen source-derived tables for a pre-execution runtime-state proxy that explains phase-wall slowdown; do not infer a target multiplier from the 3.274-second post-seal mismatch.",
    })

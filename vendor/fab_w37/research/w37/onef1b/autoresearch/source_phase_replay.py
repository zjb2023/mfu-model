"""T33A: held-out all-rank source phase replay on the code-derived 1F1B schedule."""
from __future__ import annotations

from itertools import combinations
import json
import math
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from guards import ROOT
from pp_graph import ObservedCosts, build, envelope
from pp_semantics import align_observations
from readiness import ReadinessCosts
from smoke_worker import dump, sha
from worker import csv


FIT = [85, 90]
VALIDATION = [95, 100]
ITERATIONS = FIT + VALIDATION
PHASE_KEY = ["rank", "pp_stage", "pp_lane", "phase", "microbatch"]
GROUPS = ("phase_wall", "local_runtime", "PP_readiness_completion")


class CompositeCosts:
    """Choose phase and non-phase costs from independently fitted source views."""

    def __init__(self, phase_costs, runtime_costs):
        self.phase_costs = phase_costs
        self.runtime_costs = runtime_costs
        self.last_key = ""
        self.bindings: list[dict] = []

    def _call(self, costs, method: str, item):
        value = getattr(costs, method)(item)
        self.last_key = costs.last_key
        return value

    def initial(self, action):
        return self._call(self.runtime_costs, "initial", action)

    def gap(self, action):
        return self._call(self.runtime_costs, "gap", action)

    def phase(self, action):
        return self._call(self.phase_costs, "phase", action)

    def post(self, action):
        return self._call(self.runtime_costs, "post", action)

    def sender_ready(self, message):
        return self._call(self.runtime_costs, "sender_ready", message)

    def service(self, message):
        return self._call(self.runtime_costs, "service", message)


class MaximumFitPhaseCosts(CompositeCosts):
    """Conservative phase-only ablation; all other costs use the two-fit median."""

    def __init__(self, aligned: pd.DataFrame, runtime_costs):
        super().__init__(runtime_costs, runtime_costs)
        phase = aligned[aligned.iteration.isin(FIT) & aligned.kind.eq("phase")]
        self.phase_table = phase.groupby(
            ["pp_stage", "pp_lane", "name", "microbatch"]
        ).duration_ns.max().to_dict()

    def phase(self, action):
        key = (
            action["pp_stage"], action["pp_lane"], action["name"],
            action["microbatch"],
        )
        self.last_key = f"phase_fit_max:{key}"
        return self.phase_table[key]


class OracleReadinessCosts(ObservedCosts):
    """Observed post-publication wall with a zero sender-delay decomposition."""

    def sender_ready(self, message):
        self.last_key = "validation_oracle_zero_sender_delay"
        return 0

    def service(self, message):
        self.last_key = "validation_oracle_postpublication_wall"
        return super().service(message)


class DiagnosticGroupCosts:
    """Mix fixed predictions and validation oracles for three-way Shapley audit."""

    def __init__(self, fixed, oracle, observed_groups: set[str]):
        self.fixed = fixed
        self.oracle = oracle
        self.observed_groups = observed_groups
        self.last_key = ""
        self.bindings: list[dict] = []

    def _call(self, group: str, method: str, item):
        costs = self.oracle if group in self.observed_groups else self.fixed
        value = getattr(costs, method)(item)
        self.last_key = costs.last_key
        return value

    def initial(self, action):
        return self._call("local_runtime", "initial", action)

    def gap(self, action):
        return self._call("local_runtime", "gap", action)

    def phase(self, action):
        return self._call("phase_wall", "phase", action)

    def post(self, action):
        return self._call("local_runtime", "post", action)

    def sender_ready(self, message):
        return self._call("PP_readiness_completion", "sender_ready", message)

    def service(self, message):
        return self._call("PP_readiness_completion", "service", message)


def _candidates(aligned: pd.DataFrame, pairs: pd.DataFrame):
    median = ReadinessCosts(aligned, pairs, FIT, pp=16, mb=4)
    recent = ReadinessCosts(aligned, pairs, [90], pp=16, mb=4)
    return {
        "median_85_90": median,
        "recent_90_all": recent,
        "recent_90_phase_only": CompositeCosts(recent, median),
        "recent_90_runtime_only": CompositeCosts(median, recent),
        "max_85_90_phase_only": MaximumFitPhaseCosts(aligned, median),
    }


def build_prediction_set(
    aligned: pd.DataFrame, pairs: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    graphs: dict[str, pd.DataFrame] = {}
    summaries = []
    bindings = []
    shared_edges = None
    shared_topology = None
    candidates = _candidates(aligned, pairs)
    for method, costs in candidates.items():
        nodes, edges, topology = build(16, 4, costs)
        if shared_edges is None:
            shared_edges = edges
            shared_topology = topology
        else:
            pd.testing.assert_frame_equal(shared_edges, edges, check_exact=True)
            assert topology == shared_topology
        graphs[method] = nodes
        result = envelope(nodes)
        summaries.append({
            "method": method,
            "fit_iterations": "85|90" if method != "recent_90_all" else "90",
            "predicted_onef1b_ms": result["onef1b_ms"],
            "predicted_program_ms": result["program_ms"],
            "node_count": len(nodes),
            "edge_count": len(edges),
            "candidate_topology_sha256": topology,
        })
        selected = nodes[nodes.duration_ns.gt(0)].copy()
        selected.insert(0, "method", method)
        bindings.append(selected[[
            "method", "node_id", "kind", "rank", "pp_stage", "pp_lane", "phase",
            "microbatch", "region", "cost_key", "duration_ns",
        ]])
    assert shared_edges is not None and shared_topology is not None
    parameter_rows = []
    for method, costs in {"median_85_90": candidates["median_85_90"],
                          "recent_90_all": candidates["recent_90_all"]}.items():
        for row in costs.parameter_rows:
            parameter_rows.append({"method": method, **row})
    parameters = {
        "readiness": pd.DataFrame(parameter_rows),
        "bindings": pd.concat(bindings, ignore_index=True),
    }
    return graphs, shared_edges, pd.DataFrame(summaries), parameters


def _validation_truth(phases: pd.DataFrame) -> pd.DataFrame:
    truth = phases[phases.iteration.isin(ITERATIONS)].copy()
    origin = truth.groupby("iteration").observed_start_ns.transform("min")
    truth["observed_start_relative_ns"] = truth.observed_start_ns - origin
    truth["observed_end_relative_ns"] = truth.observed_end_ns - origin
    return truth


def score_predictions(
    graphs: dict[str, pd.DataFrame], phases: pd.DataFrame, boundaries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = _validation_truth(phases)
    rows = []
    for method, nodes in graphs.items():
        predicted = nodes[nodes.kind.eq("phase_compute_communication_runtime")][
            [*PHASE_KEY, "region", "predicted_start_ns", "predicted_end_ns", "duration_ns"]
        ].copy()
        predicted_origin = int(predicted.predicted_start_ns.min())
        predicted["predicted_start_relative_ns"] = predicted.predicted_start_ns - predicted_origin
        predicted["predicted_end_relative_ns"] = predicted.predicted_end_ns - predicted_origin
        predicted.insert(0, "method", method)
        for iteration in ITERATIONS:
            frame = predicted.copy()
            frame.insert(1, "iteration", iteration)
            rows.append(frame)
    prediction_grid = pd.concat(rows, ignore_index=True)
    scored = prediction_grid.merge(
        truth[["iteration", *PHASE_KEY, "observed_start_relative_ns",
               "observed_end_relative_ns", "duration_ns"]],
        on=["iteration", *PHASE_KEY], how="left", validate="many_to_one",
        suffixes=("_predicted", "_observed"), indicator=True,
    )
    scored["truth_available"] = scored._merge.eq("both")
    assert scored.truth_available.all()
    scored["split"] = np.where(
        scored.iteration.isin(FIT), "source_fit", "source_incremental_validation"
    )
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
    groupings = [
        ["method", "split", "region", "phase"],
        ["method", "split", "region"],
        ["method", "split"],
    ]
    for keys in groupings:
        for values, group in scored.groupby(keys, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            record = dict(zip(keys, values))
            record.setdefault("region", "all")
            record.setdefault("phase", "all")
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

    actual = boundaries[boundaries.iteration.isin(ITERATIONS)][
        ["iteration", "phase_envelope_ms"]
    ].rename(columns={"phase_envelope_ms": "actual_onef1b_ms"})
    global_rows = []
    summary = {
        method: float(envelope(nodes)["onef1b_ms"]) for method, nodes in graphs.items()
    }
    for method, predicted_ms in summary.items():
        for row in actual.itertuples(index=False):
            error = predicted_ms - float(row.actual_onef1b_ms)
            global_rows.append({
                "method": method,
                "iteration": int(row.iteration),
                "split": "source_fit" if row.iteration in FIT else "source_incremental_validation",
                "actual_onef1b_ms": float(row.actual_onef1b_ms),
                "predicted_onef1b_ms": predicted_ms,
                "error_ms": error,
                "APE_pct": 100.0 * abs(error) / float(row.actual_onef1b_ms),
            })
    global_results = pd.DataFrame(global_rows)
    return scored.drop(columns="_merge"), metrics, global_results


def diagnostic_shapley(
    aligned: pd.DataFrame, pairs: pd.DataFrame, boundaries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = ReadinessCosts(aligned, pairs, FIT, pp=16, mb=4)
    actual = boundaries.set_index("iteration").phase_envelope_ms.to_dict()
    subset_rows = []
    shapley_rows = []
    for iteration in VALIDATION:
        oracle = OracleReadinessCosts(aligned, pairs, iteration)
        values: dict[tuple[str, ...], float] = {}
        for count in range(len(GROUPS) + 1):
            for chosen in combinations(GROUPS, count):
                nodes, _, _ = build(
                    16, 4, DiagnosticGroupCosts(fixed, oracle, set(chosen))
                )
                value = float(envelope(nodes)["onef1b_ms"])
                values[chosen] = value
                subset_rows.append({
                    "iteration": iteration,
                    "observed_groups": "|".join(chosen) if chosen else "none",
                    "observed_group_count": count,
                    "predicted_onef1b_ms": value,
                    "actual_onef1b_ms": float(actual[iteration]),
                    "error_ms": value - float(actual[iteration]),
                    "scope": "post-seal development diagnostic; observed substitutions are not predictions",
                })
        assert math.isclose(values[GROUPS], float(actual[iteration]), abs_tol=1e-9)
        for group in GROUPS:
            contribution = 0.0
            others = [item for item in GROUPS if item != group]
            for count in range(len(others) + 1):
                for subset in combinations(others, count):
                    before = tuple(item for item in GROUPS if item in subset)
                    after = tuple(item for item in GROUPS if item in set(subset) | {group})
                    weight = (
                        math.factorial(count)
                        * math.factorial(len(GROUPS) - count - 1)
                        / math.factorial(len(GROUPS))
                    )
                    contribution += weight * (values[after] - values[before])
            shapley_rows.append({
                "iteration": iteration,
                "component": group,
                "correction_to_actual_ms": contribution,
                "fixed_prediction_error_ms": values[()] - float(actual[iteration]),
                "scope": "Shapley allocation of observed post-seal substitutions; explanation only",
            })
        allocated = sum(
            row["correction_to_actual_ms"]
            for row in shapley_rows if row["iteration"] == iteration
        )
        assert math.isclose(allocated, values[GROUPS] - values[()], abs_tol=1e-8)
    return pd.DataFrame(subset_rows), pd.DataFrame(shapley_rows)


def _draw(
    out: Path, global_results: pd.DataFrame, phase_metrics: pd.DataFrame,
    shapley: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "w37-t33a-source-phase-replay"
    validation = global_results[global_results.split.eq("source_incremental_validation")]
    global_metrics = validation.groupby("method", sort=True).agg(
        MAPE_pct=("APE_pct", "mean"), bias_ms=("error_ms", "mean")
    ).reset_index()
    local = phase_metrics[
        phase_metrics.split.eq("source_incremental_validation")
        & phase_metrics.phase.eq("all")
        & phase_metrics.region.ne("all")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    axes[0].barh(global_metrics.method, global_metrics.MAPE_pct, color="#348ABD")
    axes[0].set_xlabel("source95/100 global 1F1B MAPE (%)")
    axes[0].set_title("Free-running all-rank replay")
    for method, group in local.groupby("method", sort=True):
        axes[1].plot(group.region, group.duration_MAE_ms, marker="o", label=method)
    axes[1].set_ylabel("phase duration MAE (ms)")
    axes[1].set_title("Schedule-region local error")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=6)
    pivot = shapley.pivot(index="component", columns="iteration", values="correction_to_actual_ms")
    pivot.plot(kind="bar", ax=axes[2], color=["#7A68A6", "#E24A33"])
    axes[2].axhline(0, color="black", linewidth=.7)
    axes[2].set_ylabel("correction to actual (ms)")
    axes[2].set_title("Post-seal causal accounting")
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("T33A source85/90-fit 1F1B phase replay")
    fig.text(
        .5, .012,
        "All 256 ranks; source95/100 is development validation. Shapley substitutions explain error and are not eligible prediction inputs.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out / "source_phase_replay.svg", metadata={"Date": None})
    fig.savefig(out / "source_phase_replay.png", dpi=150, metadata={"Software": "w37-t33a"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    begin = perf_counter()
    assert plan["diagnostic"] == "source_phase_replay_audit"
    assert plan["diagnostic_access"] == "source_only" and not plan["variants"]
    review = json.loads((ROOT / plan["resource_review_file"]).read_text())
    assert review["fit_iterations"] == FIT
    assert review["incremental_validation"] == VALIDATION
    for item in review["training_semantics"]:
        path = Path(item["path"])
        assert sha(path) == item["sha256"]

    phases = pd.read_csv(paths["source_pp_trace_events_60_100.csv"])
    apis = pd.read_csv(paths["source_pp_api_events_60_100.csv"])
    boundaries = pd.read_csv(paths["source256_profiler_entry_60_100.csv"])
    aligned, pairs = align_observations(phases, apis)
    assert aligned[aligned.kind.eq("phase")].groupby("iteration").size().eq(2048).all()
    assert aligned[aligned.kind.eq("phase")]["rank"].nunique() == 256

    graphs, edges, summaries, parameters = build_prediction_set(aligned, pairs)
    mutated_aligned = aligned.copy()
    mutated_pairs = pairs.copy()
    validation_actions = mutated_aligned.iteration.isin(VALIDATION)
    validation_pairs = mutated_pairs.iteration.isin(VALIDATION)
    numeric_aligned = [
        "start_ns", "end_ns", "duration_ns", "local_prelaunch_gap_ns",
        "api_postjoin_return_ns",
    ]
    numeric_pairs = [
        "sender_post_ns", "receiver_post_ns", "both_published_ns",
        "first_api_return_ns", "last_api_return_ns", "post_publication_upper_bound_ns",
        "sender_wait_for_receiver_ns", "receiver_wait_for_sender_ns",
    ]
    mutated_aligned.loc[validation_actions, numeric_aligned] = -999999999
    mutated_pairs.loc[validation_pairs, numeric_pairs] = -999999999
    changed_graphs, changed_edges, changed_summaries, changed_parameters = build_prediction_set(
        mutated_aligned, mutated_pairs
    )
    pd.testing.assert_frame_equal(edges, changed_edges, check_exact=True)
    pd.testing.assert_frame_equal(summaries, changed_summaries, check_exact=True)
    for name in graphs:
        pd.testing.assert_frame_equal(graphs[name], changed_graphs[name], check_exact=True)
    for name in parameters:
        pd.testing.assert_frame_equal(
            parameters[name], changed_parameters[name], check_exact=True
        )

    csv(out, "source_phase_edges.csv.gz", edges)
    csv(out, "source_phase_prediction_summary.csv", summaries)
    csv(out, "source_phase_readiness_parameters.csv", parameters["readiness"])
    csv(out, "source_phase_cost_bindings.csv.gz", parameters["bindings"])
    prediction_files = [
        "source_phase_edges.csv.gz", "source_phase_prediction_summary.csv",
        "source_phase_readiness_parameters.csv", "source_phase_cost_bindings.csv.gz",
    ]
    for method, nodes in graphs.items():
        name = f"source_phase_{method}_nodes.csv.gz"
        csv(out, name, nodes)
        prediction_files.append(name)
    dump(out / "source_phase_prediction_seal.json", {
        "status": "SEALED_SOURCE85_90_ALLRANK_FREE_RUNNING_PHASE_GRAPHS",
        "fit_iterations": FIT,
        "incremental_validation": VALIDATION,
        "methods": list(graphs),
        "source_rank_count": 256,
        "formal_topology_replaced": False,
        "candidate_topology_is_independent": True,
        "target_parameter_updates": 0,
        "validation_truth_attached_before_seal": False,
        "mutation_gate": "Changing source95/100 aligned phase/API timing leaves all graphs, parameters, bindings and summaries exact.",
        "files": [{"path": name, "sha256": sha(out / name)} for name in prediction_files],
    })

    scored, phase_metrics, global_results = score_predictions(graphs, phases, boundaries)
    subset_results, shapley = diagnostic_shapley(aligned, pairs, boundaries)
    csv(out, "source_phase_validation.csv.gz", scored)
    csv(out, "source_phase_metrics.csv", phase_metrics)
    csv(out, "source_phase_global_iteration_results.csv", global_results)
    global_metrics = global_results.groupby(["method", "split"], sort=True).agg(
        iterations=("iteration", "size"),
        onef1b_MAPE_pct=("APE_pct", "mean"),
        onef1b_bias_ms=("error_ms", "mean"),
        onef1b_MAE_ms=("error_ms", lambda values: values.abs().mean()),
    ).reset_index()
    csv(out, "source_phase_global_metrics.csv", global_metrics)
    csv(out, "source_phase_diagnostic_subsets.csv", subset_results)
    csv(out, "source_phase_diagnostic_shapley.csv", shapley)
    _draw(out, global_results, phase_metrics, shapley)

    for item in json.loads((out / "source_phase_prediction_seal.json").read_text())["files"]:
        assert sha(out / item["path"]) == item["sha256"]
    validation_global = global_metrics[
        global_metrics.split.eq("source_incremental_validation")
    ].set_index("method")
    median_mape = float(validation_global.loc["median_85_90", "onef1b_MAPE_pct"])
    assert math.isclose(median_mape, review["reproduction_reference"]["source95_100_onef1b_MAPE_pct"], abs_tol=1e-12)
    ranked = validation_global.sort_values(["onef1b_MAPE_pct", "onef1b_MAE_ms"])
    best_method = str(ranked.index[0])
    eligible_improvement = (
        best_method != "median_85_90"
        and float(ranked.onef1b_MAPE_pct.iloc[0]) < median_mape
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= review["resource"]["maximum_peak_RSS_bytes"]
    assert elapsed <= review["resource"]["target_analysis_seconds"]
    dump(out / "field_contract.json", {
        "phase_wall": "Observed F/B annotation wall; contains compute, collectives, overlap and local unresolved time and is counted once.",
        "local_runtime": "Code-ordered gap before an action plus post-join API return; it is separate from the phase wall.",
        "PP_readiness_completion": "Latent sender API-entry readiness and post-ready completion fitted only from source85/90 single-message endpoints.",
        "candidate_topology": "Independent phase-level graph derived from pinned non-interleaved schedule code; the formal v684 topology lock is not replaced.",
        "validation": "Source95/100 has prior development exposure; formal isolation and mutation gates do not make it blind.",
        "diagnostic_substitution": "Validation component substitutions and Shapley values are post-seal explanations and cannot become prediction costs.",
        "target": "T33A reads no target timing and emits no target, Step, or MFU prediction.",
    })
    dump(out / "diagnostic.json", {
        "status": "ALLRANK_SOURCE_PHASE_REPLAY_VALIDATED_NO_NEW_TARGET_CANDIDATE",
        "new_prediction": True,
        "new_global_source_prediction": True,
        "new_target_timing_read": False,
        "target_candidate_registered": False,
        "source_fit_iterations": FIT,
        "source_incremental_validation": VALIDATION,
        "source_rank_count": 256,
        "phase_rows_per_iteration": 2048,
        "candidate_node_count": int(summaries.node_count.iloc[0]),
        "candidate_edge_count": int(summaries.edge_count.iloc[0]),
        "candidate_topology_sha256": str(summaries.candidate_topology_sha256.iloc[0]),
        "formal_topology_replaced": False,
        "median85_90_source95_100_onef1b_MAPE_pct": median_mape,
        "best_method": best_method,
        "best_source95_100_onef1b_MAPE_pct": float(ranked.onef1b_MAPE_pct.iloc[0]),
        "eligible_method_improves_prior_median": eligible_improvement,
        "validation_oracle_full_reconstruction_max_error_ns": 0,
        "target_parameter_updates": 0,
        "prediction_seal_sha256": sha(out / "source_phase_prediction_seal.json"),
        "peak_RSS_bytes": peak,
        "analysis_seconds": elapsed,
        "next": (
            "Register a separate target transfer only after reviewing local phase regressions."
            if eligible_improvement else
            "Use the source-validated median phase replay as a control to audit why the PP16/MB4 to PP14/MB3 schedule transfer contracts by far more than the observed source-to-target 1F1B envelope, without fitting target timing."
        ),
    })

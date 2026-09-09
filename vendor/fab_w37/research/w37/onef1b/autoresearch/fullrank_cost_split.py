"""T31A: validate v6.8.5 full-rank source cost components before target use."""
from __future__ import annotations

import hashlib
import json
import math
import resource
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from guards import ROOT
from smoke_worker import dump, sha
from worker import csv


FIT = [85, 90]
VALIDATION = [95, 100]
ITERATIONS = FIT + VALIDATION
ELIGIBLE = ("median_85_90", "recent_90")
LEAKAGE = "formal_v685_all4_leakage_control"
COMPUTE_KEYS = [
    "phase", "layer_type", "layer_context", "execution_scope", "semantic_slot",
]
PP_KEYS = ["source_receiver_stage", "pp_lane", "microbatch"]


def _round_median(values: pd.Series) -> int:
    return int(round(float(values.astype(float).median())))


def physical_observations(observations: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    required = set(COMPUTE_KEYS + [
        "iteration", "rank", "pp_stage", "pp_lane", "microbatch", "layer_id",
        "autograd_phase", "parameter_view", "cost_group", "active_union_ns",
    ])
    assert required.issubset(observations.columns)
    table = observations[
        observations.iteration.isin(ITERATIONS)
        & observations.parameter_view.eq("window_split")
        & observations.cost_group.eq("__physical_slot_total__")
        & observations.autograd_phase.eq("physical_mixed")
    ].copy()
    order = ["iteration", "rank", "phase", "microbatch", "layer_id"] + COMPUTE_KEYS
    table = table.sort_values(order, kind="stable").reset_index(drop=True)
    table.insert(0, "compute_observation_id", [f"compute:{index:05d}" for index in range(len(table))])
    table["split"] = np.where(table.iteration.isin(FIT), "source_fit", "source_incremental_validation")
    if strict:
        assert set(table.iteration.astype(int)) == set(ITERATIONS)
        assert table[COMPUTE_KEYS].drop_duplicates().shape[0] == 58
        assert table.loc[table.iteration.isin(FIT), "rank"].nunique() == 16
        assert table.loc[table.iteration.isin(VALIDATION), "rank"].nunique() == 16
        assert table.pp_stage.nunique() == 16 and table.pp_lane.nunique() == 1
        assert not table.duplicated("compute_observation_id").any()
    return table


def pp_wall_samples(events: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    required = {
        "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch",
        "observed_start_ns", "observed_end_ns",
    }
    assert required.issubset(events.columns)
    selected = events[events.phase.eq("backward") & events.iteration.isin(ITERATIONS)].copy()
    receiver = selected.rename(columns={
        "pp_stage": "source_receiver_stage", "observed_start_ns": "receiver_bwd_start_ns",
    })[["iteration", "pp_lane", "source_receiver_stage", "microbatch", "receiver_bwd_start_ns"]]
    downstream = selected.rename(columns={
        "pp_stage": "source_downstream_stage", "observed_end_ns": "downstream_bwd_end_ns",
    })[["iteration", "pp_lane", "source_downstream_stage", "microbatch", "downstream_bwd_end_ns"]]
    samples = receiver.merge(downstream, on=["iteration", "pp_lane", "microbatch"], validate="many_to_many")
    samples = samples[
        samples.source_downstream_stage.eq(samples.source_receiver_stage + 1)
    ].copy()
    samples["observed_PP_wall_ns"] = samples.receiver_bwd_start_ns - samples.downstream_bwd_end_ns
    samples = samples.sort_values(PP_KEYS + ["iteration"], kind="stable").reset_index(drop=True)
    samples.insert(0, "PP_observation_id", [f"pp:{index:05d}" for index in range(len(samples))])
    samples["split"] = np.where(samples.iteration.isin(FIT), "source_fit", "source_incremental_validation")
    if strict:
        assert len(samples) == len(ITERATIONS) * 15 * 16 * 4
        assert samples.observed_PP_wall_ns.gt(0).all()
        assert not samples.duplicated(PP_KEYS + ["iteration"]).any()
    return samples


def entry_observations(boundaries: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    required = {"iteration", "profiler_entry_to_phase_ms", "profiler_step_ms", "phase_envelope_ms"}
    assert required.issubset(boundaries.columns)
    table = boundaries[boundaries.iteration.isin(ITERATIONS)].copy().sort_values("iteration").reset_index(drop=True)
    table["split"] = np.where(table.iteration.isin(FIT), "source_fit", "source_incremental_validation")
    if strict:
        assert list(table.iteration.astype(int)) == ITERATIONS
        assert table.profiler_entry_to_phase_ms.gt(0).all()
    return table


def _fit_table(table: pd.DataFrame, keys: list[str], value: str, method: str) -> pd.DataFrame:
    fit_iterations = FIT if method == "median_85_90" else [90]
    selected = table[table.iteration.isin(fit_iterations)]
    grouped = selected.groupby(keys, dropna=False, sort=True)[value].agg(
        predicted_cost_ns=_round_median, calibration_samples="size",
    ).reset_index()
    grouped.insert(0, "method", method)
    grouped["fit_iterations"] = "|".join(map(str, fit_iterations))
    grouped["parameter_unit"] = "ns"
    return grouped


def fit_candidates(
    compute: pd.DataFrame, pp: pd.DataFrame, entry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compute_parameters = pd.concat([
        _fit_table(compute, COMPUTE_KEYS, "active_union_ns", method) for method in ELIGIBLE
    ], ignore_index=True)
    pp_parameters = pd.concat([
        _fit_table(pp, PP_KEYS, "observed_PP_wall_ns", method) for method in ELIGIBLE
    ], ignore_index=True)
    entry_rows = []
    for method in ELIGIBLE:
        fit_iterations = FIT if method == "median_85_90" else [90]
        value = float(entry[entry.iteration.isin(fit_iterations)].profiler_entry_to_phase_ms.median())
        entry_rows.append({
            "method": method, "fit_iterations": "|".join(map(str, fit_iterations)),
            "predicted_entry_ms": value, "calibration_samples": len(fit_iterations),
        })
    entry_parameters = pd.DataFrame(entry_rows)

    compute_predictions = compute[["compute_observation_id", "iteration", "split"] + COMPUTE_KEYS].merge(
        compute_parameters[["method"] + COMPUTE_KEYS + ["predicted_cost_ns"]],
        on=COMPUTE_KEYS, how="cross" if False else "inner", validate="many_to_many",
    ).sort_values(["method", "compute_observation_id"], kind="stable").reset_index(drop=True)
    pp_predictions = pp[["PP_observation_id", "iteration", "split"] + PP_KEYS].merge(
        pp_parameters[["method"] + PP_KEYS + ["predicted_cost_ns"]],
        on=PP_KEYS, validate="many_to_many",
    ).sort_values(["method", "PP_observation_id"], kind="stable").reset_index(drop=True)
    entry_predictions = entry[["iteration", "split"]].merge(entry_parameters, how="cross")[
        ["method", "iteration", "split", "predicted_entry_ms"]
    ].sort_values(["method", "iteration"], kind="stable").reset_index(drop=True)
    assert len(compute_predictions) == len(compute) * len(ELIGIBLE)
    assert len(pp_predictions) == len(pp) * len(ELIGIBLE)
    assert len(entry_predictions) == len(entry) * len(ELIGIBLE)
    return (
        compute_parameters, pp_parameters, entry_parameters,
        compute_predictions, pp_predictions, entry_predictions,
    )


def _leakage_compute_parameters(formal: pd.DataFrame) -> pd.DataFrame:
    selected = formal[
        formal.parameter_view.eq("window_split")
        & formal.cost_group.eq("__physical_slot_total__")
        & formal.autograd_phase.eq("physical_mixed")
    ].copy()
    assert len(selected) == 58 and not selected.duplicated(COMPUTE_KEYS).any()
    selected.insert(0, "method", LEAKAGE)
    return selected[["method"] + COMPUTE_KEYS + ["median_active_union_ns"]].rename(
        columns={"median_active_union_ns": "predicted_cost_ns"}
    )


def _leakage_pp_parameters(formal: pd.DataFrame) -> pd.DataFrame:
    assert len(formal) == 15 * 16 * 4 and not formal.duplicated(PP_KEYS).any()
    selected = formal.copy()
    selected.insert(0, "method", LEAKAGE)
    return selected[["method"] + PP_KEYS + ["gradient_wall_median_ns"]].rename(
        columns={"gradient_wall_median_ns": "predicted_cost_ns"}
    )


def _metric_rows(scored: pd.DataFrame, component: str, truth: str, scale: float) -> pd.DataFrame:
    rows = []
    phases = ["all"] + (sorted(scored.phase.unique()) if "phase" in scored else [])
    for (method, split), group in scored.groupby(["method", "split"], sort=True):
        for phase in phases:
            local = group if phase == "all" else group[group.phase.eq(phase)]
            error = local.prediction_error.to_numpy(dtype=float)
            actual = local[truth].to_numpy(dtype=float)
            rows.append({
                "component": component, "method": method, "split": split, "phase": phase,
                "observations": len(local), "MAE": float(np.abs(error).mean() / scale),
                "bias": float(error.mean() / scale),
                "RMSE": float(math.sqrt(np.mean(error ** 2)) / scale),
                "WAPE_pct": float(100.0 * np.abs(error).sum() / np.abs(actual).sum()),
                "zero_truth_rows": int(np.count_nonzero(actual == 0)),
                "metric_unit": "ms",
            })
    return pd.DataFrame(rows)


def _per_iteration(scored: pd.DataFrame, component: str, truth: str, scale: float) -> pd.DataFrame:
    rows = []
    group_keys = ["method", "iteration"] + (["phase"] if "phase" in scored else [])
    for key, group in scored.groupby(group_keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_keys, key))
        error = group.prediction_error.to_numpy(dtype=float)
        actual = group[truth].to_numpy(dtype=float)
        rows.append(record | {
            "component": component, "observations": len(group),
            "MAE": float(np.abs(error).mean() / scale), "bias": float(error.mean() / scale),
            "WAPE_pct": float(100.0 * np.abs(error).sum() / np.abs(actual).sum()),
            "metric_unit": "ms",
        })
    return pd.DataFrame(rows)


def _source_binding_audit(paths: dict[str, Path], review: dict) -> tuple[pd.DataFrame, dict]:
    for item in review["builder_evidence"]:
        path = ROOT / item["path"]
        assert sha(path) == item["sha256"]
    builder_path = ROOT / review["builder_evidence"][0]["path"]
    lines = builder_path.read_text(encoding="utf-8").splitlines()
    def matching(pattern: str) -> list[int]:
        return [index + 1 for index, line in enumerate(lines) if re.search(pattern, line)]

    target = pd.read_csv(paths["dag_v685_nodes.csv.gz"], low_memory=False)
    updates = pd.read_csv(paths["formal_v685_target_compute_updates.csv"])
    source = pd.read_csv(paths["formal_v685_source_nodes.csv.gz"], low_memory=False)
    contract = json.loads(paths["prediction_contract.json"].read_text())
    target_ids = set(target.node_id.astype(str))
    update_ids = set(updates.node_id.astype(str))
    target_bound = target[target.node_id.astype(str).isin(update_ids)]
    assert len(updates) == 144 and update_ids.issubset(target_ids) and len(target_bound) == 144
    assert target_bound.source_parameter_key.notna().all()
    source_has_parameter_key = "source_parameter_key" in source.columns
    source_new_timing_rows = int(source.get("timing_source", pd.Series(index=source.index, dtype="object"))
                                 .fillna("").astype(str).str.contains("iterations_85_100_median").sum())
    source_overlap = int(source.node_id.astype(str).isin(update_ids).sum())
    rows = pd.DataFrame([
        {"graph": "target224_v685", "rows": len(target), "new_compute_update_rows": len(updates),
         "source_parameter_key_column": "source_parameter_key" in target.columns,
         "rows_with_new_compute_timing_source": int(target.timing_source.fillna("").astype(str).str.contains("iterations_85_100_median").sum()),
         "update_node_id_matches": len(target_bound), "candidate_compute_binding_valid": True},
        {"graph": "source256_v685_replay", "rows": len(source), "new_compute_update_rows": 0,
         "source_parameter_key_column": source_has_parameter_key,
         "rows_with_new_compute_timing_source": source_new_timing_rows,
         "update_node_id_matches": source_overlap, "candidate_compute_binding_valid": False},
    ])
    compute_call_lines = matching(r"nodes, compute_updates = apply_steady_compute")
    source_read_lines = matching(r"source_nodes = pd\.read_csv")
    source_pp_lines = matching(r"source_nodes, source_pp_updates = apply_source_gradient_wall")
    source_compute_lines = matching(r"apply_steady_compute\(source_nodes")
    assert len(compute_call_lines) == 1 and len(source_read_lines) == 1 and len(source_pp_lines) == 1
    assert source_compute_lines == []
    detail = {
        "status": "FORMAL_V685_SOURCE_REPLAY_DOES_NOT_BIND_REFIT_COMPUTE",
        "builder_path": str(builder_path.relative_to(ROOT)),
        "builder_sha256": sha(builder_path),
        "target_apply_steady_compute_lines": compute_call_lines,
        "source_nodes_read_lines": source_read_lines,
        "source_apply_PP_lines": source_pp_lines,
        "source_apply_steady_compute_lines": source_compute_lines,
        "formal_target_compute_updates": len(updates),
        "formal_target_compute_binding_rows": len(target_bound),
        "formal_source_compute_binding_rows": 0,
        "source_target_update_node_id_overlap": source_overlap,
        "source_has_source_parameter_key_column": source_has_parameter_key,
        "source_has_compute_exposed_column": "compute_exposed_ns_model" in source.columns,
        "source_has_compute_overlap_column": "compute_overlap_ns_model" in source.columns,
        "source_local_gap_rows": int(source.kind.eq("local_gap").sum()),
        "source_local_gap_unclassified_ns": int(source.loc[source.kind.eq("local_gap"), "unclassified_calibration_ns"].sum()),
        "source_PP_message_rows": int(source.kind.eq("pp_message").sum()),
        "source_raw_graph_ms": float(contract["prediction"]["source_raw_graph_ms"]),
        "source_reconciliation_ms": float(contract["prediction"]["source_reconciliation_ms"]),
        "reconciliation_compute_validation_status": "INVALID_FOR_REFIT_COMPUTE_BECAUSE_SOURCE_REPLAY_KEPT_OLD_LOCAL_GAPS",
    }
    return rows, detail


def _draw(out: Path, metrics: pd.DataFrame, binding: pd.DataFrame) -> None:
    import os
    os.environ["MPLCONFIGDIR"] = str(out / "mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "w37-t31a-fullrank-cost-split"
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    validation = metrics[(metrics.split == "source_incremental_validation") & (metrics.phase == "all")]
    components = ["compute_physical_slot", "PP_gradient_wall", "entry"]
    methods = [*ELIGIBLE, LEAKAGE]
    for index, method in enumerate(methods):
        local = validation[validation.method.eq(method)].set_index("component").reindex(components)
        axes[0].bar(np.arange(3) + (index - 1) * .25, local.MAE, .24, label=method)
    axes[0].set_xticks(range(3), ["compute slot", "PP wall", "entry"])
    axes[0].set_ylabel("source95/100 MAE (ms; per observation)")
    axes[0].set_title("Component validation; all4 is exposed-data control")
    axes[0].legend(fontsize=8)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[1].bar(binding.graph, binding.new_compute_update_rows, color=["#2a9d8f", "#e76f51"])
    axes[1].set_ylabel("new compute-bound rows")
    axes[1].set_title("v6.8.5 refit compute binding")
    axes[1].text(1, 2, "source replay: 0\nreconciliation cannot validate refit compute", ha="center")
    fig.suptitle("T31A source-only full-rank cost split")
    fig.text(.5, .015, "Fit: source85/90. Validation: already-exposed source95/100. No target timing or topology change.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out / "source_fullrank_cost_split.svg", metadata={"Date": None})
    fig.savefig(out / "source_fullrank_cost_split.png", dpi=150)
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    begin = perf_counter()
    assert plan["diagnostic"] == "source_fullrank_cost_split"
    assert plan["diagnostic_access"] == "source_only" and not plan["variants"]
    review = json.loads((ROOT / plan["resource_review_file"]).read_text())
    assert review["fit_iterations"] == FIT and review["incremental_validation"] == VALIDATION

    raw_compute = pd.read_csv(paths["operator_cost_group_observations.csv.gz"])
    raw_pp = pd.read_csv(paths["source_pp_trace_events_60_100.csv"])
    raw_entry = pd.read_csv(paths["source256_profiler_entry_60_100.csv"])
    compute = physical_observations(raw_compute)
    pp = pp_wall_samples(raw_pp)
    entry = entry_observations(raw_entry)
    fitted = fit_candidates(compute, pp, entry)

    mutated_compute = compute.copy()
    mutated_compute.loc[mutated_compute.iteration.isin(VALIDATION), "active_union_ns"] = -999999999
    mutated_pp = pp.copy()
    mask = mutated_pp.iteration.isin(VALIDATION)
    mutated_pp.loc[mask, ["observed_PP_wall_ns", "receiver_bwd_start_ns", "downstream_bwd_end_ns"]] = -999999999
    mutated_entry = entry.copy()
    mutated_entry.loc[mutated_entry.iteration.isin(VALIDATION), [
        "profiler_entry_to_phase_ms", "profiler_step_ms", "phase_envelope_ms",
    ]] = -999999.0
    other = fit_candidates(mutated_compute, mutated_pp, mutated_entry)
    for left, right in zip(fitted, other):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    compute_parameters, pp_parameters, entry_parameters, compute_predictions, pp_predictions, entry_predictions = fitted

    csv(out, "source_compute_parameters.csv", compute_parameters)
    csv(out, "source_PP_parameters.csv", pp_parameters)
    csv(out, "source_entry_parameters.csv", entry_parameters)
    csv(out, "source_compute_predictions_sealed.csv.gz", compute_predictions)
    csv(out, "source_PP_predictions_sealed.csv.gz", pp_predictions)
    csv(out, "source_entry_predictions_sealed.csv", entry_predictions)
    sealed_names = [
        "source_compute_parameters.csv", "source_PP_parameters.csv", "source_entry_parameters.csv",
        "source_compute_predictions_sealed.csv.gz", "source_PP_predictions_sealed.csv.gz",
        "source_entry_predictions_sealed.csv",
    ]
    dump(out / "source_fullrank_cost_prediction_seal.json", {
        "status": "SEALED_SOURCE85_90_COMPONENT_COST_CANDIDATES",
        "fit_iterations": FIT, "incremental_validation": VALIDATION,
        "eligible_methods": list(ELIGIBLE), "target_parameter_updates": 0,
        "validation_truth_attached_before_seal": False,
        "files": [{"path": name, "sha256": sha(out / name)} for name in sealed_names],
        "mutation_gate": "Changing every source95/100 compute, PP and entry truth value leaves all eligible parameters and predictions exact.",
    })

    compute_truth = compute[["compute_observation_id", "rank", "pp_stage", "pp_lane", "microbatch", "layer_id", "active_union_ns"]]
    compute_scored = compute_predictions.merge(compute_truth, on="compute_observation_id", validate="many_to_one")
    compute_scored["prediction_error"] = compute_scored.predicted_cost_ns - compute_scored.active_union_ns
    pp_truth = pp[["PP_observation_id", "observed_PP_wall_ns"]]
    pp_scored = pp_predictions.merge(pp_truth, on="PP_observation_id", validate="many_to_one")
    pp_scored["prediction_error"] = pp_scored.predicted_cost_ns - pp_scored.observed_PP_wall_ns
    entry_truth = entry[["iteration", "profiler_entry_to_phase_ms"]]
    entry_scored = entry_predictions.merge(entry_truth, on="iteration", validate="many_to_one")
    entry_scored["prediction_error"] = entry_scored.predicted_entry_ms - entry_scored.profiler_entry_to_phase_ms

    formal_compute = _leakage_compute_parameters(pd.read_csv(paths["formal_v685_compute_parameters.csv"]))
    formal_pp = _leakage_pp_parameters(pd.read_csv(paths["formal_v685_PP_parameters.csv"]))
    recomputed_all4 = compute.groupby(COMPUTE_KEYS, sort=True).active_union_ns.agg(_round_median).reset_index(name="expected")
    formal_check = formal_compute.merge(recomputed_all4, on=COMPUTE_KEYS, validate="one_to_one")
    assert (formal_check.predicted_cost_ns == formal_check.expected).all()
    recomputed_pp = pp.groupby(PP_KEYS, sort=True).observed_PP_wall_ns.agg(_round_median).reset_index(name="expected")
    formal_pp_check = formal_pp.merge(recomputed_pp, on=PP_KEYS, validate="one_to_one")
    assert (formal_pp_check.predicted_cost_ns == formal_pp_check.expected).all()

    formal_compute_pred = compute[["compute_observation_id", "iteration", "split"] + COMPUTE_KEYS].merge(
        formal_compute, on=COMPUTE_KEYS, validate="many_to_one")
    formal_compute_pred = formal_compute_pred.merge(compute_truth, on="compute_observation_id", validate="many_to_one")
    formal_compute_pred["prediction_error"] = formal_compute_pred.predicted_cost_ns - formal_compute_pred.active_union_ns
    formal_pp_pred = pp[["PP_observation_id", "iteration", "split"] + PP_KEYS].merge(
        formal_pp, on=PP_KEYS, validate="many_to_one").merge(pp_truth, on="PP_observation_id", validate="many_to_one")
    formal_pp_pred["prediction_error"] = formal_pp_pred.predicted_cost_ns - formal_pp_pred.observed_PP_wall_ns
    formal_entry_value = float(entry.profiler_entry_to_phase_ms.median())
    formal_entry_pred = entry[["iteration", "split", "profiler_entry_to_phase_ms"]].copy()
    formal_entry_pred["method"] = LEAKAGE
    formal_entry_pred["predicted_entry_ms"] = formal_entry_value
    formal_entry_pred["prediction_error"] = formal_entry_pred.predicted_entry_ms - formal_entry_pred.profiler_entry_to_phase_ms
    compute_scored = pd.concat([compute_scored, formal_compute_pred], ignore_index=True, sort=False)
    pp_scored = pd.concat([pp_scored, formal_pp_pred], ignore_index=True, sort=False)
    entry_scored = pd.concat([entry_scored, formal_entry_pred], ignore_index=True, sort=False)

    compute_metrics = _metric_rows(compute_scored, "compute_physical_slot", "active_union_ns", 1e6)
    pp_metrics = _metric_rows(pp_scored, "PP_gradient_wall", "observed_PP_wall_ns", 1e6)
    entry_metrics = _metric_rows(entry_scored, "entry", "profiler_entry_to_phase_ms", 1.0)
    metrics = pd.concat([compute_metrics, pp_metrics, entry_metrics], ignore_index=True)
    per_iteration = pd.concat([
        _per_iteration(compute_scored, "compute_physical_slot", "active_union_ns", 1e6),
        _per_iteration(pp_scored, "PP_gradient_wall", "observed_PP_wall_ns", 1e6),
        _per_iteration(entry_scored, "entry", "profiler_entry_to_phase_ms", 1.0),
    ], ignore_index=True, sort=False)
    csv(out, "source_compute_validation_results.csv.gz", compute_scored)
    csv(out, "source_PP_validation_results.csv.gz", pp_scored)
    csv(out, "source_entry_validation_results.csv", entry_scored)
    csv(out, "source_component_metrics.csv", metrics)
    csv(out, "source_component_per_iteration.csv", per_iteration)

    binding, binding_detail = _source_binding_audit(paths, review)
    csv(out, "v685_source_compute_binding_audit.csv", binding)
    dump(out / "v685_source_compute_binding_audit.json", binding_detail)
    ledger = pd.DataFrame([
        {"component": "compute", "eligible_fit": "source85/90", "validation": "source95/100 local physical slots",
         "formal_source_graph_binding": "MISSING", "double_count_rule": "replace physical-slot compute only"},
        {"component": "PP_gradient_wall", "eligible_fit": "source85/90", "validation": "source95/100 PP walls",
         "formal_source_graph_binding": "AVAILABLE_FOR_FORMAL_ALL4_ONLY", "double_count_rule": "single PP wall includes network plus software completion"},
        {"component": "entry", "eligible_fit": "source85/90", "validation": "source95/100 entry wall",
         "formal_source_graph_binding": "OUTSIDE_SOURCE_GRAPH", "double_count_rule": "report at profiler clock origin only"},
        {"component": "reconciliation", "eligible_fit": "NONE", "validation": "not identifiable without candidate source graph",
         "formal_source_graph_binding": "OLD_LOCAL_GAPS_PLUS_NEW_PP", "double_count_rule": "must not absorb unbound compute error"},
    ])
    csv(out, "component_accounting_ledger.csv", ledger)

    validation = metrics[(metrics.split == "source_incremental_validation") & (metrics.phase == "all")]
    def value(component: str, method: str, field: str = "MAE") -> float:
        return float(validation[(validation.component == component) & (validation.method == method)][field].item())
    compute_improvement = 100.0 * (value("compute_physical_slot", "median_85_90") - value("compute_physical_slot", "recent_90")) / value("compute_physical_slot", "median_85_90")
    pp_regression = 100.0 * (value("PP_gradient_wall", "recent_90") - value("PP_gradient_wall", "median_85_90")) / value("PP_gradient_wall", "median_85_90")
    entry_regression = 100.0 * (value("entry", "recent_90") - value("entry", "median_85_90")) / value("entry", "median_85_90")

    prior = Path(plan["sealed_reference"]["run_root"]) / "models" / plan["sealed_reference"]["variant"]
    prior_metrics = pd.read_csv(prior / "source_validation/metrics.csv")
    prior_incremental = prior_metrics[
        prior_metrics.variant.eq(plan["sealed_reference"]["variant"])
        & prior_metrics.split.eq("source_incremental_validation")
    ].iloc[0]
    comparison = {
        "status": "LOCAL_COMPUTE_IMPROVES_BUT_COMPONENTS_DIVERGE_AND_GLOBAL_SOURCE_BINDING_IS_MISSING",
        "eligible_validation": {
            "median_85_90": {
                "compute_MAE_ms": value("compute_physical_slot", "median_85_90"),
                "PP_MAE_ms": value("PP_gradient_wall", "median_85_90"),
                "entry_MAE_ms": value("entry", "median_85_90"),
            },
            "recent_90": {
                "compute_MAE_ms": value("compute_physical_slot", "recent_90"),
                "PP_MAE_ms": value("PP_gradient_wall", "recent_90"),
                "entry_MAE_ms": value("entry", "recent_90"),
            },
        },
        "recent90_vs_median85_90": {
            "compute_MAE_improvement_pct": compute_improvement,
            "PP_MAE_regression_pct": pp_regression,
            "entry_MAE_regression_pct": entry_regression,
        },
        "formal_v685_all4_exposed_control": {
            "compute_MAE_ms": value("compute_physical_slot", LEAKAGE),
            "PP_MAE_ms": value("PP_gradient_wall", LEAKAGE),
            "entry_MAE_ms": value("entry", LEAKAGE),
            "promotion_eligible": False,
        },
        "v6813_different_graph_control": {
            "source95_100_global_1F1B_MAPE_pct": float(prior_incremental.onef1b_mape_pct),
            "bias_ms": float(prior_incremental.bias_ms),
            "comparable_as_v685_cost_validation": False,
        },
        "single_eligible_estimator_dominates_all_components": False,
        "candidate_source_global_1F1B_available": False,
        "T31B_target_scoring_registered": False,
    }
    dump(out / "candidate_comparison.json", comparison)
    _draw(out, metrics, binding)

    for item in json.loads((out / "source_fullrank_cost_prediction_seal.json").read_text())["files"]:
        assert sha(out / item["path"]) == item["sha256"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= review["resource"]["maximum_peak_RSS_bytes"]
    assert elapsed <= review["resource"]["target_analysis_seconds"]
    dump(out / "field_contract.json", {
        "compute": "Physical window-split active union only, keyed exactly as v6.8.5; subgroups excluded from the cost total.",
        "PP": "Backward receiver start minus downstream end; one network-plus-software wall per receiver/lane/microbatch.",
        "entry": "Profiler start to first F/B envelope start; kept outside internal 1F1B.",
        "reconciliation": "No eligible estimate: v6.8.5 source replay does not bind its refit compute, so residual closure would hide the missing binding.",
        "validation": "Source95/100 was historically inspected and is development validation, not blind validation.",
        "target": "No target timing read, parameter, prediction, Step or MFU result in T31A.",
    })
    dump(out / "diagnostic.json", {
        "status": "SOURCE_FULLRANK_COST_SPLIT_PASS",
        "new_prediction": True, "new_global_prediction": False, "used_to_fit_model": True,
        "new_target_timing_read": False, "raw_trace_scanned": False,
        "source_iterations": ITERATIONS, "source_fit_iterations": FIT,
        "source_incremental_validation": VALIDATION,
        "compute_representative_ranks": sorted(compute["rank"].astype(int).unique().tolist()),
        "compute_rank_count": int(compute["rank"].nunique()),
        "compute_PP_stage_count": int(compute.pp_stage.nunique()), "compute_PP_lane_count": int(compute.pp_lane.nunique()),
        "PP_event_rank_count": int(raw_pp[raw_pp.iteration.isin(ITERATIONS)]["rank"].nunique()),
        "physical_compute_keys": 58, "physical_compute_observations": len(compute),
        "PP_keys": 15 * 16 * 4, "PP_observations": len(pp),
        "eligible_parameters_predictions_validation_truth_mutation_exact": True,
        "recent90_compute_MAE_improvement_pct": compute_improvement,
        "recent90_PP_MAE_regression_pct": pp_regression,
        "recent90_entry_MAE_regression_pct": entry_regression,
        "formal_v685_source_compute_binding_rows": 0,
        "formal_v685_source_replay_refit_compute_validated": False,
        "source_global_1F1B_candidate_available": False,
        "reconciliation_candidate_available": False,
        "promotion": "REJECT_T31B_MISSING_COMPATIBLE_SOURCE_GRAPH_AND_NO_COMPONENT_DOMINANCE",
        "formal_topology_changed": False, "target_parameter_updates": 0,
        "prediction_seal_sha256": sha(out / "source_fullrank_cost_prediction_seal.json"),
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed, "resource_gate_pass": True,
        "next": "Repair source replay validation at the clock-origin/accounting level or find an existing source graph whose compute slots bind the v6.8.5 keys; do not use reconciliation to conceal the missing compute binding.",
    })

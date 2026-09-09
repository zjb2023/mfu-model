"""T35A: fail-closed cold-start audit and separately labelled online 1F1B nowcast."""
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
from schedule_contraction import RoleMappedReadinessCosts
from smoke_worker import dump, sha
from worker import csv


FIT = [85, 90]
SOURCE_DEVELOPMENT = [95, 100]
TARGET_DEVELOPMENT = [85, 90, 95, 100]
MFU_NUMERATOR = 100 * 8.436548311982576e16 / (224 * 500e12) * 1000
PHASE_KEY = ["rank", "pp_stage", "pp_lane", "phase", "microbatch"]
PRIMARY = "online_midrun_last_half_winsor20"
CANDIDATES = {
    "cold_start_phase_transfer": {"statistic": "none", "backward_scope": "none"},
    "online_forward_only_winsor20": {"statistic": "winsor20", "backward_scope": "none"},
    PRIMARY: {"statistic": "winsor20", "backward_scope": "last_half"},
    "online_midrun_last_half_median": {"statistic": "median", "backward_scope": "last_half"},
    "online_late_all_stages_winsor20": {"statistic": "winsor20", "backward_scope": "all"},
}


class DirectionalScaleCosts:
    """Scale the existing phase wall once; preserve runtime, PP and all dependencies."""

    def __init__(self, base, forward_factor: float, backward_factor: float):
        self.base = base
        self.factors = {"forward": forward_factor, "backward": backward_factor}
        self.last_key = ""
        self.bindings: list[dict] = []

    def _delegate(self, name: str, item):
        value = getattr(self.base, name)(item)
        self.last_key = self.base.last_key
        return value

    def initial(self, item):
        return self._delegate("initial", item)

    def gap(self, item):
        return self._delegate("gap", item)

    def post(self, item):
        return self._delegate("post", item)

    def sender_ready(self, item):
        return self._delegate("sender_ready", item)

    def service(self, item):
        return self._delegate("service", item)

    def phase(self, item):
        value = self._delegate("phase", item)
        direction = item["name"]
        factor = self.factors[direction]
        self.last_key = f"online_directional_scale:{direction}:{factor:.12f}:{self.last_key}"
        return round(value * factor)


def winsorized_mean(values, proportion: float = 0.2) -> float:
    values = np.asarray(values, dtype=float)
    assert values.size and 0 <= proportion < 0.5
    lower, upper = np.quantile(values, [proportion, 1 - proportion])
    return float(np.clip(values, lower, upper).mean())


def direction_shapley(base: float, forward_only: float, backward_only: float, both: float):
    forward = 0.5 * ((forward_only - base) + (both - backward_only))
    backward = 0.5 * ((backward_only - base) + (both - forward_only))
    assert math.isclose(forward + backward, both - base, abs_tol=1e-9)
    return forward, backward


def _phase_column(frame: pd.DataFrame) -> pd.Series:
    return frame["phase"] if "phase" in frame else frame["name"]


def _prefix_mask(frame: pd.DataFrame, pp: int, method: str) -> pd.Series:
    phase = _phase_column(frame)
    config = CANDIDATES[method]
    if method == "cold_start_phase_transfer":
        return pd.Series(False, index=frame.index)
    forward = phase.eq("forward") & frame.microbatch.eq(0)
    if config["backward_scope"] == "none":
        backward = pd.Series(False, index=frame.index)
    elif config["backward_scope"] == "last_half":
        backward = (
            phase.eq("backward") & frame.microbatch.eq(0)
            & frame.pp_stage.ge(pp // 2)
        )
    else:
        backward = phase.eq("backward") & frame.microbatch.eq(0)
    return forward | backward


def prefix_projection(
    phases: pd.DataFrame, iteration: int, pp: int, base, method: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = phases[phases.iteration.eq(iteration)].copy()
    frame["phase"] = _phase_column(frame)
    prefix = frame[_prefix_mask(frame, pp, method)].copy()
    columns = [
        "method", "iteration", *PHASE_KEY, "feature_role",
        "observed_duration_ns", "source_cost_ns", "observed_to_source_ratio",
    ]
    if prefix.empty:
        return pd.DataFrame(columns=columns), {
            "method": method, "iteration": iteration, "PP": pp,
            "forward_factor": 1.0, "backward_factor": 1.0,
            "forward_samples": 0, "backward_samples": 0,
            "statistic": "none", "backward_scope": "none",
        }
    source_cost = []
    for row in prefix.to_dict("records"):
        source_cost.append(base.phase({
            "pp_stage": int(row["pp_stage"]),
            "pp_lane": int(row["pp_lane"]),
            "name": row["phase"],
            "microbatch": int(row["microbatch"]),
        }))
    prefix["source_cost_ns"] = source_cost
    prefix["observed_duration_ns"] = prefix.duration_ns.astype("int64")
    prefix["observed_to_source_ratio"] = (
        prefix.observed_duration_ns / prefix.source_cost_ns
    )
    prefix["feature_role"] = np.where(
        prefix.phase.eq("forward"), "all_stage_F_microbatch0",
        "admitted_stage_B_microbatch0",
    )
    prefix.insert(0, "method", method)
    statistic = CANDIDATES[method]["statistic"]

    def aggregate(direction: str) -> float:
        values = prefix.loc[prefix.phase.eq(direction), "observed_to_source_ratio"]
        if values.empty:
            return 1.0
        if statistic == "median":
            return float(values.median())
        assert statistic == "winsor20"
        return winsorized_mean(values, 0.2)

    factors = {
        "method": method,
        "iteration": iteration,
        "PP": pp,
        "forward_factor": aggregate("forward"),
        "backward_factor": aggregate("backward"),
        "forward_samples": int(prefix.phase.eq("forward").sum()),
        "backward_samples": int(prefix.phase.eq("backward").sum()),
        "statistic": statistic,
        "backward_scope": CANDIDATES[method]["backward_scope"],
    }
    return prefix[columns], factors


def build_prediction_set(
    phases: pd.DataFrame, base, pp: int, microbatches: int, iterations: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, factors, predictions, node_frames = [], [], [], []
    shared_edges = None
    shared_topology = None
    for method in CANDIDATES:
        for iteration in iterations:
            feature, factor = prefix_projection(phases, iteration, pp, base, method)
            costs = DirectionalScaleCosts(
                base, float(factor["forward_factor"]), float(factor["backward_factor"])
            )
            nodes, edges, topology = build(pp, microbatches, costs)
            if shared_edges is None:
                shared_edges, shared_topology = edges, topology
            else:
                pd.testing.assert_frame_equal(shared_edges, edges, check_exact=True)
                assert topology == shared_topology
            result = envelope(nodes)
            predictions.append({
                **factor,
                "predicted_onef1b_ms": result["onef1b_ms"],
                "predicted_program_ms": result["program_ms"],
                "node_count": len(nodes), "edge_count": len(edges),
                "candidate_topology_sha256": topology,
            })
            nodes.insert(0, "iteration", iteration)
            nodes.insert(0, "method", method)
            node_frames.append(nodes)
            features.append(feature)
            factors.append(factor)
    assert shared_edges is not None
    return (
        pd.concat(features, ignore_index=True), pd.DataFrame(factors),
        pd.DataFrame(predictions), pd.concat(node_frames, ignore_index=True),
    ), shared_edges


def score_phase(nodes: pd.DataFrame, phases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted = nodes[nodes.kind.eq("phase_compute_communication_runtime")][
        ["method", "iteration", *PHASE_KEY, "region", "duration_ns"]
    ].rename(columns={"duration_ns": "predicted_duration_ns"})
    truth = phases.copy()
    truth["phase"] = _phase_column(truth)
    scored = predicted.merge(
        truth[["iteration", *PHASE_KEY, "duration_ns"]],
        on=["iteration", *PHASE_KEY], how="left", validate="many_to_one",
        indicator=True,
    )
    assert scored._merge.eq("both").all()
    scored = scored.drop(columns="_merge").rename(
        columns={"duration_ns": "observed_duration_ns"}
    )
    scored["duration_error_ms"] = (
        scored.predicted_duration_ns - scored.observed_duration_ns
    ) / 1e6
    metric_rows = []
    for keys in [["method", "region", "phase"], ["method", "phase"], ["method"]]:
        for values, group in scored.groupby(keys, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            record = dict(zip(keys, values))
            record.setdefault("region", "all")
            record.setdefault("phase", "all")
            metric_rows.append({
                **record, "observations": len(group),
                "duration_MAE_ms": group.duration_error_ms.abs().mean(),
                "duration_bias_ms": group.duration_error_ms.mean(),
            })
    return scored, pd.DataFrame(metric_rows)


def score_iterations(
    predictions: pd.DataFrame, actual: dict[int, float], split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = predictions.copy()
    scored["actual_onef1b_ms"] = scored.iteration.map(actual)
    assert scored.actual_onef1b_ms.notna().all()
    scored["error_ms"] = scored.predicted_onef1b_ms - scored.actual_onef1b_ms
    scored["APE_pct"] = 100 * scored.error_ms.abs() / scored.actual_onef1b_ms
    scored["split"] = split
    metrics = scored.groupby("method", sort=True).agg(
        iterations=("iteration", "size"),
        onef1b_MAPE_pct=("APE_pct", "mean"),
        onef1b_MAE_ms=("error_ms", lambda values: values.abs().mean()),
        onef1b_bias_ms=("error_ms", "mean"),
    ).reset_index()
    return scored, metrics


def cutoff_scores(
    phases: pd.DataFrame, predictions: pd.DataFrame, pp: int,
    actual: dict[int, float], domain: str,
) -> pd.DataFrame:
    rows = []
    for predicted in predictions.itertuples(index=False):
        frame = phases[phases.iteration.eq(predicted.iteration)].copy()
        frame["phase"] = _phase_column(frame)
        selected = frame[_prefix_mask(frame, pp, predicted.method)]
        if selected.empty:
            elapsed_ms = 0.0
        else:
            elapsed_ms = (
                float(selected.observed_end_ns.max())
                - float(frame.observed_start_ns.min())
            ) / 1e6
        total = float(actual[int(predicted.iteration)])
        rows.append({
            "domain": domain, "method": predicted.method,
            "iteration": int(predicted.iteration),
            "prefix_available_elapsed_ms": elapsed_ms,
            "actual_onef1b_ms": total,
            "prefix_available_fraction_pct": 100 * elapsed_ms / total,
            "remaining_fraction_pct": 100 * (total - elapsed_ms) / total,
        })
    return pd.DataFrame(rows)


def static_capability_audit(paths: dict[str, Path]) -> tuple[dict, pd.DataFrame]:
    readiness = json.loads(paths["graph_input_readiness.json"].read_text())
    schema = json.loads(paths["runtime_input_schema.json"].read_text())
    contract = json.loads(paths["v69_runtime_readiness_contract.json"].read_text())
    workload = pd.read_csv(paths["static_stage_workload.csv"])
    blocked = readiness["blocked_required"]
    assert readiness["prediction_gate"] == "BLOCKED"
    assert set(blocked) == set(schema["required"])
    assert set(blocked).issubset({x["id"] for x in contract["predictor_safe_required_inputs"]})
    assert workload.pp_stage.nunique() == 14 and workload.profile_id.nunique() == 3
    forbidden_runtime_columns = {
        "iteration", "duration_ns", "duration_ms", "runtime_stall_ms",
        "rank_release_ns", "observed_start_ns", "observed_end_ns",
    }
    assert not forbidden_runtime_columns.intersection(workload.columns)
    rows = []
    for column in workload.columns:
        rows.append({
            "column": column, "unique_values": int(workload[column].nunique()),
            "execution_time_varying": False,
            "can_identify_iteration_runtime_readiness": False,
            "role": "static stage/workload shape only",
        })
    result = {
        "status": "COLD_START_RUNTIME_READINESS_FAIL_CLOSED",
        "capability_mode": contract["capability_modes"]["cold_start_static"],
        "prediction_gate": readiness["prediction_gate"],
        "blocked_required": blocked,
        "static_stage_rows": len(workload),
        "static_unique_profiles": int(workload.profile_id.nunique()),
        "static_iteration_varying_fields": 0,
        "target_timing_used_for_audit": False,
        "conclusion": "The admitted static table describes graph shape but supplies neither router-token variation nor an independent runtime-readiness state. No cold-start correction is identifiable.",
    }
    return result, pd.DataFrame(rows)


def _actual_envelope(phases: pd.DataFrame, iterations: list[int]) -> dict[int, float]:
    frame = phases[phases.iteration.isin(iterations)].groupby("iteration").agg(
        first=("observed_start_ns", "min"), last=("observed_end_ns", "max")
    )
    return ((frame["last"] - frame["first"]) / 1e6).to_dict()


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
    variants = ["v685_frozen", *CANDIDATES]
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
        for variant in variants:
            onef1b = (
                float(baseline_record["onef1b_ms"])
                if variant == "v685_frozen" else float(online.loc[(variant, iteration)])
            )
            predicted = {
                "entry": float(baseline_record["entry_ms"]),
                "onef1b": onef1b,
                "tail": float(baseline_record["tail_ms"]),
                "outer": float(baseline_record["outer_ms"]),
            }
            profiler = predicted["entry"] + onef1b + predicted["tail"]
            training = profiler + predicted["outer"]
            predicted_mfu = MFU_NUMERATOR / training
            actual_mfu = MFU_NUMERATOR / float(observed.actual_training_step_ms)
            rows.append({
                "variant": variant, "iteration": iteration,
                "prediction_mode": (
                    "cold_start" if variant in ["v685_frozen", "cold_start_phase_transfer"]
                    else "online_prefix_conditioned"
                ),
                "actual_onef1b_ms": actual["onef1b"], "predicted_onef1b_ms": onef1b,
                "onef1b_error_ms": onef1b - actual["onef1b"],
                "onef1b_APE_pct": 100 * abs(onef1b - actual["onef1b"]) / actual["onef1b"],
                "actual_profiler_ms": float(observed.actual_profiler_step_ms),
                "predicted_profiler_ms": profiler,
                "profiler_APE_pct": 100 * abs(profiler - float(observed.actual_profiler_step_ms)) / float(observed.actual_profiler_step_ms),
                "actual_training_ms": float(observed.actual_training_step_ms),
                "predicted_training_ms": training,
                "training_APE_pct": 100 * abs(training - float(observed.actual_training_step_ms)) / float(observed.actual_training_step_ms),
                "actual_MFU_pct_derived": actual_mfu,
                "predicted_MFU_pct": predicted_mfu,
                "MFU_error_pp": predicted_mfu - actual_mfu,
                "MFU_relative_APE_pct": 100 * abs(predicted_mfu - actual_mfu) / actual_mfu,
                "MFU_basis": "inherited FLOPs/peak and training clock; numerator not independently verified",
            })
            for phase in ["entry", "onef1b", "tail", "outer"]:
                ledger.append({
                    "variant": variant, "iteration": iteration, "phase": phase,
                    "actual_ms": actual[phase], "predicted_ms": predicted[phase],
                    "error_ms": predicted[phase] - actual[phase],
                    "prediction_changed_by_online_prefix": phase == "onef1b" and variant not in ["v685_frozen", "cold_start_phase_transfer"],
                })
    frame = pd.DataFrame(rows)
    metrics = frame.groupby(["variant", "prediction_mode"], sort=True).agg(
        iterations=("iteration", "size"),
        onef1b_MAPE_pct=("onef1b_APE_pct", "mean"),
        onef1b_bias_ms=("onef1b_error_ms", "mean"),
        profiler_MAPE_pct=("profiler_APE_pct", "mean"),
        training_MAPE_pct=("training_APE_pct", "mean"),
        MFU_relative_MAPE_pct=("MFU_relative_APE_pct", "mean"),
        MFU_bias_pp=("MFU_error_pp", "mean"),
    ).reset_index()
    return frame, pd.DataFrame(ledger), metrics


def target_direction_shapley(base, factors: pd.DataFrame) -> pd.DataFrame:
    base_nodes, _, _ = build(14, 3, base)
    base_value = float(envelope(base_nodes)["onef1b_ms"])
    rows = []
    primary = factors[factors.method.eq(PRIMARY)]
    for item in primary.itertuples(index=False):
        values = {}
        for label, forward, backward in [
            ("forward_only", item.forward_factor, 1.0),
            ("backward_only", 1.0, item.backward_factor),
            ("both", item.forward_factor, item.backward_factor),
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
                "scope": "two-factor allocation inside sealed online model; not observed-error attribution",
            })
    return pd.DataFrame(rows)


def _draw(out: Path, source_metrics: pd.DataFrame, target_metrics: pd.DataFrame,
          factors: pd.DataFrame, cutoffs: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "w37-t35a-online-prefix"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].barh(source_metrics.method, source_metrics.onef1b_MAPE_pct, color="#348ABD")
    axes[0].set_xlabel("source95/100 1F1B MAPE (%)")
    axes[0].set_title("Development confirmation")
    target = target_metrics[target_metrics.variant.isin(CANDIDATES)]
    axes[1].barh(target.variant, target.onef1b_MAPE_pct, color="#E24A33")
    axes[1].set_xlabel("target85/90/95/100 1F1B MAPE (%)")
    axes[1].set_title("Online and cold-start are separate")
    primary = factors[factors.method.eq(PRIMARY)]
    axes[2].plot(primary.iteration, primary.forward_factor, marker="o", label="F factor")
    axes[2].plot(primary.iteration, primary.backward_factor, marker="o", label="B factor")
    cuts = cutoffs[cutoffs.method.eq(PRIMARY)].set_index("iteration")
    for row in primary.itertuples(index=False):
        axes[2].annotate(f"{cuts.loc[row.iteration, 'prefix_available_fraction_pct']:.0f}%", (row.iteration, row.backward_factor), fontsize=7)
    axes[2].axhline(1, color="black", linewidth=.7)
    axes[2].set_title("Target prefix factors; labels=cutoff")
    axes[2].set_xlabel("iteration")
    axes[2].legend(fontsize=8)
    fig.suptitle("T35A directional online 1F1B nowcast")
    fig.text(.5, .012, "Target timing enters only as an explicit in-run prefix after the source estimator seal; results are development nowcasts, not cold-start predictions.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out / "online_prefix_nowcast.svg", metadata={"Date": None})
    fig.savefig(out / "online_prefix_nowcast.png", dpi=150, metadata={"Software": "w37-t35a"})
    plt.close(fig)


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    begin = perf_counter()
    assert plan["diagnostic"] == "online_prefix_runtime_audit"
    assert plan["diagnostic_access"] == "evaluator" and not plan["variants"]
    review = json.loads((ROOT / plan["resource_review_file"]).read_text())
    assert review["fit_iterations"] == FIT
    assert review["source_development_confirmation"] == SOURCE_DEVELOPMENT
    for item in review["training_semantics"]:
        assert sha(Path(item["path"])) == item["sha256"]

    source_phases = pd.read_csv(paths["source_pp_trace_events_60_100.csv"])
    source_apis = pd.read_csv(paths["source_pp_api_events_60_100.csv"])
    source_boundaries = pd.read_csv(paths["source256_profiler_entry_60_100.csv"])
    aligned, pairs = align_observations(source_phases, source_apis)
    source_base = RoleMappedReadinessCosts(aligned, pairs, FIT, pp=16, mb=4)
    source_bundle, source_edges = build_prediction_set(
        source_phases[source_phases.iteration.isin(FIT + SOURCE_DEVELOPMENT)],
        source_base, 16, 4, FIT + SOURCE_DEVELOPMENT,
    )
    source_features, source_factors, source_predictions, source_nodes = source_bundle

    mutated_source = source_phases[source_phases.iteration.isin(FIT + SOURCE_DEVELOPMENT)].copy()
    mutation = mutated_source.iteration.isin(SOURCE_DEVELOPMENT) & mutated_source.microbatch.ne(0)
    mutated_source.loc[mutation, "duration_ns"] = -999999999
    changed_bundle, changed_edges = build_prediction_set(
        mutated_source, source_base, 16, 4, FIT + SOURCE_DEVELOPMENT,
    )
    for original, changed in zip(source_bundle, changed_bundle):
        pd.testing.assert_frame_equal(original, changed, check_exact=True)
    pd.testing.assert_frame_equal(source_edges, changed_edges, check_exact=True)

    csv(out, "source_online_prefix_features.csv.gz", source_features)
    csv(out, "source_online_factors.csv", source_factors)
    csv(out, "source_online_iteration_predictions.csv", source_predictions)
    csv(out, "source_online_nodes.csv.gz", source_nodes)
    csv(out, "source_online_edges.csv.gz", source_edges)
    csv(out, "source_phase_readiness_parameters.csv", pd.DataFrame(source_base.parameter_rows))
    dump(out / "online_estimator_contract.json", {
        "status": "SEALED_SOURCE85_90_COSTS_AND_REGISTERED_PREFIX_RULES",
        "fit_iterations": FIT,
        "source_development_confirmation": SOURCE_DEVELOPMENT,
        "candidates": CANDIDATES,
        "primary_selected_before_target_prefix": PRIMARY,
        "phase_ownership": "multiply each existing F/B phase wall once; add no cost or wait",
        "target_timing_used": False,
        "formal_topology_replaced": False,
    })
    source_prediction_files = [
        "source_online_prefix_features.csv.gz", "source_online_factors.csv",
        "source_online_iteration_predictions.csv", "source_online_nodes.csv.gz",
        "source_online_edges.csv.gz", "source_phase_readiness_parameters.csv",
        "online_estimator_contract.json",
    ]
    dump(out / "source_online_prediction_seal.json", {
        "status": "SEALED_BEFORE_TARGET_PREFIX_ACCESS",
        "source_nonprefix_mutation_invariant": True,
        "target_timing_attached_before_seal": False,
        "files": [{"path": name, "sha256": sha(out / name)} for name in source_prediction_files],
    })

    source_actual = source_boundaries[
        source_boundaries.iteration.isin(FIT + SOURCE_DEVELOPMENT)
    ].set_index("iteration").phase_envelope_ms.to_dict()
    source_results, source_all_metrics = score_iterations(
        source_predictions, source_actual, "source_fit_or_development"
    )
    source_phase_results, source_phase_metrics = score_phase(source_nodes, source_phases)
    source_metrics = source_all_metrics.merge(
        source_phase_metrics[
            source_phase_metrics.region.eq("all") & source_phase_metrics.phase.eq("all")
        ][["method", "duration_MAE_ms", "duration_bias_ms"]], on="method", validate="one_to_one"
    )
    source_results["split"] = np.where(
        source_results.iteration.isin(FIT), "source_fit", "source_development_confirmation"
    )
    source_metrics_by_split = source_results.groupby(["method", "split"], sort=True).agg(
        iterations=("iteration", "size"), onef1b_MAPE_pct=("APE_pct", "mean"),
        onef1b_MAE_ms=("error_ms", lambda values: values.abs().mean()),
        onef1b_bias_ms=("error_ms", "mean"),
    ).reset_index()
    phase_dev = source_phase_results[source_phase_results.iteration.isin(SOURCE_DEVELOPMENT)]
    phase_dev_metrics = phase_dev.groupby("method", sort=True).agg(
        duration_MAE_ms=("duration_error_ms", lambda values: values.abs().mean()),
        duration_bias_ms=("duration_error_ms", "mean"),
    ).reset_index()
    dev = source_metrics_by_split[
        source_metrics_by_split.split.eq("source_development_confirmation")
    ].merge(phase_dev_metrics, on="method", validate="one_to_one")
    cold = dev.set_index("method").loc["cold_start_phase_transfer"]
    primary = dev.set_index("method").loc[PRIMARY]
    source_gate = bool(
        primary.onef1b_MAPE_pct < cold.onef1b_MAPE_pct
        and primary.duration_MAE_ms < cold.duration_MAE_ms
    )
    assert math.isclose(
        float(cold.onef1b_MAPE_pct),
        review["reproduction_reference"]["source95_100_cold_start_MAPE_pct"],
        abs_tol=1e-12,
    )
    csv(out, "source_online_iteration_results.csv", source_results)
    csv(out, "source_online_metrics.csv", dev)
    csv(out, "source_online_metrics_by_split.csv", source_metrics_by_split)
    csv(out, "source_online_phase_results.csv.gz", source_phase_results)
    csv(out, "source_online_phase_metrics.csv", source_phase_metrics)
    source_cutoffs = cutoff_scores(
        source_phases, source_predictions, 16, source_actual, "source256"
    )
    csv(out, "source_online_prefix_cutoffs.csv", source_cutoffs)
    assert source_gate

    capability, static_columns = static_capability_audit(paths)
    dump(out / "cold_start_runtime_capability_audit.json", capability)
    csv(out, "static_covariate_audit.csv", static_columns)
    for item in json.loads((out / "source_online_prediction_seal.json").read_text())["files"]:
        assert sha(out / item["path"]) == item["sha256"]

    target_phases = pd.read_csv(paths["target_phase_rank_events_60_100.csv"])
    target_base = RoleMappedReadinessCosts(aligned, pairs, FIT, pp=14, mb=3, role_transfer=True)
    target_bundle, target_edges = build_prediction_set(
        target_phases[target_phases.iteration.isin(TARGET_DEVELOPMENT)],
        target_base, 14, 3, TARGET_DEVELOPMENT,
    )
    target_features, target_factors, target_predictions, target_nodes = target_bundle
    mutated_target = target_phases[target_phases.iteration.isin(TARGET_DEVELOPMENT)].copy()
    mutated_target.loc[mutated_target.microbatch.ne(0), "duration_ns"] = -999999999
    changed_target_bundle, changed_target_edges = build_prediction_set(
        mutated_target, target_base, 14, 3, TARGET_DEVELOPMENT,
    )
    for original, changed in zip(target_bundle, changed_target_bundle):
        pd.testing.assert_frame_equal(original, changed, check_exact=True)
    pd.testing.assert_frame_equal(target_edges, changed_target_edges, check_exact=True)

    fit_ranges = source_factors[source_factors.iteration.isin(FIT)].groupby("method").agg(
        source_fit_forward_min=("forward_factor", "min"),
        source_fit_forward_max=("forward_factor", "max"),
        source_fit_backward_min=("backward_factor", "min"),
        source_fit_backward_max=("backward_factor", "max"),
    ).reset_index()
    target_factors = target_factors.merge(fit_ranges, on="method", validate="many_to_one")
    target_factors["forward_within_source_fit_range"] = (
        target_factors.forward_factor.between(
            target_factors.source_fit_forward_min, target_factors.source_fit_forward_max
        )
    )
    target_factors["backward_within_source_fit_range"] = (
        target_factors.backward_factor.between(
            target_factors.source_fit_backward_min, target_factors.source_fit_backward_max
        )
    )
    csv(out, "target_online_prefix_features.csv.gz", target_features)
    csv(out, "target_online_factors.csv", target_factors)
    csv(out, "target_online_iteration_predictions.csv", target_predictions)
    csv(out, "target_online_nodes.csv.gz", target_nodes)
    csv(out, "target_online_edges.csv.gz", target_edges)
    target_prediction_files = [
        "target_online_prefix_features.csv.gz", "target_online_factors.csv",
        "target_online_iteration_predictions.csv", "target_online_nodes.csv.gz",
        "target_online_edges.csv.gz",
    ]
    dump(out / "target_online_prediction_seal.json", {
        "status": "SEALED_PREFIX_CONDITIONED_PREDICTIONS_BEFORE_NONPREFIX_SCORING",
        "source_estimator_seal_sha256": sha(out / "source_online_prediction_seal.json"),
        "primary_selected_from_source_gate": PRIMARY,
        "target_nonprefix_mutation_invariant": True,
        "target_parameter_updates": 0,
        "formal_topology_replaced": False,
        "files": [{"path": name, "sha256": sha(out / name)} for name in target_prediction_files],
    })
    sealed_target_hashes = {
        item["path"]: item["sha256"]
        for item in json.loads((out / "target_online_prediction_seal.json").read_text())["files"]
    }

    target_actual = _actual_envelope(target_phases, TARGET_DEVELOPMENT)
    target_results, target_online_metrics = score_iterations(
        target_predictions, target_actual, "target_development"
    )
    target_phase_results, target_phase_metrics = score_phase(target_nodes, target_phases)
    baseline_record, _, formal_contract = baseline(paths)
    step_results, step_ledger, target_step_metrics = target_step_score(
        target_predictions, target_actual, paths, baseline_record
    )
    target_cutoffs = cutoff_scores(
        target_phases, target_predictions, 14, target_actual, "target224"
    )
    shapley = target_direction_shapley(target_base, target_factors)
    primary_nodes = target_nodes[target_nodes.method.eq(PRIMARY)]
    critical_rows = []
    for iteration in TARGET_DEVELOPMENT:
        frame = primary_nodes[primary_nodes.iteration.eq(iteration)].copy()
        region = frame.set_index("node_id").region.to_dict()
        for row in critical_ledger(frame, PRIMARY, f"target224_iteration{iteration}"):
            row["region"] = region[row["node_id"]]
            row["iteration"] = iteration
            critical_rows.append(row)

    csv(out, "target_online_iteration_results.csv", target_results)
    csv(out, "target_online_metrics.csv", target_online_metrics)
    csv(out, "target_online_phase_results.csv.gz", target_phase_results)
    csv(out, "target_online_phase_metrics.csv", target_phase_metrics)
    csv(out, "target_online_prefix_cutoffs.csv", target_cutoffs)
    csv(out, "target_online_direction_shapley.csv", shapley)
    csv(out, "target_online_critical_path_ledger.csv.gz", pd.DataFrame(critical_rows))
    csv(out, "target_step_iteration_results.csv", step_results)
    csv(out, "target_step_phase_ledger.csv", step_ledger)
    csv(out, "target_step_metrics.csv", target_step_metrics)
    dump(out / "formal_v685_prediction_contract.json", formal_contract)
    for name, expected in sealed_target_hashes.items():
        assert sha(out / name) == expected

    target_index = target_step_metrics.set_index("variant")
    assert math.isclose(
        float(target_index.loc["cold_start_phase_transfer", "onef1b_MAPE_pct"]),
        review["reproduction_reference"]["target_cold_start_phase_transfer_MAPE_pct"],
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(target_index.loc["v685_frozen", "onef1b_MAPE_pct"]),
        review["reproduction_reference"]["formal_v685_target_onef1b_MAPE_pct"],
        abs_tol=1e-12,
    )
    primary_target = target_index.loc[PRIMARY]
    target_development_improvement = (
        float(target_index.loc["v685_frozen", "onef1b_MAPE_pct"])
        - float(primary_target.onef1b_MAPE_pct)
    )
    _draw(out, dev, target_step_metrics, target_factors, target_cutoffs)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= review["resource"]["maximum_peak_RSS_bytes"]
    assert elapsed <= review["resource"]["target_analysis_seconds"]
    dump(out / "field_contract.json", {
        "cold_start": "No target timing is available. The v69 capability gate remains blocked by dynamic router tokens and independent runtime readiness.",
        "online_prefix": "Observed microbatch0 phase walls are explicit same-iteration conditions. The output estimates total 1F1B only after the recorded cutoff.",
        "phase_scale": "Separate robust F/B factors multiply the existing aggregate phase wall exactly once; compute, communication, overlap and unresolved runtime are not individually re-added.",
        "runtime_and_PP": "Source85/90 local runtime and PP readiness/completion remain unchanged.",
        "topology": "Pinned schedule topology is unchanged. This candidate does not replace the formal v684 topology lock.",
        "evaluation": "Source95/100 and target85/90/95/100 were exposed during development. Mutation gates prove feature isolation but do not make either set blind.",
    })
    dump(out / "diagnostic.json", {
        "status": "ONLINE_PREFIX_NOWCAST_EVALUATED_COLD_START_STILL_BLOCKED",
        "new_prediction": True,
        "online_prediction": True,
        "cold_start_prediction_improved": False,
        "source_gate_passed": source_gate,
        "primary_candidate": PRIMARY,
        "source95_100_control_MAPE_pct": float(cold.onef1b_MAPE_pct),
        "source95_100_primary_MAPE_pct": float(primary.onef1b_MAPE_pct),
        "source95_100_control_phase_MAE_ms": float(cold.duration_MAE_ms),
        "source95_100_primary_phase_MAE_ms": float(primary.duration_MAE_ms),
        "target_primary_online_onef1b_MAPE_pct": float(primary_target.onef1b_MAPE_pct),
        "target_formal_v685_cold_start_MAPE_pct": float(target_index.loc["v685_frozen", "onef1b_MAPE_pct"]),
        "target_phase_transfer_cold_start_MAPE_pct": float(target_index.loc["cold_start_phase_transfer", "onef1b_MAPE_pct"]),
        "target_online_minus_formal_improvement_pp": target_development_improvement,
        "target_primary_profiler_MAPE_pct": float(primary_target.profiler_MAPE_pct),
        "target_primary_training_MAPE_pct": float(primary_target.training_MAPE_pct),
        "target_primary_MFU_relative_MAPE_pct": float(primary_target.MFU_relative_MAPE_pct),
        "target_primary_prefix_mean_fraction_pct": float(target_cutoffs[target_cutoffs.method.eq(PRIMARY)].prefix_available_fraction_pct.mean()),
        "target_primary_all_factors_within_source_fit_range": bool(
            target_factors[target_factors.method.eq(PRIMARY)][
                ["forward_within_source_fit_range", "backward_within_source_fit_range"]
            ].all(axis=None)
        ),
        "target_parameter_updates": 0,
        "formal_topology_replaced": False,
        "source_prediction_seal_sha256": sha(out / "source_online_prediction_seal.json"),
        "target_prediction_seal_sha256": sha(out / "target_online_prediction_seal.json"),
        "peak_RSS_bytes": peak,
        "analysis_seconds": elapsed,
        "decision": "Report any error reduction only as a development online nowcast available after its prefix cutoff. The missing cold-start runtime inputs keep formal v685 unchanged.",
        "next": "Audit earlier, operationally useful target-prefix cutoffs and uncertainty coverage using source-only selection; do not promote an out-of-domain factor or target residual into the cold-start model.",
    })

#!/usr/bin/env python3
"""Create focused no-PP collective-bandwidth CSV views and audit them.

The expensive Trace/DeepEP/MTLink attribution is performed by
build_collective_bandwidth_no_pp.py.  This script only creates reproducible
subsets from its iteration-rank and type-summary CSV outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CORE_BEHAVIORS = (
    "ep_fwd_dispatch",
    "ep_fwd_combine",
    "ep_bwd_combine_backward_dispatch",
    "ep_bwd_dispatch_backward_combine",
    "cp_all_to_all",
    "dp_grad_reduce_scatter",
    "dp_param_allgather",
    "expert_dp_grad_reduce_scatter",
    "expert_dp_param_allgather",
)

# The exact six roles exposed by case_16gpu_pp2.  This view is intentionally
# narrower than CORE_BEHAVIORS so the two runs can be compared directly.
CASE16_COMPATIBLE_BEHAVIORS = (
    "ep_fwd_dispatch",
    "ep_fwd_combine",
    "ep_bwd_combine_backward_dispatch",
    "ep_bwd_dispatch_backward_combine",
    "dp_grad_reduce_scatter",
    "dp_param_allgather",
)

EXPECTED_EVENTS_PER_CELL = {
    "dp_grad_reduce_scatter": 1,
    "dp_param_allgather": 2,
    "expert_dp_grad_reduce_scatter": 1,
    "expert_dp_param_allgather": 2,
}

PP_DESCRIPTIONS = {
    "PIPELINE_MODEL_PARALLEL_GROUP",
    "MODEL_PARALLEL_GROUP",
    "EMBEDDING_GROUP",
    "POSITION_EMBEDDING_GROUP",
}
EXCLUDED_COLLECTIVES = {"send", "recv", "wait"}


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    default_output = case_root / "results" / "collective_bw_no_pp"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def write_subset(
    cells: pd.DataFrame,
    summary: pd.DataFrame,
    behaviors: tuple[str, ...],
    cell_output: Path,
    summary_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = {behavior: index for index, behavior in enumerate(behaviors)}
    selected_cells = cells[cells["behavior"].isin(behaviors)].copy()
    selected_summary = summary[summary["behavior"].isin(behaviors)].copy()
    selected_cells["_behavior_order"] = selected_cells["behavior"].map(order)
    selected_summary["_behavior_order"] = selected_summary["behavior"].map(order)
    selected_cells = selected_cells.sort_values(
        ["_behavior_order", "iteration", "rank"]
    ).drop(columns="_behavior_order")
    selected_summary = selected_summary.sort_values("_behavior_order").drop(
        columns="_behavior_order"
    )
    selected_cells.to_csv(cell_output, index=False)
    selected_summary.to_csv(summary_output, index=False)
    return selected_cells, selected_summary


def build_coverage(
    core_cells: pd.DataFrame, core_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in core_summary.itertuples(index=False):
        behavior = str(item.behavior)
        expected_per_cell = EXPECTED_EVENTS_PER_CELL.get(behavior)
        expected_event_count = (
            int(item.profiled_iteration_count)
            * int(item.rank_count)
            * expected_per_cell
            if expected_per_cell is not None
            else pd.NA
        )
        rows.append(
            {
                "behavior": behavior,
                "source": item.source,
                "collective": item.collective,
                "group_size": int(item.group_size),
                "profiled_iteration_count": int(item.profiled_iteration_count),
                "rank_count": int(item.rank_count),
                "cell_count": int(item.cell_count),
                "event_count": int(item.event_count),
                "expected_events_per_iter_rank": (
                    expected_per_cell
                    if expected_per_cell is not None
                    else pd.NA
                ),
                "expected_event_count": expected_event_count,
                "event_count_delta": (
                    int(item.event_count) - int(expected_event_count)
                    if expected_per_cell is not None
                    else pd.NA
                ),
                "zero_event_iter_rank_cell_count": (
                    int(item.profiled_iteration_count)
                    * int(item.rank_count)
                    - int(item.cell_count)
                ),
                "partial_event_iter_rank_cell_count": (
                    int(
                        (
                            core_cells.loc[
                                core_cells["behavior"].eq(behavior),
                                "event_count",
                            ]
                            < expected_per_cell
                        ).sum()
                    )
                    if expected_per_cell is not None
                    else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    cells_path = output_dir / "collective_iter_rank_bandwidth_no_pp.csv"
    summary_path = output_dir / "collective_type_bandwidth_summary_no_pp.csv"
    cells = pd.read_csv(cells_path)
    summary = pd.read_csv(summary_path)
    require_columns(
        cells,
        {
            "behavior",
            "iteration",
            "rank",
            "collective",
            "pg_description",
            "event_count",
            "duration_ns_sum",
        },
        cells_path.name,
    )
    require_columns(
        summary,
        {
            "behavior",
            "source",
            "collective",
            "pg_description",
            "group_size",
            "cell_count",
            "profiled_iteration_count",
            "rank_count",
            "event_count",
        },
        summary_path.name,
    )

    pp_rows = cells[
        cells["pg_description"].isin(PP_DESCRIPTIONS)
        | cells["pg_description"].str.contains("PIPELINE", case=False, na=False)
        | cells["collective"].str.lower().isin(EXCLUDED_COLLECTIVES)
    ]
    unknown_core = sorted(set(CORE_BEHAVIORS) - set(cells["behavior"]))
    if len(pp_rows) or unknown_core:
        raise ValueError(
            f"invalid source: pp_rows={len(pp_rows)}, "
            f"missing_core_behaviors={unknown_core}"
        )

    core_cells, core_summary = write_subset(
        cells,
        summary,
        CORE_BEHAVIORS,
        output_dir / "collective_iter_rank_bandwidth_core_no_pp.csv",
        output_dir / "collective_core_type_bandwidth_summary_no_pp.csv",
    )
    compatible_cells, compatible_summary = write_subset(
        cells,
        summary,
        CASE16_COMPATIBLE_BEHAVIORS,
        output_dir
        / "collective_iter_rank_bandwidth_case16_compatible_no_pp.csv",
        output_dir
        / "collective_case16_compatible_type_bandwidth_summary_no_pp.csv",
    )
    coverage = build_coverage(core_cells, core_summary)
    coverage.to_csv(
        output_dir / "collective_core_coverage_no_pp.csv", index=False
    )

    audit = {
        "status": "PASS",
        "pp_included": False,
        "source_cell_count": int(len(cells)),
        "source_behavior_count": int(cells["behavior"].nunique()),
        "core_behaviors": list(CORE_BEHAVIORS),
        "core_behavior_count": int(core_cells["behavior"].nunique()),
        "core_cell_count": int(len(core_cells)),
        "core_event_count": int(core_cells["event_count"].sum()),
        "case16_compatible_behaviors": list(CASE16_COMPATIBLE_BEHAVIORS),
        "case16_compatible_behavior_count": int(
            compatible_cells["behavior"].nunique()
        ),
        "case16_compatible_cell_count": int(len(compatible_cells)),
        "case16_compatible_event_count": int(
            compatible_cells["event_count"].sum()
        ),
        "iterations": sorted(int(x) for x in core_cells["iteration"].unique()),
        "rank_count": int(core_cells["rank"].nunique()),
        "known_profiler_event_deltas": {
            str(row.behavior): int(row.event_count_delta)
            for row in coverage.itertuples(index=False)
            if not pd.isna(row.event_count_delta)
        },
        "outputs": [
            "collective_iter_rank_bandwidth_core_no_pp.csv",
            "collective_core_type_bandwidth_summary_no_pp.csv",
            "collective_iter_rank_bandwidth_case16_compatible_no_pp.csv",
            "collective_case16_compatible_type_bandwidth_summary_no_pp.csv",
            "collective_core_coverage_no_pp.csv",
        ],
    }
    if (
        audit["core_behavior_count"] != len(CORE_BEHAVIORS)
        or audit["case16_compatible_behavior_count"]
        != len(CASE16_COMPATIBLE_BEHAVIORS)
        or audit["rank_count"] != 256
    ):
        audit["status"] = "FAIL"
    (output_dir / "collective_bandwidth_views_audit_no_pp.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

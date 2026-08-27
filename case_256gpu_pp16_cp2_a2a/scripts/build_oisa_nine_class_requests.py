#!/usr/bin/env python3
"""Freeze nine representative iteration-55 collective requests for OISA.

Each representative is a real, fully aligned collective call.  Rank release
offsets come from the unified Trace clock.  Selection is deterministic and
chooses the call closest to the behavior medians in payload, release span and
observed group service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


BEHAVIORS = {
    "ep_fwd_dispatch": ("ep_group", "all_to_all", "expected_tx_bytes"),
    "ep_fwd_combine": ("ep_group", "all_to_all", "expected_tx_bytes"),
    "ep_bwd_combine_backward_dispatch": (
        "ep_group", "all_to_all", "expected_tx_bytes"
    ),
    "ep_bwd_dispatch_backward_combine": (
        "ep_group", "all_to_all", "expected_tx_bytes"
    ),
    "cp_all_to_all": ("cp_group", "all_to_all", "expected_tx_bytes"),
    "dp_grad_reduce_scatter": ("pp_stage", "reduce_scatter", "logical_input_bytes"),
    "dp_param_allgather": ("pp_stage", "all_gather", "logical_input_bytes"),
    "expert_dp_grad_reduce_scatter": (
        "expert_dp_group", "reduce_scatter", "logical_input_bytes"
    ),
    "expert_dp_param_allgather": (
        "expert_dp_group", "all_gather", "logical_input_bytes"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text)


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    fabric_case = Path("/home/zjb/Desktop/fabric-data-analysis/case_256gpu_pp16_cp2_a2a")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=55)
    parser.add_argument(
        "--events",
        type=Path,
        default=fabric_case
        / "results/collective_bw_no_pp/collective_events_no_pp_pre_mtlink.parquet",
    )
    parser.add_argument(
        "--rank-topology",
        type=Path,
        default=fabric_case / "results/topology/rank_topology.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=case_root / "results/oisa_s5000_256gpu_nine_class/inputs",
    )
    parser.add_argument("--topology-id", default="s5000_256gpu_32host_1spine")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns = [
        "behavior", "iteration", "rank", "group_size", "start_ns", "end_ns",
        "duration_ns", "logical_input_bytes", "expected_tx_bytes",
    ]
    events = pd.read_parquet(args.events, columns=columns)
    events = events[
        events["iteration"].eq(args.iteration)
        & events["behavior"].isin(BEHAVIORS)
    ].copy()
    topology = pd.read_csv(
        args.rank_topology,
        usecols=["rank", "pp_stage", "cp_group", "ep_group", "expert_dp_group"],
    )
    events = events.merge(topology, on="rank", validate="many_to_one")
    events = events.sort_values(
        ["behavior", "rank", "start_ns", "end_ns"]
    ).reset_index(drop=True)
    events["slot_index"] = events.groupby(["behavior", "rank"]).cumcount()

    output = args.output
    requests_dir = output / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, object]] = []
    request_paths: list[str] = []

    for behavior, (group_column, op, payload_column) in BEHAVIORS.items():
        behavior_events = events[events["behavior"].eq(behavior)].copy()
        behavior_events["sync_group"] = behavior_events[group_column].astype("int64")
        keys = ["sync_group", "slot_index"]
        candidates = (
            behavior_events.groupby(keys, as_index=False)
            .agg(
                participant_count=("rank", "nunique"),
                group_size=("group_size", "max"),
                pp_stage=("pp_stage", "first"),
                pp_stage_count=("pp_stage", "nunique"),
                first_start_ns=("start_ns", "min"),
                ready_ns=("start_ns", "max"),
                group_end_ns=("end_ns", "max"),
                logical_input_bytes=("logical_input_bytes", "median"),
                expected_tx_bytes=("expected_tx_bytes", "median"),
            )
        )
        candidates = candidates[
            candidates["participant_count"].eq(candidates["group_size"])
            & candidates[payload_column].gt(0)
        ].copy()
        if candidates.empty:
            raise ValueError(f"no fully aligned positive-payload call for {behavior}")
        candidates["arrival_span_ns"] = candidates["ready_ns"] - candidates["first_start_ns"]
        candidates["observed_service_ns"] = candidates["group_end_ns"] - candidates["ready_ns"]
        candidates["payload_bytes"] = candidates[payload_column].round().astype("int64")
        if (candidates[["arrival_span_ns", "observed_service_ns", "payload_bytes"]] < 0).any().any():
            raise ValueError(f"invalid candidate timing for {behavior}")

        medians = {
            column: float(candidates[column].median())
            for column in ("payload_bytes", "arrival_span_ns", "observed_service_ns")
        }
        candidates["selection_score"] = 0.0
        for column, median in medians.items():
            candidates["selection_score"] += candidates[column].map(
                lambda value, center=median: abs(
                    math.log((float(value) + 1.0) / (center + 1.0))
                )
            )
        selected = candidates.sort_values(
            ["selection_score", "sync_group", "slot_index"]
        ).iloc[0]
        members = behavior_events[
            behavior_events["sync_group"].eq(int(selected["sync_group"]))
            & behavior_events["slot_index"].eq(int(selected["slot_index"]))
        ].sort_values("rank")
        if members["rank"].nunique() != int(selected["group_size"]):
            raise ValueError(f"selected member set is incomplete for {behavior}")
        origin = int(members["start_ns"].min())
        offsets = {
            str(int(row.rank)): int(row.start_ns) - origin
            for row in members.itertuples(index=False)
        }
        request_id = (
            f"iter{args.iteration}_{_safe(behavior)}_"
            f"g{int(selected['sync_group'])}_slot{int(selected['slot_index'])}"
        )
        request = {
            "request_id": request_id,
            "iteration": args.iteration,
            "kind": behavior,
            "round": int(selected["slot_index"]),
            "group_key": f"{behavior}:group={int(selected['sync_group'])}",
            "op": op,
            "group_ranks": [int(rank) for rank in members["rank"]],
            "payload_bytes": int(selected["payload_bytes"]),
            "rank_release_offsets_ns": offsets,
            "topology_id": args.topology_id,
            "traffic_matrix_id": None,
            "n_channels": 1,
            "input_time_origin_ns": origin,
        }
        request_path = requests_dir / f"{request_id}.json"
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        request_paths.append(str(request_path.relative_to(output)))
        selected_rows.append({
            "behavior": behavior,
            "op": op,
            "request_id": request_id,
            "sync_group": int(selected["sync_group"]),
            "slot_index": int(selected["slot_index"]),
            "pp_stage": int(selected["pp_stage"]),
            "group_size": int(selected["group_size"]),
            "group_ranks": ";".join(str(rank) for rank in request["group_ranks"]),
            "payload_source": payload_column,
            "payload_bytes": int(selected["payload_bytes"]),
            "arrival_span_ns": int(selected["arrival_span_ns"]),
            "observed_service_ns": int(selected["observed_service_ns"]),
            "selection_score": float(selected["selection_score"]),
            "candidate_count": int(len(candidates)),
        })

    selection = pd.DataFrame(selected_rows).sort_values("behavior")
    selection.to_csv(output / "representative_calls.csv", index=False)
    manifest = {
        "schema": "mfu-oisa-nine-class-input-v1",
        "iteration": args.iteration,
        "topology_id": args.topology_id,
        "selection": (
            "fully aligned real call nearest behavior medians in payload, "
            "rank-release span and observed service"
        ),
        "payload_semantics": {
            "all_to_all": "median per-rank expected network TX bytes",
            "reduce_scatter": "median per-rank input tensor bytes",
            "all_gather": "median per-rank shard bytes",
        },
        "request_count": len(request_paths),
        "requests": request_paths,
        "inputs": {
            "events": {"path": str(args.events), "sha256": _sha256(args.events)},
            "rank_topology": {
                "path": str(args.rank_topology),
                "sha256": _sha256(args.rank_topology),
            },
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build rank, pipeline-boundary and layer topology tables for this run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-host-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    with args.rank_host_map.open(newline="", encoding="utf-8") as handle:
        rank_host = {
            int(row["rank"]): row["host"] for row in csv.DictReader(handle)
        }
    if set(rank_host) != set(range(256)):
        raise SystemExit("rank-host map must contain exactly ranks 0..255")

    host_ranks: dict[str, list[int]] = {}
    for rank, host in rank_host.items():
        host_ranks.setdefault(host, []).append(rank)
    if any(len(ranks) != 8 for ranks in host_ranks.values()):
        raise SystemExit("each host must contain exactly 8 ranks")
    local_rank = {
        rank: index
        for ranks in host_ranks.values()
        for index, rank in enumerate(sorted(ranks))
    }

    rank_rows = []
    for rank in range(256):
        within_stage = rank % 16
        rank_rows.append(
            {
                "rank": rank,
                "host": rank_host[rank],
                "local_rank": local_rank[rank],
                "gpu_id": local_rank[rank],
                "pp_stage": rank // 16,
                "pp_lane": within_stage,
                "cp_group": (rank // 2),
                "cp_rank": within_stage % 2,
                "dp_group": (rank // 16) * 2 + (within_stage % 2),
                "dp_rank": within_stage // 2,
                "ep_group": rank // 8,
                "ep_rank": rank % 8,
                "expert_dp_group": rank % 8 + (rank // 16) * 8,
                "expert_dp_rank": (within_stage // 8),
            }
        )

    edge_rows = []
    for boundary in range(15):
        for lane in range(16):
            src = boundary * 16 + lane
            dst = (boundary + 1) * 16 + lane
            edge_rows.append(
                {
                    "pp_boundary": boundary,
                    "lane_id": lane,
                    "src_stage": boundary,
                    "dst_stage": boundary + 1,
                    "src_rank": src,
                    "dst_rank": dst,
                    "src_host": rank_host[src],
                    "dst_host": rank_host[dst],
                }
            )

    stage_counts = [2] + [4] * 14 + [2]
    layer_rows = []
    layer = 0
    for stage, count in enumerate(stage_counts):
        for stage_local_layer in range(count):
            layer_rows.append(
                {
                    "layer_id": layer,
                    "pp_stage": stage,
                    "stage_local_layer": stage_local_layer,
                    "stage_layer_count": count,
                    "virtual_chunk": 0,
                }
            )
            layer += 1
    if layer != 60:
        raise SystemExit("layer layout must total 60")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        args.output_dir / "rank_topology.csv",
        list(rank_rows[0]),
        rank_rows,
    )
    write_rows(
        args.output_dir / "pp_boundary_lanes.csv",
        list(edge_rows[0]),
        edge_rows,
    )
    write_rows(
        args.output_dir / "layer_stage_map.csv",
        list(layer_rows[0]),
        layer_rows,
    )
    print(
        f"wrote {len(rank_rows)} ranks, {len(edge_rows)} PP edges, "
        f"{len(layer_rows)} layers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the documented GPU-slot to NIC mapping used by the 256-GPU case.

The raw 256-GPU corpus contains NIC counters but no per-rank dashboard set.
This case therefore carries over the slot mapping that was raw-point validated
for all 224 ranks in the immediately preceding run on the same machine family.
The output makes that evidence level explicit and must not be described as a
fresh 256-rank dashboard validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GPU_TO_NIC = {
    0: "mlx5_0",
    1: "mlx5_1",
    2: "mlx5_2",
    3: "mlx5_3",
    4: "mlx5_6",
    5: "mlx5_7",
    6: "mlx5_8",
    7: "mlx5_9",
}


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    output_dir = case_root / "results" / "dashboard_reference"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank-topology",
        type=Path,
        default=case_root / "results" / "topology" / "rank_topology.csv",
    )
    parser.add_argument(
        "--nic-manifest",
        type=Path,
        default=case_root / "results" / "readiness" / "nic_files.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=output_dir
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topology = pd.read_csv(args.rank_topology)
    manifest = pd.read_csv(args.nic_manifest)
    entities = manifest[["host", "device"]].drop_duplicates()
    if (
        len(topology) != 256
        or topology["rank"].nunique() != 256
        or topology["host"].nunique() != 32
        or len(entities) != 256
    ):
        raise ValueError("incomplete 256-rank topology or NIC entity grid")

    mapping = topology[["rank", "host", "local_rank", "gpu_id"]].copy()
    mapping["nic_device"] = mapping["gpu_id"].map(GPU_TO_NIC)
    mapping["expected_nic_device"] = mapping["nic_device"]
    mapping["mapping_status"] = "CARRYOVER_VALIDATED_SLOT_MAP"
    mapping["dashboard_profiler_step"] = "not-audited-in-256-case"
    mapping["nic_metric"] = "xmit_bytes/recv_bytes delta"
    mapping["raw_nic_point_match_count"] = 0
    mapping["nic_point_count"] = 0
    mapping["raw_nic_max_abs_value_difference"] = 0.0
    mapping["relative_path"] = (
        "carryover:case_224gpu_pp14_cp2_a2a/"
        "results/dashboard_reference/dashboard_rank_nic_mapping.csv"
    )
    mapping_keys = set(zip(mapping["host"], mapping["nic_device"]))
    manifest_keys = set(zip(entities["host"], entities["device"]))
    if (
        mapping["nic_device"].isna().any()
        or mapping_keys != manifest_keys
        or not mapping.groupby("host")["nic_device"].nunique().eq(8).all()
    ):
        raise ValueError("carried slot mapping does not close against NIC files")

    stage = mapping[["rank", "nic_device"]].copy()
    stage["nic_point_count_in_edp_windows"] = 0
    stage["nic_xmit_gbps_median_in_edp_windows"] = float("nan")
    stage["nic_xmit_gbps_p95_in_edp_windows"] = float("nan")
    stage["nic_xmit_gbps_max_in_edp_windows"] = float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output_dir / "dashboard_rank_nic_mapping.csv", index=False)
    stage.to_csv(
        args.output_dir / "dashboard_iteration5_nic_stage_summary.csv",
        index=False,
    )
    validation = {
        "status": "PASS_WITH_CARRYOVER_MAPPING",
        "schema_version": "256gpu-rank-nic-mapping-v1",
        "rank_count": int(len(mapping)),
        "host_count": int(mapping["host"].nunique()),
        "physical_nic_entity_count": int(len(mapping_keys)),
        "mapping": GPU_TO_NIC,
        "fresh_256_dashboard_audit": False,
        "evidence": [
            "The same GPU-slot mapping was raw-point validated across all 224 ranks in case_224gpu_pp14_cp2_a2a.",
            "The 256-GPU raw corpus exposes the same eight NIC device names on every host.",
        ],
        "limitation": (
            "This closes the analysis entity grid but is not proof of current-run "
            "PCIe/NUMA/rail affinity or exclusive EDP traffic ownership."
        ),
    }
    (args.output_dir / "dashboard_nic_reference_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

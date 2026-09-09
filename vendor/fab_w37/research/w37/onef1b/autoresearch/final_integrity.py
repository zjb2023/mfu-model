"""Read-only integrity audit for the accepted W37 1F1B evidence package."""
from __future__ import annotations

import json
import os
import re
import resource
from pathlib import Path
from time import perf_counter

import pandas as pd


LOCAL_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def local_links(document: Path, root: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(document.read_text().splitlines(), 1):
        for raw_target in LOCAL_LINK.findall(line):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = target.split("#", 1)[0]
            resolved = (document.parent / path_part).resolve()
            rows.append({
                "document": document.name,
                "line": line_number,
                "target": target,
                "resolved_path": str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved),
                "inside_worktree": resolved.is_relative_to(root),
                "exists": resolved.exists(),
                "is_file": resolved.is_file(),
                "is_directory": resolved.is_dir(),
            })
    return rows


def boundary_rows(document: Path, phrases: list[str]) -> list[dict]:
    text = document.read_text()
    return [{"document": document.name, "required_phrase": phrase, "present": phrase in text}
            for phrase in phrases]


def reproduce_rows(index: pd.DataFrame, references: dict[str, dict], root: Path) -> list[dict]:
    from smoke_worker import sha

    rows = []
    for record in index.to_dict("records"):
        name = str(record["wave"]).lower()
        ref = references[name]
        checks = {
            "plan_path": str(Path(ref["plan_path"]).relative_to(root)) == record["plan_path"],
            "plan_sha256": sha(Path(ref["plan_path"])) == record["plan_sha256"] == ref["plan_sha256"],
            "wave_spec_path": str(Path(ref["wave_spec_path"]).relative_to(root)) == record["wave_spec_path"],
            "wave_spec_sha256": sha(Path(ref["wave_spec_path"])) == record["wave_spec_sha256"] == ref["wave_spec_sha256"],
            "official_run_link": str(Path(ref["official_link_path"]).relative_to(root)) == record["official_run_link"],
            "manifest_sha256": ref["manifest_sha256"] == record["diagnose_manifest_sha256"],
            "pipeline_command_path": str(Path(ref["pipeline_command_path"]).relative_to(root)) == record["pipeline_command_path"],
            "pipeline_command_sha256": sha(Path(ref["pipeline_command_path"])) == record["pipeline_command_sha256"] == ref["pipeline_command_sha256"],
        }
        rows.append({"wave": record["wave"], **{key + "_match": value for key, value in checks.items()},
                     "all_match": all(checks.values())})
    return rows


def diagnose(out: Path, paths: dict[str, Path], plan: dict) -> None:
    from guards import checked_stage
    from smoke_worker import dump, sha
    from worker import csv

    begin = perf_counter()
    root = Path(__file__).resolve().parents[4]
    references = {ref["name"]: ref for ref in plan["sealed_diagnostic_stages"]}
    assert list(references) == [f"t{n}" for n in range(35, 42)]

    evidence = []
    for name, ref in references.items():
        stage = Path(ref["stage_root"])
        acceptance_path = Path(ref["acceptance_path"])
        command_path = Path(ref["pipeline_command_path"])
        plan_path = Path(ref["plan_path"])
        wave_path = Path(ref["wave_spec_path"])
        official_link = Path(ref["official_link_path"])
        checked_stage(stage)
        acceptance = json.loads(acceptance_path.read_text())
        command = json.loads(command_path.read_text())
        command_text = " ".join(map(str, command["argv"]))
        checks = {
            "manifest": sha(stage / "run_manifest.json") == ref["manifest_sha256"],
            "acceptance": sha(acceptance_path) == ref["acceptance_sha256"],
            "acceptance_status": acceptance["status"] == ref["expected_status"],
            "acceptance_manifest": acceptance["diagnose_manifest_sha256"] == ref["manifest_sha256"],
            "pipeline_command": sha(command_path) == ref["pipeline_command_sha256"],
            "pipeline_spec": str(wave_path) in command_text,
            "plan": sha(plan_path) == ref["plan_sha256"],
            "wave_spec": sha(wave_path) == ref["wave_spec_sha256"],
            "official_symlink": official_link.is_symlink(),
            "official_target_text": os.readlink(official_link) == ref["official_link_target"],
            "official_target_run": official_link.resolve() == stage.parent.resolve(),
        }
        assert all(checks.values()), (name, checks)
        evidence.append({
            "wave": name.upper(), "run_id": acceptance["run_id"], "acceptance_status": acceptance["status"],
            "official_link": str(official_link.relative_to(root)), "official_target": os.readlink(official_link),
            **{key + "_match": value for key, value in checks.items()}, "all_match": all(checks.values()),
        })

    index = pd.read_csv(Path(plan["t41_reproduction_index_path"]))
    assert list(index["wave"]) == [f"T{n}" for n in range(35, 41)]
    reproduction = reproduce_rows(index, references, root)
    assert all(row["all_match"] for row in reproduction)

    document_rows, claims = [], []
    phrases = plan["required_boundary_phrases"]
    for document_path in map(Path, plan["documents"]):
        document_rows.extend(local_links(document_path, root))
        claims.extend(boundary_rows(document_path, phrases[document_path.name]))
    assert len(document_rows) == plan["expected_document_local_links"]
    assert all(row["inside_worktree"] and row["exists"] for row in document_rows)
    assert all(row["present"] for row in claims)

    deliverables = []
    for item in plan["required_deliverables"]:
        path = Path(item["path"])
        checks = {
            "inside_worktree": path.resolve().is_relative_to(root),
            "exists": path.exists(),
            "size": path.stat().st_size == item["size_bytes"],
        }
        assert all(checks.values()), (item["key"], checks)
        deliverables.append({
            "key": item["key"], "category": item["category"], "display_path": item["display_path"],
            "sha256": item["sha256"], "size_bytes": item["size_bytes"],
            "preflight_sha256_verified": True, **checks, "all_match": all(checks.values()),
        })

    csv(out, "evidence_chain_audit.csv", pd.DataFrame(evidence))
    csv(out, "reproduction_recompute.csv", pd.DataFrame(reproduction))
    csv(out, "documentation_link_audit.csv", pd.DataFrame(document_rows))
    csv(out, "boundary_claim_audit.csv", pd.DataFrame(claims))
    csv(out, "deliverable_inventory.csv", pd.DataFrame(deliverables))
    dump(out / "final_integrity_contract.json", {
        "status": "CURRENT_EVIDENCE_PACKAGE_COMPLETE_TARGET_METHOD_SEARCH_FROZEN",
        "official_waves_verified": [row["wave"] for row in evidence],
        "document_local_links_verified": len(document_rows),
        "required_deliverables_verified": len(deliverables),
        "t41_reproduction_entries_recomputed": len(reproduction),
        "formal_cold_start": "v685",
        "same_iteration_development": "T35",
        "target_initialized_sequential_development": "T38",
        "resume_conditions": json.loads(paths["t41_stop_resume"].read_text())["resume_conditions"],
        "new_prediction": False, "new_fit": False, "new_target_scoring": False,
        "parameter_updates": 0, "added_edges_or_waits": 0, "formal_topology_replaced": False,
        "unique_next": plan["unique_next_on_pass"],
    })
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = perf_counter() - begin
    assert peak <= plan["resource"]["maximum_peak_RSS_bytes"]
    assert elapsed <= plan["resource"]["target_analysis_seconds"]
    dump(out / "diagnostic.json", {
        "status": "FINAL_EVIDENCE_CHAIN_INTEGRITY_PASS",
        "new_prediction": False, "new_fit": False, "new_target_scoring": False,
        "target_method_search_frozen": True,
        "official_waves_verified": len(evidence), "document_local_links_verified": len(document_rows),
        "missing_document_links": 0, "boundary_claim_checks": len(claims),
        "required_deliverables_verified": len(deliverables),
        "t41_reproduction_entries_recomputed": len(reproduction),
        "hash_or_link_failures": 0, "source_parameter_updates": 0, "target_parameter_updates": 0,
        "added_edges_or_waits": 0, "formal_topology_replaced": False,
        "peak_RSS_bytes": peak, "analysis_seconds": elapsed,
        "decision": "The current T35-T41 evidence package is complete and navigable. Preserve v685/T35/T38 claim boundaries and do not reopen target method selection without a recorded resume condition.",
        "next": plan["unique_next_on_pass"],
    })

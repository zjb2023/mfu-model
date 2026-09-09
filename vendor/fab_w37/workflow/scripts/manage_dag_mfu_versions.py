#!/usr/bin/env python3
"""Validate and snapshot the DAG MFU version dependency graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import tomllib
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "workflow/config/dag_mfu_versions.toml"
DEFAULT_OUTPUT = REPO / "case_224gpu_pp14_cp2_a2a/results/mfu_accuracy_comparison_2026w36/dag_mfu_version_pipeline"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def topological_order(versions: list[dict[str, Any]]) -> list[str]:
    by_id = {item["id"]: item for item in versions}
    if len(by_id) != len(versions):
        raise ValueError("duplicate version id")
    external = {
        parent
        for item in versions
        for parent in item.get("external_parents", [])
    }
    missing = {
        parent
        for item in versions
        for parent in item.get("parents", [])
        if parent not in by_id and parent not in external
    }
    if missing:
        raise ValueError(f"undeclared parents: {sorted(missing)}")
    indegree = {key: 0 for key in by_id}
    children = {key: [] for key in by_id}
    for item in versions:
        for parent in item.get("parents", []):
            if parent not in by_id:
                continue
            indegree[item["id"]] += 1
            children[parent].append(item["id"])
    ready = [item["id"] for item in versions if indegree[item["id"]] == 0]
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(ordered) != len(versions):
        raise ValueError("cycle in DAG MFU version graph")
    return ordered


def inspect_file(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256(path) if exists else None,
    }


def fingerprint(parts: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        f"{item.get('path')}\t{item.get('sha256')}"
        for item in sorted(parts, key=lambda item: str(item.get("path")))
        if item.get("exists") and item.get("sha256")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def portable_content_fingerprint(parts: list[dict[str, Any]]) -> str:
    """Hash logical roles and content, without repository-specific absolute paths."""
    canonical = "\n".join(
        f"{item.get('role')}\t{item.get('sha256')}"
        for item in sorted(parts, key=lambda item: str(item.get("role")))
        if item.get("exists") and item.get("sha256")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_version_graph_html(
    rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> str:
    mainline = [row for row in rows if row["track"] == "predictive_mainline"]
    development = [row for row in rows if row["track"] == "target_assisted_development"]

    def card(row: dict[str, Any], kind: str) -> str:
        fingerprint = row.get(
            "candidate_content_fingerprint_sha256",
            row.get("portable_content_fingerprint_sha256", ""),
        )
        status = row.get("status") or row.get("seal_status") or "STRUCTURE_ONLY"
        return (
            f'<article class="node {kind}"><div class="tag">{html.escape(kind.upper())}</div>'
            f'<h3>{html.escape(row["id"])} · {html.escape(row["slug"])}</h3>'
            f'<p>{html.escape(row["claim"])}</p>'
            f'<div class="meta">{html.escape(str(status))}</div>'
            f'<code>{html.escape(fingerprint[:16])}…</code></article>'
        )

    main_cards = '<span class="arrow">→</span>'.join(card(row, "mainline") for row in mainline)
    branch_cards = "".join(card(row, "development") for row in development)
    candidate_cards = "".join(card(row, "candidate") for row in candidate_rows)
    dependency_rows = []
    for row in rows:
        code_roles = ", ".join(
            role for role in ("builder", "finalizer", "sealer", "evaluator")
            if role in row.get("files", {})
        )
        config_name = Path(
            row.get("files", {}).get("config", {}).get("path", "not-recorded")
        ).name
        anchor_name = Path(row.get("anchor", {}).get("path", "not-recorded")).name
        dependency_rows.append(
            "<tr>"
            f"<td><b>{html.escape(row['id'])}</b><br><span>{html.escape(row['track'])}</span></td>"
            f"<td>{html.escape(', '.join(row.get('parents', [])) or 'root')}</td>"
            f"<td><code>{html.escape(config_name)}</code></td>"
            f"<td>{html.escape(code_roles)}</td>"
            f"<td><code>{html.escape(anchor_name)}</code></td>"
            f"<td><code>{html.escape(row.get('portable_content_fingerprint_sha256', '')[:16])}…</code></td>"
            "</tr>"
        )
    for row in candidate_rows:
        dependency_rows.append(
            "<tr class='candidate-row'>"
            f"<td><b>{html.escape(row['id'])}</b><br><span>{html.escape(row['track'])}</span></td>"
            f"<td>{html.escape(', '.join(row.get('parents', [])) or 'root')}</td>"
            "<td>设计合同</td><td>尚无 builder/evaluator</td>"
            f"<td><code>{html.escape(Path(row.get('files', {}).get('design', {}).get('path', 'not-recorded')).name)}</code></td>"
            f"<td><code>{html.escape(row['candidate_content_fingerprint_sha256'][:16])}…</code></td>"
            "</tr>"
        )
    dependency_table = "".join(dependency_rows)
    payload = json.dumps(
        {"versions": rows, "candidates": candidate_rows}, ensure_ascii=False
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAG MFU 版本与数据依赖</title><style>
:root{{--bg:#071521;--panel:#102536;--ink:#eef7ff;--muted:#9ab1c5;--cyan:#55c2ff;--pink:#ff82dc;--gold:#ffd166}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#06111b,#0c2030);color:var(--ink);font-family:Inter,"Noto Sans SC",system-ui,sans-serif}}
main{{max-width:1480px;margin:auto;padding:38px 28px 72px}}h1{{margin:0 0 8px;font-size:34px}}.lead{{color:var(--muted);line-height:1.7;max-width:980px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:22px 0}}.legend span{{font-size:13px;padding:7px 11px;border-radius:99px;background:#ffffff0d;border:1px solid #ffffff1a}}
.flow{{display:flex;align-items:stretch;gap:10px;overflow-x:auto;padding:20px 4px 26px}}.arrow{{align-self:center;color:#6f91aa;font-size:24px}}
.node{{min-width:205px;max-width:230px;background:var(--panel);border:1px solid #27475d;border-radius:15px;padding:16px;box-shadow:0 10px 28px #0004}}
.node h3{{font-size:16px;margin:10px 0}}.node p{{font-size:12px;line-height:1.55;color:var(--muted);min-height:56px}}.tag{{font-size:10px;letter-spacing:.12em;font-weight:800}}.meta{{font-size:10px;color:#b9cad8;margin:12px 0 7px;word-break:break-word}}code{{font-size:10px;color:#9ed9fb}}
.mainline{{border-top:3px solid var(--cyan)}}.development{{border-top:3px solid var(--pink)}}.candidate{{border:2px dashed #8f762f;border-top:3px dashed var(--gold)}}
.branches{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:8px}}.branch{{background:#ffffff08;border:1px solid #ffffff12;border-radius:16px;padding:18px}}.branch h2{{font-size:16px;margin:0 0 14px}}.branch .node{{max-width:none}}
.pipeline{{margin:26px 0;background:#ffffff08;border:1px solid #ffffff12;border-radius:16px;padding:18px;overflow:auto}}.pipeline h2{{font-size:18px;margin:0 0 8px}}.formula{{padding:13px 16px;margin:14px 0;border-radius:10px;background:#071521;border:1px solid #29485e;color:#bfe8ff;white-space:nowrap}}table{{border-collapse:collapse;width:100%;min-width:980px;font-size:12px}}th,td{{padding:10px;border-bottom:1px solid #244054;text-align:left;vertical-align:top}}th{{color:#b9d5e8;background:#0b1f2e}}td span{{color:var(--muted)}}.candidate-row{{background:#ffd1660b}}
.notice{{margin-top:26px;padding:17px 19px;border-radius:12px;background:#ffd16613;border:1px solid #ffd1664a;color:#e9dba8;line-height:1.7}}
@media(max-width:760px){{.branches{{grid-template-columns:1fr}}main{{padding:28px 16px}}}}
</style></head><body><main><h1>DAG MFU 版本与数据依赖</h1>
<p class="lead">一个节点不是一张HTML，而是 config、builder/evaluator、结果锚点与内容指纹的集合。实线是已实现版本依赖；候选只冻结设计，不具备发布资格。</p>
<div class="legend"><span>蓝色 · source-only预测主线</span><span>粉色 · target-assisted开发支线</span><span>黄色虚线 · 未发布候选</span></div>
<section class="flow">{main_cards}</section>
<section class="branches"><div class="branch"><h2>从 v6.7 分出的开发诊断</h2>{branch_cards}</div><div class="branch"><h2>从 v6.7 分出的下一候选</h2>{candidate_cards}</div></section>
<section class="pipeline"><h2>每个版本不是文件名，而是一条可重跑的数据合同</h2>
<p class="lead">父版本与冻结输入发生变化时，Snakemake沿依赖边重跑后继；结果锚点与内容指纹用于确认“同一个版本”仍是同一份数据和代码。</p>
<div class="formula">父版本结果 + 冻结 config → builder / sealer → 结果锚点 → evaluator（如有）→ SHA256 内容锁</div>
<table><thead><tr><th>版本/轨道</th><th>父依赖</th><th>冻结配置</th><th>执行角色</th><th>结果锚点</th><th>可迁移内容指纹</th></tr></thead><tbody>{dependency_table}</tbody></table></section>
<div class="notice"><b>当前边界：</b>v6.8.4是代码语义闭合后的预测头；相对v6.8.3只增加静态1F1B/PP API可证明的阻塞发送边，未删除父图边。参数搜索必须匹配v6.8.4拓扑指纹，只能修改有来源的节点成本。224卡只做封存后的开发评估，且因目标场景此前已被分析，不宣称独立blind结果。v6.9为DESIGN_ONLY_BLOCKED_NOT_RELEASED。内容锁与Git发布门禁分离，dirty工作树不妨碍科学结果，但禁止声称已有clean Git lineage。</div>
<script type="application/json" id="version-data">{payload}</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    versions = payload["version"]
    candidates = payload.get("candidate", [])
    ordered = topological_order(versions)
    by_id = {item["id"]: item for item in versions}
    candidate_ids = {item["id"] for item in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("duplicate candidate id")
    if candidate_ids & set(by_id):
        raise ValueError(f"candidate/version id collision: {sorted(candidate_ids & set(by_id))}")
    known_candidate_parents = set(by_id) | candidate_ids
    missing_candidate_parents = {
        parent
        for item in candidates
        for parent in item.get("parents", [])
        if parent not in known_candidate_parents
    }
    if missing_candidate_parents:
        raise ValueError(f"undeclared candidate parents: {sorted(missing_candidate_parents)}")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for version_id in ordered:
        item = by_id[version_id]
        files: dict[str, dict[str, Any]] = {}
        for role in ("config", "builder", "finalizer", "sealer", "evaluator", "evaluation_config"):
            if role in item:
                files[role] = inspect_file(repo_path(item[role]))
                if not files[role]["exists"]:
                    missing.append(files[role]["path"])
        run_dir = repo_path(item["run_dir"])
        anchor = inspect_file(run_dir / item["anchor"])
        if not anchor["exists"]:
            missing.append(anchor["path"])
        provenance = inspect_file(run_dir / "provenance.json")
        evaluation_metrics = inspect_file(run_dir / "evaluator_only/metrics.json")
        reproduction_command = inspect_file(run_dir / "reproduction_command.txt")
        artifact_manifest = inspect_file(run_dir / "artifact_manifest.json")
        config_inputs: dict[str, dict[str, Any]] = {}
        config_payload = tomllib.loads(repo_path(item["config"]).read_text(encoding="utf-8"))
        for input_name, input_value in config_payload.get("inputs", {}).items():
            input_path = repo_path(str(input_value))
            if input_path.is_dir():
                config_inputs[input_name] = {
                    "path": str(input_path), "exists": True, "kind": "directory",
                }
            else:
                config_inputs[input_name] = {"kind": "file", **inspect_file(input_path)}
        seal_status = None
        if anchor["exists"] and anchor["path"].endswith("prediction_seal.json"):
            seal_status = json.loads(Path(anchor["path"]).read_text(encoding="utf-8")).get("status")
        row = {
            "id": version_id,
            "slug": item["slug"],
            "track": item["track"],
            "parents": item.get("parents", []),
            "claim": item["claim"],
            "run_dir": str(run_dir),
            "files": files,
            "anchor": anchor,
            "provenance": provenance,
            "evaluation_metrics": evaluation_metrics,
            "reproduction_command": reproduction_command,
            "artifact_manifest": artifact_manifest,
            "declared_config_inputs": config_inputs,
            "seal_status": seal_status,
        }
        row["version_fingerprint_sha256"] = fingerprint(
            list(files.values())
            + [anchor, provenance, evaluation_metrics, reproduction_command, artifact_manifest]
            + [value for value in config_inputs.values() if value.get("kind") == "file"]
        )
        portable_parts = [
            {"role": role, **value} for role, value in files.items()
        ] + [
            {"role": role, **value} for role, value in (
                ("anchor", anchor),
                ("provenance", provenance),
                ("evaluation_metrics", evaluation_metrics),
                ("reproduction_command", reproduction_command),
                ("artifact_manifest", artifact_manifest),
            )
        ] + [
            {"role": f"config_input:{name}", **value}
            for name, value in config_inputs.items() if value.get("kind") == "file"
        ]
        row["portable_content_fingerprint_sha256"] = portable_content_fingerprint(portable_parts)
        rows.append(row)

    candidate_rows: list[dict[str, Any]] = []
    for item in candidates:
        files = {
            role: inspect_file(repo_path(item[role]))
            for role in ("report", "design", "runtime_contract", "graph", "manifest")
            if role in item
        }
        for value in files.values():
            if not value["exists"]:
                missing.append(value["path"])
        fingerprint_roles = ("design", "runtime_contract", "graph")
        fingerprint_parts = [
            {"role": role, **files[role]}
            for role in fingerprint_roles
            if role in files
        ]
        candidate_rows.append({
            "id": item["id"],
            "slug": item["slug"],
            "track": item["track"],
            "parents": item.get("parents", []),
            "status": item["status"],
            "claim": item["claim"],
            "files": files,
            "candidate_content_fingerprint_sha256": portable_content_fingerprint(fingerprint_parts),
            "fingerprint_roles": list(fingerprint_roles),
            "release_eligible": False,
        })

    status = "PASS" if not missing else "PARTIAL"
    snapshot = {
        "schema": "dag-mfu-version-inventory-v4",
        "status": status,
        "manifest": inspect_file(manifest_path),
        "snapshot_semantics": "deterministic scientific content inventory; Git state is audited separately",
        "topological_order": ordered,
        "missing_required_files": sorted(set(missing)),
        "versions": rows,
        "candidates": candidate_rows,
    }
    (output / "version_inventory.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "version_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "slug", "track", "parents", "anchor_exists", "anchor_sha256",
            "version_fingerprint_sha256", "reproduction_command_exists",
            "portable_content_fingerprint_sha256", "artifact_manifest_exists", "seal_status",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"], "slug": row["slug"], "track": row["track"],
                "parents": ",".join(row["parents"]),
                "anchor_exists": row["anchor"]["exists"],
                "anchor_sha256": row["anchor"]["sha256"],
                "version_fingerprint_sha256": row["version_fingerprint_sha256"],
                "portable_content_fingerprint_sha256": row["portable_content_fingerprint_sha256"],
                "reproduction_command_exists": row["reproduction_command"]["exists"],
                "artifact_manifest_exists": row["artifact_manifest"]["exists"],
                "seal_status": row["seal_status"],
            })
    with (output / "candidate_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "slug", "track", "parents", "status",
            "candidate_content_fingerprint_sha256", "release_eligible",
        ])
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({
                "id": row["id"],
                "slug": row["slug"],
                "track": row["track"],
                "parents": ",".join(row["parents"]),
                "status": row["status"],
                "candidate_content_fingerprint_sha256": row[
                    "candidate_content_fingerprint_sha256"
                ],
                "release_eligible": row["release_eligible"],
            })

    graph = ["flowchart LR"]
    for row in rows:
        style = "target" if row["track"] == "target_assisted_development" else "main"
        graph.append(f'  {row["id"].replace(".", "_")}["{row["id"]} · {row["slug"]}"]:::{style}')
        for parent in row["parents"]:
            if parent in by_id:
                graph.append(f'  {parent.replace(".", "_")} --> {row["id"].replace(".", "_")}')
            else:
                external_id = "ext_" + parent.replace(".", "_")
                graph.append(f'  {external_id}["{parent} · external sealed backend"]:::external --> {row["id"].replace(".", "_")}')
    for row in candidate_rows:
        node_id = row["id"].replace(".", "_")
        graph.append(f'  {node_id}["{row["id"]} · {row["slug"]}<br/>CANDIDATE"]:::candidate')
        for parent in row["parents"]:
            graph.append(f'  {parent.replace(".", "_")} -.-> {node_id}')
    graph.extend([
        "  classDef main fill:#123b52,stroke:#55c2ff,color:#fff",
        "  classDef target fill:#4b244f,stroke:#ff82dc,color:#fff",
        "  classDef candidate fill:#5a4a20,stroke:#ffd166,color:#fff,stroke-dasharray: 5 5",
        "  classDef external fill:#3b3b3b,stroke:#aaa,color:#fff",
    ])
    (output / "version_graph.mmd").write_text("\n".join(graph) + "\n", encoding="utf-8")
    (output / "version_graph.html").write_text(
        render_version_graph_html(rows, candidate_rows), encoding="utf-8"
    )

    fingerprints = {row["id"]: row["portable_content_fingerprint_sha256"] for row in rows}
    lock_entries = []
    for row in rows:
        local_parents = [parent for parent in row["parents"] if parent in fingerprints]
        lock_entries.append({
            "id": row["id"],
            "track": row["track"],
            "parents": row["parents"],
            "parent_content_fingerprints": {
                parent: fingerprints[parent] for parent in local_parents
            },
            "portable_content_fingerprint_sha256": row["portable_content_fingerprint_sha256"],
            "anchor_sha256": row["anchor"]["sha256"],
            "seal_status": row["seal_status"],
        })
    lock = {
        "schema": "dag-mfu-portable-version-lock-v2",
        "status": "CONTENT_LOCK_READY_GIT_GATE_SEPARATE",
        "fingerprint_semantics": "sha256 over sorted logical-role + content-sha256; absolute paths excluded",
        "topological_order": ordered,
        "entries": lock_entries,
        "candidate_entries": [
            {
                "id": row["id"],
                "track": row["track"],
                "parents": row["parents"],
                "status": row["status"],
                "candidate_content_fingerprint_sha256": row[
                    "candidate_content_fingerprint_sha256"
                ],
                "release_eligible": False,
            }
            for row in candidate_rows
        ],
    }
    lock_path = output / "version_lock.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plan = f"""# DAG MFU Git 版本管理边界

版本内容清单状态：`{status}`。当前分支、提交和工作树状态由独立的 `git_release_gate.json` 记录，不进入科学内容锁。

## 固定原则

1. 一个模型版本由 `config + builder/finalizer/sealer/evaluator + tests + 小型 provenance/metrics` 共同定义，不能只给 HTML 命名。
2. 预测主线只允许 256 卡校准数据进入模型进程；224 卡 Trace 只由封存后的 evaluator 读取。
3. v6.8.4属于代码语义闭合的预测主线；ENTRY只取256卡Trace iter85，224卡只做封存后开发评估。参数搜索不得新增、删除或改向依赖边。因目标场景此前已被分析，不得写成独立blind实验。
4. v6.9 只登记为 `predictive_candidate`；设计、runtime contract与图结构有独立content fingerprint，但在完整builder/source-holdout/evaluator门禁前永远 `release_eligible=false`。
5. 大型 Trace、节点/边压缩表和可重建 HTML 不进入 Git 历史；由 `version_inventory.json` 中的绝对路径和 SHA256 固定。
6. 当前工作区包含大量未跟踪/既有修改，禁止 `git add -A`。迁移前按版本逐个列出精确 pathspec。

## 建议提交单元

- `dag-v6x-code`: 该版本 config、script、test 和方法文档。
- `dag-v6x-evidence`: 小型 contract、seal、metrics、provenance 和 artifact manifest；不含原始 Trace。
- `dag-mfu-pipeline`: 本版本图、Snakemake 规则、分析脚本与测试。

建议标签使用 `dag-mfu/v6.7-source256`、`dag-mfu/v6.8-source256-entry`与`dag-mfu/v6.8.4-code-derived-topology`，标签名称直接编码数据边界和状态。迁移到 `{payload['meta']['migration_target']}` 时保留相同 manifest id 和 SHA256，不依赖当前结果目录的 Git 历史。
"""
    (output / "GIT_VERSIONING.md").write_text(plan, encoding="utf-8")
    print(json.dumps({
        "status": status, "versions": len(rows), "candidates": len(candidate_rows),
        "content_lock": lock["status"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

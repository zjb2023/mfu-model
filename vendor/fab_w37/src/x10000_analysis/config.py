from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


HashMode = Literal["none", "full"]


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    experiment: str
    deepep_group: str
    world_size: int
    host_split_rank: int
    analysis_iters: tuple[int, ...]
    profiler_iters: tuple[int, ...]
    excluded_iters: tuple[int, ...]
    data_root: Path
    training_log_dir: Path
    profiler_dir: Path
    deepep_dir: Path
    nic_dir: Path
    mtlink_dir: Path
    workspace: Path
    derived_v5: Path
    figures_v5: Path
    reports: Path
    thresholds: dict[str, float]


def load_run_config(path: Path) -> RunConfig:
    """Load a v5 TOML config, resolving relative paths from its directory."""
    config_path = path.resolve()
    config_dir = config_path.parent
    with config_path.open("rb") as fh:
        doc = tomllib.load(fh)
    run = doc["run"]
    paths = doc["paths"]
    excluded = tuple(int(x) for x in run.get("excluded_iters", []))
    analysis_iters = tuple(
        i
        for i in range(int(run["analysis_iter_start"]), int(run["analysis_iter_end"]) + 1)
        if i not in excluded
    )
    profiler_iters = tuple(
        i
        for i in range(
            int(run["profiler_iter_start"]),
            int(run["profiler_iter_end"]) + 1,
            int(run["profiler_iter_step"]),
        )
        if i not in excluded
    )

    def resolved_path(key: str) -> Path:
        configured = Path(paths[key]).expanduser()
        if configured.is_absolute():
            return configured
        return (config_dir / configured).resolve()

    return RunConfig(
        run_id=str(run["run_id"]),
        experiment=str(run["experiment"]),
        deepep_group=str(run["deepep_group"]),
        world_size=int(run["world_size"]),
        host_split_rank=int(run["host_split_rank"]),
        analysis_iters=analysis_iters,
        profiler_iters=profiler_iters,
        excluded_iters=excluded,
        data_root=resolved_path("data_root"),
        training_log_dir=resolved_path("training_log_dir"),
        profiler_dir=resolved_path("profiler_dir"),
        deepep_dir=resolved_path("deepep_dir"),
        nic_dir=resolved_path("nic_dir"),
        mtlink_dir=resolved_path("mtlink_dir"),
        workspace=resolved_path("workspace"),
        derived_v5=resolved_path("derived_v5"),
        figures_v5=resolved_path("figures_v5"),
        reports=resolved_path("reports"),
        thresholds={str(k): float(v) for k, v in doc.get("thresholds", {}).items()},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, hash_mode: HashMode) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_mode == "full":
        record["sha256"] = _sha256(path)
    return record


def _require_dir(label: str, path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required input directory: {label}={path}")


def _rank_from_trace(path: Path) -> int:
    match = re.match(r"rank(\d+)\.", path.name)
    if not match:
        raise ValueError(f"Unrecognized profiler trace filename: {path.name}")
    return int(match.group(1))


def build_input_inventory(cfg: RunConfig, hash_mode: HashMode = "full") -> dict[str, Any]:
    """Discover required inputs and fail closed on missing iter/rank coverage."""
    for label, path in (
        ("training_log_dir", cfg.training_log_dir),
        ("profiler_dir", cfg.profiler_dir),
        ("deepep_dir", cfg.deepep_dir),
        ("nic_dir", cfg.nic_dir),
        ("mtlink_dir", cfg.mtlink_dir),
    ):
        _require_dir(label, path)

    training_logs = sorted(cfg.training_log_dir.glob("*.log"))
    deepep_logs = sorted(cfg.deepep_dir.glob("r*_rank*.log"))
    nic_csv = sorted(cfg.nic_dir.glob("*.csv"))
    mtlink_csv = sorted(cfg.mtlink_dir.glob("*.csv"))

    profiler_traces: list[Path] = []
    ranks_by_iter: dict[str, list[int]] = {}
    expected_ranks = list(range(cfg.world_size))
    for iter_id in cfg.profiler_iters:
        iter_dir = cfg.profiler_dir / f"iteration_{iter_id}"
        _require_dir(f"profiler iteration {iter_id}", iter_dir)
        traces = sorted(iter_dir.glob("rank*.pt.trace.json"), key=_rank_from_trace)
        ranks = [_rank_from_trace(path) for path in traces]
        if ranks != expected_ranks:
            raise FileNotFoundError(
                f"Missing required input profiler ranks for iter {iter_id}: "
                f"expected={expected_ranks}, actual={ranks}"
            )
        profiler_traces.extend(traces)
        ranks_by_iter[str(iter_id)] = ranks

    groups = {
        "training_logs": training_logs,
        "profiler_traces": profiler_traces,
        "deepep_logs": deepep_logs,
        "nic_csv": nic_csv,
        "mtlink_csv": mtlink_csv,
    }
    for label, files in groups.items():
        if not files:
            raise FileNotFoundError(f"Missing required input files: {label}")

    return {
        "run_id": cfg.run_id,
        "analysis_iters": list(cfg.analysis_iters),
        "profiler_iters": list(cfg.profiler_iters),
        "profiler_ranks_by_iter": ranks_by_iter,
        "counts": {label: len(files) for label, files in groups.items()},
        "files": {
            label: [_file_record(path, hash_mode) for path in files]
            for label, files in groups.items()
        },
        "hash_mode": hash_mode,
    }


def write_input_manifest(
    cfg: RunConfig,
    output_path: Path,
    hash_mode: HashMode = "full",
) -> dict[str, Any]:
    """Build and atomically write the input inventory as JSON."""
    inventory = build_input_inventory(cfg, hash_mode=hash_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    temp_path.replace(output_path)
    return inventory
